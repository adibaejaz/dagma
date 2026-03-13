import typing
import csv
from pathlib import Path

import numpy as np
import scipy.linalg as sla
from scipy.special import expit as sigmoid
from tqdm.auto import tqdm

try:
    from .linear_query import QueryDagmaLinear
except ImportError:
    from linear_query import QueryDagmaLinear


__all__ = ["CPDAGQueryDagmaLinear"]


class CPDAGQueryDagmaLinear(QueryDagmaLinear):
    """
    Query-augmented DAGMA linear model with optional CPDAG constraints.

    The CPDAG uses the convention:

    - ``1`` for a directed edge ``i -> j``
    - ``1`` in both ``(i, j)`` and ``(j, i)`` for an undirected edge ``i - j``
    - ``0`` otherwise
    """

    def __init__(self, loss_type: str, verbose: bool = False, dtype: type = np.float64) -> None:
        super().__init__(loss_type=loss_type, verbose=verbose, dtype=dtype)
        self.use_cpdag_objective = False

    @staticmethod
    def _validate_cpdag(cpdag: np.ndarray, d: int) -> np.ndarray:
        """
        Validate and normalize a CPDAG matrix.

        Parameters
        ----------
        cpdag : np.ndarray
            :math:`(d,d)` CPDAG adjacency matrix taking values in ``{0, 1}``.
        d : int
            Number of nodes.

        Returns
        -------
        np.ndarray
            CPDAG converted to an integer numpy array.
        """
        cpdag = np.asarray(cpdag, dtype=int)
        if cpdag.shape != (d, d):
            raise ValueError(f"cpdag should have shape {(d, d)}")
        if not np.all((cpdag == 0) | (cpdag == 1)):
            raise ValueError("cpdag should take values in {0, 1}")
        if np.any(np.diag(cpdag) != 0):
            raise ValueError("cpdag should have zero diagonal")
        return cpdag

    @staticmethod
    def _merge_edge_constraints(
        base_edges: typing.Optional[typing.List[typing.Tuple[int, int]]],
        extra_edges: typing.Iterable[typing.Tuple[int, int]],
    ) -> typing.Optional[typing.Tuple[typing.Tuple[int, int], ...]]:
        """
        Merge existing edge constraints with CPDAG-derived constraints.
        """
        merged = set(base_edges or [])
        merged.update(extra_edges)
        if not merged:
            return None
        return tuple(sorted(merged))

    @classmethod
    def _cpdag_to_constraints(
        cls,
        cpdag: np.ndarray,
    ) -> typing.Tuple[
        typing.Optional[typing.Tuple[typing.Tuple[int, int], ...]],
        typing.Optional[typing.Tuple[typing.Tuple[int, int], ...]],
    ]:
        """
        Convert a CPDAG into include/exclude constraints.

        Directed CPDAG edges are treated as required in the given orientation.
        Missing adjacencies are excluded in both directions. Undirected CPDAG
        edges constrain the skeleton but do not force an orientation here.

        Returns
        -------
        typing.Tuple[typing.Optional[tuple], typing.Optional[tuple]]
            ``(include_edges, exclude_edges)`` derived from the CPDAG.
        """
        d = cpdag.shape[0]
        include_edges = set()
        exclude_edges = set()

        for i in range(d):
            for j in range(d):
                if cpdag[i, j] == 1 and cpdag[j, i] == 0:
                    include_edges.add((i, j))
                    exclude_edges.add((j, i))

        for i in range(d):
            for j in range(i + 1, d):
                if cpdag[i, j] == 0 and cpdag[j, i] == 0:
                    exclude_edges.add((i, j))
                    exclude_edges.add((j, i))

        include_tuple = tuple(sorted(include_edges)) if include_edges else None
        exclude_tuple = tuple(sorted(exclude_edges)) if exclude_edges else None
        return include_tuple, exclude_tuple

    def fit(
        self,
        X: np.ndarray,
        lambda1: float = 0.03,
        w_threshold: float = 0.3,
        T: int = 5,
        mu_init: float = 1.0,
        mu_factor: float = 0.1,
        gamma_init: float = 1e-3,
        gamma_factor: float = 10.0,
        gamma_warmup: int = 2,
        s: typing.Union[typing.List[float], float] = [1.0, .9, .8, .7, .6],
        warm_iter: int = 3e4,
        max_iter: int = 6e4,
        lr: float = 0.0003,
        checkpoint: int = 1000,
        beta_1: float = 0.99,
        beta_2: float = 0.999,
        exclude_edges: typing.Optional[typing.List[typing.Tuple[int, int]]] = None,
        include_edges: typing.Optional[typing.List[typing.Tuple[int, int]]] = None,
        effect_src: typing.Optional[int] = None,
        effect_dst: typing.Optional[int] = None,
        effect_mode: typing.Optional[str] = None,
        cpdag: typing.Optional[np.ndarray] = None,
    ) -> np.ndarray:
        r"""
        Runs DAGMA on observational data while optionally shaping a total effect
        and constraining the solution with a CPDAG.

        Parameters
        ----------
        X : np.ndarray
            :math:`(n,d)` dataset.
        lambda1 : float
            Coefficient of the L1 penalty. Defaults to 0.03.
        w_threshold : float, optional
            Removes edges with weight value less than the given threshold. Defaults to 0.3.
        T : int, optional
            Number of DAGMA iterations. Defaults to 5.
        mu_init : float, optional
            Initial value of :math:`\mu`. Defaults to 1.0.
        mu_factor : float, optional
            Decay factor for :math:`\mu`. Defaults to 0.1.
        gamma_init : float, optional
            Initial value of :math:`\gamma` after warmup. Defaults to 1e-3.
        gamma_factor : float, optional
            Multiplicative factor applied to :math:`\gamma` after each outer iteration following warmup. Defaults to 10.0.
        gamma_warmup : int, optional
            Number of initial outer iterations for which :math:`\gamma` is held at 0. Defaults to 2.
        s : typing.Union[typing.List[float], float], optional
            Controls the domain of M-matrices. Defaults to [1.0, .9, .8, .7, .6].
        warm_iter : int, optional
            Number of iterations for :py:meth:`~dagma.linear.DagmaLinear.minimize` for :math:`t < T`. Defaults to 3e4.
        max_iter : int, optional
            Number of iterations for :py:meth:`~dagma.linear.DagmaLinear.minimize` for :math:`t = T`. Defaults to 6e4.
        lr : float, optional
            Learning rate. Defaults to 0.0003.
        checkpoint : int, optional
            If ``verbose`` is ``True``, then prints to stdout every ``checkpoint`` iterations. Defaults to 1000.
        beta_1 : float, optional
            Adam hyperparameter. Defaults to 0.99.
        beta_2 : float, optional
            Adam hyperparameter. Defaults to 0.999.
        exclude_edges : typing.Optional[typing.List[typing.Tuple[int, int]]], optional
            Tuple of edges that should be excluded from the DAG solution. Defaults to None.
        include_edges : typing.Optional[typing.List[typing.Tuple[int, int]]], optional
            Tuple of edges that should be included in the DAG solution. Defaults to None.
        effect_src : typing.Optional[int], optional
            Source node index for the total-effect query. Defaults to None.
        effect_dst : typing.Optional[int], optional
            Target node index for the total-effect query. Defaults to None.
        effect_mode : typing.Optional[str], optional
            Query objective mode. One of ``"maximize"``, ``"minimize"``, or ``"maximize_squared"``.
            Required when total-effect shaping is enabled. Defaults to None.
        cpdag : typing.Optional[np.ndarray], optional
            CPDAG adjacency matrix with values in ``{0, 1}``. Undirected edges
            are represented by ``cpdag[i, j] = cpdag[j, i] = 1``. Directed edges
            are represented by ``cpdag[i, j] = 1`` and ``cpdag[j, i] = 0``.
            Directed edges are enforced in their given orientation. Missing
            adjacencies are excluded. Undirected edges keep both orientations
            available.

        Returns
        -------
        np.ndarray
            Estimated DAG from data.
        """
        self.use_cpdag_objective = cpdag is not None
        if cpdag is not None:
            cpdag = self._validate_cpdag(cpdag, X.shape[1])
            cpdag_include, cpdag_exclude = self._cpdag_to_constraints(cpdag)
            include_edges = self._merge_edge_constraints(include_edges, cpdag_include or [])
            exclude_edges = self._merge_edge_constraints(exclude_edges, cpdag_exclude or [])

        return super().fit(
            X=X,
            lambda1=lambda1,
            w_threshold=w_threshold,
            T=T,
            mu_init=mu_init,
            mu_factor=mu_factor,
            gamma_init=gamma_init,
            gamma_factor=gamma_factor,
            gamma_warmup=gamma_warmup,
            s=s,
            warm_iter=warm_iter,
            max_iter=max_iter,
            lr=lr,
            checkpoint=checkpoint,
            beta_1=beta_1,
            beta_2=beta_2,
            exclude_edges=exclude_edges,
            include_edges=include_edges,
            effect_src=effect_src,
            effect_dst=effect_dst,
            effect_mode=effect_mode,
        )

    def _func(
        self, W: np.ndarray, mu: float, gamma: float, s: float = 1.0
    ) -> typing.Tuple[float, float, float, float]:
        """
        Evaluate the objective with likelihood-only score scaling.

        In this subclass, the objective is

        ``mu * score(W) + gamma * query(W) + h(W)``

        so ``mu`` scales only the likelihood term and there is no L1 penalty.
        """
        if not self.use_cpdag_objective:
            return super()._func(W, mu, gamma, s)
        score, _ = self._score(W)
        h, _ = self._h(W, s)
        query, _, _ = self._query(W)
        obj = mu * score + gamma * query + h
        return obj, score, h, query

    def _record_history(
        self,
        W: np.ndarray,
        mu: float,
        gamma: float,
        s: float,
        outer_iter: int,
        inner_iter: int,
    ) -> typing.Tuple[float, float, float, float]:
        """
        Store checkpointed objective components with likelihood-only score loss.

        In this subclass, ``score_loss`` means only the weighted likelihood term
        ``mu * score``. There is no L1 penalty term in the objective.
        """
        if not self.use_cpdag_objective:
            return super()._record_history(W, mu, gamma, s, outer_iter, inner_iter)
        _, score, h, query = self._func(W, mu, gamma, s)
        score_loss = mu * score
        query_loss = gamma * query
        self.loss_history["outer_iter"].append(outer_iter)
        self.loss_history["inner_iter"].append(inner_iter)
        self.loss_history["step"].append(self.loss_step_offset + inner_iter)
        self.loss_history["score_loss"].append(score_loss)
        self.loss_history["query_loss"].append(query_loss)
        self.loss_history["acyclicity_loss"].append(h)
        return score, h, query, score_loss + query_loss + h

    def minimize(
        self,
        W: np.ndarray,
        mu: float,
        gamma: float,
        max_iter: int,
        s: float,
        lr: float,
        outer_iter: int,
        tol: float = 1e-6,
        beta_1: float = 0.99,
        beta_2: float = 0.999,
        pbar: typing.Optional[tqdm] = None,
    ) -> typing.Tuple[np.ndarray, bool]:
        """
        Minimize the CPDAG-query objective with likelihood-only score scaling.
        """
        if not self.use_cpdag_objective:
            return super().minimize(
                W=W,
                mu=mu,
                gamma=gamma,
                max_iter=max_iter,
                s=s,
                lr=lr,
                outer_iter=outer_iter,
                tol=tol,
                beta_1=beta_1,
                beta_2=beta_2,
                pbar=pbar,
            )
        obj_prev = 1e16
        self.opt_m, self.opt_v = 0, 0
        self.vprint(
            f"\n\nMinimize with -- mu:{mu} -- gamma:{gamma} -- lr: {lr} -- s: {s} for {max_iter} max iterations"
        )
        mask_exc = np.ones((self.d, self.d), dtype=self.dtype)
        if self.exc_c is not None:
            mask_exc[self.exc_r, self.exc_c] = 0.0

        for iter in range(1, max_iter + 1):
            M = sla.inv(s * self.Id - W * W) + 1e-16
            while np.any(M < 0):
                if iter == 1 or s <= 0.9:
                    self.vprint(f"W went out of domain for s={s} at iteration {iter}")
                    return W, False
                W += lr * grad
                lr *= 0.5
                if lr <= 1e-16:
                    return W, True
                W -= lr * grad
                M = sla.inv(s * self.Id - W * W) + 1e-16
                self.vprint(f"Learning rate decreased to lr: {lr}")

            if self.loss_type == "l2":
                G_score = -mu * self.cov @ (self.Id - W)
            elif self.loss_type == "logistic":
                G_score = mu / self.n * self.X.T @ sigmoid(self.X @ W) - mu * self.cov
            else:
                raise ValueError(f"Unsupported loss type: {self.loss_type}")

            _, G_query, tau = self._query(W)
            Gobj = (
                G_score
                + 2 * W * M.T
                + gamma * G_query
            )

            grad = self._adam_update(Gobj, iter, beta_1, beta_2)
            W -= lr * grad
            W *= mask_exc

            if iter % self.checkpoint == 0 or iter == max_iter:
                score, h, query, obj_new = self._record_history(W, mu, gamma, s, outer_iter, iter)
                self.vprint(f"\nInner iteration {iter}")
                self.vprint(f"\th(W_est): {h:.4e}")
                self.vprint(f"\tscore(W_est): {score:.4e}")
                if gamma != 0.0:
                    self.vprint(f"\ttau(W_est): {tau:.4e}")
                    self.vprint(f"\tquery(W_est): {query:.4e}")
                    self.vprint(f"\tweighted_query(W_est): {(gamma * query):.4e}")
                self.vprint(f"\tobj(W_est): {obj_new:.4e}")
                if np.abs((obj_prev - obj_new) / obj_prev) <= tol:
                    pbar.update(max_iter - iter + 1)
                    break
                obj_prev = obj_new
            pbar.update(1)
        return W, True


def experiment(
    B_true: np.ndarray,
    cpdag: typing.Optional[np.ndarray],
    src: int,
    dst: int,
    graph_name: str = "graph",
    query_label: typing.Optional[str] = None,
    id_status: str = "unknown",
    csv_path: str = "cpdag_query_results.csv",
) -> None:
    import matplotlib.pyplot as plt
    import utils
    from timeit import default_timer as timer

    utils.set_random_seed(1)

    n, d = 5000, B_true.shape[0]
    sem_type = "gauss"
    scale  = np.random.uniform(low=0.5, high=1, size=d)
    if query_label is None:
        query_label = f"{src}->{dst}"
    if cpdag is None:
        cpdag = B_true.copy()

    W_true = utils.simulate_parameter(B_true)
    X = utils.simulate_linear_sem(W_true, n, sem_type, noise_scale=scale)

    queries = [
        (f"baseline {src}->{dst}", src, dst, None),
        (f"maximize {src}->{dst}", src, dst, "maximize"),
        (f"minimize {src}->{dst}", src, dst, "minimize"),
    ]
    settings = [
        ("with CPDAG", cpdag),
        ("without CPDAG", None),
    ]
    fig, axes = plt.subplots(3, 2, figsize=(12, 12), sharex=False)
    axes = axes.ravel()
    csv_file = Path(csv_path)
    write_header = not csv_file.exists()
    fieldnames = [
        "graph_name",
        "query_label",
        "id_status",
        "cpdag_setting",
        "query_mode",
        "src",
        "dst",
        "is_dag",
        "h_final",
        "fdr",
        "tpr",
        "fpr",
        "shd",
        "nnz",
        "true_total_effect",
        "estimated_total_effect",
        "runtime_seconds",
    ]

    ax_idx = 0
    with csv_file.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()

        for setting_label, setting_cpdag in settings:
            print(f"=== {setting_label} ===")
            for label, src, dst, mode in queries:
                ax = axes[ax_idx]
                ax_idx += 1

                model = CPDAGQueryDagmaLinear(loss_type="l2")
                start = timer()
                fit_kwargs = dict(
                    X=X,
                    lambda1=0.02,
                )
                if setting_cpdag is not None:
                    fit_kwargs["cpdag"] = setting_cpdag
                if mode is not None:
                    fit_kwargs.update(
                        effect_src=src,
                        effect_dst=dst,
                        effect_mode=mode,
                        gamma_init=1e-5,
                        gamma_factor=2.0,
                        gamma_warmup=2,
                    )
                else:
                    fit_kwargs.update(
                        gamma_init=0.0,
                        gamma_factor=1.0,
                        gamma_warmup=0,
                    )
                W_est = model.fit(**fit_kwargs)
                end = timer()
                is_dag = utils.is_dag(W_est)
                true_total_effect = model.total_effect(W_true, src, dst)
                total_effect = model.total_effect(W_est, src, dst)
                runtime = end - start

                print(f"{setting_label} | {label}")
                print(f"is_dag: {is_dag}")
                print(f"final h: {model.h_final:.4e}")
                acc = None
                if is_dag:
                    acc = utils.count_accuracy(B_true, W_est != 0)
                    print(acc)
                else:
                    print("accuracy skipped because learned W is not a DAG")
                print("learned W:")
                print(W_est)
                print(f"true total effect: {true_total_effect:.4f}")
                print(f"computed total effect: {total_effect:.4f}")
                print(f"time: {runtime:.4f}s")
                print()

                writer.writerow(
                    {
                        "graph_name": graph_name,
                        "query_label": query_label,
                        "id_status": id_status,
                        "cpdag_setting": setting_label,
                        "query_mode": "baseline" if mode is None else mode,
                        "src": src,
                        "dst": dst,
                        "is_dag": is_dag,
                        "h_final": model.h_final,
                        "fdr": "" if acc is None else acc["fdr"],
                        "tpr": "" if acc is None else acc["tpr"],
                        "fpr": "" if acc is None else acc["fpr"],
                        "shd": "" if acc is None else acc["shd"],
                        "nnz": "" if acc is None else acc["nnz"],
                        "true_total_effect": true_total_effect,
                        "estimated_total_effect": total_effect,
                        "runtime_seconds": runtime,
                    }
                )

                steps = np.array(model.loss_history["step"])
                ax.plot(steps, model.loss_history["score_loss"], label="score loss")
                ax.plot(steps, model.loss_history["query_loss"], label="query loss")
                ax.plot(steps, model.loss_history["acyclicity_loss"], label="acyclicity loss")
                ax.set_title(f"{setting_label} | {label}")
                ax.set_xlabel("inner iteration")
                ax.set_ylabel("loss")
                ax.set_yscale("symlog", linthresh=1e-8)
                ax.legend()

    fig.tight_layout()
    fig.savefig("cpdag_query_loss_curves.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_max_min_gaps(
    csv_path: str = "cpdag_query_results.csv",
    output_path: str = "cpdag_max_min_gaps.png",
) -> None:
    import matplotlib.pyplot as plt

    rows = []
    with Path(csv_path).open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["query_mode"] not in {"maximize", "minimize"}:
                continue
            rows.append(row)

    grouped = {}
    for row in rows:
        key = (row["graph_name"], row["query_label"], row["id_status"])
        grouped.setdefault(key, {"with CPDAG": {}, "without CPDAG": {}})
        grouped[key][row["cpdag_setting"]][row["query_mode"]] = float(row["estimated_total_effect"])

    labels = []
    gaps_with_cpdag = []
    gaps_without_cpdag = []
    for (graph_name, query_label, id_status), values in grouped.items():
        labels.append(f"{graph_name}\n{query_label}\n({id_status})")
        with_modes = values["with CPDAG"]
        without_modes = values["without CPDAG"]
        if "maximize" in with_modes and "minimize" in with_modes:
            gaps_with_cpdag.append(abs(with_modes["maximize"] - with_modes["minimize"]))
        else:
            gaps_with_cpdag.append(np.nan)
        if "maximize" in without_modes and "minimize" in without_modes:
            gaps_without_cpdag.append(abs(without_modes["maximize"] - without_modes["minimize"]))
        else:
            gaps_without_cpdag.append(np.nan)

    x = np.arange(len(labels))
    width = 0.38
    fig, ax = plt.subplots(figsize=(20, 6))
    ax.bar(x - width / 2, gaps_with_cpdag, width, label="with CPDAG")
    ax.bar(x + width / 2, gaps_without_cpdag, width, label="without CPDAG")
    for container in ax.containers:
        ax.bar_label(container) # Centers labels and uses white text for visibility
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("max-min total effect gap")
    ax.set_title("Max-Min Gaps by DAG Type and Query")
    ax.set_yscale("symlog", linthresh=1e-8)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def test(B_true: np.ndarray, cpdag: typing.Optional[np.ndarray] = None) -> None:
    experiment(B_true, cpdag, src=0, dst=min(2, B_true.shape[0] - 1))


if __name__ == "__main__":
    # Example DAG: X -> Y
    # CPDAG: X - Y
    B_two_node = np.array([
        [0, 1],
        [0, 0],
    ])
    CPDAG_two_node = np.array([
        [0, 1],
        [1, 0],
    ])

    # Example DAG / CPDAG: X -> Y <- Z
    B_collider = np.array([
        [0, 1, 0],
        [0, 0, 0],
        [0, 1, 0],
    ])
    CPDAG_collider = np.array([
        [0, 1, 0],
        [0, 0, 0],
        [0, 1, 0],
    ])

    # Example DAG: X -> Y -> Z
    # CPDAG: X - Y - Z
    B_chain = np.array([
        [0, 1, 0],
        [0, 0, 1],
        [0, 0, 0],
    ])
    CPDAG_chain = np.array([
        [0, 1, 0],
        [1, 0, 1],
        [0, 1, 0],
    ])

    # Example DAG / CPDAG: X   Y   Z
    B_three_disconnected = np.zeros((3, 3), dtype=int)
    CPDAG_three_disconnected = np.zeros((3, 3), dtype=int)

    # Node order: [X, Y, Z, W]
    # Example DAG / CPDAG: Z -> X -> Y, W -> X, W -> Y
    B_instrumental_variable = np.array([
        [0, 1, 0, 0],
        [0, 0, 0, 0],
        [1, 0, 0, 0],
        [1, 1, 0, 0],
    ])
    CPDAG_instrumental_variable = B_instrumental_variable.copy()

    # Node order: [X, Y, Z, W]
    # Example DAG / CPDAG: X -> Y <- Z, Y -> W
    B_descendant_of_collider = np.array([
        [0, 1, 0, 0],
        [0, 0, 0, 1],
        [0, 1, 0, 0],
        [0, 0, 0, 0],
    ])
    CPDAG_descendant_of_collider = B_descendant_of_collider.copy()

    # Node order: [X, Y, Z, W], with Z as the center.
    # Example DAG: Z -> X, Z -> Y, Z -> W
    # CPDAG: X - Z - Y and Z - W
    B_star = np.array([
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [1, 1, 0, 1],
        [0, 0, 0, 0],
    ])
    CPDAG_star = np.array([
        [0, 0, 1, 0],
        [0, 0, 1, 0],
        [1, 1, 0, 1],
        [0, 0, 1, 0],
    ])

    experiments = [
        ("TWO NODE", "non-ID", B_two_node, CPDAG_two_node, 0, 1),
        ("COLLIDER", "ID", B_collider, CPDAG_collider, 0, 1),
        ("CHAIN","non-ID", B_chain, CPDAG_chain, 0, 1),
        ("THREE DISCONNECTED", "ID", B_three_disconnected, CPDAG_three_disconnected, 0, 1),
        ("INSTRUMENTAL VARIABLE", "ID", B_instrumental_variable, CPDAG_instrumental_variable, 2, 3),
        ("INSTRUMENTAL VARIABLE", "ID", B_instrumental_variable, CPDAG_instrumental_variable, 2, 1),
        ("DESCENDANT OF COLLIDER", "ID", B_descendant_of_collider, CPDAG_descendant_of_collider, 0, 3),
        ("STAR", "non-ID", B_star, CPDAG_star, 0, 3),
        ("STAR", "non-ID", B_star, CPDAG_star, 2, 0),
    ]
    

    # for name, id_status, B_true, cpdag, src, dst in experiments:
    #     print(f"---{name}---")
    #     experiment(B_true, cpdag, src=src, dst=dst, graph_name=name, id_status=status)

    plot_max_min_gaps()
