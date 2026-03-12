import typing

import numpy as np
import scipy.linalg as sla
from scipy.special import expit as sigmoid
from tqdm.auto import tqdm

from linear import DagmaLinear


__all__ = ["QueryDagmaLinear"]


class QueryDagmaLinear(DagmaLinear):
    """
    DAGMA linear model with an additional objective term on a selected total effect.

    The base observational objective is unchanged. This subclass adds an optional
    penalty or reward on the total causal effect from one node to another, where
    the total effect is defined by the linear SEM implied by ``W``.
    """

    def __init__(self, loss_type: str, verbose: bool = False, dtype: type = np.float64) -> None:
        super().__init__(loss_type=loss_type, verbose=verbose, dtype=dtype)
        self.effect_src: typing.Optional[int] = None
        self.effect_dst: typing.Optional[int] = None
        self.effect_mode: typing.Optional[str] = None


    @staticmethod
    def total_effect(W: np.ndarray, src: int, dst: int) -> float:
        """
        Compute the total causal effect from ``src`` to ``dst``.

        Parameters
        ----------
        W : np.ndarray
            Weighted adjacency matrix with the repository convention
            ``W[i, j]`` for the edge ``i -> j``.
        src : int
            Index of the intervention/source node.
        dst : int
            Index of the outcome/target node.

        Returns
        -------
        float
            The ``(src, dst)`` entry of ``(I - W)^{-1}``, which is the total
            causal effect in the row-vector SEM convention used in this package.
        """
        A = np.eye(W.shape[0], dtype=W.dtype) - W
        return np.linalg.inv(A)[src, dst]

    
    @staticmethod
    def total_effect_and_grad(
        W: np.ndarray, src: int, dst: int
    ) -> typing.Tuple[float, np.ndarray]:
        """
        Compute the total effect and its gradient with respect to ``W``.

        Parameters
        ----------
        W : np.ndarray
            Weighted adjacency matrix with the repository convention
            ``W[i, j]`` for the edge ``i -> j``.
        src : int
            Index of the intervention/source node.
        dst : int
            Index of the outcome/target node.

        Returns
        -------
        typing.Tuple[float, np.ndarray]
            A pair ``(tau, G_tau)`` where ``tau`` is the total effect from
            ``src`` to ``dst`` and ``G_tau`` is its gradient with respect to
            ``W``.
        """
        A = np.eye(W.shape[0], dtype=W.dtype) - W
        R = np.linalg.inv(A)
        tau = R[src, dst]
        grad = np.outer(R[src, :], R[:, dst])
        return tau, grad

    def _query(self, W: np.ndarray) -> typing.Tuple[float, np.ndarray, float]:
        """
        Evaluate the configured total-effect query term.

        Parameters
        ----------
        W : np.ndarray
            Current weighted adjacency matrix.

        Returns
        -------
        typing.Tuple[float, np.ndarray, float]
            A triple ``(value, grad, tau)`` containing the scalar query term to
            add to the objective, its gradient with respect to ``W``, and the
            current total effect value itself.

        Notes
        -----
        Supported modes are:

        - ``"maximize"``: maximize the signed total effect.
        - ``"minimize"``: minimize the signed total effect.
        - ``"maximize_squared"``: maximize the squared total effect magnitude.
        """
        if self.effect_src is None or self.effect_dst is None:
            return 0.0, np.zeros_like(W), 0.0

        tau, G_tau = self.total_effect_and_grad(W, self.effect_src, self.effect_dst)
        mode = self.effect_mode

        if mode == "maximize":
            value = -tau
            grad = -G_tau
        elif mode == "minimize":
            value = tau
            grad = G_tau
        elif mode == "maximize_squared":
            value = -tau * tau
            grad = -2.0 * tau * G_tau
        else:
            raise ValueError(
                "effect_mode should be one of {'maximize', 'minimize', 'maximize_squared'}"
            )

        return value, grad, tau

    def _func(
        self, W: np.ndarray, mu: float, gamma: float, s: float = 1.0
    ) -> typing.Tuple[float, float, float, float]:
        """
        Evaluate the penalized objective, including the optional query term.

        Parameters
        ----------
        W : np.ndarray
            Current weighted adjacency matrix.
        mu : float
            Weight applied to the observational score and L1 penalty, matching
            the base DAGMA objective.
        gamma : float
            Weight applied to the query term for the current outer iteration.
        s : float, optional
            M-matrix domain parameter for the acyclicity term.

        Returns
        -------
        typing.Tuple[float, float, float, float]
            The total objective value, observational score, acyclicity value,
            and unscaled query-term contribution.
        """
        score, _ = self._score(W)
        h, _ = self._h(W, s)
        query, _, _ = self._query(W)
        obj = mu * (score + self.lambda1 * np.abs(W).sum()) + gamma * query + h
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
        Store checkpointed objective components for later inspection or plotting.
        """
        _, score, h, query = self._func(W, mu, gamma, s)
        score_loss = mu * (score + self.lambda1 * np.abs(W).sum())
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
        Minimize the query-augmented DAGMA objective by Adam updates.

        This override keeps the base optimizer structure but adds the gradient of
        the total-effect term to the objective gradient at each iteration.

        Parameters
        ----------
        W : np.ndarray
            Initial point for optimization.
        mu : float
            Weight applied to the observational score and L1 penalty.
        gamma : float
            Weight applied to the query term for the current outer iteration.
        max_iter : int
            Maximum number of gradient iterations.
        s : float
            M-matrix domain parameter for the acyclicity term.
        lr : float
            Learning rate.
        outer_iter : int
            Current outer DAGMA iteration index, used for history tracking.
        tol : float, optional
            Relative objective tolerance for early stopping.
        beta_1 : float, optional
            Adam first-moment parameter.
        beta_2 : float, optional
            Adam second-moment parameter.
        pbar : tqdm, optional
            Progress bar object from the outer training loop.

        Returns
        -------
        typing.Tuple[np.ndarray, bool]
            The optimized weighted adjacency matrix and a success flag.
        """
        obj_prev = 1e16
        self.opt_m, self.opt_v = 0, 0
        self.vprint(
            f"\n\nMinimize with -- mu:{mu} -- gamma:{gamma} -- lr: {lr} -- s: {s} -- l1: {self.lambda1} for {max_iter} max iterations"
        )
        mask_inc = np.zeros((self.d, self.d))
        if self.inc_c is not None:
            mask_inc[self.inc_r, self.inc_c] = -2 * mu * self.lambda1
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
                + mu * self.lambda1 * np.sign(W)
                + 2 * W * M.T
                + mask_inc * np.sign(W)
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
    ) -> np.ndarray:
        r"""
        Runs DAGMA on observational data while optionally shaping a total effect.

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
            Tuple of edges that should be excluded from the DAG solution, e.g., ``((1,3), (2,4), (5,1))``. Defaults to None.
        include_edges : typing.Optional[typing.List[typing.Tuple[int, int]]], optional
            Tuple of edges that should be included from the DAG solution, e.g., ``((1,3), (2,4), (5,1))``. Defaults to None.
        effect_src : typing.Optional[int], optional
            Source node index for the total-effect query. Defaults to None.
        effect_dst : typing.Optional[int], optional
            Target node index for the total-effect query. Defaults to None.
        effect_mode : typing.Optional[str], optional
            Query objective mode. One of ``"maximize"``, ``"minimize"``, or ``"maximize_squared"``.
            Required when total-effect shaping is enabled. Defaults to None.

        Returns
        -------
        np.ndarray
            Estimated DAG from data.


        .. important::

            If the output of :py:meth:`~dagma.linear_query.QueryDagmaLinear.fit` is not a DAG, then the user should try larger values
            of ``T`` (e.g., 6, 7, or 8) before raising an issue in github.

        .. warning::

            While DAGMA ensures to exclude the edges given in ``exclude_edges``, the current implementation does not guarantee that all edges
            in ``include_edges`` will be part of the final DAG.
        """
        self.effect_src = effect_src
        self.effect_dst = effect_dst
        self.effect_mode = effect_mode

        if gamma_warmup < 0:
            raise ValueError("gamma_warmup should be nonnegative")
        if gamma_init < 0.0:
            raise ValueError("gamma_init should be nonnegative")
        if gamma_factor < 0.0:
            raise ValueError("gamma_factor should be nonnegative")
        query_enabled = gamma_init != 0.0 and gamma_warmup < int(T)
        if query_enabled:
            if effect_src is None or effect_dst is None:
                raise ValueError("effect_src and effect_dst are required when total-effect shaping is enabled")
            if effect_mode is None:
                raise ValueError("effect_mode is required when total-effect shaping is enabled")
        
        self.X, self.lambda1, self.checkpoint = X, lambda1, checkpoint
        self.n, self.d = X.shape
        self.Id = np.eye(self.d).astype(self.dtype)

        if self.loss_type == 'l2':
            self.X -= X.mean(axis=0, keepdims=True)

        self.exc_r, self.exc_c = None, None
        self.inc_r, self.inc_c = None, None

        if exclude_edges is not None:
            if type(exclude_edges) is tuple and type(exclude_edges[0]) is tuple and np.all(np.array([len(e) for e in exclude_edges]) == 2):
                self.exc_r, self.exc_c = zip(*exclude_edges)
            else:
                ValueError("blacklist should be a tuple of edges, e.g., ((1,2), (2,3))")

        if include_edges is not None:
            if type(include_edges) is tuple and type(include_edges[0]) is tuple and np.all(np.array([len(e) for e in include_edges]) == 2):
                self.inc_r, self.inc_c = zip(*include_edges)
            else:
                ValueError("whitelist should be a tuple of edges, e.g., ((1,2), (2,3))")

        self.cov = X.T @ X / float(self.n)
        self.W_est = np.zeros((self.d, self.d)).astype(self.dtype)
        self.loss_history = {
            "outer_iter": [],
            "inner_iter": [],
            "step": [],
            "score_loss": [],
            "query_loss": [],
            "acyclicity_loss": [],
        }
        self.loss_step_offset = 0
        mu = mu_init
        if type(s) == list:
            if len(s) < T:
                self.vprint(f"Length of s is {len(s)}, using last value in s for iteration t >= {len(s)}")
                s = s + (T - len(s)) * [s[-1]]
        elif type(s) in [int, float]:
            s = T * [s]
        else:
            ValueError("s should be a list, int, or float.")

        with tqdm(total=(T-1)*warm_iter+max_iter) as pbar:
            for i in range(int(T)):
                self.vprint(f'\nIteration -- {i+1}:')
                gamma = 0.0 if i < gamma_warmup else gamma_init * (gamma_factor ** (i - gamma_warmup))
                lr_adam, success = lr, False
                inner_iters = int(max_iter) if i == T - 1 else int(warm_iter)
                while success is False:
                    W_temp, success = self.minimize(
                        self.W_est.copy(),
                        mu,
                        gamma,
                        inner_iters,
                        s[i],
                        lr=lr_adam,
                        outer_iter=i + 1,
                        beta_1=beta_1,
                        beta_2=beta_2,
                        pbar=pbar,
                    )
                    if success is False:
                        self.vprint(f'Retrying with larger s')
                        lr_adam *= 0.5
                        s[i] += 0.1
                self.W_est = W_temp
                self.loss_step_offset += inner_iters
                mu *= mu_factor

        self.h_final, _ = self._h(self.W_est)
        self.score_final, _ = self._score(self.W_est)
        self.query_final, _, self.effect_final = self._query(self.W_est)
        self.W_est[np.abs(self.W_est) < w_threshold] = 0
        return self.W_est
    


def test(B_true):
    import utils
    import matplotlib.pyplot as plt
    from timeit import default_timer as timer
    utils.set_random_seed(1)
    
    n, d = 5000, 3
    sem_type = 'gauss'

    # Chain DAG: X -> Y -> Z, with node order [X, Y, Z].
    W_true = utils.simulate_parameter(B_true)
    X = utils.simulate_linear_sem(W_true, n, sem_type)

    queries = [
        ("baseline X->Z", 0, 2, None),
        ("maximize X->Z", 0, 2, "maximize"),
        ("minimize X->Z", 0, 2, "minimize"),
        ("baseline X->Y", 0, 1, None),
        ("maximize X->Y", 0, 1, "maximize"),
        ("minimize X->Y", 0, 1, "minimize"),
    ]
    fig, axes = plt.subplots(3, 2, figsize=(12, 12), sharex=False)
    axes = axes.ravel()

    for ax, (label, src, dst, mode) in zip(axes, queries):
        model = QueryDagmaLinear(loss_type='l2')
        start = timer()
        fit_kwargs = dict(
            X=X,
            lambda1=0.02,
        )
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
        print(label)
        print(f'is_dag: {is_dag}')
        print(f'final h: {model.h_final:.4e}')
        if is_dag:
            acc = utils.count_accuracy(B_true, W_est != 0)
            print(acc)
        else:
            print('accuracy skipped because learned W is not a DAG')
        print("learned W:")
        print(W_est)
        print(f'true total effect: {true_total_effect:.4f}')
        print(f'computed total effect: {total_effect:.4f}')
        print(f'time: {end-start:.4f}s')
        print()

        steps = np.array(model.loss_history["step"])
        ax.plot(steps, model.loss_history["score_loss"], label="score loss")
        ax.plot(steps, model.loss_history["query_loss"], label="query loss")
        ax.plot(steps, model.loss_history["acyclicity_loss"], label="acyclicity loss")
        ax.set_title(label)
        ax.set_xlabel("inner iteration")
        ax.set_ylabel("loss")
        ax.set_yscale("symlog", linthresh=1e-8)
        ax.legend()

    fig.tight_layout()
    fig.savefig("query_loss_curves.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    
  
if __name__ == '__main__':

    B_collider = np.array([
        [0, 1, 0],
        [0, 0, 0],
        [0, 1, 0],
    ])
    B_chain = np.array([
        [0, 0, 0],
        [1, 0, 1],
        [0, 0, 0],
    ])
    print("---COLLIDER---")
    test(B_collider)

    print("---CHAIN---")
    test(B_chain)

