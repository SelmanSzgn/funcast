"""
FunCast model implementation.
"""

import numpy as np
from scipy.linalg import lstsq
from sklearn.base import BaseEstimator

from funcast.basis import get_basis
from funcast.selection import select_h_rrss

if hasattr(np, "trapezoid"):
    _trapezoid = np.trapezoid  # type: ignore[attr-defined]
else:
    _trapezoid = np.trapz  # type: ignore[attr-defined]


class FunCast(BaseEstimator):
    """
    FunCast model.

    Parameters
    ----------
    K : int, optionnal
        Number of basis functions for the future of Y. Default is 10.
    s : float, optionnal
        Smoothing coefficient. Default is 0.5.
    basis_type : str
        Type of function basis, "bspline" or "fourier". Default is "bspline".
    auto_h : bool, optionnal
        If True, h_l is optimized with RRSS. Default is True.
    h_list : list, optionnal
        Values of h_l if auto_h is False. Default is None.
    degree : int, optionnal
        B-splines degree. Default is 3.
    rcond : float or None, optionnal
        Pseudo-inverse threshold. Default is None.
    """

    def __init__(
        self,
        K: int = 10,
        s: float = 0.5,
        basis_type: str = "bspline",
        auto_h: bool = True,
        h_list: list[int] | None = None,
        degree: int = 3,
        rcond: float | None = None,
    ):
        self.K = K
        self.s = s
        self.basis_type = basis_type
        self.auto_h = auto_h
        self.h_list = h_list
        self.degree = degree
        self.rcond = rcond

    def _compute_h(
        self,
        covariates_past: list[np.ndarray],
        t_past: np.ndarray,
    ) -> list[int]:
        """
        Compute h_l for each covariate.

        Parameters
        ----------
        covariates_past : list of array-like
            Past of covariates.
        t_past : array-like
            Past timestamps.

        Returns
        -------
        h_values : list
            Values of h_l.
        """
        h_values = []
        for ell, X in enumerate(covariates_past):
            if self.auto_h:
                h = select_h_rrss(
                    X,
                    t_past,
                    basis_type=self.basis_type,
                    degree=self.degree,
                )
            else:
                if self.h_list is None or len(self.h_list) <= ell:
                    raise ValueError(
                        "h_list required and of length p+1 if auto_h=False."
                    )
                h = max(self.h_list[ell], self.degree + 1)
            h_values.append(h)
        return h_values

    def _compute_q(self, h_values: list[int]) -> list[int]:
        """
        Compute q_l.

        Parameters
        ----------
        h_values : list
            Values of h_l.

        Returns
        -------
        list
            Values of q_l.
        """
        min_q = self.degree + 1
        return [max(min_q, round((1 - self.s) * h)) for h in h_values]

    def _project_covariates(
        self,
        covariates_past: list[np.ndarray],
        t_past: np.ndarray,
        h_values: list[int],
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """
        Project each covariate on theta basis.

        Parameters
        ----------
        covariates_past : list of array-like
            Past of covariates.
        t_past : array-like
            Past timestamps.
        h_values : list
            Values of h_l.

        Returns
        -------
        tuple
            Expansion coefficients of the covariates.
        """
        C_list, theta_list = [], []
        for X, h in zip(covariates_past, h_values):
            theta = get_basis(t_past, h, self.basis_type)
            thetaT_theta = theta.T @ theta
            C, _, _, _ = lstsq(thetaT_theta, (X @ theta).T, cond=self.rcond)
            C_list.append(C.T)
            theta_list.append(theta)
        return C_list, theta_list

    def _compute_J_matrices(
        self,
        theta_list: list[np.ndarray],
        t_past: np.ndarray,
        q_values: list[int],
    ) -> list[np.ndarray]:
        """
        Compute the inner product matrix.

        Parameters
        ----------
        theta_list : list of array-like
            Basis for covariates.
        t_past : array-like
            Past timestamps.
        q_values : list
            Values of q_l.

        Returns
        -------
        J_list : list
            Inner product matrix.
        """
        J_list = []
        for theta, q_ell in zip(theta_list, q_values):
            B_ell = get_basis(t_past, q_ell, self.basis_type)
            integrand = theta[:, :, np.newaxis] * B_ell[:, np.newaxis, :]
            J = _trapezoid(integrand, x=t_past, axis=0)
            J_list.append(np.asarray(J))
        return J_list

    def _build_design_matrix(
        self,
        C_list: list[np.ndarray],
        J_list: list[np.ndarray],
        t_future: np.ndarray,
        q_values: list[int],
    ) -> np.ndarray:
        """
        Build the design matrix.

        Parameters
        ----------
        C_list : list of array-like
            Expansion coefficients of the covariates.
        J_list : list
            Inner product matrix.
        t_future : array-like
            Future timestamps.
        q_values : list
            Values of q_l.

        Returns
        -------
        X_design : array-like
            Design matrix.
        """
        n = C_list[0].shape[0]
        m2 = len(t_future)
        K = self.K
        q_total = sum(q_values)
        psi = get_basis(t_future, K, self.basis_type)
        C_full = np.concatenate(C_list, axis=1)
        X_design = np.zeros((n * m2, K * q_total))
        for j, t_j in enumerate(t_future):
            psi_t = psi[j, :]
            xi_parts = []
            for ell, (J, q_ell) in enumerate(zip(J_list, q_values)):
                h_ell = J.shape[0]
                c_start = sum(C_list[e].shape[1] for e in range(ell))
                c_end = c_start + h_ell
                c_i_ell = C_full[:, c_start:c_end]
                cJ = c_i_ell @ J
                block = np.einsum("k,nq->nkq", psi_t, cJ).reshape(n, K * q_ell)
                xi_parts.append(block)
            xi_j = np.concatenate(xi_parts, axis=1)
            X_design[j::m2, :] = xi_j
        return X_design

    def fit(
        self,
        Y_past: np.ndarray,
        Y_future: np.ndarray,
        t_past: np.ndarray,
        t_future: np.ndarray,
        covariates_past: list[np.ndarray] | None = None,
    ) -> "FunCast":
        """
        Fit FunCast.

        Parameters
        ----------
        Y_past : array-like
            Past of Y.
        Y_future : array-like
            Future of Y.
        t_past : array-like
            Past timestamps.
        t_future : array-like
            Future timestamps.
        covariates_past : list of array-like
            Past of covariates.

        Returns
        -------
        self
        """
        all_covariates = [Y_past] + (covariates_past or [])

        self.h_values_ = self._compute_h(all_covariates, t_past)
        self.q_values_ = self._compute_q(self.h_values_)
        self.C_list_, self.theta_list_ = self._project_covariates(
            all_covariates, t_past, self.h_values_
        )
        self.J_list_ = self._compute_J_matrices(
            self.theta_list_, t_past, self.q_values_
        )

        X_design = self._build_design_matrix(
            self.C_list_, self.J_list_, t_future, self.q_values_
        )
        y = Y_future.ravel()
        self.b_hat_, _, _, _ = lstsq(X_design, y, cond=self.rcond)

        self.t_past_ = t_past
        self.t_future_ = t_future
        self.m2_ = Y_future.shape[1]
        self.n_covariates_ = len(all_covariates)

        return self

    def predict(
        self,
        Y_past_new: np.ndarray,
        covariates_past_new: list[np.ndarray] | None = None,
    ) -> np.ndarray:
        """
        Predict the future of Y.

        Parameters
        ----------
        Y_past_new : array-like
            Past of Y for inference.
        covariates_past_new : list of array-like
            Past of covariates for inference.

        Returns
        -------
        Y_pred : array-like
            Prediction of Y.
        """
        all_new = [Y_past_new] + (covariates_past_new or [])

        C_new_list = []
        for X_new, theta, h in zip(all_new, self.theta_list_, self.h_values_):
            thetaT_theta = theta.T @ theta
            C_new, _, _, _ = lstsq(thetaT_theta, (X_new @ theta).T, cond=self.rcond)
            C_new_list.append(np.asarray(C_new).T)  # ← np.asarray() clarifie le type

        X_new_design = self._build_design_matrix(
            C_new_list, self.J_list_, self.t_future_, self.q_values_
        )

        n_new = Y_past_new.shape[0]
        y_pred = X_new_design @ self.b_hat_
        return np.asarray(y_pred).reshape(n_new, self.m2_)

    def score(
        self,
        Y_past: np.ndarray,
        Y_future: np.ndarray,
        covariates_past: list[np.ndarray] | None = None,
        metric: str = "rmse",
    ) -> float:
        """
        Compute RMSE and SMAPE.

        Parameters
        ----------
        Y_past : array-like
            Past of Y.
        Y_future : array-like
            Future of Y.
        covariates_past : list of array-like or None, optionnal
            Past of covariates. Default is None.
        metric : string, optionnal
            Evaluation metric, "rmse" or "smape". Default is "rmse".

        Returns
        -------
        float
            Metric value.
        """
        Y_pred = self.predict(Y_past, covariates_past)

        if metric == "rmse":
            return float(np.sqrt(np.mean((Y_future - Y_pred) ** 2)))
        elif metric == "smape":
            denom = np.abs(Y_future) + np.abs(Y_pred)
            denom = np.where(denom == 0, 1e-8, denom)
            return float(100 * np.mean(np.abs(Y_future - Y_pred) / denom))
        else:
            raise ValueError(f"metric inconnu : '{metric}'. Choisir 'rmse' ou 'smape'.")
