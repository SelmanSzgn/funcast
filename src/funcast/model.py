"""
Modèle FunCast — Prévision fonctionnelle avec covariates.

Référence : Sezgin et al. (2025), "FunCast: a forecasting model for
functional data using covariates", Statistics and Computing.
"""

import numpy as np
from scipy.linalg import lstsq
from sklearn.base import BaseEstimator

from funcast.basis import get_basis
from funcast.selection import select_h_rrss


class FunCast(BaseEstimator):
    """
    Modèle FunCast — Prévision fonctionnelle avec covariates.

    Paramètres
    ----------
    K          : int   — nb de fonctions de base ψ pour le futur de Y (Step 1)
    s          : float — coefficient de lissage ∈ [0,1] ; qℓ = (1-s)*hℓ (Step 3)
    basis_type : str   — type de base ('bspline' ou 'fourier')
    auto_h     : bool  — si True, hℓ sélectionné automatiquement via RRSS (Step 2)
    h_list     : list  — valeurs de hℓ imposées si auto_h=False (longueur = p+1)
    degree     : int   — degré des B-splines (défaut : 3 = cubique)
    rcond      : float — seuil pour la pseudo-inverse (None = auto)
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
        """Calcule hℓ pour chaque covariate. Section 5, Step 2."""
        h_values = []
        for ell, X in enumerate(covariates_past):
            if self.auto_h:
                h = select_h_rrss(
                    X, t_past,
                    basis_type=self.basis_type,
                    degree=self.degree,
                )
            else:
                if self.h_list is None or len(self.h_list) <= ell:
                    raise ValueError(
                        "h_list doit être fourni et de longueur p+1 si auto_h=False."
                    )
                h = max(self.h_list[ell], self.degree + 1)
            h_values.append(h)
        return h_values

    def _compute_q(self, h_values: list[int]) -> list[int]:
        """Calcule qℓ = max(degree+1, round((1-s)*hℓ)). Section 5, Step 3."""
        min_q = self.degree + 1
        return [max(min_q, round((1 - self.s) * h)) for h in h_values]

    def _project_covariates(
        self,
        covariates_past: list[np.ndarray],
        t_past: np.ndarray,
        h_values: list[int],
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """Projette chaque covariate Xi,ℓ sur sa base θℓ (Assumption 2)."""
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
        Calcule les matrices d'inner product Jθ,B,ℓ (Proof of Theorem 1) :

            Jθ,B,ℓ = ∫₀ᵀ θℓ(u)ᵀ Bℓ(u) du ∈ R^{hℓ × qℓ}

        Approximée par la règle des trapèzes via np.trapezoid (numpy ≥ 2.0)
        avec repli sur np.trapz.
        """
        trapezoid = getattr(np, "trapezoid", np.trapz)
        J_list = []
        for theta, q_ell in zip(theta_list, q_values):
            B_ell = get_basis(t_past, q_ell, self.basis_type)
            integrand = theta[:, :, np.newaxis] * B_ell[:, np.newaxis, :]
            J = trapezoid(integrand, x=t_past, axis=0)
            J_list.append(J)
        return J_list

    def _build_design_matrix(
        self,
        C_list: list[np.ndarray],
        J_list: list[np.ndarray],
        t_future: np.ndarray,
        q_values: list[int],
    ) -> np.ndarray:
        """Construit la matrice de design X ∈ R^{n*m2 × K*q} (Theorem 1, Eq. 8)."""
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
                block = np.einsum('k,nq->nkq', psi_t, cJ).reshape(n, K * q_ell)
                xi_parts.append(block)

            xi_j = np.concatenate(xi_parts, axis=1)

            # La ligne i*m2 + j correspond à l'observation i au temps j,
            # cohérent avec y = Y_future.ravel() (ordre C row-major).
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
        Entraîne FunCast (Section 5 + Theorem 2).

        Paramètres
        ----------
        Y_past          : array (n, m1)   — passé de Y sur [0, T]
        Y_future        : array (n, m2)   — futur de Y sur [T, T+H]
        t_past          : array (m1,)     — grille temporelle [0, T]
        t_future        : array (m2,)     — grille temporelle [T, T+H]
        covariates_past : list de (n, m1) — covariates optionnels

        Retourne
        --------
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
        Prédit le futur de Y (Proposition 1).

        Paramètres
        ----------
        Y_past_new          : array (n_new, m1)
        covariates_past_new : list de (n_new, m1)

        Retourne
        --------
        Y_pred : array (n_new, m2)
        """
        all_new = [Y_past_new] + (covariates_past_new or [])

        C_new_list = []
        for X_new, theta, h in zip(all_new, self.theta_list_, self.h_values_):
            thetaT_theta = theta.T @ theta
            C_new, _, _, _ = lstsq(
                thetaT_theta, (X_new @ theta).T, cond=self.rcond
            )
            C_new_list.append(C_new.T)

        X_new_design = self._build_design_matrix(
            C_new_list, self.J_list_, self.t_future_, self.q_values_
        )

        n_new = Y_past_new.shape[0]
        y_pred = X_new_design @ self.b_hat_
        return y_pred.reshape(n_new, self.m2_)

    def score(
        self,
        Y_past: np.ndarray,
        Y_future: np.ndarray,
        covariates_past: list[np.ndarray] | None = None,
        metric: str = "rmse",
    ) -> float:
        """
        Calcule RMSE ou SMAPE (Section 6.3).

        Paramètres
        ----------
        metric : 'rmse' ou 'smape'

        Retourne
        --------
        score : float
        """
        Y_pred = self.predict(Y_past, covariates_past)

        if metric == "rmse":
            return float(np.sqrt(np.mean((Y_future - Y_pred) ** 2)))
        elif metric == "smape":
            denom = np.abs(Y_future) + np.abs(Y_pred)
            denom = np.where(denom == 0, 1e-8, denom)
            return float(100 * np.mean(np.abs(Y_future - Y_pred) / denom))
        else:
            raise ValueError(
                f"metric inconnu : '{metric}'. Choisir 'rmse' ou 'smape'."
            )
