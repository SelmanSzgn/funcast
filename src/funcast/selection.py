"""
Sélection automatique de hℓ via le critère RRSS.

Référence : Sezgin et al. (2025), Section 5, Step 2, Eq. 10.
"""

import warnings

import numpy as np
from scipy.linalg import lstsq

from funcast.basis import get_basis


def select_h_rrss(
    X: np.ndarray,
    t: np.ndarray,
    h_candidates: list[int] | None = None,
    basis_type: str = "bspline",
    degree: int = 3,
) -> int:
    """
    Sélectionne hℓ via le critère RRSS (Eq. 10) :

        RRSS(hℓ) = sqrt( sum_i sum_j (X_ij - X̂_ij)² / (n * (m1 - hℓ)) )

    Paramètres
    ----------
    X            : array (n, m1) — réalisations du covariate
    t            : array (m1,)   — grille temporelle
    h_candidates : liste de valeurs candidates pour hℓ
    basis_type   : str
    degree       : int — degré des B-splines (pour la contrainte min)

    Retourne
    --------
    h_opt : int — valeur de hℓ minimisant le critère RRSS
    """
    n, m1 = X.shape
    min_h = degree + 1

    if h_candidates is None:
        max_h = max(min_h + 1, min(m1 // 2, 30))
        h_candidates = list(range(min_h, max_h + 1))
    else:
        h_candidates = [h for h in h_candidates if h >= min_h]

    if not h_candidates:
        warnings.warn(
            f"Aucun candidat valide pour hℓ (min requis : {min_h}). "
            f"Utilisation de h={min_h}."
        )
        return min_h

    best_h, best_rrss = h_candidates[0], np.inf

    for h in h_candidates:
        denom = m1 - h
        if denom <= 0:
            continue
        try:
            theta = get_basis(t, h, basis_type)
            thetaT_theta = theta.T @ theta
            C, _, _, _ = lstsq(thetaT_theta, (X @ theta).T)
            X_hat = (theta @ C).T
            residuals = X - X_hat
            rrss = np.sqrt(np.sum(residuals**2) / (n * denom))
            if rrss < best_rrss:
                best_rrss = rrss
                best_h = h
        except np.linalg.LinAlgError:
            continue

    return best_h
