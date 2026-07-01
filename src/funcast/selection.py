"""
Automatic selection of h_l with RRSS criterion.
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
    Selects h_l with RRSS criterion.

    Parameters
    ----------
    X : array-like
        Covariate realization.
    t : array-like
        Temporal grid.
    h_candidates : list or None, optionnal
        Candidates for h_l values. Default is None.
    basis_type : str
        Type of function basis. Default is "bspline"
    degree : int
        Spline degrees. Default is 3.

    Returns
    -------
    h_opt : int
        h_l value that minimizes RRSS.
    """
    n, m1 = X.shape
    min_h = degree + 1

    if h_candidates is None:
        max_h = max(min_h + 1, min(m1 // 2, 30))
        h_candidates = list(range(min_h, max_h + 1))
    else:
        h_candidates = [h for h in h_candidates if h >= min_h]

    if not h_candidates:
        warnings.warn(f"No valid candidate for h_l. Set h={min_h}.")
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
