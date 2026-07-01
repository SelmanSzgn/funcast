"""
Functional basis for FunCast.
"""

import numpy as np
from scipy.interpolate import BSpline


def bspline_basis(t: np.ndarray, n_basis: int, degree: int = 3) -> np.ndarray:
    """
    Create a cubic B-spline basis matrix.

    Parameters
    ----------
    t : array-like
        Evaluation points.
    n_basis : int
        Number of basis functions.
    degree : int
        Spline degrees. Default is 3.

    Returns
    -------
    B : array-like
        Basis matrix.
    """
    n_basis = max(n_basis, degree + 1)

    t_min, t_max = t.min(), t.max()
    n_inner = n_basis - degree - 1

    if n_inner > 0:
        inner_knots = np.linspace(t_min, t_max, n_inner + 2)[1:-1]
    else:
        inner_knots = np.array([])

    knots = np.concatenate(
        [np.repeat(t_min, degree + 1), inner_knots, np.repeat(t_max, degree + 1)]
    )

    B = np.zeros((len(t), n_basis))
    for j in range(n_basis):
        coef = np.zeros(n_basis)
        coef[j] = 1.0
        spl = BSpline(knots, coef, degree)
        B[:, j] = spl(t)
    return B


def fourier_basis(t: np.ndarray, n_basis: int) -> np.ndarray:
    """
    Create a Fourier basis.

    Parameters
    ----------
    t : array-like
        Evaluation points.
    n_basis : int
        Number of basis functions.

    Returns
    -------
    B : array-like
        Basis matrix.
    """
    T = t.max() - t.min()
    B = np.ones((len(t), n_basis))
    for k in range(1, n_basis):
        freq = (k + 1) // 2
        if k % 2 == 1:
            B[:, k] = np.cos(2 * np.pi * freq * t / T)
        else:
            B[:, k] = np.sin(2 * np.pi * freq * t / T)
    return B


def get_basis(t: np.ndarray, n_basis: int, basis_type: str = "bspline") -> np.ndarray:
    """
    Create a basis of functions.

    Parameters
    ----------
    t : array-like
        Evaluation points.
    n_basis : int
        Number of basis functions.
    basis_type : str
        Type of function basis, "bspline" or "fourier". Default is "bspline".

    Returns
    -------
    B : array-like
        Basis matrix.
    """
    if basis_type == "bspline":
        return bspline_basis(t, n_basis)
    elif basis_type == "fourier":
        return fourier_basis(t, n_basis)
    else:
        raise ValueError(
            f"unknown basis_type : '{basis_type}'. Choose 'bspline' or 'fourier'."
        )
