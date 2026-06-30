"""
FunCast — Prévision fonctionnelle avec covariates.

Référence : Sezgin et al. (2025), "FunCast: a forecasting model for
functional data using covariates", Statistics and Computing.
"""

from funcast.model import FunCast
from funcast.basis import get_basis, bspline_basis, fourier_basis
from funcast.selection import select_h_rrss

__version__ = "0.1.0"
__all__ = [
    "FunCast",
    "get_basis",
    "bspline_basis",
    "fourier_basis",
    "select_h_rrss",
]
