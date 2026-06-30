"""
Tests pour le module funcast.basis.

On vérifie :
- Les dimensions des matrices retournées
- Les propriétés mathématiques des bases (partition de l'unité, orthogonalité)
- Les cas limites et les erreurs attendues
"""

import numpy as np
import pytest

from funcast.basis import bspline_basis, fourier_basis, get_basis

# ==============================================================================
# Fixtures partagées
# ==============================================================================


@pytest.fixture
def grid():
    """Grille temporelle standard pour les tests."""
    return np.linspace(0, 1, 100)


@pytest.fixture
def grid_irregular():
    """Grille temporelle irrégulière."""
    rng = np.random.default_rng(42)
    t = np.sort(rng.uniform(0, 1, 80))
    return t


# ==============================================================================
# Tests B-spline
# ==============================================================================


class TestBsplineBasis:
    def test_output_shape(self, grid):
        """La matrice retournée doit avoir la bonne forme."""
        B = bspline_basis(grid, n_basis=8)
        assert B.shape == (100, 8)

    def test_minimum_basis_enforced(self, grid):
        """n_basis est clipé à degree+1 si trop petit."""
        B = bspline_basis(grid, n_basis=2, degree=3)
        assert B.shape[1] == 4

    def test_partition_of_unity(self, grid):
        """Les B-splines forment une partition de l'unité : somme des lignes = 1."""
        B = bspline_basis(grid, n_basis=10)
        row_sums = B.sum(axis=1)
        np.testing.assert_allclose(row_sums, np.ones(100), atol=1e-10)

    def test_non_negative(self, grid):
        """Les B-splines sont toujours positives ou nulles."""
        B = bspline_basis(grid, n_basis=10)
        assert np.all(B >= -1e-12)

    def test_different_degrees(self, grid):
        """Fonctionne pour différents degrés."""
        for degree in [1, 2, 3, 4]:
            B = bspline_basis(grid, n_basis=8, degree=degree)
            assert B.shape == (100, 8)
            np.testing.assert_allclose(B.sum(axis=1), np.ones(100), atol=1e-10)

    def test_two_points_minimum(self):
        """Fonctionne avec deux points d'évaluation (minimum requis)."""
        t = np.array([0.0, 1.0])
        B = bspline_basis(t, n_basis=4)
        assert B.shape == (2, 4)
        np.testing.assert_allclose(B.sum(axis=1), np.ones(2), atol=1e-10)

    def test_single_point_not_supported(self):
        """Un seul point d'évaluation n'est pas supporté par les B-splines scipy."""
        t = np.array([0.5])
        with pytest.raises(ValueError):
            bspline_basis(t, n_basis=4)

    def test_irregular_grid(self, grid_irregular):
        """Fonctionne sur une grille irrégulière."""
        B = bspline_basis(grid_irregular, n_basis=8)
        assert B.shape == (80, 8)
        np.testing.assert_allclose(B.sum(axis=1), np.ones(80), atol=1e-10)

    def test_large_n_basis(self, grid):
        """Fonctionne avec un grand nombre de bases."""
        B = bspline_basis(grid, n_basis=50)
        assert B.shape == (100, 50)
        np.testing.assert_allclose(B.sum(axis=1), np.ones(100), atol=1e-10)


# ==============================================================================
# Tests Fourier
# ==============================================================================


class TestFourierBasis:
    def test_output_shape(self, grid):
        """La matrice retournée doit avoir la bonne forme."""
        B = fourier_basis(grid, n_basis=7)
        assert B.shape == (100, 7)

    def test_first_column_is_ones(self, grid):
        """La première colonne doit être constante égale à 1."""
        B = fourier_basis(grid, n_basis=5)
        np.testing.assert_allclose(B[:, 0], np.ones(100), atol=1e-12)

    def test_cosine_columns(self, grid):
        """Les colonnes impaires (k=1,3,...) doivent être des cosinus."""
        B = fourier_basis(grid, n_basis=5)
        T = grid.max() - grid.min()
        expected_col1 = np.cos(2 * np.pi * 1 * grid / T)
        np.testing.assert_allclose(B[:, 1], expected_col1, atol=1e-12)

    def test_sine_columns(self, grid):
        """Les colonnes paires (k=2,4,...) doivent être des sinus."""
        B = fourier_basis(grid, n_basis=5)
        T = grid.max() - grid.min()
        expected_col2 = np.sin(2 * np.pi * 1 * grid / T)
        np.testing.assert_allclose(B[:, 2], expected_col2, atol=1e-12)

    def test_n_basis_one(self, grid):
        """Avec n_basis=1, retourne uniquement la constante."""
        B = fourier_basis(grid, n_basis=1)
        assert B.shape == (100, 1)
        np.testing.assert_allclose(B[:, 0], np.ones(100), atol=1e-12)

    def test_values_bounded(self, grid):
        """Les valeurs doivent être dans [-1, 1]."""
        B = fourier_basis(grid, n_basis=11)
        assert np.all(B >= -1.0 - 1e-12)
        assert np.all(B <= 1.0 + 1e-12)


# ==============================================================================
# Tests get_basis (sélecteur)
# ==============================================================================


class TestGetBasis:
    def test_bspline_dispatch(self, grid):
        """get_basis avec 'bspline' doit retourner la même chose que bspline_basis."""
        B1 = get_basis(grid, n_basis=8, basis_type="bspline")
        B2 = bspline_basis(grid, n_basis=8)
        np.testing.assert_array_equal(B1, B2)

    def test_fourier_dispatch(self, grid):
        """get_basis avec 'fourier' doit retourner la même chose que fourier_basis."""
        B1 = get_basis(grid, n_basis=7, basis_type="fourier")
        B2 = fourier_basis(grid, n_basis=7)
        np.testing.assert_array_equal(B1, B2)

    def test_unknown_basis_raises(self, grid):
        """Un basis_type inconnu doit lever une ValueError."""
        with pytest.raises(ValueError, match="basis_type inconnu"):
            get_basis(grid, n_basis=8, basis_type="wavelet")

    def test_default_is_bspline(self, grid):
        """Le type par défaut doit être 'bspline'."""
        B1 = get_basis(grid, n_basis=8)
        B2 = bspline_basis(grid, n_basis=8)
        np.testing.assert_array_equal(B1, B2)
