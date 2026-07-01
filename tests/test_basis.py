"""
Tests for module funcast.basis
"""

import numpy as np
import pytest

from funcast.basis import bspline_basis, fourier_basis, get_basis


@pytest.fixture
def grid():
    return np.linspace(0, 1, 100)


@pytest.fixture
def grid_irregular():
    rng = np.random.default_rng(42)
    t = np.sort(rng.uniform(0, 1, 80))
    return t


class TestBsplineBasis:
    def test_output_shape(self, grid):
        """Shape of the returned matrix."""
        B = bspline_basis(grid, n_basis=8)
        assert B.shape == (100, 8)

    def test_minimum_basis_enforced(self, grid):
        """n_basis is clipped to degree+1 if too small."""
        B = bspline_basis(grid, n_basis=2, degree=3)
        assert B.shape[1] == 4

    def test_partition_of_unity(self, grid):
        """B-splines form a partition of unity: row sums equal 1."""
        B = bspline_basis(grid, n_basis=10)
        row_sums = B.sum(axis=1)
        np.testing.assert_allclose(row_sums, np.ones(100), atol=1e-10)

    def test_non_negative(self, grid):
        """B-splines are always non-negative."""
        B = bspline_basis(grid, n_basis=10)
        assert np.all(B >= -1e-12)

    def test_different_degrees(self, grid):
        """Works for different degrees."""
        for degree in [1, 2, 3, 4]:
            B = bspline_basis(grid, n_basis=8, degree=degree)
            assert B.shape == (100, 8)
            np.testing.assert_allclose(B.sum(axis=1), np.ones(100), atol=1e-10)

    def test_two_points_minimum(self):
        """Works with two evaluation points (minimum required)."""
        t = np.array([0.0, 1.0])
        B = bspline_basis(t, n_basis=4)
        assert B.shape == (2, 4)
        np.testing.assert_allclose(B.sum(axis=1), np.ones(2), atol=1e-10)

    def test_single_point_not_supported(self):
        """A single evaluation point is not supported by scipy B-splines."""
        t = np.array([0.5])
        with pytest.raises(ValueError):
            bspline_basis(t, n_basis=4)

    def test_irregular_grid(self, grid_irregular):
        """Works on an irregular grid."""
        B = bspline_basis(grid_irregular, n_basis=8)
        assert B.shape == (80, 8)
        np.testing.assert_allclose(B.sum(axis=1), np.ones(80), atol=1e-10)

    def test_large_n_basis(self, grid):
        """Works with a large number of basis functions."""
        B = bspline_basis(grid, n_basis=50)
        assert B.shape == (100, 50)
        np.testing.assert_allclose(B.sum(axis=1), np.ones(100), atol=1e-10)


class TestFourierBasis:
    def test_output_shape(self, grid):
        """The returned matrix must have the correct shape."""
        B = fourier_basis(grid, n_basis=7)
        assert B.shape == (100, 7)

    def test_first_column_is_ones(self, grid):
        """The first column must be constant and equal to 1."""
        B = fourier_basis(grid, n_basis=5)
        np.testing.assert_allclose(B[:, 0], np.ones(100), atol=1e-12)

    def test_cosine_columns(self, grid):
        """Odd columns (k=1,3,...) must be cosines."""
        B = fourier_basis(grid, n_basis=5)
        T = grid.max() - grid.min()
        expected_col1 = np.cos(2 * np.pi * 1 * grid / T)
        np.testing.assert_allclose(B[:, 1], expected_col1, atol=1e-12)

    def test_sine_columns(self, grid):
        """Even columns (k=2,4,...) must be sines."""
        B = fourier_basis(grid, n_basis=5)
        T = grid.max() - grid.min()
        expected_col2 = np.sin(2 * np.pi * 1 * grid / T)
        np.testing.assert_allclose(B[:, 2], expected_col2, atol=1e-12)

    def test_n_basis_one(self, grid):
        """With n_basis=1, returns only the constant term."""
        B = fourier_basis(grid, n_basis=1)
        assert B.shape == (100, 1)
        np.testing.assert_allclose(B[:, 0], np.ones(100), atol=1e-12)

    def test_values_bounded(self, grid):
        """Values must be in the range [-1, 1]."""
        B = fourier_basis(grid, n_basis=11)
        assert np.all(B >= -1.0 - 1e-12)
        assert np.all(B <= 1.0 + 1e-12)


class TestGetBasis:
    def test_bspline_dispatch(self, grid):
        """get_basis with 'bspline' must return the same result as bspline_basis."""
        B1 = get_basis(grid, n_basis=8, basis_type="bspline")
        B2 = bspline_basis(grid, n_basis=8)
        np.testing.assert_array_equal(B1, B2)

    def test_fourier_dispatch(self, grid):
        """get_basis with 'fourier' must return the same result as fourier_basis."""
        B1 = get_basis(grid, n_basis=7, basis_type="fourier")
        B2 = fourier_basis(grid, n_basis=7)
        np.testing.assert_array_equal(B1, B2)

    def test_unknown_basis_raises(self, grid):
        """An unknown basis_type must raise a ValueError."""
        with pytest.raises(ValueError, match="unknown basis_type"):
            get_basis(grid, n_basis=8, basis_type="wavelet")

    def test_default_is_bspline(self, grid):
        """The default type must be 'bspline'."""
        B1 = get_basis(grid, n_basis=8)
        B2 = bspline_basis(grid, n_basis=8)
        np.testing.assert_array_equal(B1, B2)
