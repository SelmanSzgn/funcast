"""
Tests for module funcast.selection
"""

import numpy as np
import pytest

from funcast.selection import select_h_rrss


@pytest.fixture
def synthetic_data():
    """Simple synthetic data for testing RRSS."""
    rng = np.random.default_rng(0)
    n, m1 = 30, 60
    t = np.linspace(0, 1, m1)
    X = np.array(
        [np.sin(2 * np.pi * t) + 0.1 * rng.standard_normal(m1) for _ in range(n)]
    )
    return X, t


class TestSelectHRrss:
    def test_returns_integer(self, synthetic_data):
        """The result must be an integer."""
        X, t = synthetic_data
        h = select_h_rrss(X, t)
        assert isinstance(h, int)

    def test_respects_minimum_constraint(self, synthetic_data):
        """h must be >= degree+1."""
        X, t = synthetic_data
        degree = 3
        h = select_h_rrss(X, t, degree=degree)
        assert h >= degree + 1

    def test_h_within_candidates(self, synthetic_data):
        """The returned h must belong to the valid candidates."""
        X, t = synthetic_data
        candidates = [5, 8, 10, 15]
        h = select_h_rrss(X, t, h_candidates=candidates)
        assert h in candidates

    def test_filters_invalid_candidates(self, synthetic_data):
        """Candidates < degree+1 must be silently filtered out."""
        X, t = synthetic_data
        candidates = [1, 2, 3, 6, 10]
        h = select_h_rrss(X, t, h_candidates=candidates, degree=3)
        assert h >= 4
        assert h in [6, 10]

    def test_all_invalid_candidates_warns(self, synthetic_data):
        """If all candidates are invalid, a warning must be raised."""
        X, t = synthetic_data
        with pytest.warns(UserWarning):
            h = select_h_rrss(X, t, h_candidates=[1, 2], degree=3)
        assert h == 4

    def test_fourier_basis(self, synthetic_data):
        """Also works with the Fourier basis."""
        X, t = synthetic_data
        h = select_h_rrss(X, t, basis_type="fourier")
        assert isinstance(h, int)
        assert h >= 1

    def test_smooth_signal_low_h(self):
        """A very smooth signal should favour a small h."""
        rng = np.random.default_rng(1)
        n, m1 = 40, 80
        t = np.linspace(0, 1, m1)
        X = np.array([1.0 + 0.1 * t + 0.01 * rng.standard_normal(m1) for _ in range(n)])
        h = select_h_rrss(X, t, h_candidates=list(range(4, 20)))
        assert h <= 12

    def test_single_candidate(self, synthetic_data):
        """With a single valid candidate, must return that candidate."""
        X, t = synthetic_data
        h = select_h_rrss(X, t, h_candidates=[8])
        assert h == 8
