"""
Tests pour le module funcast.selection.

On vérifie :
- Que le critère RRSS retourne un entier valide
- Que la contrainte n_basis >= degree+1 est respectée
- Les cas limites (peu de candidats, grille courte)
"""

import numpy as np
import pytest

from funcast.selection import select_h_rrss

# ==============================================================================
# Fixtures
# ==============================================================================


@pytest.fixture
def synthetic_data():
    """Données synthétiques simples pour tester RRSS."""
    rng = np.random.default_rng(0)
    n, m1 = 30, 60
    t = np.linspace(0, 1, m1)
    # Signal lisse : facile à approximer par des splines
    X = np.array(
        [np.sin(2 * np.pi * t) + 0.1 * rng.standard_normal(m1) for _ in range(n)]
    )
    return X, t


# ==============================================================================
# Tests
# ==============================================================================


class TestSelectHRrss:
    def test_returns_integer(self, synthetic_data):
        """Le résultat doit être un entier."""
        X, t = synthetic_data
        h = select_h_rrss(X, t)
        assert isinstance(h, int)

    def test_respects_minimum_constraint(self, synthetic_data):
        """h doit être >= degree+1."""
        X, t = synthetic_data
        degree = 3
        h = select_h_rrss(X, t, degree=degree)
        assert h >= degree + 1

    def test_h_within_candidates(self, synthetic_data):
        """h retourné doit appartenir aux candidats valides."""
        X, t = synthetic_data
        candidates = [5, 8, 10, 15]
        h = select_h_rrss(X, t, h_candidates=candidates)
        assert h in candidates

    def test_filters_invalid_candidates(self, synthetic_data):
        """Les candidats < degree+1 doivent être filtrés silencieusement."""
        X, t = synthetic_data
        # degree=3 → min valide = 4 ; on passe 1, 2, 3 qui sont invalides
        candidates = [1, 2, 3, 6, 10]
        h = select_h_rrss(X, t, h_candidates=candidates, degree=3)
        assert h >= 4
        assert h in [6, 10]

    def test_all_invalid_candidates_warns(self, synthetic_data):
        """Si tous les candidats sont invalides, un warning est émis."""
        X, t = synthetic_data
        with pytest.warns(UserWarning):
            h = select_h_rrss(X, t, h_candidates=[1, 2], degree=3)
        assert h == 4  # retourne degree+1

    def test_fourier_basis(self, synthetic_data):
        """Fonctionne aussi avec la base de Fourier."""
        X, t = synthetic_data
        h = select_h_rrss(X, t, basis_type="fourier")
        assert isinstance(h, int)
        assert h >= 1

    def test_smooth_signal_low_h(self):
        """Un signal très lisse doit favoriser un petit h."""
        rng = np.random.default_rng(1)
        n, m1 = 40, 80
        t = np.linspace(0, 1, m1)
        # Signal très lisse (constante + légère tendance)
        X = np.array([1.0 + 0.1 * t + 0.01 * rng.standard_normal(m1) for _ in range(n)])
        h = select_h_rrss(X, t, h_candidates=list(range(4, 20)))
        # Un signal quasi-constant est bien approché par peu de bases
        assert h <= 12

    def test_single_candidate(self, synthetic_data):
        """Avec un seul candidat valide, retourne ce candidat."""
        X, t = synthetic_data
        h = select_h_rrss(X, t, h_candidates=[8])
        assert h == 8
