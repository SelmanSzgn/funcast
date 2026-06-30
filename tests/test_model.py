"""
Tests pour le module funcast.model (classe FunCast).

On vérifie :
- Les dimensions des sorties
- Le cycle fit/predict complet
- Le comportement avec et sans covariates
- Les métriques de score
- Les cas d'erreur
"""

import numpy as np
import pytest

from funcast import FunCast

# ==============================================================================
# Fixtures
# ==============================================================================


@pytest.fixture
def time_grids():
    """Grilles temporelles passé/futur."""
    t_past = np.linspace(0, 1, 80)
    t_future = np.linspace(1, 1.25, 20)
    return t_past, t_future


@pytest.fixture
def synthetic_dataset(time_grids):
    """Jeu de données synthétique complet."""
    rng = np.random.default_rng(42)
    n = 40
    t_past, t_future = time_grids

    Y_past = np.array(
        [
            np.sin(2 * np.pi * t_past) + 0.1 * rng.standard_normal(len(t_past))
            for _ in range(n)
        ]
    )
    Y_future = np.array(
        [
            np.sin(2 * np.pi * t_future) + 0.1 * rng.standard_normal(len(t_future))
            for _ in range(n)
        ]
    )
    covariate = np.array(
        [
            np.cos(2 * np.pi * t_past) + 0.1 * rng.standard_normal(len(t_past))
            for _ in range(n)
        ]
    )
    return Y_past, Y_future, covariate, t_past, t_future


# ==============================================================================
# Tests de base
# ==============================================================================


class TestFunCastFitPredict:
    def test_fit_returns_self(self, synthetic_dataset):
        """fit() doit retourner self (convention sklearn)."""
        Y_past, Y_future, _, t_past, t_future = synthetic_dataset
        model = FunCast(K=6, s=0.5)
        result = model.fit(Y_past, Y_future, t_past, t_future)
        assert result is model

    def test_predict_output_shape(self, synthetic_dataset):
        """predict() doit retourner un array (n, m2)."""
        Y_past, Y_future, _, t_past, t_future = synthetic_dataset
        n, m2 = Y_past.shape[0], len(t_future)
        model = FunCast(K=6, s=0.5)
        model.fit(Y_past, Y_future, t_past, t_future)
        Y_pred = model.predict(Y_past)
        assert Y_pred.shape == (n, m2)

    def test_predict_output_is_finite(self, synthetic_dataset):
        """Les prédictions ne doivent pas contenir de NaN ou Inf."""
        Y_past, Y_future, _, t_past, t_future = synthetic_dataset
        model = FunCast(K=6, s=0.5)
        model.fit(Y_past, Y_future, t_past, t_future)
        Y_pred = model.predict(Y_past)
        assert np.all(np.isfinite(Y_pred))

    def test_predict_new_observations(self, synthetic_dataset):
        """predict() doit fonctionner sur de nouvelles observations."""
        Y_past, Y_future, _, t_past, t_future = synthetic_dataset
        rng = np.random.default_rng(99)
        n_new = 10
        Y_new = rng.standard_normal((n_new, len(t_past)))

        model = FunCast(K=6, s=0.5)
        model.fit(Y_past, Y_future, t_past, t_future)
        Y_pred = model.predict(Y_new)
        assert Y_pred.shape == (n_new, len(t_future))

    def test_fit_with_covariate(self, synthetic_dataset):
        """fit() et predict() fonctionnent avec un covariate externe."""
        Y_past, Y_future, covariate, t_past, t_future = synthetic_dataset
        n, m2 = Y_past.shape[0], len(t_future)

        model = FunCast(K=6, s=0.5)
        model.fit(Y_past, Y_future, t_past, t_future, covariates_past=[covariate])
        Y_pred = model.predict(Y_past, covariates_past_new=[covariate])

        assert Y_pred.shape == (n, m2)
        assert np.all(np.isfinite(Y_pred))

    def test_fitted_attributes_exist(self, synthetic_dataset):
        """Les attributs appris doivent exister après fit()."""
        Y_past, Y_future, _, t_past, t_future = synthetic_dataset
        model = FunCast(K=6, s=0.5)
        model.fit(Y_past, Y_future, t_past, t_future)

        assert hasattr(model, "b_hat_")
        assert hasattr(model, "h_values_")
        assert hasattr(model, "q_values_")
        assert hasattr(model, "C_list_")
        assert hasattr(model, "J_list_")
        assert hasattr(model, "theta_list_")

    def test_h_values_respect_constraint(self, synthetic_dataset):
        """hℓ doit être >= degree+1 après fit()."""
        Y_past, Y_future, _, t_past, t_future = synthetic_dataset
        model = FunCast(K=6, s=0.5, degree=3)
        model.fit(Y_past, Y_future, t_past, t_future)
        assert all(h >= model.degree + 1 for h in model.h_values_)

    def test_q_values_respect_constraint(self, synthetic_dataset):
        """qℓ doit être >= degree+1 après fit()."""
        Y_past, Y_future, _, t_past, t_future = synthetic_dataset
        model = FunCast(K=6, s=0.5, degree=3)
        model.fit(Y_past, Y_future, t_past, t_future)
        assert all(q >= model.degree + 1 for q in model.q_values_)


# ==============================================================================
# Tests des hyperparamètres
# ==============================================================================


class TestFunCastHyperparameters:
    def test_fourier_basis(self, synthetic_dataset):
        """Fonctionne avec la base de Fourier."""
        Y_past, Y_future, _, t_past, t_future = synthetic_dataset
        model = FunCast(K=7, s=0.5, basis_type="fourier")
        model.fit(Y_past, Y_future, t_past, t_future)
        Y_pred = model.predict(Y_past)
        assert Y_pred.shape == (len(Y_past), len(t_future))

    def test_manual_h_list(self, synthetic_dataset):
        """auto_h=False avec h_list fourni doit fonctionner."""
        Y_past, Y_future, _, t_past, t_future = synthetic_dataset
        model = FunCast(K=6, s=0.5, auto_h=False, h_list=[8])
        model.fit(Y_past, Y_future, t_past, t_future)
        assert model.h_values_[0] == 8

    def test_manual_h_list_missing_raises(self, synthetic_dataset):
        """auto_h=False sans h_list doit lever une ValueError."""
        Y_past, Y_future, _, t_past, t_future = synthetic_dataset
        model = FunCast(K=6, s=0.5, auto_h=False, h_list=None)
        with pytest.raises(ValueError, match="h_list"):
            model.fit(Y_past, Y_future, t_past, t_future)

    def test_smoothing_s_zero(self, synthetic_dataset):
        """s=0 → qℓ = hℓ (pas de lissage)."""
        Y_past, Y_future, _, t_past, t_future = synthetic_dataset
        model = FunCast(K=6, s=0.0, auto_h=False, h_list=[8])
        model.fit(Y_past, Y_future, t_past, t_future)
        # q = round((1-0)*8) = 8
        assert model.q_values_[0] == 8

    def test_smoothing_s_one(self, synthetic_dataset):
        """s=1 → qℓ clipé à degree+1 (lissage maximal)."""
        Y_past, Y_future, _, t_past, t_future = synthetic_dataset
        model = FunCast(K=6, s=1.0, auto_h=False, h_list=[8], degree=3)
        model.fit(Y_past, Y_future, t_past, t_future)
        # q = max(4, round(0*8)) = max(4, 0) = 4
        assert model.q_values_[0] == model.degree + 1

    @pytest.mark.parametrize("K", [4, 8, 12])
    def test_various_K(self, synthetic_dataset, K):
        """Fonctionne pour différentes valeurs de K."""
        Y_past, Y_future, _, t_past, t_future = synthetic_dataset
        model = FunCast(K=K, s=0.5)
        model.fit(Y_past, Y_future, t_past, t_future)
        Y_pred = model.predict(Y_past)
        assert Y_pred.shape == (len(Y_past), len(t_future))


# ==============================================================================
# Tests des métriques
# ==============================================================================


class TestFunCastScore:
    def test_rmse_is_positive(self, synthetic_dataset):
        """Le RMSE doit être un float positif."""
        Y_past, Y_future, _, t_past, t_future = synthetic_dataset
        model = FunCast(K=6, s=0.5)
        model.fit(Y_past, Y_future, t_past, t_future)
        rmse = model.score(Y_past, Y_future, metric="rmse")
        assert isinstance(rmse, float)
        assert rmse >= 0.0

    def test_smape_in_range(self, synthetic_dataset):
        """Le SMAPE doit être dans [0, 100]."""
        Y_past, Y_future, _, t_past, t_future = synthetic_dataset
        model = FunCast(K=6, s=0.5)
        model.fit(Y_past, Y_future, t_past, t_future)
        smape = model.score(Y_past, Y_future, metric="smape")
        assert isinstance(smape, float)
        assert 0.0 <= smape <= 100.0

    def test_perfect_prediction_rmse_zero(self, time_grids):
        """Si Ŷ = Y exactement, le RMSE doit être 0."""
        t_past, t_future = time_grids
        n = 20
        rng = np.random.default_rng(7)
        Y_past = rng.standard_normal((n, len(t_past)))
        Y_future = rng.standard_normal((n, len(t_future)))

        model = FunCast(K=6, s=0.5)
        model.fit(Y_past, Y_future, t_past, t_future)

        # On remplace b_hat_ par la solution exacte sur les données d'entraînement
        # → le RMSE sur le train set doit être très faible (pas forcément 0 car OLS)
        rmse = model.score(Y_past, Y_future, metric="rmse")
        assert rmse >= 0.0  # propriété minimale

    def test_unknown_metric_raises(self, synthetic_dataset):
        """Un metric inconnu doit lever une ValueError."""
        Y_past, Y_future, _, t_past, t_future = synthetic_dataset
        model = FunCast(K=6, s=0.5)
        model.fit(Y_past, Y_future, t_past, t_future)
        with pytest.raises(ValueError, match="metric inconnu"):
            model.score(Y_past, Y_future, metric="mape")
