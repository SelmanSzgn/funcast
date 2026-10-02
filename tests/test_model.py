"""
Tests for module funcast.model
"""

import numpy as np
import pytest
from sklearn.base import clone

from funcast import FunCast


@pytest.fixture
def time_grids():
    """Past/future time grids."""
    t_past = np.linspace(0, 1, 80)
    t_future = np.linspace(1, 1.25, 20)
    return t_past, t_future


@pytest.fixture
def synthetic_dataset(time_grids):
    """Complete synthetic dataset."""
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
            np.sin(2 * np.pi * t_future)
            + 0.1 * rng.standard_normal(len(t_future))
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


class TestFunCastFitPredict:
    def test_fit_returns_self(self, synthetic_dataset):
        """fit() must return self (sklearn convention)."""
        Y_past, Y_future, _, t_past, t_future = synthetic_dataset
        model = FunCast(K=6, s=0.5)
        result = model.fit(Y_past, Y_future, t_past, t_future)
        assert result is model

    def test_predict_output_shape(self, synthetic_dataset):
        """predict() must return an array of shape (n, m2)."""
        Y_past, Y_future, _, t_past, t_future = synthetic_dataset
        n, m2 = Y_past.shape[0], len(t_future)
        model = FunCast(K=6, s=0.5)
        model.fit(Y_past, Y_future, t_past, t_future)
        Y_pred = model.predict(Y_past)
        assert Y_pred.shape == (n, m2)

    def test_predict_output_is_finite(self, synthetic_dataset):
        """Predictions must not contain NaN or Inf values."""
        Y_past, Y_future, _, t_past, t_future = synthetic_dataset
        model = FunCast(K=6, s=0.5)
        model.fit(Y_past, Y_future, t_past, t_future)
        Y_pred = model.predict(Y_past)
        assert np.all(np.isfinite(Y_pred))

    def test_predict_new_observations(self, synthetic_dataset):
        """predict() must work on new observations."""
        Y_past, Y_future, _, t_past, t_future = synthetic_dataset
        rng = np.random.default_rng(99)
        n_new = 10
        Y_new = rng.standard_normal((n_new, len(t_past)))

        model = FunCast(K=6, s=0.5)
        model.fit(Y_past, Y_future, t_past, t_future)
        Y_pred = model.predict(Y_new)
        assert Y_pred.shape == (n_new, len(t_future))

    def test_fit_with_covariate(self, synthetic_dataset):
        """fit() and predict() work with an external covariate."""
        Y_past, Y_future, covariate, t_past, t_future = synthetic_dataset
        n, m2 = Y_past.shape[0], len(t_future)

        model = FunCast(K=6, s=0.5)
        model.fit(
            Y_past, Y_future, t_past, t_future, covariates_past=[covariate]
        )
        Y_pred = model.predict(Y_past, covariates_past_new=[covariate])

        assert Y_pred.shape == (n, m2)
        assert np.all(np.isfinite(Y_pred))

    def test_fitted_attributes_exist(self, synthetic_dataset):
        """Learned attributes must exist after fit()."""
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
        """h_l must be >= degree+1 after fit()."""
        Y_past, Y_future, _, t_past, t_future = synthetic_dataset
        model = FunCast(K=6, s=0.5, degree=3)
        model.fit(Y_past, Y_future, t_past, t_future)
        assert all(h >= model.degree + 1 for h in model.h_values_)

    def test_q_values_respect_constraint(self, synthetic_dataset):
        """q_l must be >= degree+1 after fit()."""
        Y_past, Y_future, _, t_past, t_future = synthetic_dataset
        model = FunCast(K=6, s=0.5, degree=3)
        model.fit(Y_past, Y_future, t_past, t_future)
        assert all(q >= model.degree + 1 for q in model.q_values_)


class TestFunCastHyperparameters:
    def test_fourier_basis(self, synthetic_dataset):
        """Works with the Fourier basis."""
        Y_past, Y_future, _, t_past, t_future = synthetic_dataset
        model = FunCast(K=7, s=0.5, basis_type="fourier")
        model.fit(Y_past, Y_future, t_past, t_future)
        Y_pred = model.predict(Y_past)
        assert Y_pred.shape == (len(Y_past), len(t_future))

    def test_manual_h_list(self, synthetic_dataset):
        """auto_h=False with a provided h_list must work correctly."""
        Y_past, Y_future, _, t_past, t_future = synthetic_dataset
        model = FunCast(K=6, s=0.5, auto_h=False, h_list=[8])
        model.fit(Y_past, Y_future, t_past, t_future)
        assert model.h_values_[0] == 8

    def test_manual_h_list_missing_raises(self, synthetic_dataset):
        """auto_h=False without h_list must raise a ValueError."""
        Y_past, Y_future, _, t_past, t_future = synthetic_dataset
        model = FunCast(K=6, s=0.5, auto_h=False, h_list=None)
        with pytest.raises(ValueError, match="h_list"):
            model.fit(Y_past, Y_future, t_past, t_future)

    def test_smoothing_s_zero(self, synthetic_dataset):
        """s=0 → q_l = h_l (no smoothing)."""
        Y_past, Y_future, _, t_past, t_future = synthetic_dataset
        model = FunCast(K=6, s=0.0, auto_h=False, h_list=[8])
        model.fit(Y_past, Y_future, t_past, t_future)
        assert model.q_values_[0] == 8

    def test_smoothing_s_one(self, synthetic_dataset):
        """s=1 → q_l clipped to degree+1 (maximum smoothing)."""
        Y_past, Y_future, _, t_past, t_future = synthetic_dataset
        model = FunCast(K=6, s=1.0, auto_h=False, h_list=[8], degree=3)
        model.fit(Y_past, Y_future, t_past, t_future)
        # q = max(4, round(0*8)) = max(4, 0) = 4
        assert model.q_values_[0] == model.degree + 1

    @pytest.mark.parametrize("K", [4, 8, 12])
    def test_various_K(self, synthetic_dataset, K):
        """Works for different values of K."""
        Y_past, Y_future, _, t_past, t_future = synthetic_dataset
        model = FunCast(K=K, s=0.5)
        model.fit(Y_past, Y_future, t_past, t_future)
        Y_pred = model.predict(Y_past)
        assert Y_pred.shape == (len(Y_past), len(t_future))

    def test_degree_changes_fit(self, synthetic_dataset):
        """degree must influence the fitted model."""
        Y_past, Y_future, _, t_past, t_future = synthetic_dataset
        m2 = FunCast(K=6, degree=2, auto_h=False, h_list=[8])
        m3 = FunCast(K=6, degree=3, auto_h=False, h_list=[8])
        m2.fit(Y_past, Y_future, t_past, t_future)
        m3.fit(Y_past, Y_future, t_past, t_future)
        assert not np.allclose(m2.theta_list_[0], m3.theta_list_[0])

class TestFunCastCorrectness:
    @pytest.mark.parametrize("use_cov", [False, True])
    def test_recovers_model_generated_data(self, synthetic_dataset, use_cov):
        """Si Y_future est généré par le modèle, un refit le reproduit."""
        Y_past, Y_future, cov, t_past, t_future = synthetic_dataset
        covs = [cov] if use_cov else None
        params = dict(
            K=6, s=0.5, auto_h=False, h_list=[8, 8] if use_cov else [8]
        )
        gen = FunCast(**params).fit(
            Y_past, Y_future, t_past, t_future, covariates_past=covs
        )
        Y_gen = gen.predict(Y_past, covariates_past_new=covs)

        refit = FunCast(**params).fit(
            Y_past, Y_gen, t_past, t_future, covariates_past=covs
        )
        Y_back = refit.predict(Y_past, covariates_past_new=covs)
        np.testing.assert_allclose(Y_back, Y_gen, atol=1e-6)

    def test_scale_equivariance(self, synthetic_dataset):
        """Multiplier Y par 3 doit multiplier la prédiction par 3."""
        Y_past, Y_future, _, t_past, t_future = synthetic_dataset
        params = dict(K=6, s=0.5, auto_h=False, h_list=[8])
        base = FunCast(**params).fit(Y_past, Y_future, t_past, t_future)
        scaled = FunCast(**params).fit(
            3 * Y_past, 3 * Y_future, t_past, t_future
        )
        np.testing.assert_allclose(
            scaled.predict(3 * Y_past),
            3 * base.predict(Y_past),
            rtol=1e-5,
            atol=1e-6,
        )

    def test_inputs_not_modified(self, synthetic_dataset):
        """fit et predict ne doivent pas modifier les données d'entrée."""
        Y_past, Y_future, _, t_past, t_future = synthetic_dataset
        arrays = (Y_past, Y_future, t_past, t_future)
        copies = [a.copy() for a in arrays]
        FunCast(K=6).fit(Y_past, Y_future, t_past, t_future).predict(Y_past)
        for a, b in zip(arrays, copies):
            np.testing.assert_array_equal(a, b)

    def test_sklearn_clone(self):
        """Compatibilité scikit-learn : clone conserve les paramètres."""
        m = FunCast(K=5, s=0.3, degree=2)
        assert clone(m).get_params() == m.get_params()
