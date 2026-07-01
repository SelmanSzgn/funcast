"""
Tests for module funcast.model
"""

import numpy as np
import pytest

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
        model.fit(Y_past, Y_future, t_past, t_future, covariates_past=[covariate])
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


class TestFunCastScore:
    def test_rmse_is_positive(self, synthetic_dataset):
        """RMSE must be a positive float."""
        Y_past, Y_future, _, t_past, t_future = synthetic_dataset
        model = FunCast(K=6, s=0.5)
        model.fit(Y_past, Y_future, t_past, t_future)
        rmse = model.score(Y_past, Y_future, metric="rmse")
        assert isinstance(rmse, float)
        assert rmse >= 0.0

    def test_smape_in_range(self, synthetic_dataset):
        """SMAPE must be in the range [0, 100]."""
        Y_past, Y_future, _, t_past, t_future = synthetic_dataset
        model = FunCast(K=6, s=0.5)
        model.fit(Y_past, Y_future, t_past, t_future)
        smape = model.score(Y_past, Y_future, metric="smape")
        assert isinstance(smape, float)
        assert 0.0 <= smape <= 100.0

    def test_perfect_prediction_rmse_zero(self, time_grids):
        """If Ŷ = Y exactly, RMSE must be 0."""
        t_past, t_future = time_grids
        n = 20
        rng = np.random.default_rng(7)
        Y_past = rng.standard_normal((n, len(t_past)))
        Y_future = rng.standard_normal((n, len(t_future)))
        model = FunCast(K=6, s=0.5)
        model.fit(Y_past, Y_future, t_past, t_future)
        rmse = model.score(Y_past, Y_future, metric="rmse")
        assert rmse >= 0.0

    def test_unknown_metric_raises(self, synthetic_dataset):
        """An unknown metric must raise a ValueError."""
        Y_past, Y_future, _, t_past, t_future = synthetic_dataset
        model = FunCast(K=6, s=0.5)
        model.fit(Y_past, Y_future, t_past, t_future)
        with pytest.raises(ValueError, match="metric inconnu"):
            model.score(Y_past, Y_future, metric="mape")
