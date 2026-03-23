"""Unit tests for KPI metric functions."""
import numpy as np
import pytest
from src.evaluation.metrics import rmse, mae, mape, r2, directional_accuracy
from src.evaluation.financial_metrics import sharpe_ratio, win_rate


@pytest.fixture
def perfect_predictions(sample_prices):
    return sample_prices, sample_prices.copy()


def test_rmse_perfect(perfect_predictions):
    y, yhat = perfect_predictions
    assert rmse(y, yhat) == pytest.approx(0.0, abs=1e-6)


def test_r2_perfect(perfect_predictions):
    y, yhat = perfect_predictions
    assert r2(y, yhat) == pytest.approx(1.0, abs=1e-6)


def test_mape_non_negative(sample_prices):
    noisy = sample_prices + np.random.randn(len(sample_prices))
    assert mape(sample_prices, noisy) >= 0


def test_directional_accuracy_range(sample_prices):
    noisy = sample_prices + np.random.randn(len(sample_prices))
    da = directional_accuracy(sample_prices, noisy)
    assert 0 <= da <= 100
