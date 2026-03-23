"""Core forecasting KPI metrics."""
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error."""
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error."""
    return float(mean_absolute_error(y_true, y_pred))


def mape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    """Mean Absolute Percentage Error (%)."""
    return float(np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + eps))) * 100)


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """R² coefficient of determination."""
    return float(r2_score(y_true, y_pred))


def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Percentage of correct up/down direction predictions."""
    true_dir = np.diff(y_true) > 0
    pred_dir = np.diff(y_pred) > 0
    return float(np.mean(true_dir == pred_dir) * 100)


def compute_all_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute all forecasting KPIs and return as dict."""
    return {
        "rmse": rmse(y_true, y_pred),
        "mae":  mae(y_true, y_pred),
        "mape": mape(y_true, y_pred),
        "r2":   r2(y_true, y_pred),
        "directional_accuracy": directional_accuracy(y_true, y_pred),
    }
