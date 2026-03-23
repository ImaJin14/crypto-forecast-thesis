"""Financial KPIs: Sharpe Ratio, Max Drawdown, Win Rate."""
import numpy as np


def sharpe_ratio(returns: np.ndarray, risk_free_rate: float = 0.0,
                 periods_per_year: int = 252) -> float:
    """Annualized Sharpe Ratio from daily/hourly return series."""
    excess = returns - risk_free_rate / periods_per_year
    if excess.std() == 0:
        return 0.0
    return float(np.sqrt(periods_per_year) * excess.mean() / excess.std())


def max_drawdown(equity_curve: np.ndarray) -> float:
    """Maximum peak-to-trough drawdown as a fraction."""
    peak = np.maximum.accumulate(equity_curve)
    drawdown = (equity_curve - peak) / peak
    return float(drawdown.min())


def win_rate(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """% of trades where direction prediction was profitable."""
    true_dir = np.diff(y_true) > 0
    pred_dir = np.diff(y_pred) > 0
    profitable = (true_dir == pred_dir)
    return float(profitable.mean() * 100)


def profit_factor(returns: np.ndarray) -> float:
    """Gross profit / gross loss ratio."""
    gains  = returns[returns > 0].sum()
    losses = abs(returns[returns < 0].sum())
    return float(gains / losses) if losses != 0 else float("inf")
