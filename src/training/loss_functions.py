"""
╔══════════════════════════════════════════════════════════════════════╗
║   src/training/loss_functions.py                                    ║
║   Loss functions for cryptocurrency price forecasting               ║
║   Author : Muluh Penn Junior Patrick                                ║
╚══════════════════════════════════════════════════════════════════════╝
Loss functions used across all models:
  - MSELoss            : Standard mean squared error (primary)
  - HuberLoss          : Outlier-robust MSE variant
  - DirectionalLoss    : Penalises wrong up/down direction
  - CombinedLoss       : MSE + directional penalty (thesis main loss)
  - MAPELoss           : Mean absolute percentage error
  - QuantileLoss       : Prediction interval training
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger


# ─── Standard Losses ──────────────────────────────────────────────────────────

class MSELoss(nn.MSELoss):
    """Standard Mean Squared Error. Alias for nn.MSELoss for registry compatibility."""
    pass


class MAELoss(nn.L1Loss):
    """Mean Absolute Error. Alias for nn.L1Loss."""
    pass


class HuberLoss(nn.HuberLoss):
    """
    Huber Loss — quadratic for small errors, linear for large ones.
    More robust to outliers than pure MSE (useful for crypto price spikes).

    Args:
        delta : Threshold between quadratic and linear regime (default: 1.0)
    """
    def __init__(self, delta: float = 1.0, **kwargs):
        super().__init__(delta=delta, **kwargs)


# ─── Directional Loss ─────────────────────────────────────────────────────────

class DirectionalLoss(nn.Module):
    """
    Penalises predictions that get the price direction wrong (up vs down).

    This loss is unique to financial forecasting — a model that predicts
    the correct direction but wrong magnitude is more useful than one
    that's accurate in magnitude but wrong in direction.

    Loss = MSE + alpha * directional_penalty

    Where directional_penalty = mean(relu(-(y_true_dir * y_pred_dir)))
    (positive when directions differ, zero when they agree)

    Args:
        alpha : Weight of the directional penalty (default: 0.3)
    """

    def __init__(self, alpha: float = 0.3):
        super().__init__()
        self.alpha = alpha

    def forward(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
    ) -> torch.Tensor:
        # MSE component
        mse = F.mse_loss(y_pred, y_true)

        # Directional penalty
        # Sign of changes: positive = price went up, negative = went down
        true_dir = torch.sign(y_true[1:] - y_true[:-1])
        pred_dir = torch.sign(y_pred[1:] - y_pred[:-1])

        # Penalty when directions disagree
        disagreement = F.relu(-(true_dir * pred_dir))
        dir_penalty  = disagreement.mean()

        return mse + self.alpha * dir_penalty


# ─── Combined Loss (Primary thesis loss) ──────────────────────────────────────

class CombinedLoss(nn.Module):
    """
    Combined loss: MSE + Huber + Directional penalty.

    This is the primary loss function used in the thesis as it optimises
    for both magnitude accuracy (MSE/Huber) and directional accuracy
    (DirectionalLoss), which maps directly to KPIs RMSE and Directional
    Accuracy.

    Loss = w_mse * MSE + w_huber * Huber + w_dir * DirectionalPenalty

    Args:
        w_mse   : Weight for MSE component   (default: 0.5)
        w_huber : Weight for Huber component  (default: 0.3)
        w_dir   : Weight for directional comp (default: 0.2)
        delta   : Huber delta parameter       (default: 1.0)
    """

    def __init__(
        self,
        w_mse:   float = 0.5,
        w_huber: float = 0.3,
        w_dir:   float = 0.2,
        delta:   float = 1.0,
    ):
        super().__init__()
        self.w_mse   = w_mse
        self.w_huber = w_huber
        self.w_dir   = w_dir
        self.huber   = nn.HuberLoss(delta=delta)
        self.dir     = DirectionalLoss(alpha=1.0)

        # Validate weights
        total = w_mse + w_huber + w_dir
        if abs(total - 1.0) > 0.01:
            logger.warning(f"CombinedLoss weights sum to {total:.2f}, not 1.0")

    def forward(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
    ) -> torch.Tensor:
        mse_loss   = F.mse_loss(y_pred, y_true)
        huber_loss = self.huber(y_pred, y_true)
        dir_loss   = self.dir(y_pred, y_true)

        return (self.w_mse   * mse_loss +
                self.w_huber * huber_loss +
                self.w_dir   * dir_loss)

    def components(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
    ) -> dict:
        """Return individual loss components for logging."""
        return {
            "loss_mse":   F.mse_loss(y_pred, y_true).item(),
            "loss_huber": self.huber(y_pred, y_true).item(),
            "loss_dir":   self.dir(y_pred, y_true).item(),
        }


# ─── MAPE Loss ────────────────────────────────────────────────────────────────

class MAPELoss(nn.Module):
    """
    Mean Absolute Percentage Error loss.
    Useful for scale-independent training across multiple assets.

    Note: Unstable when y_true is near zero — use epsilon for stability.

    Args:
        epsilon : Small constant to avoid division by zero (default: 1e-8)
    """

    def __init__(self, epsilon: float = 1e-8):
        super().__init__()
        self.epsilon = epsilon

    def forward(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
    ) -> torch.Tensor:
        return torch.mean(
            torch.abs((y_true - y_pred) / (torch.abs(y_true) + self.epsilon))
        )


# ─── Quantile Loss ────────────────────────────────────────────────────────────

class QuantileLoss(nn.Module):
    """
    Quantile (Pinball) Loss for prediction interval estimation.

    Trains the model to predict a specific quantile of the target
    distribution rather than the mean — useful for uncertainty
    quantification and confidence intervals.

    Args:
        quantile : Target quantile in (0, 1). 0.5 = median (default: 0.5)
    """

    def __init__(self, quantile: float = 0.5):
        super().__init__()
        if not 0 < quantile < 1:
            raise ValueError(f"quantile must be in (0, 1), got {quantile}")
        self.quantile = quantile

    def forward(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
    ) -> torch.Tensor:
        errors = y_true - y_pred
        return torch.mean(
            torch.max(
                self.quantile * errors,
                (self.quantile - 1) * errors,
            )
        )


# ─── Loss Registry ────────────────────────────────────────────────────────────

LOSS_REGISTRY = {
    "mse":      MSELoss,
    "mae":      MAELoss,
    "huber":    HuberLoss,
    "directional": DirectionalLoss,
    "combined": CombinedLoss,
    "mape":     MAPELoss,
    "quantile": QuantileLoss,
}


def get_loss(name: str, **kwargs) -> nn.Module:
    """
    Instantiate a loss function by name.

    Args:
        name   : Loss name from LOSS_REGISTRY
        kwargs : Additional arguments passed to the loss constructor

    Returns:
        Instantiated loss function

    Example:
        loss = get_loss("combined", w_mse=0.6, w_dir=0.2, w_huber=0.2)
        loss = get_loss("huber", delta=0.5)
    """
    name = name.lower()
    if name not in LOSS_REGISTRY:
        raise ValueError(f"Unknown loss '{name}'. Available: {list(LOSS_REGISTRY)}")
    return LOSS_REGISTRY[name](**kwargs)
