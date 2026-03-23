"""
╔══════════════════════════════════════════════════════════════════════╗
║   src/tuning/pruner.py                                              ║
║   Optuna pruning — early termination of bad trials                  ║
║   Author : Muluh Penn Junior Patrick                                ║
╚══════════════════════════════════════════════════════════════════════╝
Pruners stop unpromising trials early to save compute time.
This is especially important for 50-trial studies on an RTX 3050.

Pruners available:
  - MedianPruner   : Stop if worse than median of completed trials (default)
  - HyperbandPruner: Successive halving (aggressive, faster)
  - NopPruner      : No pruning (for short experiments)
"""

import optuna
from optuna.pruners import MedianPruner, HyperbandPruner, NopPruner
import pytorch_lightning as pl
from loguru import logger


def get_pruner(name: str = "median", **kwargs):
    """
    Return configured Optuna pruner.

    Args:
        name : 'median', 'hyperband', or 'none'
        kwargs: Pruner-specific arguments

    Returns:
        Optuna pruner instance
    """
    name = name.lower()
    if name == "median":
        return MedianPruner(
            n_startup_trials  = kwargs.get("n_startup_trials", 5),
            n_warmup_steps    = kwargs.get("n_warmup_steps",   10),
            interval_steps    = kwargs.get("interval_steps",   1),
        )
    elif name == "hyperband":
        return HyperbandPruner(
            min_resource      = kwargs.get("min_resource",  5),
            max_resource      = kwargs.get("max_resource",  50),
            reduction_factor  = kwargs.get("reduction_factor", 3),
        )
    elif name == "none":
        return NopPruner()
    else:
        raise ValueError(f"Unknown pruner '{name}'. Choose: median, hyperband, none")


class OptunaPruningCallback(pl.Callback):
    """
    PyTorch Lightning callback that reports val_loss to Optuna each epoch
    and raises TrialPruned if the trial should be stopped early.

    Args:
        trial   : Active Optuna trial
        monitor : Metric to report (default: 'val_loss')
    """

    def __init__(self, trial: optuna.Trial, monitor: str = "val_loss"):
        super().__init__()
        self.trial   = trial
        self.monitor = monitor

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        epoch   = trainer.current_epoch
        metrics = trainer.callback_metrics

        value = metrics.get(self.monitor)
        if value is None:
            return

        value = float(value)
        self.trial.report(value, step=epoch)

        if self.trial.should_prune():
            logger.info(f"  Trial {self.trial.number} pruned at epoch {epoch} "
                        f"({self.monitor}={value:.4f})")
            raise optuna.TrialPruned()
