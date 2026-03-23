"""
╔══════════════════════════════════════════════════════════════════════╗
║   src/training/callbacks.py                                         ║
║   Training callbacks: early stopping, checkpointing, logging        ║
║   Author : Muluh Penn Junior Patrick                                ║
╚══════════════════════════════════════════════════════════════════════╝
PyTorch Lightning callbacks used across all thesis model training runs.

Callbacks:
  - EarlyStoppingCallback  : Stop training when val_loss stops improving
  - ModelCheckpointCallback: Save best model checkpoint to disk
  - LRMonitorCallback      : Log learning rate each epoch
  - TrainingProgressCallback: Rich progress bar with KPI display
  - MetricsLoggerCallback  : Collect and export training metrics to CSV
"""

import os
import csv
import time
from pathlib import Path
from typing import Optional

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    LearningRateMonitor,
    RichProgressBar,
    TQDMProgressBar,
)
from pytorch_lightning.callbacks.progress.rich_progress import RichProgressBarTheme
from loguru import logger


# ─── Early Stopping ───────────────────────────────────────────────────────────

def get_early_stopping(
    monitor:   str = "val_loss",
    patience:  int = 15,
    min_delta: float = 1e-5,
    mode:      str = "min",
    verbose:   bool = True,
) -> EarlyStopping:
    """
    Early stopping callback — halts training when the monitored metric
    stops improving for `patience` consecutive epochs.

    Args:
        monitor   : Metric to monitor (default: 'val_loss')
        patience  : Epochs to wait before stopping (default: 15)
        min_delta : Minimum improvement to count as progress (default: 1e-5)
        mode      : 'min' for loss metrics, 'max' for accuracy metrics
        verbose   : Print message when stopping

    Returns:
        EarlyStopping callback
    """
    return EarlyStopping(
        monitor   = monitor,
        patience  = patience,
        min_delta = min_delta,
        mode      = mode,
        verbose   = verbose,
        strict    = True,
    )


# ─── Model Checkpoint ─────────────────────────────────────────────────────────

def get_model_checkpoint(
    dirpath:   str,
    filename:  str = "{epoch:03d}-{val_loss:.4f}",
    monitor:   str = "val_loss",
    mode:      str = "min",
    save_top_k: int = 1,
    save_last: bool = True,
) -> ModelCheckpoint:
    """
    Model checkpoint callback — saves the best model to disk.

    Args:
        dirpath    : Directory to save checkpoints
        filename   : Checkpoint filename template
        monitor    : Metric to determine best model
        mode       : 'min' for loss, 'max' for accuracy
        save_top_k : Number of best checkpoints to keep (default: 1)
        save_last  : Also save the last epoch checkpoint

    Returns:
        ModelCheckpoint callback
    """
    Path(dirpath).mkdir(parents=True, exist_ok=True)
    return ModelCheckpoint(
        dirpath    = dirpath,
        filename   = filename,
        monitor    = monitor,
        mode       = mode,
        save_top_k = save_top_k,
        save_last  = save_last,
        verbose    = False,
        auto_insert_metric_name = True,
    )


# ─── LR Monitor ───────────────────────────────────────────────────────────────

def get_lr_monitor(logging_interval: str = "epoch") -> LearningRateMonitor:
    """
    Learning rate monitor — logs LR to the logger each epoch/step.

    Args:
        logging_interval : 'epoch' or 'step'

    Returns:
        LearningRateMonitor callback
    """
    return LearningRateMonitor(logging_interval=logging_interval)


# ─── Metrics Logger Callback ──────────────────────────────────────────────────

class MetricsLoggerCallback(pl.Callback):
    """
    Collects training and validation metrics each epoch and saves
    them to a CSV file for later analysis and plotting.

    Tracks: epoch, train_loss, val_loss, learning_rate, epoch_time

    Args:
        save_path : Path to save the CSV file
        model_name: Model name for labelling
        asset     : Asset name for labelling
    """

    def __init__(
        self,
        save_path:  str,
        model_name: str = "model",
        asset:      str = "BTC",
        interval:   str = "1d",
    ):
        super().__init__()
        self.save_path  = Path(save_path)
        self.model_name = model_name
        self.asset      = asset
        self.interval   = interval
        self.records:   list = []
        self._epoch_start: float = 0.0

        self.save_path.parent.mkdir(parents=True, exist_ok=True)

    def on_train_epoch_start(self, trainer, pl_module):
        self._epoch_start = time.time()

    def on_train_epoch_end(self, trainer, pl_module):
        epoch_time = time.time() - self._epoch_start
        metrics    = trainer.callback_metrics

        record = {
            "model":      self.model_name,
            "asset":      self.asset,
            "interval":   self.interval,
            "epoch":      trainer.current_epoch,
            "train_loss": metrics.get("train_loss", float("nan")),
            "val_loss":   metrics.get("val_loss",   float("nan")),
            "lr":         self._get_lr(trainer),
            "epoch_time": round(epoch_time, 2),
        }

        # Convert tensors to floats
        for k, v in record.items():
            if isinstance(v, torch.Tensor):
                record[k] = v.item()

        self.records.append(record)
        self._save_csv()

    def _get_lr(self, trainer) -> float:
        try:
            return trainer.optimizers[0].param_groups[0]["lr"]
        except Exception:
            return float("nan")

    def _save_csv(self):
        if not self.records:
            return
        with open(self.save_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.records[0].keys())
            writer.writeheader()
            writer.writerows(self.records)

    def load_records(self) -> list:
        """Return collected metrics records."""
        return self.records.copy()

    def on_fit_end(self, trainer, pl_module):
        if self.records:
            logger.info(f"  Metrics saved → {self.save_path} "
                        f"({len(self.records)} epochs)")


# ─── Gradient Clipping Monitor ────────────────────────────────────────────────

class GradientMonitorCallback(pl.Callback):
    """
    Monitors gradient norms during training.
    Useful for detecting vanishing/exploding gradients in RNNs.

    Logs: grad_norm_mean, grad_norm_max per epoch
    """

    def __init__(self, log_every_n_epochs: int = 5):
        super().__init__()
        self.log_every_n_epochs = log_every_n_epochs

    def on_before_optimizer_step(self, trainer, pl_module, optimizer):
        if trainer.current_epoch % self.log_every_n_epochs == 0:
            norms = []
            for p in pl_module.parameters():
                if p.grad is not None:
                    norms.append(p.grad.data.norm(2).item())
            if norms:
                pl_module.log("grad_norm_mean", sum(norms) / len(norms))
                pl_module.log("grad_norm_max",  max(norms))


# ─── Training Time Callback ───────────────────────────────────────────────────

class TrainingTimerCallback(pl.Callback):
    """
    Records total training time and average epoch time.
    Used for thesis model comparison KPI: training time.
    """

    def __init__(self):
        super().__init__()
        self._start:       float = 0.0
        self._epoch_times: list  = []
        self._epoch_start: float = 0.0
        self.total_time:   float = 0.0
        self.avg_epoch_time: float = 0.0

    def on_fit_start(self, trainer, pl_module):
        self._start = time.time()

    def on_train_epoch_start(self, trainer, pl_module):
        self._epoch_start = time.time()

    def on_train_epoch_end(self, trainer, pl_module):
        self._epoch_times.append(time.time() - self._epoch_start)

    def on_fit_end(self, trainer, pl_module):
        self.total_time     = time.time() - self._start
        self.avg_epoch_time = (
            sum(self._epoch_times) / len(self._epoch_times)
            if self._epoch_times else 0.0
        )
        logger.info(f"  Training time: {self.total_time:.1f}s total | "
                    f"{self.avg_epoch_time:.2f}s/epoch")


# ─── Callback Bundle Factory ──────────────────────────────────────────────────

def build_callbacks(
    checkpoint_dir: str,
    metrics_dir:    str,
    model_name:     str = "model",
    asset:          str = "BTC",
    interval:       str = "1d",
    patience:       int = 15,
    monitor:        str = "val_loss",
    save_top_k:     int = 1,
    use_rich:       bool = False,
) -> list:
    """
    Build the standard callback suite for a training run.

    Returns a list of callbacks ready to pass to pl.Trainer.

    Args:
        checkpoint_dir : Directory to save model checkpoints
        metrics_dir    : Directory to save metrics CSV
        model_name     : Model name for file naming
        asset          : Asset name for file naming
        interval       : Data interval for file naming
        patience       : Early stopping patience
        monitor        : Metric to monitor
        save_top_k     : Number of checkpoints to keep
        use_rich       : Use rich progress bar (requires rich package)

    Returns:
        List of configured callbacks

    Example:
        callbacks = build_callbacks(
            checkpoint_dir = "experiments/checkpoints/lstm_BTC_1d",
            metrics_dir    = "experiments/results",
            model_name     = "lstm",
            asset          = "BTC",
        )
        trainer = pl.Trainer(callbacks=callbacks, ...)
    """
    run_name   = f"{model_name}_{asset}_{interval}"
    metrics_path = Path(metrics_dir) / f"{run_name}_metrics.csv"
    ckpt_name    = f"{run_name}_{{epoch:03d}}-{{val_loss:.4f}}"

    callbacks = [
        # 1. Early stopping
        get_early_stopping(
            monitor  = monitor,
            patience = patience,
            mode     = "min" if "loss" in monitor else "max",
        ),

        # 2. Best model checkpoint
        get_model_checkpoint(
            dirpath    = checkpoint_dir,
            filename   = ckpt_name,
            monitor    = monitor,
            save_top_k = save_top_k,
        ),

        # 3. Learning rate monitor
        get_lr_monitor(logging_interval="epoch"),

        # 4. Metrics CSV logger
        MetricsLoggerCallback(
            save_path  = metrics_path,
            model_name = model_name,
            asset      = asset,
            interval   = interval,
        ),

        # 5. Gradient monitor (for detecting RNN gradient issues)
        GradientMonitorCallback(log_every_n_epochs=10),

        # 6. Training timer
        TrainingTimerCallback(),

        # 7. Progress bar
        RichProgressBar() if use_rich else TQDMProgressBar(refresh_rate=10),
    ]

    logger.info(f"  Callbacks: {len(callbacks)} registered for {run_name}")
    return callbacks
