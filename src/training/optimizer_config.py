"""
╔══════════════════════════════════════════════════════════════════════╗
║   src/training/optimizer_config.py                                  ║
║   Optimizer and learning rate scheduler configuration               ║
║   Author : Muluh Penn Junior Patrick                                ║
╚══════════════════════════════════════════════════════════════════════╝
Centralised optimizer and scheduler setup for all thesis models.

Optimizers : Adam (primary), AdamW, SGD with momentum
Schedulers : CosineAnnealing (primary), ReduceLROnPlateau, OneCycleLR,
             WarmupCosine (custom)
"""

import math
import torch
import torch.nn as nn
from torch.optim import Adam, AdamW, SGD
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    ReduceLROnPlateau,
    OneCycleLR,
    StepLR,
    ExponentialLR,
)
from loguru import logger


# ─── Optimizer Factory ────────────────────────────────────────────────────────

def get_optimizer(
    model:        nn.Module,
    name:         str   = "adam",
    lr:           float = 1e-3,
    weight_decay: float = 1e-4,
    **kwargs,
) -> torch.optim.Optimizer:
    """
    Instantiate an optimizer for a model.

    Args:
        model        : PyTorch model (parameters will be extracted)
        name         : Optimizer name: 'adam', 'adamw', 'sgd'
        lr           : Initial learning rate (default: 1e-3)
        weight_decay : L2 regularisation strength (default: 1e-4)
        **kwargs     : Additional optimizer-specific arguments

    Returns:
        Configured optimizer

    Example:
        opt = get_optimizer(model, name="adamw", lr=5e-4, weight_decay=1e-3)
    """
    name = name.lower()
    params = model.parameters()

    if name == "adam":
        optimizer = Adam(
            params,
            lr           = lr,
            weight_decay = weight_decay,
            betas        = kwargs.get("betas", (0.9, 0.999)),
            eps          = kwargs.get("eps",   1e-8),
        )
    elif name == "adamw":
        optimizer = AdamW(
            params,
            lr           = lr,
            weight_decay = weight_decay,
            betas        = kwargs.get("betas", (0.9, 0.999)),
            eps          = kwargs.get("eps",   1e-8),
        )
    elif name == "sgd":
        optimizer = SGD(
            params,
            lr           = lr,
            momentum     = kwargs.get("momentum",     0.9),
            weight_decay = weight_decay,
            nesterov     = kwargs.get("nesterov",     True),
        )
    else:
        raise ValueError(f"Unknown optimizer '{name}'. Choose: adam, adamw, sgd")

    logger.debug(f"  Optimizer: {type(optimizer).__name__} | lr={lr} | "
                 f"wd={weight_decay}")
    return optimizer


# ─── Scheduler Factory ────────────────────────────────────────────────────────

def get_scheduler(
    optimizer:  torch.optim.Optimizer,
    name:       str = "cosine",
    max_epochs: int = 200,
    steps_per_epoch: int = None,
    **kwargs,
):
    """
    Instantiate a learning rate scheduler.

    Args:
        optimizer        : Fitted optimizer
        name             : Scheduler name:
                           'cosine'    — CosineAnnealingLR (primary)
                           'plateau'   — ReduceLROnPlateau
                           'onecycle'  — OneCycleLR
                           'warmup_cosine' — Linear warmup + cosine decay
                           'step'      — StepLR
                           'exp'       — ExponentialLR
                           'none'      — No scheduling
        max_epochs       : Total training epochs (for cosine/onecycle)
        steps_per_epoch  : Steps per epoch (required for onecycle)
        **kwargs         : Scheduler-specific overrides

    Returns:
        Scheduler instance (or None if name='none')
    """
    name = name.lower()

    if name == "cosine":
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max   = kwargs.get("T_max",   max_epochs),
            eta_min = kwargs.get("eta_min", 1e-6),
        )

    elif name == "plateau":
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode     = kwargs.get("mode",     "min"),
            factor   = kwargs.get("factor",   0.5),
            patience = kwargs.get("patience", 10),
            min_lr   = kwargs.get("min_lr",   1e-6),
            verbose  = False,
        )

    elif name == "onecycle":
        if steps_per_epoch is None:
            raise ValueError("steps_per_epoch required for OneCycleLR")
        max_lr = kwargs.get("max_lr", optimizer.param_groups[0]["lr"] * 10)
        scheduler = OneCycleLR(
            optimizer,
            max_lr          = max_lr,
            epochs          = max_epochs,
            steps_per_epoch = steps_per_epoch,
            pct_start       = kwargs.get("pct_start", 0.3),
            div_factor      = kwargs.get("div_factor", 25),
            final_div_factor= kwargs.get("final_div_factor", 1e4),
        )

    elif name == "warmup_cosine":
        warmup_epochs = kwargs.get("warmup_epochs", max(5, max_epochs // 20))
        scheduler = WarmupCosineScheduler(
            optimizer,
            warmup_epochs = warmup_epochs,
            max_epochs    = max_epochs,
            min_lr        = kwargs.get("min_lr", 1e-6),
        )

    elif name == "step":
        scheduler = StepLR(
            optimizer,
            step_size = kwargs.get("step_size", max_epochs // 4),
            gamma     = kwargs.get("gamma",     0.5),
        )

    elif name == "exp":
        scheduler = ExponentialLR(
            optimizer,
            gamma = kwargs.get("gamma", 0.95),
        )

    elif name == "none":
        return None

    else:
        raise ValueError(
            f"Unknown scheduler '{name}'. "
            "Choose: cosine, plateau, onecycle, warmup_cosine, step, exp, none"
        )

    logger.debug(f"  Scheduler: {type(scheduler).__name__}")
    return scheduler


# ─── Warmup + Cosine Scheduler ────────────────────────────────────────────────

class WarmupCosineScheduler(torch.optim.lr_scheduler.LambdaLR):
    """
    Linear warmup followed by cosine annealing decay.

    The warmup phase linearly increases LR from 0 to base_lr over
    warmup_epochs, then cosine-decays to min_lr over the remaining epochs.

    This is the recommended schedule for Transformer models.

    Args:
        optimizer     : Fitted optimizer
        warmup_epochs : Number of linear warmup epochs
        max_epochs    : Total training epochs
        min_lr        : Minimum learning rate at end of decay
    """

    def __init__(
        self,
        optimizer:     torch.optim.Optimizer,
        warmup_epochs: int   = 10,
        max_epochs:    int   = 200,
        min_lr:        float = 1e-6,
    ):
        self.warmup_epochs = warmup_epochs
        self.max_epochs    = max_epochs
        base_lr            = optimizer.param_groups[0]["lr"]
        self.min_lr_ratio  = min_lr / max(base_lr, 1e-10)

        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                # Linear warmup
                return float(epoch) / float(max(warmup_epochs, 1))
            # Cosine decay
            progress = (epoch - warmup_epochs) / float(
                max(max_epochs - warmup_epochs, 1)
            )
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return self.min_lr_ratio + (1.0 - self.min_lr_ratio) * cosine

        super().__init__(optimizer, lr_lambda)


# ─── Full Optimizer + Scheduler Bundle ────────────────────────────────────────

def build_optimizer_config(
    model:           nn.Module,
    optimizer_name:  str   = "adam",
    scheduler_name:  str   = "cosine",
    lr:              float = 1e-3,
    weight_decay:    float = 1e-4,
    max_epochs:      int   = 200,
    steps_per_epoch: int   = None,
    **kwargs,
) -> dict:
    """
    Build a complete optimizer + scheduler configuration dict.

    Returns a dict compatible with PyTorch Lightning's
    configure_optimizers() return format.

    Args:
        model           : PyTorch model
        optimizer_name  : Optimizer name (default: 'adam')
        scheduler_name  : Scheduler name (default: 'cosine')
        lr              : Learning rate  (default: 1e-3)
        weight_decay    : L2 reg         (default: 1e-4)
        max_epochs      : Total epochs   (default: 200)
        steps_per_epoch : Steps per epoch (for OneCycleLR)
        **kwargs        : Passed to optimizer/scheduler constructors

    Returns:
        dict: {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss',
                'interval': 'epoch',
                'frequency': 1,
            }
        }

    Example:
        config = build_optimizer_config(model, lr=5e-4, scheduler_name="plateau")
    """
    optimizer = get_optimizer(
        model,
        name         = optimizer_name,
        lr           = lr,
        weight_decay = weight_decay,
        **{k: v for k, v in kwargs.items()
           if k in ["betas", "eps", "momentum", "nesterov"]},
    )

    scheduler = get_scheduler(
        optimizer,
        name             = scheduler_name,
        max_epochs       = max_epochs,
        steps_per_epoch  = steps_per_epoch,
        **{k: v for k, v in kwargs.items()
           if k in ["T_max","eta_min","factor","patience","min_lr",
                    "pct_start","warmup_epochs","step_size","gamma",
                    "div_factor","final_div_factor","max_lr"]},
    )

    config = {"optimizer": optimizer}

    if scheduler is not None:
        monitor  = "val_loss" if scheduler_name == "plateau" else None
        interval = "step" if scheduler_name == "onecycle" else "epoch"

        lr_config = {
            "scheduler": scheduler,
            "interval":  interval,
            "frequency": 1,
        }
        if monitor:
            lr_config["monitor"] = monitor

        config["lr_scheduler"] = lr_config

    logger.info(f"  Optimizer config: {optimizer_name} lr={lr} | "
                f"scheduler={scheduler_name} | epochs={max_epochs}")
    return config


# ─── Model-specific recommended configs ──────────────────────────────────────

MODEL_OPTIMIZER_DEFAULTS = {
    "lstm": {
        "optimizer_name": "adam",
        "scheduler_name": "cosine",
        "lr":             1e-3,
        "weight_decay":   1e-4,
    },
    "gru": {
        "optimizer_name": "adam",
        "scheduler_name": "cosine",
        "lr":             1e-3,
        "weight_decay":   1e-4,
    },
    "bilstm": {
        "optimizer_name": "adam",
        "scheduler_name": "cosine",
        "lr":             1e-3,
        "weight_decay":   1e-4,
    },
    "cnn_lstm": {
        "optimizer_name": "adamw",
        "scheduler_name": "cosine",
        "lr":             5e-4,
        "weight_decay":   1e-3,
    },
    "attention_lstm": {
        "optimizer_name": "adamw",
        "scheduler_name": "cosine",
        "lr":             5e-4,
        "weight_decay":   1e-3,
    },
    "transformer": {
        "optimizer_name": "adamw",
        "scheduler_name": "warmup_cosine",
        "lr":             1e-4,
        "weight_decay":   1e-2,
        "warmup_epochs":  10,
    },
}


def get_model_optimizer_config(
    model_name: str,
    model:      nn.Module,
    max_epochs: int = 200,
    **overrides,
) -> dict:
    """
    Get the recommended optimizer config for a specific model architecture.

    Args:
        model_name : One of the 6 thesis model names
        model      : Instantiated model
        max_epochs : Total training epochs
        **overrides: Override any default hyperparameter

    Returns:
        Optimizer config dict (Lightning-compatible)
    """
    defaults = MODEL_OPTIMIZER_DEFAULTS.get(model_name, MODEL_OPTIMIZER_DEFAULTS["lstm"])
    config   = {**defaults, **overrides}

    return build_optimizer_config(
        model      = model,
        max_epochs = max_epochs,
        **config,
    )
