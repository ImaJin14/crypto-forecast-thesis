"""
╔══════════════════════════════════════════════════════════════════════╗
║   src/tuning/search_spaces.py                                       ║
║   Hyperparameter search spaces for all thesis models                ║
║   Author : Muluh Penn Junior Patrick                                ║
╚══════════════════════════════════════════════════════════════════════╝
Defines the Optuna search space for each model architecture.
Each function takes an Optuna trial and returns a hyperparameter dict.

Usage:
    from src.tuning.search_spaces import get_search_space
    params = get_search_space("lstm", trial)
"""

from typing import Any
import optuna


def get_search_space(model_name: str, trial: optuna.Trial) -> dict[str, Any]:
    """
    Return hyperparameter dict for a given model from an Optuna trial.

    Args:
        model_name : Model architecture name
        trial      : Optuna trial object

    Returns:
        dict of hyperparameters to pass to train_model()
    """
    model_name = model_name.lower()

    # ── Shared hyperparameters (all models) ───────────────────────────────────
    shared = {
        "seq_len":      trial.suggest_categorical("seq_len",    [30, 60, 90, 120]),
        "batch_size":   trial.suggest_categorical("batch_size", [16, 32, 64]),
        "lr":           trial.suggest_float("lr",          1e-4, 1e-2, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True),
        "loss_name":    "combined",
    }

    if model_name == "lstm":
        return {**shared, **_lstm_space(trial)}
    elif model_name == "gru":
        return {**shared, **_gru_space(trial)}
    elif model_name == "bilstm":
        return {**shared, **_bilstm_space(trial)}
    elif model_name == "cnn_lstm":
        return {**shared, **_cnn_lstm_space(trial)}
    elif model_name == "attention_lstm":
        return {**shared, **_attention_lstm_space(trial)}
    elif model_name == "transformer":
        return {**shared, **_transformer_space(trial)}
    else:
        raise ValueError(f"Unknown model '{model_name}'")


# ─── Model-specific search spaces ─────────────────────────────────────────────

def _lstm_space(trial: optuna.Trial) -> dict:
    hidden_size = trial.suggest_categorical("hidden_size", [64, 128, 256, 512])
    num_layers  = trial.suggest_int("num_layers", 1, 4)
    return {
        "model_kwargs": {
            "hidden_size": hidden_size,
            "num_layers":  num_layers,
            "dropout":     trial.suggest_float("dropout", 0.1, 0.5),
        },
        "optimizer_name":  trial.suggest_categorical("optimizer", ["adam", "adamw"]),
        "scheduler_name":  trial.suggest_categorical("scheduler", ["cosine"]),
    }


def _gru_space(trial: optuna.Trial) -> dict:
    hidden_size = trial.suggest_categorical("hidden_size", [64, 128, 256, 512])
    num_layers  = trial.suggest_int("num_layers", 1, 4)
    return {
        "model_kwargs": {
            "hidden_size": hidden_size,
            "num_layers":  num_layers,
            "dropout":     trial.suggest_float("dropout", 0.1, 0.5),
        },
        "optimizer_name": trial.suggest_categorical("optimizer", ["adam", "adamw"]),
        "scheduler_name": trial.suggest_categorical("scheduler", ["cosine"]),
    }


def _bilstm_space(trial: optuna.Trial) -> dict:
    hidden_size = trial.suggest_categorical("hidden_size", [64, 128, 256])
    num_layers  = trial.suggest_int("num_layers", 1, 3)
    return {
        "model_kwargs": {
            "hidden_size": hidden_size,
            "num_layers":  num_layers,
            "dropout":     trial.suggest_float("dropout", 0.1, 0.5),
        },
        "optimizer_name": trial.suggest_categorical("optimizer", ["adam", "adamw"]),
        "scheduler_name": "cosine",
    }


def _cnn_lstm_space(trial: optuna.Trial) -> dict:
    return {
        "model_kwargs": {
            "num_filters":  trial.suggest_categorical("num_filters",  [32, 64, 128]),
            "kernel_size":  trial.suggest_categorical("kernel_size",  [3, 5, 7]),
            "hidden_size":  trial.suggest_categorical("hidden_size",  [64, 128, 256]),
            "num_layers":   trial.suggest_int("num_layers", 1, 3),
            "dropout":      trial.suggest_float("dropout", 0.1, 0.4),
        },
        "optimizer_name": trial.suggest_categorical("optimizer", ["adam", "adamw"]),
        "scheduler_name": "cosine",
    }


def _attention_lstm_space(trial: optuna.Trial) -> dict:
    hidden_size = trial.suggest_categorical("hidden_size", [64, 128, 256, 512])
    num_layers  = trial.suggest_int("num_layers", 1, 3)
    return {
        "model_kwargs": {
            "hidden_size": hidden_size,
            "num_layers":  num_layers,
            "dropout":     trial.suggest_float("dropout", 0.1, 0.4),
        },
        "optimizer_name": trial.suggest_categorical("optimizer", ["adam", "adamw"]),
        "scheduler_name": trial.suggest_categorical("scheduler", ["cosine"]),
    }


def _transformer_space(trial: optuna.Trial) -> dict:
    d_model = trial.suggest_categorical("d_model", [64, 128, 256])
    # nhead must divide d_model
    nhead_options = [h for h in [2, 4, 8] if d_model % h == 0]
    return {
        "model_kwargs": {
            "d_model":            d_model,
            "nhead":              trial.suggest_categorical("nhead", nhead_options),
            "num_encoder_layers": trial.suggest_int("num_encoder_layers", 1, 4),
            "dim_feedforward":    trial.suggest_categorical("dim_feedforward", [128, 256, 512]),
            "dropout":            trial.suggest_float("dropout", 0.05, 0.3),
        },
        "optimizer_name": "adamw",
        "scheduler_name": trial.suggest_categorical("scheduler",
                                                    ["cosine", "warmup_cosine"]),
    }


# ─── Metric to optimise ───────────────────────────────────────────────────────

OPTIMISE_METRIC = "val_loss"   # minimise validation loss during tuning
OPTIMISE_DIR    = "minimize"
