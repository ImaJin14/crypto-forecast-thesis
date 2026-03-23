"""
╔══════════════════════════════════════════════════════════════════════╗
║   src/tuning/optuna_study.py                                        ║
║   Optuna hyperparameter optimisation study                          ║
║   Author : Muluh Penn Junior Patrick                                ║
╚══════════════════════════════════════════════════════════════════════╝
Runs automated hyperparameter search for any thesis model-asset pair
using the Tree-structured Parzen Estimator (TPE) sampler.

Usage:
    from src.tuning.optuna_study import run_study

    best = run_study(
        model_name = "lstm",
        asset      = "BTC",
        interval   = "1d",
        n_trials   = 50,
    )

CLI:
    python -m src.tuning.optuna_study --model lstm --asset BTC --trials 50
"""

import os
import sys
import json
from pathlib import Path
from typing import Optional

import optuna
from optuna.samplers import TPESampler
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.tuning.search_spaces    import get_search_space, OPTIMISE_DIR
from src.tuning.pruner           import get_pruner, OptunaPruningCallback
from src.utils.seed              import set_seed
from src.utils.device            import get_device

RESULTS_DIR    = Path(os.getenv("RESULTS_DIR",    "./experiments/results"))
CHECKPOINT_DIR = Path(os.getenv("CHECKPOINT_DIR", "./experiments/checkpoints"))
TUNING_DIR     = RESULTS_DIR / "tuning"


# ─── Objective function ───────────────────────────────────────────────────────

def make_objective(
    model_name:  str,
    asset:       str,
    interval:    str   = "1d",
    horizon:     int   = 1,
    max_epochs:  int   = 50,
    seed:        int   = 42,
):
    """
    Create an Optuna objective function for a given model-asset pair.

    The objective trains a model with trial-suggested hyperparameters
    and returns the best validation loss achieved.

    Args:
        model_name : Model architecture name
        asset      : Asset ticker
        interval   : Candle interval
        horizon    : Forecast horizon
        max_epochs : Max epochs per trial (keep low for tuning: 30-50)
        seed       : Base random seed

    Returns:
        Callable objective(trial) -> float
    """
    import pytorch_lightning as pl
    import torch
    from src.training.trainer       import CryptoForecasterModule, CryptoDataModule
    from src.training.callbacks     import build_callbacks
    from src.training.optimizer_config import get_model_optimizer_config
    from src.models                 import get_model

    def objective(trial: optuna.Trial) -> float:
        # Set seed with trial offset for reproducibility but diversity
        set_seed(seed + trial.number)
        torch.set_float32_matmul_precision("medium")

        # ── 1. Sample hyperparameters ──────────────────────────────────────
        params = get_search_space(model_name, trial)

        seq_len      = params.pop("seq_len",       60)
        batch_size   = params.pop("batch_size",    32)
        lr           = params.pop("lr",            1e-3)
        weight_decay = params.pop("weight_decay",  1e-4)
        loss_name    = params.pop("loss_name",     "combined")
        optimizer_name = params.pop("optimizer_name", "adam")
        scheduler_name = params.pop("scheduler_name", "cosine")
        model_kwargs   = params.pop("model_kwargs",   {})

        logger.info(f"\n  Trial {trial.number} | {model_name} {asset} {interval}")
        logger.info(f"  Params: seq={seq_len} bs={batch_size} lr={lr:.5f} "
                    f"loss={loss_name} | {model_kwargs}")

        try:
            # ── 2. Data ───────────────────────────────────────────────────
            data_module = CryptoDataModule(
                asset      = asset,
                interval   = interval,
                seq_len    = seq_len,
                horizon    = horizon,
                batch_size = batch_size,
            )
            data_module.setup()

            # ── 3. Model ──────────────────────────────────────────────────
            model = get_model(
                model_name,
                input_size  = data_module.n_features,
                output_size = horizon,
                **model_kwargs,
            )

            # ── 4. Lightning module ───────────────────────────────────────
            module = CryptoForecasterModule(
                model          = model,
                loss_name      = loss_name,
                optimizer_name = optimizer_name,
                scheduler_name = scheduler_name,
                lr             = lr,
                weight_decay   = weight_decay,
                max_epochs     = max_epochs,
                model_name     = model_name,
            )

            # ── 5. Callbacks (with Optuna pruning) ────────────────────────
            run_name  = f"{model_name}_{asset}_{interval}_trial{trial.number}"
            ckpt_dir  = CHECKPOINT_DIR / "tuning" / run_name
            callbacks = build_callbacks(
                checkpoint_dir = str(ckpt_dir),
                metrics_dir    = str(TUNING_DIR),
                model_name     = model_name,
                asset          = asset,
                interval       = interval,
                patience       = 10,   # shorter patience for tuning
            )

            # Remove callbacks incompatible with tuning trainer (no logger, no progress bar)
            from pytorch_lightning.callbacks import (
                TQDMProgressBar, RichProgressBar, LearningRateMonitor
            )
            callbacks = [c for c in callbacks
                         if not isinstance(c, (TQDMProgressBar, RichProgressBar,
                                               LearningRateMonitor))]

            # Add Optuna pruning callback
            callbacks.append(OptunaPruningCallback(trial, monitor="val_loss"))

            # ── 6. Trainer ────────────────────────────────────────────────
            accelerator = "gpu" if __import__("torch").cuda.is_available() else "cpu"
            trainer = pl.Trainer(
                max_epochs          = max_epochs,
                accelerator         = accelerator,
                devices             = 1,
                callbacks           = callbacks,
                gradient_clip_val   = 1.0,
                log_every_n_steps   = 10,
                enable_progress_bar = False,   # clean output during tuning
                enable_model_summary= False,
                logger              = False,   # disable Lightning logger for tuning
            )

            # ── 7. Train ──────────────────────────────────────────────────
            trainer.fit(module, datamodule=data_module)

            # Use BEST checkpoint val_loss (not final epoch — avoids collapsed trials)
            ckpt_cb = next(
                (c for c in callbacks if hasattr(c, "best_model_score")), None
            )
            if ckpt_cb is not None and ckpt_cb.best_model_score is not None:
                best_val_loss = float(ckpt_cb.best_model_score)
            else:
                best_val_loss = float(trainer.callback_metrics.get("val_loss", 999.0))

            # Sanity check — reject collapsed trials (val_loss < 0.001 in < 20 epochs)
            n_epochs = trainer.current_epoch + 1
            if best_val_loss < 0.005 and n_epochs < max_epochs:
                logger.warning(f"  Trial {trial.number} likely collapsed "
                               f"(val_loss={best_val_loss:.5f}, epochs={n_epochs}) → penalised")
                best_val_loss = 0.5  # penalise collapsed trials

            logger.info(f"  Trial {trial.number} → best_val_loss={best_val_loss:.4f} "
                        f"({n_epochs} epochs)")
            return best_val_loss

        except optuna.TrialPruned:
            raise
        except Exception as e:
            logger.error(f"  Trial {trial.number} failed: {e}")
            return 999.0   # failed trial gets worst score

    return objective


# ─── Study runner ─────────────────────────────────────────────────────────────

def run_study(
    model_name:    str,
    asset:         str,
    interval:      str   = "1d",
    horizon:       int   = 1,
    n_trials:      int   = 50,
    max_epochs:    int   = 50,
    pruner_name:   str   = "median",
    seed:          int   = 42,
    resume:        bool  = True,
    show_progress: bool  = True,
) -> dict:
    """
    Run a full Optuna hyperparameter optimisation study.

    Args:
        model_name    : Model architecture to tune
        asset         : Asset to tune on
        interval      : Candle interval
        horizon       : Forecast horizon
        n_trials      : Number of trials (default: 50)
        max_epochs    : Max training epochs per trial (default: 50)
        pruner_name   : Pruning strategy ('median', 'hyperband', 'none')
        seed          : Random seed
        resume        : Resume existing study if found (default: True)
        show_progress : Show Optuna progress bar

    Returns:
        dict with keys: best_params, best_value, study, n_trials
    """
    TUNING_DIR.mkdir(parents=True, exist_ok=True)
    study_name = f"{model_name}_{asset}_{interval}_h{horizon}"
    storage    = f"sqlite:///{TUNING_DIR}/{study_name}.db"

    logger.info(f"\n{'═'*60}")
    logger.info(f"  Optuna Study: {study_name}")
    logger.info(f"  Trials: {n_trials} | Epochs/trial: {max_epochs} | Pruner: {pruner_name}")
    logger.info(f"{'═'*60}")

    # Create or resume study
    pruner  = get_pruner(pruner_name)
    sampler = TPESampler(seed=seed, n_startup_trials=10)

    study = optuna.create_study(
        study_name   = study_name,
        storage      = storage if resume else None,
        load_if_exists = resume,
        direction    = OPTIMISE_DIR,
        sampler      = sampler,
        pruner       = pruner,
    )

    # Suppress Optuna's verbose logging
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # Create objective
    objective = make_objective(
        model_name = model_name,
        asset      = asset,
        interval   = interval,
        horizon    = horizon,
        max_epochs = max_epochs,
        seed       = seed,
    )

    # Run optimisation
    study.optimize(
        objective,
        n_trials          = n_trials,
        show_progress_bar = show_progress,
        catch             = (Exception,),
    )

    # ── Results ───────────────────────────────────────────────────────────────
    best_trial  = study.best_trial
    best_params = best_trial.params
    best_value  = best_trial.value

    logger.success(f"\n  ✅  Study complete: {study_name}")
    logger.success(f"  Best val_loss : {best_value:.4f}")
    logger.success(f"  Best params   : {best_params}")

    # Save best params to JSON
    results = {
        "study_name":   study_name,
        "model":        model_name,
        "asset":        asset,
        "interval":     interval,
        "horizon":      horizon,
        "n_trials":     len(study.trials),
        "best_value":   best_value,
        "best_params":  best_params,
        "best_trial":   best_trial.number,
    }
    out_path = TUNING_DIR / f"{study_name}_best_params.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"  Best params saved → {out_path}")

    # Print summary table
    _print_study_summary(study, study_name)

    return {
        "best_params": best_params,
        "best_value":  best_value,
        "study":       study,
        "n_trials":    len(study.trials),
        "results":     results,
    }


def _print_study_summary(study: optuna.Study, name: str):
    """Print a formatted summary of completed trials."""
    trials = [t for t in study.trials
              if t.state == optuna.trial.TrialState.COMPLETE]

    print(f"\n{'─'*62}")
    print(f"  Optuna Summary: {name}  ({len(trials)} completed trials)")
    print(f"{'─'*62}")
    print(f"  {'TRIAL':>6}  {'VAL LOSS':>10}  TOP PARAMS")
    print(f"{'─'*62}")

    sorted_trials = sorted(trials, key=lambda t: t.value)
    for t in sorted_trials[:10]:
        key_params = {k: v for k, v in t.params.items()
                      if k in ["hidden_size","num_layers","lr","dropout","seq_len"]}
        print(f"  {t.number:>6}  {t.value:>10.4f}  {key_params}")
    print(f"{'─'*62}\n")


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run Optuna hyperparameter search",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src.tuning.optuna_study --model lstm --asset BTC --trials 50
  python -m src.tuning.optuna_study --model transformer --asset ETH --trials 30 --epochs 30
  python -m src.tuning.optuna_study --model gru --asset SOL --trials 20 --pruner hyperband
        """
    )
    parser.add_argument("--model",    default="lstm",
                        choices=["lstm","gru","bilstm","cnn_lstm",
                                 "attention_lstm","transformer"])
    parser.add_argument("--asset",    default="BTC",
                        choices=["BTC","ETH","SOL","SUI","XRP"])
    parser.add_argument("--interval", default="1d", choices=["1h","1d"])
    parser.add_argument("--horizon",  type=int, default=1)
    parser.add_argument("--trials",   type=int, default=50)
    parser.add_argument("--epochs",   type=int, default=50,
                        help="Max epochs per trial")
    parser.add_argument("--pruner",   default="median",
                        choices=["median","hyperband","none"])
    parser.add_argument("--no-resume", action="store_true",
                        help="Start fresh study (don't resume)")
    args = parser.parse_args()

    results = run_study(
        model_name  = args.model,
        asset       = args.asset,
        interval    = args.interval,
        horizon     = args.horizon,
        n_trials    = args.trials,
        max_epochs  = args.epochs,
        pruner_name = args.pruner,
        resume      = not args.no_resume,
    )

    print(f"\n{'═'*50}")
    print(f"  Best hyperparameters for {args.model} {args.asset}:")
    print(f"{'═'*50}")
    for k, v in results["best_params"].items():
        print(f"  {k:<25} {v}")
    print(f"\n  Best val_loss: {results['best_value']:.4f}")
    print(f"  Saved to: experiments/results/tuning/")
