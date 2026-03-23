"""
╔══════════════════════════════════════════════════════════════════════╗
║   src/evaluation/regime_analysis.py                                 ║
║   Market regime detection and performance breakdown                 ║
║   Author : Muluh Penn Junior Patrick                                ║
╚══════════════════════════════════════════════════════════════════════╝
Splits test period into Bull / Bear / Sideways market regimes and
evaluates model performance separately in each regime.

This is a standard requirement in financial ML papers — it shows
whether a model works consistently across market conditions.
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from loguru import logger

DATA_DIR    = Path(os.getenv("DATA_DIR",    "./data"))
RESULTS_DIR = Path(os.getenv("RESULTS_DIR", "./experiments/results"))


# ─── Regime Detection ─────────────────────────────────────────────────────────

def detect_regimes(
    prices:        np.ndarray,
    bull_threshold: float = 0.02,
    bear_threshold: float = -0.02,
    window:         int   = 20,
) -> np.ndarray:
    """
    Classify each day as Bull / Bear / Sideways based on rolling return.

    A period is classified as:
        Bull     : Rolling 20-day return > +2%
        Bear     : Rolling 20-day return < -2%
        Sideways : Otherwise

    Args:
        prices          : Array of closing prices
        bull_threshold  : Minimum rolling return for bull (default: +2%)
        bear_threshold  : Maximum rolling return for bear (default: -2%)
        window          : Rolling window in days (default: 20)

    Returns:
        Array of strings: 'bull', 'bear', or 'sideways'
    """
    prices = np.asarray(prices, dtype=float)
    n      = len(prices)

    rolling_returns = np.full(n, np.nan)
    for i in range(window, n):
        rolling_returns[i] = (prices[i] - prices[i - window]) / prices[i - window]

    regimes = np.full(n, "sideways", dtype=object)
    regimes[rolling_returns > bull_threshold]  = "bull"
    regimes[rolling_returns < bear_threshold]  = "bear"
    regimes[:window] = "sideways"  # not enough history

    return regimes


def load_test_prices(asset: str, interval: str) -> np.ndarray:
    """Load raw test period close prices for regime detection."""
    scaler_dir  = DATA_DIR / "processed" / "scalers"
    test_prices = scaler_dir / f"{asset}_{interval}_test_close.npy"

    if not test_prices.exists():
        logger.warning(f"  Test prices not found: {test_prices}")
        return np.array([])

    return np.load(test_prices)


# ─── Regime Performance ───────────────────────────────────────────────────────

def regime_performance(
    y_true:  np.ndarray,
    y_pred:  np.ndarray,
    regimes: np.ndarray,
) -> pd.DataFrame:
    """
    Compute MAPE, RMSE, and DA for each market regime.

    Args:
        y_true  : Actual prices (USD)
        y_pred  : Predicted prices (USD)
        regimes : Regime labels ('bull', 'bear', 'sideways')

    Returns:
        DataFrame with regime × metric performance
    """
    y_true  = np.asarray(y_true)
    y_pred  = np.asarray(y_pred)
    regimes = np.asarray(regimes)

    # Align lengths
    n = min(len(y_true), len(y_pred), len(regimes))
    y_true  = y_true[:n]
    y_pred  = y_pred[:n]
    regimes = regimes[:n]

    records = []
    for regime in ["bull", "bear", "sideways", "all"]:
        if regime == "all":
            mask = np.ones(n, dtype=bool)
        else:
            mask = regimes == regime

        if mask.sum() < 5:
            continue

        yt = y_true[mask]
        yp = y_pred[mask]
        n_regime = mask.sum()

        # Compute metrics
        errors    = yt - yp
        rmse      = float(np.sqrt(np.mean(errors ** 2)))
        mae       = float(np.mean(np.abs(errors)))
        mape      = float(np.mean(np.abs(errors / (np.abs(yt) + 1e-8))) * 100)
        r2        = float(1 - np.sum(errors**2) / (np.sum((yt - yt.mean())**2) + 1e-10))

        # Directional accuracy
        dir_correct = np.sign(np.diff(yp)) == np.sign(np.diff(yt))
        da = float(dir_correct.mean() * 100) if len(dir_correct) > 0 else 50.0

        records.append({
            "regime":  regime,
            "n_days":  int(n_regime),
            "rmse":    rmse,
            "mae":     mae,
            "mape":    mape,
            "r2":      r2,
            "da":      da,
        })

    return pd.DataFrame(records).set_index("regime")


def load_predictions_from_checkpoint(
    model_name: str,
    asset:      str,
    interval:   str,
    horizon:    int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Re-run inference with the saved best checkpoint to get predictions.

    This is a lightweight version that loads the checkpoint and runs
    the test DataLoader without full training infrastructure.
    """
    import torch, sys
    sys.path.insert(0, ".")

    from src.training.trainer import CryptoDataModule, CryptoForecasterModule
    from src.models import get_model
    from src.evaluation.financial_metrics import _load_best_params

    # Load best params
    params_file = RESULTS_DIR / "tuning" / f"{model_name}_{asset}_{interval}_h{horizon}_best_params.json"
    if not params_file.exists():
        logger.warning(f"  Best params not found: {params_file}")
        return np.array([]), np.array([])

    import json
    with open(params_file) as f:
        params = json.load(f)["best_params"]

    seq_len    = params.get("seq_len", 60)
    batch_size = params.get("batch_size", 32)
    model_kwargs = {k: v for k, v in params.items()
                    if k in ["hidden_size","num_layers","dropout",
                             "d_model","nhead","num_encoder_layers",
                             "dim_feedforward","num_filters","kernel_size"]}

    # Setup data
    data_module = CryptoDataModule(
        asset=asset, interval=interval,
        seq_len=seq_len, horizon=horizon,
        batch_size=batch_size,
    )
    data_module.setup()

    # Load model
    model = get_model(model_name, input_size=data_module.n_features,
                      output_size=horizon, **model_kwargs)

    # Find best checkpoint
    ckpt_dir  = Path("experiments/checkpoints") / f"{model_name}_{asset}_{interval}_h{horizon}"
    ckpts     = list(ckpt_dir.glob("*.ckpt")) if ckpt_dir.exists() else []
    if not ckpts:
        logger.warning(f"  No checkpoint found in {ckpt_dir}")
        return np.array([]), np.array([])

    # Load the best checkpoint (not 'last')
    best_ckpt = min([c for c in ckpts if "last" not in c.name],
                    key=lambda c: float(c.stem.split("val_loss=")[-1].split("-")[0])
                    if "val_loss=" in c.stem else float("inf"),
                    default=ckpts[0])

    module = CryptoForecasterModule.load_from_checkpoint(
        str(best_ckpt), model=model, strict=False
    )
    module.eval()

    # Inference
    device     = "cuda" if torch.cuda.is_available() else "cpu"
    module     = module.to(device)
    all_preds  = []
    all_targets = []

    with torch.no_grad():
        for batch in data_module.test_dataloader():
            x, y = batch
            x    = x.to(device)
            yhat = module(x).squeeze().cpu().numpy()
            all_preds.append(np.atleast_1d(yhat))
            all_targets.append(np.atleast_1d(y.numpy()))

    pred_log_returns   = np.concatenate(all_preds).flatten()
    target_log_returns = np.concatenate(all_targets).flatten()

    # Reconstruct USD prices
    test_close = np.load(
        DATA_DIR / "processed" / "scalers" / f"{asset}_{interval}_test_close.npy"
    )
    n = len(pred_log_returns)
    current_idx  = np.arange(n) + seq_len - 1
    current_idx  = np.clip(current_idx, 0, len(test_close) - 1)
    start_prices = test_close[current_idx]

    preds   = start_prices * np.exp(pred_log_returns)
    targets = start_prices * np.exp(target_log_returns)

    return preds, targets


def full_regime_analysis(
    asset:    str = "BTC",
    interval: str = "1d",
    horizon:  int = 1,
) -> dict:
    """
    Run full regime analysis for all available models.

    Returns:
        Dict mapping model_name → regime performance DataFrame
    """
    from src.evaluation.evaluator import MODELS

    test_prices = load_test_prices(asset, interval)
    if len(test_prices) == 0:
        logger.error("  Cannot run regime analysis without test prices")
        return {}

    regimes = detect_regimes(test_prices)
    regime_counts = {r: (regimes == r).sum() for r in ["bull", "bear", "sideways"]}
    logger.info(f"  Regime distribution: {regime_counts}")

    results = {}
    for model in MODELS:
        preds, targets = load_predictions_from_checkpoint(
            model, asset, interval, horizon
        )
        if len(preds) == 0:
            continue

        # Align regime labels with prediction window
        seq_len = 60  # approximate
        regime_aligned = regimes[seq_len : seq_len + len(preds)]
        if len(regime_aligned) < len(preds):
            regime_aligned = np.pad(regime_aligned, (0, len(preds) - len(regime_aligned)),
                                    constant_values="sideways")

        perf = regime_performance(targets, preds, regime_aligned)
        results[model] = perf
        logger.info(f"  {model}: {perf.to_dict()}")

    return results


def print_regime_table(results: dict):
    """Print formatted regime performance table."""
    if not results:
        return

    print(f"\n{'═'*80}")
    print(f"  Regime Performance Analysis")
    print(f"{'═'*80}")

    for regime in ["bull", "bear", "sideways", "all"]:
        print(f"\n  Regime: {regime.upper()}")
        print(f"  {'MODEL':<20}  {'MAPE':>8}  {'RMSE':>10}  {'DA':>8}  {'R²':>8}")
        print(f"  {'─'*56}")

        for model, df in results.items():
            if regime not in df.index:
                continue
            row = df.loc[regime]
            print(f"  {model:<20}  {row.get('mape',0):>8.2f}  "
                  f"{row.get('rmse',0):>10.0f}  "
                  f"{row.get('da',0):>8.1f}  "
                  f"{row.get('r2',0):>8.4f}")


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run regime analysis")
    parser.add_argument("--asset",    default="BTC")
    parser.add_argument("--interval", default="1d")
    parser.add_argument("--horizon",  type=int, default=1)
    args = parser.parse_args()

    results = full_regime_analysis(args.asset, args.interval, args.horizon)
    print_regime_table(results)
