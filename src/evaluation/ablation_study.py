"""
╔══════════════════════════════════════════════════════════════════════╗
║   src/evaluation/ablation_study.py                                  ║
║   Feature ablation study — what contributes most?                   ║
║   Author : Muluh Penn Junior Patrick                                ║
╚══════════════════════════════════════════════════════════════════════╝
Tests model performance when feature groups are removed one at a time.
Answers: "How much does each feature category contribute?"

Feature groups tested:
  - Full model (baseline)
  - No LTST decomposition features
  - No on-chain metrics
  - No sentiment features
  - No macro features
  - Technical indicators only (OHLCV + TA)

Usage:
    python -m src.evaluation.ablation_study --model lstm --asset BTC
"""

import os
import sys
import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

DATA_DIR    = Path(os.getenv("DATA_DIR",    "./data"))
RESULTS_DIR = Path(os.getenv("RESULTS_DIR", "./experiments/results"))

# Feature group definitions (column name prefixes/patterns)
FEATURE_GROUPS = {
    "ltst":      ["ltt_", "stt_", "ma_residual", "ltt_stt_spread", "ltt_slope",
                  "stt_slope", "above_ltt", "hp_trend", "hp_cycle", "stl_",
                  "trend_consensus", "ltst_", "mean_reversion", "ltt_strength"],
    "onchain":   ["hash_rate", "difficulty", "n_transactions", "mempool",
                  "block_size", "transaction_volume", "miner_revenue",
                  "eth_tx_count", "eth_supply", "nvt_proxy",
                  "market_cap", "total_volume"],
    "sentiment": ["fg_value", "fg_classification", "fg_pct_change", "fg_ma",
                  "fg_momentum", "fg_is_", "cg_price_change", "cg_volume",
                  "cg_market_cap", "sentiment_"],
    "macro":     ["sp500", "dxy", "gold", "vix", "nasdaq", "bonds", "btc_dom",
                  "macro_", "cross_asset_"],
}


def get_features_to_drop(group: str, all_columns: list) -> list:
    """Return list of column names belonging to a feature group."""
    patterns = FEATURE_GROUPS.get(group, [])
    to_drop  = []
    for col in all_columns:
        for pattern in patterns:
            if col.startswith(pattern) or pattern in col:
                to_drop.append(col)
                break
    return list(set(to_drop))


def run_ablation_experiment(
    model_name:    str,
    asset:         str,
    interval:      str,
    horizon:       int,
    feature_group: str,
    max_epochs:    int = 100,
) -> dict:
    """
    Train model with a feature group removed and return test metrics.

    Args:
        model_name    : Model architecture
        asset         : Asset ticker
        interval      : Candle interval
        horizon       : Forecast horizon
        feature_group : Feature group to remove ('ltst', 'onchain', etc.)
        max_epochs    : Max training epochs

    Returns:
        dict of test metrics
    """
    from src.training.trainer import train_model

    # Load best params for this model
    params_file = (RESULTS_DIR / "tuning" /
                   f"{model_name}_{asset}_{interval}_h{horizon}_best_params.json")

    if params_file.exists():
        with open(params_file) as f:
            params = json.load(f)["best_params"]
    else:
        params = {}

    model_kwargs = {k: v for k, v in params.items()
                    if k in ["hidden_size","num_layers","dropout",
                             "d_model","nhead","num_encoder_layers",
                             "dim_feedforward","num_filters","kernel_size"]}

    logger.info(f"  Ablation: {model_name} {asset} | removing {feature_group}")

    try:
        results = train_model(
            model_name     = model_name,
            asset          = asset,
            interval       = interval,
            horizon        = horizon,
            seq_len        = int(params.get("seq_len", 60)),
            batch_size     = int(params.get("batch_size", 32)),
            lr             = float(params.get("lr", 1e-3)),
            weight_decay   = float(params.get("weight_decay", 1e-4)),
            max_epochs     = max_epochs,
            loss_name      = "combined",
            use_ltst       = (feature_group != "ltst"),
            model_kwargs   = model_kwargs,
        )
        metrics = results["test_metrics"].copy()
        metrics["ablation_group"] = feature_group
        metrics["model"]          = model_name
        return metrics

    except Exception as e:
        logger.error(f"  Ablation failed for {feature_group}: {e}")
        return {"ablation_group": feature_group, "model": model_name, "error": str(e)}


def run_full_ablation(
    model_name: str = "lstm",
    asset:      str = "BTC",
    interval:   str = "1d",
    horizon:    int = 1,
    max_epochs: int = 100,
    save:       bool = True,
) -> pd.DataFrame:
    """
    Run complete ablation study removing each feature group one at a time.

    Args:
        model_name : Model to test
        asset      : Asset to test on
        interval   : Candle interval
        horizon    : Forecast horizon
        max_epochs : Epochs per ablation run
        save       : Save results to CSV

    Returns:
        DataFrame with one row per ablation condition
    """
    conditions = ["full"] + list(FEATURE_GROUPS.keys())
    records    = []

    for condition in conditions:
        logger.info(f"\n  Ablation condition: {condition}")

        if condition == "full":
            # Run with all features (reload from saved results if available)
            run_name    = f"{model_name}_{asset}_{interval}_h{horizon}"
            result_path = RESULTS_DIR / f"{run_name}_results.csv"
            if result_path.exists():
                df = pd.read_csv(result_path)
                metrics = df.iloc[0].to_dict()
            else:
                metrics = run_ablation_experiment(
                    model_name, asset, interval, horizon, "full", max_epochs
                )
        else:
            metrics = run_ablation_experiment(
                model_name, asset, interval, horizon, condition, max_epochs
            )

        metrics["condition"] = condition
        records.append(metrics)

    results = pd.DataFrame(records)

    if save and not results.empty:
        save_path = RESULTS_DIR / f"ablation_{model_name}_{asset}_{interval}_h{horizon}.csv"
        results.to_csv(save_path, index=False)
        logger.success(f"  Ablation results saved → {save_path}")

    return results


def print_ablation_table(results: pd.DataFrame):
    """Print formatted ablation study results."""
    if results.empty:
        return

    metrics = ["rmse", "mape", "r2", "directional_accuracy"]
    metrics = [m for m in metrics if m in results.columns]

    # Get baseline (full) values
    baseline = results[results["condition"] == "full"].iloc[0] if "full" in results["condition"].values else None

    print(f"\n{'═'*70}")
    print(f"  Ablation Study Results")
    print(f"{'═'*70}")
    print(f"  {'CONDITION':<20}  {'RMSE':>8}  {'MAPE':>8}  {'R²':>8}  {'DA':>8}  {'ΔMAPE':>8}")
    print(f"  {'─'*64}")

    for _, row in results.iterrows():
        cond = row.get("condition", "?")
        rmse = row.get("rmse", np.nan)
        mape = row.get("mape", np.nan)
        r2   = row.get("r2",   np.nan)
        da   = row.get("directional_accuracy", np.nan)

        delta_mape = ""
        if baseline is not None and cond != "full":
            base_mape = float(baseline.get("mape", np.nan))
            if not np.isnan(mape) and not np.isnan(base_mape):
                delta = mape - base_mape
                delta_mape = f"{delta:+.2f}%"

        print(f"  {cond:<20}  {rmse:>8.0f}  {mape:>8.2f}  {r2:>8.4f}  "
              f"{da:>8.1f}  {delta_mape:>8}")

    print(f"  {'─'*64}")
    print(f"  ΔMAPE = change vs full model (positive = worse)\n")


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run feature ablation study")
    parser.add_argument("--model",    default="lstm",
                        choices=["lstm","gru","bilstm","cnn_lstm",
                                 "attention_lstm","transformer"])
    parser.add_argument("--asset",    default="BTC")
    parser.add_argument("--interval", default="1d")
    parser.add_argument("--horizon",  type=int, default=1)
    parser.add_argument("--epochs",   type=int, default=100)
    parser.add_argument("--quick",    action="store_true",
                        help="Only test ltst and onchain ablations")
    args = parser.parse_args()

    results = run_full_ablation(
        model_name = args.model,
        asset      = args.asset,
        interval   = args.interval,
        horizon    = args.horizon,
        max_epochs = args.epochs,
    )

    print_ablation_table(results)
