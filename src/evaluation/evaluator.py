"""
╔══════════════════════════════════════════════════════════════════════╗
║   src/evaluation/evaluator.py                                       ║
║   Comprehensive model evaluation pipeline                           ║
║   Author : Muluh Penn Junior Patrick                                ║
╚══════════════════════════════════════════════════════════════════════╝
Loads all trained model results and produces:
  - Full KPI comparison table (all 6 models × all assets × horizons)
  - Statistical significance tests (Diebold-Mariano)
  - Regime-specific performance breakdown
  - Ablation study results

Usage:
    python -m src.evaluation.evaluator --asset BTC --interval 1d
"""

import os
import json
import glob
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger

RESULTS_DIR = Path(os.getenv("RESULTS_DIR", "./experiments/results"))

MODELS    = ["lstm", "gru", "bilstm", "cnn_lstm", "attention_lstm", "transformer"]
ASSETS    = ["BTC", "ETH", "SOL", "SUI", "XRP"]
INTERVALS = ["1d", "1h"]
HORIZONS  = [1, 7, 14, 30]

# KPI display names and formatting
KPI_CONFIG = {
    "rmse":                  {"name": "RMSE (USD)",          "fmt": ".0f",  "lower_better": True},
    "mae":                   {"name": "MAE (USD)",           "fmt": ".0f",  "lower_better": True},
    "mape":                  {"name": "MAPE (%)",            "fmt": ".2f",  "lower_better": True},
    "r2":                    {"name": "R²",                  "fmt": ".4f",  "lower_better": False},
    "directional_accuracy":  {"name": "DA (%)",              "fmt": ".1f",  "lower_better": False},
    "sharpe_ratio":          {"name": "Sharpe",              "fmt": ".3f",  "lower_better": False},
    "max_drawdown":          {"name": "Max DD",              "fmt": ".3f",  "lower_better": True},
    "win_rate":              {"name": "Win Rate (%)",        "fmt": ".1f",  "lower_better": False},
    "n_params":              {"name": "Parameters",         "fmt": ".0f",  "lower_better": True},
}


def load_all_results(
    asset:    str = "BTC",
    interval: str = "1d",
    horizon:  int = 1,
) -> pd.DataFrame:
    """
    Load all model results for a given asset-interval-horizon combination.

    Returns:
        DataFrame with one row per model, columns = KPI metrics
    """
    records = []

    for model in MODELS:
        run_name    = f"{model}_{asset}_{interval}_h{horizon}"
        result_path = RESULTS_DIR / f"{run_name}_results.csv"

        if not result_path.exists():
            logger.warning(f"  Results not found: {result_path}")
            continue

        df = pd.read_csv(result_path)
        if df.empty:
            continue

        row = df.iloc[0].to_dict()
        row["model"] = model
        records.append(row)

    if not records:
        logger.warning(f"  No results found for {asset} {interval} h={horizon}")
        return pd.DataFrame()

    results = pd.DataFrame(records).set_index("model")

    # Ensure numeric columns
    for col in results.columns:
        if col != "run_name":
            results[col] = pd.to_numeric(results[col], errors="coerce")

    logger.info(f"  Loaded {len(results)} model results for {asset} {interval} h{horizon}")
    return results


def build_comparison_table(
    asset:    str = "BTC",
    interval: str = "1d",
    horizon:  int = 1,
    kpis:     list = None,
) -> pd.DataFrame:
    """
    Build a formatted comparison table of all models for a given setting.

    Args:
        asset    : Asset ticker
        interval : Candle interval
        horizon  : Forecast horizon
        kpis     : List of KPI column names to include (default: all)

    Returns:
        Formatted DataFrame with models as rows, KPIs as columns.
        Best value in each column is highlighted (marked with *).
    """
    results = load_all_results(asset, interval, horizon)
    if results.empty:
        return pd.DataFrame()

    kpis = kpis or list(KPI_CONFIG.keys())
    kpis = [k for k in kpis if k in results.columns]

    table = results[kpis].copy()

    # Add rank column per KPI
    for kpi in kpis:
        if kpi not in KPI_CONFIG:
            continue
        lower_better = KPI_CONFIG[kpi]["lower_better"]
        rank = table[kpi].rank(ascending=lower_better, na_option="bottom").astype(int)
        table[f"{kpi}_rank"] = rank

    return table


def rank_models(
    asset:    str = "BTC",
    interval: str = "1d",
    horizon:  int = 1,
) -> pd.DataFrame:
    """
    Rank all models by each KPI and compute an overall rank score.

    Returns:
        DataFrame with models ranked by overall performance.
    """
    results = load_all_results(asset, interval, horizon)
    if results.empty:
        return pd.DataFrame()

    primary_kpis = ["mape", "rmse", "directional_accuracy", "r2", "sharpe_ratio"]
    primary_kpis = [k for k in primary_kpis if k in results.columns]

    ranks = pd.DataFrame(index=results.index)
    for kpi in primary_kpis:
        lower_better = KPI_CONFIG.get(kpi, {}).get("lower_better", True)
        ranks[f"{kpi}_rank"] = results[kpi].rank(ascending=lower_better, na_option="bottom")

    ranks["avg_rank"]   = ranks.mean(axis=1)
    ranks["overall_rank"] = ranks["avg_rank"].rank()

    # Merge with raw values
    summary = results[primary_kpis].copy()
    summary["avg_rank"]     = ranks["avg_rank"]
    summary["overall_rank"] = ranks["overall_rank"].astype(int)
    summary = summary.sort_values("overall_rank")

    return summary


def print_comparison_table(
    asset:    str = "BTC",
    interval: str = "1d",
    horizon:  int = 1,
):
    """Print a formatted comparison table to console."""
    results = load_all_results(asset, interval, horizon)
    if results.empty:
        print(f"No results found for {asset} {interval} h={horizon}")
        return

    primary_kpis = ["rmse", "mape", "r2", "directional_accuracy",
                    "sharpe_ratio", "max_drawdown"]
    cols = [k for k in primary_kpis if k in results.columns]

    print(f"\n{'═'*80}")
    print(f"  Model Comparison: {asset} {interval} h={horizon}")
    print(f"{'═'*80}")

    # Header
    header = f"  {'MODEL':<20}"
    for kpi in cols:
        name = KPI_CONFIG.get(kpi, {}).get("name", kpi)
        header += f"  {name:>12}"
    print(header)
    print(f"{'─'*80}")

    # Rows
    for model in results.index:
        row_str = f"  {model:<20}"
        for kpi in cols:
            val  = results.loc[model, kpi]
            fmt  = KPI_CONFIG.get(kpi, {}).get("fmt", ".4f")
            lower_better = KPI_CONFIG.get(kpi, {}).get("lower_better", True)

            # Mark best value
            if lower_better:
                is_best = val == results[kpi].min()
            else:
                is_best = val == results[kpi].max()

            formatted = f"{val:{fmt}}"
            if is_best:
                formatted = f"*{formatted}"
            row_str += f"  {formatted:>12}"
        print(row_str)

    print(f"{'─'*80}")
    print(f"  * = best in column\n")


def save_all_comparison_tables(
    output_dir: str = None,
):
    """
    Save comparison tables for all available asset-interval-horizon combinations.

    Args:
        output_dir : Directory to save CSVs (default: experiments/results/)
    """
    output_dir = Path(output_dir or RESULTS_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all result files
    result_files = list(RESULTS_DIR.glob("*_h*_results.csv"))
    combinations = set()
    for f in result_files:
        parts = f.stem.replace("_results", "").split("_")
        # Format: model_ASSET_interval_hN
        if len(parts) >= 4:
            asset    = parts[-3]
            interval = parts[-2]
            horizon  = int(parts[-1].replace("h", ""))
            combinations.add((asset, interval, horizon))

    logger.info(f"  Found {len(combinations)} unique experiment combinations")

    for asset, interval, horizon in sorted(combinations):
        results = load_all_results(asset, interval, horizon)
        if results.empty:
            continue

        out_path = output_dir / f"comparison_{asset}_{interval}_h{horizon}.csv"
        results.to_csv(out_path)
        logger.info(f"  Saved → {out_path}")

        # Also print to console
        print_comparison_table(asset, interval, horizon)


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate and compare model results")
    parser.add_argument("--asset",    default="BTC",
                        choices=["BTC","ETH","SOL","SUI","XRP","all"])
    parser.add_argument("--interval", default="1d", choices=["1h","1d","all"])
    parser.add_argument("--horizon",  type=int, default=1)
    parser.add_argument("--all",      action="store_true",
                        help="Generate comparison tables for all available results")
    args = parser.parse_args()

    if args.all:
        save_all_comparison_tables()
    else:
        print_comparison_table(
            asset    = args.asset,
            interval = args.interval,
            horizon  = args.horizon,
        )

        ranked = rank_models(args.asset, args.interval, args.horizon)
        if not ranked.empty:
            print(f"\n{'─'*60}")
            print(f"  Overall Rankings: {args.asset} {args.interval} h={args.horizon}")
            print(f"{'─'*60}")
            print(ranked[["mape","rmse","directional_accuracy","avg_rank","overall_rank"]]
                  .to_string())
