#!/usr/bin/env python3
"""
Evaluate all trained models and generate comparison tables.

Usage:
    python scripts/run_evaluation.py                    # BTC 1d h=1
    python scripts/run_evaluation.py --asset ETH --horizon 7
    python scripts/run_evaluation.py --all              # all available results
"""
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

def parse_args():
    p = argparse.ArgumentParser(description="Evaluate all models")
    p.add_argument("--asset",    default="BTC",  choices=["BTC","ETH","SOL","SUI","XRP"])
    p.add_argument("--interval", default="1d",   choices=["1h","1d"])
    p.add_argument("--horizon",  type=int, default=1)
    p.add_argument("--all",      action="store_true", help="Process all available results")
    p.add_argument("--dm",       action="store_true",
                   help="Also run Diebold-Mariano significance tests")
    return p.parse_args()

def main():
    args = parse_args()

    from src.evaluation.evaluator import (
        print_comparison_table, rank_models, save_all_comparison_tables
    )

    if args.all:
        save_all_comparison_tables()
    else:
        print_comparison_table(asset=args.asset, interval=args.interval, horizon=args.horizon)

        ranked = rank_models(args.asset, args.interval, args.horizon)
        if not ranked.empty:
            print(f"\n  Rankings:")
            print(ranked[["mape","rmse","directional_accuracy","avg_rank","overall_rank"]])

    if args.dm:
        from src.evaluation.diebold_mariano import (
            load_errors_from_results, print_dm_results
        )
        errors = load_errors_from_results(
            asset=args.asset, interval=args.interval, horizon=args.horizon
        )
        if len(errors) >= 2:
            print_dm_results(errors, h=args.horizon)
        else:
            print("  ⚠  Run save_predictions.py first to enable DM testing.")

if __name__ == "__main__":
    main()
