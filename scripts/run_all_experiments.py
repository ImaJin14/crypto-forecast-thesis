#!/usr/bin/env python3
"""
Orchestrate all 120 experiments (6 models × 5 assets × 4 horizons).
Runs tuning + full training for each combination sequentially.

Usage:
    python scripts/run_all_experiments.py                  # full 120 runs
    python scripts/run_all_experiments.py --asset BTC      # BTC only (24 runs)
    python scripts/run_all_experiments.py --dry-run        # print plan only
"""
import argparse, json, sys, subprocess
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

MODELS    = ["lstm", "gru", "bilstm", "cnn_lstm", "attention_lstm", "transformer"]
ASSETS    = ["BTC", "ETH", "SOL", "SUI", "XRP"]
HORIZONS  = [1, 7, 14, 30]
INTERVAL  = "1d"

def parse_args():
    p = argparse.ArgumentParser(description="Run all 120 experiments")
    p.add_argument("--asset",    default=None, choices=ASSETS + [None])
    p.add_argument("--model",    default=None, choices=MODELS + [None])
    p.add_argument("--trials",   type=int, default=15,
                   help="Optuna trials per study (default 15 for overnight runs)")
    p.add_argument("--epochs",   type=int, default=30,
                   help="Epochs per tuning trial (default 30)")
    p.add_argument("--dry-run",  action="store_true", help="Print plan only")
    return p.parse_args()

def main():
    args    = parse_args()
    assets  = [args.asset] if args.asset  else ASSETS
    models  = [args.model] if args.model  else MODELS
    total   = len(assets) * len(models) * len(HORIZONS)

    print(f"\n  Experiment plan: {len(models)} models × {len(assets)} assets "
          f"× {len(HORIZONS)} horizons = {total} runs")
    print(f"  Trials/study: {args.trials}  |  Epochs/trial: {args.epochs}\n")

    if args.dry_run:
        for asset in assets:
            for horizon in HORIZONS:
                for model in models:
                    print(f"  [{asset}] h={horizon:2d}  {model}")
        return

    done = 0
    for asset in assets:
        for horizon in HORIZONS:
            for model in models:
                print(f"\n{'─'*60}")
                print(f"  [{done+1}/{total}] Tuning   {model} | {asset} 1d h={horizon}")
                subprocess.run([
                    sys.executable, "scripts/run_tuning.py",
                    "--model", model, "--asset", asset,
                    "--interval", INTERVAL, "--horizon", str(horizon),
                    "--trials", str(args.trials), "--epochs", str(args.epochs),
                    "--no-resume",
                ])
                print(f"  [{done+1}/{total}] Training {model} | {asset} 1d h={horizon}")
                subprocess.run([
                    sys.executable, "scripts/run_training.py",
                    "--model", model, "--asset", asset,
                    "--interval", INTERVAL, "--horizon", str(horizon),
                    "--use-best-params",
                ])
                done += 1

    print(f"\n  ✅  All {total} experiments complete.")

if __name__ == "__main__":
    main()
