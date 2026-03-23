#!/usr/bin/env python3
"""Run Optuna hyperparameter optimization for a given model."""
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

p = argparse.ArgumentParser()
p.add_argument("--model",  default="lstm")
p.add_argument("--asset",  default="BTC")
p.add_argument("--trials", type=int, default=50)
args = p.parse_args()
print(f"\n  🔍  Optuna tuning: {args.model} on {args.asset} — {args.trials} trials")
print("  Full implementation in Phase 6 of the roadmap.\n")
