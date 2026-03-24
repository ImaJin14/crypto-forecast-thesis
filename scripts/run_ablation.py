#!/usr/bin/env python3
"""
Run feature ablation study.

Usage:
    python scripts/run_ablation.py --model lstm --asset BTC
    python scripts/run_ablation.py --model gru  --asset BTC --epochs 50
"""
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

def parse_args():
    p = argparse.ArgumentParser(description="Run feature ablation study")
    p.add_argument("--model",    default="lstm",
                   choices=["lstm","gru","bilstm","cnn_lstm","attention_lstm","transformer"])
    p.add_argument("--asset",    default="BTC",  choices=["BTC","ETH","SOL","SUI","XRP"])
    p.add_argument("--interval", default="1d",   choices=["1h","1d"])
    p.add_argument("--horizon",  type=int, default=1)
    p.add_argument("--epochs",   type=int, default=100)
    return p.parse_args()

def main():
    args = parse_args()
    from src.evaluation.ablation_study import run_full_ablation, print_ablation_table
    results = run_full_ablation(
        model_name = args.model,
        asset      = args.asset,
        interval   = args.interval,
        horizon    = args.horizon,
        max_epochs = args.epochs,
        save       = True,
    )
    print_ablation_table(results)

if __name__ == "__main__":
    main()
