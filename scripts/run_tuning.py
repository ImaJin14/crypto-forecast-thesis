#!/usr/bin/env python3
"""
Run Optuna hyperparameter search for a model.

Usage:
    python scripts/run_tuning.py --model lstm --asset BTC --trials 30 --epochs 50
    python scripts/run_tuning.py --model gru  --asset ETH --trials 20 --no-resume
"""
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.logger import setup_logger

def parse_args():
    p = argparse.ArgumentParser(description="Run Optuna hyperparameter tuning")
    p.add_argument("--model",     default="lstm",
                   choices=["lstm","gru","bilstm","cnn_lstm","attention_lstm","transformer"])
    p.add_argument("--asset",     default="BTC",  choices=["BTC","ETH","SOL","SUI","XRP"])
    p.add_argument("--interval",  default="1d",   choices=["1h","1d"])
    p.add_argument("--horizon",   type=int, default=1)
    p.add_argument("--trials",    type=int, default=30)
    p.add_argument("--epochs",    type=int, default=50)
    p.add_argument("--no-resume", action="store_true",
                   help="Start fresh study (ignore existing SQLite DB)")
    return p.parse_args()

def main():
    args = parse_args()
    setup_logger()

    # Import here to avoid circular imports at module level
    from src.tuning.optuna_study import run_study

    run_study(
        model_name = args.model,
        asset      = args.asset,
        interval   = args.interval,
        horizon    = args.horizon,
        n_trials   = args.trials,
        max_epochs = args.epochs,
        resume     = not args.no_resume,
    )

if __name__ == "__main__":
    main()
