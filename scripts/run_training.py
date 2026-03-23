#!/usr/bin/env python3
"""
Train a single model.
Usage: python scripts/run_training.py --model lstm --asset BTC --horizon 1d
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.seed   import set_seed
from src.utils.device import get_device
from src.utils.logger import setup_logger
from src.models       import get_model


def parse_args():
    p = argparse.ArgumentParser(description="Train a forecasting model")
    p.add_argument("--model",   default="lstm",
                   choices=["lstm","gru","bilstm","cnn_lstm","attention_lstm","transformer"])
    p.add_argument("--asset",   default="BTC",
                   choices=["BTC","ETH","SOL","SUI","XRP"])
    p.add_argument("--horizon", default="1d",
                   choices=["1h","1d","7d","30d"])
    p.add_argument("--config",  default=None,
                   help="Path to YAML config (optional override)")
    p.add_argument("--seed",    type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    setup_logger()
    set_seed(args.seed)
    device = get_device()

    print(f"\n  🚀  Training  | Model: {args.model} | Asset: {args.asset} | Horizon: {args.horizon}\n")

    # TODO: Load data, build DataModule, instantiate model, run Trainer
    # Full implementation in Phase 5 of the thesis roadmap
    model = get_model(args.model, input_size=50)  # placeholder input_size
    print(f"  ✔   Model instantiated: {model.__class__.__name__}")
    print(f"  ✔   Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("\n  ⚙️   Data pipeline and training loop coming in Phase 5...")


if __name__ == "__main__":
    main()
