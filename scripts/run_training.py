#!/usr/bin/env python3
"""
Train a single model with the best hyperparameters.

Usage:
    python scripts/run_training.py --model lstm --asset BTC --interval 1d --horizon 1
    python scripts/run_training.py --model gru  --asset ETH --interval 1d --horizon 7
"""
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.seed   import set_seed
from src.utils.logger import setup_logger

def parse_args():
    p = argparse.ArgumentParser(description="Train a forecasting model")
    p.add_argument("--model",    default="lstm",
                   choices=["lstm","gru","bilstm","cnn_lstm","attention_lstm","transformer"])
    p.add_argument("--asset",    default="BTC",  choices=["BTC","ETH","SOL","SUI","XRP"])
    p.add_argument("--interval", default="1d",   choices=["1h","1d"])
    p.add_argument("--horizon",  type=int, default=1)
    p.add_argument("--epochs",   type=int, default=200)
    p.add_argument("--seed",     type=int, default=42)
    p.add_argument("--use-best-params", action="store_true",
                   help="Load best params from Optuna tuning results")
    return p.parse_args()

def main():
    args = parse_args()
    setup_logger()
    set_seed(args.seed)

    from src.training.trainer import train_model

    # Load best params if available
    model_kwargs, lr, weight_decay, seq_len, batch_size = {}, 1e-3, 1e-4, 60, 32
    if args.use_best_params:
        params_file = (Path("experiments/results/tuning") /
                       f"{args.model}_{args.asset}_{args.interval}_h{args.horizon}_best_params.json")
        if params_file.exists():
            with open(params_file) as f:
                p = json.load(f)["best_params"]
            seq_len      = int(p.get("seq_len", seq_len))
            batch_size   = int(p.get("batch_size", batch_size))
            lr           = float(p.get("lr", lr))
            weight_decay = float(p.get("weight_decay", weight_decay))
            model_kwargs = {k: v for k, v in p.items()
                            if k in ["hidden_size","num_layers","dropout",
                                     "d_model","nhead","num_encoder_layers",
                                     "dim_feedforward","num_filters","kernel_size"]}
            print(f"  ✔  Loaded best params from {params_file}")
        else:
            print(f"  ⚠  No best params found at {params_file}. Using defaults.")

    results = train_model(
        model_name   = args.model,
        asset        = args.asset,
        interval     = args.interval,
        horizon      = args.horizon,
        seq_len      = seq_len,
        batch_size   = batch_size,
        lr           = lr,
        weight_decay = weight_decay,
        max_epochs   = args.epochs,
        loss_name    = "combined",
        model_kwargs = model_kwargs,
    )

    m = results["test_metrics"]
    print(f"\n  Results: RMSE=${m['rmse']:,.0f}  MAPE={m['mape']:.2f}%  "
          f"DA={m['directional_accuracy']:.1f}%  R²={m.get('r2', float('nan')):.4f}")

if __name__ == "__main__":
    main()
