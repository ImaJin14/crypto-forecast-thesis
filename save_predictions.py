"""
save_predictions.py
───────────────────
Re-runs test inference for each model using its best checkpoint and saves
prediction arrays to experiments/results/{run_name}_predictions.npy

Run once after training to enable proper Diebold-Mariano testing:
    python save_predictions.py --asset BTC --interval 1d --horizon 1

This is a standalone utility; it does NOT modify training code.
"""

import sys, os, json, argparse
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, ".")

DATA_DIR    = Path(os.getenv("DATA_DIR",    "./data"))
RESULTS_DIR = Path(os.getenv("RESULTS_DIR", "./experiments/results"))
CKPT_BASE   = Path("experiments/checkpoints")

MODELS = ["lstm", "gru", "bilstm", "cnn_lstm", "attention_lstm", "transformer"]


def save_predictions_for_model(model_name, asset, interval, horizon):
    run_name   = f"{model_name}_{asset}_{interval}_h{horizon}"
    pred_path  = RESULTS_DIR / f"{run_name}_predictions.npy"

    if pred_path.exists():
        print(f"  ✔  {model_name}: predictions already saved")
        return True

    # Load best params
    params_file = RESULTS_DIR / "tuning" / f"{run_name}_best_params.json"
    if not params_file.exists():
        print(f"  ⚠  {model_name}: no best_params.json, skipping")
        return False

    with open(params_file) as f:
        params = json.load(f).get("best_params", {})

    seq_len      = int(params.get("seq_len", 60))
    batch_size   = int(params.get("batch_size", 32))
    model_kwargs = {k: v for k, v in params.items()
                    if k in ["hidden_size","num_layers","dropout",
                             "d_model","nhead","num_encoder_layers",
                             "dim_feedforward","num_filters","kernel_size"]}

    # Find best checkpoint
    ckpt_dir = CKPT_BASE / run_name
    ckpts    = [c for c in ckpt_dir.glob("*.ckpt")
                if "last" not in c.name] if ckpt_dir.exists() else []

    if not ckpts:
        print(f"  ⚠  {model_name}: no checkpoint found in {ckpt_dir}")
        return False

    def _val_loss(p):
        try:
            return float(p.stem.split("val_loss=")[-1].split("-")[0])
        except Exception:
            return float("inf")

    best_ckpt = min(ckpts, key=_val_loss)

    try:
        from src.training.trainer import CryptoDataModule, CryptoForecasterModule
        from src.models import get_model

        dm = CryptoDataModule(
            asset=asset, interval=interval,
            seq_len=seq_len, horizon=horizon,
            batch_size=batch_size, use_ltst=True,
        )
        dm.setup()
        n_feats = dm.n_features

        # Check feature count vs checkpoint
        ckpt_data = torch.load(str(best_ckpt), map_location="cpu",
                               weights_only=False)
        state = ckpt_data.get("state_dict", {})
        for k, v in state.items():
            if ("weight_ih_l0" in k or ("weight" in k and v.dim() == 2)):
                ckpt_in = v.shape[-1]
                if ckpt_in != n_feats:
                    print(f"  ✗  {model_name}: feature mismatch "
                          f"(ckpt={ckpt_in}, current={n_feats}) — retrain needed")
                    return False
                break

        mdl    = get_model(model_name, input_size=n_feats,
                           output_size=horizon, **model_kwargs)
        module = CryptoForecasterModule.load_from_checkpoint(
            str(best_ckpt), model=mdl, strict=False
        )
        module.eval()
        device  = "cuda" if torch.cuda.is_available() else "cpu"
        module  = module.to(device)

        preds_list   = []
        targets_list = []
        with torch.no_grad():
            for x, y in dm.test_dataloader():
                x    = x.to(device)
                yhat = module(x).squeeze().cpu().numpy()
                preds_list.append(np.atleast_1d(yhat))
                targets_list.append(np.atleast_1d(y.numpy()))

        pred_lr   = np.concatenate(preds_list).flatten()
        target_lr = np.concatenate(targets_list).flatten()

        # Reconstruct USD errors
        test_close_path = DATA_DIR / "processed" / "scalers" / \
                          f"{asset}_{interval}_test_close.npy"
        if test_close_path.exists():
            raw = np.load(test_close_path)
            n   = len(pred_lr)
            idx = np.clip(np.arange(n) + seq_len - 1, 0, len(raw) - 1)
            preds   = raw[idx] * np.exp(pred_lr)
            targets = raw[idx] * np.exp(target_lr)
            errors  = targets - preds
        else:
            errors = target_lr - pred_lr   # fallback: log-return errors

        np.save(pred_path, errors)
        print(f"  ✔  {model_name}: saved {len(errors)} errors → {pred_path.name}")
        return True

    except Exception as e:
        print(f"  ✗  {model_name}: {e}")
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset",    default="BTC")
    parser.add_argument("--interval", default="1d")
    parser.add_argument("--horizon",  type=int, default=1)
    args = parser.parse_args()

    print(f"\nSaving prediction arrays: {args.asset} {args.interval} h={args.horizon}")
    print("─" * 56)
    ok = 0
    for m in MODELS:
        if save_predictions_for_model(m, args.asset, args.interval, args.horizon):
            ok += 1
    print(f"─" * 56)
    print(f"  Done: {ok}/{len(MODELS)} models saved\n")
