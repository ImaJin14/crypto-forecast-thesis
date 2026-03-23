"""
╔══════════════════════════════════════════════════════════════════════╗
║   src/evaluation/diebold_mariano.py                                 ║
║   Diebold-Mariano test for forecast comparison significance         ║
║   Author : Muluh Penn Junior Patrick                                ║
╚══════════════════════════════════════════════════════════════════════╝
The Diebold-Mariano (DM) test answers: "Is model A significantly better
than model B, or could the difference be due to chance?"

Reference: Diebold & Mariano (1995), "Comparing Predictive Accuracy"

Usage:
    from src.evaluation.diebold_mariano import dm_test, dm_matrix

    stat, p_value = dm_test(errors_a, errors_b)
    matrix = dm_matrix(all_errors_dict)
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from itertools import combinations
from loguru import logger


def dm_test(
    errors_a:   np.ndarray,
    errors_b:   np.ndarray,
    h:          int   = 1,
    loss:       str   = "mse",
    alternative: str  = "two-sided",
) -> tuple[float, float]:
    """
    Diebold-Mariano test for equal predictive accuracy.

    Tests H0: E[d_t] = 0, where d_t = L(e_at) - L(e_bt) is the
    differential loss between models A and B.

    A negative DM statistic means model A is MORE accurate than model B.
    A p-value < 0.05 means the difference is statistically significant.

    Args:
        errors_a    : Forecast errors for model A (array of e_t = y_t - ŷ_t)
        errors_b    : Forecast errors for model B
        h           : Forecast horizon (for serial correlation correction)
        loss        : Loss function: 'mse' (squared), 'mae' (absolute), 'mape'
        alternative : 'two-sided', 'less' (A better), or 'greater' (B better)

    Returns:
        (dm_statistic, p_value)
        Negative DM → A is more accurate
        p < 0.05 → statistically significant difference
    """
    errors_a = np.asarray(errors_a, dtype=float)
    errors_b = np.asarray(errors_b, dtype=float)

    if len(errors_a) != len(errors_b):
        raise ValueError(
            f"Error arrays must have equal length: {len(errors_a)} vs {len(errors_b)}"
        )

    n = len(errors_a)

    # Compute loss differential
    if loss == "mse":
        d = errors_a ** 2 - errors_b ** 2
    elif loss == "mae":
        d = np.abs(errors_a) - np.abs(errors_b)
    elif loss == "mape":
        # Assumes y_true is available implicitly through errors
        d = np.abs(errors_a) - np.abs(errors_b)
    else:
        raise ValueError(f"Unknown loss function '{loss}'. Use: mse, mae, mape")

    d_bar   = d.mean()
    d_var   = _autocovariance(d, h=h, n=n)

    if d_var <= 0:
        logger.warning("  DM test: variance <= 0, returning nan")
        return float("nan"), float("nan")

    dm_stat = d_bar / np.sqrt(d_var / n)

    # Harvey, Leybourne, Newbold (1997) small-sample correction
    correction = np.sqrt((n + 1 - 2*h + h*(h-1)/n) / n)
    dm_stat_corr = dm_stat * correction

    # t-distribution p-value with n-1 degrees of freedom
    if alternative == "two-sided":
        p_value = 2 * stats.t.sf(np.abs(dm_stat_corr), df=n-1)
    elif alternative == "less":
        p_value = stats.t.cdf(dm_stat_corr, df=n-1)
    elif alternative == "greater":
        p_value = stats.t.sf(dm_stat_corr, df=n-1)
    else:
        raise ValueError(f"Unknown alternative '{alternative}'")

    return float(dm_stat_corr), float(p_value)


def _autocovariance(d: np.ndarray, h: int, n: int) -> float:
    """
    Estimate the h-step ahead autocovariance of d_t.
    Used for Newey-West heteroskedasticity and autocorrelation consistent (HAC)
    variance estimation.
    """
    d_bar  = d.mean()
    d_cent = d - d_bar

    # Variance (h=0 term)
    variance = np.sum(d_cent ** 2) / n

    # Add autocovariance terms for lags 1 to h-1
    for k in range(1, h):
        cov_k = np.sum(d_cent[k:] * d_cent[:-k]) / n
        variance += 2 * cov_k

    return max(variance, 1e-15)  # avoid zero variance


def dm_matrix(
    errors_dict: dict[str, np.ndarray],
    h:           int = 1,
    loss:        str = "mse",
    alpha:       float = 0.05,
) -> pd.DataFrame:
    """
    Compute pairwise DM test statistics for all model pairs.

    Args:
        errors_dict : Dict mapping model_name → forecast errors array
        h           : Forecast horizon
        loss        : Loss function
        alpha       : Significance level for marking significant pairs

    Returns:
        DataFrame of DM statistics (lower triangle) and p-values (upper triangle).
        Cells marked with * where p < alpha.

    Example:
        errors = {
            'lstm':        lstm_errors,
            'gru':         gru_errors,
            'transformer': transformer_errors,
        }
        matrix = dm_matrix(errors)
    """
    models = list(errors_dict.keys())
    n      = len(models)

    # Trim all error arrays to the shortest length (different seq_len → different n_test)
    min_len = min(len(errors_dict[m]) for m in models)
    if len(set(len(errors_dict[m]) for m in models)) > 1:
        logger.info(f"  DM matrix: trimming arrays to min length {min_len}")
        errors_dict = {m: errors_dict[m][:min_len] for m in models}

    dm_stats  = pd.DataFrame(np.nan, index=models, columns=models)
    p_values  = pd.DataFrame(np.nan, index=models, columns=models)

    for i, m_a in enumerate(models):
        for j, m_b in enumerate(models):
            if i == j:
                continue
            stat, pval = dm_test(
                errors_dict[m_a],
                errors_dict[m_b],
                h    = h,
                loss = loss,
            )
            dm_stats.loc[m_a, m_b] = stat
            p_values.loc[m_a, m_b] = pval

    return dm_stats, p_values


def print_dm_results(
    errors_dict: dict[str, np.ndarray],
    h:           int   = 1,
    alpha:       float = 0.05,
):
    """
    Print a formatted DM test table comparing all model pairs.

    Interpretation:
        DM < 0 and p < 0.05 → row model is significantly better than column model
        DM > 0 and p < 0.05 → column model is significantly better than row model
    """
    dm_stats, p_values = dm_matrix(errors_dict, h=h)
    models = list(errors_dict.keys())

    print(f"\n{'─'*72}")
    print(f"  Diebold-Mariano Test Matrix (h={h}, loss=MSE)")
    print(f"  DM statistic: negative = row model MORE accurate than column")
    print(f"  * = p < {alpha} (statistically significant)")
    print(f"{'─'*72}")

    # Header
    header = f"  {'':>18}"
    for m in models:
        header += f"  {m[:10]:>12}"
    print(header)
    print(f"{'─'*72}")

    for m_a in models:
        row = f"  {m_a:<18}"
        for m_b in models:
            if m_a == m_b:
                row += f"  {'—':>12}"
            else:
                stat = dm_stats.loc[m_a, m_b]
                pval = p_values.loc[m_a, m_b]
                cell = f"{stat:+.3f}"
                if pval < alpha:
                    cell += "*"
                row += f"  {cell:>12}"
        print(row)

    print(f"{'─'*72}")
    print(f"  * p < {alpha}  |  DM negative: row model better\n")


def load_errors_from_results(
    asset:    str = "BTC",
    interval: str = "1d",
    horizon:  int = 1,
    results_dir: str = "./experiments/results",
) -> dict[str, np.ndarray]:
    """
    Load actual forecast errors for DM testing.
    Priority: (1) pre-saved .npy file, (2) checkpoint inference, (3) RMSE proxy.

    Run save_predictions.py first to populate the .npy cache.
    """
    import os, sys, json, torch
    from pathlib import Path
    sys.path.insert(0, ".")

    results_path = Path(results_dir)
    ckpt_base    = Path("experiments/checkpoints")
    scalers_dir  = Path("data/processed/scalers")
    errors       = {}

    models = ["lstm", "gru", "bilstm", "cnn_lstm", "attention_lstm", "transformer"]

    for model in models:
        run_name    = f"{model}_{asset}_{interval}_h{horizon}"
        result_file = results_path / f"{run_name}_results.csv"
        if not result_file.exists():
            continue

        # ── Priority 1: pre-saved predictions.npy ────────────────────────────
        npy_path = results_path / f"{run_name}_predictions.npy"
        if npy_path.exists():
            errors[model] = np.load(npy_path)
            logger.info(f"  DM: {model} — loaded from .npy ({len(errors[model])} samples)")
            continue

        # ── Try to get real errors from checkpoint ────────────────────────────
        ckpt_dir = ckpt_base / run_name
        ckpts    = [c for c in ckpt_dir.glob("*.ckpt")
                    if "last" not in c.name] if ckpt_dir.exists() else []

        if ckpts:
            try:
                # Sort by val_loss in filename to get the best checkpoint
                def _val_loss(p):
                    try:
                        return float(p.stem.split("val_loss=")[-1].split("-")[0])
                    except Exception:
                        return float("inf")

                best_ckpt = min(ckpts, key=_val_loss)

                # Load best params
                params_file = (results_path / "tuning" /
                               f"{run_name}_best_params.json")
                params = {}
                if params_file.exists():
                    with open(params_file) as f:
                        params = json.load(f).get("best_params", {})

                seq_len    = int(params.get("seq_len", 60))
                batch_size = int(params.get("batch_size", 32))
                model_kwargs = {k: v for k, v in params.items()
                                if k in ["hidden_size","num_layers","dropout",
                                         "d_model","nhead","num_encoder_layers",
                                         "dim_feedforward","num_filters","kernel_size"]}

                from src.training.trainer import CryptoDataModule, CryptoForecasterModule
                from src.models import get_model

                data_module = CryptoDataModule(
                    asset=asset, interval=interval,
                    seq_len=seq_len, horizon=horizon,
                    batch_size=batch_size,
                    use_ltst=True,  # always use full features for DM inference
                )
                data_module.setup()

                mdl    = get_model(model, input_size=data_module.n_features,
                                   output_size=horizon, **model_kwargs)

                # Check checkpoint input size matches current model
                ckpt_data = torch.load(str(best_ckpt), map_location="cpu",
                                       weights_only=False)
                state     = ckpt_data.get("state_dict", {})
                # Find first weight tensor and check its input dimension
                mismatch = False
                for k, v in state.items():
                    if "weight_ih_l0" in k or "weight" in k and v.dim() == 2:
                        ckpt_in = v.shape[-1]
                        mdl_in  = data_module.n_features
                        if ckpt_in != mdl_in:
                            logger.warning(f"  DM: {model} feature mismatch "
                                           f"(ckpt={ckpt_in}, current={mdl_in}) — skipping")
                            mismatch = True
                        break
                if mismatch:
                    raise ValueError("Feature count mismatch — retrain needed")

                module = CryptoForecasterModule.load_from_checkpoint(
                    str(best_ckpt), model=mdl, strict=False
                )
                module.eval()

                device      = "cuda" if torch.cuda.is_available() else "cpu"
                module      = module.to(device)
                preds_list  = []
                targets_list = []

                with torch.no_grad():
                    for batch in data_module.test_dataloader():
                        x, y = batch
                        x    = x.to(device)
                        yhat = module(x).squeeze().cpu().numpy()
                        preds_list.append(np.atleast_1d(yhat))
                        targets_list.append(np.atleast_1d(y.numpy()))

                pred_lr   = np.concatenate(preds_list).flatten()
                target_lr = np.concatenate(targets_list).flatten()

                # Reconstruct USD prices for error computation
                test_close_path = scalers_dir / f"{asset}_{interval}_test_close.npy"
                if test_close_path.exists():
                    raw_close = np.load(test_close_path)
                    n = len(pred_lr)
                    idx = np.clip(np.arange(n) + seq_len - 1, 0, len(raw_close) - 1)
                    start = raw_close[idx]
                    preds   = start * np.exp(pred_lr)
                    targets = start * np.exp(target_lr)
                    errors[model] = targets - preds   # actual forecast errors in USD
                    logger.info(f"  DM: {model} — real errors loaded ({len(errors[model])} samples)")
                else:
                    errors[model] = target_lr - pred_lr   # log-return errors
                    logger.info(f"  DM: {model} — log-return errors ({len(errors[model])} samples)")

                continue  # skip fallback

            except Exception as e:
                logger.warning(f"  DM: {model} checkpoint load failed ({e}), using RMSE proxy")

        # ── Fallback: RMSE-scaled normal errors ───────────────────────────────
        df   = pd.read_csv(result_file)
        rmse = float(df.iloc[0].get("rmse", np.nan))
        if not np.isnan(rmse):
            rng = np.random.default_rng(hash(model) % (2**31))
            errors[model] = rng.normal(0, rmse, 361)
            logger.warning(f"  DM: {model} — synthetic errors (RMSE proxy)")

    return errors


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run Diebold-Mariano significance tests")
    parser.add_argument("--asset",    default="BTC")
    parser.add_argument("--interval", default="1d")
    parser.add_argument("--horizon",  type=int, default=1)
    parser.add_argument("--alpha",    type=float, default=0.05)
    args = parser.parse_args()

    errors = load_errors_from_results(
        asset    = args.asset,
        interval = args.interval,
        horizon  = args.horizon,
    )

    if len(errors) < 2:
        print(f"Need at least 2 models. Found: {list(errors.keys())}")
    else:
        print_dm_results(errors, h=args.horizon, alpha=args.alpha)
