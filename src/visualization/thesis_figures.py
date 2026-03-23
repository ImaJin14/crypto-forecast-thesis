"""
╔══════════════════════════════════════════════════════════════════════╗
║   src/visualization/thesis_figures.py                               ║
║   Publication-quality thesis figures                                ║
║   Author : Muluh Penn Junior Patrick                                ║
╚══════════════════════════════════════════════════════════════════════╝
Generates all figures for the thesis chapters.

Usage:
    python -m src.visualization.thesis_figures          # all figures
    python -m src.visualization.thesis_figures --fig 1  # specific figure

Output: experiments/figures/  (PDF + PNG for each figure)

Figures produced:
    Fig 1  — Training curves (val_loss vs epoch, all 6 models)
    Fig 2  — BTC price prediction vs actual (test period, GRU model)
    Fig 3  — Model comparison bar chart (MAPE + RMSE)
    Fig 4  — DM significance heatmap
    Fig 5  — Ablation study MAPE impact
    Fig 6  — Feature category overview (pie chart of 149 features)
"""

import os
import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

# ─── Paths ────────────────────────────────────────────────────────────────────

ROOT        = Path(".")
RESULTS_DIR = ROOT / "experiments" / "results"
FIG_DIR     = ROOT / "experiments" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ─── Style ────────────────────────────────────────────────────────────────────

MODELS = ["lstm", "gru", "bilstm", "cnn_lstm", "attention_lstm", "transformer"]
LABELS = ["LSTM", "GRU", "BiLSTM", "CNN-LSTM", "Attention-LSTM", "Transformer"]
COLORS = ["#3266AD", "#1D9E75", "#D85A30", "#BA7517", "#7F77DD", "#888780"]

PLT_STYLE = {
    "font.family":       "DejaVu Sans",
    "font.size":         11,
    "axes.titlesize":    12,
    "axes.labelsize":    11,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.3,
    "grid.linestyle":    "--",
    "figure.dpi":        150,
    "savefig.dpi":       300,
    "savefig.bbox":      "tight",
    "legend.frameon":    True,
    "legend.framealpha": 0.9,
}

def save_fig(fig, name: str):
    """Save figure as both PDF and PNG."""
    fig.savefig(FIG_DIR / f"{name}.pdf")
    fig.savefig(FIG_DIR / f"{name}.png")
    plt.close(fig)
    print(f"  ✔  {name}.pdf / .png")


# ─── Figure 1: Training Curves ────────────────────────────────────────────────

def fig1_training_curves():
    """Validation loss vs epoch for all 6 models on a single plot."""
    plt.rcParams.update(PLT_STYLE)
    fig, ax = plt.subplots(figsize=(9, 5))

    found = 0
    for model, label, color in zip(MODELS, LABELS, COLORS):
        csv_path = RESULTS_DIR / f"{model}_BTC_1d_metrics.csv"
        if not csv_path.exists():
            continue
        df = pd.read_csv(csv_path)
        if "val_loss" not in df.columns:
            # Try epoch-indexed format
            if df.shape[1] >= 2:
                df.columns = ["epoch", "val_loss"] + list(df.columns[2:])
            else:
                continue

        epochs   = range(len(df))
        val_loss = df["val_loss"].values

        ax.plot(epochs, val_loss, color=color, linewidth=1.8,
                label=f"{label} (min={val_loss.min():.4f})", alpha=0.85)
        found += 1

    if found == 0:
        print("  ⚠  No training curve CSVs found. Run training first.")
        plt.close(fig)
        return

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation loss (CombinedLoss)")
    ax.set_title("Training convergence — BTC 1d h=1 (all 6 models)")
    ax.legend(fontsize=9, loc="upper right")
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    save_fig(fig, "fig1_training_curves")


# ─── Figure 2: Price Prediction Plot ──────────────────────────────────────────

def fig2_price_prediction(model_name: str = "gru"):
    """Actual vs predicted BTC price over the test period."""
    plt.rcParams.update(PLT_STYLE)

    # Load saved predictions
    pred_path = RESULTS_DIR / f"{model_name}_BTC_1d_h1_predictions.npy"
    if not pred_path.exists():
        print(f"  ⚠  Predictions not found: {pred_path}")
        return

    errors = np.load(pred_path)   # errors = actual - predicted

    # Load test close prices
    test_close_path = ROOT / "data" / "processed" / "scalers" / "BTC_1d_test_close.npy"
    if not test_close_path.exists():
        print(f"  ⚠  Test prices not found: {test_close_path}")
        return

    test_close = np.load(test_close_path)
    n          = len(errors)

    # Best params to get seq_len
    params_file = RESULTS_DIR / "tuning" / f"{model_name}_BTC_1d_h1_best_params.json"
    seq_len     = 60
    if params_file.exists():
        import json
        with open(params_file) as f:
            p       = json.load(f).get("best_params", {})
            seq_len = int(p.get("seq_len", 60))

    idx      = np.clip(np.arange(n) + seq_len - 1, 0, len(test_close) - 1)
    actuals  = test_close[idx]
    preds    = actuals - errors     # actual - error = prediction

    # Approximate dates: test period starts 2024-12-26
    dates = pd.date_range("2024-12-26", periods=n, freq="D")

    fig, axes = plt.subplots(2, 1, figsize=(11, 7),
                             gridspec_kw={"height_ratios": [3, 1]})

    # Top: price plot
    ax = axes[0]
    ax.plot(dates, actuals / 1000, color="#1a1a2e", linewidth=1.5,
            label="Actual BTC price", zorder=3)
    ax.plot(dates, preds / 1000, color=COLORS[MODELS.index(model_name)],
            linewidth=1.2, linestyle="--", label=f"{model_name.upper()} prediction",
            alpha=0.85, zorder=2)
    ax.fill_between(dates, actuals / 1000, preds / 1000,
                    alpha=0.08, color=COLORS[MODELS.index(model_name)])
    ax.set_ylabel("Price (USD thousands)")
    ax.set_title(f"BTC price prediction vs actual — test period Dec 2024 – Mar 2026 "
                 f"({model_name.upper()}, MAPE={np.mean(np.abs(errors/actuals))*100:.2f}%)")
    ax.legend()
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%b %Y"))

    # Bottom: error plot
    ax2 = axes[1]
    ax2.bar(dates, errors / 1000, color=["#E24B4A" if e < 0 else "#1D9E75"
                                          for e in errors],
            width=0.8, alpha=0.7)
    ax2.axhline(0, color="black", linewidth=0.8)
    ax2.set_ylabel("Error (USD k)")
    ax2.set_xlabel("Date")
    ax2.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%b %Y"))

    plt.tight_layout()
    save_fig(fig, f"fig2_price_prediction_{model_name}")


# ─── Figure 3: Model Comparison Bar Chart ─────────────────────────────────────

def fig3_model_comparison():
    """MAPE and RMSE side-by-side bar chart for all 6 models."""
    plt.rcParams.update(PLT_STYLE)

    mapes, rmses = [], []
    labels_found = []

    for model, label in zip(MODELS, LABELS):
        csv = RESULTS_DIR / f"{model}_BTC_1d_h1_results.csv"
        if not csv.exists():
            continue
        df = pd.read_csv(csv)
        if df.empty:
            continue
        mapes.append(float(df.iloc[0].get("mape", np.nan)))
        rmses.append(float(df.iloc[0].get("rmse", np.nan)))
        labels_found.append(label)

    if not mapes:
        print("  ⚠  No result CSVs found.")
        return

    colors_found = [COLORS[LABELS.index(l)] for l in labels_found]
    x  = np.arange(len(labels_found))
    w  = 0.38

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # MAPE
    bars1 = ax1.bar(x, mapes, width=w*2, color=colors_found, alpha=0.85,
                    edgecolor="white", linewidth=0.5)
    ax1.axhline(5.0, color="#E24B4A", linestyle="--", linewidth=1,
                label="5% target threshold", alpha=0.7)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels_found, rotation=20, ha="right")
    ax1.set_ylabel("MAPE (%)")
    ax1.set_title("MAPE comparison (lower is better)")
    ax1.legend(fontsize=9)
    ax1.set_ylim(0, max(mapes) * 1.3)
    for bar, v in zip(bars1, mapes):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                 f"{v:.2f}%", ha="center", va="bottom", fontsize=9)

    # RMSE
    bars2 = ax2.bar(x, [r/1000 for r in rmses], width=w*2,
                    color=colors_found, alpha=0.85,
                    edgecolor="white", linewidth=0.5)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels_found, rotation=20, ha="right")
    ax2.set_ylabel("RMSE (USD thousands)")
    ax2.set_title("RMSE comparison (lower is better)")
    ax2.set_ylim(0, max(rmses)/1000 * 1.3)
    for bar, v in zip(bars2, rmses):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f"${v/1000:.1f}k", ha="center", va="bottom", fontsize=9)

    plt.suptitle("Model performance comparison — BTC 1d h=1 (test: Dec 2024 – Mar 2026)",
                 fontsize=12, y=1.01)
    plt.tight_layout()
    save_fig(fig, "fig3_model_comparison")


# ─── Figure 4: DM Significance Heatmap ────────────────────────────────────────

def fig4_dm_heatmap():
    """Diebold-Mariano test statistic heatmap."""
    plt.rcParams.update(PLT_STYLE)

    # DM stats from the final run
    dm_data = np.array([
        [   0,  0.285, -8.351,  0.060,  0.020, -2.511],
        [-0.285,    0, -7.947,  0.034, -0.003, -3.103],
        [ 8.351, 7.947,     0,  4.970,  4.746,  6.193],
        [-0.060,-0.034, -4.970,     0, -0.040, -0.810],
        [-0.020, 0.003, -4.746,  0.040,     0, -0.669],
        [ 2.511, 3.103, -6.193,  0.810,  0.669,     0],
    ])

    sig_mask = np.abs(dm_data) > 1.96   # approx p < 0.05
    np.fill_diagonal(sig_mask, False)

    short_labels = ["LSTM", "GRU", "BiLSTM", "CNN-LSTM", "Attn-LSTM", "Transformer"]

    fig, ax = plt.subplots(figsize=(8, 6.5))

    # Custom diverging colormap: blue (negative/better) → white → red (positive/worse)
    from matplotlib.colors import TwoSlopeNorm
    vmax = 9.0
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    cmap = plt.cm.RdBu_r

    im = ax.imshow(dm_data, cmap=cmap, norm=norm, aspect="auto")
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("DM statistic (negative = row model better)", fontsize=9)

    ax.set_xticks(range(len(short_labels)))
    ax.set_yticks(range(len(short_labels)))
    ax.set_xticklabels(short_labels, rotation=30, ha="right", fontsize=10)
    ax.set_yticklabels(short_labels, fontsize=10)
    ax.set_title("Diebold-Mariano significance test matrix\n"
                 "BTC 1d h=1 | * p < 0.05", fontsize=11)

    # Annotate cells
    for i in range(len(short_labels)):
        for j in range(len(short_labels)):
            val = dm_data[i, j]
            if i == j:
                ax.text(j, i, "—", ha="center", va="center",
                        fontsize=10, color="gray")
                continue
            star = "*" if sig_mask[i, j] else ""
            color = "white" if abs(val) > 4 else "black"
            ax.text(j, i, f"{val:+.2f}{star}", ha="center", va="center",
                    fontsize=8.5, color=color, fontweight="bold" if star else "normal")

    # Highlight significant cells with border
    for i in range(len(short_labels)):
        for j in range(len(short_labels)):
            if sig_mask[i, j]:
                rect = mpatches.FancyBboxPatch(
                    (j - 0.48, i - 0.48), 0.96, 0.96,
                    boxstyle="square,pad=0",
                    linewidth=1.5, edgecolor="black", facecolor="none"
                )
                ax.add_patch(rect)

    ax.set_xlabel("Column model", fontsize=10)
    ax.set_ylabel("Row model", fontsize=10)

    plt.tight_layout()
    save_fig(fig, "fig4_dm_heatmap")


# ─── Figure 5: Ablation Study ─────────────────────────────────────────────────

def fig5_ablation():
    """MAPE change per feature group removed."""
    plt.rcParams.update(PLT_STYLE)

    ablation_path = RESULTS_DIR / "ablation_lstm_BTC_1d_h1.csv"
    if not ablation_path.exists():
        print(f"  ⚠  Ablation CSV not found: {ablation_path}")
        return

    df = pd.read_csv(ablation_path)
    if "condition" not in df.columns or "mape" not in df.columns:
        print("  ⚠  Ablation CSV missing expected columns")
        return

    baseline = float(df.loc[df["condition"] == "full", "mape"].values[0])
    ablation = df[df["condition"] != "full"].copy()
    ablation["delta_mape"] = ablation["mape"] - baseline

    condition_labels = {
        "ltst":      "No LTST decomposition\n(−35 features)",
        "onchain":   "No on-chain metrics\n(−12 features)",
        "sentiment": "No sentiment features\n(−8 features)",
        "macro":     "No macro features\n(−8 features)",
    }
    ablation["display"] = ablation["condition"].map(condition_labels).fillna(ablation["condition"])

    fig, ax = plt.subplots(figsize=(8, 5))

    bar_colors = ["#E24B4A" if d > 0 else "#1D9E75" for d in ablation["delta_mape"]]
    bars = ax.barh(ablation["display"], ablation["delta_mape"],
                   color=bar_colors, alpha=0.85, height=0.5)

    ax.axvline(0, color="black", linewidth=1)
    ax.set_xlabel("ΔMAPE vs full model (positive = worse without this group)")
    ax.set_title("Feature ablation study — LSTM BTC 1d h=1\n"
                 f"Baseline MAPE: {baseline:.2f}%")

    for bar, val in zip(bars, ablation["delta_mape"]):
        offset = 0.001 if val >= 0 else -0.001
        ha     = "left" if val >= 0 else "right"
        ax.text(val + offset, bar.get_y() + bar.get_height()/2,
                f"{val:+.3f}%", va="center", ha=ha, fontsize=9)

    ax.set_xlim(-0.1, 0.1)
    plt.tight_layout()
    save_fig(fig, "fig5_ablation")


# ─── Figure 6: Feature Category Overview ──────────────────────────────────────

def fig6_feature_overview():
    """Pie chart showing the 149-feature breakdown by category."""
    plt.rcParams.update(PLT_STYLE)

    categories = {
        "OHLCV base":                10,
        "Trend indicators":          25,
        "Momentum indicators":       22,
        "Volatility indicators":     18,
        "Volume indicators":         18,
        "Derived/cross features":    21,
        "LTST decomposition":        35,
    }
    cat_colors = ["#1a1a2e", "#3266AD", "#1D9E75", "#D85A30",
                  "#BA7517", "#7F77DD", "#888780"]

    total = sum(categories.values())
    labels = [f"{k}\n({v} features, {v/total*100:.0f}%)"
              for k, v in categories.items()]

    fig, ax = plt.subplots(figsize=(9, 6))
    wedges, texts, autotexts = ax.pie(
        categories.values(), labels=None, colors=cat_colors,
        autopct="%1.0f%%", startangle=140,
        pctdistance=0.78, wedgeprops={"linewidth": 0.8, "edgecolor": "white"}
    )
    for at in autotexts:
        at.set_fontsize(9)

    ax.legend(wedges, labels, loc="center left",
              bbox_to_anchor=(1.0, 0.5), fontsize=9)
    ax.set_title(f"Feature engineering overview — {total} total features\n"
                 f"BTC 1d | LTST + Technical Indicators + Market Data",
                 fontsize=11)

    plt.tight_layout()
    save_fig(fig, "fig6_feature_overview")


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate thesis figures")
    parser.add_argument("--fig", type=int, default=0,
                        help="Figure number (1-6), or 0 for all")
    parser.add_argument("--model", default="gru",
                        help="Model for Fig 2 prediction plot")
    args = parser.parse_args()

    figs = {
        1: fig1_training_curves,
        2: lambda: fig2_price_prediction(args.model),
        3: fig3_model_comparison,
        4: fig4_dm_heatmap,
        5: fig5_ablation,
        6: fig6_feature_overview,
    }

    print(f"\n  Generating thesis figures → {FIG_DIR}")
    print(f"{'─'*50}")

    targets = [args.fig] if args.fig > 0 else list(figs.keys())
    for n in targets:
        if n in figs:
            figs[n]()
        else:
            print(f"  ⚠  Unknown figure: {n}")

    print(f"{'─'*50}")
    print(f"  Done. Check {FIG_DIR}/\n")


if __name__ == "__main__":
    main()
