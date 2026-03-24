# Deep Learning for Cryptocurrency Price Forecasting
### A Multi-Model Approach Using Neural Networks

> **M.Tech. Thesis — 2026**
> **Author: Muluh Penn Junior Patrick**
> **Supervisor: [Supervisor Name]**
> **Institution: [The University of Bamenda]**

---

## Overview

This repository contains the complete implementation of a master's thesis comparing six deep learning architectures for multi-asset, multi-horizon cryptocurrency price forecasting. The study evaluates models across five assets and four prediction horizons using a rigorous experimental framework including hyperparameter optimisation, statistical significance testing, and feature ablation analysis.

**Research question:** Which deep learning architecture achieves the best risk-adjusted forecast accuracy for cryptocurrency price prediction, and how does performance vary across assets and prediction horizons?

---

## Key Results — BTC Daily, 1-Day Horizon

| Model | MAPE | RMSE (USD) | R² | DA (%) | Parameters |
|-------|------|-----------|-----|--------|-----------|
| **GRU** | **1.60%** | **$2,077** | **0.9831** | 50.1 | ~130K |
| LSTM | 1.61% | $2,085 | 0.9829 | 50.1 | 143K |
| CNN-LSTM | 1.72% | $2,205 | 0.9825 | 50.6 | 574K |
| Transformer | 1.67% | $2,142 | 0.9820 | 50.1 | 418K |
| Attention-LSTM | 1.67% | $2,161 | 0.9797 | 48.5 | ~285K |
| BiLSTM | 1.71% | $2,165 | 0.9816 | 49.3 | ~285K |

- All six models exceed the MAPE < 5% thesis target
- GRU and LSTM are statistically equivalent (Diebold-Mariano p > 0.05)
- GRU is significantly better than Transformer (DM = −3.10, p < 0.05)
- BiLSTM is significantly worse than all other models (DM > 4.0, p < 0.05 for all pairs)

**Test period:** December 2024 – March 2026 (451 days, BTC range $62k–$108k)

---

## Scope

| Dimension | Detail |
|-----------|--------|
| Assets | BTC, ETH, SOL, SUI, XRP |
| Models | LSTM, GRU, BiLSTM, CNN-LSTM, Attention-LSTM, Transformer |
| Horizons | h=1, h=7, h=14, h=30 (daily) |
| Features | 149 per timestep (TA + LTST + on-chain + sentiment + macro) |
| Data range | 2018-01-01 → 2026-03-21 |
| Train/Val/Test | 70% / 15% / 15% (walk-forward) |
| Tuning | Optuna TPE, 20–30 trials per model |
| Significance | Diebold-Mariano test (Harvey, Leybourne & Newbold correction) |

---

## Repository Structure

```
crypto-forecast-thesis/
│
├── data/
│   ├── raw/
│   │   ├── ohlcv/          # BTC, ETH, SOL, SUI, XRP daily and hourly OHLCV
│   │   ├── onchain/        # Blockchain.info (BTC), Etherscan (ETH)
│   │   ├── sentiment/      # Fear & Greed Index, CoinGecko market data
│   │   └── macro/          # S&P 500, DXY, Gold, VIX via Yahoo Finance
│   └── processed/
│       ├── features/       # Engineered feature DataFrames (.parquet)
│       ├── sequences/      # Sliding window sequences (.npy)
│       └── scalers/        # Fitted normalizers and test close prices
│
├── src/
│   ├── data_collection/    # API fetchers and pipeline orchestrator
│   ├── preprocessing/      # Technical indicators, LTST, normalizer, sequences
│   ├── models/             # All 6 model architectures + registry
│   ├── training/           # PyTorch Lightning trainer, loss functions, callbacks
│   ├── tuning/             # Optuna search spaces, study runner, pruner
│   ├── evaluation/         # KPI metrics, DM tests, ablation, regime analysis
│   ├── visualization/      # Thesis figure generation
│   └── utils/              # Seed, logger, device, config loader
│
├── experiments/
│   ├── configs/            # YAML experiment configurations
│   ├── checkpoints/        # Best model checkpoints (.ckpt) — gitignored
│   ├── results/            # KPI CSVs, prediction arrays, tuning results
│   └── figures/            # Publication-quality figures (PDF + PNG)
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_baseline_models.ipynb
│   ├── 04_model_training.ipynb
│   ├── 05_results_analysis.ipynb
│   └── 06_visualization_gallery.ipynb
│
├── scripts/
│   ├── run_training.py         # Train a single model
│   ├── run_tuning.py           # Run Optuna hyperparameter search
│   ├── run_evaluation.py       # Generate comparison tables + DM tests
│   ├── run_ablation.py         # Feature group ablation study
│   └── run_all_experiments.py  # Orchestrate all 120 experiments
│
├── tests/
│   ├── conftest.py             # Shared fixtures
│   ├── test_models.py          # Forward pass shape validation
│   └── test_metrics.py         # KPI metric unit tests
│
├── docs/
│   ├── data_dictionary.md      # All 149 features documented
│   ├── experiment_log.md       # Full research diary and results
│   └── api_reference.md        # Public API and CLI documentation
│
├── save_predictions.py         # Save per-sample errors for DM testing
├── generate_repo_content.py    # Regenerate all notebooks, scripts, docs
├── requirements.txt
├── environment.yml
├── pyproject.toml
├── Makefile
└── .env.example
```

---

## Setup

### Prerequisites

- Python 3.11+
- CUDA-capable GPU (recommended; tested on NVIDIA RTX 3050 with CUDA 12.8)
- 8 GB RAM minimum; 16 GB recommended for full feature matrix

### Installation

```bash
# Clone
git clone https://github.com/ImaJin14/crypto-forecast-thesis.git
cd crypto-forecast-thesis

# Option A — conda (recommended)
conda env create -f environment.yml
conda activate crypto-forecast

# Option B — pip
pip install -r requirements.txt

# Configure API keys
cp .env.example .env
# Edit .env with your Binance and CoinGecko API keys
```

### Verify

```bash
pytest tests/ -v
# Expected: all tests pass (forward pass shapes + metric computations)
```

---

## Reproducing the Results

### Step 1 — Data collection

```bash
python -m src.data_collection.pipeline
# or
make data
```

Collects OHLCV data from Binance, on-chain metrics from Blockchain.info/Etherscan, sentiment from Alternative.me/CoinGecko, and macro from Yahoo Finance. All APIs used are free-tier — no paid subscription required.

### Step 2 — Train a single model

```bash
# With tuned hyperparameters (loads from experiments/results/tuning/)
python scripts/run_training.py --model gru --asset BTC --interval 1d --horizon 1 --use-best-params

# With manual hyperparameters
python -m src.training.trainer --model lstm --asset BTC --interval 1d --epochs 200
```

### Step 3 — Hyperparameter tuning

```bash
python scripts/run_tuning.py --model lstm --asset BTC --trials 30 --epochs 50
```

Results saved to `experiments/results/tuning/{model}_{asset}_{interval}_h{horizon}_best_params.json`.

### Step 4 — Run all 120 experiments (overnight)

```bash
# Full matrix: 6 models × 5 assets × 4 horizons
python scripts/run_all_experiments.py --trials 15 --epochs 30

# BTC only (24 experiments, ~2–3 hours)
python scripts/run_all_experiments.py --asset BTC

# Preview without running
python scripts/run_all_experiments.py --dry-run
```

### Step 5 — Evaluate

```bash
# Comparison table for BTC 1d h=1
python scripts/run_evaluation.py --asset BTC --horizon 1

# With Diebold-Mariano significance tests
python scripts/run_evaluation.py --asset BTC --horizon 1 --dm

# Save prediction arrays first (required for DM testing)
python save_predictions.py --asset BTC --interval 1d --horizon 1
```

### Step 6 — Generate figures

```bash
python -m src.visualization.thesis_figures
# Saves all 6 figures to experiments/figures/ as PDF + PNG (300 DPI)
```

---

## Models

All models share the same input format `(batch, seq_len, n_features)` and output `(batch, horizon)`.

| Model | Architecture | Key hyperparameters (BTC 1d h=1 best) |
|-------|-------------|--------------------------------------|
| LSTM | Stacked LSTM | hidden=128, layers=1, seq=90, lr=0.00448 |
| GRU | Stacked GRU | hidden=128, layers=1, seq=90, lr=0.00448 |
| BiLSTM | Bidirectional LSTM | hidden=128, layers=2, seq=90 |
| CNN-LSTM | Conv1D → LSTM | filters=128, kernel=5, hidden=128, seq=60 |
| Attention-LSTM | LSTM + Bahdanau attention | hidden=256, layers=1, seq=60 |
| Transformer | Encoder + positional encoding | d=128, heads=2, layers=4, seq=90 |

```python
from src.models import get_model

model = get_model('gru', input_size=149, hidden_size=128, num_layers=1, dropout=0.296)
```

---

## Feature Engineering

149 features per timestep across five categories:

| Category | Count | Examples |
|----------|-------|---------|
| OHLCV base | 10 | open, high, low, close, volume, log_return |
| Trend indicators | 25 | EMA-9/21/50/200, MACD, ADX, Aroon, CCI |
| Momentum indicators | 22 | RSI, Stochastic, ROC, Williams %R |
| Volatility indicators | 18 | ATR, Bollinger Bands, Keltner Channel |
| Volume indicators | 18 | OBV, CMF, MFI, VWAP |
| Derived / cross-asset | 21 | candlestick patterns, lagged returns |
| LTST decomposition | 35 | LTT-50/100/200, STT-5/10/20, residuals, slopes |

Full documentation: [`docs/data_dictionary.md`](docs/data_dictionary.md)

### LTST Decomposition

Long-term/short-term trend (LTST) decomposition separates price into:

- **Long-term trend (LTT):** 50, 100, 200-period moving averages
- **Short-term trend (STT):** 5, 10, 20-period exponential moving averages
- **Residuals:** price deviation from each trend level
- **Derived signals:** slope, above/below trend, trend consensus, mean reversion signal

---

## Training Framework

Built on PyTorch Lightning with a custom combined loss function:

```
CombinedLoss = α·MSE + β·Huber + γ·DirectionalLoss
```

The directional component penalises incorrect sign prediction, directly optimising for trading signal accuracy alongside regression accuracy.

**Key design decisions:**
- Target: log returns (stationary), not price levels (avoids normalisation leakage on test set)
- Reconstruction: USD prices computed as `P_{t+1} = P_t × exp(log_return)` using saved `test_close.npy`
- Normalizer: MinMax fitted on training split only — zero leakage guarantee
- Early stopping: patience=15 epochs on validation loss

---

## Evaluation Framework

### KPIs

| Metric | Definition | Thesis Target |
|--------|-----------|--------------|
| RMSE | Root mean squared error (USD) | Minimise |
| MAE | Mean absolute error (USD) | Minimise |
| MAPE | Mean absolute percentage error | < 5% |
| R² | Coefficient of determination | > 0.85 |
| DA | Directional accuracy | > 60% |
| Sharpe | Annualised Sharpe ratio of strategy returns | Maximise |
| Max Drawdown | Worst peak-to-trough equity decline | Minimise |

### Statistical Significance

The Diebold-Mariano test (Harvey, Leybourne & Newbold 1997 small-sample correction) is applied to all pairwise model comparisons. A negative DM statistic indicates the row model is more accurate than the column model; p < 0.05 indicates statistical significance.

### Feature Ablation

Each feature group (LTST, on-chain, sentiment, macro) is removed individually and the model is retrained, measuring the change in MAPE. Results for BTC 1d h=1 show ΔMAPE < 0.01% for all groups — auxiliary features become more impactful at longer prediction horizons.

---

## Notebooks

| Notebook | Content |
|----------|---------|
| `01_data_exploration` | Price distributions, cross-asset correlations, volatility regimes, stationarity tests |
| `02_feature_engineering` | Full 149-feature pipeline, LTST decomposition visualisation, normalisation |
| `03_baseline_models` | Naïve, moving average, and ARIMA baselines vs deep learning results |
| `04_model_training` | End-to-end GRU training with tuned hyperparameters, training curve analysis |
| `05_results_analysis` | Comparison tables, DM matrix, prediction plots, ablation results, discussion |
| `06_visualization_gallery` | Reproduces all six thesis figures |

---

## Project KPIs — Current Status

| KPI | Target | Best Result | Model |
|-----|--------|-------------|-------|
| MAPE | < 5% | **1.60%** ✅ | GRU |
| R² | > 0.85 | **0.983** ✅ | GRU |
| RMSE (BTC 1d) | Minimise | **$2,077** | GRU |
| Directional Accuracy | > 60% | 50.6% ⚠️ | CNN-LSTM |
| Sharpe Ratio | Maximise | 0.502 | LSTM / GRU |

Directional accuracy at h=1 is near-random across all models, consistent with weak-form market efficiency. Improvement expected at longer horizons (h=7, 14, 30).

---

## Makefile

```bash
make setup      # Install dependencies
make data       # Run full data collection pipeline
make train      # Train MODEL=lstm ASSET=BTC HORIZON=1
make tune       # Run Optuna tuning for MODEL on ASSET
make eval       # Evaluate all models and print comparison tables
make ablation   # Run feature ablation study
make test       # Run pytest test suite
make lint       # ruff + black formatting checks
make clean      # Remove __pycache__ and .pyc files
```

---

## Environment Variables

```bash
# Required
BINANCE_API_KEY=your_key
BINANCE_API_SECRET=your_secret

# Optional (enhance data coverage)
COINGECKO_API_KEY=your_key
ETHERSCAN_API_KEY=your_key

# Experiment tracking (optional)
WANDB_API_KEY=your_key
MLFLOW_TRACKING_URI=http://localhost:5000

# Paths (defaults shown)
DATA_DIR=./data
RESULTS_DIR=./experiments/results
CHECKPOINT_DIR=./experiments/checkpoints
```

Copy `.env.example` to `.env` and populate before running data collection.

---

## Citation

If you use this code or findings in your research, please cite:

```bibtex
@mastersthesis{patrick2026deeplearning,
  author      = {Muluh Penn Junior Patrick},
  title       = {Deep Learning for Cryptocurrency Price Forecasting:
                 A Multi-Model Approach Using Neural Networks},
  school      = {[The University of Bamenda]},
  year        = {2026},
  type        = {M.Tech. Thesis}
}
```

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

## Acknowledgements

Data sourced from Binance, CoinGecko, Alternative.me, Blockchain.info, Etherscan, and Yahoo Finance via their free-tier APIs. Built with PyTorch, PyTorch Lightning, Optuna, and pandas.
