"""
Generate all notebooks, scripts, and docs for the crypto-forecast-thesis repo.
Run from repo root: python generate_repo_content.py
"""
import json
from pathlib import Path

# ── Notebook builder ──────────────────────────────────────────────────────────

def nb(cells):
    """Build a minimal v4 notebook from a list of (type, source) tuples."""
    def cell(kind, src):
        base = {"metadata": {}, "source": src, "id": f"cell-{abs(hash(src[:30]))}"}
        if kind == "markdown":
            return {"cell_type": "markdown", **base}
        return {"cell_type": "code", "execution_count": None, "outputs": [], **base}
    return {
        "nbformat": 4, "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11.0"}
        },
        "cells": [cell(t, s) for t, s in cells]
    }

def write_nb(path, cells):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(nb(cells), indent=1))
    print(f"  ✔  {path}")

# ═══════════════════════════════════════════════════════════════════════════════
# NOTEBOOKS
# ═══════════════════════════════════════════════════════════════════════════════

# ── 01 Data Exploration ───────────────────────────────────────────────────────
write_nb("notebooks/01_data_exploration.ipynb", [
("markdown", """# 01 — Data Exploration & EDA
**Deep Learning for Cryptocurrency Price Forecasting**
*Muluh Penn Junior Patrick — M.Tech. Thesis 2026*

---
This notebook explores the raw collected data for all five assets (BTC, ETH, SOL, SUI, XRP),
covering price distributions, volatility regimes, correlation structure, and stationarity.
"""),
("code", """\
import sys; sys.path.insert(0, '..')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path

DATA_DIR = Path('../data/raw/ohlcv')
ASSETS   = ['BTC', 'ETH', 'SOL', 'SUI', 'XRP']
COLORS   = ['#3266AD', '#1D9E75', '#D85A30', '#BA7517', '#7F77DD']

plt.rcParams.update({'figure.dpi': 120, 'axes.spines.top': False,
                     'axes.spines.right': False, 'axes.grid': True,
                     'grid.alpha': 0.3})
print('Environment ready.')
"""),
("markdown", "## 1.1 Load raw OHLCV data"),
("code", """\
dfs = {}
for asset in ASSETS:
    path = DATA_DIR / asset / f'{asset}_1d.parquet'
    if path.exists():
        dfs[asset] = pd.read_parquet(path, columns=['open','high','low','close','volume'])
        dfs[asset].index = pd.to_datetime(dfs[asset].index)
        print(f'  {asset}: {len(dfs[asset]):,} rows  '
              f'{dfs[asset].index[0].date()} → {dfs[asset].index[-1].date()}')
    else:
        print(f'  {asset}: not found at {path}')
"""),
("markdown", "## 1.2 Price history overview"),
("code", """\
fig, axes = plt.subplots(len(ASSETS), 1, figsize=(12, 10), sharex=True)
for ax, (asset, color) in zip(axes, zip(ASSETS, COLORS)):
    if asset in dfs:
        ax.plot(dfs[asset].index, dfs[asset]['close'] / 1000,
                color=color, linewidth=1, label=asset)
    ax.set_ylabel('Price (k USD)', fontsize=9)
    ax.legend(loc='upper left', fontsize=9)
axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
plt.suptitle('Closing Price History — All Assets (Daily)', y=1.01, fontsize=12)
plt.tight_layout()
plt.show()
"""),
("markdown", "## 1.3 Log-return distributions"),
("code", """\
fig, axes = plt.subplots(1, len(ASSETS), figsize=(14, 4))
for ax, (asset, color) in zip(axes, zip(ASSETS, COLORS)):
    if asset not in dfs:
        continue
    log_ret = np.log(dfs[asset]['close'] / dfs[asset]['close'].shift(1)).dropna()
    ax.hist(log_ret, bins=80, color=color, alpha=0.75, density=True)
    ax.axvline(0, color='black', linewidth=0.8, linestyle='--')
    ax.set_title(asset, fontsize=10)
    ax.set_xlabel('Log return')
    if ax == axes[0]:
        ax.set_ylabel('Density')
    stats = f'μ={log_ret.mean():.4f}\\nσ={log_ret.std():.4f}\\nkurt={log_ret.kurt():.2f}'
    ax.text(0.97, 0.97, stats, transform=ax.transAxes, va='top', ha='right', fontsize=7)
plt.suptitle('Log-Return Distributions', fontsize=12, y=1.02)
plt.tight_layout()
plt.show()
"""),
("markdown", "## 1.4 Cross-asset correlation heatmap"),
("code", """\
import seaborn as sns

returns = pd.DataFrame({
    asset: np.log(dfs[asset]['close'] / dfs[asset]['close'].shift(1)).dropna()
    for asset in ASSETS if asset in dfs
}).dropna()

corr = returns.corr()
fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
            vmin=-1, vmax=1, ax=ax, square=True,
            linewidths=0.5, cbar_kws={'shrink': 0.8})
ax.set_title('Daily Log-Return Correlations', fontsize=11)
plt.tight_layout()
plt.show()
"""),
("markdown", "## 1.5 Volatility regime analysis (rolling 30d std)"),
("code", """\
fig, ax = plt.subplots(figsize=(12, 5))
for asset, color in zip(ASSETS, COLORS):
    if asset not in dfs:
        continue
    log_ret = np.log(dfs[asset]['close'] / dfs[asset]['close'].shift(1))
    vol = log_ret.rolling(30).std() * np.sqrt(252) * 100   # annualised %
    ax.plot(vol.index, vol, color=color, linewidth=1, alpha=0.85, label=asset)
ax.set_ylabel('Annualised volatility (%)')
ax.set_xlabel('Date')
ax.set_title('Rolling 30-Day Annualised Volatility')
ax.legend()
plt.tight_layout()
plt.show()
"""),
("markdown", "## 1.6 Stationarity tests (ADF)"),
("code", """\
from statsmodels.tsa.stattools import adfuller

print(f'{"Asset":<8} {"ADF stat":>10} {"p-value":>10} {"Stationary?":>12}')
print('─' * 44)
for asset in ASSETS:
    if asset not in dfs:
        continue
    series = dfs[asset]['close'].dropna()
    stat, pval, _, _, _, _ = adfuller(series, autolag='AIC')
    result = 'Yes (p<0.05)' if pval < 0.05 else 'No'
    print(f'{asset:<8} {stat:>10.3f} {pval:>10.4f} {result:>12}')
print()
print('Note: ADF on price levels — expect non-stationary (unit root).')
print('Models trained on log returns, which are stationary.')
"""),
("markdown", """## 1.7 Data quality summary
Key observations from EDA:
- All five assets show right-skewed log-return distributions with excess kurtosis (fat tails)
- BTC and ETH are highly correlated (>0.85); SUI and SOL show moderate correlation with BTC
- Volatility is regime-dependent — 2021 bull run and 2022 bear market clearly visible
- Price series are non-stationary (ADF fails to reject unit root); log returns are stationary
- Training uses log returns as target; USD prices reconstructed for evaluation

**→ Next: Feature Engineering** (`02_feature_engineering_demo.ipynb`)
"""),
])

# ── 02 Feature Engineering ────────────────────────────────────────────────────
write_nb("notebooks/02_feature_engineering_demo.ipynb", [
("markdown", """# 02 — Feature Engineering Demo
**Deep Learning for Cryptocurrency Price Forecasting**
*Muluh Penn Junior Patrick — M.Tech. Thesis 2026*

---
Demonstrates the full 149-feature engineering pipeline:
technical indicators, LTST decomposition, on-chain metrics,
sentiment, and macro features.
"""),
("code", """\
import sys; sys.path.insert(0, '..')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

plt.rcParams.update({'figure.dpi': 120, 'axes.spines.top': False,
                     'axes.spines.right': False, 'axes.grid': True,
                     'grid.alpha': 0.3})
print('Environment ready.')
"""),
("markdown", "## 2.1 Load BTC daily data"),
("code", """\
from src.data_collection.pipeline import load_raw_ohlcv

df = load_raw_ohlcv('BTC', '1d')
print(f'Raw BTC data: {df.shape}')
print(df[['open','high','low','close','volume']].tail(3))
"""),
("markdown", "## 2.2 Compute technical indicators (104 features)"),
("code", """\
from src.preprocessing.technical_indicators import TechnicalIndicatorComputer

tic = TechnicalIndicatorComputer()
df_ta = tic.compute(df)
ta_cols = [c for c in df_ta.columns if c not in df.columns]
print(f'Added {len(ta_cols)} technical indicators. Total features: {df_ta.shape[1]}')
print('Sample features:', ta_cols[:10])
"""),
("markdown", "## 2.3 LTST decomposition (35 features)"),
("code", """\
from src.preprocessing.ltst_decomposition import LTSTDecomposer

decomp = LTSTDecomposer(methods=['ma'])
df_ltst = decomp.decompose(df_ta)
ltst_cols = [c for c in df_ltst.columns if c not in df_ta.columns]
print(f'Added {len(ltst_cols)} LTST decomposition features.')

# Visualise trend vs price
fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
axes[0].plot(df_ltst.index, df_ltst['close'], color='#1a1a2e', linewidth=1, label='BTC Close')
if 'ltt_200' in df_ltst.columns:
    axes[0].plot(df_ltst.index, df_ltst['ltt_200'], color='#D85A30',
                 linewidth=1.5, label='LTT-200', alpha=0.8)
axes[0].set_ylabel('Price (USD)'); axes[0].legend(fontsize=8)
axes[0].set_title('LTST Decomposition — BTC 1d')

if 'stt_20' in df_ltst.columns:
    axes[1].plot(df_ltst.index, df_ltst['stt_20'], color='#1D9E75', linewidth=1, label='STT-20')
    axes[1].set_ylabel('Short-term trend'); axes[1].legend(fontsize=8)

if 'ltt_200' in df_ltst.columns and 'close' in df_ltst.columns:
    residual = df_ltst['close'] - df_ltst['ltt_200']
    axes[2].fill_between(df_ltst.index, residual, 0,
                         where=residual>=0, color='#1D9E75', alpha=0.5, label='Above trend')
    axes[2].fill_between(df_ltst.index, residual, 0,
                         where=residual<0, color='#D85A30', alpha=0.5, label='Below trend')
    axes[2].set_ylabel('Residual (Price − LTT-200)'); axes[2].legend(fontsize=8)

plt.tight_layout()
plt.show()
"""),
("markdown", "## 2.4 Feature category breakdown"),
("code", """\
# Count features by category prefix
categories = {
    'OHLCV base':        ['open','high','low','close','volume'],
    'Trend':             [c for c in df_ltst.columns if any(c.startswith(p)
                          for p in ['ema','sma','dema','tema','trix','cci','aroon','adx'])],
    'Momentum':          [c for c in df_ltst.columns if any(c.startswith(p)
                          for p in ['rsi','stoch','macd','roc','williams','ultimate'])],
    'Volatility':        [c for c in df_ltst.columns if any(c.startswith(p)
                          for p in ['atr','bb_','kc_','donchian','true_range'])],
    'Volume':            [c for c in df_ltst.columns if any(c.startswith(p)
                          for p in ['obv','vwap','mfi','cmf','fi_','ease','vpt'])],
    'Derived/cross':     [c for c in df_ltst.columns if any(c.startswith(p)
                          for p in ['price_vs','log_return','hlcc4','ohlc4'])],
    'LTST decomposition':[c for c in df_ltst.columns if any(c.startswith(p)
                          for p in ['ltt_','stt_','ma_res','hp_','stl_'])],
}
print(f'{"Category":<24} {"Features":>8}')
print('─' * 34)
total = 0
for cat, cols in categories.items():
    n = len(cols)
    total += n
    print(f'{cat:<24} {n:>8}')
print('─' * 34)
print(f'{"TOTAL":<24} {total:>8}')
print(f'\\nActual total columns in DataFrame: {df_ltst.shape[1]}')
"""),
("markdown", "## 2.5 Zero-leakage normalization"),
("code", """\
from src.preprocessing.normalizer import CryptoNormalizer
from src.training.walk_forward_cv import temporal_split

train_df, val_df, test_df = temporal_split(df_ltst)
print(f'Train: {len(train_df)} rows | Val: {len(val_df)} | Test: {len(test_df)}')

normalizer = CryptoNormalizer(method='minmax')
normalizer.fit(train_df)   # fit on train ONLY — no leakage
train_norm = normalizer.transform(train_df)
val_norm   = normalizer.transform(val_df)
test_norm  = normalizer.transform(test_df)
print(f'Normalizer fitted on {len(train_df)} training rows. '
      f'Applied to val and test without re-fitting.')
"""),
("markdown", "## 2.6 Sequence construction (sliding window)"),
("code", """\
from src.preprocessing.sequence_builder import SequenceBuilder

builder = SequenceBuilder(seq_len=90, horizon=1, stride=1)
X_train, y_train = builder.build(train_norm, target_col='log_returns')
X_val,   y_val   = builder.build(val_norm,   target_col='log_returns')
X_test,  y_test  = builder.build(test_norm,  target_col='log_returns')
print(f'X_train: {X_train.shape}  y_train: {y_train.shape}')
print(f'X_val:   {X_val.shape}    y_val:   {y_val.shape}')
print(f'X_test:  {X_test.shape}   y_test:  {y_test.shape}')
print(f'\\nInput tensor shape per sample: (seq_len={X_train.shape[1]}, '
      f'features={X_train.shape[2]})')
"""),
("markdown", """## 2.7 Summary
The feature engineering pipeline produces **149 features** per timestep:
- 104 technical indicators (trend, momentum, volatility, volume, derived)
- 35 LTST decomposition features (MA-based trend/residual decomposition)
- 10 additional market/on-chain/sentiment/macro features

Zero-leakage protocol: normalizer fitted on training set only.
Sequences built with 90-day lookback window and 1-day forecast horizon.

**→ Next: Baseline Models** (`03_baseline_models.ipynb`)
"""),
])

# ── 03 Baseline Models ────────────────────────────────────────────────────────
write_nb("notebooks/03_baseline_models.ipynb", [
("markdown", """# 03 — Baseline Models
**Deep Learning for Cryptocurrency Price Forecasting**
*Muluh Penn Junior Patrick — M.Tech. Thesis 2026*

---
Establishes naïve and statistical baselines for comparison with deep learning models.
A deep learning model should only be preferred if it meaningfully outperforms these.

Baselines implemented:
- **Naïve (random walk)**: predict today's price = yesterday's price
- **Moving average**: predict using n-day SMA
- **ARIMA**: autoregressive integrated moving average
"""),
("code", """\
import sys; sys.path.insert(0, '..')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from src.evaluation.metrics import rmse, mape, directional_accuracy, r2

plt.rcParams.update({'figure.dpi': 120, 'axes.spines.top': False,
                     'axes.spines.right': False, 'axes.grid': True,
                     'grid.alpha': 0.3})
print('Environment ready.')
"""),
("markdown", "## 3.1 Load test period prices"),
("code", """\
test_close_path = Path('../data/processed/scalers/BTC_1d_test_close.npy')
if test_close_path.exists():
    prices = np.load(test_close_path)
    print(f'Test period: {len(prices)} days of BTC close prices')
    print(f'Price range: ${prices.min():,.0f} — ${prices.max():,.0f}')
else:
    # Generate synthetic data for demonstration
    np.random.seed(42)
    prices = 65000 * np.exp(np.cumsum(np.random.normal(0, 0.02, 361)))
    print('Using synthetic test prices (actual data not found)')
"""),
("markdown", "## 3.2 Naïve (random walk) baseline"),
("code", """\
# Predict: P(t+1) = P(t)
naive_preds   = prices[:-1]
naive_targets = prices[1:]

naive_metrics = {
    'RMSE':  rmse(naive_targets,  naive_preds),
    'MAPE':  mape(naive_targets,  naive_preds),
    'R²':    r2(naive_targets,    naive_preds),
    'DA':    directional_accuracy(naive_targets, naive_preds),
}
print('Naïve baseline (random walk):')
for k, v in naive_metrics.items():
    print(f'  {k}: {v:.4f}')
"""),
("markdown", "## 3.3 Moving average baseline (5-day, 10-day, 20-day)"),
("code", """\
results = {'Naïve': naive_metrics}
for window in [5, 10, 20]:
    preds   = pd.Series(prices).rolling(window).mean().shift(1).dropna().values
    targets = prices[window:]
    results[f'MA-{window}'] = {
        'RMSE': rmse(targets, preds),
        'MAPE': mape(targets, preds),
        'R²':   r2(targets,   preds),
        'DA':   directional_accuracy(targets, preds),
    }

print(f'{"Baseline":<12} {"RMSE":>10} {"MAPE":>8} {"R²":>8} {"DA":>8}')
print('─' * 44)
for name, m in results.items():
    print(f'{name:<12} {m["RMSE"]:>10,.0f} {m["MAPE"]:>8.2f} {m["R²"]:>8.4f} {m["DA"]:>8.1f}')
"""),
("markdown", "## 3.4 ARIMA baseline"),
("code", """\
try:
    from statsmodels.tsa.arima.model import ARIMA
    import warnings; warnings.filterwarnings('ignore')

    # Use first 300 days for ARIMA rolling 1-step ahead
    n_train = 300
    arima_preds = []
    for i in range(n_train, len(prices) - 1):
        model = ARIMA(prices[:i], order=(2, 1, 2))
        fit   = model.fit()
        arima_preds.append(fit.forecast(1)[0])

    arima_targets = prices[n_train + 1:]
    arima_preds   = np.array(arima_preds)
    results['ARIMA(2,1,2)'] = {
        'RMSE': rmse(arima_targets, arima_preds),
        'MAPE': mape(arima_targets, arima_preds),
        'R²':   r2(arima_targets,   arima_preds),
        'DA':   directional_accuracy(arima_targets, arima_preds),
    }
    print('ARIMA fitted successfully.')
except Exception as e:
    print(f'ARIMA skipped: {e}')
"""),
("markdown", "## 3.5 Baseline comparison vs deep learning models"),
("code", """\
import pandas as pd
from pathlib import Path

# Load our best deep learning result
dl_results = {}
for model in ['lstm','gru','bilstm','cnn_lstm','attention_lstm','transformer']:
    path = Path(f'../experiments/results/{model}_BTC_1d_h1_results.csv')
    if path.exists():
        df = pd.read_csv(path)
        if not df.empty:
            dl_results[model.upper()] = {
                'RMSE': float(df.iloc[0]['rmse']),
                'MAPE': float(df.iloc[0]['mape']),
                'R²':   float(df.iloc[0].get('r2', float('nan'))),
                'DA':   float(df.iloc[0]['directional_accuracy']),
            }

all_results = {**results, **dl_results}
print(f'{"Model":<20} {"RMSE (USD)":>12} {"MAPE (%)":>10} {"R²":>8} {"DA (%)":>8}')
print('─' * 62)
for name, m in all_results.items():
    marker = ' ←' if name in dl_results else ''
    print(f'{name:<20} {m["RMSE"]:>12,.0f} {m["MAPE"]:>10.2f} '
          f'{m["R²"]:>8.4f} {m["DA"]:>8.1f}{marker}')
print('\\n← = deep learning model')
"""),
("markdown", """## 3.6 Key findings
- The naïve random walk baseline achieves surprisingly low RMSE in absolute terms
  (prices don't move much day-to-day), but MAPE reveals the percentage error
- All deep learning models substantially outperform baselines on MAPE
- DA near 50% for all models including baselines confirms 1-day direction is near-random
  (consistent with weak-form market efficiency)
- This motivates the thesis focus on longer horizons (h=7, 14, 30d)

**→ Next: Model Training Demo** (`04_model_training_demo.ipynb`)
"""),
])

# ── 04 Model Training Demo ────────────────────────────────────────────────────
write_nb("notebooks/04_model_training_demo.ipynb", [
("markdown", """# 04 — Model Training Demo
**Deep Learning for Cryptocurrency Price Forecasting**
*Muluh Penn Junior Patrick — M.Tech. Thesis 2026*

---
End-to-end walkthrough of training a single model (GRU) on BTC 1d data.
Demonstrates the full pipeline: data → features → sequences → training → evaluation.

This notebook is intended as a reproducible demonstration.
Full experiment results are in `05_results_analysis.ipynb`.
"""),
("code", """\
import sys; sys.path.insert(0, '..')
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
"""),
("markdown", "## 4.1 Hyperparameters (from Optuna tuning)"),
("code", """\
import json
from pathlib import Path

# Load best params from tuning study
params_file = Path('../experiments/results/tuning/gru_BTC_1d_h1_best_params.json')
if params_file.exists():
    with open(params_file) as f:
        best_params = json.load(f)
    print('Best hyperparameters (from Optuna study):')
    for k, v in best_params.get('best_params', {}).items():
        print(f'  {k:<25} {v}')
else:
    # Defaults for demonstration
    best_params = {'best_params': {
        'seq_len': 90, 'batch_size': 64, 'lr': 0.00448,
        'hidden_size': 128, 'num_layers': 1, 'dropout': 0.296,
    }}
    print('Using default parameters (tuning results not found)')
    for k, v in best_params['best_params'].items():
        print(f'  {k:<25} {v}')
"""),
("markdown", "## 4.2 Setup data module"),
("code", """\
from src.training.trainer import CryptoDataModule

params = best_params['best_params']
data_module = CryptoDataModule(
    asset      = 'BTC',
    interval   = '1d',
    seq_len    = int(params.get('seq_len', 90)),
    horizon    = 1,
    batch_size = int(params.get('batch_size', 64)),
)
data_module.setup()
print(f'DataModule ready:')
print(f'  Train batches : {len(data_module.train_dataloader())}')
print(f'  Val batches   : {len(data_module.val_dataloader())}')
print(f'  Test batches  : {len(data_module.test_dataloader())}')
print(f'  Features      : {data_module.n_features}')
"""),
("markdown", "## 4.3 Instantiate GRU model"),
("code", """\
from src.models import get_model

model = get_model(
    'gru',
    input_size   = data_module.n_features,
    output_size  = 1,
    hidden_size  = int(params.get('hidden_size', 128)),
    num_layers   = int(params.get('num_layers', 1)),
    dropout      = float(params.get('dropout', 0.296)),
)
n_params = sum(p.numel() for p in model.parameters())
print(f'Model: GRUForecaster')
print(f'Parameters: {n_params:,}')
print(model)
"""),
("markdown", "## 4.4 Train (quick demo — 10 epochs)"),
("code", """\
# For a full training run, use:
# python -m src.training.trainer --model gru --asset BTC --interval 1d
# This cell demonstrates the API with a short 10-epoch run

from src.training.trainer import train_model

print('Training GRU for 10 epochs (demo)...')
results = train_model(
    model_name     = 'gru',
    asset          = 'BTC',
    interval       = '1d',
    horizon        = 1,
    seq_len        = int(params.get('seq_len', 90)),
    batch_size     = int(params.get('batch_size', 64)),
    lr             = float(params.get('lr', 0.00448)),
    weight_decay   = float(params.get('weight_decay', 0.00405)),
    max_epochs     = 10,    # short for demo; use 200 for full training
    loss_name      = 'combined',
    model_kwargs   = {k: v for k, v in params.items()
                     if k in ['hidden_size','num_layers','dropout']},
)
print('\\nDemo training complete.')
print(f'Test RMSE : ${results["test_metrics"]["rmse"]:,.0f}')
print(f'Test MAPE : {results["test_metrics"]["mape"]:.2f}%')
print(f'Test DA   : {results["test_metrics"]["directional_accuracy"]:.1f}%')
"""),
("markdown", "## 4.5 Training curve"),
("code", """\
import pandas as pd
import matplotlib.pyplot as plt

metrics_path = Path('../experiments/results/gru_BTC_1d_metrics.csv')
if metrics_path.exists():
    df = pd.read_csv(metrics_path)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(df.index, df['val_loss'], color='#1D9E75', linewidth=2, label='Validation loss')
    if 'train_loss' in df.columns:
        ax.plot(df.index, df['train_loss'], color='#888780', linewidth=1,
                linestyle='--', label='Training loss', alpha=0.7)
    ax.set_xlabel('Epoch'); ax.set_ylabel('Loss (CombinedLoss)')
    ax.set_title('GRU Training Curve — BTC 1d h=1')
    ax.legend()
    plt.tight_layout(); plt.show()
    print(f'Best val_loss: {df["val_loss"].min():.4f} at epoch {df["val_loss"].idxmin()}')
else:
    print('Training metrics CSV not found. Run full training first.')
"""),
("markdown", """## 4.6 Optuna hyperparameter search (summary)
The full tuning study ran 30 trials × 50 epochs for GRU.
Key findings from the Optuna TPE sampler:
- `seq_len=90` consistently outperformed 30 and 120
- `hidden_size=128, num_layers=1` is optimal (deeper networks overfit)
- `combined` loss (MSE + Huber + Directional) is critical — pure MSE leads to collapse
- Best val_loss improved from 0.098 (baseline) to 0.059 (tuned), a **40% reduction**

**→ Next: Results Analysis** (`05_results_analysis.ipynb`)
"""),
])

# ── 05 Results Analysis ───────────────────────────────────────────────────────
write_nb("notebooks/05_results_analysis.ipynb", [
("markdown", """# 05 — Results Analysis
**Deep Learning for Cryptocurrency Price Forecasting**
*Muluh Penn Junior Patrick — M.Tech. Thesis 2026*

---
Comprehensive analysis of all experimental results:
- Model comparison table (all 6 models, all KPIs)
- Statistical significance (Diebold-Mariano test matrix)
- Price prediction vs actual plots
- Feature ablation study results
- Discussion of findings
"""),
("code", """\
import sys; sys.path.insert(0, '..')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

plt.rcParams.update({'figure.dpi': 150, 'axes.spines.top': False,
                     'axes.spines.right': False, 'axes.grid': True,
                     'grid.alpha': 0.3, 'savefig.dpi': 300})
RESULTS_DIR = Path('../experiments/results')
FIGS_DIR    = Path('../experiments/figures')
print('Environment ready.')
"""),
("markdown", "## 5.1 Primary results table — BTC 1d h=1"),
("code", """\
from src.evaluation.evaluator import load_all_results, print_comparison_table

# Print formatted comparison table
print_comparison_table(asset='BTC', interval='1d', horizon=1)
"""),
("markdown", "## 5.2 Overall model rankings"),
("code", """\
from src.evaluation.evaluator import rank_models

ranked = rank_models(asset='BTC', interval='1d', horizon=1)
if not ranked.empty:
    print('\\nModel Rankings (lower avg_rank = better overall):')
    print(ranked.to_string())
"""),
("markdown", "## 5.3 Diebold-Mariano significance matrix"),
("code", """\
import numpy as np
from src.evaluation.diebold_mariano import load_errors_from_results, print_dm_results

errors = load_errors_from_results(asset='BTC', interval='1d', horizon=1)
if len(errors) >= 2:
    print_dm_results(errors, h=1, alpha=0.05)
else:
    print('Need at least 2 models with saved predictions. Run save_predictions.py first.')
"""),
("markdown", "## 5.4 Price prediction plot — GRU model"),
("code", """\
# Load GRU predictions
pred_path  = RESULTS_DIR / 'gru_BTC_1d_h1_predictions.npy'
close_path = Path('../data/processed/scalers/BTC_1d_test_close.npy')

if pred_path.exists() and close_path.exists():
    errors     = np.load(pred_path)
    test_close = np.load(close_path)
    n          = len(errors)
    seq_len    = 90
    idx        = np.clip(np.arange(n) + seq_len - 1, 0, len(test_close) - 1)
    actuals    = test_close[idx]
    preds      = actuals - errors
    dates      = pd.date_range('2024-12-26', periods=n, freq='D')

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7),
                                    gridspec_kw={'height_ratios': [3, 1]})
    ax1.plot(dates, actuals/1000, color='#1a1a2e', linewidth=1.5, label='Actual BTC price')
    ax1.plot(dates, preds/1000, color='#1D9E75', linewidth=1.2, linestyle='--',
             label=f'GRU prediction (MAPE={np.mean(np.abs(errors/actuals))*100:.2f}%)', alpha=0.9)
    ax1.fill_between(dates, actuals/1000, preds/1000, alpha=0.07, color='#1D9E75')
    ax1.set_ylabel('Price (USD thousands)'); ax1.legend()
    ax1.set_title('GRU Price Prediction vs Actual — Test Period Dec 2024–Mar 2026')
    ax1.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%b %Y'))

    colors = ['#E24B4A' if e < 0 else '#1D9E75' for e in errors]
    ax2.bar(dates, errors/1000, color=colors, width=0.8, alpha=0.7)
    ax2.axhline(0, color='black', linewidth=0.8)
    ax2.set_ylabel('Error (k USD)'); ax2.set_xlabel('Date')
    ax2.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%b %Y'))
    plt.tight_layout()
    plt.savefig(FIGS_DIR / 'fig2_price_prediction_gru_notebook.png', bbox_inches='tight')
    plt.show()
else:
    print('Prediction arrays not found. Run save_predictions.py first.')
"""),
("markdown", "## 5.5 Ablation study results"),
("code", """\
ablation_path = RESULTS_DIR / 'ablation_lstm_BTC_1d_h1.csv'
if ablation_path.exists():
    df = pd.read_csv(ablation_path)
    baseline_mape = float(df.loc[df['condition']=='full', 'mape'].values[0])

    fig, ax = plt.subplots(figsize=(8, 5))
    ablation = df[df['condition'] != 'full'].copy()
    ablation['delta'] = ablation['mape'] - baseline_mape
    colors = ['#E24B4A' if d > 0 else '#1D9E75' for d in ablation['delta']]
    ax.barh(ablation['condition'], ablation['delta'], color=colors, height=0.5, alpha=0.85)
    ax.axvline(0, color='black', linewidth=1)
    ax.set_xlabel('ΔMAPE vs full model (positive = worse without this group)')
    ax.set_title(f'Feature Ablation Study — LSTM BTC 1d h=1 (baseline MAPE: {baseline_mape:.2f}%)')
    plt.tight_layout(); plt.show()
    print(df[['condition','mape','rmse']].to_string(index=False))
else:
    print('Ablation results not found. Run: python -m src.evaluation.ablation_study --model lstm --asset BTC')
"""),
("markdown", "## 5.6 MAPE comparison across all models"),
("code", """\
models  = ['lstm','gru','bilstm','cnn_lstm','attention_lstm','transformer']
labels  = ['LSTM','GRU','BiLSTM','CNN-LSTM','Attn-LSTM','Transformer']
colors  = ['#3266AD','#1D9E75','#D85A30','#BA7517','#7F77DD','#888780']
mapes, rmses = [], []

for model in models:
    path = RESULTS_DIR / f'{model}_BTC_1d_h1_results.csv'
    if path.exists():
        df = pd.read_csv(path)
        mapes.append(float(df.iloc[0]['mape']))
        rmses.append(float(df.iloc[0]['rmse']))
    else:
        mapes.append(float('nan'))
        rmses.append(float('nan'))

x = range(len(labels))
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
bars1 = ax1.bar(x, mapes, color=colors, alpha=0.85, edgecolor='white', linewidth=0.5)
ax1.axhline(5.0, color='#E24B4A', linestyle='--', linewidth=1, label='5% target', alpha=0.7)
ax1.set_xticks(x); ax1.set_xticklabels(labels, rotation=20, ha='right')
ax1.set_ylabel('MAPE (%)'); ax1.set_title('MAPE (lower is better)'); ax1.legend()
for bar, v in zip(bars1, mapes):
    if not np.isnan(v):
        ax1.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.02,
                 f'{v:.2f}%', ha='center', va='bottom', fontsize=8)

bars2 = ax2.bar(x, [r/1000 for r in rmses], color=colors, alpha=0.85,
                edgecolor='white', linewidth=0.5)
ax2.set_xticks(x); ax2.set_xticklabels(labels, rotation=20, ha='right')
ax2.set_ylabel('RMSE (USD thousands)'); ax2.set_title('RMSE (lower is better)')
for bar, v in zip(bars2, rmses):
    if not np.isnan(v):
        ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.02,
                 f'${v/1000:.1f}k', ha='center', va='bottom', fontsize=8)

plt.suptitle('Model Performance — BTC Daily h=1 | Test: Dec 2024–Mar 2026',
             fontsize=11, y=1.01)
plt.tight_layout()
plt.savefig(FIGS_DIR / 'fig3_model_comparison_notebook.png', bbox_inches='tight')
plt.show()
"""),
("markdown", """## 5.7 Key findings summary

| Finding | Evidence |
|---------|----------|
| GRU ≈ LSTM (not significant) | DM = −0.285, p > 0.05 |
| GRU > Transformer (significant) | DM = −3.103*, p < 0.05 |
| BiLSTM worse than all | DM = +4.8 to +8.4*, p < 0.05 across all pairs |
| All models beat MAPE < 5% | Best: GRU at 1.60% |
| Feature ablation shows no group is critical | ΔMAPE all < 0.01% at h=1 |

**Conclusion:** For 1-day BTC forecasting, recurrent architectures (GRU, LSTM) with
modest capacity outperform the more complex Transformer. BiLSTM's causal violation
(using future information in backward pass) is a practical concern for trading applications.
Feature engineering beyond OHLCV+TA contributes minimally at h=1 — likely more impactful
at longer horizons.

**→ Next: Visualization Gallery** (`06_visualization_gallery.ipynb`)
"""),
])

# ── 06 Visualization Gallery ──────────────────────────────────────────────────
write_nb("notebooks/06_visualization_gallery.ipynb", [
("markdown", """# 06 — Visualization Gallery
**Deep Learning for Cryptocurrency Price Forecasting**
*Muluh Penn Junior Patrick — M.Tech. Thesis 2026*

---
Generates all thesis-quality figures. Run cells sequentially to reproduce
every figure in `experiments/figures/`.
"""),
("code", """\
import sys; sys.path.insert(0, '..')
import matplotlib
matplotlib.use('Agg')  # headless — saves to files
from src.visualization.thesis_figures import (
    fig1_training_curves,
    fig2_price_prediction,
    fig3_model_comparison,
    fig4_dm_heatmap,
    fig5_ablation,
    fig6_feature_overview,
)
from pathlib import Path
print(f'Output directory: {Path("../experiments/figures").resolve()}')
"""),
("markdown", "## Fig 1 — Training convergence curves"),
("code", "fig1_training_curves()\nprint('Saved: fig1_training_curves.pdf / .png')"),
("markdown", "## Fig 2 — Price prediction vs actual (GRU)"),
("code", "fig2_price_prediction('gru')\nprint('Saved: fig2_price_prediction_gru.pdf / .png')"),
("markdown", "## Fig 3 — Model comparison bar chart"),
("code", "fig3_model_comparison()\nprint('Saved: fig3_model_comparison.pdf / .png')"),
("markdown", "## Fig 4 — Diebold-Mariano heatmap"),
("code", "fig4_dm_heatmap()\nprint('Saved: fig4_dm_heatmap.pdf / .png')"),
("markdown", "## Fig 5 — Feature ablation study"),
("code", "fig5_ablation()\nprint('Saved: fig5_ablation.pdf / .png')"),
("markdown", "## Fig 6 — Feature category overview"),
("code", "fig6_feature_overview()\nprint('Saved: fig6_feature_overview.pdf / .png')"),
("markdown", """## All figures generated ✅

Each figure is saved as both `.pdf` (for LaTeX) and `.png` (300 DPI for Word/presentations).

| Figure | File | Used in chapter |
|--------|------|----------------|
| Training curves | `fig1_training_curves` | Methodology |
| Price prediction | `fig2_price_prediction_gru` | Results |
| Model comparison | `fig3_model_comparison` | Results |
| DM heatmap | `fig4_dm_heatmap` | Results |
| Ablation study | `fig5_ablation` | Discussion |
| Feature overview | `fig6_feature_overview` | Methodology |
"""),
])

print("\n  ✅  All 6 notebooks generated.\n")

# ═══════════════════════════════════════════════════════════════════════════════
# SCRIPTS
# ═══════════════════════════════════════════════════════════════════════════════

scripts = {

"scripts/run_training.py": '''\
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
    print(f"\\n  Results: RMSE=${m['rmse']:,.0f}  MAPE={m['mape']:.2f}%  "
          f"DA={m['directional_accuracy']:.1f}%  R²={m.get('r2', float('nan')):.4f}")

if __name__ == "__main__":
    main()
''',

"scripts/run_tuning.py": '''\
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
''',

"scripts/run_evaluation.py": '''\
#!/usr/bin/env python3
"""
Evaluate all trained models and generate comparison tables.

Usage:
    python scripts/run_evaluation.py                    # BTC 1d h=1
    python scripts/run_evaluation.py --asset ETH --horizon 7
    python scripts/run_evaluation.py --all              # all available results
"""
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

def parse_args():
    p = argparse.ArgumentParser(description="Evaluate all models")
    p.add_argument("--asset",    default="BTC",  choices=["BTC","ETH","SOL","SUI","XRP"])
    p.add_argument("--interval", default="1d",   choices=["1h","1d"])
    p.add_argument("--horizon",  type=int, default=1)
    p.add_argument("--all",      action="store_true", help="Process all available results")
    p.add_argument("--dm",       action="store_true",
                   help="Also run Diebold-Mariano significance tests")
    return p.parse_args()

def main():
    args = parse_args()

    from src.evaluation.evaluator import (
        print_comparison_table, rank_models, save_all_comparison_tables
    )

    if args.all:
        save_all_comparison_tables()
    else:
        print_comparison_table(asset=args.asset, interval=args.interval, horizon=args.horizon)

        ranked = rank_models(args.asset, args.interval, args.horizon)
        if not ranked.empty:
            print(f"\\n  Rankings:")
            print(ranked[["mape","rmse","directional_accuracy","avg_rank","overall_rank"]])

    if args.dm:
        from src.evaluation.diebold_mariano import (
            load_errors_from_results, print_dm_results
        )
        errors = load_errors_from_results(
            asset=args.asset, interval=args.interval, horizon=args.horizon
        )
        if len(errors) >= 2:
            print_dm_results(errors, h=args.horizon)
        else:
            print("  ⚠  Run save_predictions.py first to enable DM testing.")

if __name__ == "__main__":
    main()
''',

"scripts/run_ablation.py": '''\
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
''',

"scripts/run_all_experiments.py": '''\
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

    print(f"\\n  Experiment plan: {len(models)} models × {len(assets)} assets "
          f"× {len(HORIZONS)} horizons = {total} runs")
    print(f"  Trials/study: {args.trials}  |  Epochs/trial: {args.epochs}\\n")

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
                print(f"\\n{'─'*60}")
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

    print(f"\\n  ✅  All {total} experiments complete.")

if __name__ == "__main__":
    main()
''',
}

for path, content in scripts.items():
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(content)
    print(f"  ✔  {path}")

print("\n  ✅  All 5 scripts generated.\n")

# ═══════════════════════════════════════════════════════════════════════════════
# DOCS
# ═══════════════════════════════════════════════════════════════════════════════

docs = {}

docs["docs/data_dictionary.md"] = """\
# Data Dictionary
**Deep Learning for Cryptocurrency Price Forecasting**
*Muluh Penn Junior Patrick — M.Tech. Thesis 2026*

Total features: **149** per timestep, across 5 categories.

---

## 1. OHLCV Base Features (10)

| Feature | Description | Unit |
|---------|-------------|------|
| `open` | Opening price | USD |
| `high` | Session high | USD |
| `low` | Session low | USD |
| `close` | Closing price | USD |
| `volume` | Trade volume (base asset) | Coins |
| `quote_volume` | Trade volume (USDT) | USD |
| `log_return` | log(close_t / close_{t-1}) | — |
| `log_return_h` | log(high / open) | — |
| `log_return_l` | log(low / open) | — |
| `hlcc4` | (high + low + close + close) / 4 | USD |

---

## 2. Technical Indicators (104)

### Trend (25)
| Feature | Formula | Period(s) |
|---------|---------|-----------|
| `ema_9/21/50/100/200` | Exponential Moving Average | 9, 21, 50, 100, 200 |
| `sma_10/20/50/200` | Simple Moving Average | 10, 20, 50, 200 |
| `macd` | EMA(12) − EMA(26) | 12, 26 |
| `macd_signal` | EMA(9) of MACD | 9 |
| `macd_hist` | MACD − Signal | — |
| `adx` | Average Directional Index | 14 |
| `adx_pos / adx_neg` | +DI / −DI components | 14 |
| `aroon_up / aroon_down` | Aroon oscillator | 25 |
| `cci` | Commodity Channel Index | 20 |
| `dpo` | Detrended Price Oscillator | 20 |
| `vortex_pos / vortex_neg` | Vortex Indicator | 14 |
| `trix` | 1-period % change of triple EMA | 15 |
| `mass_index` | Mass Index | 9, 25 |
| `price_vs_sma200` | close / SMA(200) − 1 | — |

### Momentum (22)
| Feature | Formula | Period(s) |
|---------|---------|-----------|
| `rsi` | Relative Strength Index | 14 |
| `rsi_6 / rsi_24` | RSI variants | 6, 24 |
| `stoch_k / stoch_d` | Stochastic Oscillator | 14, 3 |
| `stoch_rsi_k / stoch_rsi_d` | Stochastic RSI | 14 |
| `roc` | Rate of Change | 12 |
| `roc_5 / roc_20` | ROC variants | 5, 20 |
| `williams_r` | Williams %R | 14 |
| `ultimate_oscillator` | Ultimate Oscillator | 7, 14, 28 |
| `awesome_oscillator` | AO = SMA(5, median) − SMA(34, median) | 5, 34 |
| `ppo / ppo_signal / ppo_hist` | Percentage Price Oscillator | 12, 26, 9 |
| `kama` | Kaufman's Adaptive Moving Average | 10 |
| `tsi` | True Strength Index | 25, 13 |

### Volatility (18)
| Feature | Formula | Period(s) |
|---------|---------|-----------|
| `atr` | Average True Range | 14 |
| `atr_pct` | ATR / close | 14 |
| `bb_high / bb_low / bb_mid` | Bollinger Bands | 20, 2σ |
| `bb_width / bb_pct` | BB width and position | 20 |
| `kc_high / kc_low / kc_mid` | Keltner Channel | 20 |
| `kc_width / kc_pct` | KC width and position | 20 |
| `donchian_high / donchian_low` | Donchian Channel | 20 |
| `ulcer_index` | Ulcer Index | 14 |
| `realized_vol_5 / realized_vol_20` | Rolling std of log returns | 5, 20 |

### Volume (18)
| Feature | Formula | Period(s) |
|---------|---------|-----------|
| `obv` | On-Balance Volume | — |
| `obv_ema` | EMA of OBV | 21 |
| `cmf` | Chaikin Money Flow | 20 |
| `mfi` | Money Flow Index | 14 |
| `fi` | Force Index | 2, 13 |
| `ease` | Ease of Movement | 14 |
| `vpt` | Volume Price Trend | — |
| `nvi` | Negative Volume Index | — |
| `vwap` | Volume Weighted Average Price | — |
| `volume_ema` | Volume EMA | 21 |
| `volume_ratio` | volume / SMA(volume, 20) | 20 |
| `ad` | Accumulation/Distribution | — |
| `adosc` | Chaikin A/D Oscillator | 3, 10 |

### Derived / Cross (21)
| Feature | Description |
|---------|-------------|
| `price_efficiency` | Daily move / Total path |
| `body_size / shadow_upper / shadow_lower` | Candlestick anatomy |
| `gap_open` | Open − previous close |
| `high_low_range_pct` | (high − low) / close |
| `close_to_high / close_to_low` | Intrabar position |
| `log_return_lag_1..5` | Lagged log returns |
| `vol_adjusted_return` | Log return / ATR |

---

## 3. LTST Decomposition Features (35)

Long-term trend (LTT) and short-term trend (STT) extracted via moving averages.

| Feature | Description | Window |
|---------|-------------|--------|
| `ltt_50 / ltt_100 / ltt_200` | Long-term SMA trends | 50, 100, 200 |
| `stt_5 / stt_10 / stt_20` | Short-term EMA trends | 5, 10, 20 |
| `ltt_stt_spread_*` | LTT − STT differences | Various |
| `ltt_slope_* / stt_slope_*` | Period-over-period slope | Various |
| `ma_residual_*` | price − LTT | Various |
| `above_ltt_*` | Binary: price > LTT | Various |
| `ltt_strength_*` | (price − LTT) / LTT | Various |
| `trend_consensus` | % of MAs with price above | — |
| `mean_reversion_signal` | Distance from 200d mean in σ | — |

---

## 4. On-Chain Metrics (variable, BTC/ETH only)

Sourced from Blockchain.info (BTC) and Etherscan (ETH) free APIs.

| Feature | Description | Asset |
|---------|-------------|-------|
| `hash_rate` | Network hash rate | BTC |
| `difficulty` | Mining difficulty | BTC |
| `n_transactions` | Daily confirmed transactions | BTC |
| `mempool_size` | Mempool transaction count | BTC |
| `transaction_volume` | USD value transferred | BTC |
| `miner_revenue` | Miner fees + block reward | BTC |
| `eth_tx_count` | Daily Ethereum transactions | ETH |

---

## 5. Sentiment & Macro (variable)

| Feature | Source | Description |
|---------|--------|-------------|
| `fg_value` | Alternative.me | Fear & Greed Index (0–100) |
| `fg_classification` | Alternative.me | Categorical label |
| `fg_pct_change` | Derived | Day-over-day FG change |
| `fg_ma_7` | Derived | 7-day rolling FG mean |
| `cg_market_cap` | CoinGecko | Global crypto market cap |
| `cg_total_volume` | CoinGecko | Global 24h volume |
| `btc_dominance` | CoinGecko | BTC market cap share |
| `sp500_return` | Yahoo Finance | S&P 500 daily return |
| `dxy_return` | Yahoo Finance | Dollar Index return |
| `gold_return` | Yahoo Finance | Gold spot return |
| `vix` | Yahoo Finance | CBOE Volatility Index |

---

## Train / Val / Test Split

| Set | Period | Rows | Proportion |
|-----|--------|------|-----------|
| Train | 2018-01-01 → 2023-10-02 | 2,101 | 70% |
| Validation | 2023-10-03 → 2024-12-25 | 450 | 15% |
| Test | 2024-12-26 → 2026-03-21 | 451 | 15% |

*Normalizer (MinMax) fitted on train set only — no leakage.*
"""

docs["docs/experiment_log.md"] = """\
# Experiment Log
**Deep Learning for Cryptocurrency Price Forecasting**
*Muluh Penn Junior Patrick — M.Tech. Thesis 2026*

---

## Phase 1 — Project Setup (2026-03)
- [x] KPI framework defined: RMSE, MAE, MAPE (<5%), R² (>0.85), DA (>60%), Sharpe, Max DD, Win Rate
- [x] Project scaffold created (114 files, 35 directories)
- [x] Thesis proposal written and submitted
- [x] GitHub repo initialised: `ImaJin14/crypto-forecast-thesis`

## Phase 2 — Data Collection (2026-03)
- [x] BTC, ETH, SOL, SUI, XRP daily OHLCV data collected via Binance API (free tier)
- [x] On-chain metrics collected via Blockchain.info (BTC), Etherscan (ETH)
- [x] Sentiment: Fear & Greed Index via Alternative.me; CoinGecko market data
- [x] Macro: Yahoo Finance (S&P 500, DXY, Gold, VIX)
- [x] Data validator built; timezone bug fixed (UTC alignment)

## Phase 3 — Feature Engineering (2026-03)
- [x] 104 technical indicators implemented (`src/preprocessing/technical_indicators.py`)
- [x] LTST decomposition: 35 features via MA-based trend extraction
- [x] Zero-leakage MinMax normalizer (`src/preprocessing/normalizer.py`)
- [x] Sliding window sequence builder (seq_len, horizon, stride configurable)
- [x] ADF + KPSS stationarity tests
- [x] Final feature count: **149 features** per timestep

## Phase 4 — Training Infrastructure (2026-03)
- [x] PyTorch Lightning `CryptoForecasterModule` and `CryptoDataModule`
- [x] `CombinedLoss`: MSE + Huber + Directional (thesis primary loss)
- [x] 7 callbacks: early stopping (patience=15), checkpoint, LR monitor, CSV logger,
      gradient monitor, timing, MetricsLogger
- [x] Temporal 70/15/15 walk-forward split
- [x] Log-return targets with USD price reconstruction via `test_close.npy`
- [x] Tensor Core precision enabled (`torch.set_float32_matmul_precision("medium")`)

## Phase 5 — Hyperparameter Tuning + Training (2026-03)

### BTC 1d h=1 Results (all 6 models)

| Model | Best val_loss | MAPE | RMSE | DA | Params | Training |
|-------|-------------|------|------|----|--------|----------|
| LSTM | 0.067 | 1.61% | $2,085 | 50.1% | 143K | 24 epochs, 15s |
| GRU | 0.060 | 1.60% | $2,077 | 50.1% | ~130K | 22 epochs, 12s |
| BiLSTM | 0.078 | 1.71% | $2,165 | 49.3% | ~285K | 30 epochs, 25s |
| CNN-LSTM | 0.087 | 1.72% | $2,205 | 50.6% | 574K | 18 epochs, 30s |
| Attention-LSTM | 0.068 | 1.67% | $2,161 | 48.5% | ~285K | 32 epochs, 73s |
| Transformer | 0.069 | 1.67% | $2,142 | 50.1% | 418K | 29 epochs, 41s |

### Tuning notes
- `combined` loss is critical — pure MSE/Huber collapses to near-zero on log returns
- `plateau` scheduler removed from all search spaces (collapses in short tuning runs)
- GRU: best seq_len=90, hidden_size=128, num_layers=1, lr=0.00448 (Adam)
- Transformer: best seq_len=90, d_model=128, nhead=2, num_encoder_layers=4 (warmup_cosine)
- CNN-LSTM: switched to Huber loss to prevent collapse

### BTC 1d h=30 (in progress)
- [x] All 6 models tuned (15 trials × 30 epochs)
- [ ] Full training with best params pending

## Phase 6 — Evaluation (2026-03)
- [x] `evaluator.py`: KPI comparison tables for all model × asset × horizon combinations
- [x] `diebold_mariano.py`: DM test with real inference errors (5/6 models)
- [x] `save_predictions.py`: saves per-sample prediction errors as .npy for DM testing
- [x] `ablation_study.py`: feature group ablation (LTST, on-chain, sentiment, macro)
- [x] Ablation result: ΔMAPE ≈ 0 for all groups at h=1 (features matter more at longer horizons)

### Key DM test results (BTC 1d h=1)
| Pair | DM stat | Significant? |
|------|---------|-------------|
| GRU vs LSTM | −0.285 | No (p>0.05) |
| GRU vs Transformer | −3.103 | **Yes (p<0.05)** |
| LSTM vs Transformer | −2.511 | **Yes (p<0.05)** |
| BiLSTM vs all others | +4.4 to +8.4 | **Yes (all pairs)** |

## Phase 7 — Visualization (2026-03)
- [x] `thesis_figures.py`: 6 publication-quality figures (PDF + PNG at 300 DPI)
  - Fig 1: Training convergence curves
  - Fig 2: Price prediction vs actual (GRU, test period)
  - Fig 3: MAPE/RMSE model comparison bar chart
  - Fig 4: Diebold-Mariano heatmap
  - Fig 5: Ablation study
  - Fig 6: Feature engineering overview

## Phase 8 — Thesis Writing (upcoming)
- [ ] Chapter 1: Introduction
- [ ] Chapter 2: Literature Review
- [ ] Chapter 3: Methodology
- [ ] Chapter 4: Results
- [ ] Chapter 5: Discussion
- [ ] Chapter 6: Conclusion
- [ ] Abstract
- [ ] References

## Known Issues / Technical Notes

1. **CNN-LSTM checkpoint mismatch**: Ablation run (use_ltst=False) overwrites the scaler
   with 114 features. Workaround: delete checkpoint and retrain before DM testing.
   Fix: ablation should use a separate scaler path.

2. **BiLSTM causal concern**: BiLSTM uses backward-pass future information, which violates
   causality for live trading. Results are valid for research comparison but BiLSTM should
   not be deployed in a live trading system.

3. **DA at 50% for h=1**: All models predict direction near-randomly at 1-day horizon,
   consistent with weak-form EMH. DA expected to improve at h=7, 14, 30.

4. **CNN-LSTM collapse with combined loss**: CNN-LSTM tends to collapse (val_loss→0)
   with `combined` loss. Switched to `huber` loss for stable training.
"""

docs["docs/api_reference.md"] = """\
# API Reference
**Deep Learning for Cryptocurrency Price Forecasting**
*Muluh Penn Junior Patrick — M.Tech. Thesis 2026*

---

## `src.models`

### `get_model(name, **kwargs) → nn.Module`
Instantiate a model by name.

```python
from src.models import get_model
model = get_model('lstm', input_size=149, hidden_size=128, num_layers=1, dropout=0.3)
```

**Available models:** `lstm`, `gru`, `bilstm`, `cnn_lstm`, `attention_lstm`, `transformer`

### `LSTMForecaster(input_size, hidden_size=256, num_layers=2, dropout=0.2, output_size=1)`
Stacked LSTM. Input: `(batch, seq_len, input_size)`. Output: `(batch, output_size)`.

### `GRUForecaster(...)` — same signature as LSTM.

### `BiLSTMForecaster(input_size, hidden_size=128, num_layers=2, dropout=0.2, output_size=1)`
Bidirectional LSTM. Output dimension = `hidden_size * 2`.

### `CNNLSTMForecaster(input_size, num_filters=64, kernel_size=3, hidden_size=128, num_layers=2, dropout=0.2, output_size=1)`
CNN feature extractor → LSTM sequence model.

### `AttentionLSTMForecaster(input_size, hidden_size=256, num_layers=2, dropout=0.2, output_size=1)`
LSTM with Bahdanau-style soft attention over the sequence.

### `TransformerForecaster(input_size, d_model=128, nhead=8, num_encoder_layers=3, dim_feedforward=512, dropout=0.1, output_size=1)`
Transformer encoder with positional encoding.

---

## `src.training.trainer`

### `train_model(model_name, asset, interval, horizon, ...) → dict`
Full training pipeline: data → features → normalise → train → evaluate.

```python
from src.training.trainer import train_model
results = train_model(
    model_name='gru', asset='BTC', interval='1d', horizon=1,
    seq_len=90, batch_size=64, lr=0.00448, max_epochs=200,
    loss_name='combined',
    model_kwargs={'hidden_size': 128, 'num_layers': 1, 'dropout': 0.296},
)
# results['test_metrics'] → dict with rmse, mape, r2, directional_accuracy, sharpe_ratio, ...
```

### `CryptoDataModule`
PyTorch Lightning DataModule. Handles data loading, feature engineering, normalisation, and sequence building.

```python
from src.training.trainer import CryptoDataModule
dm = CryptoDataModule(asset='BTC', interval='1d', seq_len=90, horizon=1, batch_size=64)
dm.setup()
# dm.train_dataloader(), dm.val_dataloader(), dm.test_dataloader()
# dm.n_features → int (number of input features)
```

---

## `src.evaluation`

### `evaluator.print_comparison_table(asset, interval, horizon)`
Print formatted KPI table for all 6 models.

### `evaluator.rank_models(asset, interval, horizon) → pd.DataFrame`
Return models ranked by average KPI rank.

### `diebold_mariano.dm_test(errors_a, errors_b, h=1, loss='mse') → (stat, p_value)`
Diebold-Mariano test for equal predictive accuracy.
- Negative stat → model A more accurate
- p < 0.05 → statistically significant difference

### `diebold_mariano.dm_matrix(errors_dict, h=1) → (dm_stats, p_values)`
Pairwise DM test for all model combinations.

### `ablation_study.run_full_ablation(model_name, asset, ...) → pd.DataFrame`
Run ablation removing each feature group (ltst, onchain, sentiment, macro) one at a time.

---

## `src.preprocessing`

### `TechnicalIndicatorComputer.compute(df) → pd.DataFrame`
Add 104 technical indicator features to OHLCV DataFrame.

### `LTSTDecomposer(methods=['ma']).decompose(df) → pd.DataFrame`
Add 35 LTST decomposition features.

### `CryptoNormalizer(method='minmax').fit(df) / .transform(df) / .inverse_transform(df)`
Zero-leakage normalizer. Always `fit` on training data only.

### `SequenceBuilder(seq_len, horizon, stride).build(df, target_col) → (X, y)`
Build sliding window sequences. Returns `(n_samples, seq_len, n_features)` and `(n_samples,)`.

---

## `src.visualization.thesis_figures`

```python
from src.visualization.thesis_figures import (
    fig1_training_curves,   # val_loss curves for all 6 models
    fig2_price_prediction,  # actual vs predicted BTC price
    fig3_model_comparison,  # MAPE + RMSE bar charts
    fig4_dm_heatmap,        # Diebold-Mariano significance matrix
    fig5_ablation,          # feature ablation ΔMAPE
    fig6_feature_overview,  # 149-feature pie chart
)
# All functions save PDF + PNG to experiments/figures/
```

---

## CLI Reference

```bash
# Training
python scripts/run_training.py --model gru --asset BTC --interval 1d --horizon 1
python scripts/run_training.py --model lstm --asset ETH --use-best-params

# Hyperparameter tuning
python scripts/run_tuning.py --model lstm --asset BTC --trials 30 --epochs 50
python scripts/run_tuning.py --model gru  --asset ETH --no-resume

# Evaluation
python scripts/run_evaluation.py --asset BTC --horizon 1 --dm
python scripts/run_evaluation.py --all

# Ablation study
python scripts/run_ablation.py --model lstm --asset BTC --epochs 100

# All 120 experiments (overnight)
python scripts/run_all_experiments.py --trials 15 --epochs 30
python scripts/run_all_experiments.py --asset BTC  # BTC only
python scripts/run_all_experiments.py --dry-run    # print plan only

# Save prediction arrays for DM testing
python save_predictions.py --asset BTC --interval 1d --horizon 1
```
"""

for path, content in docs.items():
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(content)
    print(f"  ✔  {path}")

print("\n  ✅  All 3 docs generated.")
print("\n  SUMMARY")
print("  ─" * 25)
print("  6 notebooks  →  notebooks/01..06_*.ipynb")
print("  5 scripts    →  scripts/run_*.py")
print("  3 docs       →  docs/*.md")
