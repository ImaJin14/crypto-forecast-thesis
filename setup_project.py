#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════╗
║   crypto-forecast-thesis — Project Scaffold Setup Script             ║
║   Deep Learning for Cryptocurrency Price Forecasting                 ║
║   Author : Muluh Penn Junior Patrick                                 ║
║   Usage  : python setup_project.py [--root ./crypto-forecast-thesis] ║
╚══════════════════════════════════════════════════════════════════════╝
Run from inside your cloned GitHub repo root:
    cd crypto-forecast-thesis
    python setup_project.py
"""

import os
import sys
import argparse
import textwrap
from pathlib import Path

# ─── ANSI colors ──────────────────────────────────────────────────────────────
G  = "\033[92m"   # green
B  = "\033[94m"   # blue
Y  = "\033[93m"   # yellow
C  = "\033[96m"   # cyan
R  = "\033[91m"   # red
DIM= "\033[2m"
RST= "\033[0m"
BOLD="\033[1m"

def log(msg, color=G):   print(f"{color}  ✔  {RST}{msg}")
def section(msg):        print(f"\n{BOLD}{C}{'─'*60}{RST}\n{BOLD}{C}  {msg}{RST}\n{'─'*60}")
def warn(msg):           print(f"{Y}  ⚠  {msg}{RST}")
def done(msg):           print(f"\n{BOLD}{G}  ✅  {msg}{RST}\n")

# ══════════════════════════════════════════════════════════════════════════════
#  FILE CONTENTS
# ══════════════════════════════════════════════════════════════════════════════

FILES = {}

# ── .gitignore ────────────────────────────────────────────────────────────────
FILES[".gitignore"] = textwrap.dedent("""\
    # Python
    __pycache__/
    *.py[cod]
    *.pyo
    *.pyd
    .Python
    *.egg-info/
    dist/
    build/
    .eggs/

    # Environments
    .env
    .venv/
    venv/
    ENV/

    # Data (never commit raw/processed data)
    data/raw/
    data/processed/
    data/external/

    # Model artifacts
    experiments/checkpoints/
    experiments/runs/
    *.pt
    *.pth
    *.ckpt

    # Notebooks checkpoints
    .ipynb_checkpoints/

    # IDEs
    .vscode/
    .idea/
    *.swp

    # OS
    .DS_Store
    Thumbs.db

    # MLflow / W&B
    mlruns/
    wandb/

    # Logs
    *.log
    logs/
""")

# ── requirements.txt ──────────────────────────────────────────────────────────
FILES["requirements.txt"] = textwrap.dedent("""\
    # ── Core ──────────────────────────────────────────────────────
    numpy>=1.24.0
    pandas>=2.0.0
    scipy>=1.11.0
    scikit-learn>=1.3.0

    # ── Deep Learning ─────────────────────────────────────────────
    torch>=2.1.0
    pytorch-lightning>=2.1.0
    torchmetrics>=1.2.0

    # ── Hyperparameter Tuning ─────────────────────────────────────
    optuna>=3.4.0
    optuna-integration[pytorch-lightning]

    # ── Data Collection ───────────────────────────────────────────
    ccxt>=4.1.0
    requests>=2.31.0
    yfinance>=0.2.31
    pyarrow>=14.0.0

    # ── Feature Engineering ───────────────────────────────────────
    ta>=0.11.0
    PyWavelets>=1.4.1
    statsmodels>=0.14.0

    # ── NLP / Sentiment ───────────────────────────────────────────
    transformers>=4.35.0
    torch-sentiment>=0.1.0
    vaderSentiment>=3.3.2

    # ── Experiment Tracking ───────────────────────────────────────
    mlflow>=2.8.0
    wandb>=0.16.0

    # ── Visualization ─────────────────────────────────────────────
    matplotlib>=3.8.0
    seaborn>=0.13.0
    plotly>=5.18.0
    shap>=0.44.0

    # ── Utilities ─────────────────────────────────────────────────
    python-dotenv>=1.0.0
    pyyaml>=6.0.1
    loguru>=0.7.2
    rich>=13.7.0
    tqdm>=4.66.0
    joblib>=1.3.2

    # ── Testing ───────────────────────────────────────────────────
    pytest>=7.4.0
    pytest-cov>=4.1.0
""")

# ── environment.yml ───────────────────────────────────────────────────────────
FILES["environment.yml"] = textwrap.dedent("""\
    name: crypto-forecast
    channels:
      - pytorch
      - conda-forge
      - defaults
    dependencies:
      - python=3.11
      - pip
      - numpy
      - pandas
      - scipy
      - scikit-learn
      - matplotlib
      - seaborn
      - jupyterlab
      - ipykernel
      - pytorch
      - torchvision
      - pip:
          - -r requirements.txt
""")

# ── pyproject.toml ────────────────────────────────────────────────────────────
FILES["pyproject.toml"] = textwrap.dedent("""\
    [project]
    name = "crypto-forecast-thesis"
    version = "0.1.0"
    description = "Deep learning for cryptocurrency price forecasting: a multi-model approach"
    authors = [{name = "Muluh Penn Junior Patrick"}]
    requires-python = ">=3.11"
    readme = "README.md"
    license = {text = "MIT"}

    [tool.black]
    line-length = 100
    target-version = ["py311"]

    [tool.ruff]
    line-length = 100
    select = ["E", "F", "I", "N", "W"]
    ignore = ["E501"]

    [tool.pytest.ini_options]
    testpaths = ["tests"]
    python_files = ["test_*.py"]
    addopts = "-v --cov=src --cov-report=term-missing"
""")

# ── Makefile ──────────────────────────────────────────────────────────────────
FILES["Makefile"] = textwrap.dedent("""\
    .PHONY: help setup data train eval tune ablation test lint clean

    help:
    \t@echo ""
    \t@echo "  crypto-forecast-thesis — Available Commands"
    \t@echo "  ─────────────────────────────────────────────"
    \t@echo "  make setup      Install dependencies"
    \t@echo "  make data       Run full data collection + preprocessing pipeline"
    \t@echo "  make train      Train a model (MODEL=lstm ASSET=BTC HORIZON=1d)"
    \t@echo "  make tune       Run Optuna hyperparameter search"
    \t@echo "  make eval       Evaluate all models and generate KPI tables"
    \t@echo "  make ablation   Run feature ablation study"
    \t@echo "  make test       Run pytest test suite"
    \t@echo "  make lint       Run ruff + black formatting checks"
    \t@echo "  make clean      Remove __pycache__ and .pyc files"
    \t@echo ""

    setup:
    \tpip install -r requirements.txt

    data:
    \tpython scripts/run_data_pipeline.py

    train:
    \tpython scripts/run_training.py --model $(MODEL) --asset $(ASSET) --horizon $(HORIZON)

    tune:
    \tpython scripts/run_tuning.py --model $(MODEL) --asset $(ASSET)

    eval:
    \tpython scripts/run_evaluation.py

    ablation:
    \tpython scripts/run_ablation.py

    test:
    \tpytest tests/ -v --cov=src

    lint:
    \truff check src/ scripts/ tests/
    \tblack --check src/ scripts/ tests/

    clean:
    \tfind . -type d -name __pycache__ -exec rm -rf {} +
    \tfind . -name "*.pyc" -delete
    \tfind . -name "*.pyo" -delete
""")

# ── README.md ─────────────────────────────────────────────────────────────────
FILES["README.md"] = textwrap.dedent("""\
    # 🧠 Deep Learning for Cryptocurrency Price Forecasting
    ### A Multi-Model Approach Using Neural Networks

    > **M.Tech. Thesis | 2026 | Muluh Penn Junior Patrick**

    ---

    ## Overview

    This repository contains the full implementation of a master's thesis comparing six deep
    learning architectures for multi-asset, multi-horizon cryptocurrency price forecasting.

    **Assets:** BTC · ETH · SOL · SUI · XRP
    **Models:** LSTM · GRU · CNN-LSTM · Attention-LSTM · Transformer · BiLSTM
    **Horizons:** 1h (intraday) · 1–7d (short-term) · 7–30d (medium-term)

    ## Project Structure

    ```
    crypto-forecast-thesis/
    ├── data/               # Raw, processed, and external datasets
    ├── src/                # All source code (modular)
    │   ├── data_collection/
    │   ├── preprocessing/
    │   ├── models/
    │   ├── training/
    │   ├── tuning/
    │   ├── evaluation/
    │   ├── visualization/
    │   └── utils/
    ├── experiments/        # Configs, checkpoints, results
    ├── notebooks/          # EDA through results analysis
    ├── scripts/            # CLI entry points
    ├── tests/              # pytest test suite
    └── docs/               # Thesis chapters and data dictionary
    ```

    ## Quick Start

    ```bash
    # 1. Clone
    git clone https://github.com/ImaJin14/crypto-forecast-thesis.git
    cd crypto-forecast-thesis

    # 2. Setup environment
    conda env create -f environment.yml
    conda activate crypto-forecast
    # OR
    pip install -r requirements.txt

    # 3. Configure API keys
    cp .env.example .env
    # Edit .env with your Binance, Glassnode, W&B API keys

    # 4. Collect data
    make data

    # 5. Train a model
    make train MODEL=lstm ASSET=BTC HORIZON=1d

    # 6. Evaluate all models
    make eval
    ```

    ## Key Performance Indicators

    | Metric | Target |
    |--------|--------|
    | MAPE   | < 5%   |
    | Directional Accuracy | > 60% |
    | R²     | > 0.85 |
    | Sharpe Ratio | Maximize |

    ## License

    MIT License — see [LICENSE](LICENSE) for details.
""")

# ── .env.example ──────────────────────────────────────────────────────────────
FILES[".env.example"] = textwrap.dedent("""\
    # ── Binance API ──────────────────────────────────────────
    BINANCE_API_KEY=your_binance_api_key_here
    BINANCE_API_SECRET=your_binance_api_secret_here

    # ── Glassnode (On-chain metrics) ──────────────────────────
    GLASSNODE_API_KEY=your_glassnode_api_key_here

    # ── CoinGecko (backup price source) ───────────────────────
    COINGECKO_API_KEY=your_coingecko_api_key_here

    # ── Experiment Tracking ───────────────────────────────────
    WANDB_API_KEY=your_wandb_api_key_here
    MLFLOW_TRACKING_URI=http://localhost:5000

    # ── Paths ─────────────────────────────────────────────────
    DATA_DIR=./data
    CHECKPOINT_DIR=./experiments/checkpoints
    RESULTS_DIR=./experiments/results
""")

# ── base_config.yaml ──────────────────────────────────────────────────────────
FILES["experiments/configs/base_config.yaml"] = textwrap.dedent("""\
    # Base configuration — inherited by all experiment configs

    project:
      name: "crypto-forecast-thesis"
      author: "Muluh Penn Junior Patrick"
      seed: 42

    data:
      assets: [BTC, ETH, SOL, SUI, XRP]
      intervals: [1h, 1d]
      start_date: "2020-01-01"
      end_date: "2025-12-31"
      sequence_length: 60
      train_ratio: 0.70
      val_ratio: 0.15
      test_ratio: 0.15

    features:
      price_volume: true
      technical_indicators: true
      onchain_metrics: true
      sentiment: true
      macro: true
      ltst_decomposition: true

    training:
      max_epochs: 200
      batch_size: 32
      learning_rate: 0.001
      optimizer: adam
      scheduler: cosine_annealing
      early_stopping_patience: 15
      dropout: 0.2
      weight_decay: 0.0001

    evaluation:
      metrics: [rmse, mae, mape, r2, directional_accuracy, sharpe_ratio, max_drawdown, win_rate]
      dm_test: true
      ablation: true

    logging:
      mlflow: true
      wandb: false
      log_every_n_steps: 10
""")

# ── lstm_btc_1h.yaml ──────────────────────────────────────────────────────────
FILES["experiments/configs/lstm_btc_1h.yaml"] = textwrap.dedent("""\
    # LSTM · BTC · 1-hour horizon
    defaults:
      - base_config

    model:
      name: lstm
      hidden_size: 256
      num_layers: 2
      dropout: 0.3

    data:
      target_asset: BTC
      interval: 1h
      forecast_horizon: 1

    training:
      max_epochs: 150
      batch_size: 64
      learning_rate: 0.001
""")

# ── transformer_eth_1d.yaml ───────────────────────────────────────────────────
FILES["experiments/configs/transformer_eth_1d.yaml"] = textwrap.dedent("""\
    # Transformer · ETH · daily horizon
    defaults:
      - base_config

    model:
      name: transformer
      d_model: 128
      nhead: 8
      num_encoder_layers: 3
      dim_feedforward: 512
      dropout: 0.1

    data:
      target_asset: ETH
      interval: 1d
      forecast_horizon: 7

    training:
      max_epochs: 200
      batch_size: 32
      learning_rate: 0.0005
""")

# ── ablation_no_onchain.yaml ──────────────────────────────────────────────────
FILES["experiments/configs/ablation_no_onchain.yaml"] = textwrap.dedent("""\
    # Ablation: remove on-chain features
    defaults:
      - base_config

    features:
      onchain_metrics: false

    experiment:
      ablation_group: no_onchain
      description: "Measure marginal impact of on-chain metrics"
""")

# ── src/__init__.py ────────────────────────────────────────────────────────────
FILES["src/__init__.py"] = '"""crypto-forecast-thesis source package."""\n'

# ── src/utils/seed.py ─────────────────────────────────────────────────────────
FILES["src/utils/seed.py"] = textwrap.dedent("""\
    \"\"\"Fix random seeds across all libraries for reproducibility.\"\"\"
    import os
    import random
    import numpy as np


    def set_seed(seed: int = 42) -> None:
        \"\"\"Set seed for Python, NumPy, and PyTorch.\"\"\"
        random.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        np.random.seed(seed)
        try:
            import torch
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        except ImportError:
            pass
""")

# ── src/utils/logger.py ───────────────────────────────────────────────────────
FILES["src/utils/logger.py"] = textwrap.dedent("""\
    \"\"\"Structured logging using loguru.\"\"\"
    import sys
    from loguru import logger


    def setup_logger(level: str = "INFO", log_file: str = None) -> None:
        logger.remove()
        logger.add(sys.stderr, level=level,
                   format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
                          "<level>{level: <8}</level> | "
                          "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> — "
                          "<level>{message}</level>")
        if log_file:
            logger.add(log_file, rotation="10 MB", retention="30 days", level=level)


    def get_logger(name: str = __name__):
        return logger.bind(module=name)
""")

# ── src/utils/device.py ───────────────────────────────────────────────────────
FILES["src/utils/device.py"] = textwrap.dedent("""\
    \"\"\"CUDA / MPS / CPU device detection.\"\"\"
    import torch


    def get_device() -> torch.device:
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print(f"  🚀  Using GPU: {torch.cuda.get_device_name(0)}")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
            print("  🍎  Using Apple MPS")
        else:
            device = torch.device("cpu")
            print("  💻  Using CPU")
        return device
""")

# ── src/utils/config_loader.py ────────────────────────────────────────────────
FILES["src/utils/config_loader.py"] = textwrap.dedent("""\
    \"\"\"Load and merge YAML experiment configs.\"\"\"
    from pathlib import Path
    import yaml


    def load_config(config_path: str | Path) -> dict:
        path = Path(config_path)
        with open(path) as f:
            cfg = yaml.safe_load(f)

        # Merge with base_config if 'defaults' key present
        if "defaults" in cfg:
            base_path = path.parent / "base_config.yaml"
            with open(base_path) as f:
                base = yaml.safe_load(f)
            base.update(cfg)
            cfg = base

        return cfg
""")

# ── src/evaluation/metrics.py ─────────────────────────────────────────────────
FILES["src/evaluation/metrics.py"] = textwrap.dedent("""\
    \"\"\"Core forecasting KPI metrics.\"\"\"
    import numpy as np
    from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


    def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        \"\"\"Root Mean Squared Error.\"\"\"
        return float(np.sqrt(mean_squared_error(y_true, y_pred)))


    def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        \"\"\"Mean Absolute Error.\"\"\"
        return float(mean_absolute_error(y_true, y_pred))


    def mape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
        \"\"\"Mean Absolute Percentage Error (%).\"\"\"
        return float(np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + eps))) * 100)


    def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        \"\"\"R² coefficient of determination.\"\"\"
        return float(r2_score(y_true, y_pred))


    def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        \"\"\"Percentage of correct up/down direction predictions.\"\"\"
        true_dir = np.diff(y_true) > 0
        pred_dir = np.diff(y_pred) > 0
        return float(np.mean(true_dir == pred_dir) * 100)


    def compute_all_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
        \"\"\"Compute all forecasting KPIs and return as dict.\"\"\"
        return {
            "rmse": rmse(y_true, y_pred),
            "mae":  mae(y_true, y_pred),
            "mape": mape(y_true, y_pred),
            "r2":   r2(y_true, y_pred),
            "directional_accuracy": directional_accuracy(y_true, y_pred),
        }
""")

# ── src/evaluation/financial_metrics.py ──────────────────────────────────────
FILES["src/evaluation/financial_metrics.py"] = textwrap.dedent("""\
    \"\"\"Financial KPIs: Sharpe Ratio, Max Drawdown, Win Rate.\"\"\"
    import numpy as np


    def sharpe_ratio(returns: np.ndarray, risk_free_rate: float = 0.0,
                     periods_per_year: int = 252) -> float:
        \"\"\"Annualized Sharpe Ratio from daily/hourly return series.\"\"\"
        excess = returns - risk_free_rate / periods_per_year
        if excess.std() == 0:
            return 0.0
        return float(np.sqrt(periods_per_year) * excess.mean() / excess.std())


    def max_drawdown(equity_curve: np.ndarray) -> float:
        \"\"\"Maximum peak-to-trough drawdown as a fraction.\"\"\"
        peak = np.maximum.accumulate(equity_curve)
        drawdown = (equity_curve - peak) / peak
        return float(drawdown.min())


    def win_rate(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        \"\"\"% of trades where direction prediction was profitable.\"\"\"
        true_dir = np.diff(y_true) > 0
        pred_dir = np.diff(y_pred) > 0
        profitable = (true_dir == pred_dir)
        return float(profitable.mean() * 100)


    def profit_factor(returns: np.ndarray) -> float:
        \"\"\"Gross profit / gross loss ratio.\"\"\"
        gains  = returns[returns > 0].sum()
        losses = abs(returns[returns < 0].sum())
        return float(gains / losses) if losses != 0 else float("inf")
""")

# ── src/models/base_model.py ──────────────────────────────────────────────────
FILES["src/models/base_model.py"] = textwrap.dedent("""\
    \"\"\"Abstract base class for all forecast models.\"\"\"
    from abc import ABC, abstractmethod
    import torch
    import torch.nn as nn
    import pytorch_lightning as pl


    class BaseForecaster(pl.LightningModule, ABC):
        \"\"\"Shared interface for all deep learning forecasting models.\"\"\"

        def __init__(self, input_size: int, output_size: int = 1,
                     learning_rate: float = 1e-3):
            super().__init__()
            self.input_size   = input_size
            self.output_size  = output_size
            self.learning_rate = learning_rate
            self.save_hyperparameters()

        @abstractmethod
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            \"\"\"Forward pass. x: (batch, seq_len, input_size)\"\"\"
            ...

        def _shared_step(self, batch):
            x, y = batch
            y_hat = self(x)
            loss  = nn.MSELoss()(y_hat.squeeze(), y.squeeze())
            return loss, y_hat.squeeze(), y.squeeze()

        def training_step(self, batch, batch_idx):
            loss, _, _ = self._shared_step(batch)
            self.log("train_loss", loss, prog_bar=True)
            return loss

        def validation_step(self, batch, batch_idx):
            loss, _, _ = self._shared_step(batch)
            self.log("val_loss", loss, prog_bar=True)
            return loss

        def test_step(self, batch, batch_idx):
            loss, y_hat, y = self._shared_step(batch)
            self.log("test_loss", loss)
            return {"preds": y_hat, "targets": y}

        def configure_optimizers(self):
            optimizer = torch.optim.Adam(
                self.parameters(), lr=self.learning_rate, weight_decay=1e-4
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=50, eta_min=1e-6
            )
            return {"optimizer": optimizer,
                    "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"}}
""")

# ── src/models/lstm_model.py ──────────────────────────────────────────────────
FILES["src/models/lstm_model.py"] = textwrap.dedent("""\
    \"\"\"Vanilla LSTM forecasting model.\"\"\"
    import torch
    import torch.nn as nn
    from .base_model import BaseForecaster


    class LSTMForecaster(BaseForecaster):
        \"\"\"
        Stacked LSTM for price sequence forecasting.

        Args:
            input_size:   Number of input features per timestep
            hidden_size:  LSTM hidden state dimension
            num_layers:   Number of stacked LSTM layers
            dropout:      Dropout rate between layers
            output_size:  Forecast horizon (default 1)
        \"\"\"

        def __init__(self, input_size: int, hidden_size: int = 256,
                     num_layers: int = 2, dropout: float = 0.2,
                     output_size: int = 1, learning_rate: float = 1e-3):
            super().__init__(input_size, output_size, learning_rate)
            self.hidden_size = hidden_size
            self.num_layers  = num_layers

            self.lstm = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0.0,
                batch_first=True,
            )
            self.dropout = nn.Dropout(dropout)
            self.fc      = nn.Linear(hidden_size, output_size)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # x: (batch, seq_len, input_size)
            out, _ = self.lstm(x)         # (batch, seq_len, hidden)
            out    = self.dropout(out)
            out    = self.fc(out[:, -1])  # last timestep → (batch, output)
            return out
""")

# ── src/models/gru_model.py ───────────────────────────────────────────────────
FILES["src/models/gru_model.py"] = textwrap.dedent("""\
    \"\"\"GRU forecasting model — lightweight LSTM variant.\"\"\"
    import torch
    import torch.nn as nn
    from .base_model import BaseForecaster


    class GRUForecaster(BaseForecaster):
        def __init__(self, input_size: int, hidden_size: int = 256,
                     num_layers: int = 2, dropout: float = 0.2,
                     output_size: int = 1, learning_rate: float = 1e-3):
            super().__init__(input_size, output_size, learning_rate)
            self.gru = nn.GRU(
                input_size=input_size, hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0.0,
                batch_first=True,
            )
            self.dropout = nn.Dropout(dropout)
            self.fc      = nn.Linear(hidden_size, output_size)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            out, _ = self.gru(x)
            out    = self.dropout(out[:, -1])
            return self.fc(out)
""")

# ── src/models/bilstm_model.py ────────────────────────────────────────────────
FILES["src/models/bilstm_model.py"] = textwrap.dedent("""\
    \"\"\"Bidirectional LSTM forecasting model.\"\"\"
    import torch
    import torch.nn as nn
    from .base_model import BaseForecaster


    class BiLSTMForecaster(BaseForecaster):
        def __init__(self, input_size: int, hidden_size: int = 128,
                     num_layers: int = 2, dropout: float = 0.2,
                     output_size: int = 1, learning_rate: float = 1e-3):
            super().__init__(input_size, output_size, learning_rate)
            self.lstm = nn.LSTM(
                input_size=input_size, hidden_size=hidden_size,
                num_layers=num_layers, bidirectional=True,
                dropout=dropout if num_layers > 1 else 0.0,
                batch_first=True,
            )
            self.dropout = nn.Dropout(dropout)
            self.fc      = nn.Linear(hidden_size * 2, output_size)  # *2 for bidirectional

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            out, _ = self.lstm(x)
            out    = self.dropout(out[:, -1])
            return self.fc(out)
""")

# ── src/models/cnn_lstm_model.py ──────────────────────────────────────────────
FILES["src/models/cnn_lstm_model.py"] = textwrap.dedent("""\
    \"\"\"CNN + LSTM hybrid: CNN extracts local features, LSTM models temporal dynamics.\"\"\"
    import torch
    import torch.nn as nn
    from .base_model import BaseForecaster


    class CNNLSTMForecaster(BaseForecaster):
        def __init__(self, input_size: int, num_filters: int = 64,
                     kernel_size: int = 3, hidden_size: int = 128,
                     num_layers: int = 2, dropout: float = 0.2,
                     output_size: int = 1, learning_rate: float = 1e-3):
            super().__init__(input_size, output_size, learning_rate)
            self.conv = nn.Sequential(
                nn.Conv1d(input_size, num_filters, kernel_size, padding=kernel_size // 2),
                nn.ReLU(),
                nn.Conv1d(num_filters, num_filters, kernel_size, padding=kernel_size // 2),
                nn.ReLU(),
            )
            self.lstm    = nn.LSTM(num_filters, hidden_size, num_layers,
                                   batch_first=True,
                                   dropout=dropout if num_layers > 1 else 0.0)
            self.dropout = nn.Dropout(dropout)
            self.fc      = nn.Linear(hidden_size, output_size)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # x: (batch, seq_len, features) → conv needs (batch, features, seq_len)
            x   = x.permute(0, 2, 1)
            x   = self.conv(x)
            x   = x.permute(0, 2, 1)   # back to (batch, seq_len, filters)
            out, _ = self.lstm(x)
            out    = self.dropout(out[:, -1])
            return self.fc(out)
""")

# ── src/models/attention_lstm_model.py ───────────────────────────────────────
FILES["src/models/attention_lstm_model.py"] = textwrap.dedent("""\
    \"\"\"LSTM + Bahdanau-style attention mechanism.\"\"\"
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from .base_model import BaseForecaster


    class AttentionLSTMForecaster(BaseForecaster):
        def __init__(self, input_size: int, hidden_size: int = 256,
                     num_layers: int = 2, dropout: float = 0.2,
                     output_size: int = 1, learning_rate: float = 1e-3):
            super().__init__(input_size, output_size, learning_rate)
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                                batch_first=True,
                                dropout=dropout if num_layers > 1 else 0.0)
            self.attention = nn.Linear(hidden_size, 1)
            self.dropout   = nn.Dropout(dropout)
            self.fc        = nn.Linear(hidden_size, output_size)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            out, _ = self.lstm(x)                        # (B, T, H)
            scores = self.attention(out).squeeze(-1)      # (B, T)
            weights = F.softmax(scores, dim=1).unsqueeze(2)  # (B, T, 1)
            context = (out * weights).sum(dim=1)          # (B, H)
            context = self.dropout(context)
            return self.fc(context)
""")

# ── src/models/transformer_model.py ──────────────────────────────────────────
FILES["src/models/transformer_model.py"] = textwrap.dedent("""\
    \"\"\"Transformer encoder for time-series forecasting.\"\"\"
    import math
    import torch
    import torch.nn as nn
    from .base_model import BaseForecaster


    class PositionalEncoding(nn.Module):
        def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
            super().__init__()
            self.dropout = nn.Dropout(dropout)
            pe = torch.zeros(max_len, d_model)
            pos = torch.arange(0, max_len).unsqueeze(1).float()
            div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
            pe[:, 0::2] = torch.sin(pos * div)
            pe[:, 1::2] = torch.cos(pos * div)
            self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

        def forward(self, x):
            return self.dropout(x + self.pe[:, :x.size(1)])


    class TransformerForecaster(BaseForecaster):
        def __init__(self, input_size: int, d_model: int = 128, nhead: int = 8,
                     num_encoder_layers: int = 3, dim_feedforward: int = 512,
                     dropout: float = 0.1, output_size: int = 1,
                     learning_rate: float = 1e-3):
            super().__init__(input_size, output_size, learning_rate)
            self.input_proj  = nn.Linear(input_size, d_model)
            self.pos_enc     = PositionalEncoding(d_model, dropout=dropout)
            encoder_layer    = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout, batch_first=True
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
            self.fc          = nn.Linear(d_model, output_size)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.input_proj(x)   # (B, T, d_model)
            x = self.pos_enc(x)
            x = self.transformer(x)
            return self.fc(x[:, -1]) # last token
""")

# ── src/models/__init__.py ────────────────────────────────────────────────────
FILES["src/models/__init__.py"] = textwrap.dedent("""\
    from .lstm_model          import LSTMForecaster
    from .gru_model           import GRUForecaster
    from .bilstm_model        import BiLSTMForecaster
    from .cnn_lstm_model      import CNNLSTMForecaster
    from .attention_lstm_model import AttentionLSTMForecaster
    from .transformer_model   import TransformerForecaster

    MODEL_REGISTRY = {
        "lstm":           LSTMForecaster,
        "gru":            GRUForecaster,
        "bilstm":         BiLSTMForecaster,
        "cnn_lstm":       CNNLSTMForecaster,
        "attention_lstm": AttentionLSTMForecaster,
        "transformer":    TransformerForecaster,
    }

    def get_model(name: str, **kwargs):
        if name not in MODEL_REGISTRY:
            raise ValueError(f"Unknown model '{name}'. Available: {list(MODEL_REGISTRY)}")
        return MODEL_REGISTRY[name](**kwargs)
""")

# ── scripts/run_training.py ───────────────────────────────────────────────────
FILES["scripts/run_training.py"] = textwrap.dedent("""\
    #!/usr/bin/env python3
    \"\"\"
    Train a single model.
    Usage: python scripts/run_training.py --model lstm --asset BTC --horizon 1d
    \"\"\"
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

        print(f"\\n  🚀  Training  | Model: {args.model} | Asset: {args.asset} | Horizon: {args.horizon}\\n")

        # TODO: Load data, build DataModule, instantiate model, run Trainer
        # Full implementation in Phase 5 of the thesis roadmap
        model = get_model(args.model, input_size=50)  # placeholder input_size
        print(f"  ✔   Model instantiated: {model.__class__.__name__}")
        print(f"  ✔   Parameters: {sum(p.numel() for p in model.parameters()):,}")
        print("\\n  ⚙️   Data pipeline and training loop coming in Phase 5...")


    if __name__ == "__main__":
        main()
""")

# ── scripts/run_evaluation.py ────────────────────────────────────────────────
FILES["scripts/run_evaluation.py"] = textwrap.dedent("""\
    #!/usr/bin/env python3
    \"\"\"Evaluate all trained models and generate KPI comparison tables.\"\"\"
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from src.evaluation.metrics           import compute_all_metrics
    from src.evaluation.financial_metrics import sharpe_ratio, max_drawdown, win_rate

    print("\\n  📊  Evaluation pipeline — coming in Phase 7 of the roadmap.\\n")
    print("  Metrics available:")
    import inspect, src.evaluation.metrics as m
    for name, fn in inspect.getmembers(m, inspect.isfunction):
        print(f"    ✔  {name}")
""")

# ── scripts/run_data_pipeline.py ─────────────────────────────────────────────
FILES["scripts/run_data_pipeline.py"] = textwrap.dedent("""\
    #!/usr/bin/env python3
    \"\"\"End-to-end data collection and preprocessing pipeline.\"\"\"
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))

    print("\\n  📡  Data pipeline — implementation in Phase 2.\\n")
    print("  Will collect: OHLCV · On-chain · Sentiment · Macro · LTST features")
    print("  Assets: BTC · ETH · SOL · SUI · XRP")
""")

# ── scripts/run_tuning.py ────────────────────────────────────────────────────
FILES["scripts/run_tuning.py"] = textwrap.dedent("""\
    #!/usr/bin/env python3
    \"\"\"Run Optuna hyperparameter optimization for a given model.\"\"\"
    import argparse, sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))

    p = argparse.ArgumentParser()
    p.add_argument("--model",  default="lstm")
    p.add_argument("--asset",  default="BTC")
    p.add_argument("--trials", type=int, default=50)
    args = p.parse_args()
    print(f"\\n  🔍  Optuna tuning: {args.model} on {args.asset} — {args.trials} trials")
    print("  Full implementation in Phase 6 of the roadmap.\\n")
""")

# ── tests/conftest.py ────────────────────────────────────────────────────────
FILES["tests/conftest.py"] = textwrap.dedent("""\
    \"\"\"Pytest fixtures — sample tensors and mock data.\"\"\"
    import pytest
    import torch
    import numpy as np


    @pytest.fixture
    def sample_batch():
        \"\"\"A batch of (X, y) tensors with shape (32, 60, 20).\"\"\"
        X = torch.randn(32, 60, 20)   # batch=32, seq=60, features=20
        y = torch.randn(32, 1)
        return X, y


    @pytest.fixture
    def sample_prices():
        \"\"\"1000 synthetic daily price points for metric testing.\"\"\"
        np.random.seed(42)
        prices = np.cumsum(np.random.randn(1000)) + 100
        return prices.astype(np.float32)
""")

# ── tests/test_models.py ──────────────────────────────────────────────────────
FILES["tests/test_models.py"] = textwrap.dedent("""\
    \"\"\"Forward pass shape validation for all models.\"\"\"
    import pytest
    import torch
    from src.models import MODEL_REGISTRY


    @pytest.mark.parametrize("model_name", list(MODEL_REGISTRY.keys()))
    def test_forward_pass_shape(model_name, sample_batch):
        X, y = sample_batch
        model = MODEL_REGISTRY[model_name](input_size=20)
        model.eval()
        with torch.no_grad():
            out = model(X)
        assert out.shape == (32, 1), f"{model_name}: expected (32,1), got {out.shape}"


    @pytest.mark.parametrize("model_name", list(MODEL_REGISTRY.keys()))
    def test_model_has_parameters(model_name):
        model = MODEL_REGISTRY[model_name](input_size=20)
        n_params = sum(p.numel() for p in model.parameters())
        assert n_params > 0, f"{model_name} has no parameters"
""")

# ── tests/test_metrics.py ────────────────────────────────────────────────────
FILES["tests/test_metrics.py"] = textwrap.dedent("""\
    \"\"\"Unit tests for KPI metric functions.\"\"\"
    import numpy as np
    import pytest
    from src.evaluation.metrics import rmse, mae, mape, r2, directional_accuracy
    from src.evaluation.financial_metrics import sharpe_ratio, win_rate


    @pytest.fixture
    def perfect_predictions(sample_prices):
        return sample_prices, sample_prices.copy()


    def test_rmse_perfect(perfect_predictions):
        y, yhat = perfect_predictions
        assert rmse(y, yhat) == pytest.approx(0.0, abs=1e-6)


    def test_r2_perfect(perfect_predictions):
        y, yhat = perfect_predictions
        assert r2(y, yhat) == pytest.approx(1.0, abs=1e-6)


    def test_mape_non_negative(sample_prices):
        noisy = sample_prices + np.random.randn(len(sample_prices))
        assert mape(sample_prices, noisy) >= 0


    def test_directional_accuracy_range(sample_prices):
        noisy = sample_prices + np.random.randn(len(sample_prices))
        da = directional_accuracy(sample_prices, noisy)
        assert 0 <= da <= 100
""")

# ── docs/data_dictionary.md ───────────────────────────────────────────────────
FILES["docs/data_dictionary.md"] = textwrap.dedent("""\
    # Data Dictionary

    ## Price / Volume Features (OHLCV)
    | Feature | Formula | Source |
    |---------|---------|--------|
    | open    | Opening price | Binance |
    | high    | Session high  | Binance |
    | low     | Session low   | Binance |
    | close   | Closing price | Binance |
    | volume  | Trade volume (base asset) | Binance |
    | quote_volume | Trade volume (quote asset, USDT) | Binance |

    ## Technical Indicators
    | Feature | Formula | Period |
    |---------|---------|--------|
    | rsi     | Relative Strength Index | 14 |
    | macd    | EMA(12) − EMA(26) | — |
    | macd_signal | EMA(9) of MACD | 9 |
    | bb_upper | SMA(20) + 2σ | 20 |
    | bb_lower | SMA(20) − 2σ | 20 |
    | bb_width | (upper − lower) / SMA | 20 |
    | ema_9   | Exponential MA | 9 |
    | ema_21  | Exponential MA | 21 |
    | atr     | Average True Range | 14 |
    | obv     | On-Balance Volume | — |

    ## On-Chain Metrics
    | Feature | Description | Source |
    |---------|-------------|--------|
    | active_addresses | Unique daily active addresses | Glassnode |
    | nvts    | NVT Signal (Market Cap / Tx Volume) | Glassnode |
    | hash_rate | Network hash rate | Glassnode |
    | mvrv    | Market Value to Realized Value | Glassnode |
    | sopr    | Spent Output Profit Ratio | Glassnode |

    ## Sentiment Features
    | Feature | Description | Source |
    |---------|-------------|--------|
    | fear_greed | Fear & Greed Index (0–100) | Alternative.me |
    | tweet_sentiment | FinBERT-scored tweet aggregate | Twitter/X API |
    | sentiment_lag_1 | 1-period lagged sentiment | Derived |
    | sentiment_lag_24 | 24-period lagged sentiment | Derived |

    ## Macro / Correlating Assets
    | Feature | Description | Source |
    |---------|-------------|--------|
    | sp500_return | Daily S&P 500 return | Yahoo Finance |
    | dxy_return | DXY Dollar Index return | Yahoo Finance |
    | gold_return | Gold spot return | Yahoo Finance |
    | vix | CBOE Volatility Index | Yahoo Finance |
    | btc_dom | BTC market dominance % | CoinGecko |

    ## LTST Decomposition
    | Feature | Description | Method |
    |---------|-------------|--------|
    | ltt_50  | Long-term trend (50-period) | SMA |
    | ltt_200 | Long-term trend (200-period) | SMA |
    | stt_5   | Short-term trend (5-period)  | EMA |
    | stt_20  | Short-term trend (20-period) | EMA |
    | hp_trend | HP filter trend component | Hodrick-Prescott |
    | residual | Price − HP trend | Derived |
""")

# ══════════════════════════════════════════════════════════════════════════════
#  DIRECTORY STRUCTURE
# ══════════════════════════════════════════════════════════════════════════════

DIRECTORIES = [
    "data/raw/ohlcv/BTC", "data/raw/ohlcv/ETH", "data/raw/ohlcv/SOL",
    "data/raw/ohlcv/SUI", "data/raw/ohlcv/XRP",
    "data/raw/onchain", "data/raw/sentiment", "data/raw/macro",
    "data/processed/features", "data/processed/sequences", "data/processed/scalers",
    "data/external",
    "src/data_collection", "src/preprocessing", "src/models",
    "src/training", "src/tuning", "src/evaluation", "src/visualization", "src/utils",
    "experiments/configs", "experiments/runs", "experiments/results", "experiments/checkpoints",
    "notebooks",
    "scripts",
    "tests",
    "docs/thesis",
]

# __init__.py for every src sub-package
INIT_PACKAGES = [
    "src/data_collection", "src/preprocessing", "src/training",
    "src/tuning", "src/evaluation", "src/visualization", "src/utils",
]

# placeholder stub files
STUBS = {
    "src/data_collection/binance_fetcher.py":    '"""Binance OHLCV fetcher via ccxt."""\n',
    "src/data_collection/coingecko_fetcher.py":  '"""CoinGecko backup price fetcher."""\n',
    "src/data_collection/glassnode_fetcher.py":  '"""Glassnode on-chain metrics fetcher."""\n',
    "src/data_collection/sentiment_fetcher.py":  '"""Fear & Greed + Twitter sentiment fetcher."""\n',
    "src/data_collection/macro_fetcher.py":      '"""Yahoo Finance macro data fetcher."""\n',
    "src/data_collection/data_validator.py":     '"""Data quality checks: gaps, outliers, alignment."""\n',
    "src/data_collection/pipeline.py":           '"""Orchestrates full data collection pipeline."""\n',
    "src/preprocessing/technical_indicators.py": '"""RSI, MACD, Bollinger Bands via ta library."""\n',
    "src/preprocessing/ltst_decomposition.py":   '"""Long-term / short-term trend decomposition."""\n',
    "src/preprocessing/sentiment_pipeline.py":   '"""FinBERT sentiment scoring and aggregation."""\n',
    "src/preprocessing/onchain_features.py":     '"""On-chain feature engineering."""\n',
    "src/preprocessing/macro_features.py":       '"""Macro cross-asset feature engineering."""\n',
    "src/preprocessing/normalizer.py":           '"""MinMax normalization with train-only fitting."""\n',
    "src/preprocessing/sequence_builder.py":     '"""Sliding window sequence construction."""\n',
    "src/preprocessing/stationarity.py":         '"""ADF test, differencing, stationarity checks."""\n',
    "src/training/trainer.py":                   '"""PyTorch Lightning training module."""\n',
    "src/training/loss_functions.py":            '"""MSE, Huber, and custom loss functions."""\n',
    "src/training/callbacks.py":                 '"""Early stopping, checkpointing, LR monitor."""\n',
    "src/training/optimizer_config.py":          '"""Optimizer and scheduler configuration."""\n',
    "src/training/walk_forward_cv.py":           '"""Rolling window walk-forward cross-validation."""\n',
    "src/tuning/optuna_study.py":                '"""Optuna hyperparameter study setup."""\n',
    "src/tuning/search_spaces.py":               '"""Search space definitions per model."""\n',
    "src/tuning/pruner.py":                      '"""Optuna pruning for early trial termination."""\n',
    "src/evaluation/diebold_mariano.py":         '"""Diebold-Mariano statistical significance test."""\n',
    "src/evaluation/ablation_study.py":          '"""Feature category ablation study."""\n',
    "src/evaluation/regime_analysis.py":         '"""Bull / Bear / Sideways market regime analysis."""\n',
    "src/evaluation/evaluator.py":               '"""Orchestrates full evaluation pipeline."""\n',
    "src/visualization/price_predictions.py":    '"""Predicted vs actual price overlay plots."""\n',
    "src/visualization/error_analysis.py":       '"""Residual and error distribution plots."""\n',
    "src/visualization/model_comparison.py":     '"""KPI comparison heatmaps across models."""\n',
    "src/visualization/feature_importance.py":   '"""SHAP value plots for feature importance."""\n',
    "src/visualization/training_curves.py":      '"""Training and validation loss curves."""\n',
    "src/visualization/correlation_heatmap.py":  '"""Feature correlation matrix heatmap."""\n',
    "src/models/ensemble.py":                    '"""Weighted ensemble / stacking wrapper."""\n',
    "src/utils/__init__.py":                     '"""Utility functions package."""\n',
    "src/utils/timer.py":                        '"""Benchmark training and inference time."""\n',
    "scripts/run_ablation.py":  '#!/usr/bin/env python3\n"""Run full feature ablation study."""\nprint("Ablation study — Phase 7")\n',
    "scripts/run_all_experiments.py": '#!/usr/bin/env python3\n"""Orchestrate all 120 experiments."""\nprint("Full experiment run — Phase 7")\n',
    "notebooks/01_data_exploration.ipynb":        None,
    "notebooks/02_feature_engineering_demo.ipynb": None,
    "notebooks/03_baseline_models.ipynb":          None,
    "notebooks/04_model_training_demo.ipynb":      None,
    "notebooks/05_results_analysis.ipynb":         None,
    "notebooks/06_visualization_gallery.ipynb":    None,
    "docs/api_reference.md":    '# API Reference\n\nGenerated from docstrings — coming soon.\n',
    "docs/experiment_log.md":   '# Experiment Log\n\n## 2026-03\n- [ ] Project scaffolded\n- [ ] Data collection pipeline\n',
}

MINIMAL_NOTEBOOK = """{
 "nbformat": 4,
 "nbformat_minor": 5,
 "metadata": {
  "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
  "language_info": {"name": "python", "version": "3.11.0"}
 },
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": ["# {title}\\n\\n> crypto-forecast-thesis — Muluh Penn Junior Patrick"],
   "id": "intro"
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": ["import sys\\nsys.path.insert(0, '..')\\n"],
   "id": "imports"
  }
 ]
}"""

NOTEBOOK_TITLES = {
    "notebooks/01_data_exploration.ipynb":         "01 — Data Exploration & EDA",
    "notebooks/02_feature_engineering_demo.ipynb": "02 — Feature Engineering Demo",
    "notebooks/03_baseline_models.ipynb":          "03 — Baseline Models (ARIMA, Naive)",
    "notebooks/04_model_training_demo.ipynb":      "04 — Model Training Demo",
    "notebooks/05_results_analysis.ipynb":         "05 — Results Analysis",
    "notebooks/06_visualization_gallery.ipynb":    "06 — Visualization Gallery",
}

# ══════════════════════════════════════════════════════════════════════════════
#  SCAFFOLD FUNCTION
# ══════════════════════════════════════════════════════════════════════════════

def scaffold(root: Path):
    print(f"""
{BOLD}{C}
╔══════════════════════════════════════════════════════════════╗
║  🧠  crypto-forecast-thesis  —  Project Scaffold             ║
╚══════════════════════════════════════════════════════════════╝{RST}
  Root: {root.resolve()}
""")

    # ── 1. Directories ──
    section("Creating directories")
    for d in DIRECTORIES:
        path = root / d
        path.mkdir(parents=True, exist_ok=True)
        # .gitkeep so empty dirs are tracked by git
        gitkeep = path / ".gitkeep"
        if not any(path.iterdir()) or not gitkeep.exists():
            gitkeep.touch()
        log(d, B)

    # ── 2. __init__.py files ──
    section("Creating package __init__.py files")
    for pkg in INIT_PACKAGES:
        p = root / pkg / "__init__.py"
        if not p.exists():
            p.write_text(f'"""{pkg.split("/")[-1]} package."""\n')
        log(f"{pkg}/__init__.py", C)

    # ── 3. Core files ──
    section("Writing core project files")
    for rel_path, content in FILES.items():
        p = root / rel_path
        p.parent.mkdir(parents=True, exist_ok=True)
        if not p.exists():
            p.write_text(content)
            log(rel_path)
        else:
            warn(f"Skipped (exists): {rel_path}")

    # ── 4. Stub files ──
    section("Writing stub source files")
    for rel_path, content in STUBS.items():
        p = root / rel_path
        p.parent.mkdir(parents=True, exist_ok=True)
        if not p.exists():
            if content is None:
                # Notebook
                title = NOTEBOOK_TITLES.get(rel_path, rel_path)
                p.write_text(MINIMAL_NOTEBOOK.replace("{title}", title))
            else:
                p.write_text(content)
            log(rel_path, DIM + G)

    # ── Summary ──
    py_files  = len(list(root.rglob("*.py")))
    nb_files  = len(list(root.rglob("*.ipynb")))
    dirs      = len([d for d in root.rglob("*") if d.is_dir()])
    all_files = len([f for f in root.rglob("*") if f.is_file()])

    print(f"""
{BOLD}{G}
╔══════════════════════════════════════════════════════════════╗
║  ✅  Scaffold complete!                                       ║
╠══════════════════════════════════════════════════════════════╣
║  📁  Directories  : {dirs:<39}║
║  🐍  Python files : {py_files:<39}║
║  📓  Notebooks    : {nb_files:<39}║
║  📄  Total files  : {all_files:<39}║
╚══════════════════════════════════════════════════════════════╝{RST}

{BOLD}Next steps:{RST}

  {Y}1.{RST}  cp .env.example .env  {DIM}# Add your API keys{RST}
  {Y}2.{RST}  conda env create -f environment.yml && conda activate crypto-forecast
      {DIM}# OR: pip install -r requirements.txt{RST}
  {Y}3.{RST}  make test  {DIM}# Verify everything imports correctly{RST}
  {Y}4.{RST}  make data  {DIM}# Start data collection pipeline{RST}
  {Y}5.{RST}  make train MODEL=lstm ASSET=BTC HORIZON=1d

  {C}Happy hacking, ImaJin! 🚀{RST}
""")


# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Scaffold the crypto-forecast-thesis project directory"
    )
    parser.add_argument(
        "--root", default=".",
        help="Root directory to scaffold into (default: current directory)"
    )
    args = parser.parse_args()
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    scaffold(root)
