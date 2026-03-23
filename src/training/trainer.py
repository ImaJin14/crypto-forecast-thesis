"""
╔══════════════════════════════════════════════════════════════════════╗
║   src/training/trainer.py                                           ║
║   PyTorch Lightning training module for all thesis models           ║
║   Author : Muluh Penn Junior Patrick                                ║
╚══════════════════════════════════════════════════════════════════════╝
Central training infrastructure:
  - CryptoForecasterModule : Lightning module wrapping any thesis model
  - CryptoDataModule       : Lightning data module for OHLCV sequences
  - train_model()          : High-level training entry point
  - evaluate_model()       : Post-training evaluation on test set

Usage:
    from src.training.trainer import train_model

    results = train_model(
        model_name = "lstm",
        asset      = "BTC",
        interval   = "1d",
        horizon    = 1,
        max_epochs = 200,
    )
"""

import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader, TensorDataset
from loguru import logger

# Project imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models                         import get_model, MODEL_REGISTRY
from src.training.loss_functions        import get_loss, CombinedLoss
from src.training.optimizer_config      import get_model_optimizer_config
from src.training.callbacks             import build_callbacks
from src.training.walk_forward_cv       import temporal_split
from src.preprocessing.technical_indicators import TechnicalIndicators
from src.preprocessing.ltst_decomposition  import LTSTDecomposer
from src.preprocessing.normalizer          import Normalizer
from src.preprocessing.sequence_builder    import SequenceBuilder
from src.evaluation.metrics             import compute_all_metrics
from src.evaluation.financial_metrics   import sharpe_ratio, win_rate, max_drawdown
from src.utils.seed                     import set_seed
from src.utils.device                   import get_device

DATA_DIR       = Path(os.getenv("DATA_DIR",        "./data"))
CHECKPOINT_DIR = Path(os.getenv("CHECKPOINT_DIR",  "./experiments/checkpoints"))
RESULTS_DIR    = Path(os.getenv("RESULTS_DIR",     "./experiments/results"))


# ══════════════════════════════════════════════════════════════════════════════
#  Lightning Module
# ══════════════════════════════════════════════════════════════════════════════

class CryptoForecasterModule(pl.LightningModule):
    """
    PyTorch Lightning module wrapping any thesis forecasting model.

    Handles:
    - Forward pass
    - Training / validation / test steps
    - Loss computation with component logging
    - Optimizer + scheduler configuration
    - Prediction collection for evaluation

    Args:
        model        : Instantiated PyTorch model (LSTM, GRU, etc.)
        loss_name    : Loss function name (default: 'combined')
        loss_kwargs  : Additional loss constructor arguments
        optimizer_name: Optimizer name (default: 'adam')
        scheduler_name: LR scheduler name (default: 'cosine')
        lr           : Learning rate (default: 1e-3)
        weight_decay : L2 regularisation (default: 1e-4)
        max_epochs   : Total training epochs (for scheduler)
        model_name   : Model name string (for logging)
    """

    def __init__(
        self,
        model:          nn.Module,
        loss_name:      str   = "combined",
        loss_kwargs:    dict  = None,
        optimizer_name: str   = "adam",
        scheduler_name: str   = "cosine",
        lr:             float = 1e-3,
        weight_decay:   float = 1e-4,
        max_epochs:     int   = 200,
        model_name:     str   = "model",
    ):
        super().__init__()
        self.model          = model
        self.loss_fn        = get_loss(loss_name, **(loss_kwargs or {}))
        self.optimizer_name = optimizer_name
        self.scheduler_name = scheduler_name
        self.lr             = lr
        self.weight_decay   = weight_decay
        self.max_epochs     = max_epochs
        self.model_name     = model_name

        # Collect predictions for evaluation
        self._test_preds:   list = []
        self._test_targets: list = []

        self.save_hyperparameters(ignore=["model", "loss_fn"])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    # ── Training step ─────────────────────────────────────────────────────────

    def training_step(self, batch, batch_idx):
        x, y    = batch
        y_hat   = self(x).squeeze()
        y       = y.squeeze()
        loss    = self.loss_fn(y_hat, y)

        self.log("train_loss", loss,
                 on_step=False, on_epoch=True, prog_bar=True)

        # Log individual loss components if using CombinedLoss
        if isinstance(self.loss_fn, CombinedLoss):
            components = self.loss_fn.components(y_hat, y)
            for k, v in components.items():
                self.log(k, v, on_step=False, on_epoch=True)

        return loss

    # ── Validation step ───────────────────────────────────────────────────────

    def validation_step(self, batch, batch_idx):
        x, y  = batch
        y_hat = self(x).squeeze()
        y     = y.squeeze()
        loss  = self.loss_fn(y_hat, y)

        self.log("val_loss", loss,
                 on_step=False, on_epoch=True, prog_bar=True)
        return loss

    # ── Test step ─────────────────────────────────────────────────────────────

    def test_step(self, batch, batch_idx):
        x, y  = batch
        y_hat = self(x).squeeze()
        y     = y.squeeze()
        loss  = self.loss_fn(y_hat, y)

        self.log("test_loss", loss)

        # Collect for full evaluation
        self._test_preds.append(y_hat.detach().cpu().numpy())
        self._test_targets.append(y.detach().cpu().numpy())
        return loss

    def on_test_epoch_end(self):
        self._test_preds   = []
        self._test_targets = []

    def get_test_predictions(self) -> tuple[np.ndarray, np.ndarray]:
        """Return collected test predictions and targets."""
        preds   = np.concatenate(self._test_preds)
        targets = np.concatenate(self._test_targets)
        return preds, targets

    # ── Optimizer ─────────────────────────────────────────────────────────────

    def configure_optimizers(self):
        return get_model_optimizer_config(
            model_name     = self.model_name,
            model          = self.model,
            max_epochs     = self.max_epochs,
            optimizer_name = self.optimizer_name,
            scheduler_name = self.scheduler_name,
            lr             = self.lr,
            weight_decay   = self.weight_decay,
        )


# ══════════════════════════════════════════════════════════════════════════════
#  Lightning Data Module
# ══════════════════════════════════════════════════════════════════════════════

class CryptoDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for cryptocurrency sequence data.

    Handles the full preprocessing pipeline:
    1. Load raw OHLCV parquet
    2. Compute technical indicators
    3. Add LTST decomposition features
    4. Temporal train/val/test split
    5. Normalize (fit on train only)
    6. Build sliding window sequences
    7. Create DataLoaders

    Args:
        asset         : Asset ticker (BTC, ETH, SOL, SUI, XRP)
        interval      : Candle interval (1h, 1d)
        seq_len       : Input sequence length (default: 60)
        horizon       : Forecast horizon in steps (default: 1)
        batch_size    : Training batch size (default: 32)
        target_col    : Column to predict (default: 'close')
        train_ratio   : Training fraction (default: 0.70)
        val_ratio     : Validation fraction (default: 0.15)
        num_workers   : DataLoader workers (default: 0)
        use_ltst      : Include LTST features (default: True)
        normalizer_method: Scaling method (default: 'minmax')
    """

    def __init__(
        self,
        asset:             str   = "BTC",
        interval:          str   = "1d",
        seq_len:           int   = 60,
        horizon:           int   = 1,
        batch_size:        int   = 32,
        target_col:        str   = "close",
        train_ratio:       float = 0.70,
        val_ratio:         float = 0.15,
        num_workers:       int   = 0,
        use_ltst:          bool  = True,
        normalizer_method: str   = "minmax",
    ):
        super().__init__()
        self.asset             = asset
        self.interval          = interval
        self.seq_len           = seq_len
        self.horizon           = horizon
        self.batch_size        = batch_size
        self.target_col        = target_col
        self.train_ratio       = train_ratio
        self.val_ratio         = val_ratio
        self.num_workers       = num_workers
        self.use_ltst          = use_ltst
        self.normalizer_method = normalizer_method

        self._n_features: int = 0
        self._normalizer: Optional[Normalizer] = None
        self._setup_done: bool = False   # cache flag

    # ── Setup ─────────────────────────────────────────────────────────────────

    def setup(self, stage: Optional[str] = None):
        """Load data, preprocess, and build all split datasets. Cached after first call."""
        if self._setup_done:
            return   # already preprocessed — skip recomputation
        # Load OHLCV
        ohlcv_path = DATA_DIR / "raw" / "ohlcv" / self.asset / \
                     f"{self.asset}_{self.interval}.parquet"
        if not ohlcv_path.exists():
            raise FileNotFoundError(
                f"OHLCV data not found: {ohlcv_path}\n"
                "Run: python -m src.data_collection.pipeline"
            )

        df = pd.read_parquet(ohlcv_path)
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")

        logger.info(f"  DataModule setup: {self.asset} {self.interval} "
                    f"| {len(df):,} rows")

        # Feature engineering
        logger.info("  Computing technical indicators...")
        ti = TechnicalIndicators(fillna=True)
        df = ti.compute(df)

        if self.use_ltst:
            logger.info("  Computing LTST decomposition...")
            decomp = LTSTDecomposer(fillna=True)
            df     = decomp.decompose(df, method="ma")  # fast MA method only

        # Drop non-numeric / object columns
        df = df.select_dtypes(include=[np.number])
        df = df.ffill().bfill().dropna()

        # ── Store raw close prices for inverse-transform metrics ──────────────
        # Keep unscaled close price aligned with the full index
        self._raw_close = df["close"].copy()

        # Temporal split
        train_df, val_df, test_df = temporal_split(
            df,
            train_ratio = self.train_ratio,
            val_ratio   = self.val_ratio,
            verbose     = True,
        )
        # Store test close prices for reconstructing USD metrics
        self._test_close_raw = test_df["close"].values.copy()

        # Normalize — fit on train only
        # Exclude:
        #   1. Binary/categorical features (already in [0,1] or {0,1})
        #   2. Return/log_return columns — MUST stay signed for directional accuracy
        #      MinMax scaling would shift them to [0,1] making all values positive
        exclude = [c for c in df.columns if
                   any(c.startswith(p) or c == p for p in
                       ["is_","above_","rsi_over","rsi_extreme",
                        "bb_above","bb_below","golden_","death_",
                        "stoch_cross","macd_cross","fg_is_",
                        "vol_surge","vol_dry","high_vol","low_vol",
                        "psar_bullish","ichimoku_bullish",
                        "hp_expansion","hp_contraction",
                        "trending_regime","mean_reverting_",
                        # Keep returns signed — critical for DA
                        "returns","log_returns",
                        "roc_1","roc_7","roc_14","roc_30",
                        "price_accel","fg_pct_change","fg_momentum",
                        ])]

        self._normalizer = Normalizer(
            method           = self.normalizer_method,
            exclude_features = exclude,
            clip_outliers    = True,
        )
        train_scaled = self._normalizer.fit_transform(train_df)
        val_scaled   = self._normalizer.transform(val_df)
        test_scaled  = self._normalizer.transform(test_df)

        # ── Build sequences — predict log_returns, not price levels ──────────
        # log_returns are stationary and scale-invariant — no extrapolation issue
        # After prediction, reconstruct price: P_t+1 = P_t * exp(log_return)
        builder = SequenceBuilder(
            seq_len          = self.seq_len,
            horizon          = self.horizon,
            target_col       = "log_returns",   # ← KEY CHANGE: predict returns not price
            target_transform = "raw",            # log_returns column already computed
        )

        X_train, y_train = builder.build(train_scaled)
        X_val,   y_val   = builder.build(val_scaled)
        X_test,  y_test  = builder.build(test_scaled)

        self._n_features = X_train.shape[2]

        # Save scaler and raw test prices for evaluation
        scaler_dir = DATA_DIR / "processed" / "scalers"
        scaler_dir.mkdir(parents=True, exist_ok=True)
        self._normalizer.save(
            scaler_dir / f"{self.asset}_{self.interval}_scaler.pkl"
        )
        # Save raw test close prices for price reconstruction during eval
        np.save(
            scaler_dir / f"{self.asset}_{self.interval}_test_close.npy",
            self._test_close_raw
        )

        # Create TensorDatasets
        self._train_ds = TensorDataset(
            torch.FloatTensor(X_train), torch.FloatTensor(y_train)
        )
        self._val_ds = TensorDataset(
            torch.FloatTensor(X_val), torch.FloatTensor(y_val)
        )
        self._test_ds = TensorDataset(
            torch.FloatTensor(X_test), torch.FloatTensor(y_test)
        )

        logger.success(
            f"  DataModule ready: "
            f"train={len(self._train_ds):,} | "
            f"val={len(self._val_ds):,} | "
            f"test={len(self._test_ds):,} | "
            f"features={self._n_features} | target=log_returns"
        )
        self._setup_done = True   # cache complete

    def train_dataloader(self):
        return DataLoader(self._train_ds, batch_size=self.batch_size,
                          shuffle=False, num_workers=self.num_workers,
                          pin_memory=torch.cuda.is_available())

    def val_dataloader(self):
        return DataLoader(self._val_ds, batch_size=self.batch_size,
                          shuffle=False, num_workers=self.num_workers,
                          pin_memory=torch.cuda.is_available())

    def test_dataloader(self):
        return DataLoader(self._test_ds, batch_size=self.batch_size,
                          shuffle=False, num_workers=self.num_workers,
                          pin_memory=torch.cuda.is_available())

    @property
    def n_features(self) -> int:
        return self._n_features

    @property
    def normalizer(self) -> Optional[Normalizer]:
        return self._normalizer


# ══════════════════════════════════════════════════════════════════════════════
#  High-level training entry point
# ══════════════════════════════════════════════════════════════════════════════

def train_model(
    model_name:     str   = "lstm",
    asset:          str   = "BTC",
    interval:       str   = "1d",
    horizon:        int   = 1,
    seq_len:        int   = 60,
    batch_size:     int   = 32,
    max_epochs:     int   = 200,
    lr:             float = 1e-3,
    weight_decay:   float = 1e-4,
    loss_name:      str   = "combined",
    optimizer_name: str   = "adam",
    scheduler_name: str   = "cosine",
    use_ltst:       bool  = True,
    seed:           int   = 42,
    fast_dev_run:   bool  = False,
    model_kwargs:   dict  = None,
) -> dict:
    """
    Full training pipeline for a single model-asset-horizon experiment.

    Args:
        model_name     : Architecture name (lstm, gru, bilstm, cnn_lstm,
                         attention_lstm, transformer)
        asset          : Crypto asset (BTC, ETH, SOL, SUI, XRP)
        interval       : Candle interval (1h, 1d)
        horizon        : Forecast horizon in steps ahead
        seq_len        : Input sequence length
        batch_size     : Training batch size
        max_epochs     : Maximum training epochs
        lr             : Initial learning rate
        weight_decay   : L2 regularisation strength
        loss_name      : Loss function name
        optimizer_name : Optimizer name
        scheduler_name : LR scheduler name
        use_ltst       : Include LTST features
        seed           : Random seed for reproducibility
        fast_dev_run   : Quick 1-batch test run (for debugging)
        model_kwargs   : Extra model constructor arguments

    Returns:
        dict with keys:
            model, module, trainer, data_module,
            test_metrics, checkpoint_path, run_name
    """
    set_seed(seed)
    # Enable Tensor Core optimisation on RTX GPUs
    torch.set_float32_matmul_precision("medium")
    device  = get_device()
    run_name = f"{model_name}_{asset}_{interval}_h{horizon}"

    logger.info(f"\n{'═'*60}")
    logger.info(f"  Training: {run_name}")
    logger.info(f"  Device  : {device}")
    logger.info(f"{'═'*60}")

    # ── 1. Data ───────────────────────────────────────────────────────────────
    data_module = CryptoDataModule(
        asset      = asset,
        interval   = interval,
        seq_len    = seq_len,
        horizon    = horizon,
        batch_size = batch_size,
        use_ltst   = use_ltst,
    )
    data_module.setup()

    # ── 2. Model ──────────────────────────────────────────────────────────────
    kwargs = {
        "input_size":  data_module.n_features,
        "output_size": horizon,
        **(model_kwargs or {}),
    }
    model = get_model(model_name, **kwargs)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"  Model     : {model.__class__.__name__} | "
                f"{n_params:,} parameters")

    # ── 3. Lightning module ───────────────────────────────────────────────────
    module = CryptoForecasterModule(
        model          = model,
        loss_name      = loss_name,
        optimizer_name = optimizer_name,
        scheduler_name = scheduler_name,
        lr             = lr,
        weight_decay   = weight_decay,
        max_epochs     = max_epochs,
        model_name     = model_name,
    )

    # ── 4. Callbacks ──────────────────────────────────────────────────────────
    ckpt_dir  = CHECKPOINT_DIR / run_name
    callbacks = build_callbacks(
        checkpoint_dir = str(ckpt_dir),
        metrics_dir    = str(RESULTS_DIR),
        model_name     = model_name,
        asset          = asset,
        interval       = interval,
    )

    # ── 5. Trainer ────────────────────────────────────────────────────────────
    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    trainer     = pl.Trainer(
        max_epochs          = max_epochs,
        accelerator         = accelerator,
        devices             = 1,
        callbacks           = callbacks,
        gradient_clip_val   = 1.0,       # clip gradients for RNN stability
        log_every_n_steps   = 10,
        enable_progress_bar = True,
        enable_model_summary= True,
        fast_dev_run        = fast_dev_run,
        deterministic       = False,     # True slows GPU, off for speed
    )

    # ── 6. Train ──────────────────────────────────────────────────────────────
    logger.info("  Starting training...")
    trainer.fit(module, datamodule=data_module)

    # ── 7. Test ───────────────────────────────────────────────────────────────
    logger.info("  Running test evaluation...")
    trainer.test(module, datamodule=data_module, verbose=False)

    # ── 8. Evaluate ───────────────────────────────────────────────────────────
    # Get best checkpoint path
    ckpt_callback = next(
        (c for c in callbacks if hasattr(c, "best_model_path")), None
    )
    best_ckpt = ckpt_callback.best_model_path if ckpt_callback else None

    # Compute KPI metrics
    test_metrics = _compute_test_metrics(
        module      = module,
        data_module = data_module,
        run_name    = run_name,
        n_params    = n_params,
    )

    logger.success(f"\n  ✅  {run_name} complete")
    logger.success(f"  RMSE: {test_metrics.get('rmse', 'N/A'):.4f} | "
                   f"MAPE: {test_metrics.get('mape', 'N/A'):.2f}% | "
                   f"DA: {test_metrics.get('directional_accuracy', 'N/A'):.1f}%")

    return {
        "run_name":       run_name,
        "model":          model,
        "module":         module,
        "trainer":        trainer,
        "data_module":    data_module,
        "test_metrics":   test_metrics,
        "checkpoint_path": best_ckpt,
        "n_params":       n_params,
    }


def _compute_test_metrics(
    module:      CryptoForecasterModule,
    data_module: CryptoDataModule,
    run_name:    str,
    n_params:    int,
) -> dict:
    """Run full KPI evaluation on test predictions."""
    try:
        # Re-run inference on test set
        device = next(module.parameters()).device
        module.eval()
        module._test_preds   = []
        module._test_targets = []

        with torch.no_grad():
            for batch in data_module.test_dataloader():
                x, y  = batch
                x     = x.to(device)
                y_hat = module(x).squeeze().cpu().numpy()
                y_np  = y.squeeze().cpu().numpy()
                module._test_preds.append(np.atleast_1d(y_hat))
                module._test_targets.append(np.atleast_1d(y_np))

        pred_log_returns   = np.concatenate(module._test_preds).flatten()
        target_log_returns = np.concatenate(module._test_targets).flatten()

        # ── Reconstruct next-step USD prices from log returns ─────────────────
        # The sequence builder produces:
        #   X[i] = scaled features [i : i+seq_len]
        #   y[i] = log_return at position i+seq_len
        # So the "current" price for sample i is raw_close[i + seq_len - 1]
        scaler_dir      = DATA_DIR / "processed" / "scalers"
        test_close_path = scaler_dir / f"{data_module.asset}_{data_module.interval}_test_close.npy"

        try:
            raw_close    = np.load(test_close_path)
            n            = len(pred_log_returns)
            seq_len      = data_module.seq_len

            # Current price at each prediction step = raw_close[seq_len-1 + i]
            # (the last candle in the input window)
            current_idx  = np.arange(n) + seq_len - 1
            current_idx  = np.clip(current_idx, 0, len(raw_close) - 1)
            start_prices = raw_close[current_idx]

            # Next-step price = current * exp(log_return)
            preds   = start_prices * np.exp(pred_log_returns)
            targets = start_prices * np.exp(target_log_returns)

            logger.info(f"  USD prices | "
                        f"pred: [{preds.min():.0f}, {preds.max():.0f}] | "
                        f"actual: [{targets.min():.0f}, {targets.max():.0f}]")

        except Exception as inv_err:
            logger.warning(f"  Price reconstruction failed ({inv_err}) "
                           f"— using log returns directly")
            preds   = pred_log_returns
            targets = target_log_returns

        # ── Core forecasting KPIs (on USD prices) ────────────────────────────
        metrics = compute_all_metrics(targets, preds)

        # ── Directional accuracy on raw log returns ───────────────────────────
        nonzero = np.abs(target_log_returns) > 1e-8
        if nonzero.sum() > 0:
            correct_dir = (
                np.sign(pred_log_returns[nonzero]) ==
                np.sign(target_log_returns[nonzero])
            )
            da = float(correct_dir.mean() * 100)
        else:
            da = 50.0
        metrics["directional_accuracy"] = da
        metrics["win_rate"]             = da

        # ── Financial KPIs — use STRATEGY returns ────────────────────────────
        # Strategy: go long if model predicts positive return, else go short/flat
        # Strategy return[i] = sign(pred_log_return[i]) * actual_log_return[i]
        # This is the standard financial ML evaluation approach
        pred_direction   = np.sign(pred_log_returns)
        strategy_returns = pred_direction * target_log_returns  # actual P&L from our calls
        strategy_clipped = np.clip(strategy_returns, -0.3, 0.3)

        sr_mean = float(strategy_clipped.mean())
        sr_std  = float(strategy_clipped.std())

        # Annualised Sharpe (crypto runs 365 days/year)
        metrics["sharpe_ratio"] = (
            float(np.sqrt(365) * sr_mean / sr_std)
            if sr_std > 1e-8 else 0.0
        )

        # Max drawdown of strategy equity curve
        try:
            equity_curve = np.cumprod(1 + strategy_clipped)
            if np.all(np.isfinite(equity_curve)) and len(equity_curve) > 1:
                metrics["max_drawdown"] = float(max_drawdown(equity_curve))
            else:
                metrics["max_drawdown"] = 0.0
        except Exception:
            metrics["max_drawdown"] = 0.0

        metrics["n_params"]  = n_params
        metrics["run_name"]  = run_name

        # Save results
        results_path = RESULTS_DIR / f"{run_name}_results.csv"
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([metrics]).to_csv(results_path, index=False)
        logger.info(f"  Results saved → {results_path}")

        return metrics

    except Exception as e:
        logger.error(f"  Metrics computation failed: {e}", exc_info=True)
        return {"error": str(e)}


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train a forecasting model")
    parser.add_argument("--model",      default="lstm",
                        choices=list(MODEL_REGISTRY.keys()))
    parser.add_argument("--asset",      default="BTC",
                        choices=["BTC","ETH","SOL","SUI","XRP"])
    parser.add_argument("--interval",   default="1d", choices=["1h","1d"])
    parser.add_argument("--horizon",    type=int, default=1)
    parser.add_argument("--seq_len",    type=int, default=60)
    parser.add_argument("--epochs",     type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--loss",       default="combined",
                        choices=["mse","huber","combined","directional","mape"])
    parser.add_argument("--no_ltst",    action="store_true")
    parser.add_argument("--fast",       action="store_true",
                        help="Fast dev run (1 batch) for debugging")
    args = parser.parse_args()

    results = train_model(
        model_name   = args.model,
        asset        = args.asset,
        interval     = args.interval,
        horizon      = args.horizon,
        seq_len      = args.seq_len,
        max_epochs   = args.epochs,
        batch_size   = args.batch_size,
        lr           = args.lr,
        loss_name    = args.loss,
        use_ltst     = not args.no_ltst,
        fast_dev_run = args.fast,
    )

    print(f"\n{'═'*50}")
    print(f"  Results: {results['run_name']}")
    print(f"{'═'*50}")
    for k, v in results["test_metrics"].items():
        if isinstance(v, float):
            print(f"  {k:<30} {v:.4f}")
        else:
            print(f"  {k:<30} {v}")
