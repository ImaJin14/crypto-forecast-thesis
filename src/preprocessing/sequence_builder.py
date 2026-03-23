"""
╔══════════════════════════════════════════════════════════════════════╗
║   src/preprocessing/sequence_builder.py                             ║
║   Sliding window sequence construction for deep learning            ║
║   Author : Muluh Penn Junior Patrick                                ║
╚══════════════════════════════════════════════════════════════════════╝
Converts scaled feature DataFrames into (X, y) tensor pairs for
time-series deep learning models.

Supports:
  - Single-step forecasting  : predict t+1
  - Multi-step forecasting   : predict t+1 ... t+h (sequence-to-sequence)
  - Multi-target forecasting : predict multiple assets simultaneously
  - Walk-forward CV splits   : time-series safe cross-validation

Usage:
    from src.preprocessing.sequence_builder import SequenceBuilder

    builder = SequenceBuilder(seq_len=60, horizon=1, target_col="close")
    X, y    = builder.build(scaled_df)
    dataset = builder.to_torch_dataset(X, y)
    loader  = builder.to_dataloader(X, y, batch_size=32)
"""

import os
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
from loguru import logger

try:
    import torch
    from torch.utils.data import Dataset, DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available — to_torch_dataset() will be disabled")


# ─── CryptoDataset ────────────────────────────────────────────────────────────

if TORCH_AVAILABLE:
    class CryptoDataset(Dataset):
        """
        PyTorch Dataset for cryptocurrency sequence data.

        Args:
            X : Feature sequences  (n_samples, seq_len, n_features)
            y : Target values      (n_samples, horizon) or (n_samples,)
        """

        def __init__(self, X: np.ndarray, y: np.ndarray):
            self.X = torch.FloatTensor(X)
            self.y = torch.FloatTensor(y)

        def __len__(self):
            return len(self.X)

        def __getitem__(self, idx):
            return self.X[idx], self.y[idx]

        @property
        def n_features(self) -> int:
            return self.X.shape[2]

        @property
        def seq_len(self) -> int:
            return self.X.shape[1]


# ─── SequenceBuilder ─────────────────────────────────────────────────────────

class SequenceBuilder:
    """
    Constructs supervised learning sequences from time-series DataFrames.

    Args:
        seq_len    : Input sequence length (lookback window)
        horizon    : Forecast horizon (steps ahead to predict)
        target_col : Column to predict (default: 'close')
        stride     : Step size between sequences (default: 1)
        target_transform : 'raw' | 'returns' | 'log_returns'
                           What form to predict (default: 'raw')
    """

    def __init__(
        self,
        seq_len:          int = 60,
        horizon:          int = 1,
        target_col:       str = "close",
        stride:           int = 1,
        target_transform: str = "raw",
    ):
        if target_transform not in ["raw", "returns", "log_returns"]:
            raise ValueError("target_transform must be 'raw', 'returns', or 'log_returns'")

        self.seq_len          = seq_len
        self.horizon          = horizon
        self.target_col       = target_col
        self.stride           = stride
        self.target_transform = target_transform

    # ── Core Builder ─────────────────────────────────────────────────────────

    def build(
        self,
        df:           pd.DataFrame,
        feature_cols: Optional[list] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Build (X, y) arrays from a scaled feature DataFrame.

        Args:
            df           : Scaled feature DataFrame (chronological order)
            feature_cols : Columns to use as input features.
                           Default: all numeric columns.

        Returns:
            X : np.ndarray of shape (n_samples, seq_len, n_features)
            y : np.ndarray of shape (n_samples, horizon) or (n_samples,) if horizon=1
        """
        self._validate(df)

        # Select feature columns
        if feature_cols is None:
            feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        # Remove non-numeric or object columns silently
        feature_cols = [c for c in feature_cols if c in df.columns
                        and np.issubdtype(df[c].dtype, np.number)]

        if self.target_col not in df.columns:
            raise ValueError(f"Target column '{self.target_col}' not in DataFrame")

        values     = df[feature_cols].values.astype(np.float32)
        target_raw = df[self.target_col].values.astype(np.float32)
        target     = self._apply_target_transform(target_raw)

        n_samples = (len(df) - self.seq_len - self.horizon + 1)
        if n_samples <= 0:
            raise ValueError(
                f"Not enough data to build sequences. "
                f"Need at least {self.seq_len + self.horizon} rows, got {len(df)}"
            )

        # Pre-allocate arrays
        if self.horizon == 1:
            y_shape = (n_samples,)
        else:
            y_shape = (n_samples, self.horizon)

        n_features = len(feature_cols)
        n_actual   = len(range(0, n_samples, self.stride))

        X = np.zeros((n_actual, self.seq_len, n_features), dtype=np.float32)
        y = np.zeros(y_shape[0:1] if self.horizon == 1 else (n_actual, self.horizon),
                     dtype=np.float32)

        # Trim to actual stride count
        X = np.zeros((n_actual, self.seq_len, n_features), dtype=np.float32)
        if self.horizon == 1:
            y = np.zeros(n_actual, dtype=np.float32)
        else:
            y = np.zeros((n_actual, self.horizon), dtype=np.float32)

        for i, start in enumerate(range(0, n_samples, self.stride)):
            end       = start + self.seq_len
            X[i]      = values[start:end]
            if self.horizon == 1:
                y[i]  = target[end]
            else:
                y[i]  = target[end : end + self.horizon]

        logger.info(f"  Sequences built: X={X.shape}, y={y.shape} | "
                    f"seq_len={self.seq_len}, horizon={self.horizon}, "
                    f"stride={self.stride}, features={n_features}")
        return X, y

    # ── Walk-Forward CV ───────────────────────────────────────────────────────

    def walk_forward_splits(
        self,
        df:          pd.DataFrame,
        n_splits:    int   = 5,
        train_ratio: float = 0.70,
        val_ratio:   float = 0.15,
        feature_cols: Optional[list] = None,
    ) -> list[dict]:
        """
        Generate walk-forward cross-validation splits.

        Each fold expands the training window and slides the
        validation/test windows forward in time.

        Args:
            df          : Full feature DataFrame
            n_splits    : Number of CV folds
            train_ratio : Initial training set fraction
            val_ratio   : Validation set fraction per fold
            feature_cols: Input feature columns

        Returns:
            List of dicts with keys:
                'fold', 'X_train', 'y_train',
                'X_val', 'y_val', 'train_end', 'val_end'
        """
        splits  = []
        n       = len(df)
        fold_size = int(n * (1 - train_ratio - val_ratio) / n_splits)

        for fold in range(n_splits):
            train_end = int(n * train_ratio) + fold * fold_size
            val_end   = train_end + int(n * val_ratio)
            val_end   = min(val_end, n)

            if train_end >= val_end or val_end - train_end < self.seq_len + self.horizon:
                logger.warning(f"  Fold {fold+1}: insufficient data, skipping")
                continue

            train_df = df.iloc[:train_end]
            val_df   = df.iloc[train_end:val_end]

            try:
                X_train, y_train = self.build(train_df, feature_cols)
                X_val,   y_val   = self.build(val_df,   feature_cols)
            except ValueError as e:
                logger.warning(f"  Fold {fold+1}: {e}, skipping")
                continue

            splits.append({
                "fold":      fold + 1,
                "X_train":   X_train,
                "y_train":   y_train,
                "X_val":     X_val,
                "y_val":     y_val,
                "train_end": df.index[train_end - 1] if hasattr(df.index, '__getitem__') else train_end,
                "val_end":   df.index[val_end - 1]   if hasattr(df.index, '__getitem__') else val_end,
                "n_train":   len(X_train),
                "n_val":     len(X_val),
            })
            logger.debug(f"  Fold {fold+1}: train={len(X_train):,}, val={len(X_val):,}")

        logger.info(f"  Walk-forward CV: {len(splits)} folds generated")
        return splits

    # ── PyTorch Integration ───────────────────────────────────────────────────

    def to_torch_dataset(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ):
        """Convert (X, y) arrays to a PyTorch CryptoDataset."""
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required. Install with: pip install torch")
        return CryptoDataset(X, y)

    def to_dataloader(
        self,
        X:           np.ndarray,
        y:           np.ndarray,
        batch_size:  int  = 32,
        shuffle:     bool = False,
        num_workers: int  = 0,
        pin_memory:  bool = False,
    ) -> "DataLoader":
        """
        Convert (X, y) arrays to a PyTorch DataLoader.

        NOTE: shuffle=False for time-series (preserve temporal order).

        Args:
            X          : Feature sequences
            y          : Target values
            batch_size : Batch size (default: 32)
            shuffle    : Shuffle samples (ALWAYS False for time-series)
            num_workers: Parallel data loading workers
            pin_memory : Pin memory for GPU transfer

        Returns:
            torch.utils.data.DataLoader
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required")

        if shuffle:
            logger.warning("  shuffle=True detected for time-series data — "
                           "this breaks temporal order! Setting shuffle=False.")
            shuffle = False

        dataset = self.to_torch_dataset(X, y)
        return DataLoader(
            dataset,
            batch_size  = batch_size,
            shuffle     = shuffle,
            num_workers = num_workers,
            pin_memory  = pin_memory,
        )

    # ── Save / Load sequences ─────────────────────────────────────────────────

    def save(
        self,
        X:    np.ndarray,
        y:    np.ndarray,
        path: Union[str, Path],
        name: str = "sequences",
    ) -> Path:
        """Save (X, y) arrays to disk as compressed npz."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        out  = path / f"{name}.npz"
        np.savez_compressed(out, X=X, y=y)
        size_kb = out.stat().st_size / 1024
        logger.info(f"  Sequences saved → {out} ({size_kb:.1f} KB)")
        return out

    @staticmethod
    def load(path: Union[str, Path]) -> tuple[np.ndarray, np.ndarray]:
        """Load (X, y) arrays from a saved npz file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Sequences not found: {path}")
        data = np.load(path)
        logger.info(f"  Sequences loaded ← {path}")
        return data["X"], data["y"]

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def config(self) -> dict:
        return {
            "seq_len":          self.seq_len,
            "horizon":          self.horizon,
            "target_col":       self.target_col,
            "stride":           self.stride,
            "target_transform": self.target_transform,
        }

    # ── Utilities ─────────────────────────────────────────────────────────────

    def _apply_target_transform(self, prices: np.ndarray) -> np.ndarray:
        """Apply target variable transformation."""
        if self.target_transform == "raw":
            return prices
        elif self.target_transform == "returns":
            ret      = np.zeros_like(prices)
            ret[1:]  = (prices[1:] - prices[:-1]) / (prices[:-1] + 1e-10)
            return ret
        elif self.target_transform == "log_returns":
            lr       = np.zeros_like(prices)
            lr[1:]   = np.log(prices[1:] / (prices[:-1] + 1e-10))
            return lr
        return prices

    def _validate(self, df: pd.DataFrame):
        if df.empty:
            raise ValueError("Input DataFrame is empty")
        min_rows = self.seq_len + self.horizon
        if len(df) < min_rows:
            raise ValueError(
                f"Need at least {min_rows} rows "
                f"(seq_len={self.seq_len} + horizon={self.horizon}), "
                f"got {len(df)}"
            )


# ─── Convenience functions ────────────────────────────────────────────────────

def build_sequences(
    df:          pd.DataFrame,
    seq_len:     int = 60,
    horizon:     int = 1,
    target_col:  str = "close",
    feature_cols: Optional[list] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Quick sequence building with default settings."""
    return SequenceBuilder(
        seq_len=seq_len, horizon=horizon, target_col=target_col
    ).build(df, feature_cols)


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    import sys
    sys.path.insert(0, ".")
    from src.preprocessing.technical_indicators import TechnicalIndicators
    from src.preprocessing.normalizer import Normalizer

    parser = argparse.ArgumentParser(description="Build training sequences")
    parser.add_argument("--asset",    default="BTC")
    parser.add_argument("--interval", default="1d", choices=["1h","1d"])
    parser.add_argument("--seq_len",  type=int, default=60)
    parser.add_argument("--horizon",  type=int, default=1)
    parser.add_argument("--save",     action="store_true")
    args = parser.parse_args()

    path = Path(f"data/raw/ohlcv/{args.asset}/{args.asset}_{args.interval}.parquet")
    if not path.exists():
        print(f"Data not found: {path}")
        exit(1)

    # Full pipeline test
    df     = pd.read_parquet(path)
    df     = TechnicalIndicators().compute(df)
    norm   = Normalizer(method="minmax")
    df_s, _, _ = norm.split_and_scale(df)
    builder = SequenceBuilder(seq_len=args.seq_len, horizon=args.horizon)
    X, y   = builder.build(df_s)

    print(f"\nPipeline test: {args.asset} {args.interval}")
    print(f"  Input rows  : {len(df):,}")
    print(f"  X shape     : {X.shape}   (samples, seq_len, features)")
    print(f"  y shape     : {y.shape}   (samples, horizon)")
    print(f"  X dtype     : {X.dtype}")
    print(f"  y range     : [{y.min():.4f}, {y.max():.4f}]")

    if args.save:
        out = builder.save(X, y,
                           path=f"data/processed/sequences/{args.asset}",
                           name=f"{args.asset}_{args.interval}_seq{args.seq_len}_h{args.horizon}")
        print(f"\n  Saved → {out}")
