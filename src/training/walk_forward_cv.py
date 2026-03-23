"""
╔══════════════════════════════════════════════════════════════════════╗
║   src/training/walk_forward_cv.py                                   ║
║   Walk-forward cross-validation for time-series models              ║
║   Author : Muluh Penn Junior Patrick                                ║
╚══════════════════════════════════════════════════════════════════════╝
Implements time-series safe cross-validation using expanding and
sliding window approaches.

Standard k-fold CV is INVALID for time-series because it allows
future data to leak into training. Walk-forward CV ensures the
training window always precedes the validation window.

Modes:
  - Expanding window : Training grows with each fold (default)
  - Sliding window   : Fixed-size training window slides forward

Usage:
    from src.training.walk_forward_cv import WalkForwardCV

    cv = WalkForwardCV(n_splits=5, mode="expanding")
    for fold in cv.split(df):
        train_df = fold["train"]
        val_df   = fold["val"]
        ...
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Iterator

import numpy as np
import pandas as pd
from loguru import logger


# ─── Fold dataclass ───────────────────────────────────────────────────────────

@dataclass
class Fold:
    """Container for a single walk-forward CV fold."""
    fold_num:   int
    train:      pd.DataFrame
    val:        pd.DataFrame
    train_start: pd.Timestamp
    train_end:   pd.Timestamp
    val_start:   pd.Timestamp
    val_end:     pd.Timestamp
    n_train:    int = field(init=False)
    n_val:      int = field(init=False)

    def __post_init__(self):
        self.n_train = len(self.train)
        self.n_val   = len(self.val)

    def __repr__(self):
        return (f"Fold({self.fold_num} | "
                f"train={self.n_train:,} [{self.train_start.date()} → {self.train_end.date()}] | "
                f"val={self.n_val:,} [{self.val_start.date()} → {self.val_end.date()}])")


# ─── WalkForwardCV ────────────────────────────────────────────────────────────

class WalkForwardCV:
    """
    Walk-forward cross-validation for time series.

    Generates sequential train/val splits where training data always
    precedes validation data — the only valid CV scheme for time series.

    Args:
        n_splits      : Number of CV folds (default: 5)
        mode          : 'expanding' (growing train window) or
                        'sliding'   (fixed-size train window)
        train_ratio   : Initial training set fraction (default: 0.60)
        val_ratio     : Validation set fraction per fold (default: 0.15)
        min_train_size: Minimum rows required for training
        gap           : Gap between train end and val start (prevents leakage
                        in high-frequency data where adjacent candles overlap)

    Example (expanding, 5 folds, 1000 rows):
        Fold 1: train[0:600],   val[600:750]
        Fold 2: train[0:700],   val[700:850]
        Fold 3: train[0:800],   val[800:950]
        ...

    Example (sliding, 5 folds, window=600):
        Fold 1: train[0:600],   val[600:750]
        Fold 2: train[100:700], val[700:850]
        ...
    """

    def __init__(
        self,
        n_splits:       int   = 5,
        mode:           str   = "expanding",
        train_ratio:    float = 0.60,
        val_ratio:      float = 0.15,
        min_train_size: int   = 200,
        gap:            int   = 0,
    ):
        if mode not in ["expanding", "sliding"]:
            raise ValueError(f"mode must be 'expanding' or 'sliding', got '{mode}'")

        self.n_splits       = n_splits
        self.mode           = mode
        self.train_ratio    = train_ratio
        self.val_ratio      = val_ratio
        self.min_train_size = min_train_size
        self.gap            = gap

    # ── Main split method ─────────────────────────────────────────────────────

    def split(self, df: pd.DataFrame) -> Iterator[Fold]:
        """
        Generate walk-forward CV folds from a DataFrame.

        Args:
            df : Feature DataFrame with DatetimeIndex (chronological order)

        Yields:
            Fold objects with train/val DataFrames and metadata
        """
        n          = len(df)
        n_initial  = int(n * self.train_ratio)
        n_val_each = int(n * self.val_ratio)

        if n_initial < self.min_train_size:
            raise ValueError(
                f"Initial training size {n_initial} < min_train_size "
                f"{self.min_train_size}. Reduce train_ratio or n_splits."
            )

        # Step size between folds
        remaining = n - n_initial
        step      = max(1, remaining // self.n_splits)

        generated = 0
        for i in range(self.n_splits):
            val_start_idx = n_initial + i * step
            val_end_idx   = min(val_start_idx + n_val_each, n)

            if val_end_idx > n:
                break
            if val_end_idx - val_start_idx < 10:
                logger.warning(f"  Fold {i+1}: validation set too small, skipping")
                continue

            # Training window
            if self.mode == "expanding":
                train_start_idx = 0
                train_end_idx   = val_start_idx - self.gap
            else:  # sliding
                train_end_idx   = val_start_idx - self.gap
                train_start_idx = max(0, train_end_idx - n_initial)

            if train_end_idx - train_start_idx < self.min_train_size:
                logger.warning(f"  Fold {i+1}: training set too small, skipping")
                continue

            train_df = df.iloc[train_start_idx:train_end_idx]
            val_df   = df.iloc[val_start_idx:val_end_idx]

            fold = Fold(
                fold_num    = i + 1,
                train       = train_df,
                val         = val_df,
                train_start = df.index[train_start_idx],
                train_end   = df.index[train_end_idx - 1],
                val_start   = df.index[val_start_idx],
                val_end     = df.index[val_end_idx - 1],
            )

            generated += 1
            logger.debug(f"  {fold}")
            yield fold

        logger.info(f"  Walk-forward CV: {generated} folds | mode={self.mode}")

    def get_splits_info(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preview all splits without running training.
        Returns a DataFrame summary of all fold boundaries.
        """
        rows = []
        for fold in self.split(df):
            rows.append({
                "fold":        fold.fold_num,
                "train_start": fold.train_start.date(),
                "train_end":   fold.train_end.date(),
                "val_start":   fold.val_start.date(),
                "val_end":     fold.val_end.date(),
                "n_train":     fold.n_train,
                "n_val":       fold.n_val,
            })
        return pd.DataFrame(rows).set_index("fold")

    def print_splits(self, df: pd.DataFrame):
        """Print a visual summary of all CV splits."""
        info = self.get_splits_info(df)
        print(f"\n{'─'*72}")
        print(f"  Walk-Forward CV | mode={self.mode} | {self.n_splits} folds")
        print(f"{'─'*72}")
        print(f"  {'FOLD':>4}  {'TRAIN START':<12} {'TRAIN END':<12} "
              f"{'VAL START':<12} {'VAL END':<12} {'N TRAIN':>8} {'N VAL':>6}")
        print(f"{'─'*72}")
        for fold_num, row in info.iterrows():
            print(f"  {fold_num:>4}  {str(row.train_start):<12} {str(row.train_end):<12} "
                  f"{str(row.val_start):<12} {str(row.val_end):<12} "
                  f"{row.n_train:>8,} {row.n_val:>6,}")
        print(f"{'─'*72}\n")


# ─── Time Series Train/Val/Test Split ────────────────────────────────────────

def temporal_split(
    df:          pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio:   float = 0.15,
    verbose:     bool  = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Simple chronological train/val/test split (no CV).

    Used for final model evaluation — train on 70%, validate on 15%,
    test on the final 15% (most recent data).

    Args:
        df          : Feature DataFrame with DatetimeIndex
        train_ratio : Training fraction (default: 0.70)
        val_ratio   : Validation fraction (default: 0.15)
        verbose     : Print split summary

    Returns:
        (train_df, val_df, test_df)
    """
    n       = len(df)
    n_train = int(n * train_ratio)
    n_val   = int(n * val_ratio)

    train = df.iloc[:n_train]
    val   = df.iloc[n_train : n_train + n_val]
    test  = df.iloc[n_train + n_val :]

    if verbose:
        logger.info(
            f"  Temporal split: "
            f"train={len(train):,} [{train.index[0].date()} → {train.index[-1].date()}] | "
            f"val={len(val):,} [{val.index[0].date()} → {val.index[-1].date()}] | "
            f"test={len(test):,} [{test.index[0].date()} → {test.index[-1].date()}]"
        )

    return train, val, test


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Preview walk-forward CV splits")
    parser.add_argument("--asset",    default="BTC")
    parser.add_argument("--interval", default="1d", choices=["1h","1d"])
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--mode",     default="expanding",
                        choices=["expanding","sliding"])
    args = parser.parse_args()

    path = Path(f"data/raw/ohlcv/{args.asset}/{args.asset}_{args.interval}.parquet")
    if not path.exists():
        print(f"Data not found: {path}")
        exit(1)

    df = pd.read_parquet(path)
    cv = WalkForwardCV(n_splits=args.n_splits, mode=args.mode)
    cv.print_splits(df)

    print("Simple temporal split:")
    train, val, test = temporal_split(df)
