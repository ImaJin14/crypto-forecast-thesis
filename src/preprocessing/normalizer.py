"""
╔══════════════════════════════════════════════════════════════════════╗
║   src/preprocessing/normalizer.py                                   ║
║   Feature normalization with strict train-only fitting              ║
║   Author : Muluh Penn Junior Patrick                                ║
╚══════════════════════════════════════════════════════════════════════╝
Handles all feature scaling for the thesis pipeline:
  - MinMax scaling [0, 1]    per feature
  - Standard scaling (Z)     per feature
  - Robust scaling (IQR)     per feature (outlier-resistant)
  - Log transform             for price/volume skewed features

CRITICAL: Scalers are ALWAYS fitted on training data only, then
applied to validation and test sets to prevent data leakage.

Usage:
    from src.preprocessing.normalizer import Normalizer

    norm = Normalizer(method="minmax")
    X_train = norm.fit_transform(X_train_df, split="train")
    X_val   = norm.transform(X_val_df)
    X_test  = norm.transform(X_test_df)
    norm.save("data/processed/scalers/BTC_1d_scaler.pkl")
"""

import os
import pickle
import warnings
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler

warnings.filterwarnings("ignore")

SCALER_DIR = Path(os.getenv("DATA_DIR", "./data")) / "processed" / "scalers"


# ─── Normalizer ───────────────────────────────────────────────────────────────

class Normalizer:
    """
    Feature normalization with train-only fitting to prevent data leakage.

    Args:
        method          : 'minmax', 'standard', 'robust', or 'mixed'
        feature_range   : Range for MinMax scaling (default: (0, 1))
        log_features    : List of column names to log-transform before scaling
        exclude_features: Columns to exclude from scaling (e.g. binary flags)
        clip_outliers   : Clip values outside [mean ± n_std * std] before scaling
        n_std           : Standard deviations for outlier clipping (default: 5)
    """

    METHODS = ["minmax", "standard", "robust", "mixed"]

    def __init__(
        self,
        method:           str   = "minmax",
        feature_range:    tuple = (0, 1),
        log_features:     Optional[list] = None,
        exclude_features: Optional[list] = None,
        clip_outliers:    bool  = True,
        n_std:            float = 5.0,
    ):
        if method not in self.METHODS:
            raise ValueError(f"method must be one of {self.METHODS}")

        self.method           = method
        self.feature_range    = feature_range
        self.log_features     = log_features or []
        self.exclude_features = exclude_features or []
        self.clip_outliers    = clip_outliers
        self.n_std            = n_std

        self._scalers: dict = {}       # {col: fitted_scaler}
        self._log_cols: list = []      # columns actually log-transformed
        self._fitted_cols: list = []   # columns that were scaled
        self._is_fitted: bool = False
        self._clip_bounds: dict = {}   # {col: (lower, upper)}

    # ── Fit & Transform ───────────────────────────────────────────────────────

    def fit(self, df: pd.DataFrame) -> "Normalizer":
        """
        Fit scalers on training data ONLY.

        Args:
            df : Training set DataFrame

        Returns:
            self (for chaining)
        """
        if df.empty:
            raise ValueError("Cannot fit on empty DataFrame")

        logger.info(f"  Fitting normalizer [{self.method}] on "
                    f"{len(df):,} rows × {len(df.columns)} features")

        # Determine columns to scale
        scale_cols = [
            c for c in df.select_dtypes(include=[np.number]).columns
            if c not in self.exclude_features
        ]

        self._fitted_cols = scale_cols
        self._log_cols    = [c for c in self.log_features if c in scale_cols]

        work = df[scale_cols].copy()

        # Log transform price/volume features (reduce skewness)
        for col in self._log_cols:
            min_val = work[col].min()
            shift   = max(0, -min_val + 1e-6)  # shift to make positive if needed
            work[col] = np.log1p(work[col] + shift)

        # Fit clip bounds (outlier detection on training data)
        if self.clip_outliers:
            for col in scale_cols:
                mean = work[col].mean()
                std  = work[col].std()
                self._clip_bounds[col] = (
                    mean - self.n_std * std,
                    mean + self.n_std * std,
                )
            work = self._apply_clip(work, scale_cols)

        # Fit scaler per column
        for col in scale_cols:
            scaler = self._make_scaler(col)
            vals   = work[col].values.reshape(-1, 1)
            scaler.fit(vals)
            self._scalers[col] = scaler

        self._is_fitted = True
        logger.success(f"  ✔  Normalizer fitted: {len(scale_cols)} features scaled")
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply fitted scalers to a DataFrame (val or test set).
        Non-numeric and excluded columns are returned unchanged.

        Args:
            df : DataFrame to transform

        Returns:
            Scaled DataFrame (same shape and columns)
        """
        if not self._is_fitted:
            raise RuntimeError("Normalizer must be fitted before transform. "
                               "Call fit() or fit_transform() first.")

        result = df.copy()
        cols   = [c for c in self._fitted_cols if c in result.columns]

        # Log transform
        for col in self._log_cols:
            if col in result.columns:
                min_val = result[col].min()
                shift   = max(0, -min_val + 1e-6)
                result[col] = np.log1p(result[col] + shift)

        # Clip outliers using training bounds
        if self.clip_outliers:
            result = self._apply_clip(result, cols)

        # Scale
        for col in cols:
            if col not in self._scalers:
                continue
            vals          = result[col].values.reshape(-1, 1)
            result[col]   = self._scalers[col].transform(vals).flatten()

        return result

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform in one step (use ONLY on training data)."""
        return self.fit(df).transform(df)

    def inverse_transform(
        self,
        values: Union[np.ndarray, pd.Series],
        col:    str,
    ) -> np.ndarray:
        """
        Inverse-transform scaled predictions back to original price scale.

        Args:
            values : Scaled values (1D array or Series)
            col    : Feature name (must have been fitted)

        Returns:
            Values in original scale
        """
        if not self._is_fitted:
            raise RuntimeError("Normalizer not fitted")
        if col not in self._scalers:
            raise KeyError(f"Column '{col}' was not scaled")

        vals    = np.array(values).reshape(-1, 1)
        inv     = self._scalers[col].inverse_transform(vals).flatten()

        # Reverse log transform
        if col in self._log_cols:
            inv = np.expm1(inv)

        return inv

    # ── Split-aware helpers ───────────────────────────────────────────────────

    def split_and_scale(
        self,
        df:         pd.DataFrame,
        train_ratio: float = 0.70,
        val_ratio:   float = 0.15,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split DataFrame into train/val/test and scale correctly:
        fit on train only, transform all three.

        Args:
            df          : Full feature DataFrame (chronological order)
            train_ratio : Fraction of data for training
            val_ratio   : Fraction of data for validation
            (test = 1 - train_ratio - val_ratio)

        Returns:
            (train_scaled, val_scaled, test_scaled)
        """
        n       = len(df)
        n_train = int(n * train_ratio)
        n_val   = int(n * val_ratio)

        train = df.iloc[:n_train]
        val   = df.iloc[n_train : n_train + n_val]
        test  = df.iloc[n_train + n_val :]

        logger.info(f"  Split: train={len(train):,} | val={len(val):,} | "
                    f"test={len(test):,}")

        train_scaled = self.fit_transform(train)
        val_scaled   = self.transform(val)
        test_scaled  = self.transform(test)

        logger.success(f"  ✔  Scaling complete — no data leakage")
        return train_scaled, val_scaled, test_scaled

    # ── Save / Load ───────────────────────────────────────────────────────────

    def save(self, path: Union[str, Path]) -> Path:
        """Save fitted normalizer to disk."""
        if not self._is_fitted:
            raise RuntimeError("Cannot save an unfitted normalizer")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

        size_kb = path.stat().st_size / 1024
        logger.info(f"  Normalizer saved → {path} ({size_kb:.1f} KB)")
        return path

    @classmethod
    def load(cls, path: Union[str, Path]) -> "Normalizer":
        """Load a previously saved normalizer."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Normalizer not found: {path}")
        with open(path, "rb") as f:
            obj = pickle.load(f)
        logger.info(f"  Normalizer loaded ← {path}")
        return obj

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    @property
    def n_features(self) -> int:
        return len(self._fitted_cols)

    @property
    def feature_names(self) -> list:
        return self._fitted_cols.copy()

    def get_scaler(self, col: str):
        """Return the fitted scaler for a specific column."""
        if col not in self._scalers:
            raise KeyError(f"No scaler found for column '{col}'")
        return self._scalers[col]

    # ── Utilities ─────────────────────────────────────────────────────────────

    def _make_scaler(self, col: str):
        """Return the appropriate scaler for a column."""
        if self.method == "minmax":
            return MinMaxScaler(feature_range=self.feature_range)
        elif self.method == "standard":
            return StandardScaler()
        elif self.method == "robust":
            return RobustScaler()
        elif self.method == "mixed":
            # Use robust scaler for price/volume (skewed), minmax for everything else
            if any(k in col for k in ["close", "open", "high", "low",
                                       "volume", "market_cap", "mcap"]):
                return RobustScaler()
            return MinMaxScaler(feature_range=self.feature_range)
        return MinMaxScaler(feature_range=self.feature_range)

    def _apply_clip(self, df: pd.DataFrame, cols: list) -> pd.DataFrame:
        """Clip outliers to training bounds."""
        result = df.copy()
        for col in cols:
            if col in self._clip_bounds and col in result.columns:
                lo, hi       = self._clip_bounds[col]
                result[col]  = result[col].clip(lower=lo, upper=hi)
        return result

    def summary(self) -> pd.DataFrame:
        """Return a summary of fitted scaler statistics."""
        rows = []
        for col, scaler in self._scalers.items():
            row = {"feature": col, "scaler": type(scaler).__name__}
            if hasattr(scaler, "data_min_"):
                row["train_min"] = scaler.data_min_[0]
                row["train_max"] = scaler.data_max_[0]
            elif hasattr(scaler, "mean_"):
                row["train_mean"] = scaler.mean_[0]
                row["train_std"]  = np.sqrt(scaler.var_[0])
            if col in self._clip_bounds:
                row["clip_lo"], row["clip_hi"] = self._clip_bounds[col]
            rows.append(row)
        return pd.DataFrame(rows).set_index("feature")


# ─── Convenience functions ────────────────────────────────────────────────────

def scale_features(
    df:           pd.DataFrame,
    method:       str   = "minmax",
    log_features: Optional[list] = None,
) -> tuple:
    """
    Quick scaling for a single DataFrame (fits and transforms in one call).
    Use only when you don't need separate train/val/test splits.

    Returns:
        (scaled_df, fitted_normalizer)
    """
    norm    = Normalizer(method=method, log_features=log_features)
    scaled  = norm.fit_transform(df)
    return scaled, norm


def load_or_fit_normalizer(
    path:         Union[str, Path],
    train_df:     Optional[pd.DataFrame] = None,
    method:       str = "minmax",
) -> "Normalizer":
    """
    Load existing normalizer from disk, or fit a new one if not found.

    Args:
        path     : Path to saved normalizer pickle
        train_df : Training data (required if file doesn't exist)
        method   : Scaling method if fitting from scratch

    Returns:
        Fitted Normalizer instance
    """
    path = Path(path)
    if path.exists():
        return Normalizer.load(path)

    if train_df is None:
        raise ValueError(f"Normalizer not found at {path} and no train_df provided")

    norm = Normalizer(method=method)
    norm.fit(train_df)
    norm.save(path)
    return norm


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    from pathlib import Path
    from src.preprocessing.technical_indicators import TechnicalIndicators

    parser = argparse.ArgumentParser(description="Test normalizer on OHLCV data")
    parser.add_argument("--asset",    default="BTC", choices=["BTC","ETH","SOL","SUI","XRP"])
    parser.add_argument("--interval", default="1d",  choices=["1h","1d"])
    parser.add_argument("--method",   default="minmax",
                        choices=["minmax","standard","robust","mixed"])
    args = parser.parse_args()

    path = Path(f"data/raw/ohlcv/{args.asset}/{args.asset}_{args.interval}.parquet")
    if not path.exists():
        print(f"Data not found: {path}")
        exit(1)

    df      = pd.read_parquet(path)
    df      = TechnicalIndicators().compute(df)

    # Exclude binary/categorical columns from scaling
    exclude = [c for c in df.columns if
               c.startswith(("is_", "above_", "rsi_over", "rsi_extreme",
                              "bb_above", "bb_below", "golden_", "death_",
                              "stoch_cross", "macd_cross", "fg_is_", "fg_regime"))]

    norm  = Normalizer(method=args.method, exclude_features=exclude)
    train_s, val_s, test_s = norm.split_and_scale(df)

    print(f"\nAsset    : {args.asset} {args.interval}")
    print(f"Method   : {args.method}")
    print(f"Features : {norm.n_features}")
    print(f"\nScaler summary (first 10 features):")
    print(norm.summary().head(10).to_string())
    print(f"\nTrain set (scaled, last 3 rows):")
    print(train_s[["close", "rsi_14", "macd"]].tail(3).to_string())
