"""
╔══════════════════════════════════════════════════════════════════════╗
║   src/preprocessing/ltst_decomposition.py                           ║
║   Long-Term / Short-Term trend decomposition                        ║
║   Author : Muluh Penn Junior Patrick                                ║
╚══════════════════════════════════════════════════════════════════════╝
Decomposes price time series into multi-scale trend components:
  - Long-Term Trend  (LTT) : Macro price direction (weeks–months)
  - Short-Term Trend (STT) : Micro price dynamics  (hours–days)
  - Residual               : Mean-reverting noise component

Methods:
  1. Moving Average decomposition (SMA/EMA based)
  2. Hodrick-Prescott filter     (statistical smoothing)
  3. STL decomposition           (Seasonal-Trend via LOESS)
  4. Wavelet decomposition       (PyWavelets — optional)

Usage:
    from src.preprocessing.ltst_decomposition import LTSTDecomposer

    decomposer = LTSTDecomposer()
    df = decomposer.decompose(ohlcv_df)           # all methods
    df = decomposer.decompose(ohlcv_df, method="hp")
"""

import warnings
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger

warnings.filterwarnings("ignore")

try:
    from statsmodels.tsa.filters.hp_filter import hpfilter
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    logger.warning("statsmodels not found — HP filter disabled. "
                   "Install with: pip install statsmodels")

try:
    from statsmodels.tsa.seasonal import STL
    STL_AVAILABLE = STATSMODELS_AVAILABLE
except ImportError:
    STL_AVAILABLE = False

try:
    import pywt
    PYWT_AVAILABLE = True
except ImportError:
    PYWT_AVAILABLE = False
    logger.warning("PyWavelets not found — wavelet decomposition disabled. "
                   "Install with: pip install PyWavelets")


# ─── LTSTDecomposer ───────────────────────────────────────────────────────────

class LTSTDecomposer:
    """
    Decomposes cryptocurrency price series into long-term and short-term
    trend components using multiple complementary methods.

    The decomposed components serve as distinct input features for the
    deep learning models, providing explicit multi-scale temporal context.

    Args:
        price_col : Column to decompose (default: 'close')
        fillna    : Forward-fill NaN values (default: True)
    """

    METHODS = ["ma", "hp", "stl", "wavelet"]

    def __init__(self, price_col: str = "close", fillna: bool = True):
        self.price_col = price_col
        self.fillna    = fillna

    # ── Public API ────────────────────────────────────────────────────────────

    def decompose(
        self,
        df:     pd.DataFrame,
        method: str = "all",
    ) -> pd.DataFrame:
        """
        Decompose price series and append LTST features to DataFrame.

        Args:
            df     : DataFrame with at least a 'close' column
            method : One of 'ma', 'hp', 'stl', 'wavelet', or 'all'

        Returns:
            DataFrame with original columns + LTST feature columns
        """
        if self.price_col not in df.columns:
            raise ValueError(f"Column '{self.price_col}' not found in DataFrame")

        price  = df[self.price_col].copy()
        result = df.copy()

        methods = self.METHODS if method == "all" else [method]
        logger.info(f"  LTST decomposition | methods: {methods} | "
                    f"{len(price):,} observations")

        for m in methods:
            if m == "ma":
                result = self._decompose_ma(result, price)
            elif m == "hp":
                result = self._decompose_hp(result, price)
            elif m == "stl":
                result = self._decompose_stl(result, price)
            elif m == "wavelet":
                result = self._decompose_wavelet(result, price)

        # Cross-method composite features
        result = self._add_composite_features(result, price)

        if self.fillna:
            result = result.ffill().bfill()

        n_new = len(result.columns) - len(df.columns)
        logger.info(f"  ✔  LTST: added {n_new} decomposition features")
        return result

    # ── Method 1: Moving Average Decomposition ────────────────────────────────

    def _decompose_ma(self, df: pd.DataFrame, price: pd.Series) -> pd.DataFrame:
        """
        MA-based decomposition using SMA and EMA at multiple scales.

        Long-term  trend: SMA(50), SMA(100), SMA(200), EMA(50), EMA(200)
        Short-term trend: SMA(5),  SMA(10),  SMA(20),  EMA(9),  EMA(21)
        Residual        : price − long-term trend
        """
        # Long-term trends
        for p in [50, 100, 200]:
            df[f"ltt_sma_{p}"]   = price.rolling(p, min_periods=p//2).mean()
            df[f"ltt_sma_{p}_dist"] = (price - df[f"ltt_sma_{p}"]) / (df[f"ltt_sma_{p}"] + 1e-10)

        for p in [50, 200]:
            df[f"ltt_ema_{p}"]   = price.ewm(span=p, adjust=False).mean()
            df[f"ltt_ema_{p}_dist"] = (price - df[f"ltt_ema_{p}"]) / (df[f"ltt_ema_{p}"] + 1e-10)

        # Short-term trends
        for p in [5, 10, 20]:
            df[f"stt_sma_{p}"]   = price.rolling(p, min_periods=1).mean()
            df[f"stt_sma_{p}_dist"] = (price - df[f"stt_sma_{p}"]) / (df[f"stt_sma_{p}"] + 1e-10)

        for p in [9, 21]:
            df[f"stt_ema_{p}"]   = price.ewm(span=p, adjust=False).mean()
            df[f"stt_ema_{p}_dist"] = (price - df[f"stt_ema_{p}"]) / (df[f"stt_ema_{p}"] + 1e-10)

        # Residuals (short-term noise)
        df["ma_residual_50"]  = price - df["ltt_sma_50"]
        df["ma_residual_200"] = price - df["ltt_sma_200"]

        # LTT vs STT spread (trend divergence signal)
        df["ltt_stt_spread_50_20"]   = df["ltt_sma_50"]  - df["stt_sma_20"]
        df["ltt_stt_spread_200_50"]  = df["ltt_sma_200"] - df["ltt_sma_50"]

        # Trend direction (slope proxy)
        df["ltt_slope_50"]  = df["ltt_sma_50"].diff(5)  / (df["ltt_sma_50"].shift(5) + 1e-10)
        df["ltt_slope_200"] = df["ltt_sma_200"].diff(20) / (df["ltt_sma_200"].shift(20) + 1e-10)
        df["stt_slope_20"]  = df["stt_sma_20"].diff(3)  / (df["stt_sma_20"].shift(3) + 1e-10)

        # Trend regime: above/below long-term trend
        df["above_ltt_50"]  = (price > df["ltt_sma_50"]).astype(int)
        df["above_ltt_200"] = (price > df["ltt_sma_200"]).astype(int)

        return df

    # ── Method 2: Hodrick-Prescott Filter ─────────────────────────────────────

    def _decompose_hp(self, df: pd.DataFrame, price: pd.Series) -> pd.DataFrame:
        """
        Hodrick-Prescott filter decomposition.

        Separates price into:
            hp_trend    : Smooth long-term trend component
            hp_cycle    : Cyclical / short-term deviation from trend
            hp_cycle_norm: Cycle normalised by trend level
        """
        if not STATSMODELS_AVAILABLE:
            logger.warning("  HP filter skipped — statsmodels not available")
            return df

        try:
            # Lambda selection:
            # Daily data:  λ = 1600 (standard quarterly economic)
            # Hourly data: λ = 129,600 (scaled: 1600 × 6.25² for intraday)
            # We use 1600 for daily as per Hodrick-Prescott convention
            series = price.dropna()
            if len(series) < 20:
                logger.warning("  HP filter: insufficient data (< 20 points)")
                return df

            lambda_val = 1600  # standard for daily financial data
            cycle, trend = hpfilter(series, lamb=lambda_val)

            df["hp_trend"]       = trend.reindex(df.index)
            df["hp_cycle"]       = cycle.reindex(df.index)
            df["hp_cycle_norm"]  = df["hp_cycle"] / (df["hp_trend"].abs() + 1e-10)

            # HP trend slope
            df["hp_trend_slope"] = df["hp_trend"].diff(5) / (df["hp_trend"].shift(5) + 1e-10)

            # HP cycle regime
            df["hp_expansion"]   = (df["hp_cycle"] > 0).astype(int)
            df["hp_contraction"] = (df["hp_cycle"] < 0).astype(int)

            # HP cycle momentum
            df["hp_cycle_ma7"]   = df["hp_cycle"].rolling(7,  min_periods=1).mean()
            df["hp_cycle_ma30"]  = df["hp_cycle"].rolling(30, min_periods=1).mean()

            logger.debug("  HP filter: trend + cycle extracted")

        except Exception as e:
            logger.warning(f"  HP filter failed: {e}")

        return df

    # ── Method 3: STL Decomposition ───────────────────────────────────────────

    def _decompose_stl(self, df: pd.DataFrame, price: pd.Series) -> pd.DataFrame:
        """
        STL (Seasonal and Trend decomposition using LOESS).

        For daily crypto data, uses period=7 (weekly seasonality).
        Extracts: trend, seasonal, residual components.
        """
        if not STL_AVAILABLE:
            logger.warning("  STL skipped — statsmodels not available")
            return df

        try:
            series = price.dropna()
            if len(series) < 30:
                logger.warning("  STL: insufficient data (< 30 points)")
                return df

            stl    = STL(series, period=7, robust=True)
            result = stl.fit()

            df["stl_trend"]    = pd.Series(result.trend,    index=series.index).reindex(df.index)
            df["stl_seasonal"] = pd.Series(result.seasonal, index=series.index).reindex(df.index)
            df["stl_residual"] = pd.Series(result.resid,    index=series.index).reindex(df.index)

            # Normalised components
            df["stl_trend_norm"]    = df["stl_trend"]    / (price + 1e-10)
            df["stl_seasonal_norm"] = df["stl_seasonal"] / (price + 1e-10)
            df["stl_residual_norm"] = df["stl_residual"] / (price + 1e-10)

            # Trend slope
            df["stl_trend_slope"] = df["stl_trend"].diff(5) / (df["stl_trend"].shift(5) + 1e-10)

            # Seasonal strength: how dominant is the seasonal component?
            resid_var    = df["stl_residual"].var()
            seasonal_var = (df["stl_seasonal"] + df["stl_residual"]).var()
            df["stl_seasonal_strength"] = max(0, 1 - resid_var / (seasonal_var + 1e-10))

            logger.debug("  STL: trend + seasonal + residual extracted")

        except Exception as e:
            logger.warning(f"  STL decomposition failed: {e}")

        return df

    # ── Method 4: Wavelet Decomposition ──────────────────────────────────────

    def _decompose_wavelet(self, df: pd.DataFrame, price: pd.Series) -> pd.DataFrame:
        """
        Discrete Wavelet Transform decomposition using Daubechies-4 wavelet.

        Decomposes price into approximation (low-freq = LTT) and detail
        (high-freq = STT) coefficients at multiple levels.

        Level 1: High-frequency (1-2 day cycles)
        Level 2: Medium-frequency (2-4 day cycles)
        Level 3: Low-frequency (long-term trend)
        """
        if not PYWT_AVAILABLE:
            logger.warning("  Wavelet decomposition skipped — PyWavelets not available")
            return df

        try:
            series   = price.dropna().values
            wavelet  = "db4"
            level    = min(3, pywt.dwt_max_level(len(series), wavelet))

            coeffs   = pywt.wavedec(series, wavelet, level=level)

            # Reconstruct each level independently
            def reconstruct_level(coeffs_list, keep_level):
                """Reconstruct signal keeping only one level's coefficients."""
                masked = [np.zeros_like(c) for c in coeffs_list]
                masked[keep_level] = coeffs_list[keep_level]
                return pywt.waverec(masked, wavelet)[:len(series)]

            # Approximation = long-term trend (lowest frequency)
            approx_only = [np.zeros_like(c) for c in coeffs]
            approx_only[0] = coeffs[0]
            ltt_wave = pywt.waverec(approx_only, wavelet)[:len(series)]

            # Details = short-term (highest frequency details)
            detail_only = [np.zeros_like(c) for c in coeffs]
            detail_only[-1] = coeffs[-1]  # finest detail
            stt_wave = pywt.waverec(detail_only, wavelet)[:len(series)]

            idx = price.dropna().index
            df["wave_ltt"]      = pd.Series(ltt_wave, index=idx).reindex(df.index)
            df["wave_stt"]      = pd.Series(stt_wave, index=idx).reindex(df.index)
            df["wave_residual"] = price - df["wave_ltt"]

            # Normalised
            df["wave_ltt_norm"]  = df["wave_ltt"] / (price + 1e-10)
            df["wave_stt_norm"]  = df["wave_stt"] / (price + 1e-10)

            # Energy ratio: how much energy is in long vs short term
            ltt_energy = (df["wave_ltt"] ** 2).rolling(30).mean()
            stt_energy = (df["wave_stt"] ** 2).rolling(30).mean()
            df["wave_energy_ratio"] = ltt_energy / (stt_energy + 1e-10)

            logger.debug(f"  Wavelet (db4, level={level}): LTT + STT extracted")

        except Exception as e:
            logger.warning(f"  Wavelet decomposition failed: {e}")

        return df

    # ── Composite Features ────────────────────────────────────────────────────

    def _add_composite_features(self, df: pd.DataFrame, price: pd.Series) -> pd.DataFrame:
        """
        Cross-method composite LTST features combining multiple decompositions.
        These are the highest-value features for thesis contribution.
        """
        # Trend consensus: do MA and HP agree on trend direction?
        if "hp_trend" in df.columns and "ltt_sma_50" in df.columns:
            hp_above_ma  = (df["hp_trend"] > df["ltt_sma_50"]).astype(int)
            price_above  = (price > df["hp_trend"]).astype(int)
            df["trend_consensus"] = (hp_above_ma + price_above) / 2

        # Multi-scale momentum: LTT vs STT slope comparison
        if "ltt_slope_50" in df.columns and "stt_slope_20" in df.columns:
            df["ltst_momentum_divergence"] = (
                df["stt_slope_20"] - df["ltt_slope_50"]
            )
            df["ltst_aligned"] = (
                np.sign(df["ltt_slope_50"]) == np.sign(df["stt_slope_20"])
            ).astype(int)

        # Mean reversion signal: price far from LTT → likely to revert
        if "ma_residual_200" in df.columns:
            resid     = df["ma_residual_200"]
            resid_std = resid.rolling(60, min_periods=20).std()
            df["mean_reversion_signal"] = -resid / (resid_std + 1e-10)  # negative = buy

        # Trend strength composite
        ltt_cols = [c for c in df.columns if "ltt" in c and "slope" in c]
        if ltt_cols:
            df["ltt_strength"] = df[ltt_cols].mean(axis=1).abs()

        # Regime: trending vs mean-reverting
        if "adx" in df.columns:
            df["trending_regime"]       = (df["adx"] > 25).astype(int)
            df["mean_reverting_regime"] = (df["adx"] < 20).astype(int)

        return df


# ─── Convenience function ─────────────────────────────────────────────────────

def add_ltst_features(
    df:     pd.DataFrame,
    method: str = "all",
    price_col: str = "close",
) -> pd.DataFrame:
    """Convenience wrapper — add LTST features to an OHLCV DataFrame."""
    return LTSTDecomposer(price_col=price_col).decompose(df, method=method)


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description="LTST decomposition")
    parser.add_argument("--asset",    default="BTC", choices=["BTC","ETH","SOL","SUI","XRP"])
    parser.add_argument("--interval", default="1d",  choices=["1h","1d"])
    parser.add_argument("--method",   default="all", choices=["all","ma","hp","stl","wavelet"])
    args = parser.parse_args()

    path = Path(f"data/raw/ohlcv/{args.asset}/{args.asset}_{args.interval}.parquet")
    if not path.exists():
        print(f"Data not found: {path}")
        exit(1)

    df     = pd.read_parquet(path)
    decomp = LTSTDecomposer()
    result = decomp.decompose(df, method=args.method)

    new_cols = [c for c in result.columns if c not in df.columns]
    print(f"\nAsset    : {args.asset} {args.interval}")
    print(f"Method   : {args.method}")
    print(f"New cols : {len(new_cols)}")
    print(f"\nLTST features: {new_cols}")
    print(f"\n{result[new_cols].tail(5).to_string()}")
