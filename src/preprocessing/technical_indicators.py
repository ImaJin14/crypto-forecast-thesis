"""
╔══════════════════════════════════════════════════════════════════════╗
║   src/preprocessing/technical_indicators.py                         ║
║   Technical indicator feature engineering                           ║
║   Author : Muluh Penn Junior Patrick                                ║
╚══════════════════════════════════════════════════════════════════════╝
Computes all technical indicators from OHLCV data using the `ta` library
with numpy fallbacks. No paid dependencies required.

Indicators computed:
  Trend     : EMA(9,21,50,200), SMA(20,50,200), MACD, ADX, Ichimoku
  Momentum  : RSI(14), Stochastic(%K,%D), Williams%R, ROC, CCI
  Volatility: Bollinger Bands, ATR, Keltner Channels, Historical Vol
  Volume    : OBV, VWAP, MFI, Volume SMA, Volume ratio
  Derived   : Golden/Death cross, price vs MA position, BB squeeze

Usage:
    from src.preprocessing.technical_indicators import TechnicalIndicators

    ti  = TechnicalIndicators()
    df  = ti.compute(ohlcv_df)           # add all indicators
    df  = ti.compute(ohlcv_df, groups=["trend","momentum"])
"""

import warnings
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger

warnings.filterwarnings("ignore", category=RuntimeWarning)

try:
    import ta
    TA_AVAILABLE = True
except ImportError:
    TA_AVAILABLE = False
    logger.warning("  'ta' library not found — using numpy fallbacks. "
                   "Install with: pip install ta")


# ─── TechnicalIndicators ──────────────────────────────────────────────────────

class TechnicalIndicators:
    """
    Computes technical analysis features from OHLCV DataFrames.

    Args:
        fillna : Forward-fill NaN values after computation (default True)
    """

    GROUPS = ["trend", "momentum", "volatility", "volume", "derived"]

    def __init__(self, fillna: bool = True):
        self.fillna = fillna

    # ── Public API ────────────────────────────────────────────────────────────

    def compute(
        self,
        df:     pd.DataFrame,
        groups: Optional[list] = None,
    ) -> pd.DataFrame:
        """
        Compute all technical indicators and append to DataFrame.

        Args:
            df     : OHLCV DataFrame with columns [open, high, low, close, volume]
            groups : Subset of indicator groups to compute. Default: all.
                     Options: 'trend', 'momentum', 'volatility', 'volume', 'derived'

        Returns:
            pd.DataFrame with original columns + all indicator columns
        """
        self._validate(df)
        groups = groups or self.GROUPS
        result = df.copy()

        o = result["open"]
        h = result["high"]
        l = result["low"]
        c = result["close"]
        v = result["volume"]

        logger.info(f"  Computing technical indicators: {groups}")

        if "trend" in groups:
            result = self._add_trend(result, o, h, l, c, v)

        if "momentum" in groups:
            result = self._add_momentum(result, h, l, c, v)

        if "volatility" in groups:
            result = self._add_volatility(result, h, l, c)

        if "volume" in groups:
            result = self._add_volume(result, h, l, c, v)

        if "derived" in groups:
            result = self._add_derived(result, c)

        if self.fillna:
            # Forward-fill then back-fill to handle leading NaNs
            result = result.ffill().bfill()

        n_new = len(result.columns) - len(df.columns)
        logger.info(f"  ✔  Added {n_new} indicator features "
                    f"({len(result.columns)} total)")
        return result

    # ── Trend Indicators ──────────────────────────────────────────────────────

    def _add_trend(self, df, o, h, l, c, v):
        """EMA, SMA, MACD, ADX, Ichimoku."""

        # Exponential Moving Averages
        for p in [9, 21, 50, 200]:
            df[f"ema_{p}"] = c.ewm(span=p, adjust=False).mean()

        # Simple Moving Averages
        for p in [20, 50, 200]:
            df[f"sma_{p}"] = c.rolling(p).mean()

        # MACD
        ema12 = c.ewm(span=12, adjust=False).mean()
        ema26 = c.ewm(span=26, adjust=False).mean()
        df["macd"]        = ema12 - ema26
        df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
        df["macd_hist"]   = df["macd"] - df["macd_signal"]
        df["macd_cross"]  = np.sign(df["macd_hist"]).diff().fillna(0)

        # ADX (Average Directional Index) — trend strength
        df["adx"] = self._adx(h, l, c, period=14)

        # Parabolic SAR (via ta if available)
        if TA_AVAILABLE:
            try:
                sar = ta.trend.PSARIndicator(h, l, c)
                df["psar"]         = sar.psar()
                df["psar_up"]      = sar.psar_up()
                df["psar_down"]    = sar.psar_down()
                df["psar_bullish"] = (c > df["psar"]).astype(int)
            except Exception:
                pass

        # Ichimoku Cloud (simplified)
        high9  = h.rolling(9).max()
        low9   = l.rolling(9).min()
        high26 = h.rolling(26).max()
        low26  = l.rolling(26).min()
        high52 = h.rolling(52).max()
        low52  = l.rolling(52).min()

        df["ichimoku_conversion"] = (high9  + low9)  / 2   # Tenkan-sen
        df["ichimoku_base"]       = (high26 + low26) / 2   # Kijun-sen
        df["ichimoku_span_a"]     = (df["ichimoku_conversion"] + df["ichimoku_base"]) / 2
        df["ichimoku_span_b"]     = (high52 + low52) / 2
        df["ichimoku_bullish"]    = (df["ichimoku_span_a"] > df["ichimoku_span_b"]).astype(int)

        return df

    # ── Momentum Indicators ───────────────────────────────────────────────────

    def _add_momentum(self, df, h, l, c, v):
        """RSI, Stochastic, Williams %R, ROC, CCI, MFI."""

        # RSI
        df["rsi_14"] = self._rsi(c, 14)
        df["rsi_7"]  = self._rsi(c, 7)
        df["rsi_21"] = self._rsi(c, 21)

        # RSI regime
        df["rsi_overbought"]  = (df["rsi_14"] > 70).astype(int)
        df["rsi_oversold"]    = (df["rsi_14"] < 30).astype(int)
        df["rsi_extreme_ob"]  = (df["rsi_14"] > 80).astype(int)
        df["rsi_extreme_os"]  = (df["rsi_14"] < 20).astype(int)

        # Stochastic Oscillator
        low14  = l.rolling(14).min()
        high14 = h.rolling(14).max()
        df["stoch_k"] = 100 * (c - low14) / (high14 - low14 + 1e-10)
        df["stoch_d"] = df["stoch_k"].rolling(3).mean()
        df["stoch_cross"] = np.sign(df["stoch_k"] - df["stoch_d"]).diff().fillna(0)

        # Williams %R
        df["williams_r"] = -100 * (high14 - c) / (high14 - low14 + 1e-10)

        # Rate of Change
        for p in [1, 7, 14, 30]:
            df[f"roc_{p}"] = c.pct_change(p) * 100

        # CCI (Commodity Channel Index)
        tp = (h + l + c) / 3   # typical price
        tp_sma  = tp.rolling(20).mean()
        tp_mad  = tp.rolling(20).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
        df["cci"] = (tp - tp_sma) / (0.015 * tp_mad + 1e-10)

        # Momentum (raw)
        df["momentum_10"] = c - c.shift(10)
        df["momentum_20"] = c - c.shift(20)

        # Price acceleration
        df["price_accel"] = df["roc_1"].diff()

        return df

    # ── Volatility Indicators ─────────────────────────────────────────────────

    def _add_volatility(self, df, h, l, c):
        """Bollinger Bands, ATR, Keltner Channels, Historical Volatility."""

        # Bollinger Bands (20, 2σ)
        sma20       = c.rolling(20).mean()
        std20       = c.rolling(20).std()
        df["bb_upper"]  = sma20 + 2 * std20
        df["bb_lower"]  = sma20 - 2 * std20
        df["bb_mid"]    = sma20
        df["bb_width"]  = (df["bb_upper"] - df["bb_lower"]) / (sma20 + 1e-10)
        df["bb_pct"]    = (c - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"] + 1e-10)
        df["bb_above"]  = (c > df["bb_upper"]).astype(int)
        df["bb_below"]  = (c < df["bb_lower"]).astype(int)

        # BB Squeeze: BB width below 20-period min (low volatility → breakout signal)
        df["bb_squeeze"] = (df["bb_width"] <= df["bb_width"].rolling(20).min()).astype(int)

        # ATR (Average True Range)
        df["atr_14"] = self._atr(h, l, c, 14)
        df["atr_7"]  = self._atr(h, l, c, 7)

        # ATR ratio (normalised volatility)
        df["atr_ratio"] = df["atr_14"] / (c + 1e-10)

        # Keltner Channels
        ema20 = c.ewm(span=20, adjust=False).mean()
        df["kc_upper"] = ema20 + 2 * df["atr_14"]
        df["kc_lower"] = ema20 - 2 * df["atr_14"]
        df["kc_pct"]   = (c - df["kc_lower"]) / (df["kc_upper"] - df["kc_lower"] + 1e-10)

        # Historical Volatility (rolling std of log returns)
        log_ret = np.log(c / c.shift(1))
        for w in [7, 14, 30]:
            df[f"hist_vol_{w}"] = log_ret.rolling(w).std() * np.sqrt(365)

        # True Range
        df["true_range"] = self._true_range(h, l, c)

        # Volatility regime
        df["high_vol"] = (df["hist_vol_14"] > df["hist_vol_14"].rolling(90).quantile(0.75)).astype(int)
        df["low_vol"]  = (df["hist_vol_14"] < df["hist_vol_14"].rolling(90).quantile(0.25)).astype(int)

        return df

    # ── Volume Indicators ─────────────────────────────────────────────────────

    def _add_volume(self, df, h, l, c, v):
        """OBV, VWAP, MFI, Volume SMA, Volume ratio."""

        # OBV (On-Balance Volume)
        direction   = np.sign(c.diff()).fillna(0)
        df["obv"]   = (direction * v).cumsum()
        df["obv_ema"] = df["obv"].ewm(span=20, adjust=False).mean()
        df["obv_signal"] = np.sign(df["obv"] - df["obv_ema"]).astype(int)

        # VWAP (daily reset approximation using cumulative method)
        tp           = (h + l + c) / 3
        df["vwap"]   = (tp * v).cumsum() / v.cumsum().replace(0, np.nan)
        df["vwap_ratio"] = c / df["vwap"].replace(0, np.nan)

        # Volume SMAs and ratios
        df["vol_sma_10"] = v.rolling(10).mean()
        df["vol_sma_20"] = v.rolling(20).mean()
        df["vol_ratio"]  = v / df["vol_sma_20"].replace(0, np.nan)  # relative volume

        # Volume surge
        df["vol_surge"]  = (df["vol_ratio"] > 2.0).astype(int)
        df["vol_dry"]    = (df["vol_ratio"] < 0.5).astype(int)

        # Money Flow Index (MFI)
        df["mfi"] = self._mfi(h, l, c, v, 14)
        df["mfi_overbought"] = (df["mfi"] > 80).astype(int)
        df["mfi_oversold"]   = (df["mfi"] < 20).astype(int)

        # Accumulation/Distribution Line
        clv = ((c - l) - (h - c)) / (h - l + 1e-10)
        df["adl"] = (clv * v).cumsum()

        # Chaikin Money Flow
        df["cmf"] = (clv * v).rolling(20).sum() / v.rolling(20).sum().replace(0, np.nan)

        return df

    # ── Derived / Cross Features ──────────────────────────────────────────────

    def _add_derived(self, df, c):
        """Golden/death cross, price vs MA, trend alignment."""

        # Golden / Death cross
        if "sma_50" in df.columns and "sma_200" in df.columns:
            cross = np.sign(df["sma_50"] - df["sma_200"])
            df["golden_cross"] = (cross == 1).astype(int)
            df["death_cross"]  = (cross == -1).astype(int)
            df["cross_signal"] = cross.diff().fillna(0)

        # Price position relative to MAs
        for ma in ["ema_9", "ema_21", "ema_50", "sma_20", "sma_50", "sma_200"]:
            if ma in df.columns:
                df[f"above_{ma}"] = (c > df[ma]).astype(int)
                df[f"dist_{ma}"]  = (c - df[ma]) / (df[ma] + 1e-10) * 100

        # Trend alignment score: how many MAs is price above? (0–6)
        ma_cols = [f"above_{m}" for m in
                   ["ema_9","ema_21","ema_50","sma_20","sma_50","sma_200"]
                   if f"above_{m}" in df.columns]
        if ma_cols:
            df["trend_alignment"] = df[ma_cols].sum(axis=1)
            df["trend_score"]     = df["trend_alignment"] / len(ma_cols)

        # Candle body / wick features
        if "open" in df.columns:
            o = df["open"]
            h = df["high"]
            l = df["low"]
            body  = (c - o).abs()
            range_ = h - l + 1e-10
            df["body_ratio"]      = body / range_
            df["upper_wick"]      = (h - c.clip(lower=o)) / range_
            df["lower_wick"]      = (c.clip(upper=o) - l) / range_
            df["is_bullish_candle"] = (c > o).astype(int)
            df["is_doji"]           = (df["body_ratio"] < 0.1).astype(int)
            df["is_hammer"]         = (
                (df["lower_wick"] > 2 * df["body_ratio"]) &
                (df["upper_wick"] < 0.1)
            ).astype(int)

        # Price distance from recent highs/lows
        for w in [7, 14, 30]:
            df[f"dist_high_{w}"] = (c - c.rolling(w).max()) / (c.rolling(w).max() + 1e-10)
            df[f"dist_low_{w}"]  = (c - c.rolling(w).min()) / (c.rolling(w).min() + 1e-10)

        return df

    # ── Indicator Implementations ─────────────────────────────────────────────

    @staticmethod
    def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
        delta = close.diff()
        gain  = delta.clip(lower=0).rolling(period, min_periods=1).mean()
        loss  = (-delta.clip(upper=0)).rolling(period, min_periods=1).mean()
        rs    = gain / loss.replace(0, np.nan)
        return 100 - (100 / (1 + rs))

    @staticmethod
    def _atr(high: pd.Series, low: pd.Series,
             close: pd.Series, period: int = 14) -> pd.Series:
        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low  - close.shift(1)).abs(),
        ], axis=1).max(axis=1)
        return tr.ewm(span=period, adjust=False).mean()

    @staticmethod
    def _true_range(high: pd.Series, low: pd.Series,
                    close: pd.Series) -> pd.Series:
        return pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low  - close.shift(1)).abs(),
        ], axis=1).max(axis=1)

    @staticmethod
    def _adx(high: pd.Series, low: pd.Series,
             close: pd.Series, period: int = 14) -> pd.Series:
        tr    = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low  - close.shift(1)).abs(),
        ], axis=1).max(axis=1)
        dm_pos = (high.diff()).clip(lower=0)
        dm_neg = (-low.diff()).clip(lower=0)
        # Zero out where the other direction is larger
        dm_pos = dm_pos.where(dm_pos > dm_neg, 0)
        dm_neg = dm_neg.where(dm_neg > dm_pos, 0)

        atr14   = tr.ewm(span=period, adjust=False).mean()
        dmp14   = dm_pos.ewm(span=period, adjust=False).mean()
        dmn14   = dm_neg.ewm(span=period, adjust=False).mean()
        di_pos  = 100 * dmp14 / (atr14 + 1e-10)
        di_neg  = 100 * dmn14 / (atr14 + 1e-10)
        dx      = 100 * (di_pos - di_neg).abs() / (di_pos + di_neg + 1e-10)
        return dx.ewm(span=period, adjust=False).mean()

    @staticmethod
    def _mfi(high: pd.Series, low: pd.Series, close: pd.Series,
             volume: pd.Series, period: int = 14) -> pd.Series:
        tp       = (high + low + close) / 3
        mf       = tp * volume
        pos_mf   = mf.where(tp > tp.shift(1), 0).rolling(period).sum()
        neg_mf   = mf.where(tp < tp.shift(1), 0).rolling(period).sum()
        mfr      = pos_mf / (neg_mf + 1e-10)
        return 100 - (100 / (1 + mfr))

    # ── Validation ────────────────────────────────────────────────────────────

    @staticmethod
    def _validate(df: pd.DataFrame):
        required = ["open", "high", "low", "close", "volume"]
        missing  = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required OHLCV columns: {missing}")
        if df.empty:
            raise ValueError("Input DataFrame is empty")


# ─── Convenience function ─────────────────────────────────────────────────────

def add_technical_indicators(
    df:     pd.DataFrame,
    groups: Optional[list] = None,
    fillna: bool = True,
) -> pd.DataFrame:
    """
    Convenience wrapper — add all technical indicators to an OHLCV DataFrame.

    Args:
        df     : OHLCV DataFrame
        groups : Indicator groups to compute (default: all)
        fillna : Forward-fill NaN values

    Returns:
        DataFrame with indicators appended
    """
    return TechnicalIndicators(fillna=fillna).compute(df, groups)


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Compute technical indicators")
    parser.add_argument("--asset",    default="BTC", choices=["BTC","ETH","SOL","SUI","XRP"])
    parser.add_argument("--interval", default="1d",  choices=["1h","1d"])
    parser.add_argument("--groups",   nargs="+",     default=None)
    args = parser.parse_args()

    data_path = Path(f"data/raw/ohlcv/{args.asset}/{args.asset}_{args.interval}.parquet")
    if not data_path.exists():
        print(f"Data not found: {data_path}. Run data collection pipeline first.")
        exit(1)

    df = pd.read_parquet(data_path)
    ti = TechnicalIndicators()
    result = ti.compute(df, groups=args.groups)

    new_cols = [c for c in result.columns if c not in df.columns]
    print(f"\nAsset    : {args.asset} {args.interval}")
    print(f"Rows     : {len(result):,}")
    print(f"New cols : {len(new_cols)}")
    print(f"\nSample (last 3 rows, indicator columns):")
    print(result[new_cols].tail(3).to_string())
