"""
╔══════════════════════════════════════════════════════════════════════╗
║   src/data_collection/binance_fetcher.py                            ║
║   Binance OHLCV data fetcher via ccxt                               ║
║   Author : Muluh Penn Junior Patrick                                  ║
╚══════════════════════════════════════════════════════════════════════╝
Fetches historical OHLCV candlestick data from Binance for all thesis
assets (BTC, ETH, SOL, SUI, XRP) across multiple timeframes.

Usage:
    from src.data_collection.binance_fetcher import BinanceFetcher

    fetcher = BinanceFetcher()
    df = fetcher.fetch("BTC", "1h", "2020-01-01", "2025-12-31")
    fetcher.fetch_all_assets()          # fetch + save all assets
"""

import os
import time
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import ccxt
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from loguru import logger

load_dotenv()

# ─── Constants ────────────────────────────────────────────────────────────────

ASSETS = {
    "BTC": "BTC/USDT",
    "ETH": "ETH/USDT",
    "SOL": "SOL/USDT",
    "SUI": "SUI/USDT",
    "XRP": "XRP/USDT",
}

INTERVALS = {
    "1h":  {"ccxt": "1h",  "label": "hourly",  "limit": 1000},
    "4h":  {"ccxt": "4h",  "label": "4hourly", "limit": 1000},
    "1d":  {"ccxt": "1d",  "label": "daily",   "limit": 1000},
}

# Asset start dates (when each coin had liquid Binance data)
ASSET_START_DATES = {
    "BTC": "2018-01-01",
    "ETH": "2018-01-01",
    "SOL": "2020-09-01",
    "SUI": "2023-05-03",   # SUI mainnet launch
    "XRP": "2018-01-01",
}

DATA_DIR = Path(os.getenv("DATA_DIR", "./data"))
RAW_DIR  = DATA_DIR / "raw" / "ohlcv"


# ─── BinanceFetcher ───────────────────────────────────────────────────────────

class BinanceFetcher:
    """
    Fetches OHLCV candlestick data from Binance via ccxt.

    Features:
    - Automatic pagination for long date ranges
    - Rate limit handling with exponential backoff
    - Incremental updates (only fetch new candles)
    - Data validation and quality checks
    - Saves to Parquet format per asset/interval

    Args:
        api_key    : Binance API key (optional for public data)
        api_secret : Binance API secret (optional for public data)
        data_dir   : Root directory to save data (default: ./data/raw/ohlcv)
        delay      : Seconds between requests (default: 0.5)
    """

    def __init__(
        self,
        api_key:    Optional[str] = None,
        api_secret: Optional[str] = None,
        data_dir:   Optional[Path] = None,
        delay:      float = 0.5,
    ):
        self.delay    = delay
        self.data_dir = Path(data_dir) if data_dir else RAW_DIR

        # Only pass API keys if actually set — public OHLCV needs no key
        resolved_key    = api_key    or os.getenv("BINANCE_API_KEY",    "")
        resolved_secret = api_secret or os.getenv("BINANCE_API_SECRET", "")
        config = {
            "enableRateLimit": True,
            "options": {"defaultType": "spot", "adjustForTimeDifference": True},
        }
        if resolved_key and resolved_secret:
            config["apiKey"] = resolved_key
            config["secret"] = resolved_secret
        self.exchange = ccxt.binance(config)

        logger.info("BinanceFetcher initialised — exchange: Binance (ccxt)")

    # ── Core fetch ────────────────────────────────────────────────────────────

    def fetch(
        self,
        asset:      str,
        interval:   str = "1d",
        start_date: Optional[str] = None,
        end_date:   Optional[str] = None,
        save:       bool = True,
    ) -> pd.DataFrame:
        """
        Fetch OHLCV data for a single asset and interval.

        Args:
            asset      : Ticker symbol e.g. 'BTC'
            interval   : Candle interval: '1h', '4h', '1d'
            start_date : 'YYYY-MM-DD' (default: asset listing date)
            end_date   : 'YYYY-MM-DD' (default: today)
            save       : Save to Parquet file (default: True)

        Returns:
            pd.DataFrame with columns: [open, high, low, close, volume,
                                        quote_volume, trades, timestamp]
        """
        if asset not in ASSETS:
            raise ValueError(f"Unknown asset '{asset}'. Available: {list(ASSETS)}")
        if interval not in INTERVALS:
            raise ValueError(f"Unknown interval '{interval}'. Available: {list(INTERVALS)}")

        symbol     = ASSETS[asset]
        start_date = start_date or ASSET_START_DATES[asset]
        end_date   = end_date   or datetime.now(timezone.utc).strftime("%Y-%m-%d")

        start_ms = self._date_to_ms(start_date)
        end_ms   = self._date_to_ms(end_date)

        logger.info(f"Fetching {asset} ({symbol}) | {interval} | {start_date} → {end_date}")

        # Check for existing data and resume from last candle
        existing_df  = self._load_existing(asset, interval)
        if existing_df is not None and not existing_df.empty:
            last_ts  = int(existing_df.index[-1].timestamp() * 1000)
            start_ms = last_ts + self._interval_ms(interval)
            logger.info(f"  Resuming from {existing_df.index[-1]} ({len(existing_df)} existing rows)")

        if start_ms >= end_ms:
            logger.info(f"  {asset} {interval} already up to date.")
            return existing_df

        # Paginate through all candles
        all_candles = self._fetch_paginated(symbol, interval, start_ms, end_ms)

        if not all_candles:
            logger.warning(f"  No new candles returned for {asset} {interval}")
            return existing_df if existing_df is not None else pd.DataFrame()

        # Build DataFrame
        df = self._candles_to_df(all_candles)

        # Merge with existing
        if existing_df is not None and not existing_df.empty:
            # Ensure both frames have matching UTC timezone before concat
            if df.index.tz is None:
                df.index = df.index.tz_localize("UTC")
            else:
                df.index = df.index.tz_convert("UTC")
            if existing_df.index.tz is None:
                existing_df.index = existing_df.index.tz_localize("UTC")
            else:
                existing_df.index = existing_df.index.tz_convert("UTC")
            df = pd.concat([existing_df, df])
            df = df[~df.index.duplicated(keep="last")]
            df.sort_index(inplace=True)

        # Validate
        df = self._validate(df, asset, interval)

        # Save
        if save:
            self._save(df, asset, interval)

        logger.success(f"  ✔  {asset} {interval}: {len(df):,} candles | "
                       f"{df.index[0].date()} → {df.index[-1].date()}")
        return df

    # ── Fetch all assets ──────────────────────────────────────────────────────

    def fetch_all_assets(
        self,
        intervals:  list[str] = ["1h", "1d"],
        start_date: Optional[str] = None,
        end_date:   Optional[str] = None,
    ) -> dict[str, dict[str, pd.DataFrame]]:
        """
        Fetch OHLCV data for all thesis assets and intervals.

        Returns:
            Nested dict: {asset: {interval: DataFrame}}
        """
        results = {}
        total   = len(ASSETS) * len(intervals)
        done    = 0

        logger.info(f"Starting full data collection — {len(ASSETS)} assets × "
                    f"{len(intervals)} intervals = {total} fetches")

        for asset in ASSETS:
            results[asset] = {}
            asset_start = start_date or ASSET_START_DATES[asset]

            for interval in intervals:
                done += 1
                logger.info(f"[{done}/{total}] {asset} {interval}")
                try:
                    df = self.fetch(
                        asset      = asset,
                        interval   = interval,
                        start_date = asset_start,
                        end_date   = end_date,
                        save       = True,
                    )
                    results[asset][interval] = df
                except Exception as e:
                    logger.error(f"  ✗  Failed {asset} {interval}: {e}")
                    results[asset][interval] = pd.DataFrame()

                time.sleep(self.delay)

        logger.success(f"Data collection complete — {total} fetches done.")
        self._print_summary(results)
        return results

    # ── Pagination ────────────────────────────────────────────────────────────

    def _fetch_paginated(
        self,
        symbol:    str,
        interval:  str,
        start_ms:  int,
        end_ms:    int,
    ) -> list:
        """Fetch all candles for a date range using pagination."""
        ccxt_tf  = INTERVALS[interval]["ccxt"]
        limit    = INTERVALS[interval]["limit"]
        candles  = []
        since    = start_ms
        retries  = 0
        max_retries = 5

        while since < end_ms:
            try:
                batch = self.exchange.fetch_ohlcv(
                    symbol    = symbol,
                    timeframe = ccxt_tf,
                    since     = since,
                    limit     = limit,
                )

                if not batch:
                    break

                # Filter to requested window
                batch = [c for c in batch if c[0] < end_ms]
                candles.extend(batch)

                # Advance cursor
                since = batch[-1][0] + self._interval_ms(interval)

                n_total = len(candles)
                pct     = min(100, int((since - start_ms) / (end_ms - start_ms) * 100))
                logger.debug(f"    Fetched {n_total:,} candles ({pct}%)")

                retries = 0  # reset on success
                time.sleep(self.delay)

                if len(batch) < limit:
                    break  # reached end of available data

            except ccxt.RateLimitExceeded:
                wait = 2 ** retries * 5
                logger.warning(f"  Rate limit hit — waiting {wait}s")
                time.sleep(wait)
                retries += 1
                if retries > max_retries:
                    logger.error("Max retries exceeded on rate limit.")
                    break

            except ccxt.NetworkError as e:
                wait = 2 ** retries * 3
                logger.warning(f"  Network error: {e} — retrying in {wait}s")
                time.sleep(wait)
                retries += 1
                if retries > max_retries:
                    logger.error("Max retries exceeded on network error.")
                    break

            except ccxt.ExchangeError as e:
                logger.error(f"  Exchange error: {e}")
                break

        return candles

    # ── DataFrame construction ────────────────────────────────────────────────

    def _candles_to_df(self, candles: list) -> pd.DataFrame:
        """Convert raw ccxt candle list to labelled DataFrame."""
        # ccxt OHLCV format: [timestamp_ms, open, high, low, close, volume]
        df = pd.DataFrame(candles, columns=[
            "timestamp_ms", "open", "high", "low", "close", "volume"
        ])

        # Convert timestamp to UTC DatetimeIndex
        df["timestamp"] = pd.to_datetime(df["timestamp_ms"], unit="ms", utc=True)
        df.set_index("timestamp", inplace=True)
        df.drop(columns=["timestamp_ms"], inplace=True)

        # Derived columns
        df["quote_volume"] = df["close"] * df["volume"]   # volume in USDT
        df["returns"]      = df["close"].pct_change()      # log return proxy
        df["log_returns"]  = np.log(df["close"] / df["close"].shift(1))
        df["hl_spread"]    = (df["high"] - df["low"]) / df["close"]  # intraday range
        df["body"]         = (df["close"] - df["open"]).abs() / df["close"]

        return df.sort_index()

    # ── Validation ────────────────────────────────────────────────────────────

    def _validate(self, df: pd.DataFrame, asset: str, interval: str) -> pd.DataFrame:
        """
        Run data quality checks and fixes:
        - Remove duplicate timestamps
        - Forward-fill small gaps (≤ 3 missing candles)
        - Flag and remove anomalous zero prices
        - Log quality report
        """
        original_len = len(df)
        issues       = []

        # 1. Remove duplicates
        n_dupes = df.index.duplicated().sum()
        if n_dupes > 0:
            df = df[~df.index.duplicated(keep="last")]
            issues.append(f"{n_dupes} duplicate timestamps removed")

        # 2. Remove zero / negative prices
        zero_mask = (df[["open", "high", "low", "close"]] <= 0).any(axis=1)
        n_zeros   = zero_mask.sum()
        if n_zeros > 0:
            df = df[~zero_mask]
            issues.append(f"{n_zeros} zero/negative price rows removed")

        # 3. Detect and fill small gaps
        expected_freq = self._interval_to_freq(interval)
        full_idx      = pd.date_range(df.index[0], df.index[-1], freq=expected_freq, tz="UTC")
        missing       = full_idx.difference(df.index)
        n_missing     = len(missing)

        if n_missing > 0:
            # Only forward-fill isolated gaps (≤ 3 consecutive)
            df = df.reindex(full_idx)
            df = df.ffill(limit=3)
            remaining_nan = df["close"].isna().sum()
            if remaining_nan > 0:
                df.dropna(subset=["close"], inplace=True)
            issues.append(f"{n_missing} missing candles — ffilled (limit=3), "
                          f"{remaining_nan} unfillable dropped")

        # 4. OHLC sanity: high ≥ max(open, close), low ≤ min(open, close)
        bad_ohlc = (
            (df["high"] < df[["open", "close"]].max(axis=1)) |
            (df["low"]  > df[["open", "close"]].min(axis=1))
        ).sum()
        if bad_ohlc > 0:
            issues.append(f"{bad_ohlc} OHLC sanity violations detected (kept)")

        # 5. Extreme returns (> 50% in one candle — likely data error)
        extreme = (df["returns"].abs() > 0.5).sum()
        if extreme > 0:
            issues.append(f"{extreme} extreme return candles (>50%) detected (kept)")

        # Quality report
        completeness = len(df) / len(full_idx) * 100 if len(full_idx) > 0 else 100
        logger.info(f"  Validation [{asset} {interval}]: "
                    f"{len(df):,} rows | {completeness:.1f}% complete")
        if issues:
            for issue in issues:
                logger.warning(f"    ⚠  {issue}")

        return df

    # ── I/O ───────────────────────────────────────────────────────────────────

    def _save(self, df: pd.DataFrame, asset: str, interval: str) -> Path:
        """Save DataFrame to Parquet file."""
        out_dir  = self.data_dir / asset
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{asset}_{interval}.parquet"
        df.to_parquet(out_path, compression="snappy", index=True)
        size_kb  = out_path.stat().st_size / 1024
        logger.info(f"  Saved → {out_path}  ({size_kb:.1f} KB)")
        return out_path

    def _load_existing(self, asset: str, interval: str) -> Optional[pd.DataFrame]:
        """Load existing Parquet file if it exists."""
        path = self.data_dir / asset / f"{asset}_{interval}.parquet"
        if path.exists():
            try:
                df = pd.read_parquet(path)
                if not isinstance(df.index, pd.DatetimeIndex):
                    df.index = pd.to_datetime(df.index, utc=True)
                elif df.index.tz is None:
                    df.index = df.index.tz_localize("UTC")
                else:
                    df.index = df.index.tz_convert("UTC")
                return df
            except Exception as e:
                logger.warning(f"  Could not load existing data: {e}")
        return None

    def load(self, asset: str, interval: str) -> pd.DataFrame:
        """
        Public method to load saved OHLCV data.

        Args:
            asset    : e.g. 'BTC'
            interval : e.g. '1d'

        Returns:
            pd.DataFrame with DatetimeIndex
        """
        df = self._load_existing(asset, interval)
        if df is None:
            raise FileNotFoundError(
                f"No data found for {asset} {interval}. Run fetch() first."
            )
        return df

    # ── Utilities ─────────────────────────────────────────────────────────────

    @staticmethod
    def _date_to_ms(date_str: str) -> int:
        """Convert 'YYYY-MM-DD' to millisecond UTC timestamp."""
        dt = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        return int(dt.timestamp() * 1000)

    @staticmethod
    def _interval_ms(interval: str) -> int:
        """Return milliseconds for a given interval string."""
        mapping = {
            "1m":  60_000,
            "5m":  300_000,
            "15m": 900_000,
            "1h":  3_600_000,
            "4h":  14_400_000,
            "1d":  86_400_000,
        }
        return mapping.get(interval, 86_400_000)

    @staticmethod
    def _interval_to_freq(interval: str) -> str:
        """Convert interval string to pandas freq alias."""
        mapping = {
            "1m":  "1min",
            "5m":  "5min",
            "15m": "15min",
            "1h":  "1h",
            "4h":  "4h",
            "1d":  "1D",
        }
        return mapping.get(interval, "1D")

    @staticmethod
    def _print_summary(results: dict):
        """Print a summary table of all fetched data."""
        print("\n" + "─" * 62)
        print(f"  {'ASSET':<6} {'INTERVAL':<10} {'ROWS':>8}  {'FROM':<12}  {'TO':<12}")
        print("─" * 62)
        for asset, intervals in results.items():
            for interval, df in intervals.items():
                if df is not None and not df.empty:
                    print(f"  {asset:<6} {interval:<10} {len(df):>8,}  "
                          f"{str(df.index[0].date()):<12}  "
                          f"{str(df.index[-1].date()):<12}")
                else:
                    print(f"  {asset:<6} {interval:<10} {'FAILED':>8}")
        print("─" * 62 + "\n")


# ─── Convenience functions ────────────────────────────────────────────────────

def fetch_single(asset: str, interval: str = "1d",
                 start_date: str = None, end_date: str = None) -> pd.DataFrame:
    """Quick single-asset fetch without instantiating the class."""
    fetcher = BinanceFetcher()
    return fetcher.fetch(asset, interval, start_date, end_date)


def fetch_all(intervals: list[str] = ["1h", "1d"]) -> dict:
    """Fetch all thesis assets and save to disk."""
    fetcher = BinanceFetcher()
    return fetcher.fetch_all_assets(intervals=intervals)


# ─── CLI entry point ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Fetch OHLCV data from Binance",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python binance_fetcher.py                          # fetch all assets, all intervals
  python binance_fetcher.py --asset BTC              # fetch BTC only
  python binance_fetcher.py --asset ETH --interval 1h
  python binance_fetcher.py --asset SOL --start 2022-01-01 --end 2024-01-01
        """
    )
    parser.add_argument("--asset",    default=None,
                        choices=list(ASSETS.keys()) + ["ALL"],
                        help="Asset to fetch (default: ALL)")
    parser.add_argument("--interval", default=None,
                        choices=list(INTERVALS.keys()),
                        help="Candle interval (default: both 1h and 1d)")
    parser.add_argument("--start",    default=None, help="Start date YYYY-MM-DD")
    parser.add_argument("--end",      default=None, help="End date YYYY-MM-DD")
    parser.add_argument("--no-save",  action="store_true", help="Don't save to disk")
    args = parser.parse_args()

    fetcher   = BinanceFetcher()
    intervals = [args.interval] if args.interval else ["1h", "1d"]
    assets    = [args.asset]    if args.asset and args.asset != "ALL" else list(ASSETS.keys())

    if len(assets) == 1 and len(intervals) == 1:
        df = fetcher.fetch(
            asset      = assets[0],
            interval   = intervals[0],
            start_date = args.start,
            end_date   = args.end,
            save       = not args.no_save,
        )
        print(df.tail(10).to_string())
        print(f"\nShape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
    else:
        fetcher.fetch_all_assets(
            intervals  = intervals,
            start_date = args.start,
            end_date   = args.end,
        )
