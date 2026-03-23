"""
╔══════════════════════════════════════════════════════════════════════╗
║   src/data_collection/macro_fetcher.py                              ║
║   Macro & correlating asset data fetcher via Yahoo Finance          ║
║   Author : Muluh Penn Junior Patrick                                  ║
╚══════════════════════════════════════════════════════════════════════╝
Fetches macroeconomic and cross-asset data:
  - S&P 500 (^GSPC)     — equity risk-on/off proxy
  - DXY Dollar Index     — USD strength (crypto inverse)
  - Gold (GC=F)          — safe-haven / store-of-value proxy
  - VIX (^VIX)           — market fear / volatility index
  - BTC Dominance        — from CoinGecko (market structure signal)

Usage:
    from src.data_collection.macro_fetcher import MacroFetcher

    fetcher = MacroFetcher()
    df = fetcher.fetch_all(start_date="2018-01-01")
"""

import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np
import yfinance as yf
import requests
from loguru import logger
from dotenv import load_dotenv

load_dotenv()

# ─── Constants ────────────────────────────────────────────────────────────────

MACRO_SYMBOLS = {
    "sp500":  "^GSPC",
    "dxy":    "DX-Y.NYB",
    "gold":   "GC=F",
    "vix":    "^VIX",
    "nasdaq": "^IXIC",
    "bonds":  "^TNX",    # 10-year US Treasury yield
}

DATA_DIR = Path(os.getenv("DATA_DIR", "./data"))
MACRO_DIR = DATA_DIR / "raw" / "macro"


# ─── MacroFetcher ─────────────────────────────────────────────────────────────

class MacroFetcher:
    """
    Fetches macro and cross-asset data from Yahoo Finance and CoinGecko.

    Args:
        data_dir : Root directory to save data
        delay    : Seconds between requests
    """

    def __init__(
        self,
        data_dir: Optional[Path] = None,
        delay:    float = 1.0,
    ):
        self.data_dir = Path(data_dir) if data_dir else MACRO_DIR
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.delay    = delay
        self.coingecko_key = os.getenv("COINGECKO_API_KEY", "")
        logger.info("MacroFetcher initialised")

    # ── Yahoo Finance ─────────────────────────────────────────────────────────

    def fetch_yfinance(
        self,
        name:       str,
        ticker:     str,
        start_date: str = "2018-01-01",
        end_date:   Optional[str] = None,
        interval:   str = "1d",
        save:       bool = True,
    ) -> pd.DataFrame:
        """
        Fetch a single ticker from Yahoo Finance.

        Args:
            name       : Human-readable name (e.g. 'sp500')
            ticker     : Yahoo Finance ticker (e.g. '^GSPC')
            start_date : Start date 'YYYY-MM-DD'
            end_date   : End date 'YYYY-MM-DD' (default: today)
            interval   : '1d', '1wk', '1mo'
            save       : Save to Parquet

        Returns:
            pd.DataFrame with OHLCV + derived features
        """
        end_date = end_date or datetime.now(timezone.utc).strftime("%Y-%m-%d")
        logger.info(f"Fetching {name} ({ticker}) | {start_date} → {end_date}")

        try:
            raw = yf.download(
                tickers   = ticker,
                start     = start_date,
                end       = end_date,
                interval  = interval,
                auto_adjust = True,
                progress  = False,
            )

            if raw.empty:
                logger.warning(f"  No data returned for {ticker}")
                return pd.DataFrame()

            # Flatten multi-level columns if present
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = [col[0].lower() for col in raw.columns]
            else:
                raw.columns = [col.lower() for col in raw.columns]

            # Standardise index to UTC
            raw.index = pd.to_datetime(raw.index, utc=True)
            raw.index.name = "timestamp"

            # Derived features
            raw[f"{name}_return"]     = raw["close"].pct_change()
            raw[f"{name}_log_return"] = np.log(raw["close"] / raw["close"].shift(1))
            raw[f"{name}_volatility"] = raw[f"{name}_return"].rolling(20).std()
            raw[f"{name}_ma50"]       = raw["close"].rolling(50).mean()
            raw[f"{name}_ma200"]      = raw["close"].rolling(200).mean()

            # Prefix all columns with name for clarity when merging
            rename = {col: f"{name}_{col}" for col in ["open", "high", "low", "close", "volume"]}
            raw.rename(columns=rename, inplace=True)

            if save:
                self._save(raw, name)

            logger.success(f"  ✔  {name}: {len(raw):,} rows | "
                           f"{raw.index[0].date()} → {raw.index[-1].date()}")
            return raw

        except Exception as e:
            logger.error(f"  ✗  Failed to fetch {name} ({ticker}): {e}")
            return pd.DataFrame()

    def fetch_all_yfinance(
        self,
        start_date: str = "2018-01-01",
        end_date:   Optional[str] = None,
    ) -> dict[str, pd.DataFrame]:
        """Fetch all macro symbols from Yahoo Finance."""
        results = {}
        for name, ticker in MACRO_SYMBOLS.items():
            results[name] = self.fetch_yfinance(
                name       = name,
                ticker     = ticker,
                start_date = start_date,
                end_date   = end_date,
            )
            time.sleep(self.delay)
        return results

    # ── BTC Dominance Proxy (computed from OHLCV — no API needed) ────────────

    def fetch_btc_dominance(
        self,
        start_date: str = "2018-01-01",
        save:       bool = True,
    ) -> pd.DataFrame:
        """
        Compute a BTC dominance proxy from local OHLCV data.

        True dominance = BTC_mcap / total_crypto_mcap.
        We approximate it as BTC_mcap / sum(all_asset_mcaps we hold).
        market_cap ≈ close_price × circulating_supply_proxy.

        Since we don't have circulating supply, we use a simpler proxy:
        BTC relative volume dominance among our 5 assets, which correlates
        well with true dominance direction.

        Requires OHLCV parquet files to exist (run Binance pipeline first).
        """
        logger.info("Computing BTC dominance proxy from local OHLCV...")

        ohlcv_dir = Path(os.getenv("DATA_DIR", "./data")) / "raw" / "ohlcv"
        assets    = ["BTC", "ETH", "SOL", "SUI", "XRP"]
        volumes   = {}

        for asset in assets:
            path = ohlcv_dir / asset / f"{asset}_1d.parquet"
            if path.exists():
                try:
                    df = pd.read_parquet(path)
                    if df.index.tz is None:
                        df.index = df.index.tz_localize("UTC")
                    else:
                        df.index = df.index.tz_convert("UTC")
                    # quote_volume = volume in USD terms
                    col = "quote_volume" if "quote_volume" in df.columns else "volume"
                    volumes[asset] = df[col].rename(asset)
                except Exception as e:
                    logger.warning(f"  Could not load {asset} OHLCV: {e}")

        if "BTC" not in volumes or not volumes:
            logger.warning("  BTC OHLCV not found — skipping dominance proxy")
            return pd.DataFrame()

        vol_df      = pd.DataFrame(volumes).ffill(limit=5)
        total_vol   = vol_df.sum(axis=1).replace(0, np.nan)
        btc_dom_vol = (vol_df["BTC"] / total_vol * 100).rename("btc_dominance_proxy")

        # Rolling smoothed version
        result = pd.DataFrame({
            "btc_dominance_proxy":      btc_dom_vol,
            "btc_dominance_proxy_ma7":  btc_dom_vol.rolling(7,  min_periods=1).mean(),
            "btc_dominance_proxy_ma30": btc_dom_vol.rolling(30, min_periods=1).mean(),
        })

        result = result[result.index >= pd.Timestamp(start_date, tz="UTC")]

        if save and not result.empty:
            self._save(result, "btc_dominance")
            logger.success(f"  ✔  BTC dominance proxy: {len(result):,} rows "
                           f"(computed from OHLCV, no API needed)")

        return result

    def _fetch_btc_dominance_historical(
        self,
        start_date: str,
        headers:    dict,
    ) -> pd.DataFrame:
        """Fetch historical global market cap data to compute BTC dominance."""
        try:
            # CoinGecko free tier: days=max without interval param returns auto granularity
            # For >90 days it returns daily data automatically
            url_btc  = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart"
            params   = {"vs_currency": "usd", "days": "max"}
            resp_btc = requests.get(url_btc, headers=headers, params=params, timeout=30)
            resp_btc.raise_for_status()
            btc_data = resp_btc.json()

            btc_mcap = pd.DataFrame(
                btc_data["market_caps"],
                columns=["timestamp_ms", "btc_mcap"]
            )
            btc_mcap["timestamp"] = pd.to_datetime(btc_mcap["timestamp_ms"], unit="ms", utc=True)
            btc_mcap.set_index("timestamp", inplace=True)
            btc_mcap.drop(columns=["timestamp_ms"], inplace=True)

            # Filter from start_date
            btc_mcap = btc_mcap[btc_mcap.index >= pd.Timestamp(start_date, tz="UTC")]

            # Note: true global mcap requires summing all coins.
            # We use BTC mcap as a proxy feature; true dominance requires premium API.
            btc_mcap["btc_dominance_proxy"] = btc_mcap["btc_mcap"] / btc_mcap["btc_mcap"].mean()

            logger.success(f"  ✔  BTC market cap: {len(btc_mcap):,} rows")
            return btc_mcap

        except Exception as e:
            logger.warning(f"  Historical BTC dominance fetch failed: {e}")
            return pd.DataFrame()

    # ── Merge all macro features ──────────────────────────────────────────────

    def fetch_all(
        self,
        start_date: str = "2018-01-01",
        end_date:   Optional[str] = None,
        save_merged: bool = True,
    ) -> pd.DataFrame:
        """
        Fetch all macro features and merge into a single aligned DataFrame.

        Returns:
            pd.DataFrame indexed by daily UTC timestamp with all macro features
        """
        logger.info("Fetching all macro features...")

        yf_data  = self.fetch_all_yfinance(start_date, end_date)
        btc_dom  = self.fetch_btc_dominance(start_date)

        # Extract only the derived feature columns (returns, volatility) for merging
        frames = []
        for name, df in yf_data.items():
            if df.empty:
                continue
            # Only keep derived columns (return, log_return, volatility)
            derived_cols = [c for c in df.columns if
                            any(c.endswith(s) for s in ["_return", "_log_return", "_volatility", "_ma50", "_ma200"])
                            or c in [f"{name}_close"]]
            frames.append(df[derived_cols])

        if not btc_dom.empty:
            frames.append(btc_dom)

        if not frames:
            logger.error("No macro data available.")
            return pd.DataFrame()

        # Merge all on daily index
        merged = frames[0]
        for frame in frames[1:]:
            merged = merged.join(frame, how="outer")

        # Forward-fill macro data (markets close on weekends, crypto doesn't)
        merged = merged.ffill(limit=3).bfill(limit=1)

        # Filter to date range
        start_ts = pd.Timestamp(start_date, tz="UTC")
        merged   = merged[merged.index >= start_ts]
        if end_date:
            end_ts = pd.Timestamp(end_date, tz="UTC")
            merged = merged[merged.index <= end_ts]

        if save_merged:
            path = self.data_dir / "macro_features.parquet"
            merged.to_parquet(path, compression="snappy")
            logger.success(f"  Merged macro features saved → {path} "
                           f"({len(merged):,} rows × {len(merged.columns)} cols)")

        return merged

    # ── Cross-asset correlation ───────────────────────────────────────────────

    def compute_rolling_correlation(
        self,
        macro_df:    pd.DataFrame,
        crypto_df:   pd.DataFrame,
        price_col:   str = "close",
        window:      int = 30,
    ) -> pd.DataFrame:
        """
        Compute rolling correlation between a crypto asset and macro indicators.

        Args:
            macro_df  : Macro features DataFrame
            crypto_df : Crypto OHLCV DataFrame
            price_col : Column to use from crypto_df
            window    : Rolling window in days

        Returns:
            pd.DataFrame of rolling correlations
        """
        crypto_ret = crypto_df[price_col].pct_change().rename("crypto_return")
        macro_df   = macro_df.reindex(crypto_ret.index, method="ffill")

        corr_cols  = [c for c in macro_df.columns if c.endswith("_return")]
        corr_data  = {}

        for col in corr_cols:
            combined = pd.concat([crypto_ret, macro_df[col]], axis=1).dropna()
            corr_data[f"corr_{col}"] = (
                combined["crypto_return"]
                .rolling(window)
                .corr(combined[col])
            )

        return pd.DataFrame(corr_data, index=crypto_ret.index)

    # ── I/O ───────────────────────────────────────────────────────────────────

    def _save(self, df: pd.DataFrame, name: str) -> Path:
        path = self.data_dir / f"{name}.parquet"
        df.to_parquet(path, compression="snappy")
        size_kb = path.stat().st_size / 1024
        logger.info(f"  Saved → {path}  ({size_kb:.1f} KB)")
        return path

    def load(self, name: str) -> pd.DataFrame:
        """Load saved macro data by name."""
        path = self.data_dir / f"{name}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"No data found for '{name}'. Run fetch_all() first.")
        return pd.read_parquet(path)

    def load_merged(self) -> pd.DataFrame:
        """Load the merged macro features file."""
        return self.load("macro_features")


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fetch macro/correlating asset data")
    parser.add_argument("--start",  default="2018-01-01")
    parser.add_argument("--end",    default=None)
    parser.add_argument("--symbol", default=None,
                        choices=list(MACRO_SYMBOLS.keys()) + ["all"],
                        help="Specific symbol or 'all' (default: all)")
    args = parser.parse_args()

    fetcher = MacroFetcher()

    if args.symbol and args.symbol != "all":
        df = fetcher.fetch_yfinance(
            name       = args.symbol,
            ticker     = MACRO_SYMBOLS[args.symbol],
            start_date = args.start,
            end_date   = args.end,
        )
        print(df.tail(5).to_string())
    else:
        merged = fetcher.fetch_all(start_date=args.start, end_date=args.end)
        print(f"\nMerged macro features: {merged.shape}")
        print(merged.tail(5).to_string())
