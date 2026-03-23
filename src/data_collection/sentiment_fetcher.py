"""
╔══════════════════════════════════════════════════════════════════════╗
║   src/data_collection/sentiment_fetcher.py                          ║
║   Crypto sentiment — 100% free APIs                                 ║
║   Author : Muluh Penn Junior Patrick                                 ║
╚══════════════════════════════════════════════════════════════════════╝
Sources (all free):
  1. Alternative.me   — Crypto Fear & Greed Index (full history, no key)
  2. CryptoPanic      — Crypto news sentiment     (free tier, no key needed)

Usage:
    from src.data_collection.sentiment_fetcher import SentimentFetcher

    fetcher = SentimentFetcher()
    fg_df   = fetcher.fetch_fear_greed(start_date="2018-01-01")
    cp_df   = fetcher.fetch_cryptopanic(assets=["BTC","ETH"], days=365)
    all_df  = fetcher.fetch_all(start_date="2020-01-01")
"""

import os
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np
import requests
from loguru import logger
from dotenv import load_dotenv

load_dotenv()

DATA_DIR      = Path(os.getenv("DATA_DIR", "./data"))
SENTIMENT_DIR = DATA_DIR / "raw" / "sentiment"

FEAR_GREED_URL  = "https://api.alternative.me/fng/"
CRYPTOPANIC_URL = "https://cryptopanic.com/api/v1/posts/"

CRYPTOPANIC_KEY = os.getenv("CRYPTOPANIC_API_KEY", "")

SENTIMENT_LABELS = {
    "Extreme Fear":  1,
    "Fear":          2,
    "Neutral":       3,
    "Greed":         4,
    "Extreme Greed": 5,
}

VOTE_SENTIMENT = {
    "important":  0.1,
    "liked":      0.8,
    "disliked":  -0.8,
    "lol":        0.0,
    "toxic":     -0.9,
    "saved":      0.3,
}


class SentimentFetcher:
    """
    Fetches cryptocurrency sentiment signals from free APIs only.

    Sources:
        - Crypto Fear & Greed Index (Alternative.me) — free, no key
        - CryptoPanic news sentiment                 — free tier, no key needed
    """

    def __init__(self, data_dir: Optional[Path] = None, delay: float = 1.0):
        self.data_dir = Path(data_dir) if data_dir else SENTIMENT_DIR
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.delay   = delay
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "crypto-forecast-thesis/1.0"})
        logger.info("SentimentFetcher initialised — free APIs only")

    # ══════════════════════════════════════════════════════════════════════════
    # Fear & Greed Index  (Alternative.me)
    # ══════════════════════════════════════════════════════════════════════════

    def fetch_fear_greed(
        self,
        start_date: str = "2018-02-01",
        save:       bool = True,
    ) -> pd.DataFrame:
        """
        Fetch full Crypto Fear & Greed Index history from Alternative.me.
        Free, no API key required. Available since 2018-02-01.

        Features:
            fg_value, fg_label, fg_category, fg_normalised,
            fg_ma7, fg_ma30, fg_momentum, fg_lag_1/7/14/30,
            fg_regime, fg_is_* (one-hot dummies)
        """
        logger.info(f"Fetching Fear & Greed Index | {start_date} → today")

        retries = 0
        data    = []
        while retries < 4:
            try:
                resp = self.session.get(
                    f"{FEAR_GREED_URL}?limit=0&format=json",
                    timeout=30
                )
                resp.raise_for_status()
                data = resp.json().get("data", [])
                break
            except Exception as e:
                retries += 1
                wait = 2 ** retries
                logger.warning(f"  Fear & Greed attempt {retries} failed: {e} — retry in {wait}s")
                time.sleep(wait)

        if not data:
            logger.error("Fear & Greed: no data returned after retries")
            return pd.DataFrame()

        df = self._process_fear_greed(data, start_date)

        if save and not df.empty:
            path = self.data_dir / "fear_greed.parquet"
            df.to_parquet(path, compression="snappy")
            size_kb = path.stat().st_size / 1024
            logger.success(
                f"  ✔  Fear & Greed: {len(df):,} rows → {path} ({size_kb:.1f} KB)"
            )

        return df

    def _process_fear_greed(self, records: list, start_date: str) -> pd.DataFrame:
        rows = []
        for r in records:
            try:
                ts = pd.Timestamp(int(r["timestamp"]), unit="s", tz="UTC")
                rows.append({
                    "timestamp": ts,
                    "fg_value":  int(r["value"]),
                    "fg_label":  r["value_classification"],
                })
            except Exception:
                continue

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows).set_index("timestamp").sort_index()

        df["fg_category"]   = df["fg_label"].map(SENTIMENT_LABELS).fillna(3).astype(int)
        df["fg_normalised"] = df["fg_value"] / 100.0
        df["fg_ma7"]        = df["fg_value"].rolling(7,  min_periods=1).mean()
        df["fg_ma30"]       = df["fg_value"].rolling(30, min_periods=1).mean()
        df["fg_pct_change"] = df["fg_value"].pct_change()
        df["fg_momentum"]   = df["fg_value"] - df["fg_value"].shift(7)

        for lag in [1, 7, 14, 30]:
            df[f"fg_lag_{lag}"] = df["fg_value"].shift(lag)

        def classify(v):
            if v <= 25: return "extreme_fear"
            if v <= 45: return "fear"
            if v <= 55: return "neutral"
            if v <= 75: return "greed"
            return "extreme_greed"

        df["fg_regime"] = df["fg_value"].apply(classify)
        for regime in ["extreme_fear", "fear", "neutral", "greed", "extreme_greed"]:
            df[f"fg_is_{regime}"] = (df["fg_regime"] == regime).astype(int)

        return df[df.index >= pd.Timestamp(start_date, tz="UTC")]

    # ══════════════════════════════════════════════════════════════════════════
    # CryptoPanic  (free tier)
    # ══════════════════════════════════════════════════════════════════════════

    # ══════════════════════════════════════════════════════════════════════════
    # CoinGecko Trending  (free, no key, no auth)
    # ══════════════════════════════════════════════════════════════════════════

    def fetch_coingecko_trending(
        self,
        assets:    list = None,
        start_date: str = "2020-01-01",
        save:      bool = True,
    ) -> pd.DataFrame:
        """
        Build a sentiment proxy from CoinGecko's free market data endpoint.

        Uses daily price change % and market cap rank change as a proxy for
        sentiment — assets rallying and rising in rank signal positive sentiment.

        CoinGecko /coins/markets — free, no key, 30 req/min.

        Features per asset:
            {asset}_price_change_24h     : 24h price change %
            {asset}_mcap_rank            : Market cap rank (lower = larger)
            {asset}_volume_mcap_ratio    : Volume / market cap (activity signal)
            {asset}_sentiment_votes_up   : Community bullish vote % (CoinGecko)
        """
        assets     = assets or ["BTC", "ETH", "SOL", "SUI", "XRP"]
        asset_ids  = {
            "BTC": "bitcoin", "ETH": "ethereum", "SOL": "solana",
            "SUI": "sui",     "XRP": "ripple",
        }
        ids_str    = ",".join(asset_ids[a] for a in assets if a in asset_ids)

        logger.info(f"Fetching CoinGecko market sentiment | {assets}")

        try:
            url    = "https://api.coingecko.com/api/v3/coins/markets"
            params = {
                "vs_currency":            "usd",
                "ids":                    ids_str,
                "order":                  "market_cap_desc",
                "per_page":               50,
                "page":                   1,
                "price_change_percentage": "24h,7d,30d",
                "sparkline":              "false",
            }
            resp = self.session.get(url, params=params, timeout=20)
            resp.raise_for_status()
            coins = resp.json()

        except Exception as e:
            logger.warning(f"  CoinGecko markets fetch failed: {e}")
            return pd.DataFrame()

        # Build today's snapshot row
        id_to_ticker = {v: k for k, v in asset_ids.items()}
        today        = pd.Timestamp.now(tz="UTC").normalize()
        rows         = {}

        for coin in coins:
            ticker = id_to_ticker.get(coin["id"])
            if not ticker:
                continue
            rows[f"{ticker.lower()}_price_change_24h"]  = coin.get("price_change_percentage_24h", 0)
            rows[f"{ticker.lower()}_price_change_7d"]   = coin.get("price_change_percentage_7d_in_currency", 0)
            rows[f"{ticker.lower()}_mcap_rank"]          = coin.get("market_cap_rank", 999)
            rows[f"{ticker.lower()}_sentiment_votes_up"] = coin.get("sentiment_votes_up_percentage", 50)
            vol  = coin.get("total_volume", 0) or 0
            mcap = coin.get("market_cap", 1) or 1
            rows[f"{ticker.lower()}_volume_mcap_ratio"] = vol / mcap

        if not rows:
            logger.warning("  No CoinGecko market data parsed")
            return pd.DataFrame()

        # Create single-row DataFrame for today, then try to load + append existing
        df_today = pd.DataFrame([rows], index=pd.DatetimeIndex([today], tz="UTC"))
        df_today.index.name = "timestamp"

        # Load existing snapshot history and append
        path = self.data_dir / "coingecko_sentiment.parquet"
        if path.exists():
            try:
                existing = pd.read_parquet(path)
                if existing.index.tz is None:
                    existing.index = existing.index.tz_localize("UTC")
                df_today = pd.concat([existing, df_today])
                df_today = df_today[~df_today.index.duplicated(keep="last")]
                df_today.sort_index(inplace=True)
            except Exception as e:
                logger.warning(f"  Could not load existing CoinGecko data: {e}")

        # Filter from start_date
        df_today = df_today[df_today.index >= pd.Timestamp(start_date, tz="UTC")]

        if save and not df_today.empty:
            df_today.to_parquet(path, compression="snappy")
            size_kb = path.stat().st_size / 1024
            logger.success(f"  ✔  CoinGecko sentiment: {len(df_today):,} snapshots → "
                           f"{path} ({size_kb:.1f} KB)")

        return df_today

    # ══════════════════════════════════════════════════════════════════════════
    # FETCH ALL — Fear & Greed + CoinGecko market sentiment
    # ══════════════════════════════════════════════════════════════════════════

    def fetch_all(
        self,
        assets:     list = None,
        start_date: str  = "2018-02-01",
        save:       bool = True,
    ) -> pd.DataFrame:
        """
        Fetch and merge all sentiment sources into one aligned DataFrame.

        Sources:
            1. Fear & Greed Index  (Alternative.me)  — full history since 2018
            2. CoinGecko markets   (free, no key)    — daily snapshot (append mode)
        """
        assets = assets or ["BTC", "ETH", "SOL", "SUI", "XRP"]
        logger.info("Fetching all sentiment data (free APIs only)...")

        # ── Fear & Greed ──
        fg_df = self.fetch_fear_greed(start_date=start_date, save=save)
        time.sleep(self.delay)

        # ── CoinGecko market sentiment ──
        cg_df = self.fetch_coingecko_trending(
            assets=assets, start_date=start_date, save=save
        )
        time.sleep(self.delay)

        if fg_df.empty and cg_df.empty:
            logger.error("No sentiment data available.")
            return pd.DataFrame()

        if not fg_df.empty and not cg_df.empty:
            merged = fg_df.join(cg_df, how="outer")
        else:
            merged = fg_df if not fg_df.empty else cg_df

        merged = merged.ffill(limit=3)
        merged = merged[merged.index >= pd.Timestamp(start_date, tz="UTC")]

        if save:
            path = self.data_dir / "sentiment_all.parquet"
            merged.to_parquet(path, compression="snappy")
            logger.success(
                f"  All sentiment → {path} "
                f"({len(merged):,} rows × {len(merged.columns)} cols)"
            )

        return merged

    # ── Loaders ───────────────────────────────────────────────────────────────

    def load_fear_greed(self) -> pd.DataFrame:
        return self._load("fear_greed")

    def load_coingecko_sentiment(self) -> pd.DataFrame:
        return self._load("coingecko_sentiment")

    def load_all(self) -> pd.DataFrame:
        return self._load("sentiment_all")

    def _load(self, name: str) -> pd.DataFrame:
        path = self.data_dir / f"{name}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"'{name}' not found. Run fetch_all() first.")
        return pd.read_parquet(path)


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Fetch sentiment data (free APIs only)")
    parser.add_argument("--start",  default="2018-02-01")
    parser.add_argument("--source", default="all",
                        choices=["all", "fear_greed", "cryptopanic"])
    parser.add_argument("--assets", nargs="+",
                        default=["BTC", "ETH", "SOL", "SUI", "XRP"])
    args = parser.parse_args()

    fetcher = SentimentFetcher()

    if args.source == "fear_greed":
        df = fetcher.fetch_fear_greed(start_date=args.start)
    elif args.source == "cryptopanic":
        df = fetcher.fetch_cryptopanic(assets=args.assets)
    else:
        df = fetcher.fetch_all(assets=args.assets, start_date=args.start)

    if not df.empty:
        print(f"\nShape   : {df.shape}")
        print(f"Columns : {list(df.columns)}")
        print(f"\n{df.tail(5).to_string()}")
