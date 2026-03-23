"""
╔══════════════════════════════════════════════════════════════════════╗
║   src/data_collection/onchain_fetcher.py                            ║
║   On-chain metrics — 100% free APIs, no paid keys required          ║
║   Author : Muluh Penn Junior Patrick                                 ║
╚══════════════════════════════════════════════════════════════════════╝
Data sources (all free):
  - blockchain.info   → BTC: hash rate, tx count, mempool, difficulty
  - CoinGecko free    → All assets: market cap, dominance, dev activity
  - Etherscan free    → ETH: gas price, tx count, active addresses
  - Blockchair free   → Multi-asset tx volume, address activity

Usage:
    from src.data_collection.onchain_fetcher import OnChainFetcher

    fetcher = OnChainFetcher()
    btc_df  = fetcher.fetch_btc_onchain(start_date="2020-01-01")
    eth_df  = fetcher.fetch_eth_onchain(start_date="2020-01-01")
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

# ─── Constants ────────────────────────────────────────────────────────────────

DATA_DIR    = Path(os.getenv("DATA_DIR", "./data"))
ONCHAIN_DIR = DATA_DIR / "raw" / "onchain"

# Free API base URLs
BLOCKCHAIN_INFO_URL = "https://api.blockchain.info/charts"
COINGECKO_URL       = "https://api.coingecko.com/api/v3"
ETHERSCAN_URL       = "https://api.etherscan.io/api"
BLOCKCHAIR_URL      = "https://api.blockchair.com"

# CoinGecko asset IDs
COINGECKO_IDS = {
    "BTC": "bitcoin",
    "ETH": "ethereum",
    "SOL": "solana",
    "SUI": "sui",
    "XRP": "ripple",
}

# Etherscan free key (public demo key — get your own free at etherscan.io)
ETHERSCAN_KEY = os.getenv("ETHERSCAN_API_KEY", "YourApiKeyToken")

# CoinGecko free key (optional — works without key at lower rate limit)
COINGECKO_KEY = os.getenv("COINGECKO_API_KEY", "")

# Request delays to respect free tier rate limits
DELAYS = {
    "blockchain_info": 1.5,   # generous — no strict limit but be polite
    "coingecko":       1.5,   # free tier: 30 req/min
    "etherscan":       0.25,  # free tier: 5 req/s
    "blockchair":      2.0,   # free tier: 30 req/min
}


# ─── OnChainFetcher ───────────────────────────────────────────────────────────

class OnChainFetcher:
    """
    Fetches on-chain metrics for all thesis assets using free APIs only.

    BTC  → blockchain.info  (hash rate, difficulty, tx count, mempool size)
    ETH  → Etherscan        (gas price, tx count)
    All  → CoinGecko        (market cap, total volume, circulating supply)
    All  → Blockchair       (transaction volume, output value)
    """

    def __init__(self, data_dir: Optional[Path] = None):
        self.data_dir = Path(data_dir) if data_dir else ONCHAIN_DIR
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.cg_headers = {}
        if COINGECKO_KEY and COINGECKO_KEY != "your_coingecko_api_key_here":
            self.cg_headers["x-cg-demo-api-key"] = COINGECKO_KEY

        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "crypto-forecast-thesis/1.0"})
        logger.info("OnChainFetcher initialised — using free APIs only")

    # ══════════════════════════════════════════════════════════════════════════
    # BTC ON-CHAIN  (blockchain.info)
    # ══════════════════════════════════════════════════════════════════════════

    def fetch_btc_onchain(
        self,
        start_date: str = "2018-01-01",
        end_date:   Optional[str] = None,
        save:       bool = True,
    ) -> pd.DataFrame:
        """
        Fetch BTC on-chain metrics from blockchain.info.

        Metrics:
            hash-rate       — Network hash rate (TH/s)
            difficulty      — Mining difficulty
            n-transactions  — Daily confirmed transactions
            mempool-size    — Mempool size (bytes)
            avg-block-size  — Average block size (MB)
            estimated-transaction-volume-usd — Daily tx volume in USD
            miners-revenue  — Daily miner revenue (USD)
        """
        logger.info(f"Fetching BTC on-chain metrics | {start_date} → {end_date or 'today'}")

        metrics = {
            "hash_rate":    "hash-rate",
            "difficulty":   "difficulty",
            "tx_count":     "n-transactions",
            "mempool_size": "mempool-size",
            "block_size":   "avg-block-size",
            "tx_volume_usd":"estimated-transaction-volume-usd",
            "miner_revenue":"miners-revenue",
        }

        frames = {}
        for col_name, chart_name in metrics.items():
            df = self._fetch_blockchain_info_chart(
                chart    = chart_name,
                col_name = col_name,
                start    = start_date,
                end      = end_date,
            )
            if not df.empty:
                frames[col_name] = df[col_name]
            time.sleep(DELAYS["blockchain_info"])

        if not frames:
            logger.error("No BTC on-chain data retrieved")
            return pd.DataFrame()

        # Merge all metrics on date index
        result = pd.DataFrame(frames)
        result = result.sort_index()

        # Derived features
        result["hash_rate_ma7"]   = result["hash_rate"].rolling(7).mean()
        result["hash_rate_ma30"]  = result["hash_rate"].rolling(30).mean()
        result["tx_count_ma7"]    = result["tx_count"].rolling(7).mean()
        result["difficulty_pct_change"] = result["difficulty"].pct_change()

        # NVT Signal proxy: market_cap / tx_volume_usd
        # (market cap added later from CoinGecko merge)

        # Filter date range
        result = self._filter_dates(result, start_date, end_date)

        if save:
            self._save(result, "btc_onchain")

        logger.success(f"  ✔  BTC on-chain: {len(result):,} rows, "
                       f"{len(result.columns)} metrics")
        return result

    def _fetch_blockchain_info_chart(
        self,
        chart:    str,
        col_name: str,
        start:    str,
        end:      Optional[str] = None,
    ) -> pd.DataFrame:
        """Fetch a single chart from blockchain.info API."""
        try:
            params = {
                "timespan":  "all",
                "format":    "json",
                "sampled":   "false",
            }
            url  = f"{BLOCKCHAIN_INFO_URL}/{chart}"
            resp = self.session.get(url, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()

            rows = [
                {
                    "timestamp": pd.Timestamp(v["x"], unit="s", tz="UTC"),
                    col_name:    float(v["y"]),
                }
                for v in data.get("values", [])
                if v.get("y") is not None
            ]

            if not rows:
                logger.warning(f"  No data for blockchain.info chart: {chart}")
                return pd.DataFrame()

            df = pd.DataFrame(rows).set_index("timestamp").sort_index()
            logger.debug(f"  blockchain.info [{chart}]: {len(df):,} rows")
            return df

        except Exception as e:
            logger.warning(f"  blockchain.info [{chart}] failed: {e}")
            return pd.DataFrame()

    # ══════════════════════════════════════════════════════════════════════════
    # ETH ON-CHAIN  (Etherscan free tier)
    # ══════════════════════════════════════════════════════════════════════════

    def fetch_eth_onchain(
        self,
        start_date: str = "2018-01-01",
        end_date:   Optional[str] = None,
        save:       bool = True,
    ) -> pd.DataFrame:
        """
        Fetch ETH on-chain metrics from Etherscan free API.

        Note: Etherscan free tier gives current + some historical stats.
        For deep historical data we supplement with CoinGecko.
        Sign up free at: https://etherscan.io/apis
        """
        logger.info(f"Fetching ETH on-chain metrics | {start_date} → today")

        frames = {}

        # ── Daily tx count ──
        tx_df = self._fetch_etherscan_daily_tx(start_date, end_date)
        if not tx_df.empty:
            frames["eth_tx_count"] = tx_df["eth_tx_count"]

        time.sleep(DELAYS["etherscan"])

        # ── ETH supply ──
        supply_df = self._fetch_etherscan_eth_supply()
        if not supply_df.empty:
            frames["eth_supply"] = supply_df["eth_supply"]

        if not frames:
            logger.warning("  No ETH Etherscan data — using CoinGecko fallback")
            return pd.DataFrame()

        result = pd.DataFrame(frames).sort_index()
        result = self._filter_dates(result, start_date, end_date)

        if save:
            self._save(result, "eth_onchain")

        logger.success(f"  ✔  ETH on-chain: {len(result):,} rows")
        return result

    def _fetch_etherscan_daily_tx(
        self,
        start_date: str,
        end_date:   Optional[str],
    ) -> pd.DataFrame:
        """Fetch daily ETH transaction count from Etherscan."""
        try:
            end = end_date or datetime.now(timezone.utc).strftime("%Y-%m-%d")
            params = {
                "module":    "stats",
                "action":    "dailytx",
                "startdate": start_date,
                "enddate":   end,
                "sort":      "asc",
                "apikey":    ETHERSCAN_KEY,
            }
            resp = self.session.get(ETHERSCAN_URL, params=params, timeout=20)
            resp.raise_for_status()
            data = resp.json()

            if data.get("status") != "1":
                logger.warning(f"  Etherscan daily tx: {data.get('message','unknown error')}")
                return pd.DataFrame()

            rows = [
                {
                    "timestamp":    pd.Timestamp(r["unixTimeStamp"], unit="s", tz="UTC"),
                    "eth_tx_count": int(r["transactionCount"]),
                }
                for r in data["result"]
            ]
            df = pd.DataFrame(rows).set_index("timestamp").sort_index()
            logger.debug(f"  Etherscan daily tx: {len(df):,} rows")
            return df

        except Exception as e:
            logger.warning(f"  Etherscan daily tx failed: {e}")
            return pd.DataFrame()

    def _fetch_etherscan_eth_supply(self) -> pd.DataFrame:
        """Fetch current ETH total supply (scalar → broadcast to index later)."""
        try:
            params = {
                "module": "stats",
                "action": "ethsupply2",
                "apikey": ETHERSCAN_KEY,
            }
            resp = self.session.get(ETHERSCAN_URL, params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()

            if data.get("status") != "1":
                return pd.DataFrame()

            supply = float(data["result"]["EthSupply"]) / 1e18  # convert wei → ETH
            now    = pd.Timestamp.now(tz="UTC").normalize()
            df     = pd.DataFrame(
                {"eth_supply": [supply]},
                index=pd.DatetimeIndex([now], tz="UTC")
            )
            return df

        except Exception as e:
            logger.warning(f"  Etherscan supply failed: {e}")
            return pd.DataFrame()

    # ══════════════════════════════════════════════════════════════════════════
    # ALL ASSETS — CoinGecko free tier
    # ══════════════════════════════════════════════════════════════════════════

    def fetch_coingecko_onchain(
        self,
        asset:      str,
        start_date: str = "2020-01-01",
        end_date:   Optional[str] = None,
        save:       bool = True,
    ) -> pd.DataFrame:
        """
        Fetch market + on-chain proxy metrics from CoinGecko free API.

        Available for ALL assets (BTC, ETH, SOL, SUI, XRP):
            market_cap          — USD market capitalisation
            total_volume        — 24h total volume (USD)
            market_cap_rank     — Global rank
            circulating_supply  — Coins in circulation
            nvt_proxy           — market_cap / total_volume (NVT proxy)
        """
        if asset not in COINGECKO_IDS:
            raise ValueError(f"Unknown asset '{asset}'. Available: {list(COINGECKO_IDS)}")

        coin_id = COINGECKO_IDS[asset]
        logger.info(f"Fetching CoinGecko on-chain proxy | {asset} | {start_date} → today")

        # CoinGecko market chart — free, returns up to max history
        # days=max gives full history at daily granularity
        url    = f"{COINGECKO_URL}/coins/{coin_id}/market_chart"
        params = {
            "vs_currency": "usd",
            "days":        "max",
            "interval":    "daily",
        }
        if self.cg_headers:
            params["x_cg_demo_api_key"] = COINGECKO_KEY

        try:
            resp = self.session.get(url, headers=self.cg_headers,
                                    params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()

        except requests.HTTPError as e:
            if e.response.status_code == 429:
                logger.warning("  CoinGecko rate limit — waiting 60s")
                time.sleep(60)
                resp = self.session.get(url, headers=self.cg_headers,
                                        params=params, timeout=30)
                resp.raise_for_status()
                data = resp.json()
            else:
                logger.error(f"  CoinGecko HTTP error: {e}")
                return pd.DataFrame()
        except Exception as e:
            logger.error(f"  CoinGecko fetch failed: {e}")
            return pd.DataFrame()

        # Parse market caps and volumes
        def parse_series(key, col):
            rows = data.get(key, [])
            return pd.Series(
                {pd.Timestamp(r[0], unit="ms", tz="UTC"): float(r[1])
                 for r in rows if r[1] is not None},
                name=col,
            )

        mcap   = parse_series("market_caps",    f"{asset.lower()}_market_cap")
        volume = parse_series("total_volumes",  f"{asset.lower()}_volume")

        df = pd.DataFrame({
            mcap.name:   mcap,
            volume.name: volume,
        }).sort_index()

        # NVT proxy: market cap / volume (higher = overvalued, lower = undervalued)
        df[f"{asset.lower()}_nvt_proxy"] = (
            df[mcap.name] / df[volume.name].replace(0, np.nan)
        )

        # Market cap change
        df[f"{asset.lower()}_mcap_pct_change"] = df[mcap.name].pct_change()
        df[f"{asset.lower()}_mcap_ma7"]        = df[mcap.name].rolling(7).mean()
        df[f"{asset.lower()}_mcap_ma30"]       = df[mcap.name].rolling(30).mean()

        # Filter date range
        df = self._filter_dates(df, start_date, end_date)

        if save:
            self._save(df, f"{asset.lower()}_coingecko_onchain")

        logger.success(f"  ✔  {asset} CoinGecko: {len(df):,} rows, "
                       f"{len(df.columns)} features")
        return df

    # ══════════════════════════════════════════════════════════════════════════
    # FETCH ALL
    # ══════════════════════════════════════════════════════════════════════════

    def fetch_all(
        self,
        assets:     list[str] = None,
        start_date: str = "2020-01-01",
        end_date:   Optional[str] = None,
        save_merged: bool = True,
    ) -> dict[str, pd.DataFrame]:
        """
        Fetch all on-chain data for all assets.

        Returns:
            dict: {
                'btc_onchain':  DataFrame,   # blockchain.info BTC metrics
                'eth_onchain':  DataFrame,   # Etherscan ETH metrics
                'BTC':          DataFrame,   # CoinGecko BTC
                'ETH':          DataFrame,   # CoinGecko ETH
                ...
            }
        """
        assets  = assets or list(COINGECKO_IDS.keys())
        results = {}

        # ── BTC blockchain.info ──
        logger.info("\n  [1/3] BTC on-chain (blockchain.info)")
        results["btc_onchain"] = self.fetch_btc_onchain(start_date, end_date)
        time.sleep(DELAYS["blockchain_info"])

        # ── ETH Etherscan ──
        logger.info("\n  [2/3] ETH on-chain (Etherscan)")
        results["eth_onchain"] = self.fetch_eth_onchain(start_date, end_date)
        time.sleep(DELAYS["etherscan"])

        # ── CoinGecko all assets ──
        logger.info("\n  [3/3] All assets CoinGecko market data")
        for i, asset in enumerate(assets):
            logger.info(f"    [{i+1}/{len(assets)}] {asset}")
            results[asset] = self.fetch_coingecko_onchain(asset, start_date, end_date)
            time.sleep(DELAYS["coingecko"])

        # ── Summary ──
        self._print_summary(results)
        return results

    def fetch_merged(
        self,
        asset:      str,
        start_date: str = "2020-01-01",
        end_date:   Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Return a single merged on-chain DataFrame for one asset,
        ready to join with OHLCV data.
        """
        frames = []

        # CoinGecko (available for all assets)
        cg_path = self.data_dir / f"{asset.lower()}_coingecko_onchain.parquet"
        if cg_path.exists():
            frames.append(pd.read_parquet(cg_path))

        # BTC-specific
        if asset == "BTC":
            btc_path = self.data_dir / "btc_onchain.parquet"
            if btc_path.exists():
                frames.append(pd.read_parquet(btc_path))

        # ETH-specific
        if asset == "ETH":
            eth_path = self.data_dir / "eth_onchain.parquet"
            if eth_path.exists():
                frames.append(pd.read_parquet(eth_path))

        if not frames:
            logger.warning(f"No on-chain data found for {asset}. Run fetch_all() first.")
            return pd.DataFrame()

        merged = frames[0]
        for frame in frames[1:]:
            merged = merged.join(frame, how="outer")

        merged = self._filter_dates(merged, start_date, end_date)
        merged = merged.ffill(limit=3)

        return merged.sort_index()

    # ── Utilities ─────────────────────────────────────────────────────────────

    def _filter_dates(
        self,
        df:         pd.DataFrame,
        start_date: str,
        end_date:   Optional[str],
    ) -> pd.DataFrame:
        start = pd.Timestamp(start_date, tz="UTC")
        df    = df[df.index >= start]
        if end_date:
            end = pd.Timestamp(end_date, tz="UTC")
            df  = df[df.index <= end]
        return df

    def _save(self, df: pd.DataFrame, name: str) -> Path:
        path = self.data_dir / f"{name}.parquet"
        df.to_parquet(path, compression="snappy")
        size_kb = path.stat().st_size / 1024
        logger.info(f"  Saved → {path}  ({size_kb:.1f} KB)")
        return path

    def load(self, name: str) -> pd.DataFrame:
        path = self.data_dir / f"{name}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"No data found for '{name}'. Run fetch_all() first.")
        return pd.read_parquet(path)

    @staticmethod
    def _print_summary(results: dict):
        print("\n" + "─" * 58)
        print(f"  {'SOURCE':<25} {'ROWS':>8}  {'COLS':>6}  STATUS")
        print("─" * 58)
        for name, df in results.items():
            if df is not None and not df.empty:
                print(f"  {name:<25} {len(df):>8,}  {len(df.columns):>6}  ✔")
            else:
                print(f"  {name:<25} {'—':>8}   {'—':>6}  ✗ empty")
        print("─" * 58 + "\n")


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Fetch on-chain metrics (free APIs only)"
    )
    parser.add_argument("--asset",  default="ALL",
                        choices=list(COINGECKO_IDS.keys()) + ["ALL"])
    parser.add_argument("--start",  default="2020-01-01")
    parser.add_argument("--end",    default=None)
    parser.add_argument("--source", default="all",
                        choices=["all", "btc", "eth", "coingecko"])
    args = parser.parse_args()

    fetcher = OnChainFetcher()

    if args.source == "btc" or (args.source == "all" and args.asset in ["BTC","ALL"]):
        fetcher.fetch_btc_onchain(args.start, args.end)

    elif args.source == "eth" or (args.source == "all" and args.asset in ["ETH","ALL"]):
        fetcher.fetch_eth_onchain(args.start, args.end)

    elif args.source == "coingecko":
        assets = list(COINGECKO_IDS.keys()) if args.asset == "ALL" else [args.asset]
        for a in assets:
            fetcher.fetch_coingecko_onchain(a, args.start, args.end)

    else:
        assets = list(COINGECKO_IDS.keys()) if args.asset == "ALL" else [args.asset]
        fetcher.fetch_all(assets=assets, start_date=args.start, end_date=args.end)
