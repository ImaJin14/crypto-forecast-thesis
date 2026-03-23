"""
╔══════════════════════════════════════════════════════════════════════╗
║   src/data_collection/pipeline.py                                    ║
║   Full data collection pipeline orchestrator                         ║
║   Author : Muluh Penn Junior Patrick                                 ║
╚══════════════════════════════════════════════════════════════════════╝
Orchestrates the complete data collection run:
  1. OHLCV data (Binance) for all assets and intervals
  2. Macro/correlating assets (Yahoo Finance)
  3. Sentiment data (Fear & Greed Index)
  4. Data validation report

Usage:
    python -m src.data_collection.pipeline
    python -m src.data_collection.pipeline --quick   # BTC daily only
    python -m src.data_collection.pipeline --since 2022-01-01
"""

import os
import sys
import time
import argparse
from datetime import datetime, timezone
from pathlib import Path

from loguru import logger
from dotenv import load_dotenv

load_dotenv()

# Add project root to path
ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.data_collection.binance_fetcher   import BinanceFetcher, ASSETS, ASSET_START_DATES
from src.data_collection.macro_fetcher     import MacroFetcher
from src.data_collection.sentiment_fetcher import SentimentFetcher
from src.data_collection.data_validator    import DataValidator
from src.utils.logger                      import setup_logger
from src.utils.seed                        import set_seed


# ─── Pipeline ─────────────────────────────────────────────────────────────────

class DataPipeline:
    """
    Orchestrates the full data collection process for the thesis.

    Steps:
        1. Fetch OHLCV candles from Binance (all assets × intervals)
        2. Fetch macro features from Yahoo Finance
        3. Fetch sentiment data (Fear & Greed)
        4. Validate all collected data
        5. Print summary report

    Args:
        start_date  : Global start date override (default: per-asset start)
        end_date    : End date (default: today)
        intervals   : List of candle intervals to fetch
        assets      : List of assets to fetch (default: all)
        skip_ohlcv  : Skip Binance OHLCV fetch
        skip_macro  : Skip macro data fetch
        skip_sentiment : Skip sentiment fetch
        skip_validation : Skip validation step
    """

    def __init__(
        self,
        start_date:       str  = None,
        end_date:         str  = None,
        intervals:        list = None,
        assets:           list = None,
        skip_ohlcv:       bool = False,
        skip_macro:       bool = False,
        skip_sentiment:   bool = False,
        skip_validation:  bool = False,
    ):
        self.start_date      = start_date
        self.end_date        = end_date or datetime.now(timezone.utc).strftime("%Y-%m-%d")
        self.intervals       = intervals or ["1h", "1d"]
        self.assets          = assets or list(ASSETS.keys())
        self.skip_ohlcv      = skip_ohlcv
        self.skip_macro      = skip_macro
        self.skip_sentiment  = skip_sentiment
        self.skip_validation = skip_validation

        self.binance   = BinanceFetcher()
        self.macro     = MacroFetcher()
        self.sentiment = SentimentFetcher()
        self.validator = DataValidator()

        self.results = {
            "ohlcv":     {},
            "macro":     None,
            "sentiment": None,
            "errors":    [],
            "start_time": time.time(),
        }

    def run(self) -> dict:
        """Execute the full pipeline."""
        self._print_header()

        # ── Step 1: OHLCV ──
        if not self.skip_ohlcv:
            self._run_ohlcv()
        else:
            logger.info("  [SKIP] OHLCV collection")

        # ── Step 2: Macro ──
        if not self.skip_macro:
            self._run_macro()
        else:
            logger.info("  [SKIP] Macro data collection")

        # ── Step 3: Sentiment ──
        if not self.skip_sentiment:
            self._run_sentiment()
        else:
            logger.info("  [SKIP] Sentiment collection")

        # ── Step 4: Validation ──
        if not self.skip_validation:
            self._run_validation()

        # ── Summary ──
        self._print_summary()
        return self.results

    # ── Step runners ─────────────────────────────────────────────────────────

    def _run_ohlcv(self):
        step_start = time.time()
        logger.info("\n  STEP 1 — OHLCV Data Collection (Binance)")
        logger.info(f"  Assets: {self.assets}")
        logger.info(f"  Intervals: {self.intervals}")

        total = len(self.assets) * len(self.intervals)
        done  = 0

        for asset in self.assets:
            self.results["ohlcv"][asset] = {}
            start = self.start_date or ASSET_START_DATES.get(asset, "2020-01-01")

            for interval in self.intervals:
                done += 1
                logger.info(f"  [{done}/{total}] {asset} {interval}")
                try:
                    df = self.binance.fetch(
                        asset      = asset,
                        interval   = interval,
                        start_date = start,
                        end_date   = self.end_date,
                        save       = True,
                    )
                    self.results["ohlcv"][asset][interval] = {
                        "rows":    len(df),
                        "from":    str(df.index[0].date()) if not df.empty else "—",
                        "to":      str(df.index[-1].date()) if not df.empty else "—",
                        "status":  "ok" if not df.empty else "empty",
                    }
                except Exception as e:
                    err = f"OHLCV {asset} {interval}: {e}"
                    logger.error(f"  ✗  {err}")
                    self.results["errors"].append(err)
                    self.results["ohlcv"][asset][interval] = {"status": "error", "error": str(e)}

        elapsed = time.time() - step_start
        logger.success(f"  OHLCV complete in {elapsed:.1f}s")

    def _run_macro(self):
        step_start = time.time()
        logger.info("\n  STEP 2 — Macro Data Collection (Yahoo Finance)")
        try:
            start  = self.start_date or "2018-01-01"
            df     = self.macro.fetch_all(start_date=start, end_date=self.end_date)
            self.results["macro"] = {
                "rows":    len(df),
                "cols":    len(df.columns) if not df.empty else 0,
                "status":  "ok" if not df.empty else "empty",
            }
            elapsed = time.time() - step_start
            logger.success(f"  Macro complete in {elapsed:.1f}s")
        except Exception as e:
            err = f"Macro collection failed: {e}"
            logger.error(f"  ✗  {err}")
            self.results["errors"].append(err)
            self.results["macro"] = {"status": "error", "error": str(e)}

    def _run_sentiment(self):
        step_start = time.time()
        logger.info("\n  STEP 3 — Sentiment Data Collection (Fear & Greed)")
        try:
            start  = self.start_date or "2018-02-01"
            df     = self.sentiment.fetch_all(start_date=start)
            self.results["sentiment"] = {
                "rows":   len(df),
                "cols":   len(df.columns) if not df.empty else 0,
                "status": "ok" if not df.empty else "empty",
            }
            elapsed = time.time() - step_start
            logger.success(f"  Sentiment complete in {elapsed:.1f}s")
        except Exception as e:
            err = f"Sentiment collection failed: {e}"
            logger.error(f"  ✗  {err}")
            self.results["errors"].append(err)
            self.results["sentiment"] = {"status": "error", "error": str(e)}

    def _run_validation(self):
        logger.info("\n  STEP 4 — Data Validation")
        try:
            report = self.validator.validate_all()
            self.validator.print_report(report)
            self.results["validation"] = report
        except Exception as e:
            logger.error(f"  ✗  Validation failed: {e}")

    # ── Display ───────────────────────────────────────────────────────────────

    def _print_header(self):
        print(f"""
\033[1m\033[96m
╔══════════════════════════════════════════════════════════════╗
║  📡  Data Collection Pipeline                               ║
║  crypto-forecast-thesis — Muluh Penn Junior Patrick          ║
╚══════════════════════════════════════════════════════════════╝\033[0m
  End date   : {self.end_date}
  Assets     : {', '.join(self.assets)}
  Intervals  : {', '.join(self.intervals)}
""")

    def _print_summary(self):
        elapsed = time.time() - self.results["start_time"]
        n_err   = len(self.results["errors"])
        status  = "✅ SUCCESS" if n_err == 0 else f"⚠️  DONE WITH {n_err} ERROR(S)"

        print(f"""
\033[1m\033[92m
╔══════════════════════════════════════════════════════════════╗
║  {status:<59}║
╠══════════════════════════════════════════════════════════════╣\033[0m""")

        # OHLCV summary
        for asset, intervals in self.results["ohlcv"].items():
            for interval, info in intervals.items():
                rows   = f"{info.get('rows', 0):,}"
                frm    = info.get("from", "—")
                to     = info.get("to",   "—")
                status_icon = "✔" if info.get("status") == "ok" else "✗"
                print(f"  {status_icon}  {asset:<4} {interval:<4} {rows:>9} rows  "
                      f"{frm} → {to}")

        # Macro
        if self.results.get("macro"):
            m = self.results["macro"]
            print(f"  ✔  Macro    {m.get('rows',0):>9,} rows  "
                  f"{m.get('cols',0)} features")

        # Sentiment
        if self.results.get("sentiment"):
            s = self.results["sentiment"]
            print(f"  ✔  Sentiment{s.get('rows',0):>9,} rows  "
                  f"{s.get('cols',0)} features")

        if self.results["errors"]:
            print(f"\n  Errors:")
            for err in self.results["errors"]:
                print(f"    ✗  {err}")

        print(f"""
  Total time : {elapsed:.1f}s
\033[1m\033[92m╚══════════════════════════════════════════════════════════════╝\033[0m
""")


# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the full data collection pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src.data_collection.pipeline                         # full run
  python -m src.data_collection.pipeline --quick                 # BTC daily only
  python -m src.data_collection.pipeline --asset BTC ETH         # subset assets
  python -m src.data_collection.pipeline --since 2022-01-01      # from date
  python -m src.data_collection.pipeline --skip-ohlcv            # macro + sentiment only
  python -m src.data_collection.pipeline --validate-only         # validation only
        """
    )
    parser.add_argument("--asset",          nargs="+", default=None,
                        help="Assets to fetch (default: all)")
    parser.add_argument("--interval",       nargs="+", default=None,
                        help="Intervals to fetch (default: 1h 1d)")
    parser.add_argument("--since",          default=None,
                        help="Start date YYYY-MM-DD")
    parser.add_argument("--end",            default=None,
                        help="End date YYYY-MM-DD (default: today)")
    parser.add_argument("--quick",          action="store_true",
                        help="Quick mode: BTC daily only")
    parser.add_argument("--skip-ohlcv",     action="store_true")
    parser.add_argument("--skip-macro",     action="store_true")
    parser.add_argument("--skip-sentiment", action="store_true")
    parser.add_argument("--validate-only",  action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    setup_logger()
    set_seed(42)
    args = parse_args()

    if args.validate_only:
        validator = DataValidator()
        report    = validator.validate_all()
        validator.print_report(report)
        sys.exit(0)

    if args.quick:
        assets    = ["BTC"]
        intervals = ["1d"]
    else:
        assets    = args.asset
        intervals = args.interval

    pipeline = DataPipeline(
        start_date      = args.since,
        end_date        = args.end,
        intervals       = intervals,
        assets          = assets,
        skip_ohlcv      = args.skip_ohlcv,
        skip_macro      = args.skip_macro,
        skip_sentiment  = args.skip_sentiment,
    )
    pipeline.run()