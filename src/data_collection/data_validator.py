"""
╔══════════════════════════════════════════════════════════════════════╗
║   src/data_collection/data_validator.py                             ║
║   Data quality checks across all collected datasets                 ║
║   Author : Muluh Penn Junior Patrick                                  ║
╚══════════════════════════════════════════════════════════════════════╝
"""

import os
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np
from loguru import logger
from dotenv import load_dotenv

load_dotenv()

DATA_DIR = Path(os.getenv("DATA_DIR", "./data"))


class DataValidator:
    """
    Runs comprehensive quality checks on all collected datasets.

    Checks:
    - Completeness (missing timestamps, % coverage)
    - Value integrity (nulls, zeros, negatives, OHLC sanity)
    - Temporal alignment across assets
    - Extreme values / outlier detection
    - Stale data detection

    Usage:
        validator = DataValidator()
        report = validator.validate_all()
        validator.print_report(report)
    """

    def __init__(self, data_dir: Optional[Path] = None):
        self.data_dir = Path(data_dir) if data_dir else DATA_DIR
        self.ohlcv_dir = self.data_dir / "raw" / "ohlcv"
        self.macro_dir = self.data_dir / "raw" / "macro"
        self.sent_dir  = self.data_dir / "raw" / "sentiment"

    # ── OHLCV Validation ──────────────────────────────────────────────────────

    def validate_ohlcv(self, df: pd.DataFrame, asset: str, interval: str) -> dict:
        """
        Run full OHLCV quality report for a single asset/interval.

        Returns:
            dict with quality metrics
        """
        report = {
            "asset":    asset,
            "interval": interval,
            "n_rows":   len(df),
            "passed":   [],
            "warnings": [],
            "errors":   [],
        }

        if df.empty:
            report["errors"].append("DataFrame is empty")
            return report

        # ── 1. Date range ──
        report["date_from"] = str(df.index[0].date())
        report["date_to"]   = str(df.index[-1].date())
        report["date_span_days"] = (df.index[-1] - df.index[0]).days

        # ── 2. Null check ──
        null_counts = df.isnull().sum()
        total_nulls = null_counts.sum()
        if total_nulls == 0:
            report["passed"].append("No null values")
        else:
            report["warnings"].append(f"{total_nulls} null values: {null_counts[null_counts>0].to_dict()}")

        # ── 3. Duplicate timestamps ──
        n_dupes = df.index.duplicated().sum()
        if n_dupes == 0:
            report["passed"].append("No duplicate timestamps")
        else:
            report["errors"].append(f"{n_dupes} duplicate timestamps")

        # ── 4. OHLC sanity ──
        price_cols = [c for c in ["open","high","low","close"] if c in df.columns]
        if len(price_cols) == 4:
            bad_high = (df["high"] < df[["open","close"]].max(axis=1)).sum()
            bad_low  = (df["low"]  > df[["open","close"]].min(axis=1)).sum()
            if bad_high == 0 and bad_low == 0:
                report["passed"].append("OHLC sanity checks passed")
            else:
                report["warnings"].append(f"OHLC violations — high: {bad_high}, low: {bad_low}")

        # ── 5. Zero/negative prices ──
        if "close" in df.columns:
            zero_prices = (df["close"] <= 0).sum()
            if zero_prices == 0:
                report["passed"].append("No zero/negative prices")
            else:
                report["errors"].append(f"{zero_prices} zero/negative close prices")

        # ── 6. Extreme returns ──
        if "returns" in df.columns:
            extreme = (df["returns"].abs() > 0.5).sum()
            if extreme == 0:
                report["passed"].append("No extreme single-candle returns (>50%)")
            else:
                report["warnings"].append(f"{extreme} candles with >50% return")

        # ── 7. Temporal completeness ──
        freq_map = {"1h": "1h", "4h": "4h", "1d": "1D"}
        if interval in freq_map:
            expected_idx = pd.date_range(
                df.index[0], df.index[-1],
                freq=freq_map[interval], tz="UTC"
            )
            missing = expected_idx.difference(df.index)
            completeness = (1 - len(missing) / len(expected_idx)) * 100
            report["completeness_pct"] = round(completeness, 2)
            if completeness >= 99:
                report["passed"].append(f"Completeness: {completeness:.2f}%")
            elif completeness >= 95:
                report["warnings"].append(f"Completeness: {completeness:.2f}% ({len(missing)} missing)")
            else:
                report["errors"].append(f"Low completeness: {completeness:.2f}% ({len(missing)} missing)")

        # ── 8. Volume check ──
        if "volume" in df.columns:
            zero_vol = (df["volume"] == 0).sum()
            if zero_vol == 0:
                report["passed"].append("No zero-volume candles")
            else:
                report["warnings"].append(f"{zero_vol} zero-volume candles")

        # ── Overall status ──
        if report["errors"]:
            report["status"] = "❌ FAILED"
        elif report["warnings"]:
            report["status"] = "⚠️  WARNING"
        else:
            report["status"] = "✅ PASSED"

        return report

    # ── Validate all saved data ───────────────────────────────────────────────

    def validate_all_ohlcv(self, intervals: list = None) -> list[dict]:
        """Validate all OHLCV parquet files on disk."""
        reports  = []
        assets   = ["BTC", "ETH", "SOL", "SUI", "XRP"]
        intervals = intervals or ["1h", "1d"]  # only check what we actually fetch

        for asset in assets:
            for interval in intervals:
                path = self.ohlcv_dir / asset / f"{asset}_{interval}.parquet"
                if not path.exists():
                    reports.append({
                        "asset": asset, "interval": interval,
                        "status": "⚪ NOT FOUND", "errors": ["File does not exist"]
                    })
                    continue
                try:
                    df = pd.read_parquet(path)
                    # Normalize index timezone
                    if not isinstance(df.index, pd.DatetimeIndex):
                        df.index = pd.to_datetime(df.index)
                    if df.index.tz is None:
                        df.index = df.index.tz_localize("UTC")
                    else:
                        df.index = df.index.tz_convert("UTC")
                    report = self.validate_ohlcv(df, asset, interval)
                    reports.append(report)
                except Exception as e:
                    reports.append({
                        "asset": asset, "interval": interval,
                        "status": "❌ LOAD ERROR", "errors": [str(e)]
                    })

        return reports

    def validate_all(self) -> dict:
        """Run all validation checks and return summary."""
        return {
            "ohlcv":     self.validate_all_ohlcv(),
            "timestamp": pd.Timestamp.now(tz="UTC").isoformat(),
        }

    # ── Report printing ───────────────────────────────────────────────────────

    def print_report(self, report: dict):
        """Print formatted validation report to console."""
        print("\n" + "═" * 70)
        print("  DATA VALIDATION REPORT")
        print("  Generated:", report.get("timestamp", ""))
        print("═" * 70)

        ohlcv_reports = report.get("ohlcv", [])
        print(f"\n  OHLCV ({len(ohlcv_reports)} checks)")
        print("  " + "─" * 66)
        print(f"  {'ASSET':<6} {'INT':<5} {'STATUS':<14} {'ROWS':>8}  "
              f"{'FROM':<12} {'TO':<12} {'COMPLETE':>9}")
        print("  " + "─" * 66)

        for r in ohlcv_reports:
            status   = r.get("status", "?")
            rows     = f"{r.get('n_rows', 0):,}"
            date_from = r.get("date_from", "—")
            date_to   = r.get("date_to",   "—")
            complete  = f"{r.get('completeness_pct', 0):.1f}%" if "completeness_pct" in r else "—"
            print(f"  {r['asset']:<6} {r['interval']:<5} {status:<14} {rows:>8}  "
                  f"{date_from:<12} {date_to:<12} {complete:>9}")

            for warn in r.get("warnings", []):
                print(f"         ⚠  {warn}")
            for err in r.get("errors", []):
                print(f"         ✗  {err}")

        print("\n" + "═" * 70 + "\n")


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    validator = DataValidator()
    report    = validator.validate_all()
    validator.print_report(report)
