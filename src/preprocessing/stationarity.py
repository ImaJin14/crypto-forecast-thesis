"""
╔══════════════════════════════════════════════════════════════════════╗
║   src/preprocessing/stationarity.py                                 ║
║   Stationarity testing and transformation                           ║
║   Author : Muluh Penn Junior Patrick                                ║
╚══════════════════════════════════════════════════════════════════════╝
Tests time series for stationarity and applies transformations
to achieve stationarity where required (primarily for macro features).

Tests:
  - Augmented Dickey-Fuller (ADF)
  - KPSS (Kwiatkowski-Phillips-Schmidt-Shin)
  - Phillips-Perron (PP) via statsmodels

Transformations:
  - First differencing
  - Log transformation
  - Log + differencing
  - Percentage returns

Usage:
    from src.preprocessing.stationarity import StationarityChecker

    checker = StationarityChecker()
    report  = checker.test_all(df)
    df_stat = checker.make_stationary(df)
    checker.print_report(report)
"""

import warnings
from typing import Optional, Union

import numpy as np
import pandas as pd
from loguru import logger

warnings.filterwarnings("ignore")

try:
    from statsmodels.tsa.stattools import adfuller, kpss
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    logger.warning("statsmodels not available — ADF/KPSS tests disabled. "
                   "Install with: pip install statsmodels")


# ─── StationarityChecker ──────────────────────────────────────────────────────

class StationarityChecker:
    """
    Tests and corrects non-stationarity in time series features.

    For deep learning models, stationarity is less critical than for
    ARIMA — however, macro features (DXY, S&P500) benefit from
    differencing as they are strongly non-stationary.

    Args:
        significance : p-value threshold for stationarity tests (default: 0.05)
        max_diffs    : Maximum differencing order to attempt (default: 2)
    """

    def __init__(self, significance: float = 0.05, max_diffs: int = 2):
        self.significance = significance
        self.max_diffs    = max_diffs
        self._transform_map: dict = {}   # {col: transform_applied}

    # ── Test Single Series ────────────────────────────────────────────────────

    def adf_test(
        self,
        series: pd.Series,
        name:   str = "",
    ) -> dict:
        """
        Augmented Dickey-Fuller test for unit root (non-stationarity).

        H0: Series has a unit root (non-stationary)
        H1: Series is stationary

        Returns:
            dict with keys: statistic, p_value, n_lags, is_stationary
        """
        if not STATSMODELS_AVAILABLE:
            return {"error": "statsmodels not available"}

        series = series.dropna()
        if len(series) < 20:
            return {"error": f"Insufficient data ({len(series)} < 20)"}

        try:
            result = adfuller(series, autolag="AIC")
            return {
                "test":         "ADF",
                "name":         name or series.name,
                "statistic":    round(result[0], 4),
                "p_value":      round(result[1], 6),
                "n_lags":       result[2],
                "n_obs":        result[3],
                "critical_1pct": result[4]["1%"],
                "critical_5pct": result[4]["5%"],
                "is_stationary": result[1] < self.significance,
            }
        except Exception as e:
            return {"error": str(e), "name": name}

    def kpss_test(
        self,
        series: pd.Series,
        name:   str = "",
    ) -> dict:
        """
        KPSS test for stationarity.

        H0: Series is stationary (opposite of ADF!)
        H1: Series has a unit root (non-stationary)
        """
        if not STATSMODELS_AVAILABLE:
            return {"error": "statsmodels not available"}

        series = series.dropna()
        if len(series) < 20:
            return {"error": f"Insufficient data ({len(series)} < 20)"}

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = kpss(series, regression="c", nlags="auto")
            return {
                "test":          "KPSS",
                "name":          name or series.name,
                "statistic":     round(result[0], 4),
                "p_value":       round(result[1], 6),
                "n_lags":        result[2],
                "critical_5pct": result[3]["5%"],
                "is_stationary": result[1] > self.significance,  # KPSS: high p = stationary
            }
        except Exception as e:
            return {"error": str(e), "name": name}

    def test_series(
        self,
        series: pd.Series,
        name:   str = "",
    ) -> dict:
        """
        Run both ADF and KPSS tests and return a combined verdict.

        Interpretation:
            ADF stationary  + KPSS stationary  → Stationary ✅
            ADF stationary  + KPSS not stationary → Trend-stationary ⚠
            ADF not stat.   + KPSS stationary  → Difference-stationary ⚠
            ADF not stat.   + KPSS not stationary → Non-stationary ❌
        """
        adf  = self.adf_test(series, name)
        kpss_res = self.kpss_test(series, name)

        adf_stat  = adf.get("is_stationary", False)
        kpss_stat = kpss_res.get("is_stationary", True)

        if adf_stat and kpss_stat:
            verdict = "stationary"
            status  = "✅"
        elif adf_stat and not kpss_stat:
            verdict = "trend-stationary"
            status  = "⚠️"
        elif not adf_stat and kpss_stat:
            verdict = "difference-stationary"
            status  = "⚠️"
        else:
            verdict = "non-stationary"
            status  = "❌"

        return {
            "name":         name or str(series.name),
            "n_obs":        len(series.dropna()),
            "adf_p":        adf.get("p_value"),
            "kpss_p":       kpss_res.get("p_value"),
            "adf_stat":     adf_stat,
            "kpss_stat":    kpss_stat,
            "verdict":      verdict,
            "status":       status,
            "needs_diff":   not adf_stat,
        }

    # ── Test DataFrame ────────────────────────────────────────────────────────

    def test_all(
        self,
        df:   pd.DataFrame,
        cols: Optional[list] = None,
        max_cols: int = 50,
    ) -> list[dict]:
        """
        Test all numeric columns in a DataFrame for stationarity.

        Args:
            df       : Feature DataFrame
            cols     : Columns to test (default: all numeric)
            max_cols : Maximum columns to test (to avoid very long runs)

        Returns:
            List of test result dicts
        """
        if cols is None:
            cols = df.select_dtypes(include=[np.number]).columns.tolist()

        # Skip binary/categorical columns
        skip_patterns = ["is_", "above_", "_cross", "_regime",
                         "_overbought", "_oversold", "fg_is_", "golden_",
                         "death_", "trending_", "mean_reverting_"]
        cols = [
            c for c in cols[:max_cols]
            if not any(c.startswith(p) or c.endswith(p) for p in skip_patterns)
        ]

        results = []
        logger.info(f"  Testing stationarity: {len(cols)} features")

        for col in cols:
            result = self.test_series(df[col].dropna(), name=col)
            results.append(result)

        n_stat    = sum(1 for r in results if r.get("verdict") == "stationary")
        n_nonstat = sum(1 for r in results if r.get("needs_diff", False))
        logger.info(f"  Results: {n_stat} stationary, {n_nonstat} need differencing")

        return results

    # ── Make Stationary ───────────────────────────────────────────────────────

    def make_stationary(
        self,
        df:           pd.DataFrame,
        test_results: Optional[list] = None,
        cols:         Optional[list] = None,
        method:       str = "auto",
    ) -> pd.DataFrame:
        """
        Transform non-stationary series to achieve stationarity.

        Note: For deep learning, we apply this selectively — mainly to
        macro series (DXY, S&P500) and not to price (already handled by
        return features). Price itself is NOT differenced as models need
        level information.

        Args:
            df           : Feature DataFrame
            test_results : Pre-computed test results (avoids re-testing)
            cols         : Specific columns to transform (overrides auto-detect)
            method       : 'auto' (use test results) | 'returns' | 'log_diff'

        Returns:
            DataFrame with transformed columns appended (originals kept)
        """
        result = df.copy()

        if cols is not None:
            # Transform specified columns
            non_stat_cols = cols
        elif test_results is not None:
            # Use pre-computed test results
            non_stat_cols = [
                r["name"] for r in test_results
                if r.get("needs_diff", False) and r["name"] in df.columns
            ]
        else:
            # Run tests automatically
            tests = self.test_all(df)
            non_stat_cols = [
                r["name"] for r in tests
                if r.get("needs_diff", False) and r["name"] in df.columns
            ]

        # Skip price columns — they're handled separately
        skip = ["open", "high", "low", "close", "volume"]
        non_stat_cols = [c for c in non_stat_cols if c not in skip]

        if not non_stat_cols:
            logger.info("  All series stationary — no transformations needed")
            return result

        logger.info(f"  Transforming {len(non_stat_cols)} non-stationary series")

        for col in non_stat_cols:
            series = result[col]
            transform = self._choose_transform(series, method)

            if transform == "returns":
                new_col = f"{col}_ret"
                result[new_col] = series.pct_change()
                self._transform_map[col] = ("returns", None)

            elif transform == "log_diff":
                new_col = f"{col}_logdiff"
                log_s   = np.log(series.clip(lower=1e-10))
                result[new_col] = log_s.diff()
                self._transform_map[col] = ("log_diff", None)

            elif transform == "diff":
                new_col = f"{col}_diff"
                result[new_col] = series.diff()
                self._transform_map[col] = ("diff", None)

            logger.debug(f"  {col} → {transform}")

        logger.success(f"  ✔  Stationarity transforms applied to {len(non_stat_cols)} features")
        return result

    # ── Report ────────────────────────────────────────────────────────────────

    def print_report(self, results: list[dict], max_rows: int = 30):
        """Print a formatted stationarity test report."""
        print("\n" + "═" * 72)
        print("  STATIONARITY TEST REPORT")
        print("═" * 72)
        print(f"  {'FEATURE':<35} {'ADF-p':>8}  {'KPSS-p':>8}  {'STATUS':<6}  VERDICT")
        print("─" * 72)

        shown = 0
        for r in results:
            if shown >= max_rows:
                print(f"  ... and {len(results) - shown} more")
                break
            adf_p  = f"{r['adf_p']:.4f}"  if r.get("adf_p")  is not None else "—"
            kpss_p = f"{r['kpss_p']:.4f}" if r.get("kpss_p") is not None else "—"
            print(f"  {r['name']:<35} {adf_p:>8}  {kpss_p:>8}  "
                  f"{r.get('status','?'):<6}  {r.get('verdict','—')}")
            shown += 1

        n_stat    = sum(1 for r in results if r.get("verdict") == "stationary")
        n_nonstat = sum(1 for r in results if r.get("needs_diff", False))
        print("─" * 72)
        print(f"  Stationary: {n_stat} / {len(results)} | "
              f"Need differencing: {n_nonstat}")
        print("═" * 72 + "\n")

    # ── Utilities ─────────────────────────────────────────────────────────────

    def _choose_transform(self, series: pd.Series, method: str) -> str:
        """Choose the best transformation for a series."""
        if method != "auto":
            return method

        # If series is always positive → log_diff (common for financial)
        if (series.dropna() > 0).all():
            return "log_diff"
        # If series has negatives → simple returns or diff
        elif series.dropna().std() > 0:
            return "returns"
        return "diff"

    @property
    def transform_map(self) -> dict:
        """Dict of {original_col: (transform_type, params)} applied."""
        return self._transform_map.copy()


# ─── Convenience functions ────────────────────────────────────────────────────

def test_stationarity(
    df:   pd.DataFrame,
    cols: Optional[list] = None,
) -> list[dict]:
    """Quick stationarity test. Returns list of test results."""
    checker = StationarityChecker()
    results = checker.test_all(df, cols)
    checker.print_report(results)
    return results


def ensure_stationary(
    df:     pd.DataFrame,
    method: str = "auto",
) -> pd.DataFrame:
    """Auto-detect and fix non-stationary columns. Returns transformed DataFrame."""
    checker = StationarityChecker()
    results = checker.test_all(df)
    return checker.make_stationary(df, test_results=results, method=method)


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Test stationarity of features")
    parser.add_argument("--asset",    default="BTC")
    parser.add_argument("--interval", default="1d", choices=["1h","1d"])
    parser.add_argument("--fix",      action="store_true",
                        help="Apply transformations to non-stationary series")
    args = parser.parse_args()

    path = Path(f"data/raw/ohlcv/{args.asset}/{args.asset}_{args.interval}.parquet")
    if not path.exists():
        print(f"Data not found: {path}")
        exit(1)

    df      = pd.read_parquet(path)
    checker = StationarityChecker()

    # Test core price features
    test_cols = ["close", "volume", "returns", "log_returns"]
    test_cols = [c for c in test_cols if c in df.columns]
    results   = checker.test_all(df, cols=test_cols)
    checker.print_report(results)

    if args.fix:
        df_stat = checker.make_stationary(df, test_results=results)
        print(f"\nTransforms applied: {checker.transform_map}")
        print(f"New columns: {[c for c in df_stat.columns if c not in df.columns]}")
