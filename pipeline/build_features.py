"""
Build features: resample daily → monthly, merge fundamentals, compute factors.

Pipeline steps:
  1. Resample daily prices to month-end
  2. Merge fundamentals using point-in-time (report_date) alignment
  3. Compute raw factor values via each factor module
  4. Cross-sectional percentile ranking
"""

from __future__ import annotations

import pandas as pd

from factors import size, revenue_momentum, price_momentum

FACTOR_NAMES = ["size", "rev_mom", "mom"]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _resample_to_month_end(prices: pd.DataFrame) -> pd.DataFrame:
    """Take the last observation per ticker per month."""
    prices = prices.copy()
    prices["month_end"] = prices["date"] + pd.offsets.MonthEnd(0)
    # keep last available record for each month per ticker
    monthly = (
        prices
        .sort_values("date")
        .groupby(["ticker", "month_end"])
        .last()
        .reset_index()
    )
    # drop the original daily 'date' that survived groupby().last()
    monthly = monthly.drop(columns=["date"])
    monthly = monthly.rename(columns={"month_end": "date"})
    return monthly[["date", "ticker", "close", "volume", "market_cap"]]


def _align_fundamentals(
    monthly: pd.DataFrame,
    fundamentals: pd.DataFrame,
) -> pd.DataFrame:
    """Point-in-time merge: for each (ticker, month_end), use the latest
    fundamental row whose report_date <= month_end.
    """
    fundamentals = fundamentals.sort_values("report_date").copy()
    monthly = monthly.sort_values("date").copy()

    # Use left_on / right_on to avoid column name collision
    merged = pd.merge_asof(
        monthly,
        fundamentals,
        left_on="date",
        right_on="report_date",
        by="ticker",
        direction="backward",
    )
    # drop the redundant report_date column
    if "report_date" in merged.columns:
        merged = merged.drop(columns=["report_date"])
    return merged


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------

def build(
    prices: pd.DataFrame,
    fundamentals: pd.DataFrame,
) -> pd.DataFrame:
    """Full feature build pipeline.

    Returns
    -------
    pd.DataFrame
        Columns: date, ticker, size, rev_mom, mom  (percentile-ranked)
        plus raw columns size_raw, rev_mom_raw, mom_raw and close/market_cap.
    """
    monthly = _resample_to_month_end(prices)
    panel = _align_fundamentals(monthly, fundamentals)

    # sort for shift operations
    panel = panel.sort_values(["ticker", "date"]).reset_index(drop=True)

    # --- raw factors ---
    panel["size_raw"] = size.compute(panel)
    panel["rev_mom_raw"] = revenue_momentum.compute(panel)
    panel["mom_raw"] = price_momentum.compute(panel)

    # --- cross-sectional ranking (percentile) each month ---
    for raw, ranked in zip(
        ["size_raw", "rev_mom_raw", "mom_raw"],
        FACTOR_NAMES,
    ):
        panel[ranked] = panel.groupby("date")[raw].rank(pct=True)

    # drop rows where factors are NaN (burn-in)
    panel = panel.dropna(subset=FACTOR_NAMES).reset_index(drop=True)

    return panel
