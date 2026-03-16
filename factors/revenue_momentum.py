"""
Revenue Momentum factor: year-over-year revenue growth.

    rev_mom = revenue_t / revenue_{t-4} - 1

Quarterly lag (4 quarters = 1 year).
"""

from __future__ import annotations

import pandas as pd


def compute(df: pd.DataFrame) -> pd.Series:
    """Compute revenue momentum from a monthly panel.

    The caller is expected to have already joined the latest available
    quarterly revenue (point-in-time) onto the monthly panel.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns ``ticker`` and ``revenue``.
        Should also contain a ``revenue_lag4`` column representing
        revenue from 4 quarters ago.  If absent, we compute it here
        using the ticker-level shift.

    Returns
    -------
    pd.Series
        Revenue momentum values.
    """
    if "revenue_lag4" in df.columns:
        lag = df["revenue_lag4"]
    else:
        lag = df.groupby("ticker")["revenue"].shift(4)

    rev_mom = df["revenue"] / lag - 1
    return rev_mom
