"""
Price Momentum factor: 12-month return skipping the most recent month.

    mom = return(t-12, t-1)

Standard academic momentum definition.
"""

from __future__ import annotations

import pandas as pd


def compute(df: pd.DataFrame) -> pd.Series:
    """Compute 12-month skipping momentum from a monthly panel.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns ``ticker`` and ``close``.
        Assumed to be sorted by (ticker, date) with month-end frequency.

    Returns
    -------
    pd.Series
        Momentum values aligned with the input index.
    """
    # price 1 month ago
    price_1m = df.groupby("ticker")["close"].shift(1)
    # price 12 months ago
    price_12m = df.groupby("ticker")["close"].shift(12)
    # momentum = cumulative return from t-12 to t-1
    mom = price_1m / price_12m - 1
    return mom
