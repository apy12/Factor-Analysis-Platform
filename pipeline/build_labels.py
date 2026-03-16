"""
Build labels: forward excess returns at multiple horizons.

Target = future stock return − cross-sectional mean return (market proxy).

Horizons: 1M, 3M, 6M, 12M.
Default analysis horizon: 3M.
"""

from __future__ import annotations

import pandas as pd

HORIZONS = {"1M": 1, "3M": 3, "6M": 6, "12M": 12}
DEFAULT_HORIZON = "3M"


def build(panel: pd.DataFrame, horizons: dict[str, int] | None = None) -> pd.DataFrame:
    """Add forward-return columns to a monthly panel.

    Parameters
    ----------
    panel : pd.DataFrame
        Must contain columns ``date``, ``ticker``, ``close``.
    horizons : dict[str, int], optional
        Mapping of label name → number of months forward.
        Defaults to HORIZONS.

    Returns
    -------
    pd.DataFrame
        Input panel augmented with ``fwd_ret_<key>`` and
        ``fwd_excess_ret_<key>`` columns.
    """
    if horizons is None:
        horizons = HORIZONS

    panel = panel.sort_values(["ticker", "date"]).copy()

    for label, months in horizons.items():
        col_ret = f"fwd_ret_{label}"
        col_exret = f"fwd_excess_ret_{label}"

        # forward return per ticker
        future_price = panel.groupby("ticker")["close"].shift(-months)
        panel[col_ret] = future_price / panel["close"] - 1

        # cross-sectional mean return each month (market proxy)
        mkt_ret = panel.groupby("date")[col_ret].transform("mean")
        panel[col_exret] = panel[col_ret] - mkt_ret

    return panel
