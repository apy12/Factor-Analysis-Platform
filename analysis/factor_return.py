"""
Factor Return analysis.

Long-short portfolio: long top 20 %, short bottom 20 % of factor scores
each month.  Outputs factor_return_series, cumulative_return, Sharpe_ratio.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class FactorReturnResult:
    factor_return_series: pd.Series
    cumulative_return: pd.Series
    sharpe_ratio: float


def compute(
    panel: pd.DataFrame,
    factor: str,
    return_col: str = "fwd_excess_ret_3M",
    long_pct: float = 0.80,
    short_pct: float = 0.20,
) -> FactorReturnResult:
    """Compute long-short factor returns.

    Parameters
    ----------
    panel : pd.DataFrame
        Must contain ``date``, ``factor``, and ``return_col``.
    factor : str
        Column name of the percentile-ranked factor.
    return_col : str
        Column name of the forward return.
    long_pct, short_pct : float
        Percentile thresholds for long/short legs.
    """
    df = panel.dropna(subset=[factor, return_col]).copy()

    monthly_ret: list[dict] = []
    for dt, grp in df.groupby("date"):
        long_mask = grp[factor] >= long_pct
        short_mask = grp[factor] <= short_pct
        if long_mask.sum() == 0 or short_mask.sum() == 0:
            continue
        long_ret = grp.loc[long_mask, return_col].mean()
        short_ret = grp.loc[short_mask, return_col].mean()
        monthly_ret.append({"date": dt, "factor_return": long_ret - short_ret})

    ret_df = pd.DataFrame(monthly_ret).set_index("date").sort_index()
    series = ret_df["factor_return"]
    cum = (1 + series).cumprod() - 1
    sharpe = series.mean() / series.std() * np.sqrt(12) if series.std() > 0 else 0.0

    return FactorReturnResult(
        factor_return_series=series,
        cumulative_return=cum,
        sharpe_ratio=round(sharpe, 4),
    )
