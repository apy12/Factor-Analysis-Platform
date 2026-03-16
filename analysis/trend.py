"""
Trend Detection via OLS regression on time series.

Fit y = a + b·t on IC / centrality / SHAP series.
Report slope, p_value, r_squared, trend_label.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class TrendResult:
    slope: float
    p_value: float
    r_squared: float
    trend_label: str   # 'strengthening' | 'decaying' | 'uncertain'


def compute(series: pd.Series, alpha: float = 0.05) -> TrendResult:
    """Detect trend in a time series via OLS.

    Parameters
    ----------
    series : pd.Series
        Time-indexed numeric series (e.g., monthly IC values).
    alpha : float
        Significance level for trend classification.

    Returns
    -------
    TrendResult
    """
    s = series.dropna()
    if len(s) < 5:
        return TrendResult(slope=0.0, p_value=1.0, r_squared=0.0, trend_label="uncertain")

    t = np.arange(len(s), dtype=float)
    y = s.values.astype(float)

    slope, intercept, r_value, p_value, std_err = stats.linregress(t, y)
    r_sq = r_value ** 2

    if p_value < alpha and slope > 0:
        label = "strengthening"
    elif p_value < alpha and slope < 0:
        label = "decaying"
    else:
        label = "uncertain"

    return TrendResult(
        slope=round(float(slope), 6),
        p_value=round(float(p_value), 4),
        r_squared=round(float(r_sq), 4),
        trend_label=label,
    )
