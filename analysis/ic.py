"""
Information Coefficient (IC) analysis.

IC = Spearman rank correlation between factor score and future return,
computed cross-sectionally each month.

Outputs: IC_t series, mean_IC, IC_IR (= mean / std).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


@dataclass
class ICResult:
    ic_series: pd.Series       # monthly IC
    mean_ic: float
    ic_ir: float               # IC information ratio


def compute(
    panel: pd.DataFrame,
    factor: str,
    return_col: str = "fwd_excess_ret_3M",
) -> ICResult:
    """Compute monthly cross-sectional Spearman IC.

    Parameters
    ----------
    panel : pd.DataFrame
        Must contain ``date``, ``factor``, and ``return_col``.
    factor : str
        Column name of the factor score.
    return_col : str
        Column name of the forward return.
    """
    df = panel.dropna(subset=[factor, return_col])

    ic_list: list[dict] = []
    for dt, grp in df.groupby("date"):
        if grp.shape[0] < 10:
            continue
        rho, _ = spearmanr(grp[factor], grp[return_col])
        ic_list.append({"date": dt, "ic": rho})

    if not ic_list:
        return ICResult(ic_series=pd.Series(dtype=float), mean_ic=np.nan, ic_ir=np.nan)

    ic_df = pd.DataFrame(ic_list).set_index("date").sort_index()
    series = ic_df["ic"]
    mean_ic = float(series.mean())
    std_ic = float(series.std())
    ic_ir = mean_ic / std_ic if std_ic > 0 else 0.0

    return ICResult(
        ic_series=series,
        mean_ic=round(mean_ic, 4),
        ic_ir=round(ic_ir, 4),
    )
