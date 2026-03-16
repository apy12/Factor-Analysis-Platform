"""
IC Decay Curve.

Evaluate IC at multiple forward horizons (1M, 3M, 6M, 12M) to determine
optimal holding period (peak IC horizon).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


@dataclass
class ICDecayResult:
    decay_curve: dict[str, float]    # horizon → mean IC
    peak_horizon: str | None         # horizon with highest |IC|


def compute(
    panel: pd.DataFrame,
    factor: str,
    horizons: list[str] | None = None,
) -> ICDecayResult:
    """Compute IC decay curve across multiple horizons.

    Parameters
    ----------
    panel : pd.DataFrame
        Must contain ``factor`` and ``fwd_excess_ret_<horizon>`` columns.
    factor : str
        Column name of the factor score.
    horizons : list[str], optional
        Horizon labels to evaluate.  Defaults to ["1M", "3M", "6M", "12M"].
    """
    if horizons is None:
        horizons = ["1M", "3M", "6M", "12M"]

    curve: dict[str, float] = {}
    for h in horizons:
        ret_col = f"fwd_excess_ret_{h}"
        if ret_col not in panel.columns:
            curve[h] = np.nan
            continue

        df = panel.dropna(subset=[factor, ret_col])
        ic_list: list[float] = []
        for _, grp in df.groupby("date"):
            if grp.shape[0] < 10:
                continue
            rho, _ = spearmanr(grp[factor], grp[ret_col])
            ic_list.append(rho)

        curve[h] = round(float(np.mean(ic_list)), 4) if ic_list else np.nan

    # identify peak
    valid = {k: abs(v) for k, v in curve.items() if not np.isnan(v)}
    peak = max(valid, key=valid.get) if valid else None

    return ICDecayResult(decay_curve=curve, peak_horizon=peak)
