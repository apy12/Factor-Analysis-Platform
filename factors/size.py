"""
Size factor: log(market_cap).

Smaller companies receive higher raw values *after negation* if desired,
but per the spec we simply compute log(market_cap) and let ranking handle
the cross-sectional ordering.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute(df: pd.DataFrame) -> pd.Series:
    """Compute size factor from a monthly panel containing 'market_cap'.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain column ``market_cap``.

    Returns
    -------
    pd.Series
        log(market_cap) aligned with the input index.
    """
    return np.log(df["market_cap"].clip(lower=1))
