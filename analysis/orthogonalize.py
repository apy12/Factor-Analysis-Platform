"""
Factor Orthogonalization via regression residualization.

For each factor, regress on all other factors cross-sectionally
and use residuals as the orthogonalized version.
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression


def compute(
    panel: pd.DataFrame,
    factor_cols: list[str],
) -> pd.DataFrame:
    """Orthogonalize factors via cross-sectional regression residualization.

    Parameters
    ----------
    panel : pd.DataFrame
        Must contain ``date`` and all columns in ``factor_cols``.
    factor_cols : list[str]
        Column names of the ranked factors to orthogonalize.

    Returns
    -------
    pd.DataFrame
        Input panel augmented with ``<factor>_orth`` columns.
    """
    panel = panel.copy()
    orth_cols = [f"{f}_orth" for f in factor_cols]
    for col in orth_cols:
        panel[col] = np.nan

    for dt, grp in panel.groupby("date"):
        idx = grp.index
        sub = grp[factor_cols].dropna()
        if sub.shape[0] < 10:
            continue

        for i, target in enumerate(factor_cols):
            others = [f for f in factor_cols if f != target]
            X = sub[others].values
            y = sub[target].values
            reg = LinearRegression().fit(X, y)
            residuals = y - reg.predict(X)
            panel.loc[sub.index, f"{target}_orth"] = residuals

    return panel
