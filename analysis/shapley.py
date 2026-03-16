"""
Shapley Contribution analysis.

Fits XGBoost on (factors → future_return), computes SHAP values.
Reports mean |SHAP| per factor.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import shap
from xgboost import XGBRegressor


@dataclass
class ShapleyResult:
    mean_abs_shap: dict[str, float]    # factor → mean |SHAP|
    shap_values: np.ndarray | None     # (n_samples, n_factors)


def compute(
    panel: pd.DataFrame,
    factor_cols: list[str],
    target_col: str = "fwd_excess_ret_3M",
    n_estimators: int = 100,
    max_depth: int = 4,
    seed: int = 42,
) -> ShapleyResult:
    """Compute Shapley contributions of each factor.

    Parameters
    ----------
    panel : pd.DataFrame
        Must contain ``factor_cols`` and ``target_col``.
    factor_cols : list[str]
        Column names of factors to evaluate.
    target_col : str
        Name of the target variable column.
    """
    df = panel[factor_cols + [target_col]].dropna()
    if df.shape[0] < 50:
        return ShapleyResult(
            mean_abs_shap={f: np.nan for f in factor_cols},
            shap_values=None,
        )

    X = df[factor_cols].values
    y = df[target_col].values

    model = XGBRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=seed,
        verbosity=0,
    )
    model.fit(X, y)

    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(X)

    mean_abs = np.abs(sv).mean(axis=0)
    result = {f: round(float(v), 6) for f, v in zip(factor_cols, mean_abs)}

    return ShapleyResult(mean_abs_shap=result, shap_values=sv)
