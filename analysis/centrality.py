"""
Factor Centrality (Crowding) analysis.

PCA on the factor return correlation matrix.
Centrality score = λ_i / Σλ.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


@dataclass
class CentralityResult:
    scores: dict[str, float]          # factor → centrality
    crowding_labels: dict[str, str]   # factor → 'low' / 'medium' / 'high'


def compute(
    factor_returns: dict[str, pd.Series],
    threshold_high: float = 0.50,
    threshold_medium: float = 0.30,
) -> CentralityResult:
    """Compute factor centrality via PCA on factor return correlation matrix.

    Parameters
    ----------
    factor_returns : dict[str, pd.Series]
        Mapping of factor name → monthly return series.
    """
    # build return matrix aligned on common dates
    ret_df = pd.DataFrame(factor_returns).dropna()
    if ret_df.shape[1] < 2 or ret_df.shape[0] < 3:
        scores = {f: 1.0 / max(len(factor_returns), 1) for f in factor_returns}
        labels = {f: "uncertain" for f in factor_returns}
        return CentralityResult(scores=scores, crowding_labels=labels)

    pca = PCA()
    pca.fit(ret_df.values)
    explained = pca.explained_variance_ratio_  # eigenvalue proportions

    # compute centrality: for each factor, project loadings on PCs
    # weighted by eigenvalue proportion → centrality
    loadings = pca.components_  # (n_components, n_factors)
    weighted_loadings = (loadings ** 2) * explained[:, np.newaxis]
    centrality = weighted_loadings.sum(axis=0)
    centrality = centrality / centrality.sum()

    factor_names = list(factor_returns.keys())
    scores = {f: round(float(c), 4) for f, c in zip(factor_names, centrality)}

    labels: dict[str, str] = {}
    for f, c in scores.items():
        if c >= threshold_high:
            labels[f] = "high"
        elif c >= threshold_medium:
            labels[f] = "medium"
        else:
            labels[f] = "low"

    return CentralityResult(scores=scores, crowding_labels=labels)
