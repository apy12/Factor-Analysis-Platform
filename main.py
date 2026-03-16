"""
Factor Analysis Platform — End-to-end runner.

Pipeline:
  1. Generate synthetic data
  2. Build features (resample, align, compute factors, rank)
  3. Build labels  (forward excess returns at multiple horizons)
  4. Orthogonalize factors
  5. Run all 7 diagnostics for each factor
  6. Print structured report
"""

from __future__ import annotations

import sys
import os

# Ensure the project root is on sys.path so that relative imports work
# when running as `python main.py` from any directory.
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import warnings
warnings.filterwarnings("ignore")

from data.generate_sample_data import generate_price_data, generate_fundamental_data
from pipeline.build_features import build as build_features, FACTOR_NAMES
from pipeline.build_labels import build as build_labels
from analysis import (
    factor_return,
    centrality,
    shapley,
    ic,
    ic_decay,
    orthogonalize,
    trend,
)
from report import print_factor_report


# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------
DISPLAY_FACTORS = {
    "size":    "Size",
    "rev_mom": "Revenue Momentum",
    "mom":     "Price Momentum",
}
TARGET_COL = "fwd_excess_ret_3M"


def main() -> None:
    print("=" * 60)
    print("  Factor Analysis Platform  —  SPEC v2")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. Data generation
    # ------------------------------------------------------------------
    print("\n[1/6] Generating synthetic data …")
    prices = generate_price_data()
    fundamentals = generate_fundamental_data()
    print(f"      Prices      : {prices.shape[0]:,} rows  "
          f"({prices['ticker'].nunique()} tickers)")
    print(f"      Fundamentals: {fundamentals.shape[0]:,} rows")

    # ------------------------------------------------------------------
    # 2. Feature construction
    # ------------------------------------------------------------------
    print("\n[2/6] Building features …")
    panel = build_features(prices, fundamentals)
    print(f"      Panel shape : {panel.shape}")

    # ------------------------------------------------------------------
    # 3. Label construction
    # ------------------------------------------------------------------
    print("\n[3/6] Building labels …")
    panel = build_labels(panel)
    print(f"      Horizons    : 1M, 3M, 6M, 12M")
    non_null = panel[TARGET_COL].notna().sum()
    print(f"      Valid target: {non_null:,} rows")

    # ------------------------------------------------------------------
    # 4. Orthogonalization
    # ------------------------------------------------------------------
    print("\n[4/6] Orthogonalizing factors …")
    panel = orthogonalize.compute(panel, FACTOR_NAMES)
    orth_cols = [f"{f}_orth" for f in FACTOR_NAMES]
    print(f"      Orthogonalized columns: {orth_cols}")

    # ------------------------------------------------------------------
    # 5. Factor return series (needed for centrality)
    # ------------------------------------------------------------------
    print("\n[5/6] Computing factor returns …")
    factor_ret_results: dict[str, factor_return.FactorReturnResult] = {}
    factor_ret_series: dict[str, object] = {}
    for f in FACTOR_NAMES:
        res = factor_return.compute(panel, factor=f, return_col=TARGET_COL)
        factor_ret_results[f] = res
        factor_ret_series[f] = res.factor_return_series
    print("      Done.")

    # ------------------------------------------------------------------
    # 6. All diagnostics
    # ------------------------------------------------------------------
    print("\n[6/6] Running diagnostics …\n")

    # -- Centrality (once for all factors) --
    cent_result = centrality.compute(factor_ret_series)

    # -- Shapley (once for all factors) --
    shap_result = shapley.compute(panel, factor_cols=FACTOR_NAMES, target_col=TARGET_COL)

    # -- Per-factor diagnostics --
    for f_col, f_display in DISPLAY_FACTORS.items():
        # IC
        ic_res = ic.compute(panel, factor=f_col, return_col=TARGET_COL)

        # IC Decay
        decay_res = ic_decay.compute(panel, factor=f_col)

        # Trend on IC series
        trend_res = trend.compute(ic_res.ic_series)

        # Gather centrality / shapley for this factor
        cent_score = cent_result.scores.get(f_col, 0.0)
        crowd_label = cent_result.crowding_labels.get(f_col, "unknown")
        shap_val = shap_result.mean_abs_shap.get(f_col, 0.0)

        print_factor_report(
            factor_name=f_display,
            ic_result=ic_res,
            factor_return_result=factor_ret_results[f_col],
            centrality_score=cent_score,
            crowding_label=crowd_label,
            shapley_value=shap_val,
            trend_ic=trend_res,
            ic_decay=decay_res.decay_curve,
        )

    print("=" * 60)
    print("  All 3 × 7 diagnostics complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
