"""
Structured per-factor report formatter.

Formats all diagnostic results into a readable console report.
"""

from __future__ import annotations

from typing import Any


def _section(title: str, width: int = 60) -> str:
    return f"\n{'-' * width}\n  {title}\n{'-' * width}"


def _kv(key: str, value: Any, indent: int = 4) -> str:
    pad = " " * indent
    return f"{pad}{key:<28s} {value}"


def print_factor_report(
    factor_name: str,
    ic_result: Any,
    factor_return_result: Any,
    centrality_score: float,
    crowding_label: str,
    shapley_value: float,
    trend_ic: Any,
    ic_decay: dict[str, float],
) -> None:
    """Print a structured report for a single factor."""

    print(_section(f"Factor: {factor_name}"))

    # IC
    print("\n  > Information Coefficient")
    print(_kv("Mean IC", f"{ic_result.mean_ic:.4f}"))
    print(_kv("IC IR", f"{ic_result.ic_ir:.4f}"))

    # Factor Return
    print("\n  > Factor Return")
    last_cum = (
        factor_return_result.cumulative_return.iloc[-1]
        if len(factor_return_result.cumulative_return) > 0
        else 0.0
    )
    print(_kv("Cumulative Return", f"{last_cum:.4f}"))
    print(_kv("Sharpe Ratio", f"{factor_return_result.sharpe_ratio:.4f}"))

    # Centrality
    print("\n  > Centrality (Crowding)")
    print(_kv("Centrality Score", f"{centrality_score:.4f}"))
    print(_kv("Crowding Level", crowding_label))

    # Shapley
    print("\n  > Shapley Contribution")
    print(_kv("Mean |SHAP|", f"{shapley_value:.6f}"))

    # Trend
    print("\n  > Trend Detection (IC)")
    print(_kv("Slope", f"{trend_ic.slope:.6f}"))
    print(_kv("p-value", f"{trend_ic.p_value:.4f}"))
    print(_kv("R2", f"{trend_ic.r_squared:.4f}"))
    print(_kv("Label", trend_ic.trend_label))

    # IC Decay
    print("\n  > IC Decay Curve")
    for horizon, ic_val in ic_decay.items():
        print(_kv(f"IC_{horizon}", f"{ic_val:.4f}"))

    print()
