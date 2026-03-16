# Factor Analysis Platform

A minimal quant research framework for US equity factor analysis. Evaluates factors using **5 core diagnostics** plus orthogonalization and IC decay analysis.

---

## Diagnostics (3 Factors × 7 Diagnostics)

| Diagnostic | Description |
|---|---|
| **Factor Return** | Long-short portfolio (top/bottom 20%), Sharpe ratio |
| **Centrality** | PCA-based crowding score on factor return correlations |
| **Shapley Contribution** | SHAP values via XGBoost for marginal factor attribution |
| **Information Coefficient** | Spearman rank correlation with forward returns |
| **Trend Detection** | OLS regression on IC/centrality/SHAP time series |
| **Orthogonalization** | Regression residualization to remove factor collinearity |
| **IC Decay Curve** | IC at 1M/3M/6M/12M horizons to find optimal holding period |

### Factors

- **Size** — `log(market_cap)`
- **Revenue Momentum** — YoY revenue growth (`revenue_t / revenue_{t-4} - 1`)
- **Price Momentum** — 12-month return skipping the most recent month

---

## Quick Start

```bash
pip install -r requirements.txt
python main.py
```

This generates synthetic data (50 tickers × 5 years) and runs the full diagnostic suite.

---

## Project Structure

```
factor-platform/
├── main.py                     # End-to-end runner
├── report.py                   # Structured console report formatter
├── requirements.txt
│
├── data/
│   └── generate_sample_data.py # Synthetic price + fundamental data
│
├── factors/
│   ├── size.py                 # log(market_cap)
│   ├── revenue_momentum.py     # YoY revenue growth
│   └── price_momentum.py       # 12-month skipping momentum
│
├── pipeline/
│   ├── build_features.py       # Resample, align, compute, rank
│   └── build_labels.py         # Forward excess returns (1M/3M/6M/12M)
│
└── analysis/
    ├── factor_return.py        # Long-short portfolio returns + Sharpe
    ├── centrality.py           # PCA crowding scores
    ├── shapley.py              # SHAP via XGBoost
    ├── ic.py                   # Spearman IC + IC IR
    ├── ic_decay.py             # Multi-horizon IC curve
    ├── orthogonalize.py        # Regression residualization
    └── trend.py                # OLS trend detection
```

---

## Research Pipeline

```
Raw Data → Factor Construction → Orthogonalization → Prediction Target → Analysis
                                                                          ├── Factor Return
                                                                          ├── Centrality
                                                                          ├── Shapley
                                                                          ├── IC
                                                                          └── Trend + IC Decay
```

---

## Sample Output

```
------------------------------------------------------------
  Factor: Price Momentum
------------------------------------------------------------

  > Information Coefficient
    Mean IC                      0.0113
    IC IR                        0.0880

  > Factor Return
    Cumulative Return            0.1171
    Sharpe Ratio                 0.2422

  > Centrality (Crowding)
    Centrality Score             0.3361
    Crowding Level               medium

  > Shapley Contribution
    Mean |SHAP|                  0.027864

  > Trend Detection (IC)
    Slope                        -0.001790
    p-value                      0.2280
    R2                           0.0336
    Label                        uncertain

  > IC Decay Curve
    IC_1M                        -0.0172
    IC_3M                        0.0113
    IC_6M                        -0.0094
    IC_12M                       0.0098
```

---

## Using Real Data

Replace the synthetic generators in `main.py` with your own data loaders. Required schemas:

**Price data** (daily): `date`, `ticker`, `close`, `volume`, `market_cap`

**Fundamental data** (quarterly): `ticker`, `report_date`, `revenue`

> **Important:** Fundamentals are aligned using `report_date` (publication date) to prevent look-ahead bias.

---

## Dependencies

- pandas, numpy, scipy
- scikit-learn, xgboost, shap
- statsmodels

---

## License

MIT
