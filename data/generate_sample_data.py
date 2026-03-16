"""
Synthetic data generator for the Factor Analysis Platform.

Produces:
  - Daily price data  : (date, ticker, close, volume, market_cap)
  - Quarterly fundamentals : (ticker, report_date, revenue)

~50 tickers over 5 years of daily prices.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
NUM_TICKERS = 50
START_DATE = "2018-01-02"
END_DATE = "2022-12-30"
SEED = 42


def _trading_days(start: str, end: str) -> pd.DatetimeIndex:
    """Generate weekday-only date range (proxy for trading days)."""
    return pd.bdate_range(start, end)


# ---------------------------------------------------------------------------
# Price data
# ---------------------------------------------------------------------------

def generate_price_data(
    num_tickers: int = NUM_TICKERS,
    start: str = START_DATE,
    end: str = END_DATE,
    seed: int = SEED,
) -> pd.DataFrame:
    """Return a DataFrame with columns [date, ticker, close, volume, market_cap]."""
    rng = np.random.default_rng(seed)
    dates = _trading_days(start, end)
    tickers = [f"STOCK_{i:03d}" for i in range(num_tickers)]

    rows: list[dict] = []
    for ticker in tickers:
        # random starting price and shares outstanding
        price = rng.uniform(20, 200)
        shares_out = rng.uniform(1e7, 5e8)
        drift = rng.uniform(-0.0001, 0.0003)
        vol = rng.uniform(0.01, 0.03)

        for dt in dates:
            ret = drift + vol * rng.standard_normal()
            price *= np.exp(ret)
            volume = int(rng.uniform(1e5, 5e6))
            mktcap = price * shares_out
            rows.append(
                {
                    "date": dt,
                    "ticker": ticker,
                    "close": round(price, 2),
                    "volume": volume,
                    "market_cap": round(mktcap, 2),
                }
            )

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values(["date", "ticker"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Fundamental data
# ---------------------------------------------------------------------------

def generate_fundamental_data(
    num_tickers: int = NUM_TICKERS,
    start: str = START_DATE,
    end: str = END_DATE,
    seed: int = SEED,
) -> pd.DataFrame:
    """Return a DataFrame with columns [ticker, report_date, revenue].

    report_date is the *publication* date (typically 1-2 months after quarter-end)
    to avoid look-ahead bias.
    """
    rng = np.random.default_rng(seed + 1)
    tickers = [f"STOCK_{i:03d}" for i in range(num_tickers)]

    # quarter-end dates within range
    quarter_ends = pd.date_range(start, end, freq="Q")

    rows: list[dict] = []
    for ticker in tickers:
        base_rev = rng.uniform(1e8, 5e9)
        growth = rng.uniform(-0.02, 0.06)

        for i, qe in enumerate(quarter_ends):
            # revenue with trend + noise
            rev = base_rev * (1 + growth) ** i * (1 + 0.05 * rng.standard_normal())
            # report_date = quarter-end + 30-60 day lag
            lag_days = int(rng.uniform(30, 60))
            report_date = qe + pd.Timedelta(days=lag_days)
            rows.append(
                {
                    "ticker": ticker,
                    "report_date": report_date,
                    "revenue": round(rev, 2),
                }
            )

    df = pd.DataFrame(rows)
    df["report_date"] = pd.to_datetime(df["report_date"])
    return df.sort_values(["ticker", "report_date"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# CLI helper
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    prices = generate_price_data()
    fundas = generate_fundamental_data()
    print(f"Price data  : {prices.shape}")
    print(prices.head())
    print(f"\nFundamental data : {fundas.shape}")
    print(fundas.head())
