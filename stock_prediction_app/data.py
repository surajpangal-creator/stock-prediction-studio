"""Data access helpers for the stock prediction app."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

import pandas as pd
import yfinance as yf


def fetch_market_data(
    ticker: str,
    start: datetime,
    end: datetime,
    interval: Literal["1d", "1wk", "1mo"] = "1d",
) -> pd.DataFrame:
    """Download historical price data for a ticker from Yahoo Finance."""
    data = yf.download(
        ticker,
        start=start,
        end=end,
        interval=interval,
        auto_adjust=True,
        progress=False,
    )

    if data.empty:
        raise ValueError(
            f"No pricing data returned for {ticker} between {start:%Y-%m-%d} "
            f"and {end:%Y-%m-%d}."
        )

    data.index = data.index.tz_localize(None)
    return data


def compute_target_series(
    data: pd.DataFrame, target: Literal["close", "volatility"] = "close"
) -> pd.Series:
    """Return the target series to model."""
    if target == "close":
        series = data["Close"].copy()
        series.name = "price"
        return series

    returns = data["Close"].pct_change()
    volatility = returns.rolling(window=21).std().dropna() * (252**0.5)
    if volatility.empty:
        raise ValueError("Not enough data to compute rolling volatility.")
    volatility.name = "volatility"
    return volatility

