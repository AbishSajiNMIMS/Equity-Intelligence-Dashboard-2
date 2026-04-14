"""
Data-access helpers for the dashboard.
This module downloads prices from yfinance, converts them into
clean tables, and engineers the features used by the ML models.
"""

import numpy as np
import pandas as pd
import yfinance as yf

# -----------------------------------------------------------------------------
# Sector map:
# the sidebar uses this dictionary to populate stock choices by sector.
# -----------------------------------------------------------------------------
SECTORS = {
    "Banking": ["HDFCBANK.NS", "ICICIBANK.NS", "KOTAKBANK.NS", "AXISBANK.NS", "SBIN.NS"],
    "IT": ["TCS.NS", "INFY.NS", "WIPRO.NS", "HCLTECH.NS", "TECHM.NS"],
    "FMCG": ["HINDUNILVR.NS", "NESTLEIND.NS", "BRITANNIA.NS", "DABUR.NS", "MARICO.NS"],
    "Pharma": ["SUNPHARMA.NS", "DRREDDY.NS", "CIPLA.NS", "DIVISLAB.NS", "BIOCON.NS"],
    "Energy": ["RELIANCE.NS", "ONGC.NS", "NTPC.NS", "POWERGRID.NS", "BPCL.NS"],
}


# -----------------------------------------------------------------------------
# Download and extraction helpers:
# yfinance sometimes returns flat columns and sometimes MultiIndex columns,
# so these helpers normalize both cases into one simple dataframe shape.
# -----------------------------------------------------------------------------
def _download(tickers, period="2y"):
    return yf.download(tickers, period=period, auto_adjust=True, progress=False)


def _extract_field(raw: pd.DataFrame, field: str, labels=None) -> pd.DataFrame:
    if isinstance(raw.columns, pd.MultiIndex):
        levels = [raw.columns.get_level_values(i).map(str).str.lower() for i in range(2)]
        if field.lower() in set(levels[0]):
            data = raw.xs(field, axis=1, level=0)
        elif field.lower() in set(levels[1]):
            data = raw.xs(field, axis=1, level=1)
        else:
            raise ValueError(f"Cannot find {field} in downloaded data")
    else:
        column = next((col for col in raw.columns if str(col).lower() == field.lower()), None)
        if column is None:
            raise ValueError(f"Cannot find {field} in downloaded data")
        data = raw[column]

    if isinstance(data, pd.Series):
        data = data.to_frame()
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(-1)
    if labels and len(data.columns) == len(labels):
        data.columns = labels
    return data


# -----------------------------------------------------------------------------
# Basic market-data helpers:
# these functions keep price downloads, return calculations,
# and stock metadata lookups out of the main Streamlit file.
# -----------------------------------------------------------------------------
def get_price_data(tickers: list, period: str = "2y") -> pd.DataFrame:
    return _extract_field(_download(tickers, period), "Close", tickers).dropna(how="all")


def get_daily_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return prices.pct_change().dropna()


def get_sector_returns(period: str = "2y") -> dict:
    return {
        sector: get_daily_returns(get_price_data(tickers, period)).mean(axis=1)
        for sector, tickers in SECTORS.items()
    }


def get_stock_info(ticker: str) -> dict:
    try:
        return {"name": yf.Ticker(ticker).info.get("longName", ticker)}
    except Exception:
        return {"name": ticker}


# -----------------------------------------------------------------------------
# Feature engineering:
# this section builds lag, rolling, momentum, and distance-to-average features
# that feed the logistic, ridge, and lasso models.
# -----------------------------------------------------------------------------
def engineer_features(ticker: str, period: str = "2y") -> pd.DataFrame:
    raw = _download(ticker, period)
    df = pd.DataFrame(
        {
            "close": _extract_field(raw, "Close").squeeze(),
            "volume": _extract_field(raw, "Volume").squeeze(),
        },
        index=raw.index,
    )

    df["return"] = df["close"].pct_change()
    for lag in (1, 2, 3, 5):
        df[f"lag_{lag}"] = df["return"].shift(lag)

    for window in (5, 20):
        df[f"rolling_mean_{window}"] = df["return"].rolling(window).mean()
        df[f"rolling_std_{window}"] = df["return"].rolling(window).std()

    delta = df["close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss
    df["rsi"] = 100 - (100 / (1 + rs))

    df["volume_ratio"] = df["volume"] / df["volume"].rolling(20).mean()
    for window in (20, 50):
        ma = df["close"].rolling(window).mean()
        df[f"dist_ma{window}"] = (df["close"] - ma) / ma

    df["target"] = (df["return"].shift(-1) > 0).astype(int)
    return df.dropna()
