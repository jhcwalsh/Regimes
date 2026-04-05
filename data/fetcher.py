"""
Data fetcher: pulls all seven state variables from FRED and yfinance.

Variables (per Mulliner et al. 2026):
  1. S&P 500 log price                  → yfinance ^GSPC (monthly close)
  2. Yield curve (10yr - 3m T-bill)     → FRED GS10 - TB3MS
  3. WTI crude oil price                → FRED DCOILWTICO
  4. Copper price                       → FRED PCOPPUSDM
  5. US 3-month T-bill yield            → FRED TB3MS
  6. VIX / realized volatility          → FRED VIXCLS (1990+); realised vol pre-1990
  7. Rolling 3-yr stock-bond correlation → computed from daily ^GSPC + ^TNX
"""

import os
import numpy as np
import pandas as pd
import yfinance as yf
from fredapi import Fred

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import (
    FRED_API_KEY, FRED_SERIES, SP500_TICKER, BOND_TICKER,
    CORR_LOOKBACK_YRS, CACHE_DIR
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _fred() -> Fred:
    if not FRED_API_KEY:
        raise ValueError(
            "FRED_API_KEY is not set. "
            "Get a free key at https://fred.stlouisfed.org/docs/api/api_key.html "
            "and set it as an environment variable: export FRED_API_KEY=your_key"
        )
    return Fred(api_key=FRED_API_KEY)


def _to_monthly_end(series: pd.Series) -> pd.Series:
    """Resample any series to month-end frequency, forward-filling gaps."""
    series = series.copy()
    series.index = pd.to_datetime(series.index)
    return series.resample("ME").last().ffill()


def _cache_path(name: str) -> str:
    os.makedirs(CACHE_DIR, exist_ok=True)
    return os.path.join(CACHE_DIR, f"{name}.parquet")


def _load_cache(name: str) -> pd.Series | None:
    path = _cache_path(name)
    if os.path.exists(path):
        df = pd.read_parquet(path)
        return df.iloc[:, 0]
    return None


def _save_cache(name: str, series: pd.Series) -> None:
    series.to_frame().to_parquet(_cache_path(name))


# ---------------------------------------------------------------------------
# Individual variable fetchers
# ---------------------------------------------------------------------------

def fetch_sp500_monthly(start: str = "1920-01-01") -> pd.Series:
    """S&P 500 monthly close price (log scale used in transformation)."""
    cached = _load_cache("sp500")
    if cached is not None:
        return cached

    df = yf.download(SP500_TICKER, start=start, interval="1mo", auto_adjust=True, progress=False)
    s = df["Close"].squeeze()
    s.index = pd.to_datetime(s.index).to_period("M").to_timestamp("M")
    s.name = "sp500"
    _save_cache("sp500", s)
    return s


def fetch_fred_series(start: str = "1920-01-01") -> pd.DataFrame:
    """Fetch all FRED series and return as a monthly DataFrame."""
    fred = _fred()
    frames = {}
    for name, series_id in FRED_SERIES.items():
        cached = _load_cache(name)
        if cached is not None:
            frames[name] = cached
        else:
            raw = fred.get_series(series_id, observation_start=start)
            raw.name = name
            monthly = _to_monthly_end(raw)
            _save_cache(name, monthly)
            frames[name] = monthly

    return pd.DataFrame(frames)


def fetch_stock_bond_correlation(start: str = "1960-01-01") -> pd.Series:
    """
    Rolling 3-year stock-bond correlation computed from daily returns.
    Uses ^GSPC (equity) and ^TNX (10-yr yield, inverted for bond returns).
    Pre-1962 data is not available from yfinance, so series starts ~1962.
    """
    cached = _load_cache("stock_bond_corr")
    if cached is not None:
        return cached

    window_days = int(CORR_LOOKBACK_YRS * 252)

    eq = yf.download(SP500_TICKER, start=start, interval="1d", auto_adjust=True, progress=False)["Close"].squeeze()
    bd = yf.download(BOND_TICKER,  start=start, interval="1d", auto_adjust=True, progress=False)["Close"].squeeze()

    eq_ret  = eq.pct_change().dropna()
    # Bond return ≈ negative change in yield (duration proxy, sign flip)
    bd_ret  = (-bd).pct_change().dropna()

    combined = pd.concat([eq_ret, bd_ret], axis=1).dropna()
    combined.columns = ["equity", "bond"]

    rolling_corr = combined["equity"].rolling(window=window_days).corr(combined["bond"])

    # Collapse to month-end
    monthly = rolling_corr.resample("ME").last().ffill()
    monthly.name = "stock_bond_corr"

    _save_cache("stock_bond_corr", monthly)
    return monthly


def fetch_realized_volatility_monthly(start: str = "1920-01-01") -> pd.Series:
    """
    Monthly realised volatility of S&P 500 from daily returns.
    Used to prepend VIX history before 1990 (annualised, %).
    """
    cached = _load_cache("realized_vol")
    if cached is not None:
        return cached

    df = yf.download(SP500_TICKER, start=start, interval="1d", auto_adjust=True, progress=False)
    daily_ret = df["Close"].squeeze().pct_change().dropna()
    # Annualised realised vol (%)
    monthly_vol = daily_ret.resample("ME").std() * np.sqrt(252) * 100
    monthly_vol.name = "realized_vol"
    _save_cache("realized_vol", monthly_vol)
    return monthly_vol


def build_vix_series() -> pd.Series:
    """
    Splice realised volatility (pre-1990) with VIX (1990+).
    Paper: 'VIX prepended with realized volatility before 1990'.
    """
    cached = _load_cache("vix_spliced")
    if cached is not None:
        return cached

    realised  = fetch_realized_volatility_monthly()
    fred_vix  = fetch_fred_series()["vix"] if "vix" in fetch_fred_series().columns else None

    if fred_vix is None:
        fred = _fred()
        raw = fred.get_series("VIXCLS", observation_start="1985-01-01")
        fred_vix = _to_monthly_end(raw)
        fred_vix.name = "vix"

    cutoff = pd.Timestamp("1990-01-01")
    pre  = realised[realised.index < cutoff]
    post = fred_vix[fred_vix.index >= cutoff]

    spliced = pd.concat([pre, post]).sort_index()
    spliced.name = "volatility"
    _save_cache("vix_spliced", spliced)
    return spliced


# ---------------------------------------------------------------------------
# Master assembly
# ---------------------------------------------------------------------------

def fetch_all(start: str = "1920-01-01", refresh_cache: bool = False) -> pd.DataFrame:
    """
    Fetch and assemble all seven state variables into a single monthly DataFrame.

    Columns:
        sp500            – S&P 500 log price
        yield_curve      – 10yr yield minus 3m T-bill (%)
        oil              – WTI crude oil price
        copper           – Copper price
        tbill_3m         – US 3-month T-bill yield
        volatility       – VIX / spliced realised vol
        stock_bond_corr  – Rolling 3-yr stock-bond correlation
    """
    if refresh_cache:
        for f in os.listdir(CACHE_DIR):
            if f.endswith(".parquet"):
                os.remove(os.path.join(CACHE_DIR, f))

    fred_data    = fetch_fred_series(start)
    sp500        = fetch_sp500_monthly(start)
    vol          = build_vix_series()
    sb_corr      = fetch_stock_bond_correlation(start)

    # Yield curve = 10yr minus 3m
    yield_curve  = fred_data["yield_10yr"] - fred_data["tbill_3m"]
    yield_curve.name = "yield_curve"

    # Log S&P 500
    log_sp500 = np.log(sp500)
    log_sp500.name = "sp500"

    df = pd.concat([
        log_sp500,
        yield_curve,
        fred_data["oil"],
        fred_data["copper"],
        fred_data["tbill_3m"],
        vol,
        sb_corr,
    ], axis=1)

    df.columns = ["sp500", "yield_curve", "oil", "copper", "tbill_3m", "volatility", "stock_bond_corr"]
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    return df


if __name__ == "__main__":
    print("Fetching all state variables...")
    data = fetch_all()
    print(data.tail(12).to_string())
    print(f"\nDate range: {data.index[0].date()} to {data.index[-1].date()}")
    print(f"Rows: {len(data)},  Columns: {list(data.columns)}")
