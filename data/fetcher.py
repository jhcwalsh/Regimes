"""
Data fetcher: pulls all seven state variables from FRED and yfinance.

Variables (per Mulliner et al. 2026):
  1. S&P 500 log price                  → yfinance ^GSPC daily (1927-), month-end close
  2. Yield curve (10yr - 3m T-bill)     → FRED GS10 - TB3MS
  3. WTI crude oil price                → FRED WTISPLC (monthly spot, 1946-)
  4. Copper price                       → World Bank Pink Sheet (1960-) spliced with FRED PCOPPUSDM
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
    CORR_LOOKBACK_YRS, CACHE_DIR,
    COPPER_FRED_ID, COPPER_WB_URL, COPPER_WB_COLUMN,
)

# Bump when a series' source changes so stale cache files are ignored, not reused.
CACHE_VERSION = "v2"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _fred() -> Fred:
    # Re-read from environment each call so sidebar input takes effect immediately
    key = os.environ.get("FRED_API_KEY", "") or FRED_API_KEY
    if not key:
        raise ValueError(
            "FRED_API_KEY is not set. "
            "Get a free key at https://fred.stlouisfed.org/docs/api/api_key.html "
            "and set it as an environment variable: export FRED_API_KEY=your_key"
        )
    return Fred(api_key=key)


def _to_month_period(series: pd.Series) -> pd.Series:
    """
    Normalize any series to a monthly PeriodIndex (YYYY-MM), then convert
    to timestamp at month-end.  Forward-fill to cover reporting lags.
    All series use this so that pd.concat aligns correctly.
    """
    series = series.copy()
    series.index = pd.to_datetime(series.index).to_period("M").to_timestamp("M")
    # Collapse multiple observations in same month to last value
    series = series.groupby(series.index).last()
    return series.ffill()


# Keep old name as alias so cached-parquet reads still work
def _to_monthly_end(series: pd.Series) -> pd.Series:
    return _to_month_period(series)


def _cache_path(name: str) -> str:
    os.makedirs(CACHE_DIR, exist_ok=True)
    return os.path.join(CACHE_DIR, f"{name}.{CACHE_VERSION}.parquet")


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

def monthly_close_from_daily(daily: pd.Series) -> pd.Series:
    """Last observed value in each month, indexed at month-end."""
    return daily.resample("ME").last().dropna()


def parse_pink_sheet(sheet: pd.DataFrame, column: str) -> pd.Series:
    """
    Extract one commodity from the World Bank Pink Sheet 'Monthly Prices' tab read
    with header=None: a name row, a units row, then rows keyed like '1960M01'.
    """
    header_row, col = None, None
    for r in range(min(10, len(sheet))):
        row = sheet.iloc[r].astype(str).str.strip().str.lower()
        hits = np.flatnonzero((row == column.lower()).values)
        if len(hits):
            header_row, col = r, int(hits[0])
            break
    if header_row is None:
        raise KeyError(f"column {column!r} not found in pink sheet")
    body = sheet.iloc[header_row + 2:]
    values = pd.to_numeric(body.iloc[:, col], errors="coerce")
    keys = body.iloc[:, 0].astype(str).str.strip()
    ok = values.notna() & keys.str.match(r"^\d{4}M\d{2}$")
    idx = pd.PeriodIndex(keys[ok].str.replace("M", "-", regex=False), freq="M").to_timestamp("M")
    return pd.Series(values[ok].values, index=idx, name=column.lower())


def splice_levels(old: pd.Series, new: pd.Series) -> pd.Series:
    """
    Join a long historical series to a live one: `new` wins where it exists; `old`
    is rescaled by the mean new/old ratio over the overlap so levels match.
    """
    both = pd.concat([old, new], axis=1).dropna()
    if both.empty:
        raise ValueError("splice_levels: the two series do not overlap")
    ratio = float((both.iloc[:, 1] / both.iloc[:, 0]).mean())
    return new.combine_first(old * ratio).sort_index()


def _sp500_daily(start: str = "1920-01-01") -> pd.Series:
    """Daily ^GSPC close from 1927, cached; feeds the S&P level, realised vol and the correlation."""
    cached = _load_cache("sp500_daily")
    if cached is not None:
        return cached
    s = yf.download(SP500_TICKER, start=start, interval="1d", auto_adjust=True, progress=False)["Close"].squeeze()
    s.index = pd.to_datetime(s.index)
    s.name = "sp500_daily"
    _save_cache("sp500_daily", s)
    return s


def fetch_sp500_monthly(start: str = "1920-01-01") -> pd.Series:
    """S&P 500 month-end close derived from the daily series (log scale used in transformation)."""
    cached = _load_cache("sp500")
    if cached is not None:
        return _to_month_period(cached)
    s = _to_month_period(monthly_close_from_daily(_sp500_daily(start)))
    s.name = "sp500"
    _save_cache("sp500", s)
    return s


def fetch_copper_monthly() -> pd.Series:
    """Copper, $/mt, 1960-: World Bank Pink Sheet history spliced with FRED's live series."""
    cached = _load_cache("copper")
    if cached is not None:
        return _to_month_period(cached)

    import io
    import urllib.request
    with urllib.request.urlopen(COPPER_WB_URL, timeout=60) as resp:
        raw = resp.read()
    sheet = pd.read_excel(io.BytesIO(raw), sheet_name="Monthly Prices", header=None)
    wb = parse_pink_sheet(sheet, COPPER_WB_COLUMN)

    live = _to_month_period(_fred().get_series(COPPER_FRED_ID, observation_start="1960-01-01"))

    copper = _to_month_period(splice_levels(wb, live))
    copper.name = "copper"
    _save_cache("copper", copper)
    return copper


def fetch_fred_series(start: str = "1920-01-01") -> pd.DataFrame:
    """Fetch all FRED series and return as a monthly DataFrame."""
    fred = None  # created only if a series is missing from the cache
    frames = {}
    for name, series_id in FRED_SERIES.items():
        cached = _load_cache(name)
        if cached is not None:
            frames[name] = _to_month_period(cached)
        else:
            fred = fred or _fred()
            raw = fred.get_series(series_id, observation_start=start)
            raw.name = name
            monthly = _to_month_period(raw)
            _save_cache(name, monthly)
            frames[name] = monthly

    return pd.DataFrame(frames)


def compute_stock_bond_correlation(
    eq_prices: pd.Series,
    yields: pd.Series,
    window_days: int,
) -> pd.Series:
    """
    Rolling stock-bond correlation from daily equity prices and daily 10-yr yields.

    Bond return is proxied by the negative of the daily yield change (a bond's
    price falls when its yield rises), so the correlation carries the sign of
    the stock-bond *return* correlation. Collapsed to month-end.
    """
    eq_ret = eq_prices.pct_change()
    bd_ret = -yields.diff()

    combined = pd.concat([eq_ret, bd_ret], axis=1).dropna()
    combined.columns = ["equity", "bond"]

    rolling_corr = combined["equity"].rolling(window=window_days).corr(combined["bond"])

    # Collapse to month-end with the normalised index used by every series
    monthly = _to_month_period(rolling_corr.resample("ME").last())
    monthly.name = "stock_bond_corr"
    return monthly


def fetch_stock_bond_correlation(start: str = "1960-01-01") -> pd.Series:
    """
    Rolling 3-year stock-bond correlation computed from daily returns.
    Uses ^GSPC (equity) and ^TNX (10-yr yield; negative yield change proxies bond return).
    Pre-1962 data is not available from yfinance, so series starts ~1962.
    """
    cached = _load_cache("stock_bond_corr")
    if cached is not None:
        return cached

    window_days = int(CORR_LOOKBACK_YRS * 252)

    eq = _sp500_daily()
    bd = yf.download(BOND_TICKER,  start=start, interval="1d", auto_adjust=True, progress=False)["Close"].squeeze()

    monthly = compute_stock_bond_correlation(eq, bd, window_days)
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

    daily_ret = _sp500_daily(start).pct_change().dropna()
    # Annualised realised vol (%)
    monthly_vol = daily_ret.resample("ME").std() * np.sqrt(252) * 100
    monthly_vol = _to_month_period(monthly_vol)
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

    # Normalise both to the same month-end index before splicing
    realised  = _to_month_period(realised)
    fred_vix  = _to_month_period(fred_vix)

    cutoff = pd.Timestamp("1990-01-31")
    pre  = realised[realised.index <= cutoff]
    post = fred_vix[fred_vix.index >  cutoff]

    spliced = pd.concat([pre, post]).sort_index()
    spliced = spliced[~spliced.index.duplicated(keep="last")]
    spliced.name = "volatility"
    _save_cache("vix_spliced", spliced)
    return spliced


# ---------------------------------------------------------------------------
# Master assembly
# ---------------------------------------------------------------------------

def last_observed(unfilled: pd.DataFrame) -> pd.Series:
    """Last month with a real (not forward-filled) observation, per column."""
    return unfilled.apply(lambda c: c.last_valid_index())


def fetch_all(start: str = "1920-01-01", refresh_cache: bool = False, fill: bool = True) -> pd.DataFrame:
    """
    Fetch and assemble all seven state variables into a single monthly DataFrame.

    Columns:
        sp500            – S&P 500 log price
        yield_curve      – 10yr yield minus 3m T-bill (%)
        oil              – WTI spot price (monthly, 1946-)
        copper           – Copper price, $/mt (1960-)
        tbill_3m         – US 3-month T-bill yield
        volatility       – VIX / spliced realised vol
        stock_bond_corr  – Rolling 3-yr stock-bond correlation

    fill=True forward-fills lagging series to the latest month.
    """
    if refresh_cache:
        for f in os.listdir(CACHE_DIR):
            if f.endswith(".parquet"):
                os.remove(os.path.join(CACHE_DIR, f))

    fred_data    = fetch_fred_series(start)
    sp500        = fetch_sp500_monthly(start)
    copper       = fetch_copper_monthly()
    vol          = build_vix_series()
    sb_corr      = fetch_stock_bond_correlation(start)

    # Yield curve = 10yr minus 3m
    yield_curve  = fred_data["yield_10yr"] - fred_data["tbill_3m"]
    yield_curve.name = "yield_curve"

    # Log S&P 500
    log_sp500 = np.log(sp500)
    log_sp500.name = "sp500"

    # Ensure every series has a normalised month-end index before concat
    series_list = [log_sp500, yield_curve,
                   fred_data["oil"], copper, fred_data["tbill_3m"],
                   vol, sb_corr]
    series_list = [_to_month_period(s) for s in series_list]

    df = pd.concat(series_list, axis=1)
    df.columns = ["sp500", "yield_curve", "oil", "copper", "tbill_3m", "volatility", "stock_bond_corr"]
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    # Forward-fill to cover FRED reporting lags (e.g. oil/copper released ~1m late).
    # Callers that need to know what was filled pass fill=False and use last_observed().
    if fill:
        df = df.ffill()

    return df


if __name__ == "__main__":
    print("Fetching all state variables...")
    data = fetch_all()
    print(data.tail(12).to_string())
    print(f"\nDate range: {data.index[0].date()} to {data.index[-1].date()}")
    print(f"Rows: {len(data)},  Columns: {list(data.columns)}")
