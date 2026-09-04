"""Tests for the pure computations in data/fetcher.py (no network)."""
import numpy as np
import pandas as pd

from data.fetcher import compute_stock_bond_correlation


def _daily_index(n):
    return pd.bdate_range("2000-01-03", periods=n)


def test_stock_bond_correlation_is_negative_when_stocks_rise_with_yields():
    """
    Bond prices fall when yields rise. If equity returns move one-for-one
    with yield changes, the stock-BOND-RETURN correlation must be -1, not +1.
    """
    idx = _daily_index(400)
    rng = np.random.default_rng(0)
    d_yield = rng.normal(0, 0.05, len(idx))
    yields = pd.Series(4.0 + np.cumsum(d_yield), index=idx)
    # Equity return equals the yield change each day (perfectly positive).
    eq_ret = pd.Series(d_yield, index=idx)
    eq_prices = 100 * (1 + eq_ret).cumprod()

    corr = compute_stock_bond_correlation(eq_prices, yields, window_days=250)

    last = corr.dropna().iloc[-1]
    assert last < -0.99, f"expected ~-1, got {last}"


def test_stock_bond_correlation_is_monthly_month_end():
    idx = _daily_index(400)
    rng = np.random.default_rng(1)
    yields = pd.Series(4.0 + np.cumsum(rng.normal(0, 0.05, len(idx))), index=idx)
    eq_prices = pd.Series(100 * np.cumprod(1 + rng.normal(0, 0.01, len(idx))), index=idx)

    corr = compute_stock_bond_correlation(eq_prices, yields, window_days=250)

    assert corr.name == "stock_bond_corr"
    assert all(corr.index == corr.index + pd.offsets.MonthEnd(0))


def test_fred_series_load_from_cache_without_api_key(tmp_path, monkeypatch):
    """A cached deployment must not need a FRED key just to read its own cache."""
    import data.fetcher as fetcher
    from config import FRED_SERIES

    idx = pd.date_range("2020-01-31", periods=3, freq="ME")
    for name in FRED_SERIES:
        pd.Series([1.0, 2.0, 3.0], index=idx, name=name).to_frame().to_parquet(tmp_path / f"{name}.parquet")
    monkeypatch.setattr(fetcher, "CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(fetcher, "FRED_API_KEY", "")
    monkeypatch.delenv("FRED_API_KEY", raising=False)

    df = fetcher.fetch_fred_series()

    assert list(df.columns) == list(FRED_SERIES)
    assert len(df) == 3
