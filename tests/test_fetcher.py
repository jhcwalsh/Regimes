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
