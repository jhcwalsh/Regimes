"""
Factor timing per the paper (pp. 18-19): at month T, months up to T-36 are ranked by
distance and cut into quantiles; for each factor, go long next month if the average
return in the month after that quantile's months was positive, else short.
"""
import numpy as np
import pandas as pd
import pytest

from engine.factor_timing import direction, performance, quantile_labels, run_backtest


def test_quantile_labels_1_is_most_similar_and_5_least():
    idx = pd.date_range("2000-01-31", periods=10, freq="ME")
    scores = pd.Series(np.arange(10, dtype=float), index=idx)   # 0 = closest
    q = quantile_labels(scores, 5)
    assert q.iloc[0] == 1 and q.iloc[1] == 1
    assert q.iloc[-1] == 5 and q.iloc[-2] == 5
    assert sorted(q.unique()) == [1, 2, 3, 4, 5]


def test_direction_is_sign_of_mean_next_month_return():
    idx = pd.date_range("2000-01-31", periods=6, freq="ME")
    ret = pd.Series([0.0, 2.0, -1.0, -3.0, 5.0, 0.0], index=idx)
    # next-month returns after idx[0], idx[1] are 2 and -1 -> mean +0.5 -> long
    assert direction([idx[0], idx[1]], ret) == 1
    # after idx[2], idx[3]: -3 and 5 -> +1 -> long ; after idx[1], idx[2]: -1 and -3 -> short
    assert direction([idx[1], idx[2]], ret) == -1


def _two_state_world(n_months=240):
    """Z-state alternates in 12-month blocks between A=(+1,+1) and B=(-1,-1); a factor
    earns +1% in the month after an A month and -1% after a B month."""
    idx = pd.date_range("1990-01-31", periods=n_months, freq="ME")
    state = np.array([1.0 if (i // 12) % 2 == 0 else -1.0 for i in range(n_months)])
    zs = pd.DataFrame({"a": state, "b": state}, index=idx)
    ret = pd.Series(np.roll(state, 1), index=idx, name="F")   # return at m = state of m-1
    ret.iloc[0] = 0.0
    return zs, ret.to_frame()


def test_backtest_profits_from_similar_months_and_loses_on_dissimilar_ones():
    zs, factors = _two_state_world()
    out = run_backtest(zs, factors, start="2000-01-31", n_quantiles=5, exclude_recent_months=36)
    r = out["returns"]
    assert (r["q1"] > 0).all()          # similar months predict the sign correctly every month
    assert (r["q5"] < 0).all()          # dissimilar months are the other state: wrong every month
    assert np.allclose(r["spread"], r["q1"] - r["q5"])
    assert (r["long_only"] == factors["F"].reindex(r.index)).all()
    assert r.index[0] > pd.Timestamp("2000-01-31")   # first holding month is after the first decision


def test_backtest_reports_current_positions_per_factor():
    zs, factors = _two_state_world()
    out = run_backtest(zs, factors, start="2000-01-31")
    pos = out["positions"]
    assert set(pos.index) == {"F"} and set(pos.columns) == {"q1", "q5"}
    assert pos.loc["F", "q1"] == -pos.loc["F", "q5"]


def test_performance_table():
    idx = pd.date_range("2000-01-31", periods=24, freq="ME")
    r = pd.DataFrame({"q1": np.tile([1.0, -0.5], 12), "long_only": np.tile([1.0, -0.5], 12)}, index=idx)
    p = performance(r, benchmark="long_only")
    assert p.loc["q1", "sharpe"] == pytest.approx((r["q1"].mean() / r["q1"].std()) * np.sqrt(12))
    assert p.loc["q1", "corr_to_long_only"] == pytest.approx(1.0)
    assert p.loc["q1", "max_drawdown_%"] == pytest.approx(-0.5)
    assert p.loc["q1", "ann_return_%"] == pytest.approx(r["q1"].mean() * 12)
