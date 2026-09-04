import numpy as np
import pandas as pd

from data.transformer import compute_zscore, current_zscores


def test_current_zscores_returns_last_fully_observed_month():
    idx = pd.date_range("2000-01-31", periods=3, freq="ME")
    zs = pd.DataFrame([[0.5, -0.5], [1.0, 1.0], [2.0, np.nan]], index=idx, columns=["a", "b"])
    cur = current_zscores(zs)
    assert cur.name == idx[1]
    assert cur["a"] == 1.0 and cur["b"] == 1.0


def test_zscore_is_diff_over_rolling_std_and_winsorised():
    idx = pd.date_range("2000-01-31", periods=200, freq="ME")
    rng = np.random.default_rng(0)
    raw = pd.DataFrame({"x": np.cumsum(rng.normal(size=200))}, index=idx)
    zs = compute_zscore(raw, diff_months=12, zscore_window_yrs=5, winsor_limit=3.0)
    diffs = raw.diff(12)
    expected = (diffs / diffs.rolling(60, min_periods=30).std()).clip(-3, 3)
    pd.testing.assert_frame_equal(zs, expected)
    assert zs["x"].abs().max() <= 3.0
