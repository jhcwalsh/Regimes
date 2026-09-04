"""
Regime-shift indicator per the paper's Exhibit 9:
    at time T, d_{T,t} = distance from T to each earlier month t <= T,
    C_T = EWMA over t of d_{T,t}, beta = 1 - 1/n, most weight on recent t.
"""
import numpy as np
import pandas as pd
import pytest

from engine.regime_shift import compute_regime_shift

COLS = ["a", "b"]


def _zs(rows):
    idx = pd.date_range("2000-01-31", periods=len(rows), freq="ME")
    return pd.DataFrame(rows, index=idx, columns=COLS, dtype=float)


def test_constant_state_gives_zero_indicator():
    zs = _zs([[1.0, -1.0]] * 30)
    out = compute_regime_shift(zs, lookback_years=[1])
    assert (out["ewma_1yr"].abs() < 1e-12).all()
    assert (out["mean_ewma"].abs() < 1e-12).all()


def test_indicator_matches_hand_computed_weighted_average():
    # Three months; state jumps from (0,0) to (3,4) at month 3 (distance 5).
    zs = _zs([[0.0, 0.0], [0.0, 0.0], [3.0, 4.0]])
    n = 12
    beta = 1 - 1 / n
    out = compute_regime_shift(zs, lookback_years=[1])
    # At T=2: distances to t=0,1,2 are 5,5,0 with weights beta^2, beta^1, beta^0.
    expected = (beta**2 * 5 + beta * 5 + 1 * 0) / (beta**2 + beta + 1)
    assert out["ewma_1yr"].iloc[2] == pytest.approx(expected)
    assert out["ewma_1yr"].iloc[0] == pytest.approx(0.0)


def test_step_change_spikes_then_decays():
    rows = [[0.0, 0.0]] * 24 + [[3.0, 0.0]] * 36
    zs = _zs(rows)
    out = compute_regime_shift(zs, lookback_years=[1, 2, 3, 4])
    c = out["mean_ewma"]
    assert c.iloc[23] == pytest.approx(0.0)
    assert c.iloc[24] > 1.0            # jump at the step
    assert c.iloc[59] < c.iloc[24]     # decays as the new state becomes "the past"
    assert set(out.columns) == {"ewma_1yr", "ewma_2yr", "ewma_3yr", "ewma_4yr", "mean_ewma"}


def test_rows_with_missing_variables_are_ignored():
    rows = [[0.0, 0.0]] * 5 + [[np.nan, 0.0]]
    zs = _zs(rows)
    out = compute_regime_shift(zs, lookback_years=[1])
    assert len(out) == 5
    assert out.index[-1] == zs.index[4]
