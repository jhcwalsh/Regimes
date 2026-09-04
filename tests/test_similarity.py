import numpy as np
import pandas as pd
import pytest

from engine.similarity import compute_global_scores, latest_complete_date

COLS = ["a", "b"]


def _zs(rows, start="2000-01-31"):
    idx = pd.date_range(start, periods=len(rows), freq="ME")
    return pd.DataFrame(rows, index=idx, columns=COLS, dtype=float)


def test_global_score_is_euclidean_distance_with_square_root():
    # Paper eq. (1): d = sqrt(sum of squared differences). (3, 4) -> 5.
    zs = _zs([[0.0, 0.0], [3.0, 4.0]])
    scores = compute_global_scores(zs, target_date=zs.index[-1], exclude_recent_months=0)
    assert scores.loc[zs.index[0]] == pytest.approx(5.0)


def test_latest_complete_date_skips_rows_with_missing_variables():
    zs = _zs([[0.0, 0.0], [1.0, 1.0], [2.0, np.nan], [np.nan, np.nan]])
    assert latest_complete_date(zs) == zs.index[1]


def test_default_target_is_last_complete_row_not_last_partial_row():
    # Last row has only one of two variables populated; it must not be the target.
    rows = [[0.0, 0.0], [1.0, 1.0], [5.0, 5.0], [0.1, np.nan]]
    zs = _zs(rows)
    scores = compute_global_scores(zs, exclude_recent_months=0)
    # Target should be row 2 (5, 5): distance from row 0 is sqrt(50), from row 1 is sqrt(32).
    assert scores.loc[zs.index[0]] == pytest.approx(np.sqrt(50))
    assert scores.loc[zs.index[1]] == pytest.approx(np.sqrt(32))
    assert scores.loc[zs.index[2]] == pytest.approx(0.0)
    assert np.isnan(scores.loc[zs.index[3]])


def test_explicit_incomplete_target_date_is_rejected():
    zs = _zs([[0.0, 0.0], [1.0, np.nan]])
    with pytest.raises(ValueError, match="incomplete"):
        compute_global_scores(zs, target_date=zs.index[-1], exclude_recent_months=0)
