import numpy as np
import pandas as pd

from data.fetcher import last_observed


def test_last_observed_reports_the_last_real_value_per_column_before_filling():
    idx = pd.date_range("2026-06-30", periods=4, freq="ME")
    unfilled = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0], "b": [1.0, 2.0, np.nan, np.nan]}, index=idx)
    lo = last_observed(unfilled)
    assert lo["a"] == idx[3]
    assert lo["b"] == idx[1]
