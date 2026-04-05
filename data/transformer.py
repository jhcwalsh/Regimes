"""
Transformation pipeline for the seven economic state variables.

Per Mulliner et al. (2026):
  1. Compute 12-month difference for each variable
  2. Divide by rolling 10-year standard deviation of those differences (Z-score)
  3. Winsorize to [-3, +3]

Result: a DataFrame of Z-scores aligned to the same monthly index.
"""

import numpy as np
import pandas as pd

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import DIFF_MONTHS, ZSCORE_WINDOW_YRS, WINSOR_LIMIT


def compute_zscore(
    raw: pd.DataFrame,
    diff_months: int = DIFF_MONTHS,
    zscore_window_yrs: int = ZSCORE_WINDOW_YRS,
    winsor_limit: float = WINSOR_LIMIT,
) -> pd.DataFrame:
    """
    Transform raw state variables into winsorised Z-scores.

    Parameters
    ----------
    raw : pd.DataFrame
        Monthly raw values, columns = variable names.
    diff_months : int
        Number of months for the differencing step (default 12).
    zscore_window_yrs : int
        Rolling window in years for the std denominator (default 10).
    winsor_limit : float
        Clip Z-scores to [-winsor_limit, +winsor_limit] (default 3.0).

    Returns
    -------
    pd.DataFrame
        Transformed Z-scores, same shape as input (NaN where insufficient history).
    """
    window_months = zscore_window_yrs * 12

    # Step 1: 12-month difference
    diffs = raw.diff(diff_months)

    # Step 2: Rolling std of those differences over 10 years
    rolling_std = diffs.rolling(window=window_months, min_periods=window_months // 2).std()

    # Step 3: Z-score
    zscores = diffs / rolling_std

    # Step 4: Winsorise
    zscores = zscores.clip(lower=-winsor_limit, upper=winsor_limit)

    return zscores


def describe_transformed(zscores: pd.DataFrame) -> pd.DataFrame:
    """
    Replicates Exhibit 4 style summary: mean, std, and autocorrelations.
    """
    stats = pd.DataFrame(index=zscores.columns)
    stats["mean"] = zscores.mean()
    stats["std"]  = zscores.std()
    for lag in [1, 3, 12, 36, 120]:
        stats[f"ac_{lag}m"] = zscores.apply(lambda s: s.autocorr(lag=lag))
    return stats.round(2)


def current_zscores(zscores: pd.DataFrame) -> pd.Series:
    """Return the most recent (current) row of Z-scores."""
    return zscores.dropna(how="all").iloc[-1]


if __name__ == "__main__":
    from data.fetcher import fetch_all

    raw = fetch_all()
    zs  = compute_zscore(raw)

    print("Current Z-scores:")
    print(current_zscores(zs).round(3).to_string())
    print()
    print("Descriptive statistics (replicating Exhibit 4):")
    print(describe_transformed(zs).to_string())
