"""
Regime shift detector using exponentially weighted moving averages (EWMA).

Per Mulliner et al. (2026), Exhibit 9:
  - At each time T, compute the EWMA of all global scores up to T,
    weighting recent dates more heavily.
  - Four lookback windows: 1, 2, 3, 4 years.
  - Half-life:  alpha = 1 - 1/n
                t_half = -ln(2) / ln(alpha)
  - A rapidly rising EWMA signals a regime shift.
"""

import numpy as np
import pandas as pd
from typing import Optional

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import EWMA_LOOKBACK_YEARS


# ---------------------------------------------------------------------------
# EWMA helpers
# ---------------------------------------------------------------------------

def _alpha_from_window(n_months: int) -> float:
    """Equation (2) from paper: alpha = 1 - 1/n"""
    return 1.0 - (1.0 / n_months)


def _half_life(n_months: int) -> float:
    """Equation (3) from paper: t_half = -ln(2) / ln(alpha)"""
    alpha = _alpha_from_window(n_months)
    return -np.log(2) / np.log(alpha)


def compute_ewma_regime_shift(
    global_scores: pd.Series,
    lookback_years: list[int] = EWMA_LOOKBACK_YEARS,
) -> pd.DataFrame:
    """
    Compute EWMA of global scores for multiple lookback windows.

    For each time T and each lookback window n:
      C_T = EWMA({global_score_t : t <= T}) with alpha = 1 - 1/n

    Parameters
    ----------
    global_scores : pd.Series
        Time series of minimum global scores (from compute_global_score_history).
    lookback_years : list of int
        Lookback windows in years (paper uses [1, 2, 3, 4]).

    Returns
    -------
    pd.DataFrame
        Columns: ewma_{n}yr for each lookback, plus 'mean_ewma'.
        Higher values indicate the current environment differs significantly
        from recent history → potential regime shift.
    """
    valid = global_scores.dropna().sort_index()
    result = pd.DataFrame(index=valid.index)

    for yrs in lookback_years:
        n_months = yrs * 12
        alpha    = _alpha_from_window(n_months)
        col_name = f"ewma_{yrs}yr"
        # pandas ewm with alpha directly gives C_T
        result[col_name] = valid.ewm(alpha=(1 - alpha), adjust=True).mean()

    result["mean_ewma"] = result.mean(axis=1)
    return result


def get_half_lives(lookback_years: list[int] = EWMA_LOOKBACK_YEARS) -> pd.DataFrame:
    """
    Replicates Exhibit 9 Panel A: half-lives for each lookback period.
    """
    rows = []
    for yrs in lookback_years:
        n = yrs * 12
        rows.append({
            "lookback_years":  yrs,
            "lookback_months": n,
            "alpha":           round(_alpha_from_window(n), 4),
            "half_life_months": round(_half_life(n), 1),
        })
    return pd.DataFrame(rows)


def detect_regime_shift_events(
    ewma_df: pd.DataFrame,
    threshold_pct: float = 0.90,
) -> pd.DataFrame:
    """
    Identify dates where the mean EWMA exceeds a historical percentile threshold,
    signalling a potential regime shift.

    Parameters
    ----------
    threshold_pct : float
        Percentile of mean_ewma to use as the spike threshold (default 90th).

    Returns
    -------
    pd.DataFrame of regime-shift dates with their mean_ewma values.
    """
    col = "mean_ewma"
    threshold = ewma_df[col].quantile(threshold_pct)
    events = ewma_df[ewma_df[col] >= threshold][[col]].copy()
    events["threshold"] = threshold
    return events


def current_regime_shift_score(ewma_df: pd.DataFrame) -> dict:
    """
    Return the latest regime shift reading and its historical context.
    """
    latest = ewma_df.iloc[-1]
    col = "mean_ewma"
    pct_rank = (ewma_df[col] <= latest[col]).mean()

    return {
        "date":          ewma_df.index[-1].date(),
        "mean_ewma":     round(latest[col], 4),
        "pct_rank":      round(pct_rank, 3),
        "signal":        "REGIME SHIFT" if pct_rank >= 0.90 else "STABLE",
        "by_lookback":   {
            c: round(latest[c], 4)
            for c in ewma_df.columns if c.startswith("ewma_")
        },
    }


if __name__ == "__main__":
    from data.fetcher import fetch_all
    from data.transformer import compute_zscore
    from engine.similarity import compute_global_score_history

    raw    = fetch_all()
    zs     = compute_zscore(raw)
    scores = compute_global_score_history(zs)
    ewma   = compute_ewma_regime_shift(scores)

    print("Half-lives (Exhibit 9 Panel A):")
    print(get_half_lives().to_string(index=False))

    print("\nCurrent regime shift reading:")
    reading = current_regime_shift_score(ewma)
    for k, v in reading.items():
        print(f"  {k}: {v}")

    print("\nHistoric regime shift events (>90th pct):")
    events = detect_regime_shift_events(ewma)
    print(events.to_string())
