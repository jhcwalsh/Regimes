"""
Regime shift detector (Mulliner et al. 2026, Exhibit 9).

At each month T:
  1. Compute d_{T,t}, the Euclidean distance (eq. 1) between T's Z-score
     vector and every earlier month t <= T (d_{T,T} = 0).
  2. C_T = EWMA over t of d_{T,t}, with the highest weight on the most
     recent t:   weight(t) = beta^(T - t),   beta = 1 - 1/n   (eq. 2)
     where n is the lookback in months (12, 24, 36, 48).
  3. Half-life  t_half = -ln(2) / ln(beta)                    (eq. 3)

C_T measures how far today is from the recent past. A rapid rise means the
environment is changing quickly — a potential regime shift. The paper plots
the four lookbacks and their mean; peaks sit near Oct 2008 / Jan 2009,
May 2020 and Oct 2022.
"""

import numpy as np
import pandas as pd

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import EWMA_LOOKBACK_YEARS


# ---------------------------------------------------------------------------
# EWMA helpers
# ---------------------------------------------------------------------------

def _beta_from_window(n_months: int) -> float:
    """Equation (2) from paper: beta = 1 - 1/n"""
    return 1.0 - (1.0 / n_months)


def _half_life(n_months: int) -> float:
    """Equation (3) from paper: t_half = -ln(2) / ln(beta)"""
    return -np.log(2) / np.log(_beta_from_window(n_months))


# ---------------------------------------------------------------------------
# Indicator
# ---------------------------------------------------------------------------

def compute_regime_shift(
    zscores: pd.DataFrame,
    lookback_years: list[int] = EWMA_LOOKBACK_YEARS,
) -> pd.DataFrame:
    """
    Compute C_T for every complete month and each lookback window.

    Parameters
    ----------
    zscores : pd.DataFrame
        Monthly Z-score matrix (rows = months, cols = variables). Rows with
        any missing variable are dropped before computing distances.
    lookback_years : list of int
        Lookback windows in years (paper uses [1, 2, 3, 4]).

    Returns
    -------
    pd.DataFrame
        Columns: ewma_{n}yr for each lookback, plus 'mean_ewma'.
    """
    zs = zscores.dropna(how="any").sort_index()
    X = zs.values
    n = len(X)

    # Pairwise Euclidean distances D[T, t]
    D = np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(axis=-1))

    # lag[T, t] = T - t for t <= T, used to build beta^(T - t) weights
    T_idx = np.arange(n)
    lag = T_idx[:, None] - T_idx[None, :]
    past = lag >= 0

    result = pd.DataFrame(index=zs.index)
    for yrs in lookback_years:
        beta = _beta_from_window(yrs * 12)
        W = np.where(past, beta ** np.clip(lag, 0, None), 0.0)
        result[f"ewma_{yrs}yr"] = (W * D).sum(axis=1) / W.sum(axis=1)

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
            "beta":            round(_beta_from_window(n), 4),
            "half_life_months": round(_half_life(n), 1),
        })
    return pd.DataFrame(rows)


def detect_regime_shift_events(
    ewma_df: pd.DataFrame,
    threshold_pct: float = 0.90,
) -> pd.DataFrame:
    """
    Identify dates where the mean EWMA exceeds a historical percentile threshold,
    signalling a potential regime shift. (Threshold is this project's heuristic,
    not from the paper.)

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

    raw    = fetch_all()
    zs     = compute_zscore(raw)
    ewma   = compute_regime_shift(zs)

    print("Half-lives (Exhibit 9 Panel A):")
    print(get_half_lives().to_string(index=False))

    print("\nCurrent regime shift reading:")
    reading = current_regime_shift_score(ewma)
    for k, v in reading.items():
        print(f"  {k}: {v}")

    print("\nHistoric regime shift events (>90th pct):")
    events = detect_regime_shift_events(ewma)
    print(events.to_string())
