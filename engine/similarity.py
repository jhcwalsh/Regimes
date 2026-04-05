"""
Similarity engine: Euclidean distance-based regime identification.

Per Mulliner et al. (2026):
  - For a target date T, compute the squared Euclidean distance between
    T's Z-score vector and every historical month's Z-score vector.
  - Sum across all 7 variables → global score (lower = more similar).
  - Score = 0 at T by construction.
  - Exclude the last 36 months to avoid momentum loading.
  - Rank historical months; bottom quantile = similar regimes (Q1),
    top quantile = anti-regimes (Q5).
"""

import numpy as np
import pandas as pd
from typing import Optional

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import (
    EXCLUDE_RECENT_MONTHS,
    QUANTILE_SIMILAR,
    QUANTILE_DISSIMILAR,
    N_DISPLAY_SIMILAR,
)


# ---------------------------------------------------------------------------
# Core distance calculation
# ---------------------------------------------------------------------------

def compute_global_scores(
    zscores: pd.DataFrame,
    target_date: Optional[pd.Timestamp] = None,
    exclude_recent_months: int = EXCLUDE_RECENT_MONTHS,
) -> pd.Series:
    """
    Compute the global similarity score for every historical month vs. a target date.

    Parameters
    ----------
    zscores : pd.DataFrame
        Monthly Z-score matrix (rows = months, cols = variables).
    target_date : pd.Timestamp, optional
        The month to compare against. Defaults to the most recent month.
    exclude_recent_months : int
        Exclude this many months immediately before target_date from candidates.

    Returns
    -------
    pd.Series
        Global score for each historical month (NaN for excluded/missing rows).
        Lower score = more similar to target_date.
    """
    zs = zscores.dropna(how="all")

    if target_date is None:
        target_date = zs.index[-1]

    if target_date not in zs.index:
        raise ValueError(f"target_date {target_date} not found in Z-score index.")

    target_vec = zs.loc[target_date].values  # shape (7,)

    # Squared Euclidean distance for each historical row
    diff = zs.values - target_vec            # broadcast (T, 7)
    sq_dist = np.nansum(diff ** 2, axis=1)   # sum across variables → (T,)

    scores = pd.Series(sq_dist, index=zs.index, name="global_score")

    # Mask the exclusion window: [target_date - exclude_recent_months, target_date]
    cutoff = target_date - pd.DateOffset(months=exclude_recent_months)
    scores.loc[(scores.index > cutoff) & (scores.index <= target_date)] = np.nan

    # Also mask rows that had any NaN in the Z-score (insufficient history)
    nan_rows = zs.isnull().any(axis=1)
    scores.loc[nan_rows] = np.nan

    return scores


def rank_regimes(
    global_scores: pd.Series,
    quantile_similar: float = QUANTILE_SIMILAR,
    quantile_dissimilar: float = QUANTILE_DISSIMILAR,
) -> pd.DataFrame:
    """
    Rank historical months by similarity and label similar / dissimilar quintiles.

    Returns
    -------
    pd.DataFrame with columns:
        global_score  – raw distance
        rank          – rank (1 = most similar)
        pct_rank      – percentile rank (0 = most similar)
        regime        – 'similar' | 'dissimilar' | 'neutral'
    """
    valid = global_scores.dropna().sort_values()
    n = len(valid)

    df = pd.DataFrame({
        "global_score": valid,
        "rank":         range(1, n + 1),
        "pct_rank":     np.linspace(0, 1, n),
    })

    df["regime"] = "neutral"
    df.loc[df["pct_rank"] <= quantile_similar,              "regime"] = "similar"
    df.loc[df["pct_rank"] >= (1 - quantile_dissimilar),     "regime"] = "dissimilar"

    return df


def get_similar_periods(
    zscores: pd.DataFrame,
    target_date: Optional[pd.Timestamp] = None,
    n: int = N_DISPLAY_SIMILAR,
    exclude_recent_months: int = EXCLUDE_RECENT_MONTHS,
) -> pd.DataFrame:
    """
    Return the N most similar historical months to target_date.

    Returns
    -------
    pd.DataFrame with index = date, columns = global_score, rank, regime.
    """
    scores = compute_global_scores(zscores, target_date, exclude_recent_months)
    ranked = rank_regimes(scores)
    return ranked[ranked["regime"] == "similar"].head(n)


def get_dissimilar_periods(
    zscores: pd.DataFrame,
    target_date: Optional[pd.Timestamp] = None,
    n: int = N_DISPLAY_SIMILAR,
    exclude_recent_months: int = EXCLUDE_RECENT_MONTHS,
) -> pd.DataFrame:
    """Return the N most dissimilar (anti-regime) historical months."""
    scores = compute_global_scores(zscores, target_date, exclude_recent_months)
    ranked = rank_regimes(scores)
    return ranked[ranked["regime"] == "dissimilar"].sort_values("global_score", ascending=False).head(n)


# ---------------------------------------------------------------------------
# Full historical global score time series (for dashboard chart)
# ---------------------------------------------------------------------------

def compute_global_score_history(
    zscores: pd.DataFrame,
    exclude_recent_months: int = EXCLUDE_RECENT_MONTHS,
) -> pd.Series:
    """
    Compute global scores for each month in the Z-score history,
    treating each month as the 'current' date in turn.
    This produces the time series used in regime-shift detection and charts.

    Returns
    -------
    pd.Series  index = date, values = global score at that date
               (minimum distance to any valid historical month)
    """
    zs = zscores.dropna(how="all")
    min_scores = {}

    for target_date in zs.index:
        scores = compute_global_scores(zs, target_date, exclude_recent_months)
        valid = scores.dropna()
        if len(valid) > 0:
            min_scores[target_date] = valid.min()
        else:
            min_scores[target_date] = np.nan

    return pd.Series(min_scores, name="min_global_score")


# ---------------------------------------------------------------------------
# Factor timing signal (optional, requires factor return data)
# ---------------------------------------------------------------------------

def compute_factor_signal(
    zscores: pd.DataFrame,
    factor_returns: pd.Series,
    target_date: Optional[pd.Timestamp] = None,
    quantile: float = QUANTILE_SIMILAR,
    fwd_months: int = 1,
    exclude_recent_months: int = EXCLUDE_RECENT_MONTHS,
) -> dict:
    """
    Compute the regime-based factor timing signal for a single factor.

    Parameters
    ----------
    factor_returns : pd.Series
        Monthly returns for the factor (aligned to month-end dates).
    fwd_months : int
        Number of months forward to look at subsequent returns (default 1).

    Returns
    -------
    dict with keys:
        signal       – +1 (long) or -1 (short)
        avg_fwd_ret  – average forward return in similar historical periods
        n_periods    – number of similar periods used
        similar_dates – list of similar dates
    """
    scores = compute_global_scores(zscores, target_date, exclude_recent_months)
    ranked = rank_regimes(scores, quantile_similar=quantile)
    similar_dates = ranked[ranked["regime"] == "similar"].index.tolist()

    # Subsequent returns: return in the month AFTER each similar date
    fwd_returns = []
    for d in similar_dates:
        # Find the return fwd_months after d
        try:
            loc = factor_returns.index.get_loc(d)
            if loc + fwd_months < len(factor_returns):
                fwd_returns.append(factor_returns.iloc[loc + fwd_months])
        except KeyError:
            pass

    avg_fwd = np.mean(fwd_returns) if fwd_returns else 0.0
    signal = 1 if avg_fwd >= 0 else -1

    return {
        "signal":        signal,
        "avg_fwd_ret":   avg_fwd,
        "n_periods":     len(fwd_returns),
        "similar_dates": similar_dates,
    }


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from data.fetcher import fetch_all
    from data.transformer import compute_zscore

    raw = fetch_all()
    zs  = compute_zscore(raw)

    print("Computing global scores for most recent month...")
    scores = compute_global_scores(zs)
    ranked = rank_regimes(scores)

    print(f"\nTarget date: {zs.dropna(how='all').index[-1].date()}")
    print("\nTop 20 most similar historical periods:")
    print(ranked[ranked["regime"] == "similar"].head(20).to_string())
    print("\nTop 10 most dissimilar (anti-regime) periods:")
    print(ranked[ranked["regime"] == "dissimilar"].sort_values("global_score", ascending=False).head(10).to_string())
