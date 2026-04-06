"""
Strategy timing engine: regime-conditional return statistics for hedge fund strategies.

For a given target date:
  1. Identify similar historical regime months via the global score engine.
  2. Compute forward returns of each strategy in those similar periods.
  3. Return regime-conditional stats (mean, std, Sharpe) and Black-Litterman views.

Autocorrelation in hedge fund returns (from stale pricing / illiquid holdings)
is corrected via Geltner (1994) unsmoothing before any statistics are computed.
"""

import numpy as np
import pandas as pd
from typing import Optional

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from config import (
    EXCLUDE_RECENT_MONTHS, QUANTILE_SIMILAR,
    HF_HORIZONS, UNSMOOTH_RETURNS,
)
from engine.similarity import compute_global_scores, rank_regimes


# ---------------------------------------------------------------------------
# Unsmoothing
# ---------------------------------------------------------------------------

def unsmooth_returns(r: pd.Series, rho: Optional[float] = None) -> pd.Series:
    """
    Geltner (1994) unsmoothing: removes first-order autocorrelation from
    illiquid asset return series (hedge funds, private equity, etc.).

    Formula:  r_unsmoothed_t = (r_t - rho * r_{t-1}) / (1 - rho)

    Parameters
    ----------
    r   : pd.Series  Monthly return series.
    rho : float, optional  AR(1) smoothing parameter. Estimated from data if None.

    Returns
    -------
    pd.Series  Unsmoothed returns (NaN at first observation).
    """
    r = r.copy()
    if rho is None:
        rho = float(r.autocorr(lag=1))

    # Clip to avoid numerical issues
    rho = float(np.clip(rho, -0.95, 0.95))

    if abs(rho) < 0.05:
        # Negligible smoothing — return as-is
        return r

    unsmoothed = (r - rho * r.shift(1)) / (1 - rho)
    unsmoothed.name = r.name
    return unsmoothed


def unsmooth_dataframe(returns: pd.DataFrame) -> pd.DataFrame:
    """Apply Geltner unsmoothing column-by-column."""
    return returns.apply(unsmooth_returns, axis=0)


# ---------------------------------------------------------------------------
# Regime-conditional statistics
# ---------------------------------------------------------------------------

def compute_regime_conditional_stats(
    strategy_returns: pd.DataFrame,
    global_scores: pd.Series,
    quantile: float = QUANTILE_SIMILAR,
    horizons: list = HF_HORIZONS,
) -> dict:
    """
    Compute forward return statistics for each strategy in similar-regime months.

    Parameters
    ----------
    strategy_returns : pd.DataFrame
        Monthly returns, columns = strategy names.
    global_scores : pd.Series
        Pre-computed global scores for a specific target date.
    quantile : float
        Fraction of history defining 'similar' regimes.
    horizons : list of int
        Forward-return horizons in months (e.g. [1, 3, 6, 12]).

    Returns
    -------
    dict  Keyed by horizon (int). Each value is a pd.DataFrame with:
          index = strategy name
          columns = [regime_mean_ret, regime_std, regime_sharpe,
                     unconditional_sharpe, sharpe_premium, n_similar_periods]
    """
    ranked = rank_regimes(global_scores, quantile_similar=quantile)
    similar_dates = ranked[ranked["regime"] == "similar"].index

    results = {}

    for h in horizons:
        rows = {}
        for strategy in strategy_returns.columns:
            r = strategy_returns[strategy].dropna()

            # --- Regime-conditional forward returns ---
            sim_fwd = []
            for d in similar_dates:
                try:
                    loc = r.index.get_loc(d)
                    if loc + h < len(r):
                        # Compounded return over h months
                        fwd = float((1 + r.iloc[loc + 1: loc + h + 1]).prod() - 1)
                        sim_fwd.append(fwd)
                except KeyError:
                    pass

            # --- Unconditional forward returns (all available dates) ---
            all_fwd = []
            for i in range(len(r) - h):
                fwd = float((1 + r.iloc[i + 1: i + h + 1]).prod() - 1)
                all_fwd.append(fwd)

            sim_arr = np.array(sim_fwd) if sim_fwd else np.full(1, np.nan)
            all_arr = np.array(all_fwd) if all_fwd else np.full(1, np.nan)

            # Annualisation factor
            ann = np.sqrt(12 / h)

            sim_mean   = float(np.nanmean(sim_arr))
            sim_std    = float(np.nanstd(sim_arr, ddof=1)) if len(sim_arr) > 1 else np.nan
            sim_sharpe = (sim_mean / sim_std * ann) if sim_std and sim_std > 0 else np.nan

            unc_mean   = float(np.nanmean(all_arr))
            unc_std    = float(np.nanstd(all_arr, ddof=1)) if len(all_arr) > 1 else np.nan
            unc_sharpe = (unc_mean / unc_std * ann) if unc_std and unc_std > 0 else np.nan

            rows[strategy] = {
                "regime_mean_ret_%":    round(sim_mean * 100, 3),
                "regime_std_%":         round(sim_std * 100, 3) if not np.isnan(sim_std) else np.nan,
                "regime_sharpe":        round(sim_sharpe, 3) if not np.isnan(sim_sharpe) else np.nan,
                "unconditional_sharpe": round(unc_sharpe, 3) if not np.isnan(unc_sharpe) else np.nan,
                "sharpe_premium":       round(sim_sharpe - unc_sharpe, 3)
                                        if not np.isnan(sim_sharpe) and not np.isnan(unc_sharpe)
                                        else np.nan,
                "n_similar_periods":    len(sim_fwd),
            }

        results[h] = pd.DataFrame(rows).T

    return results


# ---------------------------------------------------------------------------
# Black-Litterman views
# ---------------------------------------------------------------------------

def compute_views(
    strategy_returns: pd.DataFrame,
    zscores: pd.DataFrame,
    target_date: Optional[pd.Timestamp] = None,
    quantile: float = QUANTILE_SIMILAR,
    horizon: int = 1,
    exclude_recent_months: int = EXCLUDE_RECENT_MONTHS,
    unsmooth: bool = UNSMOOTH_RETURNS,
) -> dict:
    """
    Compute Black-Litterman views from regime-conditional expected returns.

    For each strategy:
      - 'View' = average 1-month forward return in similar historical months (annualised %).
      - 'Confidence' = scaled by sqrt(n_similar_periods) — more history → higher confidence.

    Falls back to unconditional mean if no similar periods overlap with strategy history.

    Returns
    -------
    dict with keys:
        views         : pd.Series  Annualised expected returns per strategy (%)
        confidence    : pd.Series  View confidence in [0.10, 1.00]
        n_periods     : pd.Series  Number of similar months with return data
        target_date   : pd.Timestamp
        similar_dates : list of pd.Timestamp
    """
    zs = zscores.dropna(how="all")
    if target_date is None:
        target_date = zs.index[-1]

    if unsmooth:
        strategy_returns = unsmooth_dataframe(strategy_returns)

    scores = compute_global_scores(zs, target_date, exclude_recent_months)
    ranked = rank_regimes(scores, quantile_similar=quantile)
    similar_dates = ranked[ranked["regime"] == "similar"].index

    expected_returns = {}
    n_periods_dict   = {}

    for strategy in strategy_returns.columns:
        r = strategy_returns[strategy].dropna()
        fwd = []
        for d in similar_dates:
            try:
                loc = r.index.get_loc(d)
                if loc + horizon < len(r):
                    fwd.append(float(r.iloc[loc + horizon]))
            except KeyError:
                pass

        if fwd:
            expected_returns[strategy] = np.mean(fwd) * 12 * 100  # annualised %
            n_periods_dict[strategy]   = len(fwd)
        else:
            # Fallback: unconditional mean
            expected_returns[strategy] = float(r.mean()) * 12 * 100
            n_periods_dict[strategy]   = 0

    views     = pd.Series(expected_returns)
    n_periods = pd.Series(n_periods_dict)

    # Confidence: proportional to sqrt(n_periods), normalised to [0.10, 1.00]
    max_n      = max(n_periods.max(), 1)
    confidence = (np.sqrt(n_periods) / np.sqrt(max_n)).clip(0.10, 1.00)

    return {
        "views":         views,
        "confidence":    confidence,
        "n_periods":     n_periods,
        "target_date":   target_date,
        "similar_dates": similar_dates.tolist(),
    }


# ---------------------------------------------------------------------------
# Cross-sectional regime-conditional stats table (for dashboard display)
# ---------------------------------------------------------------------------

def current_regime_stats(
    strategy_returns: pd.DataFrame,
    zscores: pd.DataFrame,
    target_date: Optional[pd.Timestamp] = None,
    quantile: float = QUANTILE_SIMILAR,
    horizon: int = 1,
    exclude_recent_months: int = EXCLUDE_RECENT_MONTHS,
    unsmooth: bool = UNSMOOTH_RETURNS,
) -> pd.DataFrame:
    """
    One-stop function: compute regime-conditional stats for current date.

    Returns a display-ready DataFrame with regime vs unconditional comparison
    for each strategy at the given horizon.
    """
    zs = zscores.dropna(how="all")
    if target_date is None:
        target_date = zs.index[-1]

    if unsmooth:
        strategy_returns = unsmooth_dataframe(strategy_returns)

    scores = compute_global_scores(zs, target_date, exclude_recent_months)
    stats_by_horizon = compute_regime_conditional_stats(
        strategy_returns, scores, quantile=quantile, horizons=[horizon]
    )
    return stats_by_horizon[horizon]
