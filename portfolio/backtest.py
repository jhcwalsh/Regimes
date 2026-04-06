"""
Walk-forward backtest engine for regime-based hedge fund portfolio construction.

Algorithm (per rebalance date T):
  1. Use only data available at T (no lookahead).
  2. Identify similar historical regime months via global similarity scores.
  3. Compute regime-conditional expected returns as Black-Litterman views.
  4. Estimate rolling covariance (36-month window, shrinkage regularisation).
  5. Optimise weights via chosen method (BL, risk parity, or tilted RP).
  6. Record actual strategy returns over the next month (or rebalance_freq months).

Includes demo_strategy_returns() to generate realistic synthetic HF data
until data/hf_returns.py is built.
"""

import warnings
import numpy as np
import pandas as pd
from typing import Optional

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from config import (
    EXCLUDE_RECENT_MONTHS, QUANTILE_SIMILAR,
    REBALANCE_FREQ_MONTHS, UNSMOOTH_RETURNS,
    BACKTEST_START,
)


# ---------------------------------------------------------------------------
# Synthetic demo data
# ---------------------------------------------------------------------------

#: Long-run target characteristics for each demo strategy.
#: Replace with real HFRI indices once data/hf_returns.py is available.
DEMO_STRATEGY_PARAMS = {
    "Equity L/S":      {"mu": 0.080, "vol": 0.070, "rho": 0.20},
    "Global Macro":    {"mu": 0.070, "vol": 0.080, "rho": 0.15},
    "Merger Arb":      {"mu": 0.050, "vol": 0.040, "rho": 0.25},
    "Distressed":      {"mu": 0.090, "vol": 0.090, "rho": 0.30},
    "CTA / Trend":     {"mu": 0.060, "vol": 0.100, "rho": 0.10},
    "Event Driven":    {"mu": 0.070, "vol": 0.070, "rho": 0.22},
    "RV Fixed Income": {"mu": 0.060, "vol": 0.050, "rho": 0.28},
    "Multi-Strategy":  {"mu": 0.070, "vol": 0.050, "rho": 0.18},
}

#: Approximate cross-strategy correlation matrix (calibrated to HFRI history).
_DEMO_CORR = np.array([
    #  ELS   GMac  MArb  Dist  CTA   EvDr  RVFI  Multi
    [1.000, 0.400, 0.350, 0.550, 0.100, 0.600, 0.200, 0.650],  # Equity L/S
    [0.400, 1.000, 0.200, 0.300, 0.450, 0.350, 0.250, 0.500],  # Global Macro
    [0.350, 0.200, 1.000, 0.400, 0.050, 0.650, 0.150, 0.450],  # Merger Arb
    [0.550, 0.300, 0.400, 1.000, 0.000, 0.650, 0.300, 0.550],  # Distressed
    [0.100, 0.450, 0.050, 0.000, 1.000, 0.050, 0.150, 0.200],  # CTA / Trend
    [0.600, 0.350, 0.650, 0.650, 0.050, 1.000, 0.200, 0.650],  # Event Driven
    [0.200, 0.250, 0.150, 0.300, 0.150, 0.200, 1.000, 0.400],  # RV Fixed Income
    [0.650, 0.500, 0.450, 0.550, 0.200, 0.650, 0.400, 1.000],  # Multi-Strategy
])

#: Crisis periods that increase volatility and reduce returns in demo data.
_CRISIS_WINDOWS = [
    ("1998-08-31", "1998-10-31"),   # LTCM
    ("2001-09-30", "2002-09-30"),   # 9/11 + tech bust
    ("2007-07-31", "2009-03-31"),   # GFC
    ("2020-02-29", "2020-05-31"),   # COVID
    ("2022-01-31", "2022-10-31"),   # Rate shock
]


def demo_strategy_returns(
    start: str = BACKTEST_START,
    end: Optional[str] = None,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate synthetic monthly hedge fund strategy returns for demonstration.

    Properties
    ----------
    - Realistic long-run means, volatilities, and cross-correlations.
    - First-order autocorrelation (AR(1) smoothing) mimicking stale pricing.
    - Crisis periods with elevated volatility and negative drift.

    Replace with real HFRI data once data/hf_returns.py is complete.

    Parameters
    ----------
    start : str  First month (month-end date string, e.g. "1994-01-31")
    end   : str  Last month  (defaults to today)
    seed  : int  Random seed for reproducibility

    Returns
    -------
    pd.DataFrame  index = month-end dates, columns = strategy names
    """
    rng       = np.random.default_rng(seed)
    strategies = list(DEMO_STRATEGY_PARAMS.keys())
    k          = len(strategies)

    dates = pd.date_range(start=start, end=end or pd.Timestamp.today(), freq="ME")
    n     = len(dates)

    # Monthly parameters
    mus  = np.array([DEMO_STRATEGY_PARAMS[s]["mu"] / 12  for s in strategies])
    vols = np.array([DEMO_STRATEGY_PARAMS[s]["vol"] / np.sqrt(12) for s in strategies])

    # Correlated covariance
    cov = np.diag(vols) @ _DEMO_CORR @ np.diag(vols)

    # Make positive-definite (Cholesky)
    eigvals, eigvecs = np.linalg.eigh(cov)
    cov = eigvecs @ np.diag(np.maximum(eigvals, 1e-10)) @ eigvecs.T
    L   = np.linalg.cholesky(cov)

    # Raw correlated innovations
    raw = mus + rng.standard_normal((n, k)) @ L.T

    # Add AR(1) smoothing per strategy
    smoothed = raw.copy()
    rhos     = [DEMO_STRATEGY_PARAMS[s]["rho"] for s in strategies]
    for j, rho in enumerate(rhos):
        for t in range(1, n):
            smoothed[t, j] = rho * smoothed[t - 1, j] + (1 - rho) * raw[t, j]

    # Crisis shocks
    for start_c, end_c in _CRISIS_WINDOWS:
        mask = (dates >= pd.Timestamp(start_c)) & (dates <= pd.Timestamp(end_c))
        idx  = np.where(mask)[0]
        if len(idx):
            smoothed[idx] *= 2.5    # Higher volatility
            smoothed[idx] -= 0.02   # Negative drift

    return pd.DataFrame(smoothed, index=dates, columns=strategies)


# ---------------------------------------------------------------------------
# Covariance helpers
# ---------------------------------------------------------------------------

def _shrink_covariance(sigma: np.ndarray, alpha: float = 0.10) -> np.ndarray:
    """
    Ledoit-Wolf-style linear shrinkage: blend sample covariance with
    a scaled identity matrix (target = average variance on diagonal).

    alpha=0.10 provides mild regularisation without over-shrinking.
    """
    n      = sigma.shape[0]
    target = np.eye(n) * np.trace(sigma) / n
    shrunk = (1 - alpha) * sigma + alpha * target
    # Ensure positive-definiteness
    eigvals, eigvecs = np.linalg.eigh(shrunk)
    shrunk = eigvecs @ np.diag(np.maximum(eigvals, 1e-8)) @ eigvecs.T
    return shrunk


# ---------------------------------------------------------------------------
# Main backtest
# ---------------------------------------------------------------------------

def run_backtest(
    strategy_returns: pd.DataFrame,
    zscores: pd.DataFrame,
    method: str = "bl",
    quantile: float = QUANTILE_SIMILAR,
    exclude_recent_months: int = EXCLUDE_RECENT_MONTHS,
    rebalance_freq: int = REBALANCE_FREQ_MONTHS,
    horizon: int = 1,
    unsmooth: bool = UNSMOOTH_RETURNS,
    min_history_months: int = 60,
    tilt_strength: float = 0.5,
) -> dict:
    """
    Walk-forward regime-based portfolio backtest.

    Parameters
    ----------
    strategy_returns    : pd.DataFrame  Monthly returns (columns = strategies)
    zscores             : pd.DataFrame  Macro Z-scores from the regime engine
    method              : str  'bl' | 'rp' | 'tilted_rp' | 'equal'
    quantile            : float  Fraction of history defining 'similar' regimes
    exclude_recent_months : int  Exclusion window (avoids momentum)
    rebalance_freq      : int  Months between rebalancing
    horizon             : int  Forward horizon for BL views (months)
    unsmooth            : bool  Apply Geltner unsmoothing to strategy returns
    min_history_months  : int  Minimum strategy return history before first trade
    tilt_strength       : float  Tilt intensity for 'tilted_rp' (0=RP, 1=max tilt)

    Returns
    -------
    dict with keys:
        portfolio_returns : pd.Series   Monthly portfolio returns
        benchmark_returns : pd.Series   Equal-weight benchmark returns
        weights_history   : pd.DataFrame  Portfolio weights at each rebalance date
        stats             : pd.DataFrame  Tearsheet vs benchmark (performance_summary)
        strategy_names    : list
    """
    from engine.strategy_timing import compute_views, unsmooth_dataframe
    from portfolio.optimizer import black_litterman, risk_parity, regime_tilted_rp
    from portfolio.risk import performance_summary

    strategies = strategy_returns.columns.tolist()
    n          = len(strategies)

    # Optionally unsmooth upfront (once, not per rebalance)
    if unsmooth:
        ret_input = unsmooth_dataframe(strategy_returns)
    else:
        ret_input = strategy_returns.copy()

    # Common index: strategy data ∩ valid Z-scores
    valid_zs     = zscores.dropna(how="all")
    common_index = strategy_returns.index.intersection(valid_zs.index).sort_values()

    if len(common_index) < min_history_months:
        raise ValueError(
            f"Only {len(common_index)} common months of data. "
            f"Need at least {min_history_months}."
        )

    # Rebalance schedule: start after min_history_months
    rebalance_dates = common_index[min_history_months::rebalance_freq]

    port_returns   = {}
    weights_hist   = {}
    current_weights = pd.Series(np.ones(n) / n, index=strategies)

    for rebalance_date in rebalance_dates:

        # --- Data available at this date (strict no-lookahead) ---
        hist_ret = ret_input.loc[:rebalance_date]
        hist_zs  = zscores.loc[:rebalance_date]

        if len(hist_ret) < min_history_months:
            continue

        # --- Views ---
        try:
            view_result = compute_views(
                hist_ret, hist_zs,
                target_date=rebalance_date,
                quantile=quantile,
                horizon=horizon,
                exclude_recent_months=exclude_recent_months,
                unsmooth=False,          # already unsmoothed above
            )
            views      = view_result["views"].reindex(strategies).fillna(0.0)
            confidence = view_result["confidence"].reindex(strategies).fillna(0.10)
        except Exception:
            # Fallback: unconditional mean as view
            views      = hist_ret.mean() * 12 * 100
            confidence = pd.Series(0.10, index=strategies)

        # --- Covariance (rolling 36m, shrinkage-regularised) ---
        cov_window = min(len(hist_ret), 36)
        sigma_m    = hist_ret.iloc[-cov_window:].cov().fillna(0).values
        sigma_ann  = _shrink_covariance(sigma_m * 12)   # annualised

        w_eq = np.ones(n) / n

        # --- Optimise ---
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                if method == "bl":
                    weights = black_litterman(
                        sigma_ann, w_eq,
                        views.values / 100,        # convert % → fraction
                        confidence.values,
                        strategy_names=strategies,
                    )
                elif method == "rp":
                    weights = risk_parity(sigma_ann, strategy_names=strategies)
                elif method == "tilted_rp":
                    unc_sharpes = (
                        hist_ret.mean() / hist_ret.std() * np.sqrt(12)
                    ).reindex(strategies).fillna(0).values
                    reg_sharpes = (
                        (views.values / 100) / np.sqrt(np.maximum(np.diag(sigma_ann), 1e-8))
                    )
                    weights = regime_tilted_rp(
                        sigma_ann, reg_sharpes, unc_sharpes,
                        strategy_names=strategies,
                        tilt_strength=tilt_strength,
                    )
                else:   # equal weight
                    weights = pd.Series(np.ones(n) / n, index=strategies)
            except Exception:
                weights = current_weights.copy()

        current_weights          = weights
        weights_hist[rebalance_date] = weights

        # --- Forward returns over next rebalance_freq months ---
        future = strategy_returns.index[strategy_returns.index > rebalance_date]
        for fwd_date in future[:rebalance_freq]:
            row = strategy_returns.loc[fwd_date].reindex(strategies).fillna(0)
            port_returns[fwd_date] = float((weights * row).sum())

    # Assemble
    port_series = pd.Series(port_returns, name=f"Regime-{method.upper()}").sort_index()

    # Equal-weight benchmark over the same period
    eq_bench = strategy_returns.reindex(port_series.index).mean(axis=1)
    eq_bench.name = "Equal Weight"

    # Tearsheet
    port_stats = performance_summary(port_series, benchmark=eq_bench, label=port_series.name)
    eq_stats   = performance_summary(eq_bench, label="Equal Weight")

    return {
        "portfolio_returns": port_series,
        "benchmark_returns": eq_bench,
        "weights_history":   pd.DataFrame(weights_hist).T,
        "stats":             pd.concat([port_stats, eq_stats], axis=1),
        "strategy_names":    strategies,
    }
