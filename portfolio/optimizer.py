"""
Portfolio optimisers for regime-based hedge fund allocation.

Three methods:
  1. black_litterman  — BL posterior weights using regime-conditional views
  2. risk_parity      — equal risk contribution (baseline)
  3. regime_tilted_rp — risk parity tilted by regime Sharpe premium
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from typing import Optional

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from config import (
    RISK_AVERSION, BL_TAU,
    MAX_STRATEGY_WEIGHT, MIN_STRATEGY_WEIGHT,
)


# ---------------------------------------------------------------------------
# Black-Litterman
# ---------------------------------------------------------------------------

def black_litterman(
    sigma: np.ndarray,
    w_eq: np.ndarray,
    views: np.ndarray,
    confidence: np.ndarray,
    delta: float = RISK_AVERSION,
    tau: float = BL_TAU,
    strategy_names: Optional[list] = None,
) -> pd.Series:
    """
    Black-Litterman posterior weights with absolute views.

    Equilibrium:  pi  = delta * Sigma * w_eq
    Views:        P   = I  (one absolute view per strategy)
                  Q   = views vector (annualised, fraction, e.g. 0.08 = 8%)
                  Omega = diag, Omega[i] = tau * sigma[i,i] / confidence[i]^2

    Posterior mean:
        mu_BL = [(tau*Sigma)^{-1} + P' Omega^{-1} P]^{-1}
                * [(tau*Sigma)^{-1} pi + P' Omega^{-1} Q]

    Optimal weights:
        w* = (delta * Sigma)^{-1} mu_BL  (normalised to sum to 1)

    Parameters
    ----------
    sigma       : (n, n) annualised covariance matrix
    w_eq        : (n,)   equilibrium weights (e.g. equal or AUM-weighted)
    views       : (n,)   expected return per strategy (annualised fraction)
    confidence  : (n,)   view confidence in [0, 1]
    delta       : risk aversion coefficient
    tau         : BL uncertainty scaling (typically 0.025 – 0.05)
    strategy_names : list, optional  Index for output Series

    Returns
    -------
    pd.Series  Optimal portfolio weights (non-negative, sum to 1)
    """
    n = len(w_eq)

    # Equilibrium returns
    pi = delta * sigma @ w_eq

    # Absolute views: P = I
    P = np.eye(n)
    Q = views.astype(float)

    # View uncertainty: lower confidence → wider uncertainty
    conf_clipped = np.clip(confidence.astype(float), 0.01, 1.0)
    omega_diag   = tau * np.diag(sigma) / (conf_clipped ** 2)
    Omega        = np.diag(omega_diag)

    # BL posterior
    tau_sigma_inv = np.linalg.inv(tau * sigma + np.eye(n) * 1e-8)
    omega_inv     = np.linalg.inv(Omega + np.eye(n) * 1e-8)

    M = np.linalg.inv(tau_sigma_inv + P.T @ omega_inv @ P)
    mu_bl = M @ (tau_sigma_inv @ pi + P.T @ omega_inv @ Q)

    # MVO weights
    weights = np.linalg.inv(delta * sigma + np.eye(n) * 1e-8) @ mu_bl

    # Clip and normalise
    weights = np.clip(weights, MIN_STRATEGY_WEIGHT, MAX_STRATEGY_WEIGHT)
    w_sum   = weights.sum()
    if w_sum > 0:
        weights /= w_sum
    else:
        weights = np.ones(n) / n

    names = strategy_names if strategy_names is not None else list(range(n))
    return pd.Series(weights, index=names)


# ---------------------------------------------------------------------------
# Risk parity
# ---------------------------------------------------------------------------

def risk_parity(
    sigma: np.ndarray,
    strategy_names: Optional[list] = None,
    w_bounds: Optional[tuple] = None,
) -> pd.Series:
    """
    Equal risk contribution (risk parity) weights.

    Minimises:  sum_i (RC_i - 1/n)^2
    where RC_i = w_i * (Sigma w)_i / (w' Sigma w)

    Parameters
    ----------
    sigma       : (n, n) covariance matrix (monthly or annualised)
    strategy_names : index for output
    w_bounds    : (min, max) weight bounds per strategy

    Returns
    -------
    pd.Series  Equal risk contribution weights
    """
    n = sigma.shape[0]
    w0 = np.ones(n) / n

    if w_bounds is None:
        w_bounds = (MIN_STRATEGY_WEIGHT, MAX_STRATEGY_WEIGHT)

    bounds = [w_bounds] * n
    target_rc = 1.0 / n

    def objective(w):
        port_var = float(w @ sigma @ w)
        if port_var <= 0:
            return 1e10
        rc = w * (sigma @ w) / port_var
        return float(np.sum((rc - target_rc) ** 2))

    constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]

    res = minimize(
        objective, w0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"ftol": 1e-12, "maxiter": 1000},
    )

    weights = res.x
    weights = np.clip(weights, w_bounds[0], w_bounds[1])
    weights /= weights.sum()

    names = strategy_names if strategy_names is not None else list(range(n))
    return pd.Series(weights, index=names)


# ---------------------------------------------------------------------------
# Regime-tilted risk parity
# ---------------------------------------------------------------------------

def regime_tilted_rp(
    sigma: np.ndarray,
    regime_sharpes: np.ndarray,
    unconditional_sharpes: np.ndarray,
    strategy_names: Optional[list] = None,
    tilt_strength: float = 0.5,
    w_bounds: Optional[tuple] = None,
) -> pd.Series:
    """
    Risk parity base allocation tilted by regime Sharpe premium.

    Tilt logic:
      1. Compute Sharpe premium = regime_sharpe - unconditional_sharpe per strategy.
      2. Cross-sectionally rank premiums, normalise to [-1, +1].
      3. Tilt base RP weight by (1 + tilt_strength * normalised_rank).
      4. Re-clip to weight bounds and normalise.

    Parameters
    ----------
    sigma                : (n, n) covariance matrix
    regime_sharpes       : (n,)   Sharpe ratios in regime-similar periods
    unconditional_sharpes: (n,)   Full-history Sharpe ratios
    tilt_strength        : float  0 = pure RP, 1 = maximum tilt
    w_bounds             : (min, max) per-strategy weight bounds

    Returns
    -------
    pd.Series  Tilted portfolio weights
    """
    n = sigma.shape[0]

    if w_bounds is None:
        w_bounds = (MIN_STRATEGY_WEIGHT, MAX_STRATEGY_WEIGHT)

    base = risk_parity(sigma, strategy_names, w_bounds)
    w    = base.values.copy()

    premium = np.nan_to_num(regime_sharpes - unconditional_sharpes, nan=0.0)

    # Rank → normalise to [-1, +1]
    from scipy.stats import rankdata
    ranks      = rankdata(premium)                       # 1 = worst premium
    normalised = 2 * (ranks - 1) / max(n - 1, 1) - 1   # [-1, +1]

    w_tilted = w * (1 + tilt_strength * normalised)
    w_tilted = np.clip(w_tilted, w_bounds[0], w_bounds[1])
    w_tilted /= w_tilted.sum()

    names = strategy_names if strategy_names is not None else list(range(n))
    return pd.Series(w_tilted, index=names)
