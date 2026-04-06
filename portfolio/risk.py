"""
Portfolio risk metrics for hedge fund allocation analysis.

Functions
---------
drawdown_series       – monthly drawdown from peak
max_drawdown          – worst peak-to-trough
calmar_ratio          – annualised return / |max drawdown|
value_at_risk         – historical VaR
cvar                  – conditional VaR (expected shortfall)
rolling_sharpe        – rolling annualised Sharpe
rolling_sortino       – rolling annualised Sortino
risk_attribution      – marginal and total risk contribution per strategy
performance_summary   – full tearsheet as pd.Series
"""

import numpy as np
import pandas as pd
from typing import Optional


# ---------------------------------------------------------------------------
# Drawdown
# ---------------------------------------------------------------------------

def drawdown_series(returns: pd.Series) -> pd.Series:
    """Drawdown from rolling peak at each month."""
    wealth = (1 + returns.fillna(0)).cumprod()
    peak   = wealth.cummax()
    dd     = (wealth - peak) / peak
    dd.name = "drawdown"
    return dd


def max_drawdown(returns: pd.Series) -> float:
    """Maximum peak-to-trough drawdown (negative number)."""
    return float(drawdown_series(returns).min())


def calmar_ratio(returns: pd.Series, periods_per_year: int = 12) -> float:
    """Annualised return divided by absolute maximum drawdown."""
    ann_ret = float((1 + returns.mean()) ** periods_per_year - 1)
    mdd     = abs(max_drawdown(returns))
    return ann_ret / mdd if mdd > 0 else np.nan


# ---------------------------------------------------------------------------
# Tail risk
# ---------------------------------------------------------------------------

def value_at_risk(returns: pd.Series, confidence: float = 0.95) -> float:
    """
    Historical VaR: the return at the (1-confidence) left tail.
    Returns a negative number (a loss).
    """
    return float(np.percentile(returns.dropna(), (1 - confidence) * 100))


def cvar(returns: pd.Series, confidence: float = 0.95) -> float:
    """
    Conditional VaR (Expected Shortfall): average return beyond VaR.
    Returns a negative number.
    """
    var  = value_at_risk(returns, confidence)
    tail = returns[returns <= var]
    return float(tail.mean()) if len(tail) > 0 else var


# ---------------------------------------------------------------------------
# Rolling metrics
# ---------------------------------------------------------------------------

def rolling_sharpe(
    returns: pd.Series,
    window: int = 36,
    rf_monthly: float = 0.0,
    periods_per_year: int = 12,
) -> pd.Series:
    """Rolling annualised Sharpe ratio."""
    excess = returns - rf_monthly
    mean_  = excess.rolling(window).mean()
    std_   = excess.rolling(window).std()
    sr     = mean_ / std_ * np.sqrt(periods_per_year)
    sr.name = "rolling_sharpe"
    return sr


def rolling_sortino(
    returns: pd.Series,
    window: int = 36,
    rf_monthly: float = 0.0,
    periods_per_year: int = 12,
) -> pd.Series:
    """Rolling annualised Sortino ratio (downside deviation denominator)."""
    excess   = returns - rf_monthly
    mean_    = excess.rolling(window).mean()
    downside = excess.where(excess < 0).rolling(window).std()
    sr       = mean_ / downside * np.sqrt(periods_per_year)
    sr.name  = "rolling_sortino"
    return sr


# ---------------------------------------------------------------------------
# Risk attribution
# ---------------------------------------------------------------------------

def risk_attribution(
    weights: np.ndarray,
    sigma: np.ndarray,
    strategy_names: Optional[list] = None,
) -> pd.DataFrame:
    """
    Marginal and percentage risk contribution per strategy.

    Risk contribution of strategy i:
        RC_i = w_i * (Sigma w)_i / sqrt(w' Sigma w)

    Returns
    -------
    pd.DataFrame with columns:
        weight, marginal_risk, risk_contribution, pct_risk
    """
    w        = np.array(weights, dtype=float)
    port_var = float(w @ sigma @ w)
    port_vol = np.sqrt(max(port_var, 1e-12))

    marginal_rc = sigma @ w / port_vol
    total_rc    = w * marginal_rc
    pct_rc      = total_rc / total_rc.sum()

    df = pd.DataFrame({
        "weight":            w,
        "marginal_risk":     marginal_rc,
        "risk_contribution": total_rc,
        "pct_risk":          pct_rc,
    })

    if strategy_names is not None:
        df.index = strategy_names

    return df.round(5)


# ---------------------------------------------------------------------------
# Performance summary tearsheet
# ---------------------------------------------------------------------------

def performance_summary(
    returns: pd.Series,
    benchmark: Optional[pd.Series] = None,
    periods_per_year: int = 12,
    label: Optional[str] = None,
) -> pd.Series:
    """
    Comprehensive performance tearsheet as a named pd.Series.

    Metrics: Ann. Return, Ann. Vol, Sharpe, Sortino, Max DD, Calmar,
             VaR (95%), CVaR (95%), Best/Worst Month, n Months.
    If benchmark provided: adds Beta, Alpha, Info Ratio, Tracking Error.

    Parameters
    ----------
    returns   : pd.Series  Monthly total returns
    benchmark : pd.Series  Monthly benchmark returns (optional)
    label     : str        Name for the output Series

    Returns
    -------
    pd.Series  String-formatted metrics
    """
    r = returns.dropna()
    if len(r) == 0:
        return pd.Series(dtype=str, name=label)

    ann_ret = float((1 + r.mean()) ** periods_per_year - 1)
    ann_vol = float(r.std() * np.sqrt(periods_per_year))
    sharpe  = ann_ret / ann_vol if ann_vol > 0 else np.nan

    downside = float(r.where(r < 0).std() * np.sqrt(periods_per_year))
    sortino  = ann_ret / downside if downside > 0 else np.nan

    mdd    = max_drawdown(r)
    calmar = calmar_ratio(r, periods_per_year)

    var95  = value_at_risk(r, 0.95)
    cvar95 = cvar(r, 0.95)

    stats = {
        "Ann. Return":     f"{ann_ret * 100:.2f}%",
        "Ann. Volatility": f"{ann_vol * 100:.2f}%",
        "Sharpe Ratio":    f"{sharpe:.3f}" if not np.isnan(sharpe) else "n/a",
        "Sortino Ratio":   f"{sortino:.3f}" if not np.isnan(sortino) else "n/a",
        "Max Drawdown":    f"{mdd * 100:.2f}%",
        "Calmar Ratio":    f"{calmar:.3f}" if not np.isnan(calmar) else "n/a",
        "VaR (95%)":       f"{var95 * 100:.2f}%",
        "CVaR (95%)":      f"{cvar95 * 100:.2f}%",
        "Best Month":      f"{r.max() * 100:.2f}%",
        "Worst Month":     f"{r.min() * 100:.2f}%",
        "Months":          str(len(r)),
    }

    if benchmark is not None:
        bm     = benchmark.reindex(r.index).dropna()
        common = r.reindex(bm.index).dropna()
        bm     = bm.reindex(common.index)

        if len(common) > 12:
            te    = float((common - bm).std() * np.sqrt(periods_per_year))
            ir    = float((common.mean() - bm.mean()) * periods_per_year / te) if te > 0 else np.nan
            beta  = float(np.cov(common, bm)[0, 1] / np.var(bm)) if np.var(bm) > 0 else np.nan
            alpha = float((common.mean() - beta * bm.mean()) * periods_per_year) if not np.isnan(beta) else np.nan
            stats["Beta"]           = f"{beta:.3f}" if not np.isnan(beta) else "n/a"
            stats["Alpha (ann.)"]   = f"{alpha * 100:.2f}%" if not np.isnan(alpha) else "n/a"
            stats["Info Ratio"]     = f"{ir:.3f}" if not np.isnan(ir) else "n/a"
            stats["Tracking Error"] = f"{te * 100:.2f}%"

    return pd.Series(stats, name=label or r.name or "Portfolio")
