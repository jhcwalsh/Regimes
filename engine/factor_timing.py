"""
Factor timing with the regime model (Mulliner et al. 2026, pp. 18-19, Exhibits 10-13).

At each decision month T:
  1. Rank every month up to T minus the 36-month exclusion by its distance to T
     and cut the ranking into quantiles (1 = most similar).
  2. For each quantile and each factor, average the factor's return in the month
     AFTER each of the quantile's months. Positive average -> long the factor next
     month, negative -> short.
  3. The quantile portfolio's return in month T+1 is the equal-weighted average of
     the six signed factor returns. The spread is quantile 1 minus quantile 5.

Returns are in percent per month, as published by Ken French.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import EXCLUDE_RECENT_MONTHS

BACKTEST_START = "1985-01-31"   # paper: performance shown for 1985-2024


def quantile_labels(scores: pd.Series, n_quantiles: int = 5) -> pd.Series:
    """Rank valid scores ascending and label them 1..n_quantiles (1 = smallest distance)."""
    valid = scores.dropna().sort_values(kind="mergesort")
    n = len(valid)
    labels = (np.arange(n) * n_quantiles) // n + 1
    return pd.Series(labels, index=valid.index, dtype=int)


def direction(months, factor_returns: pd.Series) -> int:
    """+1 if the mean return in the month after each of `months` is non-negative, else -1."""
    nxt = factor_returns.shift(-1).reindex(list(months))
    avg = float(np.nanmean(nxt.values)) if nxt.notna().any() else 0.0
    return 1 if avg >= 0 else -1


def _pairwise_distances(x: np.ndarray) -> np.ndarray:
    return np.sqrt(((x[:, None, :] - x[None, :, :]) ** 2).sum(axis=-1))


def run_backtest(
    zscores: pd.DataFrame,
    factors: pd.DataFrame,
    start: str = BACKTEST_START,
    n_quantiles: int = 5,
    exclude_recent_months: int = EXCLUDE_RECENT_MONTHS,
) -> dict:
    """
    Returns dict with
      returns   : DataFrame indexed by holding month (T+1): q1..qN, long_only, spread (percent)
      positions : DataFrame (factors x [q1, qN]) of the latest decision's signs
      decisions : DatetimeIndex of decision months used
    """
    zs = zscores.dropna(how="any").sort_index()
    factors = factors.sort_index()
    next_ret = factors.shift(-1).reindex(zs.index)        # return in the month after each month
    D = _pairwise_distances(zs.values)
    names = list(factors.columns)
    start_ts = pd.Timestamp(start)

    holding_rows, decisions, last_positions = {}, [], None
    for i, T in enumerate(zs.index):
        if T < start_ts or i - exclude_recent_months < 0:
            continue
        loc = factors.index.get_indexer([T])[0]
        if loc < 0 or loc + 1 >= len(factors.index):
            continue                                       # need the realised return at T+1
        hold = factors.index[loc + 1]
        cand = slice(0, i - exclude_recent_months + 1)     # months up to T-36
        scores = pd.Series(D[i, cand], index=zs.index[cand])
        scores = scores[next_ret.iloc[cand].notna().any(axis=1).values]
        if len(scores) < n_quantiles:
            continue
        labels = quantile_labels(scores, n_quantiles)
        realised = factors.loc[hold, names].values
        row = {}
        positions = {}
        for q in range(1, n_quantiles + 1):
            months = labels.index[labels.values == q]
            avg = next_ret.loc[months, names].mean(axis=0).values
            signs = np.where(np.nan_to_num(avg) >= 0, 1, -1)
            row[f"q{q}"] = float(np.mean(signs * realised))
            positions[f"q{q}"] = signs
        row["long_only"] = float(np.mean(realised))
        row["spread"] = row["q1"] - row[f"q{n_quantiles}"]
        holding_rows[hold] = row
        decisions.append(T)
        last_positions = pd.DataFrame({"q1": positions["q1"], f"q{n_quantiles}": positions[f"q{n_quantiles}"]},
                                      index=names)

    returns = pd.DataFrame.from_dict(holding_rows, orient="index").sort_index()
    return {"returns": returns, "positions": last_positions, "decisions": pd.DatetimeIndex(decisions)}


def performance(returns: pd.DataFrame, benchmark: str = "long_only") -> pd.DataFrame:
    """Annualised return, vol, Sharpe, max drawdown (on cumulative percent) and correlation to the benchmark."""
    rows = {}
    for col in returns.columns:
        r = returns[col].dropna()
        cum = r.cumsum()
        dd = (cum - cum.cummax()).min()
        rows[col] = {
            "ann_return_%": r.mean() * 12,
            "ann_vol_%": r.std() * np.sqrt(12),
            "sharpe": (r.mean() / r.std()) * np.sqrt(12) if r.std() > 0 else np.nan,
            "max_drawdown_%": dd,
            f"corr_to_{benchmark}": r.corr(returns[benchmark]) if benchmark in returns else np.nan,
            "months": len(r),
        }
    return pd.DataFrame(rows).T


if __name__ == "__main__":
    from data.fetcher import fetch_all
    from data.french import fetch_french_factors
    from data.transformer import compute_zscore

    zs = compute_zscore(fetch_all())
    out = run_backtest(zs, fetch_french_factors())
    print(performance(out["returns"]).round(2).to_string())
    print(out["positions"])
