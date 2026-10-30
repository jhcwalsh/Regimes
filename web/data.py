"""
Data access for the public app: cache freshness rule and memoised loading.

The engine (data/, engine/) is unchanged; this module decides *when* to refetch
and hands the app ready-made frames.
"""
from __future__ import annotations

import glob
import logging
import os
from datetime import datetime, timedelta

import pandas as pd

from config import CACHE_DIR

log = logging.getLogger(__name__)

MAX_CACHE_AGE_DAYS = 7


def needs_refresh(cache_written_at: datetime | None, now: datetime,
                  max_age_days: int = MAX_CACHE_AGE_DAYS) -> bool:
    """True when there is no cache, or the cache is older than max_age_days."""
    if cache_written_at is None:
        return True
    return (now - cache_written_at) > timedelta(days=max_age_days)


def cache_written_at(cache_dir: str = CACHE_DIR) -> datetime | None:
    """Newest modification time across the parquet cache, or None if empty."""
    files = glob.glob(os.path.join(cache_dir, "*.parquet"))
    if not files:
        return None
    return datetime.fromtimestamp(max(os.path.getmtime(f) for f in files))


def load_frames(refresh: bool) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """
    Raw state variables (forward-filled), their Z-scores, and the last month each
    variable was actually observed. Refetches when `refresh` is True.
    """
    from data.fetcher import fetch_all, last_observed
    from data.transformer import compute_zscore

    try:
        unfilled = fetch_all(refresh_cache=refresh, fill=False)
    except Exception:
        if not refresh:
            raise
        log.exception("Refresh failed; serving the previous cache")
        unfilled = fetch_all(refresh_cache=False, fill=False)
    raw = unfilled.ffill()
    return raw, compute_zscore(raw), last_observed(unfilled)


def load_factors(refresh: bool) -> pd.DataFrame:
    """
    Ken French factor returns (percent, monthly). Refetches when `refresh` is True,
    keeping the previous file if the download fails.
    """
    from data.french import fetch_french_factors
    from data.fetcher import _cache_path

    path = _cache_path("french_factors")
    if not (refresh and os.path.exists(path)):
        return fetch_french_factors()
    os.replace(path, path + ".stale")
    try:
        df = fetch_french_factors()
    except Exception:
        log.exception("Factor refresh failed; serving the previous file")
        os.replace(path + ".stale", path)
        return fetch_french_factors()
    os.remove(path + ".stale")
    return df
