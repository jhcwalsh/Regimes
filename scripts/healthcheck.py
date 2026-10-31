"""
Health check for the public app, run inside the container by scripts/check_regimes.sh.

Loads the data exactly as the app does, refreshing when the cache is older than a
week, then checks that every series is still arriving. The app serves the previous
cache when a refresh fails, so a dead source never shows on the page; this is what
notices. Prints one line per problem and exits 1, or prints "ok" and exits 0.
"""
from __future__ import annotations

import logging
import os
import sys
from datetime import datetime, timedelta

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Monthly series land about a month late; two and a half months means one is stuck.
MAX_SERIES_LAG_DAYS = 75
# Ken French posts monthly, usually one to two months behind.
MAX_FACTOR_LAG_DAYS = 120
# The app refreshes at seven days, so a cache older than this survived a failed refresh.
MAX_CACHE_AGE_DAYS = 8


def assess(last_obs: pd.Series, factors: pd.DataFrame, cache_at: datetime | None,
           now: datetime) -> list[str]:
    """Problems found, in plain words; empty when healthy."""
    problems = []
    for name, when in last_obs.items():
        if pd.isna(when) or (now - when) > timedelta(days=MAX_SERIES_LAG_DAYS):
            month = "never" if pd.isna(when) else when.strftime("%B %Y")
            problems.append(f"{name} has not updated since {month}")
    if cache_at is None or (now - cache_at) > timedelta(days=MAX_CACHE_AGE_DAYS):
        age = "missing" if cache_at is None else f"{(now - cache_at).days} days old"
        problems.append(f"data cache is {age}: the weekly refresh is failing (see output above)")
    last_factor = factors.index.max()
    if (now - last_factor) > timedelta(days=MAX_FACTOR_LAG_DAYS):
        problems.append(f"Factor returns stop at {last_factor:%B %Y}")
    return problems


def main() -> int:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    from web.data import cache_written_at, load_factors, load_frames, needs_refresh

    now = datetime.now()
    refresh = needs_refresh(cache_written_at(), now)
    try:
        _, _, last_obs = load_frames(refresh)
        factors = load_factors(refresh)
    except Exception as exc:
        print(f"data does not load, the site shows its error card: {exc}")
        return 1
    problems = assess(last_obs, factors, cache_written_at(), now)
    for p in problems:
        print(p)
    if not problems:
        print("ok")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
