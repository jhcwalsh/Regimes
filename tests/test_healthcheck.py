"""The health check's verdict on loaded data (no network, no Docker)."""
from datetime import datetime

import pandas as pd

from scripts.healthcheck import assess

NOW = datetime(2026, 10, 31, 12)


def _last_obs(**overrides):
    s = pd.Series({"sp500": "2026-10-31", "copper": "2026-09-30", "oil": "2026-09-30"})
    for k, v in overrides.items():
        s[k] = v
    return pd.to_datetime(s)


def _factors(last="2026-09-30"):
    return pd.DataFrame({"Mkt-RF": [1.0, 2.0]}, index=pd.to_datetime(["2026-01-31", last]))


def test_healthy_data_has_no_problems():
    assert assess(_last_obs(), _factors(), cache_at=datetime(2026, 10, 30), now=NOW) == []


def test_a_series_that_stopped_updating_is_reported():
    problems = assess(_last_obs(copper="2026-06-30"), _factors(), cache_at=datetime(2026, 10, 30), now=NOW)
    assert len(problems) == 1
    assert "copper" in problems[0] and "June 2026" in problems[0]


def test_an_old_cache_means_the_refresh_is_failing():
    problems = assess(_last_obs(), _factors(), cache_at=datetime(2026, 10, 20), now=NOW)
    assert len(problems) == 1
    assert "refresh" in problems[0]


def test_stale_factor_returns_are_reported():
    problems = assess(_last_obs(), _factors(last="2026-05-31"), cache_at=datetime(2026, 10, 30), now=NOW)
    assert len(problems) == 1
    assert "factor" in problems[0].lower()
