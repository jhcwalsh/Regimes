from datetime import datetime, timedelta

from web.data import needs_refresh


def test_no_cache_means_refresh():
    assert needs_refresh(None, datetime(2026, 9, 3)) is True


def test_fresh_cache_does_not_refresh():
    now = datetime(2026, 9, 3, 12)
    assert needs_refresh(now - timedelta(days=6), now) is False


def test_cache_older_than_a_week_refreshes():
    now = datetime(2026, 9, 3, 12)
    assert needs_refresh(now - timedelta(days=8), now) is True


def test_max_age_is_configurable():
    now = datetime(2026, 9, 3, 12)
    assert needs_refresh(now - timedelta(days=2), now, max_age_days=1) is True
