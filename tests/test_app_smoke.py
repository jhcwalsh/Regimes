"""
Headless render of each public view from the cached parquet data.
Skipped when the cache is absent (CI without data), since the views need real frames.
"""
import glob
import os

import pytest
from streamlit.testing.v1 import AppTest

from config import CACHE_DIR

HAS_CACHE = bool(glob.glob(os.path.join(CACHE_DIR, "*.parquet")))
pytestmark = pytest.mark.skipif(not HAS_CACHE, reason="no cached data")


def _run(view: str) -> AppTest:
    at = AppTest.from_string(f"from web.app import main\nmain(view={view!r}, allow_refresh=False)\n",
                             default_timeout=120)
    return at.run()


@pytest.mark.parametrize("view", ["now", "explore", "factors", "method"])
def test_view_renders_without_exception(view):
    at = _run(view)
    assert not at.exception, at.exception


def test_unknown_view_falls_back_to_now():
    at = _run("nonsense")
    assert not at.exception
    assert any("Which past" in m.value for m in at.markdown)
