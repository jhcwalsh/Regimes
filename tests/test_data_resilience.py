"""The site must survive a source going away: bundled copper history, stale-on-error refresh."""
import os
import urllib.request

import pandas as pd
import pytest

import data.fetcher as fetcher


def _no_network(*args, **kwargs):
    raise urllib.error.HTTPError("https://example.invalid", 403, "Forbidden", None, None)


def test_copper_history_loads_from_the_repo_without_network(monkeypatch):
    monkeypatch.setattr(urllib.request, "urlopen", _no_network)
    s = fetcher.load_copper_history()
    assert s.index.min() == pd.Timestamp("1960-01-31")
    assert s.index.max() >= pd.Timestamp("1992-12-31")  # overlaps FRED's PCOPPUSDM
    assert s.name == "copper"
    assert (s > 0).all()


def test_failed_refresh_keeps_the_old_cache(monkeypatch, tmp_path):
    monkeypatch.setattr(fetcher, "CACHE_DIR", str(tmp_path))
    old = tmp_path / f"oil.{fetcher.CACHE_VERSION}.parquet"
    pd.Series([1.0], name="oil").to_frame().to_parquet(old)

    def boom(*args, **kwargs):
        raise RuntimeError("source down")

    monkeypatch.setattr(fetcher, "_fetch_all", boom)
    with pytest.raises(RuntimeError):
        fetcher.fetch_all(refresh_cache=True)

    assert old.exists()
    assert os.listdir(tmp_path) == [old.name]  # nothing half-written left behind


def test_successful_refresh_replaces_the_old_cache(monkeypatch, tmp_path):
    monkeypatch.setattr(fetcher, "CACHE_DIR", str(tmp_path))
    old = tmp_path / f"oil.{fetcher.CACHE_VERSION}.parquet"
    pd.Series([1.0], name="oil").to_frame().to_parquet(old)
    seen = {}

    def fake(start, fill):
        seen["cache_empty"] = not any(f.endswith(".parquet") for f in os.listdir(tmp_path))
        return pd.DataFrame()

    monkeypatch.setattr(fetcher, "_fetch_all", fake)
    fetcher.fetch_all(refresh_cache=True)

    assert seen["cache_empty"]  # the refetch did not read the stale files
    assert not old.exists()


def test_load_frames_serves_the_previous_cache_when_refresh_fails(monkeypatch):
    from web import data as web_data

    calls = []
    frame = pd.DataFrame({"sp500": [1.0, 2.0]}, index=pd.to_datetime(["2026-01-31", "2026-02-28"]))

    def fake_fetch_all(refresh_cache, fill):
        calls.append(refresh_cache)
        if refresh_cache:
            raise RuntimeError("source down")
        return frame

    monkeypatch.setattr(fetcher, "fetch_all", fake_fetch_all)
    monkeypatch.setattr("data.transformer.compute_zscore", lambda raw: raw)
    raw, _, last = web_data.load_frames(refresh=True)

    assert calls == [True, False]
    assert raw.equals(frame)
    assert last["sp500"] == pd.Timestamp("2026-02-28")
