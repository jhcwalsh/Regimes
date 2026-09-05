"""
Replication checks against the paper's worked exhibits. They need the real cached
data, so they skip when the cache is absent.
"""
import glob
import os

import pandas as pd
import pytest

from config import CACHE_DIR

pytestmark = pytest.mark.skipif(not glob.glob(os.path.join(CACHE_DIR, "*.parquet")), reason="no cached data")


@pytest.fixture(scope="module")
def zscores():
    from data.fetcher import fetch_all
    from data.transformer import compute_zscore
    return compute_zscore(fetch_all())


def _similar_years(zscores, target):
    from engine.similarity import compute_global_scores, rank_regimes
    ranked = rank_regimes(compute_global_scores(zscores, pd.Timestamp(target)))
    return ranked[ranked["regime"] == "similar"].index.year


def test_history_reaches_back_to_the_early_1970s(zscores):
    assert zscores.dropna().index[0] <= pd.Timestamp("1971-12-31")


def test_january_2009_matches_the_early_1980s_recessions(zscores):
    # Paper, Exhibit 6: the similar months "include all observed recessions, including the
    # double-dip recessions in the 1980s".
    years = _similar_years(zscores, "2009-01-31")
    assert ((years >= 1980) & (years <= 1982)).sum() >= 6
    assert ((years >= 1990) & (years <= 1991)).sum() >= 3
    assert ((years >= 2001) & (years <= 2002)).sum() >= 6


def test_august_2022_matches_the_late_1970s_inflation(zscores):
    # Paper, Exhibit 8: August 2022 "is most similar to the inflation period following the
    # Iranian Revolution of 1977-1980" and also loads on 1972-1974.
    years = _similar_years(zscores, "2022-08-31")
    assert ((years >= 1977) & (years <= 1980)).sum() >= 10
    assert ((years >= 1972) & (years <= 1974)).sum() >= 1
