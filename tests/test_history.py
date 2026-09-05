"""Pure functions behind the extended data history (no network)."""
import numpy as np
import pandas as pd
import pytest

from data.fetcher import monthly_close_from_daily, parse_pink_sheet, splice_levels


def test_monthly_close_from_daily_takes_last_trading_day_at_month_end():
    idx = pd.bdate_range("2020-01-01", "2020-03-31")
    daily = pd.Series(np.arange(len(idx), dtype=float), index=idx)
    m = monthly_close_from_daily(daily)
    assert list(m.index) == [pd.Timestamp("2020-01-31"), pd.Timestamp("2020-02-29"), pd.Timestamp("2020-03-31")]
    assert m.iloc[0] == daily.loc["2020-01-31"]
    assert m.iloc[-1] == daily.iloc[-1]


def _sheet():
    # Mimics the World Bank pink sheet: titles in rows 0-3, names in row 4, units in row 5, data from row 6,
    # first column like "1960M01".
    rows = [["World Bank Commodity Price Data", None, None],
            ["monthly prices", None, None], [None, None, None], ["Updated", None, None],
            [None, "Crude oil, average", "Copper"],
            [None, "($/bbl)", "($/mt)"],
            ["1960M01", 1.63, 700.0], ["1960M02", 1.63, 710.5], ["1960M03", 1.63, "…"]]
    return pd.DataFrame(rows)


def test_parse_pink_sheet_returns_month_end_copper_in_dollars_per_tonne():
    s = parse_pink_sheet(_sheet(), "Copper")
    assert list(s.index) == [pd.Timestamp("1960-01-31"), pd.Timestamp("1960-02-29")]
    assert s.iloc[1] == pytest.approx(710.5)
    assert s.name == "copper"


def test_parse_pink_sheet_unknown_column_raises():
    with pytest.raises(KeyError):
        parse_pink_sheet(_sheet(), "Unobtainium")


def test_splice_levels_prefers_new_series_and_rescales_old_to_match():
    idx = pd.date_range("2000-01-31", periods=6, freq="ME")
    old = pd.Series([10.0, 11.0, 12.0, 13.0, np.nan, np.nan], index=idx)
    new = pd.Series([np.nan, np.nan, 24.0, 26.0, 28.0, 30.0], index=idx)   # exactly 2x old on the overlap
    out = splice_levels(old, new)
    assert out.loc[idx[0]] == pytest.approx(20.0)     # old rescaled by the overlap ratio
    assert out.loc[idx[2]] == pytest.approx(24.0)     # new wins where both exist
    assert out.loc[idx[5]] == pytest.approx(30.0)
    assert out.notna().all()


def test_splice_levels_without_overlap_raises():
    idx = pd.date_range("2000-01-31", periods=4, freq="ME")
    old = pd.Series([1.0, 2.0, np.nan, np.nan], index=idx)
    new = pd.Series([np.nan, np.nan, 3.0, 4.0], index=idx)
    with pytest.raises(ValueError):
        splice_levels(old, new)
