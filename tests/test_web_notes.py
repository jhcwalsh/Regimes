import pandas as pd

from web.app import exhibit_claim, stale_note, VAR_LABELS


def test_exhibit_claim_for_a_paper_month():
    claim = exhibit_claim(pd.Timestamp("2009-01-31"))
    assert claim and "1980s" in claim and "Exhibit 6" in claim


def test_exhibit_claim_is_none_for_other_months():
    assert exhibit_claim(pd.Timestamp("2010-05-31")) is None


def test_stale_note_names_forward_filled_variables_and_their_last_month():
    current = pd.Timestamp("2026-09-30")
    last_obs = pd.Series({
        "sp500": current, "yield_curve": pd.Timestamp("2026-08-31"), "oil": pd.Timestamp("2026-08-31"),
        "copper": pd.Timestamp("2026-07-31"), "tbill_3m": pd.Timestamp("2026-08-31"),
        "volatility": current, "stock_bond_corr": current,
    })
    note = stale_note(last_obs, current, VAR_LABELS)
    assert "Copper" in note and "July 2026" in note
    assert "August 2026" in note and "WTI crude oil" in note
    assert "S&P 500" not in note


def test_stale_note_is_empty_when_everything_is_current():
    current = pd.Timestamp("2026-09-30")
    last_obs = pd.Series({k: current for k in VAR_LABELS})
    assert stale_note(last_obs, current, VAR_LABELS) == ""
