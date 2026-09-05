import pandas as pd

from data.french import parse_french_csv

SAMPLE = """This file was created using the 202607 CRSP database.
Some notes here.

,Mkt-RF,SMB,HML,RMW,CMA,RF
196307,   -0.39,   -0.48,   -0.84,    0.64,   -1.15,    0.27
196308,    5.08,   -0.80,    1.72,    0.40,   -0.38,    0.25

 Annual Factors: January-December 
,Mkt-RF,SMB,HML,RMW,CMA,RF
  1964,   14.01,    0.17,    5.44,    2.15,    5.72,    3.54

Copyright 2026 Eugene F. Fama and Kenneth R. French
"""


def test_parse_french_csv_keeps_monthly_rows_only_at_month_end_in_percent():
    df = parse_french_csv(SAMPLE)
    assert list(df.columns) == ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF"]
    assert list(df.index) == [pd.Timestamp("1963-07-31"), pd.Timestamp("1963-08-31")]
    assert df.loc["1963-08-31", "Mkt-RF"] == 5.08
    assert df.dtypes.eq(float).all()
