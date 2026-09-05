"""
Ken French data library: the Fama-French five factors plus momentum, monthly, in percent.

Files are zipped CSVs with a text preamble, a monthly block keyed YYYYMM, then an
annual block. Only the monthly block is used. No API key is needed.
"""
from __future__ import annotations

import io
import re
import urllib.request
import zipfile

import pandas as pd

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from data.fetcher import _cache_path

BASE = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
FILES = {
    "five": "F-F_Research_Data_5_Factors_2x3_CSV.zip",   # Mkt-RF, SMB, HML, RMW, CMA, RF from 1963-07
    "mom":  "F-F_Momentum_Factor_CSV.zip",               # Mom from 1927-01
}
FACTORS = ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "Mom"]


def parse_french_csv(text: str) -> pd.DataFrame:
    """Monthly rows of a French-library CSV as a float DataFrame indexed at month-end."""
    header = None
    rows = []
    for line in text.splitlines():
        if header is None:
            if line.startswith(","):
                header = [c.strip() for c in line.split(",")[1:]]
            continue
        first = line.split(",")[0].strip()
        if re.fullmatch(r"\d{6}", first):
            rows.append([p.strip() for p in line.split(",")])
        elif line.strip() and rows:
            break  # the annual block follows the monthly one
    if header is None or not rows:
        raise ValueError("no monthly block found in French CSV")
    df = pd.DataFrame(rows, columns=["date"] + header)
    idx = pd.PeriodIndex(df["date"].str[:4] + "-" + df["date"].str[4:], freq="M").to_timestamp("M")
    out = df.drop(columns="date").astype(float)
    out.index = idx
    return out


def _download_csv(zip_name: str) -> str:
    with urllib.request.urlopen(BASE + zip_name, timeout=60) as resp:
        raw = resp.read()
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        name = next(n for n in z.namelist() if n.lower().endswith(".csv"))
        return z.read(name).decode("latin-1")


def fetch_french_factors() -> pd.DataFrame:
    """Monthly returns in percent for the six factors, cached as parquet."""
    path = _cache_path("french_factors")
    if os.path.exists(path):
        return pd.read_parquet(path)
    five = parse_french_csv(_download_csv(FILES["five"]))
    mom = parse_french_csv(_download_csv(FILES["mom"]))
    mom.columns = [c if c.lower() != "mom" else "Mom" for c in mom.columns]
    df = five.drop(columns=["RF"]).join(mom[["Mom"]], how="inner")[FACTORS]
    df.to_parquet(path)
    return df


if __name__ == "__main__":
    f = fetch_french_factors()
    print(f.index[0].date(), "->", f.index[-1].date(), f.shape)
    print(f.tail(3).to_string())
