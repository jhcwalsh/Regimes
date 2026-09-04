# Regimes

Live replication of the nonparametric economic regime identification model from:

> Mulliner A., Harvey C.R., Xia C., Fang E., Van Hemert O.
> **"Regimes"** — *Journal of Portfolio Management*, February 2026.

---

## What it does

Identifies which historical periods are most similar to current market conditions by computing Euclidean distances across seven economic state variables. Uses this similarity to:

1. Surface the most analogous historical regimes
2. Identify "anti-regimes" — the most dissimilar historical periods
3. Detect potential regime shifts: at each month, the distance to every earlier month is averaged with exponentially decaying weights (paper Exhibit 9)

## Seven State Variables

| # | Variable | Source |
|---|----------|--------|
| 1 | S&P 500 log price | Yahoo Finance |
| 2 | Yield curve (10yr − 3m T-bill) | FRED: GS10, TB3MS |
| 3 | WTI crude oil price | FRED: DCOILWTICO |
| 4 | Copper price | FRED: PCOPPUSDM |
| 5 | US 3-month T-bill yield | FRED: TB3MS |
| 6 | VIX / realised volatility (spliced 1990) | FRED: VIXCLS + Yahoo Finance |
| 7 | Rolling 3-yr stock–bond correlation | Computed from daily data |

Each variable is transformed: 12-month difference → rolling 10-year Z-score → winsorise ±3.

## Setup

```bash
pip install -r requirements.txt
```

Get a free FRED API key at [fred.stlouisfed.org](https://fred.stlouisfed.org/docs/api/api_key.html) and set it:

```bash
export FRED_API_KEY=your_key_here   # Mac/Linux
set FRED_API_KEY=your_key_here      # Windows
```

## Run the public app (regimes.lazyeconomist.com)

The site-styled app at the repo root reads the FRED key from the environment only:

```bash
streamlit run app.py
```

Views are chosen by query parameter: `?view=now` (default), `?view=explore&month=2009-01-31`, `?view=method`.
The parquet cache refreshes itself when it is more than a week old.

Deploy on the Mac mini follows the hosting runbook: `docker compose up -d --build` in `~/apps/regimes`
(port 8503, `.env` holding `FRED_API_KEY`), then a Cloudflare published route `regimes` → `localhost:8503`.
Design notes: `docs/superpowers/specs/2026-09-03-regimes-site-design.md`.

## Run the research dashboard

The original Streamlit dashboard with sidebar controls and the portfolio tab:

```bash
streamlit run dashboard/app.py
```

## Run the tests

```bash
python -m pytest tests
```

## Project structure

```
Regimes/
├── config.py              # All parameters in one place
├── data/
│   ├── fetcher.py         # FRED + yfinance data pipeline
│   └── transformer.py     # Z-score transformation
├── engine/
│   ├── similarity.py      # Euclidean distance / global score
│   └── regime_shift.py    # EWMA regime shift detector
├── app.py                 # Public app entry point (lazyeconomist.com style)
├── web/                   # Public app: views, house style, charts, data freshness
├── dashboard/
│   └── app.py             # Research dashboard (sidebar controls, portfolio tab)
├── tests/                 # pytest suite (no network needed)
├── cache/                 # Parquet cache (git-ignored)
└── requirements.txt
```

## Key parameters (`config.py`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `DIFF_MONTHS` | 12 | Annual differencing window |
| `ZSCORE_WINDOW_YRS` | 10 | Rolling std window for Z-score |
| `WINSOR_LIMIT` | 3.0 | Z-score cap |
| `EXCLUDE_RECENT_MONTHS` | 36 | Months excluded near today (avoids momentum) |
| `QUANTILE_SIMILAR` | 0.20 | Fraction defining "similar" regimes |
| `EWMA_LOOKBACK_YEARS` | [1,2,3,4] | Lookback windows for regime shift EWMA |

---

*Not investment advice. For research and educational purposes.*
