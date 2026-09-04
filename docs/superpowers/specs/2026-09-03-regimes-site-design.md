# Regimes on lazyeconomist.com — design

**Date:** 2026-09-03 · **Status:** approved (owner, in chat) · **Target:** `regimes.lazyeconomist.com`

## Goal

Publish the Regimes model (Mulliner, Harvey, Xia, Fang & Van Hemert, *JPM* Feb 2026)
as a public app under The Lazy Economist, visually indistinguishable from the rest of
the site, running as a live Python service on the owner's Mac mini behind the existing
Cloudflare Tunnel.

## Decisions (made with the owner)

| Question | Decision |
|---|---|
| Serving model | Live Python process on the Mac mini (not a static build, not Render). |
| Exposure | Existing dashboard-managed Cloudflare Tunnel; add route `regimes` → `localhost:8503`. |
| Framework | **Streamlit, restyled** to the house look (owner's choice over FastAPI + Jinja). |
| Scope, v1 | Three views: **Now**, **Explore** (any month), **Method** (variables, transform, formula, data). |
| Out of scope | HF portfolio tab (synthetic demo data), sidebar parameter sliders, API-key entry in the UI, extending data history before 1998. |

## Look and feel

Source of truth is the landing page (`jhcwalsh/LazyEconomist/index.html`) and IPS Hub's
`site_src/static/style.css`.

- **Tokens:** bg `#fbfaf7`, bg-soft `#f4f2ec`, ink `#1a1a1a`, ink-soft `#4a4a4a`,
  ink-faint `#8a8780`, rule `#e8e4dc`, accent (rust) `#b8410e`, accent-soft `#f5e6dd`.
- **Type:** Fraunces (display, weight 400, rust *italic* for emphasis), Inter Tight (body),
  JetBrains Mono (kickers, uppercase, letter-spaced; numerals in tables). Google Fonts.
- **Components:** top bar with the `the lazy *economist*.` logo linking to
  `https://lazyeconomist.com` and a nav row; mono kicker `004 · REGIMES`; serif h1 with a
  rust italic phrase; lede paragraph; stat **tiles** (big number, mono label); bordered
  **cards** with 10px radius; pill CTAs; thin rules; the footer line
  "Built quietly · lazyeconomist.com · Not investment advice".
- **Streamlit chrome hidden:** header, footer, hamburger menu, sidebar, "deploy" badge,
  toolbar, and the default tab styling. Theme set in `.streamlit/config.toml`
  (base light, `primaryColor` rust, cream backgrounds); everything else via one injected
  `<style>` block from `web/style.py`.
- **Charts:** Plotly with one house template (`web/charts.py`): cream paper and plot
  background, ink/rust/muted series colours, mono axis labels, no gridlines, no modebar,
  no zoom. Similar months are rust markers; anti-regime months muted-grey markers;
  the masked (excluded) window shaded soft.

## Views and routing

Single Streamlit script `app.py` at the repo root (runbook template). View is chosen by
query parameter `?view=now|explore|method` (default `now`) so links are shareable; the
nav row is plain HTML links. Shared page chrome (top bar, footer) is rendered by helpers
in `web/layout.py`.

### Now
1. Kicker `004 · REGIMES` · h1 "Which past looks most like *now*?" · lede naming the
   current month (latest month with all seven variables) and data-through date.
2. Four tiles: current month; number of similar months (20 % quintile); most similar
   month; regime-shift reading (mean EWMA, percentile of history).
3. Seven Z-scores as a horizontal bar chart (rust for |z| > 2, ink otherwise), with a
   one-line reading per variable.
4. Timeline of the global score against every historical month, similar months marked,
   masked 36-month window shaded, plus a compact table of the 10 most similar months.
5. Anti-regimes: same timeline emphasis inverted, table of the 10 most dissimilar months.
6. Regime-shift indicator: mean of the four EWMAs with the four lookbacks faint, the
   paper's labelled peaks annotated (Oct 08, Jan 09, May 20, Oct 22), current value marked.

### Explore
Month picker (`st.selectbox` over complete months, newest first) with preset links for
the paper's exhibits: Jan 2009, Feb 2020, Apr 2020, Aug 2022. Renders items 3–5 of *Now*
for that month.

### Method
The seven variables with sources (table); the transformation (12-month change over the
rolling 10-year standard deviation of those changes, winsorised at ±3); the distance
formula (eq. 1); regime-shift construction (Exhibit 9, eqs. 2–3, half-life table); the
raw series small-multiples; the paper citation and links; the **data-history limitation**
(free copper/oil series start in the 1990s, so scoring starts Dec 1997 and the paper's
1970s–80s analogues cannot yet appear).

## Data and configuration

- `FRED_API_KEY` from the environment only (compose `env_file: .env`). No UI entry.
  Missing key → the app renders a clear error card, not a traceback.
- Cache directory `cache/` mounted as a compose volume `./cache:/app/cache`.
- **Freshness rule** (`web/data.py`): if the newest cached month is older than 7 days
  relative to today, refetch on next load; the assembled frame and derived scores are
  memoised with `st.cache_data(ttl=24h)`. A pure function `needs_refresh(last_month, today)`
  carries the rule so it is unit-testable.
- Engine unchanged: `data/`, `engine/` modules as fixed on 2026-09-03.

## Deploy (per the Mac mini hosting runbook in `jhcwalsh/Terrarium/docs/`)

- `Dockerfile` (python:3.12-slim, `pip install -r requirements.txt`,
  `streamlit run app.py --server.port=8503 --server.address=0.0.0.0 --server.headless=true`).
- `docker-compose.yml`: service `regimes`, `ports: "8503:8503"`, `restart: unless-stopped`,
  `env_file: .env`, `volumes: ./cache:/app/cache`.
- `.dockerignore`: `.venv`, `__pycache__`, `.git`, `.env`, `.idea`, `tests`.
- Owner steps: `~/apps/regimes` clone + `docker compose up -d --build`; Cloudflare
  published route `regimes.lazyeconomist.com` → `localhost:8503`; no Access policy (public).
- Landing page: add card `004 · Regimes · Live` to `jhcwalsh/LazyEconomist/index.html`
  linking to `https://regimes.lazyeconomist.com`.
- Port 8503 to be recorded in the runbook's allocation table (owner's Terrarium repo).

## Testing

- Existing engine tests unchanged.
- `tests/test_web_data.py`: `needs_refresh` rule.
- `tests/test_web_charts.py`: house template applied (colours, fonts, no gridlines).
- `tests/test_app_smoke.py`: `streamlit.testing.v1.AppTest` renders each view from the
  cached parquet data with no exceptions; skipped if the cache is absent.

## Non-goals / follow-ups

Data-history extension (pre-1990 copper and oil), factor-timing backtest, HF portfolio
section with real data, hover tooltips beyond Plotly defaults.
