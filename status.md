# Regimes: state of play

_Updated 2026-09-05. Latest commit 5ef7eec on main._

## What exists

**A replication of** Mulliner, Harvey, Xia, Fang and Van Hemert, "Regimes",
*Journal of Portfolio Management*, February 2026, plus a public app in the
lazyeconomist.com house style at https://regimes.lazyeconomist.com.

| Layer | Status |
|---|---|
| Data (`data/`) | FRED, Yahoo, World Bank Pink Sheet and Ken French loaders. Parquet cache (`*.v2.parquet`), 7-day refresh. History complete from Dec 1970 (670 months); binding constraint is daily ^TNX from 1962. |
| Engine (`engine/`) | Z-scores, Euclidean similarity with 36-month exclusion, EWMA regime-shift score (1 to 4 year lookbacks), factor-timing backtest. All paper rules pinned by tests. |
| App (`web/`) | Streamlit, restyled. Views: Now, Explore any month, Factors, Method. Routed by `?view=`. |
| Research dashboard (`dashboard/`) | Original multi-tab dashboard including the hedge fund portfolio tab. Kept as-is. |
| Docs | `docs/methodology.md` (referenced summary), `docs/superpowers/specs/2026-09-03-regimes-site-design.md` (approved design), README. |
| Tests | 48 passing (`pytest`). Includes replication tests for the paper's Jan 2009 and Aug 2022 exhibits. |
| CI | GitHub Actions: pytest on push and PR; daily uptime check of the live URL at 13:17 UTC. |
| Hosting | Docker on the Mac mini, port 8503, restart unless-stopped, Cloudflare Tunnel route. Landing-page card on lazyeconomist.com is live. Deploy: `ssh` to the mini, `git pull`, `docker compose up -d --build`. |

## Replication results

**Similarity exhibits reproduce.** Jan 2009 finds 1980 to 1982, 1990 to 1991
and 2001 to 2002; Aug 2022 finds 1977 to 1980 and 1972 to 1974, as in the paper.
Regime-shift peaks land on 1973, 1975, 1981, 1987, 2001, 2008, 2021 and 2022.

**Factor timing reproduces the similar side, not the anti-regime side**
(1985 to 2024, six French factors, quintiles):

| Portfolio | Ours | Paper |
|---|---|---|
| Quintile 1 Sharpe | 0.87 | 0.95 |
| Quintile 1 correlation to long-only | 0.76 | 0.76 |
| Quintile 5 Sharpe | 0.44 | 0.17 |
| Long-only Sharpe | 0.97 | 1.00 |
| Q1 minus Q5 spread Sharpe | 0.34 | 0.82 |

Our spread strengthens with finer quantile cuts (0.59 at deciles) while the
paper's peaks at quintiles. Likely causes: free data in place of Bloomberg and
Man Group series (copper and the pre-1990 volatility splice especially), history
starting 1971 rather than 1966, and unreported details of the exclusion mask.

## Known limitations

- Copper on FRED lags about two months; the Now page carries the last value forward and says so.
- The Factors page runs the backtest plus six robustness runs on first load each day, roughly 40 seconds, then serves from a 24-hour cache.
- Pre-1990 volatility is a realised-vol splice onto VIX with no level adjustment.
- The 10-year yield before 1962 is unavailable daily, which caps the history.

## Next steps

Ordered by expected value.

1. **Close the anti-regime gap.** Test whether the Q5 result is data-driven: swap in
   LBMA or Bloomberg-equivalent copper if available, and level-adjust the volatility
   splice. Re-run the scorecard after each change and record the deltas.
2. **Robustness view.** Extend the Factors page with the paper's Exhibit 11 and 13
   analogues: sensitivity to the z-score lookback window and to the exclusion window.
   The quantile table already exists; reuse `_backtest` with parameters.
3. **Earlier bond history.** Splice monthly GS10 (1953) or Shiller long rates
   before 1962 to push the first complete month back toward the paper's 1966 start.
   This also thins the candidate-set excuse for early decisions.
4. **Monthly data refresh check.** Add a small CI job that fetches the latest month
   and asserts the cache still builds, so an upstream format change is caught before
   the live app shows an error card.
5. **Explore page polish.** Link exhibit months from the Method page, and show the
   paper's own similar-month list beside ours on the exhibit months.
6. **Retire the research dashboard** once the public app covers what it is used for,
   or move the hedge fund portfolio tab into a fifth view if it is wanted publicly.

## Operating notes

- Secrets: `FRED_API_KEY` from environment only, in the mini's `~/apps/regimes/.env`. Never in the repo or the UI.
- Redeploy after every push that touches the app; the container does not auto-pull.
- Runbook entry for this app is in jhcwalsh/Terrarium, `docs/mac-mini-hosting-runbook2.md`.
