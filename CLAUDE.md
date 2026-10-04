# Regimes

A live replication of Mulliner, Harvey, Xia, Fang and Van Hemert, "Regimes" (Journal of Portfolio
Management, Feb 2026): it finds the historical months most like (and least like) today across seven
economic state variables, scores regime shifts, and runs the paper's factor-timing backtest. The public
Streamlit app is live at `https://regimes.lazyeconomist.com`; an older research dashboard is kept as-is.
`status.md` is the state of play and next steps.

## Run and test

    pip install -r requirements.txt
    export FRED_API_KEY=...                 # needed to fetch data; tests do not need it
    streamlit run app.py                    # public app; views by ?view=now|explore|factors|method
    streamlit run dashboard/app.py          # research dashboard (sidebar controls, portfolio tab)
    python -m pytest -q                     # ~10 s, no network

Without a parquet cache in `cache/`, the app-smoke and exhibit-replication tests skip themselves
(`tests/test_app_smoke.py`, `tests/test_replication.py`), so a clean checkout runs 43 passed, 8 skipped
(checked 4 Oct 2026 on Python 3.12). Run them with a filled cache before claiming a data or engine
change replicates. CI (`.github/workflows/tests.yml`) runs the same offline suite on push to `main` and on PRs.

## Layout

    config.py              every parameter in one place: FRED series, transforms, exclusion window, quantiles, cache path
    app.py                 public app entry point; just calls web.app.main
    web/                   public app: app.py (views), data.py (cache freshness, memoised loads), charts.py, layout.py, style.py
    data/                  Python package, not data: fetcher.py (FRED, yfinance, World Bank copper), french.py (Ken French factors), transformer.py
    engine/                similarity.py, regime_shift.py (EWMA, Exhibit 9), factor_timing.py (Exhibits 10-13), strategy_timing.py
    portfolio/             backtest, optimiser and risk for the research dashboard's hedge fund tab
    dashboard/app.py       the original research dashboard
    tests/                 pytest; replication tests pin the paper's Jan 2009 and Aug 2022 exhibits
    docs/methodology.md    referenced summary of the method; the approved site design is in docs/superpowers/specs/
    cache/                 parquet cache, git-ignored, mounted as a volume in Docker

## Conventions and gotchas

- `FRED_API_KEY` comes from the environment only (`.env` on the Mini, shape in `.env.example`). Never in
  the repo or the public UI. The research dashboard does have a sidebar key field; the public app does not.
- Cache files are `*.v2.parquet` (`data/fetcher.CACHE_VERSION`); the public app refetches when the cache is
  over 7 days old (`web/data.MAX_CACHE_AGE_DAYS`). Bump the version when a series' source changes,
  so stale cache files are ignored rather than reused.
- Each variable: 12-month difference, rolling 10-year z-score, winsorised at +/-3. History is complete
  from Dec 1970; daily ^TNX from 1962 is the binding constraint.
- Copper on FRED lags about two months; the Now page carries the last value forward and says so.
- The Factors view runs the backtest plus six robustness runs on first load each day (~40 s), then
  serves from a 24-hour cache.
- Known replication gap: the similar (Q1) side reproduces, the anti-regime (Q5) side does not. See
  `status.md` before "fixing" it; the likely causes are data, not code.
- The research dashboard's portfolio tab runs on synthetic demo hedge fund returns
  (`portfolio.backtest.demo_strategy_returns`), and labels them as such.

## Deploy

Docker on the Mac mini: `~/apps/regimes`, `docker compose up -d --build`, port **8503**, `.env` holding
`FRED_API_KEY`, `restart: unless-stopped` (needed for reboot recovery), `./cache` mounted so a rebuild
keeps the data. Cloudflare tunnel route `regimes.lazyeconomist.com` -> `localhost:8503`. The container
does not auto-pull, so redeploy after every push that touches the app. `.github/workflows/uptime.yml`
pings the live URL daily at 13:17 UTC and is the only alerting. `status.md` points to the runbook copy
in jhcwalsh/Terrarium `docs/mac-mini-hosting-runbook2.md`.

<!-- BEGIN shared: James's house rules. Identical in every repo; change it everywhere, not here alone. -->
## How James works

This section and the two after it are the same in all of James's repos. **Where this repo's
own rules above are stricter or more specific - about branches, pushing, deploying, editing,
or spending on model calls - this repo's rules win.**

**Talking to James**
- Plain English, and short. Lead with the outcome. If something failed or was not checked, say that first.
- Don't take yourself too seriously, and make suggestions: if there is a better way or a next step worth doing, say so in a line.
- Numbers go in a small table, parallel items in a list. No walls of text.

**Just do it, or ask first**
- Just do anything reversible that the request implies: reading, editing, running tests, committing.
- Ask first before deploying; touching live data, money, or other people's details; deleting
  files or branches you did not create; adding a heavy dependency; or widening the job beyond
  what was asked.
- If a request is ambiguous and the readings lead to different work, ask one short question.
  Otherwise pick the sensible reading, say which one, and carry on.

**Done means verified**
- Run the project's tests before saying done, and say what passed and what did not. Never
  skip, xfail or delete a failing test to get green; report it.
- For anything with a UI, see it working in a browser, not just building.
- Regenerate generated files rather than hand-editing them.
- Secrets stay out of git: `.env` is local, `.env.example` shows its shape.

## Git, PRs and deploys

- Commit message: a plain sentence saying what changed, then a short body saying why. One topic per commit.
- The default branch is `main` in some repos and `master` in others; check before assuming.
- Local sessions on James's machine: committing to the default branch is normal for work he
  asked for, unless this repo's rules say otherwise. Push only when he says so.
- Cloud sessions: work on the session's `claude/...` branch, open a draft PR, never push to the
  default branch.
- **In some repos a push to the default branch IS a deploy** (the Mini polls and rebuilds). Read
  this repo's Deploy section before any push to it.
- Apps run on the Mac mini (`JHCW-mini.local`) behind a Cloudflare tunnel. Where an app is
  deployed by hand it is `deploy <app>` from James's PowerShell. Never deploy unless asked. A
  cloud session cannot reach the Mini: say so and give him the command.
- One host port per app on the Mini: 8501 test, 8502 terrarium, 8503 regimes, 8504 glidepath,
  8505 states, 8506 brief, 8507 lampmold. A new app takes the next free port, checked on the box.
  `MacMiniHosting/mac-mini-hosting-runbook2.md` on James's machine is the authority for the Mini.

## Session notes

Keep SESSION_NOTES.md in the repo root up to date.

**When:** Before every git commit, update it and include it in
the same commit. Also update it when I say "wrap up".

**What:** Newest entry at the top, dated. Cover:
- Done: what changed, and which files
- Why: key decisions and reasoning
- Broken / unsure: known bugs, hacks, untested bits
- Next: the 1–3 most sensible next steps

**How much:** A new entry for meaningful work, or a one-line
append to today's entry for small fixes. Each entry under ~20
lines, plain English, no code dumps.

**Housekeeping:** Keep the last 10 entries in full and fold
older ones into a short "History" section at the bottom.

**Enforced:** `.claude/hooks/require-session-notes.sh` runs before every Bash command
(`.claude/settings.json`) and refuses a commit in this repo unless SESSION_NOTES.md is staged,
or staged by the same command. Merge commits pass, and so does a command that only mentions a
commit. If it blocks you, write the entry; do not work around the hook.
<!-- END shared -->
