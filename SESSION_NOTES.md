# Session notes

## 2026-10-30 — Site down: copper source 403, refresh wiped the cache

**Done**
- Diagnosed regimes.lazyeconomist.com showing "Data load failed: HTTP Error 403". The weekly
  refresh (27 Oct) deleted the whole cache, then the World Bank Pink Sheet download returned 403:
  it now refuses Python's default client. Nothing could load until a human stepped in.
- Copper history (1960-2024) now ships in `data/copper_history.csv`; no runtime World Bank
  download (`data/fetcher.py`). FRED still supplies copper live. Values match the old series to 1e-7.
- A failed refresh now puts the old cache back and the app serves it (`fetch_all` in
  `data/fetcher.py`, `load_frames`/`load_factors` in `web/data.py`).
- Security pass: a junk `?month=` crashed Explore with a public traceback; fixed in `web/app.py`.
  `.streamlit/config.toml` now shows only the exception type to visitors.
- Tests: `tests/test_data_resilience.py`, bad-month cases in `tests/test_app_smoke.py`. 58 pass.

**Why**
- The Pink Sheet snapshot is frozen (last modified Jan 2025) and only pre-1992 copper comes from
  it, so bundling it removes a fragile dependency for no loss. Faking a browser user-agent was
  rejected: it dodges the publisher's bot check.

**Broken / unsure**
- Not deployed yet. Verified on the Mini in a throwaway container only.
- Container runs as root; port 8503 is bound on all interfaces, not just localhost.

**Next**
- Push and `deploy regimes`, then check the live site in a browser.
- Run the container as a non-root user; bind 127.0.0.1:8503 if cloudflared runs on the host.
- Add a health check that alerts when the error card is showing.
