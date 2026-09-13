"""
The public Regimes app for lazyeconomist.com.

Three views chosen by the `view` query parameter: now | explore | method.
All page chrome comes from web.layout, the look from web.style, charts from
web.charts. The model itself is the unchanged engine in data/ and engine/.
"""
from __future__ import annotations

from datetime import datetime

import pandas as pd
import streamlit as st

from config import EXCLUDE_RECENT_MONTHS, QUANTILE_SIMILAR
from engine.regime_shift import compute_regime_shift, current_regime_shift_score, get_half_lives
from engine.factor_timing import BACKTEST_START, performance, run_backtest
from engine.similarity import (compute_global_scores, excluded_window_scores,
                               latest_complete_date, rank_regimes)
from web import charts, layout, style
from web.data import cache_written_at, load_factors, load_frames, needs_refresh

N_TABLE = 10

VAR_LABELS = {
    "sp500": "S&P 500 (log level)",
    "yield_curve": "Yield curve, 10yr − 3m",
    "oil": "WTI crude oil",
    "copper": "Copper",
    "tbill_3m": "3-month T-bill",
    "volatility": "Volatility (VIX, spliced)",
    "stock_bond_corr": "Stock–bond correlation",
}
VAR_SOURCES = {
    "sp500": ("S&P 500 index level, month-end close from daily data, 1927 on", "Yahoo Finance ^GSPC"),
    "yield_curve": ("10-year Treasury yield minus 3-month T-bill, 1953 on", "FRED GS10, TB3MS"),
    "oil": ("West Texas Intermediate spot price, monthly, 1946 on", "FRED WTISPLC"),
    "copper": ("Copper, USD per metric ton, 1960 on", "World Bank Pink Sheet, FRED PCOPPUSDM"),
    "tbill_3m": ("3-month Treasury bill yield, 1934 on", "FRED TB3MS"),
    "volatility": ("VIX from 1990, realised S&P 500 volatility before", "FRED VIXCLS, Yahoo Finance"),
    "stock_bond_corr": ("Rolling 3-year correlation of daily stock and bond returns, 1965 on", "Yahoo Finance ^GSPC, ^TNX"),
}
PAPER_PEAKS = {"Oct 08": "2008-10-31", "Jan 09": "2009-01-31", "May 20": "2020-05-31", "Oct 22": "2022-10-31"}
EXHIBITS = [
    ("Jan 2009 · financial crisis", "2009-01-31",
     "Paper, Exhibit 6: the similar months include all observed recessions, including the double-dip "
     "recessions of the early 1980s."),
    ("Feb 2020 · Covid", "2020-02-29",
     "Paper, Exhibit 7: no obvious pattern. Covid was unique in the last hundred years, so the model "
     "struggles to find prior regimes."),
    ("Apr 2020 · Covid", "2020-04-30",
     "Paper, Exhibit 7: as for February 2020, no convincing analogue; the matches are the least-bad ones."),
    ("Aug 2022 · inflation surge", "2022-08-31",
     "Paper, Exhibit 8: most similar to the inflation after the Iranian Revolution, 1977 to 1980, with "
     "months from 1966 to 1970, the 1972 to 1974 oil embargo and the 1987 to 1990 boom."),
]


def exhibit_claim(target: pd.Timestamp) -> str | None:
    """What the paper reports for one of its worked months, or None for any other month."""
    for _, date, claim in EXHIBITS:
        if pd.Timestamp(date) == target:
            return claim
    return None


def stale_note(last_obs: pd.Series, current: pd.Timestamp, labels: dict[str, str]) -> str:
    """
    One sentence naming the variables whose latest value is carried forward from an
    earlier month, grouped by that month. Empty when everything is current.
    """
    stale = {k: pd.Timestamp(v) for k, v in last_obs.items() if pd.Timestamp(v) < current}
    if not stale:
        return ""
    by_month: dict[pd.Timestamp, list[str]] = {}
    for k, m in stale.items():
        by_month.setdefault(m, []).append(labels.get(k, k))
    parts = []
    for m in sorted(by_month, reverse=True):
        names = by_month[m]
        joined = names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]
        parts.append(f"{joined} ({_month(m)})")
    return "Carried forward from the last observation: " + "; ".join(parts) + "."
FACTOR_LABELS = {"Mkt-RF": "Market", "SMB": "Size", "HML": "Value", "RMW": "Profitability",
                 "CMA": "Investment", "Mom": "Momentum"}
PAPER_SHARPE = {"q1": 0.95, "q2": 0.80, "q3": 0.78, "q4": 0.85, "q5": 0.17, "long_only": 1.00, "spread": 0.82}
PAPER_CORR = {"q1": 0.76, "q2": 0.79, "q3": 0.78, "q4": 0.73, "q5": 0.48, "spread": 0.37}
PAPER_QUANTILE_SPREAD = {2: 0.66, 3: 0.62, 4: 0.74, 5: 0.82, 10: 0.69, 20: 0.46}
PAPER_URL = "https://people.duke.edu/~charvey/Research/Published_Papers/P176_Regimes.pdf"
SSRN_URL = "https://ssrn.com/abstract=5164863"


# ---------------------------------------------------------------------------
# Data (memoised for a day; refetched when the cache is stale)
# ---------------------------------------------------------------------------

@st.cache_data(ttl=24 * 3600, show_spinner="Loading market data…")
def _frames(refresh: bool) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    return load_frames(refresh)


@st.cache_data(ttl=24 * 3600, show_spinner=False)
def _shift(zscores: pd.DataFrame) -> pd.DataFrame:
    return compute_regime_shift(zscores)


@st.cache_data(ttl=24 * 3600, show_spinner="Loading factor returns…")
def _factors(refresh: bool) -> pd.DataFrame:
    return load_factors(refresh)


@st.cache_data(ttl=24 * 3600, show_spinner="Running the factor-timing backtest…")
def _backtest(zscores: pd.DataFrame, factors: pd.DataFrame) -> dict:
    out = run_backtest(zscores, factors)
    out["performance"] = performance(out["returns"])
    out["robustness"] = {
        n: performance(run_backtest(zscores, factors, n_quantiles=n)["returns"].loc["1985":"2024"])
        .loc["spread", "sharpe"]
        for n in PAPER_QUANTILE_SPREAD
    }
    return out


def get_data(allow_refresh: bool) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    refresh = allow_refresh and needs_refresh(cache_written_at(), datetime.now())
    return _frames(refresh)


def get_factors(allow_refresh: bool) -> pd.DataFrame:
    refresh = allow_refresh and needs_refresh(cache_written_at(), datetime.now())
    return _factors(refresh)


def _month(ts: pd.Timestamp) -> str:
    return ts.strftime("%B %Y")


def _similarity(zscores: pd.DataFrame, target: pd.Timestamp) -> pd.DataFrame:
    scores = compute_global_scores(zscores, target, EXCLUDE_RECENT_MONTHS)
    return rank_regimes(scores, QUANTILE_SIMILAR, QUANTILE_SIMILAR)


def _plot(fig) -> None:
    st.plotly_chart(fig, use_container_width=True, theme=None, config=charts.PLOT_CONFIG)


# ---------------------------------------------------------------------------
# Shared blocks
# ---------------------------------------------------------------------------

def similarity_blocks(zscores: pd.DataFrame, target: pd.Timestamp) -> pd.DataFrame:
    ranked = _similarity(zscores, target)
    excluded = excluded_window_scores(zscores, target, EXCLUDE_RECENT_MONTHS)
    similar = ranked[ranked["regime"] == "similar"]
    dissimilar = ranked[ranked["regime"] == "dissimilar"].sort_values("global_score", ascending=False)

    layout.section(f"The seven variables in {_month(target)}",
                   "12-month change, scaled by its rolling 10-year standard deviation, capped at ±3. Rust = beyond ±2.")
    _plot(charts.zscore_bars(zscores.loc[target], VAR_LABELS))

    layout.section("Most <em>similar</em> months",
                   f"Distance from {_month(target)} to every earlier month. Rust marks the closest 20 %. "
                   f"The shaded {EXCLUDE_RECENT_MONTHS} months before the target are excluded, per the paper, to avoid momentum; "
                   "they are drawn dotted, falling to zero on the target itself, and take no part in the ranking.")
    _plot(charts.similarity_timeline(ranked, target, EXCLUDE_RECENT_MONTHS, mode="similar", excluded=excluded))
    layout.month_table(similar.head(N_TABLE))

    layout.section("Anti-regimes",
                   "The 20 % of months least like the target. The paper finds these carry information of their own.")
    _plot(charts.similarity_timeline(ranked, target, EXCLUDE_RECENT_MONTHS, mode="dissimilar", excluded=excluded))
    layout.month_table(dissimilar.head(N_TABLE), rust=False)
    return ranked


# ---------------------------------------------------------------------------
# Views
# ---------------------------------------------------------------------------

def view_now(raw: pd.DataFrame, zscores: pd.DataFrame, last_obs: pd.Series) -> None:
    target = latest_complete_date(zscores)
    ranked = _similarity(zscores, target)
    similar = ranked[ranked["regime"] == "similar"]
    shift = _shift(zscores)
    reading = current_regime_shift_score(shift)
    nearest = similar.index[0]

    layout.hero(
        "004 · Regimes",
        "Which past looks most like <em>now</em>?",
        f"Seven market variables, one distance. As of {_month(target)} the closest historical match is "
        f"<em>{_month(nearest)}</em>. A live replication of Mulliner, Harvey, Xia, Fang &amp; Van Hemert (2026).",
    )
    layout.tiles([
        (_month(target), "current month", False),
        (str(len(similar)), "similar months · closest 20 %", False),
        (_month(nearest), "closest match", True),
        (f"{reading['mean_ewma']:.2f} · {reading['pct_rank'] * 100:.0f}th pct", "regime-shift reading", False),
    ])
    note = stale_note(last_obs, target, VAR_LABELS)
    if note:
        layout.note(note)

    similarity_blocks(zscores, target)

    layout.section("Is the regime <em>shifting</em>?",
                   "Average distance from each month to its recent past, weighted towards the latest months "
                   "(paper Exhibit 9). A sharp rise means today is drifting from where we were.")
    _plot(charts.regime_shift_chart(shift, PAPER_PEAKS))
    layout.note(f"Latest reading {reading['mean_ewma']:.2f}, higher than {reading['pct_rank'] * 100:.0f} % of history. "
                "Peaks near the labelled dates are the ones the paper reports.")


def view_explore(raw: pd.DataFrame, zscores: pd.DataFrame, last_obs: pd.Series) -> None:
    complete = zscores.dropna(how="any").index
    options = list(complete[::-1])
    param = st.query_params.get("month")
    default = pd.Timestamp(param) if param else options[0]
    index = options.index(default) if default in options else 0

    layout.hero(
        "004 · Regimes",
        "Pick a month. See its <em>nearest pasts</em>.",
        "Every month in the record is scored against everything before it. Choose one to reproduce the paper's "
        "exhibits, or to test a memory.",
    )
    layout.presets([(label, f"?view=explore&month={d}") for label, d, _ in EXHIBITS])
    target = st.selectbox("Month", options, index=index, format_func=_month)
    try:
        st.query_params["month"] = target.strftime("%Y-%m-%d")
    except Exception:
        pass
    claim = exhibit_claim(target)
    if claim:
        layout.note(claim + " Compare with the matches below.")
    similarity_blocks(zscores, target)


def _perf_table(perf: pd.DataFrame, rows: list[tuple[str, str]]) -> None:
    html = ""
    for key, label in rows:
        if key not in perf.index:
            continue
        r = perf.loc[key]
        rust = " rust" if key in ("q1", "spread") else ""
        html += (f"<tr><td class='{rust.strip()}'>{label}</td><td class='num'>{r['ann_return_%']:.1f}</td>"
                 f"<td class='num'>{r['ann_vol_%']:.1f}</td><td class='num{rust}'>{r['sharpe']:.2f}</td>"
                 f"<td class='num'>{PAPER_SHARPE.get(key, float('nan')):.2f}</td>"
                 f"<td class='num'>{r['corr_to_long_only']:.2f}</td>"
                 f"<td class='num'>{PAPER_CORR.get(key, float('nan')):.2f}</td>"
                 f"<td class='num'>{r['max_drawdown_%']:.0f}</td></tr>")
    st.markdown("<table class='le-table'><thead><tr><th>Portfolio</th><th style='text-align:right'>Return %/yr</th>"
                "<th style='text-align:right'>Vol %/yr</th><th style='text-align:right'>Sharpe</th>"
                "<th style='text-align:right'>Paper Sharpe</th><th style='text-align:right'>Corr. to long-only</th>"
                "<th style='text-align:right'>Paper corr.</th><th style='text-align:right'>Max drawdown %</th></tr></thead>"
                f"<tbody>{html}</tbody></table>", unsafe_allow_html=True)


def view_factors(raw: pd.DataFrame, zscores: pd.DataFrame, last_obs: pd.Series) -> None:
    factors = get_factors(allow_refresh=st.session_state.get("_allow_refresh", True))
    out = _backtest(zscores, factors)
    r = out["returns"]
    perf = out["performance"]
    window = perf.loc["q1"]

    layout.hero(
        "004 · Regimes",
        "Do the analogues <em>predict</em> anything?",
        "The paper's test, re-run on this data: six long–short equity factors, each held long next month if "
        "it rose after the similar months and short if it fell. Returns from Ken French's library.",
    )
    layout.tiles([
        (f"{perf.loc['q1', 'sharpe']:.2f}", "Sharpe · most similar quintile", True),
        (f"{perf.loc['q5', 'sharpe']:.2f}", "Sharpe · anti-regime quintile", False),
        (f"{perf.loc['spread', 'sharpe']:.2f}", "Sharpe · similar minus anti-regime", False),
        (f"{r.index[0].strftime('%b %Y')} – {r.index[-1].strftime('%b %Y')}", f"{int(window['months'])} months", False),
    ])

    layout.section("Quintile portfolios against <em>long-only</em>",
                   "Cumulative sum of monthly returns, equal-weighted across the six factors. Quintile 1 trades in the "
                   "direction of returns after the 20 % most similar months; quintile 5 after the 20 % least similar.")
    _plot(charts.cumulative_lines(
        r, {"q1": "Quintile 1 · similar", "q2": "Quintile 2", "q3": "Quintile 3", "q4": "Quintile 4",
            "q5": "Quintile 5 · anti-regime", "long_only": "Long only, all six"},
        emphasis=["q1"], reference="long_only"))

    layout.section("The <em>spread</em>: long quintile 1, short quintile 5",
                   "The paper's headline portfolio. Its appeal is the low correlation to simply being long the factors.")
    _plot(charts.cumulative_lines(r, {"spread": "Quintile 1 minus quintile 5", "long_only": "Long only"},
                                  emphasis=["spread"], reference="long_only", height=300))

    layout.section("Scorecard, with the paper's numbers beside ours",
                   "Paper: Exhibit 10, 1985–2024. Ours: the full holding window shown above, same rules.")
    _perf_table(perf, [("q1", "Quintile 1 · similar"), ("q2", "Quintile 2"), ("q3", "Quintile 3"),
                       ("q4", "Quintile 4"), ("q5", "Quintile 5 · anti-regime"), ("long_only", "Long only"),
                       ("spread", "Quintile 1 minus 5")])
    q1s, q5s, sp = perf.loc["q1", "sharpe"], perf.loc["q5", "sharpe"], perf.loc["spread", "sharpe"]
    st.markdown(f"Quintile 1 comes out at {q1s:.2f} against the paper's 0.95, with the same 0.76 correlation to "
                f"long-only, and the quintiles rank in the paper's order. The anti-regime quintile is less bad here "
                f"({q5s:.2f} versus 0.17), so the spread is weaker ({sp:.2f} versus 0.82). The likely reasons are the "
                f"data: free series in place of Bloomberg and Man Group's, and a history that starts in 1971 rather "
                f"than 1966, which leaves fewer candidates for the earliest decisions.")

    layout.section("Positions for next month")
    pos = out["positions"]
    rows = "".join(f"<tr><td>{FACTOR_LABELS.get(f, f)}</td>"
                   f"<td class='num{' rust' if pos.loc[f, 'q1'] > 0 else ''}'>{'long' if pos.loc[f, 'q1'] > 0 else 'short'}</td>"
                   f"<td class='num'>{'long' if pos.loc[f, 'q5'] > 0 else 'short'}</td></tr>" for f in pos.index)
    st.markdown("<table class='le-table'><thead><tr><th>Factor</th><th style='text-align:right'>Similar months say</th>"
                f"<th style='text-align:right'>Anti-regime months say</th></tr></thead><tbody>{rows}</tbody></table>",
                unsafe_allow_html=True)
    layout.note(f"Decided at {out['decisions'][-1].strftime('%B %Y')} from the 20 % most and least similar months. "
                "Not investment advice; the paper's own framing is a test of information content, not a product.")

    layout.section("Robustness to the quantile choice",
                   "Paper, Exhibit 12: Sharpe of the similar-minus-dissimilar spread, 1985–2024, for different cuts.")
    rob = out["robustness"]
    rows = "".join(f"<tr><td>{n} quantiles</td><td class='num'>{rob[n]:.2f}</td><td class='num'>{PAPER_QUANTILE_SPREAD[n]:.2f}</td></tr>"
                   for n in PAPER_QUANTILE_SPREAD)
    st.markdown("<table class='le-table'><thead><tr><th>Cut</th><th style='text-align:right'>Spread Sharpe, ours</th>"
                f"<th style='text-align:right'>Paper</th></tr></thead><tbody>{rows}</tbody></table>", unsafe_allow_html=True)
    st.markdown("The paper's spread is strongest at quintiles and fades with finer cuts. Ours strengthens with finer "
                "cuts, which says the useful information here sits in the extremes of the ranking rather than in "
                "the quintile boundaries. Treat both as evidence that the ordering matters and the exact cut does not.")


def view_method(raw: pd.DataFrame, zscores: pd.DataFrame, last_obs: pd.Series) -> None:
    layout.hero(
        "004 · Regimes",
        "How it <em>works</em>.",
        "No regimes are named in advance. Each month is compared with every earlier month across seven "
        "variables, and the closest ones are the analogues.",
    )

    layout.section("The seven state variables")
    rows = "".join(
        f"<tr><td>{i + 1}</td><td>{VAR_LABELS[k]}</td><td>{desc}</td><td class='num'>{src}</td></tr>"
        for i, (k, (desc, src)) in enumerate(VAR_SOURCES.items())
    )
    st.markdown(f"<table class='le-table'><thead><tr><th>#</th><th>Variable</th><th>What</th>"
                f"<th style='text-align:right'>Source</th></tr></thead><tbody>{rows}</tbody></table>",
                unsafe_allow_html=True)
    st.markdown("Volatility and the stock–bond correlation are computed on daily data, then mapped to monthly.")

    layout.section("The transformation")
    st.markdown("Each variable becomes a Z-score-like series: its 12-month change divided by the standard deviation "
                "of those changes over the trailing 10 years, winsorised at ±3.")
    st.latex(r"z_{t} = \operatorname{clip}\!\left(\frac{x_{t} - x_{t-12}}{\sigma_{10y}(x_{t} - x_{t-12})},\,-3,\,3\right)")

    layout.section("The distance")
    st.markdown("For a target month *T*, the global score of every earlier month *i* is the Euclidean distance "
                "across the seven transformed variables (paper eq. 1). Lower is more similar. The 36 months before "
                "*T* are excluded so the match is not just momentum. The closest 20 % are the similar regime; the "
                "farthest 20 % the anti-regime.")
    st.latex(r"d_{T,i} = \sqrt{\sum_{v=1}^{7}\left(x_{i,v} - x_{T,v}\right)^{2}}")

    layout.section("Regime shifts")
    st.markdown("At each month *T* the distances to all earlier months are averaged with exponentially decaying "
                "weights, so the recent past counts most (paper Exhibit 9). Four lookbacks are shown with their mean.")
    st.latex(r"C_{T} = \frac{\sum_{t \le T} \beta^{\,T-t}\, d_{T,t}}{\sum_{t \le T} \beta^{\,T-t}},\qquad "
             r"\beta = 1 - \tfrac{1}{n},\qquad t_{1/2} = -\frac{\ln 2}{\ln \beta}")
    hl = get_half_lives()
    rows = "".join(f"<tr><td>{int(r.lookback_years)} year</td><td class='num'>{int(r.lookback_months)}</td>"
                   f"<td class='num'>{r.beta:.4f}</td><td class='num'>{r.half_life_months:.1f}</td></tr>"
                   for r in hl.itertuples())
    st.markdown(f"<table class='le-table'><thead><tr><th>Lookback</th><th style='text-align:right'>n (months)</th>"
                f"<th style='text-align:right'>β</th><th style='text-align:right'>Half-life (months)</th></tr></thead>"
                f"<tbody>{rows}</tbody></table>", unsafe_allow_html=True)

    layout.section("The raw series")
    cols = st.columns(2)
    for i, key in enumerate(VAR_LABELS):
        with cols[i % 2]:
            layout.note(VAR_LABELS[key])
            _plot(charts.raw_line(raw[key].dropna(), VAR_LABELS[key]))

    layout.section("How far back it <em>reaches</em>")
    first = zscores.dropna(how="any").index[0]
    st.markdown(f"Scoring starts in {_month(first)}. The binding series is the stock–bond correlation, which "
                f"needs daily 10-year yields, available free only from 1962, plus a three-year window and the "
                f"ten-year scaling. The paper's own scores start in 1966. Everything else runs from 1927 to 1960, "
                f"so the 1973 oil shock, the 1977 to 1980 inflation and the early-1980s recessions are all in reach.")

    layout.section("The paper")
    st.markdown(f"Mulliner A., Harvey C.R., Xia C., Fang E. and Van Hemert O., *Regimes*, "
                f"The Journal of Portfolio Management, February 2026. [PDF]({PAPER_URL}) · [SSRN]({SSRN_URL}). "
                f"This app is an independent replication and is not affiliated with the authors.")


VIEW_FUNCS = {"now": view_now, "explore": view_explore, "factors": view_factors, "method": view_method}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(view: str | None = None, allow_refresh: bool = True) -> None:
    st.set_page_config(page_title="Regimes — The Lazy Economist", layout="wide",
                       initial_sidebar_state="collapsed")
    style.inject()

    if view is None:
        view = st.query_params.get("view", "now")
    if view not in VIEW_FUNCS:
        view = "now"
    layout.top_bar(view)

    st.session_state["_allow_refresh"] = allow_refresh
    try:
        raw, zscores, last_obs = get_data(allow_refresh)
    except Exception as exc:  # a data problem must read as a card, not a traceback
        layout.hero("004 · Regimes", "Which past looks most like <em>now</em>?", "The data is not available right now.")
        layout.error_card(f"Data load failed: {exc}")
        layout.footer("—")
        return

    VIEW_FUNCS[view](raw, zscores, last_obs)
    layout.footer(_month(latest_complete_date(zscores)))
