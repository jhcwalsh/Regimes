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
from engine.similarity import compute_global_scores, latest_complete_date, rank_regimes
from web import charts, layout, style
from web.data import cache_written_at, load_frames, needs_refresh

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


def get_data(allow_refresh: bool) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    refresh = allow_refresh and needs_refresh(cache_written_at(), datetime.now())
    return _frames(refresh)


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
    similar = ranked[ranked["regime"] == "similar"]
    dissimilar = ranked[ranked["regime"] == "dissimilar"].sort_values("global_score", ascending=False)

    layout.section(f"The seven variables in {_month(target)}",
                   "12-month change, scaled by its rolling 10-year standard deviation, capped at ±3. Rust = beyond ±2.")
    _plot(charts.zscore_bars(zscores.loc[target], VAR_LABELS))

    layout.section("Most <em>similar</em> months",
                   f"Distance from {_month(target)} to every earlier month. Rust marks the closest 20 %. "
                   f"The shaded {EXCLUDE_RECENT_MONTHS} months before the target are excluded, per the paper, to avoid momentum.")
    _plot(charts.similarity_timeline(ranked, target, EXCLUDE_RECENT_MONTHS, mode="similar"))
    layout.month_table(similar.head(N_TABLE))

    layout.section("Anti-regimes",
                   "The 20 % of months least like the target. The paper finds these carry information of their own.")
    _plot(charts.similarity_timeline(ranked, target, EXCLUDE_RECENT_MONTHS, mode="dissimilar"))
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


VIEW_FUNCS = {"now": view_now, "explore": view_explore, "method": view_method}


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

    try:
        raw, zscores, last_obs = get_data(allow_refresh)
    except Exception as exc:  # a data problem must read as a card, not a traceback
        layout.hero("004 · Regimes", "Which past looks most like <em>now</em>?", "The data is not available right now.")
        layout.error_card(f"Data load failed: {exc}")
        layout.footer("—")
        return

    VIEW_FUNCS[view](raw, zscores, last_obs)
    layout.footer(_month(latest_complete_date(zscores)))
