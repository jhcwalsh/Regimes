"""
Plotly charts in the house style. Pure functions: DataFrame in, Figure out.

Tokens mirror lazyeconomist.com: cream ground, ink lines, rust emphasis,
muted grey for history, mono tick labels, no gridlines.
"""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

BG = "#fbfaf7"
BG_SOFT = "#f4f2ec"
INK = "#1a1a1a"
INK_SOFT = "#4a4a4a"
MUTED = "#8a8780"
RULE = "#e8e4dc"
RUST = "#b8410e"
RUST_SOFT = "#f5e6dd"

MONO = "JetBrains Mono, ui-monospace, Consolas, monospace"
SERIF = "Fraunces, Georgia, serif"

PLOT_CONFIG = {"displayModeBar": False, "scrollZoom": False, "staticPlot": False}


def house_layout(height: int = 340) -> dict:
    axis = dict(showgrid=False, zeroline=False, linecolor=RULE, tickfont=dict(family=MONO, size=11, color=MUTED),
                title_font=dict(family=MONO, size=11, color=MUTED))
    return dict(
        paper_bgcolor=BG, plot_bgcolor=BG,
        font=dict(family=MONO, size=11, color=INK_SOFT),
        margin=dict(l=56, r=16, t=16, b=44),
        height=height,
        xaxis=axis, yaxis=axis,
        legend=dict(orientation="h", y=-0.18, x=0, font=dict(family=MONO, size=11, color=MUTED), bgcolor="rgba(0,0,0,0)"),
        hoverlabel=dict(bgcolor=BG_SOFT, bordercolor=RULE, font=dict(family=MONO, size=11, color=INK)),
        dragmode=False,
    )


def _fig(height: int = 340) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(house_layout(height))
    return fig


def zscore_bars(z: pd.Series, labels: dict[str, str] | None = None, extreme: float = 2.0) -> go.Figure:
    """Horizontal bars of the current Z-scores; |z| above `extreme` in rust."""
    labels = labels or {}
    names = [labels.get(k, k) for k in z.index]
    colours = [RUST if abs(v) > extreme else INK for v in z.values]
    fig = _fig(height=40 * len(z) + 40)
    fig.add_trace(go.Bar(
        x=z.values, y=names, orientation="h",
        marker=dict(color=colours), name="Z-score",
        text=[f"{v:+.2f}" for v in z.values], textposition="outside",
        textfont=dict(family=MONO, size=11, color=INK),
        hovertemplate="%{y}: %{x:+.2f}<extra></extra>",
    ))
    fig.add_vline(x=0, line_color=RULE)
    for lim in (-3, 3):
        fig.add_vline(x=lim, line_color=RULE, line_dash="dot")
    fig.update_layout(showlegend=False, margin=dict(l=210, r=24, t=8, b=44),
                      xaxis=dict(range=[-3.6, 3.6], title="Z-score"),
                      yaxis=dict(autorange="reversed", tickfont=dict(family=MONO, size=11, color=INK)))
    return fig


def similarity_timeline(ranked: pd.DataFrame, target: pd.Timestamp, exclude_months: int,
                        mode: str = "similar") -> go.Figure:
    """
    Global score for every historical month against `target`.
    mode='similar' emphasises the similar quintile in rust; mode='dissimilar'
    emphasises the anti-regime quintile. The masked window before the target
    is shaded.
    """
    series = ranked["global_score"].sort_index()
    fig = _fig()
    fig.add_trace(go.Scatter(
        x=series.index, y=series.values, mode="lines", name="Global score",
        line=dict(color=MUTED, width=1), hovertemplate="%{x|%b %Y}: %{y:.2f}<extra></extra>",
    ))
    if mode == "similar":
        pts, name, colour = ranked[ranked["regime"] == "similar"], "Similar months", RUST
    else:
        pts, name, colour = ranked[ranked["regime"] == "dissimilar"], "Anti-regime months", INK
    fig.add_trace(go.Scatter(
        x=pts.index, y=pts["global_score"], mode="markers", name=name,
        marker=dict(color=colour, size=6), hovertemplate="%{x|%b %Y}: %{y:.2f}<extra></extra>",
    ))
    start = target - pd.DateOffset(months=exclude_months)
    fig.add_shape(type="rect", x0=start, x1=target, y0=0, y1=1, xref="x", yref="paper",
                  fillcolor=RULE, opacity=0.5, line_width=0, layer="below")
    fig.update_layout(yaxis=dict(title="Distance (lower = more similar)", rangemode="tozero"))
    return fig


def regime_shift_chart(ewma: pd.DataFrame, peaks: dict[str, str] | None = None) -> go.Figure:
    """Mean of the four EWMAs in ink, each lookback faint, paper peaks annotated."""
    fig = _fig(height=360)
    for col in [c for c in ewma.columns if c.startswith("ewma_")]:
        yrs = col.split("_")[1].replace("yr", "")
        fig.add_trace(go.Scatter(
            x=ewma.index, y=ewma[col], mode="lines", name=f"{yrs}-year lookback",
            line=dict(color=MUTED, width=1, dash="dot"), opacity=0.6,
            hovertemplate="%{x|%b %Y}: %{y:.2f}<extra></extra>",
        ))
    fig.add_trace(go.Scatter(
        x=ewma.index, y=ewma["mean_ewma"], mode="lines", name="Mean of four",
        line=dict(color=INK, width=2), hovertemplate="%{x|%b %Y}: %{y:.2f}<extra></extra>",
    ))
    for i, (label, date) in enumerate((peaks or {}).items()):
        ts = pd.Timestamp(date)
        if ts in ewma.index:
            fig.add_annotation(x=ts, y=float(ewma.loc[ts, "mean_ewma"]), text=label, showarrow=True,
                               arrowhead=0, arrowcolor=MUTED, ax=0, ay=-28 - 18 * (i % 2),
                               font=dict(family=MONO, size=10, color=RUST))
    last = ewma.index[-1]
    fig.add_trace(go.Scatter(
        x=[last], y=[float(ewma["mean_ewma"].iloc[-1])], mode="markers", name="Now",
        marker=dict(color=RUST, size=9), hovertemplate="%{x|%b %Y}: %{y:.2f}<extra></extra>",
    ))
    fig.update_layout(yaxis=dict(title="EWMA of distance to the recent past", rangemode="tozero"))
    return fig


def cumulative_lines(returns: pd.DataFrame, labels: dict[str, str], emphasis: list[str] = (),
                     reference: str | None = None, height: int = 360) -> go.Figure:
    """Cumulative sum of monthly percent returns, one line per column of `labels`.
    Emphasised columns in rust, the reference dotted in ink, the rest muted."""
    fig = _fig(height=height)
    for col, label in labels.items():
        if col not in returns:
            continue
        y = returns[col].fillna(0).cumsum()
        if col in emphasis:
            line = dict(color=RUST, width=2.2)
        elif col == reference:
            line = dict(color=INK, width=1.6, dash="dot")
        else:
            line = dict(color=MUTED, width=1)
        fig.add_trace(go.Scatter(x=y.index, y=y.values, mode="lines", name=label, line=line,
                                 hovertemplate="%{x|%b %Y}: %{y:.0f}%<extra></extra>"))
    fig.add_hline(y=0, line_color=RULE)
    fig.update_layout(yaxis=dict(title="Cumulative return, % (sum of monthly)"))
    return fig


def raw_line(series: pd.Series, title: str, height: int = 180) -> go.Figure:
    fig = _fig(height=height)
    fig.add_trace(go.Scatter(x=series.index, y=series.values, mode="lines", name=title,
                             line=dict(color=INK, width=1), hovertemplate="%{x|%b %Y}: %{y:.2f}<extra></extra>"))
    fig.update_layout(showlegend=False, margin=dict(l=48, r=12, t=8, b=28))
    return fig
