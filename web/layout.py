"""Page chrome and small HTML components in the house style."""
from __future__ import annotations

from html import escape

import pandas as pd
import streamlit as st

VIEWS = [("now", "Now"), ("explore", "Explore"), ("method", "Method")]
HOME = "https://lazyeconomist.com"


def top_bar(active: str) -> None:
    links = "".join(
        f'<li><a href="?view={key}" target="_self" class="{"active" if key == active else ""}">{label}</a></li>'
        for key, label in VIEWS
    )
    st.markdown(
        f'<div class="le-nav"><a class="le-logo" href="{HOME}">the lazy <em>economist</em><span class="dot">.</span></a>'
        f'<ul>{links}<li><a href="{HOME}#apps">All apps</a></li></ul></div>',
        unsafe_allow_html=True,
    )


def hero(kicker: str, title_html: str, lede_html: str) -> None:
    st.markdown(
        f'<div class="le-kicker">{escape(kicker)}</div>'
        f'<h1 class="le-h1">{title_html}</h1>'
        f'<p class="le-lede">{lede_html}</p>',
        unsafe_allow_html=True,
    )


def tiles(items: list[tuple[str, str, bool]]) -> None:
    """items: (number, label, rust?)"""
    html = "".join(
        f'<div class="le-tile"><div class="num{" rust" if rust else ""}">{escape(num)}</div>'
        f'<div class="lbl">{escape(lbl)}</div></div>'
        for num, lbl, rust in items
    )
    st.markdown(f'<div class="le-tiles">{html}</div>', unsafe_allow_html=True)


def section(title_html: str, note_text: str = "") -> None:
    st.markdown(f"## {title_html}", unsafe_allow_html=True)
    if note_text:
        note(note_text)


def note(text: str) -> None:
    st.markdown(f'<p class="le-note">{escape(text)}</p>', unsafe_allow_html=True)


def error_card(text: str) -> None:
    st.markdown(f'<div class="le-error">{escape(text)}</div>', unsafe_allow_html=True)


def month_table(df: pd.DataFrame, score_col: str = "global_score", rust: bool = True) -> None:
    rows = "".join(
        f'<tr><td>{d.strftime("%B %Y")}</td><td class="num">{int(r["rank"])}</td>'
        f'<td class="num{" rust" if rust else ""}">{r[score_col]:.2f}</td></tr>'
        for d, r in df.iterrows()
    )
    st.markdown(
        '<table class="le-table"><thead><tr><th>Month</th><th style="text-align:right">Rank</th>'
        f'<th style="text-align:right">Distance</th></tr></thead><tbody>{rows}</tbody></table>',
        unsafe_allow_html=True,
    )


def presets(items: list[tuple[str, str]]) -> None:
    """items: (label, href)"""
    html = "".join(f'<a href="{href}" target="_self">{escape(label)}</a>' for label, href in items)
    st.markdown(f'<div class="le-presets">{html}</div>', unsafe_allow_html=True)


def footer(data_through: str) -> None:
    st.markdown(
        f'<div class="le-footer"><div class="mark">the lazy <em>economist</em>. — © 2026</div>'
        f'<div class="meta">Data through {escape(data_through)} · FRED &amp; Yahoo Finance · Not investment advice · '
        f'<a href="{HOME}">lazyeconomist.com</a></div></div>',
        unsafe_allow_html=True,
    )
