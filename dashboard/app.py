"""
Streamlit dashboard for live regime identification.

Run with:
    streamlit run dashboard/app.py

Sections:
  1. Current Z-scores for all 7 state variables
  2. Most similar historical periods (with timeline chart)
  3. Anti-regime (most dissimilar) periods
  4. Regime shift EWMA detector
  5. Raw state variable charts
"""

import os
import sys
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import date

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from config import (
    FRED_API_KEY, QUANTILE_SIMILAR, QUANTILE_DISSIMILAR, N_DISPLAY_SIMILAR,
    EXCLUDE_RECENT_MONTHS, EWMA_LOOKBACK_YEARS
)
from data.fetcher import fetch_all
from data.transformer import compute_zscore, current_zscores, describe_transformed
from engine.similarity import compute_global_scores, rank_regimes, get_similar_periods, get_dissimilar_periods
from engine.regime_shift import compute_ewma_regime_shift, get_half_lives, current_regime_shift_score, detect_regime_shift_events

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Regime Identification",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.title("Regimes")
st.sidebar.caption("Mulliner, Harvey, Xia, Fang & Van Hemert (JPM 2026)")
st.sidebar.divider()

api_key_input = st.sidebar.text_input(
    "FRED API Key",
    value=FRED_API_KEY,
    type="password",
    help="Get a free key at fred.stlouisfed.org",
)
if api_key_input:
    os.environ["FRED_API_KEY"] = api_key_input

refresh = st.sidebar.button("Refresh Data", type="primary")
if refresh:
    st.cache_data.clear()  # Wipe Streamlit in-memory cache so fresh data is loaded

st.sidebar.divider()
quantile_q = st.sidebar.slider(
    "Similar / Dissimilar quantile",
    min_value=0.10, max_value=0.30, value=QUANTILE_SIMILAR, step=0.05,
    help="Fraction of history used as 'similar' (Q1) and 'dissimilar' (Q5)",
)
exclude_mo = st.sidebar.slider(
    "Exclude recent months",
    min_value=0, max_value=60, value=EXCLUDE_RECENT_MONTHS, step=6,
    help="Months before today excluded from regime candidates (avoids momentum)",
)

# ---------------------------------------------------------------------------
# Data loading — NOT cached via @st.cache_data so API key changes take effect
# immediately. We store in session_state to avoid refetching on every widget
# interaction.
# ---------------------------------------------------------------------------
if not api_key_input:
    st.warning("Enter your FRED API Key in the sidebar to load live data.")
    st.stop()

if refresh or "raw" not in st.session_state:
    with st.spinner("Fetching market data..."):
        try:
            raw     = fetch_all(refresh_cache=refresh)
            zscores = compute_zscore(raw)
            st.session_state["raw"]     = raw
            st.session_state["zscores"] = zscores
        except Exception as e:
            st.error(f"Data load failed: {e}")
            st.stop()

raw     = st.session_state["raw"]
zscores = st.session_state["zscores"]

# Diagnostic expander — helps debug missing values
with st.sidebar.expander("Diagnostics"):
    st.write("**Raw data tail (last 2 rows):**")
    st.dataframe(raw.tail(2).T.round(3))
    cur_zs = current_zscores(zscores)
    st.write("**Current Z-scores:**")
    st.dataframe(cur_zs.round(3).to_frame("Z-Score"))

zs_clean = zscores.dropna(how="all")
target_date = zs_clean.index[-1]

# ---------------------------------------------------------------------------
# Compute similarity scores
# ---------------------------------------------------------------------------
with st.spinner("Computing similarity scores..."):
    scores = compute_global_scores(zscores, target_date, exclude_mo)
    ranked = rank_regimes(scores, quantile_q, quantile_q)


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("Economic Regime Identification")
st.caption(
    f"Nonparametric similarity model — current date: **{target_date.strftime('%B %Y')}** "
    f"| Data through: {raw.index[-1].strftime('%B %Y')}"
)

col_status1, col_status2, col_status3, col_status4 = st.columns(4)

current_zs = current_zscores(zscores)
n_similar   = (ranked["regime"] == "similar").sum()
n_dissim    = (ranked["regime"] == "dissimilar").sum()

# Regime shift score
if "ewma_df" not in st.session_state or refresh:
    with st.spinner("Computing regime shift indicator..."):
        from engine.similarity import compute_global_score_history
        hist_scores = compute_global_score_history(zscores, exclude_mo)
        ewma_df     = compute_ewma_regime_shift(hist_scores)
        shift_reading = current_regime_shift_score(ewma_df)
        st.session_state["ewma_df"]       = ewma_df
        st.session_state["shift_reading"] = shift_reading

ewma_df       = st.session_state["ewma_df"]
shift_reading = st.session_state["shift_reading"]

with col_status1:
    st.metric("Similar Periods Found", n_similar)
with col_status2:
    st.metric("Anti-Regime Periods", n_dissim)
with col_status3:
    pct = shift_reading["pct_rank"] * 100
    st.metric("Regime Shift Score", f"{shift_reading['mean_ewma']:.2f}", f"{pct:.0f}th pct")
with col_status4:
    signal_color = "🔴" if shift_reading["signal"] == "REGIME SHIFT" else "🟢"
    st.metric("Status", f"{signal_color} {shift_reading['signal']}")

st.divider()

# ---------------------------------------------------------------------------
# Tab layout
# ---------------------------------------------------------------------------
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Current Z-Scores",
    "Similar Periods",
    "Anti-Regimes",
    "Regime Shift Detector",
    "Raw Variables",
    "HF Portfolio",
])

# ── Tab 1: Current Z-Scores ────────────────────────────────────────────────
with tab1:
    st.subheader("Current Economic State Variables (Z-Scores)")
    st.caption(
        "12-month change normalised by rolling 10-year std dev, winsorised to ±3. "
        "Values near ±3 indicate extreme conditions relative to recent history."
    )

    var_labels = {
        "sp500":           "S&P 500 (log)",
        "yield_curve":     "Yield Curve (10yr − 3m)",
        "oil":             "WTI Crude Oil",
        "copper":          "Copper",
        "tbill_3m":        "3-Month T-Bill",
        "volatility":      "Volatility (VIX)",
        "stock_bond_corr": "Stock–Bond Correlation",
    }

    # Gauge chart
    fig_gauge = go.Figure()
    for col in current_zs.index:
        val = current_zs[col]
        label = var_labels.get(col, col)
        color = (
            "#d62728" if val > 2 else
            "#ff7f0e" if val > 1 else
            "#2ca02c" if val > -1 else
            "#1f77b4" if val > -2 else
            "#9467bd"
        )
        fig_gauge.add_trace(go.Bar(
            x=[label], y=[val],
            marker_color=color,
            name=label,
            text=[f"{val:.2f}"],
            textposition="outside",
        ))

    fig_gauge.add_hline(y=0,  line_dash="dash", line_color="gray", opacity=0.5)
    fig_gauge.add_hline(y=3,  line_dash="dot",  line_color="red",  opacity=0.4)
    fig_gauge.add_hline(y=-3, line_dash="dot",  line_color="red",  opacity=0.4)
    fig_gauge.update_layout(
        height=400, showlegend=False,
        yaxis=dict(range=[-3.5, 3.5], title="Z-Score"),
        title="Current Z-Scores vs Historical Distribution",
    )
    st.plotly_chart(fig_gauge, use_container_width=True)

    # Time series of Z-scores
    st.subheader("Z-Score History")
    selected_var = st.selectbox("Select variable", options=list(var_labels.keys()),
                                format_func=lambda x: var_labels[x])
    fig_ts = go.Figure()
    fig_ts.add_trace(go.Scatter(
        x=zscores.index, y=zscores[selected_var],
        mode="lines", name=var_labels[selected_var],
        line=dict(color="#1f77b4"),
    ))
    fig_ts.add_hline(y=0,  line_dash="dash", line_color="gray",   opacity=0.5)
    fig_ts.add_hline(y=3,  line_dash="dot",  line_color="red",    opacity=0.4, annotation_text="+3")
    fig_ts.add_hline(y=-3, line_dash="dot",  line_color="red",    opacity=0.4, annotation_text="−3")
    fig_ts.update_layout(height=350, yaxis_title="Z-Score", xaxis_title="Date",
                         title=f"{var_labels[selected_var]} — Z-Score History")
    st.plotly_chart(fig_ts, use_container_width=True)

    # Table
    st.subheader("Current Values Summary")
    summary_df = pd.DataFrame({
        "Variable":     [var_labels.get(c, c) for c in current_zs.index],
        "Z-Score":      current_zs.values.round(3),
        "Interpretation": [
            "Extreme high" if v > 2 else
            "Elevated"     if v > 1 else
            "Neutral"      if v > -1 else
            "Depressed"    if v > -2 else
            "Extreme low"
            for v in current_zs.values
        ],
    })
    st.dataframe(summary_df, use_container_width=True, hide_index=True)


# ── Tab 2: Similar Periods ─────────────────────────────────────────────────
with tab2:
    st.subheader(f"Most Similar Historical Periods to {target_date.strftime('%B %Y')}")
    st.caption(
        f"Bottom {quantile_q*100:.0f}% of global scores = most similar. "
        f"Excluding last {exclude_mo} months. Lower global score = more similar."
    )

    similar = ranked[ranked["regime"] == "similar"].head(N_DISPLAY_SIMILAR)

    # Timeline chart
    fig_timeline = go.Figure()
    fig_timeline.add_trace(go.Scatter(
        x=ranked.index, y=ranked["global_score"],
        mode="lines", name="Global Score",
        line=dict(color="lightblue", width=1),
    ))
    sim_points = ranked[ranked["regime"] == "similar"]
    fig_timeline.add_trace(go.Scatter(
        x=sim_points.index, y=sim_points["global_score"],
        mode="markers", name="Similar Periods",
        marker=dict(color="#2ca02c", size=8),
    ))
    fig_timeline.update_layout(
        height=350, title="Global Similarity Score — Full History",
        yaxis_title="Global Score (lower = more similar)",
        xaxis_title="Date",
    )
    st.plotly_chart(fig_timeline, use_container_width=True)

    # Table of most similar periods
    st.subheader(f"Top {len(similar)} Most Similar Months")
    display_similar = similar[["global_score", "rank"]].copy()
    display_similar.index = display_similar.index.strftime("%b %Y")
    display_similar.columns = ["Global Score", "Rank"]
    st.dataframe(display_similar.round(3), use_container_width=True)

    # Historical context bar
    st.subheader("Year Distribution of Similar Periods")
    years = sim_points.index.year
    year_counts = pd.Series(years).value_counts().sort_index()
    fig_years = px.bar(
        x=year_counts.index, y=year_counts.values,
        labels={"x": "Year", "y": "Similar months"},
        title="How many similar months fell in each year?",
        color_discrete_sequence=["#2ca02c"],
    )
    fig_years.update_layout(height=300)
    st.plotly_chart(fig_years, use_container_width=True)


# ── Tab 3: Anti-Regimes ────────────────────────────────────────────────────
with tab3:
    st.subheader("Anti-Regimes — Most Dissimilar Historical Periods")
    st.caption(
        "Top quantile of global scores = most dissimilar. "
        "Paper shows anti-regime periods also carry predictive information."
    )

    dissim = ranked[ranked["regime"] == "dissimilar"].sort_values("global_score", ascending=False).head(N_DISPLAY_SIMILAR)

    # Chart
    fig_anti = go.Figure()
    fig_anti.add_trace(go.Scatter(
        x=ranked.index, y=ranked["global_score"],
        mode="lines", name="Global Score",
        line=dict(color="lightblue", width=1),
    ))
    anti_points = ranked[ranked["regime"] == "dissimilar"]
    fig_anti.add_trace(go.Scatter(
        x=anti_points.index, y=anti_points["global_score"],
        mode="markers", name="Anti-Regime Periods",
        marker=dict(color="#d62728", size=8),
    ))
    fig_anti.update_layout(
        height=350, title="Global Similarity Score — Anti-Regime Periods Highlighted",
        yaxis_title="Global Score",
        xaxis_title="Date",
    )
    st.plotly_chart(fig_anti, use_container_width=True)

    # Table
    st.subheader(f"Top {len(dissim)} Most Dissimilar Months")
    display_dissim = dissim[["global_score", "rank"]].copy()
    display_dissim.index = display_dissim.index.strftime("%b %Y")
    display_dissim.columns = ["Global Score", "Rank"]
    st.dataframe(display_dissim.round(3), use_container_width=True)


# ── Tab 4: Regime Shift Detector ───────────────────────────────────────────
with tab4:
    st.subheader("Regime Shift Detector — EWMA of Global Scores")
    st.caption(
        "Per Exhibit 9 of the paper. Rising EWMA indicates the current environment "
        "is increasingly different from recent history — a potential regime shift. "
        "Historically peaked at: Jan 2009, May 2020, Oct 2022."
    )

    # Half-life table
    with st.expander("Half-life parameters (Exhibit 9 Panel A)"):
        st.dataframe(get_half_lives(), use_container_width=True, hide_index=True)

    # EWMA chart
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    fig_ewma = go.Figure()
    for i, yrs in enumerate(EWMA_LOOKBACK_YEARS):
        col = f"ewma_{yrs}yr"
        fig_ewma.add_trace(go.Scatter(
            x=ewma_df.index, y=ewma_df[col],
            mode="lines", name=f"{yrs}-Year Lookback",
            line=dict(color=colors[i], width=1.2, dash="dot"),
            opacity=0.7,
        ))
    fig_ewma.add_trace(go.Scatter(
        x=ewma_df.index, y=ewma_df["mean_ewma"],
        mode="lines", name="Mean of 4",
        line=dict(color="black", width=2.5),
    ))

    # Mark 90th pct threshold
    threshold = ewma_df["mean_ewma"].quantile(0.90)
    fig_ewma.add_hline(y=threshold, line_dash="dash", line_color="red",
                       opacity=0.6, annotation_text="90th pct")

    # Annotate known regime shift events
    known_events = {
        "Jan 2009": "2009-01-31",
        "May 2020": "2020-05-31",
        "Oct 2022": "2022-10-31",
    }
    for label, dt_str in known_events.items():
        try:
            dt = pd.Timestamp(dt_str)
            if dt in ewma_df.index:
                val = ewma_df.loc[dt, "mean_ewma"]
                fig_ewma.add_annotation(
                    x=dt, y=val, text=label, showarrow=True,
                    arrowhead=2, ax=0, ay=-30, font=dict(size=10),
                )
        except Exception:
            pass

    fig_ewma.update_layout(
        height=450,
        title="EWMA Regime Shift Indicator",
        yaxis_title="EWMA of Global Score",
        xaxis_title="Date",
        legend=dict(x=0.01, y=0.99),
    )
    st.plotly_chart(fig_ewma, use_container_width=True)

    # Current reading
    st.subheader("Current Regime Shift Reading")
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.metric("Mean EWMA", f"{shift_reading['mean_ewma']:.3f}")
    with col_b:
        st.metric("Historical Percentile", f"{shift_reading['pct_rank']*100:.1f}th")
    with col_c:
        st.metric("Signal", shift_reading["signal"])

    with st.expander("EWMA by lookback window"):
        by_lb = pd.DataFrame([shift_reading["by_lookback"]]).T
        by_lb.columns = ["EWMA Value"]
        st.dataframe(by_lb.round(4), use_container_width=True)

    # Historic high-signal dates
    st.subheader("Historic Regime Shift Events (> 90th percentile)")
    events = detect_regime_shift_events(ewma_df)
    if not events.empty:
        events_display = events.copy()
        events_display.index = events_display.index.strftime("%b %Y")
        st.dataframe(events_display.round(3), use_container_width=True)


# ── Tab 5: Raw Variables ───────────────────────────────────────────────────
with tab5:
    st.subheader("Raw State Variables")
    st.caption("Unprocessed time series used as inputs to the model.")

    raw_labels = {
        "sp500":           "S&P 500 (log price)",
        "yield_curve":     "Yield Curve: 10yr − 3m T-Bill (%)",
        "oil":             "WTI Crude Oil ($/bbl)",
        "copper":          "Copper (USD/metric ton)",
        "tbill_3m":        "3-Month T-Bill Yield (%)",
        "volatility":      "Volatility — VIX / Realised (%)",
        "stock_bond_corr": "Stock–Bond Correlation (3yr rolling)",
    }

    fig_raw = make_subplots(
        rows=4, cols=2,
        subplot_titles=list(raw_labels.values()),
        vertical_spacing=0.08,
    )
    positions = [(1,1),(1,2),(2,1),(2,2),(3,1),(3,2),(4,1)]
    for (r, c), col in zip(positions, raw_labels.keys()):
        if col in raw.columns:
            fig_raw.add_trace(
                go.Scatter(x=raw.index, y=raw[col], mode="lines",
                           name=raw_labels[col], showlegend=False,
                           line=dict(width=1)),
                row=r, col=c,
            )

    fig_raw.update_layout(height=900, title_text="Raw Economic State Variables")
    st.plotly_chart(fig_raw, use_container_width=True)

# ── Tab 6: HF Portfolio ────────────────────────────────────────────────────
with tab6:
    st.subheader("Regime-Based Hedge Fund Portfolio Construction")
    st.caption(
        "Applies the regime similarity scores to allocate across hedge fund strategies. "
        "Similar historical months determine expected return 'views', fed into a "
        "Black-Litterman or risk parity optimiser. "
        "**Demo mode** uses synthetic returns calibrated to HFRI long-run statistics — "
        "replace with real strategy data via `data/hf_returns.py`."
    )

    st.info(
        "⚠️ **Demo mode:** synthetic HF returns are used. "
        "Build `data/hf_returns.py` (HFRI indices) to replace with live data.",
        icon="📋",
    )

    # --- Portfolio controls ---
    col_pc1, col_pc2, col_pc3, col_pc4 = st.columns(4)
    with col_pc1:
        port_method = st.selectbox(
            "Optimisation method",
            options=["bl", "rp", "tilted_rp", "equal"],
            format_func=lambda x: {
                "bl": "Black-Litterman",
                "rp": "Risk Parity",
                "tilted_rp": "Tilted Risk Parity",
                "equal": "Equal Weight",
            }[x],
        )
    with col_pc2:
        port_horizon = st.selectbox(
            "View horizon (months)",
            options=[1, 3, 6, 12],
            index=0,
        )
    with col_pc3:
        port_quantile = st.slider(
            "Regime quantile",
            min_value=0.10, max_value=0.30, value=float(quantile_q), step=0.05,
            help="Fraction of history used as 'similar' for computing views",
        )
    with col_pc4:
        tilt_str = st.slider(
            "Tilt strength",
            min_value=0.0, max_value=1.0, value=0.5, step=0.1,
            help="Only used for Tilted Risk Parity (0=pure RP, 1=max tilt)",
            disabled=(port_method != "tilted_rp"),
        )

    run_bt = st.button("Run Backtest", type="primary")

    # Cache key: invalidate when any control changes
    bt_key = f"bt_{port_method}_{port_horizon}_{port_quantile}_{tilt_str}"

    if run_bt or bt_key not in st.session_state:
        with st.spinner("Running walk-forward backtest (this may take ~30 seconds)…"):
            try:
                from portfolio.backtest import run_backtest, demo_strategy_returns
                from engine.strategy_timing import current_regime_stats

                hf_returns = demo_strategy_returns()

                bt = run_backtest(
                    hf_returns,
                    zscores,
                    method=port_method,
                    quantile=port_quantile,
                    exclude_recent_months=exclude_mo,
                    horizon=port_horizon,
                    tilt_strength=tilt_str,
                )

                # Current allocation
                regime_tbl = current_regime_stats(
                    hf_returns, zscores,
                    quantile=port_quantile,
                    horizon=port_horizon,
                    exclude_recent_months=exclude_mo,
                )

                st.session_state[bt_key] = {
                    "bt": bt,
                    "regime_tbl": regime_tbl,
                    "hf_returns": hf_returns,
                }
            except Exception as e:
                st.error(f"Backtest failed: {e}")
                st.stop()

    if bt_key in st.session_state:
        cached      = st.session_state[bt_key]
        bt          = cached["bt"]
        regime_tbl  = cached["regime_tbl"]
        hf_returns  = cached["hf_returns"]

        port_ret = bt["portfolio_returns"]
        bm_ret   = bt["benchmark_returns"]
        wt_hist  = bt["weights_history"]
        stats_df = bt["stats"]
        strategies = bt["strategy_names"]

        # ── Current allocation ──────────────────────────────────────────
        st.subheader("Current Regime-Implied Allocation")

        if not wt_hist.empty:
            latest_weights = wt_hist.iloc[-1].sort_values(ascending=True)
            fig_alloc = go.Figure(go.Bar(
                x=latest_weights.values * 100,
                y=latest_weights.index,
                orientation="h",
                marker_color="#1f77b4",
                text=[f"{v*100:.1f}%" for v in latest_weights.values],
                textposition="outside",
            ))
            fig_alloc.update_layout(
                height=350,
                title=f"Portfolio Weights — {wt_hist.index[-1].strftime('%b %Y')}",
                xaxis_title="Weight (%)",
                xaxis=dict(range=[0, 55]),
                margin=dict(l=140),
            )
            st.plotly_chart(fig_alloc, use_container_width=True)

        # ── Regime-conditional stats table ──────────────────────────────
        st.subheader(f"Regime-Conditional Statistics ({port_horizon}m horizon)")
        st.caption(
            "Regime Sharpe: forward returns in historically similar months. "
            "Sharpe Premium: regime Sharpe minus full-history unconditional Sharpe."
        )

        rc_display = regime_tbl.copy()
        rc_display = rc_display.rename(columns={
            "regime_mean_ret_%":    "Regime Mean Ret (%)",
            "regime_std_%":         "Regime Std (%)",
            "regime_sharpe":        "Regime Sharpe",
            "unconditional_sharpe": "Uncond. Sharpe",
            "sharpe_premium":       "Sharpe Premium",
            "n_similar_periods":    "N Similar",
        })
        st.dataframe(rc_display.round(3), use_container_width=True)

        st.divider()

        # ── Equity curve ────────────────────────────────────────────────
        st.subheader("Backtest Equity Curve")

        equity_port = (1 + port_ret).cumprod()
        equity_bm   = (1 + bm_ret).cumprod()

        fig_eq = go.Figure()
        fig_eq.add_trace(go.Scatter(
            x=equity_port.index, y=equity_port.values,
            mode="lines", name=port_ret.name,
            line=dict(color="#1f77b4", width=2),
        ))
        fig_eq.add_trace(go.Scatter(
            x=equity_bm.index, y=equity_bm.values,
            mode="lines", name="Equal Weight",
            line=dict(color="#aaaaaa", width=1.5, dash="dot"),
        ))
        fig_eq.update_layout(
            height=380,
            title="Cumulative Growth of $1",
            yaxis_title="Portfolio Value ($)",
            xaxis_title="Date",
            legend=dict(x=0.01, y=0.99),
        )
        st.plotly_chart(fig_eq, use_container_width=True)

        # ── Drawdown ─────────────────────────────────────────────────────
        from portfolio.risk import drawdown_series
        dd_port = drawdown_series(port_ret) * 100
        dd_bm   = drawdown_series(bm_ret)   * 100

        fig_dd = go.Figure()
        fig_dd.add_trace(go.Scatter(
            x=dd_port.index, y=dd_port.values,
            mode="lines", name=port_ret.name,
            line=dict(color="#d62728", width=1.5),
            fill="tozeroy", fillcolor="rgba(214,39,40,0.15)",
        ))
        fig_dd.add_trace(go.Scatter(
            x=dd_bm.index, y=dd_bm.values,
            mode="lines", name="Equal Weight",
            line=dict(color="#aaaaaa", width=1, dash="dot"),
        ))
        fig_dd.update_layout(
            height=280,
            title="Drawdown (%)",
            yaxis_title="Drawdown (%)",
            xaxis_title="Date",
        )
        st.plotly_chart(fig_dd, use_container_width=True)

        # ── Weight history heat-map ──────────────────────────────────────
        st.subheader("Weight History")
        fig_wt = go.Figure(go.Heatmap(
            z=wt_hist.values.T * 100,
            x=[d.strftime("%Y-%m") for d in wt_hist.index],
            y=wt_hist.columns.tolist(),
            colorscale="Blues",
            colorbar=dict(title="Weight (%)"),
            zmin=0, zmax=40,
        ))
        fig_wt.update_layout(
            height=320,
            title="Portfolio Weights Over Time (%)",
            xaxis_title="Date",
        )
        st.plotly_chart(fig_wt, use_container_width=True)

        # ── Performance table ─────────────────────────────────────────────
        st.subheader("Performance Summary")
        st.dataframe(stats_df, use_container_width=True)

        st.caption(
            "**Notes:** Walk-forward backtest — no lookahead bias. "
            "Regime-conditional views use only data available at each rebalance date. "
            "Covariance estimated from trailing 36-month returns with shrinkage regularisation. "
            "Demo returns are synthetic; live results will differ with real HFRI data."
        )


# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.divider()
st.caption(
    "Replication of: Mulliner A., Harvey C.R., Xia C., Fang E., Van Hemert O. "
    "'Regimes.' *Journal of Portfolio Management*, Feb 2026. "
    "Data: FRED, Yahoo Finance. Not investment advice."
)
