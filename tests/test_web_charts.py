import numpy as np
import pandas as pd
import plotly.graph_objects as go

from web import charts
from web.charts import BG, INK, RUST, MUTED, house_layout


def _ranked():
    idx = pd.date_range("2000-01-31", periods=120, freq="ME")
    df = pd.DataFrame({"global_score": np.linspace(1, 8, 120)}, index=idx)
    df["rank"] = range(1, 121)
    df["regime"] = "neutral"
    df.iloc[:24, df.columns.get_loc("regime")] = "similar"
    df.iloc[-24:, df.columns.get_loc("regime")] = "dissimilar"
    return df


def test_house_layout_uses_site_tokens_and_no_gridlines():
    fig = go.Figure()
    fig.update_layout(house_layout())
    assert fig.layout.paper_bgcolor == BG
    assert fig.layout.plot_bgcolor == BG
    assert "JetBrains Mono" in fig.layout.font.family
    assert fig.layout.xaxis.showgrid is False
    assert fig.layout.yaxis.showgrid is False


def test_zscore_bars_colours_extremes_rust():
    z = pd.Series({"sp500": 0.5, "oil": 2.4, "copper": -2.6, "vix": -0.1})
    fig = charts.zscore_bars(z)
    colours = list(fig.data[0].marker.color)
    assert colours[list(z.index).index("oil")] == RUST
    assert colours[list(z.index).index("copper")] == RUST
    assert colours[list(z.index).index("sp500")] == INK


def test_similarity_timeline_marks_similar_months_and_masked_window():
    ranked = _ranked()
    target = pd.Timestamp("2012-12-31")
    fig = charts.similarity_timeline(ranked, target=target, exclude_months=36, mode="similar")
    names = [t.name for t in fig.data]
    assert "Global score" in names and "Similar months" in names
    marks = fig.data[names.index("Similar months")]
    assert len(marks.x) == 24 and marks.marker.color == RUST
    # masked window bounded by a faint tint ending at the target month, plus a
    # rule at its left edge; the tint must stay light enough to read through
    rect, rule = fig.layout.shapes
    assert pd.Timestamp(rect.x1) == target and rect.opacity <= 0.25
    assert pd.Timestamp(rule.x0) == pd.Timestamp(rule.x1) == target - pd.DateOffset(months=36)
    assert "excluded" in fig.layout.annotations[0].text


def test_regime_shift_chart_has_mean_and_four_lookbacks():
    idx = pd.date_range("2000-01-31", periods=60, freq="ME")
    ew = pd.DataFrame({f"ewma_{y}yr": np.random.default_rng(y).random(60) for y in [1, 2, 3, 4]}, index=idx)
    ew["mean_ewma"] = ew.mean(axis=1)
    fig = charts.regime_shift_chart(ew)
    names = [t.name for t in fig.data]
    assert "Mean of four" in names
    assert sum(n.endswith("lookback") for n in names) == 4
    assert fig.data[names.index("Mean of four")].line.color == INK


def test_cumulative_lines_emphasises_named_series_and_cumsums_percent():
    idx = pd.date_range("2000-01-31", periods=5, freq="ME")
    r = pd.DataFrame({"q1": [1.0, 1.0, -1.0, 2.0, 0.0], "long_only": [0.5] * 5, "q5": [0.0] * 5}, index=idx)
    fig = charts.cumulative_lines(r, {"q1": "Quintile 1", "q5": "Quintile 5", "long_only": "Long only"},
                                  emphasis=["q1"], reference="long_only")
    names = [t.name for t in fig.data]
    assert names == ["Quintile 1", "Quintile 5", "Long only"]
    q1 = fig.data[0]
    assert list(q1.y) == [1.0, 2.0, 1.0, 3.0, 3.0]
    assert q1.line.color == RUST
    assert fig.data[2].line.dash == "dot"


def test_similarity_timeline_draws_the_excluded_window_as_a_dotted_continuation():
    ranked = _ranked()
    target = pd.Timestamp("2012-12-31")
    # The 36 masked months, falling to zero on the target itself.
    idx = pd.date_range("2010-01-31", periods=36, freq="ME")
    excluded = pd.Series(np.linspace(8.0, 0.0, 36), index=idx)

    fig = charts.similarity_timeline(ranked, target=target, exclude_months=36,
                                     mode="similar", excluded=excluded)
    names = [t.name for t in fig.data]
    assert "Excluded from ranking" in names
    trace = fig.data[names.index("Excluded from ranking")]
    assert trace.line.dash == "dot" and trace.line.color == MUTED
    # Joined to the last ranked point so the curve reads as continuous...
    assert pd.Timestamp(trace.x[0]) == ranked.index[-1]
    assert trace.y[0] == ranked["global_score"].sort_index().iloc[-1]
    # ...and it reaches the target at zero rather than stopping short.
    assert pd.Timestamp(trace.x[-1]) == target and trace.y[-1] == 0.0


def test_similarity_timeline_without_excluded_scores_draws_no_extra_trace():
    fig = charts.similarity_timeline(_ranked(), target=pd.Timestamp("2012-12-31"), exclude_months=36)
    assert "Excluded from ranking" not in [t.name for t in fig.data]
