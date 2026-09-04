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
    # masked window drawn as a shaded region ending at the target month
    shapes = fig.layout.shapes
    assert len(shapes) == 1 and pd.Timestamp(shapes[0].x1) == target


def test_regime_shift_chart_has_mean_and_four_lookbacks():
    idx = pd.date_range("2000-01-31", periods=60, freq="ME")
    ew = pd.DataFrame({f"ewma_{y}yr": np.random.default_rng(y).random(60) for y in [1, 2, 3, 4]}, index=idx)
    ew["mean_ewma"] = ew.mean(axis=1)
    fig = charts.regime_shift_chart(ew)
    names = [t.name for t in fig.data]
    assert "Mean of four" in names
    assert sum(n.endswith("lookback") for n in names) == 4
    assert fig.data[names.index("Mean of four")].line.color == INK
