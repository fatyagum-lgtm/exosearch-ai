"""Offline standalone research dashboard with actual experiment data."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from jinja2 import Environment, FileSystemLoader, select_autoescape

DARK_BG = "#070d1b"
BLUE = "#80a8ff"
TEAL = "#68e4d5"
CORAL = "#fa9a91"


def _style(fig: go.Figure, height: int = 300) -> go.Figure:
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)", font={"family": "Arial, sans-serif", "color": "#acbcd3"},
        margin={"l": 45, "r": 30, "t": 35, "b": 42}, height=height,
        legend={"orientation": "h", "y": -0.3},
    )
    fig.update_xaxes(showgrid=True, gridcolor="#20314a", zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="#20314a", zeroline=False)
    return fig


def _charts(exp: dict, ranking: pd.DataFrame) -> dict[str, str]:
    models = exp["models"]
    names = list(models)
    metric_labels = ["Average precision", "F1", "Recall", "ROC AUC"]
    metric_keys = ["average_precision", "f1", "recall", "roc_auc"]
    fig = go.Figure()
    for idx, name in enumerate(names):
        scores = models[name]["test"]
        fig.add_trace(go.Bar(name=name.replace("_", " ").title(), x=metric_labels,
                             y=[scores[k] for k in metric_keys], marker_color=[TEAL, BLUE][idx % 2]))
    fig.update_layout(barmode="group", yaxis_range=[0, 1.03])
    comparison = _style(fig, 345).to_html(full_html=False, include_plotlyjs=False, config={"displaylogo": False})

    fig = go.Figure()
    for idx, name in enumerate(names):
        curve = models[name]["test"]["pr_curve"]
        fig.add_trace(go.Scatter(x=curve["recall"], y=curve["precision"], name=name.replace("_", " ").title(),
                                 line={"color": [TEAL, BLUE][idx % 2], "width": 3}, mode="lines"))
    fig.update_layout(xaxis_title="Recall", yaxis_title="Precision", xaxis_range=[0, 1], yaxis_range=[0, 1.03])
    pr = _style(fig, 345).to_html(full_html=False, include_plotlyjs=False, config={"displaylogo": False})

    winner = models[exp["winner"]]
    confusion = winner["test"]["confusion_matrix"]
    fig = go.Figure(go.Heatmap(
        z=confusion, x=["Predicted FP / FA", "Predicted CP / KP"],
        y=["Actual FP / FA", "Actual CP / KP"], colorscale=[[0, "#142744"], [1, "#36c6bc"]],
        showscale=False, text=confusion, texttemplate="%{text}", textfont={"size": 22}
    ))
    confusion_plot = _style(fig, 300).to_html(full_html=False, include_plotlyjs=False, config={"displaylogo": False})

    feat = winner["importance"][:9][::-1]
    fig = go.Figure(go.Bar(x=[v["importance"] for v in feat], y=[v["feature"] for v in feat],
                           marker_color=TEAL, orientation="h"))
    fig.update_layout(xaxis_title="Absolute coefficient / relative tree importance")
    importance = _style(fig, 360).to_html(full_html=False, include_plotlyjs=False, config={"displaylogo": False})

    fig = go.Figure()
    if not ranking.empty:
        size = pd.to_numeric(ranking["pl_rade"], errors="coerce").fillna(1).clip(0.1, 30)
        fig.add_trace(go.Scatter(
            x=ranking["pl_orbper"], y=ranking["pl_rade"], mode="markers",
            text=ranking["toi"].map(lambda x: "TOI " + str(x)),
            customdata=ranking["planet_likeness_score"],
            hovertemplate="%{text}<br>Period: %{x:.2f} d<br>Radius: %{y:.2f} R⊕<br>Score: %{customdata:.3f}<extra></extra>",
            marker={"color": ranking["planet_likeness_score"], "colorscale": [[0, CORAL], [1, TEAL]],
                    "size": np.sqrt(size.to_numpy()) * 7 + 6, "opacity": 0.83, "showscale": True,
                    "colorbar": {"title": "Score", "thickness": 10}},
        ))
    fig.update_layout(xaxis_title="Orbital period (days)", yaxis_title="Estimated radius (Earth radii)")
    fig.update_xaxes(type="log")
    fig.update_yaxes(type="log")
    scatter = _style(fig, 410).to_html(full_html=False, include_plotlyjs=False, config={"displaylogo": False})
    return {"comparison": comparison, "pr": pr, "confusion": confusion_plot,
            "importance": importance, "scatter": scatter}


def build_dashboard(directory: str | Path) -> Path:
    folder = Path(directory)
    experiment = json.loads((folder / "experiment.json").read_text(encoding="utf-8"))
    rank = pd.read_csv(folder / "candidate_ranking.csv")
    rank = rank.loc[rank["planet_likeness_score"].notna()].copy()
    charts = _charts(experiment, rank)
    # escape all strings in the rendered table (no HTML injection from catalog rows)
    rows = []
    for _, obj in rank.head(250).iterrows():
        def clean(key):
            v = obj[key]
            return "—" if pd.isna(v) else str(v)
        rows.append({"toi": clean("toi"), "tid": clean("tid"), "disposition": clean("tfopwg_disp"),
                     "score": f"{obj['planet_likeness_score']:.3f}",
                     "period": clean("pl_orbper"), "radius": clean("pl_rade")})
    templates = Path(__file__).resolve().parent / "templates"
    env = Environment(loader=FileSystemLoader(str(templates)), autoescape=select_autoescape(["html"]))
    page = env.get_template("template.html").render(
        exp=experiment, charts=charts, rows=rows,
        demo=experiment["data_type"].startswith("SYNTHETIC"),
        total_ranked=len(rank), best=experiment["models"][experiment["winner"]]["test"],
    )
    # Bundle Plotly JS once, offline. The HTML and script live together in the same directory.
    plotlyjs = folder / "plotly.min.js"
    if not plotlyjs.is_file():
        from plotly.offline.offline import get_plotlyjs
        plotlyjs.write_text(get_plotlyjs(), encoding="utf-8")
    destination = folder / "index.html"
    destination.write_text(page, encoding="utf-8")
    return destination
