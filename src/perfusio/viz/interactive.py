"""Plotly interactive figures for ``perfusio``.

Functions
---------
- :func:`trajectory_figure` — animated multi-run trajectory explorer.
- :func:`forecast_figure` — GP posterior with shaded uncertainty ribbon.
- :func:`pareto_scatter` — interactive Pareto front scatter.
- :func:`acquisition_surface` — 2-D acquisition function heat-map.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def trajectory_figure(
    runs: list[dict[str, Any]],
    species: str = "VCD",
    title: str = "Perfusion run trajectories",
) -> Any:
    """Interactive Plotly line chart of multiple run trajectories.

    Parameters
    ----------
    runs:
        List of run dicts with keys ``run_id`` and ``trajectory`` (ndarray).
    species:
        Species to display.
    title:
        Figure title.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    import plotly.graph_objects as go

    species_idx = {
        "VCD": 0,
        "Via": 1,
        "Glc": 2,
        "Gln": 3,
        "Glu": 4,
        "Lac": 5,
        "Amm": 6,
        "Pyr": 7,
        "Titer": 8,
    }
    k = species_idx.get(species, 0)

    fig = go.Figure()
    for run in runs:
        traj = np.asarray(run["trajectory"])
        days = list(range(traj.shape[0]))
        fig.add_trace(
            go.Scatter(
                x=days,
                y=traj[:, k].tolist(),
                mode="lines+markers",
                name=f"Run {run['run_id']}",
                line={"width": 1.5},
                marker={"size": 4},
            )
        )

    fig.update_layout(
        title=title,
        xaxis_title="Culture day",
        yaxis_title=species,
        template="simple_white",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02},
    )
    return fig


def forecast_figure(
    days: list[int] | np.ndarray,
    mean: np.ndarray,
    q10: np.ndarray,
    q90: np.ndarray,
    observed_days: list[int] | np.ndarray | None = None,
    observed_values: np.ndarray | None = None,
    species: str = "VCD",
    title: str = "3-step forecast",
) -> Any:
    """Plotly figure with GP mean and 80% prediction interval ribbon.

    Parameters
    ----------
    days:
        Forecast horizon days.
    mean, q10, q90:
        Model predictions, 1-D arrays.
    observed_days, observed_values:
        Optional measured data to overlay.
    species:
        Species label.
    title:
        Figure title.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    import plotly.graph_objects as go

    days = np.asarray(days)
    fig = go.Figure()

    # Uncertainty ribbon
    fig.add_trace(
        go.Scatter(
            x=np.concatenate([days, days[::-1]]).tolist(),
            y=np.concatenate([q90, q10[::-1]]).tolist(),
            fill="toself",
            fillcolor="rgba(78, 154, 241, 0.2)",
            line={"color": "rgba(255,255,255,0)"},
            name="80% PI",
            showlegend=True,
        )
    )
    # Mean line
    fig.add_trace(
        go.Scatter(
            x=days.tolist(),
            y=np.asarray(mean).tolist(),
            mode="lines",
            name="Mean",
            line={"color": "#4E9AF1", "width": 2},
        )
    )
    # Observations
    if observed_days is not None and observed_values is not None:
        fig.add_trace(
            go.Scatter(
                x=list(observed_days),
                y=np.asarray(observed_values).tolist(),
                mode="markers",
                name="Observed",
                marker={"color": "#E0775C", "size": 8, "symbol": "circle"},
            )
        )

    fig.update_layout(
        title=title,
        xaxis_title="Culture day",
        yaxis_title=species,
        template="simple_white",
    )
    return fig


def pareto_scatter(
    titer: np.ndarray,
    vcv: np.ndarray,
    is_pareto: np.ndarray | None = None,
    hover_text: list[str] | None = None,
    title: str = "Pareto front — Titer vs. VCV",
) -> Any:
    """Interactive Pareto front scatter plot.

    Parameters
    ----------
    titer:
        Titer values for all evaluated points.
    vcv:
        VCV (viable cell volume) values.
    is_pareto:
        Boolean mask; Pareto-optimal points are highlighted.
    hover_text:
        Optional hover labels per point.
    title:
        Figure title.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    import plotly.graph_objects as go

    titer = np.asarray(titer)
    vcv = np.asarray(vcv)
    fig = go.Figure()

    if is_pareto is not None:
        mask = np.asarray(is_pareto, dtype=bool)
        # Non-Pareto background
        fig.add_trace(
            go.Scatter(
                x=titer[~mask].tolist(),
                y=vcv[~mask].tolist(),
                mode="markers",
                name="Feasible",
                marker={"color": "#D0D0D0", "size": 7},
                text=[hover_text[i] for i, m in enumerate(mask) if not m] if hover_text else None,
                hoverinfo="text+x+y",
            )
        )
        # Pareto front
        order = np.argsort(titer[mask])
        fig.add_trace(
            go.Scatter(
                x=titer[mask][order].tolist(),
                y=vcv[mask][order].tolist(),
                mode="lines+markers",
                name="Pareto front",
                line={"color": "#4E9AF1", "width": 2},
                marker={"color": "#4E9AF1", "size": 9},
                text=[hover_text[i] for i, m in enumerate(mask) if m] if hover_text else None,
                hoverinfo="text+x+y",
            )
        )
    else:
        fig.add_trace(
            go.Scatter(
                x=titer.tolist(),
                y=vcv.tolist(),
                mode="markers",
                name="Points",
                marker={"color": "#4E9AF1", "size": 7},
            )
        )

    fig.update_layout(
        title=title,
        xaxis_title="Titer (mg/L)",
        yaxis_title="VCV (10⁶ cells·mL⁻¹)",
        template="simple_white",
    )
    return fig


def acquisition_surface(
    x1: np.ndarray,
    x2: np.ndarray,
    acq_values: np.ndarray,
    x1_label: str = "Perfusion rate (vvd)",
    x2_label: str = "Bleed rate (vvd)",
    title: str = "Acquisition function surface",
) -> Any:
    """2-D acquisition function heat-map as an interactive contour plot.

    Parameters
    ----------
    x1, x2:
        1-D arrays of grid values (will be meshed internally).
    acq_values:
        2-D array of acquisition values, shape ``(len(x1), len(x2))``.
    x1_label, x2_label:
        Axis labels.
    title:
        Figure title.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    import plotly.graph_objects as go

    fig = go.Figure(
        go.Contour(
            z=np.asarray(acq_values).tolist(),
            x=np.asarray(x1).tolist(),
            y=np.asarray(x2).tolist(),
            colorscale="Blues",
            contours={"showlabels": True},
            colorbar={"title": "Acq. value"},
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title=x1_label,
        yaxis_title=x2_label,
        template="simple_white",
    )
    return fig
