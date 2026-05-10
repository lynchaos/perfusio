"""Plotly Dash live dashboard — Gadiyar Fig. 5 single mini-bioreactor view.

Launch with::

    python -m perfusio.viz.dashboard.app

or via the CLI::

    perfusio run --dashboard

The app streams data from a :class:`~perfusio.connectors.base.BioreactorConnectorBase`
at a configurable refresh interval and displays:
- VCD and viability over time (left panel)
- Glucose and lactate (middle panel)
- Titer + 3-step forecast ribbon (right panel)
- Current BED recommendation (card below panels)
"""

from __future__ import annotations

import threading
from collections import deque
from typing import Any

import dash
import plotly.graph_objects as go
from dash import Input, Output, dcc, html

app = dash.Dash(
    __name__,
    title="perfusio — live bioreactor dashboard",
    suppress_callback_exceptions=True,
)
app.server.secret_key = "perfusio-dashboard"  # dev only

# ── State store (thread-safe deques) ──────────────────────────────────────────

MAX_DAYS = 30
_days: deque[int] = deque(maxlen=MAX_DAYS)
_vcd: deque[float] = deque(maxlen=MAX_DAYS)
_via: deque[float] = deque(maxlen=MAX_DAYS)
_glc: deque[float] = deque(maxlen=MAX_DAYS)
_lac: deque[float] = deque(maxlen=MAX_DAYS)
_titer: deque[float] = deque(maxlen=MAX_DAYS)
_titer_q90: deque[float] = deque(maxlen=MAX_DAYS)
_titer_q10: deque[float] = deque(maxlen=MAX_DAYS)
_last_bed: dict[str, Any] = {"control": "—", "value": "—", "acqf": "—"}
_lock = threading.Lock()

# ── Layout ────────────────────────────────────────────────────────────────────

app.layout = html.Div(
    style={"fontFamily": "Inter, Helvetica, sans-serif", "padding": "1rem"},
    children=[
        html.H2(
            "perfusio — Self-Driving Bioreactor Dashboard",
            style={"marginBottom": "0.5rem"},
        ),
        html.Div(
            id="status-badge",
            children="⚡ Live",
            style={
                "display": "inline-block",
                "background": "#4E9AF1",
                "color": "white",
                "padding": "2px 10px",
                "borderRadius": "8px",
                "fontSize": "0.8rem",
                "marginBottom": "1rem",
            },
        ),
        dcc.Interval(id="interval", interval=5_000, n_intervals=0),  # 5 s refresh
        html.Div(
            style={"display": "flex", "gap": "1rem"},
            children=[
                dcc.Graph(id="vcd-chart", style={"flex": 1}),
                dcc.Graph(id="metabolite-chart", style={"flex": 1}),
                dcc.Graph(id="titer-chart", style={"flex": 1}),
            ],
        ),
        html.Div(
            id="bed-card",
            style={
                "marginTop": "1rem",
                "padding": "0.75rem 1rem",
                "border": "1px solid #E0E0E0",
                "borderRadius": "8px",
                "background": "#F8FAFF",
            },
            children=[
                html.Strong("Latest BED recommendation: "),
                html.Span(id="bed-text"),
            ],
        ),
    ],
)

# ── Callbacks ─────────────────────────────────────────────────────────────────


@app.callback(
    Output("vcd-chart", "figure"),
    Output("metabolite-chart", "figure"),
    Output("titer-chart", "figure"),
    Output("bed-text", "children"),
    Input("interval", "n_intervals"),
)
def _update(_n: int) -> tuple[go.Figure, go.Figure, go.Figure, str]:
    with _lock:
        days = list(_days)
        vcd = list(_vcd)
        via = list(_via)
        glc = list(_glc)
        lac = list(_lac)
        titer = list(_titer)
        tq10 = list(_titer_q10)
        tq90 = list(_titer_q90)
        bed = dict(_last_bed)

    tmpl = "simple_white"

    # VCD + viability
    f_vcd = go.Figure()
    f_vcd.add_trace(go.Scatter(x=days, y=vcd, name="VCD", line={"color": "#4E9AF1"}))
    f_vcd.add_trace(
        go.Scatter(x=days, y=via, name="Viability (%)", line={"color": "#E0775C", "dash": "dot"}),
    )
    f_vcd.update_layout(title="VCD & Viability", xaxis_title="Day", template=tmpl)

    # Metabolites
    f_met = go.Figure()
    f_met.add_trace(go.Scatter(x=days, y=glc, name="Glc (g/L)", line={"color": "#6DC473"}))
    f_met.add_trace(go.Scatter(x=days, y=lac, name="Lac (mmol/L)", line={"color": "#C97ED8"}))
    f_met.update_layout(title="Glucose & Lactate", xaxis_title="Day", template=tmpl)

    # Titer + forecast ribbon
    f_titer = go.Figure()
    if tq10 and tq90:
        all_days = days + days[::-1]
        all_q = tq90 + tq10[::-1]
        f_titer.add_trace(
            go.Scatter(
                x=all_days,
                y=all_q,
                fill="toself",
                fillcolor="rgba(78,154,241,0.15)",
                line={"color": "rgba(0,0,0,0)"},
                name="80% PI",
            )
        )
    f_titer.add_trace(go.Scatter(x=days, y=titer, name="Titer (mg/L)", line={"color": "#E5B84E"}))
    f_titer.update_layout(title="Titer + Forecast", xaxis_title="Day", template=tmpl)

    bed_str = f"{bed['control']} → {bed['value']} ({bed['acqf']})"
    return f_vcd, f_met, f_titer, bed_str


# ── Data injection helpers (called by DigitalTwin) ────────────────────────────


def push_sample(
    day: int,
    sample: dict[str, float | None],
    forecast_mean: dict[str, float] | None = None,
    forecast_q10: dict[str, float] | None = None,
    forecast_q90: dict[str, float] | None = None,
) -> None:
    """Thread-safe push of a new daily sample into the dashboard store."""
    with _lock:
        _days.append(day)
        _vcd.append(float(sample.get("VCD") or 0.0))
        _via.append(float(sample.get("Via") or 0.0))
        _glc.append(float(sample.get("Glc") or 0.0))
        _lac.append(float(sample.get("Lac") or 0.0))
        _titer.append(float(sample.get("Titer") or 0.0))
        _titer_q10.append(float((forecast_q10 or {}).get("Titer") or 0.0))
        _titer_q90.append(float((forecast_q90 or {}).get("Titer") or 0.0))


def push_bed_decision(control: str, value: float, acqf_name: str) -> None:
    """Update the last BED recommendation card."""
    with _lock:
        _last_bed["control"] = control
        _last_bed["value"] = f"{value:.3f}"
        _last_bed["acqf"] = acqf_name


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port="8050")
