"""Visualization sub-package for ``perfusio``.

Public API
----------
- :mod:`~perfusio.viz.theme` — custom OKLCH colour palette and rcParams.
- :mod:`~perfusio.viz.static` — Matplotlib reproductions of Gadiyar Figs. 4/6/7/8.
- :mod:`~perfusio.viz.interactive` — Plotly figures.
- :mod:`~perfusio.viz.dashboard` — Plotly Dash app (Fig. 5).
- :func:`~perfusio.viz.pareto_explorer.pareto_scatter` — interactive Pareto front.
"""

from perfusio.viz.theme import PALETTE, apply_theme

__all__ = ["PALETTE", "apply_theme"]
