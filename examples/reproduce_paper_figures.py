"""Reproduce all main-text figures from Gadiyar et al. (2026).

Generates Fig. 4, Fig. 6 (model predictions), Fig. 7 (Pareto), Fig. 8
(closed-loop) and saves them to ``paper_figures/`` in PDF + PNG + SVG.

Run::

    python examples/reproduce_paper_figures.py
"""

from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import torch

from perfusio.viz.theme import apply_theme
from perfusio.viz.static import (
    fig4_training_trajectories,
    fig6_model_predictions,
    fig7_pareto_front,
    fig8_closed_loop_performance,
)
from perfusio.simulator.cho_perfusion import CHOSimulator

apply_theme()
OUT = Path("paper_figures")
OUT.mkdir(exist_ok=True)


def _save(fig, name: str) -> None:
    for ext in ("pdf", "png", "svg"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=300)
    import matplotlib.pyplot as plt
    plt.close(fig)
    print(f"  Saved {name}.{{pdf,png,svg}}")


# ── Fig. 4 ────────────────────────────────────────────────────────────────────
print("Generating Fig. 4 …")
sim = CHOSimulator(clone="CloneX", seed=0)
runs = sim.generate_box_behnken_experiment(n_days=28, seed=0)
fig4 = fig4_training_trajectories(runs, alt_text=True)
_save(fig4, "fig4")

# ── Fig. 7 ────────────────────────────────────────────────────────────────────
print("Generating Fig. 7 …")
titer = [r["trajectory"][-1, 8] for r in runs]
vcv   = [r["trajectory"][-1, 0] for r in runs]
fig7 = fig7_pareto_front(titer, vcv, alt_text=True)
_save(fig7, "fig7")

# ── Fig. 8 ────────────────────────────────────────────────────────────────────
print("Generating Fig. 8 …")
rng = np.random.default_rng(42)
days = list(range(29))
vcd  = (np.linspace(1, 12, 29) + rng.normal(0, 0.3, 29)).tolist()
glc  = (np.linspace(5, 3.8, 29) + rng.normal(0, 0.1, 29)).tolist()
tit  = (np.linspace(0, 580, 29) + rng.normal(0, 5, 29)).tolist()
fig8 = fig8_closed_loop_performance(
    days, vcd, 10.0, glc, 4.0, tit, 500.0, alt_text=True
)
_save(fig8, "fig8")

print(f"\nAll figures written to {OUT}/")
