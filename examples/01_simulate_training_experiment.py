"""Example 01 — Simulate a 24-run Box-Behnken training experiment.

This script:
1. Instantiates a CHO Clone X simulator.
2. Runs the full Box-Behnken DoE (24 runs × 28 days).
3. Saves trajectories to ``runs/`` as JSON files.
4. Plots the VCD trajectories (Fig. 4 reproduction).

Run from the repository root::

    python examples/01_simulate_training_experiment.py
"""

from __future__ import annotations

import json
from pathlib import Path

from perfusio.simulator.cho_perfusion import CHOSimulator
from perfusio.viz.static import fig4_training_trajectories
from perfusio.viz.theme import apply_theme

# ── Configuration ─────────────────────────────────────────────────────────────

CLONE = "CloneX"
SEED = 0
N_DAYS = 28
OUT_DIR = Path("runs")

# ── Main ──────────────────────────────────────────────────────────────────────

apply_theme()

print(f"Simulating Box-Behnken experiment: {CLONE}, {N_DAYS} days, seed={SEED}")
sim = CHOSimulator(clone=CLONE, seed=SEED)
runs = sim.generate_box_behnken_experiment(n_days=N_DAYS, seed=SEED)

OUT_DIR.mkdir(parents=True, exist_ok=True)
for run in runs:
    payload = dict(run)
    payload["trajectory"] = run["trajectory"].tolist()
    payload["noisy_samples"] = [
        {k: (float(v) if v is not None else None) for k, v in s.items()}
        for s in run["noisy_samples"]
    ]
    fpath = OUT_DIR / f"run_{run['run_id']:03d}.json"
    fpath.write_text(json.dumps(payload, indent=2))

print(f"Saved {len(runs)} runs to {OUT_DIR}/")

# ── Fig. 4 ────────────────────────────────────────────────────────────────────

fig = fig4_training_trajectories(runs, species="VCD", alt_text=True)
fig.savefig("fig4_training_trajectories.pdf")
print("Saved fig4_training_trajectories.pdf")
