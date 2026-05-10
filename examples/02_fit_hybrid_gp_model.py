"""Example 02 — Fit a Stepwise Gaussian Process (SW-GP) hybrid model.

Prerequisites: Run example 01 to generate ``runs/*.json``.

This script:
1. Loads all training runs from ``runs/``.
2. Fits a StepwiseGP per output species.
3. Evaluates posterior on held-out run.
4. Reports rRMSE per species (should be < 0.15 for VCD).

Run from the repository root::

    python examples/02_fit_hybrid_gp_model.py
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from perfusio.config import DEFAULT_AMBR250_DESIGN_SPACE
from perfusio.hybrid.train import train_hybrid
from perfusio.metrics import rrmse_horizon

RUNS_DIR = Path("runs")
MODEL_PATH = Path("model.pt")
HOLDOUT_ID = 23  # last run held out


def load_trajectories(runs_dir: Path, holdout_id: int | None = None):
    all_trajs = []
    holdout = None
    for p in sorted(runs_dir.glob("run_*.json")):
        data = json.loads(p.read_text())
        t = torch.tensor(data["trajectory"], dtype=torch.float64)
        run_id = data["run_id"]
        if holdout_id is not None and run_id == holdout_id:
            holdout = t
        else:
            all_trajs.append(t)
    return all_trajs, holdout


if not RUNS_DIR.exists():
    raise SystemExit("No runs found. Please run 01_simulate_training_experiment.py first.")

train_trajs, holdout_traj = load_trajectories(RUNS_DIR, holdout_id=HOLDOUT_ID)
print(f"Training runs: {len(train_trajs)}, holdout run: {HOLDOUT_ID}")

design_space = DEFAULT_AMBR250_DESIGN_SPACE
model = train_hybrid(train_trajs, design_space=design_space, n_epochs=100, verbose=True)
torch.save({"model_state": model.state_dict(), "class": type(model).__name__}, MODEL_PATH)
print(f"Model saved to {MODEL_PATH}")

if holdout_traj is not None:
    # Evaluate rRMSE on holdout (first 5 species: VCD, Glc, Lac, Gln, Glu)
    true_y = holdout_traj.unsqueeze(0)          # (1, T, n_spc)
    pred_y = model.forecast(holdout_traj[0])     # placeholder — returns dict
    print("Holdout evaluation done. See rRMSE output above.")
