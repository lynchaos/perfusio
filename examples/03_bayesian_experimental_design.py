"""Example 03 — Bayesian Experimental Design: single next experiment.

Prerequisites: model.pt from example 02.

This script:
1. Loads the trained hybrid model.
2. Builds an EI acquisition function.
3. Optimises over the design space.
4. Prints the recommended next operating point.

Run::

    python examples/03_bayesian_experimental_design.py
"""

from __future__ import annotations

from pathlib import Path

import torch

from perfusio.config import DEFAULT_AMBR250_DESIGN_SPACE
from perfusio.bed.acquisitions import build_acquisition
from perfusio.bed.search import optimise_acquisition


DS = DEFAULT_AMBR250_DESIGN_SPACE
BOUNDS = DS.bounds_tensor()   # (2, n_controls)

# ── Build a minimal single-output surrogate (placeholder) ─────────────────────
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
import gpytorch

torch.manual_seed(0)
n_obs = 12
dim = BOUNDS.shape[1]
train_X = torch.rand(n_obs, dim, dtype=torch.float64)
train_X = BOUNDS[0] + train_X * (BOUNDS[1] - BOUNDS[0])
train_Y = torch.randn(n_obs, 1, dtype=torch.float64)  # simulated titer

gp = SingleTaskGP(train_X, train_Y)
mll = gpytorch.mlls.ExactMarginalLogLikelihood(gp.likelihood, gp)
fit_gpytorch_mll(mll)
gp.eval()

# ── Optimise EI acquisition ───────────────────────────────────────────────────
best_f = float(train_Y.max())
acqf = build_acquisition("LogEI", gp, best_f=best_f)
candidate, value = optimise_acquisition(acqf, bounds=BOUNDS, q=1)

print("Next recommended experiment:")
for name, val in zip(DS.control_names, candidate.squeeze(0).tolist()):
    print(f"  {name:20s}: {val:.4f}")
print(f"LogEI value: {float(value):.4f}")
