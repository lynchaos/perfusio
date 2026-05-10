"""Example 04 — Multi-objective BED: Pareto front exploration.

Simultaneously optimises Titer and VCD viability (dual objectives).
Plots the resulting Pareto front (Fig. 7 reproduction).

Run::

    python examples/04_multiobjective_bed.py
"""

from __future__ import annotations

import gpytorch
import torch
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.utils.multi_objective.box_decompositions.non_dominated import (
    FastNondominatedPartitioning,
)

from perfusio.bed.acquisitions import build_acquisition
from perfusio.bed.pareto import compute_pareto_front, hypervolume
from perfusio.bed.search import optimise_acquisition
from perfusio.config import DEFAULT_AMBR250_DESIGN_SPACE
from perfusio.viz.static import fig7_pareto_front
from perfusio.viz.theme import apply_theme

apply_theme()

DS = DEFAULT_AMBR250_DESIGN_SPACE
BOUNDS = DS.bounds_tensor()
dim = BOUNDS.shape[1]
n_ref = 16

torch.manual_seed(0)
train_X = BOUNDS[0] + torch.rand(n_ref, dim) * (BOUNDS[1] - BOUNDS[0])
# Two simulated objectives: Titer (↑) and VCD viability fraction (↑)
train_Y = torch.rand(n_ref, 2, dtype=torch.float64)
train_Y[:, 0] = train_Y[:, 0] * 600  # Titer [mg/L]
train_Y[:, 1] = train_Y[:, 1] * 0.95  # VCV fraction

gp = SingleTaskGP(train_X, train_Y)
mll = gpytorch.mlls.ExactMarginalLogLikelihood(gp.likelihood, gp)
fit_gpytorch_mll(mll)
gp.eval()

ref_point = torch.zeros(2, dtype=torch.float64)
partitioning = FastNondominatedPartitioning(ref_point=ref_point, Y=train_Y)

acqf = build_acquisition("qNEHVI", gp, ref_point=ref_point, partitioning=partitioning)
candidate, value = optimise_acquisition(acqf, bounds=BOUNDS, q=1)

mask = compute_pareto_front(train_Y)
hv = hypervolume(train_Y[mask], ref_point)
print(f"Pareto front size: {mask.sum()}, HV: {hv:.2f}")

titer = train_Y[:, 0].tolist()
vcv = train_Y[:, 1].tolist()
fig = fig7_pareto_front(titer, vcv, pareto_mask=mask.tolist(), alt_text=True)
fig.savefig("fig7_pareto_front.pdf")
print("Saved fig7_pareto_front.pdf")
