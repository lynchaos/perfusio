# T04 — Multi-Objective BED and Pareto Exploration

**Goal:** Simultaneously optimise Titer and VCD viability as competing
objectives, track the Pareto front, and reproduce Fig. 7.

**Script:** `examples/04_multiobjective_bed.py`

## What the script does

1. Trains a two-output `SingleTaskGP` on 16 random initial runs.
2. Computes the Pareto front via `compute_pareto_front(train_Y)`.
3. Calculates the hypervolume indicator.
4. Builds a `qNEHVI` acquisition using BoTorch's
   `FastNondominatedPartitioning`.
5. Optimises and recommends the next batch.
6. Plots and saves `fig7_pareto_front.pdf`.

## Running

```bash
python examples/04_multiobjective_bed.py
# → fig7_pareto_front.pdf
```

## Key code

```python
from perfusio.bed.acquisitions import build_acquisition
from perfusio.bed.pareto import compute_pareto_front, hypervolume
from perfusio.viz.static import fig7_pareto_front
from botorch.utils.multi_objective.box_decompositions.non_dominated import (
    FastNondominatedPartitioning,
)

ref_point = torch.zeros(2, dtype=torch.float64)
partitioning = FastNondominatedPartitioning(ref_point=ref_point, Y=train_Y)
acqf = build_acquisition("qNEHVI", gp, ref_point=ref_point, partitioning=partitioning)

mask = compute_pareto_front(train_Y)   # shape (N,) boolean
hv   = hypervolume(train_Y[mask], ref_point)
```

## Pareto dominance criterion

A point $\mathbf{y}^*$ dominates $\mathbf{y}$ if:

$$
y^*_i \geq y_i \;\forall i \quad \text{and} \quad y^*_j > y_j \;\text{for at least one } j
$$

`compute_pareto_front` implements this correctly using:

```python
(Y[i] <= Y).all(dim=1) & (Y[i] < Y).any(dim=1)
```

## Multi-objective acquisitions

| Acquisition | Notes |
|-------------|-------|
| `qEHVI` | Fast; assumes noiseless observations |
| `qNEHVI` | Handles noisy observations; recommended default |
| `qNParEGO` | Scalarisation; faster for 3+ objectives |

## Next step

Proceed to [T05 — Transfer Learning Across Cell Lines](T05-transfer.md).
