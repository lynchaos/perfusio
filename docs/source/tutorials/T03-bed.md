# T03 — Bayesian Experimental Design (Single Objective)

**Goal:** Use a trained surrogate GP to recommend the next experiment via
`LogEI` acquisition, maximising titer.

**Script:** `examples/03_bayesian_experimental_design.py`

**Prerequisite:** `model.pt` from T02 (or the script's built-in
`SingleTaskGP` placeholder).

## What the script does

1. Builds (or loads) a GP surrogate over the 6-dimensional control space.
2. Calls `build_acquisition("LogEI", gp, best_f=best_f)` to construct a
   numerically-stable Expected Improvement acquisition.
3. Optimises the acquisition via `optimise_acquisition(acqf, bounds, q=1)`
   (20 restarts, 512 raw samples).
4. Prints the recommended next operating point.

## Running

```bash
python examples/03_bayesian_experimental_design.py
```

## Key code

```python
from perfusio.bed.acquisitions import build_acquisition
from perfusio.bed.search import optimise_acquisition
from perfusio.config import DEFAULT_AMBR250_DESIGN_SPACE

DS = DEFAULT_AMBR250_DESIGN_SPACE
bounds = DS.bounds_tensor()  # shape (2, n_controls)

acqf = build_acquisition("LogEI", surrogate_gp, best_f=current_best)
candidate, value = optimise_acquisition(acqf, bounds=bounds, q=1)
```

## Available acquisition functions

See [Choosing an Acquisition Function](../choosing-acquisition.md) for a
full guide. The most common choices:

| Use case | Recommended acquisition |
|----------|------------------------|
| Single run, noisy | `LogEI` |
| Batch of 4 | `qLogEI` |
| Exploration emphasis | `UCB` with high `β` |

## Acquisition parameters

`build_acquisition` accepts keyword arguments forwarded to BoTorch:

```python
build_acquisition("UCB", model, beta=2.0)
build_acquisition("qLogEI", model, best_f=0.42, num_samples=128)
build_acquisition("qNEHVI", model, ref_point=ref_pt, partitioning=part)
```

## Next step

Proceed to [T04 — Multi-Objective BED](T04-multiobjective.md).
