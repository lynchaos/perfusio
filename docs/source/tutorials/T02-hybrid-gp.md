# T02 — Fit the Hybrid GP Model

**Goal:** Train a `HybridStateSpaceModel` (mechanistic skeleton + SW-GP
residual) on the Box-Behnken dataset from T01 and evaluate held-out
prediction accuracy.

**Script:** `examples/02_fit_hybrid_gp_model.py`

**Prerequisite:** Run T01 to generate `runs/*.json`.

## What the script does

1. Loads 26 training runs from `runs/` (one run withheld as `run_023.json`).
2. Calls `train_hybrid(trajectories, design_space, n_epochs=100)` which:
   - Constructs `HybridStateSpaceModel` (mechanistic prior + `MultiTaskRateGP`).
   - Fits the GP via L-BFGS with Adam fallback.
3. Saves the model checkpoint to `model.pt`.
4. Reports rRMSE on the held-out run.

## Running

```bash
python examples/02_fit_hybrid_gp_model.py
# → model.pt
```

Alternatively, via the CLI (trains on all runs):

```bash
perfusio train --runs runs/ --model-out model.pt
```

## Key code

```python
from perfusio.hybrid.train import train_hybrid
from perfusio.config import DEFAULT_AMBR250_DESIGN_SPACE
import torch, json
from pathlib import Path

trajectories = [
    torch.tensor(json.loads(p.read_text())["trajectory"], dtype=torch.float64)
    for p in sorted(Path("runs").glob("run_*.json"))
]

model = train_hybrid(trajectories, design_space=DEFAULT_AMBR250_DESIGN_SPACE, n_epochs=100)
torch.save({"model_state": model.state_dict()}, "model.pt")
```

## GP architecture

| Component | Detail |
|-----------|--------|
| Kernel | `PerfusionKernel` = Matérn-5/2 (state) × Linear (day) × Index (task) |
| Input dim | 17 (9 species + 6 controls + day + task_id) |
| Tasks | 9 (one per output species) |
| Likelihood | `GaussianLikelihood` (scalar, shared noise) |
| Training | L-BFGS → Adam fallback; 100 epochs |
| Rollout | Monte Carlo (100 paths) or moment-matching |

## Expected metrics (held-out run 023)

| Species | rRMSE target |
|---------|-------------|
| VCD | < 0.10 |
| Glucose | < 0.15 |
| Titer | < 0.20 |

## Next step

Proceed to [T03 — Single-Objective BED](T03-bed.md).
