# T01 — Simulate a Training Experiment

**Goal:** Generate a 27-run Box-Behnken training dataset using the CHO
`CloneX` simulator and produce Fig. 4 (training trajectories).

**Script:** `examples/01_simulate_training_experiment.py`

## What the script does

1. Instantiates `CHOSimulator(clone="CloneX", seed=0)`.
2. Calls `generate_box_behnken_experiment(n_days=28, seed=0)`, which returns
   27 runs (4-factor Box-Behnken design with 3 centre points).
3. Serialises each run to `runs/run_NNN.json`.
4. Plots VCD trajectories and saves `fig4_training_trajectories.pdf`.

## Running

```bash
python examples/01_simulate_training_experiment.py
# → runs/run_000.json … run_026.json
# → fig4_training_trajectories.pdf
```

Alternatively, via the CLI:

```bash
perfusio simulate --clone CloneX --n-days 28 --out runs/
```

## Key code

```python
from perfusio.simulator.cho_perfusion import CHOSimulator
from perfusio.viz.static import fig4_training_trajectories

sim = CHOSimulator(clone="CloneX", seed=0)
runs = sim.generate_box_behnken_experiment(n_days=28, seed=0)
# len(runs) == 27  (24 edge + 3 centre)

fig = fig4_training_trajectories(runs, species="VCD")
fig.savefig("fig4.pdf")
```

## Design space

The four factors and their ranges used in the Box-Behnken design:

| Factor | Low | Centre | High | Units |
|--------|-----|--------|------|-------|
| Perfusion rate | 0.5 | 1.0 | 1.5 | vvd |
| Bleed rate | 0.05 | 0.10 | 0.15 | vvd |
| Glucose setpoint | 3.0 | 4.5 | 6.0 | g L⁻¹ |
| Temperature | 36.0 | 36.5 | 37.0 | °C |

## Expected output

- VCD reaches ~8–12 × 10⁶ cells mL⁻¹ at steady state under optimal conditions.
- Titer accumulates to ~400–600 mg L⁻¹ by day 28 for high-performing runs.

## Next step

Proceed to [T02 — Fit the Hybrid GP Model](T02-hybrid-gp.md).
