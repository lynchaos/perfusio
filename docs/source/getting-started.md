# Getting Started

## Installation

```bash
pip install perfusio
```

For GPU support:

```bash
pip install perfusio[gpu]
```

For the interactive dashboard:

```bash
pip install perfusio[dash]
```

### From source (development)

```bash
git clone https://github.com/lynchaos/perfusio.git
cd perfusio
pip install -e ".[dev]"
```

## Quick-start

```python
from perfusio.simulator.cho_perfusion import CHOSimulator
from perfusio.viz.theme import apply_theme
from perfusio.viz.static import fig4_training_trajectories

apply_theme()

sim = CHOSimulator(clone="CloneX", seed=0)
runs = sim.generate_box_behnken_experiment(n_days=28, seed=0)
# generate_box_behnken_experiment returns 27 runs (4-factor BBD, 3 centre points)
# Pass clone_labels to split left/right subplots by clone identity;
# omit it to split evenly by run index.
fig = fig4_training_trajectories(runs)
fig.savefig("fig4.pdf")
```

## Reproducing the paper figures

```bash
python examples/reproduce_paper_figures.py
```

All figures are written to `paper_figures/` in PDF, PNG, and SVG format.

## Running the CLI

```bash
# Generates 27 runs (4-factor Box-Behnken design, 3 centre points)
perfusio simulate --clone CloneX --n-days 28 --out runs/
perfusio train   --runs runs/ --model-out model.pt
perfusio run     --model model.pt --days 28 --clone CloneX --dashboard
perfusio reproduce-figures --data runs/ --out paper_figures/
```

## Citation

If you use `perfusio`, please cite both the original methodology paper and this library:

```bibtex
@article{gadiyar2026,
  author  = {Gadiyar, Chiraag J. and others},
  title   = {Self-Driving Development of Perfusion Processes for Monoclonal Antibody Production},
  journal = {Biotechnology and Bioengineering},
  year    = {2026},
  volume  = {123},
  number  = {2},
  pages   = {391--405},
}
```
