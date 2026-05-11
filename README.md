# perfusio

`perfusio` is an open, peer-reviewable reference implementation of the self-driving perfusion methodology of Gadiyar et al. (2026) and Hutter et al. (2021), providing stepwise Gaussian Process hybrid models, entity-embedding transfer learning, 11 Bayesian experimental design acquisitions, and a fully-instrumented online digital twin for CHO cell perfusion bioprocesses.

[![CI](https://github.com/lynchaos/perfusio/actions/workflows/ci.yml/badge.svg)](https://github.com/lynchaos/perfusio/actions/workflows/ci.yml)
[![Docs](https://github.com/lynchaos/perfusio/actions/workflows/docs.yml/badge.svg)](https://lynchaos.github.io/perfusio)
[![PyPI](https://img.shields.io/pypi/v/perfusio)](https://pypi.org/project/perfusio)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/pypi/pyversions/perfusio)](https://pypi.org/project/perfusio)

---

## Quick-start

```bash
pip install perfusio
python -c "from perfusio.simulator.cho_perfusion import CHOSimulator; print('OK')"
```

### Reproduce all paper figures

```bash
pip install perfusio
python examples/reproduce_paper_figures.py
# → paper_figures/{fig4,fig7,fig8}.{pdf,png,svg}
```

### CLI

```bash
# Generates 27 runs (4-factor Box-Behnken design, 3 centre points)
perfusio simulate --clone CloneX --n-days 28 --out runs/
perfusio train   --runs runs/ --model-out model.pt
perfusio run     --model model.pt --connector ambr250 --dashboard
perfusio reproduce-figures --out paper_figures/
```

---

## What's inside

| Module | Description |
|--------|-------------|
| `perfusio.mechanistic` | CHO kinetics ODEs (dual-Monod, Pirt, Warburg, Luedeking–Piret) |
| `perfusio.gp` | PerfusionKernel, JackknifeEnsemble, StepwiseGP |
| `perfusio.embedding` | Entity-embedding transfer (Hutter 2021) |
| `perfusio.hybrid` | Hybrid state-space model + online retraining |
| `perfusio.bed` | 11 BED acquisitions + Pareto / HV utilities |
| `perfusio.simulator` | CHOSimulator, Box-Behnken / LHC DoE, noise model |
| `perfusio.twin` | DigitalTwin, audit, notifications, scheduler |
| `perfusio.connectors` | OPC UA, SQL, filesystem, ambr®250 emulator |
| `perfusio.metrics` | rRMSE, PI coverage, CRPS, IGD+, ε-indicator |
| `perfusio.viz` | Static figures (Matplotlib) + interactive (Plotly/Dash) |
| `perfusio.cli` | Typer CLI |

---

## Installation

### CPU (default)

```bash
pip install perfusio
```

### GPU

```bash
pip install perfusio[gpu]
```

### Dashboard

```bash
pip install perfusio[dash]
perfusio run --dashboard
```

### Development

```bash
git clone https://github.com/lynchaos/perfusio.git
cd perfusio
pip install -e ".[dev]"
pre-commit install
pytest
```

---

## Citation

If you use `perfusio`, please cite:

```bibtex
@article{gadiyar2026,
  author  = {Gadiyar, Chiraag J. and others},
  title   = {Self-Driving Development of Perfusion Processes
             for Monoclonal Antibody Production},
  journal = {Biotechnology and Bioengineering},
  year    = {2026},
  volume  = {123},
  number  = {2},
  pages   = {391--405},
  doi     = {10.1002/bit.28631},
}

@article{hutter2021,
  author  = {Hutter, Clemens and von Stosch, Moritz and Cruz Bournazou, Mariano N. and Butt{\'e}, Alessandro},
  title   = {Knowledge transfer across cell lines using hybrid Gaussian
             process models with entity embedding vectors},
  journal = {Biotechnology and Bioengineering},
  year    = {2021},
  volume  = {118},
  number  = {12},
  pages   = {4710--4725},
  doi     = {10.1002/bit.27907},
}
```

---

## Limitations

See [LIMITATIONS.md](LIMITATIONS.md) for a full list of out-of-scope items,
including CQA modelling, GMP validation, and proprietary media formulations.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## Code of Conduct

See [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).

## Maintainer

**Kemal Yaylali** — [kemal.yaylali.uk](https://kemal.yaylali.uk)

- GitHub: [@lynchaos](https://github.com/lynchaos)
- X / Twitter: [@kmlyyll](https://x.com/kmlyyll)
- LinkedIn: [kemalyaylali](https://www.linkedin.com/in/kemalyaylali/)
- Support: [support@yaylali.uk](mailto:support@yaylali.uk)

## Licence

Apache-2.0 — see [LICENSE](LICENSE).
