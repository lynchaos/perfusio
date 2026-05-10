# Changelog

All notable changes to `perfusio` are documented in this file.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/)
and the project uses [Semantic Versioning](https://semver.org/).

---

## [Unreleased]

## [0.1.0] — 2024-01-01

### Added

- `perfusio.mechanistic` — CHO kinetics ODEs (dual-Monod, Pirt maintenance,
  Warburg metabolic switching, Luedeking–Piret product formation).
- `perfusio.gp` — PerfusionKernel, JackknifeEnsemble (K=20), StepwiseGP
  with MC and moment-matching rollout.
- `perfusio.embedding` — EntityEmbedding (d=4), TransferLearner with
  differential learning-rate fine-tuning.
- `perfusio.hybrid` — HybridStateSpaceModel, `train_hybrid`, `retrain_online`,
  `forecast_run`.
- `perfusio.bed` — 11 acquisition functions (PI, EI, LogEI, UCB, qEI,
  qLogEI, qUCB, qEHVI, qNEHVI, qNParEGO), Pareto front, hypervolume,
  BEDPolicy.
- `perfusio.simulator` — CHOSimulator, Box-Behnken / central-composite /
  Latin hypercube DoE, NoiseModel.
- `perfusio.twin` — DigitalTwin, AuditLogger (SHA-256), AlarmNotifier,
  DailyScheduler.
- `perfusio.connectors` — OPCUAConnector (asyncua, auto-reconnect, cert auth),
  SQLStore (8-table Alembic schema), FilesystemStore (CSV/Parquet/Excel),
  Ambr250Emulator (5-reactor ensemble).
- `perfusio.metrics` — rRMSE horizon (Gadiyar Eqs. 5–6), PI coverage,
  sharpness, CRPS, hypervolume indicator, IGD+, ε-indicator.
- `perfusio.viz` — Static (Matplotlib) and interactive (Plotly/Dash)
  visualisation; Fig. 4, 6, 7, 8 reproduction.
- `perfusio.cli` — Typer commands: `simulate`, `train`, `run`,
  `reproduce-figures`.
- Full test suite: >92% line / >85% branch coverage; CI on Linux/macOS/Windows
  × Python 3.11/3.12/3.13.
- Sphinx documentation with MathJax theory pages.
- Apache-2.0 licence; CITATION.cff citing all five source papers.
