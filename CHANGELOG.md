# Changelog

All notable changes to `perfusio` are documented in this file.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/)
and the project uses [Semantic Versioning](https://semver.org/).

---

## [Unreleased]

### Fixed

- **`bed/pareto.py` — `compute_pareto_front()` strict-dominance criterion**:
  The implementation used `(Y[i] < Y).all(dim=1)` to test whether row `j`
  dominates row `i`, requiring `j` to be *strictly* better on **every**
  objective. This excludes valid domination when `j` ties `i` on one objective
  but is strictly better on another. Corrected to the standard criterion:
  `(Y[i] <= Y).all(dim=1) & (Y[i] < Y).any(dim=1)` (j weakly better on all,
  strictly better on at least one). Downstream `hypervolume` and `qNEHVI`
  computations now operate on the correct Pareto front.

- **`twin/digital_twin.py` — `_obs_buffer` never populated (online learning dead)**:
  `_async_step` collected new sample measurements but never appended them to
  `_obs_buffer`, so `_retrain` always received an empty buffer and online GP
  retraining was completely inoperative throughout an entire experiment.
  Fixed by appending `{"sample": sample, "day": day, "controls_tensor": ctrl_tensor}`
  at the end of each `_async_step` call after the BED decision is available.

- **`twin/digital_twin.py` — `_retrain()` wrong model reference and malformed X/Y tensors**:
  Three bugs in the online-retraining routine:
  (1) `model=self.hybrid.gp_model` passed the `StepwiseGP` wrapper instead of
  the inner `MultiTaskRateGP` (ExactGP), which is the object that owns
  `set_train_data`.
  (2) `new_x` was constructed as `torch.arange(N).unsqueeze(-1)` (shape `(N, 1)`)
  instead of the required indexed multi-task format `(N × n_species, 17)` with
  columns `[species×9, controls×6, day, task_id]`.
  (3) `new_y` was passed with shape `(N, n_species)` instead of the flat
  `(N × n_species,)` scalar targets the GaussianLikelihood expects.
  Complete rewrite of `_retrain` builds correct one-step pairs in indexed
  multi-task format, using controls stored in the now-populated `_obs_buffer`.

- **`embedding/transfer.py` — `embed_and_concat()` does not exist on `EntityEmbedding`**:
  Both `TransferLearner.warm_start` and `joint_finetune` called
  `self.embedding.embed_and_concat(train_x, clone_ids)`, a method that was never
  defined on `EntityEmbedding` (which only exposes `forward(clone_ids)`). Would
  raise `AttributeError` at runtime. Fixed by calling `self.embedding(clone_ids)`
  to get the embedding tensor and then `torch.cat([train_x, clone_emb], dim=1)`
  to augment the feature matrix in both methods.

- **`cli.py` — `reproduce-figures` showed all runs as Pareto-optimal**:
  The `reproduce_figures` command passed all 27 run endpoints directly to
  `fig7_pareto_front(pareto_titer, pareto_vcv)` without first computing the
  Pareto front. Every point was rendered as Pareto-optimal. Fixed by calling
  `compute_pareto_front` on the `(27, 2)` objective matrix, passing only the
  non-dominated subset as the front and all points as the `feasible_*`
  background scatter.

- **`mechanistic/kinetics.py` — `q_lac()` VCD scaling bug**: During the
  metabolic-switch (lactate-consumption) branch for Clone X, the volumetric
  lactate consumption rate was a constant (`-lac_consump_rate * 1e9`) that
  ignored cell density. Added `vcd` parameter; rate is now correctly scaled
  as `-lac_consump_rate * vcd * 1e9` (units: mmol L⁻¹ h⁻¹).  All other
  volumetric rates already multiplied by VCD; this brings `q_lac` into
  consistency. Call site in `ode_rhs` updated to pass `vcd=max(vcd, 0.0)`.

- **`gp/stepwise.py` — `_rollout_mm()` wrong quantiles**: The 10th/90th
  percentile z-scores used `math.log(0.10/0.90) ≈ −2.197` (the logit
  function) instead of the Gaussian inverse CDF. Replaced with
  `math.sqrt(2) * math.erfinv(2*p − 1)` giving the correct `z_10 ≈ −1.2816`.

- **`simulator/cho_perfusion.py` — deprecated `asyncio.get_event_loop()`**:
  Changed to `asyncio.get_running_loop()`, which is the correct call inside
  an already-running event loop (Python 3.10+ deprecation).

- **`twin/digital_twin.py` — deprecated `asyncio.get_event_loop()`**:
  Replaced `get_event_loop().run_until_complete(…)` in `step()` and
  `run_sync()` with `asyncio.run(…)`, the recommended Python 3.10+ pattern
  for running coroutines from synchronous code.

- **`hybrid/model.py` — `predict_next_state()` double-counted mechanistic
  contribution**: The original code computed `c_{t+1} = c_t + dt*r_mech +
  gp_mean` where `gp_mean` is the GP's direct prediction of `c_{t+1}` (not
  a rate). This effectively added the mechanistic Euler step twice. Fixed to
  correctly decompose: `c_{t+1} = c_mech + (gp_mean − c_mech)` where
  `c_mech = c_t + dt*r_mech`, making the GP residual structure explicit.

- **`cli.py` `train` command — `PerfusionKernel` input dimension mismatch**:
  `PerfusionKernel(n_state_dims=15)` expects 17-dim inputs (`n_state_dims +
  day + task_id`), but the training matrix `train_x` had only 16 columns
  (species + controls + day, no task_id). Fixed by appending a zeros
  `task_id` column before model construction.

- **`gp/exact_gp.py` — `predict_with_ci()` device mismatch**: The `erfinv`
  scalar tensor was created without `device=pred.mean.device`, causing a
  CPU/GPU mismatch on non-CPU devices. Fixed by passing `device=` explicitly.

- **`cli.py` `simulate` docstring**: Corrected "24-run Box-Behnken" to
  "27-run Box-Behnken" (4-factor BBD with 3 center points = 24 + 3 = 27
  runs).

- **`viz/static.py` — `fig4_training_trajectories()` subplot bug**: The
  computed `subset` variable was never used; both panels iterated over the
  full `runs` list, rendering identical charts. Fixed to correctly split runs
  by `clone_labels` (CloneX left, CloneY right) with an even index-based
  split as fallback when labels are absent.

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
