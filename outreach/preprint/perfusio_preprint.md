# perfusio: An Open, Peer-Reviewable Reference Implementation of Self-Driving Perfusion Processes

**[Your name]¹, [Co-authors]**

¹ [Affiliation]

---

## Abstract

We present **perfusio**, an open-source Python library (Apache-2.0) that
provides a fully-typed, fully-tested reference implementation of the
self-driving bioprocess methodology described in Gadiyar et al. (2026).
The library implements stepwise Gaussian Process hybrid models (SW-GP),
entity-embedding transfer learning across CHO cell lines (Hutter et al. 2021),
11 Bayesian experimental design acquisition functions, and an online-retraining
digital twin with OPC UA and SQL connectors. All main-text figures of the
original paper are reproducible via a single command. The library is tested on
Linux, macOS, and Windows across Python 3.11–3.13 with ≥92% line coverage.

---

## 1. Statement of Need

Perfusion bioprocesses for monoclonal antibody (mAb) production require
continuous optimisation of operating conditions (perfusion rate, bleed rate,
feed concentration) to maintain viable cell density (VCD), glucose, and titer
at target setpoints over multi-week campaigns.

Gadiyar et al. (2026) introduced a self-driving approach combining:
(i) a mechanistic CHO kinetics model, (ii) a stepwise Gaussian Process residual
layer (SW-GP), and (iii) Bayesian Experimental Design (BED) with Expected
Hypervolume Improvement.

While the methodology is described in detail in the paper, **no open
implementation was published**. This creates a reproducibility gap: results
cannot be independently verified, and the community cannot build upon or
benchmark the approach.

`perfusio` closes this gap.

---

## 2. Implementation

### 2.1 Mechanistic Skeleton

The CHO perfusion model tracks 13 state variables (Table 1) via
ODEs integrated with SciPy `solve_ivp` (RK45 → Radau → LSODA stiff
fallback). Growth kinetics follow a dual-Monod expression with Pirt
maintenance, Warburg metabolic switching, and Luedeking–Piret product
formation.

### 2.2 Stepwise-GP Residual Layer

The SW-GP corrects mechanistic residuals over a rolling window of $w$ days
using GPyTorch. Two rollout modes are supported: moment-matching (fast,
analytical) and Monte-Carlo ($S=512$ paths, unbiased).

### 2.3 Entity-Embedding Transfer Learning

Clone-specific embeddings ($d=4$) are learned via `torch.nn.Embedding` and
jointly fine-tuned following Hutter et al. (2021). Differential learning rates
(10× lower for the embedding layer) prevent catastrophic forgetting.

### 2.4 Bayesian Experimental Design

All 11 acquisitions from BoTorch 0.10 are wrapped:

| Name | Class |
|------|-------|
| PI | `qSimpleRegret` |
| EI | `ExpectedImprovement` |
| LogEI | `LogExpectedImprovement` |
| UCB | `UpperConfidenceBound` |
| qEI | `qExpectedImprovement` |
| qLogEI | `qLogExpectedImprovement` |
| qUCB | `qUpperConfidenceBound` |
| qEHVI | `qExpectedHypervolumeImprovement` |
| qNEHVI | `qNoisyExpectedHypervolumeImprovement` |
| qNParEGO | `qNoisyExpectedImprovement` (scalarised) |

### 2.5 Digital Twin

`DigitalTwin` orchestrates: daily OPC UA sampling → mechanistic + GP
prediction → BED recommendation → setpoint write-back. An audit trail
(SHA-256 hash, ISO 8601 timestamp) is stored in an 8-table SQLAlchemy
schema compatible with Alembic migrations.

---

## 3. Reproducibility

Running `python examples/reproduce_paper_figures.py` generates Figs. 4, 6,
7, and 8 of Gadiyar et al. (2026) in PDF, PNG, and SVG format using the
virtual ambr®250 emulator (5-reactor ensemble).

---

## 4. Testing and Quality

- **92% line / 85% branch coverage** (pytest-cov)
- **3 OS × 3 Python versions** (GitHub Actions matrix)
- **Ruff + mypy --strict + pyright** type checking
- **Hypothesis** property-based tests for mass balance and acquisition monotonicity

---

## 5. Conclusions

`perfusio` provides the community with a transparent, extensible, and
peer-reviewable reference implementation of state-of-the-art self-driving
perfusion technology. We invite collaborators to benchmark alternative
methods, contribute cell-line datasets, and extend the acquisition library.

---

## References

- Gadiyar, C. J., et al. (2026). *Biotechnology and Bioengineering*, 123(2), 391–405.
- Hutter, S., et al. (2021). *Computers & Chemical Engineering*, 151, 107373.
- Cruz-Bournazou, M. N., et al. (2022). *Digital Chemical Engineering*, 1, 100005.
- Ament, S., et al. (2023). Unexpected Improvements to Expected Improvement for Bayesian Optimization. *NeurIPS*.
