# Bayesian Experimental Design

## Objective Functions

### Single-objective: Target Tracking

$$
\text{OFV}(\mathbf{x}) = -\sum_{k} w_k \left(\hat{x}_k - x_k^*\right)^2
$$

where $\hat{x}_k$ is the model prediction and $x_k^*$ is the target for species $k$.

### Multi-objective: Hypervolume

The multi-objective OFV is the **Expected Hypervolume Improvement** (EHVI):

$$
\text{qNEHVI}(\mathbf{x}) = \mathbb{E}\left[\text{HVI}\left(\mathbf{f}(\mathbf{x}),\;\mathcal{P},\;\mathbf{r}\right)\right]
$$

where $\mathcal{P}$ is the current Pareto front and $\mathbf{r}$ is the reference point.

## Acquisition Functions

| Name | Type | Notes |
|------|------|-------|
| `PI` | Single-obj | Probability of Improvement |
| `EI` | Single-obj | Expected Improvement |
| `LogEI` | Single-obj | Numerically stable log-EI (Ament et al. 2023) |
| `UCB` | Single-obj | Upper Confidence Bound, $\beta$-parameterised |
| `qEI` | Batch | Monte-Carlo EI |
| `qLogEI` | Batch | Log-space MC EI |
| `qUCB` | Batch | MC UCB |
| `qEHVI` | Multi-obj | Expected HV Improvement |
| `qNEHVI` | Multi-obj | Noisy EHVI |
| `qNParEGO` | Multi-obj | Pareto-EGO scalarisation |

All acquisitions are optimised via `botorch.optim.optimize_acqf` with
`num_restarts=20`, `raw_samples=512`.

## Pareto Optimality

A point $\mathbf{y}$ is *Pareto-optimal* if there is no $\mathbf{y}'$ such that
$y'_i \geq y_i$ for all $i$ and $y'_j > y_j$ for at least one $j$.

The **hypervolume indicator** quantifies the quality of the Pareto front:

$$
\text{HV}(\mathcal{P},\;\mathbf{r}) = \lambda\left(\bigcup_{\mathbf{y}\in\mathcal{P}} [\mathbf{r},\mathbf{y}]\right)
$$

where $\lambda$ denotes the Lebesgue measure.
