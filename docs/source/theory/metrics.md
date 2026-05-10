# Evaluation Metrics

## Relative Root-Mean-Square Error (rRMSE)

Following Gadiyar et al. (2026) Equations 5–6, the rRMSE over a horizon of
$H$ steps is:

$$
\text{rRMSE}(h) = \frac{1}{N_{\text{tail}}}
  \sum_{i=1}^{N_{\text{tail}}}
  \sqrt{\frac{\sum_{t=T-H+1}^{T} \left(\hat{x}^{(i)}_t - x^{(i)}_t\right)^2}
             {\sum_{t=T-H+1}^{T} \left(x^{(i)}_t\right)^2}}
$$

A value < 0.10 (10%) is considered *excellent*; < 0.15 *good*.

## Prediction Interval Coverage

$$
\text{PIC}_{90} = \frac{1}{T}\sum_{t=1}^{T} \mathbf{1}\left[x_t \in [q_{0.05}, q_{0.95}]\right]
$$

Ideal value: 90%. Values < 80% indicate under-coverage (model over-confident).

## Sharpness

Mean half-width of the 90% prediction interval:

$$
\text{Sharpness} = \frac{1}{T}\sum_{t=1}^{T}(q_{0.95,t} - q_{0.05,t})
$$

Lower is better (sharper), subject to maintaining coverage.

## CRPS

Continuous Ranked Probability Score (energy form):

$$
\text{CRPS}(F, y) = \mathbb{E}_{X\sim F}|X - y| - \tfrac{1}{2}\mathbb{E}_{X,X'\sim F}|X - X'|
$$

Computed via MC samples. Lower is better.

## Multi-Objective Metrics

### IGD+

Improved Inverted Generational Distance:

$$
\text{IGD}^+(\mathcal{A}, \mathcal{R}) = \frac{1}{|\mathcal{R}|}
  \sum_{\mathbf{r}\in\mathcal{R}} d^+(\mathbf{r}, \mathcal{A})
$$

where $d^+(\mathbf{r},\mathcal{A}) = \min_{\mathbf{a}\in\mathcal{A}} \|\max(\mathbf{r}-\mathbf{a},\mathbf{0})\|$.

### $\epsilon$-Indicator

$$
I_\epsilon(\mathcal{A}, \mathcal{R}) = \max_{\mathbf{r}\in\mathcal{R}}
  \min_{\mathbf{a}\in\mathcal{A}} \max_i(r_i - a_i)
$$

$I_\epsilon \leq 0$ means $\mathcal{A}$ weakly dominates $\mathcal{R}$.
