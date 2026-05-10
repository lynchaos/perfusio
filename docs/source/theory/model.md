# Model Theory

## Overview

`perfusio` implements the self-driving bioprocess methodology of Gadiyar et al. (2026).
The core model is a **hybrid state-space model** that combines:

1. A **mechanistic skeleton** (ODEs for CHO cell culture kinetics), and
2. A **Gaussian Process residual** layer (Stepwise-GP, SW-GP) that corrects for model misspecification.

## Mechanistic Model

The CHO perfusion model tracks $n = 9$ state variables:

$$
\mathbf{x}(t) \in \mathbb{R}^{9}
$$

| Index | Symbol | Units |
|-------|--------|-------|
| 0 | $X_v$ (VCD) | $10^6$ cells mL$^{-1}$ |
| 1 | $X_d$ (Viability) | fraction |
| 2 | $S_{\text{glc}}$ (Glucose) | g L$^{-1}$ |
| 3 | $S_{\text{gln}}$ (Glutamine) | mmol L$^{-1}$ |
| 4 | $S_{\text{glu}}$ (Glutamate) | mmol L$^{-1}$ |
| 5 | $S_{\text{lac}}$ (Lactate) | mmol L$^{-1}$ |
| 6 | $S_{\text{amm}}$ (Ammonia) | mmol L$^{-1}$ |
| 7 | $S_{\text{pyr}}$ (Pyruvate) | mmol L$^{-1}$ |
| 8 | $P$ (Titer) | mg L$^{-1}$ |

### Growth Kinetics

Specific growth rate (dual Monod with inhibition):

$$
\mu = \mu_{\max}
  \cdot \frac{S_{\text{glc}}}{K_S + S_{\text{glc}}}
  \cdot \frac{S_{\text{gln}}}{K_N + S_{\text{gln}}}
  \cdot \left(1 - \frac{X_v}{X_{\max}}\right)
$$

with $\mu_{\max} = 0.040\;\mathrm{h}^{-1}$, $K_S = 0.15$, $K_N = 0.04$.

### Glucose Consumption

Pirt maintenance + growth-coupled:

$$
q_S = \frac{\mu}{Y_{XS}} + m_S
$$

with Warburg switch when $S_{\text{lac}} > L_{\text{thresh}}$.

### Product Formation

Luedeking–Piret kinetics:

$$
q_P = \alpha \mu + \beta
$$

### Mass Balances (continuous perfusion)

$$
\frac{dX_v}{dt} = (\mu - \mu_d - D_b)\,X_v
$$

$$
\frac{dS}{dt} = D_f\,(S_f - S) - D_h\,S - q_S\,X_v
$$

where $D_f$ is the perfusion (feed) dilution rate and $D_h = D_f - D_b$ the
harvest dilution rate.

## Stepwise-GP Residual Layer

The SW-GP predicts the next absolute state $\mathbf{c}_{t+1}$ directly (not a rate
residual), trained on one-step pairs $(\mathbf{c}_t, \mathbf{u}_t, t) \to \mathbf{c}_{t+1}$:

$$
\mathbf{c}_{t+1} = f_{\text{GP}}(\mathbf{c}_t,\;\mathbf{u}_t,\;t)
$$

The hybrid model decomposes predictions as:

$$
\mathbf{c}_{t+1} = \underbrace{\mathbf{c}_t + \Delta t\,\mathbf{r}_{\text{mech}}}_{\text{mechanistic Euler}} + \underbrace{(\hat{\mathbf{c}}_{t+1} - \mathbf{c}_{\text{mech}})}_{\text{GP residual}}
$$

where $\hat{\mathbf{c}}_{t+1}$ is the GP posterior mean and $\mathbf{c}_{\text{mech}}$ is the
mechanistic Euler prediction. This form ensures the mechanistic prior anchors
extrapolation while the GP corrects in-distribution errors.

Rollout is performed either via **moment-matching** (fast, propagates mean and
variance analytically) or **Monte Carlo** (unbiased, default $S = 100$ paths).

## References

- Gadiyar, C. J., et al. (2026). *Biotechnology and Bioengineering*, 123(2), 391–405.
- Hutter, S., et al. (2021). *Computers & Chemical Engineering*, 151, 107373.
- Cruz-Bournazou, M. N., et al. (2022). *Digital Chemical Engineering*, 1, 100005.
