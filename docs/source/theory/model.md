# Model Theory

## Overview

`perfusio` implements the self-driving bioprocess methodology of Gadiyar et al. (2026).
The core model is a **hybrid state-space model** that combines:

1. A **mechanistic skeleton** (ODEs for CHO cell culture kinetics), and
2. A **Gaussian Process residual** layer (Stepwise-GP, SW-GP) that corrects for model misspecification.

## Mechanistic Model

The CHO perfusion model tracks $n = 13$ state variables:

$$
\mathbf{x}(t) \in \mathbb{R}^{13}
$$

| Index | Symbol | Units |
|-------|--------|-------|
| 0 | $X_v$ | $10^6$ cells mL$^{-1}$ |
| 1 | $X_d$ | $10^6$ cells mL$^{-1}$ |
| 2 | $S_{\text{glc}}$ | g L$^{-1}$ |
| 3 | $S_{\text{lac}}$ | g L$^{-1}$ |
| 4 | $S_{\text{gln}}$ | mmol L$^{-1}$ |
| 5 | $S_{\text{glu}}$ | mmol L$^{-1}$ |
| 6 | $S_{\text{asn}}$ | mmol L$^{-1}$ |
| 7 | $S_{\text{asp}}$ | mmol L$^{-1}$ |
| 8 | $P$ | mg L$^{-1}$ |
| 9 | $O_2$ | % |
| 10 | $CO_2$ | mmHg |
| 11 | $\text{osm}$ | mOsm kg$^{-1}$ |
| 12 | $V$ | L |

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

The SW-GP corrects residuals over a rolling window of $w$ past days:

$$
\Delta \mathbf{x}(t) = f_{\text{GP}}(\mathbf{x}(t-w:t),\;\mathbf{u})
$$

Rollout is performed either via **moment-matching** (fast, closed form for
squared-exponential kernels) or **Monte Carlo** (unbiased, $S = 512$ paths).

## References

- Gadiyar, C. J., et al. (2026). *Biotechnology and Bioengineering*, 123(2), 391–405.
- Hutter, S., et al. (2021). *Computers & Chemical Engineering*, 151, 107373.
- Cruz-Bournazou, M. N., et al. (2022). *Digital Chemical Engineering*, 1, 100005.
