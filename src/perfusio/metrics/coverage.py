"""Probabilistic forecast calibration metrics.

Provides:
- :func:`pi_coverage`: empirical coverage of prediction intervals.
- :func:`sharpness`: mean width of prediction intervals (lower = sharper).
- :func:`crps`: Continuous Ranked Probability Score (lower = better).

All functions accept batched inputs to facilitate per-species analysis.

References
----------
.. [Gneiting2007] Gneiting, T., & Raftery, A. E. (2007). Strictly Proper
   Scoring Rules, Prediction, and Estimation. JASA 102(477).
"""

from __future__ import annotations

import torch
from torch import Tensor


def pi_coverage(
    true: Tensor,
    q_lo: Tensor,
    q_hi: Tensor,
) -> Tensor:
    """Empirical coverage of prediction intervals.

    Parameters
    ----------
    true:
        Observed values, shape ``(..., n_species)``.
    q_lo:
        Lower quantile predictions, same shape as *true*.
    q_hi:
        Upper quantile predictions, same shape as *true*.

    Returns
    -------
    Tensor
        Coverage probability per species, shape ``(n_species,)``.
        Value 1.0 = all observations within the PI.

    Examples
    --------
    >>> import torch
    >>> from perfusio.metrics import pi_coverage
    >>> y = torch.tensor([[1.0, 2.0], [1.5, 2.5], [0.5, 1.5]])
    >>> lo = torch.zeros_like(y)
    >>> hi = torch.full_like(y, 3.0)
    >>> pi_coverage(y, lo, hi)
    tensor([1., 1.])
    """
    inside = (true >= q_lo) & (true <= q_hi)
    return inside.float().mean(dim=tuple(range(inside.ndim - 1)))


def sharpness(q_lo: Tensor, q_hi: Tensor) -> Tensor:
    """Mean prediction interval width per species.

    Parameters
    ----------
    q_lo:
        Lower quantile, shape ``(..., n_species)``.
    q_hi:
        Upper quantile, shape ``(..., n_species)``.

    Returns
    -------
    Tensor
        Mean width per species, shape ``(n_species,)``.
    """
    width = (q_hi - q_lo).abs()
    return width.mean(dim=tuple(range(width.ndim - 1)))


def crps(
    true: Tensor,
    samples: Tensor,
) -> Tensor:
    """Energy-form Continuous Ranked Probability Score (CRPS).

    Uses the energy score decomposition:

    .. math::
        \\text{CRPS} = \\mathbb{E}[|X - y|] - \\frac{1}{2}\\mathbb{E}[|X - X'|]

    where :math:`X, X'` are i.i.d. draws from the predictive distribution.

    Parameters
    ----------
    true:
        Observations, shape ``(N, n_species)``.
    samples:
        Posterior predictive samples, shape ``(N, S, n_species)``
        where *S* is the number of samples.

    Returns
    -------
    Tensor
        Mean CRPS per species, shape ``(n_species,)``.

    Notes
    -----
    An energy-score estimate is used rather than the exact Gaussian CRPS to
    support non-Gaussian predictive distributions from the MC rollout.
    """
    # E[|X - y|]  mean over samples and observations
    diff_xy = (samples - true.unsqueeze(1)).abs().mean(dim=(0, 1))  # (n_species,)

    # E[|X - X'|]  using all pairs of samples (unbiased)
    S = samples.shape[1]
    # Efficient: E[|X-X'|] = 2 * sum_{i<j} |x_i - x_j| / (S*(S-1))
    # Vectorised as: mean over all pairs (i,j) with i != j
    pair_diff = (samples.unsqueeze(2) - samples.unsqueeze(1)).abs()  # (N, S, S, n_spc)
    # mask diagonal
    eye = torch.eye(S, dtype=torch.bool, device=samples.device)
    pair_diff = pair_diff.masked_fill(eye.unsqueeze(0).unsqueeze(-1), 0.0)
    e_pair = pair_diff.sum(dim=(1, 2)) / (S * (S - 1))  # (N, n_species)
    e_pair_mean = e_pair.mean(dim=0)  # (n_species,)

    return diff_xy - 0.5 * e_pair_mean
