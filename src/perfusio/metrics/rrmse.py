"""Relative Root Mean Squared Error over a prediction horizon.

Implements Gadiyar et al. (2026) Equations 5–6:

.. math::
    \\text{rRMSE}_{k,h} =
        \\frac{1}{\\sigma_k}
        \\sqrt{
            \\frac{1}{R}
            \\sum_{r=1}^{R}
            \\left(
                \\hat{y}_{r,h,k} - y_{r,h,k}
            \\right)^2
        }

where :math:`h` is the prediction horizon step, :math:`k` is the species
index, :math:`R` is the number of runs and :math:`\\sigma_k` is the
standard deviation of the measured species computed across all runs and all
time steps (the "normalisation scale" of Gadiyar Eq. 6).

A truncated tail variant is also provided: only the first ``n_tail`` time
steps are excluded from normalisation to avoid division-by-small-variance
artefacts at inoculation.

References
----------
.. [Gadiyar2026] Gadiyar et al. (2026), Eqs. (5)–(6).
"""

from __future__ import annotations

from torch import Tensor


def rrmse_horizon(
    true: Tensor,
    pred: Tensor,
    horizon: int | None = None,
    n_tail: int = 0,
    eps: float = 1e-8,
) -> Tensor:
    """Relative RMSE across runs, time steps and species.

    Parameters
    ----------
    true:
        Ground-truth observations, shape ``(n_runs, T, n_species)``.
    pred:
        Model predictions (mean), shape ``(n_runs, T, n_species)``.
    horizon:
        If given, only the first ``horizon`` time steps are evaluated.
        This allows head-to-head comparison of 1-step vs. 3-step models.
    n_tail:
        Number of leading time steps to *exclude* from the normalisation
        denominator (avoids division by near-zero early-run variance).
    eps:
        Small constant added to the normalisation denominator for numerical
        stability.

    Returns
    -------
    Tensor
        Shape ``(T, n_species)`` if ``horizon`` is None, else
        ``(horizon, n_species)``.  Element ``[h, k]`` is the
        :math:`\\text{rRMSE}_{k,h}`.

    Examples
    --------
    >>> import torch
    >>> from perfusio.metrics import rrmse_horizon
    >>> y = torch.randn(5, 10, 9)
    >>> yhat = y + 0.1 * torch.randn_like(y)
    >>> rr = rrmse_horizon(y, yhat, horizon=3)
    >>> rr.shape
    torch.Size([3, 9])
    """
    if true.shape != pred.shape:
        msg = f"true and pred must have the same shape; got {true.shape} and {pred.shape}."
        raise ValueError(msg)

    if horizon is not None:
        true = true[:, :horizon, :]
        pred = pred[:, :horizon, :]

    # σ_k — normalisation: std over (runs × time) for each species
    # exclude n_tail leading steps from the std estimate
    if n_tail > 0:
        sigma = true[:, n_tail:, :].std(dim=(0, 1))  # (n_species,)
    else:
        sigma = true.std(dim=(0, 1))  # (n_species,)
    sigma = sigma.clamp(min=eps)  # numerical safety

    # MSE over runs for each (time, species)
    squared_err = (pred - true) ** 2  # (n_runs, T, n_species)
    mse = squared_err.mean(dim=0)  # (T, n_species)
    rmse = mse.sqrt()  # (T, n_species)

    # Normalise: broadcast sigma over time axis
    rrmse = rmse / sigma.unsqueeze(0)  # (T, n_species)
    return rrmse
