"""Step-wise GP (SW-GP) for multi-step-ahead trajectory prediction.

The SW-GP predicts the state at time :math:`t+1` from the state at time
:math:`t` (one-step GP).  For multi-step-ahead forecasting, predictions are
chained (rollout) with uncertainty propagated by:

1. **Monte Carlo sampling**: sample from the one-step posterior, feed the
   sample as the next step's input, repeat *h* times, and report percentiles
   over *n_samples* trajectories.

2. **Moment matching** (fast approximation): propagate mean and variance
   analytically through the one-step GP (Girard et al., 2003).  Used when
   speed is critical (BED inner loop).

Both approaches are exposed through a common ``predict_quantiles`` interface.

References
----------
.. [Girard2003] Girard, A., et al. (2003). Gaussian Process Priors With
   Uncertain Inputs Application to Multiple-Step Ahead Time Series
   Forecasting. NeurIPS.
.. [Gadiyar2026] Gadiyar et al. (2026), §3.2 (SW-GP rollout for BED OFV).
"""

from __future__ import annotations

import math

import gpytorch
import torch
from torch import Tensor


class StepwiseGP:
    """Step-wise GP with multi-step rollout.

    Wraps a fitted one-step GP model and provides :meth:`predict_quantiles`
    for horizon-*h* forecasts.

    Parameters
    ----------
    model:
        Fitted GPyTorch one-step GP model.
    likelihood:
        Corresponding likelihood module.
    n_species:
        Number of state species modelled.
    control_names:
        Ordered list of control variable names.

    Notes
    -----
    The one-step GP takes inputs of shape ``(N, n_species + n_controls + 1)``
    — species concentrations, controls, and culture day — and outputs
    ``(N, n_species)`` species concentration at the next time step.
    """

    def __init__(
        self,
        model: gpytorch.models.ExactGP,
        likelihood: gpytorch.likelihoods.Likelihood,
        n_species: int,
        control_names: list[str],
    ) -> None:
        self.model = model
        self.likelihood = likelihood
        self.n_species = n_species
        self.control_names = control_names

    def predict_quantiles(
        self,
        x0: Tensor,
        controls: Tensor,
        horizon: int,
        n_samples: int = 200,
        seed: int | None = None,
        method: str = "mc",
    ) -> dict[str, Tensor]:
        """Multi-step-ahead prediction with uncertainty quantification.

        Parameters
        ----------
        x0:
            Initial state, shape ``(n_species,)`` or ``(B, n_species)``.
        controls:
            Control trajectories, shape ``(horizon, n_controls)`` or
            ``(B, horizon, n_controls)``.
        horizon:
            Number of steps ahead to predict.
        n_samples:
            Number of MC samples (ignored when ``method="moment_matching"``).
        seed:
            Random seed for MC sampling.
        method:
            ``"mc"`` (default) or ``"moment_matching"``.

        Returns
        -------
        dict[str, Tensor]
            Keys: ``"mean"`` shape ``(horizon, n_species)``,
            ``"q10"`` / ``"q50"`` / ``"q90"`` same shape.
        """
        if method == "mc":
            return self._rollout_mc(x0, controls, horizon, n_samples, seed)
        if method == "moment_matching":
            return self._rollout_mm(x0, controls, horizon)
        msg = f"Unknown method '{method}'. Use 'mc' or 'moment_matching'."
        raise ValueError(msg)

    def _build_input(self, c: Tensor, u: Tensor, day: float) -> Tensor:
        """Concatenate state, controls, and day into GP input.

        Parameters
        ----------
        c:
            Species state, shape ``(..., n_species)``.
        u:
            Controls, shape ``(..., n_controls)``.
        day:
            Culture day (scalar).

        Returns
        -------
        Tensor
            Shape ``(..., n_species + n_controls + 1)``.
        """
        day_t = torch.full((*c.shape[:-1], 1), day, dtype=c.dtype, device=c.device)
        return torch.cat([c, u, day_t], dim=-1)

    def _predict_one_step(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Return posterior mean and std-dev for one step.

        The GP uses the indexed multi-task paradigm: the last input column
        carries the task_id (species index).  We query each species separately
        by appending the corresponding task_id and then stack the results.

        Parameters
        ----------
        x:
            GP input *without* task_id, shape ``(N, d_input)`` where
            ``d_input = n_species + n_controls + 1 (day)``.

        Returns
        -------
        tuple[Tensor, Tensor]
            Mean and stddev, both shape ``(N, n_species)``.
        """
        self.model.eval()
        self.likelihood.eval()
        N = x.shape[0]
        means: list[Tensor] = []
        stds: list[Tensor] = []
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            for task_id in range(self.n_species):
                task_col = torch.full(
                    (N, 1), float(task_id), dtype=x.dtype, device=x.device
                )
                x_task = torch.cat([x, task_col], dim=-1)  # (N, d_input+1)
                pred = self.likelihood(self.model(x_task))
                means.append(pred.mean)   # (N,)
                stds.append(pred.stddev)  # (N,)
        return torch.stack(means, dim=-1), torch.stack(stds, dim=-1)  # (N, n_species)

    def _rollout_mc(
        self,
        x0: Tensor,
        controls: Tensor,
        horizon: int,
        n_samples: int,
        seed: int | None,
    ) -> dict[str, Tensor]:
        """Monte Carlo rollout for multi-step prediction."""
        g = torch.Generator(device=x0.device)
        if seed is not None:
            g.manual_seed(seed)

        # x0: (n_species,) → expand to (n_samples, n_species)
        c = x0.unsqueeze(0).expand(n_samples, -1).clone()

        trajectories: list[Tensor] = []

        for h in range(horizon):
            u_h = controls[h].unsqueeze(0).expand(n_samples, -1)
            x_h = self._build_input(c, u_h, float(h + 1))
            mu, sigma = self._predict_one_step(x_h)
            # Sample: c_{t+1} = mu + eps * sigma
            eps = torch.randn_like(sigma, generator=g)
            c = mu + eps * sigma
            c = c.clamp(min=0.0)  # physical non-negativity constraint
            trajectories.append(c.clone())

        # Stack: (horizon, n_samples, n_species)
        traj = torch.stack(trajectories, dim=0)
        return {
            "mean": traj.mean(dim=1),
            "q10": traj.quantile(0.10, dim=1),
            "q50": traj.quantile(0.50, dim=1),
            "q90": traj.quantile(0.90, dim=1),
        }

    def _rollout_mm(
        self,
        x0: Tensor,
        controls: Tensor,
        horizon: int,
    ) -> dict[str, Tensor]:
        """Moment-matching rollout (Gaussian approximation).

        Propagates mean and variance through the one-step GP.
        The covariance cross-terms between input uncertainty and kernel are
        approximated as zero (independent noise assumption per step).
        """
        c_mean = x0.clone()  # (n_species,)
        c_var = torch.zeros_like(c_mean)  # (n_species,)

        means_out: list[Tensor] = []
        sigma_out: list[Tensor] = []

        for h in range(horizon):
            u_h = controls[h].unsqueeze(0)
            x_h = self._build_input(c_mean.unsqueeze(0), u_h, float(h + 1))
            mu, sigma = self._predict_one_step(x_h)
            mu = mu.squeeze(0)
            sigma = sigma.squeeze(0)
            c_mean = mu
            c_var = sigma**2 + c_var  # accumulate variance (independent approx)
            c_mean = c_mean.clamp(min=0.0)
            means_out.append(c_mean.clone())
            sigma_out.append(c_var.sqrt().clone())

        means_t = torch.stack(means_out, dim=0)  # (horizon, n_species)
        sigmas_t = torch.stack(sigma_out, dim=0)

        z10 = torch.tensor(math.sqrt(2) * math.erfinv(2 * 0.10 - 1), dtype=means_t.dtype)  # ≈ -1.2816
        z90 = -z10  # ≈ +1.2816
        return {
            "mean": means_t,
            "q10": means_t + z10 * sigmas_t,
            "q50": means_t,
            "q90": means_t + z90 * sigmas_t,
        }
