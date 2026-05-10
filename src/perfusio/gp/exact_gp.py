"""Exact GP model for multi-task rate prediction.

Implements a batched ``ExactGP`` from GPyTorch that simultaneously models
``n_tasks`` output variables (metabolite rates) using a shared kernel and
task-specific hyperparameters.

The model can be used with either a zero mean, linear mean, or a mechanistic
prior mean (:class:`~perfusio.gp.means.MechanisticPriorMean`).

References
----------
.. [Gardner2018] Gardner, J. R., et al. (2018). GPyTorch: Blackbox
   Matrix-Matrix Gaussian Process Inference with GPU Acceleration. NeurIPS.
"""

from __future__ import annotations

from typing import cast

import gpytorch
import torch
from gpytorch.distributions import MultivariateNormal
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.means import Mean
from gpytorch.models import ExactGP
from torch import Tensor

from perfusio.gp.kernels import PerfusionKernel


class MultiTaskRateGP(ExactGP):
    """Multi-output exact GP for metabolite rate prediction.

    Each of the ``n_tasks`` outputs shares the same :class:`PerfusionKernel`
    structure but has its own output scale and noise level.

    Parameters
    ----------
    train_x:
        Training inputs, shape ``(N * n_tasks, d_input)`` in the indexed
        multi-task format — last column is the task_id (0..n_tasks-1).
    train_y:
        Training targets, shape ``(N * n_tasks,)`` — one scalar per row.
    likelihood:
        Gaussian likelihood (single-output; task correlation is captured
        by the :class:`~perfusio.gp.kernels.PerfusionKernel` IndexKernel).
    mean_module:
        Mean function module.  Should be a GPyTorch ``Mean`` or
        :class:`~perfusio.gp.means.MechanisticPriorMean`.
    kernel:
        Composite kernel.  If ``None``, constructs a default
        :class:`~perfusio.gp.kernels.PerfusionKernel`.
    n_tasks:
        Number of output tasks (species).
    n_state_dims:
        Input dimensionality (species + controls); needed to build default kernel.

    Examples
    --------
    >>> import torch, gpytorch
    >>> from perfusio.gp import MultiTaskRateGP
    >>> N, d, T = 50, 10, 5
    >>> # Indexed format: N*T rows, last col = task_id
    >>> X = torch.randn(N * T, d + 2)  # d state + 1 day + 1 task_id
    >>> Y = torch.randn(N * T)          # flat scalar targets
    >>> likelihood = gpytorch.likelihoods.GaussianLikelihood()
    >>> mean = gpytorch.means.ZeroMean()
    >>> gp = MultiTaskRateGP(X, Y, likelihood, mean, n_tasks=T, n_state_dims=d)
    """

    def __init__(
        self,
        train_x: Tensor,
        train_y: Tensor,
        likelihood: GaussianLikelihood,
        mean_module: Mean,
        kernel: PerfusionKernel | None = None,
        n_tasks: int = 9,
        n_state_dims: int = 9,
    ) -> None:
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = mean_module
        if kernel is None:
            self.covar_module: PerfusionKernel = PerfusionKernel(
                n_tasks=n_tasks,
                n_state_dims=n_state_dims,
            )
        else:
            self.covar_module = kernel

    def forward(self, x: Tensor) -> MultivariateNormal:
        """Compute the prior distribution at *x*.

        Parameters
        ----------
        x:
            Input tensor, shape ``(N, d_input)``.

        Returns
        -------
        MultivariateNormal
            The GP prior (mean + covariance).
        """
        mean_x = cast(Tensor, self.mean_module(x))
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)

    def predict_with_ci(
        self,
        x_new: Tensor,
        ci_levels: tuple[float, float, float] = (0.10, 0.50, 0.90),
    ) -> dict[str, Tensor]:
        """Return posterior mean and credible-interval quantiles.

        Parameters
        ----------
        x_new:
            Test inputs, shape ``(M, d_input)``.
        ci_levels:
            Probability levels for quantiles (e.g. 10/50/90th percentiles).

        Returns
        -------
        dict[str, Tensor]
            Keys ``"mean"`` and ``"q{p*100:.0f}"`` for each level.
            All tensors have shape ``(M,)`` (or ``(M, n_tasks)`` for multitask).
        """
        self.eval()
        assert self.likelihood is not None
        self.likelihood.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            pred = self.likelihood(self(x_new))
        out: dict[str, Tensor] = {"mean": pred.mean}
        for p in ci_levels:
            z = torch.tensor(2 * p - 1, dtype=pred.mean.dtype, device=pred.mean.device)
            out[f"q{int(p * 100)}"] = pred.mean + torch.erfinv(z) * pred.stddev * (2.0**0.5)
        return out
