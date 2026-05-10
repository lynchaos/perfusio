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

import gpytorch
import torch
from gpytorch.distributions import MultitaskMultivariateNormal, MultivariateNormal
from gpytorch.likelihoods import MultitaskGaussianLikelihood
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
        Training inputs, shape ``(N, d_input)``.
    train_y:
        Training targets, shape ``(N, n_tasks)``.
    likelihood:
        Multitask Gaussian likelihood.
    mean_module:
        Mean function module.  Should be a GPyTorch ``Mean`` or
        :class:`~perfusio.gp.means.MechanisticPriorMean`.
    kernel:
        Composite kernel.  If ``None``, constructs a default
        :class:`~perfusio.gp.kernels.PerfusionKernel`.
    n_tasks:
        Number of output tasks.
    n_state_dims:
        Input dimensionality (species + controls); needed to build default kernel.

    Examples
    --------
    >>> import torch, gpytorch
    >>> from perfusio.gp import MultiTaskRateGP
    >>> N, d, T = 50, 10, 5
    >>> X = torch.randn(N, d + 2)  # d state + 1 day + 1 task index
    >>> Y = torch.randn(N, T)
    >>> lik = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=T)
    >>> mean = gpytorch.means.ZeroMean()
    >>> gp = MultiTaskRateGP(X, Y, lik, mean, n_tasks=T, n_state_dims=d)
    """

    def __init__(
        self,
        train_x: Tensor,
        train_y: Tensor,
        likelihood: MultitaskGaussianLikelihood,
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
        mean_x = self.mean_module(x)
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
        self.likelihood.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            pred = self.likelihood(self(x_new))
        out: dict[str, Tensor] = {"mean": pred.mean}
        for p in ci_levels:
            out[f"q{int(p * 100)}"] = pred.mean + torch.erfinv(
                torch.tensor(2 * p - 1, dtype=pred.mean.dtype)
            ) * pred.stddev * (2.0 ** 0.5)
        return out
