"""Kernel definitions for the hybrid SW-GP model.

The primary kernel is a product of:

1. **Matérn-5/2** over the metabolite state space (captures smooth, non-periodic
   autocorrelation in bioreactor trajectories).
2. **Linear kernel** over the operating day variable (models the secular trend
   within a run).
3. **Categorical / Index kernel** for multi-task output (one task per species /
   rate variable), sharing a common latent representation via an index kernel.

These choices follow Cruz-Bournazou et al. (2022) §2.3 and Rasmussen &
Williams (2006) §4.2.

References
----------
.. [RW2006] Rasmussen, C. E., & Williams, C. K. I. (2006). Gaussian Processes
   for Machine Learning. MIT Press. §4.2.
.. [CruzBournazou2022] Cruz-Bournazou, M. N., et al. (2022). Digitalizing
   bioprocess development. Current Opinion in Biotechnology, 76, 102764.
"""

from __future__ import annotations

from typing import Any

import gpytorch
from gpytorch.kernels import (
    IndexKernel,
    LinearKernel,
    MaternKernel,
    RBFKernel,
    ScaleKernel,
)
from torch import Tensor


class PerfusionKernel(gpytorch.kernels.Kernel):
    """Composite kernel for CHO perfusion rate GP models.

    The kernel factorises as:

    .. math::
        k(\\mathbf{x}, \\mathbf{x}') =
          k_{\\text{Matern5/2}}(\\mathbf{x}_{\\text{state}},
                                \\mathbf{x}'_{\\text{state}})
        \\times k_{\\text{Linear}}(x_{\\text{day}}, x'_{\\text{day}})
        \\times k_{\\text{Index}}(i, j)

    where :math:`\\mathbf{x}_{\\text{state}}` are metabolite concentrations,
    :math:`x_{\\text{day}}` is the culture day, and :math:`i,j` index the
    output task (species).

    Parameters
    ----------
    n_tasks:
        Number of output tasks (species / rate variables).
    n_state_dims:
        Dimensionality of the metabolite state input.
    rank:
        Rank of the index kernel covariance factor. Default 1.
    ard_num_dims:
        If > 1 uses automatic relevance determination for the Matérn kernel.

    Notes
    -----
    Input convention:
    ``x[:, :n_state_dims]``   — metabolite concentrations (normalised)
    ``x[:, n_state_dims]``    — culture day
    ``x[:, n_state_dims + 1]`` — task index (integer)
    """

    is_stationary: bool = False  # pyright: ignore[reportIncompatibleMethodOverride]  # linear kernel breaks stationarity

    def __init__(
        self,
        n_tasks: int,
        n_state_dims: int,
        rank: int = 1,
        ard_num_dims: int = 1,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.n_state_dims = n_state_dims

        self.state_kernel = ScaleKernel(
            MaternKernel(nu=2.5, ard_num_dims=ard_num_dims, active_dims=tuple(range(n_state_dims)))
        )
        self.day_kernel = ScaleKernel(LinearKernel(active_dims=(n_state_dims,)))
        self.task_kernel = IndexKernel(
            num_tasks=n_tasks,
            rank=rank,
            active_dims=(n_state_dims + 1,),
        )

    def forward(self, x1: Tensor, x2: Tensor, **params: Any) -> Tensor:  # type: ignore[override]
        k_state = self.state_kernel(x1, x2, **params).to_dense()
        k_day = self.day_kernel(x1, x2, **params).to_dense()
        k_task = self.task_kernel(x1, x2, **params).to_dense()
        return k_state * k_day * k_task


class ResidualKernel(gpytorch.kernels.Kernel):
    """RBF-based residual kernel for the SW-GP correction term.

    Used in :class:`~perfusio.hybrid.model.HybridStateSpaceModel` to capture
    smooth, local residuals that the mechanistic model cannot explain.

    Parameters
    ----------
    n_state_dims:
        Dimensionality of the input (state + controls concatenated).
    ard_num_dims:
        If > 1, uses ARD (one length-scale per input dimension).
    """

    is_stationary: bool = True  # pyright: ignore[reportIncompatibleMethodOverride]

    def __init__(
        self,
        n_state_dims: int,
        ard_num_dims: int = 1,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.base = ScaleKernel(
            RBFKernel(ard_num_dims=ard_num_dims, active_dims=tuple(range(n_state_dims)))
        )

    def forward(self, x1: Tensor, x2: Tensor, **params: Any) -> Tensor:  # type: ignore[override]
        return self.base(x1, x2, **params).to_dense()
