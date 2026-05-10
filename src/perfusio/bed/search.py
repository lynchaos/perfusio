"""Acquisition function optimisation over the process design space.

Wraps BoTorch's ``optimize_acqf`` with:
- Explicit lower/upper bounds from :class:`~perfusio.config.DesignSpace`.
- Restarts from Sobol quasi-random initial candidates.
- Batch support for ``q > 1`` (parallelised reactor experiments).

References
----------
.. [Balandat2020] Balandat et al. (2020). BoTorch: A Framework for Efficient
   Monte-Carlo Bayesian Optimization. NeurIPS.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from botorch.optim import optimize_acqf
from torch import Tensor

if TYPE_CHECKING:
    import botorch.acquisition as ba
    from perfusio.config import DesignSpace


def optimise_acquisition(
    acqf: "ba.AcquisitionFunction",
    design_space: "DesignSpace",
    q: int = 1,
    n_restarts: int = 10,
    n_raw_samples: int = 512,
    fixed_features: dict[int, float] | None = None,
    seed: int | None = None,
    dtype: torch.dtype = torch.float64,
    device: torch.device | None = None,
) -> tuple[Tensor, Tensor]:
    """Optimise *acqf* over the control design space.

    Parameters
    ----------
    acqf:
        BoTorch acquisition function.
    design_space:
        :class:`~perfusio.config.DesignSpace` with control bounds.
    q:
        Batch size (number of concurrent recommendations).  Use ``q > 1``
        only with MC acquisitions (``qEI``, ``qUCB``, etc.).
    n_restarts:
        Number of restarts for ``optimize_acqf``.
    n_raw_samples:
        Number of Sobol candidates for initialisation.
    fixed_features:
        Dict mapping feature indices to fixed values (e.g. fix temperature).
    seed:
        Random seed.
    dtype:
        Tensor dtype.  Default ``torch.float64``.
    device:
        Compute device.  Default CPU.

    Returns
    -------
    tuple[Tensor, Tensor]
        ``(candidates, acqf_values)`` — the optimal candidate(s) (shape
        ``(q, n_controls)``) and their acquisition values (shape ``(q,)``).
    """
    if device is None:
        device = torch.device("cpu")

    bounds_lo, bounds_hi = design_space.bounds_tensor
    bounds = torch.stack([bounds_lo, bounds_hi]).to(dtype=dtype, device=device)
    n_controls = design_space.n_controls

    gen_kwargs: dict[str, object] = {}
    if seed is not None:
        gen_kwargs["seed"] = seed

    if fixed_features:
        from botorch.optim import optimize_acqf_mixed
        candidates, values = optimize_acqf_mixed(
            acq_function=acqf,
            bounds=bounds,
            fixed_features_list=[fixed_features],
            q=q,
            num_restarts=n_restarts,
            raw_samples=n_raw_samples,
            options={"maxiter": 200},
        )
    else:
        candidates, values = optimize_acqf(
            acq_function=acqf,
            bounds=bounds,
            q=q,
            num_restarts=n_restarts,
            raw_samples=n_raw_samples,
            options={"maxiter": 200, "batch_limit": min(5, n_restarts)},
        )

    return candidates, values
