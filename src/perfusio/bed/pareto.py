"""Pareto front computation and hypervolume indicator.

Provides:
- :func:`compute_pareto_front`: identify non-dominated points.
- :func:`hypervolume`: Hypervolume dominated by the Pareto front w.r.t. a
  reference point.

These are used by the multi-objective BED loop to assess diversity and
progress of the Pareto set.

References
----------
.. [Daulton2020] Daulton, S., et al. (2020). Differentiable Expected
   Hypervolume Improvement for Parallel Multi-Objective Bayesian Optimization.
   NeurIPS.
"""

from __future__ import annotations

import torch
from torch import Tensor


def compute_pareto_front(Y: Tensor) -> Tensor:
    """Return the indices of non-dominated points in *Y*.

    Assumes *maximisation* for all objectives (negate objectives that should be
    minimised before calling).

    Parameters
    ----------
    Y:
        Objective matrix, shape ``(N, n_objectives)``.  Higher is better for
        all objectives.

    Returns
    -------
    Tensor
        Boolean mask of shape ``(N,)`` — ``True`` for non-dominated points.

    Examples
    --------
    >>> import torch
    >>> from perfusio.bed.pareto import compute_pareto_front
    >>> Y = torch.tensor([[1.0, 2.0], [2.0, 1.0], [1.5, 1.5], [0.5, 0.5]])
    >>> compute_pareto_front(Y)
    tensor([ True,  True,  True, False])
    """
    N = Y.shape[0]
    is_pareto = torch.ones(N, dtype=torch.bool)

    for i in range(N):
        if not is_pareto[i]:
            continue
        # Check if point i is dominated by any other non-dominated point.
        # j dominates i iff j >= i on ALL objectives AND j > i on AT LEAST ONE.
        dominated = (Y[i] <= Y).all(dim=1) & (Y[i] < Y).any(dim=1)
        dominated[i] = False  # a point cannot dominate itself
        if dominated.any():
            is_pareto[i] = False

    return is_pareto


def hypervolume(Y: Tensor, ref_point: Tensor) -> float:
    """Compute the hypervolume indicator for a set of points.

    Uses the BoTorch ``Hypervolume`` class for efficiency and correctness.

    Parameters
    ----------
    Y:
        Objective matrix of non-dominated points, shape ``(N, n_objectives)``.
        Higher is better (negate minimisation objectives before calling).
    ref_point:
        Reference point, shape ``(n_objectives,)``.  Must be dominated by all
        points in *Y*.

    Returns
    -------
    float
        Hypervolume dominated by *Y* w.r.t. *ref_point*.
    """
    try:
        from botorch.utils.multi_objective.hypervolume import Hypervolume as BotorchHV

        hv_calc = BotorchHV(ref_point=ref_point)
        return float(hv_calc.compute(Y))
    except ImportError as err:
        # Fallback: 2-objective Monte Carlo estimate (covers the common case)
        if Y.shape[1] != 2:
            msg = "Fallback hypervolume only supports 2 objectives; install botorch."
            raise ImportError(msg) from err
        return _hv_2d(Y, ref_point)


def _hv_2d(Y: Tensor, ref_point: Tensor) -> float:
    """Exact 2-objective hypervolume via sweep-line algorithm."""
    # Filter dominated by ref
    mask = (ref_point < Y).all(dim=1)
    Y = Y[mask]
    if Y.shape[0] == 0:
        return 0.0

    # Sort by first objective descending
    order = Y[:, 0].argsort(descending=True)
    Y_sorted = Y[order]

    hv = 0.0
    prev_y2 = float(ref_point[1])
    for i in range(len(Y_sorted)):
        x_width = float(Y_sorted[i, 0] - ref_point[0])
        y_height = float(Y_sorted[i, 1] - prev_y2)
        if y_height > 0:
            hv += x_width * y_height
            prev_y2 = float(Y_sorted[i, 1])

    return hv
