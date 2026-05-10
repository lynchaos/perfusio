"""Multi-objective performance metrics.

Provides:
- :func:`hypervolume_indicator`: hypervolume dominated by a Pareto front.
- :func:`igd_plus`: Inverted Generational Distance+ (IGD+).
- :func:`epsilon_indicator`: Additive ε-indicator.

These metrics are used to assess the quality of the Pareto front returned by
the multi-objective BED loop (qEHVI / qNEHVI / qNParEGO).

References
----------
.. [Ishibuchi2015] Ishibuchi, H., et al. (2015). Modified Distance Calculation
   in Generational Distance and Inverted Generational Distance. EMO.
.. [Zitzler2003] Zitzler, E., Thiele, L., & Bader, J. (2010). On Set-Based
   Multiobjective Optimization. IEEE TEC 14(1).
"""

from __future__ import annotations

from torch import Tensor


def hypervolume_indicator(Y: Tensor, ref_point: Tensor) -> float:
    """Hypervolume dominated by *Y* w.r.t. *ref_point*.

    Delegates to :func:`~perfusio.bed.pareto.hypervolume`.

    Parameters
    ----------
    Y:
        Objective matrix (higher is better), shape ``(N, n_objectives)``.
    ref_point:
        Reference point, shape ``(n_objectives,)``.

    Returns
    -------
    float
    """
    from perfusio.bed.pareto import hypervolume

    return hypervolume(Y, ref_point)


def igd_plus(
    approx: Tensor,
    reference: Tensor,
) -> float:
    """Inverted Generational Distance+ (IGD+).

    Measures the average *modified* distance from each reference-set point
    to the nearest point in the approximation set.

    .. math::
        \\text{IGD+}(A, R) =
            \\frac{1}{|R|}
            \\sum_{r \\in R}
            \\min_{a \\in A}
            \\left\\|
                \\max(r - a, 0)
            \\right\\|_2

    Parameters
    ----------
    approx:
        Approximation Pareto front, shape ``(M, n_obj)``.
    reference:
        True (or best-known) Pareto front, shape ``(K, n_obj)``.

    Returns
    -------
    float
        Mean IGD+ (lower is better, 0 = perfect approximation).
    """
    # For each reference point r, find min_a || max(r-a, 0) ||_2
    # Shape: (K, M, n_obj)
    diff = reference.unsqueeze(1) - approx.unsqueeze(0)  # (K, M, n_obj)
    diff_clipped = diff.clamp(min=0.0)  # only penalise shortfall
    dists = diff_clipped.norm(dim=2)  # (K, M)
    min_dists = dists.min(dim=1).values  # (K,)
    return float(min_dists.mean().item())


def epsilon_indicator(
    approx: Tensor,
    reference: Tensor,
) -> float:
    """Additive ε-indicator.

    The smallest ε such that the approximation set ε-dominates the reference
    set (i.e. every reference point has an approximation point within ε).

    .. math::
        \\varepsilon(A, R) = \\max_{r \\in R} \\min_{a \\in A} \\max_k (r_k - a_k)

    Parameters
    ----------
    approx:
        Approximation Pareto front, shape ``(M, n_obj)``.
    reference:
        True (or best-known) Pareto front, shape ``(K, n_obj)``.

    Returns
    -------
    float
        ε-indicator (lower is better).
    """
    # For each r and a: max over objectives of (r_k - a_k)
    diff = reference.unsqueeze(1) - approx.unsqueeze(0)  # (K, M, n_obj)
    max_diff = diff.max(dim=2).values  # (K, M)
    min_over_a = max_diff.min(dim=1).values  # (K,)
    return float(min_over_a.max().item())
