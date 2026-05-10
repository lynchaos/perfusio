"""Design-of-Experiments (DoE) plan generators.

Provides:
- :func:`box_behnken`: Box-Behnken design for 3–7 factors.
- :func:`central_composite`: Central Composite Design (CCD).
- :func:`latin_hypercube`: Maximin Latin Hypercube (using SciPy's QMC engine).

All functions return a :class:`numpy.ndarray` of shape ``(n_runs, n_factors)``
in the **normalised** space ``[-1, 1]`` unless ``scale_to_bounds`` is True.

References
----------
.. [BoxBehnken1960] Box, G. E. P., & Behnken, D. W. (1960). Some new three
   level designs for the study of quantitative variables. Technometrics, 2(4).
.. [Montgomery2017] Montgomery, D. C. (2017). Design and Analysis of
   Experiments. 9th ed. Wiley. §14.
"""

from __future__ import annotations

import itertools

import numpy as np


def box_behnken(n_factors: int, center_points: int = 3) -> np.ndarray:
    """Generate a Box-Behnken design matrix.

    Parameters
    ----------
    n_factors:
        Number of continuous factors (3 ≤ k ≤ 7).
    center_points:
        Number of centre-point replicates.

    Returns
    -------
    np.ndarray
        Design matrix, shape ``(n_runs, n_factors)`` in normalised [-1, 1].

    Raises
    ------
    ValueError
        If ``n_factors`` is not in [3, 7].
    """
    if not 3 <= n_factors <= 7:
        msg = f"Box-Behnken requires 3 ≤ n_factors ≤ 7, got {n_factors}."
        raise ValueError(msg)

    # Generate edge midpoints: all pairs of factors at ±1, others at 0
    rows = []
    for i, j in itertools.combinations(range(n_factors), 2):
        for xi, xj in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
            row = np.zeros(n_factors)
            row[i] = xi
            row[j] = xj
            rows.append(row)

    # Add centre points
    centre = np.zeros((center_points, n_factors))
    design = np.vstack([np.array(rows), centre])
    return design


def central_composite(
    n_factors: int,
    alpha: str = "rotatable",
    center_points: tuple[int, int] = (4, 4),
) -> np.ndarray:
    """Generate a Central Composite Design (CCD) matrix.

    Parameters
    ----------
    n_factors:
        Number of factors.
    alpha:
        Axial distance strategy: ``"rotatable"`` or ``"orthogonal"``.
    center_points:
        ``(n_center_factorial, n_center_star)`` — centre-point replicates.

    Returns
    -------
    np.ndarray
        Design matrix, shape ``(n_runs, n_factors)`` in normalised space.
        Factorial points at ±1; axial points at ±alpha_val; centre at 0.
    """
    # Factorial part: 2^n_factors full-factorial (or 2^(n-1) fraction for k>4)
    levels = np.array(list(itertools.product([-1, 1], repeat=n_factors)))

    # Axial (star) points
    if alpha == "rotatable":
        alpha_val = (2**n_factors) ** 0.25
    else:
        alpha_val = n_factors**0.5

    star = []
    for i in range(n_factors):
        for sign in [-1, 1]:
            row = np.zeros(n_factors)
            row[i] = sign * alpha_val
            star.append(row)
    star_arr = np.array(star)

    # Centre points
    n_cf, n_cs = center_points
    centre = np.zeros((n_cf + n_cs, n_factors))

    return np.vstack([levels, star_arr, centre])


def latin_hypercube(
    n_runs: int,
    n_factors: int,
    seed: int = 0,
    optimise: bool = True,
) -> np.ndarray:
    """Generate a maximin Latin Hypercube Sample in [0, 1]^n_factors.

    Parameters
    ----------
    n_runs:
        Number of runs.
    n_factors:
        Number of factors.
    seed:
        Random seed.
    optimise:
        If True, use ``scipy.stats.qmc.LatinHypercube`` with ``optimization
        = "random-cd"`` for improved space filling (SciPy ≥ 1.7).

    Returns
    -------
    np.ndarray
        LHC sample, shape ``(n_runs, n_factors)`` in [0, 1].

    Notes
    -----
    Map to physical bounds externally: ``x_phys = lo + (hi - lo) * x_lhc``.
    """
    from scipy.stats.qmc import LatinHypercube

    engine = LatinHypercube(
        d=n_factors,
        seed=seed,
        optimization="random-cd" if optimise else None,
    )
    return engine.random(n=n_runs)


def scale_to_bounds(
    design: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
    normalised: bool = True,
) -> np.ndarray:
    """Scale a normalised (or [0,1]) design to physical bounds.

    Parameters
    ----------
    design:
        Design matrix, shape ``(n_runs, n_factors)``.
    lo:
        Lower bounds, shape ``(n_factors,)``.
    hi:
        Upper bounds, shape ``(n_factors,)``.
    normalised:
        If ``True``, input is in [-1, 1]; else in [0, 1].

    Returns
    -------
    np.ndarray
        Scaled design in physical units, same shape.
    """
    if normalised:
        t = (design + 1.0) / 2.0  # map [-1,1] → [0,1]
    else:
        t = design
    return lo + (hi - lo) * t
