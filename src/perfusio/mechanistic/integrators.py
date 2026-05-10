"""ODE integration utilities for the mechanistic CHO model.

Uses ``scipy.integrate.solve_ivp`` with automatic stiff fallback:
- Primary solver: ``RK45`` (non-stiff, fast).
- Fallback: ``Radau`` (implicit, handles stiff metabolite ODEs).
- Second fallback: ``LSODA`` (automatic stiffness detection).

All solutions are reported at daily (or sub-daily if requested) time points,
regardless of the internal adaptive step chosen by the solver.

References
----------
.. [SciPy] Virtanen et al. (2020). SciPy 1.0. Nature Methods, 17, 261–272.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np
from scipy.integrate import solve_ivp

if TYPE_CHECKING:
    from perfusio.mechanistic.kinetics import CHOKinetics

#: Tolerances for the ODE solver.
_RTOL = 1e-6
_ATOL = 1e-9


def integrate_run(
    kinetics: CHOKinetics,
    y0: list[float],
    controls: dict[str, float],
    n_days: int,
    dt_hours: float = 24.0,
) -> np.ndarray:
    """Integrate the CHO ODE system over *n_days* daily steps.

    Parameters
    ----------
    kinetics:
        Kinetic parameter set.
    y0:
        Initial state vector, length ``n_species``.  Order must match
        :attr:`~perfusio.mechanistic.kinetics.CHOKinetics.STATE_ORDER`.
    controls:
        Control variable dict (constant throughout the run).
    n_days:
        Number of daily steps to simulate.
    dt_hours:
        Reporting interval in hours.  Default 24.0.

    Returns
    -------
    np.ndarray
        Shape ``(n_days + 1, n_species)`` — states at each reporting point
        including the initial state.

    Raises
    ------
    RuntimeError
        If all ODE solvers fail to converge.

    Notes
    -----
    The integration time span is ``[0, n_days * dt_hours]`` hours.
    Reporting points are at every ``dt_hours`` interval.
    Negative concentrations are clipped to zero after each step.
    """
    t_end = n_days * dt_hours
    t_eval = np.arange(0, t_end + dt_hours * 0.01, dt_hours)

    def rhs(t: float, y: np.ndarray) -> list[float]:
        # Clip negatives before evaluating kinetics to avoid sqrt/log issues
        y_clipped = np.maximum(y, 0.0)
        return kinetics.ode_rhs(t, y_clipped.tolist(), controls)

    result = _try_integrate(rhs, y0, t_eval, t_end)
    # Clip negative values in the solution (cannot be negative physically)
    sol = np.maximum(result, 0.0)
    return sol


def _try_integrate(
    rhs: object,
    y0: list[float],
    t_eval: np.ndarray,
    t_end: float,
) -> np.ndarray:
    """Try ODE solvers in order of preference; fall back on failure.

    Parameters
    ----------
    rhs:
        Right-hand side function compatible with ``scipy.integrate.solve_ivp``.
    y0:
        Initial conditions.
    t_eval:
        Evaluation time points.
    t_end:
        End time.

    Returns
    -------
    np.ndarray
        Solution array, shape ``(len(t_eval), n_species)``.

    Raises
    ------
    RuntimeError
        If all solvers fail.
    """
    solvers = ["RK45", "Radau", "LSODA"]
    y0_arr = np.array(y0, dtype=np.float64)

    for method in solvers:
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            sol = solve_ivp(
                rhs,
                (0.0, t_end),
                y0_arr,
                method=method,
                t_eval=t_eval,
                rtol=_RTOL,
                atol=_ATOL,
                max_step=1.0,  # max 1-hour internal step for accuracy
                dense_output=False,
            )
        if sol.success:
            # sol.y has shape (n_species, len(t_eval)); transpose to (T, n_species)
            return sol.y.T

    msg = (
        "All ODE solvers (RK45, Radau, LSODA) failed to converge. "
        "Check kinetic parameters and initial conditions."
    )
    raise RuntimeError(msg)
