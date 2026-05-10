"""High-level CHO perfusion mechanistic model.

This module wraps :class:`~perfusio.mechanistic.kinetics.CHOKinetics` and
:func:`~perfusio.mechanistic.integrators.integrate_run` into a clean
:class:`CHOPerfusionModel` API suitable for use as a mechanistic prior in
the hybrid SW-GP.

The model computes, for any process state :math:`s_i`, the mechanistic
prediction of daily rates :math:`\\hat{R}^{\\text{mech}}_k(s_i)`.  The
hybrid model's GP then learns the *residual*:

.. math::
    R_k(s_i) = \\hat{R}^{\\text{mech}}_k(s_i) + \\epsilon_k(s_i)

where :math:`\\epsilon_k \\sim \\mathcal{GP}(0, k(s_i, s_j))`.

References
----------
.. [Gadiyar2026] Gadiyar et al. (2026), §3.1 ("Hybrid model structure").
"""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor

from perfusio.mechanistic.integrators import integrate_run
from perfusio.mechanistic.kinetics import CHOKinetics


class CHOPerfusionModel:
    """Deterministic mechanistic CHO perfusion model.

    Parameters
    ----------
    kinetics:
        Kinetic parameter set.  If ``None``, uses the default clone X
        parameters (consumes lactate at low glucose).
    volume_L:
        Nominal reactor working volume [L].  Default 0.250 (ambr®250).
    dt_hours:
        Reporting interval [h].  Default 24.0 (daily samples).

    Examples
    --------
    >>> model = CHOPerfusionModel()
    >>> y0 = {"VCD": 1.0, "Via": 99.0, "Glc": 5.0, "Gln": 4.0,
    ...        "Glu": 0.5, "Lac": 2.0, "Amm": 0.5, "Pyr": 0.0,
    ...        "Titer": 0.0}
    >>> controls = {"perfusion_rate": 1.0, "bleed_rate": 0.15,
    ...              "temperature": 37.0}
    >>> traj = model.simulate(y0, controls, n_days=14, seed=42)
    >>> traj.shape  # (15, 9) — 14 daily steps + initial state
    (15, 9)
    """

    STATE_SPECIES: list[str] = CHOKinetics.STATE_ORDER

    def __init__(
        self,
        kinetics: CHOKinetics | None = None,
        volume_L: float = 0.250,
        dt_hours: float = 24.0,
    ) -> None:
        self.kinetics = kinetics if kinetics is not None else CHOKinetics()
        self.volume_L = volume_L
        self.dt_hours = dt_hours

    def simulate(
        self,
        y0: dict[str, float],
        controls: dict[str, float],
        n_days: int,
        seed: int | None = None,
    ) -> np.ndarray:
        """Run a deterministic simulation over *n_days* daily steps.

        Parameters
        ----------
        y0:
            Initial state, mapping species name → value.
        controls:
            Control variable values (constant over the run).
        n_days:
            Number of daily steps to simulate.
        seed:
            Ignored (deterministic ODE; kept for API consistency).

        Returns
        -------
        np.ndarray
            Shape ``(n_days + 1, n_species)`` where species follow
            :attr:`STATE_SPECIES` ordering.  Row 0 is the initial state.
        """
        y0_vec = [y0.get(k, 0.0) for k in self.STATE_SPECIES]
        controls_with_volume = {**controls, "volume_L": self.volume_L}
        trajectory = integrate_run(
            kinetics=self.kinetics,
            y0=y0_vec,
            controls=controls_with_volume,
            n_days=n_days,
            dt_hours=self.dt_hours,
        )
        return trajectory

    def predict_rates(
        self,
        c: Tensor,
        controls: dict[str, float],
    ) -> Tensor:
        """Predict net volumetric rates at state *c* using the kinetic model.

        Parameters
        ----------
        c:
            State tensor, shape ``(n_species,)``.  Species ordered as
            :attr:`STATE_SPECIES`.
        controls:
            Control variable values.

        Returns
        -------
        Tensor
            Rate tensor, shape ``(n_species,)``.
            Units are [species-unit h⁻¹].
        """
        y = c.detach().cpu().numpy().tolist()
        dy = self.kinetics.ode_rhs(
            t=0.0,
            y=y,
            controls={**controls, "volume_L": self.volume_L},
        )
        return torch.tensor(dy, dtype=torch.float64)

    def predict_rates_batch(
        self,
        c_batch: Tensor,
        controls_batch: Tensor,
        control_names: list[str],
    ) -> Tensor:
        """Vectorised rate prediction for a batch of states.

        Parameters
        ----------
        c_batch:
            Batch of states, shape ``(N, n_species)``.
        controls_batch:
            Batch of controls, shape ``(N, n_controls)``.
        control_names:
            Ordered list of control names matching columns of *controls_batch*.

        Returns
        -------
        Tensor
            Shape ``(N, n_species)``.
        """
        N = c_batch.shape[0]
        rates = torch.empty_like(c_batch)
        for i in range(N):
            c_i = c_batch[i]
            ctrl_dict = {k: float(controls_batch[i, j]) for j, k in enumerate(control_names)}
            rates[i] = self.predict_rates(c_i, ctrl_dict)
        return rates
