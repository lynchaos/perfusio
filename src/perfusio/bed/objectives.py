"""Objective function values (OFVs) for Bayesian Experimental Design.

Implements Gadiyar et al. (2026) Eq. 7:

.. math::
    \\text{OFV}(\\mathbf{u}) =
      \\sum_{j=1}^{3}
      \\left( \\hat{y}_{t+j}(\\mathbf{u}) - y_{\\text{target}} \\right)^2

where :math:`\\hat{y}_{t+j}` is the *j*-step-ahead hybrid model prediction
(50th-percentile / mean) for the chosen objective species (e.g. VCD, Titer,
viability).

The OFV is used to evaluate candidate control setpoints :math:`\\mathbf{u}`
proposed by the acquisition function.

References
----------
.. [Gadiyar2026] Gadiyar et al. (2026), §3.2, Eq. (7).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch
from torch import Tensor

if TYPE_CHECKING:
    from perfusio.hybrid.model import HybridStateSpaceModel


@dataclass
class TargetSpec:
    """Specification for a single tracked species target."""

    species_index: int    # column index in the state tensor
    target: float         # target value
    weight: float = 1.0   # relative weight in the OFV sum


class TargetTrackingOFV:
    """3-step-ahead squared-error OFV from Gadiyar et al. (2026) Eq. 7.

    Parameters
    ----------
    hybrid:
        Fitted hybrid model for forecasting.
    targets:
        List of :class:`TargetSpec` objects — one per tracked species.
    horizon:
        Prediction horizon in days.  Default 3 (Gadiyar §3.2).
    n_samples:
        MC rollout samples.  Default 100 (fast for BED inner loop).
    seed:
        Fixed random seed for reproducible OFV evaluations.

    Examples
    --------
    >>> from perfusio.bed.objectives import TargetTrackingOFV, TargetSpec
    >>> # assume `hybrid` is a fitted HybridStateSpaceModel
    >>> ofv = TargetTrackingOFV(
    ...     hybrid=hybrid,
    ...     targets=[TargetSpec(0, target=30.0, weight=1.0)],  # VCD → 30e6 cells/mL
    ... )
    >>> # ofv.evaluate(c0, u_candidate)  # returns scalar Tensor
    """

    def __init__(
        self,
        targets: list[TargetSpec],
        hybrid: "HybridStateSpaceModel | None" = None,
        horizon: int = 3,
        n_samples: int = 100,
        seed: int = 42,
    ) -> None:
        self.hybrid = hybrid
        self.targets = targets
        self.horizon = horizon
        self.n_samples = n_samples
        self.seed = seed

    def evaluate(self, c0: Tensor, u: Tensor) -> Tensor:
        """Evaluate the OFV for a candidate control vector.

        Parameters
        ----------
        c0:
            Current state, shape ``(n_species,)``.
        u:
            Candidate control vector, shape ``(n_controls,)``.
            Applied *constantly* over the horizon.

        Returns
        -------
        Tensor
            Scalar OFV value (lower = better).
        """
        from perfusio.hybrid.forecast import forecast_run

        u_seq = u.unsqueeze(0).expand(self.horizon, -1)
        preds = forecast_run(
            self.hybrid, c0, u_seq,
            horizon=self.horizon,
            n_samples=self.n_samples,
            seed=self.seed,
        )
        # Use posterior mean for OFV (Gadiyar Eq. 7)
        mean_traj = preds["mean"]  # (horizon, n_species)

        ofv = torch.tensor(0.0, dtype=mean_traj.dtype)
        for spec in self.targets:
            k = spec.species_index
            for j in range(self.horizon):
                ofv = ofv + spec.weight * (mean_traj[j, k] - spec.target) ** 2
        return ofv

    def score_trajectories(self, Y: Tensor) -> Tensor:
        """Score a batch of pre-computed trajectories against targets.

        Useful for evaluating acquisition function surrogates directly on GP
        prediction samples without running the full hybrid model.

        Parameters
        ----------
        Y:
            Pre-computed trajectories, shape ``(B, horizon, n_species)``.

        Returns
        -------
        Tensor
            OFV values, shape ``(B,)`` — higher is better (negative SSE).
        """
        total = torch.zeros(Y.shape[0], dtype=Y.dtype, device=Y.device)
        for spec in self.targets:
            vals = Y[:, :, spec.species_index]           # (B, horizon)
            sse = ((vals - spec.target) ** 2).mean(dim=-1)  # (B,)
            total = total - spec.weight * sse
        return total

    def evaluate_batch(self, c0: Tensor, u_batch: Tensor) -> Tensor:
        """Evaluate OFV for a batch of candidate controls.

        Parameters
        ----------
        c0:
            Current state, shape ``(n_species,)``.
        u_batch:
            Candidate controls, shape ``(B, n_controls)``.

        Returns
        -------
        Tensor
            OFV values, shape ``(B,)``.
        """
        return torch.stack([self.evaluate(c0, u_batch[i]) for i in range(u_batch.shape[0])])


class MultiObjectiveOFV:
    """Vector-valued OFV for multi-objective BED.

    Returns one OFV per objective (e.g. maximise titer AND minimise ammonium).
    Used with multi-objective acquisition functions (qEHVI, qNEHVI, qNParEGO).

    Parameters
    ----------
    hybrid:
        Fitted hybrid model.
    objectives:
        List of ``(species_index, target, direction)`` tuples where
        ``direction`` is ``+1`` for maximisation and ``-1`` for minimisation
        (the sign is applied to the OFV before passing to botorch acqf).
    horizon:
        Prediction horizon.
    n_samples:
        MC rollout samples.
    seed:
        Random seed.
    """

    def __init__(
        self,
        hybrid: "HybridStateSpaceModel",
        objectives: list[tuple[int, float, float]],
        horizon: int = 3,
        n_samples: int = 100,
        seed: int = 42,
    ) -> None:
        self.hybrid = hybrid
        self.objectives = objectives
        self.horizon = horizon
        self.n_samples = n_samples
        self.seed = seed

    def evaluate(self, c0: Tensor, u: Tensor) -> Tensor:
        """Return the multi-objective OFV vector for one candidate control.

        Parameters
        ----------
        c0:
            Current state, shape ``(n_species,)``.
        u:
            Candidate control, shape ``(n_controls,)``.

        Returns
        -------
        Tensor
            Shape ``(n_objectives,)``.  Positive values = better for botorch
            (all objectives are negated and then passed as maximisation).
        """
        from perfusio.hybrid.forecast import forecast_run

        u_seq = u.unsqueeze(0).expand(self.horizon, -1)
        preds = forecast_run(
            self.hybrid, c0, u_seq,
            horizon=self.horizon,
            n_samples=self.n_samples,
            seed=self.seed,
        )
        mean_traj = preds["mean"]  # (horizon, n_species)

        vals = []
        for k, target, direction in self.objectives:
            # Sum of squared deviations (lower = better)
            sse = sum((mean_traj[j, k] - target) ** 2 for j in range(self.horizon))
            # Negate so that botorch maximises (smaller SSE → larger objective)
            vals.append(-float(direction) * sse)  # type: ignore[arg-type]

        return torch.tensor(vals, dtype=mean_traj.dtype)
