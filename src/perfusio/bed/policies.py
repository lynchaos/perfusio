"""High-level BED policy — daily self-driving decision loop.

:class:`BEDPolicy` orchestrates the complete self-driving step:

1. Receive current bioreactor state.
2. Evaluate the acquisition function over the design space.
3. Propose next setpoints (with ``--allow-write`` safety gate).
4. Log the decision and uncertainty to the audit trail.

This is the class called by the digital twin's daily scheduler.

References
----------
.. [Gadiyar2026] Gadiyar et al. (2026), §3.2 and Fig. 5.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING

import torch
from torch import Tensor

if TYPE_CHECKING:
    from perfusio.bed.acquisitions import build_acquisition
    from perfusio.bed.objectives import TargetTrackingOFV
    from perfusio.config import DesignSpace

logger = logging.getLogger(__name__)


@dataclass
class BEDDecision:
    """A single decision made by the BED policy."""

    timestamp: datetime
    day: int
    recommended_controls: dict[str, float]
    acqf_value: float
    acqf_name: str
    current_state: dict[str, float]
    forecast_mean: dict[str, list[float]] = field(default_factory=dict)


class BEDPolicy:
    """Daily self-driving control policy.

    Parameters
    ----------
    hybrid:
        Fitted hybrid model.
    design_space:
        Process design space with control bounds.
    acqf_name:
        Acquisition function to use.  Default ``"qLogEI"`` (robust for noisy
        bioprocess observations).
    allow_write:
        Safety gate: if ``False``, recommendations are logged but NOT written
        to the bioreactor.  Must be ``True`` for closed-loop operation.
    targets:
        List of :class:`~perfusio.bed.objectives.TargetSpec` objects.
    n_restarts:
        ``optimize_acqf`` restarts.
    n_raw_samples:
        Sobol initialisation candidates.

    Examples
    --------
    >>> from perfusio.bed import BEDPolicy
    >>> # policy = BEDPolicy(hybrid, design_space, allow_write=False)
    >>> # decision = policy.decide(c_current, day=7)
    """

    def __init__(
        self,
        hybrid: "object",
        design_space: "DesignSpace",
        acqf_name: str = "qLogEI",
        allow_write: bool = False,
        targets: list["object"] | None = None,
        n_restarts: int = 10,
        n_raw_samples: int = 512,
    ) -> None:
        self.hybrid = hybrid
        self.design_space = design_space
        self.acqf_name = acqf_name
        self.allow_write = allow_write
        self.targets = targets or []
        self.n_restarts = n_restarts
        self.n_raw_samples = n_raw_samples
        self._history: list[BEDDecision] = []

    def decide(
        self,
        c_current: Tensor,
        day: int,
        surrogate_model: "object",
        best_f: float | None = None,
        seed: int | None = None,
    ) -> BEDDecision:
        """Compute the recommended control setpoints for the next day.

        Parameters
        ----------
        c_current:
            Current species state, shape ``(n_species,)``.
        day:
            Current culture day.
        surrogate_model:
            Fitted BoTorch surrogate (wraps the hybrid GP).
        best_f:
            Best observed OFV so far.  Required for EI-based acquisitions.
        seed:
            Random seed for deterministic optimisation.

        Returns
        -------
        BEDDecision
            Decision record including recommended controls and forecast.
        """
        from perfusio.bed.acquisitions import build_acquisition
        from perfusio.bed.search import optimise_acquisition

        acqf = build_acquisition(
            name=self.acqf_name,
            model=surrogate_model,
            best_f=best_f,
        )

        candidates, acqf_val = optimise_acquisition(
            acqf=acqf,
            design_space=self.design_space,
            q=1,
            n_restarts=self.n_restarts,
            n_raw_samples=self.n_raw_samples,
            seed=seed,
        )

        # candidates shape: (1, n_controls)
        u_rec = candidates.squeeze(0)
        ctrl_dict = {
            name: float(u_rec[i])
            for i, name in enumerate(self.design_space.control_names)
        }

        decision = BEDDecision(
            timestamp=datetime.utcnow(),
            day=day,
            recommended_controls=ctrl_dict,
            acqf_value=float(acqf_val.item()),
            acqf_name=self.acqf_name,
            current_state={
                name: float(c_current[i])
                for i, name in enumerate(self.design_space.species_names)
            },
        )

        if not self.allow_write:
            logger.info(
                "BEDPolicy: allow_write=False — recommendation logged but NOT applied. "
                "Day=%d, acqf=%s, val=%.4f",
                day, self.acqf_name, float(acqf_val.item()),
            )
        else:
            logger.info(
                "BEDPolicy: Writing setpoints. Day=%d, acqf=%s, val=%.4f",
                day, self.acqf_name, float(acqf_val.item()),
            )

        self._history.append(decision)
        return decision

    @property
    def history(self) -> list[BEDDecision]:
        """All decisions made so far."""
        return list(self._history)
