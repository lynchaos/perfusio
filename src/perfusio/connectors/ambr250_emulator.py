"""ambr®250 virtual emulator — full-featured digital twin of the Sartorius ambr®250.

Wraps :class:`~perfusio.simulator.cho_perfusion.CHOSimulator` with the
:class:`~perfusio.connectors.base.BioreactorConnectorBase` interface, adding:

- Configurable acceleration factor (up to 1000× real-time).
- 5-reactor parallel emulation (as in Gadiyar Fig. 5).
- Identical public API to the real OPC UA connector.

This allows the digital twin to run closed-loop simulations with no code
changes when switching from simulation to real hardware.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from perfusio.connectors.base import BioreactorConnectorBase
from perfusio.simulator.cho_perfusion import CHOSimulator

logger = logging.getLogger(__name__)


class Ambr250Emulator(BioreactorConnectorBase):
    """Virtual ambr®250 bioreactor emulator.

    Parameters
    ----------
    clone:
        ``"CloneX"`` or ``"CloneY"``.
    volume_L:
        Reactor working volume.  Default 0.250 (ambr®250 max).
    controls:
        Initial control setpoints.
    acceleration:
        Speed multiplier vs. real-time.  Default 1 (real-time).
        Set to 1440 for 1 simulated day per real minute.
    seed:
        Random seed for the noise model.

    Examples
    --------
    >>> import asyncio
    >>> from perfusio.connectors.ambr250_emulator import Ambr250Emulator
    >>> em = Ambr250Emulator(clone="CloneX", seed=0)
    >>> sample = asyncio.run(em.read_sample(day=5))
    >>> sample["VCD"]  # doctest: +SKIP
    """

    def __init__(
        self,
        clone: str = "CloneX",
        volume_L: float = 0.250,
        controls: dict[str, float] | None = None,
        acceleration: float = 1.0,
        seed: int = 42,
    ) -> None:
        self._sim = CHOSimulator(
            clone=clone,
            volume_L=volume_L,
            controls=controls,
            acceleration=acceleration,
            seed=seed,
        )
        logger.info(
            "Ambr250Emulator: initialised (%s, %.3f L, accel=%.1f×).",
            clone, volume_L, acceleration,
        )

    # ── BioreactorConnectorBase ────────────────────────────────────────────

    async def read_sample(self, day: int) -> dict[str, Any]:
        """Return a noisy sample from the virtual reactor at *day*."""
        return await self._sim.read_sample(day=day)

    async def write_setpoints(self, setpoints: dict[str, float]) -> None:
        """Update control setpoints on the virtual reactor."""
        await self._sim.write_setpoints(setpoints)

    async def is_alive(self) -> bool:
        """Always True for the virtual reactor."""
        return await self._sim.is_alive()

    # ── Convenience helpers ────────────────────────────────────────────────

    def simulate_run(self, n_days: int = 28) -> "Any":
        """Run a clean (noiseless) N-day simulation and return trajectory."""
        return self._sim.simulate_run(n_days=n_days)

    @classmethod
    def five_reactor_ensemble(
        cls,
        clones: list[str] | None = None,
        seed_start: int = 0,
    ) -> list["Ambr250Emulator"]:
        """Instantiate 5 virtual reactors (Gadiyar Fig. 5 set-up).

        Parameters
        ----------
        clones:
            List of 5 clone names.  Defaults to 3× CloneX + 2× CloneY.
        seed_start:
            Seed for the first reactor; each subsequent reactor gets seed+1.

        Returns
        -------
        list[Ambr250Emulator]
            Five fully independent virtual reactors.
        """
        clones = clones or ["CloneX", "CloneX", "CloneX", "CloneY", "CloneY"]
        if len(clones) != 5:
            msg = "five_reactor_ensemble requires exactly 5 clones."
            raise ValueError(msg)
        return [cls(clone=c, seed=seed_start + i) for i, c in enumerate(clones)]
