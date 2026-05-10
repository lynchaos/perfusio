"""CHO perfusion simulator — virtual ambr®250 bioreactor.

Provides a fully self-contained, stochastic in-silico CHO bioreactor that
replicates the operating conditions described in Gadiyar et al. (2026) §3.3:

- Clone X: Warburg metabolic switch (lactate consumption at Glc < 2 g/L)
- Clone Y: no switch (always produces lactate)
- 24-run Box-Behnken training experiment generator
- 5-reactor closed-loop demonstration (Gadiyar Fig. 5)

The simulator implements the same :class:`~perfusio.connectors.base.BioreactorConnector`
interface as the real OPC UA connector, so the digital twin and self-driving
loop can run identically against both the virtual and real hardware.

References
----------
.. [Gadiyar2026] Gadiyar et al. (2026), §3.3 (in-silico validation).
"""

from __future__ import annotations

import asyncio
from typing import Any

import numpy as np

from perfusio.mechanistic.integrators import integrate_run
from perfusio.mechanistic.kinetics import CHOKinetics
from perfusio.simulator.noise import NoiseModel

# Initial state for a CHO perfusion run at inoculation (day 0)
_DEFAULT_Y0: dict[str, float] = {
    "VCD": 1.0,  # 10⁶ cells/mL
    "Via": 99.0,  # %
    "Glc": 5.0,  # g/L
    "Gln": 4.0,  # mmol/L
    "Glu": 0.5,  # mmol/L
    "Lac": 2.0,  # mmol/L
    "Amm": 0.5,  # mmol/L
    "Pyr": 0.0,  # mmol/L
    "Titer": 0.0,  # mg/L
}

# Default control setpoints for a standard perfusion run
_DEFAULT_CONTROLS: dict[str, float] = {
    "perfusion_rate": 1.0,  # vvd
    "bleed_rate": 0.15,  # vvd
    "glucose_setpoint": 5.0,  # g/L (target for glucose feed control)
    "temperature": 37.0,  # °C
    "agitation": 250.0,  # rpm
    "pyruvate_feed": 0.0,  # mmol/L
}


class CHOSimulator:
    """Virtual ambr®250 bioreactor implementing the BioreactorConnector interface.

    Runs the CHO kinetic ODE in accelerated simulation time (up to 1000×
    real-time), applies measurement noise, and exposes the same async API
    as the OPC UA connector.

    Parameters
    ----------
    clone:
        ``"CloneX"`` (Warburg switch) or ``"CloneY"`` (no switch).
    y0:
        Initial state dict.  Defaults to :attr:`DEFAULT_Y0`.
    controls:
        Initial control setpoints.  Updated via :meth:`write_setpoints`.
    volume_L:
        Reactor working volume [L].  Default 0.250 (ambr®250).
    noise:
        Noise model.  Defaults to :class:`~perfusio.simulator.noise.NoiseModel`.
    acceleration:
        Simulation speed multiplier vs. real-time.  Default 1 (real-time).
        Set to 1440 for 1 simulated day per real minute.
    seed:
        Random seed.

    Examples
    --------
    >>> import asyncio
    >>> from perfusio.simulator import CHOSimulator
    >>> sim = CHOSimulator(clone="CloneX", seed=42)
    >>> asyncio.run(sim.read_sample(day=3))  # doctest: +SKIP
    """

    STATE_SPECIES: list[str] = list(_DEFAULT_Y0.keys())

    def __init__(
        self,
        clone: str = "CloneX",
        y0: dict[str, float] | None = None,
        controls: dict[str, float] | None = None,
        volume_L: float = 0.250,
        noise: NoiseModel | None = None,
        acceleration: float = 1.0,
        seed: int = 42,
    ) -> None:
        if clone not in ("CloneX", "CloneY"):
            msg = f"clone must be 'CloneX' or 'CloneY', got '{clone}'."
            raise ValueError(msg)
        self.clone = clone
        self._kinetics = CHOKinetics(consumes_lactate=(clone == "CloneX"))
        self._y0 = {**_DEFAULT_Y0, **(y0 or {})}
        self._controls = {**_DEFAULT_CONTROLS, **(controls or {})}
        self.volume_L = volume_L
        self._noise = noise or NoiseModel(seed=seed)
        self.acceleration = acceleration
        self._current_day = 0
        self._trajectory: np.ndarray | None = None  # cached ODE solution
        self._alive = True
        self._seed = seed

    # ── BioreactorConnector interface ──────────────────────────────────────

    async def read_sample(self, day: int) -> dict[str, float | None]:
        """Return a (noisy) sample for the given culture day.

        Parameters
        ----------
        day:
            Culture day (1-indexed).

        Returns
        -------
        dict[str, float | None]
            Observed species values.  ``None`` indicates missing data.
        """
        # Simulate forward to the requested day if not cached
        if self._trajectory is None or day >= self._trajectory.shape[0]:
            await self._run_to_day(day)

        clean_state = self._clean_state_at_day(day)
        return self._noise.apply(clean_state, day=day)

    async def write_setpoints(self, setpoints: dict[str, float]) -> None:
        """Update control setpoints (resets the cached ODE trajectory).

        Parameters
        ----------
        setpoints:
            New setpoints dict.  Only keys present in the dict are updated.
        """
        changed = {k: v for k, v in setpoints.items() if self._controls.get(k) != v}
        if changed:
            self._controls.update(changed)
            self._trajectory = None  # invalidate cache

    async def is_alive(self) -> bool:
        """Return True if the virtual reactor is still running."""
        return self._alive

    # ── Public simulation helpers ──────────────────────────────────────────

    def simulate_run(
        self,
        n_days: int = 28,
        controls: dict[str, float] | None = None,
        seed: int | None = None,
    ) -> np.ndarray:
        """Run a complete N-day simulation deterministically.

        Parameters
        ----------
        n_days:
            Number of daily time steps.
        controls:
            Override control dict.  If None, uses current setpoints.
        seed:
            Ignored (ODE is deterministic).

        Returns
        -------
        np.ndarray
            Shape ``(n_days + 1, n_species)`` — clean (noiseless) trajectory.
        """
        ctrl = {**(controls or self._controls), "volume_L": self.volume_L}
        y0_vec = [self._y0[k] for k in self.STATE_SPECIES]
        return integrate_run(
            kinetics=self._kinetics,
            y0=y0_vec,
            controls=ctrl,
            n_days=n_days,
        )

    def generate_box_behnken_experiment(
        self,
        n_days: int = 28,
        seed: int = 0,
    ) -> list[dict[str, Any]]:
        """Generate a 24-run Box-Behnken training dataset.

        Performs a Box-Behnken design over 4 key control factors:
        - perfusion_rate:  [0.5, 1.0, 1.5] vvd
        - bleed_rate:      [0.10, 0.15, 0.20] vvd
        - temperature:     [36.5, 37.0, 37.5] °C
        - glucose_setpoint:[4.0, 5.0, 6.0] g/L

        Returns
        -------
        list[dict]
            Each element is ``{"run_id": int, "controls": dict,
            "trajectory": np.ndarray, "noisy_samples": list[dict]}``.
        """
        from perfusio.simulator.doe import box_behnken, scale_to_bounds

        factor_names = ["perfusion_rate", "bleed_rate", "temperature", "glucose_setpoint"]
        lo = np.array([0.5, 0.10, 36.5, 4.0])
        hi = np.array([1.5, 0.20, 37.5, 6.0])

        design = box_behnken(n_factors=4, center_points=3)
        physical = scale_to_bounds(design, lo, hi, normalised=True)

        runs = []
        for run_id, row in enumerate(physical):
            ctrl_override = {k: float(row[i]) for i, k in enumerate(factor_names)}
            ctrl = {**self._controls, **ctrl_override}

            traj = self.simulate_run(n_days=n_days, controls=ctrl)

            # Generate noisy daily samples
            nm = NoiseModel(seed=seed + run_id)
            noisy_samples = []
            for day in range(1, n_days + 1):
                clean = {k: float(traj[day, j]) for j, k in enumerate(self.STATE_SPECIES)}
                noisy_samples.append(nm.apply(clean, day=day))

            runs.append(
                {
                    "run_id": run_id,
                    "controls": ctrl,
                    "trajectory": traj,
                    "noisy_samples": noisy_samples,
                }
            )

        return runs

    # ── Private helpers ────────────────────────────────────────────────────

    async def _run_to_day(self, day: int) -> None:
        """Integrate ODE up to *day* and cache the trajectory."""
        # Simulate asynchronously to avoid blocking
        loop = asyncio.get_running_loop()
        traj = await loop.run_in_executor(
            None,
            lambda: self.simulate_run(n_days=max(day, 30)),
        )
        self._trajectory = traj

    def _clean_state_at_day(self, day: int) -> dict[str, float]:
        """Extract clean state at the given day from the cached trajectory."""
        assert self._trajectory is not None
        row = self._trajectory[min(day, self._trajectory.shape[0] - 1)]
        return {k: float(row[i]) for i, k in enumerate(self.STATE_SPECIES)}
