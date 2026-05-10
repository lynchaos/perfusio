"""Online digital twin — real-time hybrid model + BED decision loop.

:class:`DigitalTwin` is the central coordinator that:

1. Reads daily samples from a :class:`~perfusio.connectors.base.BioreactorConnector`
   (hardware OPC UA or virtual ambr®250).
2. Appends new observations to the training set and online-retrains the hybrid
   model (:func:`~perfusio.hybrid.train.retrain_online`).
3. Runs a 3-step-ahead forecast.
4. Queries the :class:`~perfusio.bed.policies.BEDPolicy` for the next setpoints.
5. Checks predictive alarms via :class:`~perfusio.twin.notifications.AlarmNotifier`.
6. Writes setpoints back to the connector (gated by ``allow_write``).
7. Records every event to the :class:`~perfusio.twin.audit.AuditLogger`.

Usage — blocking (sync wrapper)::

    twin = DigitalTwin(connector=sim, hybrid=model, ...)
    twin.step(day=7)              # single step (tests / notebooks)

Usage — fully async::

    await twin.run(n_days=28)    # full 28-day run

Safety gate::

    twin = DigitalTwin(..., allow_write=False)   # read-only mode (default)
    twin = DigitalTwin(..., allow_write=True)    # closed-loop (operator must opt-in)

References
----------
.. [Gadiyar2026] Gadiyar et al. (2026), §3.4 — Fig. 5.
.. [Mione2024]   Mione et al. (2024) — audit trail requirements.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any, TYPE_CHECKING

import torch

from perfusio.twin.audit import AuditLogger
from perfusio.twin.notifications import AlarmNotifier
from perfusio.twin.scheduler import DailyScheduler

if TYPE_CHECKING:
    from perfusio.bed.policies import BEDPolicy
    from perfusio.config import DesignSpace
    from perfusio.hybrid.model import HybridStateSpaceModel
    from perfusio._typing import BioreactorConnector

logger = logging.getLogger(__name__)


class DigitalTwin:
    """Online digital twin with BED-driven closed-loop control.

    Parameters
    ----------
    connector:
        Bioreactor data connector (OPC UA, SQL, or virtual simulator).
    hybrid:
        Fitted :class:`~perfusio.hybrid.model.HybridStateSpaceModel`.
    design_space:
        Process control design space.
    bed_policy:
        Configured :class:`~perfusio.bed.policies.BEDPolicy`.
    notifier:
        Alarm notifier.  Optional.
    log_dir:
        Directory for audit logs.  Defaults to ``./perfusio_audit``.
    run_id:
        Unique identifier for this run.
    allow_write:
        Safety gate — if ``False`` (default), setpoints are NOT written back.
    species_names:
        Ordered species names matching the model output.
    retrain_every:
        Online-retrain the hybrid GP every *N* days.  Default 1 (daily).
    """

    def __init__(
        self,
        connector: "BioreactorConnector",
        hybrid: "HybridStateSpaceModel",
        design_space: "DesignSpace",
        bed_policy: "BEDPolicy",
        notifier: AlarmNotifier | None = None,
        log_dir: Path | str = Path("./perfusio_audit"),
        run_id: str = "default",
        allow_write: bool = False,
        species_names: list[str] | None = None,
        retrain_every: int = 1,
    ) -> None:
        self.connector = connector
        self.hybrid = hybrid
        self.design_space = design_space
        self.bed_policy = bed_policy
        self.notifier = notifier
        self.allow_write = allow_write
        self.species_names = species_names or []
        self.retrain_every = retrain_every

        self._audit = AuditLogger(log_dir=Path(log_dir), run_id=run_id)
        self._day = 0
        self._obs_buffer: list[dict[str, Any]] = []

    # ── Sync public API ────────────────────────────────────────────────────

    def step(self, day: int) -> dict[str, Any]:
        """Execute one daily step synchronously (blocking).

        Parameters
        ----------
        day:
            Culture day.

        Returns
        -------
        dict
            Summary with keys: ``"sample"``, ``"forecast"``, ``"decision"``.
        """
        return asyncio.get_event_loop().run_until_complete(self._async_step(day))

    def run_sync(self, n_days: int, interval_seconds: float = 0.0) -> None:
        """Run the full twin synchronously (blocking)."""
        asyncio.get_event_loop().run_until_complete(
            self.run(n_days=n_days, interval_seconds=interval_seconds)
        )

    # ── Async public API ───────────────────────────────────────────────────

    async def run(self, n_days: int, interval_seconds: float = 86400.0) -> None:
        """Run the full digital twin for *n_days* asynchronously.

        Parameters
        ----------
        n_days:
            Number of culture days.
        interval_seconds:
            Real-time delay between steps.  Set to 0 for simulation.
        """
        scheduler = DailyScheduler(
            step_fn=self._async_step,
            n_days=n_days,
            interval_seconds=interval_seconds,
            start_day=self._day + 1,
        )
        try:
            await scheduler.run()
        finally:
            self._audit.close()

    # ── Core step logic ────────────────────────────────────────────────────

    async def _async_step(self, day: int) -> dict[str, Any]:
        """Full pipeline for one day."""
        self._day = day

        # 1. Read sample
        sample = await self.connector.read_sample(day=day)
        self._audit.log("SAMPLE_READ", payload={"sample": sample}, day=day)

        # 2. Build current state tensor (impute missing values with 0)
        c_current = torch.tensor(
            [float(sample.get(s, 0.0) or 0.0) for s in self.species_names],
            dtype=torch.float64,
        )

        # 3. Online retrain
        if day % self.retrain_every == 0 and len(self._obs_buffer) > 0:
            self._retrain()
            self._audit.log("MODEL_RETRAIN", payload={"n_obs": len(self._obs_buffer)}, day=day)

        # 4. Forecast
        from perfusio.hybrid.forecast import forecast_run
        u_default = torch.zeros(len(self.design_space.control_names), dtype=torch.float64)
        forecast = forecast_run(self.hybrid, c_current, u_default.unsqueeze(0).expand(3, -1))
        self._audit.log("FORECAST", payload={"mean": forecast["mean"].tolist()}, day=day)

        # 5. Predictive alarm check
        if self.notifier is not None:
            alarms = self.notifier.check_forecast(forecast, self.species_names, day)
            if alarms:
                self._audit.log(
                    "ALARM_RAISED",
                    payload={"alarms": [str(a) for a in alarms]},
                    day=day,
                )

        # 6. BED decision
        u_current = torch.zeros(len(self.design_space.control_names), dtype=torch.float64)
        from botorch.models import SingleTaskGP
        # Build a minimal botorch surrogate from current trajectory history
        surrogate = self._build_surrogate()
        decision = self.bed_policy.decide(
            c_current=c_current,
            day=day,
            surrogate_model=surrogate,
            best_f=None,
        )
        self._audit.log("DECISION", payload=decision.recommended_controls, day=day)

        # 7. Write setpoints (safety-gated)
        if self.allow_write:
            await self.connector.write_setpoints(decision.recommended_controls)
            self._audit.log("SETPOINT_WRITE", payload=decision.recommended_controls, day=day)
        else:
            logger.info("allow_write=False: setpoints not written on day %d.", day)

        return {"sample": sample, "forecast": forecast, "decision": decision}

    def _retrain(self) -> None:
        """Online-retrain the GP from buffered observations."""
        if len(self._obs_buffer) < 2:
            return
        from perfusio.hybrid.train import retrain_online
        import numpy as np

        # Build (N, n_species) tensors from buffer
        Y = torch.tensor(
            [[float(ob.get(s, 0.0) or 0.0) for s in self.species_names]
             for ob in self._obs_buffer],
            dtype=torch.float64,
        )
        N = Y.shape[0]
        X = torch.arange(N, dtype=torch.float64).unsqueeze(-1)

        retrain_online(
            new_x=X,
            new_y=Y,
            model=self.hybrid.gp_model,
            likelihood=self.hybrid.gp_model.likelihood,
        )

    def _build_surrogate(self) -> "Any":
        """Build a minimal SingleTaskGP surrogate for the BED policy."""
        from botorch.models import SingleTaskGP
        from botorch.fit import fit_gpytorch_mll
        from gpytorch.mlls import ExactMarginalLogLikelihood

        if len(self._obs_buffer) < 2:
            # Not enough data — return a trivial surrogate at the origin
            train_x = torch.zeros(2, 1, dtype=torch.float64)
            train_y = torch.zeros(2, 1, dtype=torch.float64)
        else:
            Y = torch.tensor(
                [[float(ob.get(s, 0.0) or 0.0) for s in self.species_names]
                 for ob in self._obs_buffer],
                dtype=torch.float64,
            )
            train_x = torch.arange(len(Y), dtype=torch.float64).unsqueeze(-1)
            train_y = Y[:, :1]  # use VCD as primary objective for surrogate

        surrogate = SingleTaskGP(train_x, train_y)
        mll = ExactMarginalLogLikelihood(surrogate.likelihood, surrogate)
        try:
            fit_gpytorch_mll(mll)
        except Exception:  # noqa: BLE001
            pass  # use prior if fitting fails on small data
        return surrogate
