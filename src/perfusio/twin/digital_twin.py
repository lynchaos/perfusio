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
from typing import TYPE_CHECKING, Any

import torch

from perfusio.twin.audit import AuditLogger
from perfusio.twin.notifications import AlarmNotifier
from perfusio.twin.scheduler import DailyScheduler

if TYPE_CHECKING:
    from perfusio._typing import BioreactorConnector
    from perfusio.bed.policies import BEDPolicy
    from perfusio.config import DesignSpace
    from perfusio.hybrid.model import HybridStateSpaceModel

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
        connector: BioreactorConnector,
        hybrid: HybridStateSpaceModel,
        design_space: DesignSpace,
        bed_policy: BEDPolicy,
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
        return asyncio.run(self._async_step(day))

    def run_sync(self, n_days: int, interval_seconds: float = 0.0) -> None:
        """Run the full twin synchronously (blocking)."""
        asyncio.run(self.run(n_days=n_days, interval_seconds=interval_seconds))

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

        # 8. Buffer observation (species + controls applied this day) for online retraining
        ctrl_tensor = torch.tensor(
            [
                float(decision.recommended_controls.get(k, 0.0))
                for k in self.design_space.control_names
            ],
            dtype=torch.float64,
        )
        self._obs_buffer.append({"sample": sample, "day": day, "controls_tensor": ctrl_tensor})

        return {"sample": sample, "forecast": forecast, "decision": decision}

    def _retrain(self) -> None:
        """Online-retrain the GP from buffered observations."""
        if len(self._obs_buffer) < 2:
            return

        from perfusio.hybrid.train import retrain_online

        n_species = len(self.species_names)
        n_controls = len(self.design_space.control_names)

        # Build one-step base rows: [species(9), controls(6), day(1)] → (N, 16)
        base_rows = []
        for ob in self._obs_buffer:
            sp = torch.tensor(
                [float(ob["sample"].get(s, 0.0) or 0.0) for s in self.species_names],
                dtype=torch.float64,
            )
            ctrl = ob.get("controls_tensor", torch.zeros(n_controls, dtype=torch.float64))
            day_t = torch.tensor([float(ob["day"])], dtype=torch.float64)
            base_rows.append(torch.cat([sp, ctrl, day_t]))  # (16,)

        # Input rows are t=0..N-2; targets are species at t=1..N-1
        base_x = torch.stack(base_rows[:-1])   # (N-1, 16)
        new_y_base = torch.stack([
            torch.tensor(
                [float(ob["sample"].get(s, 0.0) or 0.0) for s in self.species_names],
                dtype=torch.float64,
            )
            for ob in self._obs_buffer[1:]
        ])  # (N-1, n_species)

        N = base_x.shape[0]

        # Build indexed multi-task format: (N*n_species, 17) and (N*n_species,)
        xs_flat, ys_flat = [], []
        for task_id in range(n_species):
            task_col = torch.full((N, 1), float(task_id), dtype=torch.float64)
            xs_flat.append(torch.cat([base_x, task_col], dim=1))  # (N, 17)
            ys_flat.append(new_y_base[:, task_id])                 # (N,)

        new_x = torch.cat(xs_flat, dim=0)  # (N * n_species, 17)
        new_y = torch.cat(ys_flat, dim=0)  # (N * n_species,)

        retrain_online(
            new_x=new_x,
            new_y=new_y,
            model=self.hybrid.gp_model.model,       # MultiTaskRateGP (ExactGP)
            likelihood=self.hybrid.gp_model.likelihood,
        )

    def _build_surrogate(self) -> Any:
        """Build a minimal SingleTaskGP surrogate for the BED policy."""
        from botorch.fit import fit_gpytorch_mll
        from botorch.models import SingleTaskGP
        from gpytorch.mlls import ExactMarginalLogLikelihood

        if len(self._obs_buffer) < 2:
            # Not enough data — return a trivial surrogate at the origin
            train_x = torch.zeros(2, 1, dtype=torch.float64)
            train_y = torch.zeros(2, 1, dtype=torch.float64)
        else:
            Y = torch.tensor(
                [
                    [float(ob.get(s, 0.0) or 0.0) for s in self.species_names]
                    for ob in self._obs_buffer
                ],
                dtype=torch.float64,
            )
            train_x = torch.arange(len(Y), dtype=torch.float64).unsqueeze(-1)
            train_y = Y[:, :1]  # use VCD as primary objective for surrogate

        surrogate = SingleTaskGP(train_x, train_y)
        mll = ExactMarginalLogLikelihood(surrogate.likelihood, surrogate)
        try:
            fit_gpytorch_mll(mll)
        except Exception:
            pass  # use prior if fitting fails on small data
        return surrogate
