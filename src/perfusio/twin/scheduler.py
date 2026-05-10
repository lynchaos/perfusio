"""Daily decision scheduler for the digital twin.

Runs a 24-hour async cadence: read sample → update model → forecast →
decide setpoints → write setpoints → log audit trail.

The scheduler is designed to be cancellable (``asyncio.CancelledError`` is
handled gracefully) and restartable.

References
----------
.. [Gadiyar2026] Gadiyar et al. (2026), Fig. 5 — closed-loop timing diagram.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable, Coroutine
from typing import Any

logger = logging.getLogger(__name__)


class DailyScheduler:
    """Async daily-cadence scheduler for the digital twin.

    Parameters
    ----------
    step_fn:
        Coroutine factory ``step_fn(day: int) -> None`` — the work to
        execute each simulated day.  Raises ``StopIteration`` to end run.
    n_days:
        Total number of days to run.
    interval_seconds:
        Real-time interval between steps (default 86400 = 24 h real time).
        Set to a small value for testing (e.g. 1 s).
    start_day:
        First day number.

    Examples
    --------
    >>> import asyncio
    >>> from perfusio.twin.scheduler import DailyScheduler
    >>> async def my_step(day: int) -> None:
    ...     print(f"Day {day}")
    >>> scheduler = DailyScheduler(my_step, n_days=3, interval_seconds=0)
    >>> asyncio.run(scheduler.run())
    Day 1
    Day 2
    Day 3
    """

    def __init__(
        self,
        step_fn: Callable[[int], Coroutine[Any, Any, None]],
        n_days: int,
        interval_seconds: float = 86400.0,
        start_day: int = 1,
    ) -> None:
        self.step_fn = step_fn
        self.n_days = n_days
        self.interval_seconds = interval_seconds
        self.start_day = start_day
        self._running = False

    async def run(self) -> None:
        """Execute the scheduler for all days."""
        self._running = True
        try:
            for day in range(self.start_day, self.start_day + self.n_days):
                if not self._running:
                    logger.info("DailyScheduler: stop requested, halting at day %d.", day)
                    break
                logger.info("DailyScheduler: executing day %d.", day)
                try:
                    await self.step_fn(day)
                except StopIteration:
                    logger.info("DailyScheduler: step_fn raised StopIteration at day %d.", day)
                    break
                except asyncio.CancelledError:
                    logger.info("DailyScheduler: cancelled at day %d.", day)
                    raise
                except Exception:
                    logger.exception("DailyScheduler: unhandled exception at day %d.", day)
                    raise

                if self.interval_seconds > 0:
                    await asyncio.sleep(self.interval_seconds)
        finally:
            self._running = False

    def stop(self) -> None:
        """Request a graceful stop after the current day completes."""
        self._running = False
