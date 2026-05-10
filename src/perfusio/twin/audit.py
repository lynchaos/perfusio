"""Immutable audit logger for the digital twin.

Produces two artefacts per run:
- ``audit_events.csv`` — human-readable event log (append-only).
- ``audit_events.parquet`` — machine-readable columnar store for analysis.

Records conform to the audit-trail requirements discussed in:

    Mione, A., et al. (2024). Regulatory considerations for self-driving
    bioprocesses. *Process Biochemistry*.

The audit log is *append-only* by design.  Past records are never modified.

References
----------
.. [Mione2024] Mione et al. (2024).
"""

from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any


class AuditLogger:
    """Append-only CSV + Parquet audit recorder.

    Parameters
    ----------
    log_dir:
        Directory in which to write audit files.
    run_id:
        Unique run identifier (written to every record).
    flush_every:
        Flush buffer to disk after this many events.

    Examples
    --------
    >>> from pathlib import Path
    >>> from perfusio.twin.audit import AuditLogger
    >>> audit = AuditLogger(log_dir=Path("/tmp/audit"), run_id="run001")
    >>> audit.log("SETPOINT_CHANGE", {"perfusion_rate": 1.2})
    >>> audit.close()
    """

    COLUMNS = [
        "timestamp",
        "run_id",
        "event_type",
        "day",
        "user",
        "payload_json",
    ]

    def __init__(
        self,
        log_dir: Path,
        run_id: str,
        flush_every: int = 10,
    ) -> None:
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id
        self.flush_every = flush_every
        self._buffer: list[dict[str, Any]] = []
        self._csv_path = self.log_dir / "audit_events.csv"

        # Write header if the CSV file is new
        if not self._csv_path.exists():
            with open(self._csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=self.COLUMNS)
                writer.writeheader()

    def log(
        self,
        event_type: str,
        payload: dict[str, Any],
        day: int = 0,
        user: str = "system",
    ) -> None:
        """Append one audit event.

        Parameters
        ----------
        event_type:
            E.g. ``"SETPOINT_CHANGE"``, ``"MODEL_RETRAIN"``,
            ``"ALARM_RAISED"``, ``"DECISION_MADE"``.
        payload:
            Serialisable dict with event-specific data.
        day:
            Culture day at which the event occurred.
        user:
            Actor (``"system"`` for automated events).
        """
        record: dict[str, Any] = {
            "timestamp": datetime.utcnow().isoformat(),
            "run_id": self.run_id,
            "event_type": event_type,
            "day": day,
            "user": user,
            "payload_json": json.dumps(payload),
        }
        self._buffer.append(record)

        if len(self._buffer) >= self.flush_every:
            self._flush()

    def close(self) -> None:
        """Flush remaining events and write a Parquet snapshot."""
        self._flush()
        self._write_parquet()

    def _flush(self) -> None:
        """Flush buffer to CSV."""
        if not self._buffer:
            return
        with open(self._csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.COLUMNS)
            writer.writerows(self._buffer)
        self._buffer = []

    def _write_parquet(self) -> None:
        """Write the full CSV as a Parquet snapshot (requires pyarrow)."""
        try:
            import pandas as pd

            df = pd.read_csv(self._csv_path)
            parquet_path = self.log_dir / "audit_events.parquet"
            df.to_parquet(parquet_path, index=False)
        except ImportError:
            pass  # parquet is optional; CSV is always written
