"""Filesystem connector — CSV / Parquet / Excel data store.

Reads bioreactor samples from flat-file archives (e.g. exported from an
ambr®250 UNICORN workstation or a LIMS export).  Supports write-back to
a rolling CSV log for closed-loop applications.

Expected CSV format::

    day,VCD,Via,Glc,Gln,Glu,Lac,Amm,Pyr,Titer
    1,2.3,98.1,4.8,...

or long-form Parquet with columns ``day``, ``species``, ``value``.
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Any

from perfusio.connectors.base import BioreactorConnectorBase

logger = logging.getLogger(__name__)


class FilesystemStore(BioreactorConnectorBase):
    """Read bioreactor data from CSV / Parquet / Excel files.

    Parameters
    ----------
    source_path:
        Path to the source data file (CSV, Parquet, or Excel).
    setpoint_log:
        Optional path to append setpoint changes as CSV rows.
    wide_format:
        If ``True`` (default), source is wide-format (one column per species).
        If ``False``, source is long-format with ``day``, ``species``, ``value``
        columns.

    Examples
    --------
    >>> from pathlib import Path
    >>> store = FilesystemStore(Path("data/run01.csv"))
    """

    def __init__(
        self,
        source_path: Path | str,
        setpoint_log: Path | str | None = None,
        wide_format: bool = True,
    ) -> None:
        self.source_path = Path(source_path)
        self.setpoint_log = Path(setpoint_log) if setpoint_log else None
        self.wide_format = wide_format
        self._data: dict[int, dict[str, Any]] | None = None

    # ── BioreactorConnectorBase ────────────────────────────────────────────

    async def read_sample(self, day: int) -> dict[str, Any]:
        """Return the sample recorded for the given culture day."""
        data = self._load_if_needed()
        return dict(data.get(day, {}))

    async def write_setpoints(self, setpoints: dict[str, float]) -> None:
        """Append setpoints to the setpoint log CSV."""
        if self.setpoint_log is None:
            logger.warning("FilesystemStore: no setpoint_log configured; write skipped.")
            return
        is_new = not self.setpoint_log.exists()
        with open(self.setpoint_log, "a", newline="") as f:
            writer = csv.writer(f)
            if is_new:
                writer.writerow(["control", "value"])
            for k, v in setpoints.items():
                writer.writerow([k, v])

    async def is_alive(self) -> bool:
        return self.source_path.exists()

    # ── Internals ──────────────────────────────────────────────────────────

    def _load_if_needed(self) -> dict[int, dict[str, Any]]:
        if self._data is not None:
            return self._data

        suffix = self.source_path.suffix.lower()
        if suffix == ".csv":
            self._data = self._load_csv()
        elif suffix in (".parquet", ".pq"):
            self._data = self._load_parquet()
        elif suffix in (".xlsx", ".xls"):
            self._data = self._load_excel()
        else:
            msg = f"Unsupported file format: {suffix}"
            raise ValueError(msg)
        return self._data

    def _load_csv(self) -> dict[int, dict[str, Any]]:
        result: dict[int, dict[str, Any]] = {}
        with open(self.source_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                day = int(row["day"])
                if self.wide_format:
                    result[day] = {
                        k: float(v) if v not in ("", "NA", "NaN", "None") else None
                        for k, v in row.items()
                        if k != "day"
                    }
                else:
                    result.setdefault(day, {})[str(row["species"])] = (
                        float(row["value"])
                        if row["value"] not in ("", "NA", "NaN", "None")
                        else None
                    )
        return result

    def _load_parquet(self) -> dict[int, dict[str, Any]]:
        import pandas as pd

        df = pd.read_parquet(self.source_path)
        result: dict[int, dict[str, Any]] = {}
        if self.wide_format:
            for _, row in df.iterrows():
                day = int(row["day"])
                result[day] = {
                    str(k): (float(v) if v == v else None) for k, v in row.items() if k != "day"
                }
        else:
            for _, row in df.iterrows():
                day = int(row["day"])
                val = row.get("value")
                result.setdefault(day, {})[str(row["species"])] = (
                    float(val) if val is not None and val == val else None
                )
        return result

    def _load_excel(self) -> dict[int, dict[str, Any]]:
        import pandas as pd

        df = pd.read_excel(self.source_path)
        result: dict[int, dict[str, Any]] = {}
        if self.wide_format:
            for _, row in df.iterrows():
                day = int(row["day"])
                result[day] = {
                    str(k): (float(v) if v == v else None) for k, v in row.items() if k != "day"
                }
        else:
            for _, row in df.iterrows():
                day = int(row["day"])
                val = row.get("value")
                result.setdefault(day, {})[str(row["species"])] = (
                    float(val) if val is not None and val == val else None
                )
        return result
