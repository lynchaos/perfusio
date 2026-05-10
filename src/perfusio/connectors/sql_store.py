"""SQLAlchemy 2.0 data store for bioreactor run data.

Schema (8 tables):
- ``experiment``      — top-level experiment metadata
- ``reactor``         — individual reactor/vessel metadata
- ``sample``          — offline daily sample measurements
- ``setpoint_history``— historical setpoint changes
- ``model_run``       — GP/hybrid model training runs
- ``forecast``        — multi-step forecasts stored for audit
- ``audit_event``     — structured audit events (mirror of CSV log)
- ``alarm_event``     — predictive alarms

Uses Alembic-compatible ``MetaData`` declarations.  Run migrations with::

    alembic upgrade head

References
----------
.. [Mione2024] Mione et al. (2024) — regulatory data integrity requirements.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    create_engine,
    select,
)
from sqlalchemy.orm import Session

from perfusio.connectors.base import BioreactorConnectorBase

# ── ORM metadata ──────────────────────────────────────────────────────────────

metadata = MetaData()

experiment_table = Table(
    "experiment",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("name", String(128), unique=True, nullable=False),
    Column("created_at", DateTime, default=datetime.utcnow),
    Column("description", Text),
)

reactor_table = Table(
    "reactor",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("experiment_id", Integer, ForeignKey("experiment.id"), nullable=False),
    Column("name", String(64), nullable=False),
    Column("clone", String(32)),
    Column("volume_L", Float),
)

sample_table = Table(
    "sample",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("reactor_id", Integer, ForeignKey("reactor.id"), nullable=False),
    Column("day", Integer, nullable=False),
    Column("sampled_at", DateTime, default=datetime.utcnow),
    Column("species", String(32), nullable=False),
    Column("value", Float),  # NULL = missing
)

setpoint_history_table = Table(
    "setpoint_history",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("reactor_id", Integer, ForeignKey("reactor.id"), nullable=False),
    Column("day", Integer, nullable=False),
    Column("written_at", DateTime, default=datetime.utcnow),
    Column("control", String(64), nullable=False),
    Column("value", Float, nullable=False),
    Column("source", String(32), default="BED"),
)

model_run_table = Table(
    "model_run",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("experiment_id", Integer, ForeignKey("experiment.id"), nullable=False),
    Column("day", Integer),
    Column("trained_at", DateTime, default=datetime.utcnow),
    Column("mll", Float),
    Column("n_obs", Integer),
    Column("hyperparams_json", Text),
)

forecast_table = Table(
    "forecast",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("reactor_id", Integer, ForeignKey("reactor.id"), nullable=False),
    Column("day", Integer),
    Column("horizon", Integer),
    Column("species", String(32)),
    Column("mean", Float),
    Column("q10", Float),
    Column("q90", Float),
)

audit_event_table = Table(
    "audit_event",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("run_id", String(128), nullable=False),
    Column("event_type", String(64), nullable=False),
    Column("day", Integer),
    Column("user", String(64), default="system"),
    Column("created_at", DateTime, default=datetime.utcnow),
    Column("payload_json", Text),
)

alarm_event_table = Table(
    "alarm_event",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("reactor_id", Integer, ForeignKey("reactor.id"), nullable=False),
    Column("day", Integer),
    Column("species", String(32)),
    Column("predicted_value", Float),
    Column("threshold", Float),
    Column("direction", String(8)),
    Column("lead_days", Integer),
    Column("created_at", DateTime, default=datetime.utcnow),
)


# ── SQLStore connector ────────────────────────────────────────────────────────


class SQLStore(BioreactorConnectorBase):
    """SQLAlchemy-backed connector and data archive.

    Parameters
    ----------
    db_url:
        SQLAlchemy database URL.  E.g.:
        - ``"sqlite:///perfusio.db"``
        - ``"postgresql+psycopg2://user:pass@host/db"``
    experiment_name:
        Name of the experiment row to read/write.
    reactor_name:
        Name of the reactor row.
    echo:
        If ``True``, echo SQL to stdout.

    Examples
    --------
    >>> store = SQLStore("sqlite:///test.db", "Exp1", "R1")
    >>> store.create_tables()
    """

    def __init__(
        self,
        db_url: str,
        experiment_name: str,
        reactor_name: str,
        echo: bool = False,
    ) -> None:
        self.db_url = db_url
        self.experiment_name = experiment_name
        self.reactor_name = reactor_name
        self._engine = create_engine(db_url, echo=echo, future=True)
        self._reactor_id: int | None = None

    def create_tables(self) -> None:
        """Create all tables if they do not already exist."""
        metadata.create_all(self._engine)

    # ── BioreactorConnectorBase ────────────────────────────────────────────

    async def read_sample(self, day: int) -> dict[str, Any]:
        """Read all species measurements for the given culture day."""
        reactor_id = self._get_or_create_reactor()
        with Session(self._engine) as s:
            rows = s.execute(
                select(sample_table.c.species, sample_table.c.value)
                .where(sample_table.c.reactor_id == reactor_id)
                .where(sample_table.c.day == day)
            ).fetchall()
        return {row.species: row.value for row in rows}

    async def write_setpoints(self, setpoints: dict[str, float]) -> None:
        """Persist setpoint changes to the ``setpoint_history`` table."""
        reactor_id = self._get_or_create_reactor()
        with Session(self._engine) as s:
            for control, value in setpoints.items():
                s.execute(
                    setpoint_history_table.insert().values(
                        reactor_id=reactor_id,
                        day=0,  # caller should supply day; default 0
                        control=control,
                        value=float(value),
                        source="BED",
                    )
                )
            s.commit()

    async def is_alive(self) -> bool:
        try:
            with Session(self._engine) as s:
                s.execute(select(1))
            return True
        except Exception:
            return False

    # ── Write helpers ──────────────────────────────────────────────────────

    def write_sample(self, day: int, sample: dict[str, float | None]) -> None:
        """Insert a sample record into the database."""
        reactor_id = self._get_or_create_reactor()
        with Session(self._engine) as s:
            for species, value in sample.items():
                s.execute(
                    sample_table.insert().values(
                        reactor_id=reactor_id,
                        day=day,
                        species=species,
                        value=float(value) if value is not None else None,
                    )
                )
            s.commit()

    def log_audit_event(
        self,
        run_id: str,
        event_type: str,
        payload: dict[str, Any],
        day: int = 0,
        user: str = "system",
    ) -> None:
        """Insert an audit event record."""
        with Session(self._engine) as s:
            s.execute(
                audit_event_table.insert().values(
                    run_id=run_id,
                    event_type=event_type,
                    day=day,
                    user=user,
                    payload_json=json.dumps(payload),
                )
            )
            s.commit()

    # ── Internals ──────────────────────────────────────────────────────────

    def _get_or_create_reactor(self) -> int:
        if self._reactor_id is not None:
            return self._reactor_id

        with Session(self._engine) as s:
            exp = s.execute(
                select(experiment_table.c.id).where(experiment_table.c.name == self.experiment_name)
            ).fetchone()
            if exp is None:
                result = s.execute(experiment_table.insert().values(name=self.experiment_name))
                s.commit()
                exp_id = result.inserted_primary_key[0]  # pyright: ignore[reportAttributeAccessIssue]
            else:
                exp_id = exp.id

            reactor = s.execute(
                select(reactor_table.c.id).where(
                    reactor_table.c.experiment_id == exp_id,
                    reactor_table.c.name == self.reactor_name,
                )
            ).fetchone()
            if reactor is None:
                result = s.execute(
                    reactor_table.insert().values(
                        experiment_id=exp_id,
                        name=self.reactor_name,
                    )
                )
                s.commit()
                self._reactor_id = result.inserted_primary_key[0]  # pyright: ignore[reportAttributeAccessIssue]
            else:
                self._reactor_id = reactor.id

        assert self._reactor_id is not None
        return self._reactor_id
