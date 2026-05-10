"""Abstract base class for bioreactor connectors.

All connectors — OPC UA, SQL, filesystem, and the ambr®250 emulator — must
implement this interface.  The digital twin calls only the methods defined
here, so the twin is fully connector-agnostic.

The interface matches the :class:`~perfusio._typing.BioreactorConnector`
structural protocol, providing both inheritance and Protocol compatibility.
"""

from __future__ import annotations

import abc
from typing import Any


class BioreactorConnectorBase(abc.ABC):
    """Abstract base class for bioreactor data connectors.

    Concrete connectors should subclass this and implement all abstract methods.
    The async methods are designed for use with ``asyncio``; blocking connectors
    should wrap their I/O in :func:`asyncio.get_event_loop().run_in_executor`.
    """

    @abc.abstractmethod
    async def read_sample(self, day: int) -> dict[str, Any]:
        """Read daily offline sample for the given culture day.

        Parameters
        ----------
        day:
            Culture day (1-indexed).

        Returns
        -------
        dict[str, Any]
            Species name → measurement value mapping.  ``None`` values indicate
            missing measurements.
        """

    @abc.abstractmethod
    async def write_setpoints(self, setpoints: dict[str, float]) -> None:
        """Write control setpoints to the bioreactor.

        Parameters
        ----------
        setpoints:
            Control variable name → new value mapping.

        Notes
        -----
        Implementations MUST honour the ``allow_write`` gate at the
        :class:`~perfusio.twin.digital_twin.DigitalTwin` level.  The connector
        itself does not enforce this gate.
        """

    @abc.abstractmethod
    async def is_alive(self) -> bool:
        """Return True if the connection to the bioreactor is healthy."""
