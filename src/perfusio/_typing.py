"""Custom types, TypeAliases, and Protocols for ``perfusio``.

This module is the single place where ``Any`` may appear (in Protocol stubs
and adapter contracts). All other modules must import from here rather than
using ``typing.Any`` directly.

Notes
-----
Only import from this module using ``TYPE_CHECKING`` guards in runtime code
to avoid circular imports and keep start-up cost low.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, Protocol, TypeAlias, runtime_checkable

import numpy as np
import torch
from torch import Tensor

# ---------------------------------------------------------------------------
# Scalar types
# ---------------------------------------------------------------------------

#: A Python float, int, or zero-dimensional torch tensor.
Numeric: TypeAlias = float | int | Tensor

#: A 1-D array of float32/float64 (numpy or torch).
Vector: TypeAlias = Tensor | np.ndarray[Any, np.dtype[np.floating[Any]]]

#: A 2-D array of float64.
Matrix: TypeAlias = Tensor | np.ndarray[Any, np.dtype[np.floating[Any]]]

#: A path-like object or string.
PathLike: TypeAlias = Path | str

# ---------------------------------------------------------------------------
# Shape annotations (documentation only — not enforced at runtime)
# ---------------------------------------------------------------------------
#: Batch dimension over reactors.
_B = int
#: Number of species / tasks.
_K = int
#: Time steps.
_T = int
#: State dimension.
_D = int
#: GP input dimension.
_Din = int

# ---------------------------------------------------------------------------
# Acquisition function type
# ---------------------------------------------------------------------------

#: A callable that maps (candidate_points: Tensor) -> Tensor (acquisition values).
AcquisitionFn: TypeAlias = Callable[[Tensor], Tensor]

# ---------------------------------------------------------------------------
# Protocols
# ---------------------------------------------------------------------------


@runtime_checkable
class BioreactorConnector(Protocol):
    """Protocol that all bioreactor data connectors must satisfy.

    Both the real OPC UA client and the ambr®250 emulator implement this
    protocol so that the :class:`~perfusio.twin.digital_twin.DigitalTwin`
    can be tested offline.
    """

    async def read_sample(self, reactor_id: str) -> Mapping[str, float]:
        """Return the latest at-line sample for *reactor_id*.

        Parameters
        ----------
        reactor_id:
            Unique identifier for the bioreactor (e.g. ``"R01"``).

        Returns
        -------
        Mapping[str, float]
            Species name → measured value, e.g.
            ``{"VCD": 12.3, "Via": 97.1, "Glc": 1.8}``.

        Raises
        ------
        ConnectionError
            If the connector is offline and cannot recover.
        """
        ...

    async def write_setpoints(self, reactor_id: str, setpoints: Mapping[str, float]) -> None:
        """Write control setpoints to the reactor controller.

        Parameters
        ----------
        reactor_id:
            Target reactor.
        setpoints:
            Control variable name → desired value.

        Raises
        ------
        PermissionError
            If the connector is in read-only mode
            (``--allow-write`` flag not provided).
        ConnectionError
            If the write fails.
        """
        ...

    async def is_alive(self) -> bool:
        """Return ``True`` if the connection is healthy."""
        ...


@runtime_checkable
class Fittable(Protocol):
    """Any object with a ``fit`` method."""

    def fit(self, *args: Any, **kwargs: Any) -> None:
        """Fit the model in-place."""
        ...


@runtime_checkable
class Predictable(Protocol):
    """Any object with a ``predict`` method returning a Tensor."""

    def predict(self, x: Tensor) -> Tensor:
        """Return point predictions at *x*."""
        ...


# ---------------------------------------------------------------------------
# Callback / hook types
# ---------------------------------------------------------------------------

#: A hook called after every digital-twin step with the step result.
StepCallback: TypeAlias = Callable[..., None]

#: Factory producing a fresh model instance (used by ensemble).
ModelFactory: TypeAlias = Callable[[], Any]

# ---------------------------------------------------------------------------
# Device / dtype helpers
# ---------------------------------------------------------------------------

#: Default torch dtype — double precision everywhere.
DEFAULT_DTYPE: torch.dtype = torch.float64
DEFAULT_DEVICE: torch.device = torch.device("cpu")
