"""State, StateBatch, and Trajectory data structures for ``perfusio``.

These are the primary data containers that flow through the entire library.
They are deliberately plain dataclasses (not Pydantic models) so that PyTorch
operations can be applied directly without serialisation overhead.

All tensors are stored as ``torch.float64`` on the CPU by default.  When GPU
support is desired, call ``.to(device)`` on the container.

Design notes
------------
- ``State`` holds the full bioreactor state at a single time step.
- ``StateBatch`` holds states for *B* reactors at a single time step.
- ``Trajectory`` holds the full time series for a *single* reactor run.

The distinction matters for the digital twin: the BED optimiser works on
``StateBatch`` (one entry per candidate reactor), while the GP is trained on
``Trajectory`` objects (one per historical run).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch
from torch import Tensor

if TYPE_CHECKING:
    pass


@dataclass
class State:
    """Bioreactor state at a single discrete time step.

    All numeric fields are plain Python floats (or ``None`` for missing
    measurements), so that ``State`` objects can be serialised easily.

    Parameters
    ----------
    day:
        Process day (0-indexed integer; ``0`` = inoculation day).
    species:
        Mapping of species name → measured (or simulated) value.
        Missing measurements are represented as ``float("nan")``.
    controls:
        Mapping of control variable name → applied setpoint.
    volume_L:
        Reactor working volume in litres.  For ambr®250, this is typically
        0.250 L.  Must be > 0.
    reactor_id:
        Unique string identifier for the reactor vessel (e.g. ``"R01"``).
    clone_id:
        Cell-line / clone identifier used by :class:`~perfusio.embedding.clones.CloneRegistry`.
    run_id:
        Experiment / run identifier.

    Notes
    -----
    The ``species`` dict uses the canonical species names defined in
    :mod:`perfusio.chemistry.species`; see that module for the full registry.

    Examples
    --------
    >>> s = State(
    ...     day=5,
    ...     species={"VCD": 18.3, "Via": 97.1, "Glc": 3.2, "Amm": 4.1},
    ...     controls={"perfusion_rate": 1.0, "bleed_rate": 0.15},
    ...     volume_L=0.250,
    ...     reactor_id="R01",
    ...     clone_id="CX",
    ...     run_id="EXP001",
    ... )
    >>> s.day
    5
    """

    day: int
    species: dict[str, float]
    controls: dict[str, float]
    volume_L: float
    reactor_id: str = "R00"
    clone_id: str = "unknown"
    run_id: str = "unknown"

    def to_tensor(
        self,
        species_names: list[str],
        control_names: list[str],
        *,
        dtype: torch.dtype = torch.float64,
        device: torch.device | str = "cpu",
    ) -> Tensor:
        """Serialise this state to a flat 1-D tensor.

        Parameters
        ----------
        species_names:
            Ordered list of species to include; missing ones are ``nan``.
        control_names:
            Ordered list of controls to include; missing ones are ``nan``.
        dtype:
            Target dtype (default ``float64``).
        device:
            Target device.

        Returns
        -------
        Tensor
            Shape ``(len(species_names) + len(control_names) + 1,)``.
            The trailing +1 is the process day cast to float.
        """
        s_vals = [self.species.get(k, float("nan")) for k in species_names]
        c_vals = [self.controls.get(k, float("nan")) for k in control_names]
        all_vals = s_vals + c_vals + [float(self.day)]
        return torch.tensor(all_vals, dtype=dtype, device=device)

    @classmethod
    def from_tensor(
        cls,
        t: Tensor,
        species_names: list[str],
        control_names: list[str],
        *,
        day: int,
        volume_L: float = 0.250,
        reactor_id: str = "R00",
        clone_id: str = "unknown",
        run_id: str = "unknown",
    ) -> State:
        """Reconstruct a :class:`State` from a flat tensor.

        Parameters
        ----------
        t:
            Flat 1-D tensor as produced by :meth:`to_tensor`.
        species_names:
            Species ordering used when the tensor was created.
        control_names:
            Control variable ordering.
        day:
            Process day (not stored in the tensor to avoid dtype issues).
        volume_L:
            Working volume.
        reactor_id:
            Reactor ID.
        clone_id:
            Clone ID.
        run_id:
            Run ID.

        Returns
        -------
        State
        """
        ns = len(species_names)
        nc = len(control_names)
        vals = t.detach().cpu().tolist()
        species = {k: float(vals[i]) for i, k in enumerate(species_names)}
        controls = {k: float(vals[ns + i]) for i, k in enumerate(control_names)}
        return cls(
            day=day,
            species=species,
            controls=controls,
            volume_L=volume_L,
            reactor_id=reactor_id,
            clone_id=clone_id,
            run_id=run_id,
        )


@dataclass
class StateBatch:
    """Bioreactor states for *B* reactors at a single time step.

    Parameters
    ----------
    day:
        Shared process day.
    species:
        Tensor of shape ``(B, n_species)``; rows are reactors, columns are
        species in canonical order.
    controls:
        Tensor of shape ``(B, n_controls)``.
    volume_L:
        Tensor of shape ``(B,)`` with working volumes in litres.
    reactor_ids:
        List of length *B* with reactor identifiers.
    clone_ids:
        List of length *B* with clone identifiers.
    run_ids:
        List of length *B* with run identifiers.
    species_names:
        Column names for the *species* tensor.
    control_names:
        Column names for the *controls* tensor.

    Examples
    --------
    >>> import torch
    >>> sb = StateBatch(
    ...     day=10,
    ...     species=torch.zeros(4, 11),
    ...     controls=torch.zeros(4, 6),
    ...     volume_L=torch.full((4,), 0.250),
    ...     reactor_ids=["R01", "R02", "R03", "R04"],
    ...     clone_ids=["CX"] * 4,
    ...     run_ids=["EXP001"] * 4,
    ...     species_names=["VCD", "VCV", "Via", "Diam", "Glc", "Gln",
    ...                     "Glu", "Lac", "Amm", "Pyr", "Titer"],
    ...     control_names=["perfusion_rate", "bleed_rate", "glucose_setpoint",
    ...                     "temperature", "agitation", "pyruvate_feed"],
    ... )
    >>> sb.batch_size
    4
    """

    day: int
    species: Tensor  # (B, n_species)
    controls: Tensor  # (B, n_controls)
    volume_L: Tensor  # (B,)
    reactor_ids: list[str]
    clone_ids: list[str]
    run_ids: list[str]
    species_names: list[str]
    control_names: list[str]

    @property
    def batch_size(self) -> int:
        """Number of reactors in this batch."""
        return int(self.species.shape[0])

    def to_input_tensor(self) -> Tensor:
        """Concatenate species + controls into a single ``(B, d)`` tensor.

        Returns
        -------
        Tensor
            Shape ``(B, n_species + n_controls + 1)`` where the last column
            is the process day as float.
        """
        day_col = torch.full(
            (self.batch_size, 1),
            float(self.day),
            dtype=self.species.dtype,
            device=self.species.device,
        )
        return torch.cat([self.species, self.controls, day_col], dim=-1)


@dataclass
class Trajectory:
    """Full time-series data for a single bioreactor run.

    A ``Trajectory`` is the primary input to GP training.  Each row
    corresponds to one daily observation; the number of rows equals the run
    length in days.

    Parameters
    ----------
    species:
        Float tensor of shape ``(T, n_species)`` with measured species values.
        Missing observations are ``float("nan")``.
    controls:
        Float tensor of shape ``(T, n_controls)`` with applied setpoints.
    volume_L:
        Float tensor of shape ``(T,)`` with reactor working volumes.
    days:
        Integer tensor of shape ``(T,)`` with process days (0-indexed).
    species_names:
        Column names for *species*.
    control_names:
        Column names for *controls*.
    run_id:
        Unique run identifier.
    clone_id:
        Clone / cell-line identifier.
    reactor_id:
        Reactor vessel identifier.
    metadata:
        Arbitrary string → string metadata (e.g. medium lot, operator).

    Notes
    -----
    All tensors should be on the same device and have the same dtype.
    Use :meth:`to` to move the trajectory to a different device.

    Examples
    --------
    >>> import torch
    >>> traj = Trajectory(
    ...     species=torch.randn(20, 11),
    ...     controls=torch.ones(20, 6),
    ...     volume_L=torch.full((20,), 0.250),
    ...     days=torch.arange(20),
    ...     species_names=["VCD", "VCV", "Via", "Diam", "Glc", "Gln",
    ...                     "Glu", "Lac", "Amm", "Pyr", "Titer"],
    ...     control_names=["perfusion_rate", "bleed_rate", "glucose_setpoint",
    ...                     "temperature", "agitation", "pyruvate_feed"],
    ...     run_id="EXP001",
    ...     clone_id="CX",
    ...     reactor_id="R01",
    ... )
    >>> traj.n_days
    20
    """

    species: Tensor  # (T, n_species)
    controls: Tensor  # (T, n_controls)
    volume_L: Tensor  # (T,)
    days: Tensor  # (T,) int
    species_names: list[str]
    control_names: list[str]
    run_id: str = "unknown"
    clone_id: str = "unknown"
    reactor_id: str = "R00"
    metadata: dict[str, str] = field(default_factory=dict)

    @property
    def n_days(self) -> int:
        """Number of time steps in the trajectory."""
        return int(self.species.shape[0])

    @property
    def n_species(self) -> int:
        """Number of measured species."""
        return int(self.species.shape[1])

    @property
    def n_controls(self) -> int:
        """Number of control variables."""
        return int(self.controls.shape[1])

    def to(self, device: torch.device | str) -> Trajectory:
        """Return a copy with all tensors moved to *device*.

        Parameters
        ----------
        device:
            Target PyTorch device.

        Returns
        -------
        Trajectory
        """
        return Trajectory(
            species=self.species.to(device),
            controls=self.controls.to(device),
            volume_L=self.volume_L.to(device),
            days=self.days.to(device),
            species_names=self.species_names,
            control_names=self.control_names,
            run_id=self.run_id,
            clone_id=self.clone_id,
            reactor_id=self.reactor_id,
            metadata=self.metadata.copy(),
        )

    def slice_days(self, start: int, end: int) -> Trajectory:
        """Return a sub-trajectory for days in ``[start, end)``.

        Parameters
        ----------
        start:
            First day to include (0-indexed).
        end:
            Exclusive upper bound.

        Returns
        -------
        Trajectory
        """
        mask = (self.days >= start) & (self.days < end)
        return Trajectory(
            species=self.species[mask],
            controls=self.controls[mask],
            volume_L=self.volume_L[mask],
            days=self.days[mask],
            species_names=self.species_names,
            control_names=self.control_names,
            run_id=self.run_id,
            clone_id=self.clone_id,
            reactor_id=self.reactor_id,
            metadata=self.metadata.copy(),
        )

    def species_index(self, name: str) -> int:
        """Return the column index for *name* in the species tensor.

        Parameters
        ----------
        name:
            Species name, must be in :attr:`species_names`.

        Returns
        -------
        int

        Raises
        ------
        KeyError
            If *name* is not a registered species.
        """
        try:
            return self.species_names.index(name)
        except ValueError as exc:
            msg = f"Species '{name}' not found in trajectory. " f"Available: {self.species_names}"
            raise KeyError(msg) from exc

    def get_species(self, name: str) -> Tensor:
        """Return a 1-D tensor of values for species *name*.

        Parameters
        ----------
        name:
            Species name.

        Returns
        -------
        Tensor
            Shape ``(T,)``.
        """
        return self.species[:, self.species_index(name)]

    def valid_mask(self) -> Tensor:
        """Boolean mask ``(T, n_species)`` that is ``True`` where measurement is present.

        Returns
        -------
        Tensor
            dtype ``bool``, shape ``(T, n_species)``.
        """
        return ~torch.isnan(self.species)

    def to_input_tensor(self) -> Tensor:
        """Stack species + controls + day into ``(T, d)`` tensor.

        Returns
        -------
        Tensor
            Shape ``(T, n_species + n_controls + 1)``.
        """
        day_col = self.days.unsqueeze(1).to(dtype=self.species.dtype)
        return torch.cat([self.species, self.controls, day_col], dim=-1)
