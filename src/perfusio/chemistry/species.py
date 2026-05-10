"""Species registry for perfusion bioreactor models.

This module defines the canonical set of process variables tracked by
``perfusio``. All modules reference species by their canonical short name
(e.g. ``"VCD"``) to ensure consistent ordering in tensors.

Biological context
------------------
The species list covers all variables measured in a typical ambr®250 perfusion
run using a Cedex Bio HT or Nova BioProfile FLEX2 at-line analyser, plus
diameter measured by a Vi-Cell or Cedex HiRes.

References
----------
.. [Gadiyar2026] Gadiyar et al. (2026), §2.1 — "Process variables" table.

Notes
-----
``DCD`` (dead cell density) and ``Lys`` (lysed cell density) are derived from
VCD, total cell count, and viability; they are included as species for GP
modelling but are not independent degrees of freedom.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import ClassVar


@dataclass(frozen=True)
class Species:
    """Metadata for a single bioreactor species.

    Parameters
    ----------
    short_name:
        Canonical identifier used in tensor column names (e.g. ``"VCD"``).
    full_name:
        Human-readable name for axis labels.
    unit:
        Physical unit string for display (e.g. ``"10⁶ cells mL⁻¹"``).
    lo:
        Physically plausible lower bound.
    hi:
        Physically plausible upper bound.
    is_measured:
        Whether this species is directly measured (True) or derived (False).
    is_control:
        Whether this is a state variable that can be indirectly controlled
        via a feed (e.g. Glc via glucose setpoint).

    Examples
    --------
    >>> sp = Species(
    ...     short_name="VCD",
    ...     full_name="Viable Cell Density",
    ...     unit="10⁶ cells mL⁻¹",
    ...     lo=0.0,
    ...     hi=120.0,
    ... )
    >>> sp.short_name
    'VCD'
    """

    short_name: str
    full_name: str
    unit: str
    lo: float
    hi: float
    is_measured: bool = True
    is_control: bool = False


class SpeciesEnum(str, Enum):
    """Enumeration of all canonical species short names."""

    VCD = "VCD"
    VCV = "VCV"
    VIA = "Via"
    DIAM = "Diam"
    GLC = "Glc"
    GLN = "Gln"
    GLU = "Glu"
    LAC = "Lac"
    AMM = "Amm"
    PYR = "Pyr"
    TITER = "Titer"
    DCD = "DCD"
    LYS = "Lys"


class SpeciesRegistry:
    """Singleton registry of all process variables.

    This is the single source of truth for the canonical species list used
    throughout ``perfusio``. Obtain the default registry via
    :attr:`SpeciesRegistry.DEFAULT`.

    Attributes
    ----------
    DEFAULT:
        The library-wide default registry matching the paper's species table.

    Examples
    --------
    >>> reg = SpeciesRegistry.DEFAULT
    >>> reg.names
    ['VCD', 'VCV', 'Via', 'Diam', 'Glc', 'Gln', 'Glu', 'Lac', 'Amm', 'Pyr', 'Titer', 'DCD', 'Lys']
    >>> reg["VCD"].unit
    '10⁶ cells mL⁻¹'
    """

    DEFAULT: ClassVar["SpeciesRegistry"]

    def __init__(self, species: list[Species]) -> None:
        self._by_name: dict[str, Species] = {s.short_name: s for s in species}
        if len(self._by_name) != len(species):
            msg = "Duplicate short_name in species list."
            raise ValueError(msg)

    def __getitem__(self, name: str) -> Species:
        """Retrieve a :class:`Species` by its short name.

        Parameters
        ----------
        name:
            Canonical short name (e.g. ``"VCD"``).

        Raises
        ------
        KeyError
            If the species is not registered.
        """
        if name not in self._by_name:
            msg = f"Species '{name}' not in registry. Available: {self.names}"
            raise KeyError(msg)
        return self._by_name[name]

    def __contains__(self, name: object) -> bool:
        return name in self._by_name

    def __len__(self) -> int:
        return len(self._by_name)

    @property
    def names(self) -> list[str]:
        """Ordered list of all short names."""
        return list(self._by_name.keys())

    @property
    def measured(self) -> list[Species]:
        """Species that are directly measured at-line."""
        return [s for s in self._by_name.values() if s.is_measured]

    @property
    def derived(self) -> list[Species]:
        """Species derived from measured values."""
        return [s for s in self._by_name.values() if not s.is_measured]

    def subset(self, names: list[str]) -> "SpeciesRegistry":
        """Return a sub-registry containing only the listed species.

        Parameters
        ----------
        names:
            Ordered list of short names to include.

        Returns
        -------
        SpeciesRegistry
        Raises
        ------
        KeyError
            If any requested name is not in this registry.
        """
        return SpeciesRegistry([self[n] for n in names])

    def index(self, name: str) -> int:
        """Return the 0-based column index for *name*.

        Parameters
        ----------
        name:
            Short name of the species.

        Returns
        -------
        int
        Raises
        ------
        KeyError
            If *name* is not in this registry.
        """
        return self.names.index(self[name].short_name)


# ---------------------------------------------------------------------------
# Default registry — matches the paper's species table (Gadiyar et al. 2026)
# ---------------------------------------------------------------------------
_DEFAULT_SPECIES: list[Species] = [
    Species(
        short_name="VCD",
        full_name="Viable Cell Density",
        unit="10⁶ cells mL⁻¹",
        lo=0.0,
        hi=120.0,
        is_measured=True,
    ),
    Species(
        short_name="VCV",
        full_name="Viable Cell Volume",
        unit="%",
        lo=0.0,
        hi=60.0,
        is_measured=True,
    ),
    Species(
        short_name="Via",
        full_name="Cell Viability",
        unit="%",
        lo=0.0,
        hi=100.0,
        is_measured=True,
    ),
    Species(
        short_name="Diam",
        full_name="Mean Cell Diameter",
        unit="μm",
        lo=14.0,
        hi=26.0,
        is_measured=True,
    ),
    Species(
        short_name="Glc",
        full_name="Glucose",
        unit="g L⁻¹",
        lo=0.0,
        hi=20.0,
        is_measured=True,
        is_control=True,
    ),
    Species(
        short_name="Gln",
        full_name="Glutamine",
        unit="mmol L⁻¹",
        lo=0.0,
        hi=10.0,
        is_measured=True,
    ),
    Species(
        short_name="Glu",
        full_name="Glutamate",
        unit="mmol L⁻¹",
        lo=0.0,
        hi=8.0,
        is_measured=True,
    ),
    Species(
        short_name="Lac",
        full_name="Lactate",
        unit="mmol L⁻¹",
        lo=0.0,
        hi=40.0,
        is_measured=True,
    ),
    Species(
        short_name="Amm",
        full_name="Ammonium",
        unit="mmol L⁻¹",
        lo=0.0,
        hi=25.0,
        is_measured=True,
    ),
    Species(
        short_name="Pyr",
        full_name="Pyruvate",
        unit="mmol L⁻¹",
        lo=0.0,
        hi=10.0,
        is_measured=True,
        is_control=True,
    ),
    Species(
        short_name="Titer",
        full_name="mAb Product Titer",
        unit="mg L⁻¹",
        lo=0.0,
        hi=5000.0,
        is_measured=True,
    ),
    Species(
        short_name="DCD",
        full_name="Dead Cell Density",
        unit="10⁶ cells mL⁻¹",
        lo=0.0,
        hi=30.0,
        is_measured=False,  # derived: DCD = TCD - VCD
    ),
    Species(
        short_name="Lys",
        full_name="Lysed Cell Density",
        unit="10⁶ cells mL⁻¹",
        lo=0.0,
        hi=15.0,
        is_measured=False,  # derived from cell counting + viability
    ),
]

SpeciesRegistry.DEFAULT = SpeciesRegistry(_DEFAULT_SPECIES)
