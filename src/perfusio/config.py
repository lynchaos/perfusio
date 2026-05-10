"""Pydantic-v2 configuration models for ``perfusio``.

This module provides validated, strongly-typed configuration objects used
throughout the library. All physical bounds are stored here so that they
serve as a single source of truth for the design space.

Classes
-------
SpeciesBounds
    Lower and upper measurement bounds for a single process variable.
DesignSpace
    Full description of the operating envelope for a perfusion experiment,
    including control variable ranges and target criteria.
RunConfig
    Runtime configuration (device, dtype, seeds, logging level).
AlarmConfig
    Thresholds that trigger predictive constraint-violation notifications.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import torch
from pydantic import BaseModel, Field, model_validator

from perfusio._typing import DEFAULT_DTYPE


class SpeciesBounds(BaseModel):
    """Lower and upper physical bounds for a single measured species.

    Parameters
    ----------
    lo:
        Minimum physically plausible value (e.g. 0.0 for VCD).
    hi:
        Maximum physically plausible value.
    unit:
        SI or derived unit string, e.g. ``"10^6 cells mL^-1"``.
    description:
        Human-readable name, used in axis labels.

    Examples
    --------
    >>> b = SpeciesBounds(lo=0.0, hi=100.0, unit="10^6 cells mL^-1",
    ...                   description="Viable Cell Density")
    >>> b.lo
    0.0
    """

    lo: float = Field(..., description="Lower bound.")
    hi: float = Field(..., gt=0, description="Upper bound (must be > 0).")
    unit: str = Field(..., description="Physical unit.")
    description: str = Field(default="", description="Human-readable label.")

    @model_validator(mode="after")
    def _lo_lt_hi(self) -> SpeciesBounds:
        if self.lo >= self.hi:
            msg = f"lo ({self.lo}) must be strictly less than hi ({self.hi})."
            raise ValueError(msg)
        return self


class ControlBounds(BaseModel):
    """Bounds on a single manipulated (control) variable.

    Parameters
    ----------
    lo:
        Minimum allowable setpoint.
    hi:
        Maximum allowable setpoint.
    unit:
        Physical unit of the control variable.
    description:
        Human-readable name.
    hard_lower:
        If ``True``, the optimizer will never suggest values below *lo*.
        Used to encode cell-death constraints (e.g. stir < 700 rpm kills cells).
    hard_upper:
        Analogous upper hard constraint.

    Notes
    -----
    Hard constraints are encoded as ``bounds`` passed to
    :func:`botorch.optim.optimize_acqf`, *not* as penalty terms in the
    acquisition function. Penalty-based approaches violate the BoTorch API
    and can produce numerical artefacts.
    """

    lo: float = Field(..., description="Lower bound.")
    hi: float = Field(..., description="Upper bound.")
    unit: str = Field(..., description="Physical unit.")
    description: str = Field(default="", description="Human-readable label.")
    hard_lower: bool = Field(default=True)
    hard_upper: bool = Field(default=True)

    @model_validator(mode="after")
    def _lo_lt_hi(self) -> ControlBounds:
        if self.lo >= self.hi:
            msg = f"lo ({self.lo}) must be strictly less than hi ({self.hi})."
            raise ValueError(msg)
        return self


class DesignSpace(BaseModel):
    """Full description of the operating envelope for a perfusion experiment.

    This object encodes both the measurable state bounds and the control
    variable ranges that are passed to the Bayesian Experimental Design
    optimizer. It is the single source of truth for all ``optimize_acqf``
    calls.

    Parameters
    ----------
    name:
        Experiment name (used in audit logs and figure titles).
    control_bounds:
        Mapping of control variable name → :class:`ControlBounds`.
    species_bounds:
        Mapping of species name → :class:`SpeciesBounds`.
    viability_min:
        Minimum acceptable viability (%) for feasibility constraints.
        Default 95.0 matches the paper's constraint.
    vcv_target:
        Target volumetric cell volume (%) for the single-objective
        VCV-tracking use case (Gadiyar et al. 2026, §3.5).
    titer_target:
        Optional target product concentration [mg L⁻¹].

    Examples
    --------
    >>> from perfusio.config import DesignSpace, ControlBounds, SpeciesBounds
    >>> ds = DesignSpace(
    ...     name="ambr250_run_01",
    ...     control_bounds={
    ...         "perfusion_rate": ControlBounds(lo=0.5, hi=2.0, unit="vvd"),
    ...         "bleed_rate": ControlBounds(lo=0.05, hi=0.30, unit="vvd"),
    ...     },
    ...     species_bounds={
    ...         "VCD": SpeciesBounds(lo=0.0, hi=120.0, unit="1e6 cells/mL",
    ...                              description="Viable Cell Density"),
    ...     },
    ... )
    """

    name: str = Field(..., description="Experiment or run name.")
    control_bounds: dict[str, ControlBounds] = Field(
        default_factory=dict,
        description="Control variable name → bounds.",
    )
    species_bounds: dict[str, SpeciesBounds] = Field(
        default_factory=dict,
        description="Species name → measurement bounds.",
    )
    viability_min: float = Field(
        default=95.0,
        ge=0.0,
        le=100.0,
        description="Minimum viable cell viability [%] for feasibility constraints.",
    )
    vcv_target: float | None = Field(
        default=None,
        ge=0.0,
        le=100.0,
        description="VCV setpoint [%] for the single-objective tracking objective.",
    )
    titer_target: float | None = Field(
        default=None,
        ge=0.0,
        description="mAb titer target [mg L^-1].",
    )

    @property
    def n_controls(self) -> int:
        """Number of control variables."""
        return len(self.control_bounds)

    @property
    def bounds_tensor(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (lower, upper) tensors of shape (n_controls,) for BoTorch.

        Returns
        -------
        tuple[Tensor, Tensor]
            ``lower`` and ``upper`` bound tensors, dtype float64.
        """
        keys = list(self.control_bounds.keys())
        lo = torch.tensor([self.control_bounds[k].lo for k in keys], dtype=torch.float64)
        hi = torch.tensor([self.control_bounds[k].hi for k in keys], dtype=torch.float64)
        return lo, hi

    @property
    def control_names(self) -> list[str]:
        """Ordered list of control variable names."""
        return list(self.control_bounds.keys())

    @property
    def species_names(self) -> list[str]:
        """Ordered list of species names."""
        return list(self.species_bounds.keys())


class RunConfig(BaseModel):
    """Runtime configuration for ``perfusio`` computations.

    Parameters
    ----------
    seed:
        Global random seed. ``None`` for non-deterministic execution.
    device:
        Torch device string (``"cpu"``, ``"cuda:0"``, …). Defaults to ``"cpu"``
        to ensure CI compatibility without GPUs.
    dtype:
        Torch dtype. Defaults to ``torch.float64`` (required for numerical
        stability of GP marginal log-likelihood optimisation — never downcast).
    n_mc_samples:
        Number of Monte Carlo samples for uncertainty propagation in rollout.
        Default 256 (adequate for 80% PI; increase to 1024 for publication).
    n_restarts:
        Number of random restarts for acquisition function optimisation.
        Default 10.
    raw_samples:
        Initial Sobol samples for warm-starting acquisition optimisation.
        Default 512.
    log_level:
        ``structlog`` log level. One of ``"DEBUG"``, ``"INFO"``, ``"WARNING"``,
        ``"ERROR"``.
    allow_write:
        If ``False`` (default), the OPC UA connector operates in read-only
        mode and will raise ``PermissionError`` on any ``write_setpoints``
        call. Must be explicitly set to ``True`` to enable hardware writes.

    Notes
    -----
    ``dtype = torch.float64`` is a hard requirement for the GP marginal
    log-likelihood optimisation. The LBFGS optimiser with strong-Wolfe line
    search becomes numerically unstable in float32 for datasets with more
    than ~100 training points.
    """

    seed: int | None = Field(default=None, description="Global random seed.")
    device: str = Field(default="cpu", description="Torch device.")
    dtype: Literal["float32", "float64"] = Field(
        default="float64",
        description="Torch dtype. Must be float64 for GP stability.",
    )
    n_mc_samples: int = Field(default=256, ge=16, description="MC paths for rollout.")
    n_restarts: int = Field(default=10, ge=1, description="Acqf optimisation restarts.")
    raw_samples: int = Field(default=512, ge=64, description="Sobol samples for acqf warm-start.")
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(default="INFO")
    allow_write: bool = Field(
        default=False,
        description="Allow writing setpoints to physical connectors.",
    )

    @property
    def torch_device(self) -> torch.device:
        """Parsed :class:`torch.device`."""
        return torch.device(self.device)

    @property
    def torch_dtype(self) -> torch.dtype:
        """Parsed :class:`torch.dtype`."""
        return DEFAULT_DTYPE if self.dtype == "float64" else torch.float32

    def make_generator(self) -> torch.Generator | None:
        """Create a seeded :class:`torch.Generator`, or ``None`` if seed is ``None``.

        Returns
        -------
        torch.Generator or None
        """
        if self.seed is None:
            return None
        gen = torch.Generator(device=self.torch_device)
        gen.manual_seed(self.seed)
        return gen


class AlarmConfig(BaseModel):
    """Thresholds that trigger predictive constraint-violation notifications.

    Parameters
    ----------
    viability_warning:
        Predicted viability [%] below which a WARNING alarm is raised.
    viability_critical:
        Predicted viability [%] below which a CRITICAL alarm is raised.
    ammonium_warning:
        Predicted ammonium [mmol L⁻¹] above which a WARNING is raised.
    ammonium_critical:
        Critical ammonium threshold.
    glucose_warning:
        Predicted glucose [g L⁻¹] below which a WARNING is raised (impending
        starvation).
    glucose_critical:
        Critical glucose starvation threshold.
    notification_channel:
        Where to send alarms. Supported: ``"log"`` (structlog), ``"file"``.
        ``"email"`` and ``"slack"`` require environment variable configuration.
    alarm_file:
        Path to append alarm records when ``notification_channel == "file"``.

    Notes
    -----
    This class deliberately *does not* support email or Slack credentials
    inline. Provide them via environment variables ``PERFUSIO_SMTP_*`` or
    ``PERFUSIO_SLACK_WEBHOOK_URL``, respectively.
    """

    viability_warning: float = Field(default=97.0, ge=0.0, le=100.0)
    viability_critical: float = Field(default=95.0, ge=0.0, le=100.0)
    ammonium_warning: float = Field(default=8.0, ge=0.0)
    ammonium_critical: float = Field(default=12.0, ge=0.0)
    glucose_warning: float = Field(default=1.0, ge=0.0)
    glucose_critical: float = Field(default=0.3, ge=0.0)
    notification_channel: Literal["log", "file", "email", "slack"] = Field(default="log")
    alarm_file: Path | None = Field(default=None)

    @model_validator(mode="after")
    def _critical_below_warning(self) -> AlarmConfig:
        if self.viability_critical > self.viability_warning:
            msg = "viability_critical must be ≤ viability_warning."
            raise ValueError(msg)
        if self.ammonium_critical < self.ammonium_warning:
            msg = "ammonium_critical must be ≥ ammonium_warning."
            raise ValueError(msg)
        if self.glucose_critical > self.glucose_warning:
            msg = "glucose_critical must be ≤ glucose_warning."
            raise ValueError(msg)
        return self


# ---------------------------------------------------------------------------
# Default design space for the paper's ambr250 training experiment
# (Table 1 of Gadiyar et al. 2026)
# ---------------------------------------------------------------------------

DEFAULT_AMBR250_DESIGN_SPACE = DesignSpace(
    name="ambr250_default",
    control_bounds={
        "perfusion_rate": ControlBounds(
            lo=0.5,
            hi=2.0,
            unit="vvd",
            description="Perfusion rate [vessel volumes per day]",
        ),
        "bleed_rate": ControlBounds(
            lo=0.05,
            hi=0.30,
            unit="vvd",
            description="Bleed rate [vessel volumes per day]",
        ),
        "glucose_setpoint": ControlBounds(
            lo=2.0,
            hi=8.0,
            unit="g/L",
            description="Glucose feed setpoint",
        ),
        "temperature": ControlBounds(
            lo=35.0,
            hi=37.5,
            unit="°C",
            description="Culture temperature",
        ),
        "agitation": ControlBounds(
            lo=700.0,
            hi=1050.0,
            unit="rpm",
            description="Agitation speed (> 1050 rpm causes cell damage)",
            hard_upper=True,
        ),
        "pyruvate_feed": ControlBounds(
            lo=0.0,
            hi=5.0,
            unit="mmol/L",
            description="Pyruvate feed concentration for NH₄⁺ scavenging",
        ),
    },
    species_bounds={
        "VCD": SpeciesBounds(
            lo=0.0,
            hi=120.0,
            unit="10^6 cells/mL",
            description="Viable Cell Density",
        ),
        "VCV": SpeciesBounds(
            lo=0.0,
            hi=60.0,
            unit="%",
            description="Viable Cell Volume",
        ),
        "Via": SpeciesBounds(
            lo=0.0,
            hi=100.0,
            unit="%",
            description="Cell Viability",
        ),
        "Diam": SpeciesBounds(
            lo=14.0,
            hi=26.0,
            unit="μm",
            description="Mean Cell Diameter",
        ),
        "Glc": SpeciesBounds(
            lo=0.0,
            hi=20.0,
            unit="g/L",
            description="Glucose concentration",
        ),
        "Gln": SpeciesBounds(
            lo=0.0,
            hi=10.0,
            unit="mmol/L",
            description="Glutamine concentration",
        ),
        "Glu": SpeciesBounds(
            lo=0.0,
            hi=8.0,
            unit="mmol/L",
            description="Glutamate concentration",
        ),
        "Lac": SpeciesBounds(
            lo=0.0,
            hi=40.0,
            unit="mmol/L",
            description="Lactate concentration",
        ),
        "Amm": SpeciesBounds(
            lo=0.0,
            hi=25.0,
            unit="mmol/L",
            description="Ammonium concentration",
        ),
        "Pyr": SpeciesBounds(
            lo=0.0,
            hi=10.0,
            unit="mmol/L",
            description="Pyruvate concentration",
        ),
        "Titer": SpeciesBounds(
            lo=0.0,
            hi=5000.0,
            unit="mg/L",
            description="mAb product titer",
        ),
    },
    viability_min=95.0,
    vcv_target=30.0,
)
