"""Mechanistic sub-package for ``perfusio``.

This package provides the first-principles CHO kinetic model that serves as
the mechanistic backbone of the hybrid SW-GP model.  Its primary role is to
provide the :class:`~perfusio.gp.means.MechanisticPriorMean` with rate
predictions that reflect biological knowledge, so that the GP learns only
the *residual* between the mechanistic prediction and reality.

Public API
----------
- :class:`~perfusio.mechanistic.kinetics.CHOKinetics`
- :class:`~perfusio.mechanistic.models.CHOPerfusionModel`
- :func:`~perfusio.mechanistic.integrators.integrate_run`
"""

from perfusio.mechanistic.integrators import integrate_run
from perfusio.mechanistic.kinetics import CHOKinetics
from perfusio.mechanistic.models import CHOPerfusionModel

__all__ = [
    "CHOKinetics",
    "CHOPerfusionModel",
    "integrate_run",
]
