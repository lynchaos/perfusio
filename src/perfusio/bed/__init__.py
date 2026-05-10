"""Bayesian Experimental Design (BED) sub-package for ``perfusio``.

Implements the full BED decision loop described in Gadiyar et al. (2026) §3.2,
including:

- Objective function values (OFV) — target-tracking and multi-objective.
- All 11 BoTorch acquisition functions (PI, EI, LogEI, UCB, qEI, qLogEI,
  qUCB, qEHVI, qNEHVI, qNParEGO) plus constrained variants.
- Acquisition optimisation over the design space.
- Pareto front computation and hypervolume indicator.
- High-level :class:`BEDPolicy` daily decision loop.

Public API
----------
- :class:`~perfusio.bed.objectives.TargetTrackingOFV`
- :class:`~perfusio.bed.objectives.MultiObjectiveOFV`
- :func:`~perfusio.bed.acquisitions.build_acquisition`
- :func:`~perfusio.bed.search.optimise_acquisition`
- :func:`~perfusio.bed.pareto.compute_pareto_front`
- :class:`~perfusio.bed.policies.BEDPolicy`
"""

from perfusio.bed.acquisitions import build_acquisition
from perfusio.bed.objectives import MultiObjectiveOFV, TargetTrackingOFV
from perfusio.bed.pareto import compute_pareto_front, hypervolume
from perfusio.bed.policies import BEDPolicy
from perfusio.bed.search import optimise_acquisition

__all__ = [
    "BEDPolicy",
    "MultiObjectiveOFV",
    "TargetTrackingOFV",
    "build_acquisition",
    "compute_pareto_front",
    "hypervolume",
    "optimise_acquisition",
]
