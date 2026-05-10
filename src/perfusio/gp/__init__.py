"""Gaussian Process sub-package for ``perfusio``.

Public API
----------
- :class:`~perfusio.gp.kernels.PerfusionKernel`
- :class:`~perfusio.gp.means.MechanisticPriorMean`
- :class:`~perfusio.gp.exact_gp.MultiTaskRateGP`
- :class:`~perfusio.gp.ensemble.JackknifeEnsemble`
- :class:`~perfusio.gp.stepwise.StepwiseGP`
"""

from perfusio.gp.ensemble import JackknifeEnsemble
from perfusio.gp.exact_gp import MultiTaskRateGP
from perfusio.gp.kernels import PerfusionKernel
from perfusio.gp.means import MechanisticPriorMean, ZeroMeanMultiTask
from perfusio.gp.stepwise import StepwiseGP

__all__ = [
    "JackknifeEnsemble",
    "MultiTaskRateGP",
    "PerfusionKernel",
    "MechanisticPriorMean",
    "ZeroMeanMultiTask",
    "StepwiseGP",
]
