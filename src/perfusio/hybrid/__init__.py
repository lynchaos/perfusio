"""Hybrid state-space model sub-package for ``perfusio``.

The hybrid model combines a deterministic mechanistic prior (CHO ODE) with a
Gaussian Process correction term, following Gadiyar et al. (2026) §3.1:

.. math::
    R_k(\\mathbf{x}) = \\hat{R}^{\\text{mech}}_k(\\mathbf{x}) + \\epsilon_k(\\mathbf{x})

where :math:`\\epsilon_k \\sim \\mathcal{GP}(0, k_\\epsilon)`.

Public API
----------
- :class:`~perfusio.hybrid.model.HybridStateSpaceModel`
- :func:`~perfusio.hybrid.train.train_hybrid`
- :func:`~perfusio.hybrid.forecast.forecast_run`
"""

from perfusio.hybrid.forecast import forecast_run
from perfusio.hybrid.model import HybridStateSpaceModel
from perfusio.hybrid.train import train_hybrid

__all__ = [
    "HybridStateSpaceModel",
    "train_hybrid",
    "forecast_run",
]
