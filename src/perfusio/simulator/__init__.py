"""In-silico simulator sub-package for ``perfusio``.

Provides a fully self-contained virtual bioreactor environment for testing the
self-driving methodology without physical hardware.

Public API
----------
- :class:`~perfusio.simulator.cho_perfusion.CHOSimulator`
- :func:`~perfusio.simulator.doe.box_behnken`
- :func:`~perfusio.simulator.doe.central_composite`
- :func:`~perfusio.simulator.doe.latin_hypercube`
- :class:`~perfusio.simulator.noise.NoiseModel`
"""

from perfusio.simulator.cho_perfusion import CHOSimulator
from perfusio.simulator.doe import box_behnken, central_composite, latin_hypercube
from perfusio.simulator.noise import NoiseModel

__all__ = [
    "CHOSimulator",
    "box_behnken",
    "central_composite",
    "latin_hypercube",
    "NoiseModel",
]
