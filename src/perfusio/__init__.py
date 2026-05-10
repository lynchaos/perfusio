"""perfusio — open reference implementation of self-driving perfusion bioprocess development.

``perfusio`` implements the methodology of Gadiyar et al. (2026) and Hutter et al. (2021),
providing step-wise Gaussian-process hybrid models, entity-embedding transfer learning,
Bayesian Experimental Design, and an online-retraining digital twin for CHO perfusion
bioreactors. It is intended as an open, citable complement to commercial platforms.

References
----------
.. [Gadiyar2026] Gadiyar, C. J., Müller, C., Vuillemin, T., Bielser, J.-M., Souquet, J.,
   Fagnani, A., Sokolov, M., von Stosch, M., Feidl, F., Butté, A., &
   Cruz Bournazou, M. N. (2026). Self-Driving Development of Perfusion Processes
   for Monoclonal Antibody Production. *Biotechnology and Bioengineering*, 123(2),
   391–405. https://doi.org/10.1002/bit.70093

.. [Hutter2021] Hutter, S., von Stosch, M., Cruz Bournazou, M. N., & Butté, A. (2021).
   Knowledge transfer across cell lines using hybrid Gaussian process models with
   entity embedding vectors. *Biotechnology and Bioengineering*, 118(12), 4710–4725.
   https://doi.org/10.1002/bit.27907
"""

from __future__ import annotations

__version__ = "0.1.0"
__author__ = "perfusio Contributors"
__license__ = "Apache-2.0"

from perfusio.config import DesignSpace, RunConfig
from perfusio.states import State, StateBatch, Trajectory

__all__ = [
    "__version__",
    "DesignSpace",
    "RunConfig",
    "State",
    "StateBatch",
    "Trajectory",
]
