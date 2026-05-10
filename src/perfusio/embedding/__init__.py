"""Entity-embedding transfer-learning sub-package for ``perfusio``.

Implements the cross-clone transfer approach of Hutter et al. (2021):
each CHO cell line is represented by a learned dense vector (entity embedding)
that is concatenated with the process state before entering the GP kernel.

Public API
----------
- :class:`~perfusio.embedding.clones.CloneRegistry`
- :class:`~perfusio.embedding.clones.EntityEmbedding`
- :class:`~perfusio.embedding.transfer.TransferLearner`

References
----------
.. [Hutter2021] Hutter, S., et al. (2021). Leveraging prior knowledge with
   transfer learning in biological process model identification.
   Biotechnology and Bioengineering, 118(12), 4759–4775.
"""

from perfusio.embedding.clones import CloneRegistry, EntityEmbedding
from perfusio.embedding.transfer import TransferLearner

__all__ = [
    "CloneRegistry",
    "EntityEmbedding",
    "TransferLearner",
]
