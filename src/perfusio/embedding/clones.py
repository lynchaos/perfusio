"""Clone registry and entity-embedding module.

Each CHO cell line (clone) is assigned a learnable dense embedding vector that
captures clone-specific characteristics (growth rate, metabolism, productivity)
in a low-dimensional latent space.  The embedding is learned jointly with the
GP hyperparameters during MLL training and fine-tuned during transfer.

Architecture
------------
::

    clone_id  →  Embedding(n_clones, embed_dim)  →  e ∈ ℝ^{embed_dim}
    e is concatenated with (c, u, t) → GP input

Following Hutter et al. (2021) §2.3, the default embedding dimension is 4,
which was found to balance expressiveness and identifiability for ≤ 20 clones.

References
----------
.. [Hutter2021] Hutter et al. (2021), §2.3.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor


@dataclass
class CloneInfo:
    """Metadata for a single CHO cell line."""

    clone_id: int          # integer index (0-based)
    name: str              # e.g. "CloneX", "CloneY"
    consumes_lactate: bool = True  # True = Warburg switch (clone X)
    description: str = ""


class CloneRegistry:
    """Registry that maps clone names to integer IDs and metadata.

    Parameters
    ----------
    clones:
        List of :class:`CloneInfo` objects.  The ``clone_id`` field must form
        a contiguous range starting at 0.

    Examples
    --------
    >>> from perfusio.embedding import CloneRegistry
    >>> reg = CloneRegistry.default()
    >>> reg["CloneX"].clone_id
    0
    >>> len(reg)
    2
    """

    def __init__(self, clones: list[CloneInfo]) -> None:
        self._by_name: dict[str, CloneInfo] = {c.name: c for c in clones}
        self._by_id: dict[int, CloneInfo] = {c.clone_id: c for c in clones}

    @classmethod
    def default(cls) -> "CloneRegistry":
        """Return the default two-clone registry (Clone X and Clone Y).

        Clone X switches from lactate production to consumption at low glucose
        (Warburg reversal).  Clone Y does not.

        Returns
        -------
        CloneRegistry
        """
        return cls(
            [
                CloneInfo(
                    clone_id=0,
                    name="CloneX",
                    consumes_lactate=True,
                    description="CHO clone X — Warburg switch below 2 g/L glucose",
                ),
                CloneInfo(
                    clone_id=1,
                    name="CloneY",
                    consumes_lactate=False,
                    description="CHO clone Y — no Warburg switch",
                ),
            ]
        )

    def __getitem__(self, key: str | int) -> CloneInfo:
        if isinstance(key, str):
            return self._by_name[key]
        return self._by_id[key]

    def __len__(self) -> int:
        return len(self._by_name)

    @property
    def n_clones(self) -> int:
        """Number of registered clones."""
        return len(self._by_name)

    def clone_id(self, name: str) -> int:
        """Return the integer ID for a clone by name."""
        return self._by_name[name].clone_id


class EntityEmbedding(nn.Module):
    """Learnable dense entity embedding for CHO cell lines.

    Parameters
    ----------
    n_clones:
        Number of distinct clones (vocabulary size).
    embed_dim:
        Embedding dimensionality.  Default 4 (Hutter et al. 2021, §2.3).

    Examples
    --------
    >>> emb = EntityEmbedding(n_clones=2, embed_dim=4)
    >>> clone_ids = torch.tensor([0, 1, 0])  # batch of three samples
    >>> e = emb(clone_ids)
    >>> e.shape
    torch.Size([3, 4])
    """

    def __init__(self, n_clones: int, embed_dim: int = 4) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.embedding = nn.Embedding(n_clones, embed_dim)
        # Initialise near zero for stable warm-start
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.01)

    def forward(self, clone_ids: Tensor) -> Tensor:
        """Look up embeddings for a batch of clone IDs.

        Parameters
        ----------
        clone_ids:
            Integer tensor of clone IDs, shape ``(N,)`` or ``(B,)``.

        Returns
        -------
        Tensor
            Embedding matrix, shape ``(N, embed_dim)``.
        """
        return self.embedding(clone_ids)

    def embed_and_concat(self, x: Tensor, clone_ids: Tensor) -> Tensor:
        """Concatenate state features with clone embeddings.

        Parameters
        ----------
        x:
            Feature tensor, shape ``(N, d)``.
        clone_ids:
            Clone ID tensor, shape ``(N,)``.

        Returns
        -------
        Tensor
            Augmented features, shape ``(N, d + embed_dim)``.
        """
        e = self.forward(clone_ids)
        return torch.cat([x, e], dim=-1)
