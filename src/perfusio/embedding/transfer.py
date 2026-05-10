"""Transfer learning orchestration for entity-embedded GP models.

Implements the two-stage transfer protocol of Hutter et al. (2021):

1. **Warm start** (``warm_start``): Train on the *source* clone dataset.
   The :class:`~perfusio.embedding.clones.EntityEmbedding` is initialised and
   the full model (embedding + GP kernel) is optimised.

2. **Joint fine-tuning** (``joint_finetune``): Load weights from the source
   run, optionally freeze the kernel backbone, and fine-tune on a small
   *target* clone dataset with a higher learning rate on the embedding.

This two-stage approach allows the model to learn shared process dynamics
from data-rich source clones and rapidly adapt to a new clone with as few as
3–5 runs (Hutter et al. §3.2).

References
----------
.. [Hutter2021] Hutter et al. (2021), §2.3 and §3.2.
"""

from __future__ import annotations

from pathlib import Path

import torch
import gpytorch
from torch import Tensor

from perfusio.embedding.clones import EntityEmbedding
from perfusio.gp.ensemble import _train_gp


class TransferLearner:
    """Two-stage transfer learning manager.

    Parameters
    ----------
    embedding:
        Shared :class:`EntityEmbedding` module.
    model:
        GPyTorch model (must accept augmented inputs from the embedding).
    likelihood:
        Corresponding GPyTorch likelihood.
    lr_backbone:
        Learning rate for GP kernel hyperparameters during fine-tuning.
    lr_embedding:
        Learning rate for embedding parameters during fine-tuning.
        Typically 10× the backbone lr.

    Examples
    --------
    >>> from perfusio.embedding import TransferLearner, EntityEmbedding
    >>> emb = EntityEmbedding(n_clones=2, embed_dim=4)
    >>> # attach model and likelihood (constructed externally)
    >>> # tl = TransferLearner(emb, model, likelihood)
    """

    def __init__(
        self,
        embedding: EntityEmbedding,
        model: gpytorch.models.ExactGP,
        likelihood: gpytorch.likelihoods.Likelihood,
        lr_backbone: float = 0.01,
        lr_embedding: float = 0.10,
    ) -> None:
        self.embedding = embedding
        self.model = model
        self.likelihood = likelihood
        self.lr_backbone = lr_backbone
        self.lr_embedding = lr_embedding

    def warm_start(
        self,
        train_x: Tensor,
        train_y: Tensor,
        clone_ids: Tensor,
        n_iter: int = 300,
    ) -> None:
        """Train the full model on source-clone data.

        Parameters
        ----------
        train_x:
            Source features (without embedding), shape ``(N, d_raw)``.
        train_y:
            Targets, shape ``(N,)`` or ``(N, n_tasks)``.
        clone_ids:
            Integer clone IDs, shape ``(N,)``.
        n_iter:
            Number of optimisation steps.
        """
        x_aug = self.embedding.embed_and_concat(train_x, clone_ids)
        _train_gp(
            self.model,
            self.likelihood,
            x_aug,
            train_y,
            n_iter=n_iter,
            lr=self.lr_backbone,
        )

    def joint_finetune(
        self,
        train_x: Tensor,
        train_y: Tensor,
        clone_ids: Tensor,
        n_iter: int = 100,
        freeze_backbone: bool = False,
    ) -> None:
        """Fine-tune on target-clone data with differential learning rates.

        Parameters
        ----------
        train_x:
            Target features (without embedding), shape ``(M, d_raw)``.
            ``M`` may be as small as the number of experiments × time points.
        train_y:
            Targets, shape ``(M,)`` or ``(M, n_tasks)``.
        clone_ids:
            Clone IDs for the target data, shape ``(M,)``.
        n_iter:
            Number of fine-tuning steps.
        freeze_backbone:
            If ``True``, freeze all GP kernel/mean parameters and update only
            the embedding.  Useful when the target dataset is very small (< 3
            runs) to avoid catastrophic forgetting.
        """
        x_aug = self.embedding.embed_and_concat(train_x, clone_ids)

        # Build param groups with different learning rates
        embed_params = list(self.embedding.parameters())
        embed_ids = {id(p) for p in embed_params}

        backbone_params = [
            p for p in list(self.model.parameters()) + list(self.likelihood.parameters())
            if id(p) not in embed_ids
        ]

        if freeze_backbone:
            for p in backbone_params:
                p.requires_grad_(False)

        optimizer = torch.optim.Adam(
            [
                {"params": embed_params, "lr": self.lr_embedding},
                {"params": backbone_params, "lr": self.lr_backbone},
            ]
        )
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)

        self.model.train()
        self.likelihood.train()
        for _ in range(n_iter):
            optimizer.zero_grad()
            output = self.model(x_aug)
            loss = -mll(output, train_y)
            loss.backward()
            optimizer.step()

        # Unfreeze backbone if it was frozen
        if freeze_backbone:
            for p in backbone_params:
                p.requires_grad_(True)

    def save(self, path: str | Path) -> None:
        """Save embedding and model state to disk.

        Parameters
        ----------
        path:
            File path.  Saves a dict with ``"embedding"``, ``"model"``,
            ``"likelihood"`` state dicts.
        """
        torch.save(
            {
                "embedding": self.embedding.state_dict(),
                "model": self.model.state_dict(),
                "likelihood": self.likelihood.state_dict(),
            },
            path,
        )

    def load(self, path: str | Path) -> None:
        """Load state from disk (must match the architecture).

        Parameters
        ----------
        path:
            Path to checkpoint saved by :meth:`save`.
        """
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        self.embedding.load_state_dict(ckpt["embedding"])
        self.model.load_state_dict(ckpt["model"])
        self.likelihood.load_state_dict(ckpt["likelihood"])
