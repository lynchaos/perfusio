"""Training routines for the hybrid SW-GP model.

Provides:

- :func:`train_hybrid`: Main training entry point.  Fits the GP residual on
  pre-computed mechanistic residuals, then optionally runs jackknife ensemble.
- :func:`retrain_online`: Lightweight incremental update when new daily data
  arrive (fast re-optimisation of GP hyperparameters only).

Optimisation strategy
---------------------
1. **L-BFGS** (primary): batch optimiser, fast for small datasets (≤ 200 pts).
2. **Adam fallback**: used when L-BFGS diverges (detected by NaN/Inf loss).

References
----------
.. [Gadiyar2026] Gadiyar et al. (2026), §3.3 (model retrain cadence).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, cast

import gpytorch
import torch
from torch import Tensor
from torch.optim import LBFGS, Adam

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


def train_hybrid(
    train_x: Tensor,
    train_y: Tensor,
    model: gpytorch.models.ExactGP,
    likelihood: gpytorch.likelihoods.Likelihood,
    n_iter_lbfgs: int = 50,
    n_iter_adam: int = 200,
    lr_adam: float = 0.05,
) -> None:
    """Fit the hybrid GP residual model by maximising the MLL.

    Attempts L-BFGS first; falls back to Adam if L-BFGS produces NaN loss.

    Parameters
    ----------
    train_x:
        Training inputs, shape ``(N, d)``.
    train_y:
        Residual targets (observed − mechanistic), shape ``(N,)`` or
        ``(N, n_tasks)``.
    model:
        GPyTorch ExactGP model.
    likelihood:
        Corresponding GPyTorch likelihood.
    n_iter_lbfgs:
        Number of L-BFGS steps.
    n_iter_adam:
        Number of Adam fallback steps.
    lr_adam:
        Adam learning rate.
    """
    model.train()
    likelihood.train()
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    # --- Attempt L-BFGS ---
    all_params = list(model.parameters()) + list(likelihood.parameters())
    opt_lbfgs = LBFGS(all_params, lr=0.1, max_iter=n_iter_lbfgs, line_search_fn="strong_wolfe")
    lbfgs_ok = True

    def closure() -> Tensor:
        opt_lbfgs.zero_grad()
        output = model(train_x)
        loss = -cast(Tensor, mll(output, train_y))
        loss.backward()  # type: ignore[no-untyped-call]
        return loss

    try:
        final_loss = opt_lbfgs.step(closure)  # type: ignore[no-untyped-call]
        if torch.isnan(final_loss) or torch.isinf(final_loss):
            lbfgs_ok = False
            logger.warning("L-BFGS produced NaN/Inf loss; falling back to Adam.")
    except Exception as exc:
        lbfgs_ok = False
        logger.warning("L-BFGS failed (%s); falling back to Adam.", exc)

    if lbfgs_ok:
        return

    # --- Adam fallback ---
    opt_adam = Adam(all_params, lr=lr_adam)
    for step in range(n_iter_adam):
        opt_adam.zero_grad()
        output = model(train_x)
        loss = -cast(Tensor, mll(output, train_y))
        loss.backward()  # type: ignore[no-untyped-call]
        opt_adam.step()
        if step % 50 == 0:
            logger.debug("Adam step %d | loss=%.4f", step, loss.item())


def retrain_online(
    new_x: Tensor,
    new_y: Tensor,
    model: gpytorch.models.ExactGP,
    likelihood: gpytorch.likelihoods.Likelihood,
    n_iter: int = 50,
    lr: float = 0.05,
) -> None:
    """Incremental re-optimisation when new daily data arrive.

    Uses Adam only (L-BFGS is overkill for a single new data point).
    The existing GP training data are *not* replaced; the model is updated
    in-place by calling ``set_train_data`` to append the new observations.

    Parameters
    ----------
    new_x:
        New inputs, shape ``(M, d)``.
    new_y:
        New targets, shape ``(M,)`` or ``(M, n_tasks)``.
    model:
        Model to update in-place.
    likelihood:
        Corresponding likelihood.
    n_iter:
        Number of Adam steps.
    lr:
        Learning rate.

    Notes
    -----
    ``strict=False`` is passed to ``set_train_data`` to allow appending
    to a different-length dataset.
    """
    # Append new observations to the training set
    assert model.train_inputs is not None
    old_x = cast(Tensor, model.train_inputs[0])
    old_y = cast(Tensor, model.train_targets)
    augmented_x = torch.cat([old_x, new_x], dim=0)
    augmented_y = torch.cat([old_y, new_y], dim=0)
    model.set_train_data(inputs=augmented_x, targets=augmented_y, strict=False)

    # Fine-tune hyperparameters
    model.train()
    likelihood.train()
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    opt = Adam(list(model.parameters()) + list(likelihood.parameters()), lr=lr)

    for _ in range(n_iter):
        opt.zero_grad()
        output = model(augmented_x)
        loss = -cast(Tensor, mll(output, augmented_y))
        loss.backward()  # type: ignore[no-untyped-call]
        opt.step()
