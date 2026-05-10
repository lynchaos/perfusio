"""Jackknife ensemble of GP models for uncertainty quantification.

The ensemble trains *K* GP models, each on a random subsample of the training
data (jackknife-after-bootstrap), then aggregates their predictions as
10th / 50th / 90th percentiles.  This provides calibrated predictive intervals
even for small training sets (< 30 observations per species), where the
Normal approximation of a single GP may be overconfident.

The approach is analogous to the jackknife+ of Barber et al. (2021) but
applied at the GP model level rather than at the residual level.

References
----------
.. [Barber2021] Barber, R. F., et al. (2021). Predictive inference with the
   jackknife+. Annals of Statistics, 49(1), 486–507.
.. [Gadiyar2026] Gadiyar et al. (2026), §3.2 (ensemble calibration).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import gpytorch
import torch
from torch import Tensor


@dataclass
class EnsembleMember:
    """Container for a single jackknife ensemble member."""

    model: gpytorch.models.ExactGP
    likelihood: gpytorch.likelihoods.Likelihood
    train_idx: Tensor  # indices used to train this member


class JackknifeEnsemble:
    """Jackknife ensemble over K GP models.

    Parameters
    ----------
    K:
        Number of ensemble members. Default 20 (Gadiyar §3.2).
    subsample_fraction:
        Fraction of training data used per member. Default 0.8.
    seed:
        Random seed for reproducibility.

    Examples
    --------
    >>> from perfusio.gp import JackknifeEnsemble, MultiTaskRateGP
    >>> import gpytorch, torch
    >>> ens = JackknifeEnsemble(K=5)
    """

    def __init__(
        self,
        K: int = 20,
        subsample_fraction: float = 0.80,
        seed: int = 0,
    ) -> None:
        self.K = K
        self.subsample_fraction = subsample_fraction
        self.seed = seed
        self.members: list[EnsembleMember] = []

    def fit(
        self,
        train_x: Tensor,
        train_y: Tensor,
        model_factory: "object",
        n_iter: int = 200,
        lr: float = 0.05,
    ) -> "JackknifeEnsemble":
        """Train all K ensemble members.

        Parameters
        ----------
        train_x:
            Training inputs, shape ``(N, d)``.
        train_y:
            Training targets, shape ``(N, n_tasks)`` or ``(N,)``.
        model_factory:
            Callable ``(x_sub, y_sub) -> (model, likelihood)`` that constructs
            a fresh GP model for a given subsample.
        n_iter:
            MLL training iterations per member.
        lr:
            Adam learning rate.

        Returns
        -------
        JackknifeEnsemble
            Self (for chaining).
        """
        rng = torch.Generator()
        rng.manual_seed(self.seed)

        N = train_x.shape[0]
        sub_n = max(1, int(N * self.subsample_fraction))
        self.members = []

        for k in range(self.K):
            perm = torch.randperm(N, generator=rng)[:sub_n]
            x_sub = train_x[perm]
            y_sub = train_y[perm]

            model, likelihood = model_factory(x_sub, y_sub)  # type: ignore[operator]
            _train_gp(model, likelihood, x_sub, y_sub, n_iter=n_iter, lr=lr)
            self.members.append(EnsembleMember(model=model, likelihood=likelihood, train_idx=perm))

        return self

    def predict(
        self,
        x_new: Tensor,
        quantiles: tuple[float, float, float] = (0.10, 0.50, 0.90),
    ) -> dict[str, Tensor]:
        """Ensemble posterior mean and quantiles.

        Parameters
        ----------
        x_new:
            Test inputs, shape ``(M, d)``.
        quantiles:
            Probability levels. Default ``(0.10, 0.50, 0.90)``.

        Returns
        -------
        dict[str, Tensor]
            Keys: ``"mean"`` (ensemble mean), and ``"q{p*100:.0f}"`` for each
            quantile level.  All tensors have the same shape as
            ``x_new[..., 0]``.

        Raises
        ------
        RuntimeError
            If :meth:`fit` has not been called yet.
        """
        if not self.members:
            msg = "JackknifeEnsemble has not been fitted yet. Call .fit() first."
            raise RuntimeError(msg)

        all_means: list[Tensor] = []
        for member in self.members:
            member.model.eval()
            member.likelihood.eval()
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                pred = member.likelihood(member.model(x_new))
            all_means.append(pred.mean)

        stacked = torch.stack(all_means, dim=0)  # (K, M, ...) or (K, M)
        ensemble_mean = stacked.mean(dim=0)

        out: dict[str, Tensor] = {"mean": ensemble_mean}
        for p in quantiles:
            label = f"q{int(p * 100)}"
            q_val = torch.quantile(stacked, p, dim=0)
            out[label] = q_val

        return out


def _train_gp(
    model: gpytorch.models.ExactGP,
    likelihood: gpytorch.likelihoods.Likelihood,
    train_x: Tensor,
    train_y: Tensor,
    n_iter: int = 200,
    lr: float = 0.05,
) -> None:
    """Train a GPyTorch model by maximising the marginal log-likelihood.

    Parameters
    ----------
    model:
        GPyTorch model.
    likelihood:
        GPyTorch likelihood.
    train_x, train_y:
        Training data.
    n_iter:
        Number of Adam optimiser steps.
    lr:
        Learning rate.
    """
    model.train()
    likelihood.train()

    optimiser = torch.optim.Adam(
        list(model.parameters()) + list(likelihood.parameters()),
        lr=lr,
    )
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    for _ in range(n_iter):
        optimiser.zero_grad()
        output = model(train_x)
        loss = -mll(output, train_y)
        loss.backward()
        optimiser.step()
