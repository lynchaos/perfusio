"""Mean function definitions for the GP module.

Provides three mean functions:

1. :class:`ZeroMeanMultiTask` — zero mean function used when a pure data-driven
   GP is desired or when the training set is large.
2. :class:`LinearMean` — affine mean function for improved extrapolation.
3. :class:`MechanisticPriorMean` — wraps the :class:`~perfusio.mechanistic.models.CHOPerfusionModel`
   to provide a first-principles prior mean.  The GP then learns only the
   *residual* between the mechanistic prediction and the observations.

References
----------
.. [Gadiyar2026] Gadiyar et al. (2026), §3.1 ("Hybrid model").
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import gpytorch
import torch
from torch import Tensor

if TYPE_CHECKING:
    from perfusio.mechanistic.models import CHOPerfusionModel


class ZeroMeanMultiTask(gpytorch.means.ZeroMean):
    """Zero mean function — thin subclass for consistent naming."""


class LinearMean(gpytorch.means.Mean):
    """Affine (linear + bias) mean function.

    Parameters
    ----------
    input_size:
        Dimensionality of the input.
    batch_shape:
        Batch shape (for multi-output GP).

    Notes
    -----
    Parameters are learnable via MLL training.
    """

    def __init__(
        self,
        input_size: int,
        batch_shape: torch.Size | None = None,
    ) -> None:
        super().__init__()
        batch_shape = batch_shape or torch.Size()
        self.register_parameter(
            "weights",
            torch.nn.Parameter(torch.zeros(*batch_shape, input_size, 1)),
        )
        self.register_parameter(
            "bias",
            torch.nn.Parameter(torch.zeros(*batch_shape, 1)),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Evaluate the linear mean.

        Parameters
        ----------
        x:
            Input tensor, shape ``(..., input_size)``.

        Returns
        -------
        Tensor
            Mean predictions, shape ``(...,)``.
        """
        return (x.matmul(self.weights) + self.bias.unsqueeze(-1)).squeeze(-1)


class MechanisticPriorMean(gpytorch.means.Mean):
    """GP mean function backed by a mechanistic CHO model.

    At any input point :math:`\\mathbf{x} = (\\mathbf{c}, \\mathbf{u}, t)`,
    this mean function evaluates the mechanistic rate
    :math:`\\hat{R}^{\\text{mech}}_k(\\mathbf{x})` from
    :class:`~perfusio.mechanistic.models.CHOPerfusionModel`.

    The GP thus models:

    .. math::
        R_k(\\mathbf{x}) = \\hat{R}^{\\text{mech}}_k(\\mathbf{x}) + \\epsilon_k(\\mathbf{x})

    where :math:`\\epsilon_k \\sim \\mathcal{GP}(0, k_{\\epsilon})`.

    Parameters
    ----------
    mech_model:
        Instantiated :class:`~perfusio.mechanistic.models.CHOPerfusionModel`.
    n_species:
        Number of species whose rates are modelled.
    species_names:
        Ordered list of species names matching kinetics.STATE_ORDER.
    control_names:
        Ordered list of control names.
    task_index:
        Index of the target task (species index into STATE_ORDER).
        Used when the GP is single-task, predicting one species at a time.
        Pass ``None`` for multi-task models that predict all species jointly.
    scale:
        Optional learnable scale parameter to allow the GP to up- or
        down-weight the mechanistic prior.

    Notes
    -----
    The input layout expected is:
    ``x[:, :n_species]``      — normalised species concentrations
    ``x[:, n_species:n_species+n_controls]``  — controls
    ``x[:, -1]``              — culture day (hours ÷ 24 internally)

    Gradient flow is blocked through the mechanistic model (``torch.no_grad``)
    since the kinetics are not themselves differentiable PyTorch modules.
    """

    def __init__(
        self,
        mech_model: CHOPerfusionModel,
        n_species: int,
        species_names: list[str],
        control_names: list[str],
        task_index: int | None = None,
        scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.mech_model = mech_model
        self.n_species = n_species
        self.species_names = species_names
        self.control_names = control_names
        self.task_index = task_index
        self.register_parameter(
            "log_scale",
            torch.nn.Parameter(torch.tensor(scale).log()),
        )

    @property
    def _scale(self) -> Tensor:
        return self.log_scale.exp()

    def forward(self, x: Tensor) -> Tensor:
        """Return mechanistic rate predictions for a batch of inputs.

        Parameters
        ----------
        x:
            Input batch, shape ``(N, n_species + n_controls + 1)``.

        Returns
        -------
        Tensor
            Shape ``(N,)`` if single-task, ``(N, n_species)`` if multi-task.
        """
        c_batch = x[:, : self.n_species]
        ctrl_batch = x[:, self.n_species : self.n_species + len(self.control_names)]

        with torch.no_grad():
            rates = self.mech_model.predict_rates_batch(
                c_batch.double(),
                ctrl_batch.double(),
                self.control_names,
            )

        rates = rates.to(x.dtype)

        if self.task_index is not None:
            return self._scale * rates[:, self.task_index]

        return self._scale * rates
