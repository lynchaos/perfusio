"""Hybrid State-Space Model: mechanistic prior + GP residual.

The :class:`HybridStateSpaceModel` is the central model object used by the
self-driving loop.  It holds:

- A :class:`~perfusio.mechanistic.models.CHOPerfusionModel` (mechanistic prior)
- A :class:`~perfusio.gp.stepwise.StepwiseGP` (data-driven residual)
- Optionally an :class:`~perfusio.embedding.clones.EntityEmbedding` for
  cross-clone transfer.

Prediction combines the mechanistic and GP contributions as:

.. math::
    \\hat{c}_{t+1} = c_t + \\Delta t \\cdot
      \\left[ \\hat{R}^{\\text{mech}}(c_t, u_t) + \\hat{\\epsilon}^{\\text{GP}}(c_t, u_t) \\right]

References
----------
.. [Gadiyar2026] Gadiyar et al. (2026), §3.1 and §3.2.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import Tensor

if TYPE_CHECKING:
    from perfusio.embedding.clones import EntityEmbedding
    from perfusio.gp.stepwise import StepwiseGP
    from perfusio.mechanistic.models import CHOPerfusionModel


class HybridStateSpaceModel:
    """Combined mechanistic + GP hybrid model.

    Parameters
    ----------
    mech_model:
        Fitted (or default) mechanistic model.
    gp_model:
        Fitted one-step SW-GP model.
    embedding:
        Optional entity embedding (None for single-clone scenarios).
    dt_hours:
        Time step in hours.  Default 24 (daily).
    species_names:
        Ordered species names corresponding to state columns.
    control_names:
        Ordered control names corresponding to the control input.

    Examples
    --------
    >>> from perfusio.mechanistic import CHOPerfusionModel
    >>> mech = CHOPerfusionModel()
    >>> # gp_model = ... (fitted StepwiseGP)
    >>> # hybrid = HybridStateSpaceModel(mech, gp_model)
    """

    def __init__(
        self,
        mech_model: "CHOPerfusionModel",
        gp_model: "StepwiseGP",
        embedding: "EntityEmbedding | None" = None,
        dt_hours: float = 24.0,
        species_names: list[str] | None = None,
        control_names: list[str] | None = None,
    ) -> None:
        self.mech_model = mech_model
        self.gp_model = gp_model
        self.embedding = embedding
        self.dt_hours = dt_hours
        self.species_names = species_names or []
        self.control_names = control_names or []

    def predict_next_state(
        self,
        c_t: Tensor,
        u_t: Tensor,
        day: int,
        clone_id: int | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Predict next-day species concentrations.

        Parameters
        ----------
        c_t:
            Current state, shape ``(n_species,)``.
        u_t:
            Current controls, shape ``(n_controls,)``.
        day:
            Current culture day (1-indexed).
        clone_id:
            Integer clone index for embedding lookup.  Required only when
            ``self.embedding`` is not None.

        Returns
        -------
        tuple[Tensor, Tensor, Tensor]
            ``(mean, q10, q90)`` — posterior mean and 80% credible interval
            for :math:`c_{t+1}`.  All shape ``(n_species,)``.
        """
        # Step 1: Mechanistic rate prediction
        ctrl_dict = {k: float(u_t[i]) for i, k in enumerate(self.control_names)}
        r_mech = self.mech_model.predict_rates(c_t, ctrl_dict)  # (n_species,)

        # Step 2: Build GP input (augment with embedding if present)
        x_in = torch.cat([c_t, u_t, torch.tensor([float(day)], dtype=c_t.dtype)])
        if self.embedding is not None and clone_id is not None:
            cid = torch.tensor([clone_id], dtype=torch.long)
            e = self.embedding(cid).squeeze(0)
            x_in = torch.cat([x_in, e])

        # StepwiseGP expects (1, d_input)
        x_in = x_in.unsqueeze(0)
        u_h = u_t.unsqueeze(0)
        preds = self.gp_model.predict_quantiles(
            c_t, u_h.unsqueeze(0), horizon=1, n_samples=100, seed=0
        )

        gp_mean = preds["mean"][0]  # (n_species,)
        gp_q10 = preds["q10"][0]
        gp_q90 = preds["q90"][0]

        # Step 3: Euler integration with hybrid rate
        dt = self.dt_hours
        c_mean = (c_t + dt * r_mech + gp_mean).clamp(min=0.0)
        c_q10 = (c_t + dt * r_mech + gp_q10).clamp(min=0.0)
        c_q90 = (c_t + dt * r_mech + gp_q90).clamp(min=0.0)

        return c_mean, c_q10, c_q90

    def predict_trajectory(
        self,
        c0: Tensor,
        controls: Tensor,
        n_days: int,
        clone_id: int | None = None,
        n_samples: int = 200,
        seed: int = 0,
    ) -> dict[str, Tensor]:
        """Rollout the hybrid model over *n_days* steps.

        Parameters
        ----------
        c0:
            Initial state, shape ``(n_species,)``.
        controls:
            Control sequence, shape ``(n_days, n_controls)`` (constant or
            varying by day).
        n_days:
            Prediction horizon.
        clone_id:
            Clone ID for embedding.
        n_samples:
            MC samples for GP rollout.
        seed:
            Random seed.

        Returns
        -------
        dict[str, Tensor]
            Keys ``"mean"``, ``"q10"``, ``"q90"`` each shape ``(n_days, n_species)``.
        """
        preds = self.gp_model.predict_quantiles(
            c0, controls, horizon=n_days, n_samples=n_samples, seed=seed
        )
        return preds
