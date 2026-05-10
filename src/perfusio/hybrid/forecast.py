"""Multi-step-ahead trajectory forecasting with the hybrid model.

Exposes :func:`forecast_run` as the primary public entry point.  It wraps the
:class:`~perfusio.hybrid.model.HybridStateSpaceModel` MC rollout and returns
structured outputs that can be directly consumed by the BED OFV function and
the visualisation layer.

References
----------
.. [Gadiyar2026] Gadiyar et al. (2026), §3.2 (3-step-ahead horizon for BED).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import Tensor

if TYPE_CHECKING:
    from perfusio.hybrid.model import HybridStateSpaceModel


def forecast_run(
    hybrid: HybridStateSpaceModel,
    c0: Tensor,
    controls: Tensor,
    horizon: int = 3,
    n_samples: int = 200,
    seed: int = 0,
    clone_id: int | None = None,
) -> dict[str, Tensor]:
    """Forecast species trajectories *horizon* steps ahead.

    Parameters
    ----------
    hybrid:
        Fitted :class:`~perfusio.hybrid.model.HybridStateSpaceModel`.
    c0:
        Initial state, shape ``(n_species,)``.
    controls:
        Control sequence to apply.  Shape ``(horizon, n_controls)``.
        If shape ``(n_controls,)`` is given, the same controls are applied
        at every step.
    horizon:
        Number of future days to predict.  Default 3 (Gadiyar §3.2).
    n_samples:
        MC rollout samples for uncertainty quantification.
    seed:
        Random seed.
    clone_id:
        Clone ID for embedding lookup.

    Returns
    -------
    dict[str, Tensor]
        A dict with keys:

        - ``"days"`` — shape ``(horizon,)`` — day indices (1-indexed from c0).
        - ``"mean"`` — shape ``(horizon, n_species)`` — posterior mean.
        - ``"q10"``  — shape ``(horizon, n_species)`` — 10th percentile.
        - ``"q50"``  — shape ``(horizon, n_species)`` — 50th percentile (median).
        - ``"q90"``  — shape ``(horizon, n_species)`` — 90th percentile.
    """
    # Broadcast controls if a single control vector was provided
    if controls.dim() == 1:
        controls = controls.unsqueeze(0).expand(horizon, -1)

    preds = hybrid.predict_trajectory(
        c0=c0,
        controls=controls,
        n_days=horizon,
        clone_id=clone_id,
        n_samples=n_samples,
        seed=seed,
    )

    days = torch.arange(1, horizon + 1, dtype=torch.float64)
    return {
        "days": days,
        "mean": preds["mean"],
        "q10": preds["q10"],
        "q50": preds.get("q50", preds["mean"]),
        "q90": preds["q90"],
    }
