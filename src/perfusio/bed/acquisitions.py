"""BoTorch acquisition function factory for Bayesian Experimental Design.

Provides :func:`build_acquisition`, a single entry-point that constructs any of
the 11 acquisition functions supported by ``perfusio``:

Single-objective (analytic):
    ``PI``, ``EI``, ``LogEI``, ``UCB``

Single-objective (Monte Carlo):
    ``qEI``, ``qLogEI``, ``qUCB``

Multi-objective (Monte Carlo):
    ``qEHVI``, ``qNEHVI``, ``qNParEGO``

Constrained variants are automatically applied when ``constraints`` are passed.

References
----------
.. [Balandat2020] Balandat, M., et al. (2020). BoTorch: A Framework for
   Efficient Monte-Carlo Bayesian Optimization. NeurIPS.
"""

from __future__ import annotations

from collections.abc import Callable

import botorch.acquisition as ba
import botorch.acquisition.multi_objective as bamo
import torch
from botorch.acquisition.multi_objective.parego import qLogNParEGO
from botorch.models.model import Model
from torch import Tensor

# Supported acquisition names
_SINGLE_OBJ_ANALYTIC = {"PI", "EI", "LogEI", "UCB"}
_SINGLE_OBJ_MC = {"qEI", "qLogEI", "qUCB"}
_MULTI_OBJ_MC = {"qEHVI", "qNEHVI", "qNParEGO"}
ALL_ACQUISITIONS = _SINGLE_OBJ_ANALYTIC | _SINGLE_OBJ_MC | _MULTI_OBJ_MC


def build_acquisition(
    name: str,
    model: Model,
    best_f: float | None = None,
    beta: float = 0.2,
    ref_point: list[float] | None = None,
    partitioning: object | None = None,
    sampler: object | None = None,
    constraints: list[Callable[[Tensor], Tensor]] | None = None,
    **kwargs: object,
) -> ba.AcquisitionFunction:
    """Construct a BoTorch acquisition function by name.

    Parameters
    ----------
    name:
        One of: ``PI``, ``EI``, ``LogEI``, ``UCB``,
        ``qEI``, ``qLogEI``, ``qUCB``, ``qEHVI``, ``qNEHVI``, ``qNParEGO``.
    model:
        Fitted BoTorch / GPyTorch surrogate model.
    best_f:
        Incumbent best observed value.  Required for ``PI``, ``EI``,
        ``LogEI``, ``qEI``, ``qLogEI``.
    beta:
        UCB exploration parameter.  Required for ``UCB``, ``qUCB``.
    ref_point:
        Reference point for hypervolume-based acquisitions (``qEHVI``,
        ``qNEHVI``).  Should be strictly dominated by all feasible points.
    partitioning:
        Pre-computed box decomposition (``NondominatedPartitioning``).
        Required for ``qEHVI``.
    sampler:
        MC sampler (``IIDNormalSampler`` or ``SobolQMCNormalSampler``).
        Defaults to ``SobolQMCNormalSampler(512)`` for MC acquisitions.
    constraints:
        Optional list of constraint callables ``c(X) <= 0`` to enforce.
        Applied as soft-constraint penalties where available.
    **kwargs:
        Additional keyword arguments forwarded to the acquisition constructor.

    Returns
    -------
    botorch.acquisition.AcquisitionFunction

    Raises
    ------
    ValueError
        If *name* is not a recognised acquisition type.
    """
    if name not in ALL_ACQUISITIONS:
        msg = f"Unknown acquisition '{name}'. " f"Choose one of: {sorted(ALL_ACQUISITIONS)}"
        raise ValueError(msg)

    if sampler is None and name in (_SINGLE_OBJ_MC | _MULTI_OBJ_MC):
        from botorch.sampling.normal import SobolQMCNormalSampler

        sampler = SobolQMCNormalSampler(sample_shape=torch.Size([512]))

    # ── Analytic single-objective ──────────────────────────────────────────
    if name == "PI":
        _require(best_f, "PI", "best_f")
        return ba.analytic.ProbabilityOfImprovement(model=model, best_f=best_f, **kwargs)

    if name == "EI":
        _require(best_f, "EI", "best_f")
        return ba.analytic.ExpectedImprovement(model=model, best_f=best_f, **kwargs)

    if name == "LogEI":
        _require(best_f, "LogEI", "best_f")
        return ba.analytic.LogExpectedImprovement(model=model, best_f=best_f, **kwargs)

    if name == "UCB":
        return ba.analytic.UpperConfidenceBound(model=model, beta=beta, **kwargs)

    # ── MC single-objective ────────────────────────────────────────────────
    if name == "qEI":
        _require(best_f, "qEI", "best_f")
        if constraints:
            return ba.monte_carlo.qExpectedImprovement(
                model=model, best_f=best_f, sampler=sampler, constraints=constraints, **kwargs
            )
        return ba.monte_carlo.qExpectedImprovement(
            model=model, best_f=best_f, sampler=sampler, **kwargs
        )

    if name == "qLogEI":
        _require(best_f, "qLogEI", "best_f")
        from botorch.acquisition.logei import qLogExpectedImprovement

        return qLogExpectedImprovement(model=model, best_f=best_f, sampler=sampler, **kwargs)

    if name == "qUCB":
        return ba.monte_carlo.qUpperConfidenceBound(
            model=model, beta=beta, sampler=sampler, **kwargs
        )

    # ── Multi-objective MC ─────────────────────────────────────────────────
    if name == "qEHVI":
        _require(ref_point, "qEHVI", "ref_point")
        _require(partitioning, "qEHVI", "partitioning")
        return bamo.qExpectedHypervolumeImprovement(
            model=model,
            ref_point=ref_point,
            partitioning=partitioning,
            sampler=sampler,
            constraints=constraints,
            **kwargs,
        )

    if name == "qNEHVI":
        _require(ref_point, "qNEHVI", "ref_point")
        # X_baseline defaults to the model's training inputs when not supplied
        _x_bl = kwargs.pop(
            "X_baseline",
            model.train_inputs[0]
            if hasattr(model, "train_inputs") and model.train_inputs
            else None,  # type: ignore[attr-defined]
        )
        if _x_bl is None:
            msg = "Acquisition 'qNEHVI' requires 'X_baseline' or a fitted model with train_inputs."
            raise ValueError(msg)
        # Strip leading batch dim produced by batched multi-output GPs
        if isinstance(_x_bl, Tensor) and _x_bl.dim() == 3:
            _x_bl = _x_bl[0]
        return bamo.qNoisyExpectedHypervolumeImprovement(
            model=model,
            ref_point=ref_point,
            X_baseline=_x_bl,
            sampler=sampler,
            constraints=constraints,
            **kwargs,
        )

    if name == "qNParEGO":
        _x_bl = kwargs.pop(
            "X_baseline",
            model.train_inputs[0]
            if hasattr(model, "train_inputs") and model.train_inputs
            else None,  # type: ignore[attr-defined]
        )
        if _x_bl is None:
            msg = (
                "Acquisition 'qNParEGO' requires 'X_baseline' or a fitted model with train_inputs."
            )
            raise ValueError(msg)
        # Strip leading batch dim produced by batched multi-output GPs
        if isinstance(_x_bl, Tensor) and _x_bl.dim() == 3:
            _x_bl = _x_bl[0]
        return qLogNParEGO(
            model=model,
            X_baseline=_x_bl,
            sampler=sampler,
            constraints=constraints,
            **kwargs,
        )

    # Should never reach here — kept for static analysis
    msg = f"Unhandled acquisition name: {name}"
    raise AssertionError(msg)


def _require(value: object | None, acq_name: str, param_name: str) -> None:
    if value is None:
        msg = f"Acquisition '{acq_name}' requires '{param_name}' to be provided."
        raise ValueError(msg)
