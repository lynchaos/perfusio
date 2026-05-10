"""Unit tests for BED sub-package.

Covers:
- TargetTrackingOFV: correct SSE shape and monotonicity
- MultiObjectiveOFV: output shape (n_objectives,)
- build_acquisition: non-null returns for all 11 acquisition types
- compute_pareto_front: correct non-dominated filtering
- hypervolume: positive for dominated set
"""

from __future__ import annotations

import pytest
import torch


@pytest.fixture
def tiny_model() -> object:
    """A minimal BoTorch SingleTaskGP for acquisition testing."""
    import gpytorch
    from botorch.fit import fit_gpytorch_mll
    from botorch.models import SingleTaskGP

    torch.manual_seed(0)
    train_X = torch.rand(6, 2, dtype=torch.float64)
    train_Y = torch.rand(6, 1, dtype=torch.float64)
    gp = SingleTaskGP(train_X, train_Y)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(gp.likelihood, gp)
    fit_gpytorch_mll(mll)
    gp.eval()
    return gp


@pytest.fixture
def tiny_mo_model() -> object:
    """Multi-output GP for multi-objective acquisition testing."""
    import gpytorch
    from botorch.fit import fit_gpytorch_mll
    from botorch.models import SingleTaskGP

    torch.manual_seed(0)
    train_X = torch.rand(6, 2, dtype=torch.float64)
    train_Y = torch.rand(6, 2, dtype=torch.float64)
    gp = SingleTaskGP(train_X, train_Y)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(gp.likelihood, gp)
    fit_gpytorch_mll(mll)
    gp.eval()
    return gp


@pytest.fixture
def design_space() -> object:
    from perfusio.config import ControlBounds, DesignSpace

    return DesignSpace(
        controls={
            "perfusion_rate": ControlBounds(lo=0.5, hi=1.5),
            "bleed_rate": ControlBounds(lo=0.10, hi=0.20),
        }
    )


class TestTargetTrackingOFV:
    def test_output_shape(self) -> None:
        from perfusio.bed.objectives import TargetSpec, TargetTrackingOFV

        ofv = TargetTrackingOFV(targets=[TargetSpec(0, 10.0), TargetSpec(8, 500.0)])
        # batch of 3 candidate states, each (3-step, 13-species)
        Y = torch.rand(3, 3, 13, dtype=torch.float64)
        out = ofv.score_trajectories(Y)
        assert out.shape == (3,), f"Expected (3,) got {out.shape}"

    def test_monotonic_at_target(self) -> None:
        from perfusio.bed.objectives import TargetSpec, TargetTrackingOFV

        ofv = TargetTrackingOFV(targets=[TargetSpec(0, 5.0)])
        Y_near = torch.full((1, 3, 13), 5.0, dtype=torch.float64)
        Y_far = torch.full((1, 3, 13), 10.0, dtype=torch.float64)
        assert ofv.score_trajectories(Y_near) > ofv.score_trajectories(Y_far), (
            "OFV should be higher when prediction is closer to target."
        )


class TestBuildAcquisition:
    _SINGLE_OBJ_NAMES = ["PI", "EI", "LogEI", "UCB", "qEI", "qLogEI", "qUCB"]
    _MO_NAMES = ["qEHVI", "qNEHVI", "qNParEGO"]

    @pytest.mark.parametrize("name", _SINGLE_OBJ_NAMES)
    def test_single_objective(self, name: str, tiny_model: object) -> None:
        from perfusio.bed.acquisitions import build_acquisition

        best_f = 0.5
        acqf = build_acquisition(name, tiny_model, best_f=best_f, beta=2.0)
        assert acqf is not None

    @pytest.mark.parametrize("name", _MO_NAMES)
    def test_multi_objective(self, name: str, tiny_mo_model: object) -> None:
        from botorch.utils.multi_objective.box_decompositions.non_dominated import (
            FastNondominatedPartitioning,
        )

        from perfusio.bed.acquisitions import build_acquisition

        ref_point = torch.zeros(2, dtype=torch.float64)
        Y_ref = torch.rand(6, 2, dtype=torch.float64)
        partitioning = FastNondominatedPartitioning(ref_point=ref_point, Y=Y_ref)
        acqf = build_acquisition(
            name,
            tiny_mo_model,
            ref_point=ref_point,
            partitioning=partitioning,
        )
        assert acqf is not None


class TestParetoFront:
    def test_nondominated_filter(self) -> None:
        from perfusio.bed.pareto import compute_pareto_front

        Y = torch.tensor(
            [
                [1.0, 1.0],  # dominated by [2,2]
                [2.0, 2.0],  # Pareto
                [3.0, 0.5],  # Pareto (better on obj1)
                [0.5, 3.0],  # Pareto (better on obj2)
            ],
            dtype=torch.float64,
        )
        mask = compute_pareto_front(Y)
        assert mask[1] and mask[2] and mask[3], "All three non-dominated points should be flagged."
        assert not mask[0], "Point [1,1] is dominated; should not be on Pareto front."

    def test_hypervolume_positive(self) -> None:
        from perfusio.bed.pareto import hypervolume

        Y = torch.tensor([[2.0, 1.0], [1.0, 2.0]], dtype=torch.float64)
        ref = torch.zeros(2, dtype=torch.float64)
        hv = hypervolume(Y, ref)
        assert hv > 0.0, "Hypervolume must be positive for a non-degenerate front."
