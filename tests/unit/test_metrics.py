"""Unit tests for metrics sub-package."""

from __future__ import annotations

import pytest
import torch

from perfusio.metrics import (
    crps,
    epsilon_indicator,
    hypervolume_indicator,
    igd_plus,
    pi_coverage,
    rrmse_horizon,
    sharpness,
)


class TestRrmse:
    def test_zero_error(self) -> None:
        y = torch.randn(4, 10, 9, dtype=torch.float64)
        rr = rrmse_horizon(y, y)
        assert (rr.abs() < 1e-6).all(), "rRMSE should be 0 for perfect predictions."

    def test_shape_horizon(self) -> None:
        y = torch.randn(5, 10, 9)
        yhat = y + 0.1
        rr = rrmse_horizon(y, yhat, horizon=3)
        assert rr.shape == (3, 9)

    def test_mismatched_shapes(self) -> None:
        y = torch.randn(3, 5, 9)
        yhat = torch.randn(3, 5, 8)
        with pytest.raises(ValueError):
            rrmse_horizon(y, yhat)


class TestCoverage:
    def test_full_coverage(self) -> None:
        y = torch.ones(10, 3)
        lo = torch.zeros(10, 3)
        hi = torch.full((10, 3), 2.0)
        cov = pi_coverage(y, lo, hi)
        assert torch.allclose(cov, torch.ones(3)), "All inside → coverage = 1.0"

    def test_zero_coverage(self) -> None:
        y = torch.ones(10, 3) * 5.0
        lo = torch.zeros(10, 3)
        hi = torch.ones(10, 3)
        cov = pi_coverage(y, lo, hi)
        assert torch.allclose(cov, torch.zeros(3)), "All outside → coverage = 0.0"

    def test_sharpness_monotone(self) -> None:
        lo = torch.zeros(10, 2)
        hi_wide = torch.full((10, 2), 4.0)
        hi_narrow = torch.full((10, 2), 1.0)
        assert (sharpness(lo, hi_wide) > sharpness(lo, hi_narrow)).all()

    def test_crps_positive(self) -> None:
        y = torch.randn(8, 3)
        samples = torch.randn(8, 50, 3)  # 50 posterior samples
        score = crps(y, samples)
        assert (score >= 0).all(), "CRPS must be non-negative."


class TestMultiObjective:
    def test_hypervolume_positive(self) -> None:
        Y = torch.tensor([[2.0, 1.0], [1.0, 2.0]], dtype=torch.float64)
        ref = torch.zeros(2, dtype=torch.float64)
        hv = hypervolume_indicator(Y, ref)
        assert hv > 0.0

    def test_igd_plus_zero_perfect(self) -> None:
        front = torch.tensor([[1.0, 2.0], [2.0, 1.0]], dtype=torch.float64)
        score = igd_plus(front, front)
        assert score < 1e-10, "IGD+ should be 0 when approx == reference."

    def test_epsilon_indicator_zero(self) -> None:
        front = torch.tensor([[2.0, 1.0], [1.0, 2.0]], dtype=torch.float64)
        eps = epsilon_indicator(front, front)
        assert eps <= 0.0, "ε-indicator should be ≤0 when approx dominates reference."
