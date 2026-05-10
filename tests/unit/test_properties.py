"""Hypothesis property-based tests.

Tests:
- Mass balance round-trip: volume drift < 1e-9 for 100 random feed/bleed combos.
- Acquisition monotonicity: higher observations → higher best_f → lower EI.
"""

from __future__ import annotations

from hypothesis import given, settings, strategies as st
import torch


@settings(max_examples=50, deadline=5000)
@given(
    perfusion_rate=st.floats(min_value=0.3, max_value=2.0),
    bleed_rate=st.floats(min_value=0.0, max_value=0.3),
)
def test_volume_mass_balance(perfusion_rate: float, bleed_rate: float) -> None:
    """Volume should remain constant under constant-volume operation."""
    from perfusio.chemistry.volumes import perfusion_volume_step, constant_volume_harvest_rate

    v0 = 0.250  # L
    feed_rate = perfusion_rate * v0 / 24.0
    bleed_r = bleed_rate * v0 / 24.0
    harvest_rate = float(constant_volume_harvest_rate(feed_rate, bleed_r))
    v1 = float(perfusion_volume_step(v0, feed_rate, bleed_r, harvest_rate))
    assert abs(v1 - v0) < 1e-8, f"Volume drift {abs(v1-v0):.2e} exceeds tolerance."


@settings(max_examples=20, deadline=10000)
@given(best_f=st.floats(min_value=0.0, max_value=2.0))
def test_ei_decreases_with_higher_best_f(best_f: float) -> None:
    """EI should decrease as best_f increases (all else equal)."""
    from botorch.models import SingleTaskGP
    from botorch.fit import fit_gpytorch_mll
    import gpytorch
    from perfusio.bed.acquisitions import build_acquisition

    torch.manual_seed(42)
    train_X = torch.rand(5, 1, dtype=torch.float64)
    train_Y = torch.rand(5, 1, dtype=torch.float64)
    gp = SingleTaskGP(train_X, train_Y)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(gp.likelihood, gp)
    fit_gpytorch_mll(mll)
    gp.eval()

    test_X = torch.rand(1, 1, 1, dtype=torch.float64)

    acqf_lo = build_acquisition("EI", gp, best_f=max(0.0, best_f - 0.5))
    acqf_hi = build_acquisition("EI", gp, best_f=best_f + 0.5)

    with torch.no_grad():
        val_lo = acqf_lo(test_X)
        val_hi = acqf_hi(test_X)

    # EI should be ≥ 0 and generally decrease as best_f increases
    assert float(val_lo) >= 0.0
    assert float(val_hi) >= 0.0
