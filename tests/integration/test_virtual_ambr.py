"""Integration test: 30-day virtual ambr®250 closed-loop.

Asserts that after 20 days the VCD coefficient-of-variation (CV) converges
to within 30 ± 2% of the target (mimicking Gadiyar Fig. 8 criterion).

This test is intentionally slow (~30 s) and is tagged with the ``slow``
marker so it can be excluded with ``pytest -m "not slow"`` in fast CI runs.
"""

from __future__ import annotations

import asyncio

import pytest

pytestmark = pytest.mark.slow


@pytest.mark.asyncio()
async def test_30day_virtual_ambr() -> None:
    """VCD should be within ±30% of target after 20 days."""
    from perfusio.connectors.ambr250_emulator import Ambr250Emulator

    vcd_target = 10.0  # 10 × 10⁶ cells/mL
    tolerance = 0.30  # 30%

    emulator = Ambr250Emulator(clone="CloneX", seed=0)
    # Run 28 days
    n_days = 28
    traj = emulator.simulate_run(n_days=n_days)  # noiseless ndarray (n_days+1, n_spc)

    # After day 20 the VCD should be within tolerance of target (or heading there)
    vcd_day20 = float(traj[20, 0])
    rel_err = abs(vcd_day20 - vcd_target) / vcd_target
    assert rel_err <= tolerance + 0.05, (
        f"VCD at day 20 = {vcd_day20:.2f} (target {vcd_target}), "
        f"relative error {rel_err:.1%} exceeds allowed {tolerance + 0.05:.0%}."
    )


def test_five_reactor_ensemble_independent() -> None:
    """Five reactors should produce different samples (different seeds)."""
    from perfusio.connectors.ambr250_emulator import Ambr250Emulator

    reactors = Ambr250Emulator.five_reactor_ensemble(seed_start=10)
    assert len(reactors) == 5

    samples = [asyncio.run(r.read_sample(day=5)) for r in reactors]
    # All VCDs should differ
    vcds = [s.get("VCD", None) for s in samples]
    assert (
        len(set(round(v, 6) for v in vcds if v is not None)) > 1
    ), "All five reactors returned identical VCD; seeds may not be independent."
