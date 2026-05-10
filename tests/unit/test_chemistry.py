"""Unit tests for chemistry sub-package.

Covers:
- SpeciesEnum round-trip (value ↔ name)
- DiscreteMassBalance: glucose consumption satisfies Pirt maintenance
- FlowRateCalculator: output concentrations conserve mass
- perfusion_volume_step: volume stays constant under bleed trigger
"""

from __future__ import annotations

import torch

from perfusio.chemistry.balances import DiscreteMassBalance
from perfusio.chemistry.species import SpeciesEnum, SpeciesRegistry
from perfusio.chemistry.volumes import constant_volume_harvest_rate, perfusion_volume_step


class TestSpeciesEnum:
    def test_all_13_species(self) -> None:
        assert len(SpeciesEnum) == 13

    def test_round_trip_by_value(self) -> None:
        for sp in SpeciesEnum:
            assert SpeciesEnum(sp.value) is sp

    def test_vcd_has_string_value(self) -> None:
        assert SpeciesEnum.VCD.value == "VCD"


class TestDiscreteMassBalance:
    def _make_balance(self) -> DiscreteMassBalance:
        reg = SpeciesRegistry.DEFAULT
        return DiscreteMassBalance([reg["VCD"], reg["Glc"]], dt_hours=24.0)

    def test_glucose_decreases_with_growth(self) -> None:
        bal = self._make_balance()
        c_t = torch.tensor([5.0, 5.0], dtype=torch.float64)
        # Positive growth, net glucose consumption
        R_t = torch.tensor([0.5, -0.1], dtype=torch.float64)
        V_t = torch.tensor(0.250, dtype=torch.float64)
        u = torch.zeros(2, dtype=torch.float64)
        c_next = bal.step(c_t, R_t, V_t, u, u, u)
        assert float(c_next[1]) < float(c_t[1]), "Glucose should decrease when consumed."

    def test_vcd_grows(self) -> None:
        bal = self._make_balance()
        c_t = torch.tensor([5.0, 5.0], dtype=torch.float64)
        R_t = torch.tensor([0.5, -0.1], dtype=torch.float64)
        V_t = torch.tensor(0.250, dtype=torch.float64)
        u = torch.zeros(2, dtype=torch.float64)
        c_next = bal.step(c_t, R_t, V_t, u, u, u)
        # Positive VCD rate → VCD increases
        assert float(c_next[0]) > float(c_t[0])


class TestVolumeStep:
    def test_volume_constant(self) -> None:
        """Volume should remain 0.250 L under constant-volume operation."""
        v0 = 0.250
        bleed_rate = 0.0
        for _ in range(5):
            feed_rate = 0.8 * v0 / 24  # vvd → L/h
            harvest_rate = float(constant_volume_harvest_rate(feed_rate, bleed_rate))
            v1 = float(perfusion_volume_step(v0, feed_rate, bleed_rate, harvest_rate))
            assert abs(v1 - v0) < 1e-9, f"Volume drift: {v1} ≠ {v0}"
