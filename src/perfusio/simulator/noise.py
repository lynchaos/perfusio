"""Noise model for the in-silico simulator.

Simulates typical analytical measurement uncertainty for a CHO perfusion
bioreactor operating in an ambr®250 platform.

Calibrated to match Gadiyar et al. (2026) §3.3:
- VCD: 5% CV (coefficient of variation)
- Viability: 2% CV
- Metabolites (Glc, Gln, Glu, Lac, Amm, Pyr): 7% CV
- Titer: 8% CV
- Missing-data probability: 5% per species per day
- Measurement time jitter: ±1 hour (uniform)

References
----------
.. [Gadiyar2026] Gadiyar et al. (2026), §3.3 — in-silico noise parameterisation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


@dataclass
class NoiseModel:
    """Additive proportional Gaussian noise + missing-data model.

    Parameters
    ----------
    cv_by_species:
        Dict mapping species name → coefficient of variation (0–1).
        Species not listed use ``default_cv``.
    default_cv:
        Fallback CV for unlisted species.
    missing_prob:
        Probability that a single (species, day) measurement is missing.
    jitter_hours:
        Half-width of uniform jitter on measurement time [h].
    seed:
        Random seed.

    Examples
    --------
    >>> from perfusio.simulator import NoiseModel
    >>> nm = NoiseModel(seed=0)
    >>> clean = {"VCD": 20.0, "Glc": 5.0}
    >>> noisy = nm.apply(clean, day=7)
    """

    cv_by_species: dict[str, float] = field(default_factory=lambda: {
        "VCD":   0.05,
        "VCV":   0.05,
        "Via":   0.02,
        "Diam":  0.03,
        "Glc":   0.07,
        "Gln":   0.07,
        "Glu":   0.07,
        "Lac":   0.07,
        "Amm":   0.07,
        "Pyr":   0.07,
        "Titer": 0.08,
    })
    default_cv: float = 0.07
    missing_prob: float = 0.05
    jitter_hours: float = 1.0
    seed: int = 42

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.seed)

    def apply(
        self,
        state: dict[str, float],
        day: int,  # noqa: ARG002 — future use for time-varying noise
    ) -> dict[str, float | None]:
        """Apply noise and missing-data to a clean state observation.

        Parameters
        ----------
        state:
            Clean simulated state, dict mapping species name → value.
        day:
            Culture day (not used for noise level; reserved for future use).

        Returns
        -------
        dict[str, float | None]
            Noisy observation.  Missing values are represented as ``None``.
        """
        noisy: dict[str, float | None] = {}
        for species, value in state.items():
            if self._rng.random() < self.missing_prob:
                noisy[species] = None
                continue
            cv = self.cv_by_species.get(species, self.default_cv)
            sigma = abs(value) * cv
            noise = self._rng.normal(loc=0.0, scale=sigma)
            noisy[species] = max(value + noise, 0.0)
        return noisy

    def time_jitter(self) -> float:
        """Return a random measurement time offset [h]."""
        return float(self._rng.uniform(-self.jitter_hours, self.jitter_hours))
