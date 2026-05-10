"""Example 06 — Run the digital twin in online mode (file-system connector).

Simulates a 14-day run from a CSV file and drives the DigitalTwin
with a FilesystemStore connector (no hardware required).

Run::

    python examples/06_online_digital_twin_filesystem.py
"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from perfusio.config import DEFAULT_AMBR250_DESIGN_SPACE, RunConfig
from perfusio.connectors.filesystem import FilesystemStore
from perfusio.twin.digital_twin import DigitalTwin

DS = DEFAULT_AMBR250_DESIGN_SPACE

# ── Create a fake CSV with 14 daily samples ──────────────────────────────────
with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as fh:
    csv_path = Path(fh.name)

rng = np.random.default_rng(0)
days = np.arange(0, 15, 1.0)
df = pd.DataFrame(
    {
        "day": days,
        "VCD": rng.uniform(5, 12, len(days)),
        "viability": rng.uniform(0.88, 0.97, len(days)),
        "glucose": rng.uniform(3.5, 6.0, len(days)),
        "lactate": rng.uniform(0.8, 2.5, len(days)),
        "glutamine": rng.uniform(0.5, 1.5, len(days)),
        "glutamate": rng.uniform(0.2, 0.8, len(days)),
        "titer": rng.uniform(100, 600, len(days)),
        "perfusion_rate": rng.uniform(0.5, 1.5, len(days)),
        "bleed_rate": rng.uniform(0.05, 0.15, len(days)),
    }
)
df.to_csv(csv_path, index=False)
print(f"Fake CSV written: {csv_path}")

# ── Digital twin ──────────────────────────────────────────────────────────────
connector = FilesystemStore(source_path=csv_path)
run_cfg = RunConfig(duration_days=14, sampling_interval_hours=24.0)
twin = DigitalTwin(connector=connector, design_space=DS, run_config=run_cfg)

asyncio.run(twin.run())
print("Digital twin run complete.")
