# T06 — Online Digital Twin (Filesystem Connector)

**Goal:** Run the closed-loop `DigitalTwin` driven by a CSV file (no
hardware required) to simulate a 14-day self-driving experiment.

**Script:** `examples/06_online_digital_twin_filesystem.py`

## What the script does

1. Writes a synthetic 14-day CSV with daily samples for 10 process variables.
2. Creates a `FilesystemStore(source_path=csv_path)` connector.
3. Instantiates `DigitalTwin(connector, design_space, run_config)`.
4. Runs `asyncio.run(twin.run())` which loops:
   - Reads the next daily sample from the CSV.
   - Appends to the observation buffer.
   - Calls the BED policy to recommend the next control setpoints.
   - Writes setpoints back via the connector.
   - Retrains the GP every `retrain_every` steps.

## Running

```bash
python examples/06_online_digital_twin_filesystem.py
```

## Key code

```python
import asyncio
from perfusio.connectors.filesystem import FilesystemStore
from perfusio.twin.digital_twin import DigitalTwin
from perfusio.config import DEFAULT_AMBR250_DESIGN_SPACE, RunConfig

connector = FilesystemStore(source_path="my_data.csv")
run_cfg   = RunConfig(duration_days=14, sampling_interval_hours=24.0)
twin      = DigitalTwin(connector=connector,
                        design_space=DEFAULT_AMBR250_DESIGN_SPACE,
                        run_config=run_cfg)
asyncio.run(twin.run())
```

## Digital twin control loop

```
Day t:
  1. connector.read_sample()       → State
  2. _obs_buffer.append(...)       → online learning buffer
  3. hybrid.predict_next_state()   → 28-day forecast
  4. BEDPolicy.decide()            → recommended controls
  5. connector.write_setpoints()   → push to reactor / CSV
  6. (every N days) _retrain()     → update GP with new data
```

## Online retraining

The GP is retrained using indexed multi-task format:

- Input `X`: shape `(N × 9, 17)` — one row per species per time step,
  columns are `[species×9, controls×6, day, task_id]`.
- Target `y`: shape `(N × 9,)` — scalar next-step values.

```python
# Controlled via RunConfig
run_cfg = RunConfig(
    duration_days=28,
    sampling_interval_hours=24.0,
    retrain_every=3,          # retrain GP every 3 days
    acquisition="LogEI",
)
```

## Audit trail

All sampling events and setpoint changes are automatically logged by
`AuditLogger` with ISO 8601 timestamps and SHA-256 state hashes.
Logs are written to `audit.jsonl` in the working directory.

## Next step

Proceed to [T07 — Real ambr®250 via OPC UA](T07-real-ambr.md).
