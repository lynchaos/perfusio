# T07 — Real ambr®250 via OPC UA

**Goal:** Connect the `DigitalTwin` to a live ambr®250 microbioreactor
system over OPC UA and run a self-driving experiment.

**Script:** `examples/07_real_ambr_opcua.py`

> **Hardware required.** This tutorial requires a running OPC UA server
> at the configured endpoint (default: `opc.tcp://localhost:4840/`).
> For offline testing, substitute a `FilesystemStore` connector as shown
> in [T06](T06-digital-twin.md).

## What the script does

1. Parses `--endpoint` and `--days` command-line arguments.
2. Creates an `OPCUAConnector(endpoint=...)` with auto-reconnect and
   exponential backoff.
3. Checks server reachability via `connector.is_alive()`.
4. Instantiates and runs `DigitalTwin` for the configured number of days.

## Running

```bash
# Against a real OPC UA server
python examples/07_real_ambr_opcua.py \
    --endpoint opc.tcp://192.168.1.10:4840/ \
    --days 28

# Against localhost (e.g. a simulation server)
python examples/07_real_ambr_opcua.py --days 14
```

## Key code

```python
from perfusio.connectors.opcua_client import OPCUAConnector
from perfusio.twin.digital_twin import DigitalTwin
from perfusio.config import DEFAULT_AMBR250_DESIGN_SPACE, RunConfig
import asyncio

async def main():
    connector = OPCUAConnector(endpoint="opc.tcp://localhost:4840/")
    twin = DigitalTwin(
        connector=connector,
        design_space=DEFAULT_AMBR250_DESIGN_SPACE,
        run_config=RunConfig(duration_days=28, sampling_interval_hours=24.0),
    )
    if not await connector.is_alive():
        raise RuntimeError("OPC UA server unreachable")
    await twin.run()

asyncio.run(main())
```

## OPC UA connector features

| Feature | Detail |
|---------|--------|
| Protocol | asyncua (async OPC UA client) |
| Authentication | Anonymous, username/password, or X.509 certificate |
| Reconnect | Exponential backoff (max 5 retries, 2× delay) |
| Replay mode | Pass `replay_csv=Path(...)` to replay offline data |
| Engineering limits | All reads validated; out-of-range values rejected with reason codes |

## Node configuration

By default, `OPCUAConnector` reads from node IDs matching the
ambr®250 standard namespace. To customise for your instrument, subclass
and override `_node_map()`:

```python
from perfusio.connectors.opcua_client import OPCUAConnector

class MyConnector(OPCUAConnector):
    def _node_map(self) -> dict[str, str]:
        return {
            "VCD":      "ns=2;s=Reactor1.VCD",
            "glucose":  "ns=2;s=Reactor1.Glucose",
            # ... remaining species and controls
        }
```

## Security considerations

- Use certificate-based authentication in production environments.
- Ensure the OPC UA server enforces engineering-limit validation on
  incoming setpoints independently of `perfusio` output.
- Review [Regulatory Considerations](../regulatory-considerations.md)
  before deploying in a GMP context.
