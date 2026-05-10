"""Example 07 — Connect to a real ambr®250 via OPC UA.

**Requires** a live OPC UA server at ``opc.tcp://localhost:4840/``.
For testing purposes use the built-in Ambr250Emulator OPC UA server
(not included in this repo; see docs/tutorials/T07-real-ambr-opcua.rst).

Run::

    python examples/07_real_ambr_opcua.py --endpoint opc.tcp://localhost:4840/
"""

from __future__ import annotations

import asyncio
import argparse

from perfusio.config import DEFAULT_AMBR250_DESIGN_SPACE, RunConfig
from perfusio.connectors.opcua_client import OPCUAConnector
from perfusio.twin.digital_twin import DigitalTwin

parser = argparse.ArgumentParser(description="Connect to ambr®250 via OPC UA.")
parser.add_argument("--endpoint", default="opc.tcp://localhost:4840/", help="OPC UA endpoint URL.")
parser.add_argument("--days", type=int, default=28, help="Run duration in days.")
args = parser.parse_args()

DS = DEFAULT_AMBR250_DESIGN_SPACE
run_cfg = RunConfig(duration_days=args.days, sampling_interval_hours=24.0)


async def main() -> None:
    connector = OPCUAConnector(endpoint=args.endpoint)
    twin = DigitalTwin(connector=connector, design_space=DS, run_config=run_cfg)
    print(f"Connecting to {args.endpoint}…")
    alive = await connector.is_alive()
    if not alive:
        print("OPC UA server unreachable. Start the server and retry.")
        return
    await twin.run()
    print("Run complete.")


asyncio.run(main())
