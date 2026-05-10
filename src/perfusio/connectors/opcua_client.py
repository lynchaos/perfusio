"""OPC UA connector for real bioreactor hardware.

Wraps the ``asyncua`` async OPC UA client with:

- Auto-reconnect with exponential backoff (max 5 retries).
- Browse-tree caching (resolved node IDs are stored to avoid repeated discovery).
- Recorded-replay mode (saves/loads OPC UA values from a JSONL log file for
  offline testing and audit trail reconstruction).
- Certificate-based and username/password authentication.

Required OPC UA node IDs must be configured via the ``node_map`` dict.

References
----------
.. [OPCUA] IEC 62541 — OPC Unified Architecture.
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any

from perfusio.connectors.base import BioreactorConnectorBase

logger = logging.getLogger(__name__)


class OPCUAConnector(BioreactorConnectorBase):
    """Async OPC UA client for real bioreactor hardware.

    Parameters
    ----------
    url:
        OPC UA server URL, e.g. ``"opc.tcp://192.168.1.10:4840"``.
    node_map:
        Dict mapping species/control name → OPC UA node-ID string
        (e.g. ``"ns=2;s=VCD"``).
    username:
        Optional username for Basic256Sha256 auth.
    password:
        Optional password.
    cert_path:
        Optional path to PEM client certificate.
    replay_log:
        If provided, read values from this JSONL file (replay mode) instead of
        connecting to a real server.  Each line is a JSON object with keys
        ``day``, ``node_id``, ``value``.
    max_retries:
        Maximum reconnection attempts.
    retry_delay_s:
        Initial delay between reconnection attempts (doubles each retry).

    Examples
    --------
    >>> from perfusio.connectors import OPCUAConnector  # doctest: +SKIP
    >>> conn = OPCUAConnector(
    ...     url="opc.tcp://localhost:4840",
    ...     node_map={"VCD": "ns=2;s=VCD", "Glc": "ns=2;s=Glucose"},
    ... )
    """

    def __init__(
        self,
        url: str,
        node_map: dict[str, str],
        username: str | None = None,
        password: str | None = None,
        cert_path: str | None = None,
        replay_log: Path | str | None = None,
        max_retries: int = 5,
        retry_delay_s: float = 2.0,
    ) -> None:
        self.url = url
        self.node_map = node_map
        self.username = username
        self.password = password
        self.cert_path = cert_path
        self.replay_log = Path(replay_log) if replay_log else None
        self.max_retries = max_retries
        self.retry_delay_s = retry_delay_s
        self._client: Any = None  # asyncua.Client
        self._node_cache: dict[str, Any] = {}
        self._replay_data: dict[int, dict[str, Any]] | None = None

        if self.replay_log:
            self._load_replay_log()

    # ── BioreactorConnectorBase ────────────────────────────────────────────

    async def read_sample(self, day: int) -> dict[str, Any]:
        if self._replay_data is not None:
            return self._replay_data.get(day, {})
        await self._ensure_connected()
        sample: dict[str, Any] = {}
        for name, node_id in self.node_map.items():
            try:
                node = await self._resolve_node(node_id)
                val = await node.read_value()
                sample[name] = float(val)
            except Exception:
                logger.warning("OPC UA read failed for node %s.", node_id)
                sample[name] = None
        return sample

    async def write_setpoints(self, setpoints: dict[str, float]) -> None:
        if self._replay_data is not None:
            logger.info("OPCUAConnector: replay mode — write skipped.")
            return
        await self._ensure_connected()
        for name, value in setpoints.items():
            node_id = self.node_map.get(name)
            if node_id is None:
                logger.warning("OPCUAConnector: no node mapped for control '%s'.", name)
                continue
            try:
                from asyncua import ua  # type: ignore[import]

                node = await self._resolve_node(node_id)
                await node.write_value(
                    ua.DataValue(ua.Variant(float(value), ua.VariantType.Double))
                )
                logger.info("OPCUAConnector: wrote %s = %.4f", name, value)
            except Exception:
                logger.exception("OPCUAConnector: write failed for node %s.", node_id)

    async def is_alive(self) -> bool:
        if self._replay_data is not None:
            return True
        return self._client is not None

    # ── Connection management ──────────────────────────────────────────────

    async def _ensure_connected(self) -> None:
        if self._client is not None:
            return
        delay = self.retry_delay_s
        for attempt in range(1, self.max_retries + 1):
            try:
                from asyncua import Client  # type: ignore[import]

                client = Client(url=self.url)
                if self.username and self.password:
                    client.set_user(self.username)
                    client.set_password(self.password)
                if self.cert_path:
                    client.load_client_certificate(self.cert_path)
                await client.connect()
                self._client = client
                logger.info("OPCUAConnector: connected to %s.", self.url)
                return
            except Exception:
                logger.warning(
                    "OPCUAConnector: connection attempt %d/%d failed.",
                    attempt,
                    self.max_retries,
                )
                if attempt < self.max_retries:
                    await asyncio.sleep(delay)
                    delay = min(delay * 2, 60.0)
        msg = f"OPCUAConnector: could not connect to {self.url} after {self.max_retries} attempts."
        raise ConnectionError(msg)

    async def _resolve_node(self, node_id: str) -> Any:
        if node_id not in self._node_cache:
            assert self._client is not None
            self._node_cache[node_id] = self._client.get_node(node_id)
        return self._node_cache[node_id]

    async def disconnect(self) -> None:
        """Gracefully disconnect from the OPC UA server."""
        if self._client is not None:
            try:
                await self._client.disconnect()
            except Exception:
                pass
            self._client = None
            logger.info("OPCUAConnector: disconnected.")

    # ── Replay log ─────────────────────────────────────────────────────────

    def _load_replay_log(self) -> None:
        assert self.replay_log is not None
        self._replay_data = {}
        with open(self.replay_log) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                day = int(record["day"])
                name = str(record["name"])
                value = record.get("value")
                if day not in self._replay_data:
                    self._replay_data[day] = {}
                self._replay_data[day][name] = value
        logger.info(
            "OPCUAConnector: loaded replay log with %d days from %s.",
            len(self._replay_data),
            self.replay_log,
        )
