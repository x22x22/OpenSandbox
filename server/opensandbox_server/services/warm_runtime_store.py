"""Durable store for the warm runtime ledger.

The control plane must survive a restart without reissuing live leases, so the
ledger state is persisted. JSON + atomic replace keeps the PoC dependency-free
and crash-safe at the file level; a real deployment can swap in a database
store behind the same ``WarmRuntimeStore`` protocol.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:  # pragma: no cover - typing only
    from opensandbox_server.services.warm_runtime_pool import WarmRuntime


class WarmRuntimeStore(Protocol):
    def load(self) -> tuple[dict[str, "WarmRuntime"], dict[str, int]]: ...

    def save(
        self, runtimes: dict[str, "WarmRuntime"], metrics: dict[str, int]
    ) -> None: ...


_METRIC_KEYS = ("pool_hits", "pool_misses", "cleanup_failures", "reconcile_conflicts")


class JsonFileWarmRuntimeStore:
    """Atomic JSON persistence for the warm runtime ledger."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def load(self) -> tuple[dict[str, "WarmRuntime"], dict[str, int]]:
        from opensandbox_server.services.warm_runtime_pool import _decode_state

        if not self.path.exists():
            return {}, dict.fromkeys(_METRIC_KEYS, 0)
        raw = json.loads(self.path.read_text(encoding="utf-8"))
        runtimes = _decode_state(raw.get("runtimes", {}))
        metrics = {
            key: int(raw.get("metrics", {}).get(key, 0)) for key in _METRIC_KEYS
        }
        return runtimes, metrics

    def save(
        self, runtimes: dict[str, "WarmRuntime"], metrics: dict[str, int]
    ) -> None:
        from opensandbox_server.services.warm_runtime_pool import _encode_state

        payload: dict[str, Any] = {
            "version": 1,
            "runtimes": _encode_state(runtimes),
            "metrics": {key: metrics.get(key, 0) for key in _METRIC_KEYS},
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temp_path = tempfile.mkstemp(
            dir=str(self.path.parent), prefix=".warm-pool-", suffix=".tmp"
        )
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_path, self.path)
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
