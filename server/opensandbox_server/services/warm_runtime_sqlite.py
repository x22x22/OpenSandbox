"""SQLite-backed warm runtime ledger with row-level updates.

The JSON store rewrites the whole ledger on every claim, so throughput decays as
the pool grows (measured: 490/s at 100 runtimes down to 110/s at 500). This store
keeps the same domain semantics but performs claim/release as single-row updates
inside a `BEGIN IMMEDIATE` transaction, and aggregates status with `GROUP BY`.

It implements the same public surface as ``WarmRuntimePool`` so callers can swap
implementations without touching route or client code.
"""

from __future__ import annotations

import dataclasses
import json
import sqlite3
import time
import uuid
from pathlib import Path
from threading import RLock

from opensandbox_server.services.warm_runtime_pool import (
    ClaimLease,
    RuntimeProfile,
    WarmRuntime,
    WarmRuntimeState,
    WorkspaceBinding,
)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS warm_runtimes (
    runtime_id TEXT PRIMARY KEY,
    profile_key TEXT NOT NULL,
    profile_json TEXT NOT NULL,
    state TEXT NOT NULL,
    generation INTEGER NOT NULL DEFAULT 0,
    lease_id TEXT,
    fencing_token TEXT,
    expires_at REAL,
    binding_json TEXT,
    updated_at REAL NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_warm_ready
    ON warm_runtimes (profile_key, state, updated_at);
CREATE INDEX IF NOT EXISTS idx_warm_lease ON warm_runtimes (lease_id);
CREATE TABLE IF NOT EXISTS warm_metrics (
    key TEXT PRIMARY KEY,
    value INTEGER NOT NULL DEFAULT 0
);
"""

_METRIC_KEYS = ("pool_hits", "pool_misses", "cleanup_failures", "reconcile_conflicts")


#: 单次 reconcile 默认允许的最大隔离数量，防止部分视图误杀健康 runtime。
DEFAULT_MAX_RECONCILE_QUARANTINE = 50


def _profile_key(profile: RuntimeProfile) -> str:
    """Stable identity for profile matching, including policy dimensions."""
    return json.dumps(
        {
            "node_id": profile.node_id,
            "profile_id": profile.profile_id,
            "image_digest": profile.image_digest,
            "runtime_version": profile.runtime_version,
            "cpu": profile.cpu,
            "memory_mb": profile.memory_mb,
            "policy_version": profile.policy_version,
            "mount_policy": profile.mount_policy,
            "network_policy": profile.network_policy,
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def _binding_to_json(binding: WorkspaceBinding) -> str:
    return json.dumps(
        {
            "tenant_id": binding.tenant_id,
            "project_id": binding.project_id,
            "session_id": binding.session_id,
            "workspace_volume_id": binding.workspace_volume_id,
            "runtime_data_volume_id": binding.runtime_data_volume_id,
            "artifacts_volume_id": binding.artifacts_volume_id,
            "mount_signature": binding.mount_signature,
        },
        sort_keys=True,
        separators=(",", ":"),
    )


class SqliteWarmRuntimePool:
    """Row-level ledger; claim cost is independent of total pool size."""

    def __init__(self, path: str | Path, *, lease_ttl_seconds: float = 300.0) -> None:
        self.lease_ttl_seconds = lease_ttl_seconds
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = RLock()
        self._conn = sqlite3.connect(
            self.path, check_same_thread=False, timeout=30.0, isolation_level=None
        )
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.executescript(_SCHEMA)
            for key in _METRIC_KEYS:
                self._conn.execute(
                    "INSERT OR IGNORE INTO warm_metrics (key, value) VALUES (?, 0)", (key,)
                )

    # ------------------------------------------------------------------ writes

    def register_ready(self, runtime_id: str, profile: RuntimeProfile) -> None:
        with self._lock, self._write():
            row = self._conn.execute(
                "SELECT state, generation FROM warm_runtimes WHERE runtime_id = ?",
                (runtime_id,),
            ).fetchone()
            if row is not None and row["state"] == WarmRuntimeState.LEASED.value:
                raise ValueError("cannot replace a leased runtime")
            generation = int(row["generation"]) if row is not None else 0
            self._conn.execute(
                """
                INSERT INTO warm_runtimes
                    (runtime_id, profile_key, profile_json, state, generation,
                     lease_id, fencing_token, expires_at, binding_json, updated_at)
                VALUES (?, ?, ?, ?, ?, NULL, NULL, NULL, NULL, ?)
                ON CONFLICT(runtime_id) DO UPDATE SET
                    profile_key=excluded.profile_key,
                    profile_json=excluded.profile_json,
                    state=excluded.state,
                    generation=excluded.generation,
                    lease_id=NULL, fencing_token=NULL, expires_at=NULL,
                    binding_json=NULL, updated_at=excluded.updated_at
                """,
                (
                    runtime_id,
                    _profile_key(profile),
                    json.dumps(dataclasses.asdict(profile), sort_keys=True),
                    WarmRuntimeState.READY.value,
                    generation,
                    time.time(),
                ),
            )

    def claim(
        self,
        profile: RuntimeProfile,
        binding: WorkspaceBinding,
        *,
        now: float | None = None,
    ) -> ClaimLease | None:
        issued_at = time.time() if now is None else now
        with self._lock, self._write():
            row = self._conn.execute(
                """
                SELECT runtime_id, generation FROM warm_runtimes
                WHERE profile_key = ? AND state = ?
                ORDER BY updated_at, runtime_id LIMIT 1
                """,
                (_profile_key(profile), WarmRuntimeState.READY.value),
            ).fetchone()
            if row is None:
                self._bump("pool_misses")
                return None

            generation = int(row["generation"]) + 1
            lease = ClaimLease(
                lease_id=f"lease-{uuid.uuid4().hex}",
                runtime_id=row["runtime_id"],
                generation=generation,
                fencing_token=uuid.uuid4().hex,
                profile=profile,
                binding=binding,
                expires_at=issued_at + self.lease_ttl_seconds,
            )
            self._conn.execute(
                """
                UPDATE warm_runtimes SET
                    state = ?, generation = ?, lease_id = ?, fencing_token = ?,
                    expires_at = ?, binding_json = ?, updated_at = ?
                WHERE runtime_id = ?
                """,
                (
                    WarmRuntimeState.LEASED.value,
                    generation,
                    lease.lease_id,
                    lease.fencing_token,
                    lease.expires_at,
                    _binding_to_json(binding),
                    issued_at,
                    lease.runtime_id,
                ),
            )
            self._bump("pool_hits")
            return lease

    def begin_claim(
        self,
        profile: RuntimeProfile,
        binding: WorkspaceBinding,
        *,
        claim_ttl_seconds: float = 60.0,
        now: float | None = None,
    ) -> ClaimLease | None:
        """第一阶段：READY -> CLAIMING，占用名额但尚未完成绑定。

        用于「领取后还要做 create-time volume attach / readiness 探测」的路径。
        调用方必须在 claim TTL 内 ``confirm_claim``，否则会被视为领取超时。
        """
        issued_at = time.time() if now is None else now
        with self._lock, self._write():
            row = self._conn.execute(
                """
                SELECT runtime_id, generation FROM warm_runtimes
                WHERE profile_key = ? AND state = ?
                ORDER BY updated_at, runtime_id LIMIT 1
                """,
                (_profile_key(profile), WarmRuntimeState.READY.value),
            ).fetchone()
            if row is None:
                self._bump("pool_misses")
                return None

            generation = int(row["generation"]) + 1
            lease = ClaimLease(
                lease_id=f"lease-{uuid.uuid4().hex}",
                runtime_id=row["runtime_id"],
                generation=generation,
                fencing_token=uuid.uuid4().hex,
                profile=profile,
                binding=binding,
                expires_at=issued_at + claim_ttl_seconds,
            )
            self._conn.execute(
                """
                UPDATE warm_runtimes SET
                    state = ?, generation = ?, lease_id = ?, fencing_token = ?,
                    expires_at = ?, binding_json = ?, updated_at = ?
                WHERE runtime_id = ?
                """,
                (
                    WarmRuntimeState.CLAIMING.value,
                    generation,
                    lease.lease_id,
                    lease.fencing_token,
                    lease.expires_at,
                    _binding_to_json(binding),
                    issued_at,
                    lease.runtime_id,
                ),
            )
            self._bump("pool_hits")
            return lease

    def confirm_claim(
        self,
        lease_id: str,
        fencing_token: str,
        *,
        lease_ttl_seconds: float = 300.0,
        now: float | None = None,
    ) -> ClaimLease:
        """第二阶段：CLAIMING -> LEASED，绑定完成后才真正占用。"""
        confirmed_at = time.time() if now is None else now
        with self._lock, self._write():
            row = self._conn.execute(
                "SELECT * FROM warm_runtimes WHERE lease_id = ?", (lease_id,)
            ).fetchone()
            if row is None or row["state"] != WarmRuntimeState.CLAIMING.value:
                raise ValueError("lease is not awaiting confirmation")
            if row["fencing_token"] != fencing_token:
                raise ValueError("fencing token mismatch")

            expires_at = confirmed_at + lease_ttl_seconds
            self._conn.execute(
                """
                UPDATE warm_runtimes SET state = ?, expires_at = ?, updated_at = ?
                WHERE runtime_id = ?
                """,
                (WarmRuntimeState.LEASED.value, expires_at, confirmed_at, row["runtime_id"]),
            )
            return ClaimLease(
                lease_id=lease_id,
                runtime_id=row["runtime_id"],
                generation=int(row["generation"]),
                fencing_token=fencing_token,
                profile=RuntimeProfile(**json.loads(row["profile_json"])),
                binding=WorkspaceBinding(**json.loads(row["binding_json"])),
                expires_at=expires_at,
            )

    def abort_claim(
        self,
        lease_id: str,
        fencing_token: str,
        *,
        cleanup_ok: bool,
    ) -> None:
        """放弃 CLAIMING：cleanup 成功回 READY，否则进 QUARANTINE。"""
        with self._lock, self._write():
            row = self._conn.execute(
                "SELECT runtime_id, state, fencing_token FROM warm_runtimes WHERE lease_id = ?",
                (lease_id,),
            ).fetchone()
            if row is None or row["state"] != WarmRuntimeState.CLAIMING.value:
                raise ValueError("lease is not awaiting confirmation")
            if row["fencing_token"] != fencing_token:
                raise ValueError("fencing token mismatch")

            if cleanup_ok:
                self._conn.execute(
                    """
                    UPDATE warm_runtimes SET
                        state = ?, generation = generation + 1,
                        lease_id = NULL, fencing_token = NULL, expires_at = NULL,
                        binding_json = NULL, updated_at = ?
                    WHERE runtime_id = ?
                    """,
                    (WarmRuntimeState.READY.value, time.time(), row["runtime_id"]),
                )
            else:
                self._conn.execute(
                    "UPDATE warm_runtimes SET state = ?, updated_at = ? WHERE runtime_id = ?",
                    (WarmRuntimeState.QUARANTINE.value, time.time(), row["runtime_id"]),
                )
                self._bump("cleanup_failures")

    def release(self, lease_id: str, fencing_token: str, *, cleanup_ok: bool) -> None:
        with self._lock, self._write():
            row = self._conn.execute(
                """
                SELECT runtime_id, state, generation, fencing_token
                FROM warm_runtimes WHERE lease_id = ?
                """,
                (lease_id,),
            ).fetchone()
            if row is None or row["state"] != WarmRuntimeState.LEASED.value:
                raise ValueError("lease is not active")
            if row["fencing_token"] != fencing_token:
                raise ValueError("fencing token mismatch")

            if cleanup_ok:
                self._conn.execute(
                    """
                    UPDATE warm_runtimes SET
                        state = ?, generation = generation + 1,
                        lease_id = NULL, fencing_token = NULL, expires_at = NULL,
                        binding_json = NULL, updated_at = ?
                    WHERE runtime_id = ?
                    """,
                    (WarmRuntimeState.READY.value, time.time(), row["runtime_id"]),
                )
            else:
                self._conn.execute(
                    """
                    UPDATE warm_runtimes SET state = ?, updated_at = ?
                    WHERE runtime_id = ?
                    """,
                    (
                        WarmRuntimeState.QUARANTINE.value,
                        time.time(),
                        row["runtime_id"],
                    ),
                )
                self._bump("cleanup_failures")

    def drain(self, runtime_id: str) -> None:
        """把空闲 runtime 置为 DRAINING，使其退出可领取集合。

        只允许 READY → DRAINING：正在被租用的 runtime 不能被抽干，否则会中断
        用户的会话；此时应等 release 之后再 drain。
        """
        with self._lock, self._write():
            row = self._conn.execute(
                "SELECT state FROM warm_runtimes WHERE runtime_id = ?", (runtime_id,)
            ).fetchone()
            if row is None:
                raise KeyError(runtime_id)
            if row["state"] != WarmRuntimeState.READY.value:
                raise ValueError(
                    f"cannot drain runtime in state {row['state']}"
                )
            self._conn.execute(
                "UPDATE warm_runtimes SET state = ?, updated_at = ? WHERE runtime_id = ?",
                (WarmRuntimeState.DRAINING.value, time.time(), runtime_id),
            )

    def expire_leases(self, *, now: float | None = None) -> int:
        current_time = time.time() if now is None else now
        with self._lock, self._write():
            cursor = self._conn.execute(
                """
                UPDATE warm_runtimes SET state = ?, updated_at = ?
                WHERE state IN (?, ?) AND expires_at IS NOT NULL AND expires_at <= ?
                """,
                (
                    WarmRuntimeState.QUARANTINE.value,
                    current_time,
                    WarmRuntimeState.LEASED.value,
                    WarmRuntimeState.CLAIMING.value,
                    current_time,
                ),
            )
            return cursor.rowcount or 0

    def reconcile(
        self,
        actual_runtime_ids: set[str],
        *,
        force: bool = False,
        max_quarantine: int = DEFAULT_MAX_RECONCILE_QUARANTINE,
    ) -> int:
        conflicts = 0
        with self._lock, self._write():
            # 先算影响面：reconcile 语义是"不在实际集合里的都隔离"，调用方若只提供
            # 部分节点/部分页的数据，就会误杀其它节点的健康实例。因此默认对大批量
            # 隔离做保护，必须显式 force 才允许。
            pending = self._count_reconcile_candidates(actual_runtime_ids)
            if pending > max_quarantine and not force:
                raise ValueError(
                    f"refusing to quarantine {pending} runtimes "
                    f"(max_quarantine={max_quarantine}); pass force=True to proceed"
                )
            if actual_runtime_ids:
                placeholders = ",".join("?" * len(actual_runtime_ids))
                cursor = self._conn.execute(
                    f"""
                    UPDATE warm_runtimes SET state = ?, updated_at = ?
                    WHERE runtime_id NOT IN ({placeholders})
                      AND state NOT IN (?, ?)
                    """,
                    (
                        WarmRuntimeState.QUARANTINE.value,
                        time.time(),
                        *sorted(actual_runtime_ids),
                        WarmRuntimeState.QUARANTINE.value,
                        WarmRuntimeState.DRAINING.value,
                    ),
                )
            else:
                cursor = self._conn.execute(
                    """
                    UPDATE warm_runtimes SET state = ?, updated_at = ?
                    WHERE state NOT IN (?, ?)
                    """,
                    (
                        WarmRuntimeState.QUARANTINE.value,
                        time.time(),
                        WarmRuntimeState.QUARANTINE.value,
                        WarmRuntimeState.DRAINING.value,
                    ),
                )
            conflicts = cursor.rowcount or 0
            if conflicts:
                self._add("reconcile_conflicts", conflicts)
        return conflicts

    def refill(self, profile: RuntimeProfile, target: int, factory) -> int:
        created = 0
        with self._lock:
            ready = self._conn.execute(
                "SELECT COUNT(*) AS c FROM warm_runtimes WHERE profile_key = ? AND state = ?",
                (_profile_key(profile), WarmRuntimeState.READY.value),
            ).fetchone()["c"]
            while ready < max(0, target):
                self.register_ready(factory(), profile)
                ready += 1
                created += 1
        return created

    # ------------------------------------------------------------------- reads

    def get(self, runtime_id: str) -> WarmRuntime:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM warm_runtimes WHERE runtime_id = ?", (runtime_id,)
            ).fetchone()
        if row is None:
            raise KeyError(runtime_id)
        return _row_to_runtime(row)

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT state, COUNT(*) AS c FROM warm_runtimes GROUP BY state"
            ).fetchall()
            metrics = {
                row["key"]: int(row["value"])
                for row in self._conn.execute("SELECT key, value FROM warm_metrics")
            }
        states = {state.value: 0 for state in WarmRuntimeState}
        for row in rows:
            states[row["state"]] = int(row["c"])
        return {
            "states": states,
            "poolHits": metrics.get("pool_hits", 0),
            "poolMisses": metrics.get("pool_misses", 0),
            "cleanupFailures": metrics.get("cleanup_failures", 0),
            "reconcileConflicts": metrics.get("reconcile_conflicts", 0),
        }

    def _count_reconcile_candidates(self, actual_runtime_ids: set[str]) -> int:
        if actual_runtime_ids:
            placeholders = ",".join("?" * len(actual_runtime_ids))
            row = self._conn.execute(
                f"""
                SELECT COUNT(*) AS c FROM warm_runtimes
                WHERE runtime_id NOT IN ({placeholders}) AND state NOT IN (?, ?)
                """,
                (
                    *sorted(actual_runtime_ids),
                    WarmRuntimeState.QUARANTINE.value,
                    WarmRuntimeState.DRAINING.value,
                ),
            ).fetchone()
        else:
            row = self._conn.execute(
                """
                SELECT COUNT(*) AS c FROM warm_runtimes WHERE state NOT IN (?, ?)
                """,
                (WarmRuntimeState.QUARANTINE.value, WarmRuntimeState.DRAINING.value),
            ).fetchone()
        return int(row["c"])

    # ------------------------------------------------------------------ helper

    def _write(self):
        from contextlib import contextmanager

        @contextmanager
        def _tx():
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                yield
            except BaseException:
                self._conn.execute("ROLLBACK")
                raise
            else:
                self._conn.execute("COMMIT")

        return _tx()

    def _bump(self, key: str) -> None:
        self._add(key, 1)

    def _add(self, key: str, delta: int) -> None:
        self._conn.execute(
            "UPDATE warm_metrics SET value = value + ? WHERE key = ?", (delta, key)
        )


def _row_to_runtime(row: sqlite3.Row) -> WarmRuntime:
    lease = None
    if row["lease_id"] is not None:
        profile = RuntimeProfile(**json.loads(row["profile_json"]))
        lease = ClaimLease(
            lease_id=row["lease_id"],
            runtime_id=row["runtime_id"],
            generation=int(row["generation"]),
            fencing_token=row["fencing_token"],
            profile=profile,
            binding=WorkspaceBinding(**json.loads(row["binding_json"])),
            expires_at=float(row["expires_at"]),
        )
    return WarmRuntime(
        runtime_id=row["runtime_id"],
        profile=RuntimeProfile(**json.loads(row["profile_json"])),
        state=WarmRuntimeState(row["state"]),
        generation=int(row["generation"]),
        lease=lease,
        updated_at=float(row["updated_at"]),
    )
