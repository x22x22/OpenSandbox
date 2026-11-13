"""Provider-neutral warm runtime claim ledger.

This module owns only control-plane state. Docker/Kubernetes services remain
responsible for creating the sandbox and attaching claim-time volumes.
"""

from __future__ import annotations

import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from threading import RLock
from typing import TYPE_CHECKING, Any, Callable, Iterator

if TYPE_CHECKING:  # pragma: no cover - typing only
    from opensandbox_server.services.warm_runtime_store import WarmRuntimeStore


class WarmRuntimeState(str, Enum):
    READY = "READY"
    CLAIMING = "CLAIMING"
    LEASED = "LEASED"
    CLEANING = "CLEANING"
    QUARANTINE = "QUARANTINE"
    DRAINING = "DRAINING"


@dataclass(frozen=True, slots=True)
class RuntimeProfile:
    node_id: str
    profile_id: str
    image_digest: str
    runtime_version: str
    cpu: int
    memory_mb: int
    policy_version: str
    mount_policy: str
    network_policy: str = ""


@dataclass(frozen=True, slots=True)
class WorkspaceBinding:
    tenant_id: str
    project_id: str
    session_id: str
    workspace_volume_id: str
    runtime_data_volume_id: str
    artifacts_volume_id: str
    mount_signature: str


@dataclass(frozen=True, slots=True)
class ClaimLease:
    lease_id: str
    runtime_id: str
    generation: int
    fencing_token: str
    profile: RuntimeProfile
    binding: WorkspaceBinding
    expires_at: float


@dataclass(slots=True)
class WarmRuntime:
    runtime_id: str
    profile: RuntimeProfile
    state: WarmRuntimeState = WarmRuntimeState.READY
    generation: int = 0
    lease: ClaimLease | None = None
    updated_at: float = 0.0
    #: 槽位实际绑定的容器/沙箱 id。暖模板本身没有容器，领取后创建出来再回填。
    bound_runtime_id: str | None = None


class WarmRuntimePool:
    """Thread-safe ledger with atomic claim and bounded lifecycle transitions."""

    def __init__(
        self,
        *,
        lease_ttl_seconds: float = 300.0,
        store: "WarmRuntimeStore | None" = None,
    ) -> None:
        self.lease_ttl_seconds = lease_ttl_seconds
        self._store = store
        self._lock = RLock()
        self._runtimes: dict[str, WarmRuntime] = {}
        self.pool_hits = 0
        self.pool_misses = 0
        self.cleanup_failures = 0
        self.reconcile_conflicts = 0
        if store is not None:
            runtimes, metrics = store.load()
            self._runtimes = runtimes
            self.pool_hits = metrics.get("pool_hits", 0)
            self.pool_misses = metrics.get("pool_misses", 0)
            self.cleanup_failures = metrics.get("cleanup_failures", 0)
            self.reconcile_conflicts = metrics.get("reconcile_conflicts", 0)

    def _metrics(self) -> dict[str, int]:
        return {
            "pool_hits": self.pool_hits,
            "pool_misses": self.pool_misses,
            "cleanup_failures": self.cleanup_failures,
            "reconcile_conflicts": self.reconcile_conflicts,
        }

    def _persist(self) -> None:
        if self._store is None:
            return
        self._store.save(self._runtimes, self._metrics())

    def _reload(self) -> None:
        if self._store is None:
            return
        runtimes, metrics = self._store.load()
        self._runtimes = runtimes
        self.pool_hits = metrics.get("pool_hits", 0)
        self.pool_misses = metrics.get("pool_misses", 0)
        self.cleanup_failures = metrics.get("cleanup_failures", 0)
        self.reconcile_conflicts = metrics.get("reconcile_conflicts", 0)

    @contextmanager
    def _transaction(self) -> Iterator[None]:
        """带跨进程锁的读-改-写事务。

        有 store 时先取排他锁并重新加载最新状态，退出时落盘；没有 store 时
        退化为原有的进程内 ``RLock`` 语义。
        """
        with self._lock:
            if self._store is None:
                yield
                return
            with self._store.exclusive():
                self._reload()
                yield
                self._persist()

    def register_ready(self, runtime_id: str, profile: RuntimeProfile) -> None:
        with self._transaction():
            current = self._runtimes.get(runtime_id)
            if current and current.state is WarmRuntimeState.LEASED:
                raise ValueError("cannot replace a leased runtime")
            generation = current.generation if current else 0
            self._runtimes[runtime_id] = WarmRuntime(
                runtime_id=runtime_id,
                profile=profile,
                generation=generation,
                updated_at=time.time(),
            )

    def claim(
        self,
        profile: RuntimeProfile,
        binding: WorkspaceBinding,
        *,
        now: float | None = None,
    ) -> ClaimLease | None:
        with self._transaction():
            current = next(
                (
                    runtime
                    for runtime in self._runtimes.values()
                    if runtime.state is WarmRuntimeState.READY
                    and runtime.profile == profile
                ),
                None,
            )
            if current is None:
                self.pool_misses += 1
                return None
            self.pool_hits += 1
            current.state = WarmRuntimeState.CLAIMING
            current.generation += 1
            issued_at = time.time() if now is None else now
            lease = ClaimLease(
                lease_id=f"lease-{uuid.uuid4().hex}",
                runtime_id=current.runtime_id,
                generation=current.generation,
                fencing_token=uuid.uuid4().hex,
                profile=profile,
                binding=binding,
                expires_at=issued_at + self.lease_ttl_seconds,
            )
            current.lease = lease
            current.state = WarmRuntimeState.LEASED
            current.updated_at = issued_at
            return lease

    def begin_claim(
        self,
        profile: RuntimeProfile,
        binding: WorkspaceBinding,
        *,
        claim_ttl_seconds: float = 60.0,
        now: float | None = None,
    ) -> ClaimLease | None:
        """第一阶段：READY -> CLAIMING，占用名额但尚未完成绑定。"""
        issued_at = time.time() if now is None else now
        with self._transaction():
            current = next(
                (
                    runtime
                    for runtime in self._runtimes.values()
                    if runtime.state is WarmRuntimeState.READY
                    and runtime.profile == profile
                ),
                None,
            )
            if current is None:
                self.pool_misses += 1
                return None
            self.pool_hits += 1
            current.generation += 1
            lease = ClaimLease(
                lease_id=f"lease-{uuid.uuid4().hex}",
                runtime_id=current.runtime_id,
                generation=current.generation,
                fencing_token=uuid.uuid4().hex,
                profile=profile,
                binding=binding,
                expires_at=issued_at + claim_ttl_seconds,
            )
            current.lease = lease
            current.state = WarmRuntimeState.CLAIMING
            current.updated_at = issued_at
            return lease

    def confirm_claim(
        self,
        lease_id: str,
        fencing_token: str,
        *,
        lease_ttl_seconds: float = 300.0,
        now: float | None = None,
    ) -> ClaimLease:
        """第二阶段：CLAIMING -> LEASED。"""
        confirmed_at = time.time() if now is None else now
        with self._transaction():
            runtime = self._find_claiming_lease(lease_id)
            if runtime.state is not WarmRuntimeState.CLAIMING:
                raise ValueError("lease is not awaiting confirmation")
            if runtime.lease is None or runtime.lease.fencing_token != fencing_token:
                raise ValueError("fencing token mismatch")
            confirmed = ClaimLease(
                lease_id=runtime.lease.lease_id,
                runtime_id=runtime.runtime_id,
                generation=runtime.lease.generation,
                fencing_token=fencing_token,
                profile=runtime.lease.profile,
                binding=runtime.lease.binding,
                expires_at=confirmed_at + lease_ttl_seconds,
            )
            runtime.lease = confirmed
            runtime.state = WarmRuntimeState.LEASED
            runtime.updated_at = confirmed_at
            return confirmed

    def abort_claim(self, lease_id: str, fencing_token: str, *, cleanup_ok: bool) -> None:
        """放弃 CLAIMING：cleanup 成功回 READY，否则进 QUARANTINE。"""
        with self._transaction():
            runtime = self._find_claiming_lease(lease_id)
            if runtime.state is not WarmRuntimeState.CLAIMING:
                raise ValueError("lease is not awaiting confirmation")
            if runtime.lease is None or runtime.lease.fencing_token != fencing_token:
                raise ValueError("fencing token mismatch")
            if cleanup_ok:
                runtime.generation += 1
                runtime.lease = None
                runtime.state = WarmRuntimeState.READY
            else:
                self.cleanup_failures += 1
                runtime.state = WarmRuntimeState.QUARANTINE
            runtime.updated_at = time.time()

    def release(self, lease_id: str, fencing_token: str, *, cleanup_ok: bool) -> None:
        with self._transaction():
            runtime = self._find_lease(lease_id)
            if runtime.lease is None or runtime.lease.fencing_token != fencing_token:
                raise ValueError("fencing token mismatch")
            runtime.state = WarmRuntimeState.CLEANING
            if cleanup_ok:
                runtime.generation += 1
                runtime.lease = None
                runtime.state = WarmRuntimeState.READY
            else:
                self.cleanup_failures += 1
                runtime.state = WarmRuntimeState.QUARANTINE
            runtime.updated_at = time.time()

    def bind_runtime(
        self, lease_id: str, fencing_token: str, bound_runtime_id: str
    ) -> None:
        """把租约槽位绑定到真实创建出来的容器 id。

        暖模板没有容器，领取后由调用方创建租户容器，再把它的 id 回填到这里，
        这样 release/cleanup 才知道该清理哪一个真实运行时。
        """
        if not bound_runtime_id:
            raise ValueError("bound runtime id is required")
        with self._transaction():
            runtime = self._find_claiming_lease(lease_id)
            if runtime.lease is None or runtime.lease.fencing_token != fencing_token:
                raise ValueError("fencing token mismatch")
            if runtime.state not in {
                WarmRuntimeState.CLAIMING,
                WarmRuntimeState.LEASED,
            }:
                raise ValueError(
                    f"cannot bind runtime in state {runtime.state.value}"
                )
            runtime.bound_runtime_id = bound_runtime_id
            runtime.updated_at = time.time()

    def drain(self, runtime_id: str) -> None:
        """把空闲 runtime 置为 DRAINING，使其退出可领取集合（仅 READY 可 drain）。"""
        with self._transaction():
            runtime = self._runtimes.get(runtime_id)
            if runtime is None:
                raise KeyError(runtime_id)
            if runtime.state is not WarmRuntimeState.READY:
                raise ValueError(f"cannot drain runtime in state {runtime.state.value}")
            runtime.state = WarmRuntimeState.DRAINING
            runtime.updated_at = time.time()

    def expire_leases(self, *, now: float | None = None) -> int:
        current_time = time.time() if now is None else now
        expired = 0
        with self._transaction():
            for runtime in self._runtimes.values():
                if (
                    runtime.state
                    in {WarmRuntimeState.LEASED, WarmRuntimeState.CLAIMING}
                    and runtime.lease is not None
                    and runtime.lease.expires_at <= current_time
                ):
                    runtime.state = WarmRuntimeState.QUARANTINE
                    expired += 1
        return expired

    def reconcile(
        self,
        actual_runtime_ids: set[str],
        *,
        actual_bound_runtime_ids: set[str] | None = None,
        force: bool = False,
        max_quarantine: int = 50,
    ) -> int:
        conflicts = 0
        with self._transaction():
            pending = sum(
                1
                for runtime in self._runtimes.values()
                if runtime.runtime_id not in actual_runtime_ids
                and runtime.state
                not in {WarmRuntimeState.QUARANTINE, WarmRuntimeState.DRAINING}
            )
            if pending > max_quarantine and not force:
                raise ValueError(
                    f"refusing to quarantine {pending} runtimes "
                    f"(max_quarantine={max_quarantine}); pass force=True to proceed"
                )
            for runtime in self._runtimes.values():
                if runtime.runtime_id not in actual_runtime_ids and runtime.state not in {
                    WarmRuntimeState.QUARANTINE,
                    WarmRuntimeState.DRAINING,
                }:
                    runtime.state = WarmRuntimeState.QUARANTINE
                    conflicts += 1
            if actual_bound_runtime_ids is not None:
                for runtime in self._runtimes.values():
                    if runtime.bound_runtime_id is None:
                        continue
                    if runtime.state in {
                        WarmRuntimeState.QUARANTINE,
                        WarmRuntimeState.DRAINING,
                    }:
                        continue
                    if runtime.bound_runtime_id not in actual_bound_runtime_ids:
                        runtime.state = WarmRuntimeState.QUARANTINE
                        runtime.updated_at = time.time()
                        conflicts += 1

            self.reconcile_conflicts += conflicts
        return conflicts

    def refill(self, profile: RuntimeProfile, target: int, factory: Callable[[], str]) -> int:
        created = 0
        with self._transaction():
            ready = sum(
                runtime.state is WarmRuntimeState.READY and runtime.profile == profile
                for runtime in self._runtimes.values()
            )
            while ready < max(0, target):
                self.register_ready(factory(), profile)
                ready += 1
                created += 1
        return created

    def get(self, runtime_id: str) -> WarmRuntime:
        with self._lock:
            self._reload()
            return self._runtimes[runtime_id]

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            self._reload()
            states = {state.value: 0 for state in WarmRuntimeState}
            for runtime in self._runtimes.values():
                states[runtime.state.value] += 1
            return {
                "states": states,
                "poolHits": self.pool_hits,
                "poolMisses": self.pool_misses,
                "cleanupFailures": self.cleanup_failures,
                "reconcileConflicts": self.reconcile_conflicts,
            }

    def _find_claiming_lease(self, lease_id: str) -> WarmRuntime:
        """与 SQLite 实现保持同一错误语义：不存在或已结束都视为未在等待确认。"""
        try:
            return self._find_lease(lease_id)
        except ValueError as exc:
            raise ValueError("lease is not awaiting confirmation") from exc

    def _find_lease(self, lease_id: str) -> WarmRuntime:
        for runtime in self._runtimes.values():
            if runtime.lease and runtime.lease.lease_id == lease_id:
                return runtime
        raise ValueError("lease is not active")


def _encode_state(runtimes: dict[str, WarmRuntime]) -> dict[str, Any]:
    """Serialize ledger state to primitives (no secrets, no host paths)."""
    encoded: dict[str, Any] = {}
    for runtime_id, runtime in runtimes.items():
        lease = runtime.lease
        encoded[runtime_id] = {
            "profile": {
                "node_id": runtime.profile.node_id,
                "profile_id": runtime.profile.profile_id,
                "image_digest": runtime.profile.image_digest,
                "runtime_version": runtime.profile.runtime_version,
                "cpu": runtime.profile.cpu,
                "memory_mb": runtime.profile.memory_mb,
                "policy_version": runtime.profile.policy_version,
                "mount_policy": runtime.profile.mount_policy,
                "network_policy": runtime.profile.network_policy,
            },
            "state": runtime.state.value,
            "generation": runtime.generation,
            "updated_at": runtime.updated_at,
            "bound_runtime_id": runtime.bound_runtime_id,
            "lease": None
            if lease is None
            else {
                "lease_id": lease.lease_id,
                "generation": lease.generation,
                "fencing_token": lease.fencing_token,
                "expires_at": lease.expires_at,
                "binding": {
                    "tenant_id": lease.binding.tenant_id,
                    "project_id": lease.binding.project_id,
                    "session_id": lease.binding.session_id,
                    "workspace_volume_id": lease.binding.workspace_volume_id,
                    "runtime_data_volume_id": lease.binding.runtime_data_volume_id,
                    "artifacts_volume_id": lease.binding.artifacts_volume_id,
                    "mount_signature": lease.binding.mount_signature,
                },
            },
        }
    return encoded


def _decode_state(encoded: dict[str, Any]) -> dict[str, WarmRuntime]:
    runtimes: dict[str, WarmRuntime] = {}
    for runtime_id, payload in encoded.items():
        profile = RuntimeProfile(**payload["profile"])
        lease_payload = payload.get("lease")
        lease = None
        if lease_payload:
            lease = ClaimLease(
                lease_id=lease_payload["lease_id"],
                runtime_id=runtime_id,
                generation=int(lease_payload["generation"]),
                fencing_token=lease_payload["fencing_token"],
                profile=profile,
                binding=WorkspaceBinding(**lease_payload["binding"]),
                expires_at=float(lease_payload["expires_at"]),
            )
        runtimes[runtime_id] = WarmRuntime(
            runtime_id=runtime_id,
            profile=profile,
            state=WarmRuntimeState(payload["state"]),
            generation=int(payload["generation"]),
            lease=lease,
            updated_at=float(payload.get("updated_at", 0.0)),
            bound_runtime_id=payload.get("bound_runtime_id"),
        )
    return runtimes
