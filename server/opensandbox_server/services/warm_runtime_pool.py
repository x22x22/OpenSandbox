"""Provider-neutral warm runtime claim ledger.

This module owns only control-plane state. Docker/Kubernetes services remain
responsible for creating the sandbox and attaching claim-time volumes.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from enum import Enum
from threading import RLock
from typing import Callable


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


class WarmRuntimePool:
    """Thread-safe ledger with atomic claim and bounded lifecycle transitions."""

    def __init__(self, *, lease_ttl_seconds: float = 300.0) -> None:
        self.lease_ttl_seconds = lease_ttl_seconds
        self._runtimes: dict[str, WarmRuntime] = {}
        self._lock = RLock()
        self.pool_hits = 0
        self.pool_misses = 0
        self.cleanup_failures = 0
        self.reconcile_conflicts = 0

    def register_ready(self, runtime_id: str, profile: RuntimeProfile) -> None:
        with self._lock:
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
        with self._lock:
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

    def release(self, lease_id: str, fencing_token: str, *, cleanup_ok: bool) -> None:
        with self._lock:
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

    def expire_leases(self, *, now: float | None = None) -> int:
        current_time = time.time() if now is None else now
        expired = 0
        with self._lock:
            for runtime in self._runtimes.values():
                if (
                    runtime.state is WarmRuntimeState.LEASED
                    and runtime.lease is not None
                    and runtime.lease.expires_at <= current_time
                ):
                    runtime.state = WarmRuntimeState.QUARANTINE
                    expired += 1
        return expired

    def reconcile(self, actual_runtime_ids: set[str]) -> int:
        conflicts = 0
        with self._lock:
            for runtime in self._runtimes.values():
                if runtime.runtime_id not in actual_runtime_ids and runtime.state not in {
                    WarmRuntimeState.QUARANTINE,
                    WarmRuntimeState.DRAINING,
                }:
                    runtime.state = WarmRuntimeState.QUARANTINE
                    conflicts += 1
            self.reconcile_conflicts += conflicts
        return conflicts

    def refill(self, profile: RuntimeProfile, target: int, factory: Callable[[], str]) -> int:
        created = 0
        with self._lock:
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
            return self._runtimes[runtime_id]

    def snapshot(self) -> dict[str, object]:
        with self._lock:
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

    def _find_lease(self, lease_id: str) -> WarmRuntime:
        for runtime in self._runtimes.values():
            if runtime.lease and runtime.lease.lease_id == lease_id:
                return runtime
        raise ValueError("lease is not active")
