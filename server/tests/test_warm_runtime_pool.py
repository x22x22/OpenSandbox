from __future__ import annotations

import concurrent.futures

import pytest

from opensandbox_server.services.warm_runtime_pool import (
    RuntimeProfile,
    WarmRuntimePool,
    WarmRuntimeState,
    WorkspaceBinding,
)


def profile(name: str = "medium") -> RuntimeProfile:
    return RuntimeProfile(
        node_id="node-a",
        profile_id=name,
        image_digest="sha256:image-v1",
        runtime_version="opencode-1.14.41",
        cpu=4,
        memory_mb=4096,
        policy_version="policy-v1",
        mount_policy="create-time-volume-attach",
    )


def binding(session_id: str = "session-a") -> WorkspaceBinding:
    return WorkspaceBinding(
        tenant_id="tenant-a",
        project_id="project-a",
        session_id=session_id,
        workspace_volume_id=f"workspace-{session_id}",
        runtime_data_volume_id=f"runtime-{session_id}",
        artifacts_volume_id=f"artifacts-{session_id}",
        mount_signature=f"mount-{session_id}",
    )


def test_claim_is_atomic_and_profile_scoped() -> None:
    pool = WarmRuntimePool()
    pool.register_ready("runtime-1", profile())

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        leases = list(
            executor.map(
                lambda session: pool.claim(profile(), binding(session)),
                ("session-a", "session-b"),
            )
        )

    assert sum(lease is not None for lease in leases) == 1
    assert pool.pool_hits == 1
    assert pool.pool_misses == 1


def test_profile_mismatch_is_a_pool_miss() -> None:
    pool = WarmRuntimePool()
    pool.register_ready("runtime-1", profile())

    assert pool.claim(profile("large"), binding()) is None
    assert pool.pool_misses == 1


def test_cleanup_failure_quarantines_runtime() -> None:
    pool = WarmRuntimePool()
    pool.register_ready("runtime-1", profile())
    lease = pool.claim(profile(), binding())
    assert lease is not None

    pool.release(lease.lease_id, lease.fencing_token, cleanup_ok=False)

    assert pool.get("runtime-1").state is WarmRuntimeState.QUARANTINE
    assert pool.claim(profile(), binding("session-b")) is None
    assert pool.cleanup_failures == 1


def test_stale_fencing_token_cannot_release_new_generation() -> None:
    pool = WarmRuntimePool()
    pool.register_ready("runtime-1", profile())
    first = pool.claim(profile(), binding())
    assert first is not None
    pool.release(first.lease_id, first.fencing_token, cleanup_ok=True)
    second = pool.claim(profile(), binding("session-b"))
    assert second is not None

    with pytest.raises(ValueError, match="fencing"):
        pool.release(second.lease_id, first.fencing_token, cleanup_ok=True)


def test_expired_lease_and_reconcile_are_quarantine_transitions() -> None:
    pool = WarmRuntimePool(lease_ttl_seconds=10)
    pool.register_ready("runtime-1", profile())
    lease = pool.claim(profile(), binding(), now=100)
    assert lease is not None

    assert pool.expire_leases(now=110) == 1
    assert pool.get("runtime-1").state is WarmRuntimeState.QUARANTINE
    assert pool.reconcile(set()) == 0


def test_refill_uses_factory_and_is_bounded_by_target() -> None:
    pool = WarmRuntimePool()
    created = 0

    def factory() -> str:
        nonlocal created
        created += 1
        return f"runtime-{created}"

    assert pool.refill(profile(), 3, factory) == 3
    assert pool.refill(profile(), 3, factory) == 0
    assert created == 3
