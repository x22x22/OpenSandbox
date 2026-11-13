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


def test_network_policy_is_part_of_profile_matching() -> None:
    pool = WarmRuntimePool()
    registered = RuntimeProfile(
        node_id="node-a",
        profile_id="medium",
        image_digest="sha256:image-v1",
        runtime_version="opencode-1.14.41",
        cpu=4,
        memory_mb=4096,
        policy_version="policy-v1",
        mount_policy="create-time-volume-attach",
        network_policy="allow-all",
    )
    restricted = RuntimeProfile(
        node_id="node-a",
        profile_id="medium",
        image_digest="sha256:image-v1",
        runtime_version="opencode-1.14.41",
        cpu=4,
        memory_mb=4096,
        policy_version="policy-v1",
        mount_policy="create-time-volume-attach",
        network_policy="egress-restricted",
    )
    pool.register_ready("runtime-1", registered)

    assert pool.claim(restricted, binding()) is None
    assert pool.claim(registered, binding()) is not None


def test_drain_removes_runtime_from_claimable_set() -> None:
    pool = WarmRuntimePool()
    pool.register_ready("runtime-1", profile())

    pool.drain("runtime-1")

    assert pool.get("runtime-1").state is WarmRuntimeState.DRAINING
    assert pool.claim(profile(), binding()) is None


def test_drain_leased_runtime_is_refused() -> None:
    pool = WarmRuntimePool()
    pool.register_ready("runtime-1", profile())
    assert pool.claim(profile(), binding()) is not None

    with pytest.raises(ValueError, match="cannot drain"):
        pool.drain("runtime-1")


def test_reconcile_does_not_touch_draining_runtime() -> None:
    pool = WarmRuntimePool()
    pool.register_ready("runtime-1", profile())
    pool.drain("runtime-1")

    assert pool.reconcile(set()) == 0
    assert pool.get("runtime-1").state is WarmRuntimeState.DRAINING


def test_memory_two_phase_claim_and_timeout() -> None:
    pool = WarmRuntimePool()
    pool.register_ready("runtime-1", profile())

    pending = pool.begin_claim(profile(), binding(), claim_ttl_seconds=30)
    assert pending is not None
    assert pool.get("runtime-1").state is WarmRuntimeState.CLAIMING
    assert pool.claim(profile(), binding()) is None

    assert pool.expire_leases(now=pending.expires_at) == 1
    assert pool.get("runtime-1").state is WarmRuntimeState.QUARANTINE


def test_memory_confirm_then_release_round_trip() -> None:
    pool = WarmRuntimePool()
    pool.register_ready("runtime-1", profile())

    pending = pool.begin_claim(profile(), binding())
    assert pending is not None
    confirmed = pool.confirm_claim(pending.lease_id, pending.fencing_token)
    assert pool.get("runtime-1").state is WarmRuntimeState.LEASED

    pool.release(confirmed.lease_id, confirmed.fencing_token, cleanup_ok=True)
    assert pool.get("runtime-1").state is WarmRuntimeState.READY


def test_memory_abort_claim_paths() -> None:
    pool = WarmRuntimePool()
    pool.register_ready("runtime-1", profile())

    pending = pool.begin_claim(profile(), binding())
    assert pending is not None
    pool.abort_claim(pending.lease_id, pending.fencing_token, cleanup_ok=True)
    assert pool.get("runtime-1").state is WarmRuntimeState.READY

    with pytest.raises(ValueError, match="not awaiting confirmation"):
        pool.abort_claim(pending.lease_id, pending.fencing_token, cleanup_ok=True)


def test_ready_runtime_carries_no_tenant_binding() -> None:
    pool = WarmRuntimePool()
    pool.register_ready("runtime-1", profile())
    lease = pool.claim(profile(), binding())
    assert lease is not None

    pool.release(lease.lease_id, lease.fencing_token, cleanup_ok=True)

    restored = pool.get("runtime-1")
    assert restored.state is WarmRuntimeState.READY
    assert restored.lease is None


def test_quarantined_runtime_keeps_audit_but_is_not_reissuable() -> None:
    pool = WarmRuntimePool()
    pool.register_ready("runtime-1", profile())
    lease = pool.claim(profile(), binding())
    assert lease is not None
    pool.release(lease.lease_id, lease.fencing_token, cleanup_ok=False)

    assert pool.claim(profile(), binding()) is None
    assert pool.get("runtime-1").state is WarmRuntimeState.QUARANTINE


def test_successive_tenants_get_distinct_fencing_tokens() -> None:
    pool = WarmRuntimePool()
    pool.register_ready("runtime-1", profile())
    first = pool.claim(profile(), binding())
    assert first is not None
    pool.release(first.lease_id, first.fencing_token, cleanup_ok=True)
    second = pool.claim(profile(), binding())
    assert second is not None

    assert second.generation > first.generation
    assert second.fencing_token != first.fencing_token
