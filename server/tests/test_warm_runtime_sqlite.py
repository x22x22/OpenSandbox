from __future__ import annotations

import pytest

from opensandbox_server.services.warm_runtime_pool import (
    RuntimeProfile,
    WarmRuntimeState,
    WorkspaceBinding,
)
from opensandbox_server.services.warm_runtime_sqlite import SqliteWarmRuntimePool


def profile(name: str = "medium", network: str = "") -> RuntimeProfile:
    return RuntimeProfile(
        node_id="node-a",
        profile_id=name,
        image_digest="sha256:image-v1",
        runtime_version="opencode-1.14.41",
        cpu=4,
        memory_mb=4096,
        policy_version="policy-v1",
        mount_policy="create-time-volume-attach",
        network_policy=network,
    )


def binding(session: str = "session-a") -> WorkspaceBinding:
    return WorkspaceBinding(
        tenant_id="tenant-a",
        project_id=f"project-{session}",
        session_id=session,
        workspace_volume_id=f"ws-{session}",
        runtime_data_volume_id=f"rt-{session}",
        artifacts_volume_id=f"art-{session}",
        mount_signature=f"m-{session}",
    )


def pool(tmp_path) -> SqliteWarmRuntimePool:
    return SqliteWarmRuntimePool(tmp_path / "pool.db", lease_ttl_seconds=300)


def test_claim_is_atomic_and_profile_scoped(tmp_path) -> None:
    ledger = pool(tmp_path)
    ledger.register_ready("runtime-1", profile())

    first = ledger.claim(profile(), binding("s1"))
    second = ledger.claim(profile(), binding("s2"))

    assert first is not None
    assert second is None
    assert ledger.snapshot()["states"]["LEASED"] == 1
    assert ledger.snapshot()["poolMisses"] == 1


def test_profile_mismatch_is_a_miss(tmp_path) -> None:
    ledger = pool(tmp_path)
    ledger.register_ready("runtime-1", profile(network="allow-all"))

    assert ledger.claim(profile(network="egress-restricted"), binding()) is None
    assert ledger.claim(profile(network="allow-all"), binding()) is not None


def test_cleanup_failure_quarantines_and_blocks_reissue(tmp_path) -> None:
    ledger = pool(tmp_path)
    ledger.register_ready("runtime-1", profile())
    lease = ledger.claim(profile(), binding("s1"))
    assert lease is not None

    ledger.release(lease.lease_id, lease.fencing_token, cleanup_ok=False)

    assert ledger.get("runtime-1").state is WarmRuntimeState.QUARANTINE
    assert ledger.claim(profile(), binding("s2")) is None
    assert ledger.snapshot()["cleanupFailures"] == 1


def test_release_success_bumps_generation(tmp_path) -> None:
    ledger = pool(tmp_path)
    ledger.register_ready("runtime-1", profile())
    lease = ledger.claim(profile(), binding("s1"))
    assert lease is not None

    ledger.release(lease.lease_id, lease.fencing_token, cleanup_ok=True)

    restored = ledger.get("runtime-1")
    assert restored.state is WarmRuntimeState.READY
    assert restored.generation == lease.generation + 1
    assert restored.lease is None


def test_stale_fencing_token_is_rejected(tmp_path) -> None:
    ledger = pool(tmp_path)
    ledger.register_ready("runtime-1", profile())
    first = ledger.claim(profile(), binding("s1"))
    assert first is not None
    ledger.release(first.lease_id, first.fencing_token, cleanup_ok=True)
    second = ledger.claim(profile(), binding("s2"))
    assert second is not None

    with pytest.raises(ValueError, match="fencing"):
        ledger.release(second.lease_id, first.fencing_token, cleanup_ok=True)


def test_lease_expiration_boundary(tmp_path) -> None:
    ledger = pool(tmp_path)
    ledger.register_ready("runtime-1", profile())
    lease = ledger.claim(profile(), binding(), now=100.0)
    assert lease is not None

    assert ledger.expire_leases(now=lease.expires_at - 0.001) == 0
    assert ledger.get("runtime-1").state is WarmRuntimeState.LEASED
    assert ledger.expire_leases(now=lease.expires_at) == 1
    assert ledger.get("runtime-1").state is WarmRuntimeState.QUARANTINE


def test_reconcile_only_touches_missing_runtimes(tmp_path) -> None:
    ledger = pool(tmp_path)
    ledger.register_ready("runtime-1", profile())
    ledger.register_ready("runtime-2", profile())

    assert ledger.reconcile({"runtime-1"}) == 1

    assert ledger.get("runtime-1").state is WarmRuntimeState.READY
    assert ledger.get("runtime-2").state is WarmRuntimeState.QUARANTINE
    assert ledger.snapshot()["reconcileConflicts"] == 1


def test_restart_keeps_lease_and_metrics(tmp_path) -> None:
    path = tmp_path / "pool.db"
    first = SqliteWarmRuntimePool(path, lease_ttl_seconds=300)
    first.register_ready("runtime-1", profile())
    lease = first.claim(profile(), binding("s1"))
    assert lease is not None

    restarted = SqliteWarmRuntimePool(path, lease_ttl_seconds=300)
    restored = restarted.get("runtime-1")

    assert restored.state is WarmRuntimeState.LEASED
    assert restored.lease is not None
    assert restored.lease.lease_id == lease.lease_id
    assert restored.lease.fencing_token == lease.fencing_token
    assert restarted.claim(profile(), binding("s2")) is None
    assert restarted.snapshot()["poolHits"] == 1


def test_refill_is_bounded_by_target(tmp_path) -> None:
    ledger = pool(tmp_path)
    created = 0

    def factory() -> str:
        nonlocal created
        created += 1
        return f"runtime-{created}"

    assert ledger.refill(profile(), 3, factory) == 3
    assert ledger.refill(profile(), 3, factory) == 0
    assert ledger.snapshot()["states"]["READY"] == 3


def test_snapshot_does_not_scale_with_row_count(tmp_path) -> None:
    """快照按状态聚合，不是把全部行读进内存再计数。"""
    ledger = pool(tmp_path)
    for i in range(50):
        ledger.register_ready(f"runtime-{i}", profile())

    snapshot = ledger.snapshot()

    assert snapshot["states"]["READY"] == 50
    # 只应有状态数量级的行（GROUP BY 结果），而不是 50 行
    rows = ledger._conn.execute(
        "SELECT state, COUNT(*) FROM warm_runtimes GROUP BY state"
    ).fetchall()
    assert len(rows) == 1
