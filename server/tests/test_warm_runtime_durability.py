from __future__ import annotations

import pytest

from opensandbox_server.services.warm_runtime_pool import (
    RuntimeProfile,
    WarmRuntimePool,
    WarmRuntimeState,
    WorkspaceBinding,
)
from opensandbox_server.services.warm_runtime_store import JsonFileWarmRuntimeStore


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


def binding(session: str = "session-a") -> WorkspaceBinding:
    return WorkspaceBinding(
        tenant_id="tenant-a",
        project_id="project-a",
        session_id=session,
        workspace_volume_id=f"ws-{session}",
        runtime_data_volume_id=f"rt-{session}",
        artifacts_volume_id=f"art-{session}",
        mount_signature=f"m-{session}",
    )


def pool(path) -> WarmRuntimePool:
    return WarmRuntimePool(store=JsonFileWarmRuntimeStore(path), lease_ttl_seconds=300)


def test_restart_preserves_lease_and_generation(tmp_path) -> None:
    store_path = tmp_path / "pool.json"

    first = pool(store_path)
    first.register_ready("runtime-1", profile())
    lease = first.claim(profile(), binding("s1"))
    assert lease is not None

    # simulate control-plane restart: brand new pool instance over the same store
    restarted = pool(store_path)
    restored = restarted.get("runtime-1")

    assert restored.state is WarmRuntimeState.LEASED
    assert restored.generation == lease.generation
    assert restored.lease is not None
    assert restored.lease.lease_id == lease.lease_id
    assert restored.lease.fencing_token == lease.fencing_token
    assert restored.lease.binding.session_id == "s1"


def test_restart_does_not_reissue_live_lease(tmp_path) -> None:
    store_path = tmp_path / "pool.json"

    first = pool(store_path)
    first.register_ready("runtime-1", profile())
    lease = first.claim(profile(), binding("s1"))
    assert lease is not None

    restarted = pool(store_path)

    assert restarted.claim(profile(), binding("s2")) is None
    assert restarted.snapshot()["states"]["LEASED"] == 1


def test_restart_keeps_stale_fencing_rejected(tmp_path) -> None:
    store_path = tmp_path / "pool.json"

    first = pool(store_path)
    first.register_ready("runtime-1", profile())
    lease = first.claim(profile(), binding("s1"))
    assert lease is not None
    first.release(lease.lease_id, lease.fencing_token, cleanup_ok=True)
    second = first.claim(profile(), binding("s2"))
    assert second is not None

    restarted = pool(store_path)
    with pytest.raises(ValueError, match="fencing"):
        restarted.release(second.lease_id, lease.fencing_token, cleanup_ok=True)


def test_restart_then_reconcile_quarantines_missing_runtime(tmp_path) -> None:
    store_path = tmp_path / "pool.json"

    first = pool(store_path)
    first.register_ready("runtime-1", profile())
    first.register_ready("runtime-2", profile())

    restarted = pool(store_path)
    # only runtime-1 still exists in the real world after restart
    assert restarted.reconcile({"runtime-1"}) == 1

    assert restarted.get("runtime-2").state is WarmRuntimeState.QUARANTINE
    assert restarted.get("runtime-1").state is WarmRuntimeState.READY


def test_restart_preserves_quarantine_and_metrics(tmp_path) -> None:
    store_path = tmp_path / "pool.json"

    first = pool(store_path)
    first.register_ready("runtime-1", profile())
    lease = first.claim(profile(), binding("s1"))
    assert lease is not None
    first.release(lease.lease_id, lease.fencing_token, cleanup_ok=False)

    restarted = pool(store_path)

    assert restarted.get("runtime-1").state is WarmRuntimeState.QUARANTINE
    snapshot = restarted.snapshot()
    assert snapshot["cleanupFailures"] == 1
    assert snapshot["poolHits"] == 1


def test_store_file_contains_no_secret_material(tmp_path) -> None:
    store_path = tmp_path / "pool.json"
    first = pool(store_path)
    first.register_ready("runtime-1", profile())
    first.claim(profile(), binding("s1"))

    text = store_path.read_text(encoding="utf-8")

    assert "api_key" not in text.lower()
    assert "password" not in text.lower()
    assert "token" not in text.lower() or "fencing_token" in text
    assert "runtime-1" in text
