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


def _claim_worker(args: tuple[str, str, str, bool]) -> str:
    """跨进程 worker：两个控制面进程竞争同一个 READY runtime。

    ``slow`` 会先 load 再等待，用来制造「两个进程都读到 READY」的确定性竞争窗口；
    若没有跨进程锁，两个进程都会发放同一个 runtime。
    """
    store_path, session_id, gate_path, slow = args
    import os
    import time

    from opensandbox_server.services.warm_runtime_pool import (
        RuntimeProfile,
        WarmRuntimePool,
        WorkspaceBinding,
    )
    from opensandbox_server.services.warm_runtime_store import (
        JsonFileWarmRuntimeStore,
    )

    class _SlowStore(JsonFileWarmRuntimeStore):
        def load(self):
            result = super().load()
            if slow:
                time.sleep(0.25)
            return result

    prof = RuntimeProfile(
        node_id="node-a",
        profile_id="medium",
        image_digest="sha256:image-v1",
        runtime_version="opencode-1.14.41",
        cpu=4,
        memory_mb=4096,
        policy_version="policy-v1",
        mount_policy="create-time-volume-attach",
    )
    bind = WorkspaceBinding(
        tenant_id="tenant-a",
        project_id="project-a",
        session_id=session_id,
        workspace_volume_id=f"ws-{session_id}",
        runtime_data_volume_id=f"rt-{session_id}",
        artifacts_volume_id=f"art-{session_id}",
        mount_signature=f"m-{session_id}",
    )
    pool = WarmRuntimePool(store=_SlowStore(store_path))
    deadline = time.time() + 10
    while not os.path.exists(gate_path) and time.time() < deadline:
        time.sleep(0.01)
    lease = pool.claim(prof, bind)
    return "" if lease is None else lease.runtime_id


def test_two_processes_cannot_claim_the_same_runtime(tmp_path) -> None:
    import multiprocessing

    store_path = tmp_path / "pool.json"
    gate = tmp_path / "gate"
    pool(store_path).register_ready("runtime-1", profile())

    context = multiprocessing.get_context("spawn")
    with context.Pool(processes=2) as workers:
        async_result = workers.map_async(
            _claim_worker,
            [
                (str(store_path), "session-1", str(gate), True),
                (str(store_path), "session-2", str(gate), True),
            ],
        )
        gate.write_text("go", encoding="utf-8")
        results = async_result.get(timeout=90)

    winners = [runtime_id for runtime_id in results if runtime_id]
    assert winners == ["runtime-1"], f"expected exactly one winner, got {results}"

    final = pool(store_path)
    assert final.snapshot()["states"]["LEASED"] == 1


def test_exclusive_lock_is_mutually_exclusive_across_processes(tmp_path) -> None:
    """确定性验证 flock 互斥：B 的临界区必须晚于 A 的临界区结束。"""
    import multiprocessing

    store_path = tmp_path / "pool.json"
    log_path = tmp_path / "log.txt"
    context = multiprocessing.get_context("spawn")
    with context.Pool(processes=2) as workers:
        workers.map(
            _lock_probe,
            [
                (str(store_path), str(log_path), "A", 0.0),
                (str(store_path), str(log_path), "B", 0.05),
            ],
        )

    lines = [line for line in log_path.read_text().splitlines() if line]
    assert len(lines) == 4, lines
    order = [line.split(":")[1] for line in lines]
    # A 先进入；B 只能等 A 退出后才进入
    assert order == ["A", "A", "B", "B"], lines


def _lock_probe(args: tuple[str, str, str, float]) -> None:
    import time

    from opensandbox_server.services.warm_runtime_store import (
        JsonFileWarmRuntimeStore,
    )

    store_path, log_path, label, delay = args
    store = JsonFileWarmRuntimeStore(store_path)
    time.sleep(delay)
    with store.exclusive():
        with open(log_path, "a", encoding="utf-8") as handle:
            handle.write(f"in:{label}\n")
        time.sleep(0.2)
        with open(log_path, "a", encoding="utf-8") as handle:
            handle.write(f"out:{label}\n")


def test_reconcile_refuses_mass_quarantine_from_partial_view(tmp_path) -> None:
    """回归：只提供部分实际集合时不得批量隔离健康 runtime。"""
    ledger = pool(tmp_path / "pool.json")
    for i in range(60):
        ledger.register_ready(f"runtime-{i}", profile())

    # 只见到 1 个 runtime 的部分视图，其余 59 个不得被误杀
    with pytest.raises(ValueError, match="refusing to quarantine"):
        ledger.reconcile({"runtime-0"})

    assert ledger.snapshot()["states"]["READY"] == 60
    assert ledger.snapshot()["states"]["QUARANTINE"] == 0


def test_reconcile_force_allows_mass_quarantine(tmp_path) -> None:
    ledger = pool(tmp_path / "pool.json")
    for i in range(60):
        ledger.register_ready(f"runtime-{i}", profile())

    quarantined = ledger.reconcile({"runtime-0"}, force=True)

    assert quarantined == 59
    assert ledger.get("runtime-0").state is WarmRuntimeState.READY


def test_reconcile_small_diff_still_works_without_force(tmp_path) -> None:
    ledger = pool(tmp_path / "pool.json")
    ledger.register_ready("runtime-1", profile())
    ledger.register_ready("runtime-2", profile())

    assert ledger.reconcile({"runtime-1"}) == 1
