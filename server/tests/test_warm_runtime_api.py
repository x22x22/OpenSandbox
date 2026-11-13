from opensandbox_server.api import warm_runtime
from opensandbox_server.api.warm_runtime import (
    ClaimWarmRuntimeRequest,
    RegisterWarmRuntimeRequest,
    ReconcileWarmRuntimeRequest,
    RuntimeProfileRequest,
    WorkspaceBindingRequest,
    claim_warm_runtime,
    reconcile_warm_runtimes,
    register_warm_runtime,
    release_warm_runtime,
    warm_runtime_status,
)
from opensandbox_server.services.warm_runtime_pool import WarmRuntimePool


def _profile() -> RuntimeProfileRequest:
    return RuntimeProfileRequest(
        node_id="node-a",
        profile_id="medium",
        image_digest="sha256:image-v1",
        runtime_version="opencode-1.14.41",
        cpu=4,
        memory_mb=4096,
        policy_version="policy-v1",
        mount_policy="create-time-volume-attach",
    )


def _binding() -> WorkspaceBindingRequest:
    return WorkspaceBindingRequest(
        tenant_id="tenant-a",
        project_id="project-a",
        session_id="session-a",
        workspace_volume_id="workspace-a",
        runtime_data_volume_id="runtime-a",
        artifacts_volume_id="artifacts-a",
        mount_signature="mount-a",
    )


def test_warm_runtime_api_claim_release_and_status(monkeypatch):
    monkeypatch.setattr(warm_runtime, "_pool", WarmRuntimePool())
    register_warm_runtime(RegisterWarmRuntimeRequest(runtime_id="runtime-1", profile=_profile()))

    result = claim_warm_runtime(ClaimWarmRuntimeRequest(profile=_profile(), binding=_binding()))
    assert result["poolHit"] is True
    lease = result["lease"]
    assert lease["runtimeId"] == "runtime-1"

    release_warm_runtime(
        lease["leaseId"],
        fencing_token=lease["fencingToken"],
        cleanup_ok=True,
    )
    status = warm_runtime_status()
    assert status["states"]["READY"] == 1
    assert status["states"]["LEASED"] == 0


def test_warm_runtime_api_pool_miss_and_reconcile(monkeypatch):
    monkeypatch.setattr(warm_runtime, "_pool", WarmRuntimePool())

    result = claim_warm_runtime(ClaimWarmRuntimeRequest(profile=_profile(), binding=_binding()))
    assert result == {"lease": None, "poolHit": False}
    assert reconcile_warm_runtimes(ReconcileWarmRuntimeRequest(runtime_ids=set())) == {
        "quarantined": 0
    }


def test_warm_runtime_routes_register_through_fastapi(monkeypatch):
    """走真实 FastAPI 路由注册，防止 204 响应体断言在 CI 才暴露。"""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    monkeypatch.setattr(warm_runtime, "_pool", WarmRuntimePool())
    app = FastAPI()
    app.include_router(warm_runtime.router)
    client = TestClient(app)

    registered = client.post(
        "/warm-runtimes/register",
        json={"runtime_id": "runtime-1", "profile": _profile().model_dump()},
    )
    assert registered.status_code == 204
    assert registered.content == b""

    claimed = client.post(
        "/warm-runtimes/claim",
        json={"profile": _profile().model_dump(), "binding": _binding().model_dump()},
    )
    assert claimed.status_code == 200
    lease = claimed.json()["lease"]
    assert claimed.json()["poolHit"] is True

    stale = client.post(
        f"/warm-runtimes/leases/{lease['leaseId']}/release",
        params={"fencing_token": "stale", "cleanupOk": "true"},
    )
    assert stale.status_code == 409

    released = client.post(
        f"/warm-runtimes/leases/{lease['leaseId']}/release",
        params={"fencing_token": lease["fencingToken"], "cleanupOk": "true"},
    )
    assert released.status_code == 204
    assert released.content == b""
    assert client.get("/warm-runtimes/status").json()["states"]["READY"] == 1


def test_route_can_run_on_sqlite_backend(tmp_path, monkeypatch):
    """真实路由 + SQLite 行级 ledger：状态语义必须与内存实现一致。"""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from opensandbox_server.services.warm_runtime_sqlite import SqliteWarmRuntimePool

    monkeypatch.setattr(
        warm_runtime,
        "_pool",
        SqliteWarmRuntimePool(tmp_path / "route.db", lease_ttl_seconds=300),
    )
    app = FastAPI()
    app.include_router(warm_runtime.router)
    client = TestClient(app)

    assert (
        client.post(
            "/warm-runtimes/register",
            json={"runtime_id": "runtime-1", "profile": _profile().model_dump()},
        ).status_code
        == 204
    )

    claimed = client.post(
        "/warm-runtimes/claim",
        json={"profile": _profile().model_dump(), "binding": _binding().model_dump()},
    )
    assert claimed.status_code == 200
    assert claimed.json()["poolHit"] is True
    lease = claimed.json()["lease"]

    # 同一 profile 再领取必须 miss
    again = client.post(
        "/warm-runtimes/claim",
        json={"profile": _profile().model_dump(), "binding": _binding().model_dump()},
    )
    assert again.json()["poolHit"] is False

    assert (
        client.post(
            f"/warm-runtimes/leases/{lease['leaseId']}/release",
            params={"fencing_token": lease["fencingToken"], "cleanupOk": "false"},
        ).status_code
        == 204
    )
    status = client.get("/warm-runtimes/status").json()
    assert status["states"]["QUARANTINE"] == 1
    assert status["cleanupFailures"] == 1


def test_build_pool_selects_backend_from_env(tmp_path, monkeypatch):
    from opensandbox_server.services.warm_runtime_sqlite import SqliteWarmRuntimePool

    monkeypatch.setenv("WARM_RUNTIME_STORE", "sqlite")
    monkeypatch.setenv("WARM_RUNTIME_DB_PATH", str(tmp_path / "picked.db"))
    assert isinstance(warm_runtime._build_pool(), SqliteWarmRuntimePool)

    monkeypatch.setenv("WARM_RUNTIME_STORE", "json")
    monkeypatch.setenv("WARM_RUNTIME_STORE_PATH", str(tmp_path / "picked.json"))
    assert not isinstance(warm_runtime._build_pool(), SqliteWarmRuntimePool)

    monkeypatch.setenv("WARM_RUNTIME_STORE", "json")
    monkeypatch.delenv("WARM_RUNTIME_STORE_PATH")
    import pytest

    with pytest.raises(RuntimeError, match="requires WARM_RUNTIME_STORE_PATH"):
        warm_runtime._build_pool()

    monkeypatch.delenv("WARM_RUNTIME_STORE")
    assert not isinstance(warm_runtime._build_pool(), SqliteWarmRuntimePool)


def test_reconcile_rejects_scope_too_broad(monkeypatch):
    """部分视图触发批量隔离保护时必须返回 409，而不是静默误杀。"""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    monkeypatch.setattr(warm_runtime, "_pool", WarmRuntimePool())
    app = FastAPI()
    app.include_router(warm_runtime.router)
    client = TestClient(app)

    for i in range(60):
        client.post(
            "/warm-runtimes/register",
            json={"runtime_id": f"runtime-{i}", "profile": _profile().model_dump()},
        )

    rejected = client.post(
        "/warm-runtimes/reconcile", json={"runtime_ids": ["runtime-0"]}
    )
    assert rejected.status_code == 409
    assert rejected.json()["detail"] == "reconcile_scope_too_broad"
    assert client.get("/warm-runtimes/status").json()["states"]["READY"] == 60

    forced = client.post(
        "/warm-runtimes/reconcile?force=true", json={"runtime_ids": ["runtime-0"]}
    )
    assert forced.status_code == 200
    assert forced.json()["quarantined"] == 59


def test_drain_endpoint(monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    monkeypatch.setattr(warm_runtime, "_pool", WarmRuntimePool())
    app = FastAPI()
    app.include_router(warm_runtime.router)
    client = TestClient(app)

    client.post(
        "/warm-runtimes/register",
        json={"runtime_id": "runtime-1", "profile": _profile().model_dump()},
    )

    drained = client.post("/warm-runtimes/runtime-1/drain")
    assert drained.status_code == 204
    assert client.get("/warm-runtimes/status").json()["states"]["DRAINING"] == 1

    claimed = client.post(
        "/warm-runtimes/claim",
        json={"profile": _profile().model_dump(), "binding": _binding().model_dump()},
    )
    assert claimed.json()["poolHit"] is False

    again = client.post("/warm-runtimes/runtime-1/drain")
    assert again.status_code == 409
    assert again.json()["detail"] == "runtime_not_drainable"

    missing = client.post("/warm-runtimes/missing/drain")
    assert missing.status_code == 404
    assert missing.json()["detail"] == "runtime_not_found"


def test_two_phase_claim_endpoints(monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    monkeypatch.setattr(warm_runtime, "_pool", WarmRuntimePool())
    app = FastAPI()
    app.include_router(warm_runtime.router)
    client = TestClient(app)

    client.post(
        "/warm-runtimes/register",
        json={"runtime_id": "runtime-1", "profile": _profile().model_dump()},
    )

    begun = client.post(
        "/warm-runtimes/begin-claim",
        json={"profile": _profile().model_dump(), "binding": _binding().model_dump()},
    )
    assert begun.status_code == 200
    assert begun.json()["poolHit"] is True
    lease = begun.json()["lease"]
    assert client.get("/warm-runtimes/status").json()["states"]["CLAIMING"] == 1

    # CLAIMING 期间不可被再次领取
    again = client.post(
        "/warm-runtimes/begin-claim",
        json={"profile": _profile().model_dump(), "binding": _binding().model_dump()},
    )
    assert again.json()["poolHit"] is False

    stale = client.post(
        f"/warm-runtimes/leases/{lease['leaseId']}/confirm",
        params={"fencing_token": "stale"},
    )
    assert stale.status_code == 409
    assert stale.json()["detail"] == "lease_not_confirmable"

    confirmed = client.post(
        f"/warm-runtimes/leases/{lease['leaseId']}/confirm",
        params={"fencing_token": lease["fencingToken"]},
    )
    assert confirmed.status_code == 200
    assert client.get("/warm-runtimes/status").json()["states"]["LEASED"] == 1


def test_abort_claim_endpoint(monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    monkeypatch.setattr(warm_runtime, "_pool", WarmRuntimePool())
    app = FastAPI()
    app.include_router(warm_runtime.router)
    client = TestClient(app)

    client.post(
        "/warm-runtimes/register",
        json={"runtime_id": "runtime-1", "profile": _profile().model_dump()},
    )
    begun = client.post(
        "/warm-runtimes/begin-claim",
        json={"profile": _profile().model_dump(), "binding": _binding().model_dump()},
    )
    lease = begun.json()["lease"]

    aborted = client.post(
        f"/warm-runtimes/leases/{lease['leaseId']}/abort",
        params={"fencing_token": lease["fencingToken"], "cleanupOk": "true"},
    )
    assert aborted.status_code == 204
    assert client.get("/warm-runtimes/status").json()["states"]["READY"] == 1
