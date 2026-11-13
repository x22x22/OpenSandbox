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
