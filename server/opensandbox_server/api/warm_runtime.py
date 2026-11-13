"""Runtime-neutral warm pool control-plane PoC endpoints.

These endpoints manage only the ledger. Provider creation and claim-time volume
attachment remain in Docker/Kubernetes services and are intentionally separate.
"""

from __future__ import annotations

import os
from typing import Annotated

from fastapi import APIRouter, HTTPException, Query, Response, status
from pydantic import BaseModel, Field

from opensandbox_server.services.warm_runtime_pool import (
    RuntimeProfile,
    WarmRuntimePool,
    WorkspaceBinding,
)
from opensandbox_server.services.warm_runtime_store import JsonFileWarmRuntimeStore

router = APIRouter(prefix="/warm-runtimes", tags=["Warm Runtimes"])


def _build_pool() -> WarmRuntimePool:
    """Use a durable ledger when a store path is configured, else in-memory."""
    store_path = (os.getenv("WARM_RUNTIME_STORE_PATH") or "").strip()
    if not store_path:
        return WarmRuntimePool()
    return WarmRuntimePool(store=JsonFileWarmRuntimeStore(store_path))


_pool = _build_pool()


class RuntimeProfileRequest(BaseModel):
    node_id: str = Field(min_length=1)
    profile_id: str = Field(min_length=1)
    image_digest: str = Field(min_length=1)
    runtime_version: str = Field(min_length=1)
    cpu: int = Field(gt=0)
    memory_mb: int = Field(gt=0)
    policy_version: str = Field(min_length=1)
    mount_policy: str = Field(min_length=1)

    def to_domain(self) -> RuntimeProfile:
        return RuntimeProfile(**self.model_dump())


class WorkspaceBindingRequest(BaseModel):
    tenant_id: str = Field(min_length=1)
    project_id: str = Field(min_length=1)
    session_id: str = Field(min_length=1)
    workspace_volume_id: str = Field(min_length=1)
    runtime_data_volume_id: str = Field(min_length=1)
    artifacts_volume_id: str = Field(min_length=1)
    mount_signature: str = Field(min_length=1)

    def to_domain(self) -> WorkspaceBinding:
        return WorkspaceBinding(**self.model_dump())


class RegisterWarmRuntimeRequest(BaseModel):
    runtime_id: str = Field(min_length=1)
    profile: RuntimeProfileRequest


class ClaimWarmRuntimeRequest(BaseModel):
    profile: RuntimeProfileRequest
    binding: WorkspaceBindingRequest


class ReconcileWarmRuntimeRequest(BaseModel):
    runtime_ids: set[str] = Field(default_factory=set)


@router.post("/register", status_code=status.HTTP_204_NO_CONTENT)
def register_warm_runtime(request: RegisterWarmRuntimeRequest) -> Response:
    _pool.register_ready(request.runtime_id, request.profile.to_domain())
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post("/claim")
def claim_warm_runtime(request: ClaimWarmRuntimeRequest) -> dict[str, object]:
    lease = _pool.claim(request.profile.to_domain(), request.binding.to_domain())
    if lease is None:
        return {"lease": None, "poolHit": False}
    return {
        "poolHit": True,
        "lease": {
            "leaseId": lease.lease_id,
            "runtimeId": lease.runtime_id,
            "generation": lease.generation,
            "fencingToken": lease.fencing_token,
            "expiresAt": lease.expires_at,
        },
    }


@router.post("/leases/{lease_id}/release", status_code=status.HTTP_204_NO_CONTENT)
def release_warm_runtime(
    lease_id: str,
    fencing_token: Annotated[str, Query(min_length=1)],
    cleanup_ok: Annotated[bool, Query(alias="cleanupOk")],
) -> Response:
    try:
        _pool.release(lease_id, fencing_token, cleanup_ok=cleanup_ok)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="lease_conflict") from exc
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.get("/status")
def warm_runtime_status() -> dict[str, object]:
    return _pool.snapshot()


@router.post("/reconcile")
def reconcile_warm_runtimes(request: ReconcileWarmRuntimeRequest) -> dict[str, int]:
    return {"quarantined": _pool.reconcile(request.runtime_ids)}
