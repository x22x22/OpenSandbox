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
    ClaimLease,
    RuntimeProfile,
    WarmRuntimePool,
    WorkspaceBinding,
)
from opensandbox_server.services.warm_runtime_sqlite import SqliteWarmRuntimePool
from opensandbox_server.services.warm_runtime_store import JsonFileWarmRuntimeStore

router = APIRouter(prefix="/warm-runtimes", tags=["Warm Runtimes"])


def _lease_payload(lease: ClaimLease) -> dict[str, object]:
    return {
        "leaseId": lease.lease_id,
        "runtimeId": lease.runtime_id,
        "generation": lease.generation,
        "fencingToken": lease.fencing_token,
        "expiresAt": lease.expires_at,
    }

# 默认在内存里跑 PoC；生产由环境变量选择持久化后端，避免默认写磁盘。
#   WARM_RUNTIME_STORE=sqlite + WARM_RUNTIME_DB_PATH   -> 行级更新，推荐
#   WARM_RUNTIME_STORE=json   + WARM_RUNTIME_STORE_PATH -> 全量重写，仅小规模
_DEFAULT_SQLITE_PATH = "warm-runtimes.sqlite3"


def _build_pool() -> "WarmRuntimePool | SqliteWarmRuntimePool":
    backend = (os.getenv("WARM_RUNTIME_STORE") or "").strip().lower()
    if backend == "sqlite":
        db_path = (os.getenv("WARM_RUNTIME_DB_PATH") or "").strip() or _DEFAULT_SQLITE_PATH
        return SqliteWarmRuntimePool(db_path)
    if backend == "json":
        store_path = (os.getenv("WARM_RUNTIME_STORE_PATH") or "").strip()
        if not store_path:
            raise RuntimeError("WARM_RUNTIME_STORE=json requires WARM_RUNTIME_STORE_PATH")
        return WarmRuntimePool(store=JsonFileWarmRuntimeStore(store_path))
    return WarmRuntimePool()


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
    network_policy: str = ""

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
    return {"poolHit": True, "lease": _lease_payload(lease)}


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


@router.post("/begin-claim")
def begin_warm_runtime_claim(request: ClaimWarmRuntimeRequest) -> dict[str, object]:
    """两阶段领取第一步：READY -> CLAIMING，返回待确认 lease。"""
    lease = _pool.begin_claim(
        request.profile.to_domain(), request.binding.to_domain()
    )
    if lease is None:
        return {"poolHit": False, "lease": None}
    return {"poolHit": True, "lease": _lease_payload(lease)}


@router.post("/leases/{lease_id}/confirm")
def confirm_warm_runtime_claim(
    lease_id: str,
    fencing_token: Annotated[str, Query(min_length=1)],
) -> dict[str, object]:
    """两阶段领取第二步：CLAIMING -> LEASED。"""
    try:
        lease = _pool.confirm_claim(lease_id, fencing_token)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT, detail="lease_not_confirmable"
        ) from exc
    return {"lease": _lease_payload(lease)}


@router.post("/leases/{lease_id}/abort", status_code=status.HTTP_204_NO_CONTENT)
def abort_warm_runtime_claim(
    lease_id: str,
    fencing_token: Annotated[str, Query(min_length=1)],
    cleanup_ok: Annotated[bool, Query(alias="cleanupOk")] = True,
) -> Response:
    """放弃待确认 lease：cleanup 成功回 READY，否则进 QUARANTINE。"""
    try:
        _pool.abort_claim(lease_id, fencing_token, cleanup_ok=cleanup_ok)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT, detail="lease_not_confirmable"
        ) from exc
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post("/{runtime_id}/drain", status_code=status.HTTP_204_NO_CONTENT)
def drain_warm_runtime(runtime_id: str) -> Response:
    """让空闲 runtime 退出可领取集合（READY -> DRAINING）。"""
    try:
        _pool.drain(runtime_id)
    except KeyError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="runtime_not_found"
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT, detail="runtime_not_drainable"
        ) from exc
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.get("/status")
def warm_runtime_status() -> dict[str, object]:
    return _pool.snapshot()


@router.post("/reconcile")
def reconcile_warm_runtimes(
    request: ReconcileWarmRuntimeRequest,
    force: Annotated[bool, Query()] = False,
) -> dict[str, int]:
    try:
        quarantined = _pool.reconcile(request.runtime_ids, force=force)
    except ValueError as exc:
        # 保护触发：调用方只给了部分视图，拒绝大批量隔离健康 runtime。
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT, detail="reconcile_scope_too_broad"
        ) from exc
    return {"quarantined": quarantined}
