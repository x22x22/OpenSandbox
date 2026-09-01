"""沙箱 OOM 证据暴露：SandboxStatus 新字段 + cgroup oom_kill 读取（TDD）。

prod2 2026-08-31 实测：沙箱容器 cgroup OOM（CONSTRAINT_MEMCG）杀掉容器内
opencode 进程（容器本身存活），State.OOMKilled 检测不到，必须读 cgroup
memory.events 的 oom_kill 计数才能证明进程级 OOM。

test5 2026-09-01 回归：v1 兜底 exec 成功（exit_code=0，输出 oom_kill 3）时
旧实现 ``int(getattr(...,-1) or -1)`` 把 0 错判为 -1 → 返回 None，导致
oomKillCount 永远为空；修复 exit_code 解析后 v1/v2 均正确返回计数。
"""

from __future__ import annotations

from threading import Lock
from types import SimpleNamespace

from opensandbox_server.api.schema import SandboxStatus
from opensandbox_server.services.docker.docker_service import DockerSandboxService


def _service_without_init() -> DockerSandboxService:
    """绕过 __init__（会连 Docker daemon），只挂 _read_oom_kill_count 需要的属性。"""
    svc = object.__new__(DockerSandboxService)
    svc._oom_kill_cache_lock = Lock()
    svc._oom_kill_cache = {}
    svc._oom_kill_cache_ttl_seconds = 5.0
    return svc


def _container_with_exec(exec_results) -> SimpleNamespace:
    """构造带 id 与 exec_run 的伪容器；exec_run 依次返回给定结果。"""
    it = iter(exec_results)

    def exec_run(*_args, **_kwargs):
        return next(it)

    return SimpleNamespace(id="c0ffee1234", exec_run=exec_run)


def _exec_result(exit_code: int, output: bytes) -> SimpleNamespace:
    return SimpleNamespace(exit_code=exit_code, output=output)


# ---------------------------------------------------------------------------
# _read_oom_kill_count 解析（2026-09-01 回归：exit_code=0 被 or -1 误判为失败）
# ---------------------------------------------------------------------------

def test_read_oom_kill_count_v1_success_exit_zero() -> None:
    """v1 兜底：exec 成功 exit_code=0，输出 oom_kill 3 → 返回 3（曾误判为 None）。"""
    svc = _service_without_init()
    container = _container_with_exec(
        [
            # 第一个 exec（memory.events）在 v1 主机上不存在 → exit 1 无输出
            _exec_result(1, b"cat: /sys/fs/cgroup/memory.events: No such file or directory\n"),
            # 第二个 exec（memory.oom_control）成功 → exit 0，输出 oom_kill 3
            _exec_result(0, b"oom_kill_disable 0\nunder_oom 0\noom_kill 3\noom_kill_local 3\n"),
        ]
    )
    assert svc._read_oom_kill_count(container) == 3


def test_read_oom_kill_count_v2_success_exit_zero() -> None:
    """v2 主机：memory.events exit 0 输出 oom_kill 2 → 返回 2，不再走兜底。"""
    svc = _service_without_init()
    container = _container_with_exec(
        [_exec_result(0, b"oom_kill 2\noom_kill_current 2\noom_group_kill 0\n")]
    )
    assert svc._read_oom_kill_count(container) == 2


def test_read_oom_kill_count_zero_is_valid() -> None:
    """未发生 OOM：exit 0 输出 oom_kill 0 → 返回 0（不是 None）。"""
    svc = _service_without_init()
    container = _container_with_exec(
        [_exec_result(0, b"oom_kill 0\n")]
    )
    assert svc._read_oom_kill_count(container) == 0


def test_read_oom_kill_count_degrades_to_none_on_failure() -> None:
    """exec 失败（容器刚终止等）→ 返回 None，调用方优雅降级。"""
    svc = _service_without_init()
    container = _container_with_exec(
        [_exec_result(1, b"error\n"), _exec_result(1, b"error\n")]
    )
    assert svc._read_oom_kill_count(container) is None


def test_read_oom_kill_count_caches_short_ttl() -> None:
    """5 秒 TTL 内重复查询走缓存，不再 exec。"""
    svc = _service_without_init()
    calls = []

    def exec_run(*_args, **_kwargs):
        calls.append(1)
        return _exec_result(0, b"oom_kill 1\n")

    container = SimpleNamespace(id="c0ffee5678", exec_run=exec_run)
    assert svc._read_oom_kill_count(container) == 1
    assert svc._read_oom_kill_count(container) == 1
    assert len(calls) == 1


def test_sandbox_status_has_oom_evidence_fields() -> None:
    status = SandboxStatus(
        state="Running",
        oom_killed=False,
        oom_kill_count=1,
        memory_limit_bytes=2147483648,
    )
    assert status.oom_killed is False
    assert status.oom_kill_count == 1
    assert status.memory_limit_bytes == 2147483648
    # alias 兼容（JSON 序列化用 camelCase）
    assert status.model_dump(by_alias=True)["oomKillCount"] == 1
    assert status.model_dump(by_alias=True)["oomKilled"] is False


def test_sandbox_status_oom_fields_optional() -> None:
    status = SandboxStatus(state="Running")
    assert status.oom_killed is None
    assert status.oom_kill_count is None
    assert status.memory_limit_bytes is None
