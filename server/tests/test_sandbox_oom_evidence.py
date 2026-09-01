"""沙箱 OOM 证据暴露：SandboxStatus 新字段 + cgroup oom_kill 读取（TDD）。

prod2 2026-08-31 实测：沙箱容器 cgroup OOM（CONSTRAINT_MEMCG）杀掉容器内
opencode 进程（容器本身存活），State.OOMKilled 检测不到，必须读 cgroup
memory.events 的 oom_kill 计数才能证明进程级 OOM。
"""

from __future__ import annotations

from opensandbox_server.api.schema import SandboxStatus


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
