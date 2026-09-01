# Copyright 2026 Alibaba Group Holding Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Server-side Prometheus metrics for sandbox lifecycle operations.

Zero-dependency implementation (no prometheus_client package): records
Docker operation counters/gauges/histograms for every sandbox lifecycle
action (create/start/kill/remove/...) and renders the standard Prometheus
text exposition format so netdata / Prometheus can scrape the "/metrics"
endpoint directly.

This complements the optional OTel export: it always records regardless of
the [otel] config and is directly scrapeable by netdata's go.d
prometheus collector (no OTLP endpoint required).
"""

from __future__ import annotations

import threading
from typing import Dict, Tuple

__all__ = [
    "record_op_start",
    "record_op_end",
    "render_text",
    "reset_for_tests",
]

# Duration histogram buckets in milliseconds (rendered as seconds).
_BUCKETS_MS: Tuple[float, ...] = (
    10.0,
    25.0,
    50.0,
    100.0,
    250.0,
    500.0,
    1000.0,
    2500.0,
    5000.0,
    10000.0,
    30000.0,
    60000.0,
    120000.0,
    300000.0,
)

_lock = threading.Lock()
# (action, status) -> count
_counters: Dict[Tuple[str, str], int] = {}
# action -> number of operations currently in flight
_in_progress: Dict[str, int] = {}
# (action, status) -> bucket cumulative counts (len == len(_BUCKETS_MS))
_bucket_counts: Dict[Tuple[str, str], list] = {}
# (action, status) -> (sum_ms, count)
_hist_sums: Dict[Tuple[str, str], Tuple[float, int]] = {}


def _escape_label(value: str) -> str:
    """Escape a Prometheus label value per text exposition format."""
    return (
        value.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\n", "\\n")
    )


def record_op_start(action: str) -> None:
    """Mark the beginning of a sandbox Docker operation."""
    with _lock:
        _in_progress[action] = _in_progress.get(action, 0) + 1


def record_op_end(action: str, status: str, duration_ms: float) -> None:
    """Record the outcome and duration of a finished sandbox Docker operation.

    status is typically "success" or "failure". duration_ms is the
    wall-clock duration of the Docker API call in milliseconds.
    """
    duration_ms = float(duration_ms) if duration_ms >= 0 else 0.0
    with _lock:
        remaining = _in_progress.get(action, 1) - 1
        if remaining <= 0:
            _in_progress.pop(action, None)
        else:
            _in_progress[action] = remaining

        key = (action, status)
        _counters[key] = _counters.get(key, 0) + 1

        buckets = _bucket_counts.get(key)
        if buckets is None:
            buckets = [0] * len(_BUCKETS_MS)
            _bucket_counts[key] = buckets
        for i, bound in enumerate(_BUCKETS_MS):
            if duration_ms <= bound:
                buckets[i] += 1

        sum_ms, count = _hist_sums.get(key, (0.0, 0))
        _hist_sums[key] = (sum_ms + duration_ms, count + 1)


def render_text() -> str:
    """Render all recorded metrics as Prometheus text exposition format."""
    with _lock:
        counters_snapshot = dict(sorted(_counters.items()))
        in_progress_snapshot = dict(sorted(_in_progress.items()))
        buckets_snapshot = {k: list(v) for k, v in sorted(_bucket_counts.items())}
        sums_snapshot = dict(sorted(_hist_sums.items()))

    out: list[str] = []
    out.append(
        "# HELP opensandbox_sandbox_ops_total Total sandbox Docker operations by action and status."
    )
    out.append("# TYPE opensandbox_sandbox_ops_total counter")
    for (action, status), value in counters_snapshot.items():
        out.append(
            'opensandbox_sandbox_ops_total{action="' + _escape_label(action)
            + '",status="' + _escape_label(status) + '"} ' + str(value)
        )

    out.append(
        "# HELP opensandbox_sandbox_ops_in_progress Sandbox Docker operations currently in flight by action."
    )
    out.append("# TYPE opensandbox_sandbox_ops_in_progress gauge")
    # Emit a stable gauge (0 when idle) for every action ever observed.
    all_actions = {action for (action, _status) in counters_snapshot}
    all_actions.update(in_progress_snapshot)
    for action in sorted(all_actions):
        value = in_progress_snapshot.get(action, 0)
        out.append(
            'opensandbox_sandbox_ops_in_progress{action="' + _escape_label(action) + '"} ' + str(value)
        )

    out.append(
        "# HELP opensandbox_sandbox_ops_duration_seconds Sandbox Docker operation duration in seconds."
    )
    out.append("# TYPE opensandbox_sandbox_ops_duration_seconds histogram")
    all_keys = sorted(set(buckets_snapshot) | set(sums_snapshot))
    for key in all_keys:
        action, status = key
        buckets = buckets_snapshot.get(key, [0] * len(_BUCKETS_MS))
        sum_ms, count = sums_snapshot.get(key, (0.0, 0))
        metric_name = "opensandbox_sandbox_ops_duration_seconds"
        label_body = (
            'action="' + _escape_label(action)
            + '",status="' + _escape_label(status) + '"'
        )
        for i, bound in enumerate(_BUCKETS_MS):
            out.append(
                metric_name + '_bucket{' + label_body
                + ',le="' + format(bound / 1000.0, ".6g") + '"} ' + str(buckets[i])
            )
        out.append(metric_name + '_bucket{' + label_body + ',le="+Inf"} ' + str(count))
        out.append(metric_name + '_sum{' + label_body + '} ' + format(sum_ms / 1000.0, ".6g"))
        out.append(metric_name + '_count{' + label_body + '} ' + str(count))

    return "\n".join(out) + "\n"


def reset_for_tests() -> None:
    """Clear all recorded metrics (used by unit tests only)."""
    with _lock:
        _counters.clear()
        _in_progress.clear()
        _bucket_counts.clear()
        _hist_sums.clear()
