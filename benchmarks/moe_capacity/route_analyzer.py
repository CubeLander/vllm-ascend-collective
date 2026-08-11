# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fail-closed route / graph / shutdown-lifecycle gate for one server log.

The analyzer partitions the log at the explicit API shutdown marker, which is
required: a live or incomplete log without the marker fails closed.  The
serving part must show exactly one expected MoE communication family, every
observed live token count ``M`` at or below the scheduler capacity ``C``,
ACL-graph capture evidence, and zero tracebacks.  After the shutdown marker,
only two known vLLM 0.21 teardown shapes are accepted:

1. the exact single post-SIGTERM ``AsyncLLM`` ``EngineDeadError`` traceback;
2. the known multi-trace CANN worker-teardown family, accepted only when the
   server was explicitly idle before shutdown and every lifecycle marker is
   present (EngineCore request-complete teardown, worker SIGTERM /
   parent-exit / graph-cleanup, ForkAwareLocal and connection-refused
   races).

Any pre-shutdown traceback or unknown post-shutdown family fails the gate.
Pure standard library only.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

# ``MoE comm method selected: soc=%s, method=%s, num_tokens=%d, mc2_capacity=%s``
SELECT = re.compile(
    r"MoE comm method selected:.*method=MoECommType\.([A-Z0-9_]+), "
    r"num_tokens=(\d+), mc2_capacity=([^\s]+)"
)
SHUTDOWN_MARKER = "[shutdown] API server: shutdown triggered"
TRACEBACK_MARKER = "Traceback (most recent call last)"
KNOWN_SHUTDOWN_ERROR = "AsyncLLM output_handler failed."
KNOWN_SHUTDOWN_EXCEPTION = "vllm.v1.engine.exceptions.EngineDeadError:"
IDLE_MARKER = "Running: 0 reqs, Waiting: 0 reqs"
SCHEDULER_DOMAIN_FAIL_FAST = "outside the scheduler domain"
GRAPH_CAPTURE_LINE = "Capturing a aclgraph"
GRAPH_COMPILE_LINE = "Start compiling function"
IDLE_TAIL_LINES = 300

# Every marker of the known CANN worker-teardown family must be present.
WORKER_TEARDOWN_MARKERS = (
    "request processing complete; starting resource teardown",
    "WorkerProc handling signal 15",
    "Parent process exited, terminating worker queues",
    "Cleaned up profiling KV cache and CUDA graphs",
    "ForkAwareLocal' object has no attribute 'connection'",
    "ConnectionRefusedError: [Errno 111] Connection refused",
)


def analyze_log(text: str, expected_family: str, capacity: int) -> dict:
    """Classify one server log and return the full gate payload."""
    if capacity < 1:
        raise ValueError(f"capacity must be >= 1, got {capacity}")
    before_shutdown, marker, shutdown_tail = text.partition(SHUTDOWN_MARKER)
    serving_text = before_shutdown if marker else text
    rows = [(family, int(tokens)) for family, tokens, _ in SELECT.findall(serving_text)]
    families = Counter(family for family, _ in rows)
    token_hist = Counter(tokens for _, tokens in rows)
    issues: list[str] = []
    if not rows:
        issues.append("no communication-selection rows")
    if set(families) != {expected_family}:
        issues.append(f"families={dict(families)} expected_only={expected_family}")
    if any(tokens > capacity for _, tokens in rows):
        issues.append("observed token domain above scheduler capacity")
    if SCHEDULER_DOMAIN_FAIL_FAST in serving_text:
        issues.append("scheduler-domain fail-fast occurred during legal serving")
    capture_lines = serving_text.count(GRAPH_CAPTURE_LINE)
    compile_lines = serving_text.count(GRAPH_COMPILE_LINE)
    if capture_lines <= 0:
        issues.append("no ACL graph capture lines")
    serving_tracebacks = serving_text.count(TRACEBACK_MARKER)
    shutdown_tracebacks = shutdown_tail.count(TRACEBACK_MARKER) if marker else 0
    known_shutdown_engine_dead = bool(
        marker and KNOWN_SHUTDOWN_ERROR in shutdown_tail and KNOWN_SHUTDOWN_EXCEPTION in shutdown_tail
    )
    serving_tail = "\n".join(serving_text.splitlines()[-IDLE_TAIL_LINES:])
    shutdown_idle_before_marker = IDLE_MARKER in serving_tail
    known_worker_teardown = bool(
        marker
        and shutdown_tracebacks > 1
        and known_shutdown_engine_dead
        and shutdown_idle_before_marker
        and all(lifecycle in shutdown_tail for lifecycle in WORKER_TEARDOWN_MARKERS)
    )
    if serving_tracebacks:
        issues.append(f"server traceback before shutdown ({serving_tracebacks})")
    known_shutdown_lifecycle = bool((shutdown_tracebacks == 1 and known_shutdown_engine_dead) or known_worker_teardown)
    if shutdown_tracebacks and not known_shutdown_lifecycle:
        issues.append(f"unknown traceback during shutdown ({shutdown_tracebacks})")
    if not marker:
        issues.append("explicit API shutdown marker missing")
    return {
        "valid": not issues,
        "expected_family": expected_family,
        "capacity": capacity,
        "family_counts": dict(families),
        "token_histogram": {str(k): v for k, v in sorted(token_hist.items())},
        "max_tokens": max(token_hist, default=None),
        "selection_rows": len(rows),
        "capture_lines": capture_lines,
        "compile_lines": compile_lines,
        "shutdown_marker_present": bool(marker),
        "serving_tracebacks": serving_tracebacks,
        "shutdown_tracebacks": shutdown_tracebacks,
        "known_shutdown_engine_dead": known_shutdown_engine_dead,
        "shutdown_idle_before_marker": shutdown_idle_before_marker,
        "known_worker_teardown": known_worker_teardown,
        "known_shutdown_lifecycle": known_shutdown_lifecycle,
        "issues": issues,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("log", type=Path, help="server log to gate")
    ap.add_argument(
        "--expected-family",
        required=True,
        help="single accepted MoECommType family, e.g. FUSED_MC2 or ALLGATHER",
    )
    ap.add_argument("--capacity", type=int, required=True, help="scheduler capacity C")
    ap.add_argument("--output", type=Path, required=True, help="gate payload JSON destination")
    args = ap.parse_args()
    if not re.fullmatch(r"[A-Z0-9_]+", args.expected_family):
        ap.error("--expected-family must match [A-Z0-9_]+")
    payload = analyze_log(args.log.read_text(errors="replace"), args.expected_family, args.capacity)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, sort_keys=True))
    return 0 if payload["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
