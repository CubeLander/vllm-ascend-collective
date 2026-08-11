# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fail-closed validator for one token-id client result against its trace.

Checks exact request counts, exact per-request input/output lengths, empty
per-request errors, exact token totals, and finite positive headline metrics.
Lengths are compared in request order: the client's ``asyncio.gather``
preserves trace order, and order-sensitive comparison catches per-request
mismatches that sorting would hide.  Pure standard library only.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

REQUIRED_METRICS = ("request_throughput", "mean_ttft_ms", "mean_tpot_ms", "mean_itl_ms")


def read_trace(path: Path) -> list[dict]:
    """Parse and validate a trace JSONL file produced by ``gen_trace``."""
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    check_rows(rows)
    return rows


def check_rows(rows: list[dict]) -> None:
    """Fail closed on empty traces and out-of-domain row values."""
    if not rows:
        raise ValueError("trace is empty")
    for index, row in enumerate(rows):
        for key in ("timestamp", "input_length", "output_length"):
            if key not in row:
                raise ValueError(f"trace row {index} missing {key!r}")
        ts = row["timestamp"]
        if not isinstance(ts, (int, float)) or isinstance(ts, bool) or ts < 0:
            raise ValueError(f"trace row {index} has invalid timestamp {ts!r}")
        for key in ("input_length", "output_length"):
            value = row[key]
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"trace row {index} has invalid {key} {value!r}")


def validate_result(data: dict, rows: list[dict]) -> list[str]:
    """Return the list of validation issues (empty when the result is valid)."""
    issues: list[str] = []
    expected_in = [int(row["input_length"]) for row in rows]
    expected_out = [int(row["output_length"]) for row in rows]
    n = len(rows)
    if data.get("completed") != n:
        issues.append(f"completed={data.get('completed')} expected={n}")
    if data.get("failed") != 0:
        issues.append(f"failed={data.get('failed')}")
    # asyncio.gather preserves trace order; compare in request order so a
    # per-request mismatch cannot hide inside a permutation.
    input_lens = data.get("input_lens")
    if not isinstance(input_lens, list) or [int(value) for value in input_lens] != expected_in:
        issues.append("input_lens mismatch")
    output_lens = data.get("output_lens")
    if not isinstance(output_lens, list) or [int(value) for value in output_lens] != expected_out:
        issues.append("output_lens mismatch")
    if data.get("total_input_tokens") != sum(expected_in):
        issues.append("total_input_tokens mismatch")
    if data.get("total_output_tokens") != sum(expected_out):
        issues.append("total_output_tokens mismatch")
    if data.get("errors") != [""] * n:
        issues.append("request errors present")
    for key in REQUIRED_METRICS:
        value = data.get(key)
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or value <= 0:
            issues.append(f"bad metric {key}={value!r}")
    return issues


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("result", type=Path, help="token-id client result JSON")
    ap.add_argument("--trace", type=Path, required=True, help="trace JSONL the result was produced from")
    args = ap.parse_args()
    data = json.loads(args.result.read_text())
    rows = read_trace(args.trace)
    issues = validate_result(data, rows)
    if issues:
        print("INVALID: " + "; ".join(issues))
        return 1
    input_tokens = sum(int(row["input_length"]) for row in rows)
    output_tokens = sum(int(row["output_length"]) for row in rows)
    print(f"VALID requests={len(rows)} input_tokens={input_tokens} output_tokens={output_tokens}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
