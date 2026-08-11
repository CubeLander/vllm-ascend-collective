# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Deterministic timed-trace JSONL workload generator.

Every preset is a pure function of ``(preset, capacity, seed)`` and emits
JSONL rows with the schema::

    {"timestamp": float, "input_length": int, "output_length": int, "prompt_seed": int}

``capacity`` is the scheduler capacity ``C`` (the MoE scheduler-domain token
limit).  Capacity-derived prompt lengths scale with ``C``; at the proven
``C=512`` the generator reproduces the exact campaign boundary/mixed/sawtooth
shapes:

- ``boundary-one``:     one request, 1-token prompt.
- ``boundary-quarter``: one request, ``C // 4`` prompt (128 at C=512).
- ``boundary-full``:    one request, ``C`` prompt (512 at C=512).
- ``steady``:           deterministic fixed-interval analogue of the campaign
                        decode workload.  The proving campaign ran ``vllm
                        bench random --request-rate 4``; this preset keeps the
                        same 32 requests of 462-in / 256-out with exact 0.25 s
                        arrivals, so the arrival timing is *not* byte-exact
                        with the campaign.  The lengths are a legacy decode
                        point and are *not* derived from ``C``.
- ``mixed``:            24 short-prompt long-output decode cohort requests
                        (``C // 8`` prompt, 64 at C=512) entering decode
                        before four staggered full-``C`` prefills.
- ``sawtooth``:         8 long decode anchors (``C // 8`` prompt) plus four
                        periodic bursts, each with 4 full-``C`` prefills.

No model tokens or text are embedded: prompts are synthesized later by the
token-id client from the served model's vocabulary.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

PRESETS = (
    "boundary-one",
    "boundary-quarter",
    "boundary-full",
    "steady",
    "mixed",
    "sawtooth",
)

# Fixed decode lengths from the proving campaign (independent of capacity).
STEADY_NUM_REQUESTS = 32
STEADY_INPUT_LENGTH = 462
STEADY_OUTPUT_LENGTH = 256
STEADY_QPS = 4.0

# Sawtooth shape constants (proven at C=512).
SAWTOOTH_ANCHORS = 8
SAWTOOTH_ANCHOR_OUTPUT = 512
SAWTOOTH_BURST_STARTS = (1.0, 3.0, 5.0, 7.0)
SAWTOOTH_BURST_REQUESTS = 4
SAWTOOTH_BURST_OUTPUT = 16

# Mixed shape constants (proven at C=512).
MIXED_COHORT_REQUESTS = 24
MIXED_COHORT_OUTPUT = 256
MIXED_PREFILLS = 4
MIXED_PREFILL_OUTPUT = 32

_BOUNDARY_OUTPUT = 16


def _row(ts: float, inp: int, out: int, seed: int) -> dict:
    return {
        "timestamp": round(ts, 6),
        "input_length": inp,
        "output_length": out,
        "prompt_seed": seed,
    }


def _fraction(capacity: int, denominator: int) -> int:
    """Capacity-derived prompt length, clamped to at least one token."""
    return max(1, capacity // denominator)


def build_rows(preset: str, capacity: int, seed: int) -> list[dict]:
    """Return the deterministic trace rows for one preset.

    The result is a pure function of ``(preset, capacity, seed)``.
    """
    if capacity < 1:
        raise ValueError(f"capacity must be >= 1, got {capacity}")
    rows: list[dict] = []
    if preset == "mixed":
        # A short-prompt, long-output cohort enters decode before four
        # full-C prefills arrive.  The prefill timestamps are deliberately
        # staggered.
        cohort_prompt = _fraction(capacity, 8)
        for i in range(MIXED_COHORT_REQUESTS):
            rows.append(_row(0.05 * i, cohort_prompt, MIXED_COHORT_OUTPUT, seed))
        for i in range(MIXED_PREFILLS):
            rows.append(_row(1.50 + 0.50 * i, capacity, MIXED_PREFILL_OUTPUT, seed))
    elif preset == "sawtooth":
        # Long-lived decode anchors, then four repeatable capacity bursts.
        anchor_prompt = _fraction(capacity, 8)
        for i in range(SAWTOOTH_ANCHORS):
            rows.append(_row(0.05 * i, anchor_prompt, SAWTOOTH_ANCHOR_OUTPUT, seed))
        for burst, ts in enumerate(SAWTOOTH_BURST_STARTS):
            for j in range(SAWTOOTH_BURST_REQUESTS):
                rows.append(_row(ts + 0.02 * j, capacity, SAWTOOTH_BURST_OUTPUT, seed + burst))
    elif preset == "boundary-one":
        rows.append(_row(0.0, 1, _BOUNDARY_OUTPUT, seed))
    elif preset == "boundary-quarter":
        rows.append(_row(0.0, _fraction(capacity, 4), _BOUNDARY_OUTPUT, seed))
    elif preset == "boundary-full":
        rows.append(_row(0.0, capacity, _BOUNDARY_OUTPUT, seed))
    elif preset == "steady":
        # Fixed-interval analogue of the campaign's `vllm bench random
        # --request-rate 4` decode workload: same lengths, deterministic
        # 0.25 s arrivals.  Lengths are intentionally not C-derived.
        for i in range(STEADY_NUM_REQUESTS):
            rows.append(_row(i / STEADY_QPS, STEADY_INPUT_LENGTH, STEADY_OUTPUT_LENGTH, seed))
    else:
        raise ValueError(f"unsupported preset: {preset}")
    return rows


def render_trace(rows: list[dict]) -> str:
    """Render rows as canonical sorted-key JSONL text (byte-stable)."""
    return "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)


def write_trace(rows: list[dict], path: Path) -> str:
    """Atomically write the trace and return its sha256 hex digest."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = render_trace(rows).encode()
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)
    return hashlib.sha256(payload).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("preset", choices=PRESETS)
    ap.add_argument("output", type=Path)
    ap.add_argument("--capacity", type=int, default=512, help="scheduler capacity C (default: 512)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rows = build_rows(args.preset, args.capacity, args.seed)
    digest = write_trace(rows, args.output)
    print(
        f"preset={args.preset} capacity={args.capacity} seed={args.seed} "
        f"requests={len(rows)} sha256={digest} path={args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
