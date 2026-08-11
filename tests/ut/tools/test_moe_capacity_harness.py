# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Offline tests for the reusable MoE capacity harness.

Covers deterministic and capacity-scaled trace presets, fail-closed result
validation, the route/log analyzer's shutdown-lifecycle classification, and
the pure helpers of the token-id client and HBM sampler.  Everything here
imports only the standard library plus the new package.
"""

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.moe_capacity import (  # noqa: E402
    gen_trace,
    route_analyzer,
    sample_hbm,
    token_id_client,
)
from benchmarks.moe_capacity.validate_result import read_trace, validate_result  # noqa: E402


def _expected_sawtooth_c512(seed: int = 0) -> list[dict]:
    """Oracle mirrored from the proven campaign generator at C=512."""
    rows: list[dict] = []
    for i in range(8):
        rows.append({"timestamp": round(0.05 * i, 6), "input_length": 64, "output_length": 512, "prompt_seed": seed})
    for burst, ts in enumerate((1.0, 3.0, 5.0, 7.0)):
        for j in range(4):
            rows.append(
                {
                    "timestamp": round(ts + 0.02 * j, 6),
                    "input_length": 512,
                    "output_length": 16,
                    "prompt_seed": seed + burst,
                }
            )
    return rows


def _expected_mixed_c512(seed: int = 0) -> list[dict]:
    rows: list[dict] = []
    for i in range(24):
        rows.append({"timestamp": round(0.05 * i, 6), "input_length": 64, "output_length": 256, "prompt_seed": seed})
    for i in range(4):
        rows.append(
            {"timestamp": round(1.50 + 0.50 * i, 6), "input_length": 512, "output_length": 32, "prompt_seed": seed}
        )
    return rows


# ---------------------------------------------------------------------------
# Trace generation: determinism, proven C=512 shapes, capacity scaling.
# ---------------------------------------------------------------------------


def test_sawtooth_is_deterministic_and_proven_at_c512():
    first = gen_trace.build_rows("sawtooth", 512, 0)
    second = gen_trace.build_rows("sawtooth", 512, 0)
    assert first == second == _expected_sawtooth_c512()


def test_sawtooth_scales_with_capacity():
    rows = gen_trace.build_rows("sawtooth", 1024, 3)
    assert len(rows) == 24
    anchors, bursts = rows[:8], rows[8:]
    assert all(r["input_length"] == 128 and r["output_length"] == 512 for r in anchors)
    assert [r["timestamp"] for r in anchors] == [round(0.05 * i, 6) for i in range(8)]
    for burst, start in enumerate((1.0, 3.0, 5.0, 7.0)):
        chunk = bursts[burst * 4 : (burst + 1) * 4]
        assert [r["timestamp"] for r in chunk] == [round(start + 0.02 * j, 6) for j in range(4)]
        assert all(r["input_length"] == 1024 and r["output_length"] == 16 for r in chunk)
        assert all(r["prompt_seed"] == 3 + burst for r in chunk)


def test_sawtooth_small_capacity_clamps_to_one():
    rows = gen_trace.build_rows("sawtooth", 2, 0)
    assert all(r["input_length"] >= 1 for r in rows)
    assert rows[0]["input_length"] == 1  # 2 // 8 clamped to one token


def test_boundary_presets_proven_at_c512_and_scaled():
    for preset, expected in (("boundary-one", 1), ("boundary-quarter", 128), ("boundary-full", 512)):
        rows = gen_trace.build_rows(preset, 512, 5)
        assert rows == [{"timestamp": 0.0, "input_length": expected, "output_length": 16, "prompt_seed": 5}]
    assert gen_trace.build_rows("boundary-quarter", 1024, 0)[0]["input_length"] == 256
    assert gen_trace.build_rows("boundary-full", 1024, 0)[0]["input_length"] == 1024
    assert gen_trace.build_rows("boundary-quarter", 2, 0)[0]["input_length"] == 1


def test_mixed_proven_at_c512_and_scaled():
    assert gen_trace.build_rows("mixed", 512, 0) == _expected_mixed_c512()
    rows = gen_trace.build_rows("mixed", 256, 1)
    assert len(rows) == 28
    assert all(r["input_length"] == 32 for r in rows[:24])
    assert all(r["input_length"] == 256 for r in rows[24:])


def test_steady_is_the_fixed_decode_workload():
    rows = gen_trace.build_rows("steady", 512, 0)
    assert len(rows) == 32
    assert all(r["input_length"] == 462 and r["output_length"] == 256 for r in rows)
    assert [r["timestamp"] for r in rows] == [round(i / 4.0, 6) for i in range(32)]


def test_trace_rows_schema_and_order():
    for preset in gen_trace.PRESETS:
        rows = gen_trace.build_rows(preset, 512, 0)
        for row in rows:
            assert set(row) == {"timestamp", "input_length", "output_length", "prompt_seed"}
        timestamps = [row["timestamp"] for row in rows]
        assert timestamps == sorted(timestamps)


def test_invalid_capacity_and_preset_rejected():
    for bad_call in (lambda: gen_trace.build_rows("steady", 0, 0), lambda: gen_trace.build_rows("nope", 512, 0)):
        try:
            bad_call()
        except ValueError:
            pass
        else:
            raise AssertionError("expected ValueError")


def test_write_trace_roundtrip_is_byte_stable(tmp_path):
    rows = gen_trace.build_rows("sawtooth", 512, 0)
    digest_a = gen_trace.write_trace(rows, tmp_path / "a.jsonl")
    digest_b = gen_trace.write_trace(rows, tmp_path / "b.jsonl")
    assert digest_a == digest_b
    assert (tmp_path / "a.jsonl").read_bytes() == (tmp_path / "b.jsonl").read_bytes()
    parsed = [json.loads(line) for line in (tmp_path / "a.jsonl").read_text().splitlines()]
    assert parsed == rows


def test_gen_trace_cli_writes_trace(tmp_path, monkeypatch):
    output = tmp_path / "nested" / "trace.jsonl"
    monkeypatch.setattr(sys, "argv", ["gen_trace", "boundary-full", str(output), "--capacity", "64", "--seed", "1"])
    assert gen_trace.main() == 0
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert rows == [{"timestamp": 0.0, "input_length": 64, "output_length": 16, "prompt_seed": 1}]


# ---------------------------------------------------------------------------
# Result validation: exact pass/fail gates.
# ---------------------------------------------------------------------------


def _valid_result(rows: list[dict]) -> dict:
    input_lens = [int(row["input_length"]) for row in rows]
    output_lens = [int(row["output_length"]) for row in rows]
    return {
        "completed": len(rows),
        "failed": 0,
        "input_lens": input_lens,
        "output_lens": output_lens,
        "total_input_tokens": sum(input_lens),
        "total_output_tokens": sum(output_lens),
        "errors": [""] * len(rows),
        "request_throughput": 1.5,
        "mean_ttft_ms": 12.0,
        "mean_tpot_ms": 30.0,
        "mean_itl_ms": 29.0,
    }


def test_validate_result_passes_on_exact_match():
    rows = gen_trace.build_rows("sawtooth", 512, 0)
    assert validate_result(_valid_result(rows), rows) == []


def test_validate_result_rejects_permuted_lengths():
    # asyncio.gather preserves trace order; a permutation of the expected
    # lengths must fail even though the sorted multisets match.
    rows = gen_trace.build_rows("sawtooth", 512, 0)
    good = _valid_result(rows)
    permuted_in = dict(good, input_lens=list(reversed(good["input_lens"])))
    assert any("input_lens mismatch" in issue for issue in validate_result(permuted_in, rows))
    permuted_out = dict(good, output_lens=list(reversed(good["output_lens"])))
    assert any("output_lens mismatch" in issue for issue in validate_result(permuted_out, rows))


def test_validate_result_fails_on_each_corruption():
    rows = gen_trace.build_rows("mixed", 512, 0)
    good = _valid_result(rows)
    mutations = {
        "completed": dict(good, completed=len(rows) - 1),
        "failed": dict(good, failed=2),
        "input_lens": dict(good, input_lens=good["input_lens"][:-1] + [good["input_lens"][-1] + 1]),
        "output_lens": dict(good, output_lens=good["output_lens"][:-1] + [good["output_lens"][-1] + 1]),
        "total_input_tokens": dict(good, total_input_tokens=good["total_input_tokens"] + 1),
        "total_output_tokens": dict(good, total_output_tokens=good["total_output_tokens"] + 1),
        "errors": dict(good, errors=["boom"] * len(rows)),
        "nan_metric": dict(good, mean_ttft_ms=float("nan")),
        "zero_metric": dict(good, request_throughput=0.0),
        "missing_metric": {key: value for key, value in good.items() if key != "mean_itl_ms"},
    }
    for name, corrupted in mutations.items():
        issues = validate_result(corrupted, rows)
        assert issues, f"mutation {name} was not rejected"


def test_read_trace_requires_lengths(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text(json.dumps({"timestamp": 0.0, "input_length": 1}) + "\n")
    try:
        read_trace(path)
    except ValueError as exc:
        assert "output_length" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_read_trace_rejects_empty_trace(tmp_path):
    path = tmp_path / "empty.jsonl"
    for content in ("", "\n   \n"):
        path.write_text(content)
        try:
            read_trace(path)
        except ValueError as exc:
            assert "empty" in str(exc)
        else:
            raise AssertionError("expected ValueError")


def test_read_trace_rejects_bad_rows(tmp_path):
    bad_rows = [
        {"timestamp": 0.0, "input_length": 0, "output_length": 16},
        {"timestamp": 0.0, "input_length": 8, "output_length": -1},
        {"timestamp": -0.5, "input_length": 8, "output_length": 16},
        {"input_length": 8, "output_length": 16},
    ]
    path = tmp_path / "bad.jsonl"
    for row in bad_rows:
        path.write_text(json.dumps(row) + "\n")
        try:
            read_trace(path)
        except ValueError:
            pass
        else:
            raise AssertionError(f"row not rejected: {row}")


# ---------------------------------------------------------------------------
# Route analyzer: shutdown-lifecycle classification.
# ---------------------------------------------------------------------------


def _select_line(family: str, num_tokens: int, capacity: int = 512) -> str:
    return (
        f"DEBUG MoE comm method selected: soc=ASCEND910_4, method=MoECommType.{family}, "
        f"num_tokens={num_tokens}, mc2_capacity={capacity}"
    )


def _serving_log(family: str = "FUSED_MC2", token_counts=(2, 4, 8, 512), idle=True, pre_traceback=False) -> str:
    lines = [
        "INFO server started",
        "INFO Capturing a aclgraph for size 2",
        "INFO Capturing a aclgraph for size 4",
    ]
    if pre_traceback:
        lines += [
            "Traceback (most recent call last):",
            '  File "serving.py", line 1, in step',
            "RuntimeError: serving failure",
        ]
    lines += [_select_line(family, tokens) for tokens in token_counts]
    if idle:
        lines.append("INFO Running: 0 reqs, Waiting: 0 reqs")
    return "\n".join(lines) + "\n"


def _engine_dead_tail() -> str:
    return (
        "Traceback (most recent call last):\n"
        '  File "vllm/v1/engine/async_llm.py", line 100, in output_handler\n'
        "vllm.v1.engine.exceptions.EngineDeadError: AsyncLLM output_handler failed.\n"
    )


def _worker_teardown_tail() -> str:
    lines = [
        "Traceback (most recent call last):",
        '  File "vllm/v1/engine/async_llm.py", line 100, in output_handler',
        "vllm.v1.engine.exceptions.EngineDeadError: AsyncLLM output_handler failed.",
    ]
    for rank in range(4):
        lines += [
            "Traceback (most recent call last):",
            f'  File "cann/knowledge_base_{rank}.py", line 1, in flush',
            "EOFError: end of file",
        ]
    lines += [
        "INFO EngineCore: request processing complete; starting resource teardown",
        "INFO WorkerProc handling signal 15",
        "INFO Parent process exited, terminating worker queues",
        "INFO Cleaned up profiling KV cache and CUDA graphs",
        "AttributeError: 'ForkAwareLocal' object has no attribute 'connection'",
        "ConnectionRefusedError: [Errno 111] Connection refused",
    ]
    return "\n".join(lines) + "\n"


def test_route_valid_single_shutdown():
    text = _serving_log() + route_analyzer.SHUTDOWN_MARKER + "\n" + _engine_dead_tail()
    payload = route_analyzer.analyze_log(text, "FUSED_MC2", 512)
    assert payload["valid"], payload["issues"]
    assert payload["family_counts"] == {"FUSED_MC2": 4}
    assert payload["max_tokens"] == 512
    assert payload["capture_lines"] == 2
    assert payload["serving_tracebacks"] == 0
    assert payload["shutdown_tracebacks"] == 1
    assert payload["known_shutdown_engine_dead"] is True
    assert payload["known_shutdown_lifecycle"] is True
    assert payload["known_worker_teardown"] is False


def test_route_valid_known_worker_teardown():
    text = _serving_log() + route_analyzer.SHUTDOWN_MARKER + "\n" + _worker_teardown_tail()
    payload = route_analyzer.analyze_log(text, "FUSED_MC2", 512)
    assert payload["valid"], payload["issues"]
    assert payload["shutdown_tracebacks"] == 5
    assert payload["shutdown_idle_before_marker"] is True
    assert payload["known_shutdown_engine_dead"] is True
    assert payload["known_worker_teardown"] is True
    assert payload["known_shutdown_lifecycle"] is True


def test_route_invalid_pre_shutdown_traceback():
    text = _serving_log(pre_traceback=True) + route_analyzer.SHUTDOWN_MARKER + "\n" + _engine_dead_tail()
    payload = route_analyzer.analyze_log(text, "FUSED_MC2", 512)
    assert not payload["valid"]
    assert any("before shutdown" in issue for issue in payload["issues"])


def test_route_invalid_unknown_shutdown_family():
    unknown_tail = (
        'Traceback (most recent call last):\n  File "somewhere.py", line 1, in run\nRuntimeError: unclassified crash\n'
    )
    text = _serving_log() + route_analyzer.SHUTDOWN_MARKER + "\n" + unknown_tail
    payload = route_analyzer.analyze_log(text, "FUSED_MC2", 512)
    assert not payload["valid"]
    assert payload["known_shutdown_lifecycle"] is False
    assert any("unknown traceback during shutdown" in issue for issue in payload["issues"])


def test_route_worker_teardown_requires_idle_server():
    text = _serving_log(idle=False) + route_analyzer.SHUTDOWN_MARKER + "\n" + _worker_teardown_tail()
    payload = route_analyzer.analyze_log(text, "FUSED_MC2", 512)
    assert not payload["valid"]
    assert payload["known_worker_teardown"] is False


def test_route_rejects_wrong_family_oversized_tokens_and_missing_capture():
    def with_shutdown(text: str) -> str:
        return text + route_analyzer.SHUTDOWN_MARKER + "\n"

    wrong_family = route_analyzer.analyze_log(with_shutdown(_serving_log(family="ALLGATHER")), "FUSED_MC2", 512)
    assert not wrong_family["valid"]
    oversized = route_analyzer.analyze_log(with_shutdown(_serving_log(token_counts=(2, 600))), "FUSED_MC2", 512)
    assert not oversized["valid"]
    assert any("above scheduler capacity" in issue for issue in oversized["issues"])
    no_capture = route_analyzer.analyze_log(with_shutdown(_select_line("FUSED_MC2", 8) + "\n"), "FUSED_MC2", 512)
    assert not no_capture["valid"]
    assert any("graph capture" in issue for issue in no_capture["issues"])


def test_route_missing_shutdown_marker_fails_closed():
    payload = route_analyzer.analyze_log(_serving_log(), "FUSED_MC2", 512)
    assert not payload["valid"]
    assert payload["shutdown_marker_present"] is False
    assert any("shutdown marker missing" in issue for issue in payload["issues"])


# ---------------------------------------------------------------------------
# Pure helpers of the token-id client and the HBM sampler.
# ---------------------------------------------------------------------------


def test_percentile_matches_linear_interpolation():
    assert token_id_client.percentile([1.0, 2.0, 3.0, 4.0], 50) == 2.5
    assert token_id_client.percentile([10.0], 99) == 10.0
    assert token_id_client.percentile([], 99) == 0.0


def test_make_prompt_is_deterministic_and_in_vocab():
    vocab = list(range(1000, 1100))
    first = token_id_client.make_prompt(vocab, 16, 7, 3)
    second = token_id_client.make_prompt(vocab, 16, 7, 3)
    other = token_id_client.make_prompt(vocab, 16, 7, 4)
    assert first == second
    assert first != other
    assert len(first) == 16
    assert set(first) <= set(vocab)


def test_token_id_client_read_trace_shares_contract(tmp_path):
    zero = tmp_path / "zero.jsonl"
    zero.write_text(json.dumps({"timestamp": 0.0, "input_length": 0, "output_length": 16}) + "\n")
    empty = tmp_path / "empty.jsonl"
    empty.write_text("")
    for path in (zero, empty):
        try:
            token_id_client.read_trace(path)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError for {path.name}")


class _FakeOutput:
    def __init__(self, success, prompt_len, output_tokens, ttft, itl, latency, error="", generated_text=""):
        self.success = success
        self.prompt_len = prompt_len
        self.output_tokens = output_tokens
        self.ttft = ttft
        self.itl = itl
        self.latency = latency
        self.error = error
        self.generated_text = generated_text


def test_summarize_reports_exact_receipts():
    rows = gen_trace.build_rows("boundary-full", 512, 0) + gen_trace.build_rows("boundary-one", 512, 0)
    prompts = [[7] * 512, [9]]
    outputs = [
        _FakeOutput(True, 512, 16, 0.5, [0.03] * 15, 1.0),
        _FakeOutput(False, 1, 0, 0.0, [], 0.1, error="boom"),
    ]
    result = token_id_client.summarize(rows, prompts, outputs, duration=2.0)
    assert result["completed"] == 1
    assert result["failed"] == 1
    assert result["total_input_tokens"] == 513
    assert result["total_output_tokens"] == 16
    assert result["requested_input_lens"] == [512, 1]
    assert result["requested_output_lens"] == [16, 16]
    assert len(result["requested_prompt_sha256"]) == 2
    assert result["errors"] == ["", "boom"]
    for key in ("request_throughput", "mean_ttft_ms", "mean_tpot_ms", "mean_itl_ms"):
        assert result[key] > 0


def test_parse_npu_smi_info_reads_device_table():
    table = "\n".join(
        (
            "+======================+===============+=========================================================+",
            "| 0     910B3          | OK            | 98.1        46                0    / 0                  |",
            "| 0                    | 0000:C1:00.0  | 0           0    / 0          3705 / 65536              |",
            "+======================+===============+=========================================================+",
            "| 1     910B3          | OK            | 90.0        45                0    / 0                  |",
            "| 1                    | 0000:C2:00.0  | 37          0    / 0          40000 / 65536             |",
            "+======================+===============+=========================================================+",
        )
    )
    readings = sample_hbm.parse_npu_smi_info(table)
    assert readings[0] == (0.0, 3705, 65536)
    assert readings[1] == (37.0, 40000, 65536)
