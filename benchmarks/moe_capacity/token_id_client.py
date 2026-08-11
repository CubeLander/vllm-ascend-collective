# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Timed streaming benchmark client that sends token-id prompts directly.

``vllm bench serve``'s timed-trace path expands token hashes, decodes them to
text, and the server tokenizes that text again.  That round trip is not
length-preserving for arbitrary BPE tokens (a requested 128-token boundary
became 136 server tokens in the proving campaign), so it cannot control the
scheduler token domain ``M``.  This client therefore sends the OpenAI
completions endpoint explicit ``list[int]`` prompts while retaining vLLM's
streaming request implementation and metrics.  The result JSON records the
usage-reported prompt lengths, the exact requested hashes/lengths, and
TTFT/TPOT/ITL/E2E plus throughput aggregates.

Runtime dependencies (``aiohttp``, ``transformers``, and vLLM's benchmark
helpers) are imported lazily so the pure helpers stay importable offline.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import random
import time
from collections.abc import Sequence
from pathlib import Path
from statistics import mean, median

from .validate_result import read_trace

CLIENT_NAME = "benchmarks/moe_capacity/token_id_client"

# Mixed into every prompt RNG so trace seeds never collide with serving seeds.
_PROMPT_RNG_SALT = 0x54524143454C4F4F


def percentile(values: Sequence[float], value: float) -> float:
    """Linear-interpolation percentile, matching numpy's default method."""
    ordered = sorted(values) if values else [0.0]
    if len(ordered) == 1:
        return float(ordered[0])
    rank = (len(ordered) - 1) * (value / 100.0)
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return float(ordered[int(rank)])
    fraction = rank - low
    return float(ordered[low] * (1.0 - fraction) + ordered[high] * fraction)


def make_prompt(vocab: Sequence[int], length: int, seed: int, index: int) -> list[int]:
    """Deterministically draw ``length`` token ids for request ``index``."""
    rng = random.Random((seed << 32) ^ index ^ _PROMPT_RNG_SALT)
    return [vocab[rng.randrange(len(vocab))] for _ in range(length)]


def vocab_from_tokenizer(tokenizer) -> list[int]:
    """Sorted non-special token ids of a HuggingFace-style tokenizer."""
    special = set(tokenizer.all_special_ids)
    vocab = sorted(set(tokenizer.get_vocab().values()) - special)
    if not vocab:
        raise RuntimeError("tokenizer has no non-special token ids")
    return vocab


def prompt_sha256(prompt: Sequence[int]) -> str:
    return hashlib.sha256(json.dumps(list(prompt), separators=(",", ":")).encode()).hexdigest()


def build_prompts(rows: list[dict], vocab: Sequence[int], default_seed: int) -> list[list[int]]:
    return [
        make_prompt(vocab, int(row["input_length"]), int(row.get("prompt_seed", default_seed)), index)
        for index, row in enumerate(rows)
    ]


def summarize(rows: list[dict], prompts: list[list[int]], outputs: Sequence, duration: float) -> dict:
    """Aggregate request outputs into the canonical result payload."""
    succeeded = [output for output in outputs if output.success]
    ttfts = [output.ttft for output in succeeded]
    itls = [output.itl for output in succeeded]
    flat_itls = [value for values in itls for value in values]
    e2els = [output.latency for output in succeeded]
    tpots = [
        (output.latency - output.ttft) / (output.output_tokens - 1) for output in succeeded if output.output_tokens > 1
    ]
    input_lens = [output.prompt_len for output in outputs]
    output_lens = [output.output_tokens if output.success else 0 for output in outputs]

    def ms(values: Sequence[float], op) -> float:
        return float(op(list(values) or [0.0]) * 1000)

    return {
        "completed": len(succeeded),
        "failed": len(outputs) - len(succeeded),
        "num_prompts": len(outputs),
        "duration": duration,
        "input_lens": input_lens,
        "output_lens": output_lens,
        "total_input_tokens": sum(input_lens),
        "total_output_tokens": sum(output_lens),
        "request_throughput": len(succeeded) / duration,
        "output_throughput": sum(output_lens) / duration,
        "total_token_throughput": (sum(input_lens) + sum(output_lens)) / duration,
        "mean_ttft_ms": ms(ttfts, mean),
        "median_ttft_ms": ms(ttfts, median),
        "p99_ttft_ms": percentile(ttfts, 99) * 1000,
        "mean_tpot_ms": ms(tpots, mean),
        "median_tpot_ms": ms(tpots, median),
        "p99_tpot_ms": percentile(tpots, 99) * 1000,
        "mean_itl_ms": ms(flat_itls, mean),
        "median_itl_ms": ms(flat_itls, median),
        "p99_itl_ms": percentile(flat_itls, 99) * 1000,
        "mean_e2el_ms": ms(e2els, mean),
        "median_e2el_ms": ms(e2els, median),
        "p99_e2el_ms": percentile(e2els, 99) * 1000,
        "ttfts": ttfts,
        "itls": itls,
        "errors": [output.error for output in outputs],
        "generated_texts": [output.generated_text for output in outputs],
        "requested_input_lens": [len(prompt) for prompt in prompts],
        "requested_output_lens": [int(row["output_length"]) for row in rows],
        "requested_prompt_sha256": [prompt_sha256(prompt) for prompt in prompts],
        "client": CLIENT_NAME,
    }


def write_result(result: dict, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


async def run_requests(base_url: str, model: str, seed: int, rows: list[dict], prompts: list[list[int]]):
    """Send every trace row at its timestamp through vLLM's streaming helper."""
    import aiohttp
    from vllm.benchmarks.lib.endpoint_request_func import (
        RequestFuncInput,
        async_request_openai_completions,
    )

    timeout = aiohttp.ClientTimeout(total=6 * 60 * 60)
    connector = aiohttp.TCPConnector(limit=0, limit_per_host=0, ttl_dns_cache=300)
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        started = time.perf_counter()

        async def one(index: int):
            row = rows[index]
            deadline = started + float(row["timestamp"])
            delay = deadline - time.perf_counter()
            if delay > 0:
                await asyncio.sleep(delay)
            request = RequestFuncInput(
                # Runtime accepts CompletionRequest.prompt=list[int].  The
                # benchmark dataclass annotation is narrower than the API.
                prompt=prompts[index],  # type: ignore[arg-type]
                api_url=f"{base_url}/v1/completions",
                prompt_len=len(prompts[index]),
                output_len=int(row["output_length"]),
                model=model,
                ignore_eos=True,
                extra_body={"temperature": 0, "seed": seed},
                request_id=f"moe-capacity-{seed}-{index}",
            )
            return await async_request_openai_completions(request, session)

        outputs = await asyncio.gather(*(one(index) for index in range(len(rows))))
        duration = time.perf_counter() - started
    return outputs, duration


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trace", type=Path, required=True)
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    from transformers import AutoTokenizer

    rows = read_trace(args.trace)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    prompts = build_prompts(rows, vocab_from_tokenizer(tokenizer), args.seed)
    outputs, duration = asyncio.run(run_requests(args.base_url, args.model, args.seed, rows, prompts))
    result = summarize(rows, prompts, outputs, duration)
    write_result(result, args.output)
    print(
        f"completed={result['completed']} failed={result['failed']} "
        f"duration={duration:.3f}s input={result['total_input_tokens']} "
        f"output={result['total_output_tokens']} throughput={result['request_throughput']:.4f}"
    )
    if result["failed"] or result["input_lens"] != result["requested_input_lens"]:
        print("ERROR: request failure or server prompt-token count mismatch")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
