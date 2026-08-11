# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Reusable graph-mode MoE capacity benchmark harness.

This package turns the proven task-local campaign
(``benchmark-artifacts/moe-fused-capacity-mixed-20260811``) into a small,
host-independent repo asset.  It contains:

- :mod:`benchmarks.moe_capacity.gen_trace`: deterministic timed-trace JSONL
  workload presets (boundary / steady / mixed / sawtooth), parameterized by
  scheduler capacity ``C`` and seed.
- :mod:`benchmarks.moe_capacity.token_id_client`: an OpenAI-completions
  streaming client that sends explicit token-id lists so the requested input
  length equals the server-reported prompt-token count.
- :mod:`benchmarks.moe_capacity.validate_result`: pure result validator for
  exact counts, lengths, errors, and finite metrics.
- :mod:`benchmarks.moe_capacity.route_analyzer`: fail-closed route / graph /
  shutdown-lifecycle gate for one server log.
- :mod:`benchmarks.moe_capacity.sample_hbm`: generic host-side HBM sampler
  driven by an injectable ``npu-smi`` command.

The pure helpers (trace generation, validation, log analysis, table parsing)
import only the Python standard library, so offline unit tests run without
Ascend, vLLM, transformers, aiohttp, or numpy installed.

This ``__init__`` intentionally imports nothing: each module is a standalone
CLI (``python -m benchmarks.moe_capacity.<module> ...``) and importing the
submodules here would make ``python -m`` re-import them and emit runpy
warnings.
"""
