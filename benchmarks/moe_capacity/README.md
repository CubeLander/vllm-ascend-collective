# MoE Capacity Harness

Reusable workload, measurement, and acceptance assets for graph-mode fused-MoE
capacity campaigns (see #215 and its follow-up #217).  The package generalizes
the proven task-local campaign
`benchmark-artifacts/moe-fused-capacity-mixed-20260811/campaign-w8a8-tp4-c512`
without hardcoded hosts, models, containers, or device paths: everything is
parameterized through CLI flags.

## Scope and invariants

- Benchmark-side assets only.  Nothing under this directory may change
  production source, add environment variables, or relax a gate.
- Every trace is a pure function of `(preset, capacity, seed)`; every gate is
  fail-closed and emits machine-readable evidence.
- No model tokens or text are embedded in traces.  Prompts are synthesized at
  run time from the served model's vocabulary by the token-id client.

## Graph-primary methodology

Cells are served under ACL-graph capture (for example
`{"cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes": [...]}`),
and the route gate requires graph-capture evidence in the server log.  The
campaign compares a frozen `baseline` configuration against a `candidate`
configuration of the same source revision.

`C` is the scheduler capacity: the MoE scheduler-domain token limit that the
server profiles at startup (`mc2_capacity` in the selection log).  `M` is the
live batch token count observed in
`MoE comm method selected: ... num_tokens=M` rows.  Legal serving requires
`M <= C` for every selection row.  `M > C` cannot occur through live serving
because the scheduler chunks serving batches to `C`; it is covered by the
repository's fail-fast unit test instead, and the route gate additionally
treats any observed `M > C` or any serving-time "outside the scheduler domain"
fail-fast as invalid.

## Exact token-ID rationale

Timed requests are sent to `/v1/completions` as explicit token-id lists
(`prompt=list[int]`).  Decoding arbitrary sampled BPE ids back to text and
retokenizing that text is not length preserving: in the proving campaign a
requested 128-token boundary arrived as 136 server tokens.  Direct token-id
prompts make the controlled input length equal to the usage-reported prompt
length and therefore equal to the scheduler evidence; the client still
verifies `input_lens == requested_input_lens` and fails otherwise.

## Presets

`gen_trace` writes JSONL rows
`{"timestamp", "input_length", "output_length", "prompt_seed"}`.
Capacity-derived prompt lengths scale with `C` (floored, minimum 1) and
reproduce the proven boundary/mixed/sawtooth shapes exactly at `C=512`:

- `boundary-one`: one request, 1-token prompt, 16 output tokens.
- `boundary-quarter`: one request, `C // 4` prompt (128 at C=512).
- `boundary-full`: one request, `C` prompt (512 at C=512).
- `steady`: a deterministic fixed-interval analogue of the campaign's decode
  workload.  The proving campaign ran `vllm bench random --request-rate 4`;
  this preset keeps the same 32 requests of 462-in / 256-out but uses exact
  0.25 s arrivals, so it is not a byte-exact reproduction of the campaign
  timing.  The lengths are a legacy decode point and are deliberately not
  derived from `C`.
- `mixed`: 24 short-prompt long-output cohort requests (`C // 8` prompt, 256
  output) entering decode before four staggered full-`C` prefills (32 output).
- `sawtooth`: 8 long decode anchors (`C // 8` prompt, 512 output) plus four
  periodic bursts at t=1/3/5/7 s, each with 4 full-`C` prefills (16 output);
  burst requests carry `prompt_seed = seed + burst`.

## Fresh-process and crossed-order guidance

- Start a fresh server process for every cell; never reuse a warm process
  across configurations or seeds.
- Cross arms and seeds deterministically (ABBA-style), for example
  `pair1-baseline-seed0, pair1-candidate-seed0, pair2-candidate-seed1,
  pair2-baseline-seed1`.
- If the baseline cannot be measured (see below), repeat each candidate cell
  with independent seeds instead of comparing against a stale arm.
- After every cell, stop the server, verify process cleanup, and re-probe the
  devices back to their idle/HBM baseline before the next cell.

## Fail-closed gates

- Route gate (`route_analyzer`): exactly one expected communication family,
  every `M <= C`, graph capture present, zero pre-shutdown tracebacks.  The
  explicit API shutdown marker is required: a live or incomplete log fails
  closed.  The log is partitioned at that marker; afterwards only two known
  teardown shapes are accepted: the single post-SIGTERM `AsyncLLM`
  `EngineDeadError` traceback, and the known multi-trace CANN worker-teardown
  family, which additionally requires pre-shutdown
  `Running: 0 reqs, Waiting: 0 reqs`, EngineCore request-complete teardown,
  worker SIGTERM / parent-exit / graph-cleanup, and the ForkAwareLocal /
  connection-refused markers.  Unknown shutdown families fail.
- Correctness gate (`validate_result`): exact completed/failed counts, exact
  per-request input/output lengths compared in request order, empty
  per-request errors, exact token totals, and finite positive
  throughput/TTFT/TPOT/ITL metrics.  Trace readers reject empty traces and
  rows with non-positive lengths or negative timestamps before sending or
  validating.
- HBM gate: host-side sampling with `sample_hbm` plus controller-checked idle
  baseline before and after each cell.  The sampler only records; availability
  policy stays with the caller.

## Baseline graph-infeasibility rule

If the frozen baseline cannot reach graph-server readiness (for example graph
capture specializes a supposedly dynamic shape), preserve that negative
feasibility cell as evidence.  Characterize the candidate in a candidate-only
mode that repeats every workload and retains absolute
throughput/latency/HBM evidence, but emit no speedup delta.  Never substitute
an eager or non-equivalent control, and never report an infinite or implied
speedup against an infeasible baseline.

## CLI examples

Run from the repository root.

```bash
# 1. Generate a deterministic trace (capacity C=512, seed 0).
python -m benchmarks.moe_capacity.gen_trace sawtooth out/sawtooth.jsonl --capacity 512 --seed 0

# 2. Send it at a running graph-mode server with exact token-id prompts.
python -m benchmarks.moe_capacity.token_id_client \
  --trace out/sawtooth.jsonl \
  --base-url http://127.0.0.1:8140 \
  --model <served-model-name> \
  --tokenizer <tokenizer-path> \
  --seed 0 \
  --output out/sawtooth.result.json

# 3. Validate exact counts / lengths / errors / metrics.
python -m benchmarks.moe_capacity.validate_result out/sawtooth.result.json --trace out/sawtooth.jsonl

# 4. Gate the server log (candidate arm shown; baseline uses its own family).
python -m benchmarks.moe_capacity.route_analyzer out/server.log \
  --expected-family FUSED_MC2 --capacity 512 --output out/route_gate.json

# 5. Sample host-side HBM/AICore during the cell.
python -m benchmarks.moe_capacity.sample_hbm \
  --devices 2,3,4,5 --output out/hbm.csv \
  --tag-file out/hbm.tag --stop-file out/hbm.stop --interval 2
```

## Host adapter boundary

`sample_hbm` parses the canonical `npu-smi info` table and accepts an injected
command (`--command`), explicit device list, interval, tag file, and stop
file.  Hosts with different tooling should provide a wrapper emitting that
table format or adapt `parse_npu_smi_info`; device-availability probing policy
(idle baselines, shared-container rules) intentionally stays outside this
package.
