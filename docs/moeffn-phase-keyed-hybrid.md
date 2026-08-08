# Phase-keyed hybrid MoE comm policy (experimental, default OFF)

Status: **experimental mechanism, default disabled**. This document records the
design, the historical failure it must not reproduce, the safety invariants and
their audit, the exact execution envelope, the selector truth table, and the
unit-test contract. It is a mechanism/design note, not a performance promise.

## Motivation

TraceLoom P2 (see `TRACELOOM_CUDA_HANDOFF_20260804.md`) observed that a fused
MoE comm path can be a meaningful target inside decode ACL graphs
(+12.36%/+12.95% on the decode-graph target) while graph-external large units
show -11.48% and anchor count moves 1186 -> 994. These numbers are the
prototype motivation only; no per-phase performance promise is made here.

The production experiment
`research/co-resident-async-ep/experiments/p0g-production-gmm2-cutthrough/README.md`
(L270) documented the hard architecture gate: a naive `num_tokens >= 16`
selector that swapped the Python comm object (`FusedMC2CommImpl` <-> `AllGatherCommImpl`)
inside one `torch.compile` model graph failed at startup with
`'AllGatherCommImpl' object has no attribute 'expert_token_nums'` — the compiled
template kept the first object's layout. That patch was withdrawn. A safe hybrid
needs either per-path compiled graphs or a stable data-driven interface; this
prototype provides the former by keeping exactly one comm object family inside
every compiled/ACL template.

## Configuration

Additional-config only, default off. There is deliberately **no environment
variable**: repo convention funnels new switches through
`vllm_ascend/ascend_config.py`, and this switch is a config-user choice.

```bash
--additional-config '{"enable_fused_mc2":1,"moe_phase_hybrid_policy":"mixed_prefill_fused"}'
```

`enable_fused_mc2: 1` is REQUIRED, not optional: the fused comm impl and the
NZ weight layout are only initialized in that mode. The startup validation
raises unless both are set, so the experiment cannot silently stay on the
baseline family.

`AscendConfig` parses the value through
`parse_moe_phase_hybrid_policy` (`MoEPhaseHybridPolicy` enum) at startup:

| value | meaning |
|---|---|
| `off` (default when missing) | stock selector, byte-for-byte; no phase keying, no skip_compiled |
| `mixed_prefill_fused` | only the `MIXED` forward phase may select `MoECommType.FUSED_MC2` (the existing `dispatch_ffn_combine` BF16 path); `PURE_PREFILL` and `MIXED` both bypass the compiled model |

The accepted values are exactly `off` and `mixed_prefill_fused`. Bool-like
aliases (`0`/`1`/`true`/`false`/`on`/`none`/...) are deliberately rejected so
the switch cannot be set ambiguously; any unrecognized value raises
`ValueError` at `AscendConfig` construction and fails closed at startup.

## Forward phase classification

A host-computed **forward phase** (never a device-tensor `.item()` sync)
distinguishes, from the v1 runner's `attn_state`/`with_prefill` CPU state:

- `PURE_DECODE` — every request is decoding / speculative decode
  (`attn_state` in `{DecodeOnly, SpecDecoding}`).
- `PURE_PREFILL` — every request is a brand-new prefill
  (`PrefillNoCache`: zero computed tokens).
- `MIXED` — at least one request is prefilling alongside requests that
  already have computed tokens (`ChunkedPrefill` / `PrefillCacheHit`).

**Only `MIXED` is in the fused experiment.** `PURE_PREFILL` is deliberately
NOT part of the fused path: the existing evidence supports the
mixed/interleaved unit only, and extending pure prefill to fused requires
separate measurement. Under the policy, `PURE_PREFILL` keeps the baseline
comm family but still bypasses the compiled model (see the safety argument),
so it never reuses a decode-compiled artifact.

## Execution envelope (fail closed)

The minimal safe envelope is **v1 model runner + `cudagraph_mode ==
FULL_DECODE_ONLY` + `data_parallel_size == 1`**. `validate_moe_phase_hybrid_policy`
runs at startup (platform `check_and_update_config`, after all platform mode
fallbacks) and raises otherwise:

- `FULL` / `FULL_AND_PIECEWISE` / `PIECEWISE`: prefill or mixed waves can be
  dispatched into a graph template there. That would either replay a baseline
  template under a fused intent or trigger runtime capture after capture is
  disabled. `skip_compiled` alone does not protect that envelope, so these
  modes are rejected.
- `NONE` (no graphs): rejected — the invariant "decode is the only
  compiled/ACL-graph path" requires decode graphs to exist.
- `enable_fused_mc2 != 1`: rejected — the fused comm impl and the NZ weight
  layout are only initialized with `enable_fused_mc2 == 1`; without it the
  policy would silently stay on the baseline family and the experiment would
  not actually be enabled.
- V2 runner: rejected — upstream V2 owns `skip_compiled`; vllm-ascend cannot
  force fused mixed-prefill forwards to bypass the compiled model from the
  platform hook.
- DP > 1: rejected — DP ranks can schedule different phases for the same step,
  while FUSED_MC2 collectives span the EP group (which includes all DP ranks).
  A per-rank phase-dependent comm choice is not DP-consistent, so the policy
  fails closed instead of deadlocking.

The same validation is invoked defensively inside
`set_ascend_forward_context` on every hybrid-active forward, so a runner that
somehow bypasses the startup check fails on its first forward rather than at
request time.

## Selector semantics: baseline family, not stock

The blocker that made the first revision ineffective: under
`enable_fused_mc2=1` (A2 or A3), the **stock** selector already returns
`FUSED_MC2` for decode — so "decode keeps stock" would leave decode fused and
the hybrid would be a no-op with the stale-layout risk still present.

The corrected rule: when the policy is enabled,

- `MIXED` → `FUSED_MC2` when the SoC capability gate holds, else fail closed
  to the **baseline** family;
- `PURE_DECODE` and `PURE_PREFILL` → the **baseline** family, defined as what
  the stock selector would choose with `enable_fused_mc2 == 0` (explicit
  `_select_moe_comm_method_baseline`, written out rather than simulated by
  mutating global config). For the A2/TP2 experiment that is `ALLGATHER`.

So under the policy the decode comm object is **exactly the object the decode
ACL graph captured with** — the same family as an `enable_fused_mc2=0` run —
and only MIXED (which never enters a graph) runs fused.

## Why this cannot reproduce the historical stale-layout failure

Two independent layers enforce "one compiled/captured template = one comm
object":

1. **Comm object stability (baseline selector)**: with the policy enabled,
   decode and pure prefill always select the `enable_fused_mc2=0` baseline
   family — never fused. The compiled/ACL decode template therefore embeds the
   same comm object family for the whole process; no `num_tokens >= N`
   threshold (the historical bug) and no phase can swap the object inside a
   template. There is deliberately **no token-count-dependent selection
   anywhere** in the hybrid path.
2. **Compiled path + dispatch (`skip_compiled=True` + cudagraph NONE)**:
   under the policy, BOTH `PURE_PREFILL` and `MIXED` (a) bypass the compiled
   model (`ForwardContext.skip_compiled`, honored by upstream
   `vllm/compilation/decorators.py`) and (b) force eager dispatch
   (`force_eager` in `_determine_batch_execution_and_padding` →
   `CUDAGraphMode.NONE`). This is required even for `PURE_PREFILL`, which
   keeps the baseline comm family: on A3 the decode baseline can be MC2 while
   a large pure-prefill baseline is ALLTOALL, so a pure-prefill wave must
   never reuse the decode-compiled MC2 artifact. `force_eager` is the
   belt-and-braces guarantee that neither a pure-prefill nor a mixed wave
   ever enters FULL/PIECEWISE capture or replay; decode is the only compiled
   path.

**No ACL cache-key change is needed.** The safety invariant is "prefill never
enters a graph"; decode is the only compiled/ACL-graph path and its comm family
never changes. `ACLGraphWrapper`/`compute_acl_graph_cache_key` are untouched.

Additionally the policy is **deliberately token-count independent**: per-wave
token thresholds are exactly what historically swapped the Python comm object
inside one compiled template. No `num_tokens >= N` logic exists in the hybrid
selector.

## Capture / dummy-run ordering audit

- Capture warmup (`_warmup_and_capture` -> `_dummy_run`) only runs decode-style
  dummy runs (`with_prefill=False` -> `PURE_DECODE`), so every captured graph
  is a baseline decode template, exactly matching runtime decode under the
  policy.
- `_dummy_run` defaults to the same host phase classification
  (`PURE_PREFILL` if `with_prefill` else `PURE_DECODE`) but accepts an explicit
  profile-only phase override; graph capture never uses that override.
- Under the policy, `profile_run` uses the existing MC2-capacity dummy with an
  explicit `MIXED` override. It stays eager/`skip_compiled`, warms
  `dispatch_ffn_combine`, and includes its workspace in memory profiling,
  rather than making the first real MIXED request pay first-use cost.

## Weight-layout compatibility audit

`UnquantizedFusedMoEMethod.process_weights_after_loading` lays weights out at
load time from the static `enable_fused_mc2` config: with
`enable_fused_mc2 != 0` the expert weights are cast to `ACL_FORMAT_FRACTAL_NZ`
(and `maybe_trans_nz` otherwise). The layout is therefore fixed per process, not
per wave — there is no NZ conversion toggle per wave, and **both** the baseline
paths (MC2 / ALLTOALL / ALLGATHER) and the fused path consume the same loaded
layout. Production already mixes `FUSED_MC2` with `ALLTOALL`/`MC2` on A3 with
`enable_fused_mc2=1`, so the mixed-layout consumption is known to be
numerically compatible.

**Disclosure:** the hybrid process therefore forces NZ, whereas historical
stock BF16 decode ran on the default ND layout. A dedicated 2x910B2
stock-`ALLGATHER` discriminator has now measured that distinction with ACL
graph enabled, 8 warmups and 30 bracketed samples per arm. Both layouts were
finite and bitwise identical (`max_abs_diff=0.0`) in every cell, but NZ was
materially faster than ND:

| global capacity | rank-critical NZ vs ND median |
|---:|---:|
| 2 | -3.82% |
| 4 | -8.25% |
| 8 | -13.42% |
| 16 | -16.68% |
| 32 | -15.79% |

The hybrid decode arm is therefore **baseline communication on NZ weights**,
not performance-equivalent to the historical ND stock arm. Any end-to-end
hybrid improvement must be reported against a fresh control and must not be
attributed solely to phase selection. The discriminator used the production
tensor shapes and stock operator sequence but substituted the available v2
fake-id routing primitive in the isolated micro environment; it settles
layout compatibility and shows a material layout effect, not full real-model
latency. Its immutable local receipt is
`.lumi-workbench/artifacts/nd-vs-nz-20260808-final/full-env/summary.json`.
The receipt's inherited `npu_devices_env="6,7"` is stale container metadata;
the before/after host snapshots record the actually mounted physical devices
4 and 5.

## Selector truth table

`select_moe_comm_method(num_tokens, vllm_config, is_draft_model=False, *, phase=None)`:

| policy | phase | EP / SoC gate | result |
|---|---|---|---|
| OFF (default) | any / None | any | stock (unchanged, incl. `None` for non-MoE) |
| ON | `None` (legacy callers, e.g. spec-decode proposers) | any | stock |
| ON | draft model | any | stock |
| ON | `MIXED` | A3: `enable_fused_mc2==1` and EP<=32; A2: fused-A2 gate | `FUSED_MC2` |
| ON | `MIXED` | gate fails | baseline (`enable_fused_mc2=0` family), fail closed |
| ON | `PURE_DECODE` | any | baseline (A2/TP2: `ALLGATHER`) |
| ON | `PURE_PREFILL` | any | baseline comm (A2/TP2: `ALLGATHER`) **and** bypass compiled (skip_compiled + eager NONE) |
| ON | any | non-MoE | `None` |

Baseline family by SoC (identical to stock with `enable_fused_mc2=0`): A2 →
`MC2` when `num_experts_per_device <= 24`, EP>=16 and tokens fit MC2 capacity,
else `ALLGATHER`; A3 → `MC2` within MC2 capacity else `ALLTOALL`; 310P →
`ALLGATHER`; A5 → unchanged from stock (A5 has no fused branch).

## Files touched

- `vllm_ascend/ascend_config.py` — `MoEPhaseHybridPolicy` enum +
  `parse_moe_phase_hybrid_policy` (additional-config only, default `off`),
  `AscendConfig.moe_phase_hybrid_policy`.
- `vllm_ascend/ascend_forward_context.py` — host phase classification,
  hybrid-active predicate (MIXED only), startup validation, phase-aware
  selector with explicit baseline selector (`_select_moe_comm_method_baseline`;
  stock logic kept intact as `_select_moe_comm_method_stock`),
  `should_bypass_compiled_for_moe_phase_hybrid` (PURE_PREFILL + MIXED bypass
  the compiled model), `skip_compiled` merge.
- `vllm_ascend/worker/model_runner_v1.py` — host phase computation, forced
  eager dispatch for PURE_PREFILL + MIXED under the policy, phase threading
  into `set_ascend_forward_context` (runtime + dummy runs).
- `vllm_ascend/platform.py` — startup fail-closed validation after mode
  resolution; defensive V2 hook validation.
- `tests/ut/test_ascend_forward_context.py`, `tests/ut/test_ascend_config.py`
  — unit tests (CPU, no NPU).
- `docs/moeffn-phase-keyed-hybrid.md` — this note.

## Tests

Coverage includes the selector truth table (A2/TP2 decode+pure-prefill →
baseline `ALLGATHER`, MIXED → `FUSED_MC2`; A3 capacity/fail-closed cases;
A5/310P; draft/non-MoE/`phase=None`), default compatibility (policy OFF
identical to stock for every phase), strict config parsing (default OFF,
`mixed_prefill_fused` accepted, bool-like aliases raise), 3-way phase
classification, fail-closed validation (DP / V2 / cudagraph mode /
`enable_fused_mc2 != 1`), the hybrid-active predicate (MIXED only), the
bypass-compiled predicate (PURE_PREFILL + MIXED), and `skip_compiled`
propagation.

## Real-model mechanism gate

The prototype passed a bounded A2/TP2 real-model gate on two Ascend 910B2
devices with Qwen3-30B-A3B BF16, EP2/DP1, `FULL_DECODE_ONLY`, and capture sizes
2, 4, 8, 16, and 32:

- startup, memory profiling, and 14 ACL graph captures completed without
  stale-comm errors, OOM, 507015, or traceback;
- the server log independently records 208
  `PURE_DECODE -> baseline ALLGATHER` selections, 8
  `PURE_PREFILL -> baseline` selections, and 18
  `MIXED -> FUSED_MC2` selections;
- an 8-request random serving gate at 462 input / 16 output tokens and QPS 4
  completed 8/8 with exact requested lengths and no request errors; and
- a deterministic completion after MIXED/FUSED traffic succeeded, showing
  that the captured decode graph remained reusable after the eager fused
  phase.

This establishes mechanism, numerical-serving correctness, and graph-lifetime
safety for the declared envelope. It is not a performance comparison. The
local raw receipt is
`.lumi-workbench/artifacts/moe-phase-hybrid-realmodel-20260808/` and records
the exact commands, hashes, phase logs, client JSON, and before/after device
state.

## Same-NZ performance pilot

A bounded control-vs-hybrid pilot compared the committed prototype against the
original stock selector while holding `enable_fused_mc2=1` in both arms. Both
arms therefore used the same NZ weights; only
`moe_phase_hybrid_policy=off|mixed_prefill_fused` changed. Each arm used one
fresh server followed by three paired seeds, with 24 measured requests plus 4
warmups per seed at 462 input / 16 output tokens and QPS 4. All six measured
trials completed 24/24 with no request errors.

| metric (median over 3 seeds) | stock selector | mixed-only hybrid | delta |
|---|---:|---:|---:|
| request throughput | 3.71 req/s | 3.69 req/s | -0.7% |
| median TTFT | 180.7 ms | 248.6 ms | +37.5% |
| p99 TTFT | 283.5 ms | 497.0 ms | +75.3% |
| median TPOT | 30.6 ms | 43.7 ms | +43.0% |
| mean TPOT | 34.6 ms | 50.3 ms | +45.6% |
| median ITL | 19.5 ms | 20.9 ms | +7.3% |

All three paired seeds show the same direction. The hybrid log records 704
`PURE_DECODE -> ALLGATHER`, 26 `PURE_PREFILL -> ALLGATHER`, and 122
`MIXED -> FUSED_MC2` debug selections; the stock arm selected FUSED_MC2 under
its existing policy. These debug counts are selector invocations across both
ranks, not a direct time decomposition.

**Go/no-go:** do not enable this policy by default for the tested traffic. It
proves that the requested phase split is mechanically possible, but it does
not preserve latency at this workload. The pilot does not isolate a single
cause: the hybrid changes both the decode communication family and the
compiled/eager boundary for PURE_PREFILL/MIXED. In particular, attributing the
regression to the fused MIXED kernel alone would overstate the evidence. A
future attempt would need phase-aware compiled artifacts (or another safe way
to retain compilation) and a workload deliberately rich in naturally
co-scheduled MIXED waves before more campaign investment is justified. Raw
local receipts are under
`.lumi-workbench/artifacts/moe-phase-hybrid-pilot-20260808/`.

## Residual risks and next experiments

- The same-NZ pilot is negative at 462->16 / QPS 4: throughput is flat and
  latency regresses. Keep the policy experimental and default-off; do not
  merge or promote it as an optimization without a new mechanism that avoids
  the eager-boundary cost and fresh evidence on mixed-rich traffic.
- Decode layout disclosure: the 2x910B2 discriminator found NZ stock decode
  3.82--16.68% faster than ND across capacities 2--32. Historical ND parity is
  disproven; use a fresh control and name the hybrid arm "baseline comm on NZ
  weights" rather than unchanged stock decode.
- Only the MIXED phase is included by design. Extending to `PURE_PREFILL`
  fused requires separate measurement and its own validation.
- The TraceLoom numbers suggest the profit region may not match the current
  `MIXED` direction on every SoC/capacity; the mechanism is what is delivered.
  Per-phase/per-capacity crossover must be measured before any change to the
  default.
- Token-count thresholds inside a phase remain forbidden; if a capacity-based
  selector is ever needed it must become a new policy value with its own
  validation, not a per-wave object swap.
- V2 runner support requires upstream `skip_compiled` plumbing (out of scope).
- DP > 1 support requires a cross-rank phase agreement protocol (out of scope).
