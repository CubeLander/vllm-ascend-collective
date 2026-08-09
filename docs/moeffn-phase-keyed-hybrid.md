# Phase-keyed hybrid MoE comm policy (experimental, default OFF)

Status: **experimental mechanism, default disabled**. This document records the
corrected design, the historical failure it must not reproduce, the retired
over-narrow pilot that is retained only as a negative prototype, the safety
invariants and their audit, the exact execution envelope, the selector truth
table, the 2x2 dispatch matrix, and the unit-test contract. It is a
mechanism/design note, not a performance promise.

## Motivation

TraceLoom P2 (see `TRACELOOM_CUDA_HANDOFF_20260804.md`) observed the structural
replacement of the stock MoE path by `DispatchFFNCombineBF16` inside decode ACL
graphs. Its historical selector statistic rose +12.36%/+12.95%, but that
statistic compared a six-task stock compute motif with one larger fused task;
it was not an equivalent-boundary graph-latency measurement. Graph-external
large units moved -11.48% and anchor count moved 1186 -> 994. These numbers are
prototype motivation only; no per-phase performance promise is made here.

The production experiment
`research/co-resident-async-ep/experiments/p0g-production-gmm2-cutthrough/README.md`
(L270) documented the hard architecture gate: a naive `num_tokens >= 16`
selector that swapped the Python comm object (`FusedMC2CommImpl` <-> `AllGatherCommImpl`)
inside one `torch.compile` model graph failed at startup with
`'AllGatherCommImpl' object has no attribute 'expert_token_nums'` — the compiled
template kept the first object's layout. That patch was withdrawn. A safe hybrid
needs exactly one comm object family inside every compiled/ACL template. The
corrected policy achieves this by giving **decode** and **non-decode** separate
execution lanes (see "Two independent decisions" below), so no template ever
changes comm family at runtime. The current canary design additionally makes
the outer compiled graph **opaque with respect to MoE**: under the policy the
Ascend runner selects the already-registered opaque
`torch.ops.vllm.moe_forward` / `torch.ops.vllm.moe_forward_shared` custom ops
(upstream `runner/moe_runner.py`) instead of the raw PrivateUse1
`_moe_forward` functions, so `torch.compile` traces only the op call and the
live `_forward_impl` dispatch happens at op runtime. That is what lets every
phase share one compiled outer artifact while the comm family is chosen per
phase at op runtime.

## Configuration

Additional-config only, default off. There is deliberately **no environment
variable**: repo convention funnels new switches through
`vllm_ascend/ascend_config.py`, and this switch is a config-user choice.

```bash
--additional-config '{"enable_fused_mc2":1,"moe_phase_hybrid_policy":"non_decode_fused"}'
```

`enable_fused_mc2: 1` is REQUIRED, not optional: the fused comm impl and the
NZ weight layout are only initialized in that mode. The startup validation
raises unless both are set, so the experiment cannot silently stay on the
baseline family.

`AscendConfig` parses the value through
`parse_moe_phase_hybrid_policy` (`MoEPhaseHybridPolicy` enum) at startup:

| value | meaning |
|---|---|
| `off` (default when missing) | stock selector, byte-for-byte; no phase keying, no dispatch changes |
| `non_decode_fused` | `PURE_DECODE` selects the baseline comm family, dispatched at opaque-op runtime inside the one compiled outer artifact (`skip_compiled=False`, `force_eager=False`) that FULL_DECODE_ONLY ACL graphs capture and replay; `PURE_PREFILL` and `MIXED` run `FUSED_MC2` through the same single compiled outer artifact outside the ACL graphs (`skip_compiled=False`, `force_eager=True`). If the static fused-MC2 capability gate is unavailable, selection **raises a deterministic `ValueError`** (fail fast) instead of ever falling back to the baseline family |

The accepted values are exactly `off` and `non_decode_fused`. Bool-like
aliases (`0`/`1`/`true`/`false`/`on`/`none`/...) are deliberately rejected so
the switch cannot be set ambiguously. The retired `mixed_prefill_fused`
spelling of the earlier pilot is **also rejected**: it encoded the wrong
over-narrow semantics, and silently mapping it onto the corrected behavior
would change the graph execution that the pilot actually measured. Any
unrecognized value raises `ValueError` at `AscendConfig` construction and
fails closed at startup.

## Forward phase classification

A host-computed **forward phase** (never a device-tensor `.item()` sync)
distinguishes, from the v1 runner's `attn_state`/`with_prefill` CPU state:

- `PURE_DECODE` — every request is decoding / speculative decode
  (`attn_state` in `{DecodeOnly, SpecDecoding}`).
- `PURE_PREFILL` — every request is a brand-new prefill
  (`PrefillNoCache`: zero computed tokens).
- `MIXED` — at least one request is prefilling alongside requests that
  already have computed tokens (`ChunkedPrefill` / `PrefillCacheHit`).

`PURE_PREFILL` and `MIXED` are the **fused non-decode** lanes; `PURE_DECODE`
is the **baseline** lane. There is no token-count-dependent selection
anywhere: per-wave token thresholds are exactly what historically swapped the
comm object inside one compiled template.

## Two independent decisions (the 2x2 matrix)

The corrected policy keeps two independent per-phase controls and never
derives one from the other:

| phase | `skip_compiled` (torch.compile/AOT) | `force_eager` (ACL dispatch) | result |
|---|---|---|---|
| `PURE_DECODE` | `False` | `False` | baseline comm family selected at opaque-op runtime inside the one compiled outer artifact, captured/replayed by FULL_DECODE_ONLY ACL graphs |
| `PURE_PREFILL` | `False` | `True` | fused `FUSED_MC2` via the same one compiled outer artifact, ACL mode NONE |
| `MIXED` | `False` | `True` | fused `FUSED_MC2` via the same one compiled outer artifact, ACL mode NONE |
| policy OFF / `phase=None` / draft | stock | stock | byte-for-byte stock behavior |

- `skip_compiled` is **retired under the canary**: it stays `False` for every
  phase. The outer compiled graph contains only opaque
  `torch.ops.vllm.moe_forward` calls and the live `_forward_impl` dispatch
  happens at op runtime, so decode shares the one compiled outer artifact with
  the non-decode phases; the comm family (baseline vs `FUSED_MC2`) is chosen
  per phase at op runtime, never baked into the compiled graph. An unmatched
  decode wave therefore safely runs the compiled outer graph with the
  baseline family dispatched at op runtime.
- `force_eager=True` (only `PURE_PREFILL`/`MIXED` under the policy) forces
  `CUDAGraphMode.NONE` in `_determine_batch_execution_and_padding`, so the
  outer `ACLGraphWrapper` falls through and the wave runs the fused compiled
  artifact. This is the belt-and-braces guarantee that a non-decode wave is
  never dispatched into (or replayed from) a FULL template, even when the
  runner's `uniform_decode` heuristic misclassifies a 1-token prefill as a
  uniform decode wave because `speculative_config` is absent.
- The two controls never combine on one phase; in particular
  `skip_compiled` is **never fed into** `force_eager`.

## Execution envelope (fail closed)

The minimal safe envelope is **v1 model runner + `cudagraph_mode ==
FULL_DECODE_ONLY` + `data_parallel_size == 1`**. `validate_moe_phase_hybrid_policy`
runs at startup (platform `check_and_update_config`, after all platform mode
fallbacks) and raises otherwise:

- `FULL` / `FULL_AND_PIECEWISE` / `PIECEWISE`: prefill or mixed waves can be
  dispatched into a graph template there. That would either replay a raw
  baseline template under a fused intent or trigger runtime capture after
  capture is disabled. `force_eager` alone does not protect that envelope,
  so these modes are rejected.
- `NONE` (no graphs): rejected — the invariant "decode is the only
  compiled/ACL-graph path" requires decode graphs to exist.
- `enable_fused_mc2 != 1`: rejected — the fused comm impl and the NZ weight
  layout are only initialized with `enable_fused_mc2 == 1`; without it the
  policy would silently stay on the baseline family and the experiment would
  not actually be enabled.
- SoC/EP capability gate: the static fused-MC2 gate (A3 `enable_fused_mc2==1`
  and EP<=32; A2 fused-A2 conditions) cannot be checked at platform startup
  because the EP process group is not initialized yet. It is enforced **fail
  fast** by `select_moe_comm_method` at the first non-decode forward — under
  this policy that is the unconditional fused `MIXED` profile dummy, which
  runs before `super().profile_run()` and before any decode warmup/capture —
  so an unsupported SoC/EP configuration raises a clear deterministic error
  before any compiled artifact exists. Non-decode never falls back to the
  token-dependent baseline family.
- V2 runner: rejected — upstream V2 owns `skip_compiled`/`force_eager` itself;
  vllm-ascend cannot enforce the phase-keyed matrix from the platform hook.
- DP > 1: rejected — DP ranks can schedule different phases for the same step,
  while the fused MC2 collectives span all DP ranks; a per-rank phase-
  dependent comm choice is not DP-consistent.

## Selector truth table

`select_moe_comm_method(num_tokens, vllm_config, is_draft_model=False, *, phase=None)`:

| policy | phase | EP / SoC gate | result |
|---|---|---|---|
| OFF (default) | any / None | any | stock (unchanged, incl. `None` for non-MoE) |
| ON | `None` (legacy callers, e.g. spec-decode proposers) | any | stock |
| ON | draft model | any | stock |
| ON | `PURE_PREFILL` / `MIXED` | A3: `enable_fused_mc2==1` and EP<=32; A2: fused-A2 gate | `FUSED_MC2` |
| ON | `PURE_PREFILL` / `MIXED` | gate fails | **raise `ValueError` (fail fast)** — never a baseline fallback |
| ON | `PURE_DECODE` | any | baseline (A2/TP2: `ALLGATHER`) |
| ON | any | non-MoE | `None` |

Baseline family by SoC (identical to stock with `enable_fused_mc2=0`): A2 →
`MC2` when `num_experts_per_device <= 24`, EP>=16 and tokens fit MC2 capacity,
else `ALLGATHER`; A3 → `MC2` within MC2 capacity else `ALLTOALL`; 310P →
`ALLGATHER`; A5 → stock (no fused branch there).

## Why this cannot reproduce the historical stale-layout failure

1. **One template = one comm family.** The decode ACL templates are captured
   from the compiled outer artifact with the baseline selector, and the comm
   family is dispatched at opaque-op runtime during capture; runtime decode
   replays exactly what was captured. The fused non-decode path runs through
   the same one torch.compile/AOT artifact outside the ACL graphs; the op
   runtime dispatch always selects the same family for a given process (the
   SoC gate result is static). No phase and no token threshold can swap a comm
   object inside any template.
2. **The compiled outer artifact is phase-agnostic.** It contains only opaque
   `torch.ops.vllm.moe_forward` calls, so the same artifact serves decode and
   non-decode; the baseline/fused choice happens in `_forward_impl` at op
   runtime. An unmatched/runtime decode wave that falls through the ACL
   wrapper (mode NONE) safely runs the compiled outer graph with the baseline
   family dispatched at op runtime.

**No ACL cache-key change is needed.** `ACLGraphWrapper`/`compute_acl_graph_cache_key`
are untouched. Additionally the policy is **deliberately token-count
independent**; no `num_tokens >= N` logic exists in the hybrid selector.

## Capture / dummy-run / profile ordering audit

The requirement is that the fused non-decode compile and its workspace are
deterministically established **before** any decode warmup/capture, and that
memory profiling includes both the FUSED workspace and the baseline
workspace.

- `profile_run` (v1 runner): when the policy is active the fused non-decode
  dummy is **unconditional and first**: it runs even when
  `max_num_tokens <= mc2_tokens_capacity` (the historical MC2-capacity
  condition could skip it), with size `min(self.max_num_tokens,
  mc2_tokens_capacity)` (valid for `_dummy_run`'s `num_tokens <=
  max_num_batched_tokens` assertion), `with_prefill=True`,
  `moe_forward_phase=MIXED`, `is_profile=True` (ACL NONE). It compiles the
  fused artifact and allocates its workspace before `super().profile_run()`
  and before any decode warmup/capture; if the static fused gate is
  unavailable it is also the deterministic fail-fast point. In the
  `--kv-cache-memory` fast path the same `profile_run` still runs, so the
  fused compile happens there too. Policy OFF keeps the historical
  conditional MC2-capacity dummy byte-for-byte (with its historical
  `PURE_PREFILL` phase).
- Capture warmup (`_warmup_and_capture` -> `_dummy_run`, and `capture_model`)
  only runs decode-style dummy runs (`with_prefill=False` -> `PURE_DECODE`),
  so every captured graph is a decode template of the compiled outer artifact
  with the baseline family selected at opaque-op runtime — exactly matching
  runtime decode under the policy, with no retrace. The decode warmup
  allocates the baseline workspace inside the memory-profiling window, so the
  profile records both workspaces.
- `_dummy_run` defaults to the same host phase classification
  (`PURE_PREFILL` if `with_prefill` else `PURE_DECODE`), accepts an explicit
  phase override, and applies the same phase-keyed `force_eager` as the
  runtime forward, so a non-decode dummy can never capture a fused template
  into the decode graph pool.
- `compile_or_warm_up_model` (worker): the generic compile-size warmups pass
  the explicit non-decode phase under the policy. This keeps the fused
  non-decode lane deterministically compiled first and its workspace
  established before decode warmup/capture. Policy off keeps the historical
  `_dummy_run(size)` behavior.

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
stock-`ALLGATHER` discriminator has measured that distinction with ACL graph
enabled, 8 warmups and 30 bracketed samples per arm. Both layouts were
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

## Retired `mixed_prefill_fused` pilot (negative prototype)

The earlier experimental value `mixed_prefill_fused` and its committed
prototype are **retired and rejected by the corrected design**. What the pilot
actually tested was a different, over-narrow policy:

- only `MIXED` selected `FUSED_MC2`; `PURE_PREFILL` kept the baseline comm
  family; and
- `PURE_PREFILL`/`MIXED` **bypassed the compiled model entirely**
  (`skip_compiled=True` + eager dispatch), so the non-decode lanes ran raw
  eager with no torch.compile artifact.

The corrected `non_decode_fused` policy changes both decisions: non-decode
now selects `FUSED_MC2` for `PURE_PREFILL` **and** `MIXED`, and keeps the
torch.compile/AOT artifact (the fused path runs *compiled*, just outside the
ACL graphs); decode is the only lane that goes raw, and it goes raw inside the
ACL graphs where its baseline comm family is captured and replayed exactly.
Because the compiled/eager boundary is the opposite of what the pilot
measured, **the pilot's negative result does not reject the corrected design.**

The pilot evidence is retained here as a negative prototype of the naive
boundary cost and of the old envelope:

The bounded control-vs-hybrid pilot compared the old prototype against the
original stock selector while holding `enable_fused_mc2=1` in both arms. Each
arm used one fresh server followed by three paired seeds, with 24 measured
requests plus 4 warmups per seed at 462 input / 16 output tokens and QPS 4.
All six measured trials completed 24/24 with no request errors.

| metric (median over 3 seeds) | stock selector | mixed-only hybrid | delta |
|---|---:|---:|---:|
| request throughput | 3.71 req/s | 3.69 req/s | -0.7% |
| median TTFT | 180.7 ms | 248.6 ms | +37.5% |
| p99 TTFT | 283.5 ms | 497.0 ms | +75.3% |
| median TPOT | 30.6 ms | 43.7 ms | +43.0% |
| mean TPOT | 34.6 ms | 50.3 ms | +45.6% |
| median ITL | 19.5 ms | 20.9 ms | +7.3% |

The old hybrid log records 704 `PURE_DECODE -> ALLGATHER`, 26
`PURE_PREFILL -> ALLGATHER`, and 122 `MIXED -> FUSED_MC2` debug selections;
the stock arm selected FUSED_MC2 under its existing policy. These debug
counts are selector invocations across both ranks, not a direct time
decomposition. The pilot's regression was never attributed to the fused
MIXED kernel alone: it changed both the decode communication family and the
compiled/eager boundary. Raw local receipts are under
`.lumi-workbench/artifacts/moe-phase-hybrid-pilot-20260808/`.

## Raw-decode ACL graph gate and crossed result (retired as measured fallback)

**Retired as the current design.** The intermediate raw-decode mechanism moved
decode **out of the compiled artifact and into raw ops inside FULL_DECODE_ONLY
ACL graphs** (`skip_compiled=True` on `PURE_DECODE`). It is kept in this
document only as a **measured fallback**: the crossed comparison below is the
evidence for what the raw capture costs, and the current opaque-MoE canary is
the remaining discriminator that the raw design could not resolve (a
*compiled* baseline decode ACL graph alongside the compiled fused non-decode
lane). Under the canary, `skip_compiled` is retired and decode runs the one
compiled outer artifact with the baseline family selected at opaque-op
runtime, so the raw design's compile/raw confound is gone.

The raw mechanism gate passed on 2x910B2 with Qwen3-30B-A3B BF16, TP2/EP2,
DP1, and capture sizes 2/4/8/16/32. The explicit MIXED profile dummy selected
FUSED_MC2 and compiled first; every decode warmup/capture selected baseline
ALLGATHER; 14 ACL graphs captured; runtime PURE_PREFILL/MIXED remained fused
outside ACL while 93 decode steps replayed baseline graphs. The 8-request
462/16 correctness gate and a deterministic post-MIXED completion passed with
no stale object, unexpected recompile, OOM, 507015, or request error. The raw
local receipt is
`.lumi-workbench/artifacts/nondecode-fused-realmodel-gate-20260808/`.

A subsequent same-NZ, crossed-order comparison used the historical 462 input /
256 output / QPS 4 workload. Both arms kept `enable_fused_mc2=1` and therefore
used the same NZ weights and fused compiled non-decode lane. The corrected arm
changed only pure-decode execution from the control's compiled FUSED_MC2 graph
to a raw baseline-ALLGATHER ACL graph. All four cells completed 32/32 requests
at exact requested lengths with zero failures:

| metric | pair 1 (control first) | pair 2 (corrected first) | mean direction |
|---|---:|---:|---:|
| request throughput | -8.0% | -6.0% | **-7.0%** |
| median TTFT | +6.2% | +19.3% | **+13.1%** |
| mean TPOT | +13.2% | +13.1% | **+13.1%** |
| median TPOT | +14.9% | +13.6% | **+14.2%** |
| median ITL | +19.2% | +14.6% | **+16.9%** |

The stable metrics agree across reversed order: **the raw-baseline decode graph
is slower than the compiled-fused control on this workload.** This result does
not establish that a *compiled* baseline decode graph is slower: the raw arm
changed both the decode comm family and the decode compiled/raw boundary, and
TraceLoom's earlier local fused-window regression cannot compensate for losing
compile transformations elsewhere in the graph. The canary's whole point is to
measure the remaining discriminator without that confound. The raw comparison
receipt is `.lumi-workbench/artifacts/nondecode-fused-462x256-crossed-20260808/`.

## Opaque-MoE canary result

The policy-local opaque boundary passed its 2x910B2 NPU canary. Each worker
compiled exactly once; the retained FX graph contains 48
`torch.ops.vllm.moe_forward.default` calls per rank and no raw
`_moe_forward` body. Runtime safely alternated PURE_PREFILL/MIXED FUSED_MC2
with PURE_DECODE baseline ALLGATHER, then captured/replayed decode ACL graphs,
without a recompile, stale object, PrivateUse1 custom-op error, OOM, or 507015.
The short correctness gate completed 4/4 exact 462/16 requests plus a
deterministic post-MIXED completion. Receipt:
`.lumi-workbench/artifacts/opaque-moe-npu-canary-20260808/`.

The same historical 462/256/QPS4 crossed comparison was then repeated with
the compile/raw confound removed. Both arms used the same compile settings;
the corrected arm used compiled opaque baseline decode graphs plus fused
non-decode, while the control kept stock compiled FUSED_MC2 decode. All four
cells completed 32/32 with no errors, but the corrected arm remained slower
in both orders:

| metric | pair 1 | pair 2 |
|---|---:|---:|
| request throughput | -5.1% | -4.7% |
| mean TPOT | +3.36 ms (+10.1%) | +2.46 ms (+7.3%) |
| median TPOT | +3.54 ms | +2.75 ms |
| median ITL | +2.96 ms | +3.13 ms |

Therefore the compile boundary explained part, but not all, of the raw
prototype's loss. With compilation preserved, replacing stock FUSED_MC2
decode by baseline ALLGATHER still regresses the whole serving path. The
earlier TraceLoom result was local: its historical 48-position statistic rose,
but the stock selector spanned six visible compute tasks whereas the fused
selector spanned one larger `DispatchFFNCombineBF16` task. The corrected
whole-body direction was explicitly inconclusive. It did not establish either
that the fused kernel itself regressed or that replacing the complete fused
decode path would improve the graph. Receipt:
`.lumi-workbench/artifacts/opaque-moe-462x256-crossed-20260808/`.

That comparison still changed the compiler boundary relative to the stock
control. A final exact isolation therefore compared two opaque arms with the
same code, weights, compile settings, graph dispatch, and fused nondecode lane:
`opaque_fused_control` used FUSED_MC2 for PURE_DECODE, while
`non_decode_fused` used ALLGATHER. In crossed order, FUSED minus baseline was
+0.8% / +5.0% request throughput and -0.2% / -7.2% mean TPOT (mean-of-pairs
+2.9% throughput, -1.9% TPOT). Pair magnitude varied, so the defensible verdict
is **performance-neutral-to-positive for FUSED decode**, not a precise speedup.
The decode-family switch alone does not reproduce the earlier 3 ms penalty;
compiler opacity was the dominant confound in that comparison. Receipt:
`.lumi-workbench/artifacts/opaque-exact-decode-control-20260808/`.

## Unit-test contract

`tests/ut/test_ascend_forward_context.py`, `tests/ut/ops/test_moe_opaque_canary.py`
and `tests/ut/test_ascend_config.py` encode: the selector truth table (decode
baseline on A2/TP2 and A3; `PURE_PREFILL`/`MIXED` fused under gate;
**gate-fail raises `ValueError` fail-fast, never a baseline fallback**;
`enable_fused_mc2 != 1` rejected at startup; A5/310P; draft/non-MoE/
`phase=None`), default compatibility (policy OFF identical to stock for every
phase), strict config parsing (default OFF, `non_decode_fused` accepted,
bool-like aliases and the retired `mixed_prefill_fused` spelling raise),
fail-closed validation (DP / V2 / cudagraph mode / `enable_fused_mc2 != 1`),
the hybrid-active predicate (`PURE_PREFILL` + `MIXED` only), the
bypass-compiled predicate **retired to constant `False` under the opaque
canary** (`skip_compiled` is false for every phase), the force-eager predicate
(non-decode only), the **2x2 matrix invariant** (the two controls never
combine on one phase: decode is `(False, False)`, non-decode is
`(False, True)`), `skip_compiled` propagation through
`set_ascend_forward_context`, the explicit profile/warmup ordering helper
(`get_moe_phase_hybrid_warmup_phase`), and the opaque-selection contract
(`AscendMoERunner._select_forward`: policy ON returns
`torch.ops.vllm.moe_forward` / `torch.ops.vllm.moe_forward_shared`; policy OFF
delegates to `super()._select_forward()` byte-for-byte, which on
PrivateUse1/Ascend is the raw `_moe_forward` / `_moe_forward_shared`
selection; an unavailable policy config cannot change stock selection; the
info-once marker fires at most once). Runner-level tests assert that under
the active policy `profile_run` issues the fused `MIXED` dummy
**unconditionally and first** (including `max_num_tokens <= capacity` and
non-MC2 selector results, with size `min(max_num_tokens, capacity)`), while
policy OFF keeps the historical conditional `PURE_PREFILL` dummy; worker-level
tests assert compile warmups pass the explicit `MIXED` phase under the policy
and stay positional `_dummy_run(size)` when OFF, including the empty-warmup
case.

## Residual risks and next experiments

- Both discriminators are negative: raw baseline decode loses 6--8% request
  throughput, and compiled opaque baseline decode still loses 4.7--5.1% on
  462x256. Keep `non_decode_fused` experimental/default-off; do not promote a
  baseline-decode phase switch for this workload.
- The opaque op itself is mechanism-correct on the tested A2/TP2 envelope and
  removes the stale compiled-object limitation. It remains useful experimental
  infrastructure, but it is not a performance win by itself and should not be
  generalized until its wider NPU compatibility is separately justified.
- Stale out-of-scope text: the `MoEPhaseHybridPolicy` enum docstring in
  `vllm_ascend/ascend_config.py`, the worker warmup comment in
  `vllm_ascend/worker/worker.py`, and the platform comment in
  `vllm_ascend/platform.py` still describe the retired raw-capture decode
  (`skip_compiled=True`); they were outside this canary's assigned file set
  and should be corrected when the design settles.
- Decode layout disclosure: the hybrid decode arm is baseline comm on NZ
  weights, not the historical ND stock decode. Compare against a fresh
  control (same `enable_fused_mc2=1` arm with `moe_phase_hybrid_policy=off`).
- Token-count thresholds inside a phase remain forbidden; if a capacity-based
  selector is ever needed it must become a new policy value with its own
  validation, not a per-wave object swap.
- V2 runner support requires upstream `skip_compiled`/`force_eager` plumbing
  (out of scope).
- DP > 1 support requires a cross-rank phase agreement protocol (out of scope).
