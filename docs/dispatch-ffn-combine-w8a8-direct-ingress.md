# Dispatch-FFN-Combine W8A8 direct ingress

Status: accepted single-node compile-time mechanism. Correctness passes on
EP2, EP4, and EP8; a DeepSeek-V4-Flash W8A8 TP8/EP8 smoke run selected
`FUSED_MC2` on all ranks and completed. Production-shaped operator
measurements retain the direct path only where it shows a repeatable
kernel-level benefit. Warmed layerwise eager and graph-replay measurements
pass the Phase 2 mechanism gate. Fresh-server end-to-end brackets on the
shared host are quantitatively inconclusive because enclosing baseline drift
exceeds the expected effect; they remain functional, warmup, and
large-regression smoke rather than evidence against the layerwise win.

The evaluated candidate defines both
`DISPATCH_FFN_COMBINE_W8A8_DIRECT_INGRESS` and
`DISPATCH_FFN_COMBINE_W8A8_DIRECT_INGRESS_SPARSE_FALLBACK`. The first enables
the direct data path; the second enables the EP2/EP4 graph preference exchange
and is rejected at compile time unless the first is also defined. The shared
epoch helper is hardened independently so an impossible generation skew fails
fast rather than waiting forever.

## Performance acceptance policy

Phase 2 performance is decided by warmed, production-shaped layerwise evidence
before collective generalization or service-level promotion work. A direct
shape must win in eager microbenchmarks on the distributed critical path and
remain positive in repeated decode-graph replay; fallback shapes must preserve
baseline parity. Correctness, changing-generation reuse, graph masking, and
fail-fast protocol gates remain mandatory.

Fresh-server end-to-end runs on a shared machine are smoke tests for path
selection, semantic completion, stability, and large regressions. They do not
arbitrate a low-single-digit kernel effect when the two enclosing executions
of the same baseline drift by more than that effect. A quantitative
service-level gate becomes authoritative only in an isolated environment with
stable enclosing baselines and matched routing inputs.

The present evidence satisfies the layerwise mechanism criterion: EP8 exact
`M=8` improves the all-rank operator median by about 4%, warmed eager rank-0
`M=8` improves 2.38%, graph-contained `DispatchFFNCombine` improves 3.64% per
layer, and graph replay median improves 1.53%. The decreasing magnitude is
consistent with dilution by surrounding graph work rather than disappearance
of the operator benefit.

Mechanism acceptance and deployment scope are separate decisions. The current
guards are compile time and do not prove a host predicate that restricts the
protocol to one node. Single-node EP2, EP4, and EP8 are covered; multi-node EP
is not. The rollout boundary is therefore a named single-node A2 candidate
package, while unknown and multi-node topologies retain the macro-off package.
This is an operability boundary, not a request for another shared-host
end-to-end performance campaign.

## Purpose

The BF16 direct-ingress work removed receiver-side, per-expert pulls after the
route-count exchange. This prototype transfers the same ownership and epoch
protocol to the fused W8A8 path without discarding W8A8's dynamic per-token
scale.

Each source already learns the complete route-count matrix. It can therefore
derive both its source-row prefix and the final expert-major destination row.
No remote allocator or second reservation exchange is necessary.

## Protocol

```text
quantize and route into source-owned peer staging
  -> exchange tagged count rows
  -> derive disjoint destination prefixes
  -> push each INT8 row and its FP32 dynamic scale to final input
  -> join local producers and complete their DDR writes
  -> publish one source-owned uint32 epoch
  -> wait for every source epoch
  -> order completed epoch reads before local GMM1 consumption
```

The direct input and scale buffers occupy peer-window space beginning at the
old return offset. Their row capacity is bounded by
`min(max_output_size, EP * M * topK)`, and the return buffer moves after them.
If the resulting layout overlaps the control region, construction fails closed
to the existing gather path.

The epoch is a cache-line-separated, source-owned unsigned 32-bit modular
counter. A waiter accepts the requested generation or one generation ahead,
temporarily tolerates one generation behind during publication, and traps on
any larger skew. A larger skew means the single payload buffer cannot contain
the requested wave, so recovery inside the operator would be unsound.

## W8A8-specific data movement

`moe_init_routing_quant_v2` writes an INT8 row plus aligned padding containing
that row's dynamic scale. Direct ingress reuses the existing per-token copy
primitive to push the compact INT8 payload and extract the scale into a
separate destination scale array. GMM1 therefore sees the same input and scale
layout as the existing gather path.

Producer MTE3 writes are completed with a DDR data-sync barrier before the
local join. The publishing core orders the scalar epoch store, and the receiver
orders completed epoch reads before Cube consumption. DCCI is restricted to
the scalar control accesses rather than scanning DMA payloads.

## Selection policy

W8A8's existing gather overlaps per-expert ingress with GMM1, while direct
ingress seals the complete wave. The crossover depends on EP fan-out as well
as active token count, so the final policy is deliberately asymmetric:

| Execution shape | Gather | Direct ingress |
|---|---|---|
| EP2/EP4 exact | `M <= 2` | `M >= 3` |
| EP2/EP4 graph mask | active tokens `<= 1` | active tokens `>= 2` |
| EP8 exact | `M <= 7` | `M >= 8` |
| EP8 graph mask | all measured shapes | disabled |

Exact-shape and all EP8 graph fallbacks are chosen during buffer construction.
They retain the original peer layout and pay no distributed selector cost.

For EP2/EP4 graph execution, padded `M` hides the active prefix length. Each
rank publishes one preference bit in the count row's first padding lane. The
existing tagged count exchange carries it for free, and every receiver takes
an all-source AND over only `EP` integers. This uniform decision prevents peers
from entering different protocols.

## Reproducible opt-in package

The repository's existing `--ops-compile-options` surface is sufficient; a
second feature configuration mechanism is unnecessary. From the repository
root, build the single-node candidate with:

```bash
bash csrc/build.sh \
  --ops=dispatch_ffn_combine \
  --soc=ascend910b \
  --vendor_name=custom_transformer \
  --pkg \
  --ops-compile-options \
  '-DDISPATCH_FFN_COMBINE_W8A8_DIRECT_INGRESS;-DDISPATCH_FFN_COMBINE_W8A8_DIRECT_INGRESS_SPARSE_FALLBACK'
```

Do not trust an incremental build directory to identify the macro set.
Generated operator files can retain earlier options. Every candidate receipt
must record the source commit, dirty-worktree state, complete
`OPS_COMPILE_OPTIONS`, generated `custom_compile_options.ini` row, and SHA-256
hashes of all installed object variants. Build from a clean operator output or
an isolated output directory.

### W8A8 object receipt, 2026-08-06

The first packaging gate is complete. An isolated three-variant `opc` compile
used repository checkpoint `9b8d9ef56`; the tracked operator and shared epoch
helper are unchanged from implementation commit `d03603239`. The tracked
worktree was clean.

The generated option set was:

```text
-DDISPATCH_FFN_COMBINE_W8A8_DIRECT_INGRESS;-DDISPATCH_FFN_COMBINE_W8A8_DIRECT_INGRESS_SPARSE_FALLBACK;-Wno-ignored-attributes;-Wno-ignored-attributes;-Wno-ignored-attributes
```

The isolated generated kernel and helper were byte-identical to the tracked
files:

| Variant suffix | Bytes | Object SHA-256 | JSON SHA-256 |
|---|---:|---|---|
| `9907fbc1444e58de3ad65d22316d7b8f` | 224,800 | `d0bc902eb004559e92d36cc1d83f998cd14175ae92b5ac703c093210b65ca49a` | `e0968dce418c6416bf325b68c106b52fc536d69ed14317e0ef872ad666f36b59` |
| `af7cdba33254528b52a5326f36d262f5` | 224,824 | `4e531dbfbf4fe689665cdd4612d0bf485f9041b0d7638350891faff1d64c8b89` | `53f691985d16811ad3f9f21a73a386c8a9588a4f174b252f936ab164cfff6c25` |
| `b12576f77f0efc7337f4f4527a4d909f` | 224,824 | `4279e7fc78ba84f99021c5b1ddc3b7db67fdaec9636add96747c75f251c5e2df` | `da68362b7ed90297442f8f31d50b2a1d0a4b0bf62dbec8944a8023988438b4a1` |

All six files byte-match the final selector-v6 objects used for EP2, EP4, EP8,
layerwise eager, graph replay, profiler, and real-model smoke evidence. Those
results therefore transfer without another noisy performance run. An
installer-level receipt is still required on the final common integration
commit.

After installation, rerun EP2, EP4, and EP8 changing-generation correctness,
the eager and graph selector-boundary cases with workload warmup, and the
DeepSeek-V4-Flash semantic and short stability smoke. Canary only on a declared
single-node A2 service pool. Rollback drains the canary, reinstalls the recorded
macro-off package, verifies restored object hashes, and repeats the semantic
smoke.

## Correctness and fault evidence

The bounded regression exercises changing dense routes, exact output and
expert-count oracles, an all-zero count row followed by peer-window reuse, and
a graph-sized buffer with one active token. EP2, EP4, and EP8 pass against the
final candidate objects:

```text
3 passed, 21 warnings in 159.14s
```

The EP8 case uses 64 local experts, covers 512 global expert IDs, and changes
routes across generations. All eight devices were idle before the run and
returned idle afterward.

The epoch-skew gate is also complete. A disposable object advanced rank 0 by
two extra generations on its first direct wave. Both EP2 ranks terminated with
device error `507015` in less than 60 seconds; the test did not reach its
90-second timeout. Restoring the clean object by recorded SHA-256 hashes and
rerunning EP2 then passed in 62.63 seconds. The injection exists only in the
ignored workbench receipt, not in tracked source.

## Production-shaped operator measurements

The microbenchmarks use the DeepSeek-V4-Flash W8A8 operator shape: hidden size
4096, FFN size 2048, 256 global experts, top-K 6, and
`max_output_size=131072`. They measure the slower rank's device-event time.

### EP4 masked graph

For graph `M=32`, the selector was bracketed against the original gather
objects in fresh processes:

| Active tokens | Gather median | Selector median | Improvement |
|---:|---:|---:|---:|
| 1 | 550.07 us | 457.22 us | 20.31% |
| 2 | 601.61 us | 463.02 us | 29.93% |
| 4 | 553.76 us | 489.12 us | 13.22% |
| 7 | 560.34 us | 520.96 us | 7.56% |

Paired-generation wins were 40/40, 40/40, 39/40, and 36/40 respectively. A
separately compiled direct-only object measured 496.43 us at active 1 and
464.86 us at active 2. This distinguishes the intended selection: active 1
uses fallback, while active 2 is at direct-only parity.

Exact EP4 `M=1` and `M=2` were remeasured in selector/baseline/selector order.
The observations were 530.31/521.66/482.61 us and
532.13/527.47/499.39 us respectively. Selector-run medians improve on the
enclosed baseline by 2.9% and 2.2%, but the individual observations span a
small regression through a larger win. The exact-small fallback is therefore
at parity within fresh-process variance.

### EP8 crossover

EP4's graph result does not transfer directly to EP8: the sealed wave has twice
the coordination fan-out. Exact-shape measurements locate a useful crossover
at `M=8`:

| Exact tokens | Gather median | Final selector median | Improvement |
|---:|---:|---:|---:|
| 4 | 415.42 us | 413.71 us | 0.41% |
| 7 | 425.52 us | 441.00 us | -3.51% |
| 8 | 462.41 us | 444.79 us | 3.96% |

`M=4` and `M=7` take the gather fallback; their variation is process noise.
`M=8` takes direct ingress and retains the earlier candidate's approximately
4--5% median win. All measured outputs were exact zero for the zero-weight
oracle.

The earlier EP8 masked candidate used direct ingress and regressed at nearly
every tested point: -1.0%, -7.6%, -4.7%, -15.7%, -5.6%, and -7.3% at active
1, 2, 4, 7, 8, and 16; active 32's 1.7% apparent win was within noise. The
final candidate therefore binds gather for every EP8 graph-mask call. A
same-window fallback/baseline bracket measured deltas from -2.4% to +6.3%
across active 1, 2, 4, 7, 8, 16, and 32, with six of seven non-negative and
all outputs exact. This is baseline parity, not a direct-ingress speedup.

These are operator microbenchmarks, not an end-to-end model throughput claim.

## Profiler evidence and rejected prefix candidate

A one-wave torch-NPU Level1 collection profiled rank 0 while every peer still
participated in a fresh process group. It is mechanism evidence rather than a
distributed critical-path measurement:

| Shape | Kernel duration | AIC MAC | AIC scalar | AIC MTE2 | AIV scalar |
|---|---:|---:|---:|---:|---:|
| EP4 graph `M=32`, active 2 | 251.19 us | 2.9% | 41.4% | 14.6% | 33.6% |
| EP8 exact `M=8` | 225.07 us | 5.4% | 33.0% | 31.8% | 29.0% |

The surviving paths are control/scalar-heavy rather than MAC- or payload-
bandwidth-bound. EP8 also carries material MTE2 pressure. Further work should
therefore attribute waits and scalar control before changing copy granularity.

A disposable candidate replaced on-demand nested prefix calculation with
contiguous per-core expert blocks and incrementally advanced prefixes. It
passed the full EP2/EP4/EP8 exact-output suite, but a
candidate/baseline/candidate bracket rejected it:

| Shape | Candidate 1 | Baseline | Candidate 2 | Candidate regression |
|---|---:|---:|---:|---:|
| EP8 exact `M=8` | 460.16 us | 391.27 us | 418.39 us | 10.9% |
| EP4 graph active 2 | 469.09 us | 438.53 us | 484.27 us | 8.0% |

The dense prefix preparation does work for empty targets and gives up the
cyclic expert distribution. It is not present in tracked source. Do not replace
sparse on-demand prefix calculation with this form of dense per-core scan.

A second disposable candidate removed the logically redundant observation of
the publishing rank's own epoch. The local producer barrier already precedes
publication, so EP2/EP4/EP8 correctness still passed. Performance did not: a
candidate/v6/candidate EP8 exact-`M=8` bracket measured 425.04, 388.87, and
413.17 us, and the paired candidate median regressed v6 by 7.63%. The self poll
is therefore retained; its position in the ordered polling loop appears to
provide useful pacing or instruction-layout behavior that the source-level
invariant alone does not predict.

## Real-model EP8 smoke

DeepSeek-V4-Flash W8A8 was started at TP8/EP8 with the production fused-MC2
selection. The W8A8 branch needed the packed compressed-KV runtime commits to
initialize its KV cache, and the source checkout needed the matching current
extension binary. With that bounded runtime composition:

- all eight ranks selected `FUSED_MC2`;
- warmup exercised fused calls at 8 and 768 tokens;
- a semantic request returned the expected capital continuation;
- all 8/8 benchmark requests completed;
- the server exited cleanly and all devices returned idle.

The smoke observed 8.59 output tokens/s and 624.74 ms mean TPOT, but it was not
an A/B run and is evidence only for real-model correctness and selectability.

## Real-model end-to-end bracket

A subsequent baseline/candidate/baseline bracket used the same packed-KV
DeepSeek-V4-Flash W8A8 runtime and a fresh server for each leg. Every server
was launched with `--enforce-eager`, so this bracket covers the mixed/eager
path rather than ACLGraph decode. Every server first completed a semantic
request. Each workload then ran 16 same-shape
requests plus two benchmark-internal warmups, all excluded from measurement,
before 32 measured requests with four further benchmark-internal warmups.
Inputs were 64 tokens, outputs were 16 tokens, and prefix caching was disabled.
The two measured workloads were arrival rate 2 and an unbounded burst with up
to eight concurrent sequences.

All three legs selected `FUSED_MC2`, returned the same semantic continuation,
completed every request, shut down cleanly, and left all eight devices idle.
The baseline object hashes matched across both baseline legs, and the cleanup
trap restored the final candidate hashes. The burst produced exact `M=8`
fused calls, so it exercised EP8 direct ingress rather than only its fallback.

The primary comparison is the candidate against the mean of the two enclosing
baselines:

| Workload | Metric | Baseline 1 | Candidate | Baseline 2 | Improvement | Baseline drift |
|---|---|---:|---:|---:|---:|---:|
| arrival rate 2 | output throughput | 15.57 tok/s | 15.90 tok/s | 16.02 tok/s | +0.70% | +2.89% |
| arrival rate 2 | mean TPOT | 425.35 ms | 415.66 ms | 413.25 ms | +0.87% | -2.84% |
| arrival rate 2 | mean E2EL | 12359.17 ms | 11916.52 ms | 11770.25 ms | +1.23% | -4.77% |
| burst | output throughput | 19.43 tok/s | 18.95 tok/s | 19.18 tok/s | -1.82% | -1.26% |
| burst | mean TPOT | 366.95 ms | 383.26 ms | 378.77 ms | -2.79% | +3.22% |
| burst | mean E2EL | 16645.37 ms | 16830.80 ms | 16637.69 ms | -1.14% | -0.05% |

Positive improvement means higher throughput or lower latency. The low-load
differences are smaller than the baseline drift and establish parity, not a
gain. In the saturated workload the candidate is below both baselines on
throughput and TPOT, but the 1.8--2.8% deltas are comparable to run-to-run
variation. The honest conclusion is therefore no demonstrated end-to-end
speedup, with a possible small saturated regression. The approximately 4%
EP8 `M=8` isolated-kernel win does not currently justify enabling this path in
a production model build.

## Warmed real-model msprof comparison

The end-to-end result was followed by matched dynamic `msprof` collections on
rank 0. Candidate and baseline each used a fresh server, the same semantic and
16-request burst warmup, and four excluded benchmark warmups. After attaching,
an additional start/stop window primed the profiler itself; only the second
start/stop window was interpreted. This avoids attributing `msprof`'s first-task
overhead to either operator.

The captured request produced 43 fused calls at each of `M=64`, `M=1`, and
`M=449`, followed by 215 calls at `M=8` and 86 at `M=7`. Server logs
independently confirmed the same shape sequence. The candidate takes direct
ingress at `M=64`, `M=449`, and `M=8`, and gather fallback at `M=1` and `M=7`.

| Shape | Baseline mean | Candidate mean | Mean improvement | Median improvement | P95 improvement |
|---:|---:|---:|---:|---:|---:|
| `M=1` | 182.02 us | 181.73 us | 0.16% | 0.28% | 0.45% |
| `M=7` | 205.14 us | 205.66 us | -0.25% | -0.26% | 0.48% |
| `M=8` | 207.18 us | 201.88 us | 2.56% | 2.38% | 4.67% |
| `M=64` | 829.44 us | 359.04 us | 56.71% | 3.09% | 74.13% |
| `M=449` | 572.69 us | 566.07 us | 1.16% | 1.79% | 1.16% |

The `M=64` mean and P95 are dominated by a baseline-only transient in the first
12 layers, including one 4.73 ms call. Its 3.09% median delta is the defensible
steady comparison; the large tail reduction is an observation to reproduce,
not a promotion claim. The fallback shapes are at parity, while warmed `M=8`
direct ingress improves mean, median, and P95. The large prefill wave is also
slightly positive rather than revealing the suspected large-wave regression.

Across the 215 warmed `M=8` candidate calls, duration was 201.88 us mean,
199.76 us median, and 224.77 us P95. The average utilization signature was
4.1% AIC MAC, 33.6% AIC scalar, 26.5% AIC MTE2, and 29.7% AIV scalar,
consistent with the earlier one-wave profile. `DispatchFFNCombine` accounted
for 28.0% of summed device task time in the target window and remained its
largest operator.

The profiled eight-request workload itself was effectively tied: candidate
output throughput was 13.53 tokens/s versus 13.47 for baseline, and mean TPOT
was 451.38 ms versus 455.70 ms. Profiling overhead and the small request count
make these mechanism checks rather than throughput evidence. They do show that
the warmed rank-0 kernel does not explain the earlier end-to-end regression;
the remaining possibilities are cross-rank critical-path behavior and ordinary
fresh-server variance.

## Cross-rank msopprof attribution

An eight-rank exact-`M=8` application-replay `msopprof` collection added source
and pipeline evidence. Source mode placed the highest dynamic basic-block
counts in the routing quantizer, around the temporary smooth-input allocation,
before direct ingress begins. Direct-ingress source lines themselves had only
static execution counts. These counts are instrumented frequencies rather than
time, so they locate executed work but cannot time the hardware epoch wait.

Pipeline mode showed a different and more useful invariant. Across ranks, the
active AIC scalar envelope was about 63--78 us, AIV scalar 60--72 us, MTE1
27--45 us, MTE2 60--107 us, and MTE3 2.7--5.0 us. Instrumented task durations
ranged from 201 us to 20.2 ms even though those active envelopes remained
small. Absolute per-rank task duration is distorted by profiling and the
wait-ID counters accumulate with tool-specific semantics, but the stable
active work inside much larger tasks supports a synchronization/communication
critical path rather than an arithmetic bottleneck.

`TimelineDetail` cannot close this gap in application replay: this CANN build
accepts it only with kernel replay, while replaying one distributed
communication kernel outside its coordinated eight-rank execution is unsafe
and non-representative. This is a tool boundary, not evidence for a particular
epoch implementation.

The most direct follow-up was tested rather than inferred. A disposable
notification-array candidate made each source push its epoch into a
cache-line-separated slot in every destination window, then made each receiver
poll only local HBM. It preserved the existing modular fail-fast gate and
passed an EP8 exact-`M=8` smoke run. In a candidate/v6/candidate bracket its
paired median was 462.76 us versus 417.83 us for v6, a 10.75% regression; P90
regressed 25.23%. The extra remote stores and DCCI publication cost more than
the removed remote polling. Keep the source-owned remote-read protocol and do
not pursue another scalar barrier rewrite without stronger timing evidence.

## Graph-enabled TraceLoom comparison

The repository's TraceLoom submodule was advanced to `d059b0b`, built from
source, and passed all 50 enabled native tests; 11 fixture-dependent tests were
disabled because their external assets are not present. The two warmed eager
profiles above contain no ACLGraph capture or replay evidence, as expected
from `--enforce-eager`. They remain the strict eager control.

A new matched candidate/baseline collection removed `--enforce-eager` and used
`FULL_DECODE_ONLY` with capture size 8. Dynamic `msprof` attached to rank 0
before graph capture, then retained capture, a semantic request, the same-shape
warmup, and a formal eight-request by eight-output burst in one clock domain.
Both runs completed, restored the candidate objects, and left all eight devices
idle. Server logs independently showed eager prefill and `FULL` decode at
eight tokens and eight requests.

TraceLoom isolated the formal tail as seven replay iterations. Every replay
enclosed exactly 43 `DispatchFFNCombine` calls, matching the model's 43 MoE
layers:

| Formal graph metric | Baseline | Candidate | Candidate latency reduction |
|---|---:|---:|---:|
| complete decode-loop average | 51.95 ms | 51.26 ms | 1.32% |
| graph replay median | 38.70 ms | 38.11 ms | 1.53% |
| graph replay mean | 39.44 ms | 38.09 ms | 3.42% |
| `DispatchFFNCombine` mean per layer | 212.69 us | 204.94 us | 3.64% |

The baseline replay mean contains one 44.67 ms outlier. The 1.53% replay
median and 1.32% complete-loop reduction are therefore the robust graph
results. The operator saves about 333 us of summed device task duration per
replay; the complete graph saves about 686 us per iteration. This is direct
evidence that the operator win survives inside the decode graph rather than
being erased by neighboring work.

The same TraceLoom profile separates the two formal graph-external eager
waves. Median operator duration avoids the known first-layer transient:

| Eager wave | Baseline DFC median | Candidate DFC median | DFC reduction | Baseline DFC span | Candidate DFC span | Span reduction |
|---|---:|---:|---:|---:|---:|---:|
| `M=64` | 359.05 us | 346.77 us | 3.42% | 428.71 ms | 439.24 ms | -2.46% |
| `M=449` | 584.91 us | 572.69 us | 2.09% | 676.52 ms | 655.94 ms | 3.04% |

`M=64` again improves inside the operator but does not shorten its complete
DFC-to-DFC wave in this single pair, so no prefill claim is made from it.
`M=449` is directionally consistent at both levels.

The profiled request result agrees with the graph loop rather than the earlier
eager-only saturated bracket: candidate request throughput was 4.951 versus
4.882 requests/s (+1.41%), mean TTFT was 1164.00 versus 1179.63 ms (-1.33%),
and mean TPOT was 62.44 versus 63.37 ms (-1.46%). This is one candidate/baseline
pair under profiling rather than standalone promotion evidence.

TraceLoom found 69 ACLGraph envelopes in each complete collection but promoted
zero exact replay compositions. The primary cause is a TraceLoom coverage gap:
the exact promoter currently admits only head/repeated-layer/tail compositions,
whereas these whole-model captures are an exact periodic sequence with
`pattern_length=1` and `shape_policy=unclassified`. Even body-matching regions
therefore remain legacy envelopes. This semantic-shape gate is tracked in
[TraceLoom issue #25](https://github.com/vLLM-HUST/vllm-hust-perf-analyzer/issues/25).

Separately, candidate had 10 and baseline one `unrecognized_body_mismatch`
regions, so the capability state is `evidence_incomplete`. The formal legacy
envelopes are nevertheless internally aligned, contain exactly 43 DFC children
each, and agree with server logs and request-level timing. Treat them as strong
paired mechanism evidence, not exact capture-body proof.

## Fresh-server graph-enabled brackets

Two subsequent baseline/candidate/baseline brackets used the same
`FULL_DECODE_ONLY` capture and no profiler. Every leg used identical runtime
commits and commands, selected `FUSED_MC2`, exercised `num_tokens=8` full-graph
decode, completed all requests, restored the candidate objects, and left all
eight devices idle. The enclosing baseline object hashes were identical.

The first bracket measured 32 requests per workload. Its saturated burst
favored candidate against the enclosing baseline mean, but the baseline drift
was already larger than the claimed effect:

| Saturated metric | Baseline 1 | Candidate | Baseline 2 | Candidate improvement | Baseline drift |
|---|---:|---:|---:|---:|---:|
| output throughput | 62.19 tok/s | 61.85 tok/s | 55.75 tok/s | +4.88% | -10.35% |
| mean TPOT | 70.97 ms | 70.86 ms | 76.51 ms | +3.90% | +7.80% |
| median TPOT | 72.30 ms | 72.18 ms | 78.48 ms | +4.27% | +8.54% |

The second bracket lengthened each saturated measurement to 128 requests. It
reversed the apparent result while retaining large baseline drift:

| Saturated metric | Baseline 1 | Candidate | Baseline 2 | Candidate improvement | Baseline drift |
|---|---:|---:|---:|---:|---:|
| output throughput | 71.12 tok/s | 54.95 tok/s | 60.61 tok/s | -16.58% | -14.79% |
| mean TPOT | 77.35 ms | 81.27 ms | 75.91 ms | -6.06% | -1.85% |
| median TPOT | 79.60 ms | 78.52 ms | 72.91 ms | -2.97% | -8.40% |

The longer arrival-rate-2 workload exceeded the observed service rate and
accumulated a queue, so it is not interpreted as a low-load latency result.
Its output-throughput comparison was tied within drift (-0.34% candidate with
+9.55% baseline drift).

The generated continuation also is not deterministic enough to make these
request streams token-identical. Inputs and output lengths match exactly, but
the two baseline legs differ on 26/32 saturated continuations in the short
bracket and 110/128 in the long bracket. Candidate divergence from baseline 1
is similar at 24/32 and 106/128, so this is not a candidate-specific correctness
signal; it does mean different generated tokens can induce different expert
routing across otherwise matched runs.

The fresh-server graph evidence therefore establishes no repeatable
end-to-end gain and no stable regression magnitude. It does not overturn the
paired TraceLoom mechanism result, which directly observes a faster candidate
graph loop. Because enclosing baseline drift reaches 10--15%, these brackets
are classified as successful functional smoke and an invalid quantitative
comparison, not as a negative operator-performance gate. Keep the feature
compile-time opt-in until an isolated service-level campaign is needed to
decide the production default.

## Build discipline and remaining gates

Build the candidate by adding both macros to `OPS_COMPILE_OPTIONS`. A clean
macro-disabled build retains the original gather data path. Both configurations
were compiled from the tracked headers into new empty output directories and
produced all three registered Ascend 910B JSON/object pairs.

The generated Ninja rule only checks whether its target objects already exist;
an `opc` environment failure can therefore print a traceback yet leave the
rule apparently successful against stale objects. For experimental rebuilds,
compile into a new empty output directory, require all expected JSON/object
pairs, and compare their hashes before installing them. On this machine,
`ASCEND_OPP_PATH` must point at the task-local readable OPP overlay because the
system vendor configuration is not readable by the workspace user.

The remaining engineering gates are:

1. treat scalar epoch-protocol tuning as closed for the current design: source
   mode cannot time the wait, pipeline mode identifies synchronization and
   communication as the broad residual, and dense-prefix preparation,
   self-epoch-poll removal, and local-poll notification all regress;
2. treat warmed layerwise eager and graph replay as the completed Phase 2
   mechanism gate; retain shared-host fresh-server runs as functional smoke,
   and defer quantitative service promotion to an isolated environment with
   stable enclosing baselines;
3. resolve TraceLoom's generic periodic-composition gap and the remaining body
   mismatches only if exact graph-body attribution or a cross-rank critical-path
   claim becomes necessary; the aligned legacy envelopes are sufficient for the
   present mechanism result;
4. preserve the current compile-time opt-in until isolated service evidence
   supports a production default, then decide whether to upstream the policy
   as-is or expose it through the operator build configuration.
