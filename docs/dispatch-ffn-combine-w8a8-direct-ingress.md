# Dispatch-FFN-Combine W8A8 direct ingress

Status: experimental compile-time prototype. Correctness passes on EP2, EP4,
and EP8; a DeepSeek-V4-Flash W8A8 TP8/EP8 smoke run selected `FUSED_MC2` on
all ranks and completed. Production-shaped operator measurements retain the
direct path only where it shows a repeatable benefit.

The evaluated candidate defines both
`DISPATCH_FFN_COMBINE_W8A8_DIRECT_INGRESS` and
`DISPATCH_FFN_COMBINE_W8A8_DIRECT_INGRESS_SPARSE_FALLBACK`. The first enables
the direct data path; the second enables the EP2/EP4 graph preference exchange
and is rejected at compile time unless the first is also defined. The shared
epoch helper is hardened independently so an impossible generation skew fails
fast rather than waiting forever.

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

The remaining promotion gates are:

1. source-attribute the remaining wait/scalar control only if another kernel
   optimization is pursued; the first dense-prefix candidate is closed;
2. run a controlled end-to-end A/B only if the promotion decision needs a
   costly fresh-server throughput campaign;
3. decide whether to upstream the compile-time policy as-is or first expose it
   through the operator build configuration.
