# Dispatch-FFN-Combine Phase 0 measurements

Status: completed baseline and first bounded optimization experiment.

## Question

Does sparse decode pay a fixed ingress/control cost for every local expert,
including experts that received no routes?

The source-derived hypothesis was yes: the AIV ingress loop iterated all 64
local experts, called an all-core barrier for each one, and advanced an AIC
progress flag even when the expert's received-row count was zero.

## Method

The experiment used two 910B2 devices in the live-source Ascend development
image, CANN 9.0.0, EP=2, 64 local experts per rank, top-k=8,
hidden/FFN size 256, and `max_output_size=4096`.

A temporary compile-time instrumentation block measured, with device system
cycle timestamps:

- active-mask processing;
- route construction;
- cross-rank count exchange;
- prefix/count publication;
- ingress peer copies and per-expert synchronization;
- activation-side waiting/work;
- return writes; and
- final completion plus unpermute.

It also counted active routes `R`, unique token-destination-rank pairs `U`,
active local experts, and nonempty source-expert fragments. A host HCCL barrier
was placed immediately before every measured invocation to remove reference
calculation and Python process skew from the count-exchange phase. Every case
checked numerical output and local expert counts.

The experimental instrumentation was removed after measurement. It is not
part of the production patch.

## Baseline result

The ingress phase was effectively constant at about 150--160 us over radically
different route sparsity:

| Case | Physical M | Active routes | Active local experts, rank 0 | Nonempty fragments, rank 0 | Ingress, rank 0 |
|---|---:|---:|---:|---:|---:|
| one-token spread | 1 | 8 | 11 | 11 | 150--151 us |
| eight-token spread | 8 | 64 | 57 | 69 | 151--152 us |
| eight-token, four experts total | 8 | 64 | 2 | 4 | 157 us |
| 64-token, four experts total | 64 | 512 | 2 | 4 | 156--160 us |
| graph M=64, one active token | 64 | 8 | 11 | 11 | 147--149 us |
| graph M=64, 16 active tokens | 64 | 128 | 64 | 128 | 151 us |

Payload volume and active-expert count therefore did not explain ingress
latency. The fixed 64-step barrier/progress protocol did.

`R/U` was 4 for all top-k=8 spread cases in this synthetic matrix. This is
evidence that rank-deduplicated ingress could reduce hidden-vector traffic in
this route family, but it does not establish that receiver fan-out is cheaper.
The transport choice remains open.

## First bounded change

The first patch deliberately preserves the existing count exchange, memory
ownership, expert ordering, AIV-to-AIC progress count, and final rank-wide
completion protocol. It changes only two vacuous operations:

1. do not enter `CopyGMToGM` for a zero-row source fragment; and
2. when the total received-row count for a local expert is zero, publish its
   AIV-to-AIC progress signal without an all-core barrier, because there are no
   ingress writes whose visibility that barrier could protect.

The empty-expert branch is uniform across all AIV cores because every core
reads the same cumulative local-expert count. The next nonempty expert still
synchronizes all producers before publishing readiness.

## A/B result

With the same temporary instrumentation, the patch made ingress scale with
the active local-expert set:

| Case | Active local experts, rank 0 | Baseline ingress | Patched ingress | Baseline total | Patched total |
|---|---:|---:|---:|---:|---:|
| one-token spread | 11 | 150--151 us | 87--91 us | 244--261 us | 182--194 us |
| eight-token, four experts total | 2 | 157 us | 26--28 us | 290--294 us | 163--167 us |
| 64-token, four experts total | 2 | 156--160 us | 19--24 us | 274--304 us | 178--180 us |
| graph M=64, one active token | 11 | 147--149 us | 90--91 us | 272--287 us | 203--213 us |
| graph M=64, four active tokens | 34 | 149--150 us | 117 us | 271--284 us | 241--242 us |
| graph M=64, 16 active tokens | 64 | 151 us | 149 us | 281--283 us | 274--277 us |

For two-active-expert cases, some of the saved ingress time moved into the
activation phase as explicit waiting for GMM1: ingress no longer hides AIC
work behind unnecessary AIV barriers. Total device time still fell by roughly
38--44% in those instrumented calls.

Release binaries were also compared without device `printf`. Two A/B orders
were run to expose environmental drift. Using the slower rank's median host
latency as the distributed critical-path estimate:

| Case | First order | Reverse order | Judgment |
|---|---:|---:|---|
| eight-token, four experts total | -18.8% | -14.6% | stable sparse-path win |
| graph M=64, one active token | -11.2% | -13.2% | stable graph-decode win |
| dense 64-token spread | -0.5% | -5.2% | no stable regression |
| dense 256-token spread | +12.0% | -4.2% | noisy/inconclusive |

Absolute timings drifted substantially between process launches, so the
dense-path numbers are not a performance claim. The sparse-path direction
survived both A/B orders and agrees with the internal phase attribution.

## Correctness and gate result

The existing two-rank end-to-end test passed with an ordinary release binary.
It covers repeated reuse of the same HCCL window, changing nonuniform count
generations, a rank with zero received experts, and graph padding with one
active token. The larger Phase 0 matrix additionally covered M in
`{1, 2, 4, 8, 16, 32, 64, 256}`, dense spread, four-expert concentration, and
graph-padded active sizes `{1, 4, 16}`.

Phase 0's source hypothesis is accepted: empty-expert AIV coordination
explains a material part of the small-M fixed floor. The bounded barrier skip
is worth retaining.

A second bounded Phase 1 change retained the same progress protocol but made
GMM1, GMM2, and combine bypass tensor-address construction and zero-shape
scheduler setup for empty experts. It explicitly preserves activation split
events and packed-weight offsets. Across two A/B orders against the
barrier-skip-only binary, the slower-rank median improved by 14.5--19.7% for
the four-expert decode case and 11.5--14.5% for graph M=64 with one active
token. Dense cases are controls because they take no new empty-expert branch;
their variation is environmental noise rather than attributable benefit.

Phase 1 remains partial: the loops still traverse expert indices, progress
flags still advance for empty experts, and egress retains rank-wide
completion.

## Next decision

The next experiment should compact the shared AIC/AIV expert schedule while
retaining the current transport and return layout. It must demonstrate that
removing empty-expert flag advancement and the remaining index traversal saves
more than worklist construction costs. Transport redesign remains gated on
production distributions of `R/U` and nonempty source-expert fragments.
