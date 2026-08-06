# Dispatch-FFN-Combine Phase 3 EP2 readiness discriminator

Status: complete and closed at EP2. The disposable prototype passed
correctness and showed an eager staggered-route gain, but graph replay remained
at parity. Per the preregistered stop rule, no EP4/EP8 generalization or
production implementation follows.

## Question

Phase 2 seals every source's complete ingress wave before GMM1 can consume the
first local expert. Phase 3 asks one narrow question:

> On EP2, can GMM1 of an early expert overlap the remaining ingress of a later
> expert enough to repay per-expert publication and observation?

The experiment is successful only if it proves both the causal overlap and a
repeatable warmed layerwise latency reduction. Correctness without a latency
win closes this path early.

## Fixed surroundings

The discriminator keeps these Phase 2 choices unchanged:

- the dense tagged count exchange and its prefix calculation;
- source-owned routing staging and expert-major final input layout;
- source-rank ordering inside each destination expert;
- the existing GMM1, SwiGLU, GMM2, direct-return, and unpermute layouts;
- the existing layer-final egress join; and
- the current fallback selector and default compile-time behavior.

It does not introduce compact metadata, receiver allocation, direct-return
completion changes, unpermute removal, fragment tiling, multi-node handling,
or an EP4/EP8 protocol.

## Implemented EP2 protocol

The experiment was enabled under a disposable compile guard and only when
`EP == 2`. Other configurations retained the Phase 2 sealed-wave path. The
implementation narrowed the preregistered per-expert design further: only
expert 0 was released early, and the remaining experts reused the proven Phase
2 tail seal.

```text
count matrix and final offsets become available
  -> every source reads, but does not publish, the next sealed-wave epoch
  -> ordinary source workers push destination expert 0 first
  -> every source-owned expert-0 lane publishes that epoch, even if empty
  -> every receiver AIV observes the required source lanes for expert 0
  -> both AIV subcores of each physical AI Core release their paired AIC
  -> ordinary source workers continue the unchanged mapping for the tail
  -> the existing Phase 2 epoch seals the complete tail wave
  -> remaining expert flags and the existing compute/egress path continue
```

The initial AIV-to-AIC cumsum flag moved before ingress. GMM1 could then wait
on the existing compact per-scheduled-expert progress sequence while the AIV
source workers continued pushing later experts. The GMM order remained fixed;
the experiment overlapped an earlier expert with a later ingress tail rather
than adding out-of-order expert scheduling.

## State ownership and layout

Readiness remained source-owned, matching the proven Phase 2 direction. Each
source window reserved the existing sealed-wave generation line plus one
cache line per `(destination rank, local expert)` candidate lane:

```text
one cache line: existing sealed-wave generation register
EP * experts_per_rank cache lines: ready[dst_rank][local_expert]
```

A worker that owns `(dst_rank, local_expert)` performs the payload copy, waits
for that copy's MTE3 completion, executes a DDR data-sync barrier, and then
publishes the generation to its local source-owned ready slot. The destination
polls that slot through the source's peer window. Cache-line separation avoids
unrelated DCCI operations sharing a control line.

The destination already has the complete count matrix and polls only sources
whose expert-0 fragment is nonempty. Nevertheless, every source-owned expert-0
lane advances every generation, including when its payload is empty. This is
not redundant bookkeeping: when routes change, a lane may be dormant in one
generation and required in the next. Leaving the dormant epoch behind makes
the next consumer observe a legitimate old state that is indistinguishable
from lost protocol progress under the fail-fast modular rule.

The unsigned modular wait rule itself remained unchanged: the requested
generation and its allowed one-generation look-ahead are accepted, one behind
may still be publication in flight, and an impossible larger skew traps.

The EP2 prototype fails closed to Phase 2 if the additional control region
does not fit before the reserved peer-window control boundary.

## AIV signaling constraint

The preregistered single receiver coordinator is invalid on this architecture.
The CANN [`CrossCoreSetFlag` contract][cross-core-set] defines mode 2 within
one physical AI Core: both AIV subcores must execute the set before their
paired AIC's wait proceeds. A single global coordinator therefore releases at
most its own paired AIC and deadlocks the others.

The first correct design had every AIV poll the two possible source-owned
expert-0 lanes, order the observations with a DDR barrier, and publish its own
mode-2 flag. A second design made one AIV poll remote lanes and publish a local
relay cache line, while every AIV polled that relay and still published its own
paired flag. The relay reduced duplicate remote polling but added local
control traffic; graph timing showed no benefit.

## Causal route and instrumentation

Use exactly two active destination experts: group 0 and a later group assigned
to the same source worker after group 0 under the existing Phase 2 mapping.
Give both experts real rows. The source worker completes and publishes group 0,
then performs the later expert's larger copy. The sealed-wave control cannot
release group 0 until both copies complete; the Phase 3 candidate can.

The causal condition represented by this route is:

```text
GMM1 start(group 0) < receiver ready(later group)
```

Release-object device events, not diagnostic host time, decided performance.

## Correctness result

One EP2 communicator and peer window covered changing nonuniform routes,
graph-padding masks, empty lanes, and repeated peer-window reuse. The direct
all-AIV design passed the tracked structural oracle:

```text
1 passed, 19 warnings in 42.83s
```

A separate 28-generation case alternated 20 ms launch skew between peers and
also passed exact output and expert-count checks:

```text
1 passed, 19 warnings in 43.58s
```

The relay design passed the same tracked oracle:

```text
1 passed, 21 warnings in 37.39s
```

Together the cases covered:

- both sources contributing to both tested experts;
- one source absent from the early expert;
- local-only and remote-only early fragments;
- an empty destination expert between the two scheduled experts;
- alternating route templates across repeated generations;
- graph-padding masks; and
- the existing impossible-generation-skew fail-fast contract.

Outputs and expert counts must match the existing structural oracle. The
ordinary Phase 2 object must be restored and checksum-verified after every
disposable run.

## Performance result

The primary EP2 shape used `H=7168`, `FFN=2048`, `M=64`, top-k 4, and 64 local
experts. Every process and communicator was warmed before 50--100 device-event
samples. The distributed critical path is the slower rank. Phase 2 and the
first correct Phase 3 candidate were bracketed in B-C-C-B order.

| Case | Phase 2 before | Candidate 1 | Candidate 2 | Phase 2 after |
|---|---:|---:|---:|---:|
| staggered eager | 753.52 us | 739.32 us | 726.91 us | 775.30 us |
| balanced eager | 700.58 us | 692.85 us | 658.80 us | 854.87 us |
| production graph | 3751.44 us | 3767.81 us | 3743.67 us | 3771.74 us |

The staggered eager case improved by roughly 3--5% under both candidate runs,
and the balanced control did not expose a candidate regression. The second
baseline run suffered broad shared-machine interference, so it is supporting
rather than decisive evidence. Production graph replay remained at parity.

The deliberately staggered route was then measured under graph replay. The
cleanest baseline/candidate comparison was 615.38 us versus 617.49 us on the
slower-rank raw-sample median. A second candidate observation reached 606.76
us, but its reverse baseline was heavily perturbed and could not substantiate
the direction. Finally, the local-relay design measured 615.32 us in a clean
100-sample repeat: effectively identical to the 615.38 us baseline.

The eager mechanism win is real enough to preserve as evidence, but it does
not cross the graph-replay promotion gate. Phase 3 is therefore archived at
EP2, the production source and installed objects remain on Phase 2, and no
EP4/EP8 readiness state will be built.

## Borrowed lesson and local choice

DeepEP and NCCL EP both demonstrate expert- or lane-shaped readiness, but they
also retain packing, barriers, or transport-specific release semantics. This
experiment borrows only the narrower dependency model. It deliberately keeps
the Ascend Phase 2 source-owned remote-read direction and explicit DDR/DCCI
ordering rather than importing a remote-notification protocol whose local
W8A8 ablation already regressed.

[cross-core-set]: https://www.hiascend.com/document/detail/zh/canncommercial/80RC3/apiref/ascendcopapi/atlasascendc_api_07_0260.html
