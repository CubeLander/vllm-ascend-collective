# Dispatch-FFN-Combine sparse-dataflow roadmap

Status: active experimental implementation. Phase 0, the bounded Phase 1
sparse-schedule changes, and the single-node Phase 2 direct-ingress mechanism
are complete. Phase 3 and Phase 4 both proved their narrowed EP2 mechanisms
correct but are closed because their eager gains did not survive warmed graph
replay. Neither will be generalized to EP4 or EP8. Phase 5 is not promoted
without new evidence of exposed fragment-level latency. Phase 6 now has a full
production-opset single-node A2 opt-in package plus process-local and warmed
real-model canaries; topology-blind default enablement remains closed until
multi-node evidence or a proven host topology gate exists. See
`dispatch-ffn-combine-phase0-measurements.md`,
`dispatch-ffn-combine-phase2-direct-ingress.md`,
`dispatch-ffn-combine-phase3-ep2-readiness.md`, and
`dispatch-ffn-combine-phase4-ep2-egress.md`. The production policy and build
receipt requirements are in
`dispatch-ffn-combine-phase6-production-policy.md`.

## Operating rule: prove the micro-logic before the collective

Every new communication or scheduling mechanism starts as the smallest EP2
prototype that can prove or reject one causal hypothesis. It keeps unrelated
layout, metadata, ingress, compute, and egress behavior fixed. The prototype
must expose the dependency directly, include an adversarial delayed-peer case,
and pass repeated-generation correctness before performance is interpreted.

Performance promotion uses warmed layerwise eager and repeated graph-replay
microbenchmarks at production shapes. The distributed critical path is the
slower rank, not one convenient rank or a host-launch average. On a shared
machine, fresh-server end-to-end measurements remain valuable functional and
large-regression smoke tests, but they do not arbitrate a small kernel win when
the enclosing baseline drifts by more than the claimed effect.

The expansion ladder is deliberately one-way:

```text
EP2 causal prototype
  -> EP2 correctness and delayed-peer discrimination
  -> warmed eager and graph-replay win
  -> EP4 correctness and performance
  -> EP8 single-node coverage
  -> multi-node and production policy
```

If the EP2 micro-logic does not produce a repeatable layerwise win, archive the
candidate and record the closed path. Do not pay for generalized collective
state, metadata, topology handling, or service-level campaigns first. A
negative small prototype is a successful early result.

## Objective

Turn `DispatchFFNCombineBF16` from a monolithic, internally
bulk-synchronous MoE kernel into a sparse asynchronous dataflow kernel.  The
target is not merely fewer host launches.  It is to make fixed device-side
protocol cost proportional to the ranks, experts, and token fragments that
actually participate.

The initial candidate steady-state flow was:

```text
route locally
  -> exchange compact metadata
  -> compute final remote destinations
  -> write tokens directly to those destinations
  -> publish fine-grained readiness
  -> consume ready local-expert work
  -> write results directly to precomputed return slots
  -> consume locally ready results
```

The prior-art and local-source study in
`dispatch-ffn-combine-communication-protocol-study.md` changes this from a
prescription into one candidate. Direct expert placement, rank-deduplicated
placement, and receiver-reserved placement have different latency, bandwidth,
and packing costs. The protocol must be selected from measured route
cardinality rather than assumed in advance.

This note deliberately preserves design choices that still need discussion.
It is a roadmap, not approval to modify the production operator.

## Why revisit the operator

Retained production profiles show that the fused operator has a large fixed
floor in decode graph replay:

| Graph input M | Fused operator, one layer | Stock six-task sum, one layer | Ratio |
|---:|---:|---:|---:|
| 1 | 288.1 us | 92.3 us | 3.12x |
| 2 | 290.7 us | 133.9 us | 2.17x |
| 4 | 295.2 us | 174.9 us | 1.69x |
| 8 | 299.4 us | 265.6 us | 1.13x |

The fused duration grows by only about 11 us from M=1 to M=8.  Its roughly
285 us floor therefore dominates small-M execution.  Across 48 layers, the
M=8 difference alone predicts about 1.62 ms of additional latency, consistent
with the observed decode-graph regression.

The same production toggle improves large, non-graph mixed-prefill
iterations.  The optimization target is consequently not "remove fusion".
It is to retain direct, low-fragmentation execution while eliminating the
fixed bulk-synchronous protocol underneath it.

## Current conservative structure

The current kernel establishes correctness through a sequence of global
phase boundaries:

1. apply the active mask and synchronize all participating cores;
2. run generic routing/sort and synchronize again;
3. publish and gather token counts across ranks;
4. compute cumulative offsets;
5. iterate every local expert slot while coordinating AIV and AIC progress;
6. run GMM1, activation, and GMM2 through cross-core flags and additional
   all-core barriers;
7. combine into peer-visible storage;
8. perform a rank-wide completion protocol; and
9. unpermute only after completion is globally visible.

This is robust for a generic dense workload, but it charges sparse decode for
empty experts, unrelated cores, and unrelated ranks.  Fusion removed the
outer launch boundaries without removing the inner bulk-synchronous ones.

`max_output_size=131072` also reserves a very large workspace address range.
The current implementation does not appear to scan or clear that entire
range on every invocation, so it is not the primary explanation for the fixed
latency.  Its memory-footprint and address-locality effects should be measured
separately rather than conflated with synchronization cost.

## Candidate target protocol

This section describes the receiver-reserved direct-expert candidate. It is
not yet the default design. A fixed per-source expert slab can avoid the
metadata round at the price of receiver packing or segmented GMM, while a
rank-deduplicated layout can send a hidden vector once per destination rank at
the price of receiver fan-out. Phase 0 must distinguish these choices.

### 1. First exchange: compact routing metadata

Each source rank first computes its local routes.  It groups only nonempty
traffic by `(destination_rank, destination_expert)` and exchanges compact
metadata such as:

```text
destination rank
destination local-expert id
token count
source rank
source fragment id
return-slot base or return descriptor
```

The metadata exchange supplies enough information for each destination rank
to compute final receive offsets.  No token payload needs to move in this
round, and no dense `EP x EP x experts_per_rank` count matrix should be
required when most entries are zero.

Open choice: use an all-rank compact exchange, pairwise exchange, or an EP2
specialization.  The invariant is that the receiver owns final allocation of
its local expert buffers and communicates the resulting destinations back to
the relevant producers.

### 2. Precompute direct destinations

After the metadata exchange, every routed token has a concrete destination:

```text
remote window + expert buffer base + row offset
```

Every result also has a concrete return destination. Prefer a source-owned
return slot whose identity is established while routing, so the destination
rank can write the result directly. The current fused operator already does
this for its `offsetD` fragments; the remaining egress questions are
completion granularity and whether the final local unpermute can be fused or
avoided.

The desired descriptor is logically similar to:

```text
RouteDescriptor {
    remote_token_address;
    remote_ready_address;
    local_source_row;
    return_address;
    return_ready_address;
    expert_id;
    fragment_length;
    generation;
}
```

Actual encoding should be compact and aligned for device-side vectorized
loading.  Addresses may be represented as window-relative offsets rather than
raw pointers.

### 3. Direct payload writes

The producer writes token rows directly into their final remote expert
buffers.  This removes the present pattern of publishing a complete count
matrix, deriving global cumulative state, staging tokens, and then copying
again into expert-contiguous storage.

Publication must obey a strict ordering invariant:

```text
payload and metadata visible before readiness is visible
```

The implementation must use the narrowest device/HCCL primitive that provides
this release ordering.  A readiness signal without a proven memory-ordering
contract is not sufficient.

### 4. Consume only active local experts

The metadata round naturally produces a compact active-expert worklist:

```text
[(local_expert_id, row_begin, row_count, readiness), ...]
```

GMM1, activation, GMM2, and return dispatch should iterate this list rather
than all `experts_per_rank` slots.  Empty experts must not consume scheduler
iterations or advance cross-core compute flags. A generation-bearing
readiness lane is different: if it remains part of the next consumer's wait
set, its epoch must advance once per invocation even when that invocation has
no payload. Otherwise a dormant generation turns a later valid publication
into an apparent impossible skew.

Readiness may be defined per expert or per expert fragment.  Per-fragment
readiness enables earlier compute but increases control traffic; per-expert
readiness is simpler and is probably the first safe implementation target.

### 5. Lightweight completion

Rank-wide completion is broader than the true dependency.  A consumer needs
to know only that the payload it will read is complete.

Candidate protocols, from simpler to more aggressive:

1. **Per-expert expected/arrived counts.** The metadata exchange fixes the
   number of source fragments for each active expert.  The final producer
   makes that expert ready.
2. **Per-fragment sequence numbers.** Each fragment carries a generation and
   publishes a sequence after its payload.  Compute and return consumption can
   pipeline at fragment granularity.
3. **Source-owned return slots and ready bitmap.** The destination writes each
   result directly to its source-owned final slot and sets only the
   corresponding readiness bit.  The source waits for its own routed tokens,
   not for a rank-wide completion condition.

The first implementation should prefer per-expert ingress readiness and
source-owned return slots.  Fragment pipelining should be added only if the
measured expert-level critical path leaves meaningful headroom.

## Synchronization budget

Every synchronization in the redesigned kernel must name its producer,
consumer, protected state, and scope.  The expected scopes are:

| Dependency | Maximum necessary scope |
|---|---|
| route descriptor construction | routing workers for the local source batch |
| destination allocation | metadata owner for the affected destination expert |
| token payload publication | producer and consumer of that expert/fragment |
| GMM1 to activation | cores processing the same ready expert tile |
| activation to GMM2 | cores processing the same ready expert tile |
| result publication | producer and source-owned return-slot consumer |
| layer completion | local consumers whose outputs feed the next layer |

An all-core or all-rank barrier is acceptable only when a narrower dependency
cannot preserve one of the correctness invariants below and the reason is
recorded.

## Correctness invariants

1. Every active `(token, top-k slot)` is delivered exactly once.
2. Every inactive graph-padding row is delivered zero times.
3. A destination expert consumes a row only after its payload is visible.
4. A source consumes a result only after the complete result row is visible.
5. Return placement preserves the original token/top-k identity and weighted
   reduction semantics.
6. Buffer reuse is protected against ABA across graph replays; readiness
   carries a generation or uses an equivalently proven alternating-state
   protocol.
7. No producer may overrun the fixed graph-safe workspace capacity.
8. Overflow, malformed metadata, or generation mismatch must produce an
   identifiable unsupported result rather than silent corruption or a hang.
9. Progress must not depend on an inactive core or rank publishing a signal.
10. EP2 specialization and generic EP execution must have the same externally
    observable numerical contract.
11. Every reusable generation-bearing readiness lane advances exactly once per
    invocation, including invocations whose payload on that lane is empty.

## Development phases

### Phase 0: measurement and protocol accounting

Status: complete for the synthetic EP2 matrix. The experiment identified the
per-local-expert AIV barrier loop as a material fixed cost. Production route
distributions and byte counters remain future evidence, not blockers for the
bounded Phase 1 change.

- Add experimental, removable phase timestamps/counters around routing,
  metadata/count exchange, ingress movement, expert compute, result movement,
  completion, and unpermute.
- Measure M in `{1, 2, 4, 8, 16, 32}` plus representative mixed-prefill
  shapes.
- Count active experts, nonzero rank-expert pairs, bytes moved, readiness
  publications, waits, and wait iterations.
- Count active `(token, expert)` routes `R` and unique
  `(token, destination rank)` pairs `U`; report `R / U` to expose the payload
  duplication versus receiver-packing tradeoff.
- Establish the stock, current fused, and graph-replay baselines before
  changing behavior.

Gate: explain most of the roughly 285 us small-M fixed floor with measured
phase evidence.  Do not redesign blindly if instrumentation contradicts the
source-derived hypothesis.

### Phase 1: compact active-expert execution

Status: complete at the evidence-supported boundary. Empty source fragments
no longer enter the copy helper, empty local experts no longer execute a
vacuous all-AIV barrier, and GMM1, GMM2, and combine skip zero-shape
tensor/scheduler setup. AIV-to-AIC progress flags are compacted to scheduled
nonempty experts.

The loops deliberately still scan expert indices. A bracketed 100-sample
discrimination reduced `experts_per_rank` from 64 to 8, which removes 56 scan
steps and also shrinks count metadata, but produced no sparse-path latency
improvement. A shared worklist would add construction, global reads, and
activation-split synchronization risk without a measured benefit. Revisit it
only if production evidence exposes a materially larger expert-cardinality
floor.

- Retain the existing metadata/count exchange and memory ownership.
- Materialize a compact active-expert worklist only if a future cardinality
  experiment demonstrates a scan-bound regime.
- Make GMM1, activation, GMM2, and combine skip empty experts and avoid flag
  advancement for them.
- Preserve the current rank-wide completion protocol initially.

This isolates the value of removing empty-expert control work without also
changing communication semantics.

Gate: matched correctness; no deadlock across repeated graph replay; stable
small-M reduction attributable to active-expert count.

### Phase 2: discriminate and prototype ingress placement

Status: complete at the single-node mechanism boundary. Direct final placement
is implemented behind
`DISPATCH_FFN_COMBINE_DIRECT_INGRESS`. It reuses the existing complete count
matrix so every source can compute disjoint final destination prefixes without
a returned reservation round. A dedicated source-owned epoch publishes one
sealed request wave; the old per-expert receiver-pull copies and ingress
barriers are bypassed. Tracked EP2 and EP4 correctness, 2,048-generation EP8
reuse with 512 global experts, unsigned wrap, explicit payload ordering, and
fail-fast generation divergence are validated on one node.

The first bracketed 50-sample result improves six route families by 10.6--23.3%
on the slower-rank device median. Graph M=64 with one active token is
inconclusive to negative because the new epoch cannot be amortized. A naive
runtime route-matrix scan was measured and rejected; it added about 50 us of
rank skew. The zero-scan mask selector protects that route family without a
new communication round. Multi-node coverage and production-default policy
remain rollout gates rather than reasons to keep Phase 2 mechanism work open.

- Compare the existing source-pull layout, a fixed
  `[expert][source][slot]` direct-write layout, and rank-deduplicated ingress.
- Add receiver-reserved compact metadata only if fixed slabs or receiver
  fan-out leave a measured packing/fragmentation bottleneck worth an extra
  metadata round trip.
- Preserve source/top-k identity and return-slot descriptors across all
  candidates so egress can be evaluated independently.

Gate: byte/count accounting proves one delivery per active route; destination
calculation remains graph-safe and allocation-free in replay; the selected
protocol wins in its intended decode or prefill regime for an explained
reason.

### Phase 3: per-expert ingress readiness

Status: complete and closed at EP2. A disposable candidate released expert 0
before the remaining ingress wave, passed changing-route and delayed-peer
correctness, and improved the deliberately staggered eager case by roughly
3--5%. Warmed graph replay was instead at parity: the cleanest Phase 2 and
candidate critical medians were 615.38 us and 617.49 us. A second local-relay
design measured 615.32 us against that 615.38 us baseline. The improvement is
therefore not repeatable in graph execution, so the stop rule rejects generic
EP4/EP8 state and keeps the Phase 2 sealed wave.

The experiment also established two protocol constraints. First,
`CrossCoreSetFlag<0x2>` is scoped to one physical AI Core, and both AIV
subcores must signal their paired AIC; one global AIV coordinator deadlocks.
Second, source-owned expert readiness must advance on an empty generation as
well as a nonempty one. The full implementation and measurement record is in
`dispatch-ffn-combine-phase3-ep2-readiness.md`.

1. Keep the Phase 2 count exchange, final input layout, GMM schedule, return
   path, and layer-final join unchanged.
2. On EP2, choose at least two destination experts and deliberately delay one
   source fragment. Replace only the sealed pre-compute ingress boundary with
   per-expert expected/arrived state.
3. Demonstrate that an independent expert starts before the delayed expert
   without an early read, hang, or cross-replay ABA.
4. Compare against the Phase 2 sealed-wave binary with warmed layerwise eager
   and repeated graph-replay measurements on the slower rank.
5. Expand to EP4 only after the EP2 critical path improves repeatably; expand
   to EP8 only after EP4 preserves the mechanism and benefit.

Stop rule: if the EP2 micro-logic is at parity or slower after correctness and
warmup are controlled, archive per-expert readiness and do not build its
generic collective form. Do not combine this discriminator with direct-return
completion, unpermute removal, compact-metadata redesign, or fragment-level
pipelining.

### Phase 4: direct return slots and lightweight egress completion

Status: complete and closed at EP2. A disposable local-only candidate removed
the entire cross-rank completion handshake while retaining explicit return
write completion, the local all-AIV join, and the existing unpermute. It passed
nonzero-output, changing-generation, empty-rank, and graph-padding correctness.

The 64-expert production-shaped case was at parity to slightly slower. In the
more favorable four-expert sparse case, the clean reverse comparison improved
eager by about 2.2% but regressed graph replay by about 0.8%. Because this was
the maximum possible EP2 completion saving, an active-rank mask cannot improve
the result enough to cross the joint gate. No egress epoch or EP4/EP8
completion state will be built. The full record is in
`dispatch-ffn-combine-phase4-ep2-egress.md`.

- Retain and document the current source-owned `offsetD` return placement;
  verify its exact `(token, top-k slot)` identity contract.
- Publish per-token, per-fragment, or per-source readiness without a
  rank-wide completion handshake.
- Fuse or eliminate local unpermute where the existing direct placement and
  routing metadata already establish enough of the final layout.

Gate: output identity and weighted combine remain exact within the accepted
  numerical tolerance; a source waits only for work it actually emitted.

### Phase 5: optional fragment-level pipeline

Status: not promoted. Phase 3 found no graph-replay benefit from releasing a
whole early expert, and Phase 4 found no graph-replay benefit even when its
entire rank-wide egress handshake was removed. Fragment-level descriptors and
readiness would add more control traffic than either rejected ceiling. Reopen
only if a future profile exposes a large expert whose payload transfer, GMM,
or return sits uncovered on the graph critical path.

- Split large active experts into independently ready fragments.
- Pipeline ingress, GMM, and return only when Phase 4 profiles show sufficient
  exposed latency.
- Bound descriptor count and synchronization traffic so the optimization does
  not recreate the decode fixed-cost problem.

This phase is optional.  Expert-level readiness may already be the best
complexity/performance point.

### Phase 6: production policy and fallback

Status: policy, full production-opset BF16 plus W8A8 package receipt,
isolated-prefix installation, process-local EP2/EP4/EP8 canary, and warmed W8A8
real-model smoke complete. Publication of the matched wheel/full-package bundle
and a bounded service-pool canary remain. Treat Phase 2 direct ingress as the
last mechanism that crossed its warmed eager and graph gates. BF16 and W8A8
are accepted as profitable mechanisms inside their measured selector
envelopes. Later closed experiments remain evidence, not production branches.

- Preserve the measured BF16 and W8A8 selector tables rather than inventing a
  dtype-independent threshold.
- Use the existing compile-options surface to produce a named single-node A2
  candidate package with a clean-build, full-runtime inventory, and object-hash
  receipt. A narrow operator package is a test receipt, never a deployable
  overlay, because the installer replaces complete vendor subtrees.
- Pair that package with a recorded, source-matched `vllm_ascend_C`; never
  permit a missing local extension to fall through to an unrelated machine
  copy.
- Treat shared-host end-to-end runs as semantic, warmup, and large-regression
  smoke; use warmed layerwise eager and graph replay for mechanism acceptance.
- Keep unknown and multi-node topologies on the macro-off package until they
  gain direct evidence or a uniform host-side topology gate.
- Treat unsupported topology, capacity, or protocol state as an explicit
  unsupported path and fall back safely; never silently reinterpret it as
  noise.

## Experiment matrix

At minimum, evaluate:

- M: `1, 2, 4, 8, 16, 32`, representative mixed-prefill sizes, and near
  capacity;
- active experts: one, few, balanced many, and worst-case skew;
- EP: 2 first, followed by the smallest generic-EP configuration worth
  supporting;
- graph mode: repeated same-template replay, alternating templates, and active
  mask padding;
- route patterns: local-only, remote-only, bidirectional, hot expert, and
  empty destination rank;
- arrival skew: intentionally delayed source fragments;
- repeated generations sufficient to expose stale readiness and counter wrap;
- current fused wave scheduling versus cut-through scheduling as a separate
  ablation, so fusion and pipeline effects are not conflated.

Report device envelope, per-phase duration, active work cardinality, protocol
event count, and macro serving metrics.  A throughput result alone is not
sufficient to validate the mechanism.

## Success criteria

The roadmap succeeds when:

1. small-M latency scales with active routes/experts rather than total local
   experts;
2. decode graph replay is no slower than the stock path across repeated
   matched captures;
3. the mixed-prefill benefit of fusion is preserved or improved;
4. no rank-wide completion wait remains unless supported by a demonstrated
   layer-level dependency;
5. graph replay uses bounded preallocated state with a proven generation
   protocol;
6. TraceLoom exposes the expected structural replacement and the measured
   critical-path movement without relying on semantic labels.

## Decisions intentionally left open

- whether compact metadata is needed at all, and its exact layout if so;
- all-rank versus pairwise metadata transport;
- EP2 specialization boundary;
- per-expert versus per-fragment ingress readiness;
- readiness primitive and its formal memory-ordering contract;
- whether direct return placement can remove unpermute completely;
- whether the generic path should coexist with, or be replaced by, a
  decode-specialized kernel;
- graph capture policy for variable mixed-prefill shapes.

These decisions should be resolved by the smallest discriminating experiment,
not by expanding the implementation in advance.
