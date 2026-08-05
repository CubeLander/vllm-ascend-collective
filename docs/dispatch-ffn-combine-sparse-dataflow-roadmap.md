# Dispatch-FFN-Combine sparse-dataflow roadmap

Status: design note; no implementation is authorized yet.

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
iterations or advance cross-core flags.

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

## Development phases

### Phase 0: measurement and protocol accounting

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

- Retain the existing metadata/count exchange and memory ownership.
- Materialize a compact active-expert worklist.
- Make GMM1, activation, GMM2, and combine skip empty experts and avoid flag
  advancement for them.
- Preserve the current rank-wide completion protocol initially.

This isolates the value of removing empty-expert control work without also
changing communication semantics.

Gate: matched correctness; no deadlock across repeated graph replay; stable
small-M reduction attributable to active-expert count.

### Phase 2: discriminate and prototype ingress placement

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

- Replace the pre-compute rank-wide/phase-wide ingress boundary with
  per-expert expected/arrived state.
- Start an expert as soon as all of its declared fragments are visible.
- Retain a simple local layer-final join only where the following layer truly
  requires the entire local output.

Gate: adversarial source skew and delayed-fragment tests cannot trigger early
reads, hangs, or cross-replay ABA.

### Phase 4: direct return slots and lightweight egress completion

- Retain and document the current source-owned `offsetD` return placement;
  verify its exact `(token, top-k slot)` identity contract.
- Publish per-token, per-fragment, or per-source readiness without a
  rank-wide completion handshake.
- Fuse or eliminate local unpermute where the existing direct placement and
  routing metadata already establish enough of the final layout.

Gate: output identity and weighted combine remain exact within the accepted
  numerical tolerance; a source waits only for work it actually emitted.

### Phase 5: optional fragment-level pipeline

- Split large active experts into independently ready fragments.
- Pipeline ingress, GMM, and return only when Phase 4 profiles show sufficient
  exposed latency.
- Bound descriptor count and synchronization traffic so the optimization does
  not recreate the decode fixed-cost problem.

This phase is optional.  Expert-level readiness may already be the best
complexity/performance point.

### Phase 6: production policy and fallback

- Determine the measured crossover among stock MC2, current fused, and sparse
  fused execution.
- Keep a size-/shape-aware fallback until the sparse path wins reliably at
  small M.
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
