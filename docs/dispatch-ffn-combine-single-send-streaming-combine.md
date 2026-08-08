# Dispatch-FFN-Combine single-send packets and expert-scoped completion

Status: design landing point for a new experimental branch. No production path
or default is changed.

## Outcome and corrected scope

The target is not merely to make receiver pull cheaper. It is to preserve the
current direct path's deterministic placement while changing the unit of
communication and completion to the smallest true dependency:

- send one hidden row per `(source token, destination rank)`, not per route;
- carry a compact destination-local route descriptor;
- expand the hidden row locally into the destination's expert-major GMM input;
- return each route result directly to a disjoint `(source token, top-k slot)`
  position;
- publish completion per `(receiver, logical expert)`; and
- combine a source token as soon as every logical expert in its top-k set has
  returned.

The earlier EP2 rank-deduplicated prototype at commits `65b1c8469` and
`01f5d2df3` does not reject this design. It rejected a deliberately cheap
receiver implementation that rescanned raw `M * top-k` metadata once per local
expert. Even with that scan, its captured M=236 path was only 1.9% slower than
sealed-wave direct ingress. The next experiment must remove the rescan rather
than polish it.

## Initial boundary

The first implementation is intentionally narrow:

- BF16, EP2, graph-static M, and the existing fused GMM/SwiGLU/GMM path;
- stable logical-expert ownership and unique top-k expert IDs;
- no EPLB, expert replication, migration, or remapping;
- the existing route-count exchange may remain for the first discriminator;
- no remote atomic allocator and no global source-wave ingress epoch;
- no change to the production default until exactness and bracketed graph
  timing both pass.

“Send once” means once per unique destination rank. If a token routes to
experts on two ranks, both ranks need its hidden row unless the platform gains
true multicast or a forwarding hop.

## Data model

For each `(source rank, destination rank)`, routing emits a graph-static packet
lane:

```text
PacketLane[src][dst] {
    hidden[M, K]                         // only touched token slots are valid
    destination_mask[M]
    route_offsets[M + 1]
    routes[R_dst] {
        logical_expert_id
        source_topk_slot
    }
}
```

A token's hidden row appears once in a destination lane even when several of
its experts live on that destination. Route metadata remains route-expanded,
but it is small and destination-local. Probabilities can remain at the source;
the destination needs the top-k slot to address the return value, not the
probability itself.

The first prototype may use fixed token-indexed packet slots. This avoids a
reservation round and makes graph addresses stable. Source routing also emits
an expert-grouped compact reference list:

```text
ExpertRouteRefs[src][dst][logical_expert] -> packet token indices and top-k slots
```

The receiver therefore performs one bulk local gather per nonempty expert. It
does not rescan all token metadata for every expert.

For `R` route rows and `U` unique `(source token, destination rank)` pairs, the
remote hidden payload changes from `R * K` to `U * K`. The measured synthetic
families have `R / U = 4`. The destination still performs `R * K` local writes
to form the expert-major input; communication deduplication does not eliminate
the layout copies required by distinct expert weights.

## Ingress publication without source-wave HOL

Each packet lane owns an independent generation-tagged publication slot:

```text
packet_ready[src][dst] = generation
```

The producer order is:

```text
write packet payload and route refs
  -> device-memory ordering / visibility boundary
  -> publish packet_ready[src][dst]
```

The destination can consume `src -> dst0` without waiting for the same source
to finish `src -> dst1`. This removes the current source-wide sealed-wave HOL.
A generation sequence is still required: disjoint addresses prove write
ownership, but not payload visibility or replay identity. A boolean ready bit
would permit stale graph-replay data and ABA failure.

The slot should be cache-line isolated by writer ownership. Packing slots from
independent writers into one hardware cache line risks false sharing and
visibility ambiguity.

## Direct return indexed by source token and top-k slot

The route descriptor follows the row through GMM1 and GMM2. The GMM2 epilogue
writes directly to:

```text
Return[src][source_token_id][source_topk_slot][K]
```

These positions are disjoint under the unique-top-k invariant. They require no
remote allocation or atomic reduction. This is a layout change from the
current expert-grouped return buffer, where `BlockEpilogue2` writes source
fragments by expert and a full `expandedRowIdx` unpermute follows the final
cross-rank barrier.

Position uniqueness does not prove completion. The source must not read a
return slot until the producing expert's remote writes are globally visible.

## Fletcher's logical-expert completion slots

Ignoring EPLB makes the logical expert the natural completion identity. Each
source receiver owns:

```text
expert_done[receiver_rank][logical_expert_id] = generation
```

The physical device executing an expert does not publish its rank as semantic
metadata. It uses the logical expert ID carried by the route and writes the
corresponding slot in each source receiver's peer window.

The signal is per `(receiver, logical expert)`, not merely per expert. An expert
may return rows to several source ranks; receiver A should not wait for the
expert's writes to receiver B. Publication order is:

```text
finish all GMM2 return writes for (logical expert, receiver)
  -> device-memory ordering / visibility boundary
  -> publish expert_done[receiver][logical_expert] = generation
```

An empty `(expert, receiver)` fragment needs no signal because no source token
waits on it. Slots must be generation-tagged and assigned cache lines so that
independent expert writers do not race through the same cache-maintenance
unit.

### Existing-kernel hinge

Today, GMM2 output tiles are split across AIC cores and AIV subcores, and
`BlockEpilogue2` can split one tile across several source-rank fragments. A
logical expert is done for a receiver only after every intersecting epilogue
tile has completed its remote MTE3 writes. The first implementation must reuse
or extend the local GMM2 progress sequence; it must not let the first tile
publish expert completion.

The cheapest correct first mechanism is a designated completion owner per
logical expert. It observes local completion of all epilogue tiles, then
publishes the per-receiver expert slots. A finer per-fragment counter is a later
optimization only if the owner becomes measurable overhead.

## Progressive token combine

Each source already owns `expert_idx[M, top-k]` and `probs[M, top-k]`. A source
AIV core owns a stable token range and keeps a pending-token worklist. On a
polling round it snapshots the small logical-expert done vector into UB, then
checks each pending token:

```text
token_ready(t) = all(
    expert_done[rank][expert_idx[t, slot]] == generation
    for slot in active top-k slots
)
```

For a ready token, the core reads its disjoint return slots, applies the source
probabilities, reduces top-k contributions, writes the final output row, and
removes the token from the pending list. Token ownership avoids atomic ready
counters. Snapshotting the expert vector avoids repeated remote scalar loads
for every token.

This removes the current whole-rank `CrossRankSyncV2Wait -> full unpermute`
boundary. It does not promise that many tokens become ready early. With the
current synthetic spread routing and experts processed in increasing local
index, the median token waits until local expert wave 58.5 of 63; none are ready
by wave 48 and only about 31% are ready by wave 56. Progressive combine is
therefore an independent overlap hypothesis, not a free consequence of
per-expert signaling. Real route/load order and GMM2 scheduling must decide it.

## Staged discriminators

### Stage A: single-send ingress

Implement destination-local packet lanes and grouped compact route refs, but
retain the existing final return and batch unpermute. This isolates the value
of replacing remote route-expanded hidden rows with one remote hidden row plus
local expansion.

Acceptance:

- exact EP2 output and expert counts across changing routes, active masks, and
  repeated graph replay;
- no receiver per-expert scan of raw `M * top-k` metadata;
- forced-build bracket against Phase-1 and sealed-wave direct ingress;
- primary discriminator: captured M=236, `R / U = 4`.

Stop if local expansion does not reach direct-ingress graph latency within 3%
or if it introduces unstable rank skew. Do not build streaming combine to
rescue an ingress path that is already clearly slower.

### Stage B: logical-expert done return

Change return addressing to `(source token, top-k slot)` and publish
per-receiver logical-expert completion. Initially wait for all required expert
slots before running the existing batch-shaped combine equivalent. This
isolates correctness, visibility, and the removal of the global rank barrier.

Acceptance:

- exact return-slot ownership and generation reuse;
- no source-wide or rank-wide completion required before a completed expert is
  visible;
- adversarial skew where one expert or receiver is delayed;
- no stale return consumption under captured replay.

### Stage C: progressive combine

Add token-owned pending worklists and UB snapshots of the done vector. Compare
full-batch completion, combine overlap, and critical-path output readiness.

Stop if fewer than 25% of production-route tokens become ready before the last
10% of expert work, or if polling/compaction costs more than the removed tail.
A negative Stage C does not invalidate Stages A or B.

## Decision posture

This document is the implementation landing point, not a claim that every
stage will win. Stage A is the next reversible experiment. Fletcher's
per-logical-expert signal is adopted as the Stage B completion model because it
aligns signaling with semantic work rather than physical rank boundaries.
Progressive combine remains deliberately separable because top-k readiness may
concentrate near the end of the expert schedule.
