# Dispatch-FFN-Combine communication protocol study

Status: prior-art and local-source study, followed by an experimental
compile-time direct-ingress prototype. See
`dispatch-ffn-combine-phase2-direct-ingress.md` for implementation evidence.

Snapshot: 2026-08-05. The external source trees inspected for this study were
DeepEP commit `01dc3aaac82068020353dce2c302e38153c0bfaa` and NCCL commit
`5067397c2676d5aed50042fc39e5c8ee96eb0027`. CANN observations refer to the
locally installed CANN 9.0.0 sources.

## Problem frame

The operator implements a sparse distributed scatter -> expert compute ->
gather. Its difficult state is not the token payload itself, but the compact
description of:

- which `(token, top-k slot)` instances exist;
- which destination expert owns each instance;
- where each instance and its result may be written;
- when those bytes become visible to the consumer; and
- when graph-reused state may be overwritten without an ABA error.

The decode objective is to make fixed control work proportional to active
routes, ranks, and experts. The prefill objective also cares about payload
bandwidth and long contiguous expert batches. Those objectives can prefer
different wire layouts, so this study does not assume that one protocol must
serve both regimes.

## What the current Ascend operator actually does

The current fused operator is a **source-owned staging plus receiver-pull**
protocol:

1. `moe_init_routing_v2` expands active `(token, top-k slot)` instances and
   places expert-grouped rows in the source rank's peer-visible `offsetA`
   region.
2. `TaggedTokenPerExpertGatherAndGetSumPreRank` publishes a dense row of
   counts to every other rank. It encodes readiness by adding
   `INGRESS_READY_MAGIC` to every count cache line; receivers poll all source
   rows, strip the tag, and derive prefix sums.
3. For every local expert, cores pull that expert's rows from every source
   rank into a local contiguous input. The loop and its all-core boundary run
   even when many `(source, expert)` fragments are empty.
4. GMM1, activation, and GMM2 iterate the fixed local-expert schedule.
5. The GMM2 epilogue already writes each source fragment directly into that
   source rank's `offsetD` return region. Return placement is therefore
   substantially more direct than the initial roadmap assumed.
6. A toggling rank-wide completion handshake waits for all ranks before local
   unpermute and reduction consume `offsetD`.

This refines the optimization target. Direct return placement does not need
to be invented; its broad completion condition and the final local unpermute
are the remaining questions. On ingress, the expensive choices are the dense
count publication, polling, receiver pull/packing, empty-expert loops, and
associated all-core boundaries.

## CANN 9.0.0 built-in MoE protocol

The built-in `moe_distribute_dispatch_v2` A2 path uses a different but still
bulk-synchronous protocol:

- it counts and sorts routes by destination expert;
- it builds one expert-grouped buffer per destination rank;
- inter-server transport uses `BatchWrite` for a rank-sized region containing
  counts, payload, and a trailing data-ready flag;
- the single-server path directly copies the same region into the peer window
  and writes the trailing flag after the payload;
- the receiver polls a status-ready flag and then the payload-dependent
  trailing flag for every source rank before a full-core join; and
- it copies received rank regions into final expert-contiguous output.

The path alternates two window buffers, which is a useful graph-reuse pattern,
but it still publishes and waits at source-rank granularity. The layered CANN
path goes lower: it constructs RMA work-queue entries and doorbells, embeds
per-token/status flags, and explicitly cleans or advances protocol state.

The inspected public host API exposes `HcclBatchPut` and `HcclBatchGet`, while
the inspected symmetric-window API exposes peer-pointer lookup. Neither
header documents a device-side `put-with-signal` operation or its memory
ordering. The official `BatchWrite` documentation is also explicit that
`Wait(handle)` proves local send completion, not peer receipt. The installed
device kernels consequently remain the stronger nearby evidence: they use
ordered payload/flag writes, explicit pipeline events, cache maintenance,
polling, and double-buffered state. This does not prove that a narrower HCCL
primitive is unavailable elsewhere; it means the redesign must not assume one
until its contract is found or tested.

## DeepEP protocols

### Legacy low-latency path: direct expert-addressed ingress

DeepEP's legacy low-latency dispatch assigns a source-local atomic slot for
each destination expert and writes directly to a fixed layout equivalent to:

```text
[destination local expert][source rank][source slot]
```

After issuing all payload writes, the source publishes a count for each
`(destination expert, source rank)` channel. The receiver waits for those
counts and packs the source fragments into a contiguous local-expert tensor.
Combine returns data to slots derived from dispatch metadata and uses
expert-channel finish flags.

This is direct destination addressing without a metadata round trip. Its
price is a hidden-state copy for every expert route and receiver-side packing.

### Current elastic path: rank deduplication plus a notify phase

Current DeepEP separates notification/count exchange from payload dispatch.
It counts both expert routes and unique destination ranks, exchanges those
counts, computes prefix sums, then assigns one destination-rank slot per
token. If multiple top-k experts live on the same rank, the hidden vector is
sent once and the top-k metadata tells the receiver how to fan it out.

The path also supports reusing previously computed destination slot indices.
That is directly relevant to graph replay: a stable template can cache
placement decisions while runtime state still uses a safe generation or
phase. Current DeepEP ends dispatch with a GPU barrier that flushes stores
before announcing completion, so it is not evidence that all-rank completion
is unnecessary; it is evidence that rank deduplication and cached placement
can coexist with a conservative correctness boundary.

## NCCL EP low-latency protocol

NCCL EP's current low-latency dispatch makes the same central bandwidth
choice as current DeepEP:

- it deduplicates a token's destinations by rank;
- it allocates one source-rank slot per destination rank;
- its header carries the token identity and all top-k routes;
- the receiver reads that header and packs the token into each active local
  expert; and
- count notification remains expert-lane shaped, partly to flush independent
  communication channels.

The receiver waits until all local-expert count lanes for a source rank have
arrived before it consumes that rank's payload. The source contains a TODO
about whether the direct P2P release/acquire pairing is sufficient across SMs,
which is a useful warning against treating a flag store as self-evidently
safe.

NCCL EP combine is more direct. It writes each expert result into the origin
rank's precomputed `(token, top-k return slot)` and then publishes finish
signals per expert communication lane. The origin still performs a grid join
before reduction. This closely matches the current Ascend operator's direct
return placement, while offering a finer completion namespace than its final
all-rank toggle.

## Memory-ordering lesson

The desired semantic primitive is not merely a flag. It is:

```text
publish(payload) happens-before publish(readiness)
observe(readiness) happens-before consume(payload)
```

NVSHMEM documents `put_signal` as updating the remote signal after the data
transfer and pairs it with wait/test operations. NCCL GIN distinguishes
strong signals, which cover all preceding puts to the peer/context, from weak
signals, which cover only the attached put. DeepEP explicitly flushes all
used QPs and applies a system-scope fence before a world barrier that mixes
NVLink stores and GIN signals.

The transferable rule is to bind readiness to a documented release/acquire
contract at the actual transport scope. A local pipeline event, a cache clean,
an RDMA completion, and remote visibility are different facts and must not be
silently substituted for one another.

Primary references:

- NVSHMEM signaling operations:
  <https://docs.nvidia.com/nvshmem/api/gen/api/signal.html>
- NVSHMEM memory ordering:
  <https://docs.nvidia.com/nvshmem/api/gen/api/ordering.html>
- NCCL device GIN:
  <https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/device_gin.html>
- AscendC HCCL `BatchWrite`:
  <https://www.hiascend.com/document/detail/en/canncommercial/850/API/ascendcopapi/atlasascendc_api_07_10132.html>
- DeepEP repository:
  <https://github.com/deepseek-ai/DeepEP>
- NCCL EP source:
  <https://github.com/NVIDIA/nccl/tree/master/contrib/nccl_ep>

## The central tradeoff: route copies versus receiver packing

Define:

```text
R = number of active (token, expert) routes
U = number of unique (token, destination rank) pairs
duplication factor = R / U
```

For top-k routing, `R / U` can be materially greater than one when several
chosen experts live on the same rank. Four useful protocol families follow:

| Family | Hidden payload | Destination allocation | Receiver work | Fixed coordination |
|---|---|---|---|---|
| current Ascend source-pull | once per route, peer-read | source expert layout | pull and concatenate | dense count rows + rank completion |
| DeepEP-legacy direct expert | once per route, peer-write | fixed source slab per expert | pack source slabs | per expert-source count |
| DeepEP/NCCL EP rank-dedup | once per unique destination rank | source atomic rank slot | inspect header and fan out | notify/count phase + barrier/lane signals |
| receiver-reserved direct expert | once per route, peer-write | receiver prefix/reservation | little or no packing | metadata round trip + readiness |

The last family was the initial roadmap hypothesis. It removes packing only
if the receiver can assign final contiguous expert offsets before payload
arrival, but the extra round trip may dominate decode. The legacy fixed-slab
layout removes that round trip, but retains packing or requires segmented GMM.

For the current Ascend operator, per-expert direct writes do **not** introduce
a new route-duplication cost relative to the baseline: the current routing
path already materializes one hidden row per `(token, top-k slot)`. However,
rank deduplication remains a credible bandwidth optimization for larger M and
must be measured before choosing one permanent wire format.

## Revised design judgment

No inspected implementation supports the strong claim that a two-round
receiver-reservation protocol is unconditionally best. The evidence instead
supports a decode/prefill split or an adaptive choice:

1. **First remove work that is unambiguously unnecessary.** Build an active
   expert list and skip empty expert compute/control while retaining current
   communication semantics.
2. **Measure `R`, `U`, nonempty source-expert fragments, bytes, and wait time.**
   `R / U` is the discriminating statistic between direct-expert and
   rank-deduplicated ingress.
3. **Prototype the cheapest decode protocol.** Compare a fixed
   `[expert][source][slot]` direct-write layout against the existing
   source-pull path. Do not pay for a reservation round until measurement
   shows receiver packing or fragmented GMM is more expensive.
4. **Preserve or improve the rank-deduplicated option for prefill.** Large M
   may value bandwidth and contiguous batching more than one extra local copy.
5. **Refine egress independently.** Current direct return placement should be
   retained. Replace the final all-rank toggle only after a per-source or
   per-fragment visibility primitive has a proven contract.
6. **Use rolling generations or double buffers.** Avoid clearing a global
   readiness matrix on every replay. Counters must tolerate wrap under a
   bounded outstanding-generation assumption, or alternating buffers must
   prove that reuse cannot race a previous consumer.

## Prototype resolution

The first implementation did not need a second receiver-reservation round.
The existing complete count matrix lets every source compute the same final
expert-major destination prefixes, so sources can push disjoint fragments
directly and publish one source-owned sealed-wave epoch. This removes
receiver-side per-expert pulls and their barriers while keeping the streaming
direct-return path unchanged.

The EP2 prototype passes repeated changing-generation correctness and improves
six of seven bracketed route families. The exception is the smallest
graph-padded route family, where an extra wave epoch is not amortized. Thus the
study's adaptive-policy judgment is retained, but the direct-placement family
now has positive local evidence rather than remaining only a hypothesis.

## Next experiment

For the remaining policy and hardening questions, retain removable phase and
cardinality counters for:

- route count `R` and unique token-destination count `U`;
- active local experts and nonempty `(source, local expert)` fragments;
- dense count publication/poll duration;
- ingress peer-copy duration and bytes;
- each local expert's compute duration, including empty slots;
- return-write duration; and
- final rank-completion wait plus unpermute duration.

Run M in `{1, 2, 4, 8, 16, 32}` and representative mixed-prefill shapes. This
single measurement separates three currently entangled hypotheses: empty
expert control cost, ingress protocol cost, and final completion/unpermute
cost. It also decides whether the next prototype should optimize route
latency or rank-deduplicated bandwidth.

## Open questions

- Does an installed CANN/HCCL device API provide a documented strong
  payload-plus-signal operation that is not exposed by the inspected headers?
- Can fixed source slabs feed the local grouped GEMMs without a full compacting
  copy, or does segmented execution lose more than it saves at small M?
- At Qwen's observed routes, what are the distributions of `R / U`, active
  experts, and nonempty source-expert fragments in decode and mixed prefill?
- Can one source-owned readiness word cover all of that source's return slots
  without delaying unrelated local tokens?
- Does graph replay keep routes stable enough to reuse slot indices, or only
  shapes and capacities?

Those questions now bound the uncertainty. More protocol design should wait
for the phase/cardinality evidence rather than adding speculative machinery.
