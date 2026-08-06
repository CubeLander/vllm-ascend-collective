# Dispatch-FFN-Combine Phase 3 EP2 readiness discriminator

Status: preregistered micro-logic experiment. This note authorizes neither a
generic collective implementation nor production enablement.

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

## Smallest protocol

Enable the experiment under a disposable compile guard and only when
`EP == 2`. Other configurations execute the Phase 2 sealed-wave path.

```text
count matrix and final offsets become available
  -> source core 0 advances a local generation register
  -> local AIV join distributes that generation
  -> source workers push disjoint destination-expert fragments
  -> each completed nonempty fragment publishes a source-owned ready epoch
  -> one receiver-coordinator AIV observes sources for experts in GMM order
  -> coordinator releases each ready scheduled expert to AIC
  -> source workers and coordinator join after the last expert
  -> existing activation, GMM2, return, and layer-final completion continue
```

The initial AIV-to-AIC cumsum flag moves before ingress. GMM1 may then wait on
the existing compact per-scheduled-expert progress sequence while the AIV
source workers continue pushing later experts. The GMM order remains fixed;
the experiment overlaps an earlier expert with a later ingress tail rather
than adding out-of-order expert scheduling.

## State ownership and layout

Keep readiness source-owned, matching the proven Phase 2 direction. For each
source window, reserve:

```text
one cache line: source generation register
EP * experts_per_rank cache lines: ready[dst_rank][local_expert]
```

A worker that owns `(dst_rank, local_expert)` performs the payload copy, waits
for that copy's MTE3 completion, executes a DDR data-sync barrier, and then
publishes the generation to its local source-owned ready slot. The destination
polls that slot through the source's peer window. Cache-line separation avoids
unrelated DCCI operations sharing a control line.

The destination already has the complete count matrix. It polls only sources
with a nonzero fragment for that expert; an empty fragment publishes nothing.
The existing unsigned modular wait rule remains unchanged: the requested
generation and its allowed one-generation look-ahead are accepted, one behind
may still be publication in flight, and an impossible larger skew traps.

The EP2 prototype fails closed to Phase 2 if the additional control region
does not fit before the reserved peer-window control boundary.

## AIV role split

Reserve one AIV as the receiver coordinator. Preserve the Phase 2 target-to-core
mapping for every other worker; redistribute only the coordinator's otherwise
owned targets to core 0. This avoids turning a new work partition into the
measured mechanism.

Source workers never wait for peer readiness. The coordinator visits local
experts in existing GMM order, skips unscheduled experts from the shared count
matrix, waits for every nonempty source fragment of the next scheduled expert,
orders those observations with a DDR barrier, and publishes exactly one
existing AIV-to-AIC progress flag. A final local AIV join waits for both all
source workers and the coordinator, but GMM1 can already be executing.

## Causal route and instrumentation

Use exactly two active destination experts: group 0 and a later group assigned
to the same source worker after group 0 under the existing Phase 2 mapping.
Give both experts real rows. The source worker completes and publishes group 0,
then performs the later expert's larger copy. The sealed-wave control cannot
release group 0 until both copies complete; the Phase 3 candidate can.

A disposable diagnostic build records same-device system-cycle timestamps for:

1. receiver observation of each tested expert's final required fragment;
2. the AIC start immediately after each tested expert's progress wait; and
3. completion of the final local AIV join.

The causal condition is:

```text
GMM1 start(group 0) < receiver ready(later group)
```

These timestamps prove ordering only. Release-object device events decide
performance; an instrumented object is never used for the latency comparison.

## Correctness gate

Before timing, one EP2 communicator and peer window must cover:

- both sources contributing to both tested experts;
- one source absent from the early expert;
- local-only and remote-only early fragments;
- an empty destination expert between the two scheduled experts;
- alternating route templates across repeated generations;
- graph-padding masks; and
- an injected impossible generation skew that traps rather than hangs.

Outputs and expert counts must match the existing structural oracle. The
ordinary Phase 2 object must be restored and checksum-verified after every
disposable run.

## Performance gate and stop rule

Bracket the Phase 3 release object with the Phase 2 sealed-wave object in both
orders. Warm every process and communicator before collecting at least 50
device-event samples. Report the slower rank's median and a robust spread for:

- the deliberately staggered two-expert discriminator;
- a balanced two-expert control with little ingress skew; and
- one production-shaped graph-replay case.

Promotion requires a repeatable critical-path win in the staggered case and
no material regression in the balanced control. The graph case must preserve
the direction after warmup. Host time is supporting evidence only.

If the EP2 candidate is at parity or slower, or if its benefit exists only in
the diagnostic object, archive it and keep Phase 2 sealed-wave ingress. Do not
build EP4/EP8 state. If it wins, the next task is EP4 correctness and the same
mechanism test, not generic collective redesign.

## Borrowed lesson and local choice

DeepEP and NCCL EP both demonstrate expert- or lane-shaped readiness, but they
also retain packing, barriers, or transport-specific release semantics. This
experiment borrows only the narrower dependency model. It deliberately keeps
the Ascend Phase 2 source-owned remote-read direction and explicit DDR/DCCI
ordering rather than importing a remote-notification protocol whose local
W8A8 ablation already regressed.
