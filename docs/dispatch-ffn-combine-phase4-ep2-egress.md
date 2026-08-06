# Dispatch-FFN-Combine Phase 4 EP2 egress discriminator

Status: preregistered completion-cost ceiling experiment. This note authorizes
neither a generic active-rank protocol nor production enablement.

## Question

The existing GMM2 epilogue already writes each result directly into its
source rank's final `offsetD` return slots. After all local return writers
finish, every rank still toggles completion state to every other rank and
waits for all ranks before unpermute.

Phase 4 first asks a smaller question than how to replace that collective:

> On EP2, when both sources route only to their own local experts, is removing
> the now-vacuous cross-rank completion handshake enough to improve both
> warmed eager and graph-replay latency?

This is a cost-ceiling experiment. A local-only source has no remote return
writer, so removing the handshake captures the largest benefit an exact
active-rank dependency set could provide on EP2. If this ceiling does not win,
building rank masks, generations, or generic completion state cannot be
justified.

## Existing dependency

The current egress path is:

```text
GMM2 produces local expert tiles
  -> AIV epilogue writes each tile directly to its source-owned offsetD slot
  -> every AIV completes its return-write pipeline
  -> all local AIVs join
  -> every rank publishes one alternating completion state to every rank
  -> every rank waits for every peer's matching state
  -> local unpermute reads offsetD and reduces top-k rows
```

The first four edges are necessary even for local-only routes: unpermute must
not read a local return slot before all local writers finish. The cross-rank
publication and wait are necessary only for ranks that can write one of this
source's return slots.

The complete count matrix already identifies that dependency: source `s`
needs destination rank `d` exactly when any
`count[s][d][local_expert]` is nonzero. Computing and distributing that set is
deliberately outside the first experiment.

## Smallest disposable candidate

Build one EP2-only object under a disposable compile guard. The object is
valid only for the preregistered local-only route and is never installed as a
production or tracked implementation.

Replace only the rank-wide completion section with:

```text
all return writers finish BlockEpilogue2::Finalize
  -> each AIV executes a DDR data-sync barrier
  -> all local AIVs join
  -> the existing local unpermute runs unchanged
```

Keep unchanged:

- Phase 2 count exchange, direct ingress, and sealed-wave protocol;
- GMM1, activation, GMM2, and return-slot address calculation;
- the `offsetD` layout and `expandedRowIdx` identity mapping;
- the number of AIVs participating in unpermute;
- output weighting, reduction, dtype, and capacity behavior; and
- every default and non-EP2 compile path.

The explicit DDR barrier replaces the ordering previously obtained indirectly
from later MTE3 completion-state traffic. The local all-AIV join remains; the
experiment removes no writer-to-local-consumer dependency.

## Correctness gate

Use nonzero weights that encode the selected global expert in output column
zero. Both ranks route exclusively to their own 64-expert shard. Across one
communicator and peer-window lifetime, cover:

- repeated eager generations with rotating local experts;
- nonuniform local expert counts;
- one rank with an empty active-token set;
- graph padding with one and several active tokens; and
- output and local expert-count comparison against the structural oracle.

This candidate is intentionally not run on a remote route. A passing
local-only test proves only the narrowed dependency and the local visibility
edge; it is not evidence for a generic completion protocol.

The known-good Phase 2 object must be restored and checksum-verified after
every candidate run.

## Performance gate and stop rule

Use the production-shaped BF16 layer microbenchmark with `H=7168`,
`FFN=2048`, `M=64`, top-k 4, and 64 local experts. Give every rank the same
amount of local expert work. Warm the communicator and graph before collecting
at least 100 device-event samples.

Bracket the completion-bypass object with the Phase 2 object in B-C-C-B order
and report the slower rank's per-iteration device median and robust spread for:

- local-only eager execution; and
- the same local-only route under repeated graph replay.

The ceiling passes only if both candidate observations improve both eager and
graph replay, and the effect is larger than the nearby baseline drift. Host
time is supporting evidence only.

If the ceiling is at parity or slower in either mode, close Phase 4 at EP2 and
retain the existing rank-wide completion. Do not implement an active-rank mask
or generalize to EP4/EP8.

If the ceiling wins, the next experiment is still EP2: derive the single peer
dependency without a new dense scan, retain a source-owned modular generation,
and validate alternating local-only and remote generations. Only that correct
runtime protocol may be considered for EP4.
