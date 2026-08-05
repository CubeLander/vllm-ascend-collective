# Dispatch-FFN-Combine Phase 2 direct-ingress prototype

Status: experimental compile-time prototype; correctness passed on EP2 and the
first bracketed performance result is positive for six of seven route families.
The production path remains unchanged unless
`DISPATCH_FFN_COMBINE_DIRECT_INGRESS` is defined.

## Question

Can the existing dense count exchange be reused to remove receiver-side
per-expert pulls and barriers without adding a second destination-reservation
round trip?

The key observation is that every source already receives the complete route
count matrix. Once that matrix is visible, every source can independently
derive both:

- its source-row prefix in the existing expert-grouped staging buffer; and
- the final destination-row prefix in the receiver's expert-major input.

No remote atomic allocator, per-expert doorbell, or returned reservation
descriptor is required for this candidate.

## Protocol

The experimental ingress is a sealed source wave:

```text
route locally into source-owned offsetA
  -> publish and gather the existing count matrix
  -> each source computes exact final destination prefixes
  -> each source pushes every nonempty fragment into remote final input
  -> all local producer cores join
  -> one source-owned epoch publishes the complete source wave
  -> every destination waits for every source epoch
  -> invalidate only the received contiguous input rows
  -> publish the existing compact AIV-to-AIC expert progress sequence
```

Destination rows remain expert-major. Within an expert, lower source ranks own
the preceding rows. Across experts, all rows of lower local-expert indices own
the preceding region. These ownership rules make writes disjoint and remove
remote allocation.

The direct input buffer reuses peer-window space beginning at the old return
offset. Its graph-static capacity is bounded by
`min(max_output_size, EP * M * topK)` rows rather than the policy capacity
alone. The return buffer moves immediately after it. If input plus return data
cannot fit below the count-control region, the experiment fails closed to the
original path.

The count matrix and ingress epoch have disjoint, cache-line-aligned storage.
The epoch is source-owned and independent of the existing egress completion
state. A waiter accepts the requested epoch or one generation ahead, matching
the existing barrier's bounded-skew rule and avoiding an exact-value ABA hang.
Payload cache maintenance is distributed by hardware cache line, so distinct
cores never issue DCCI against the same line.

Egress is intentionally unchanged. GMM2 continues to stream each completed
source fragment directly into that source rank's return region; this prototype
does not turn reply transmission into a sealed batch.

## Failure that exposed the generation boundary

The first direct-placement build reused the existing rank synchronization for
both ingress and egress. The repeated two-rank test failed on the first changing
route generation: 64 of 16384 output elements mismatched, all on the rank with
zero received experts. Adding payload DCCI did not change the failure.

A focused generation trace showed that expert counts and later generations
were correct while the empty destination rank consumed stale return data. The
shared synchronization counter allowed ingress and egress phases to alias by
one generation. Giving ingress its own source-owned epoch removed the failure.
This is evidence that phase identity, not destination-prefix arithmetic, was
the defect.

## Correctness result

The ordinary two-rank end-to-end test passed after the dedicated epoch was
added, and passed again after cache maintenance was narrowed from the full
graph-static input reservation to the rows actually received:

```text
1 passed, 16 warnings in 36.90s
```

The test includes repeated peer-window reuse, changing nonuniform route
generations, a rank with zero received experts, late sparse tensor-list routes,
and graph padding with one active token.

## First performance discrimination

The Phase 1 compact-signal binary was run before and after the direct binary,
with 50 repetitions per case. The table uses the slower rank's median device
event as a distributed critical-path estimate. Percentages show the direct
binary relative to the two bracketing Phase 1 runs.

| Case | Phase 1 before | Direct ingress | Phase 1 after | Direct judgment |
|---|---:|---:|---:|---|
| decode M=8, spread | 467.75 us | 369.40 us | 481.47 us | 21.0--23.3% faster |
| prefill M=64, spread | 480.67 us | 396.90 us | 499.60 us | 17.4--20.6% faster |
| prefill M=256, spread | 605.42 us | 469.19 us | 565.51 us | 17.0--22.5% faster |
| decode M=8, four experts | 351.35 us | 278.22 us | 311.10 us | 10.6--20.8% faster |
| prefill M=64, four experts | 322.71 us | 283.34 us | 327.79 us | 12.2--13.6% faster |
| graph M=64, active=1 | 332.76 us | 325.50 us | 294.17 us | inconclusive; -2.2% to +10.7% |
| graph M=64, active=16 | 508.84 us | 416.03 us | 510.90 us | 18.2--18.6% faster |

Slower-rank host medians agree for the six positive cases, with improvements
from 2.7% to 19.4%. For graph `active=1`, host medians regress by 3.5--14.4%.
The sealed-wave epoch is therefore not free: the smallest graph-padded route
family has too little ingress work to amortize it.

A trial runtime selector summed the full route matrix on one core and kept the
single-active-token case on the pull path. The serialized scalar scan added
roughly 50 us of rank skew and degraded the otherwise positive cases, so it was
rejected rather than committed. A production selector must reuse a cheaply
available cardinality or compute it during routing; it must not add a new dense
scan.

## msopprof evidence

MC2 cannot be safely replayed as an isolated kernel or range: the peer must
participate in every replay. The working method is application replay with the
profiler injected into exactly one rank and a fresh uninstrumented peer for
each replay. Rank 0 and rank 1 are profiled separately.

The Phase 1 operator is control/wait dominated rather than bandwidth bound:

- 24 Cube kernels had a 65.12 us median, including 27.95 us on wait id 0 and
  6.99 us on wait id 10, with median Cube utilization reported as zero;
- 48 Vector kernels had a 91.13 us median, including 38.72 us on wait id 14;
- Vector utilization was 0.65%, MTE2 4.49%, and MTE3 1.48%; and
- measured GM/L1/UB bandwidth ratios were all below 0.1%.

This does not identify every wait site because the release objects lack
`.debug_line`, but it independently supports reducing protocol coordination
rather than tuning payload bandwidth first. A disposable `-g` build is needed
for source-level PC attribution.

## Gate result and next work

The direct-placement mechanism passes the Phase 2 EP2 correctness gate and has
a strong first performance signal. It is not ready to become the production
default because:

1. generic EP and larger expert topology bounds are not yet validated;
2. the graph `active=1` crossover needs a zero-scan policy input;
3. the source-owned epoch assumes a fresh common initial generation and a
   bounded one-wave skew; wrap and communicator reuse need adversarial tests;
4. DCCI should be ablated now that the phase-alias defect is understood; and
5. the dense count exchange remains, even though per-expert receiver pulls and
   their barriers are gone.

The next highest-value experiment is source-attributed msopprof on a disposable
debug build, followed by a DCCI ablation and a generic-EP correctness run. Only
then should the prototype's compile-time guard or dispatch policy be widened.
