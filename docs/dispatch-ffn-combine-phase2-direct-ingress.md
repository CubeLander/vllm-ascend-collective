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

A separate long-reuse harness then held one EP2 communicator and peer window
open for 2,048 consecutive generations. It varied active-token counts over
`0, 1, 2, 7, 16, 31, 64` and rotated through all-empty, one-destination,
local-only, peer-only, and mixed expert routes. Every generation passed exact
output and expert-count oracles. This closes the ordinary changing-generation
stress gate on EP2; it does not exercise signed epoch wrap or a larger EP
topology.

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

This independently supports reducing protocol coordination rather than tuning
payload bandwidth first.

A disposable debug object built by invoking `opc` directly with
`--op_debug_level=1 --op_debug_config=dump_loc,ccec_g` supplied the missing
DWARF line table. Source-mode collection then extracted 36,108 PC/source
relations and 375,915 address-to-line relations. Joining its `fdata` with the
basic-block map attributed 650 nonzero blocks and 95,832 control-flow visits
to source lines, including the direct-ingress prefix, publication, and
consumer phases.

Two boundaries matter:

- basic-block visit counts are control-flow evidence, not per-line cycles, so
  they must not be presented as latency-hotspot percentages; and
- the debug object has a slightly different `.text` section from the release
  object, so it is suitable for source attribution but not comparative timing.

The normal package-build path did not preserve `ccec_g`: its generated options
contained two `ALL` rows and the later ordinary option row shadowed the debug
row. Direct `opc` compilation was therefore required for this disposable
artifact. The conservative release object was restored after collection.

## Gate result and next work

The direct-placement mechanism passes the Phase 2 EP2 correctness gate and has
a strong first performance signal. It is not ready to become the production
default because:

1. generic EP and larger expert topology bounds are not yet validated;
2. the graph `active=1` crossover needs a zero-scan policy input;
3. the source-owned epoch assumes a fresh common initial generation and a
   bounded one-wave skew; 2,048-generation communicator reuse passes, but
   signed wrap still needs an adversarial test;
4. the DCCI ablation passes the current test but is not yet strong enough to
   replace the conservative visibility boundary; and
5. the dense count exchange remains, even though per-expert receiver pulls and
   their barriers are gone.

### DCCI ablation

The received-row DCCI loop can be removed in a disposable build with
`DISPATCH_FFN_COMBINE_DIRECT_INGRESS_SKIP_DCCI`. That build passed the complete
two-rank repeated-generation test. A 100-sample run was bracketed by two builds
with DCCI:

| Case | DCCI before | No DCCI | DCCI after | Judgment |
|---|---:|---:|---:|---|
| decode M=8, spread | 428.57 us | 410.65 us | 441.16 us | 4.2--6.9% faster |
| prefill M=256, spread | 532.41 us | 526.40 us | 547.83 us | 1.1--3.9% faster |
| graph M=64, active=1 | 340.28 us | 352.15 us | 353.67 us | inconclusive |

Removing DCCI is directionally useful for the larger route families but does
not explain the smallest-wave floor. More importantly, repeated correctness is
not a substitute for a documented peer-write-to-Cube visibility contract. The
conservative direct prototype therefore keeps DCCI by default and retains the
skip macro only as an explicit experimental ablation.

The next highest-value experiment is a generic-EP correctness run, followed by
a zero-scan crossover input for the one-active-token graph case. Exact
per-source cycle attribution would sharpen the mechanism diagnosis, but the
current Source product exposes visits rather than cycles and should not block
the generic-topology gate. Only after those gates should the prototype's
compile-time guard or dispatch policy be widened.
