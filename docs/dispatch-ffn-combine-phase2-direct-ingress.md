# Dispatch-FFN-Combine Phase 2 direct-ingress prototype

Status: experimental compile-time prototype; exactness passed repeated EP2 and
EP4 validation, and the zero-scan sparse selector has hardware crossover
evidence. The production path remains unchanged unless
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
output and expert-count oracles.

The same harness then passed 2,048 generations on EP4, including empty ranks,
one-destination waves, and local-only, peer-only, and mixed routes. Its capacity
is derived as `EP * M * topK`: retaining EP2's 512-row constant at EP4 caused
the expected one-destination truncation at 1,024 rows, while expert counts
remained exact. Correcting that test precondition produced a clean 28-wave
pilot and the full 2,048-wave pass. A tracked four-rank regression with 64
local experts per rank (256 global experts) also passes. These results close
the current EP4 and larger-topology correctness gates; signed epoch wrap and
topologies beyond EP4/256 experts remain untested.

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

### Zero-scan sparse fallback candidate

`DISPATCH_FFN_COMBINE_DIRECT_INGRESS_SPARSE_FALLBACK` is a compile-time
candidate for the remaining `active=1` crossover. It is deliberately coupled
to `DISPATCH_FFN_COMBINE_DIRECT_INGRESS` and leaves the default build
unchanged.

The production uniform-token wrapper supplies a prefix-valid `mc2_mask`.
Consequently, `mask[1]` exactly separates zero or one active token from two or
more whenever the graph capacity is greater than one. A source publishes that
local direct-ingress preference in the first padding lane after its real
expert counts. The existing tagged count-row exchange carries the lane to
every peer, so the selector adds no communication round and reads only one
integer per source, O(EP), rather than scanning the O(EP^2 * experts-per-rank)
count matrix. Every rank enables direct ingress only when every source's
preference is nonzero; this all-source AND prevents ranks from choosing
different protocols even if a lower-level caller supplies differing masks.

The buffer layout is chosen before this dynamic decision. When the selector
falls back, the original pull loop writes into the already selected direct
input buffer, while AIC and egress retain the matching direct-buffer offsets.
The selector therefore changes only how input rows arrive, not their final
layout or the AIC/AIV address contract.

The candidate passed a clean Ascend 910B package compile for all four registered
dtype/format variants. The generated driver was checked to contain both direct
ingress macros and retain DCCI; all four selector objects were 609,656 bytes and
matched the known compile-only checkpoint hashes.

Five fresh EP2 processes bracketed selector and direct-only objects in
S-D-S-D-S order, with 10 warmups and 50 device-event samples per case. Exact
output passed throughout. At graph `active=1`, selector observations were
291.02, 290.69, and 295.63 us versus direct-only observations of 309.59 and
298.84 us. The median reduction was 4.34%: consistently positive, but narrowly
below the preregistered 5% useful-effect target. At `active=16`, selector and
direct-only medians were 429.18 and 431.46 us respectively, a 0.53% difference
inside the 5% parity bound. The selector therefore demonstrates the intended
crossover without regressing the direct path, but the smallest-wave gain should
not be reported as meeting a >=5% target.

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

The direct-placement mechanism passes the Phase 2 EP2 and EP4 correctness
gates and has a strong first performance signal. It is not ready to become the
production default because:

1. EP4 with 256 global experts is validated, but wider generic EP/topology
   bounds are not;
2. the graph `active=1` selector is consistently faster, but its 4.34% median
   gain narrowly misses the preregistered 5% useful-effect target;
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

The next highest-value correctness experiment is an adversarial epoch-wrap
test, followed by a wider generic EP/topology bound if production targets need
it. Exact per-source cycle attribution would sharpen the mechanism diagnosis,
but the current Source product exposes visits rather than cycles. Neither the
compile-time guard nor dispatch policy should be widened until the epoch and
visibility boundaries are resolved.
