# Dispatch-FFN-Combine EP2 rank-deduplicated pull discriminator

Status: preregistered experimental discriminator. The production default and
the existing direct-ingress candidate remain unchanged.

## Decision

Test whether an EP2 receiver-pull path that stages each source token once with
its top-k expert metadata is a better production direction than sealed-wave
direct ingress. The decision is whether to invest in a complete
rank-deduplicated protocol, not whether this deliberately bounded prototype is
production ready.

## Claim

For identical inputs, routes, weights, and graph mode, a receiver can reconstruct
the existing expert-major GMM1 input from one peer-visible hidden row per source
token plus top-k expert IDs. On production-like mixed-prefill routes, avoiding
route-expanded remote payload and source-wave completion may reduce the slower
rank's warmed device latency without changing exact output.

The prototype is allowed to retain the existing `moe_init_routing_v2` pass for
counts and return metadata. Consequently, a positive result is conservative:
it demonstrates value before removing the now-redundant expanded-payload work.

## Fixed parity contract

- EP is exactly 2; other EP sizes fail closed to the existing path.
- BF16 inputs and outputs, weight layout, top-k, active mask, expert order,
  GMM1/SwiGLU/GMM2, direct return, unpermute, and final completion are unchanged.
- The existing dense count exchange remains unchanged.
- Each source stages `x[M, K]` once and stages the masked
  `expert_id[M, top_k]` metadata in its peer-visible window.
- The receiver constructs the same expert-major, source-major row order as the
  existing pull path. No segmented GMM, reservation round, remote atomic, or
  new host-visible synchronization is introduced.
- Correctness is exact output plus exact expert-token counts across changing
  routes and repeated peer-window reuse.

## Comparison and metrics

Compare three separately built binaries with identical route cases:

1. Phase-1 compact-signal receiver pull;
2. sealed-wave direct ingress with its sparse fallback; and
3. the rank-deduplicated receiver-pull prototype.

Use bracketing baseline runs and report the slower rank's median device event.
Record per-rank medians, rank skew, hidden rows staged, route count `R`, unique
token-destination pairs `U`, and `R / U`. Include decode shapes and the observed
production mixed-prefill range near `M=235..237`. Warming and graph replay are
required before a performance decision.

## Stop rules

- Stop immediately on any output, expert-count, generation-reuse, or graph
  correctness failure.
- Close the implementation path if the prototype does not beat both bracketing
  Phase-1 runs on the production-like mixed-prefill discriminator, or if a win
  exists only in eager mode and disappears under warmed graph replay.
- A decode regression is acceptable only if an already available, zero-scan
  runtime fact can select the path without a new dense scan or communication
  round.
- Do not generalize to EP4/EP8, remove the legacy path, or change the production
  default until EP2 proves a useful effect.

## Interpretation labels

- **REPRODUCED**: exact EP2 behavior and a repeatable warmed graph win under the
  fixed comparison.
- **NOT_REPRODUCED**: faithful EP2 evidence shows parity or regression.
- **INSUFFICIENT_EVIDENCE**: the prototype cannot preserve parity or the
  comparison is invalidated by build, device, or route mismatch.


## Prototype receipt, 2026-08-07

The compile-guarded EP2 prototype was built from source commit
`68c83f69d457a4506aeb6bf653dba96a2e2b3fb9` with 96 requested outer build
jobs and these operator options:

```text
-UDISPATCH_FFN_COMBINE_PROFILE
-UDISPATCH_FFN_COMBINE_DIRECT_INGRESS
-UDISPATCH_FFN_COMBINE_DIRECT_INGRESS_SPARSE_FALLBACK
-DDISPATCH_FFN_COMBINE_RANK_DEDUP_PULL
-Wno-ignored-attributes
```

The generated narrow package had SHA-256
`9b2181d9abe2b2cd1438f5180a0182784f9386fac7573e2dea2cd5aa771725a5`.
The tracked EP2 exactness test passed on physical NPUs 4 and 5 in an isolated
operator prefix. It covered repeated uniform routes, changing non-uniform
routes, the one-live-token active mask, sparse high-numbered experts, and both
packed and tensor-list weight ABIs (`1 passed` in 41.62 s). The prototype
therefore clears the correctness gate for performance discrimination; it is
not yet a performance result.

One build-system observation is worth retaining in provenance: the outer
Ninja build received `-j96`, but the nested clean ascend-protobuf external
project invoked plain `cmake --build .` and compiled serially. This affected
build wall time only, not the generated operator object.

## Bracketed performance result, 2026-08-07

Three narrow packages were forced through fresh generated operator sources and
kernel objects from the same tracked source. A first incremental baseline
package retained the preceding rank-dedup compile-option row, so it was
rejected before timing. The accepted package SHA-256 values are:

| Binary | Compile-time ingress | SHA-256 |
|---|---|---|
| Phase-1 baseline | direct off; rank-dedup off | `6e3844fdaa0fc564a18138d63bfdf4dd95e21118245248e2c87b448409402c21` |
| Rank-deduplicated pull | rank-dedup on; direct off | `9b2181d9abe2b2cd1438f5180a0182784f9386fac7573e2dea2cd5aa771725a5` |
| Sealed-wave direct ingress | direct plus sparse fallback; rank-dedup off | `a987e21a17a32d646bef36df678be736862e83feddbfc7c25b879ba45b59afa4` |

Every synthetic family has `R / U = 4.0`: each accepted token contributes
eight routes but reaches both EP2 destinations, so rank-deduplicated staging
reduces hidden-row communication fourfold. Runs used physical NPUs 4 and 5,
an isolated operator prefix, three shape-local warmups, 50 samples, and the
slower rank's median device event as the critical-path proxy.

| Route family | Baseline before | Rank-dedup pull | Direct ingress | Baseline after | Rank-dedup judgment |
|---|---:|---:|---:|---:|---|
| decode M=8, spread | 505.83 us | 383.60 us | 366.92 us | 439.61 us | beats both baselines; direct 4.4% faster |
| prefill M=64, spread | 553.14 us | 419.31 us | 427.90 us | 470.57 us | beats both baselines; 2.0% faster than direct |
| prefill M=256, spread | 599.57 us | 816.97 us | 773.37 us | 686.96 us | regresses both baselines; direct 5.3% faster |
| decode M=8, four experts | 252.68 us | 294.48 us | 288.49 us | 302.20 us | baseline drift crosses result |
| prefill M=64, four experts | 275.84 us | 363.81 us | 311.89 us | 323.94 us | regresses both baselines; direct 14.3% faster |
| graph-shaped M=64, active=1 | 291.92 us | 285.34 us | 313.81 us | 337.18 us | beats both baselines; direct fallback is mixed |
| graph-shaped M=64, active=16 | 495.37 us | 406.41 us | 425.25 us | 512.63 us | beats both baselines; 4.4% faster than direct |

The broad eager matrix shows real useful work, not a no-op prototype, but also
the expected receiver-scan crossover: scanning source metadata independently
for every local expert becomes expensive at large M and when few experts own
most routes. Fourfold fewer hidden rows is therefore not itself a latency
claim.

The preregistered production-like discriminator was then run at M=236 with 100
samples and the same `R / U = 4.0` route shape:

| Mode | Baseline before | Rank-dedup pull | Direct ingress | Baseline after |
|---|---:|---:|---:|---:|
| warmed eager device median | 756.45 us | 663.08 us | 580.08 us | 669.46 us |
| captured graph replay device median | 438.71 us | 380.95 us | 373.64 us | 417.30 us |

The captured-graph row used a fresh `torch.npu.NPUGraph` per rank and binary,
three eager warmups, three replay warmups, 100 timed replays, exact output and
expert-count checks, and events around `graph.replay()`. Rank-deduplicated pull
beats both Phase-1 brackets in both modes, but it does not beat sealed-wave
direct ingress: direct is 12.5% faster in eager mode and 1.9% faster under
captured replay.

### Decision

Exact receiver reconstruction is **REPRODUCED**. The stronger investment
hypothesis—that rank-deduplicated receiver pull is a better production
direction than the existing direct-ingress path—is **NOT REPRODUCED** on the
preregistered EP2 discriminator. The experiment does reveal a real mechanism:
source-token deduplication can remove enough payload/control work to beat the
legacy pull path, especially for graph-shaped sparse activity. But the naive
receiver must rescan route metadata per expert, and direct placement remains
equal or better on the production-like M=236 route.

Do not widen or productionize this prototype. Retain the branch and artifact as
a negative design discriminator. If this idea is revisited, the new question
must be narrower—whether routing can emit a destination-local compact token
index once, avoiding both route-expanded hidden payload and per-expert metadata
rescans—rather than polishing this receiver scan.
