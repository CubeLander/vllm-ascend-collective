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
