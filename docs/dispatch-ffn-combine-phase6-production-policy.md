# Dispatch-FFN-Combine Phase 6 production policy

Status: single-node A2 opt-in policy defined. The BF16 and W8A8 Phase 2
direct-ingress mechanisms have passed their warmed layerwise eager and graph
replay gates. They are profitable mechanisms inside their measured selector
envelopes. This does not yet make either compile-time feature a topology-blind
global default.

## Separate performance acceptance from deployment scope

Mechanism performance is decided at the layer boundary. The required evidence
is a production-shaped, warmed distributed eager benchmark plus repeated graph
replay, with the slower rank treated as the critical path. The BF16 and W8A8
Phase 2 candidates satisfy that rule.

Fresh-server end-to-end runs on a shared host serve a different purpose. They
prove real-model selectability, semantic completion, warmup, and absence of a
large regression. They do not overturn a low-single-digit layerwise win when
the enclosing baseline drifts by more than the claimed effect.

Deployment scope is an operability decision. The feature guards are compile
time, while the kernel has no proven host-level predicate that restricts the
new protocol to one node. Single-node EP2, EP4, and EP8 are validated;
multi-node EP is not. Therefore the current production boundary is a named
single-node A2 candidate package, not an unconditional repository default.

## Mechanisms admitted to the profile

Only the Phase 2 sealed-wave direct-ingress implementations are admitted.
Phase 3 per-expert readiness and Phase 4 egress-handshake removal were correct
EP2 experiments but failed the joint eager-plus-graph promotion gate. Phase 5
fragment pipelining was consequently not promoted. Their experimental macros
must not appear in a production build.

### BF16 selector

The BF16 candidate is enabled by both
`DISPATCH_FFN_COMBINE_DIRECT_INGRESS` and
`DISPATCH_FFN_COMBINE_DIRECT_INGRESS_SPARSE_FALLBACK`.

| Execution shape | Existing pull ingress | Direct ingress |
|---|---|---|
| exact shape | disabled | all measured M |
| graph mask | active tokens `<= 1` | active tokens `>= 2` |
| unsafe direct-buffer layout | fallback | disabled |

The graph decision is uniform across ranks: each source publishes one
preference in the already exchanged count row and every rank takes an
all-source AND. It adds no communication round and no dense route-matrix scan.

### W8A8 selector

The corresponding W8A8 branch is enabled by
`DISPATCH_FFN_COMBINE_W8A8_DIRECT_INGRESS` and
`DISPATCH_FFN_COMBINE_W8A8_DIRECT_INGRESS_SPARSE_FALLBACK`.

| Execution shape | Existing gather | Direct ingress |
|---|---|---|
| EP2/EP4 exact | `M <= 2` | `M >= 3` |
| EP2/EP4 graph mask | active tokens `<= 1` | active tokens `>= 2` |
| EP8 exact | `M <= 7` | `M >= 8` |
| EP8 graph mask | all measured shapes | disabled |

W8A8's asymmetric policy is intentional. Its existing gather overlaps expert
ingress with GMM1, while the direct path seals a complete wave; EP8 graph
replay did not amortize the larger fan-out.

## Reproducible opt-in build profile

The repository already exposes the necessary build surface through
`--ops-compile-options`; no second configuration mechanism is justified. From
the repository root, the BF16 candidate package is built with:

```bash
bash csrc/build.sh \
  --ops=dispatch_ffn_combine_bf16 \
  --soc=ascend910b \
  --vendor_name=custom_transformer \
  --pkg \
  --ops-compile-options \
  '-UDISPATCH_FFN_COMBINE_PROFILE;-DDISPATCH_FFN_COMBINE_DIRECT_INGRESS;-DDISPATCH_FFN_COMBINE_DIRECT_INGRESS_SPARSE_FALLBACK'
```

On the W8A8 direct-ingress branch, substitute the W8A8 operator and guards:

```bash
bash csrc/build.sh \
  --ops=dispatch_ffn_combine \
  --soc=ascend910b \
  --vendor_name=custom_transformer \
  --pkg \
  --ops-compile-options \
  '-DDISPATCH_FFN_COMBINE_W8A8_DIRECT_INGRESS;-DDISPATCH_FFN_COMBINE_W8A8_DIRECT_INGRESS_SPARSE_FALLBACK'
```

The build directory is not a source of truth. Incremental CMake and generated
operator files can retain an earlier macro set. A release receipt must record:

1. source commit and dirty-worktree state;
2. the complete `OPS_COMPILE_OPTIONS` value;
3. the generated `custom_compile_options.ini` row;
4. SHA-256 hashes for every installed object variant; and
5. the exact destination package and installation timestamp.

A candidate package must be produced from a clean operator build or an
isolated output directory. Reusing an unverified cache is a release failure.

### BF16 object receipt, 2026-08-06

The first packaging gate is complete for BF16. An isolated four-variant `opc`
compile used repository checkpoint `598f1ecfb`, whose tracked operator source
is unchanged from kernel commit `161ca3bd2` and shared epoch-helper commit
`54d5478ce`. All tracked files were clean; the unrelated untracked
`extra-info/` directory was excluded from the build.

The generated option row was:

```text
ALL,,-UDISPATCH_FFN_COMBINE_PROFILE;-DDISPATCH_FFN_COMBINE_DIRECT_INGRESS;-DDISPATCH_FFN_COMBINE_DIRECT_INGRESS_SPARSE_FALLBACK;-Wno-ignored-attributes;-Wno-ignored-attributes
```

The generated kernel source and shared epoch helper were byte-identical to the
tracked files. Every object was 605,560 bytes:

| Variant suffix | Object SHA-256 | JSON SHA-256 |
|---|---|---|
| `2cdb81c6f496f276126540d98f0dc828` | `57245716904cf1708a65abd2f885afb497429c72bcd47119714c710da96c4c53` | `3c2dbf3c53f55966125a4142976ab28b3eda49209ab287612f81733933c1bac5` |
| `6f342c7338f87a7ad09f5a9dd3c8d8fd` | `f38dbeccc5a9594c89db7f94e97834cc6773757aed5ea30f31f53f126380f754` | `44bad2e8de6f8b4d52eccdc25002f9e720a9d8866284407f660b35fde99efa4b` |
| `8506bed211987078143317c8ace3482e` | `01c8066cb0b573844a338b87772b5960709a316c327fb7ea63cefb77f0196834` | `643d07e8d48261e853e38abf1822b4dd8ae20a8d0f26b1cb1b36d664cf924bdc` |
| `dc184900a0beaeecc753318ebc545c55` | `c26f3db0e533431dac8add91c62661719e37a8848b508d733f1ae4ebc3eeaab3` | `57e78fca020f38980874ebc8b69fb0f7a8ada98ca239607e79be005d946d2457` |

All eight files byte-match the installed Phase 2 candidate that supplied the
Phase 3 and Phase 4 baseline. The prior EP2, EP4, EP8, long-generation,
fail-fast, eager, and graph evidence therefore transfers without a new noisy
performance run. Packaging still needs an installer-level receipt on the final
common integration commit.

## Rollout and rollback gates

The smallest useful rollout is deliberately narrower than another noisy
end-to-end performance campaign:

1. build all registered variants with the admitted macro pair and verify the
   generated options and object manifest;
2. run the existing EP2, EP4, and EP8 correctness suites, including changing
   generations, empty ranks, graph padding, and fail-fast skew;
3. rerun the selector-boundary eager and graph cases after package
   installation, with normal workload warmup;
4. run a real-model semantic and short stability smoke using the matching
   dtype path; and
5. canary only on a declared single-node A2 service pool.

Rollback is package-level and does not require a protocol recovery path:
drain the canary, reinstall the recorded macro-off package, verify the restored
object hashes, and repeat the semantic smoke. An impossible epoch skew remains
a fail-fast upstream-service error.

An unknown or multi-node topology uses the macro-off package. A global default
requires either multi-node correctness and performance evidence or a proven
host-side topology gate that selects the candidate package or kernel without
allowing peers to choose different protocols.

## Remaining engineering boundary

The next work is packaging and canary evidence, not another communication
mechanism. BF16 and W8A8 currently live on separately rebased development
branches, so their final integration must preserve both selector tables and
repeat the clean-build receipt on the common target commit. Branch integration
must not silently turn either dtype's selector into the other's policy.
