# Dispatch-FFN-Combine Phase 6 production policy

Status: single-node A2 opt-in policy, full production-opset BF16 plus W8A8
package receipt, process-local EP2/EP4/EP8 canary, and warmed real-model W8A8
smoke complete. The Phase 2 direct-ingress mechanisms have passed their warmed
layerwise eager and graph replay gates. They are profitable mechanisms inside
their measured selector envelopes. This does not yet make either compile-time
feature a topology-blind global default.

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
  --vendor_name=custom \
  --pkg \
  --ops-compile-options \
  '-UDISPATCH_FFN_COMBINE_PROFILE;-DDISPATCH_FFN_COMBINE_DIRECT_INGRESS;-DDISPATCH_FFN_COMBINE_DIRECT_INGRESS_SPARSE_FALLBACK'
```

On the W8A8 direct-ingress branch, substitute the W8A8 operator and guards:

```bash
bash csrc/build.sh \
  --ops=dispatch_ffn_combine \
  --soc=ascend910b \
  --vendor_name=custom \
  --pkg \
  --ops-compile-options \
  '-DDISPATCH_FFN_COMBINE_W8A8_DIRECT_INGRESS;-DDISPATCH_FFN_COMBINE_W8A8_DIRECT_INGRESS_SPARSE_FALLBACK'
```

For object-receipt and process-local testing, a narrow common package compiles
both operators and both selector pairs in one clean package:

```bash
bash csrc/build.sh \
  --ops=dispatch_ffn_combine_bf16,dispatch_ffn_combine \
  --soc=ascend910b \
  --vendor_name=custom \
  --pkg \
  --ops-compile-options \
  '-UDISPATCH_FFN_COMBINE_PROFILE;-DDISPATCH_FFN_COMBINE_DIRECT_INGRESS;-DDISPATCH_FFN_COMBINE_DIRECT_INGRESS_SPARSE_FALLBACK;-DDISPATCH_FFN_COMBINE_W8A8_DIRECT_INGRESS;-DDISPATCH_FFN_COMBINE_W8A8_DIRECT_INGRESS_SPARSE_FALLBACK'
```

That two-operator package is deliberately **not** a production overlay. The
installer replaces the vendor's complete `op_proto`, `op_impl`, and `op_api`
trees rather than merging individual operators. Production packaging must
therefore compile the complete operator composition of the runtime it will
replace. The 2026-08-06 A2 receipt used this exact composition:

```bash
PRODUCTION_OPS='add_rms_norm_bias,apply_top_k_top_p_custom,causal_conv1d,chunk_fwd_o,chunk_gated_delta_rule_fwd_h,compressor,compressor_metadata,copy_and_expand_eagle_inputs,dequant_swiglu_quant,dispatch_ffn_combine,dispatch_ffn_combine_bf16,fused_gdn_gating,grouped_matmul_swiglu_quant,grouped_matmul_swiglu_quant_v2,grouped_matmul_swiglu_quant_weight_nz_tensor_list,hamming_dist_top_k,hc_post,hc_pre,hc_pre_inv_rms,hc_pre_sinkhorn,inplace_partial_rotary_mul,lightning_indexer,lightning_indexer_quant,matmul_allreduce_add_rmsnorm,moe_gating_top_k,moe_gating_top_k_hash,moe_grouped_matmul,moe_init_routing_custom,ngram_spec_decode,recurrent_gated_delta_rule,reshape_and_cache_bnsd,rms_norm_dynamic_quant,scatter_nd_update_v2,sparse_attn_sharedkv,sparse_flash_attention,store_kv_block,transpose_kv_cache_by_block,vllm_quant_lightning_indexer,kv_quant_sparse_attn_sharedkv_metadata,sparse_attn_sharedkv_metadata,vllm_quant_lightning_indexer_metadata'

bash csrc/build.sh \
  --ops="$PRODUCTION_OPS" \
  --soc=ascend910b \
  --vendor_name=custom \
  --pkg \
  --ops-compile-options \
  '-UDISPATCH_FFN_COMBINE_PROFILE;-DDISPATCH_FFN_COMBINE_DIRECT_INGRESS;-DDISPATCH_FFN_COMBINE_DIRECT_INGRESS_SPARSE_FALLBACK;-DDISPATCH_FFN_COMBINE_W8A8_DIRECT_INGRESS;-DDISPATCH_FFN_COMBINE_W8A8_DIRECT_INGRESS_SPARSE_FALLBACK'
```

The three trailing metadata operators supply AICPU contents but no AICore
kernel directory. `--ops=ALL` is not an equivalent release recipe in this
checkpoint: it links both `lightning_indexer` implementations and fails on
duplicate definitions. The explicit list is the measured runtime composition,
not a workaround that silently drops installed operators.

The `custom` vendor argument is intentional: this build system appends
`_transformer`, so it installs the expected internal vendor
`custom_transformer`. Passing `custom_transformer` would incorrectly produce
`custom_transformer_transformer`.

The build directory is not a source of truth. Incremental CMake and generated
operator files can retain an earlier macro set. A release receipt must record:

1. source commit and dirty-worktree state;
2. the complete `OPS_COMPILE_OPTIONS` value;
3. the generated `custom_compile_options.ini` row;
4. SHA-256 hashes for every installed object variant;
5. the imported `vllm_ascend_C` path and SHA-256 hash; and
6. the exact destination package and installation timestamp.

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
performance run.

### Narrow common integration object receipt, 2026-08-06

The BF16 and W8A8 branches were combined at source checkpoint `3cf1379ef` on
`agent/moe-direct-ingress-integration`. The tracked operator sources were
clean at build start and exactly matched their accepted dtype branches. The
package used the combined command above, and its generated option row was:

```text
ALL,,-UDISPATCH_FFN_COMBINE_PROFILE;-DDISPATCH_FFN_COMBINE_DIRECT_INGRESS;-DDISPATCH_FFN_COMBINE_DIRECT_INGRESS_SPARSE_FALLBACK;-DDISPATCH_FFN_COMBINE_W8A8_DIRECT_INGRESS;-DDISPATCH_FFN_COMBINE_W8A8_DIRECT_INGRESS_SPARSE_FALLBACK;-Wno-ignored-attributes
```

The package is named `cann-ops-transformer-custom_linux-aarch64.run`, installs
under internal vendor `custom_transformer`, and has SHA-256
`fe8418efbcf850f9b237d1dda6dd6ecb366a6b669649d1b3792dccfa82cffa2d`.
The generated kernel and shared epoch-helper sources for both dtypes were
byte-identical to the tracked files. All seven packaged objects byte-match the
previous isolated receipts:

| Operator and variant suffix | Object SHA-256 |
|---|---|
| W8A8 `9907fbc1444e58de3ad65d22316d7b8f` | `d0bc902eb004559e92d36cc1d83f998cd14175ae92b5ac703c093210b65ca49a` |
| W8A8 `af7cdba33254528b52a5326f36d262f5` | `4e531dbfbf4fe689665cdd4612d0bf485f9041b0d7638350891faff1d64c8b89` |
| W8A8 `b12576f77f0efc7337f4f4527a4d909f` | `4279e7fc78ba84f99021c5b1ddc3b7db67fdaec9636add96747c75f251c5e2df` |
| BF16 `2cdb81c6f496f276126540d98f0dc828` | `57245716904cf1708a65abd2f885afb497429c72bcd47119714c710da96c4c53` |
| BF16 `6f342c7338f87a7ad09f5a9dd3c8d8fd` | `f38dbeccc5a9594c89db7f94e97834cc6773757aed5ea30f31f53f126380f754` |
| BF16 `8506bed211987078143317c8ace3482e` | `01c8066cb0b573844a338b87772b5960709a316c327fb7ea63cefb77f0196834` |
| BF16 `dc184900a0beaeecc753318ebc545c55` | `c26f3db0e533431dac8add91c62661719e37a8848b508d733f1ae4ebc3eeaab3` |

After removing the package-only `filePath` field, all seven JSON manifests
also match the accepted dtype receipts. This machine required a user-readable
empty OPP vendor-registry overlay because the system
`opp/vendors/config.ini` is unreadable; that is a host permission condition,
not a source or package deviation. A quiet installation to an isolated,
explicit prefix succeeded, created only vendor `custom_transformer`, and
retained all seven object hashes. The shared CANN OPP tree was not modified.

Installing the same package over an isolated copy of the complete runtime
proved why this artifact is receipt-only: the copy fell from 898 files and
183 AICore objects to 127 files and seven objects, while `libcust_opapi.so`
fell from 938,664 to 69,152 bytes. A narrow package must never be promoted as
an overlay, even when both target operators pass in an empty prefix.

### Full production-opset package receipt, 2026-08-06

The explicit production composition above was built from integration
checkpoint `911c4dedf` plus the tracked packaging correction in
`matmul_allreduce_add_rmsnorm`: its compile-option key now names the actual
operator, and its kernel receives the repository CATLASS include path. Without
that correction, the complete package fails while compiling
`catlass/catlass.hpp`; an incremental generated-options cache can conceal the
bad key. The accepted integration artifact therefore verifies the regenerated
option row and installed inventory rather than trusting the cache; final
publication still requires a clean output directory.

The resulting `cann-ops-transformer-custom_linux-aarch64.run` has SHA-256
`7e705ac1fba659c79c111a2e3ad03af16dbc9a623a791b44ef983e5c53a8f424`.
Its isolated installation contains 899 files, 183 AICore objects, 38 AICore
kernel directories, and a 938,664-byte `libcust_opapi.so`. The 38 directory
names exactly equal the pre-existing production runtime composition. All seven
BF16 and W8A8 dispatch objects have the hashes in the narrow receipt above.
The full package therefore preserves production operator coverage while
carrying the independently accepted dispatch kernels.

The build also required a user-readable OPP registry overlay because the
system `opp/vendors/config.ini` is unreadable. Neither the shared runtime nor
the system CANN tree was modified. Stale empty `kv_cache_block_gather` and
`lightning_indexer_vllm` directories left by the rejected `ALL` configuration
were removed from the isolated build output before the final manifest was
generated; neither name appears in the accepted composition.

### Process-local and real-model canary, 2026-08-06

The isolated installation was activated through its generated environment
script, without modifying the shared CANN OPP tree. The first run exposed a
separate release invariant: this source worktree had no local
`vllm_ascend_C`, so Python silently loaded the stale machine fallback from
`/vllm-workspace/vllm-ascend`. W8A8 EP2 and EP4 happened to pass, while BF16
EP2 and EP4 terminated with `SIGSEGV`. Replacing the operator package and its
OPAPI library with the known-good BF16-only versions did not change that
failure, distinguishing the extension fallback from an operator-package
defect.

The canary was repeated with the W8A8 worktree extension whose torch-binding
sources have zero diff from the common integration branch. Its
`vllm_ascend_C` SHA-256 is
`4814a008bf509f5f768ca9a54ca10c90ba7da847ec6f12032c4807df22ea21c5`.
With that matching extension and the full production-opset package:

- the tracked BF16 and W8A8 EP2 plus EP4 suites passed, four tests across two
  invocations;
- the BF16 EP8 pilot passed 56 changing generations across empty, sparse,
  local-only, peer-only, one-destination, and mixed waves;
- the W8A8 EP8 changing-route regression passed; and
- all eight devices returned idle after each campaign.

The W8A8 real-model canary used runtime checkpoint `c6a6fbdb2`, which combines
the fused-MC2 selector with the separately verified packed compressed-KV fixes,
plus the matching extension above and the full package. After model and service
warmup, a deterministic completion succeeded and an eight-request 64-input by
16-output smoke completed all requests. The smoke reported 12.32 output
tokens/s, 389.06 ms mean TPOT, and 0.770 requests/s. Those values are semantic
and large-regression evidence only; shared-host end-to-end throughput remains
outside the mechanism acceptance gate.

The deployable unit is therefore the matching Python extension or wheel, its
declared runtime dependencies, and the full operator package. A release must
refuse a missing or unrecorded local extension rather than falling through to
an unrelated machine copy. Publishing that matched wheel/package bundle is
still required before service rollout; the semantic canary itself is complete.

## Rollout and rollback gates

The smallest useful rollout is deliberately narrower than another noisy
end-to-end performance campaign:

1. build the complete declared runtime operator composition with the admitted
   macro pairs and verify the generated options, inventory, and object
   manifest;
2. run the existing EP2, EP4, and EP8 correctness suites, including changing
   generations, empty ranks, graph padding, and fail-fast skew;
3. rerun the selector-boundary eager and graph cases after package
   installation, with normal workload warmup;
4. run a real-model semantic and short stability smoke using the matching
   dtype path and declared runtime dependency set; and
5. canary only on a declared single-node A2 service pool.

Rollback is deployable-unit-level and does not require a protocol recovery
path: drain the canary, restore the recorded macro-off package and matching
extension, verify both sets of hashes, and repeat the semantic smoke. An
impossible epoch skew remains a fail-fast upstream-service error.

An unknown or multi-node topology uses the macro-off package. A global default
requires either multi-node correctness and performance evidence or a proven
host-side topology gate that selects the candidate package or kernel without
allowing peers to choose different protocols.

## Remaining engineering boundary

The next work is publication of a matched wheel/full-package bundle and a
bounded service-pool canary, not another communication mechanism. The common
integration branch preserves both dtype selector tables, its full package
objects are identical to their independently accepted receipts, and
process-local EP2/EP4/EP8 plus real-model package activation are verified.
Runtime installation must not silently turn either dtype's selector into the
other's policy, omit unrelated production operators, or load an unrelated
extension. The existing macro-off full package must remain the recorded
rollback target.
