# MoE FFN direct-ingress handoff

Updated: 2026-08-05 UTC

Repository: `CubeLander/vllm-ascend-hust`

Published handoff branch: `agent/moe-ffn-handoff-20260805`

Active continuation branch: `agent/moe-gmm2-cutthrough-revival`

Latest kernel checkpoint: `161ca3bd2` (not yet published)

Status: experimental and compile-time guarded. The production default is
unchanged. Hardware revival on a healthy Ascend 910B host has now validated the
zero-scan sparse fallback through EP8, explicit payload ordering, and 512
global experts on one node.

## State in one page

This line of work targets `DispatchFFNCombineBF16`. The original receive path
pulled each remote expert fragment after a per-expert readiness protocol. The
current prototype reuses the already exchanged dense route-count matrix to
derive exact destination offsets and lets each source write directly into the
receiver's final expert-major input. A dedicated source-owned ingress epoch
seals each source wave. Reply transmission is deliberately still streaming:
GMM2 sends each completed source fragment without waiting to batch the whole
reply.

The sealed-wave direct-ingress mechanism has passed tracked EP2 and EP4
end-to-end tests and 2,048-generation exact-output stress runs at both EP
sizes. Its bracketed critical-path timing is positive in six of seven original
route families. The zero-scan selector now addresses the graph-padded window
with one active token, where the fixed epoch protocol was not reliably
amortized.

A zero-scan selector now keeps that tiny route on the legacy pull path. It
uses the production prefix-mask contract and one padding lane in each already
exchanged count row, adds no communication round, and reads O(EP) scalars
rather than scanning the dense count matrix. All four registered Ascend 910B
dtype/format variants compile with the selector. EP2 hardware exactness,
bracketed crossover timing, EP4 exactness, and repeated reuse are now evidence;
unsigned epoch wrap is also hardware-validated. Wider production topologies
up to single-node EP8 are now covered.

## Code and evidence map

| Area | Location | Meaning |
|---|---|---|
| Kernel and compile guards | `csrc/mc2/dispatch_ffn_combine_bf16/op_kernel/dispatch_ffn_combine_bf16_kernel.hpp` | Direct placement, epochs, memory ordering, fallback selector |
| Protocol and experimental receipts | `docs/dispatch-ffn-combine-phase2-direct-ingress.md` | Authoritative design, correctness, timing, msopprof, and remaining gates |
| Tracked EP2/EP4 regression | `tests/e2e/nightly/single_node/ops/multicard_ops_a2/test_dispatch_ffn_combine_bf16.py` | Dense and changing routes, graph padding, sparse tensor-list weights |
| Production mask construction | `vllm_ascend/ascend_forward_context.py` | Builds a true-prefix/false-suffix `mc2_mask` |
| Production wrapper contract | `vllm_ascend/ops/fused_moe/token_dispatcher.py` | Passes `mc2_mask` only in uniform-token mode (`global_bs == 0`) |
| Live-source container helper | `tools/docker/source_dev_container.sh` | Mounts this checkout and an exact vLLM source into a task-specific container |

The ignored `.lumi-workbench/` contains local harnesses and raw receipts. It
is intentionally not in Git and therefore does **not** travel with this
branch. Its useful script names and behavior are recorded below so a new local
workbench can recreate only what the next experiment needs:

- `profile_dispatch_ffn_combine_bf16.py`: two-rank exact-output and host/device
  timing matrix; supports repeated named cases and bracketed A/B runs;
- `stress_direct_ingress_generations.py`: one communicator and peer window for
  2,048 changing generations over active counts `0, 1, 2, 7, 16, 31, 64`;
- `msopprof_dispatch_ffn_combine_bf16.py` plus its shell wrapper: safe
  application replay with one instrumented rank and one fresh uninstrumented
  peer; and
- `attribute_msopprof_fdata.py`: joins Source-mode `fdata`, `bbbmap`, and the
  matching DWARF object. Its output is control-flow visit attribution, not a
  cycle profile.

## Commit spine

Read these in chronological order when recovering intent:

| Commit | Contribution |
|---|---|
| `1dd0e6547` | Enable fused BF16 MC2 on Ascend 910B |
| `9f8f24276` | Make fused graph-capture shapes safe |
| `4e49ba921` | Mask padded fused-expert rows |
| `82eef1a16` | Avoid resetting the fused count matrix |
| `4a49efe3f` | Record prior-art study of sparse communication protocols |
| `3ce576609` | Clarify HCCL completion semantics |
| `a2f832bd9` | Skip empty ingress barriers |
| `3257f1100` | Bypass empty expert schedules |
| `0d5a82aff` | Compact expert-progress signals |
| `4c3ca56a0`, `56c083a3f` | Cover sparse tensor-list and late expert routes |
| `2764fb690` | Gate the next optimization work on evidence |
| `dc85a56dd` | Implement sealed-wave direct ingress |
| `9f32f32b0` | Record correctness and first bracketed timing evidence |
| `d26df5379` | Retain the DCCI ablation behind an explicit macro |
| `c16592cc8` | Normalize bare image IDs in the source-container helper |
| `3e5cd5b0e` | Record 2,048-generation stress and source attribution |
| `5cd9c9fe4` | Add the zero-scan sparse fallback selector |
| `8a7adf477` | Add tracked EP4 direct-ingress validation |
| `3ecb4e684` | Make MC2 epoch arithmetic explicitly wrap-safe |
| `592187815` | Record repeated EP8 direct-ingress validation |
| `161ca3bd2` | Replace payload DCCI with explicit memory barriers |

## Protocol and invariants

The direct ingress is a sealed source wave:

```text
route locally into source-owned staging
  -> publish and gather the existing count matrix
  -> each source derives exact source and destination prefixes
  -> each source pushes nonempty fragments to remote final input
  -> every producer core completes MTE3 payload writes, then local cores join
  -> one source-owned ingress epoch publishes and completes the source wave
  -> each destination waits for every source epoch
  -> order the completed epoch reads before MTE2/Cube consumption
  -> publish compact AIV-to-AIC expert progress
```

The following are correctness constraints rather than tuning preferences:

1. Destination rows are expert-major. Within one expert, lower source ranks
   own earlier rows; across experts, lower local-expert indices own earlier
   regions. These prefixes make all remote writes disjoint.
2. The ingress epoch has cache-line-aligned storage separate from both the
   count matrix and egress completion state. Reusing egress synchronization
   caused a one-generation phase alias and stale return data on an empty
   receiver.
3. A waiter accepts the requested epoch or one generation ahead, matching the
   existing bounded-skew rule. Epochs are explicit unsigned 32-bit modular
   counters; a disposable `UINT32_MAX - 2` seed crossed wrap with exact output.
   One generation behind is the only waitable state. Any other modular
   distance traps because the requested single-buffer payload has been lost.
4. Every producer core executes a DDR data-sync barrier before the local AIV
   join. The publisher completes its scalar epoch store, and the receiver
   executes a DDR data-sync barrier after observing every source epoch. Scalar
   epoch accesses retain DCCI; the DMA payload does not use scalar cache
   maintenance.
5. Buffer layout is chosen before the dynamic selector. On fallback, the
   legacy pull loop writes into the already selected direct input buffer, so
   AIC and egress offsets remain coherent.
6. **Do not batch egress.** The reply side intentionally remains streaming to
   avoid expert-completion tail latency. Direct ingress seals only the request
   wave.

## Zero-scan sparse fallback

Compile guard:

```text
DISPATCH_FFN_COMBINE_DIRECT_INGRESS_SPARSE_FALLBACK
```

It requires:

```text
DISPATCH_FFN_COMBINE_DIRECT_INGRESS
```

The selector depends on a concrete production contract:

- `ascend_forward_context.py` writes `mc2_mask[:num_actual_tokens] = True` and
  the remaining captured rows `False`;
- `token_dispatcher.py` supplies that mask only when `global_bs == 0`; and
- therefore, for graph capacity greater than one, `mask[1]` separates local
  cardinality 0/1 from cardinality at least 2 without scanning routes.

Each source publishes this local preference in the first padding lane after
its real expert counts. The existing tagged count-row exchange carries it to
all ranks. Every rank ANDs one preference per source and chooses direct ingress
only if all sources prefer it. This prevents protocol disagreement even if a
lower-level caller provides different masks on different ranks.

A discarded selector scanned the full count matrix on one core. It added
roughly 50 us of rank skew and harmed the otherwise positive families. Do not
reintroduce a serialized O(EP² × local-experts) scan or a new selection round.

## Evidence so far

### Correctness

- The ordinary tracked EP2 test passed after separating the ingress epoch and
  again after narrowing DCCI to rows actually received: `1 passed, 16 warnings
  in 36.90s`.
- One EP2 communicator and peer window survived 2,048 consecutive generations
  with exact output and expert-count oracles. It rotated through all-empty,
  one-destination, local-only, peer-only, and mixed routes.
- One EP4 communicator and peer window survived the same 2,048-generation
  matrix after sizing the test capacity for the 1,024-row one-destination
  maximum. A tracked EP4 regression also passed with 64 local experts per rank
  (256 global experts).
- The selector passed EP2 exactness and five fresh-process S-D-S-D-S crossover
  runs. At active=1 its median was 291.02 us versus 304.22 us direct-only,
  4.34% faster but narrowly below the preregistered 5% useful-effect target.
  At active=16 it was 0.53% faster, inside the 5% parity bound.
- A byte-identical unsigned-counter rebuild passed tracked EP2+EP4 again. A
  disposable ingress seed at `UINT32_MAX - 2` crossed wrap during 14 changing
  EP2 generations with exact output and expert counts; the hook was removed.
- A 56-generation pilot and 2,048-generation run passed on EP8 with exact
  output/counts, all eight devices, and the one-destination hotspot rotated
  across every rank. The explicit-DSB candidate also passed 2,048 EP8
  generations with 64 experts per rank (512 global experts). Multi-node EP
  remains open.

### Bracketed critical-path timing

The slower rank's median device event is the distributed critical-path
estimate. Each direct run had a Phase 1 compact-signal run before and after it,
50 samples per case.

| Case | Phase 1 before | Direct ingress | Phase 1 after | Judgment |
|---|---:|---:|---:|---|
| decode M=8, spread | 467.75 us | 369.40 us | 481.47 us | 21.0--23.3% faster |
| prefill M=64, spread | 480.67 us | 396.90 us | 499.60 us | 17.4--20.6% faster |
| prefill M=256, spread | 605.42 us | 469.19 us | 565.51 us | 17.0--22.5% faster |
| decode M=8, four experts | 351.35 us | 278.22 us | 311.10 us | 10.6--20.8% faster |
| prefill M=64, four experts | 322.71 us | 283.34 us | 327.79 us | 12.2--13.6% faster |
| graph M=64, active=1 | 332.76 us | 325.50 us | 294.17 us | inconclusive; -2.2% to +10.7% |
| graph M=64, active=16 | 508.84 us | 416.03 us | 510.90 us | 18.2--18.6% faster |

The host medians agree for the six positive cases, improving 2.7--19.4%. For
graph `active=1`, host medians regress 3.5--14.4%. This is why a tiny-wave
fallback is evidence-driven rather than speculative complexity.

### Visibility barrier closure

The explicit `DISPATCH_FFN_COMBINE_DIRECT_INGRESS_SKIP_DCCI` ablation passed
the tracked EP2+EP4 tests, a 56-generation EP8 pilot, and 2,048 changing EP8
generations. It is directionally positive for larger routes, but graph
`active=1` remains inconclusive:

| Case | DCCI before | No DCCI | DCCI after | Judgment |
|---|---:|---:|---:|---|
| decode M=8, spread | 428.57 us | 410.65 us | 441.16 us | 4.2--6.9% faster |
| prefill M=256, spread | 532.41 us | 526.40 us | 547.83 us | 1.1--3.9% faster |
| graph M=64, active=1 | 340.28 us | 352.15 us | 353.67 us | inconclusive |

CANN 9.0.X documents that DMA access to GM has no DataCache consistency issue,
that scalar reads after external modification require DCCI, that A2 `DataCopy`
supports inter-device transfer, and that `DataSyncBarrier` waits for preceding
GM/memory-access instructions. The tracked candidate therefore replaces the
byte-scaled payload DCCI with an explicit sequence: per-producer DDR DSB, local
AIV join, completed scalar epoch publication, all-source epoch observation,
then receiver DDR DSB before MTE2/Cube consumption. Scalar epoch accesses keep
DCCI.

The clean tracked-source build was byte-identical to the disposable DSB
prototype. It passed the tracked EP2+EP4 suite (`2 passed, 21 warnings in
72.33s`), a 56-generation EP8 pilot, and 2,048 changing EP8 generations with
exact output and expert-count oracles. A focused graph `active=16` S-DSB-S
bracket with 50 warmups and 200 samples measured 389.26 us for DSB versus
402.52 and 457.54 us for payload DCCI, a 3.3--14.9% improvement. `active=1`
falls back before this path and is not discriminating. The detailed protocol
note links the authoritative API pages and records the sequence. A separate
2,048-generation EP8 run with 64 experts per rank (512 global experts) passed
all route families and exact oracles, closing the previous above-256-expert gap
on a single node.

### msopprof mechanism evidence

The Phase 1 operator is control/wait dominated, not payload-bandwidth bound:

- 24 Cube kernels: 65.12 us median, including 27.95 us on wait id 0 and
  6.99 us on wait id 10;
- 48 Vector kernels: 91.13 us median, including 38.72 us on wait id 14;
- Vector utilization 0.65%, MTE2 4.49%, MTE3 1.48%; and
- measured GM/L1/UB bandwidth ratios below 0.1.

Safe MC2 profiling uses `msopprof --replay-mode=application`, instruments one
rank, and creates a fresh uninstrumented peer for every replay. Kernel or range
replay is unsafe because the remote participant would be absent.

A disposable direct-`opc` debug object produced 36,108 PC/source relations and
375,915 address-to-line relations. Joining `fdata` with its basic-block map
attributed 650 nonzero blocks and 95,832 visits. Those are control-flow visits,
not per-line cycles. The debug `.text` differs from release; it is valid for
source attribution, never comparative timing. The conservative release object
was restored after collection.

Public progress receipts:

- HUST PR 199: <https://github.com/vLLM-HUST/vllm-ascend-hust/pull/199#issuecomment-5189677655>
- Upstream RFC 6434: <https://github.com/vllm-project/vllm-ascend/issues/6434#issuecomment-5189678017>

## Clean build on the next machine

First fetch the published state without depending on the divergent historical
remote branch:

```bash
git fetch origin agent/moe-ffn-handoff-20260805
git switch -c agent/moe-gmm2-cutthrough-revival FETCH_HEAD
git log -1 --oneline
```

Read the workspace and repository `AGENTS.md` files. Inspect `npu-smi info`,
choose exactly two genuinely idle physical devices, and create a uniquely
named live-source container:

```bash
NPU_DEVICES=<id0,id1> \
CONTAINER_NAME=moe-direct-ingress-selector \
bash tools/docker/source_dev_container.sh start

NPU_DEVICES=<id0,id1> \
CONTAINER_NAME=moe-direct-ingress-selector \
bash tools/docker/source_dev_container.sh verify

NPU_DEVICES=<id0,id1> \
CONTAINER_NAME=moe-direct-ingress-selector \
bash tools/docker/source_dev_container.sh verify-npu
```

Inside the container, confirm both packages resolve from `/workspace`, then
build the selector for Ascend 910B:

```bash
bash csrc/build.sh --pkg --soc=ascend910b \
  --ops=dispatch_ffn_combine_bf16 -j8 \
  --ops-compile-options \
  '-UDISPATCH_FFN_COMBINE_PROFILE;-DDISPATCH_FFN_COMBINE_DIRECT_INGRESS;-DDISPATCH_FFN_COMBINE_DIRECT_INGRESS_SPARSE_FALLBACK;-Wno-ignored-attributes'
```

Prefer a fresh `csrc/build` from a clean checkout for compile-option
experiments. An incremental build previously refreshed
`csrc/build/impl/dynamic/dispatch_ffn_combine_bf16.py` but left the per-op
driver below stale with an obsolete DCCI-ablation macro:

```text
csrc/build/binary/ascend910b/src/dispatch_ffn_combine_bf16/DispatchFFNCombineBF16.py
```

Before trusting the package, inspect that generated driver and verify the
selector macro is present and the obsolete ablation macro is absent:

```bash
rg 'DIRECT_INGRESS|SKIP_DCCI' \
  csrc/build/binary/ascend910b/src/dispatch_ffn_combine_bf16/DispatchFFNCombineBF16.py
```

The known-good explicit-DSB selector objects are 605,560 bytes. Their hashes
are:

```text
57245716904cf1708a65abd2f885afb497429c72bcd47119714c710da96c4c53
f38dbeccc5a9594c89db7f94e97834cc6773757aed5ea30f31f53f126380f754
01c8066cb0b573844a338b87772b5960709a316c327fb7ea63cefb77f0196834
c26f3db0e533431dac8add91c62661719e37a8848b508d733f1ae4ebc3eeaab3
```

Do not treat matching hashes as hardware validation; they only help detect a
stale compile-option cache. Back up the installed release op and JSON before
installing an experimental package, and restore plus checksum-verify them
after every experiment.

## Recovery experiment order

Use this order so a failure has a small search space:

1. Run the tracked EP2 test against the selector package:

   ```bash
   pytest -sv \
     tests/e2e/nightly/single_node/ops/multicard_ops_a2/test_dispatch_ffn_combine_bf16.py
   ```

2. Recreate the local 2,048-generation stress harness from the behavior above
   and run it with one communicator/window. It must cover empty destinations,
   local-only, peer-only, one-destination, and mixed routes with exact output
   and expert-count oracles.
3. Use the timing harness in `--mode host`, 50--100 samples, and bracket the
   selector with the existing direct-ingress build in both orders. The first
   discriminating cases are `graph-64-active-1` and
   `graph-64-active-16`. Compare the slower rank's device-event median and
   retain host medians as a second view.
4. The expected selector behavior is legacy pull for active 0/1 and direct
   ingress for active at least 2, with identical protocol choice on every
   rank. Add temporary diagnostics only if the performance result cannot
   distinguish the branch.
5. Only after EP2 exactness, 2,048-wave reuse, and the crossover timing pass,
   run the same exact-output protocol with `DIRECT_INGRESS_STRESS_WORLD_SIZE=4`
   on four idle NPUs.
6. Restore the release package, stop/remove only this task's container, and
   confirm the physical devices returned to their prior idle state.

Promotion gates remain: multi-node topology coverage and the narrowly missed
5% tiny-wave target. Single-node EP8 with 512 global experts passes, and the
visibility gate is closed by explicit documented memory-completion barriers.
A disposable two-wave skew injection produced an AICore trap (`507015`) and a
nonzero process exit rather than a hang; the restored object then passed EP2.
Larger skew is therefore fail-fast by design, not a recovery gate. The dense
count exchange is still present and can be revisited later, but removing it is
not part of the current closed optimization line.

## Why this host was abandoned

Rootful Podman remained usable for CPU-only package compilation, but every
container command emitted `Peer netns reference is invalid`. Newly created
containers saw zero NPUs, with `DrvMngGetConsoleLogLevel ret4` and DCMI
`-8020`, even with host networking. Physical devices 2 and 3 were idle at the
last check, but not usable from a fresh task container. Existing unrelated
containers were not entered, modified, or removed; only task-owned containers
were cleaned up.

This is an environment blocker, not operator evidence. Resume from the clean
hardware gates above rather than debugging the protocol against this host.

## Boundaries to preserve

- Keep claims observation-backed: six positive original route families, a
  4.34% selector median gain on the tiny graph route, active=16 parity, and
  exact EP2/EP4 reuse. Do not round the tiny-route result up to the 5% target.
- Treat unknown behavior as a signal to investigate, not noise to discard.
- Do not force-push or merge the divergent historical source branch merely to
  publish experiments; the dedicated handoff branch is the transferable
  checkpoint.
- Do not add a full dense selector scan, a new selection handshake, or a
  sealed-batch egress reply.
- Do not promote direct ingress or the selector to the production default
  before the stated gates pass.
- Preserve local workbench hygiene: helpers and raw profiler products remain
  ignored and disposable; tracked code, tests, docs, and this handoff are the
  durable source of truth.
