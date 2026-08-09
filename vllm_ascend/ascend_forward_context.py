import math
from contextlib import contextmanager
from enum import Enum
from typing import Any

import torch
import vllm.envs as envs_vllm
from vllm.config import CUDAGraphMode, VllmConfig
from vllm.distributed import get_dp_group, get_ep_group, get_tensor_model_parallel_world_size
from vllm.forward_context import BatchDescriptor, get_forward_context, set_forward_context
from vllm.logger import logger

from vllm_ascend.ascend_config import MoEPhaseHybridPolicy, get_ascend_config, parse_moe_phase_hybrid_policy
from vllm_ascend.utils import (
    AscendDeviceType,
    enable_sp,
    flashcomm2_enable,
    get_ascend_device_type,
    has_layer_idx,
    is_drafter_moe_model,
    is_moe_model,
    speculative_enable_dispatch_gmm_combine_decode,
)


class MoECommType(Enum):
    ALLGATHER = 0
    MC2 = 1
    ALLTOALL = 2
    FUSED_MC2 = 3


MAX_A2_FUSED_MC2_EP_SIZE = 8
MIN_A2_FUSED_MC2_LOCAL_EXPERTS = 3


class MoEForwardPhase(Enum):
    """Host-computed forward phase used by the phase-keyed hybrid MoE policy.

    The classification is derived from host-side scheduler state
    (``attn_state``/``with_prefill`` in the v1 runner). It never synchronizes
    a device tensor with ``.item()``.

    - ``PURE_DECODE``: every request is decoding (or speculative decode).
    - ``PURE_PREFILL``: every request is a brand-new prefill
      (``PrefillNoCache``: zero computed tokens).
    - ``MIXED``: at least one request is prefilling while others may already
      have computed tokens (``ChunkedPrefill`` / ``PrefillCacheHit``).
    """

    PURE_DECODE = 0
    PURE_PREFILL = 1
    MIXED = 2


def get_moe_phase_hybrid_policy(vllm_config: VllmConfig | None = None) -> MoEPhaseHybridPolicy:
    """Resolve the phase-keyed hybrid MoE policy (additional_config only).

    The authoritative value is ``additional_config.moe_phase_hybrid_policy``
    (see :func:`vllm_ascend.ascend_config.parse_moe_phase_hybrid_policy`);
    there is deliberately no environment-variable fallback. ``AscendConfig``
    validates the same value at startup, so a typo fails closed during
    initialization. ``vllm_config=None`` (legacy callers) falls back to the
    initialized ``AscendConfig`` singleton.
    """
    if vllm_config is not None:
        additional_config = getattr(vllm_config, "additional_config", None) or {}
        return parse_moe_phase_hybrid_policy(additional_config.get("moe_phase_hybrid_policy", "off"))
    return get_ascend_config().moe_phase_hybrid_policy


def classify_forward_phase(attn_state: Any, with_prefill: bool) -> MoEForwardPhase:
    """Classify the host forward phase from the runner's host-side signals.

    The runner computes ``attn_state`` (``AscendAttentionState``) and
    ``with_prefill`` from CPU scheduler buffers; no device tensor sync is
    performed here.
    """
    if not with_prefill:
        return MoEForwardPhase.PURE_DECODE
    # Lazy import avoids a cycle: attention_v1 already imports this module.
    from vllm_ascend.attention.attention_v1 import AscendAttentionState

    if attn_state == AscendAttentionState.PrefillNoCache:
        return MoEForwardPhase.PURE_PREFILL
    return MoEForwardPhase.MIXED


def is_moe_phase_hybrid_active(
    forward_phase: MoEForwardPhase | None,
    vllm_config: VllmConfig,
    is_draft_model: bool = False,
) -> bool:
    """Whether the current forward runs under the fused non-decode path.

    True only when the policy is enabled, the forward is the main (non-draft)
    MoE model, and the phase is ``PURE_PREFILL`` or ``MIXED``. Both non-decode
    phases select ``FUSED_MC2`` (under the SoC gate) and run the fused
    torch.compile artifact outside the ACL graphs; ``PURE_DECODE`` is
    deliberately excluded because it selects the baseline comm family (see
    :func:`select_moe_comm_method`) inside the FULL_DECODE_ONLY ACL graphs.
    """
    return (
        get_moe_phase_hybrid_policy(vllm_config)
        in (MoEPhaseHybridPolicy.NON_DECODE_FUSED, MoEPhaseHybridPolicy.OPAQUE_FUSED_CONTROL)
        and not is_draft_model
        and forward_phase in (MoEForwardPhase.PURE_PREFILL, MoEForwardPhase.MIXED)
        and is_moe_model(vllm_config)
    )


def should_skip_compiled_for_moe_phase_hybrid(
    forward_phase: MoEForwardPhase | None,
    vllm_config: VllmConfig,
    is_draft_model: bool = False,
) -> bool:
    """Whether the forward must bypass the torch.compile/AOT artifact.

    RETIRED under the opaque-MoE canary: always returns ``False``. The outer
    compiled graph contains only opaque ``torch.ops.vllm.moe_forward`` calls
    and the live ``_forward_impl`` dispatch happens at op runtime, so decode
    no longer needs ``skip_compiled=True`` to stay on the raw baseline: every
    phase shares the one compiled outer artifact and the op runtime selects
    the comm family per phase (``PURE_DECODE`` baseline, ``PURE_PREFILL`` /
    ``MIXED`` fused). The flag is deliberately independent from
    :func:`should_force_eager_for_moe_phase_hybrid` and is never fed into
    ``force_eager``; ``skip_compiled`` only disables the compiled model call
    inside the model forward, it does not choose the ACL graph mode.
    """
    return False


def should_force_eager_for_moe_phase_hybrid(
    forward_phase: MoEForwardPhase | None,
    vllm_config: VllmConfig,
    is_draft_model: bool = False,
) -> bool:
    """Whether the forward must force eager dispatch (cudagraph NONE).

    Under the policy this is true for ``PURE_PREFILL`` and ``MIXED``: the
    fused comm path runs outside the ACL graphs, and ``force_eager`` is the
    belt-and-braces guarantee that a non-decode wave is never dispatched into
    (or replayed from) a FULL template. Without it, the runner's
    ``uniform_decode`` heuristic can classify a 1-token prefill as a uniform
    decode wave when ``speculative_config`` is absent. ``PURE_DECODE`` keeps
    ``force_eager=False`` so FULL capture/replay of the compiled outer
    artifact proceeds (the opaque MoE op dispatches to the baseline comm
    family at op runtime).
    """
    return (
        get_moe_phase_hybrid_policy(vllm_config)
        in (MoEPhaseHybridPolicy.NON_DECODE_FUSED, MoEPhaseHybridPolicy.OPAQUE_FUSED_CONTROL)
        and not is_draft_model
        and forward_phase in (MoEForwardPhase.PURE_PREFILL, MoEForwardPhase.MIXED)
        and is_moe_model(vllm_config)
    )


def get_moe_phase_hybrid_warmup_phase(
    vllm_config: VllmConfig,
    is_draft_model: bool = False,
) -> MoEForwardPhase | None:
    """Explicit non-decode phase for profile/compile warmups under the policy.

    Returns ``MoEForwardPhase.MIXED`` when the policy is active on the main
    MoE model, so profile and compile-size warmups deterministically execute
    the fused non-decode path before decode warmup/capture. Returns ``None``
    (caller keeps its historical default phase) when the policy is off, the
    model is not MoE, or the forward is a draft model.
    """
    if not is_moe_phase_hybrid_active(MoEForwardPhase.MIXED, vllm_config, is_draft_model):
        return None
    return MoEForwardPhase.MIXED


def validate_moe_phase_hybrid_policy(vllm_config: VllmConfig) -> None:
    """Fail closed when the phase-keyed hybrid policy cannot be safe.

    - Data parallel: DP ranks can schedule different phases (decode vs
      prefill) for the same step, but the fused MC2 collectives span the EP
      group which includes every DP rank. A per-rank phase-dependent comm
      choice would deadlock or corrupt shapes, so DP > 1 raises at startup.
    - V2 runner: the upstream V2 runner owns the ``skip_compiled`` /
      ``force_eager`` decisions itself, so vllm-ascend cannot enforce the
      phase-keyed matrix (baseline decode via opaque-op runtime dispatch
      inside FULL_DECODE_ONLY ACL graphs, fused non-decode outside them)
      from the platform hook. V2 raises at startup.
    - Cudagraph envelope: only ``FULL_DECODE_ONLY`` keeps non-decode waves
      out of graph templates structurally (backed by ``force_eager`` at the
      runner), so pure decode is the only graph path. FULL /
      FULL_AND_PIECEWISE / PIECEWISE can dispatch prefill into a template
      and are rejected; ``NONE`` (no decode graphs) is rejected because the
      policy's invariant requires decode graphs to exist.
    - SoC/EP capability gate: the static fused-MC2 gate (A3 EP<=32, A2
      fused-A2 conditions) cannot be evaluated here because the EP process
      group is not initialized at platform startup. It is enforced **fail
      fast** by :func:`select_moe_comm_method` at the first non-decode
      forward -- under this policy that is the unconditional fused MIXED
      profile dummy, before any decode capture or fused compilation -- and
      never by a baseline fallback.
    """
    if get_moe_phase_hybrid_policy(vllm_config) == MoEPhaseHybridPolicy.OFF:
        return
    # The fused impl and the NZ weight layout are only initialized when
    # enable_fused_mc2 == 1; without it the policy silently falls back to the
    # baseline family and the experiment is not actually enabled.
    if get_ascend_config().enable_fused_mc2 != 1:
        raise ValueError(
            "additional_config.moe_phase_hybrid_policy=non_decode_fused "
            "requires additional_config.enable_fused_mc2 == 1: the fused "
            "comm impl and the NZ weight layout are only initialized in that "
            "mode. Without it the policy silently selects the baseline "
            "family and the experiment is not enabled."
        )
    if vllm_config.parallel_config.data_parallel_size > 1:
        raise ValueError(
            "additional_config.moe_phase_hybrid_policy=non_decode_fused "
            "requires data_parallel_size == 1: DP ranks can schedule "
            "different phases for the same step, and FUSED_MC2 collectives "
            "span all DP ranks, so a per-rank phase-dependent comm choice is "
            "not DP-consistent."
        )
    if vllm_config.use_v2_model_runner:
        raise ValueError(
            "additional_config.moe_phase_hybrid_policy=non_decode_fused "
            "is not supported on the V2 model runner: the upstream runner "
            "owns skip_compiled/force_eager, so vllm-ascend cannot enforce "
            "the phase-keyed matrix (baseline decode via opaque-op runtime "
            "dispatch inside FULL_DECODE_ONLY ACL graphs, fused non-decode "
            "outside them), which would recreate the stale-layout failure."
        )
    cudagraph_mode = vllm_config.compilation_config.cudagraph_mode
    if cudagraph_mode != CUDAGraphMode.FULL_DECODE_ONLY:
        raise ValueError(
            "additional_config.moe_phase_hybrid_policy=non_decode_fused "
            f"requires cudagraph_mode == FULL_DECODE_ONLY, got {cudagraph_mode}. "
            "FULL / FULL_AND_PIECEWISE / PIECEWISE can dispatch prefill or "
            "mixed waves into a graph template, which would either replay a "
            "stock template under a fused intent or require unsafe runtime "
            "capture; force_eager alone does not protect that envelope."
        )


_mrv2_in_profile_run: bool = False


@contextmanager
def override_mrv2_in_profile_run(enabled: bool):
    """Override MRv2's extra profile-run marker for one forward path.

    MRv2 builds the base forward context inside upstream vLLM, so Ascend's
    platform hook cannot tell whether the current forward is the extra MC2
    profile dummy run. A module-level bool keeps this MRv2-only state
    scoped to the current forward path without using contextvars.ContextVar,
    which is incompatible with torch.compile(fullgraph=True).
    """
    global _mrv2_in_profile_run
    old = _mrv2_in_profile_run
    _mrv2_in_profile_run = enabled
    try:
        yield
    finally:
        _mrv2_in_profile_run = old


def get_mrv2_in_profile_run() -> bool:
    return _mrv2_in_profile_run


@contextmanager
def set_ascend_forward_context(
    attn_metadata: Any,
    vllm_config: VllmConfig,
    num_tokens: int = 0,
    num_tokens_across_dp: torch.Tensor | None = None,
    in_profile_run: bool = False,
    num_actual_tokens: int | None = None,
    aclgraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,
    batch_descriptor: BatchDescriptor | None = None,
    model_instance: torch.nn.Module = None,
    is_draft_model=False,
    skip_compiled: bool = False,
    max_tokens_across_pcp: int = 0,
    draft_attn_metadatas=None,
    has_sinks=False,
    input_ids=None,
    eplb_heat_collection_status: bool = False,
    forward_phase: MoEForwardPhase | None = None,
):
    """A context manager that stores the current forward context,
    can be attention metadata, etc.
    We add some additional param into forward_context.
    """
    # Phase-keyed hybrid MoE policy (experimental, default OFF). Under the
    # opaque-MoE canary, NO phase bypasses the torch.compile/AOT artifact:
    # the outer compiled graph contains only opaque torch.ops.vllm.moe_forward
    # calls, and the live _forward_impl dispatch happens at op runtime, so
    # PURE_DECODE and PURE_PREFILL/MIXED all share the one compiled outer
    # artifact. The comm family is selected per phase at op runtime
    # (baseline vs FUSED_MC2); force_eager is decided by the runner, never
    # derived from skip_compiled. skip_compiled here only reflects upstream
    # encoder-input handling (has_encoder_input).


    forward_context_kwargs = {
        "attn_metadata": attn_metadata,
        "vllm_config": vllm_config,
        "num_tokens": num_tokens,
        "num_tokens_across_dp": num_tokens_across_dp,
        "cudagraph_runtime_mode": aclgraph_runtime_mode,
        "batch_descriptor": batch_descriptor,
        "skip_compiled": skip_compiled,
    }
    with set_forward_context(**forward_context_kwargs):
        forward_context = get_forward_context()
        forward_context.draft_attn_metadatas = draft_attn_metadatas

        forward_context.input_ids = input_ids

        from vllm_ascend.ops.fused_moe.moe_comm_method import get_moe_comm_method

        max_num_tokens = int(num_tokens_across_dp.max().item()) if num_tokens_across_dp is not None else num_tokens
        moe_comm_type = select_moe_comm_method(max_num_tokens, vllm_config, is_draft_model, phase=forward_phase)

        forward_context.moe_comm_type = moe_comm_type
        forward_context.moe_comm_method = get_moe_comm_method(moe_comm_type)

        tp_world_size = get_tensor_model_parallel_world_size()

        forward_context.in_profile_run = in_profile_run

        # NOTE: This cannot be set using set_forward_context
        # due to multiple warmups before actual capturing
        forward_context.capturing = False

        # TODO: remove it when fia merge in fiav2
        forward_context.sinks = has_sinks

        # TODO: remove it when torch_npu.npu_mm_reduce_scatter_base supports tp_size >= 16.
        mmrs_fusion = tp_world_size <= 8

        # set for sequence parallelism, 1000 is the batch size concurrency threshold
        # for enabling the flashcomm_v1 or sequence_parallelism feature.
        # Currently, it is an empirical value. In normal scenarios, if the concurrency
        # exceeds this threshold, the performance benefits can be maximized.
        # Conversely, if the concurrency is below the threshold,
        # the performance may degrade due to the switching of communication methods.

        # main model and drafter model may have different architecture
        is_context_moe_model = is_drafter_moe_model(vllm_config) if is_draft_model else is_moe_model(vllm_config)
        if is_context_moe_model:
            flash_comm_v1_enabled = enable_sp(vllm_config) and num_tokens is not None
            mmrs_fusion = False
        elif is_draft_model:
            # TODO: for dense drafter, `sp` is redundant and is not compatible with `dp` and `graph`.
            # Disable it to avoid more problems.
            flash_comm_v1_enabled = False
        else:
            flash_comm_v1_enabled = enable_sp(vllm_config) and num_tokens is not None and num_tokens > 1000
        forward_context.mmrs_fusion = mmrs_fusion
        forward_context.num_tokens = num_tokens
        forward_context.flash_comm_v1_enabled = flash_comm_v1_enabled
        # TODO(Levi-JQ): another PR to normalize the enabling logic for sp/fc2
        forward_context.flashcomm_v2_enabled = flashcomm2_enable() and tp_world_size > 1 and num_tokens is not None

        forward_context.pad_size = 0
        if forward_context.flash_comm_v1_enabled or forward_context.flashcomm_v2_enabled:
            pad_size = (tp_world_size - (num_tokens % tp_world_size)) % tp_world_size
            forward_context.pad_size = pad_size

        # set this for rope forward_oot using
        forward_context.is_first_layer = True

        # set layer_idx to enable optimization features that depend on this information.
        # This is only applicable to models that contain these necessary attributes.
        forward_context.layer_idx = None
        if has_layer_idx(model_instance):
            forward_context.layer_idx = model_instance.model.start_layer

        forward_context.prefetch_mlp_gate_up_proj = False
        forward_context.prefetch_mlp_down_proj = False
        forward_context.model_instance = model_instance
        forward_context.is_draft_model = is_draft_model
        forward_context.is_draft_model_prefill = False

        if num_tokens is None and attn_metadata is not None:
            num_tokens = attn_metadata.num_actual_tokens

        dp_world_size = get_dp_group().world_size
        if dp_world_size > 1 and forward_context.dp_metadata is not None:
            dp_meta = forward_context.dp_metadata
            max_tokens_across_dp = dp_meta.num_tokens_across_dp_cpu.max().item()
            if forward_context.flash_comm_v1_enabled or forward_context.flashcomm_v2_enabled:
                padded_length = (max_tokens_across_dp + tp_world_size - 1) // tp_world_size * tp_world_size
                pad_size = padded_length - num_tokens
                forward_context.padded_length = padded_length
                forward_context.pad_size = pad_size
        else:
            max_tokens_across_dp = num_tokens

        forward_context.max_tokens_across_dp = max_tokens_across_dp
        forward_context.max_tokens_across_pcp = max_tokens_across_pcp

        forward_context.eplb_heat_collection_status = eplb_heat_collection_status

        if num_tokens is not None:
            if num_actual_tokens is None:
                num_actual_tokens = num_tokens
            # NOTE: token num which need to pad to when mc2
            forward_context.padded_num_tokens = math.ceil(max_tokens_across_dp / tp_world_size) * tp_world_size
            reserved_mc2_mask = get_mc2_mask()
            if reserved_mc2_mask is not None:
                mc2_mask = reserved_mc2_mask[: forward_context.padded_num_tokens]
                mc2_mask[:num_actual_tokens] = True
                mc2_mask[num_actual_tokens:] = False
                forward_context.mc2_mask = mc2_mask
        try:
            yield
        finally:
            pass


_mc2_tokens_capacity: int | None = None
_reserved_mc2_mask: torch.Tensor | None = None


def set_mc2_tokens_capacity(vllm_config, max_num_reqs, uniform_decode_query_len):
    global _mc2_tokens_capacity
    if _mc2_tokens_capacity is not None:
        return
    if get_ascend_config().enable_prefill_mc2:
        max_num_tokens = vllm_config.scheduler_config.max_num_batched_tokens
    elif vllm_config.compilation_config.cudagraph_capture_sizes:
        max_num_tokens = vllm_config.compilation_config.max_cudagraph_capture_size
    else:
        max_num_tokens = max_num_reqs * uniform_decode_query_len
    tp_size = vllm_config.parallel_config.tensor_parallel_size
    # Use integer arithmetic for ceiling division.
    num_tokens_per_tp_rank = (max_num_tokens + tp_size - 1) // tp_size
    # NOTE: To save memory, we cap the max number of tokens to 512.
    num_tokens_per_tp_rank = min(num_tokens_per_tp_rank, 512)
    _mc2_tokens_capacity = num_tokens_per_tp_rank * tp_size


def get_mc2_tokens_capacity():
    return _mc2_tokens_capacity


def set_mc2_mask(vllm_config, device):
    global _reserved_mc2_mask
    if _reserved_mc2_mask is not None:
        return
    if is_moe_model(vllm_config):
        _reserved_mc2_mask = torch.zeros(
            vllm_config.scheduler_config.max_num_batched_tokens, dtype=torch.bool, device=device
        )
    else:
        _reserved_mc2_mask = None


def get_mc2_mask():
    return _reserved_mc2_mask


def _moe_phase_hybrid_non_decode_fused_available(vllm_config: VllmConfig) -> bool:
    """SoC capability gate for the fused non-decode path of the hybrid policy.

    Reuses exactly the same conditions the stock selector applies for the
    FUSED_MC2 / dispatch_ffn_combine BF16 path. Deliberately token-count
    independent: per-wave token thresholds are what historically swapped the
    Python comm object inside one compiled template.
    """
    if not is_moe_model(vllm_config):
        return False
    if not vllm_config.parallel_config.enable_expert_parallel or get_ep_group().world_size == 1:
        return False
    soc_version = get_ascend_device_type()
    ascend_config = get_ascend_config()
    if soc_version in {AscendDeviceType.A3}:
        # dispatch_ffn_combine (BF16) is supported up to EP size 32.
        return ascend_config.enable_fused_mc2 == 1 and get_ep_group().world_size <= 32
    if soc_version in {AscendDeviceType.A2}:
        num_experts = vllm_config.model_config.get_num_experts()
        ep_world_size = (
            vllm_config.parallel_config.world_size_across_dp // vllm_config.parallel_config.pipeline_parallel_size
        )
        num_experts_per_device = num_experts // ep_world_size
        quant_type = getattr(
            vllm_config.model_config.hf_text_config,
            "moe_quantize",
            getattr(vllm_config.model_config.hf_text_config, "quantize", None),
        )
        return (
            ascend_config.enable_fused_mc2 == 1
            and quant_type in (None, "w8a8_dynamic")
            and get_ep_group().world_size <= MAX_A2_FUSED_MC2_EP_SIZE
            and num_experts_per_device >= MIN_A2_FUSED_MC2_LOCAL_EXPERTS
        )
    return False


def _moe_phase_hybrid_non_decode_fused_error(vllm_config: VllmConfig, phase: MoEForwardPhase) -> ValueError:
    """Build the deterministic fail-fast error for an unsupported gate.

    Under ``NON_DECODE_FUSED`` a non-decode phase must never select the
    baseline comm family: the baseline family is token-threshold dependent,
    so a baseline wave inside the single fused torch.compile lane could swap
    the comm object per wave and recreate the stale-template failure. The
    message names the gate requirement so the operator can fix the
    configuration (or turn the policy off) instead of discovering a
    nondeterministic runtime mismatch.
    """
    # The gate helper just ran, so the EP group is resolved; these reads are
    # only for a precise diagnostic message.
    ep_world_size = get_ep_group().world_size
    num_experts = vllm_config.model_config.get_num_experts()
    ep_world_size_from_config = (
        vllm_config.parallel_config.world_size_across_dp // vllm_config.parallel_config.pipeline_parallel_size
    )
    num_experts_per_device = num_experts // max(ep_world_size_from_config, 1)
    quant_type = getattr(
        vllm_config.model_config.hf_text_config,
        "moe_quantize",
        getattr(vllm_config.model_config.hf_text_config, "quantize", None),
    )
    return ValueError(
        "additional_config.moe_phase_hybrid_policy=non_decode_fused: "
        f"{phase.name} requires the static fused-MC2 capability gate, which "
        "is unavailable on this configuration "
        f"(soc={get_ascend_device_type()}, enable_fused_mc2="
        f"{get_ascend_config().enable_fused_mc2}, ep_world_size={ep_world_size}, "
        f"quant={quant_type!r}, local_experts={num_experts_per_device}). "
        "A non-decode wave must NEVER fall back to the token-dependent "
        "baseline comm family inside the single fused torch.compile lane; "
        "fix the gate (A3: enable_fused_mc2==1 and EP<=32; A2: "
        "enable_fused_mc2==1, EP<=8, quant in {None, w8a8_dynamic}, and "
        ">=3 local experts) or set moe_phase_hybrid_policy=off."
    )


def select_moe_comm_method(
    num_tokens: int,
    vllm_config: VllmConfig,
    is_draft_model: bool = False,
    *,
    phase: MoEForwardPhase | None = None,
) -> MoECommType | None:
    """Select the MoE communication method, optionally phase-keyed.

    With the hybrid policy OFF (default), with ``phase=None`` (legacy
    callers such as the spec-decode proposers), or for non-MoE / draft
    models, this returns exactly the stock selector result.

    With ``MoEPhaseHybridPolicy.NON_DECODE_FUSED``:

    - ``PURE_PREFILL`` and ``MIXED`` select ``MoECommType.FUSED_MC2``; when
      the static SoC capability gate does not hold, selection **raises a
      deterministic ``ValueError``** (fail fast) instead of falling back to
      the token-dependent baseline family -- a baseline non-decode wave
      inside the single fused torch.compile lane is exactly the stale-template
      failure this policy exists to prevent.
    - ``PURE_DECODE`` selects the baseline family -- the stock selector with
      ``enable_fused_mc2 == 0`` -- so the decode path inside the
      FULL_DECODE_ONLY ACL graphs keeps exactly the comm object it captured
      with (never the fused path the stock selector would pick under
      ``enable_fused_mc2=1``). The comm family is dispatched at opaque-op
      runtime inside the one compiled outer artifact; decode never retraces.
    """
    policy = get_moe_phase_hybrid_policy(vllm_config)
    if (
        policy == MoEPhaseHybridPolicy.OFF
        or phase is None
        or is_draft_model
        or not is_moe_model(vllm_config)
    ):
        # Non-MoE models return None (no comm family at all), so they never
        # enter the fail-fast non-decode branch.
        return _select_moe_comm_method_stock(num_tokens, vllm_config, is_draft_model)
    wants_fused = phase in (MoEForwardPhase.PURE_PREFILL, MoEForwardPhase.MIXED) or (
        policy == MoEPhaseHybridPolicy.OPAQUE_FUSED_CONTROL and phase is MoEForwardPhase.PURE_DECODE
    )
    if wants_fused:
        if _moe_phase_hybrid_non_decode_fused_available(vllm_config):
            logger.debug(
                "MoE phase-keyed hybrid: phase=%s selects %s",
                phase,
                MoECommType.FUSED_MC2,
            )
            return MoECommType.FUSED_MC2
        # Fail fast, never fail closed: a baseline non-decode wave would swap
        # the comm object inside the single fused torch.compile lane (the
        # baseline family is token-threshold dependent), which is exactly the
        # stale-template failure this policy exists to prevent. The first
        # non-decode forward -- under the policy that is the unconditional
        # fused MIXED profile dummy -- raises before any decode capture or
        # fused compilation, so the error is deterministic and precedes
        # compilation/first profile.
        raise _moe_phase_hybrid_non_decode_fused_error(vllm_config, phase)
    # PURE_DECODE: the baseline family, dispatched at opaque-op runtime
    # inside the compiled outer artifact that the FULL_DECODE_ONLY ACL graphs
    # capture and replay (never the fused path the stock selector would pick
    # under enable_fused_mc2=1).
    baseline_type = _select_moe_comm_method_baseline(num_tokens, vllm_config, is_draft_model)
    logger.debug(
        "MoE phase-keyed hybrid: phase=%s selects baseline %s",
        phase,
        baseline_type,
    )
    return baseline_type


def _select_moe_comm_method_baseline(
    num_tokens: int, vllm_config: VllmConfig, is_draft_model: bool = False
) -> MoECommType | None:
    """Baseline MoE communication family: the stock selector with ``enable_fused_mc2 == 0``.

    This is the explicit, auditable version of "what the stock selector
    would choose if fused MC2 were disabled". The phase-keyed hybrid policy
    uses it for ``PURE_DECODE`` only (a non-decode phase with an unavailable
    fused gate raises fail-fast instead), so under the policy the decode
    comm object is the baseline family (ALLGATHER on the A2/TP2 experiment,
    MC2 or ALLTOALL on A3), never the fused path the stock selector would
    pick with ``enable_fused_mc2=1``. It is written out explicitly rather
    than simulated by temporarily mutating the global config, so the choice
    is deterministic and unit-testable.

    Rules are identical to :func:`_select_moe_comm_method_stock` with the
    fused-MC2 branches removed:

    1. Non-MoE models return `None`.
    2. Without expert parallel (or EP==1), fall back to all-gather.
    3. On A2 with expert parallel, pick MC2 when tokens fit the MC2 capacity
       and the EP group is large enough, else all-gather (the fused-A2 branch
       never applies).
    4. On A3 with expert parallel, use MC2 within capacity, else all-to-all
       (no fused branch).
    5. On 310P, always use all-gather.
    6. On A5 with expert parallel, identical to stock: MC2 when tokens fit
       the MC2 capacity and the world size is large enough; otherwise
       all-gather when EP is smaller than num of topK experts, else
       all-to-all.

    Args:
        num_tokens (int): The number of tokens in the current batch.
        vllm_config (VllmConfig): Runtime configuration for the model.
        is_draft_model (bool): Whether the model runs in MTP mode.

    Raises:
        ValueError: If the soc version is unsupported.

    Returns:
        MoECommType | None: The selected baseline communication method.
    """
    if not is_moe_model(vllm_config):
        return None
    mc2_tokens_capacity = get_mc2_tokens_capacity()
    soc_version = get_ascend_device_type()

    if not vllm_config.parallel_config.enable_expert_parallel or get_ep_group().world_size == 1:
        moe_comm_type = MoECommType.ALLGATHER
    elif soc_version in {AscendDeviceType.A2}:
        num_experts = vllm_config.model_config.get_num_experts()
        ep_world_size = (
            vllm_config.parallel_config.world_size_across_dp // vllm_config.parallel_config.pipeline_parallel_size
        )
        num_experts_per_device = num_experts // ep_world_size
        if num_experts_per_device <= 24 and ep_world_size >= 16 and num_tokens <= mc2_tokens_capacity:
            moe_comm_type = MoECommType.MC2
        else:
            moe_comm_type = MoECommType.ALLGATHER
    elif soc_version in {AscendDeviceType.A3}:
        # enable_fused_mc2 == 0: the fused branch never applies; the MC2
        # capacity alone decides MC2 vs ALLTOALL.
        if num_tokens <= mc2_tokens_capacity:
            moe_comm_type = MoECommType.MC2
        else:
            moe_comm_type = MoECommType.ALLTOALL
    elif soc_version in {AscendDeviceType._310P}:
        moe_comm_type = MoECommType.ALLGATHER
    elif soc_version in {AscendDeviceType.A5}:
        num_experts_per_tok = getattr(
            vllm_config.model_config.hf_text_config,
            "num_experts_per_tok",
            getattr(vllm_config.model_config.hf_text_config, "top_k_experts", 1),
        )
        world_size = vllm_config.parallel_config.world_size_across_dp
        if num_tokens <= mc2_tokens_capacity and world_size > 1:
            moe_comm_type = MoECommType.MC2
        elif world_size <= num_experts_per_tok:
            moe_comm_type = MoECommType.ALLGATHER
        else:
            moe_comm_type = MoECommType.ALLTOALL
    else:
        raise ValueError(f"Unsupported soc_version: {soc_version}")
    logger.debug(
        "MoE baseline comm method selected: soc=%s, method=%s, num_tokens=%d, mc2_capacity=%s",
        soc_version,
        moe_comm_type,
        num_tokens,
        mc2_tokens_capacity,
    )
    return moe_comm_type


def _select_moe_comm_method_stock(num_tokens: int, vllm_config: VllmConfig, is_draft_model=False) -> MoECommType | None:
    """Stock MoE communication method selection.

    This is the historical, phase-agnostic selector. It is kept byte-for-byte
    intact and is the only behavior visible when the phase-keyed hybrid policy
    is OFF. Selection considers parallel settings, device generation, token
    count, and quantization.

    1. Non-MoE models return `None`.
    2. Without expert parallel, fall back to all-gather.
    3. On A2 with expert parallel, use the BF16 or W8A8 dynamic fused MC2 path
       when explicitly enabled for a small EP group with enough local experts
       for the kernel pipeline. Otherwise, pick MC2 when tokens fit the MC2
       capacity and the DP size is large enough, or fall back to all-gather.
    4. On A3 with expert parallel, prefer fused MC2 when using w8a8_dynamic
       quantization with small EP size, no dynamic_eplb, and not in MTP
       mode; otherwise use MC2 within capacity or all-to-all.
    5. On 310P, always use all-gather.
    6. On A5 with expert parallel, use MC2 when tokens fit the MC2 capacity
       and the EP size is large enough; otherwise use all-gather when
       EP size is smaller than num of topK experts or all-to-all.

    Args:
        num_tokens (int): The number of tokens in the current batch.
        vllm_config (VllmConfig): Runtime configuration for the model.
        is_draft_model (bool): Whether the model runs in MTP mode.

    Raises:
        ValueError: If the soc version is unsupported.

    Returns:
        MoECommType | None: The selected MoE communication method.
    """
    if not is_moe_model(vllm_config):
        return None
    mc2_tokens_capacity = get_mc2_tokens_capacity()
    soc_version = get_ascend_device_type()
    quant_type = getattr(
        vllm_config.model_config.hf_text_config,
        "moe_quantize",
        getattr(vllm_config.model_config.hf_text_config, "quantize", None),
    )

    if not vllm_config.parallel_config.enable_expert_parallel or get_ep_group().world_size == 1:
        moe_comm_type = MoECommType.ALLGATHER
    elif soc_version in {AscendDeviceType.A2}:
        num_experts = vllm_config.model_config.get_num_experts()
        ep_world_size = (
            vllm_config.parallel_config.world_size_across_dp // vllm_config.parallel_config.pipeline_parallel_size
        )
        num_experts_per_device = num_experts // ep_world_size
        fused_a2_enable = (
            get_ascend_config().enable_fused_mc2 == 1
            and quant_type in (None, "w8a8_dynamic")
            and get_ep_group().world_size <= MAX_A2_FUSED_MC2_EP_SIZE
            and num_experts_per_device >= MIN_A2_FUSED_MC2_LOCAL_EXPERTS
        )
        if fused_a2_enable:
            moe_comm_type = MoECommType.FUSED_MC2
        elif num_experts_per_device <= 24 and ep_world_size >= 16 and num_tokens <= mc2_tokens_capacity:
            moe_comm_type = MoECommType.MC2
        else:
            moe_comm_type = MoECommType.ALLGATHER

    elif soc_version in {AscendDeviceType.A3}:
        # TODO: drop the EP-size guard when dispatch_ffn_combine supports larger EP sizes
        # TODO: drop speculative method guard when dispatch_gmm_combine_decode supports w16a16
        fused_mc2_enable = get_ascend_config().enable_fused_mc2
        dispatch_ffn_combine_enable = get_ep_group().world_size <= 32
        if num_tokens <= mc2_tokens_capacity:
            fused_decode_enable = fused_mc2_enable
            if fused_mc2_enable == 1:
                fused_decode_enable = fused_mc2_enable and dispatch_ffn_combine_enable
            elif fused_mc2_enable == 2:
                fused_decode_enable = (
                    fused_mc2_enable
                    and speculative_enable_dispatch_gmm_combine_decode(vllm_config)
                    and quant_type == "w8a8_dynamic"
                )
            moe_comm_type = MoECommType.FUSED_MC2 if fused_decode_enable else MoECommType.MC2
        else:
            fused_prefill_enable = fused_mc2_enable
            if fused_mc2_enable == 1:
                fused_prefill_enable = fused_mc2_enable and dispatch_ffn_combine_enable
            elif fused_mc2_enable == 2:
                fused_prefill_enable = False
            moe_comm_type = MoECommType.FUSED_MC2 if fused_prefill_enable else MoECommType.ALLTOALL
    elif soc_version in {AscendDeviceType._310P}:
        moe_comm_type = MoECommType.ALLGATHER
    elif soc_version in {AscendDeviceType.A5}:
        num_experts_per_tok = getattr(
            vllm_config.model_config.hf_text_config,
            "num_experts_per_tok",
            getattr(vllm_config.model_config.hf_text_config, "top_k_experts", 1),
        )
        world_size = vllm_config.parallel_config.world_size_across_dp
        if num_tokens <= mc2_tokens_capacity and world_size > 1:
            moe_comm_type = MoECommType.MC2
        elif world_size <= num_experts_per_tok:
            moe_comm_type = MoECommType.ALLGATHER
        else:
            moe_comm_type = MoECommType.ALLTOALL
    else:
        raise ValueError(f"Unsupported soc_version: {soc_version}")
    logger.debug(
        "MoE comm method selected: soc=%s, method=%s, num_tokens=%d, mc2_capacity=%s",
        soc_version,
        moe_comm_type,
        num_tokens,
        mc2_tokens_capacity,
    )
    return moe_comm_type


class _ExtraForwardContextProxy:
    """Unified forward-context access for v1/v2 model runners."""

    extra_attrs = (
        "capturing",
        "moe_comm_type",
        "moe_comm_method",
        "mmrs_fusion",
        "num_tokens",
        "flash_comm_v1_enabled",
        "flashcomm_v2_enabled",
        "pad_size",
        "padded_length",
        "num_tokens_across_dp",
        "mc2_mask",
        "is_draft_model",
        "is_draft_model_prefill",
        "prefetch_mlp_gate_up_proj",
        "prefetch_mlp_down_proj",
        "model_instance",
        "layer_idx",
        "max_tokens_across_dp",
        "max_tokens_across_pcp",
        "num_accept_tokens",
        "in_profile_run",
        "padded_num_tokens",
        "sinks",
        "eplb_heat_collection_status",
    )

    def check_extra_attr(self, name: str):
        if name not in self.extra_attrs:
            raise AttributeError(
                f"{name} is not extra forward context attribute, "
                "please get/set it from vllm's _forward_context directly."
            )

    @staticmethod
    def _ctx():
        return get_forward_context()

    def __getattr__(self, name: str) -> Any:
        self.check_extra_attr(name)
        ctx = self._ctx()
        if envs_vllm.VLLM_USE_V2_MODEL_RUNNER:
            # Unset known extras default to None so optional flags (e.g. `sinks`)
            # can be read with truthiness checks before the V2 path populates them.
            return ctx.additional_kwargs.get(name)
        return getattr(ctx, name, None)

    def __setattr__(self, name: str, value: Any) -> None:
        self.check_extra_attr(name)
        ctx = self._ctx()
        if envs_vllm.VLLM_USE_V2_MODEL_RUNNER:
            ctx.additional_kwargs[name] = value
        else:
            setattr(ctx, name, value)


# usage: from vllm_ascend.ascend_forward_context import _EXTRA_CTX
_EXTRA_CTX = _ExtraForwardContextProxy()
