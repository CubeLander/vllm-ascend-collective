from contextlib import contextmanager
from enum import Enum
from typing import Any

import torch
import vllm.envs as envs_vllm
from vllm.config import CUDAGraphMode, VllmConfig
from vllm.distributed import get_dp_group, get_ep_group, get_tensor_model_parallel_world_size
from vllm.forward_context import BatchDescriptor, get_forward_context, set_forward_context
from vllm.logger import logger

from vllm_ascend.ascend_config import get_ascend_config
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
MAX_A2_FUSED_MC2_LOCAL_EXPERTS = 64
MAX_A2_FUSED_MC2_TOP_K = 8
MAX_A2_FUSED_MC2_LOCAL_TOKENS = 512


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
):
    """A context manager that stores the current forward context,
    can be attention metadata, etc.
    We add some additional param into forward_context.
    """
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
        selection_num_tokens = max_num_tokens if _is_a2_fused_bf16_configured(vllm_config) else num_tokens
        moe_comm_type = select_moe_comm_method(selection_num_tokens, vllm_config, is_draft_model)

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
            forward_context.padded_num_tokens = get_mc2_padded_num_tokens(
                max_tokens_across_dp,
                tp_world_size,
                moe_comm_type,
            )
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
_mc2_tokens_limit: int | None = None
_reserved_mc2_mask: torch.Tensor | None = None


def _get_quant_type(vllm_config: VllmConfig):
    hf_text_config = vllm_config.model_config.hf_text_config
    return getattr(hf_text_config, "moe_quantize", getattr(hf_text_config, "quantize", None))


def _get_num_experts_per_token(vllm_config: VllmConfig) -> int:
    hf_text_config = vllm_config.model_config.hf_text_config
    return getattr(hf_text_config, "num_experts_per_tok", getattr(hf_text_config, "top_k_experts", 1))


def _is_a2_fused_bf16_configured(vllm_config: VllmConfig) -> bool:
    # This predicate is also reached by utility/unit-test call paths before
    # Ascend's process-global configuration is initialized.  Such a path
    # cannot have selected the fused family, so keep the probe conservative
    # instead of turning an ordinary forward-context query into an ordering
    # requirement on init_ascend_config().
    if get_ascend_device_type() is not AscendDeviceType.A2:
        return False
    try:
        ascend_config = get_ascend_config()
    except RuntimeError as error:
        if "Ascend config is not initialized" not in str(error):
            raise
        return False

    parallel_config = vllm_config.parallel_config
    ep_world_size = parallel_config.world_size_across_dp // parallel_config.pipeline_parallel_size
    num_experts = vllm_config.model_config.get_num_experts()
    num_local_experts = num_experts // ep_world_size if ep_world_size else 0
    top_k = _get_num_experts_per_token(vllm_config)
    eplb_config = getattr(ascend_config, "eplb_config", None)
    dynamic_eplb = bool(getattr(eplb_config, "dynamic_eplb", False))
    redundant_experts = int(getattr(eplb_config, "num_redundant_experts", 0))
    return (
        ascend_config.enable_fused_mc2 == 1
        and _get_quant_type(vllm_config) is None
        and getattr(vllm_config.model_config, "dtype", None) == torch.bfloat16
        and parallel_config.enable_expert_parallel
        and 1 < ep_world_size <= MAX_A2_FUSED_MC2_EP_SIZE
        and num_experts % ep_world_size == 0
        and MIN_A2_FUSED_MC2_LOCAL_EXPERTS <= num_local_experts <= MAX_A2_FUSED_MC2_LOCAL_EXPERTS
        and 1 <= top_k <= MAX_A2_FUSED_MC2_TOP_K
        and not dynamic_eplb
        and redundant_experts == 0
    )


def _round_up_to_multiple(value: int, multiple: int) -> int:
    return (value + multiple - 1) // multiple * multiple


def _compute_mc2_tokens_limit(vllm_config, max_num_reqs, uniform_decode_query_len) -> int:
    if _is_a2_fused_bf16_configured(vllm_config) or get_ascend_config().enable_prefill_mc2:
        return vllm_config.scheduler_config.max_num_batched_tokens
    if vllm_config.compilation_config.cudagraph_capture_sizes:
        return vllm_config.compilation_config.max_cudagraph_capture_size
    return max_num_reqs * uniform_decode_query_len


def _compute_mc2_tokens_capacity(vllm_config, max_num_reqs, uniform_decode_query_len) -> int:
    max_num_tokens = _compute_mc2_tokens_limit(vllm_config, max_num_reqs, uniform_decode_query_len)
    tp_size = vllm_config.parallel_config.tensor_parallel_size
    local_capacity = (max_num_tokens + tp_size - 1) // tp_size
    if _is_a2_fused_bf16_configured(vllm_config):
        if local_capacity > MAX_A2_FUSED_MC2_LOCAL_TOKENS:
            raise ValueError(
                "A2 BF16 fused MC2 requires ceil(max_num_batched_tokens / "
                f"tensor_parallel_size) <= {MAX_A2_FUSED_MC2_LOCAL_TOKENS}, got {local_capacity}."
            )
    else:
        local_capacity = min(local_capacity, 512)
    return local_capacity * tp_size


def set_mc2_tokens_capacity(vllm_config, max_num_reqs, uniform_decode_query_len):
    global _mc2_tokens_capacity, _mc2_tokens_limit
    a2_fused_bf16 = _is_a2_fused_bf16_configured(vllm_config)
    if get_ascend_device_type() is AscendDeviceType.A2 and get_ascend_config().enable_fused_mc2 == 1 and not a2_fused_bf16:
        if getattr(vllm_config, "speculative_config", None) is not None:
            raise ValueError("A2 BF16 fused MC2 does not support draft/MTP execution.")
        raise ValueError(
            "A2 enable_fused_mc2=1 requires unquantized BF16, EP 2..8, 3..64 local experts, "
            "top-k 1..8, and no dynamic or redundant EPLB experts."
        )
    max_num_tokens = _compute_mc2_tokens_limit(vllm_config, max_num_reqs, uniform_decode_query_len)
    requested_capacity = _compute_mc2_tokens_capacity(vllm_config, max_num_reqs, uniform_decode_query_len)
    if _mc2_tokens_capacity is None:
        _mc2_tokens_capacity = requested_capacity
        _mc2_tokens_limit = max_num_tokens
    elif not a2_fused_bf16:
        # Preserve the historical process-global first-run behavior for every
        # ordinary MC2 implementation. Only this new immutable A2 family needs
        # strict agreement between compiled runners.
        return
    elif _mc2_tokens_capacity != requested_capacity or _mc2_tokens_limit != max_num_tokens:
        raise RuntimeError(
            "MC2 capacity was already initialized for a different execution domain "
            f"(limit/capacity {_mc2_tokens_limit}/{_mc2_tokens_capacity} != "
            f"{max_num_tokens}/{requested_capacity})."
        )


def get_mc2_tokens_capacity():
    return _mc2_tokens_capacity


def get_mc2_tokens_limit():
    return _mc2_tokens_limit


def set_mc2_mask(vllm_config, device):
    global _reserved_mc2_mask
    if _reserved_mc2_mask is not None:
        if _is_a2_fused_bf16_configured(vllm_config):
            tp_size = vllm_config.parallel_config.tensor_parallel_size
            scheduler_capacity = _round_up_to_multiple(
                vllm_config.scheduler_config.max_num_batched_tokens,
                tp_size,
            )
            expected_capacity = max(_mc2_tokens_capacity or 0, scheduler_capacity)
            if _reserved_mc2_mask.numel() != expected_capacity or _reserved_mc2_mask.device != torch.device(device):
                raise RuntimeError(
                    "A2 BF16 fused MC2 active-mask storage was already initialized "
                    "for a different capacity or device."
                )
        return
    if is_moe_model(vllm_config):
        if _mc2_tokens_capacity is None:
            raise RuntimeError("MC2 token capacity must be initialized before its active mask")
        tp_size = vllm_config.parallel_config.tensor_parallel_size
        scheduler_capacity = _round_up_to_multiple(
            vllm_config.scheduler_config.max_num_batched_tokens,
            tp_size,
        )
        mask_capacity = max(_mc2_tokens_capacity, scheduler_capacity)
        _reserved_mc2_mask = torch.zeros(mask_capacity, dtype=torch.bool, device=device)
    else:
        _reserved_mc2_mask = None


def get_mc2_mask():
    return _reserved_mc2_mask


def get_mc2_padded_num_tokens(
    max_num_tokens: int,
    tp_world_size: int,
    moe_comm_type: MoECommType | None,
) -> int:
    padded_num_tokens = _round_up_to_multiple(max_num_tokens, tp_world_size)
    return padded_num_tokens


def select_moe_comm_method(num_tokens: int, vllm_config: VllmConfig, is_draft_model=False) -> MoECommType | None:
    """Select the MoE communication method according to parallel settings,
    device generation, token count, and quantization.

    1. Non-MoE models return `None`.
    2. Without expert parallel, fall back to all-gather.
    3. On A2 with expert parallel, use the floating-point fused MC2 path when
       explicitly enabled for a small EP group with enough local experts for
       the kernel pipeline. Otherwise, pick MC2 when tokens fit the MC2
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
    quant_type = _get_quant_type(vllm_config)

    if not vllm_config.parallel_config.enable_expert_parallel or get_ep_group().world_size == 1:
        moe_comm_type = MoECommType.ALLGATHER
    elif soc_version in {AscendDeviceType.A2}:
        num_experts = vllm_config.model_config.get_num_experts()
        ep_world_size = (
            vllm_config.parallel_config.world_size_across_dp // vllm_config.parallel_config.pipeline_parallel_size
        )
        num_experts_per_device = num_experts // ep_world_size
        fused_float_enable = _is_a2_fused_bf16_configured(vllm_config) and not is_draft_model
        if fused_float_enable:
            mc2_tokens_limit = get_mc2_tokens_limit()
            if mc2_tokens_limit is None:
                raise RuntimeError("A2 BF16 fused MC2 token domain was not initialized")
            if num_tokens > mc2_tokens_limit:
                raise RuntimeError(
                    "A2 BF16 fused MC2 received a batch outside the scheduler domain "
                    f"profiled at startup ({num_tokens} > {mc2_tokens_limit})."
                )
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
