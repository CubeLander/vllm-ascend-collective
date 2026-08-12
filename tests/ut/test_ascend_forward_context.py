# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for ascend_forward_context module.

Verifies that get_mrv2_in_profile_run() and override_mrv2_in_profile_run()
work correctly with and without torch.compile(fullgraph=True).
"""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from vllm_ascend.ascend_forward_context import (
    MoECommType,
    _compute_mc2_tokens_capacity,
    get_mc2_padded_num_tokens,
    get_mrv2_in_profile_run,
    override_mrv2_in_profile_run,
    select_moe_comm_method,
    set_mc2_mask,
    set_mc2_tokens_capacity,
)
from vllm_ascend.utils import AscendDeviceType


class ModelWithProfileFlag(torch.nn.Module):
    """Minimal model that reads the MRv2 profile flag during forward."""

    def forward(self, x):
        if get_mrv2_in_profile_run():
            x = x * 2
        return x + 1


def test_default_false():
    """Default state must be False (no profile run)."""
    assert get_mrv2_in_profile_run() is False


def test_override_true():
    """Inside override_mrv2_in_profile_run(True), get() returns True."""
    with override_mrv2_in_profile_run(True):
        assert get_mrv2_in_profile_run() is True
    # Restored to default after exit
    assert get_mrv2_in_profile_run() is False


def test_override_false():
    """override_mrv2_in_profile_run(False) keeps the value as False."""
    with override_mrv2_in_profile_run(False):
        assert get_mrv2_in_profile_run() is False


def test_override_nested():
    """Nested context managers must restore correctly."""
    with override_mrv2_in_profile_run(True):
        assert get_mrv2_in_profile_run() is True
        with override_mrv2_in_profile_run(False):
            assert get_mrv2_in_profile_run() is False
        assert get_mrv2_in_profile_run() is True
    assert get_mrv2_in_profile_run() is False


def test_override_twice():
    """Sequential overrides work independently."""
    with override_mrv2_in_profile_run(True):
        assert get_mrv2_in_profile_run() is True
    assert get_mrv2_in_profile_run() is False
    with override_mrv2_in_profile_run(True):
        assert get_mrv2_in_profile_run() is True
    assert get_mrv2_in_profile_run() is False


def test_dynamo_fullgraph_compatible():
    """get_mrv2_in_profile_run() must work inside torch.compile(fullgraph=True).

    This is the core regression test for T22: ContextVar.get() is
    incompatible with torch.compile(fullgraph=True), but a module-level
    bool is not.  The fix replaced the original ContextVar with a plain
    module-level variable so that models compiled with fullgraph=True
    can read the profile-run flag without triggering a Dynamo error.
    """
    model = ModelWithProfileFlag()
    compiled = torch.compile(model, fullgraph=True, backend="eager")

    x = torch.randn(3, 3)

    # Default (flag=False): read + add 1, no multiply
    out_off = compiled(x)
    expected_off = x + 1
    assert torch.allclose(out_off, expected_off), f"Expected {expected_off}, got {out_off}"

    # Flag=True: read + multiply by 2 + add 1
    with override_mrv2_in_profile_run(True):
        out_on = compiled(x)
        expected_on = x * 2 + 1
        assert torch.allclose(out_on, expected_on), f"Expected {expected_on}, got {out_on}"


def test_dynamo_fullgraph_compatible_after_exit():
    """After exiting override context, compiled model still works.

    Verifies that the module-level flag restoration doesn't interfere
    with subsequent torch.compile calls.
    """
    model = ModelWithProfileFlag()
    compiled = torch.compile(model, fullgraph=True, backend="eager")

    x = torch.randn(3, 3)

    with override_mrv2_in_profile_run(True):
        compiled(x)  # warm up with flag=True

    # After exit, flag is False again
    out = compiled(x)
    expected = x + 1
    assert torch.allclose(out, expected), f"Expected {expected}, got {out}"


def test_override_isolated_between_calls():
    """override_mrv2_in_profile_run must be scoped per forward call.

    Two sequential forward calls with different override states should
    each see the correct flag value.
    """
    model = ModelWithProfileFlag()
    compiled = torch.compile(model, fullgraph=True, backend="eager")

    x = torch.randn(3, 3)

    # First call: flag=False
    out1 = compiled(x)
    assert torch.allclose(out1, x + 1)

    # Second call: flag=True
    with override_mrv2_in_profile_run(True):
        out2 = compiled(x)
        assert torch.allclose(out2, x * 2 + 1)

    # Third call: flag=False (restored)
    out3 = compiled(x)
    assert torch.allclose(out3, x + 1)


def _make_moe_config(ep_size: int, quant_type=None, num_experts: int = 16):
    hf_text_config = SimpleNamespace(num_experts_per_tok=8)
    if quant_type is not None:
        hf_text_config.moe_quantize = quant_type
    return SimpleNamespace(
        model_config=SimpleNamespace(
            dtype=torch.bfloat16,
            hf_text_config=hf_text_config,
            get_num_experts=lambda: num_experts,
        ),
        parallel_config=SimpleNamespace(
            enable_expert_parallel=True,
            world_size_across_dp=ep_size,
            pipeline_parallel_size=1,
            tensor_parallel_size=ep_size,
        ),
    )


def _make_capacity_config(max_num_batched_tokens=1024, tp_size=2, top_k=8, num_experts=None):
    if num_experts is None:
        num_experts = tp_size * 8
    return SimpleNamespace(
        scheduler_config=SimpleNamespace(max_num_batched_tokens=max_num_batched_tokens),
        compilation_config=SimpleNamespace(cudagraph_capture_sizes=[1024], max_cudagraph_capture_size=1024),
        parallel_config=SimpleNamespace(
            tensor_parallel_size=tp_size,
            enable_expert_parallel=True,
            world_size_across_dp=tp_size,
            pipeline_parallel_size=1,
        ),
        model_config=SimpleNamespace(
            dtype=torch.bfloat16,
            hf_text_config=SimpleNamespace(num_experts_per_tok=top_k),
            get_num_experts=lambda: num_experts,
        ),
    )


def _a2_fused_config():
    return SimpleNamespace(
        enable_fused_mc2=1,
        enable_prefill_mc2=False,
        eplb_config=SimpleNamespace(dynamic_eplb=False, num_redundant_experts=0),
    )


def test_a2_fused_capacity_covers_scheduler_domain():
    config = _make_capacity_config(max_num_batched_tokens=1024, tp_size=2)
    with (
        patch("vllm_ascend.ascend_forward_context.get_ascend_config", return_value=_a2_fused_config()),
        patch("vllm_ascend.ascend_forward_context.get_ascend_device_type", return_value=AscendDeviceType.A2),
    ):
        assert _compute_mc2_tokens_capacity(config, 32, 1) == 1024


def test_a2_fused_capacity_rejects_local_domain_above_512():
    config = _make_capacity_config(max_num_batched_tokens=1025, tp_size=2)
    with (
        patch("vllm_ascend.ascend_forward_context.get_ascend_config", return_value=_a2_fused_config()),
        patch("vllm_ascend.ascend_forward_context.get_ascend_device_type", return_value=AscendDeviceType.A2),
        pytest.raises(ValueError, match="supported limit 512|<= 512"),
    ):
        _compute_mc2_tokens_capacity(config, 32, 1)


def test_a2_fused_capacity_rejects_incompatible_runner_reinitialization(monkeypatch):
    import vllm_ascend.ascend_forward_context as forward_context

    monkeypatch.setattr(forward_context, "_mc2_tokens_capacity", None)
    monkeypatch.setattr(forward_context, "_mc2_tokens_limit", None)
    config = _make_capacity_config(max_num_batched_tokens=512, tp_size=2)
    with (
        patch("vllm_ascend.ascend_forward_context.get_ascend_config", return_value=_a2_fused_config()),
        patch("vllm_ascend.ascend_forward_context.get_ascend_device_type", return_value=AscendDeviceType.A2),
    ):
        set_mc2_tokens_capacity(config, 32, 1)
        config.scheduler_config.max_num_batched_tokens = 1024
        with pytest.raises(RuntimeError, match="different execution domain"):
            set_mc2_tokens_capacity(config, 32, 1)


def test_mc2_mask_covers_tp_rounded_scheduler_domain(monkeypatch):
    import vllm_ascend.ascend_forward_context as forward_context

    monkeypatch.setattr(forward_context, "_mc2_tokens_capacity", 512)
    monkeypatch.setattr(forward_context, "_reserved_mc2_mask", None)
    config = _make_capacity_config(max_num_batched_tokens=513, tp_size=8)
    with patch("vllm_ascend.ascend_forward_context.is_moe_model", return_value=True):
        set_mc2_mask(config, "cpu")
    assert forward_context.get_mc2_mask().shape == (520,)


@pytest.mark.parametrize(("num_tokens", "tp_size", "expected"), [(1, 1, 1), (1, 2, 2), (5, 2, 6)])
def test_a2_fused_padding_is_only_tp_alignment(num_tokens, tp_size, expected):
    with patch("vllm_ascend.ascend_forward_context.get_ascend_device_type", return_value=AscendDeviceType.A2):
        assert get_mc2_padded_num_tokens(num_tokens, tp_size, MoECommType.FUSED_MC2) == expected


@pytest.mark.parametrize(
    ("fused_mode", "quant_type", "ep_size", "local_experts", "expected"),
    [
        (1, None, 2, 8, MoECommType.FUSED_MC2),
        (1, None, 2, 2, MoECommType.ALLGATHER),
        (0, None, 2, 8, MoECommType.ALLGATHER),
        (1, "w8a8_dynamic", 2, 8, MoECommType.ALLGATHER),
        (1, None, 16, 8, MoECommType.MC2),
    ],
)
def test_select_moe_comm_method_a2_fused_float(fused_mode, quant_type, ep_size, local_experts, expected):
    vllm_config = _make_moe_config(ep_size, quant_type, num_experts=ep_size * local_experts)
    ep_group = SimpleNamespace(world_size=ep_size)
    ascend_config = SimpleNamespace(
        enable_fused_mc2=fused_mode,
        eplb_config=SimpleNamespace(dynamic_eplb=False, num_redundant_experts=0),
    )

    with (
        patch("vllm_ascend.ascend_forward_context.is_moe_model", return_value=True),
        patch("vllm_ascend.ascend_forward_context.get_mc2_tokens_capacity", return_value=64),
        patch("vllm_ascend.ascend_forward_context.get_mc2_tokens_limit", return_value=64),
        patch("vllm_ascend.ascend_forward_context.get_ascend_device_type", return_value=AscendDeviceType.A2),
        patch("vllm_ascend.ascend_forward_context.get_ep_group", return_value=ep_group),
        patch("vllm_ascend.ascend_forward_context.get_ascend_config", return_value=ascend_config),
    ):
        assert select_moe_comm_method(32, vllm_config) is expected


@pytest.mark.parametrize(
    ("dtype", "top_k", "is_draft", "dynamic_eplb", "expected"),
    [
        (torch.bfloat16, 8, False, False, MoECommType.FUSED_MC2),
        (torch.float16, 8, False, False, MoECommType.ALLGATHER),
        (torch.bfloat16, 9, False, False, MoECommType.ALLGATHER),
        (torch.bfloat16, 8, True, False, MoECommType.ALLGATHER),
        (torch.bfloat16, 8, False, True, MoECommType.ALLGATHER),
    ],
)
def test_select_moe_comm_method_a2_bf16_contract(dtype, top_k, is_draft, dynamic_eplb, expected):
    vllm_config = _make_moe_config(ep_size=2, num_experts=16)
    vllm_config.model_config.dtype = dtype
    vllm_config.model_config.hf_text_config.num_experts_per_tok = top_k
    ep_group = SimpleNamespace(world_size=2)
    ascend_config = SimpleNamespace(
        enable_fused_mc2=1,
        eplb_config=SimpleNamespace(dynamic_eplb=dynamic_eplb, num_redundant_experts=0),
    )

    with (
        patch("vllm_ascend.ascend_forward_context.is_moe_model", return_value=True),
        patch("vllm_ascend.ascend_forward_context.get_mc2_tokens_capacity", return_value=64),
        patch("vllm_ascend.ascend_forward_context.get_mc2_tokens_limit", return_value=64),
        patch("vllm_ascend.ascend_forward_context.get_ascend_device_type", return_value=AscendDeviceType.A2),
        patch("vllm_ascend.ascend_forward_context.get_ep_group", return_value=ep_group),
        patch("vllm_ascend.ascend_forward_context.get_ascend_config", return_value=ascend_config),
    ):
        assert select_moe_comm_method(32, vllm_config, is_draft_model=is_draft) is expected


@pytest.mark.parametrize("num_tokens", [1, 64])
def test_select_moe_comm_method_a2_bf16_keeps_family_inside_startup_domain(num_tokens):
    vllm_config = _make_moe_config(ep_size=2, num_experts=16)
    ep_group = SimpleNamespace(world_size=2)
    with (
        patch("vllm_ascend.ascend_forward_context.is_moe_model", return_value=True),
        patch("vllm_ascend.ascend_forward_context.get_mc2_tokens_capacity", return_value=64),
        patch("vllm_ascend.ascend_forward_context.get_mc2_tokens_limit", return_value=64),
        patch("vllm_ascend.ascend_forward_context.get_ascend_device_type", return_value=AscendDeviceType.A2),
        patch("vllm_ascend.ascend_forward_context.get_ep_group", return_value=ep_group),
        patch("vllm_ascend.ascend_forward_context.get_ascend_config", return_value=_a2_fused_config()),
    ):
        assert select_moe_comm_method(num_tokens, vllm_config) is MoECommType.FUSED_MC2


def test_select_moe_comm_method_a2_bf16_rejects_above_startup_domain():
    vllm_config = _make_moe_config(ep_size=2, num_experts=16)
    ep_group = SimpleNamespace(world_size=2)
    with (
        patch("vllm_ascend.ascend_forward_context.is_moe_model", return_value=True),
        patch("vllm_ascend.ascend_forward_context.get_mc2_tokens_capacity", return_value=64),
        patch("vllm_ascend.ascend_forward_context.get_mc2_tokens_limit", return_value=64),
        patch("vllm_ascend.ascend_forward_context.get_ascend_device_type", return_value=AscendDeviceType.A2),
        patch("vllm_ascend.ascend_forward_context.get_ep_group", return_value=ep_group),
        patch("vllm_ascend.ascend_forward_context.get_ascend_config", return_value=_a2_fused_config()),
        pytest.raises(RuntimeError, match="outside the scheduler domain"),
    ):
        select_moe_comm_method(65, vllm_config)
