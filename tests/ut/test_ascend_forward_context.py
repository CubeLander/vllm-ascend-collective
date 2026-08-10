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
    get_mrv2_in_profile_run,
    override_mrv2_in_profile_run,
    select_moe_comm_method,
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
    hf_text_config = SimpleNamespace()
    if quant_type is not None:
        hf_text_config.moe_quantize = quant_type
    return SimpleNamespace(
        model_config=SimpleNamespace(
            hf_text_config=hf_text_config,
            get_num_experts=lambda: num_experts,
        ),
        parallel_config=SimpleNamespace(
            enable_expert_parallel=True,
            world_size_across_dp=ep_size,
            pipeline_parallel_size=1,
        ),
    )


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
    ascend_config = SimpleNamespace(enable_fused_mc2=fused_mode)

    with (
        patch("vllm_ascend.ascend_forward_context.is_moe_model", return_value=True),
        patch("vllm_ascend.ascend_forward_context.get_mc2_tokens_capacity", return_value=64),
        patch("vllm_ascend.ascend_forward_context.get_ascend_device_type", return_value=AscendDeviceType.A2),
        patch("vllm_ascend.ascend_forward_context.get_ep_group", return_value=ep_group),
        patch("vllm_ascend.ascend_forward_context.get_ascend_config", return_value=ascend_config),
    ):
        assert select_moe_comm_method(32, vllm_config) is expected


def test_select_moe_comm_method_a2_fused_float_over_capacity_falls_back():
    """Fused float MC2 must fail closed above mc2_tokens_capacity.

    Regression test for the A2 review: the fused selector previously ignored
    mc2_tokens_capacity, so a fused MC2 kernel could be selected for token
    counts every other MC2 path treats as out-of-capacity. Above capacity the
    selector must fall back to the existing non-fused path (all-gather, since
    plain MC2 is also gated by the same capacity), while the capacity
    boundary itself keeps the fused path.
    """
    vllm_config = _make_moe_config(ep_size=2, num_experts=16)
    ep_group = SimpleNamespace(world_size=2)
    ascend_config = SimpleNamespace(enable_fused_mc2=1)

    with (
        patch("vllm_ascend.ascend_forward_context.is_moe_model", return_value=True),
        patch("vllm_ascend.ascend_forward_context.get_mc2_tokens_capacity", return_value=64),
        patch("vllm_ascend.ascend_forward_context.get_ascend_device_type", return_value=AscendDeviceType.A2),
        patch("vllm_ascend.ascend_forward_context.get_ep_group", return_value=ep_group),
        patch("vllm_ascend.ascend_forward_context.get_ascend_config", return_value=ascend_config),
    ):
        # Within capacity, including the boundary: fused path unchanged.
        assert select_moe_comm_method(32, vllm_config) is MoECommType.FUSED_MC2
        assert select_moe_comm_method(64, vllm_config) is MoECommType.FUSED_MC2
        # Over capacity: fail closed to the non-fused all-gather path.
        assert select_moe_comm_method(65, vllm_config) is MoECommType.ALLGATHER
        assert select_moe_comm_method(128, vllm_config) is MoECommType.ALLGATHER
