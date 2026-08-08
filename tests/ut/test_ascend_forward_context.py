# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for ascend_forward_context module.

Verifies that get_mrv2_in_profile_run() and override_mrv2_in_profile_run()
work correctly with and without torch.compile(fullgraph=True).
"""

from contextlib import ExitStack, contextmanager
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from vllm.config import CUDAGraphMode

from vllm_ascend.ascend_forward_context import (
    MoECommType,
    MoEForwardPhase,
    _select_moe_comm_method_stock,
    classify_forward_phase,
    get_moe_phase_hybrid_warmup_phase,
    get_mrv2_in_profile_run,
    is_moe_phase_hybrid_active,
    override_mrv2_in_profile_run,
    select_moe_comm_method,
    set_ascend_forward_context,
    should_force_eager_for_moe_phase_hybrid,
    should_skip_compiled_for_moe_phase_hybrid,
    validate_moe_phase_hybrid_policy,
)
from vllm_ascend.attention.attention_v1 import AscendAttentionState
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
        (1, "w8a8_dynamic", 2, 8, MoECommType.FUSED_MC2),
        (1, "w4a8_dynamic", 2, 8, MoECommType.ALLGATHER),
        (1, None, 16, 8, MoECommType.MC2),
    ],
)
def test_select_moe_comm_method_a2_fused(fused_mode, quant_type, ep_size, local_experts, expected):
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


# ---------------------------------------------------------------------------
# Phase-keyed hybrid MoE policy
# ---------------------------------------------------------------------------


def _with_hybrid_policy(vllm_config, value):
    """Attach an additional_config moe_phase_hybrid_policy value to the config."""
    additional_config = dict(getattr(vllm_config, "additional_config", None) or {})
    additional_config["moe_phase_hybrid_policy"] = value
    vllm_config.additional_config = additional_config
    return vllm_config


@contextmanager
def _patches(*mocks):
    """Enter several unittest.mock patches as one context manager."""
    with ExitStack() as stack:
        for mock in mocks:
            stack.enter_context(mock)
        yield


def test_classify_forward_phase_truth_table():
    assert classify_forward_phase(AscendAttentionState.DecodeOnly, False) is MoEForwardPhase.PURE_DECODE
    assert classify_forward_phase(AscendAttentionState.SpecDecoding, False) is MoEForwardPhase.PURE_DECODE
    assert classify_forward_phase(AscendAttentionState.PrefillNoCache, True) is MoEForwardPhase.PURE_PREFILL
    assert classify_forward_phase(AscendAttentionState.ChunkedPrefill, True) is MoEForwardPhase.MIXED
    assert classify_forward_phase(AscendAttentionState.PrefillCacheHit, True) is MoEForwardPhase.MIXED
    # with_prefill is the primary discriminator: a speculative-decode batch is
    # never treated as prefill even if attn_state were stale.
    assert classify_forward_phase(AscendAttentionState.ChunkedPrefill, False) is MoEForwardPhase.PURE_DECODE


def _make_hybrid_moe_config(
    ep_size: int,
    quant_type=None,
    num_experts: int = 16,
    *,
    cudagraph_mode=CUDAGraphMode.FULL_DECODE_ONLY,
    dp_size: int = 1,
    use_v2: bool = False,
):
    cfg = _make_moe_config(ep_size, quant_type, num_experts)
    cfg.parallel_config.data_parallel_size = dp_size
    cfg.compilation_config = SimpleNamespace(cudagraph_mode=cudagraph_mode)
    cfg.use_v2_model_runner = use_v2
    return cfg


def test_is_moe_phase_hybrid_active_non_decode_only():
    """PURE_PREFILL and MIXED are hybrid-active; PURE_DECODE is excluded."""
    vllm_config = _with_hybrid_policy(_make_hybrid_moe_config(8, num_experts=64), "non_decode_fused")
    with patch("vllm_ascend.ascend_forward_context.is_moe_model", return_value=True):
        assert is_moe_phase_hybrid_active(MoEForwardPhase.PURE_PREFILL, vllm_config) is True
        assert is_moe_phase_hybrid_active(MoEForwardPhase.MIXED, vllm_config) is True
        assert is_moe_phase_hybrid_active(MoEForwardPhase.PURE_DECODE, vllm_config) is False
        assert is_moe_phase_hybrid_active(None, vllm_config) is False
        # Draft model never hybrid-active (drafter keeps stock comm path).
        assert is_moe_phase_hybrid_active(MoEForwardPhase.MIXED, vllm_config, is_draft_model=True) is False
        with patch("vllm_ascend.ascend_forward_context.is_moe_model", return_value=False):
            assert is_moe_phase_hybrid_active(MoEForwardPhase.MIXED, vllm_config) is False


def test_should_skip_compiled_for_moe_phase_hybrid_decode_only():
    """skip_compiled under the policy applies ONLY to PURE_DECODE: decode
    runs the raw baseline comm ops so FULL_DECODE_ONLY ACL graphs capture and
    replay exactly those raw ops. PURE_PREFILL/MIXED keep the compiled
    (fused torch.compile) path."""
    vllm_config = _with_hybrid_policy(_make_hybrid_moe_config(8, num_experts=64), "non_decode_fused")
    with patch("vllm_ascend.ascend_forward_context.is_moe_model", return_value=True):
        assert should_skip_compiled_for_moe_phase_hybrid(MoEForwardPhase.PURE_DECODE, vllm_config) is True
        assert should_skip_compiled_for_moe_phase_hybrid(MoEForwardPhase.PURE_PREFILL, vllm_config) is False
        assert should_skip_compiled_for_moe_phase_hybrid(MoEForwardPhase.MIXED, vllm_config) is False
        assert should_skip_compiled_for_moe_phase_hybrid(None, vllm_config) is False
        assert (
            should_skip_compiled_for_moe_phase_hybrid(MoEForwardPhase.PURE_DECODE, vllm_config, is_draft_model=True)
            is False
        )
        with patch("vllm_ascend.ascend_forward_context.is_moe_model", return_value=False):
            assert should_skip_compiled_for_moe_phase_hybrid(MoEForwardPhase.PURE_DECODE, vllm_config) is False
    off_config = _with_hybrid_policy(_make_hybrid_moe_config(8, num_experts=64), "off")
    with patch("vllm_ascend.ascend_forward_context.is_moe_model", return_value=True):
        assert should_skip_compiled_for_moe_phase_hybrid(MoEForwardPhase.PURE_DECODE, off_config) is False
        assert should_skip_compiled_for_moe_phase_hybrid(MoEForwardPhase.PURE_PREFILL, off_config) is False


def test_should_force_eager_for_moe_phase_hybrid_non_decode_only():
    """force_eager under the policy applies ONLY to PURE_PREFILL and MIXED:
    non-decode waves must never be dispatched into (or replayed from) a
    FULL template, even when uniform_decode misclassifies a 1-token prefill.
    PURE_DECODE keeps force_eager=False so FULL capture/replay proceeds."""
    vllm_config = _with_hybrid_policy(_make_hybrid_moe_config(8, num_experts=64), "non_decode_fused")
    with patch("vllm_ascend.ascend_forward_context.is_moe_model", return_value=True):
        assert should_force_eager_for_moe_phase_hybrid(MoEForwardPhase.PURE_PREFILL, vllm_config) is True
        assert should_force_eager_for_moe_phase_hybrid(MoEForwardPhase.MIXED, vllm_config) is True
        assert should_force_eager_for_moe_phase_hybrid(MoEForwardPhase.PURE_DECODE, vllm_config) is False
        assert should_force_eager_for_moe_phase_hybrid(None, vllm_config) is False
        assert (
            should_force_eager_for_moe_phase_hybrid(MoEForwardPhase.MIXED, vllm_config, is_draft_model=True) is False
        )
        with patch("vllm_ascend.ascend_forward_context.is_moe_model", return_value=False):
            assert should_force_eager_for_moe_phase_hybrid(MoEForwardPhase.MIXED, vllm_config) is False
    off_config = _with_hybrid_policy(_make_hybrid_moe_config(8, num_experts=64), "off")
    with patch("vllm_ascend.ascend_forward_context.is_moe_model", return_value=True):
        assert should_force_eager_for_moe_phase_hybrid(MoEForwardPhase.MIXED, off_config) is False
        assert should_force_eager_for_moe_phase_hybrid(MoEForwardPhase.PURE_PREFILL, off_config) is False


def test_moe_phase_hybrid_2x2_matrix_invariant():
    """The two phase-keyed decisions are independent and never combine.

    Under the policy each phase gets exactly one of the two controls:
    PURE_DECODE -> (skip_compiled=True, force_eager=False); PURE_PREFILL /
    MIXED -> (skip_compiled=False, force_eager=True). skip_compiled is never
    fed into force_eager. Policy off and phase=None get neither control.
    """
    vllm_config = _with_hybrid_policy(_make_hybrid_moe_config(8, num_experts=64), "non_decode_fused")
    with patch("vllm_ascend.ascend_forward_context.is_moe_model", return_value=True):
        for phase, expected_skip, expected_eager in [
            (MoEForwardPhase.PURE_DECODE, True, False),
            (MoEForwardPhase.PURE_PREFILL, False, True),
            (MoEForwardPhase.MIXED, False, True),
            (None, False, False),
        ]:
            skip = should_skip_compiled_for_moe_phase_hybrid(phase, vllm_config)
            eager = should_force_eager_for_moe_phase_hybrid(phase, vllm_config)
            assert skip is expected_skip, f"phase={phase} skip_compiled"
            assert eager is expected_eager, f"phase={phase} force_eager"
            assert not (skip and eager), f"phase={phase} combines both controls"
    off_config = _with_hybrid_policy(_make_hybrid_moe_config(8, num_experts=64), "off")
    with patch("vllm_ascend.ascend_forward_context.is_moe_model", return_value=True):
        for phase in MoEForwardPhase:
            assert should_skip_compiled_for_moe_phase_hybrid(phase, off_config) is False
            assert should_force_eager_for_moe_phase_hybrid(phase, off_config) is False


def test_get_moe_phase_hybrid_warmup_phase():
    """Profile/compile warmups get an explicit non-decode phase only under
    the active policy; otherwise the caller keeps its historical default."""
    vllm_config = _with_hybrid_policy(_make_hybrid_moe_config(8, num_experts=64), "non_decode_fused")
    with patch("vllm_ascend.ascend_forward_context.is_moe_model", return_value=True):
        assert get_moe_phase_hybrid_warmup_phase(vllm_config) is MoEForwardPhase.MIXED
        assert get_moe_phase_hybrid_warmup_phase(vllm_config, is_draft_model=True) is None
    with patch("vllm_ascend.ascend_forward_context.is_moe_model", return_value=False):
        assert get_moe_phase_hybrid_warmup_phase(vllm_config) is None
    off_config = _with_hybrid_policy(_make_hybrid_moe_config(8, num_experts=64), "off")
    with patch("vllm_ascend.ascend_forward_context.is_moe_model", return_value=True):
        assert get_moe_phase_hybrid_warmup_phase(off_config) is None


def test_is_moe_phase_hybrid_active_off_policy():
    vllm_config = _with_hybrid_policy(_make_hybrid_moe_config(8, num_experts=64), "off")
    with patch("vllm_ascend.ascend_forward_context.is_moe_model", return_value=True):
        assert is_moe_phase_hybrid_active(MoEForwardPhase.MIXED, vllm_config) is False
        assert is_moe_phase_hybrid_active(MoEForwardPhase.PURE_PREFILL, vllm_config) is False


def test_validate_moe_phase_hybrid_policy_off_passes_everything():
    """Policy OFF must never raise regardless of the execution envelope."""
    vllm_config = _with_hybrid_policy(
        _make_hybrid_moe_config(
            8,
            num_experts=64,
            cudagraph_mode=CUDAGraphMode.FULL,
            dp_size=4,
            use_v2=True,
        ),
        "off",
    )
    validate_moe_phase_hybrid_policy(vllm_config)  # must not raise


@pytest.mark.parametrize("enable_fused_mc2", [0, 2])
def test_validate_moe_phase_hybrid_policy_fail_closed_fused_mc2_disabled(enable_fused_mc2):
    """The fused impl and NZ weights only exist with enable_fused_mc2==1;
    without it the policy would silently stay on the baseline family."""
    vllm_config = _with_hybrid_policy(_make_hybrid_moe_config(8, num_experts=64), "non_decode_fused")
    with (
        patch(
            "vllm_ascend.ascend_forward_context.get_ascend_config",
            return_value=SimpleNamespace(enable_fused_mc2=enable_fused_mc2),
        ),
        pytest.raises(ValueError, match="enable_fused_mc2 == 1"),
    ):
        validate_moe_phase_hybrid_policy(vllm_config)


def test_validate_moe_phase_hybrid_policy_fail_closed_dp():
    vllm_config = _with_hybrid_policy(_make_hybrid_moe_config(8, num_experts=64, dp_size=2), "non_decode_fused")
    with (
        patch("vllm_ascend.ascend_forward_context.get_ascend_config", return_value=SimpleNamespace(enable_fused_mc2=1)),
        pytest.raises(ValueError, match="data_parallel_size == 1"),
    ):
        validate_moe_phase_hybrid_policy(vllm_config)


def test_validate_moe_phase_hybrid_policy_fail_closed_v2():
    vllm_config = _with_hybrid_policy(_make_hybrid_moe_config(8, num_experts=64, use_v2=True), "non_decode_fused")
    with (
        patch("vllm_ascend.ascend_forward_context.get_ascend_config", return_value=SimpleNamespace(enable_fused_mc2=1)),
        pytest.raises(ValueError, match="V2 model runner"),
    ):
        validate_moe_phase_hybrid_policy(vllm_config)


@pytest.mark.parametrize(
    "mode",
    [
        CUDAGraphMode.NONE,
        CUDAGraphMode.FULL,
        CUDAGraphMode.PIECEWISE,
        CUDAGraphMode.FULL_AND_PIECEWISE,
    ],
)
def test_validate_moe_phase_hybrid_policy_fail_closed_mode(mode):
    vllm_config = _with_hybrid_policy(
        _make_hybrid_moe_config(8, num_experts=64, cudagraph_mode=mode), "non_decode_fused"
    )
    with (
        patch("vllm_ascend.ascend_forward_context.get_ascend_config", return_value=SimpleNamespace(enable_fused_mc2=1)),
        pytest.raises(ValueError, match="FULL_DECODE_ONLY"),
    ):
        validate_moe_phase_hybrid_policy(vllm_config)


def test_validate_moe_phase_hybrid_policy_full_decode_only_passes():
    vllm_config = _with_hybrid_policy(_make_hybrid_moe_config(8, num_experts=64), "non_decode_fused")
    with (
        patch("vllm_ascend.ascend_forward_context.get_ascend_config", return_value=SimpleNamespace(enable_fused_mc2=1)),
    ):
        validate_moe_phase_hybrid_policy(vllm_config)  # must not raise


def _a2_selector_env(enable_fused_mc2: int, ep_size: int, quant_type=None):
    ep_group = SimpleNamespace(world_size=ep_size)
    ascend_config = SimpleNamespace(enable_fused_mc2=enable_fused_mc2)
    return (
        patch("vllm_ascend.ascend_forward_context.is_moe_model", return_value=True),
        patch("vllm_ascend.ascend_forward_context.get_mc2_tokens_capacity", return_value=64),
        patch("vllm_ascend.ascend_forward_context.get_ascend_device_type", return_value=AscendDeviceType.A2),
        patch("vllm_ascend.ascend_forward_context.get_ep_group", return_value=ep_group),
        patch("vllm_ascend.ascend_forward_context.get_ascend_config", return_value=ascend_config),
    )


def _a3_selector_env(enable_fused_mc2: int, ep_size: int):
    ep_group = SimpleNamespace(world_size=ep_size)
    ascend_config = SimpleNamespace(enable_fused_mc2=enable_fused_mc2)
    return (
        patch("vllm_ascend.ascend_forward_context.is_moe_model", return_value=True),
        patch("vllm_ascend.ascend_forward_context.get_mc2_tokens_capacity", return_value=64),
        patch("vllm_ascend.ascend_forward_context.get_ascend_device_type", return_value=AscendDeviceType.A3),
        patch("vllm_ascend.ascend_forward_context.get_ep_group", return_value=ep_group),
        patch("vllm_ascend.ascend_forward_context.get_ascend_config", return_value=ascend_config),
    )


def test_select_moe_comm_method_hybrid_off_identical_for_all_phases():
    """Default compatibility: with policy OFF every phase returns the stock result."""
    vllm_config = _with_hybrid_policy(_make_hybrid_moe_config(16, num_experts=256), "off")
    with _patches(*_a3_selector_env(1, 16)):
        stock = select_moe_comm_method(32, vllm_config)
        for phase in MoEForwardPhase:
            assert select_moe_comm_method(32, vllm_config, phase=phase) is stock
        assert select_moe_comm_method(32, vllm_config, phase=None) is stock


def test_select_moe_comm_method_hybrid_a2_tp2_experiment():
    """A2 TP2 experiment (EP=2, enable_fused_mc2=1): decode must select the
    baseline ALLGATHER -- NOT the stock FUSED_MC2 -- while PURE_PREFILL and
    MIXED select FUSED_MC2. This is the core regression for the decode
    baseline blocker."""
    vllm_config = _with_hybrid_policy(_make_hybrid_moe_config(2, num_experts=16), "non_decode_fused")
    with _patches(*_a2_selector_env(1, 2)):
        assert select_moe_comm_method(32, vllm_config, phase=MoEForwardPhase.PURE_DECODE) is MoECommType.ALLGATHER
        assert select_moe_comm_method(32, vllm_config, phase=MoEForwardPhase.PURE_PREFILL) is MoECommType.FUSED_MC2
        assert select_moe_comm_method(32, vllm_config, phase=MoEForwardPhase.MIXED) is MoECommType.FUSED_MC2
        # Stock would have picked FUSED_MC2 for decode here; the policy must
        # not leak fused into the decode path.
        assert _select_moe_comm_method_stock(32, vllm_config) is MoECommType.FUSED_MC2


def test_select_moe_comm_method_hybrid_a2_fused_disabled_fail_fast():
    """A2 enable_fused_mc2=0: decode keeps the baseline ALLGATHER, but a
    non-decode phase must NEVER fall back to baseline -- the selector raises
    fail-fast because the static fused gate is unavailable."""
    vllm_config = _with_hybrid_policy(_make_hybrid_moe_config(2, num_experts=16), "non_decode_fused")
    with _patches(*_a2_selector_env(0, 2)):
        assert select_moe_comm_method(32, vllm_config, phase=MoEForwardPhase.PURE_DECODE) is MoECommType.ALLGATHER
        for phase in (MoEForwardPhase.PURE_PREFILL, MoEForwardPhase.MIXED):
            with pytest.raises(ValueError, match="requires the static fused-MC2 capability gate"):
                select_moe_comm_method(32, vllm_config, phase=phase)


def test_select_moe_comm_method_hybrid_a2_gate_fails_fail_fast():
    """A2 EP=16 is beyond the fused gate (EP<=8): decode keeps the baseline
    MC2 family, but PURE_PREFILL and MIXED raise fail-fast instead of
    falling back to it (a baseline non-decode wave inside the single fused
    lane is exactly the stale-template failure the policy prevents)."""
    vllm_config = _with_hybrid_policy(_make_hybrid_moe_config(16, num_experts=128), "non_decode_fused")
    with _patches(*_a2_selector_env(1, 16)):
        # baseline: experts/device=8 <= 24, EP>=16, tokens within capacity -> MC2
        assert select_moe_comm_method(32, vllm_config, phase=MoEForwardPhase.PURE_DECODE) is MoECommType.MC2
        for phase in (MoEForwardPhase.PURE_PREFILL, MoEForwardPhase.MIXED):
            with pytest.raises(ValueError, match="requires the static fused-MC2 capability gate"):
                select_moe_comm_method(32, vllm_config, phase=phase)
    # Policy OFF in the same unsupported environment stays stock (no raise).
    off_config = _with_hybrid_policy(_make_hybrid_moe_config(16, num_experts=128), "off")
    with _patches(*_a2_selector_env(1, 16)):
        assert select_moe_comm_method(32, off_config, phase=MoEForwardPhase.PURE_PREFILL) is MoECommType.MC2


def test_select_moe_comm_method_hybrid_a3_decode_baseline_not_stock_fused():
    """A3 enable_fused_mc2==1, EP<=32, within capacity: stock decode would be
    FUSED_MC2, but the policy must select the baseline MC2 for decode, while
    PURE_PREFILL and MIXED select FUSED_MC2."""
    vllm_config = _with_hybrid_policy(_make_hybrid_moe_config(8, num_experts=64), "non_decode_fused")
    with _patches(*_a3_selector_env(1, 8)):
        assert _select_moe_comm_method_stock(32, vllm_config) is MoECommType.FUSED_MC2
        assert select_moe_comm_method(32, vllm_config, phase=MoEForwardPhase.PURE_DECODE) is MoECommType.MC2
        assert select_moe_comm_method(32, vllm_config, phase=MoEForwardPhase.PURE_PREFILL) is MoECommType.FUSED_MC2
        assert select_moe_comm_method(32, vllm_config, phase=MoEForwardPhase.MIXED) is MoECommType.FUSED_MC2


def test_select_moe_comm_method_hybrid_a3_capacity_beyond():
    """A3 beyond MC2 capacity: decode baseline is ALLTOALL; non-decode stays
    fused."""
    vllm_config = _with_hybrid_policy(_make_hybrid_moe_config(8, num_experts=64), "non_decode_fused")
    with _patches(*_a3_selector_env(1, 8)):
        assert select_moe_comm_method(128, vllm_config, phase=MoEForwardPhase.PURE_DECODE) is MoECommType.ALLTOALL
        assert select_moe_comm_method(128, vllm_config, phase=MoEForwardPhase.PURE_PREFILL) is MoECommType.FUSED_MC2
        assert select_moe_comm_method(128, vllm_config, phase=MoEForwardPhase.MIXED) is MoECommType.FUSED_MC2


def test_select_moe_comm_method_hybrid_a3_enable_fused_mc2_eq_2_fail_fast():
    """A3 enable_fused_mc2==2 is outside the non-decode gate (==1 required):
    decode keeps the baseline MC2, but non-decode phases raise fail-fast."""
    vllm_config = _with_hybrid_policy(_make_hybrid_moe_config(8, num_experts=64), "non_decode_fused")
    with _patches(*_a3_selector_env(2, 8)):
        assert select_moe_comm_method(32, vllm_config, phase=MoEForwardPhase.PURE_DECODE) is MoECommType.MC2
        for phase in (MoEForwardPhase.PURE_PREFILL, MoEForwardPhase.MIXED):
            with pytest.raises(ValueError, match="requires the static fused-MC2 capability gate"):
                select_moe_comm_method(32, vllm_config, phase=phase)


def test_select_moe_comm_method_hybrid_a3_ep_beyond_gate_fail_fast():
    """A3 EP=64 exceeds the dispatch_ffn_combine gate (EP<=32): decode keeps
    the baseline MC2 family, non-decode raises fail-fast."""
    vllm_config = _with_hybrid_policy(_make_hybrid_moe_config(64, num_experts=512), "non_decode_fused")
    with _patches(*_a3_selector_env(1, 64)):
        assert select_moe_comm_method(32, vllm_config, phase=MoEForwardPhase.PURE_DECODE) is MoECommType.MC2
        for phase in (MoEForwardPhase.PURE_PREFILL, MoEForwardPhase.MIXED):
            with pytest.raises(ValueError, match="requires the static fused-MC2 capability gate"):
                select_moe_comm_method(32, vllm_config, phase=phase)


def test_select_moe_comm_method_hybrid_a5_decode_stock_non_decode_fail_fast():
    """A5 has no fused branch: decode keeps stock/baseline, but non-decode
    phases under the policy raise fail-fast (never a baseline fallback)."""
    vllm_config = _with_hybrid_policy(_make_hybrid_moe_config(4, num_experts=64), "non_decode_fused")
    ep_group = SimpleNamespace(world_size=4)
    ascend_config = SimpleNamespace(enable_fused_mc2=1)
    with (
        patch("vllm_ascend.ascend_forward_context.is_moe_model", return_value=True),
        patch("vllm_ascend.ascend_forward_context.get_mc2_tokens_capacity", return_value=64),
        patch("vllm_ascend.ascend_forward_context.get_ascend_device_type", return_value=AscendDeviceType.A5),
        patch("vllm_ascend.ascend_forward_context.get_ep_group", return_value=ep_group),
        patch("vllm_ascend.ascend_forward_context.get_ascend_config", return_value=ascend_config),
    ):
        stock = select_moe_comm_method(32, vllm_config)
        assert select_moe_comm_method(32, vllm_config, phase=MoEForwardPhase.PURE_DECODE) is stock
        for phase in (MoEForwardPhase.PURE_PREFILL, MoEForwardPhase.MIXED):
            with pytest.raises(ValueError, match="requires the static fused-MC2 capability gate"):
                select_moe_comm_method(32, vllm_config, phase=phase)


def test_select_moe_comm_method_hybrid_non_moe_returns_none():
    vllm_config = _with_hybrid_policy(_make_hybrid_moe_config(8, num_experts=64), "non_decode_fused")
    with (
        patch("vllm_ascend.ascend_forward_context.is_moe_model", return_value=False),
        patch("vllm_ascend.ascend_forward_context.get_ascend_device_type", return_value=AscendDeviceType.A3),
    ):
        assert select_moe_comm_method(32, vllm_config, phase=MoEForwardPhase.MIXED) is None


def test_select_moe_comm_method_hybrid_draft_model_stock():
    """Draft models are outside the hybrid envelope: always stock."""
    vllm_config = _with_hybrid_policy(_make_hybrid_moe_config(8, num_experts=64), "non_decode_fused")
    with _patches(*_a3_selector_env(1, 8)):
        draft_prefill = select_moe_comm_method(32, vllm_config, is_draft_model=True, phase=MoEForwardPhase.MIXED)
        assert draft_prefill is MoECommType.FUSED_MC2  # stock (A3==1 within capacity), phase ignored
        assert (
            select_moe_comm_method(32, vllm_config, is_draft_model=False, phase=MoEForwardPhase.MIXED)
            is MoECommType.FUSED_MC2
        )


def test_select_moe_comm_method_hybrid_phase_none_stock():
    """Legacy callers without a phase keep stock selection."""
    vllm_config = _with_hybrid_policy(_make_hybrid_moe_config(8, num_experts=64), "non_decode_fused")
    with _patches(*_a3_selector_env(1, 8)):
        assert select_moe_comm_method(32, vllm_config) is MoECommType.FUSED_MC2  # stock


def _make_forward_context_mocks():
    forward_context = SimpleNamespace(dp_metadata=None)
    return (
        forward_context,
        (
            patch("vllm_ascend.ascend_forward_context.get_forward_context", return_value=forward_context),
            patch("vllm_ascend.ascend_forward_context.get_dp_group", return_value=SimpleNamespace(world_size=1)),
            patch("vllm_ascend.ascend_forward_context.get_tensor_model_parallel_world_size", return_value=1),
            patch(
                "vllm_ascend.ascend_forward_context.get_ascend_config", return_value=SimpleNamespace(enable_fused_mc2=1)
            ),
            patch("vllm_ascend.ascend_forward_context.is_moe_model", return_value=True),
            patch("vllm_ascend.ascend_forward_context.is_drafter_moe_model", return_value=False),
            patch("vllm_ascend.ascend_forward_context.enable_sp", return_value=False),
            patch("vllm_ascend.ascend_forward_context.flashcomm2_enable", return_value=False),
            patch("vllm_ascend.ascend_forward_context.has_layer_idx", return_value=False),
            patch("vllm_ascend.ascend_forward_context.get_mc2_mask", return_value=None),
            patch("vllm_ascend.ascend_forward_context.get_mc2_tokens_capacity", return_value=64),
            patch("vllm_ascend.ascend_forward_context.get_ascend_device_type", return_value=AscendDeviceType.A3),
            patch("vllm_ascend.ascend_forward_context.get_ep_group", return_value=SimpleNamespace(world_size=8)),
        ),
    )


@pytest.mark.parametrize(
    ("policy_value", "phase", "expected_skip_compiled", "expected_comm"),
    [
        ("off", MoEForwardPhase.PURE_DECODE, False, MoECommType.FUSED_MC2),
        ("off", MoEForwardPhase.PURE_PREFILL, False, MoECommType.FUSED_MC2),
        ("off", MoEForwardPhase.MIXED, False, MoECommType.FUSED_MC2),
        ("non_decode_fused", MoEForwardPhase.PURE_DECODE, True, MoECommType.MC2),
        ("non_decode_fused", MoEForwardPhase.PURE_PREFILL, False, MoECommType.FUSED_MC2),
        ("non_decode_fused", MoEForwardPhase.MIXED, False, MoECommType.FUSED_MC2),
        ("non_decode_fused", None, False, MoECommType.FUSED_MC2),
    ],
)
def test_set_ascend_forward_context_hybrid_skip_compiled(policy_value, phase, expected_skip_compiled, expected_comm):
    """skip_compiled=True under the policy ONLY for PURE_DECODE, so the
    FULL_DECODE_ONLY ACL graphs capture raw baseline ops. PURE_PREFILL/MIXED
    keep the compiled (fused torch.compile) path and select FUSED_MC2."""
    captured = {}

    @contextmanager
    def _fake_set_forward_context(**kwargs):
        captured.update(kwargs)
        yield

    vllm_config = _with_hybrid_policy(_make_hybrid_moe_config(8, num_experts=64), policy_value)
    forward_context, mocks = _make_forward_context_mocks()
    with _patches(
        patch("vllm_ascend.ascend_forward_context.set_forward_context", side_effect=_fake_set_forward_context),
        *mocks,
        set_ascend_forward_context(
            attn_metadata=None,
            vllm_config=vllm_config,
            num_tokens=32,
            model_instance=SimpleNamespace(),
            forward_phase=phase,
        ),
    ):
        pass

    assert captured["skip_compiled"] is expected_skip_compiled
    # Comm family follows the selector: PURE_DECODE -> baseline (A3 MC2
    # within capacity), PURE_PREFILL/MIXED -> FUSED_MC2, off/None -> stock.
    assert forward_context.moe_comm_type is expected_comm
