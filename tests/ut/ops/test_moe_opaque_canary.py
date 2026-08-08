# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Focused CPU tests for the policy-local opaque-MoE canary.

The canary is an Ascend-side, ``moe_phase_hybrid_policy=non_decode_fused``
override of ``AscendMoERunner._select_forward``: it selects the
already-registered opaque ``torch.ops.vllm.moe_forward`` /
``torch.ops.vllm.moe_forward_shared`` custom-op entries instead of the raw
PrivateUse1 ``_moe_forward`` functions the stock selector returns on Ascend.
Policy OFF must keep ``super()._select_forward()`` byte-for-byte.

These tests only inspect the selection contract; they never run the opaque
op or trace a model. They require the non-0.23.0 ``AscendMoERunner`` (the
0.23.0 legacy class in ``fused_moe_0_23_0.py`` is a separate class without
the override).
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from vllm.model_executor.layers.fused_moe.runner.moe_runner import (
    _moe_forward,
    _moe_forward_shared,
)
from vllm.platforms import current_platform

from vllm_ascend.ascend_forward_context import MoEPhaseHybridPolicy
from vllm_ascend.ops.fused_moe import fused_moe as fused_moe_module
from vllm_ascend.ops.fused_moe.fused_moe import AscendMoERunner, MoERunner
from vllm_ascend.utils import vllm_version_is

if vllm_version_is("0.23.0"):
    pytest.skip(
        "Opaque-MoE canary UTs require the non-0.23.0 AscendMoERunner "
        "(the 0.23.0 legacy class has no _select_forward override).",
        allow_module_level=True,
    )


def _make_runner(has_shared_experts: bool = False) -> AscendMoERunner:
    """Build a runner shell without running the full MoERunner __init__.

    Only the attributes the selection contract reads are populated.
    """
    runner = AscendMoERunner.__new__(AscendMoERunner)
    runner.layer_name = "test.model.layers.0.moe"
    runner._shared_experts = object() if has_shared_experts else None
    return runner


def _policy_on_singleton(monkeypatch) -> None:
    """Policy ON resolved through the AscendConfig singleton fallback."""
    monkeypatch.setattr(
        fused_moe_module,
        "get_ascend_config",
        lambda: SimpleNamespace(moe_phase_hybrid_policy=MoEPhaseHybridPolicy.NON_DECODE_FUSED),
    )


def _policy_off_singleton(monkeypatch) -> None:
    """Policy OFF resolved through the AscendConfig singleton fallback."""
    monkeypatch.setattr(
        fused_moe_module,
        "get_ascend_config",
        lambda: SimpleNamespace(moe_phase_hybrid_policy=MoEPhaseHybridPolicy.OFF),
    )


@pytest.mark.parametrize("has_shared_experts", [False, True])
def test_select_forward_policy_on_selects_opaque_op(monkeypatch, has_shared_experts):
    """Policy ON returns the opaque torch.ops.vllm.moe_forward[_shared] entry."""
    _policy_on_singleton(monkeypatch)
    runner = _make_runner(has_shared_experts)
    expected = torch.ops.vllm.moe_forward_shared if has_shared_experts else torch.ops.vllm.moe_forward
    assert runner._select_forward() is expected


@pytest.mark.parametrize("has_shared_experts", [False, True])
def test_select_forward_policy_on_via_live_vllm_config(monkeypatch, has_shared_experts):
    """Policy ON resolved from the live vllm config (production path)."""
    monkeypatch.setattr(
        fused_moe_module,
        "get_current_vllm_config_or_none",
        lambda: SimpleNamespace(additional_config={"moe_phase_hybrid_policy": "non_decode_fused"}),
    )
    runner = _make_runner(has_shared_experts)
    expected = torch.ops.vllm.moe_forward_shared if has_shared_experts else torch.ops.vllm.moe_forward
    assert runner._select_forward() is expected


def test_select_forward_policy_off_delegates_to_super(monkeypatch):
    """Policy OFF must call super()._select_forward() and return its result."""
    _policy_off_singleton(monkeypatch)
    runner = _make_runner(has_shared_experts=False)
    super_entry = MagicMock(return_value="super-forward-entry")
    monkeypatch.setattr(fused_moe_module.MoERunner, "_select_forward", super_entry)
    assert runner._select_forward() == "super-forward-entry"
    # The super() call goes through once (not the opaque branch). Note the
    # MagicMock is not descriptor-bound via super(), so only the call count
    # and the return-value passthrough are asserted here; the real super
    # behavior on PrivateUse1 is covered by the raw-selection identity test.
    assert super_entry.call_count == 1


@pytest.mark.skipif(
    current_platform.dispatch_key != "PrivateUse1",
    reason="raw PrivateUse1 _moe_forward selection only holds on Ascend",
)
@pytest.mark.parametrize("has_shared_experts", [False, True])
def test_select_forward_policy_off_preserves_raw_privateuse1_selection(monkeypatch, has_shared_experts):
    """Policy OFF keeps the stock raw PrivateUse1 selection on Ascend."""
    _policy_off_singleton(monkeypatch)
    runner = _make_runner(has_shared_experts)
    expected = _moe_forward_shared if has_shared_experts else _moe_forward
    assert runner._select_forward() is expected


@pytest.mark.skipif(
    current_platform.dispatch_key != "PrivateUse1",
    reason="raw PrivateUse1 _moe_forward selection only holds on Ascend",
)
def test_select_forward_uninitialized_config_keeps_stock_selection(monkeypatch):
    """An unavailable policy config must never change stock selection."""
    monkeypatch.setattr(
        fused_moe_module,
        "get_ascend_config",
        MagicMock(side_effect=RuntimeError("Ascend config is not initialized")),
    )
    runner = _make_runner(has_shared_experts=False)
    assert runner._select_forward() is MoERunner._select_forward(runner)


def test_select_forward_policy_on_logs_info_once(monkeypatch):
    """The opaque selection marker is logged at most once per process."""
    _policy_on_singleton(monkeypatch)
    info = MagicMock()
    monkeypatch.setattr(fused_moe_module.logger, "info", info)
    # Reset the module-level once-flag so this test measures the marker
    # independently of earlier policy-ON tests.
    monkeypatch.setattr(fused_moe_module, "_opaque_moe_canary_logged", False)
    _make_runner(has_shared_experts=False)._select_forward()
    _make_runner(has_shared_experts=True)._select_forward()
    assert info.call_count == 1
    message = info.call_args.args[0]
    assert "torch.ops.vllm.moe_forward" in message
    assert "non_decode_fused" in message
