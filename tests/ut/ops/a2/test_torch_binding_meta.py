# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm_ascend.utils import enable_custom_op

enable_custom_op()


def _moe_gating_top_k(x: torch.Tensor):
    return torch.ops._C_ascend.moe_gating_top_k(
        x,
        4,
        1,
        1,
        0,
        1,
        0,
        False,
        1.0,
        1e-20,
        None,
    )


def _moe_init_routing(x: torch.Tensor, expert_idx: torch.Tensor):
    return torch.ops._C_ascend.npu_moe_init_routing_custom(
        x,
        expert_idx,
        active_num=-1,
        expert_num=64,
        expert_tokens_num_type=1,
        expert_tokens_num_flag=True,
        active_expert_range=[0, 32],
        quant_mode=-1,
    )


def _dispatch_ffn_combine(x: torch.Tensor, x_active_mask: torch.Tensor):
    num_tokens, hidden_size = x.shape
    top_k = 2
    empty = torch.empty(0, device=x.device)
    expert_idx = torch.empty((num_tokens, top_k), device=x.device, dtype=torch.int32)
    probs = torch.empty((num_tokens, top_k), device=x.device, dtype=torch.float32)
    out = torch.empty((num_tokens, hidden_size), device=x.device, dtype=x.dtype)
    expert_token_nums = torch.empty(4, device=x.device, dtype=torch.int32)
    return torch.ops._C_ascend.dispatch_ffn_combine(
        x,
        [empty],
        [empty],
        expert_idx,
        [empty],
        [empty],
        [empty],
        [empty],
        probs,
        "meta-test-group",
        128,
        out,
        expert_token_nums,
        x_active_mask,
    )


def test_moe_gating_top_k_preserves_dynamic_token_shape():
    compiled_graphs: list[torch.fx.GraphModule] = []

    def record_graph(graph_module: torch.fx.GraphModule, _example_inputs):
        compiled_graphs.append(graph_module)
        return graph_module.forward

    torch._dynamo.reset()
    compiled = torch.compile(
        _moe_gating_top_k,
        backend=record_graph,
        dynamic=True,
        fullgraph=True,
    )

    try:
        for num_tokens in (7, 13):
            x = torch.empty((num_tokens, 64), device="meta", dtype=torch.float32)
            y, expert_idx, out = compiled(x)

            assert y.shape == (num_tokens, 4)
            assert expert_idx.shape == (num_tokens, 4)
            assert out.shape == (num_tokens, 64)
    finally:
        torch._dynamo.reset()

    assert len(compiled_graphs) == 1
    graph = compiled_graphs[0]
    input_node = next(
        node
        for node in graph.graph.nodes
        if node.op == "placeholder" and isinstance(node.meta.get("example_value"), torch.Tensor)
    )
    meta_node = next(
        node
        for node in graph.graph.nodes
        if node.op == "call_function" and node.target == torch.ops._C_ascend.moe_gating_top_k
    )

    input_token_dim = input_node.meta["example_value"].shape[0]
    meta_outputs = meta_node.meta["example_value"]
    assert isinstance(input_token_dim, torch.SymInt)
    assert all(isinstance(output.shape[0], torch.SymInt) for output in meta_outputs)
    assert all(str(output.shape[0]) == str(input_token_dim) for output in meta_outputs)


def test_moe_init_routing_dropless_preserves_dynamic_token_shape():
    compiled_graphs: list[torch.fx.GraphModule] = []

    def record_graph(graph_module: torch.fx.GraphModule, _example_inputs):
        compiled_graphs.append(graph_module)
        return graph_module.forward

    torch._dynamo.reset()
    compiled = torch.compile(
        _moe_init_routing,
        backend=record_graph,
        dynamic=True,
        fullgraph=True,
    )

    try:
        for num_tokens in (7, 13):
            x = torch.empty((num_tokens, 64), device="meta", dtype=torch.bfloat16)
            expert_idx = torch.empty((num_tokens, 4), device="meta", dtype=torch.int32)
            expanded_x, expanded_row_idx, expert_tokens, expanded_scale = compiled(x, expert_idx)

            assert expanded_x.shape == (num_tokens * 4, 64)
            assert expanded_row_idx.shape == (num_tokens * 4,)
            assert expert_tokens.shape == (32,)
            assert expanded_scale.shape == (num_tokens * 4,)
    finally:
        torch._dynamo.reset()

    assert len(compiled_graphs) == 1


def test_dispatch_ffn_combine_preserves_dynamic_mask_shape():
    compiled_graphs: list[torch.fx.GraphModule] = []

    def record_graph(graph_module: torch.fx.GraphModule, _example_inputs):
        compiled_graphs.append(graph_module)
        return graph_module.forward

    torch._dynamo.reset()
    compiled = torch.compile(
        _dispatch_ffn_combine,
        backend=record_graph,
        dynamic=True,
        fullgraph=True,
    )

    try:
        for num_tokens in (7, 13):
            x = torch.empty((num_tokens, 64), device="meta", dtype=torch.bfloat16)
            x_active_mask = torch.empty(num_tokens, device="meta", dtype=torch.bool)
            out, expert_token_nums = compiled(x, x_active_mask)

            assert out.shape == x.shape
            assert expert_token_nums.shape == (4,)
    finally:
        torch._dynamo.reset()

    assert len(compiled_graphs) == 1


def test_dispatch_ffn_combine_rejects_mismatched_mask_shape():
    x = torch.empty((7, 64), device="meta", dtype=torch.bfloat16)
    x_active_mask = torch.empty(6, device="meta", dtype=torch.bool)

    with pytest.raises(RuntimeError, match="x_active_mask must have one entry per input token"):
        _dispatch_ffn_combine(x, x_active_mask)
