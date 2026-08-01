# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

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
