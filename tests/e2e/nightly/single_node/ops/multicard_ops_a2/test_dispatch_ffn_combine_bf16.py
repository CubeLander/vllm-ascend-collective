# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import random

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
import torch_npu
from torch.distributed.distributed_c10d import _get_default_group

from vllm_ascend.utils import enable_custom_op

enable_custom_op()


def _get_hcomm_name(rank: int) -> str:
    default_group = _get_default_group()
    backend = default_group._get_backend(torch.device("npu"))
    return backend.get_hccl_comm_name(rank)


def _run_rank(rank: int, world_size: int, port: int) -> None:
    torch_npu.npu.set_device(rank)
    dist.init_process_group(
        backend="hccl",
        rank=rank,
        world_size=world_size,
        init_method=f"tcp://127.0.0.1:{port}",
    )

    try:
        # Keep EP * local_experts exactly on the 128-entry alignment
        # boundary. Reserving the masked-row sentinel must therefore expand
        # the peer token-count stride rather than alias the next rank.
        local_experts = 64
        tokens = 64
        top_k = 4
        hidden_size = 256
        ffn_size = 256
        gate_up_size = 2 * ffn_size

        torch_npu.npu.config.allow_internal_format = True

        x = torch.zeros((tokens, hidden_size), dtype=torch.bfloat16)
        x[:, 0] = 1

        # Every expert receives the same one-hot gate/up input. GMM2 then
        # encodes the global expert id in output column zero, which makes
        # cross-rank dispatch and combine errors directly observable.
        weight1 = torch.zeros((local_experts, hidden_size, gate_up_size), dtype=torch.bfloat16)
        weight1[:, 0, 0] = 1
        weight1[:, 0, ffn_size] = 1
        weight2 = torch.zeros((local_experts, ffn_size, hidden_size), dtype=torch.bfloat16)
        for local_expert in range(local_experts):
            global_expert = rank * local_experts + local_expert
            weight2[local_expert, 0, 0] = global_expert + 1

        global_experts = world_size * local_experts
        expert_idx = torch.arange(tokens * top_k, dtype=torch.int32).reshape(tokens, top_k)
        expert_idx = (expert_idx + rank * top_k) % global_experts
        probs = torch.arange(1, top_k + 1, dtype=torch.float32).repeat(tokens, 1)
        probs /= probs.sum(dim=-1, keepdim=True)

        expected = torch.zeros((tokens, hidden_size), dtype=torch.bfloat16)
        silu_one = F.silu(torch.tensor(1, dtype=torch.bfloat16))
        expected[:, 0] = ((expert_idx + 1) * probs).sum(dim=-1).to(torch.bfloat16) * silu_one
        expected_tokens_per_local_expert = torch.full(
            (local_experts,), tokens * top_k * world_size // global_experts, dtype=torch.int32
        )

        probs_cpu = probs
        x = x.npu()
        expert_idx = expert_idx.npu()
        probs = probs.npu()
        weight1_nz = [torch_npu.npu_format_cast(weight1.npu(), 29)]
        weight2_nz = [torch_npu.npu_format_cast(weight2.npu(), 29)]
        # Keep the mandatory empty scale/bias placeholders on the same device
        # as the NPU input/weights, matching the real call path (which keeps
        # them on-device for torch.compile) instead of forcing implicit
        # host-to-device transfers.
        scale1 = [torch.empty(0, dtype=torch.int64).npu()]
        scale2 = [torch.empty(0, dtype=torch.int64).npu()]
        empty_bias = [torch.empty(0, dtype=torch.float32).npu()]

        out = torch.empty_like(x)
        expert_token_nums = torch.zeros(local_experts, dtype=torch.int32).npu()
        for _ in range(3):
            out.fill_(torch.nan)
            expert_token_nums.fill_(-1)
            torch.ops._C_ascend.dispatch_ffn_combine(
                x=x,
                weight1=weight1_nz,
                weight2=weight2_nz,
                expert_idx=expert_idx,
                scale1=scale1,
                scale2=scale2,
                bias1=empty_bias,
                bias2=empty_bias,
                probs=probs,
                group=_get_hcomm_name(rank),
                max_output_size=512,
                out=out,
                expert_token_nums=expert_token_nums,
            )
            torch_npu.npu.synchronize()

            torch.testing.assert_close(out.cpu(), expected, rtol=0.02, atol=0.02)
            torch.testing.assert_close(expert_token_nums.cpu(), expected_tokens_per_local_expert)

        # Reuse the same HCCL window with non-uniform, changing count rows.
        # This catches stale producer publications that a repeated uniform
        # assignment cannot distinguish from the current generation.
        for generation, active_experts in enumerate((8, 17, 65, 127), start=1):
            routes_by_rank = []
            for source_rank in range(world_size):
                source_routes = torch.arange(tokens * top_k, dtype=torch.int32)
                source_routes = (source_routes * 11 + source_rank * 23 + generation * 7) % active_experts
                source_routes = (source_routes + generation * 29) % global_experts
                routes_by_rank.append(source_routes.reshape(tokens, top_k))

            changing_expert_idx = routes_by_rank[rank]
            changing_expected = torch.zeros_like(expected)
            changing_expected[:, 0] = ((changing_expert_idx + 1) * probs_cpu).sum(dim=-1).to(torch.bfloat16) * silu_one
            changing_expected_counts = torch.bincount(
                torch.cat([routes.reshape(-1) for routes in routes_by_rank]),
                minlength=global_experts,
            ).to(torch.int32)
            changing_expected_counts = changing_expected_counts[rank * local_experts : (rank + 1) * local_experts]

            out.fill_(torch.nan)
            expert_token_nums.fill_(-1)
            torch.ops._C_ascend.dispatch_ffn_combine(
                x=x,
                weight1=weight1_nz,
                weight2=weight2_nz,
                expert_idx=changing_expert_idx.npu(),
                scale1=scale1,
                scale2=scale2,
                bias1=empty_bias,
                bias2=empty_bias,
                probs=probs,
                group=_get_hcomm_name(rank),
                max_output_size=512,
                out=out,
                expert_token_nums=expert_token_nums,
            )
            torch_npu.npu.synchronize()

            torch.testing.assert_close(out.cpu(), changing_expected, rtol=0.02, atol=0.02)
            torch.testing.assert_close(expert_token_nums.cpu(), changing_expected_counts)

        # Graph replay pads a one-token runtime batch to a captured shape.
        # The inactive rows must not contribute expert work even when their
        # route IDs name otherwise valid experts.
        active_tokens = 1
        x_active_mask = torch.zeros(tokens, dtype=torch.bool)
        x_active_mask[:active_tokens] = True
        masked_routes = []
        for source_rank in range(world_size):
            source_expert_idx = torch.arange(tokens * top_k, dtype=torch.int32).reshape(tokens, top_k)
            source_expert_idx = (source_expert_idx + source_rank * top_k) % global_experts
            masked_routes.append(source_expert_idx[:active_tokens].reshape(-1))
        expected_masked_counts = torch.bincount(torch.cat(masked_routes), minlength=global_experts).to(torch.int32)
        expected_masked_counts = expected_masked_counts[rank * local_experts : (rank + 1) * local_experts]

        masked_expert_idx = expert_idx.clone()
        out.fill_(torch.nan)
        expert_token_nums.fill_(-1)
        torch.ops._C_ascend.dispatch_ffn_combine(
            x=x,
            weight1=weight1_nz,
            weight2=weight2_nz,
            expert_idx=masked_expert_idx,
            scale1=scale1,
            scale2=scale2,
            bias1=empty_bias,
            bias2=empty_bias,
            probs=probs,
            group=_get_hcomm_name(rank),
            max_output_size=512,
            x_active_mask=x_active_mask.npu(),
            out=out,
            expert_token_nums=expert_token_nums,
        )
        torch_npu.npu.synchronize()

        torch.testing.assert_close(out[:active_tokens].cpu(), expected[:active_tokens], rtol=0.02, atol=0.02)
        torch.testing.assert_close(expert_token_nums.cpu(), expected_masked_counts)
    finally:
        dist.destroy_process_group()


@torch.inference_mode()
def test_dispatch_ffn_combine_bf16_two_ranks():
    world_size = 2
    port = 29501 + random.randint(0, 10000)
    mp.spawn(_run_rank, args=(world_size, port), nprocs=world_size, join=True)
