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
        # EP * local_experts is exactly 128: the active-mask sentinel must
        # expand the count-row stride instead of aliasing the next rank.
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

        x = x.npu()
        expert_idx = expert_idx.npu()
        probs = probs.npu()
        weight1_nz = [torch_npu.npu_format_cast(weight1.npu(), 29)]
        weight2_nz = [torch_npu.npu_format_cast(weight2.npu(), 29)]
        scale1 = [torch.empty(0, dtype=torch.int64)]
        scale2 = [torch.empty(0, dtype=torch.int64)]
        empty_bias = [torch.empty(0, dtype=torch.float32)]

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

        # Model graph replay: the physical batch is retained while only one
        # logical row is active. Inactive valid route IDs must contribute no
        # expert work, and the custom op must not overwrite its route input.
        x_active_mask = torch.zeros(tokens, dtype=torch.bool)
        x_active_mask[0] = True
        expected_masked_counts = torch.zeros(global_experts, dtype=torch.int32)
        for source_rank in range(world_size):
            source_routes = torch.arange(tokens * top_k, dtype=torch.int32).reshape(tokens, top_k)
            source_routes = (source_routes + source_rank * top_k) % global_experts
            expected_masked_counts += torch.bincount(source_routes[0], minlength=global_experts).to(torch.int32)
        expected_masked_counts = expected_masked_counts[
            rank * local_experts : (rank + 1) * local_experts
        ]

        expert_idx_before = expert_idx.clone()
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
            x_active_mask=x_active_mask.npu(),
            out=out,
            expert_token_nums=expert_token_nums,
        )
        torch_npu.npu.synchronize()
        torch.testing.assert_close(out[:1].cpu(), expected[:1], rtol=0.02, atol=0.02)
        torch.testing.assert_close(expert_token_nums.cpu(), expected_masked_counts)
        torch.testing.assert_close(expert_idx.cpu(), expert_idx_before.cpu())

        # The production domain includes a single physical row. This catches
        # idle-core tiling that previously divided by a zero per-core length.
        x_one = x[:1].clone()
        expert_idx_one = expert_idx[:1].clone()
        probs_one = probs[:1].clone()
        expected_one = expected[:1].clone()
        expected_one_counts = torch.zeros(global_experts, dtype=torch.int32)
        for source_rank in range(world_size):
            source_routes = (torch.arange(top_k, dtype=torch.int32) + source_rank * top_k) % global_experts
            expected_one_counts += torch.bincount(source_routes, minlength=global_experts).to(torch.int32)
        expected_one_counts = expected_one_counts[rank * local_experts : (rank + 1) * local_experts]
        out_one = torch.empty_like(x_one)
        expert_token_nums.fill_(-1)
        torch.ops._C_ascend.dispatch_ffn_combine(
            x=x_one,
            weight1=weight1_nz,
            weight2=weight2_nz,
            expert_idx=expert_idx_one,
            scale1=scale1,
            scale2=scale2,
            bias1=empty_bias,
            bias2=empty_bias,
            probs=probs_one,
            group=_get_hcomm_name(rank),
            max_output_size=world_size * top_k,
            out=out_one,
            expert_token_nums=expert_token_nums,
        )
        torch_npu.npu.synchronize()
        torch.testing.assert_close(out_one.cpu(), expected_one, rtol=0.02, atol=0.02)
        torch.testing.assert_close(expert_token_nums.cpu(), expected_one_counts)

        # Reuse after a masked generation must reset the full padded count
        # matrix: zero-count experts may not retain stale state.
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
    finally:
        dist.destroy_process_group()


@torch.inference_mode()
def test_dispatch_ffn_combine_bf16_two_ranks():
    world_size = 2
    port = 29501 + random.randint(0, 10000)
    mp.spawn(_run_rank, args=(world_size, port), nprocs=world_size, join=True)
