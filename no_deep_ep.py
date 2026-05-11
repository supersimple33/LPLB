"""
Test the distributed planner.

Run with torchrun or something on 2 nodes, each with 8 GPUs.
"""
import os

import torch
import torch.distributed

from lplb import Planner
from utils import (
    SQUARE_4P2E,
    bench_kineto,
)


def test_dist_planner_solve(
    r2o: torch.Tensor,
    n_logical_experts: int,
    ep_size: int,
    n_redundants_per_rank: int,
) -> None:
    r2o = r2o.int().cuda()
    planner = Planner(
        r2o,
        n_logical_experts + n_redundants_per_rank * ep_size,
        n_logical_experts,
        ep_size=ep_size,  # Add this explicit argument
        group=torch.distributed.GroupMember.WORLD,
    )

    global_rank = torch.distributed.get_rank()
    torch.cuda.random.manual_seed(42 + global_rank)
    workload_history = torch.randn(n_logical_experts, device='cuda') * 0.15 + 1
    workload_history = workload_history.clamp_min(0).mul(2**20).int()
    torch.distributed.broadcast(workload_history, src=0)
    phy2log, _log2phy, _logcnt = planner.update_redundancy_mapping(workload_history)

    current_workload = torch.randn(n_logical_experts, device='cuda') * 1 + 1
    current_workload = current_workload.clamp_min(0).mul(2**12).int()
    # print(f'[{global_rank}] before solve {current_workload=}')
    avail_counter = torch.zeros((), dtype=torch.int, device='cuda')
    probs, global_workload_kernel = planner.solve_probs(current_workload, avail_counter)

    # print(f'[{global_rank}] {probs=}')

    global_workload = current_workload.clone()
    torch.distributed.all_reduce(global_workload)

    assert avail_counter == planner.n_group

    phy_experts_workload = global_workload[phy2log].reshape(
        ep_size // r2o.shape[0], r2o.shape[0], -1
    )
    # print(f'[{global_rank}] {phy_experts_workload=}')

    dup_workload = (
        phy_experts_workload[:, :, : planner.combined_redundant_experts * r2o.shape[1]]
        .reshape(
            ep_size // r2o.shape[0], r2o.shape[0], planner.combined_redundant_experts, r2o.shape[1]
        )
        .sum(2)
    )
    # print(f'[{global_rank}] before balancing {dup_workload=}')
    dup_workload = dup_workload * probs + (dup_workload * (1 - probs)).gather(
        dim=1, index=r2o.expand(*probs.shape).long()
    )
    # print(f'[{global_rank}] after balancing {dup_workload=}')
    dup_workload = dup_workload.sum(2)
    fixed_workload = phy_experts_workload[
        :,
        :,
        planner.combined_redundant_experts * r2o.shape[1] : -planner.combined_redundant_experts
        * r2o.shape[1],
    ].sum(2)
    actual_workload = dup_workload + fixed_workload
    # print(f'[{global_rank}] {actual_workload=}')

    balance = float(actual_workload.max() / actual_workload.mean())
    print(f'[{global_rank}] balance coefficient: {balance}')

    def f_to_bench() -> None:
        planner.solve_probs(current_workload, avail_counter)

    time = bench_kineto(f_to_bench, 'kernel_solve', 1000)
    if global_rank == 0:
        print(f'average time cost: {time * 1e6} us')


if __name__ == '__main__':
    # 1. Initialize the distributed environment
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend="nccl")
    
    # 2. Set the device for this specific process to avoid everything piling onto GPU 0
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)

    # 3. Run your tests
    test_dist_planner_solve(SQUARE_4P2E, 64, 4, 2)
    #test_dist_planner_solve(HYPERCUBE_16P2E, 256, 16, 4)
    
    # 4. Clean up (optional but good practice)
    torch.distributed.destroy_process_group()