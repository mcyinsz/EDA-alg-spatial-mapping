"""Greedy heuristic + Kernighan-Lin local search for spatial accelerator mapping."""

from __future__ import annotations

import time
from copy import deepcopy
from typing import List, Tuple

import numpy as np

from model import (
    Workload,
    Accelerator,
    MappingSolution,
    compute_latency,
    manhattan_distance,
    avg_inter_layer_distance,
)


def _greedy_partitioning(wl: Workload, acc: Accelerator) -> List[int]:
    """Greedy partitioning: allocate cores proportional to compute demand."""
    L = wl.num_layers
    min_cores = [acc.min_cores_for_layer(wl, i) for i in range(L)]
    total = acc.total_cores
    flops = [wl.layer_flops(i) for i in range(L)]
    total_flops = sum(flops)

    partitioning = list(min_cores)
    remaining = total - sum(partitioning)

    # Greedily assign remaining cores to the layer with highest per-core compute time
    for _ in range(remaining):
        best_layer = -1
        best_reduction = -1.0
        for i in range(L):
            current_time = flops[i] / (partitioning[i] * acc.core_flops_per_cycle)
            new_time = flops[i] / ((partitioning[i] + 1) * acc.core_flops_per_cycle)
            reduction = current_time - new_time
            if reduction > best_reduction:
                best_reduction = reduction
                best_layer = i
        if best_layer >= 0:
            partitioning[best_layer] += 1
    return partitioning


def _contiguous_placement(
    partitioning: List[int],
    acc: Accelerator,
) -> List[List[Tuple[int, int]]]:
    """Place layers as contiguous blocks on the mesh, column-major."""
    H, W = acc.rows, acc.cols
    positions = [(r, c) for c in range(W) for r in range(H)]
    idx = 0
    placement = []
    for c in partitioning:
        placement.append(positions[idx : idx + c])
        idx += c
    return placement


def _kl_refinement(
    sol: MappingSolution,
    wl: Workload,
    acc: Accelerator,
    max_passes: int = 10,
) -> MappingSolution:
    """Kernighan-Lin style refinement: iteratively swap core assignments between layers.

    For each pair of layers, try swapping individual cores to reduce total latency.
    """
    L = wl.num_layers
    current = deepcopy(sol)

    for _pass in range(max_passes):
        improved = False
        current_cost = compute_latency(current, wl, acc)["total_time"]

        # Try swapping cores between each pair of layers
        for i in range(L):
            for j in range(i + 1, L):
                if not current.placement[i] or not current.placement[j]:
                    continue
                # Try swapping each core in layer i with each in layer j
                best_swap = None
                best_cost = current_cost

                for ci_idx in range(len(current.placement[i])):
                    for cj_idx in range(len(current.placement[j])):
                        # Swap
                        trial = deepcopy(current)
                        trial.placement[i][ci_idx], trial.placement[j][cj_idx] = (
                            trial.placement[j][cj_idx],
                            trial.placement[i][ci_idx],
                        )
                        trial_cost = compute_latency(trial, wl, acc)["total_time"]
                        if trial_cost < best_cost:
                            best_cost = trial_cost
                            best_swap = (ci_idx, cj_idx)

                if best_swap is not None:
                    ci_idx, cj_idx = best_swap
                    current.placement[i][ci_idx], current.placement[j][cj_idx] = (
                        current.placement[j][cj_idx],
                        current.placement[i][ci_idx],
                    )
                    current_cost = best_cost
                    improved = True

        if not improved:
            break

    return current


def _partition_refinement(
    sol: MappingSolution,
    wl: Workload,
    acc: Accelerator,
    max_iters: int = 20,
) -> MappingSolution:
    """Refine partitioning by moving cores between layers greedily."""
    L = wl.num_layers
    min_cores = [acc.min_cores_for_layer(wl, i) for i in range(L)]
    current = deepcopy(sol)

    for _ in range(max_iters):
        improved = False
        current_cost = compute_latency(current, wl, acc)["total_time"]

        for src in range(L):
            if current.partitioning[src] <= min_cores[src]:
                continue
            for dst in range(L):
                if src == dst:
                    continue
                trial = deepcopy(current)
                trial.partitioning[src] -= 1
                trial.partitioning[dst] += 1

                # Move one core position from src to dst
                if trial.placement[src]:
                    core = trial.placement[src].pop()
                    trial.placement[dst].append(core)

                trial_cost = compute_latency(trial, wl, acc)["total_time"]
                if trial_cost < current_cost:
                    current = trial
                    current_cost = trial_cost
                    improved = True
                    break
            if improved:
                break

        if not improved:
            break

    return current


def solve_greedy(
    wl: Workload,
    acc: Accelerator,
    kl_passes: int = 10,
    part_refine_iters: int = 20,
) -> Tuple[MappingSolution, dict, dict]:
    """Greedy partitioning + contiguous placement + KL refinement + partition refinement.

    Returns (solution, latency_dict, info).
    """
    t0 = time.time()

    # Phase 1: Greedy partitioning
    partitioning = _greedy_partitioning(wl, acc)

    # Phase 2: Contiguous placement
    placement = _contiguous_placement(partitioning, acc)
    sol = MappingSolution(partitioning=partitioning, placement=placement)
    init_cost = compute_latency(sol, wl, acc)["total_time"]

    # Phase 3: KL placement refinement
    sol = _kl_refinement(sol, wl, acc, max_passes=kl_passes)
    after_kl_cost = compute_latency(sol, wl, acc)["total_time"]

    # Phase 4: Partition refinement
    sol = _partition_refinement(sol, wl, acc, max_iters=part_refine_iters)
    final_cost = compute_latency(sol, wl, acc)["total_time"]

    info = {
        "solve_time_s": time.time() - t0,
        "init_cost": init_cost,
        "after_kl_cost": after_kl_cost,
        "final_cost": final_cost,
        "kl_passes": kl_passes,
        "part_refine_iters": part_refine_iters,
    }
    return sol, compute_latency(sol, wl, acc), info
