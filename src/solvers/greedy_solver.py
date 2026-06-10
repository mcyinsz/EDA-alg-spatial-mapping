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
    """Greedy partitioning: add cores only when total latency improves."""
    L = wl.num_layers
    min_cores = [acc.min_cores_for_layer(wl, i) for i in range(L)]
    total = acc.total_cores

    partitioning = list(min_cores)
    positions = [(r, c) for c in range(acc.cols) for r in range(acc.rows)]
    placement = _assign_from_positions(partitioning, positions)
    sol = MappingSolution(partitioning=partitioning, placement=placement)
    current_cost = compute_latency(sol, wl, acc)["total_time"]

    while sum(partitioning) < total:
        best_layer = -1
        best_cost = current_cost
        for i in range(L):
            trial_part = list(partitioning)
            trial_part[i] += 1
            if sum(trial_part) > total:
                continue
            trial_place = _assign_from_positions(trial_part, positions)
            trial_sol = MappingSolution(partitioning=trial_part, placement=trial_place)
            trial_cost = compute_latency(trial_sol, wl, acc)["total_time"]
            if trial_cost < best_cost:
                best_cost = trial_cost
                best_layer = i
        if best_layer < 0:
            break
        partitioning[best_layer] += 1
        placement = _assign_from_positions(partitioning, positions)
        sol = MappingSolution(partitioning=partitioning, placement=placement)
        current_cost = best_cost

    return partitioning


def _assign_from_positions(
    partitioning: List[int],
    positions: List[Tuple[int, int]],
) -> List[List[Tuple[int, int]]]:
    """Assign mesh positions sequentially to layers."""
    placement = []
    idx = 0
    for c in partitioning:
        placement.append(positions[idx : idx + c])
        idx += c
    return placement


def _contiguous_placement(
    partitioning: List[int],
    acc: Accelerator,
) -> List[List[Tuple[int, int]]]:
    """Place layers as contiguous blocks on the mesh, column-major."""
    positions = [(r, c) for c in range(acc.cols) for r in range(acc.rows)]
    return _assign_from_positions(partitioning, positions)


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
    """Refine partitioning by moving or removing cores greedily."""
    L = wl.num_layers
    min_cores = [acc.min_cores_for_layer(wl, i) for i in range(L)]
    current = deepcopy(sol)
    budget = max(max_iters, acc.total_cores)

    for _ in range(budget):
        improved = False
        current_cost = compute_latency(current, wl, acc)["total_time"]

        # Try removing a core from a layer (release to idle pool)
        for src in range(L):
            if current.partitioning[src] <= min_cores[src]:
                continue
            trial = deepcopy(current)
            trial.partitioning[src] -= 1
            if trial.placement[src]:
                trial.placement[src].pop()
            trial_cost = compute_latency(trial, wl, acc)["total_time"]
            if trial_cost < current_cost:
                current = trial
                current_cost = trial_cost
                improved = True
                break

        if improved:
            continue

        # Try moving a core from src to dst
        for src in range(L):
            if current.partitioning[src] <= min_cores[src]:
                continue
            for dst in range(L):
                if src == dst:
                    continue
                trial = deepcopy(current)
                trial.partitioning[src] -= 1
                trial.partitioning[dst] += 1
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
