"""Simulated Annealing solver for spatial accelerator mapping.

Jointly optimizes partitioning (core allocation) and placement (physical positions).
"""

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
    avg_inter_layer_distance,
)


def _random_placement(
    partitioning: List[int],
    acc: Accelerator,
    rng: np.random.RandomState,
) -> List[List[Tuple[int, int]]]:
    """Generate a random placement for a given partitioning."""
    positions = acc.core_positions()
    rng.shuffle(positions)
    idx = 0
    placement = []
    for c in partitioning:
        placement.append(positions[idx : idx + c])
        idx += c
    return placement


def _neighbor_partitioning(
    partitioning: List[int],
    total_cores: int,
    min_cores: List[int],
    rng: np.random.RandomState,
) -> List[int]:
    """Perturb partitioning: move, discard, or enable an idle core.

    Operations (chosen uniformly when feasible):
    - transfer: move 1 core from layer src to layer dst
    - discard: remove 1 core from layer src (core becomes idle)
    - enable_idle: assign 1 idle core to layer dst (total < K)
    """
    L = len(partitioning)
    new_part = list(partitioning)
    used = sum(new_part)

    ops = []
    if any(new_part[i] > min_cores[i] for i in range(L)):
        ops.append("transfer")
        ops.append("discard")
    if used < total_cores:
        ops.append("enable_idle")

    if not ops:
        return new_part

    op = rng.choice(ops)

    if op == "discard":
        candidates = [i for i in range(L) if new_part[i] > min_cores[i]]
        src = rng.choice(candidates)
        new_part[src] -= 1
        return new_part

    if op == "enable_idle":
        dst = rng.randint(0, L)
        new_part[dst] += 1
        return new_part

    # transfer: move 1 core from src to dst
    candidates = [i for i in range(L) if new_part[i] > min_cores[i]]
    if not candidates:
        return new_part
    src = rng.choice(candidates)
    new_part[src] -= 1
    dst = rng.randint(0, L)
    while dst == src:
        dst = rng.randint(0, L)
    new_part[dst] += 1
    if sum(new_part) > total_cores:
        new_part[dst] -= 1
        new_part[src] += 1
        return list(partitioning)
    return new_part


def _neighbor_placement(
    placement: List[List[Tuple[int, int]]],
    acc: Accelerator,
    rng: np.random.RandomState,
    swap_count: int = 2,
) -> List[List[Tuple[int, int]]]:
    """Perturb placement: swap core positions between layers."""
    new_place = [list(p) for p in placement]
    # Flatten all core positions
    all_cores = []
    layer_map = []  # (layer_idx, within_layer_idx)
    for i, layer_cores in enumerate(new_place):
        for j, pos in enumerate(layer_cores):
            all_cores.append(pos)
            layer_map.append((i, j))

    for _ in range(swap_count):
        a, b = rng.randint(0, len(all_cores), size=2)
        if a != b:
            all_cores[a], all_cores[b] = all_cores[b], all_cores[a]

    # Rebuild placement
    idx = 0
    for i in range(len(new_place)):
        for j in range(len(new_place[i])):
            new_place[i][j] = all_cores[idx]
            idx += 1
    return new_place


def solve_sa(
    wl: Workload,
    acc: Accelerator,
    max_iters: int = 5000,
    init_temp: float = 100.0,
    cooling_rate: float = 0.995,
    seed: int = 42,
    perturbation: str = "joint",  # "partitioning", "placement", "joint"
    log_interval: int = 100,
) -> Tuple[MappingSolution, dict, dict]:
    """Run simulated annealing.

    Returns (best_solution, latency_dict, info).
    """
    rng = np.random.RandomState(seed)
    L = wl.num_layers
    min_cores = [acc.min_cores_for_layer(wl, i) for i in range(L)]

    # Initial solution: minimum required cores per layer (idle cores allowed)
    total = acc.total_cores
    partitioning = list(min_cores)

    placement = _random_placement(partitioning, acc, rng)
    sol = MappingSolution(partitioning=partitioning, placement=placement)
    cost = compute_latency(sol, wl, acc)["total_time"]

    best_sol = deepcopy(sol)
    best_cost = cost

    T = init_temp
    history = []  # (iteration, current_cost, best_cost, temperature)
    t0 = time.time()

    for it in range(max_iters):
        # Generate neighbor
        op = rng.choice(["partitioning", "placement", "joint"])
        if op == "partitioning":
            new_part = _neighbor_partitioning(partitioning, total, min_cores, rng)
            # Reassign positions: keep old positions, redistribute
            all_positions = [p for layer in placement for p in layer]
            # Add or remove positions as needed
            new_total = sum(new_part)
            all_mesh = acc.core_positions()
            used_set = set(all_positions)
            unused = [p for p in all_mesh if p not in used_set]

            if new_total > len(all_positions):
                rng.shuffle(unused)
                all_positions.extend(unused[: new_total - len(all_positions)])
            elif new_total < len(all_positions):
                all_positions = all_positions[:new_total]

            rng.shuffle(all_positions)
            idx = 0
            new_place = []
            for c in new_part:
                new_place.append(all_positions[idx : idx + c])
                idx += c
            new_sol = MappingSolution(partitioning=new_part, placement=new_place)

        elif op == "placement":
            new_part = list(partitioning)
            n_swaps = max(1, int(np.sqrt(sum(partitioning))))
            new_place = _neighbor_placement(placement, acc, rng, swap_count=n_swaps)
            new_sol = MappingSolution(partitioning=new_part, placement=new_place)

        else:  # joint
            new_part = _neighbor_partitioning(partitioning, total, min_cores, rng)
            all_positions = [p for layer in placement for p in layer]
            all_mesh = acc.core_positions()
            used_set = set(all_positions)
            unused = [p for p in all_mesh if p not in used_set]
            new_total = sum(new_part)
            if new_total > len(all_positions):
                rng.shuffle(unused)
                all_positions.extend(unused[: new_total - len(all_positions)])
            elif new_total < len(all_positions):
                all_positions = all_positions[:new_total]
            rng.shuffle(all_positions)
            idx = 0
            new_place = []
            for c in new_part:
                new_place.append(all_positions[idx : idx + c])
                idx += c
            n_swaps = max(1, int(np.sqrt(new_total)))
            new_place = _neighbor_placement(new_place, acc, rng, swap_count=n_swaps)
            new_sol = MappingSolution(partitioning=new_part, placement=new_place)

        new_cost = compute_latency(new_sol, wl, acc)["total_time"]
        delta = new_cost - cost

        # Metropolis criterion
        if delta < 0 or (T > 1e-10 and rng.random() < np.exp(-delta / T)):
            partitioning = new_sol.partitioning
            placement = new_sol.placement
            sol = new_sol
            cost = new_cost

            if cost < best_cost:
                best_sol = deepcopy(sol)
                best_cost = cost

        T *= cooling_rate

        if it % log_interval == 0:
            history.append((it, cost, best_cost, T))

    best_lat = compute_latency(best_sol, wl, acc)
    info = {
        "solve_time_s": time.time() - t0,
        "iterations": max_iters,
        "init_temp": init_temp,
        "cooling_rate": cooling_rate,
        "final_temp": T,
        "history": history,
    }
    return best_sol, best_lat, info
