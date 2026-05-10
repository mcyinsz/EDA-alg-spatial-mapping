"""Evolutionary Algorithm solver (nested ES) for spatial accelerator mapping.

Based on: "Evolutionary Mapping of Neural Networks to Spatial Accelerators" (arXiv:2602.04717)
Adapted to use analytical cost model instead of hardware-in-the-loop.
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
    manhattan_distance,
)


def _decode_partitioning(
    genotype: List[int],
    min_cores: List[int],
) -> List[int]:
    """Decode partitioning genotype to actual core counts per layer.

    genotype: [x_1, ..., x_L, C_unused] where x_i = extra cores for layer i.
    """
    L = len(min_cores)
    return [min_cores[i] + genotype[i] for i in range(L)]


def _decode_placement(
    permutation: List[Tuple[int, int]],
    partitioning: List[int],
) -> List[List[Tuple[int, int]]]:
    """Assign first N positions from permutation to layers sequentially."""
    placement = []
    idx = 0
    for c in partitioning:
        placement.append(permutation[idx : idx + c])
        idx += c
    return placement


def _build_mapping(
    part_geno: List[int],
    place_perm: List[Tuple[int, int]],
    min_cores: List[int],
) -> MappingSolution:
    partitioning = _decode_partitioning(part_geno, min_cores)
    placement = _decode_placement(place_perm, partitioning)
    return MappingSolution(partitioning=partitioning, placement=placement)


def _reorder_placement(
    old_perm: List[Tuple[int, int]],
    old_part_geno: List[int],
    new_part_geno: List[int],
    min_cores: List[int],
    all_positions: List[Tuple[int, int]],
    rng: np.random.RandomState,
) -> List[Tuple[int, int]]:
    """Reordering operator: transfer spatial locality when partitioning changes.

    Preserves cores assigned to each layer where possible,
    fills deficits from unused pool, releases surplus to unused pool.
    """
    L = len(min_cores)
    old_part = _decode_partitioning(old_part_geno, min_cores)
    new_part = _decode_partitioning(new_part_geno, min_cores)

    # Extract old layer assignments
    old_assignments = []
    idx = 0
    for i in range(L):
        old_assignments.append(old_perm[idx : idx + old_part[i]])
        idx += old_part[i]
    old_unused = set(old_perm[idx:])

    new_perm = []
    unused_pool = set(old_unused)

    for i in range(L):
        old_cores = old_assignments[i]
        new_count = new_part[i]
        if new_count <= len(old_cores):
            # Keep first new_count cores, release rest
            new_perm.extend(old_cores[:new_count])
            for c in old_cores[new_count:]:
                unused_pool.add(c)
        else:
            # Keep all old cores, draw deficit from unused pool
            new_perm.extend(old_cores)
            deficit = new_count - len(old_cores)
            pool_list = list(unused_pool)
            rng.shuffle(pool_list)
            for c in pool_list[:deficit]:
                new_perm.append(c)
                unused_pool.discard(c)

    # Append remaining unused
    for p in all_positions:
        if p not in set(new_perm):
            new_perm.append(p)

    return new_perm


def _mutate_partitioning(
    geno: List[int],
    L: int,
    rng: np.random.RandomState,
    p_mut: float = 0.3,
    delta_max: int = 2,
) -> List[int]:
    """Mutate partitioning genotype."""
    new_geno = list(geno)
    for i in range(L):
        if rng.random() < p_mut:
            if rng.random() < 0.5 and new_geno[-1] > 0:
                # Add cores from unused pool
                delta = min(rng.randint(1, delta_max + 1), new_geno[-1])
                new_geno[i] += delta
                new_geno[-1] -= delta
            elif new_geno[i] > 0:
                # Move cores to unused pool
                delta = min(rng.randint(1, delta_max + 1), new_geno[i])
                new_geno[i] -= delta
                new_geno[-1] += delta
    return new_geno


def _mutate_placement(
    perm: List[Tuple[int, int]],
    C_used: int,
    rng: np.random.RandomState,
    alpha: float = 0.3,
) -> List[Tuple[int, int]]:
    """Mutate placement permutation (swap-based)."""
    new_perm = list(perm)
    k = max(1, int(alpha * C_used))
    op = rng.choice(["swap", "invert", "scramble"])

    if op == "swap":
        for _ in range(k):
            a = rng.randint(0, C_used)
            b = rng.randint(0, len(new_perm))
            new_perm[a], new_perm[b] = new_perm[b], new_perm[a]
    elif op == "invert":
        start = rng.randint(0, C_used)
        end = min(start + k, C_used)
        new_perm[start:end] = new_perm[start:end][::-1]
    else:  # scramble
        start = rng.randint(0, C_used)
        end = min(start + k, C_used)
        segment = new_perm[start:end]
        rng.shuffle(segment)
        new_perm[start:end] = segment

    return new_perm


def solve_ea(
    wl: Workload,
    acc: Accelerator,
    lambda_part: int = 4,
    lambda_place: int = 4,
    generations: int = 50,
    seed: int = 42,
    log_interval: int = 1,
) -> Tuple[MappingSolution, dict, dict]:
    """Nested (1+λ)-ES: outer level optimizes partitioning, inner level optimizes placement.

    Returns (best_solution, latency_dict, info).
    """
    rng = np.random.RandomState(seed)
    L = wl.num_layers
    min_cores = [acc.min_cores_for_layer(wl, i) for i in range(L)]
    all_positions = acc.core_positions()
    C_tot = acc.total_cores
    C_min = sum(min_cores)
    C_extra = C_tot - C_min

    # Initial partitioning: distribute extra cores equally
    init_geno = [0] * L + [C_extra]
    for i in range(C_extra):
        layer = i % L
        init_geno[layer] += 1
        init_geno[-1] -= 1

    # Initial placement: random permutation
    init_perm = list(all_positions)
    rng.shuffle(init_perm)

    init_sol = _build_mapping(init_geno, init_perm, min_cores)
    init_cost = compute_latency(init_sol, wl, acc)["total_time"]

    best_part_geno = list(init_geno)
    best_perm = list(init_perm)
    best_cost = init_cost

    history = []
    eval_count = 0
    t0 = time.time()

    current_part_geno = list(init_geno)
    current_perm = list(init_perm)
    current_cost = init_cost

    for gen in range(generations):
        # --- Partitioning evolution step ---
        best_offspring_part = None
        best_offspring_perm = None
        best_offspring_cost = current_cost

        for _ in range(lambda_part):
            new_part_geno = _mutate_partitioning(current_part_geno, L, rng)
            # Reorder placement to match new partitioning
            reordered_perm = _reorder_placement(
                current_perm, current_part_geno, new_part_geno, min_cores, all_positions, rng
            )
            new_part = _decode_partitioning(new_part_geno, min_cores)
            C_used = sum(new_part)
            if C_used > C_tot:
                continue

            new_sol = _build_mapping(new_part_geno, reordered_perm, min_cores)
            new_cost = compute_latency(new_sol, wl, acc)["total_time"]
            eval_count += 1

            if new_cost < best_offspring_cost:
                best_offspring_cost = new_cost
                best_offspring_part = list(new_part_geno)
                best_offspring_perm = list(reordered_perm)

        # Elitist selection for partitioning
        if best_offspring_part is not None and best_offspring_cost <= current_cost:
            current_part_geno = best_offspring_part
            current_perm = best_offspring_perm
            current_cost = best_offspring_cost

        # --- Placement evolution step ---
        current_part = _decode_partitioning(current_part_geno, min_cores)
        C_used = sum(current_part)

        for _ in range(lambda_place):
            new_perm = _mutate_placement(current_perm, C_used, rng)
            new_sol = _build_mapping(current_part_geno, new_perm, min_cores)
            new_cost = compute_latency(new_sol, wl, acc)["total_time"]
            eval_count += 1

            if new_cost < current_cost:
                current_perm = list(new_perm)
                current_cost = new_cost

        # Update global best
        if current_cost < best_cost:
            best_part_geno = list(current_part_geno)
            best_perm = list(current_perm)
            best_cost = current_cost

        if gen % log_interval == 0:
            history.append((gen, current_cost, best_cost, eval_count))

    best_sol = _build_mapping(best_part_geno, best_perm, min_cores)
    best_lat = compute_latency(best_sol, wl, acc)
    info = {
        "solve_time_s": time.time() - t0,
        "generations": generations,
        "lambda_part": lambda_part,
        "lambda_place": lambda_place,
        "total_evaluations": eval_count,
        "history": history,
    }
    return best_sol, best_lat, info
