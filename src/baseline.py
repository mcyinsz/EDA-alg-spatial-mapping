"""Baseline heuristic placements for spatial accelerator mapping."""

from __future__ import annotations

from typing import List, Tuple
import numpy as np

from model import (
    Workload,
    Accelerator,
    MappingSolution,
    compute_latency,
)


def _min_partitioning(wl: Workload, acc: Accelerator) -> List[int]:
    """Compute minimum partitioning (each layer gets minimum required cores)."""
    return [acc.min_cores_for_layer(wl, i) for i in range(wl.num_layers)]


def _assign_positions(
    partitioning: List[int],
    positions: List[Tuple[int, int]],
) -> List[List[Tuple[int, int]]]:
    """Assign positions sequentially to layers based on partitioning."""
    placement = []
    idx = 0
    for c in partitioning:
        placement.append(positions[idx : idx + c])
        idx += c
    return placement


def random_placement(
    wl: Workload,
    acc: Accelerator,
    seed: int = 42,
) -> Tuple[MappingSolution, dict]:
    """Random placement with minimum partitioning."""
    rng = np.random.RandomState(seed)
    partitioning = _min_partitioning(wl, acc)
    positions = acc.core_positions()
    rng.shuffle(positions)
    used = positions[: sum(partitioning)]
    placement = _assign_positions(partitioning, used)
    sol = MappingSolution(partitioning=partitioning, placement=placement)
    return sol, compute_latency(sol, wl, acc)


def packed_row_major(
    wl: Workload,
    acc: Accelerator,
) -> Tuple[MappingSolution, dict]:
    """Pack layers sequentially in row-major order."""
    partitioning = _min_partitioning(wl, acc)
    positions = acc.core_positions()
    placement = _assign_positions(partitioning, positions)
    sol = MappingSolution(partitioning=partitioning, placement=placement)
    return sol, compute_latency(sol, wl, acc)


def packed_col_major(
    wl: Workload,
    acc: Accelerator,
) -> Tuple[MappingSolution, dict]:
    """Pack layers sequentially in column-major order."""
    partitioning = _min_partitioning(wl, acc)
    positions = [
        (r, c) for c in range(acc.cols) for r in range(acc.rows)
    ]
    placement = _assign_positions(partitioning, positions)
    sol = MappingSolution(partitioning=partitioning, placement=placement)
    return sol, compute_latency(sol, wl, acc)


def spread_row_major(
    wl: Workload,
    acc: Accelerator,
) -> Tuple[MappingSolution, dict]:
    """Spread layers across mesh in row-major, maximizing inter-layer distance."""
    partitioning = _min_partitioning(wl, acc)
    total = sum(partitioning)
    H, W = acc.rows, acc.cols
    # Interleave positions: stride by total/num_layers
    positions = acc.core_positions()
    used = []
    stride = max(1, len(positions) // total)
    for i in range(total):
        used.append(positions[(i * stride) % len(positions)])
    placement = _assign_positions(partitioning, used)
    sol = MappingSolution(partitioning=partitioning, placement=placement)
    return sol, compute_latency(sol, wl, acc)


def equal_partitioning(
    wl: Workload,
    acc: Accelerator,
    placement_fn: str = "packed_row",
) -> Tuple[MappingSolution, dict]:
    """Distribute cores equally across layers (up to available cores)."""
    L = wl.num_layers
    min_part = _min_partitioning(wl, acc)
    min_total = sum(min_part)
    extra = acc.total_cores - min_total
    per_layer_extra = extra // L
    remainder = extra % L
    partitioning = []
    for i in range(L):
        partitioning.append(min_part[i] + per_layer_extra + (1 if i < remainder else 0))

    positions = acc.core_positions()
    if placement_fn == "packed_col":
        positions = [(r, c) for c in range(acc.cols) for r in range(acc.rows)]
    placement = _assign_positions(partitioning, positions)
    sol = MappingSolution(partitioning=partitioning, placement=placement)
    return sol, compute_latency(sol, wl, acc)


BASELINES = {
    "random": random_placement,
    "packed_row": packed_row_major,
    "packed_col": packed_col_major,
    "spread_row": spread_row_major,
    "equal_partition": equal_partitioning,
}
