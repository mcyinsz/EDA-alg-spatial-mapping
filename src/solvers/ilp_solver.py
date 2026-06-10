"""ILP solver for the spatial accelerator mapping problem.

Fixed partitioning + optimal placement via ILP.
Uses big-M linearization for the quadratic comm cost term.
"""

from __future__ import annotations

import time
from typing import List, Tuple

import numpy as np
import pulp

from model import (
    Workload,
    Accelerator,
    MappingSolution,
    compute_latency,
    manhattan_distance,
)


def _greedy_partitioning(wl: Workload, acc: Accelerator) -> List[int]:
    """Greedy partitioning: allocate cores proportional to compute demand."""
    L = wl.num_layers
    min_cores = [acc.min_cores_for_layer(wl, i) for i in range(L)]
    flops = [wl.layer_flops(i) for i in range(L)]

    partitioning = list(min_cores)
    remaining = acc.total_cores - sum(partitioning)

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


def solve_ilp(
    wl: Workload,
    acc: Accelerator,
    time_limit: int = 30,
    mip_gap: float = 0.05,
) -> Tuple[MappingSolution, dict, dict]:
    """Solve the mapping problem using ILP (fixed partitioning + optimal placement).

    Returns (solution, latency_dict, solver_info).
    Raises RuntimeError if infeasible or no solution found.
    """
    L = wl.num_layers
    positions = acc.core_positions()
    K = len(positions)

    # Precompute distance matrix
    dist = np.zeros((K, K), dtype=np.float64)
    for a in range(K):
        for b in range(K):
            dist[a, b] = manhattan_distance(positions[a], positions[b])

    # Per-layer parameters
    min_cores = [acc.min_cores_for_layer(wl, i) for i in range(L)]
    layer_act = [wl.layer_activation_bytes(i) for i in range(L)]
    compute_coeff = [wl.layer_flops(i) / acc.core_flops_per_cycle for i in range(L)]

    # --- Fixed partitioning (greedy by compute demand) ---
    partitioning = _greedy_partitioning(wl, acc)
    assert sum(partitioning) <= K
    assert all(partitioning[i] >= min_cores[i] for i in range(L))

    # --- ILP: optimize placement given fixed partitioning ---
    prob = pulp.LpProblem("SpatialMappingPlacement", pulp.LpMinimize)

    # z[i,k] = 1 if core k assigned to layer i
    z = {}
    for i in range(L):
        for k in range(K):
            z[i, k] = pulp.LpVariable(f"z_{i}_{k}", cat=pulp.LpBinary)

    # Each core assigned to at most one layer
    for k in range(K):
        prob += pulp.lpSum(z[i, k] for i in range(L)) <= 1, f"one_layer_{k}"

    # Each layer gets exactly partitioning[i] cores
    for i in range(L):
        prob += (
            pulp.lpSum(z[i, k] for k in range(K)) == partitioning[i],
            f"layer_count_{i}",
        )

    # Linearize product z[i,k]*z[i+1,l] with aux variable u[i,k,l]
    u = {}
    for i in range(L - 1):
        for k in range(K):
            for l in range(K):
                if dist[k, l] > 0:
                    u[i, k, l] = pulp.LpVariable(f"u_{i}_{k}_{l}", cat=pulp.LpBinary)
                    prob += u[i, k, l] <= z[i, k], f"u_ub1_{i}_{k}_{l}"
                    prob += u[i, k, l] <= z[i + 1, l], f"u_ub2_{i}_{k}_{l}"
                    prob += u[i, k, l] >= z[i, k] + z[i + 1, l] - 1, f"u_lb_{i}_{k}_{l}"

    # Linearize z[i,k]*z[i,l] for intra-layer distance with w[i,k,l]
    w = {}
    if acc.intra_comm_enabled:
        for i in range(L):
            if partitioning[i] <= 1:
                continue
            for k in range(K):
                for l in range(k + 1, K):
                    if dist[k, l] > 0:
                        w[i, k, l] = pulp.LpVariable(f"w_{i}_{k}_{l}", cat=pulp.LpBinary)
                        prob += w[i, k, l] <= z[i, k], f"w_ub1_{i}_{k}_{l}"
                        prob += w[i, k, l] <= z[i, l], f"w_ub2_{i}_{k}_{l}"
                        prob += w[i, k, l] >= z[i, k] + z[i, l] - 1, f"w_lb_{i}_{k}_{l}"

    # Objective: total latency (compute is fixed, minimize inter + intra comm)
    total_compute_time = sum(compute_coeff[i] / partitioning[i] for i in range(L))

    # Inter-layer comm
    inter_comm_exprs = []
    for i in range(L - 1):
        weight = acc.comm_beta * layer_act[i] / partitioning[i]
        inter_comm_exprs.append(
            weight * pulp.lpSum(
                dist[k, l] * u[i, k, l]
                for k in range(K) for l in range(K)
                if dist[k, l] > 0 and (i, k, l) in u
            )
        )

    # Intra-layer comm
    intra_comm_exprs = []
    for i in range(L):
        if partitioning[i] <= 1 or (i, 0, 1) not in w:
            continue
        kappa = wl.layer_tp_comm_factor(i)
        x_i = partitioning[i]
        intra_weight = (
            acc.comm_beta * kappa * layer_act[i]
            * (1.0 - 1.0 / x_i) * 2.0 / (x_i * (x_i - 1))
        )
        if not acc.intra_serialized:
            intra_weight /= max(x_i, 1)
        intra_comm_exprs.append(
            intra_weight * pulp.lpSum(
                dist[k, l] * w[i, k, l]
                for k in range(K) for l in range(k + 1, K)
                if dist[k, l] > 0 and (i, k, l) in w
            )
        )

    prob += total_compute_time + pulp.lpSum(inter_comm_exprs) + pulp.lpSum(intra_comm_exprs), "objective"

    # Solve
    solver = pulp.PULP_CBC_CMD(timeLimit=time_limit, gapRel=mip_gap, msg=0)
    t0 = time.time()
    status = prob.solve(solver)
    solve_time = time.time() - t0
    status_str = pulp.LpStatus[status]

    solver_info = {
        "status": status_str,
        "solve_time_s": solve_time,
        "partitioning_fixed": partitioning,
    }

    # Validate solver status
    if status_str == "Infeasible":
        raise RuntimeError(f"ILP infeasible (status={status_str})")

    # Extract and validate placement
    placement = []
    for i in range(L):
        cores_i = []
        for k in range(K):
            val = z[i, k].varValue
            if val is not None and val > 0.5:
                cores_i.append(positions[k])
        placement.append(cores_i)

    # Verify extracted solution matches partitioning
    for i in range(L):
        if len(placement[i]) != partitioning[i]:
            raise RuntimeError(
                f"ILP solution invalid: layer {i} needs {partitioning[i]} cores "
                f"but got {len(placement[i])} (status={status_str})"
            )

    sol = MappingSolution(partitioning=partitioning, placement=placement)
    lat = compute_latency(sol, wl, acc)
    solver_info["objective_value"] = pulp.value(prob.objective)
    solver_info["ilp_total_time"] = lat["total_time"]

    return sol, lat, solver_info
