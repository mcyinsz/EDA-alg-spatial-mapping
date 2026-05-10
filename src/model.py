"""Spatial accelerator model: workload, architecture, and cost function."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple
import numpy as np


@dataclass
class LayerSpec:
    """Explicit specification for a single compute layer (for Conv2D etc.)."""

    flops: float
    weight_bytes: float
    output_activation_bytes: float


def conv2d_spec(
    cin: int, cout: int, kernel_size: int,
    h_in: int, w_in: int, stride: int = 1, pad: int = 0,
) -> LayerSpec:
    """Create a LayerSpec for a Conv2D layer (fp32)."""
    h_out = (h_in + 2 * pad - kernel_size) // stride + 1
    w_out = (w_in + 2 * pad - kernel_size) // stride + 1
    k = kernel_size
    return LayerSpec(
        flops=2.0 * cout * cin * k * k * h_out * w_out,
        weight_bytes=float(cout * cin * k * k * 4),
        output_activation_bytes=float(cout * h_out * w_out * 4),
    )


def linear_spec(in_dim: int, out_dim: int) -> LayerSpec:
    """Create a LayerSpec for a linear (fully-connected) layer (fp32)."""
    return LayerSpec(
        flops=2.0 * in_dim * out_dim,
        weight_bytes=float(in_dim * out_dim * 4),
        output_activation_bytes=float(out_dim * 4),
    )


@dataclass
class Workload:
    """DNN workload defined by layer dimensions or explicit LayerSpecs.

    Use layer_dims for sequential linear layers (MLP, Transformer).
    Use layer_specs for heterogeneous layers (ConvNet, mixed).
    """

    name: str
    layer_dims: List[int] = field(default_factory=list)
    sparsity: float = 0.0
    layer_specs: List[LayerSpec] = field(default_factory=list)
    tp_comm_factors: List[float] = field(default_factory=list)

    @property
    def num_layers(self) -> int:
        if self.layer_specs:
            return len(self.layer_specs)
        return max(0, len(self.layer_dims) - 1)

    def layer_flops(self, i: int) -> float:
        if self.layer_specs:
            return self.layer_specs[i].flops
        return 2.0 * self.layer_dims[i] * self.layer_dims[i + 1] * (1.0 - self.sparsity)

    def layer_weight_bytes(self, i: int) -> float:
        if self.layer_specs:
            return self.layer_specs[i].weight_bytes
        return 4.0 * self.layer_dims[i] * self.layer_dims[i + 1] * (1.0 - self.sparsity)

    def layer_activation_bytes(self, i: int) -> float:
        if self.layer_specs:
            return self.layer_specs[i].output_activation_bytes
        return 4.0 * self.layer_dims[i + 1]

    def layer_tp_comm_factor(self, i: int) -> float:
        """κ_i: tensor-parallel communication factor for layer i."""
        if self.tp_comm_factors:
            return self.tp_comm_factors[i]
        if self.layer_specs:
            return 0.5
        return 1.0


@dataclass
class Accelerator:
    """2D mesh spatial accelerator model."""

    name: str
    rows: int  # H
    cols: int  # W
    core_memory: float = 256 * 1024  # bytes per core (e.g. 256 KB)
    core_flops_per_cycle: float = 64.0  # FLOPs per cycle per core
    comm_beta: float = 1.0  # comm cost coefficient (cycles per byte per hop)
    frequency_ghz: float = 1.0  # clock frequency in GHz
    intra_comm_enabled: bool = True  # include intra-layer TP communication

    @property
    def total_cores(self) -> int:
        return self.rows * self.cols

    def core_positions(self) -> List[Tuple[int, int]]:
        """All (row, col) positions on the mesh."""
        return [(r, c) for r in range(self.rows) for c in range(self.cols)]

    def min_cores_for_layer(self, wl: Workload, layer_idx: int) -> int:
        """Minimum cores needed for a layer (memory constraint)."""
        weight_bytes = wl.layer_weight_bytes(layer_idx)
        act_bytes = wl.layer_activation_bytes(layer_idx)
        per_core_need = weight_bytes + act_bytes
        if per_core_need <= self.core_memory:
            return 1
        return int(np.ceil(per_core_need / self.core_memory))


def manhattan_distance(p1: Tuple[int, int], p2: Tuple[int, int]) -> int:
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])


def avg_inter_layer_distance(
    cores_i: List[Tuple[int, int]],
    cores_j: List[Tuple[int, int]],
) -> float:
    """Average pairwise Manhattan distance between two groups of cores."""
    if not cores_i or not cores_j:
        return 0.0
    total = 0
    for ci in cores_i:
        for cj in cores_j:
            total += manhattan_distance(ci, cj)
    return total / (len(cores_i) * len(cores_j))


def avg_intra_layer_distance(cores: List[Tuple[int, int]]) -> float:
    """Average pairwise Manhattan distance within a group of cores."""
    n = len(cores)
    if n <= 1:
        return 0.0
    total = 0
    for i in range(n):
        for j in range(i + 1, n):
            total += manhattan_distance(cores[i], cores[j])
    return 2.0 * total / (n * (n - 1))


@dataclass
class MappingSolution:
    """A complete partitioning + placement solution."""

    partitioning: List[int]  # x_i = cores for layer i
    placement: List[List[Tuple[int, int]]]  # physical positions for each layer's cores

    def total_cores_used(self) -> int:
        return sum(self.partitioning)


def compute_latency(
    sol: MappingSolution,
    wl: Workload,
    acc: Accelerator,
    intra_comm_enabled: Optional[bool] = None,
) -> dict:
    """Compute detailed latency breakdown for a mapping solution.

    Returns dict with compute_time, inter_comm_time, intra_comm_time,
    comm_time, total_time (all in cycles).

    Set intra_comm_enabled=False to disable intra-layer comm (inter-only formulation).
    Defaults to acc.intra_comm_enabled.
    """
    if intra_comm_enabled is None:
        intra_comm_enabled = acc.intra_comm_enabled
    L = wl.num_layers
    assert len(sol.partitioning) == L
    assert len(sol.placement) == L

    # Validate feasibility
    for i in range(L):
        c_i = sol.partitioning[i]
        assert c_i > 0, f"Layer {i} has 0 cores"
        assert len(sol.placement[i]) == c_i, (
            f"Layer {i}: partitioning={c_i} but placement has {len(sol.placement[i])} cores"
        )
        min_c = acc.min_cores_for_layer(wl, i)
        assert c_i >= min_c, (
            f"Layer {i}: {c_i} cores < min required {min_c}"
        )
    assert sum(sol.partitioning) <= acc.total_cores, (
        f"Total cores used {sum(sol.partitioning)} > available {acc.total_cores}"
    )
    # C1: no core assigned to more than one layer
    all_positions = [p for layer in sol.placement for p in layer]
    if len(all_positions) != len(set(all_positions)):
        from collections import Counter
        counts = Counter(all_positions)
        dupes = [p for p, c in counts.items() if c > 1]
        raise AssertionError(f"C1 violation: core(s) assigned to multiple layers: {dupes[:5]}")

    compute_times = []
    inter_comm_times = []
    intra_comm_times = []

    for i in range(L):
        c_i = sol.partitioning[i]
        t_comp = wl.layer_flops(i) / (c_i * acc.core_flops_per_cycle)
        compute_times.append(t_comp)

    for i in range(L - 1):
        c_i = sol.partitioning[i]
        act_bytes = wl.layer_activation_bytes(i)
        avg_dist = avg_inter_layer_distance(sol.placement[i], sol.placement[i + 1])
        t_comm = act_bytes * acc.comm_beta * avg_dist / max(c_i, 1)
        inter_comm_times.append(t_comm)

    for i in range(L):
        c_i = sol.partitioning[i]
        if not intra_comm_enabled or c_i <= 1:
            intra_comm_times.append(0.0)
        else:
            kappa = wl.layer_tp_comm_factor(i)
            act_bytes = wl.layer_activation_bytes(i)
            avg_d = avg_intra_layer_distance(sol.placement[i])
            vol = kappa * act_bytes * (1.0 - 1.0 / c_i)
            t_intra = acc.comm_beta * vol * avg_d
            intra_comm_times.append(t_intra)

    total_compute = sum(compute_times)
    total_inter_comm = sum(inter_comm_times)
    total_intra_comm = sum(intra_comm_times)
    total = total_compute + total_inter_comm + total_intra_comm

    return {
        "compute_time": total_compute,
        "inter_comm_time": total_inter_comm,
        "intra_comm_time": total_intra_comm,
        "comm_time": total_inter_comm + total_intra_comm,
        "total_time": total,
        "compute_per_layer": compute_times,
        "inter_comm_per_layer": inter_comm_times,
        "intra_comm_per_layer": intra_comm_times,
    }


def cycles_to_us(cycles: float, freq_ghz: float = 1.0) -> float:
    """Convert cycles to microseconds."""
    return cycles / (freq_ghz * 1000.0)
