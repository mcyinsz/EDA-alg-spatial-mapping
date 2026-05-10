"""Visualization utilities for spatial accelerator mapping results."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import numpy as np

from model import (
    Workload, Accelerator, MappingSolution, compute_latency,
    manhattan_distance, avg_inter_layer_distance, avg_intra_layer_distance,
)

RESULTS_DIR = Path(__file__).parent.parent / "results"


# Consistent color palette for layers (up to 12 layers)
LAYER_COLORS = [
    "#e6194b", "#3cb44b", "#4363d8", "#f58231",
    "#911eb4", "#42d4f4", "#f032e6", "#bfef45",
    "#fabed4", "#469990", "#dcbeff", "#9A6324",
]

# Solver colors (consistent across all figures)
SOLVER_COLORS = {"ILP": "#e6194b", "SA": "#3cb44b", "EA": "#4363d8", "Greedy": "#f58231"}
SOLVER_DISPLAY = {"ILP": "ILP", "SA": "SA", "EA": "EA", "Greedy": "Greedy+KL"}


_RESULTS_PATH = None  # Set via --results command line arg
_OUTPUT_DIR = None     # Set automatically based on input file


def _load_results() -> dict:
    global _OUTPUT_DIR
    path = Path(_RESULTS_PATH) if _RESULTS_PATH else RESULTS_DIR / "experiment_results.json"
    if not path.exists():
        raise FileNotFoundError(f"No results at {path}. Run experiment.py first.")
    print(f"Loading results from {path}")
    with open(path) as f:
        data = json.load(f)
    # If reading inter_only results, write to results_inter_only/ subdirectory
    if path.stem.endswith("_inter_only"):
        _OUTPUT_DIR = RESULTS_DIR / "inter_only"
    else:
        _OUTPUT_DIR = RESULTS_DIR
    return data


def _out(name: str) -> str:
    """Resolve output path, respecting subdirectory for inter-only results."""
    return str(_OUTPUT_DIR / name)


def _layer_color(i: int) -> str:
    return LAYER_COLORS[i % len(LAYER_COLORS)]


def _best_run(runs: list) -> Optional[dict]:
    valid = [r for r in runs if "total_time_cycles" in r]
    return min(valid, key=lambda r: r["total_time_cycles"]) if valid else None


def _reconstruct_solution(partitioning: list, placement_data: Optional[dict],
                          wl: Workload, acc: Accelerator) -> Optional[MappingSolution]:
    """Try to reconstruct a MappingSolution from experiment result data."""
    if not placement_data or "placement" not in placement_data:
        # Generate placement from partitioning using contiguous layout
        H, W = acc.rows, acc.cols
        positions = [(r, c) for c in range(W) for r in range(H)]
        idx = 0
        placement = []
        for c in partitioning:
            placement.append(positions[idx:idx + c])
            idx += c
        return MappingSolution(partitioning=partitioning, placement=placement)

    placement = []
    for layer_cores in placement_data["placement"]:
        placement.append([tuple(p) for p in layer_cores])
    return MappingSolution(partitioning=partitioning, placement=placement)


# ============================================================
# Figure 1: Workload + Objective Overview
# ============================================================

def plot_workload_overview(
    wl: Workload,
    acc: Accelerator,
    title: str = "",
    save_path: Optional[str] = None,
):
    """Left: workload layer structure. Right: accelerator mesh. Center: objective function."""
    L = wl.num_layers

    fig = plt.figure(figsize=(14, 5))
    gs = gridspec.GridSpec(1, 3, width_ratios=[3, 1.2, 3], wspace=0.35)

    # --- Left: Workload layer structure ---
    ax_wl = fig.add_subplot(gs[0])

    flops = [wl.layer_flops(i) for i in range(L)]
    weights = [wl.layer_weight_bytes(i) / 1024 for i in range(L)]
    activations = [wl.layer_activation_bytes(i) / 1024 for i in range(L - 1)]

    max_flops = max(flops) if flops else 1
    max_weight = max(weights) if weights else 1

    bar_height = 0.6
    y_positions = list(range(L))

    # Draw layer rectangles: width = flops (normalized), color intensity = weight
    for i in range(L):
        width_norm = flops[i] / max_flops
        color = _layer_color(i)
        # Alpha encodes weight size relative to max
        alpha = 0.3 + 0.7 * (weights[i] / max_weight) if max_weight > 0 else 0.8
        ax_wl.barh(i, width_norm, height=bar_height, color=color, alpha=alpha,
                    edgecolor="black", linewidth=0.8)
        ax_wl.text(width_norm + 0.02, i,
                    f"L{i}\n{flops[i]/1e6:.1f}M FLOPs\n{weights[i]:.0f} KB",
                    va="center", fontsize=7)

    # Draw activation arrows between layers
    if activations:
        max_act = max(activations)
        for i in range(L - 1):
            thickness = 0.5 + 2.5 * (activations[i] / max_act) if max_act > 0 else 1
            ax_wl.annotate("", xy=(0, i + 0.4), xytext=(0, i + 0.6),
                            arrowprops=dict(arrowstyle="->", color="gray",
                                            lw=thickness, alpha=0.5))
            ax_wl.text(-0.08, i + 0.5, f"{activations[i]:.0f}KB",
                        ha="right", va="center", fontsize=6, color="gray")

    ax_wl.set_xlim(-0.3, 1.5)
    ax_wl.set_ylim(-0.5, L - 0.5)
    ax_wl.set_yticks(y_positions)
    ax_wl.set_yticklabels([f"L{i}" for i in range(L)], fontsize=8)
    ax_wl.set_xlabel("Relative FLOPs (width) / Weight bytes (color depth)", fontsize=8)
    ax_wl.invert_yaxis()
    ax_wl.set_title("Workload Structure", fontsize=11, fontweight="bold")

    # --- Center: Accelerator mesh ---
    ax_acc = fig.add_subplot(gs[1])
    H, W = acc.rows, acc.cols
    mesh = np.zeros((H, W))
    ax_acc.imshow(mesh, cmap="Greys", vmin=0, vmax=1, aspect="equal")
    for r in range(H):
        for c in range(W):
            rect = plt.Rectangle((c - 0.5, r - 0.5), 1, 1,
                                   fill=False, edgecolor="gray", linewidth=0.5)
            ax_acc.add_patch(rect)
    ax_acc.set_xlim(-0.6, W - 0.4)
    ax_acc.set_ylim(-0.6, H - 0.4)
    ax_acc.invert_yaxis()
    ax_acc.set_xlabel("Col", fontsize=8)
    ax_acc.set_ylabel("Row", fontsize=8)
    ax_acc.set_title(f"{acc.name}\n{H}×{W}={acc.total_cores} cores",
                      fontsize=9, fontweight="bold")
    ax_acc.text(0.5, -0.15,
                f"Mem/core: {acc.core_memory/1024:.0f}KB\n"
                f"Perf: {acc.core_flops_per_cycle:.0f} FLOPs/cyc",
                transform=ax_acc.transAxes, ha="center", fontsize=7, color="gray")

    # --- Right: Objective function ---
    ax_obj = fig.add_subplot(gs[2])
    ax_obj.axis("off")

    obj_text = (
        r"$\min_{\mathbf{x}, \{S_i\}} \; T = T_{\mathrm{comp}} + T_{\mathrm{inter}} + T_{\mathrm{intra}}$"
        "\n\n"
        r"$T_{\mathrm{comp},i} = \dfrac{F_i}{x_i \cdot P}$"
        "\n\n"
        r"$T_{\mathrm{inter},i} = \beta \cdot \dfrac{A_i}{x_i} \cdot \bar{d}_{\mathrm{inter}}(S_i, S_{i+1})$"
        "\n\n"
        r"$T_{\mathrm{intra},i} = \beta \cdot \kappa_i A_i (1-\frac{1}{x_i}) \cdot \bar{d}_{\mathrm{intra}}(S_i)$"
        "\n\n"
        "Constraints:\n"
        r"• $|S_i| = x_i$, $S_i \cap S_j = \emptyset$" + "\n"
        r"• $x_i \geq x_i^{\min}$ (memory)" + "\n"
        r"• $\sum x_i \leq H \times W$"
    )
    ax_obj.text(0.05, 0.95, obj_text, transform=ax_obj.transAxes,
                fontsize=11, verticalalignment="top",
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow",
                          edgecolor="gray", alpha=0.8))
    ax_obj.set_title("Objective Function", fontsize=11, fontweight="bold")

    if title:
        fig.suptitle(title, fontsize=13, fontweight="bold")

    plt.tight_layout(rect=[0, 0, 1, 0.95] if title else None)
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.close(fig)


# ============================================================
# Figure 2: 4-Panel Mapping Detail (Placement + Partitioning + Comm + Breakdown)
# ============================================================

def plot_mapping_detail(
    sol: MappingSolution,
    wl: Workload,
    acc: Accelerator,
    title: str = "",
    save_path: Optional[str] = None,
):
    """4-panel figure: placement map, partitioning bar, communication overlay, latency breakdown."""
    L = wl.num_layers
    H, W = acc.rows, acc.cols
    lat = compute_latency(sol, wl, acc)

    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(2, 2, hspace=0.35, wspace=0.35)

    # --- Top-left: Mesh Placement Map ---
    ax_place = fig.add_subplot(gs[0, 0])
    mesh = np.full((H, W), -1)
    for i, layer_cores in enumerate(sol.placement):
        for r, c in layer_cores:
            mesh[r, c] = i

    # Draw colored cells
    for r in range(H):
        for c in range(W):
            li = mesh[r, c]
            if li >= 0:
                color = _layer_color(li)
                rect = plt.Rectangle((c - 0.45, r - 0.45), 0.9, 0.9,
                                      facecolor=color, edgecolor="black",
                                      linewidth=0.8, alpha=0.85)
                ax_place.add_patch(rect)
                ax_place.text(c, r, f"L{li}", ha="center", va="center",
                               fontsize=max(5, min(8, 60 // max(H, W))),
                               fontweight="bold", color="white")
            else:
                rect = plt.Rectangle((c - 0.45, r - 0.45), 0.9, 0.9,
                                      facecolor="white", edgecolor="lightgray",
                                      linewidth=0.3)
                ax_place.add_patch(rect)

    # Draw centroid-to-centroid lines
    centroids = []
    for i in range(L):
        cores = sol.placement[i]
        cr = np.mean([p[0] for p in cores])
        cc = np.mean([p[1] for p in cores])
        centroids.append((cr, cc))

    for i in range(L - 1):
        act_bytes = wl.layer_activation_bytes(i)
        max_act = max(wl.layer_activation_bytes(j) for j in range(L - 1))
        lw = 1 + 3 * (act_bytes / max_act) if max_act > 0 else 1.5
        dist = avg_inter_layer_distance(sol.placement[i], sol.placement[i + 1])
        max_dist = max(H + W, 1)
        alpha = 0.2 + 0.6 * (dist / max_dist)
        ax_place.plot([centroids[i][1], centroids[i + 1][1]],
                       [centroids[i][0], centroids[i + 1][0]],
                       "k-", linewidth=lw, alpha=alpha)

    ax_place.set_xlim(-0.6, W - 0.4)
    ax_place.set_ylim(-0.6, H - 0.4)
    ax_place.set_aspect("equal")
    ax_place.invert_yaxis()
    ax_place.set_xlabel("Column")
    ax_place.set_ylabel("Row")
    ax_place.set_title("Mesh Placement", fontsize=10, fontweight="bold")

    handles = [mpatches.Patch(color=_layer_color(i), label=f"L{i} ({sol.partitioning[i]}c)")
               for i in range(L)]
    ax_place.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.02, 1),
                     fontsize=max(5, min(7, 60 // L)))

    # --- Top-right: Partitioning Bar ---
    ax_part = fig.add_subplot(gs[0, 1])
    min_cores = [acc.min_cores_for_layer(wl, i) for i in range(L)]
    y_pos = list(range(L))

    # Full bar = allocated cores
    bars = ax_part.barh(y_pos, sol.partitioning, height=0.6,
                         color=[_layer_color(i) for i in range(L)],
                         edgecolor="black", linewidth=0.8, alpha=0.85)

    # Overlay minimum cores
    ax_part.barh(y_pos, min_cores, height=0.6,
                  color="none", edgecolor="black", linewidth=1.5, linestyle="--",
                  label="Minimum required")

    for i, (alloc, mn) in enumerate(zip(sol.partitioning, min_cores)):
        extra = alloc - mn
        if extra > 0:
            ax_part.text(alloc + 0.1, i, f"+{extra}", va="center", fontsize=7,
                          color="red", fontweight="bold")

    ax_part.set_yticks(y_pos)
    ax_part.set_yticklabels([f"L{i}" for i in range(L)], fontsize=8)
    ax_part.invert_yaxis()
    ax_part.set_xlabel("Number of Cores")
    ax_part.set_title("Partitioning", fontsize=10, fontweight="bold")
    ax_part.legend(fontsize=7)
    ax_part.grid(True, axis="x", alpha=0.3)

    # --- Bottom-left: Communication Overlay (inter + intra) ---
    ax_comm = fig.add_subplot(gs[1, 0])

    # Compute inter-layer comm volumes
    inter_comm_volumes = []
    for i in range(L - 1):
        act_bytes = wl.layer_activation_bytes(i)
        avg_d = avg_inter_layer_distance(sol.placement[i], sol.placement[i + 1])
        vol = act_bytes * avg_d / max(sol.partitioning[i], 1)
        inter_comm_volumes.append(vol)

    # Compute intra-layer comm volumes
    intra_comm_volumes = []
    for i in range(L):
        c_i = sol.partitioning[i]
        if c_i <= 1:
            intra_comm_volumes.append(0.0)
        else:
            kappa = wl.layer_tp_comm_factor(i)
            act_bytes = wl.layer_activation_bytes(i)
            avg_d = avg_intra_layer_distance(sol.placement[i])
            vol = kappa * act_bytes * (1.0 - 1.0 / c_i) * avg_d
            intra_comm_volumes.append(vol)

    max_vol = max(max(inter_comm_volumes) if inter_comm_volumes else 0,
                  max(intra_comm_volumes) if intra_comm_volumes else 0, 1e-10)

    for r in range(H):
        for c in range(W):
            li = mesh[r, c]
            if li >= 0:
                rect = plt.Rectangle((c - 0.45, r - 0.45), 0.9, 0.9,
                                      facecolor=_layer_color(li),
                                      edgecolor="gray", linewidth=0.3, alpha=0.3)
            else:
                rect = plt.Rectangle((c - 0.45, r - 0.45), 0.9, 0.9,
                                      facecolor="white", edgecolor="lightgray",
                                      linewidth=0.3)
            ax_comm.add_patch(rect)

    # Draw intra-layer comm halos (dashed circles around each layer's centroid)
    for i in range(L):
        if intra_comm_volumes[i] <= 0:
            continue
        vol_norm = intra_comm_volumes[i] / max_vol
        lw = 1 + 5 * vol_norm
        color_val = plt.cm.Oranges(0.3 + 0.7 * vol_norm)
        circle = plt.Circle((centroids[i][1], centroids[i][0]),
                              radius=0.5 + 1.5 * vol_norm,
                              fill=False, edgecolor=color_val,
                              linewidth=lw, linestyle="--", alpha=0.7)
        ax_comm.add_patch(circle)

    # Draw inter-layer comm bands between layer centroids
    for i in range(L - 1):
        vol_norm = inter_comm_volumes[i] / max_vol
        lw = 2 + 8 * vol_norm
        color_val = plt.cm.YlOrRd(0.3 + 0.7 * vol_norm)
        ax_comm.plot([centroids[i][1], centroids[i + 1][1]],
                      [centroids[i][0], centroids[i + 1][0]],
                      color=color_val, linewidth=lw, alpha=0.8,
                      solid_capstyle="round")

    ax_comm.set_xlim(-0.6, W - 0.4)
    ax_comm.set_ylim(-0.6, H - 0.4)
    ax_comm.set_aspect("equal")
    ax_comm.invert_yaxis()
    ax_comm.set_xlabel("Column")
    ax_comm.set_ylabel("Row")
    ax_comm.set_title("Communication Overlay\n(solid = inter-layer, dashed = intra-layer TP)",
                        fontsize=9, fontweight="bold")

    # Add comm volume legend
    sm = plt.cm.ScalarMappable(cmap="YlOrRd",
                                norm=plt.Normalize(vmin=0, vmax=max_vol))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax_comm, shrink=0.6, pad=0.02)
    cbar.set_label("Comm Volume (bytes×hops)", fontsize=7)

    # --- Bottom-right: Latency Breakdown ---
    ax_lat = fig.add_subplot(gs[1, 1])

    comp_per_layer = lat["compute_per_layer"]
    inter_per_layer = lat["inter_comm_per_layer"]  # length L-1
    intra_per_layer = lat["intra_comm_per_layer"]   # length L
    # Pad inter-comm to length L
    inter_padded = np.zeros(L)
    for i in range(len(inter_per_layer)):
        inter_padded[i] = inter_per_layer[i]

    x = np.arange(L)
    ax_lat.bar(x, comp_per_layer, label="Compute",
                color="#4363d8", alpha=0.8, width=0.7)
    ax_lat.bar(x, inter_padded, bottom=comp_per_layer, label="Inter-layer Comm",
                color="#f58231", alpha=0.8, width=0.7)
    ax_lat.bar(x, intra_per_layer, bottom=comp_per_layer + inter_padded,
                label="Intra-layer Comm", color="#e6194b", alpha=0.6, width=0.7)

    # Add total bar
    total_comp = sum(comp_per_layer)
    total_inter = sum(inter_padded)
    total_intra = sum(intra_per_layer)
    ax_lat.bar([L], [total_comp], color="#4363d8", alpha=0.5, width=0.5)
    ax_lat.bar([L], [total_inter], bottom=[total_comp], color="#f58231", alpha=0.5, width=0.5)
    ax_lat.bar([L], [total_intra], bottom=[total_comp + total_inter],
                color="#e6194b", alpha=0.4, width=0.5)
    ax_lat.text(L, total_comp + total_inter + total_intra + total_comp * 0.02,
                f"Total\n{total_comp + total_inter + total_intra:.0f}",
                ha="center", va="bottom", fontsize=7, fontweight="bold")

    ax_lat.set_xticks(list(range(L + 1)))
    ax_lat.set_xticklabels([f"L{i}" for i in range(L)] + ["Total"], fontsize=7)
    ax_lat.set_ylabel("Latency (cycles)")
    ax_lat.set_title("Latency Breakdown", fontsize=10, fontweight="bold")
    ax_lat.legend(fontsize=7, loc="upper right")
    ax_lat.grid(True, axis="y", alpha=0.3)

    if title:
        fig.suptitle(title, fontsize=12, fontweight="bold")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.close(fig)


# ============================================================
# Figure 3: Link Load Heatmap (edge-based NoC pressure)
# ============================================================

def plot_link_load_heatmap(
    sol: MappingSolution,
    wl: Workload,
    acc: Accelerator,
    title: str = "NoC Link Load",
    save_path: Optional[str] = None,
):
    """Heatmap of communication load on mesh links (horizontal and vertical edges)."""
    L = wl.num_layers
    H, W = acc.rows, acc.cols

    # Track load on horizontal links (between (r,c) and (r,c+1))
    # and vertical links (between (r,c) and (r+1,c))
    h_load = np.zeros((H, W - 1))  # horizontal links
    v_load = np.zeros((H - 1, W))  # vertical links

    for i in range(L - 1):
        act_bytes = wl.layer_activation_bytes(i)
        vol_per_core = act_bytes / max(sol.partitioning[i], 1)

        for r1, c1 in sol.placement[i]:
            for r2, c2 in sol.placement[i + 1]:
                # Route: first horizontal, then vertical (deterministic)
                payload = vol_per_core
                cr, cc = r1, c1
                # Horizontal phase
                dc = 1 if c2 > c1 else -1
                while cc != c2:
                    if 0 <= cr < H and 0 <= min(cc, cc + dc) < W - 1:
                        link_c = min(cc, cc + dc)
                        h_load[cr, link_c] += payload
                    cc += dc
                # Vertical phase
                dr = 1 if r2 > r1 else -1
                while cr != r2:
                    if 0 <= min(cr, cr + dr) < H - 1 and 0 <= cc < W:
                        link_r = min(cr, cr + dr)
                        v_load[link_r, cc] += payload
                    cr += dr

    # Intra-layer communication: each core talks to all others in same layer
    for i in range(L):
        c_i = sol.partitioning[i]
        if c_i <= 1:
            continue
        kappa = wl.layer_tp_comm_factor(i)
        act_bytes = wl.layer_activation_bytes(i)
        # Each core sends kappa * act_bytes / c_i data to each of (c_i - 1) others
        payload_per_pair = kappa * act_bytes / c_i

        for idx_a, (r1, c1) in enumerate(sol.placement[i]):
            for idx_b, (r2, c2) in enumerate(sol.placement[i]):
                if idx_a >= idx_b:
                    continue
                cr, cc = r1, c1
                dc = 1 if c2 > c1 else -1
                while cc != c2:
                    if 0 <= cr < H and 0 <= min(cc, cc + dc) < W - 1:
                        h_load[cr, min(cc, cc + dc)] += payload_per_pair
                    cc += dc
                dr = 1 if r2 > r1 else -1
                while cr != r2:
                    if 0 <= min(cr, cr + dr) < H - 1 and 0 <= cc < W:
                        v_load[min(cr, cr + dr), cc] += payload_per_pair
                    cr += dr

    # Create composite visualization
    fig, (ax_h, ax_v) = plt.subplots(1, 2, figsize=(12, 5))

    max_load = max(h_load.max(), v_load.max(), 1e-10)

    # Horizontal links
    im_h = ax_h.imshow(h_load, cmap="YlOrRd", aspect="auto",
                         vmin=0, vmax=max_load, interpolation="nearest")
    ax_h.set_xlabel("Link Column Index")
    ax_h.set_ylabel("Row")
    ax_h.set_title("Horizontal Link Load", fontsize=10, fontweight="bold")
    plt.colorbar(im_h, ax=ax_h, shrink=0.8, label="Traffic Volume")

    # Vertical links
    im_v = ax_v.imshow(v_load, cmap="YlOrRd", aspect="auto",
                         vmin=0, vmax=max_load, interpolation="nearest")
    ax_v.set_xlabel("Column")
    ax_v.set_ylabel("Link Row Index")
    ax_v.set_title("Vertical Link Load", fontsize=10, fontweight="bold")
    plt.colorbar(im_v, ax=ax_v, shrink=0.8, label="Traffic Volume")

    fig.suptitle(title + " (inter + intra-layer)", fontsize=12, fontweight="bold")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.close(fig)


# ============================================================
# Figure 4: Per-Core Utilization Heatmap
# ============================================================

def plot_utilization_heatmap(
    sol: MappingSolution,
    wl: Workload,
    acc: Accelerator,
    title: str = "Per-Core Compute Utilization",
    save_path: Optional[str] = None,
):
    """Heatmap showing per-core compute utilization on the mesh."""
    H, W = acc.rows, acc.cols
    L = wl.num_layers

    util = np.zeros((H, W))
    layer_id = np.full((H, W), -1, dtype=int)

    max_per_core_flops = acc.core_flops_per_cycle

    for i in range(L):
        cores = sol.placement[i]
        c_i = sol.partitioning[i]
        flops = wl.layer_flops(i)
        per_core_flops = flops / c_i  # FLOPs assigned per core
        per_core_intensity = per_core_flops / max_per_core_flops  # relative to max throughput
        for r, c in cores:
            util[r, c] = per_core_intensity
            layer_id[r, c] = i

    fig, ax = plt.subplots(1, 1, figsize=(max(5, W), max(5, H)))

    im = ax.imshow(util, cmap="YlGnBu", interpolation="nearest",
                     vmin=0, vmax=max(util.max(), 1e-10))
    plt.colorbar(im, ax=ax, shrink=0.8, label="Compute Intensity\n(FLOPs/cycle relative to max)")

    for r in range(H):
        for c in range(W):
            li = layer_id[r, c]
            if li >= 0:
                ax.text(c, r, f"L{li}\n{util[r,c]:.1f}", ha="center", va="center",
                         fontsize=max(5, min(8, 60 // max(H, W))),
                         fontweight="bold",
                         color="white" if util[r, c] > util.max() * 0.5 else "black")

    ax.set_xlabel("Column")
    ax.set_ylabel("Row")
    ax.set_title(title, fontsize=11, fontweight="bold")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.close(fig)


# ============================================================
# Figure 5: Partitioning Comparison Across Solvers
# ============================================================

def plot_partitioning_comparison(
    results: dict,
    config_key: str,
    wl: Workload,
    acc: Accelerator,
    save_path: Optional[str] = None,
):
    """Grouped bar chart comparing partitioning (x_i) across solvers for one config."""
    L = wl.num_layers
    solver_names = ["ILP", "SA", "EA", "Greedy"]
    exp = results[config_key]

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    n_solvers = len(solver_names)
    width = 0.8 / n_solvers

    for s_idx, sname in enumerate(solver_names):
        runs = exp["solvers"].get(sname, [])
        best = _best_run(runs)
        if best and "partitioning" in best:
            part = best["partitioning"]
            x = np.arange(L) + (s_idx - n_solvers / 2 + 0.5) * width
            ax.bar(x, part, width * 0.9, color=SOLVER_COLORS[sname],
                    alpha=0.85, label=SOLVER_DISPLAY[sname])

    # Min cores reference line
    min_cores = [acc.min_cores_for_layer(wl, i) for i in range(L)]
    ax.plot(range(L), min_cores, "k--", linewidth=1.5, alpha=0.6, label="Minimum required")

    ax.set_xticks(range(L))
    ax.set_xticklabels([f"L{i}" for i in range(L)], fontsize=9)
    ax.set_ylabel("Allocated Cores")
    ax.set_title(f"Partitioning Comparison: {config_key}", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.close(fig)


# ============================================================
# Retained & upgraded existing plots
# ============================================================

def plot_convergence(
    histories: dict,
    title: str = "Convergence",
    save_path: Optional[str] = None,
):
    """Plot convergence curves for SA and EA."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    for solver_name, hist in histories.items():
        if not hist:
            continue
        iters = [h[0] for h in hist]
        best_costs = [h[2] for h in hist]
        color = SOLVER_COLORS.get(solver_name, "gray")
        ax.plot(iters, best_costs, label=solver_name, linewidth=1.5, color=color)

    ax.set_xlabel("Generation / Iteration")
    ax.set_ylabel("Best Latency (cycles)")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.close(fig)


def plot_latency_comparison(
    results: dict,
    save_path: Optional[str] = None,
):
    """Bar chart comparing final latencies across solvers for each config."""
    configs = list(results.keys())
    solver_names = ["ILP", "SA", "EA", "Greedy"]
    baseline_name = "packed_row"

    n_configs = len(configs)
    n_solvers = len(solver_names) + 1
    width = 0.8 / n_solvers

    fig, ax = plt.subplots(1, 1, figsize=(max(12, n_configs * 2.5), 6))

    for idx, config in enumerate(configs):
        exp = results[config]
        bl_cost = exp["baselines"].get(baseline_name, {}).get("total_time_us", 0)
        x = idx + (0 - n_solvers / 2 + 0.5) * width
        ax.bar(x, bl_cost, width * 0.9, color="gray", alpha=0.6,
                label="Baseline" if idx == 0 else "")

        for s_idx, sname in enumerate(solver_names):
            runs = exp["solvers"].get(sname, [])
            costs = [r["total_time_us"] for r in runs if "total_time_us" in r]
            cost = min(costs) if costs else 0
            x = idx + (s_idx + 1 - n_solvers / 2 + 0.5) * width
            ax.bar(x, cost, width * 0.9, color=SOLVER_COLORS[sname], alpha=0.8,
                    label=SOLVER_DISPLAY[sname] if idx == 0 else "")

    ax.set_xticks(range(n_configs))
    ax.set_xticklabels(configs, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Total Latency (us)")
    ax.set_title("Solver Comparison: Minimum Latency Across Configurations")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.close(fig)


def plot_compute_comm_breakdown(
    results: dict,
    save_path: Optional[str] = None,
):
    """Stacked bar chart showing compute vs inter-comm vs intra-comm breakdown per solver."""
    configs = list(results.keys())
    solver_names = ["ILP", "SA", "EA", "Greedy"]

    fig, axes = plt.subplots(1, len(solver_names),
                              figsize=(5 * len(solver_names), 5), sharey=True)
    if len(solver_names) == 1:
        axes = [axes]

    for ax, sname in zip(axes, solver_names):
        comp_vals, inter_vals, intra_vals, labels = [], [], [], []
        for config in configs:
            exp = results[config]
            runs = exp["solvers"].get(sname, [])
            if runs and "total_time_us" in runs[0]:
                best = min(runs, key=lambda r: r.get("total_time_us", float("inf")))
                comp_us = best.get("compute_time_cycles", 0) / 1000.0
                inter_us = best.get("inter_comm_time_cycles", 0) / 1000.0
                intra_us = best.get("intra_comm_time_cycles", 0) / 1000.0
                comp_vals.append(comp_us)
                inter_vals.append(inter_us)
                intra_vals.append(intra_us)
            else:
                comp_vals.append(0)
                inter_vals.append(0)
                intra_vals.append(0)
            labels.append(config)

        x = range(len(labels))
        ax.bar(x, comp_vals, label="Compute", color="#4363d8", alpha=0.8)
        ax.bar(x, inter_vals, bottom=comp_vals, label="Inter-layer Comm", color="#f58231", alpha=0.8)
        ax.bar(x, intra_vals, bottom=np.array(comp_vals) + np.array(inter_vals),
                label="Intra-layer Comm", color="#e6194b", alpha=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
        ax.set_title(SOLVER_DISPLAY[sname])
        ax.legend(fontsize=7)
        ax.grid(True, axis="y", alpha=0.3)

    axes[0].set_ylabel("Time (us)")
    fig.suptitle("Compute vs Inter-Comm vs Intra-Comm Breakdown", fontsize=13)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.close(fig)


def plot_algorithm_wins_by_type(
    results: dict,
    save_path: Optional[str] = None,
):
    """Grouped bar chart: number of configs where each algorithm is best, by workload type."""
    solver_names = ["ILP", "SA", "EA", "Greedy"]

    type_keys = {"MLP": [], "Transformer": [], "ConvNet": []}
    for config in results:
        if config.startswith("ConvNet"):
            type_keys["ConvNet"].append(config)
        elif config.startswith("Transformer"):
            type_keys["Transformer"].append(config)
        else:
            type_keys["MLP"].append(config)

    wins = {t: {s: 0 for s in solver_names} for t in type_keys}
    for wtype, configs in type_keys.items():
        for config in configs:
            exp = results[config]
            best_cost = float("inf")
            best_solver = None
            for sname in solver_names:
                runs = exp["solvers"].get(sname, [])
                costs = [r["total_time_us"] for r in runs if "total_time_us" in r]
                if costs and min(costs) < best_cost:
                    best_cost = min(costs)
                    best_solver = sname
            if best_solver:
                wins[wtype][best_solver] += 1

    types = list(type_keys.keys())
    x = np.arange(len(types))
    width = 0.18

    fig, ax = plt.subplots(figsize=(8, 5))
    for i, sname in enumerate(solver_names):
        vals = [wins[t][sname] for t in types]
        bars = ax.bar(x + i * width - 1.5 * width, vals, width * 0.9,
                       label=SOLVER_DISPLAY[sname], color=SOLVER_COLORS[sname], alpha=0.85)
        for bar, v in zip(bars, vals):
            if v > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.08,
                         str(v), ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(types, fontsize=12)
    ax.set_ylabel("Number of Best Configs", fontsize=11)
    ax.set_title("Algorithm Wins by Workload Type", fontsize=13)
    ax.legend(fontsize=10)
    ax.set_ylim(0, max(max(v for v in wins[t].values()) for t in types) + 1.5)
    ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.close(fig)


def plot_normalized_heatmap(
    results: dict,
    save_path: Optional[str] = None,
):
    """Heatmap of normalized latency (best=1.0) for each solver across all configs."""
    solver_names = ["ILP", "SA", "EA", "Greedy"]
    configs = list(results.keys())

    raw = np.full((len(configs), len(solver_names)), np.nan)
    for i, config in enumerate(configs):
        exp = results[config]
        for j, sname in enumerate(solver_names):
            runs = exp["solvers"].get(sname, [])
            costs = [r["total_time_us"] for r in runs if "total_time_us" in r]
            if costs:
                raw[i, j] = min(costs)

    row_mins = np.nanmin(raw, axis=1, keepdims=True)
    normed = raw / row_mins

    short_labels = []
    for c in configs:
        parts = c.split("/")
        wl = parts[0].replace("Medium-", "Med-").replace("Small-", "S-").replace(
            "Large-", "L-").replace("Sparse-", "Sp-").replace("Transformer-", "Tf-")
        short_labels.append(wl + "\n" + parts[1].replace("-mesh", ""))

    fig, ax = plt.subplots(figsize=(7, max(6, len(configs) * 0.42)))
    im = ax.imshow(normed, cmap="RdYlGn_r", aspect="auto", vmin=1.0, vmax=2.0)

    ax.set_xticks(range(len(solver_names)))
    ax.set_xticklabels([SOLVER_DISPLAY[s] for s in solver_names], fontsize=10)
    ax.set_yticks(range(len(short_labels)))
    ax.set_yticklabels(short_labels, fontsize=8)

    for i in range(len(configs)):
        for j in range(len(solver_names)):
            v = normed[i, j]
            if np.isnan(v):
                ax.text(j, i, "—", ha="center", va="center", fontsize=8, color="gray")
            else:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                         fontsize=7, fontweight="bold",
                         color="white" if v > 1.4 else "black")

    cbar = plt.colorbar(im, ax=ax, shrink=0.6)
    cbar.set_label("Normalized Latency (best = 1.0)", fontsize=9)
    ax.set_title("Solver Performance Heatmap (Normalized to Best)", fontsize=12)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.close(fig)


def plot_speedup_over_baseline(
    results: dict,
    save_path: Optional[str] = None,
):
    """Bar chart: speedup of best solver over best baseline, grouped by workload type."""
    solver_names = ["ILP", "SA", "EA", "Greedy"]

    type_configs = {"MLP": [], "Transformer": [], "ConvNet": []}
    for config in results:
        if config.startswith("ConvNet"):
            type_configs["ConvNet"].append(config)
        elif config.startswith("Transformer"):
            type_configs["Transformer"].append(config)
        else:
            type_configs["MLP"].append(config)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    type_labels = ["MLP", "Transformer", "ConvNet"]

    for ax_idx, wtype in enumerate(type_labels):
        configs = type_configs[wtype]
        short_names, speedups, best_solver_names = [], [], []

        for config in configs:
            exp = results[config]
            bl_costs = [bres["total_time_us"] for bres in exp["baselines"].values()
                        if "total_time_us" in bres]
            best_bl = min(bl_costs) if bl_costs else 1.0

            best_cost, best_sname = float("inf"), ""
            for sname in solver_names:
                runs = exp["solvers"].get(sname, [])
                costs = [r["total_time_us"] for r in runs if "total_time_us" in r]
                if costs and min(costs) < best_cost:
                    best_cost = min(costs)
                    best_sname = sname

            speedups.append(best_bl / best_cost if best_cost > 0 else 1.0)
            best_solver_names.append(best_sname)
            parts = config.split("/")
            short_names.append(parts[1].replace("-mesh", ""))

        x = np.arange(len(short_names))
        ax = axes[ax_idx]
        colors = [SOLVER_COLORS[s] for s in best_solver_names]
        bar_objs = ax.bar(x, speedups, color=colors, alpha=0.85, width=0.6)

        for bar_obj, sp, sname in zip(bar_objs, speedups, best_solver_names):
            ax.text(bar_obj.get_x() + bar_obj.get_width() / 2,
                     bar_obj.get_height() + 0.02,
                     f"{sp:.2f}x\n({SOLVER_DISPLAY[sname]})",
                     ha="center", va="bottom", fontsize=8, fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels(short_names, fontsize=9)
        ax.set_title(wtype, fontsize=12)
        ax.axhline(y=1.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
        ax.grid(True, axis="y", alpha=0.3)

    axes[0].set_ylabel("Speedup over Best Baseline", fontsize=11)
    fig.suptitle("Best Solver Speedup over Best Baseline by Workload Type", fontsize=13)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.close(fig)


# ============================================================
# Small multiples: compare solvers on one config
# ============================================================

def plot_solver_comparison_small_multiples(
    results: dict,
    config_key: str,
    wl: Workload,
    acc: Accelerator,
    save_path: Optional[str] = None,
):
    """Small multiples: one placement subplot per solver, same mesh, same colors."""
    L = wl.num_layers
    H, W = acc.rows, acc.cols
    solver_names = ["SA", "Greedy", "EA", "ILP"]
    exp = results[config_key]

    valid_solvers = []
    for sname in solver_names:
        runs = exp["solvers"].get(sname, [])
        best = _best_run(runs)
        if best and "partitioning" in best:
            valid_solvers.append((sname, best))

    n = len(valid_solvers)
    if n == 0:
        return

    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    if n == 1:
        axes = [axes]

    for ax, (sname, best_run) in zip(axes, valid_solvers):
        part = best_run["partitioning"]
        sol = _reconstruct_solution(part, best_run, wl, acc)
        lat = compute_latency(sol, wl, acc)

        mesh = np.full((H, W), -1)
        for i, layer_cores in enumerate(sol.placement):
            for r, c in layer_cores:
                mesh[r, c] = i

        for r in range(H):
            for c in range(W):
                li = mesh[r, c]
                if li >= 0:
                    color = _layer_color(li)
                    rect = plt.Rectangle((c - 0.45, r - 0.45), 0.9, 0.9,
                                          facecolor=color, edgecolor="black",
                                          linewidth=0.5, alpha=0.85)
                    ax.add_patch(rect)
                else:
                    rect = plt.Rectangle((c - 0.45, r - 0.45), 0.9, 0.9,
                                          facecolor="white", edgecolor="lightgray",
                                          linewidth=0.3)
                    ax.add_patch(rect)

        ax.set_xlim(-0.6, W - 0.4)
        ax.set_ylim(-0.6, H - 0.4)
        ax.set_aspect("equal")
        ax.invert_yaxis()
        ax.set_title(f"{SOLVER_DISPLAY[sname]}\n{lat['total_time']:.0f} cycles", fontsize=9)

    fig.suptitle(f"Placement Comparison: {config_key}", fontsize=11, fontweight="bold")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.close(fig)


# ============================================================
# Main: generate all plots
# ============================================================

# Representative configs for detailed figures
REPRESENTATIVE_CONFIGS = [
    "Small-MLP/4x4-mesh",
    "Medium-MLP/4x4-mesh",
    "Sparse-MLP/4x4-mesh",
    "Large-MLP/8x8-mesh",
    "Transformer-S/6x6-mesh",
    "Transformer-L/6x6-mesh",
    "ConvNet/4x4-mesh",
]

# Workload and accelerator definitions (must match experiment.py)
from experiment import WORKLOADS, ACCELERATORS


def generate_all_plots():
    """Load experiment results and generate all plots."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results = _load_results()
    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- Summary charts (kept from before) ---
    print("Generating summary charts...")
    plot_latency_comparison(results, _out("latency_comparison.png"))
    plot_compute_comm_breakdown(results, _out("compute_comm_breakdown.png"))
    plot_algorithm_wins_by_type(results, _out("algorithm_wins_by_type.png"))
    plot_normalized_heatmap(results, _out("normalized_heatmap.png"))
    plot_speedup_over_baseline(results, _out("speedup_over_baseline.png"))

    # --- New: Workload overview for representative workloads ---
    print("Generating workload overview figures...")
    for config_key in REPRESENTATIVE_CONFIGS:
        if config_key not in results:
            continue
        wl_name, acc_name = config_key.split("/")
        wl = WORKLOADS[wl_name]
        acc = ACCELERATORS[acc_name]
        safe = config_key.replace("/", "_")

        plot_workload_overview(
            wl, acc,
            title=f"Problem Overview: {config_key}",
            save_path=_out(f"overview_{safe}.png"),
        )

    # --- New: 4-panel mapping detail for representative configs (best solver) ---
    print("Generating mapping detail figures...")
    solver_names = ["SA", "Greedy", "EA", "ILP"]
    for config_key in REPRESENTATIVE_CONFIGS:
        if config_key not in results:
            continue
        wl_name, acc_name = config_key.split("/")
        wl = WORKLOADS[wl_name]
        acc = ACCELERATORS[acc_name]
        safe = config_key.replace("/", "_")
        exp = results[config_key]

        # Find best solver
        best_cost = float("inf")
        best_run_data = None
        best_sname = None
        for sname in solver_names:
            runs = exp["solvers"].get(sname, [])
            br = _best_run(runs)
            if br and "total_time_cycles" in br and br["total_time_cycles"] < best_cost:
                best_cost = br["total_time_cycles"]
                best_run_data = br
                best_sname = sname

        if best_run_data and "partitioning" in best_run_data:
            part = best_run_data["partitioning"]
            sol = _reconstruct_solution(part, best_run_data, wl, acc)
            plot_mapping_detail(
                sol, wl, acc,
                title=f"Mapping Detail: {config_key} (Best: {SOLVER_DISPLAY[best_sname]})",
                save_path=_out(f"mapping_detail_{safe}.png"),
            )

            # Link load heatmap for this config
            plot_link_load_heatmap(
                sol, wl, acc,
                title=f"NoC Link Load: {config_key}",
                save_path=_out(f"link_load_{safe}.png"),
            )

            # Utilization heatmap
            plot_utilization_heatmap(
                sol, wl, acc,
                title=f"Per-Core Utilization: {config_key}",
                save_path=_out(f"utilization_{safe}.png"),
            )

            # Partitioning comparison
            plot_partitioning_comparison(
                results, config_key, wl, acc,
                save_path=_out(f"partitioning_{safe}.png"),
            )

        # Solver comparison small multiples
        plot_solver_comparison_small_multiples(
            results, config_key, wl, acc,
            save_path=_out(f"solver_compare_{safe}.png"),
        )

    # --- Convergence plots ---
    print("Generating convergence plots...")
    for config_key, exp in results.items():
        safe = config_key.replace("/", "_")
        histories = {}
        for sname, runs in exp["solvers"].items():
            for run in runs:
                if "history" in run and run["history"]:
                    histories[sname] = run["history"]
        if histories:
            plot_convergence(
                histories,
                title=f"Convergence: {config_key}",
                save_path=_out(f"convergence_{safe}.png"),
            )

    print(f"\nAll plots saved to {_OUTPUT_DIR}/")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate plots from experiment results.")
    parser.add_argument("--results", type=str, default=None,
                        help="Path to experiment results JSON (default: results/experiment_results.json)")
    args = parser.parse_args()
    if args.results:
        _RESULTS_PATH = args.results
    generate_all_plots()
