"""Experiment runner: evaluate all solvers across multiple workloads and accelerators."""

from __future__ import annotations

import json
import sys
import os
import time
from pathlib import Path

import numpy as np

# Ensure src is importable
sys.path.insert(0, os.path.dirname(__file__))

from model import (
    Workload, Accelerator, LayerSpec, conv2d_spec, linear_spec,
    compute_latency, cycles_to_us, reset_eval_counter, get_eval_count,
)
from baseline import BASELINES
from solvers.ilp_solver import solve_ilp
from solvers.sa_solver import solve_sa
from solvers.ea_solver import solve_ea
from solvers.greedy_solver import solve_greedy

RESULTS_DIR = Path(__file__).parent.parent / "results"


# === Transformer helper ===
def _transformer_workload_dims(d_model: int, ffn_dim: int, num_layers: int) -> list:
    """Layer dims for a stack of transformer encoder layers.

    Each layer: Q, K, V projections + output projection + FFN expand + FFN contract = 6 linear ops.
    """
    single = [d_model, d_model, d_model, d_model, d_model, ffn_dim, d_model]
    dims = list(single)
    for _ in range(1, num_layers):
        dims.extend(single[1:])
    return dims


# === Workload definitions ===
WORKLOADS = {
    "Small-MLP": Workload(
        name="Small-MLP",
        layer_dims=[128, 256, 128, 64],
        sparsity=0.0,
    ),
    "Medium-MLP": Workload(
        name="Medium-MLP",
        layer_dims=[256, 512, 1024, 512, 256],
        sparsity=0.0,
    ),
    "Large-MLP": Workload(
        name="Large-MLP",
        layer_dims=[256, 512, 1024, 1024, 512, 256],
        sparsity=0.0,
    ),
    "Sparse-MLP": Workload(
        name="Sparse-MLP",
        layer_dims=[256, 512, 1024, 512, 256],
        sparsity=0.5,
    ),
    "Transformer-S": Workload(
        name="Transformer-S",
        layer_dims=_transformer_workload_dims(d_model=128, ffn_dim=512, num_layers=2),
    ),
    "Transformer-L": Workload(
        name="Transformer-L",
        layer_dims=_transformer_workload_dims(d_model=256, ffn_dim=1024, num_layers=2),
    ),
    "ConvNet": Workload(
        name="ConvNet",
        layer_specs=[
            # Conv1: 3→64, 5×5, 32×32, pad=2
            conv2d_spec(cin=3, cout=64, kernel_size=5, h_in=32, w_in=32, pad=2),
            # Conv2: 64→128, 3×3, 16×16, pad=1 (after 2×2 pool)
            conv2d_spec(cin=64, cout=128, kernel_size=3, h_in=16, w_in=16, pad=1),
            # Conv3: 128→256, 3×3, 8×8, pad=1 (after 2×2 pool)
            conv2d_spec(cin=128, cout=256, kernel_size=3, h_in=8, w_in=8, pad=1),
            # FC: 256→10 (after global average pooling)
            linear_spec(in_dim=256, out_dim=10),
        ],
        tp_comm_factors=[0.5, 0.5, 0.5, 1.0],
    ),
}

# === Accelerator definitions ===
ACCELERATORS = {
    "4x4-mesh": Accelerator(
        name="4x4-mesh",
        rows=4,
        cols=4,
        core_memory=512 * 1024,  # 512 KB per core
        core_flops_per_cycle=64,
        comm_beta=1.0,
        frequency_ghz=1.0,
    ),
    "6x6-mesh": Accelerator(
        name="6x6-mesh",
        rows=6,
        cols=6,
        core_memory=512 * 1024,
        core_flops_per_cycle=64,
        comm_beta=1.0,
        frequency_ghz=1.0,
    ),
    "8x8-mesh": Accelerator(
        name="8x8-mesh",
        rows=8,
        cols=8,
        core_memory=512 * 1024,
        core_flops_per_cycle=64,
        comm_beta=1.0,
        frequency_ghz=1.0,
    ),
}

# === Solver configurations ===
SOLVER_CONFIGS = {
    "ILP": {
        "fn": lambda wl, acc: solve_ilp(wl, acc, time_limit=60),
        "runs": 1,
        "max_cores": 16,  # Intra-layer ILP terms scale quadratically with mesh size
    },
    "SA": {
        "fn": lambda wl, acc, seed=0: solve_sa(
            wl, acc, max_iters=3000, init_temp=100.0, cooling_rate=0.995, seed=seed
        ),
        "runs": 5,
    },
    "EA": {
        "fn": lambda wl, acc, seed=0: solve_ea(
            wl, acc, lambda_part=4, lambda_place=4, generations=40, seed=seed
        ),
        "runs": 5,
    },
    "Greedy": {
        "fn": lambda wl, acc: solve_greedy(wl, acc, kl_passes=8, part_refine_iters=15),
        "runs": 1,
    },
}


def run_single(wl_name: str, acc_name: str, solver_name: str) -> dict:
    """Run a single experiment configuration."""
    wl = WORKLOADS[wl_name]
    acc = ACCELERATORS[acc_name]
    cfg = SOLVER_CONFIGS[solver_name]

    # Skip ILP for large meshes
    max_cores = cfg.get("max_cores", float("inf"))
    if acc.total_cores > max_cores:
        print(f"  Skipping (mesh too large: {acc.total_cores} > {max_cores})")
        return [{"run": 0, "error": "skipped_large_mesh"}]

    results = []

    for run in range(cfg["runs"]):
        t0 = time.time()
        reset_eval_counter()
        try:
            if solver_name in ("SA", "EA"):
                sol, lat, info = cfg["fn"](wl, acc, seed=run * 42)
            else:
                sol, lat, info = cfg["fn"](wl, acc)

            result = {
                "run": run,
                "total_time_cycles": lat["total_time"],
                "compute_time_cycles": lat["compute_time"],
                "comm_time_cycles": lat["comm_time"],
                "inter_comm_time_cycles": lat["inter_comm_time"],
                "intra_comm_time_cycles": lat["intra_comm_time"],
                "total_time_us": cycles_to_us(lat["total_time"], acc.frequency_ghz),
                "solve_time_s": info.get("solve_time_s", time.time() - t0),
                "eval_count": get_eval_count(),
                "partitioning": sol.partitioning,
                "placement": [[[r, c] for r, c in layer] for layer in sol.placement],
                "history": info.get("history", []),
            }
        except Exception as e:
            result = {"run": run, "error": str(e)}
            print(f"  ERROR: {e}")

        results.append(result)
        print(f"  Run {run}: {result.get('total_time_us', 'N/A')} us")

    return results


def run_baselines(wl_name: str, acc_name: str) -> dict:
    """Run all baseline heuristics."""
    wl = WORKLOADS[wl_name]
    acc = ACCELERATORS[acc_name]
    results = {}
    for name, fn in BASELINES.items():
        try:
            sol, lat = fn(wl, acc)
            results[name] = {
                "total_time_cycles": lat["total_time"],
                "compute_time_cycles": lat["compute_time"],
                "comm_time_cycles": lat["comm_time"],
                "inter_comm_time_cycles": lat["inter_comm_time"],
                "intra_comm_time_cycles": lat["intra_comm_time"],
                "total_time_us": cycles_to_us(lat["total_time"], acc.frequency_ghz),
                "partitioning": sol.partitioning,
                "placement": [[[r, c] for r, c in layer] for layer in sol.placement],
            }
        except Exception as e:
            results[name] = {"error": str(e)}
    return results


def _is_feasible(wl_name: str, acc_name: str) -> bool:
    """Check if workload fits on the accelerator."""
    wl = WORKLOADS[wl_name]
    acc = ACCELERATORS[acc_name]
    min_cores = sum(acc.min_cores_for_layer(wl, i) for i in range(wl.num_layers))
    return min_cores <= acc.total_cores


def run_all_experiments():
    """Run the full experimental sweep."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    all_results = {}

    total = len(WORKLOADS) * len(ACCELERATORS) * (len(SOLVER_CONFIGS) + 1)
    count = 0

    for wl_name in WORKLOADS:
        for acc_name in ACCELERATORS:
            key = f"{wl_name}/{acc_name}"
            print(f"\n{'='*60}")
            print(f"  {key}")
            print(f"{'='*60}")

            if not _is_feasible(wl_name, acc_name):
                print(f"  SKIPPED (workload doesn't fit)")
                continue

            exp = {"baselines": {}, "solvers": {}}

            # Baselines
            count += 1
            print(f"[{count}/{total}] Baselines...")
            exp["baselines"] = run_baselines(wl_name, acc_name)
            for bname, bres in exp["baselines"].items():
                print(f"  {bname}: {bres.get('total_time_us', 'N/A')} us")

            # Solvers
            for solver_name in SOLVER_CONFIGS:
                count += 1
                print(f"\n[{count}/{total}] {solver_name}...")
                exp["solvers"][solver_name] = run_single(wl_name, acc_name, solver_name)

            all_results[key] = exp

    # Save results
    if not all(a.intra_comm_enabled for a in ACCELERATORS.values()):
        suffix = "_inter_only"
    elif all(not a.intra_serialized for a in ACCELERATORS.values()):
        suffix = "_intra_parallel"
    else:
        suffix = ""
    out_path = RESULTS_DIR / f"experiment_results{suffix}.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")

    # Print summary table
    print_summary(all_results)
    return all_results


def print_summary(results: dict):
    """Print a summary table of results."""
    print(f"\n{'='*80}")
    print("SUMMARY TABLE (Total Latency in microseconds)")
    print(f"{'='*80}")
    print(f"{'Config':<25} {'Best Baseline':>15} {'ILP':>12} {'SA':>12} {'EA':>12} {'Greedy':>12}")
    print("-" * 88)

    for key, exp in results.items():
        # Best baseline
        baseline_costs = []
        for bname, bres in exp["baselines"].items():
            if "total_time_us" in bres:
                baseline_costs.append(bres["total_time_us"])
        best_bl = min(baseline_costs) if baseline_costs else float("inf")

        # Solver results (best across runs)
        solver_costs = {}
        for sname, runs in exp["solvers"].items():
            costs = [r["total_time_us"] for r in runs if "total_time_us" in r]
            solver_costs[sname] = min(costs) if costs else float("inf")

        print(
            f"{key:<25} {best_bl:>15.2f} "
            f"{solver_costs.get('ILP', float('inf')):>12.2f} "
            f"{solver_costs.get('SA', float('inf')):>12.2f} "
            f"{solver_costs.get('EA', float('inf')):>12.2f} "
            f"{solver_costs.get('Greedy', float('inf')):>12.2f}"
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run spatial mapping experiments.")
    parser.add_argument("--inter-only", action="store_true",
                        help="Disable intra-layer communication (inter-only formulation)")
    parser.add_argument("--intra-parallel", action="store_true",
                        help="Use parallel-injection intra comm model (divide by x_i)")
    parser.add_argument("--max-cores-ilp", type=int, default=None,
                        help="Override ILP max_cores threshold")
    args = parser.parse_args()

    if args.inter_only:
        print("=" * 60)
        print("  RUNNING IN INTER-ONLY MODE (no intra-layer comm)")
        print("=" * 60)
        for acc in ACCELERATORS.values():
            acc.intra_comm_enabled = False

    if args.intra_parallel:
        print("=" * 60)
        print("  RUNNING IN INTRA-PARALLEL MODE (intra term / x_i)")
        print("=" * 60)
        for acc in ACCELERATORS.values():
            acc.intra_serialized = False

    if args.max_cores_ilp is not None:
        SOLVER_CONFIGS["ILP"]["max_cores"] = args.max_cores_ilp
    elif args.inter_only:
        # Without intra-layer w variables, ILP can handle larger meshes
        SOLVER_CONFIGS["ILP"]["max_cores"] = 49

    run_all_experiments()
