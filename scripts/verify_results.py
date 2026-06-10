"""Verify experiment results by recomputing latency from saved placements.

Reads all experiment_results*.json files, reconstructs MappingSolutions,
and checks that compute/inter/intra/total match the saved values.
"""
import json
import sys
import os
from pathlib import Path
from itertools import product

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from model import (
    Workload, Accelerator, MappingSolution, compute_latency,
    conv2d_spec, linear_spec,
)

# Must match experiment.py definitions
WORKLOADS = {
    "Small-MLP": Workload(name="Small-MLP", layer_dims=[128, 256, 128, 64]),
    "Medium-MLP": Workload(name="Medium-MLP", layer_dims=[256, 512, 1024, 512, 256]),
    "Large-MLP": Workload(name="Large-MLP", layer_dims=[256, 512, 1024, 1024, 512, 256]),
    "Sparse-MLP": Workload(name="Sparse-MLP", layer_dims=[256, 512, 1024, 512, 256], sparsity=0.5),
    "ConvNet": Workload(
        name="ConvNet",
        layer_specs=[
            conv2d_spec(cin=3, cout=64, kernel_size=5, h_in=32, w_in=32, pad=2),
            conv2d_spec(cin=64, cout=128, kernel_size=3, h_in=16, w_in=16, pad=1),
            conv2d_spec(cin=128, cout=256, kernel_size=3, h_in=8, w_in=8, pad=1),
            linear_spec(in_dim=256, out_dim=10),
        ],
        tp_comm_factors=[0.5, 0.5, 0.5, 1.0],
    ),
}


def _transformer_dims(d_model, ffn_dim, num_layers):
    single = [d_model, d_model, d_model, d_model, d_model, ffn_dim, d_model]
    dims = list(single)
    for _ in range(1, num_layers):
        dims.extend(single[1:])
    return dims


WORKLOADS["Transformer-S"] = Workload(
    name="Transformer-S",
    layer_dims=_transformer_dims(d_model=128, ffn_dim=512, num_layers=2),
)
WORKLOADS["Transformer-L"] = Workload(
    name="Transformer-L",
    layer_dims=_transformer_dims(d_model=256, ffn_dim=1024, num_layers=2),
)

ACCELERATORS = {
    "4x4-mesh": Accelerator(name="4x4-mesh", rows=4, cols=4, core_memory=512*1024,
                              core_flops_per_cycle=64, comm_beta=1.0, frequency_ghz=1.0),
    "6x6-mesh": Accelerator(name="6x6-mesh", rows=6, cols=6, core_memory=512*1024,
                              core_flops_per_cycle=64, comm_beta=1.0, frequency_ghz=1.0),
    "8x8-mesh": Accelerator(name="8x8-mesh", rows=8, cols=8, core_memory=512*1024,
                              core_flops_per_cycle=64, comm_beta=1.0, frequency_ghz=1.0),
}

TOLERANCE = 1e-3  # relative tolerance


def verify_file(path: Path):
    """Verify one results JSON file."""
    print(f"\n{'='*60}")
    print(f"  Verifying: {path.name}")
    print(f"{'='*60}")

    with open(path) as f:
        data = json.load(f)

    # Determine formulation from filename
    is_inter_only = "_inter_only" in path.stem
    is_intra_parallel = "_intra_parallel" in path.stem
    for acc in ACCELERATORS.values():
        acc.intra_comm_enabled = not is_inter_only
        acc.intra_serialized = not is_intra_parallel

    total, passed, failed = 0, 0, 0

    for key, exp in data.items():
        wl_name, acc_name = key.split("/")
        if wl_name not in WORKLOADS or acc_name not in ACCELERATORS:
            continue
        wl = WORKLOADS[wl_name]
        acc = ACCELERATORS[acc_name]

        # Check baselines
        for bname, bres in exp.get("baselines", {}).items():
            if "error" in bres or "partitioning" not in bres:
                continue
            total += 1
            ok = _check_solution(bres, wl, acc, f"{key}/baseline/{bname}")
            if ok:
                passed += 1
            else:
                failed += 1

        # Check solver runs
        for sname, runs in exp.get("solvers", {}).items():
            for run_data in runs:
                if "error" in run_data or "partitioning" not in run_data:
                    continue
                total += 1
                ok = _check_solution(run_data, wl, acc, f"{key}/{sname}/run{run_data.get('run', '?')}")
                if ok:
                    passed += 1
                else:
                    failed += 1

    print(f"\n  Results: {passed}/{total} PASSED", end="")
    if failed:
        print(f", {failed} FAILED")
    else:
        print(" -- ALL OK")

    return failed == 0


def _check_solution(data: dict, wl: Workload, acc: Accelerator, label: str) -> bool:
    """Recompute latency and compare with saved values."""
    part = data["partitioning"]
    placement_data = data.get("placement")

    if not placement_data:
        print(f"  SKIP {label} (no placement data)")
        return True

    placement = [[tuple(p) for p in layer] for layer in placement_data]

    try:
        sol = MappingSolution(partitioning=part, placement=placement)
        lat = compute_latency(sol, wl, acc)
    except Exception as e:
        print(f"  FAIL {label}: compute_latency error: {e}")
        return False

    checks = [
        ("compute_time_cycles", lat["compute_time"]),
        ("inter_comm_time_cycles", lat["inter_comm_time"]),
        ("intra_comm_time_cycles", lat["intra_comm_time"]),
        ("total_time_cycles", lat["total_time"]),
    ]

    all_ok = True
    for saved_key, recomputed in checks:
        saved = data.get(saved_key)
        if saved is None:
            continue
        rel_err = abs(saved - recomputed) / max(abs(recomputed), 1e-10)
        if rel_err > TOLERANCE:
            print(f"  FAIL {label}: {saved_key} saved={saved:.2f} recomputed={recomputed:.2f} "
                  f"rel_err={rel_err:.6f}")
            all_ok = False

    return all_ok


def main():
    results_dir = Path(__file__).parent.parent / "results"
    all_ok = True

    for name in [
        "experiment_results.json",
        "experiment_results_inter_only.json",
        "experiment_results_intra_parallel.json",
    ]:
        path = results_dir / name
        if path.exists():
            ok = verify_file(path)
            all_ok = all_ok and ok
        else:
            print(f"SKIP: {path} not found")

    print(f"\n{'='*60}")
    if all_ok:
        print("  OVERALL: ALL VERIFICATIONS PASSED")
    else:
        print("  OVERALL: SOME VERIFICATIONS FAILED")
    print(f"{'='*60}")

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
