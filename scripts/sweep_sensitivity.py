"""Communication-cost sensitivity study: κ wins, β threshold, and core-utilization.

Also emits report-facing mapping figures, e.g.
``results/mapping_detail_Large-MLP_8x8-mesh_beta0.1.png`` (Greedy+KL, β=0.1, 64/64).

Quick regenerate of that figure only::

    python scripts/sweep_sensitivity.py --mapping-only
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from experiment import WORKLOADS, ACCELERATORS, _is_feasible
from model import cycles_to_us, reset_eval_counter
from solvers.greedy_solver import solve_greedy
from solvers.sa_solver import solve_sa
from solvers.ea_solver import solve_ea

RESULTS_DIR = Path(__file__).parent.parent / "results"
LOGS_DIR = Path(__file__).parent.parent / "logs"

KAPPA_MULTIPLIERS = [0.0, 0.25, 0.5, 1.0, 2.0]
BETA_MULTIPLIERS = [0.0, 0.1, 0.25, 0.5, 1.0]
SOLVERS = {
    "SA": lambda wl, acc, seed: solve_sa(wl, acc, seed=seed, max_iters=3000),
    "EA": lambda wl, acc, seed: solve_ea(wl, acc, seed=seed, generations=40),
    "Greedy": lambda wl, acc, seed: solve_greedy(wl, acc),
}
RUNS = 5
FOCUS_CONFIGS = [
    "Large-MLP/8x8-mesh",
    "Medium-MLP/8x8-mesh",
    "Small-MLP/8x8-mesh",
]
# Report Fig. mapping (right panel): cheap comm, intra comm still enabled
REPORT_SENSITIVITY_CONFIG = "Large-MLP/8x8-mesh"
REPORT_BETA_CHEAP = 0.1


def _scaled_workload(wl_name: str, kappa_mult: float):
    wl = copy.copy(WORKLOADS[wl_name])
    wl.kappa_scale = kappa_mult
    return wl


def _scaled_accel(acc_name: str, beta_mult: float, intra_enabled: bool = True):
    acc = copy.copy(ACCELERATORS[acc_name])
    acc.comm_beta = beta_mult
    acc.intra_comm_enabled = intra_enabled
    return acc


def sweep_kappa_wins() -> dict:
    """Global κ sweep: solver win counts over all 19 feasible configs."""
    sweep = {}
    for kappa_mult in KAPPA_MULTIPLIERS:
        print(f"\n{'='*60}\n  κ multiplier = {kappa_mult}\n{'='*60}")
        wins = {s: 0 for s in SOLVERS}
        per_config = {}

        for wl_name in WORKLOADS:
            for acc_name in ACCELERATORS:
                key = f"{wl_name}/{acc_name}"
                if not _is_feasible(wl_name, acc_name):
                    continue

                wl = _scaled_workload(wl_name, kappa_mult)
                acc = ACCELERATORS[acc_name]
                best_solver = None
                best_us = float("inf")
                solver_us = {}

                for sname, fn in SOLVERS.items():
                    run_costs = []
                    for run in range(RUNS):
                        reset_eval_counter()
                        seed = run * 42
                        if sname == "Greedy":
                            sol, lat, _ = fn(wl, acc, 0)
                        else:
                            sol, lat, _ = fn(wl, acc, seed)
                        run_costs.append(cycles_to_us(lat["total_time"], acc.frequency_ghz))
                    best = min(run_costs)
                    solver_us[sname] = best
                    if best < best_us:
                        best_us = best
                        best_solver = sname

                wins[best_solver] += 1
                per_config[key] = {"best": best_solver, "latencies_us": solver_us}
                print(f"  {key}: best={best_solver} ({best_us:.2f} us)")

        sweep[str(kappa_mult)] = {"wins": wins, "configs": per_config}

    out = RESULTS_DIR / "sensitivity_kappa.json"
    with open(out, "w") as f:
        json.dump(sweep, f, indent=2)
    print(f"\nSaved {out}")
    _plot_kappa_wins(sweep, RESULTS_DIR / "sensitivity_kappa.png")
    return sweep


def sweep_core_usage() -> dict:
    """κ and β sweeps on focus configs: cores used, latency, partition."""
    study = {"kappa_sweep": {}, "beta_sweep": {}, "inter_only": {}}

    for key in FOCUS_CONFIGS:
        wl_name, acc_name = key.split("/")
        acc = ACCELERATORS[acc_name]
        total = acc.total_cores
        study["kappa_sweep"][key] = []
        study["beta_sweep"][key] = []

        print(f"\n--- Core usage: {key} (K={total}) ---")
        for kappa in KAPPA_MULTIPLIERS:
            wl = _scaled_workload(wl_name, kappa)
            sol, lat, _ = solve_greedy(wl, acc)
            used = sum(sol.partitioning)
            row = {
                "kappa": kappa,
                "cores_used": used,
                "cores_total": total,
                "utilization": used / total,
                "partitioning": sol.partitioning,
                "latency_us": cycles_to_us(lat["total_time"], acc.frequency_ghz),
            }
            study["kappa_sweep"][key].append(row)
            print(f"  κ={kappa:4}  cores={used:2}/{total}  util={used/total:.0%}  "
                  f"part={sol.partitioning}  {row['latency_us']:.2f} us")

        for beta in BETA_MULTIPLIERS:
            wl = WORKLOADS[wl_name]
            acc_b = _scaled_accel(acc_name, beta)
            sol, lat, _ = solve_greedy(wl, acc_b)
            used = sum(sol.partitioning)
            row = {
                "beta": beta,
                "cores_used": used,
                "cores_total": total,
                "utilization": used / total,
                "partitioning": sol.partitioning,
                "latency_us": cycles_to_us(lat["total_time"], acc_b.frequency_ghz),
            }
            study["beta_sweep"][key].append(row)
            print(f"  β={beta:4}  cores={used:2}/{total}  util={used/total:.0%}  "
                  f"part={sol.partitioning}  {row['latency_us']:.2f} us")

        wl = WORKLOADS[wl_name]
        acc_io = _scaled_accel(acc_name, 1.0, intra_enabled=False)
        sol, lat, _ = solve_greedy(wl, acc_io)
        study["inter_only"][key] = {
            "cores_used": sum(sol.partitioning),
            "cores_total": total,
            "utilization": sum(sol.partitioning) / total,
            "partitioning": sol.partitioning,
            "latency_us": cycles_to_us(lat["total_time"], acc_io.frequency_ghz),
        }
        print(f"  inter-only  cores={sum(sol.partitioning)}/{total}  "
              f"part={sol.partitioning}  {study['inter_only'][key]['latency_us']:.2f} us")

    out = RESULTS_DIR / "sensitivity_comm.json"
    with open(out, "w") as f:
        json.dump(study, f, indent=2)
    print(f"\nSaved {out}")
    _plot_core_usage(study, RESULTS_DIR / "sensitivity_core_usage.png")
    return study


def _plot_kappa_wins(sweep: dict, out_path: Path):
    multipliers = sorted(float(k) for k in sweep)
    colors = {"SA": "#3cb44b", "EA": "#4363d8", "Greedy": "#f58231"}
    fig, ax = plt.subplots(figsize=(7, 4))
    x = range(len(multipliers))
    width = 0.25
    for i, sname in enumerate(SOLVERS):
        counts = [sweep[str(m)]["wins"][sname] for m in multipliers]
        ax.bar([xi + i * width for xi in x], counts, width, label=sname, color=colors[sname])
    ax.set_xticks([xi + width for xi in x])
    ax.set_xticklabels([str(m) for m in multipliers])
    ax.set_xlabel("κ global multiplier")
    ax.set_ylabel("Configs with best latency")
    ax.set_title("Solver wins vs κ (19 configs)")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved {out_path}")


def _plot_core_usage(study: dict, out_path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    key = "Large-MLP/8x8-mesh"
    kappa_rows = study["kappa_sweep"][key]
    beta_rows = study["beta_sweep"][key]
    total = kappa_rows[0]["cores_total"]

    ax = axes[0]
    ax.plot([r["kappa"] for r in kappa_rows],
            [r["cores_used"] for r in kappa_rows], "o-", color="#f58231", linewidth=2)
    ax.axhline(total, color="gray", linestyle="--", label=f"K={total}")
    ax.set_xlabel("κ multiplier")
    ax.set_ylabel("Cores used (Greedy+KL)")
    ax.set_title("Large-MLP/8×8: cores vs κ")
    ax.set_ylim(0, total + 4)
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot([r["beta"] for r in beta_rows],
            [r["cores_used"] for r in beta_rows], "s-", color="#4363d8", linewidth=2)
    ax.axhline(total, color="gray", linestyle="--", label=f"K={total}")
    ax.set_xlabel("β (comm cost coefficient)")
    ax.set_ylabel("Cores used (Greedy+KL)")
    ax.set_title("Large-MLP/8×8: cores vs β")
    ax.set_ylim(0, total + 4)
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved {out_path}")


def plot_report_sensitivity_mappings() -> dict:
    """Greedy+KL mapping figure for report sensitivity example (β=0.1, full mesh)."""
    from visualize import plot_mapping_detail

    wl_name, acc_name = REPORT_SENSITIVITY_CONFIG.split("/")
    wl = WORKLOADS[wl_name]
    acc_b = _scaled_accel(acc_name, REPORT_BETA_CHEAP, intra_enabled=True)
    sol, lat, _ = solve_greedy(wl, acc_b)
    used = sum(sol.partitioning)
    total = acc_b.total_cores
    us = cycles_to_us(lat["total_time"], acc_b.frequency_ghz)

    safe = REPORT_SENSITIVITY_CONFIG.replace("/", "_")
    out = RESULTS_DIR / f"mapping_detail_{safe}_beta{REPORT_BETA_CHEAP}.png"
    plot_mapping_detail(
        sol, wl, acc_b,
        title=(
            f"Mapping Detail: {REPORT_SENSITIVITY_CONFIG} "
            f"(Greedy+KL, β={REPORT_BETA_CHEAP}, {used}/{total} cores, {us:.2f} μs)"
        ),
        save_path=str(out),
    )

    meta = {
        "config": REPORT_SENSITIVITY_CONFIG,
        "beta": REPORT_BETA_CHEAP,
        "intra_comm_enabled": True,
        "solver": "Greedy+KL",
        "cores_used": used,
        "cores_total": total,
        "partitioning": sol.partitioning,
        "latency_us": us,
        "figure": out.name,
    }
    print(f"\n--- Report sensitivity mapping (β={REPORT_BETA_CHEAP}) ---")
    print(f"  cores={used}/{total}  part={sol.partitioning}  {us:.2f} us")
    print(f"  Saved {out}")
    return meta


def run_all():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    kappa = sweep_kappa_wins()
    comm = sweep_core_usage()
    report_mapping = plot_report_sensitivity_mappings()
    comm["report_mapping"] = report_mapping
    with open(RESULTS_DIR / "sensitivity_comm.json", "w") as f:
        json.dump(comm, f, indent=2)
    summary = {
        "kappa_wins": {k: v["wins"] for k, v in kappa.items()},
        "core_usage": comm,
        "report_mapping": report_mapping,
    }
    out = RESULTS_DIR / "sensitivity_study.json"
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved combined summary {out}")
    return summary


def main():
    parser = argparse.ArgumentParser(description="Communication sensitivity sweeps and report figures")
    parser.add_argument(
        "--mapping-only",
        action="store_true",
        help="Only regenerate report β=0.1 mapping figure (skip κ/β sweeps)",
    )
    args = parser.parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    if args.mapping_only:
        plot_report_sensitivity_mappings()
        return
    run_all()


if __name__ == "__main__":
    main()
