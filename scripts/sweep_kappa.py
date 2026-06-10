"""κ sensitivity sweep: vary global κ multiplier and count solver wins."""

from __future__ import annotations

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
from solvers.sa_solver import solve_sa
from solvers.ea_solver import solve_ea
from solvers.greedy_solver import solve_greedy

RESULTS_DIR = Path(__file__).parent.parent / "results"
KAPPA_MULTIPLIERS = [0.0, 0.25, 0.5, 1.0, 2.0]
SOLVERS = {
    "SA": lambda wl, acc, seed: solve_sa(wl, acc, seed=seed, max_iters=3000),
    "EA": lambda wl, acc, seed: solve_ea(wl, acc, seed=seed, generations=40),
    "Greedy": lambda wl, acc, seed: solve_greedy(wl, acc),
}
RUNS = 5


def _scaled_workload(wl_name: str, kappa_mult: float):
    wl = WORKLOADS[wl_name]
    scaled = copy.copy(wl)
    scaled.kappa_scale = kappa_mult
    return scaled


def run_sweep():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
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

    out_json = RESULTS_DIR / "sensitivity_kappa.json"
    with open(out_json, "w") as f:
        json.dump(sweep, f, indent=2)
    print(f"\nSaved {out_json}")

    _plot_wins(sweep, RESULTS_DIR / "sensitivity_kappa.png")
    return sweep


def _plot_wins(sweep: dict, out_path: Path):
    multipliers = [float(k) for k in sweep.keys()]
    multipliers.sort()
    solvers = list(SOLVERS.keys())
    colors = {"SA": "#3cb44b", "EA": "#4363d8", "Greedy": "#f58231"}

    fig, ax = plt.subplots(figsize=(8, 5))
    x = range(len(multipliers))
    width = 0.25
    for i, sname in enumerate(solvers):
        counts = [sweep[str(m)]["wins"][sname] for m in multipliers]
        ax.bar([xi + i * width for xi in x], counts, width, label=sname, color=colors[sname])

    ax.set_xticks([xi + width for xi in x])
    ax.set_xticklabels([str(m) for m in multipliers])
    ax.set_xlabel("κ global multiplier")
    ax.set_ylabel("Configurations with best latency")
    ax.set_title("Solver wins vs κ multiplier (19 feasible configs)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved {out_path}")


if __name__ == "__main__":
    run_sweep()
