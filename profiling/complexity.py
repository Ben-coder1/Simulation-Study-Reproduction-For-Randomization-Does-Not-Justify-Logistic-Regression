import csv
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt

from configuration import RESULTS_DIR, SEED
from src.simulation import Simulation
from src.dgps import EXPERIMENT_DGPS, INDIVIDUAL_DGPS
from src.methods import PARAMETER_ESTIMATORS, TRUTH_PARAMETER_FINDERS


N_EXPERIMENTS = 1000
N_INDIVIDUALS = range(1000, 15001, 1000)

# This script is one directory below the project root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROFILES_DIR = PROJECT_ROOT / "profiles"
PROFILES_DIR.mkdir(parents=True, exist_ok=True)

CSV_PATH = PROFILES_DIR / "original_runtime_by_individuals.csv"
PLOT_PATH = PROFILES_DIR / "original_runtime_by_individuals.png"
def build_simulation(n_individuals):
    return Simulation(
        individual_dgp=INDIVIDUAL_DGPS["individual_uniform"](
            n_individuals,
            3 / 4,
            1 / 2,
        ),
        experiment_dgp=EXPERIMENT_DGPS["base"](
            N_EXPERIMENTS,
            3 / 4,
        ),
        param_estimators=[
            PARAMETER_ESTIMATORS["logit"](),
            PARAMETER_ESTIMATORS["logit_plugin"](),
        ],
        true_parameter_finder=TRUTH_PARAMETER_FINDERS[
            "differential_log_odds"
        ](),
    )


def save_results(results):
    """Save the completed timings and their plot."""
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    PLOT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with CSV_PATH.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            ["n_individuals", "n_experiments", "runtime_seconds"]
        )
        for n_individuals, elapsed in results:
            writer.writerow(
                [n_individuals, N_EXPERIMENTS, f"{elapsed:.6f}"]
            )

    counts = [row[0] for row in results]
    runtimes = [row[1] for row in results]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(counts, runtimes, marker="o", linewidth=2)
    ax.set_xlabel("Number of individuals")
    ax.set_ylabel("Runtime (seconds)")
    ax.set_title(
        f"Simulation runtime with {N_EXPERIMENTS:,} experiments"
    )
    ax.set_xticks(counts)
    ax.tick_params(axis="x", rotation=45)
    ax.grid(True, linestyle="--", alpha=0.4)

    fig.tight_layout()
    fig.savefig(PLOT_PATH, dpi=300)
    plt.close(fig)


def main():
    results = []

    print(
        f"Benchmarking with {N_EXPERIMENTS:,} experiments.",
        flush=True,
    )

    for n_individuals in N_INDIVIDUALS:
        print(
            f"Running n_individuals={n_individuals:,}...",
            flush=True,
        )

        # Object construction is excluded from the timed interval.
        simulation = build_simulation(n_individuals)

        start = perf_counter()
        simulation_result = simulation.run_simulation(rng=SEED)
        elapsed = perf_counter() - start

        results.append((n_individuals, elapsed))
        print(f"  Runtime: {elapsed:.3f} seconds", flush=True)

        del simulation_result
        del simulation

        # Update both files after every completed run.
        save_results(results)

    print(f"\nCSV saved: {CSV_PATH.resolve()}")
    print(f"Plot saved: {PLOT_PATH.resolve()}")

    return results


if __name__ == "__main__":
    complexity_results = main()