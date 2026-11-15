import csv
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np

# Adjust these imports if your simulation modules have different names.
import src.simulation as original_module
import src_optimized.simulation as optimized_module

from src.simulation import run_from_config as run_original_simulations
from src_optimized.simulation import (
    run_from_config as run_optimized_simulations,
)


# Assumes this script is one directory below the project root.
# If it is directly in the project root, use .parent instead.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROFILES_DIR = PROJECT_ROOT / "profiles"

TIMINGS_PATH = PROFILES_DIR / "performance_comparisons_timings.csv"
SUMMARY_PATH = PROFILES_DIR / "performance_comparisons_summary.txt"
PLOT_PATH = PROFILES_DIR / "performance_comparisons.png"

CSV_FIELDS = [
    "n_individuals",
    "n_experiments",
    "repeat",
    "seed",
    "method",
    "runtime_seconds",
    "status",
    "error",
]


def make_simulation_config(
    n_individuals: int,
    n_experiments: int,
) -> dict:
    """Generate simulation settings without editing configuration.py."""
    return {
        "individual_dgp": "individual_uniform",
        "experiment_dgp": "base",
        "n_individuals": n_individuals,
        "n_experiments": n_experiments,
        "treatment_threshold": 3 / 4,
        "control_threshold": 1 / 2,
        "p_treatment": 3 / 4,
        "param_estimators": [
            "logit",
            "logit_plugin",
        ],
        "true_parameter_finder": "differential_log_odds",
    }


def measure_one_run(
    module,
    run_function,
    simulation_config: dict,
    seed: int,
    pickle_path: Path,
) -> float:
    """
    Run one simulation with temporarily overridden module settings.

    Timing includes data generation, estimation, and pickle serialization.
    Configuration files and normal pickle outputs are not modified.
    """
    with (
        patch.object(module, "SIMULATIONS", [simulation_config]),
        patch.object(module, "SEED", seed),
        patch.object(module, "PICKLE_LOCATION", pickle_path),
    ):
        start = perf_counter()
        results = run_function()
        elapsed = perf_counter() - start

    del results
    return elapsed


def collect_runtime_data(
    csv_path: str | Path = TIMINGS_PATH,
    individual_counts: tuple[int, ...] = (100, 500, 1000, 5000),
    experiment_counts: tuple[int, ...] = (10, 50, 100, 250, 500, 1000),
    repeats: int = 1,
    base_seed: int = 12345,
) -> Path:
    """
    Run performance comparisons and write measurements directly to CSV.

    Both implementations receive the same seed for each paired run.
    Failed runs are recorded and excluded from runtime plots.
    """
    if repeats < 1:
        raise ValueError("repeats must be positive.")
    if not individual_counts or not experiment_counts:
        raise ValueError("Population and experiment counts must be nonempty.")
    if any(n <= 0 for n in (*individual_counts, *experiment_counts)):
        raise ValueError("Population and experiment counts must be positive.")

    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    methods = [
        ("original", original_module, run_original_simulations),
        ("optimized", optimized_module, run_optimized_simulations),
    ]

    pair_number = 0

    with (
        TemporaryDirectory(prefix="performance_comparisons_") as temp_dir,
        csv_path.open("w", newline="", encoding="utf-8") as file,
    ):
        writer = csv.DictWriter(file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        file.flush()

        for n_individuals in individual_counts:
            for n_experiments in experiment_counts:
                config = make_simulation_config(
                    n_individuals=n_individuals,
                    n_experiments=n_experiments,
                )

                for repeat in range(repeats):
                    seed = base_seed + repeat

                    # Alternate order across pairs, even when repeats=1.
                    run_order = (
                        methods if pair_number % 2 == 0 else methods[::-1]
                    )
                    pair_number += 1

                    for name, module, run_function in run_order:
                        print(
                            f"\nPerformance comparisons — {name}: "
                            f"{n_individuals:,} individuals, "
                            f"{n_experiments:,} experiments, "
                            f"repeat {repeat + 1}/{repeats}"
                        )

                        row = {
                            "n_individuals": n_individuals,
                            "n_experiments": n_experiments,
                            "repeat": repeat + 1,
                            "seed": seed,
                            "method": name,
                            "runtime_seconds": "",
                            "status": "ok",
                            "error": "",
                        }

                        try:
                            elapsed = measure_one_run(
                                module=module,
                                run_function=run_function,
                                simulation_config=config,
                                seed=seed,
                                pickle_path=(
                                    Path(temp_dir) / f"{name}.pkl"
                                ),
                            )

                            row["runtime_seconds"] = elapsed
                            print(f"Runtime: {elapsed:.4f} seconds")

                        except Exception as exc:
                            row["status"] = "failed"
                            row["error"] = (
                                f"{type(exc).__name__}: {exc}"
                            )
                            print(f"FAILED: {row['error']}")

                        writer.writerow(row)
                        file.flush()

    print(f"\nTimings saved to: {csv_path.resolve()}")
    return csv_path


def graph_runtime_data(
    csv_path: str | Path = TIMINGS_PATH,
    plot_path: str | Path = PLOT_PATH,
    summary_path: str | Path = SUMMARY_PATH,
    show: bool = False,
) -> None:
    """
    Read the saved CSV, write a summary, and generate runtime graphs.

    Repeated successful measurements are summarized by their median.
    """
    plot_path = Path(plot_path)
    summary_path = Path(summary_path)

    plot_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    measurements = {}
    individual_counts = set()
    configurations = set()
    failures = []

    with Path(csv_path).open(
        "r", newline="", encoding="utf-8"
    ) as file:
        for row in csv.DictReader(file):
            n_ind = int(row["n_individuals"])
            n_exp = int(row["n_experiments"])
            method = row["method"]

            individual_counts.add(n_ind)
            configurations.add((n_ind, n_exp))

            if row["status"] != "ok":
                failures.append(
                    f"{method}, individuals={n_ind}, "
                    f"experiments={n_exp}, repeat={row['repeat']}: "
                    f"{row['error']}"
                )
                continue

            key = (n_ind, n_exp, method)
            measurements.setdefault(key, []).append(
                float(row["runtime_seconds"])
            )

    if not individual_counts:
        raise ValueError("The timings CSV contains no measurements.")

    medians = {
        key: float(np.median(values))
        for key, values in measurements.items()
    }

    # Write the summary before generating figures.
    lines = [
        "Performance comparisons — original vs optimized simulations",
        "",
        "Times include generation, estimation, and pickle serialization.",
        "Reported times are medians across successful repeats.",
        "Speedup = original runtime / optimized runtime.",
        "The optimized pipeline also avoids a second logistic fit.",
        "Failed runs are excluded from timing statistics.",
        "",
        f"{'Individuals':>12} {'Experiments':>12} "
        f"{'Original (s)':>14} {'Optimized (s)':>14} {'Speedup':>10}",
    ]

    for n_ind, n_exp in sorted(configurations):
        original = medians.get((n_ind, n_exp, "original"))
        optimized = medians.get((n_ind, n_exp, "optimized"))

        original_text = (
            f"{original:.4f}" if original is not None else "N/A"
        )
        optimized_text = (
            f"{optimized:.4f}" if optimized is not None else "N/A"
        )
        speedup_text = (
            f"{original / optimized:.2f}x"
            if original is not None
            and optimized is not None
            and optimized > 0
            else "N/A"
        )

        lines.append(
            f"{n_ind:>12,} {n_exp:>12,} "
            f"{original_text:>14} {optimized_text:>14} "
            f"{speedup_text:>10}"
        )

    if failures:
        lines.extend(["", "Failed runs:"])
        lines.extend(failures)

    summary_path.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    def draw_population(ax, n_ind):
        plotted = False

        for method, label in [
            ("original", "Original"),
            ("optimized", "Optimized"),
        ]:
            # Include failed configurations as gaps rather than
            # connecting a line across missing measurements.
            counts = sorted(
                n_exp
                for population, n_exp in configurations
                if population == n_ind
            )
            runtimes = [
                medians.get((n_ind, n_exp, method), np.nan)
                for n_exp in counts
            ]

            if np.isfinite(runtimes).any():
                ax.plot(
                    counts,
                    runtimes,
                    marker="o",
                    linewidth=2,
                    label=label,
                )
                plotted = True

        ax.set_title(f"{n_ind:,} individuals")
        ax.set_xlabel("Number of experiments")
        ax.set_ylabel("Median runtime (seconds)")
        ax.grid(True, alpha=0.3)

        if plotted:
            ax.legend()
        else:
            ax.text(
                0.5,
                0.5,
                "No successful runs",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )

    populations = sorted(individual_counts)

    # Combined figure.
    ncols = min(2, len(populations))
    nrows = (len(populations) + ncols - 1) // ncols

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(7 * ncols, 4.5 * nrows),
        squeeze=False,
    )

    for ax, n_ind in zip(axes.flat, populations):
        draw_population(ax, n_ind)

    for ax in list(axes.flat)[len(populations):]:
        ax.set_visible(False)

    fig.suptitle("Performance comparisons — original vs optimized")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(plot_path, dpi=200)

    if show:
        plt.show()

    plt.close(fig)

    # Separate figure for each population size.
    for n_ind in populations:
        fig, ax = plt.subplots(figsize=(8, 5))
        draw_population(ax, n_ind)
        ax.set_title(
            f"Performance comparisons — {n_ind:,} individuals"
        )
        fig.tight_layout()

        individual_plot_path = plot_path.with_name(
            f"{plot_path.stem}_{n_ind}_individuals{plot_path.suffix}"
        )
        fig.savefig(individual_plot_path, dpi=200)
        plt.close(fig)

        print(f"Individual graph: {individual_plot_path.resolve()}")

    print(f"Combined graph:   {plot_path.resolve()}")
    print(f"Summary:          {summary_path.resolve()}")


if __name__ == "__main__":
    csv_path = collect_runtime_data(
        csv_path=TIMINGS_PATH,
        individual_counts=(100, 500, 1000, 5000),
        experiment_counts=(10, 50, 100, 250, 500, 1000),
        repeats=1,  # Increase to 3 for less noisy measurements.
        base_seed=12345,
    )

    graph_runtime_data(
        csv_path=csv_path,
        plot_path=PLOT_PATH,
        summary_path=SUMMARY_PATH,
        show=False,
    )