"""Compare peak Python allocations for every simulation implementation.

This is deliberately a separate measure from elapsed runtime.  ``tracemalloc``
reports allocations made through Python's allocator, so native-library memory
outside that allocator is not included.
"""

import csv
import gc
import tracemalloc
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np

import src.simulation as original_module
import src_optimized.non_vectorized_simulation as non_vectorized_module
import src_optimized.optimized_simulation as optimized_module
from performance_analysis.progress import comparison_progress
from src.simulation import run_from_config as run_original_simulations
from src_optimized import simulate_parallel
from src_optimized.non_vectorized_simulation import (
    run_from_config as run_non_vectorized_simulations,
)
from src_optimized.optimized_simulation import (
    run_from_config as run_optimized_simulations,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROFILES_DIR = PROJECT_ROOT / "performance_results" / "memory"
MEMORY_PATH = PROFILES_DIR / "memory_comparisons_grid.csv"
SUMMARY_PATH = PROFILES_DIR / "memory_comparisons_summary.txt"
PLOT_PATH = PROFILES_DIR / "memory_comparisons.png"

CSV_FIELDS = [
    "n_individuals",
    "n_experiments",
    "repeat",
    "seed",
    "method",
    "peak_python_memory_mib",
    "status",
    "error",
]

METHOD_LABELS = [
    ("original", "Original"),
    ("optimized", "Optimized / vectorized"),
    ("non_vectorized", "Non-vectorized"),
    ("parallel", "Vectorized, 2 threads"),
]


def make_simulation_config(n_individuals: int, n_experiments: int) -> dict:
    """Create a benchmark configuration without changing project settings."""
    return {
        "individual_dgp": "individual_uniform",
        "experiment_dgp": "base",
        "n_individuals": n_individuals,
        "n_experiments": n_experiments,
        "treatment_threshold": 3 / 4,
        "control_threshold": 1 / 2,
        "p_treatment": 3 / 4,
        "param_estimators": ["logit", "logit_plugin"],
        "true_parameter_finder": "differential_log_odds",
    }


def measure_peak_memory(module, run_function, config, seed, pickle_path) -> float:
    """Run one pipeline and return its peak traced allocation in MiB."""
    gc.collect()
    tracemalloc.start()
    try:
        if module is simulate_parallel:
            result = run_function(
                workers=2,
                simulation_configs=[config],
                seed=seed,
                pickle_path=pickle_path,
            )
        else:
            with (
                patch.object(module, "SIMULATIONS", [config]),
                patch.object(module, "SEED", seed),
                patch.object(module, "PICKLE_LOCATION", pickle_path),
            ):
                result = run_function()
        _, peak_bytes = tracemalloc.get_traced_memory()
        del result
        return peak_bytes / (1024**2)
    finally:
        tracemalloc.stop()


def collect_memory_data(
    csv_path: str | Path = MEMORY_PATH,
    individual_counts: tuple[int, ...] = (100, 500, 1000, 5000),
    experiment_counts: tuple[int, ...] = (10, 50, 100, 250, 500, 1000),
    repeats: int = 1,
    base_seed: int = 12345,
) -> Path:
    """Write peak-allocation measurements for the full grid and all methods."""
    if repeats < 1:
        raise ValueError("repeats must be positive.")
    if not individual_counts or not experiment_counts:
        raise ValueError("Population and experiment counts must be nonempty.")
    if any(n <= 0 for n in (*individual_counts, *experiment_counts)):
        raise ValueError("Population and experiment counts must be positive.")

    methods = [
        ("original", original_module, run_original_simulations),
        ("optimized", optimized_module, run_optimized_simulations),
        ("non_vectorized", non_vectorized_module, run_non_vectorized_simulations),
        ("parallel", simulate_parallel, simulate_parallel.run_from_config),
    ]
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    comparison_number = 0
    total = len(individual_counts) * len(experiment_counts) * repeats * len(methods)
    last = ""

    with (
        TemporaryDirectory(prefix="memory_comparisons_") as temp_dir,
        csv_path.open("w", newline="", encoding="utf-8") as file,
        comparison_progress("Memory grid", total) as progress,
    ):
        writer = csv.DictWriter(file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for n_individuals in individual_counts:
            for n_experiments in experiment_counts:
                config = make_simulation_config(n_individuals, n_experiments)
                for repeat in range(repeats):
                    seed = base_seed + repeat
                    offset = comparison_number % len(methods)
                    comparison_number += 1
                    for name, module, run_function in (
                        methods[offset:] + methods[:offset]
                    ):
                        current = (
                            f"{name}, n={n_individuals:,}, experiments={n_experiments:,}, "
                            f"repeat {repeat + 1}/{repeats}"
                        )
                        progress.set_postfix_str(current + last)
                        row = {
                            "n_individuals": n_individuals,
                            "n_experiments": n_experiments,
                            "repeat": repeat + 1,
                            "seed": seed,
                            "method": name,
                            "peak_python_memory_mib": "",
                            "status": "ok",
                            "error": "",
                        }
                        try:
                            peak = measure_peak_memory(
                                module,
                                run_function,
                                config,
                                seed,
                                Path(temp_dir) / f"{name}.pkl",
                            )
                            row["peak_python_memory_mib"] = peak
                            last = f", last={peak:.2f} MiB"
                        except Exception as exc:
                            row["status"] = "failed"
                            row["error"] = f"{type(exc).__name__}: {exc}"
                            progress.write(f"FAILED ({current}): {row['error']}")
                            last = ", last=failed"
                        writer.writerow(row)
                        file.flush()
                        progress.set_postfix_str(current + last, refresh=False)
                        progress.update()
    return csv_path


def graph_memory_data(
    csv_path: str | Path = MEMORY_PATH,
    plot_path: str | Path = PLOT_PATH,
    summary_path: str | Path = SUMMARY_PATH,
    show: bool = False,
) -> None:
    """Create a faceted, population-by-population memory comparison figure."""
    measurements, populations, configurations, failures = {}, set(), set(), []
    with Path(csv_path).open("r", newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            n_ind, n_exp, method = (
                int(row["n_individuals"]),
                int(row["n_experiments"]),
                row["method"],
            )
            populations.add(n_ind)
            configurations.add((n_ind, n_exp))
            if row["status"] != "ok":
                failures.append(
                    f"{method}, individuals={n_ind}, experiments={n_exp}: {row['error']}"
                )
                continue
            measurements.setdefault((n_ind, n_exp, method), []).append(
                float(row["peak_python_memory_mib"])
            )
    if not populations:
        raise ValueError("The memory CSV contains no measurements.")
    medians = {key: float(np.median(values)) for key, values in measurements.items()}

    lines = [
        "Memory comparisons — simulation implementations",
        "Values are median peak Python allocations (MiB) across successful repeats.",
        "tracemalloc excludes native allocations outside Python's allocator.",
        "The two-thread peak covers the entire run and both workers.",
        "",
    ]
    for n_ind, n_exp in sorted(configurations):
        values = [
            medians.get((n_ind, n_exp, method), np.nan) for method, _ in METHOD_LABELS
        ]
        lines.append(
            f"individuals={n_ind:,}, experiments={n_exp:,}: "
            + ", ".join(
                f"{label}={value:.3f} MiB"
                for (_, label), value in zip(METHOD_LABELS, values)
            )
        )
    if failures:
        lines.extend(["", "Failed runs:", *failures])
    Path(summary_path).write_text("\n".join(lines) + "\n", encoding="utf-8")

    populations = sorted(populations)
    ncols = min(2, len(populations))
    nrows = (len(populations) + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(7 * ncols, 4.5 * nrows), squeeze=False
    )
    for ax, n_ind in zip(axes.flat, populations):
        counts = sorted(
            n_exp for population, n_exp in configurations if population == n_ind
        )
        for method, label in METHOD_LABELS:
            values = [medians.get((n_ind, n_exp, method), np.nan) for n_exp in counts]
            ax.plot(counts, values, marker="o", linewidth=2, label=label)
        ax.set_title(f"{n_ind:,} individuals")
        ax.set_xlabel("Number of experiments")
        ax.set_ylabel("Median peak Python memory (MiB)")
        ax.grid(True, alpha=0.3)
    for ax in list(axes.flat)[len(populations) :]:
        ax.set_visible(False)
    # Use one shared key instead of repeating it in every population panel.
    handles, labels = axes.flat[0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            ncol=len(handles),
            bbox_to_anchor=(0.5, 0.93),
        )
    fig.suptitle("Memory comparisons — simulation implementations")
    fig.text(
        0.5,
        0.018,
        "Only two lines are visible because the two vectorized series overlap "
        "and the two non-vectorized series overlap.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.055, 1, 0.88))
    fig.savefig(plot_path, dpi=200)
    if show:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    path = collect_memory_data(
        experiment_counts=(10, 50, 100, 250, 500, 1000, 2000), repeats=5
    )
    graph_memory_data(path)
    print(
        f"Memory grid: {path.resolve()}\nGraph: {PLOT_PATH.resolve()}\nSummary: {SUMMARY_PATH.resolve()}"
    )
