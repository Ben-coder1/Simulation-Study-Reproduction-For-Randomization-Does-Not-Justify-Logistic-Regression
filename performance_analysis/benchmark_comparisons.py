"""Benchmark all simulation pipelines independently."""

import csv
import gc
from pathlib import Path
from statistics import median
from tempfile import TemporaryDirectory
from time import perf_counter
from unittest.mock import patch

import matplotlib.pyplot as plt

from performance_analysis.progress import comparison_progress
from src import simulation as original_module
from src_optimized import non_vectorized_simulation, optimized_simulation, simulate_parallel


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROFILES_DIR = PROJECT_ROOT / "performance_results" / "benchmarks"

TIMINGS_PATH = PROFILES_DIR / "benchmark_comparison_timings.csv"
SUMMARY_PATH = PROFILES_DIR / "benchmark_comparison_summary.txt"
PLOT_PATH = PROFILES_DIR / "benchmark_comparison.png"

N_REPEATS = 10

RUNNERS = {
    "original": original_module,
    "optimized": optimized_simulation,
    "non_vectorized": non_vectorized_simulation,
    "parallel": simulate_parallel,
}

DISPLAY_NAMES = {
    "original": "Original",
    "optimized": "Optimized / vectorized",
    "non_vectorized": "Non-vectorized",
    "parallel": "Vectorized, 2 threads",
}


def timed_run(version):
    """Time a pipeline without depending on its returned result format.

    Keyword arguments:
        version -- Key of the simulation pipeline to run.

    Returns:
        Elapsed wall-clock seconds.

    Raises:
        KeyError: If version is absent from RUNNERS.
    """
    gc.collect()

    module = RUNNERS[version]
    with TemporaryDirectory(prefix="benchmark_comparison_") as directory:
        pickle_path = Path(directory) / "simulation.pkl"
        if module is simulate_parallel:
            start = perf_counter()
            result = module.run_from_config(workers=2, pickle_path=pickle_path)
            elapsed = perf_counter() - start
        else:
            with patch.object(module, "PICKLE_LOCATION", pickle_path):
                start = perf_counter()
                result = module.run_from_config()
                elapsed = perf_counter() - start

    # Exclude disposal of returned results from the timed interval.
    del result

    return elapsed


def save_results(timings):
    """Write completed timings, a summary, and a comparison plot.

    Keyword arguments:
        timings -- Completed runtime lists keyed by implementation name.
    """
    PROFILES_DIR.mkdir(parents=True, exist_ok=True)

    with TIMINGS_PATH.open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.writer(file)
        writer.writerow(["repeat", "first_version", "method", "runtime_seconds"])

        for index in range(len(timings["original"])):
            order = list(RUNNERS)
            first_version = order[index % len(order)]
            for version in order:
                writer.writerow(
                    [
                        index + 1,
                        first_version,
                        version,
                        f"{timings[version][index]:.6f}",
                    ]
                )

    lines = [
        "End-to-end benchmark comparison",
        "Scope: each version's run_from_config() entry point.",
        "Includes data generation, estimation, and pipeline file saving.",
        "Excludes imports, warm-up, and benchmark report generation.",
        "No numerical-equivalence checks are performed.",
        f"Completed repeats per version: {len(timings['original'])}",
        "",
    ]

    original_median = median(timings["original"])
    vectorized_median = median(timings["optimized"])
    for version, values in timings.items():
        lines.append(
            f"{DISPLAY_NAMES[version]}: "
            f"median={median(values):.6f} s, "
            f"min={min(values):.6f} s, "
            f"max={max(values):.6f} s, "
            f"speedup_vs_original={original_median / median(values):.3f}x"
        )
    lines.append(
        f"Two-thread speedup vs full-batch vectorized: "
        f"{vectorized_median / median(timings['parallel']):.3f}x"
    )

    SUMMARY_PATH.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    repeats = list(range(1, len(timings["original"]) + 1))

    fig, ax = plt.subplots(figsize=(8, 5))

    for version, marker, linestyle in (
        ("original", "o", "-"),
        ("optimized", "s", "--"),
        ("non_vectorized", "^", ":"),
        ("parallel", "D", "-."),
    ):
        ax.plot(
            repeats,
            timings[version],
            marker=marker,
            linestyle=linestyle,
            label=DISPLAY_NAMES[version],
        )

    ax.set_xlabel("Benchmark repeat")
    ax.set_ylabel("End-to-end runtime (seconds)")
    ax.set_title("End-to-end simulation pipeline runtime")
    ax.set_xticks(repeats)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()

    fig.tight_layout()
    fig.savefig(PLOT_PATH, dpi=300)
    plt.close(fig)


def main():
    """Warm up and time each simulation pipeline repeatedly."""
    timings = {version: [] for version in RUNNERS}

    versions = list(RUNNERS)
    with comparison_progress("Benchmark warm-up", len(versions)) as progress:
        for version in versions:
            progress.set_postfix_str(version)
            timed_run(version)
            progress.update()

    total = N_REPEATS * len(versions)
    last = ""
    with comparison_progress("Benchmark runs", total) as progress:
        for repeat in range(N_REPEATS):
            offset = repeat % len(versions)
            order = versions[offset:] + versions[:offset]

            # Record a repeat only after all implementations succeed.
            pair = {}
            for version in order:
                current = f"{version}, repeat {repeat + 1}/{N_REPEATS}"
                progress.set_postfix_str(current + last)
                pair[version] = timed_run(version)
                last = f", last={pair[version]:.3f} s"
                progress.set_postfix_str(current + last, refresh=False)
                progress.update()

            for version in timings:
                timings[version].append(pair[version])

            # Preserve results after every completed repeat.
            save_results(timings)

    print("\nEnd-to-end runtime summary:")

    for version, values in timings.items():
        print(
            f"{DISPLAY_NAMES[version]}: "
            f"median={median(values):.3f} s, "
            f"min={min(values):.3f} s, "
            f"max={max(values):.3f} s"
        )

    for version in ("optimized", "non_vectorized", "parallel"):
        print(
            f"Median speedup ({DISPLAY_NAMES[version]} vs original): "
            f"{median(timings['original']) / median(timings[version]):.2f}x"
        )
    print(
        f"Median speedup (two threads vs full-batch vectorized): "
        f"{median(timings['optimized']) / median(timings['parallel']):.2f}x"
    )

    print(f"\nTimings: {TIMINGS_PATH.resolve()}")
    print(f"Summary: {SUMMARY_PATH.resolve()}")
    print(f"Plot: {PLOT_PATH.resolve()}")


if __name__ == "__main__":
    main()
