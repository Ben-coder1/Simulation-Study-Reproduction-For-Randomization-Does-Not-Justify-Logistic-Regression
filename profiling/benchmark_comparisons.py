"""Benchmark original and optimized pipelines independently."""

import csv
import gc
from pathlib import Path
from statistics import median
from time import perf_counter

import matplotlib.pyplot as plt

# Adjust the original import if its module has a different location.
from src.simulation import run_from_config as run_original
from src_optimized.simulation import run_from_config as run_optimized


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROFILES_DIR = PROJECT_ROOT / "profiles"

TIMINGS_PATH = PROFILES_DIR / "benchmark_comparison_timings.csv"
SUMMARY_PATH = PROFILES_DIR / "benchmark_comparison_summary.txt"
PLOT_PATH = PROFILES_DIR / "benchmark_comparison.png"

N_REPEATS = 20

RUNNERS = {
    "original": run_original,
    "optimized": run_optimized,
}


def timed_run(version):
    """Time a pipeline without depending on its returned result format."""
    gc.collect()

    start = perf_counter()
    result = RUNNERS[version]()
    elapsed = perf_counter() - start

    # Exclude disposal of returned results from the timed interval.
    del result

    return elapsed


def save_results(timings):
    PROFILES_DIR.mkdir(parents=True, exist_ok=True)

    with TIMINGS_PATH.open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "repeat",
                "first_version",
                "original_seconds",
                "optimized_seconds",
            ]
        )

        for index, (original, optimized) in enumerate(
            zip(timings["original"], timings["optimized"]),
            start=1,
        ):
            writer.writerow(
                [
                    index,
                    "original" if index % 2 else "optimized",
                    f"{original:.6f}",
                    f"{optimized:.6f}",
                ]
            )

    original_median = median(timings["original"])
    optimized_median = median(timings["optimized"])
    speedup = original_median / optimized_median

    lines = [
        "End-to-end benchmark comparison",
        "Scope: each version's run_from_config() entry point.",
        "Includes data generation, estimation, and pipeline file saving.",
        "Excludes imports, warm-up, and benchmark report generation.",
        "No numerical-equivalence checks are performed.",
        f"Completed repeats per version: {len(timings['original'])}",
        "",
    ]

    for version, values in timings.items():
        lines.append(
            f"{version}: "
            f"median={median(values):.6f} s, "
            f"min={min(values):.6f} s, "
            f"max={max(values):.6f} s"
        )

    lines.append(f"\nMedian speedup: {speedup:.3f}x")

    SUMMARY_PATH.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    repeats = list(range(1, len(timings["original"]) + 1))

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(
        repeats,
        timings["original"],
        marker="o",
        linestyle="-",
        label="Original",
    )
    ax.plot(
        repeats,
        timings["optimized"],
        marker="s",
        linestyle="--",
        label="Optimized",
    )

    ax.set_xlabel("Benchmark repeat")
    ax.set_ylabel("End-to-end runtime (seconds)")
    ax.set_title("Original versus optimized pipeline runtime")
    ax.set_xticks(repeats)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()

    fig.tight_layout()
    fig.savefig(PLOT_PATH, dpi=300)
    plt.close(fig)


def main():
    timings = {"original": [], "optimized": []}

    print(
        "Warming up both versions; these times are discarded.",
        flush=True,
    )
    timed_run("original")
    timed_run("optimized")

    for repeat in range(N_REPEATS):
        order = (
            ["original", "optimized"]
            if repeat % 2 == 0
            else ["optimized", "original"]
        )

        # Only record a paired repeat after both versions succeed.
        pair = {}

        for version in order:
            pair[version] = timed_run(version)

            print(
                f"Repeat {repeat + 1}/{N_REPEATS}: "
                f"{version} = {pair[version]:.3f} seconds",
                flush=True,
            )

        for version in timings:
            timings[version].append(pair[version])

        # Preserve results after every completed pair.
        save_results(timings)

    print("\nEnd-to-end runtime summary:")

    for version, values in timings.items():
        print(
            f"{version}: "
            f"median={median(values):.3f} s, "
            f"min={min(values):.3f} s, "
            f"max={max(values):.3f} s"
        )

    print(
        "\nMedian speedup: "
        f"{median(timings['original']) / median(timings['optimized']):.2f}x"
    )

    print(f"\nTimings: {TIMINGS_PATH.resolve()}")
    print(f"Summary: {SUMMARY_PATH.resolve()}")
    print(f"Plot: {PLOT_PATH.resolve()}")


if __name__ == "__main__":
    main()