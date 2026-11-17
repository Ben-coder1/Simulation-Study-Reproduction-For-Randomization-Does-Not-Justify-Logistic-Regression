"""Profile the baseline simulation and save readable reports."""

import cProfile
import pstats
import subprocess
import sys
from pathlib import Path

from src.simulation import run_from_config


PROFILE_DIR = Path(__file__).resolve().parents[1] / "performance_results" / "profiles" / "base"
PROFILE_LOCATION = PROFILE_DIR / "simulation.prof"
REPORT_LOCATION = PROFILE_DIR / "simulation_profile.txt"
FLAME_GRAPH_LOCATION = PROFILE_DIR / "simulation_flamegraph.svg"


def save_profile_outputs(profiler):
    """Save cProfile data, cumulative statistics, and a flame graph.

    Keyword arguments:
        profiler -- cProfile instance containing collected call statistics.
    """
    # Save cProfile data.
    profiler.dump_stats(str(PROFILE_LOCATION))

    # Save a readable report.
    with REPORT_LOCATION.open("w", encoding="utf-8") as report:
        stats = pstats.Stats(profiler, stream=report)
        stats.strip_dirs()
        stats.sort_stats(pstats.SortKey.CUMULATIVE)
        stats.print_stats()

    print(f"Profile saved: {PROFILE_LOCATION.resolve()}", flush=True)
    print(f"Report saved: {REPORT_LOCATION.resolve()}", flush=True)

    # Generate the flame graph using the same Python environment.
    print("Generating flame graph...", flush=True)

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "flameprof",
            str(PROFILE_LOCATION),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    if result.returncode != 0:
        print(
            "Flame graph generation failed.\n"
            "Make sure flameprof is installed in this environment:\n"
            f'"{sys.executable}" -m pip install flameprof\n\n'
            f"Error:\n{result.stderr}",
            file=sys.stderr,
            flush=True,
        )
        return

    FLAME_GRAPH_LOCATION.write_text(
        result.stdout,
        encoding="utf-8",
    )

    print(
        f"Flame graph saved: {FLAME_GRAPH_LOCATION.resolve()}",
        flush=True,
    )


def main():
    """Run the baseline simulation under cProfile and save results."""
    PROFILE_DIR.mkdir(parents=True, exist_ok=True)

    print("Starting simulation profiling...", flush=True)
    print(
        f"Output directory: {PROFILE_DIR.resolve()}",
        flush=True,
    )

    profiler = cProfile.Profile()

    try:
        profiler.enable()
        results = run_from_config()
    finally:
        # Save outputs even if the simulation raises an exception
        # or is interrupted with Ctrl+C.
        profiler.disable()
        save_profile_outputs(profiler)

    print("Simulation and profiling complete.", flush=True)
    return results


if __name__ == "__main__":
    simulation_results = main()
