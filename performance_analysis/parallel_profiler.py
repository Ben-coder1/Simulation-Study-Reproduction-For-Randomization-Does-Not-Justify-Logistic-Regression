"""Profile the vectorized estimator with two fitting threads."""

import cProfile
import pstats
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Lock
from time import perf_counter
from unittest.mock import patch

from src_optimized.optimized_methods import fast_LogitModelPlugIn
from src_optimized.simulate_parallel import run_from_config


PROFILE_DIR = Path(__file__).resolve().parents[1] / "performance_results" / "profiles" / "parallel"
PROFILE_PATH = PROFILE_DIR / "parallel_simulation.prof"
REPORT_PATH = PROFILE_DIR / "parallel_simulation_profile.txt"
FLAMEGRAPH_PATH = PROFILE_DIR / "parallel_simulation_flamegraph.svg"


def main():
    profiles = []
    profile_lock = Lock()
    estimate = fast_LogitModelPlugIn.generate_parameter_estimates

    def profile_fit(estimator, experiments):
        # Profile on the calling worker so its fit appears in the report.
        profiler = cProfile.Profile()
        with profiler:
            result = estimate(estimator, experiments)
        with profile_lock:
            profiles.append(profiler)
        return result

    with (
        TemporaryDirectory(prefix="parallel_profiler_") as directory,
        patch.object(fast_LogitModelPlugIn, "generate_parameter_estimates", profile_fit),
    ):
        start = perf_counter()
        run_from_config(workers=2, pickle_path=Path(directory) / "simulation.pkl")
        elapsed = perf_counter() - start

    PROFILE_DIR.mkdir(parents=True, exist_ok=True)
    with REPORT_PATH.open("w", encoding="utf-8") as report:
        stats = pstats.Stats(*profiles, stream=report)
        stats.dump_stats(str(PROFILE_PATH))
        print("Vectorized simulation with two fitting threads", file=report)
        print(f"Profiled pipeline wall time: {elapsed:.3f} s", file=report)
        print(f"Summed fit profile time: {stats.total_tt:.3f} s", file=report)
        print(f"Profiled fit chunks: {len(profiles)}", file=report)
        print("The flamegraph aggregates fit time across both workers.", file=report)
        stats.strip_dirs().sort_stats(pstats.SortKey.CUMULATIVE).print_stats()

    flamegraph = subprocess.run(
        [sys.executable, "-m", "flameprof", str(PROFILE_PATH)],
        stdout=subprocess.PIPE,
        text=True,
        check=True,
    )
    FLAMEGRAPH_PATH.write_text(flamegraph.stdout, encoding="utf-8")
    print(f"Profile: {PROFILE_PATH}\nReport: {REPORT_PATH}\nFlamegraph: {FLAMEGRAPH_PATH}")


if __name__ == "__main__":
    main()
