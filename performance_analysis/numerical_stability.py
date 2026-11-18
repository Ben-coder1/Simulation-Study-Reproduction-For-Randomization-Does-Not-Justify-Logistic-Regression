"""Check numerical stability across the optimized simulation grid."""

import csv
import warnings
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
from unittest.mock import patch

import numpy as np
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from tqdm.auto import tqdm
from tqdm.std import tqdm as tqdm_base

import src.simulation as original_module
import src_optimized.non_vectorized_simulation as non_vectorized_module
import src_optimized.optimized_simulation as optimized_module
from src_optimized import simulate_parallel

# Store diagnostic reports separately from timing measurements.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROFILES_DIR = PROJECT_ROOT / "performance_results" / "numerical_stability"

CSV_PATH = PROFILES_DIR / "convergence_comparisons_results.csv"
SUMMARY_PATH = PROFILES_DIR / "convergence_comparisons_summary.txt"

# Match the timing grid while using distinct seeds for each condition.
INDIVIDUAL_COUNTS = (100, 500, 1000, 5000)
EXPERIMENT_COUNTS = (10, 50, 100, 250, 500, 1000, 2000)
REPEATS = 5
BASE_SEED = 12345

METHODS = [
    ("original", original_module),
    ("optimized", optimized_module),
    ("non_vectorized", non_vectorized_module),
    ("parallel", simulate_parallel),
]

CSV_FIELDS = [
    "source",
    "configuration_index",
    "n_individuals",
    "n_experiments",
    "repeat",
    "seed",
    "method",
    "runtime_seconds",
    "status",
    "possible_separation",
    "error",
    "warnings",
    "nonfinite_outputs",
]


def make_simulation_config(
    n_individuals: int,
    n_experiments: int,
) -> dict:
    """Use the same DGP settings as the performance comparison."""
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


def find_nonfinite_outputs(results) -> list[dict]:
    """Identify nonfinite true parameters, estimates, and coefficients."""
    problems = []

    for simulation_index, result in enumerate(results):
        if not np.isfinite(np.asarray(result["true_parameter"])).all():
            problems.append(
                {
                    "simulation_index": simulation_index,
                    "output": "true_parameter",
                }
            )

        for estimator_name, outputs in result["estimator_results"].items():
            for output_name in ("estimates", "auxiliary"):
                values = np.asarray(outputs[output_name])
                finite = np.isfinite(values)

                if finite.all():
                    continue

                # The first axis represents experiments.
                if values.ndim > 1:
                    finite_rows = finite.all(axis=tuple(range(1, values.ndim)))
                else:
                    finite_rows = finite

                problems.append(
                    {
                        "simulation_index": simulation_index,
                        "estimator": estimator_name,
                        "output": output_name,
                        "experiment_indices": np.flatnonzero(~finite_rows).tolist(),
                    }
                )

    return problems


def classify_error(exc: Exception) -> str:
    """Classify recognized exceptions and diagnostic messages."""
    message = str(exc).lower()
    exception_name = type(exc).__name__.lower()

    if (
        "did not converge" in message
        or "failed to converge" in message
        or "convergencewarning" in exception_name
    ):
        return "nonconvergence"

    if "separation" in message or "separation" in exception_name:
        return "fitting_error"

    if isinstance(exc, ValueError):
        return "input_error"

    if isinstance(exc, (RuntimeError, np.linalg.LinAlgError)):
        return "fitting_error"

    return "unexpected_error"


def format_warnings(warning_records: list[dict]) -> str:
    """Format each distinct warning once."""
    distinct = {
        (warning["category"], warning["message"]) for warning in warning_records
    }

    return " | ".join(
        f"{category}: {message}" for category, message in sorted(distinct)
    )


def format_nonfinite_outputs(problems: list[dict]) -> str:
    parts = []

    for problem in problems:
        text = f"simulation={problem['simulation_index']}, output={problem['output']}"

        if "estimator" in problem:
            text += f", estimator={problem['estimator']}"

        if "experiment_indices" in problem:
            text += f", experiment_indices={problem['experiment_indices']}"

        parts.append(text)

    return " | ".join(parts)


def check_one_run(
    *,
    method: str,
    module,
    configs: list[dict],
    seed: int,
    pickle_path: Path,
    source: str,
    configuration_index: int | None,
    repeat: int,
) -> dict:
    """
    Run the existing pipeline with temporary module settings.

    The full configured list is passed together to preserve its original
    RNG spawning. Grid runs each receive a single configuration.

    Status is recorded at the pipeline-run level, not as a count of
    experiments that failed to converge.
    """
    single_config = configs[0] if len(configs) == 1 else None

    report = {
        "source": source,
        "configuration_index": configuration_index,
        "n_individuals": (single_config["n_individuals"] if single_config else None),
        "n_experiments": (single_config["n_experiments"] if single_config else None),
        "repeat": repeat,
        "seed": seed,
        "method": method,
        "runtime_seconds": None,
        "status": None,
        "possible_separation": False,
        "error": "",
        "warnings": [],
        "nonfinite_outputs": [],
        "configurations": configs,
    }

    original_tqdm_init = tqdm_base.__init__

    def disabled_tqdm_init(self, *args, **kwargs):
        kwargs["disable"] = True
        original_tqdm_init(self, *args, **kwargs)

    start = perf_counter()

    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")

        try:
            with patch.object(tqdm_base, "__init__", disabled_tqdm_init):
                if module is simulate_parallel:
                    results = module.run_from_config(
                        workers=2,
                        simulation_configs=configs,
                        seed=seed,
                        pickle_path=pickle_path,
                    )
                else:
                    with (
                        patch.object(module, "SIMULATIONS", configs),
                        patch.object(module, "SEED", seed),
                        patch.object(module, "PICKLE_LOCATION", pickle_path),
                    ):
                        results = module.run_from_config()

            report["nonfinite_outputs"] = find_nonfinite_outputs(results)
            del results

            report["status"] = (
                "nonfinite_output" if report["nonfinite_outputs"] else "passed"
            )

        except Exception as exc:
            # Record this failed pipeline run, then continue with
            # subsequent runs. Experiments remaining inside the failed
            # pipeline are not necessarily fitted or diagnosed.
            report["status"] = classify_error(exc)
            report["error"] = f"{type(exc).__name__}: {exc}"

        finally:
            report["runtime_seconds"] = perf_counter() - start

        report["warnings"] = [
            {
                "category": warning.category.__name__,
                "message": str(warning.message),
            }
            for warning in caught_warnings
        ]

        # Some implementations warn rather than raise on nonconvergence.
        # Recognize statsmodels convergence warnings and common messages.
        convergence_warning = any(
            issubclass(warning.category, ConvergenceWarning)
            or "did not converge" in str(warning.message).lower()
            or "failed to converge" in str(warning.message).lower()
            for warning in caught_warnings
        )

        if convergence_warning and report["status"] in (
            "passed",
            "nonfinite_output",
        ):
            report["status"] = "nonconvergence"

    diagnostic_text = " ".join(
        [
            report["error"],
            *[
                f"{warning['category']}: {warning['message']}"
                for warning in report["warnings"]
            ],
        ]
    ).lower()

    # True means separation was explicitly mentioned in a diagnostic.
    # It does not prove separation; False does not rule separation out.
    report["possible_separation"] = "separation" in diagnostic_text

    return report


def print_report(report: dict) -> None:
    context = (
        f"{report['source']} — {report['method']}, "
        f"repeat={report['repeat']}, seed={report['seed']}"
    )

    if report["n_individuals"] is not None:
        context += (
            f", individuals={report['n_individuals']:,}, "
            f"experiments={report['n_experiments']:,}"
        )
    else:
        context += f", configured_simulations={len(report['configurations'])}"

    tqdm.write(
        f"{report['status'].upper()}: {context} ({report['runtime_seconds']:.4f}s)"
    )

    if report["error"]:
        tqdm.write(f"  {report['error']}")

    if report["possible_separation"]:
        tqdm.write(
            "  Separation mentioned by estimator; "
            "not an independent separation diagnosis."
        )

    warning_text = format_warnings(report["warnings"])
    if warning_text:
        tqdm.write(f"  Warnings: {warning_text}")

    nonfinite_text = format_nonfinite_outputs(report["nonfinite_outputs"])
    if nonfinite_text:
        tqdm.write(f"  Nonfinite outputs: {nonfinite_text}")


def write_summary(reports: list[dict]) -> None:
    final_grid_seed = (
        BASE_SEED + len(INDIVIDUAL_COUNTS) * len(EXPERIMENT_COUNTS) * REPEATS - 1
    )

    lines = [
        "Convergence comparisons — simulation implementations",
        "",
        "Includes existing configured runs and an additional size grid.",
        f"Grid individual counts: {INDIVIDUAL_COUNTS}",
        f"Grid experiment counts: {EXPERIMENT_COUNTS}",
        f"Grid repeats: {REPEATS}",
        f"Grid seeds: {BASE_SEED} through {final_grid_seed}",
        "",
        "Grid sizes, DGP settings, repeats, and run-order rotation",
        "match the supplied performance comparison.",
        "Each grid configuration/repeat uses a distinct seed, shared",
        "across all implementations. The parallel runner uses two threads.",
        "Seed formula: BASE_SEED + configuration_index * REPEATS + repeat",
        "Here configuration_index and repeat are zero-based.",
        "These grid seeds differ from the original performance script",
        "after the first configuration unless its formula is updated.",
        "Matching seeds do not guarantee identical data across",
        "implementations if their RNG usage differs.",
        "Distinct seeds do not by themselves prove statistical independence.",
        "",
        "Statuses describe entire pipeline runs, not counts or percentages",
        "of experiments that failed to converge.",
        "Passed means no recognized nonconvergence, caught error, or",
        "nonfinite checked output was detected.",
        "Silent fitting failures may not be detected by this wrapper.",
        "Other warnings are reported separately and do not necessarily",
        "change a passed status.",
        "Separation flags reflect explicit diagnostic mentions, not a",
        "dedicated complete/quasi-complete separation test.",
        "A separation warning alone does not change a passed status.",
        "Experiment indices in diagnostics are zero-based and relative to",
        "the chunk for errors raised by a parallel fit.",
        "A failed batch may stop before all experiments are diagnosed.",
        "The wrapper uses each implementation's fitting defaults; it",
        "does not enforce matching tolerances or iteration limits.",
        "Configured runs preserve each module's configuration list, seed,",
        "and normal RNG spawning; an exception may stop that pipeline.",
        "Runtime includes the pipeline and diagnostic checks and is",
        "not intended to replace the performance benchmark.",
        "",
        "Summary by source and implementation:",
    ]

    for source in ("config", "performance_grid"):
        for method, _ in METHODS:
            subset = [
                report
                for report in reports
                if report["source"] == source and report["method"] == method
            ]

            counts = Counter(report["status"] for report in subset)

            separation_count = sum(report["possible_separation"] for report in subset)

            status_text = ", ".join(
                f"{status}={count}" for status, count in sorted(counts.items())
            )

            lines.append(
                f"  {source} / {method}: "
                f"runs={len(subset)}, {status_text}, "
                f"separation_mentions={separation_count}"
            )

    lines.extend(["", "Configured settings:"])

    for report in reports:
        if report["source"] != "config":
            continue

        lines.append(f"\n{report['method']}, seed={report['seed']}:")

        for idx, config in enumerate(report["configurations"]):
            lines.append(f"  simulation={idx}: {config}")

    lines.extend(["", "All run results:"])

    for report in reports:
        lines.append(
            f"\n{report['source']} / {report['method']}: "
            f"individuals={report['n_individuals']}, "
            f"experiments={report['n_experiments']}, "
            f"repeat={report['repeat']}, seed={report['seed']}, "
            f"status={report['status']}, "
            f"runtime={report['runtime_seconds']:.4f}s"
        )

        if report["error"]:
            lines.append(f"  Error: {report['error']}")

        if report["possible_separation"]:
            lines.append("  Separation mentioned by estimator; not a proven diagnosis.")

        warning_text = format_warnings(report["warnings"])
        if warning_text:
            lines.append(f"  Warnings: {warning_text}")

        nonfinite_text = format_nonfinite_outputs(report["nonfinite_outputs"])
        if nonfinite_text:
            lines.append(f"  Nonfinite outputs: {nonfinite_text}")

    SUMMARY_PATH.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def run_convergence_checks() -> list[dict]:
    PROFILES_DIR.mkdir(parents=True, exist_ok=True)

    reports = []

    total = len(METHODS) + len(INDIVIDUAL_COUNTS) * len(
        EXPERIMENT_COUNTS
    ) * REPEATS * len(METHODS)

    with (
        TemporaryDirectory(prefix="convergence_comparisons_") as temp_dir,
        CSV_PATH.open("w", newline="", encoding="utf-8") as file,
        tqdm(
            total=total,
            desc="Convergence comparisons",
            unit="run",
            dynamic_ncols=True,
        ) as progress,
    ):
        writer = csv.DictWriter(file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        file.flush()

        def record(report: dict) -> None:
            reports.append(report)

            row = {field: report[field] for field in CSV_FIELDS}

            row["warnings"] = format_warnings(report["warnings"])
            row["nonfinite_outputs"] = format_nonfinite_outputs(
                report["nonfinite_outputs"]
            )

            writer.writerow(row)
            file.flush()

            print_report(report)
            progress.update(1)

        # Run each implementation's existing configuration and seed.
        # Keep its full list together to preserve normal RNG spawning.
        # Pickle output is redirected to temporary files.
        for name, module in METHODS:
            progress.set_postfix(
                source="config",
                method=name,
            )

            record(
                check_one_run(
                    method=name,
                    module=module,
                    configs=list(module.SIMULATIONS),
                    seed=module.SEED,
                    pickle_path=(Path(temp_dir) / f"{name}_config.pkl"),
                    source="config",
                    configuration_index=None,
                    repeat=1,
                )
            )

        # Additional grid: same sizes, DGP settings, repeats, and
        # run-order rotation as the performance comparison, but with
        # distinct seeds for each configuration/repeat.
        comparison_number = 0
        configuration_index = 0

        for n_individuals in INDIVIDUAL_COUNTS:
            for n_experiments in EXPERIMENT_COUNTS:
                config = make_simulation_config(
                    n_individuals=n_individuals,
                    n_experiments=n_experiments,
                )

                for repeat in range(REPEATS):
                    # Distinct seed for each configuration/repeat.
                    # All methods in this comparison receive this seed.
                    seed = BASE_SEED + configuration_index * REPEATS + repeat

                    # Rotate which implementation runs first.
                    offset = comparison_number % len(METHODS)
                    run_order = METHODS[offset:] + METHODS[:offset]
                    comparison_number += 1

                    for name, module in run_order:
                        progress.set_postfix(
                            source="grid",
                            method=name,
                            individuals=n_individuals,
                            experiments=n_experiments,
                            repeat=repeat + 1,
                        )

                        record(
                            check_one_run(
                                method=name,
                                module=module,
                                configs=[config],
                                seed=seed,
                                pickle_path=(Path(temp_dir) / f"{name}_grid.pkl"),
                                source="performance_grid",
                                configuration_index=configuration_index,
                                repeat=repeat + 1,
                            )
                        )

                configuration_index += 1

    write_summary(reports)

    counts = Counter(report["status"] for report in reports)

    print("\nConvergence-comparison summary")
    print(f"Total pipeline runs: {len(reports)}")

    for status, count in sorted(counts.items()):
        print(f"{status}: {count}")

    print(
        "Runs mentioning separation: "
        f"{sum(report['possible_separation'] for report in reports)}"
    )

    print(f"\nCSV:     {CSV_PATH.resolve()}")
    print(f"Summary: {SUMMARY_PATH.resolve()}")

    return reports


if __name__ == "__main__":
    convergence_reports = run_convergence_checks()

    # Exit unsuccessfully if any run has a non-passed status.
    # Other warnings, including separation mentions alone, do not
    # independently trigger a nonzero exit status.
    if any(report["status"] != "passed" for report in convergence_reports):
        raise SystemExit(1)
