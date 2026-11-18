import warnings
from itertools import product

import numpy as np
from statsmodels.tools.sm_exceptions import ConvergenceWarning

from performance_analysis.progress import comparison_progress
from src.dgps import ExperimentDGP, IndividualConditionalOutcomesUniformDGP
from src.methods import LogitModelEstimator, LogitModelPlugIn
from src_optimized.optimized_methods import (
    fast_LogitModelEstimator,
    fast_LogitModelPlugIn,
)


def _run_logit_accuracy_grid(
    *,
    plugin: bool,
    individual_counts: tuple[int, ...] = (100, 500, 1000, 5000),
    experiment_counts: tuple[int, ...] = (1, 10, 100, 1000, 2000),
    n_regenerations: int = 3,
    p_treatment: float = 3 / 4,
    treatment_threshold: float = 3 / 4,
    control_threshold: float = 1 / 2,
    seed: int = 12345,
) -> None:
    """Compare old and fast estimators over multiple sizes and populations."""

    if not individual_counts or any(n < 100 for n in individual_counts):
        raise ValueError("Every individual count must be at least 100.")

    if not experiment_counts or any(n < 1 for n in experiment_counts):
        raise ValueError("Every experiment count must be positive.")

    if n_regenerations < 1:
        raise ValueError("n_regenerations must be positive.")

    # Initialize once: never reset inside any of the loops.
    rng = np.random.default_rng(seed)

    estimate_name = "plug-in" if plugin else "treatment coefficient"

    grid = product(
        individual_counts,
        experiment_counts,
        range(1, n_regenerations + 1),
    )
    total = len(individual_counts) * len(experiment_counts) * n_regenerations

    with comparison_progress(f"Logit {estimate_name} accuracy", total) as progress:
        for n_individuals, n_experiments, regeneration in grid:
            context = (
                f"{estimate_name}: "
                f"n_individuals={n_individuals}, "
                f"n_experiments={n_experiments}, "
                f"population={regeneration}/{n_regenerations}"
            )

            progress.set_postfix(
                individuals=n_individuals,
                experiments=n_experiments,
                draw=f"{regeneration}/{n_regenerations}",
            )

            individuals = IndividualConditionalOutcomesUniformDGP(
                n=n_individuals,
                treatment_outcome_threshold=treatment_threshold,
                control_outcome_threshold=control_threshold,
            ).generate_individuals(rng=rng)

            # Individuals are shared within this batch, while treatment
            # assignments vary across experiments.
            experiments = ExperimentDGP(
                n_exp=n_experiments,
                p_treatment=p_treatment,
                conditional_outcomes=individuals,
            ).generate_experiments(rng=rng)

            # Check the requirement for both estimator variants.
            for idx, experiment in enumerate(experiments):
                for group in (0, 1):
                    group_outcomes = experiment.outcomes[
                        experiment.assignments == group
                    ]

                    if not (
                        np.any(group_outcomes == 0) and np.any(group_outcomes == 1)
                    ):
                        raise AssertionError(
                            f"{context}: experiment {idx}, group {group} "
                            "is missing outcome 0 or 1."
                        )

            # Fresh estimator objects for each comparison.
            if plugin:
                old_estimator = LogitModelPlugIn()
                fast_estimator = fast_LogitModelPlugIn(
                    max_iter=35,
                    tol=1e-8,
                )
            else:
                old_estimator = LogitModelEstimator()
                fast_estimator = fast_LogitModelEstimator(
                    max_iter=35,
                    tol=1e-8,
                )

            # A nonconverged reference fit is not an accuracy target.
            with warnings.catch_warnings():
                warnings.simplefilter("error", ConvergenceWarning)

                old_estimates, old_beta = old_estimator.generate_parameter_estimates(
                    experiments
                )
                fast_estimates, fast_beta = fast_estimator.generate_parameter_estimates(
                    experiments
                )

            expected_shapes = {
                f"old {estimate_name}": (
                    old_estimates,
                    (n_experiments,),
                ),
                f"fast {estimate_name}": (
                    fast_estimates,
                    (n_experiments,),
                ),
                "old beta": (old_beta, (n_experiments, 3)),
                "fast beta": (fast_beta, (n_experiments, 3)),
            }

            for name, (values, expected_shape) in expected_shapes.items():
                if values.shape != expected_shape:
                    raise AssertionError(
                        f"{context}: {name}: expected shape "
                        f"{expected_shape}, got {values.shape}."
                    )

                if not np.isfinite(values).all():
                    raise AssertionError(f"{context}: {name} contains NaN or infinity.")

            # Absolute tolerance only, for every coefficient.
            np.testing.assert_allclose(
                fast_beta,
                old_beta,
                rtol=0.0,
                atol=1e-4,
                err_msg=(f"{context}: coefficient matrices differ by more than 1e-4."),
            )

            np.testing.assert_allclose(
                fast_estimates,
                old_estimates,
                rtol=0.0,
                atol=1e-4,
                err_msg=(
                    f"{context}: {estimate_name} estimates differ by more than 1e-4."
                ),
            )

            if not plugin:
                # Assignment estimates must match the assignment column.
                np.testing.assert_array_equal(
                    old_estimates,
                    old_beta[:, 1],
                    err_msg=f"{context}: old assignment column mismatch.",
                )
                np.testing.assert_array_equal(
                    fast_estimates,
                    fast_beta[:, 1],
                    err_msg=f"{context}: fast assignment column mismatch.",
                )

            progress.update(1)


def test_logit_estimator_accuracy() -> None:
    _run_logit_accuracy_grid(
        plugin=False,
        seed=12345,
    )


def test_logit_plugin_accuracy() -> None:
    _run_logit_accuracy_grid(
        plugin=True,
        seed=123456,
    )
