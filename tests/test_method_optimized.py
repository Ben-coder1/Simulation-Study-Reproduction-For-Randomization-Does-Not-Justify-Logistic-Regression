import warnings
from src.dgps import IndividualConditionalOutcomesUniformDGP, ExperimentDGP
from src_optimized.optimized_methods import fast_LogitModelEstimator, fast_LogitModelPlugIn
from src.methods import LogitModelEstimator, LogitModelPlugIn

import numpy as np
from statsmodels.tools.sm_exceptions import ConvergenceWarning

def test_logit_estimator_accuracy(
    n_individuals: int = 5_000,
    n_experiments: int = 1_000,
    p_treatment: float = 3/4,
    treatment_threshold: float = 3/4,
    control_threshold: float = 1/2,
    seed: int = 12345,
) -> None:
    """Compare batched and statsmodels coefficients within 1e-4."""

    rng = np.random.default_rng(seed)

    # Generate one population shared across all experiments.
    individuals = IndividualConditionalOutcomesUniformDGP(
        n=n_individuals,
        treatment_outcome_threshold=treatment_threshold,
        control_outcome_threshold=control_threshold,
    ).generate_individuals(rng=rng)

    # Assignments change across experiments; covariates remain identical.
    experiments = ExperimentDGP(
        n_exp=n_experiments,
        p_treatment=p_treatment,
        conditional_outcomes=individuals,
    ).generate_experiments(rng=rng)

    # Verify that the generated data meet the fast estimator's requirement:
    # each assignment group contains both outcome types.
    for idx, experiment in enumerate(experiments):
        for group in (0, 1):
            group_outcomes = experiment.outcomes[
                experiment.assignments == group
            ]
            if not (
                np.any(group_outcomes == 0)
                and np.any(group_outcomes == 1)
            ):
                raise AssertionError(
                    f"Generated experiment {idx}, group {group}: "
                    "missing outcome 0 or 1."
                )

    old_estimator = LogitModelEstimator()
    fast_estimator = fast_LogitModelEstimator(
        max_iter=35,
        tol=1e-8,
    )

    # A reference fit that did not converge is not a valid accuracy target.
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)

        old_assignment, old_beta = (
            old_estimator.generate_parameter_estimates(experiments)
        )
        fast_assignment, fast_beta = (
            fast_estimator.generate_parameter_estimates(experiments)
        )

    # One covariate -> intercept, assignment, covariate coefficient.
    expected_shapes = {
        "old assignment": (old_assignment, (n_experiments,)),
        "fast assignment": (fast_assignment, (n_experiments,)),
        "old beta": (old_beta, (n_experiments, 3)),
        "fast beta": (fast_beta, (n_experiments, 3)),
    }

    for name, (values, expected_shape) in expected_shapes.items():
        if values.shape != expected_shape:
            raise AssertionError(
                f"{name}: expected shape {expected_shape}, "
                f"got {values.shape}."
            )
        if not np.isfinite(values).all():
            raise AssertionError(f"{name}: contains NaN or infinity.")

    # Absolute tolerance only: every coefficient must differ by <= 1e-4.
    np.testing.assert_allclose(
        fast_beta,
        old_beta,
        rtol=0.0,
        atol=1e-4,
        err_msg="Fast and old coefficient matrices differ by more than 1e-4.",
    )

    np.testing.assert_allclose(
        fast_assignment,
        old_assignment,
        rtol=0.0,
        atol=1e-4,
        err_msg="Fast and old assignment coefficients differ by more than 1e-4.",
    )

    # Verify each estimator returns its assignment column correctly.
    np.testing.assert_array_equal(old_assignment, old_beta[:, 1])
    np.testing.assert_array_equal(fast_assignment, fast_beta[:, 1])

    max_errors = np.max(np.abs(fast_beta - old_beta), axis=0)

    print(
        f"PASS: {n_experiments:,} experiments, "
        f"{n_individuals:,} individuals each, "
        f"p_treatment={p_treatment}."
    )
    print("Maximum absolute coefficient differences:")
    for name, error in zip(
        ("intercept", "assignment", "covariate"),
        max_errors,
    ):
        print(f"  {name}: {error:.3e}")


def test_logit_plugin_accuracy(
    n_individuals: int = 5_000,
    n_experiments: int = 1_000,
    p_treatment: float = 3 / 4,
    treatment_threshold: float = 3 / 4,
    control_threshold: float = 1 / 2,
    seed: int = 123456,
) -> None:
    """Compare old and fast plug-in estimates and coefficients within 1e-4."""

    rng = np.random.default_rng(seed)

    # One shared population; assignments vary across experiments.
    individuals = IndividualConditionalOutcomesUniformDGP(
        n=n_individuals,
        treatment_outcome_threshold=treatment_threshold,
        control_outcome_threshold=control_threshold,
    ).generate_individuals(rng=rng)

    experiments = ExperimentDGP(
        n_exp=n_experiments,
        p_treatment=p_treatment,
        conditional_outcomes=individuals,
    ).generate_experiments(rng=rng)

    old_estimator = LogitModelPlugIn()
    fast_estimator = fast_LogitModelPlugIn(
        max_iter=35,
        tol=1e-8,
    )

    # Do not compare against a reference fit that failed to converge.
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)

        old_plugin, old_beta = (
            old_estimator.generate_parameter_estimates(experiments)
        )
        fast_plugin, fast_beta = (
            fast_estimator.generate_parameter_estimates(experiments)
        )

    # Check output shapes and reject nonfinite results.
    expected_shapes = {
        "old plug-in": (old_plugin, (n_experiments,)),
        "fast plug-in": (fast_plugin, (n_experiments,)),
        "old beta": (old_beta, (n_experiments, 3)),
        "fast beta": (fast_beta, (n_experiments, 3)),
    }

    for name, (values, expected_shape) in expected_shapes.items():
        if values.shape != expected_shape:
            raise AssertionError(
                f"{name}: expected shape {expected_shape}, "
                f"got {values.shape}."
            )

        if not np.isfinite(values).all():
            raise AssertionError(f"{name}: contains NaN or infinity.")

    # Compare every coefficient: intercept, assignment, and covariate.
    np.testing.assert_allclose(
        fast_beta,
        old_beta,
        rtol=0.0,
        atol=1e-4,
        err_msg=(
            "Fast and old plug-in coefficient matrices "
            "differ by more than 1e-4."
        ),
    )

    # Compare every marginal differential-log-odds estimate.
    np.testing.assert_allclose(
        fast_plugin,
        old_plugin,
        rtol=0.0,
        atol=1e-4,
        err_msg="Fast and old plug-in estimates differ by more than 1e-4.",
    )

    max_beta_errors = np.max(np.abs(fast_beta - old_beta), axis=0)
    max_plugin_error = np.max(np.abs(fast_plugin - old_plugin))

    print(
        f"\nPASS: plug-in comparison for {n_experiments:,} experiments, "
        f"{n_individuals:,} individuals each, "
        f"p_treatment={p_treatment}."
    )

    print("Maximum absolute coefficient differences:")
    for name, error in zip(
        ("intercept", "assignment", "covariate"),
        max_beta_errors,
    ):
        print(f"  {name}: {error:.3e}")

    print(f"Maximum absolute plug-in difference: {max_plugin_error:.3e}")