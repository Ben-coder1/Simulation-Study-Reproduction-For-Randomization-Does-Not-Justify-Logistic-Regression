"""Compare the two-thread vectorized fits against the original fits."""

import pickle
import warnings
from concurrent.futures import ThreadPoolExecutor
from itertools import product

import numpy as np
import pytest
from statsmodels.tools.sm_exceptions import ConvergenceWarning

from configuration import SIMULATIONS
from performance_analysis.progress import comparison_progress
from src import simulation as original_simulation
from src.dgps import ExperimentDGP, IndividualConditionalOutcomesUniformDGP
from src.methods import LogitModelEstimator, LogitModelPlugIn
from src_optimized.optimized_methods import fast_LogitModelPlugIn
from src_optimized.simulate_parallel import _ParallelEstimator, run_from_config


def assert_fit_matches(actual, expected, shape, context):
    assert actual.shape == expected.shape == shape, context
    assert np.isfinite(actual).all() and np.isfinite(expected).all(), context
    np.testing.assert_allclose(
        actual, expected, atol=1e-4, rtol=0, equal_nan=False, err_msg=context
    )


@pytest.mark.parametrize(
    ("reference_class", "seed"),
    [(LogitModelEstimator, 12345), (LogitModelPlugIn, 123456)],
)
def test_parallel_accuracy_grid(reference_class, seed):
    rng = np.random.default_rng(seed)
    grid = product((100, 500, 1000, 5000), (1, 10, 100, 1000, 2000), range(3))
    # Check the uneven split after reproducing the existing grid's RNG stream.
    cases = (*grid, (500, 11, 0))
    estimate_name = (
        "plug-in" if reference_class is LogitModelPlugIn else "treatment coefficient"
    )
    with (
        comparison_progress(
            f"Parallel logit {estimate_name} accuracy", len(cases)
        ) as progress,
        ThreadPoolExecutor(max_workers=2) as executor,
    ):
        parallel = _ParallelEstimator(fast_LogitModelPlugIn(), executor, workers=2)
        for n_individuals, n_experiments, regeneration in cases:
            draw = "uneven split" if n_experiments == 11 else f"{regeneration + 1}/3"
            progress.set_postfix(
                individuals=n_individuals, experiments=n_experiments, draw=draw
            )
            individuals = IndividualConditionalOutcomesUniformDGP(
                n=n_individuals,
                treatment_outcome_threshold=3 / 4,
                control_outcome_threshold=1 / 2,
            ).generate_individuals(rng=rng)
            experiments = ExperimentDGP(
                n_exp=n_experiments,
                p_treatment=3 / 4,
                conditional_outcomes=individuals,
            ).generate_experiments(rng=rng)

            with warnings.catch_warnings():
                warnings.simplefilter("error", ConvergenceWarning)
                expected, expected_beta = (
                    reference_class().generate_parameter_estimates(experiments)
                )
                plug_in, actual_beta = parallel.generate_parameter_estimates(
                    experiments
                )

            actual = (
                plug_in if reference_class is LogitModelPlugIn else actual_beta[:, 1]
            )
            context = (
                f"n={n_individuals}, experiments={n_experiments}, draw={regeneration}"
            )
            assert_fit_matches(actual, expected, (n_experiments,), context)
            assert_fit_matches(actual_beta, expected_beta, (n_experiments, 3), context)
            progress.update()


def test_parallel_run_from_config_matches_original(tmp_path, monkeypatch):
    original_path = tmp_path / "original.pkl"
    parallel_path = tmp_path / "parallel.pkl"
    monkeypatch.setattr(original_simulation, "PICKLE_LOCATION", original_path)

    with comparison_progress("Parallel run from config", 3) as progress:
        progress.set_postfix_str("original run")
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            original_simulation.run_from_config()
        progress.update()
        progress.set_postfix_str("threaded run")
        run_from_config(workers=2, pickle_path=parallel_path)
        progress.update()
        progress.set_postfix_str("compare saved outputs")

        with (
            original_path.open("rb") as original_file,
            parallel_path.open("rb") as parallel_file,
        ):
            expected_results = pickle.load(original_file)
            actual_results = pickle.load(parallel_file)

        assert len(actual_results) == len(expected_results) == len(SIMULATIONS)
        for config, actual, expected in zip(
            SIMULATIONS, actual_results, expected_results
        ):
            assert (
                actual.keys()
                == expected.keys()
                == {"true_parameter", "estimator_results"}
            )
            np.testing.assert_array_equal(
                actual["true_parameter"], expected["true_parameter"]
            )
            actual_estimators = actual["estimator_results"]
            expected_estimators = expected["estimator_results"]
            assert (
                actual_estimators.keys()
                == expected_estimators.keys()
                == {"LogitModelPlugIn", "LogitModelEstimator"}
            )
            n_experiments = config["n_experiments"]
            for name, outputs in actual_estimators.items():
                reference = expected_estimators[name]
                assert outputs.keys() == reference.keys() == {"estimates", "auxiliary"}
                context = f"n={config['n_individuals']}, {name}"
                assert_fit_matches(
                    outputs["estimates"],
                    reference["estimates"],
                    (n_experiments,),
                    context,
                )
                assert_fit_matches(
                    outputs["auxiliary"],
                    reference["auxiliary"],
                    (n_experiments, 3),
                    context,
                )

            treatment = actual_estimators["LogitModelEstimator"]
            np.testing.assert_array_equal(
                treatment["estimates"], treatment["auxiliary"][:, 1]
            )
        progress.update()
