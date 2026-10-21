import pytest
from scipy import stats
import numpy as np
from src.simulation import run_from_config
from src.dgps import IndividualConditionalOutcomesUniformDGP, ExperimentDGP, IndividualLevelConditionalOutcomes

def test_individual_conditional_outcomes_dgp_same_from_same_seed():
    n = 100
    treatment_threshold = 0.75
    control_threshold = 0.5

    dgp = IndividualConditionalOutcomesUniformDGP(n, treatment_threshold, control_threshold)
    individuals = dgp.generate_individuals(rng=42)
    dgp2 = IndividualConditionalOutcomesUniformDGP(n, treatment_threshold, control_threshold)
    individuals2 = dgp2.generate_individuals(rng=42)

    assert np.array_equal(individuals.treatment_outcomes, individuals2.treatment_outcomes)
    assert np.array_equal(individuals.control_outcomes, individuals2.control_outcomes)


def test_experiment_dgp_generate_experiment_same():
    individuals_basic = IndividualConditionalOutcomesUniformDGP(1000, 0.75, 0.5).generate_individuals(rng=42)
    n_exp = 1000
    p_treatment = 0.75
    experiment_dgp = ExperimentDGP(n_exp, p_treatment, individuals_basic)
    experiment_results = experiment_dgp.generate_experiment(rng=42)

    experiment_dgp2 = ExperimentDGP(n_exp, p_treatment, individuals_basic)
    experiment_results2 = experiment_dgp2.generate_experiment(rng=42)

    assert np.array_equal(experiment_results.assignments, experiment_results2.assignments)
    assert np.array_equal(experiment_results.covariates, experiment_results2.covariates)
    assert np.array_equal(experiment_results.outcomes, experiment_results2.outcomes)


def test_experiment_dgp_generate_experiments_same():
    individuals_basic = IndividualConditionalOutcomesUniformDGP(1000, 0.75, 0.5).generate_individuals(rng=42)
    n_exp = 100
    p_treatment = 0.75
    experiment_dgp = ExperimentDGP(n_exp, p_treatment, individuals_basic)
    experiment_results_list = experiment_dgp.generate_experiments(rng=42)

    experiment_dgp2 = ExperimentDGP(n_exp, p_treatment, individuals_basic)
    experiment_results_list2 = experiment_dgp2.generate_experiments(rng=42)

    assert len(experiment_results_list) == len(experiment_results_list2)
    for exp1, exp2 in zip(experiment_results_list, experiment_results_list2):
        assert np.array_equal(exp1.assignments, exp2.assignments)
        assert np.array_equal(exp1.covariates, exp2.covariates)
        assert np.array_equal(exp1.outcomes, exp2.outcomes)

def test_run_from_config_reproducibility():
    results1 = run_from_config()
    results2 = run_from_config()

    assert len(results1) == len(results2)

    for result1, result2 in zip(results1, results2):
        assert result1.keys() == result2.keys()

        np.testing.assert_equal(
            result1["true_parameter"],
            result2["true_parameter"],
        )

        estimator_results1 = result1["estimator_results"]
        estimator_results2 = result2["estimator_results"]

        assert (
            estimator_results1.keys()
            == estimator_results2.keys()
        )

        for estimator_name in estimator_results1:
            estimator1 = estimator_results1[estimator_name]
            estimator2 = estimator_results2[estimator_name]

            assert estimator1.keys() == estimator2.keys()

            np.testing.assert_array_equal(
                estimator1["estimates"],
                estimator2["estimates"],
            )

            np.testing.assert_array_equal(
                estimator1["auxiliary"],
                estimator2["auxiliary"],
            )




   