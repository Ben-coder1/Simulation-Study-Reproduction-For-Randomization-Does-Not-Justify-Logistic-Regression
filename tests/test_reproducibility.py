import pytest
from scipy import stats
import numpy as np
from src.simulation import run_from_config
from src.DGP import IndividualConditionalOutcomesUniformDGP, ExperimentDGP, IndividualLevelConditionalOutcomes

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
    for r1, r2 in zip(results1, results2):
        model_param_estimates1, plug_in_estimates1, true_parameter1 = r1
        model_param_estimates2, plug_in_estimates2, true_parameter2 = r2
        assert np.array_equal(model_param_estimates1, model_param_estimates2)
        assert np.array_equal(plug_in_estimates1, plug_in_estimates2)
        assert np.array_equal(true_parameter1, true_parameter2)




   