import pytest
from scipy import stats
import numpy as np

from src.DGP import IndividualConditionalOutcomesUniformDGP, ExperimentDGP, IndividualLevelConditionalOutcomes

def test_individual_conditional_outcomes_dgp():
    n = 100
    treatment_threshold = 0.75
    control_threshold = 0.5

    dgp = IndividualConditionalOutcomesUniformDGP(n, treatment_threshold, control_threshold)
    individuals = dgp.generate_individuals(rng=42)

    assert individuals.n == n
    assert np.isin(individuals.treatment_outcomes, [0, 1]).all()
    assert np.isin(individuals.control_outcomes, [0, 1]).all()

def test_individual_conditional_outcomes_dgp_crazy_values():
    n = 100
    treatment_threshold = 1.99
    control_threshold = -.1

    dgp = IndividualConditionalOutcomesUniformDGP(n, treatment_threshold, control_threshold)
    individuals = dgp.generate_individuals(rng=43)

    assert individuals.n == n
    assert np.isin(individuals.treatment_outcomes, [0, 1]).all()
    assert np.isin(individuals.control_outcomes, [0, 1]).all()

def test_individual_conditional_outcomes_dgp_negative():
    n = -1
    treatment_threshold = 0.75
    control_threshold = 0.5

    with pytest.raises(ValueError):
        dgp = IndividualConditionalOutcomesUniformDGP(n, treatment_threshold, control_threshold)

def test_individual_conditional_outcomes_dgp_float():
    n = .5
    treatment_threshold = .75
    control_threshold = .5

    with pytest.raises(TypeError):
        dgp = IndividualConditionalOutcomesUniformDGP(n, treatment_threshold, control_threshold)



def test_individual_conditional_outcomes_dgp_zero():
    n = 0
    treatment_threshold = 0.75
    control_threshold = 0.5

    with pytest.raises(ValueError):
        dgp = IndividualConditionalOutcomesUniformDGP(n, treatment_threshold, control_threshold)

def test_statistical_properties_individuals():
    n=10000
    treatment_threshold = 0.75
    control_threshold = 0.5

    dgp = IndividualConditionalOutcomesUniformDGP(n, treatment_threshold, control_threshold)
    individuals = dgp.generate_individuals(rng=100)

    n_treated_successes = int(np.sum(individuals.treatment_outcomes))
    n_control_successes = int(np.sum(individuals.control_outcomes))

    p_success_treatment = 1-((treatment_threshold**2)/2) if treatment_threshold <= 1 else ((2-treatment_threshold)**2)/2


    result = stats.binomtest(n_treated_successes, n, p=p_success_treatment)
    assert result.pvalue > 0.01

    result = stats.binomtest(n_control_successes, n, p=1-control_threshold)
    assert result.pvalue > 0.01






def test_experiment_dgp_basic():
    individuals_basic = IndividualConditionalOutcomesUniformDGP(1000, 0.75, 0.5).generate_individuals(rng=42)
    n_exp = 1000
    p_treatment = 0.75
    experiment_dgp = ExperimentDGP(n_exp, p_treatment, individuals_basic)
    assert experiment_dgp.n_ind == 1000
    assert experiment_dgp.conditional_outcomes is individuals_basic
    assert experiment_dgp.n_exp == n_exp
    assert experiment_dgp.p_treatment == p_treatment

def test_default_construction():
    n_exp = 1000
    p_treatment = 0.75
    experiment_dgp = ExperimentDGP(n_exp, p_treatment)
    assert experiment_dgp.n_ind is None
    assert experiment_dgp.conditional_outcomes is None
    assert experiment_dgp.n_exp == n_exp
    assert experiment_dgp.p_treatment == p_treatment

def test_bad_construction():
    n_exp = 1000
    p_treatment = 1.5
    with pytest.raises(ValueError):
        experiment_dgp = ExperimentDGP(n_exp, p_treatment)

    
def test_bad_Individuals():
    n_exp = 1000
    p_treatment = 0.75
    with pytest.raises(TypeError):
        experiment_dgp = ExperimentDGP(n_exp, p_treatment, 100)

def test_no_individuals_set():
    n_exp = 1000
    p_treatment = 0.75
    experiment_dgp = ExperimentDGP(n_exp, p_treatment)
    with pytest.raises(ValueError):
        experiment_dgp.generate_experiment(rng=42)

def test_set_conditional_outcomes():
    individuals_basic = IndividualConditionalOutcomesUniformDGP(1000, 0.75, 0.5).generate_individuals(rng=42)
    n_exp = 1000
    p_treatment = 0.75
    experiment_dgp = ExperimentDGP(n_exp, p_treatment)
    experiment_dgp.set_conditional_outcomes(individuals_basic)
    assert experiment_dgp.n_ind == 1000
    assert experiment_dgp.conditional_outcomes is individuals_basic

def test_set_conditional_outcomes_bad():
    n_exp = 1000
    p_treatment = 0.75
    experiment_dgp = ExperimentDGP(n_exp, p_treatment)
    with pytest.raises(TypeError):
        experiment_dgp.set_conditional_outcomes(100)

def test_experiment_dgp_generate_experiment():
    individuals_basic = IndividualConditionalOutcomesUniformDGP(1000, 0.75, 0.5).generate_individuals(rng=42)
    n_exp = 1000
    p_treatment = 0.75
    experiment_dgp = ExperimentDGP(n_exp, p_treatment, individuals_basic)
    experiment_results = experiment_dgp.generate_experiment(rng=42)

    assert isinstance(experiment_results.assignments, np.ndarray)
    assert isinstance(experiment_results.covariates, np.ndarray)
    assert isinstance(experiment_results.outcomes, np.ndarray)

    assert experiment_results.assignments.shape == (1000,)
    assert experiment_results.covariates.shape == (1000, 1)
    assert experiment_results.outcomes.shape == (1000,)

    assert np.isin(experiment_results.assignments, [0, 1]).all()
    assert np.isin(experiment_results.outcomes, [0, 1]).all()

def test_experiment_dgp_generate_experiment_assignments_match_outcomes_all_treatment_success():
    individuals_basic = IndividualLevelConditionalOutcomes(
        n=1000,
        covariates=np.random.default_rng(42).uniform(0, 1, size=(1000, 1)),
        treatment_outcomes = np.ones(1000, dtype=int),
        control_outcomes = np.zeros(1000, dtype=int)
    )
    experiment_dgp = ExperimentDGP(n_exp=1000, p_treatment=1.0, conditional_outcomes=individuals_basic)
    experiment_results = experiment_dgp.generate_experiment(rng=42)

    assert np.all(experiment_results.outcomes[experiment_results.assignments == 1] == 1)
    assert np.all(experiment_results.outcomes[experiment_results.assignments == 0] == 0)


def test_experiment_dgp_generate_experiment_assignments_match_outcomes_general():
    individuals_basic = IndividualLevelConditionalOutcomes(
        n=1000,
        covariates=np.random.default_rng(42).uniform(0, 1, size=(1000, 1)),
        treatment_outcomes = np.random.default_rng(42).binomial(1, 0.75, size=1000),
        control_outcomes = np.random.default_rng(42).binomial(1, 0.5, size=1000)
    )
    experiment_dgp = ExperimentDGP(n_exp=1000, p_treatment=1.0, conditional_outcomes=individuals_basic)
    experiment_results = experiment_dgp.generate_experiment(rng=42)

    assert np.all(experiment_results.outcomes[experiment_results.assignments == 1] == individuals_basic.treatment_outcomes[experiment_results.assignments == 1])
    assert np.all(experiment_results.outcomes[experiment_results.assignments == 0] == individuals_basic.control_outcomes[experiment_results.assignments == 0])
    
    



def test_experiment_dgp_generate_experiments():
    individuals_basic = IndividualConditionalOutcomesUniformDGP(1000, 0.75, 0.5).generate_individuals(rng=42)
    n_exp = 100
    p_treatment = 0.75
    experiment_dgp = ExperimentDGP(n_exp, p_treatment, individuals_basic)
    experiment_results_list = experiment_dgp.generate_experiments(rng=42)

    assert len(experiment_results_list) == n_exp

    for experiment_results in experiment_results_list:
        assert isinstance(experiment_results.assignments, np.ndarray)
        assert isinstance(experiment_results.covariates, np.ndarray)
        assert isinstance(experiment_results.outcomes, np.ndarray)

        assert experiment_results.assignments.shape == (1000,)
        assert experiment_results.covariates.shape == (1000, 1)
        assert experiment_results.outcomes.shape == (1000,)

        assert np.isin(experiment_results.assignments, [0, 1]).all()
        assert np.isin(experiment_results.outcomes, [0, 1]).all()

