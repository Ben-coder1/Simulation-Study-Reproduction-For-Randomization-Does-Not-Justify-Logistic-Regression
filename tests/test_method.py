import pytest
from src.DGP import ExperimentResults
from src.Methods import LogitModelEstimator, DifferentialLogOddsFinder, differential_log_odds, LogitModelPlugIn
import statsmodels.api as sm
import numpy as np

def test_differential_log_odds():
    treatment_results = [1,1,1,0]
    control_results = [1,0,0,0]

    diff_log_odds = differential_log_odds(treatment_results, control_results)
    assert diff_log_odds == pytest.approx(2.1972245773362196, rel=1e-9)

def test_differential_log_odds_zero_control():
    treatment_results = [1,1,1,0]
    control_results = [0,0,0,0]

    with pytest.warns(UserWarning):
        diff_log_odds = differential_log_odds(treatment_results, control_results)
    
    assert np.isnan(diff_log_odds)
    
def test_differential_log_odds_1_control():
    treatment_results = [1,1,1,0]
    control_results = [1,1,1,1]

    with pytest.warns(UserWarning):
        diff_log_odds = differential_log_odds(treatment_results, control_results)
    
    assert np.isnan(diff_log_odds)
    


class ThresholdPredictor:
    """A simple substitute for a fitted statsmodels result."""

    def __init__(
        self,
        treated_threshold,
        control_threshold,
        low_probability=0.25,
        high_probability=0.75,
    ):
        self.treated_threshold = treated_threshold
        self.control_threshold = control_threshold
        self.low_probability = low_probability
        self.high_probability = high_probability

    def predict(self, features):
        treatment = features[:, 1]
        covariate = features[:, 2]

        # Use a different threshold depending on treatment status.
        threshold = np.where(
            treatment == 1,
            self.treated_threshold,
            self.control_threshold,
        )

        return np.where(
            covariate > threshold,
            self.high_probability,
            self.low_probability,
        )


def test_plug_in_with_equal_thresholds_returns_zero():
    predictor = ThresholdPredictor(
        treated_threshold=1,
        control_threshold=1,
    )

    covariates = np.array([0, 1, 2, 3])

    result = LogitModelPlugIn().plug_in_differential_log_odds(
        fitted_result=predictor,
        covariates=covariates,
    )

    assert result == pytest.approx(0.0)


def test_plug_in_with_different_thresholds():
    predictor = ThresholdPredictor(
        treated_threshold=1,
        control_threshold=2,
    )

    covariates = np.array([0, 1, 2, 3])

    result = LogitModelPlugIn().plug_in_differential_log_odds(
        fitted_result=predictor,
        covariates=covariates,
    )

    expected = np.log(5 / 3)

    assert result == pytest.approx(expected)

def test_plug_in_with_all_zeros_warns_and_returns_nan():
    predictor = ThresholdPredictor(
        treated_threshold=1,
        control_threshold=2,
        low_probability=0.0,
        high_probability=0.0,
    )

    covariates = np.array([0, 1, 2, 3])

    with pytest.warns(Warning):
        result = LogitModelPlugIn().plug_in_differential_log_odds(
            fitted_result=predictor,
            covariates=covariates,
        )

    assert np.isnan(result)

def test_generate_parameter_estimates():
    # Four observations in each treatment/covariate group:
    #
    # treatment=0, covariate=0: outcome rate 1/4
    # treatment=0, covariate=1: outcome rate 1/4
    # treatment=1, covariate=0: outcome rate 3/4
    # treatment=1, covariate=1: outcome rate 3/4

    experiment_result = ExperimentResults(
        assignments=np.array(
            [
                0, 0, 0, 0,  # Control, covariate 0
                0, 0, 0, 0,  # Control, covariate 1
                1, 1, 1, 1,  # Treated, covariate 0
                1, 1, 1, 1,  # Treated, covariate 1
            ]
        ),
        covariates=np.array(
            [
                [0], [0], [0], [0],
                [1], [1], [1], [1],
                [0], [0], [0], [0],
                [1], [1], [1], [1],
            ]
        ),
        outcomes=np.array(
            [
                0, 0, 0, 1,  # Control rate = 1/4
                0, 0, 0, 1,  # Control rate = 1/4
                0, 1, 1, 1,  # Treated rate = 3/4
                0, 1, 1, 1,  # Treated rate = 3/4
            ]
        ),
    )

    plug_in_estimate, beta_mat = (
        LogitModelPlugIn().generate_parameter_estimates(
            experiment_results=[experiment_result]
        )
    )

    assert len(beta_mat) == 1
    assert len(plug_in_estimate) == 1

    # Parameter order:
    # [intercept, treatment, covariate]
    expected_beta = np.array(
        [
            -np.log(3),
            2 * np.log(3),
            0.0,
        ]
    )

    np.testing.assert_allclose(
        beta_mat[0],
        expected_beta,
        atol=1e-6,
    )

    # Treated probability = 0.75
    # Control probability = 0.25
    #
    # Differential log odds:
    # log(0.75 / 0.25) - log(0.25 / 0.75)
    # = log(3) - log(1/3)
    # = 2*log(3)
    expected_plug_in = 2 * np.log(3)

    assert plug_in_estimate[0] == pytest.approx(
        expected_plug_in,
        abs=1e-6,
    )


def test_generate_parameter_estimates_when_covariate_matters():
    experiment_result = ExperimentResults(
        assignments=np.array(
            [
                0, 0, 0, 0,  # Control, covariate 0
                1, 1, 1, 1,  # Treated, covariate 0
                0, 0, 0, 0,  # Control, covariate 1
                1, 1, 1, 1,  # Treated, covariate 1
            ]
        ),
        covariates=np.array(
            [
                [0], [0], [0], [0],
                [0], [0], [0], [0],
                [1], [1], [1], [1],
                [1], [1], [1], [1],
            ]
        ),
        outcomes=np.array(
            [
                0, 0, 0, 1,  # Control, covariate 0: rate 1/4
                0, 0, 1, 1,  # Treated, covariate 0: rate 1/2
                0, 0, 1, 1,  # Control, covariate 1: rate 1/2
                0, 1, 1, 1,  # Treated, covariate 1: rate 3/4
            ]
        ),
    )

    plug_in_estimate, beta_mat = (
        LogitModelPlugIn().generate_parameter_estimates(
            experiment_results=[experiment_result]
        )
    )

    assert len(beta_mat) == 1
    assert len(plug_in_estimate) == 1

    # Coefficient order:
    # [intercept, assignment, covariate]
    expected_beta = np.array(
        [
            -np.log(3),  # Intercept
            np.log(3),   # Treatment effect
            np.log(3),   # Covariate effect
        ]
    )

    np.testing.assert_allclose(
        beta_mat[0],
        expected_beta,
        atol=1e-6,
        rtol=0,
    )

    # There are equal numbers of covariate=0 and covariate=1.
    #
    # If everyone is treated:
    #   covariate=0 -> probability 0.50
    #   covariate=1 -> probability 0.75
    #   average probability = 0.625
    #
    # If everyone is control:
    #   covariate=0 -> probability 0.25
    #   covariate=1 -> probability 0.50
    #   average probability = 0.375
    treated_probability = 0.625
    control_probability = 0.375

    treated_log_odds = np.log(
        treated_probability / (1 - treated_probability)
    )
    control_log_odds = np.log(
        control_probability / (1 - control_probability)
    )

    expected_plug_in = treated_log_odds - control_log_odds

    assert plug_in_estimate[0] == pytest.approx(
        expected_plug_in,
        abs=1e-6,
    )



def experiment_without_covariate_effect():
    """
    Control outcome probability: 0.25
    Treated outcome probability: 0.75

    The probability is the same for covariate=0 and covariate=1.
    """

    return ExperimentResults(
        assignments=np.array(
            [
                0, 0, 0, 0,  # Control, covariate=0
                0, 0, 0, 0,  # Control, covariate=1
                1, 1, 1, 1,  # Treated, covariate=0
                1, 1, 1, 1,  # Treated, covariate=1
            ]
        ),
        covariates=np.array(
            [
                [0], [0], [0], [0],
                [1], [1], [1], [1],
                [0], [0], [0], [0],
                [1], [1], [1], [1],
            ]
        ),
        outcomes=np.array(
            [
                0, 0, 0, 1,  # Control, covariate=0: 1/4
                0, 0, 0, 1,  # Control, covariate=1: 1/4
                0, 1, 1, 1,  # Treated, covariate=0: 3/4
                0, 1, 1, 1,  # Treated, covariate=1: 3/4
            ]
        ),
    )


def experiment_with_covariate_effect():
    """
    Outcome probabilities:

                  covariate=0    covariate=1
    Control           0.25            0.50
    Treated           0.50            0.75
    """

    return ExperimentResults(
        assignments=np.array(
            [
                0, 0, 0, 0,  # Control, covariate=0
                1, 1, 1, 1,  # Treated, covariate=0
                0, 0, 0, 0,  # Control, covariate=1
                1, 1, 1, 1,  # Treated, covariate=1
            ]
        ),
        covariates=np.array(
            [
                [0], [0], [0], [0],
                [0], [0], [0], [0],
                [1], [1], [1], [1],
                [1], [1], [1], [1],
            ]
        ),
        outcomes=np.array(
            [
                0, 0, 0, 1,  # Control, covariate=0: 1/4
                0, 0, 1, 1,  # Treated, covariate=0: 1/2
                0, 0, 1, 1,  # Control, covariate=1: 1/2
                0, 1, 1, 1,  # Treated, covariate=1: 3/4
            ]
        ),
    )


def test_generate_parameter_estimates_all_treated_returns_nans():
    experiment = ExperimentResults(
        assignments=np.array(
            [1, 1, 1, 1, 1, 1, 1, 1]
        ),
        covariates=np.array(
            [
                [0], [0], [0], [0],
                [1], [1], [1], [1],
            ]
        ),
        outcomes=np.array(
            [0, 0, 0, 1, 0, 1, 1, 1]
        ),
    )

    with pytest.warns(
        UserWarning,
        match="both treated and control",
    ):
        plug_in_estimate, beta_mat = (
            LogitModelPlugIn().generate_parameter_estimates(
                experiment_results=[experiment]
            )
        )

    assert len(beta_mat) == 1
    assert len(plug_in_estimate) == 1

    np.testing.assert_array_equal(
        beta_mat[0],
        np.array([np.nan, np.nan, np.nan]),
    )
    assert np.isnan(plug_in_estimate[0])

def test_generate_parameter_estimates_all_control_returns_nans():
    experiment = ExperimentResults(
        assignments=np.array(
            [0, 0, 0, 0, 0, 0, 0, 0]
        ),
        covariates=np.array(
            [
                [0], [0], [0], [0],
                [1], [1], [1], [1],
            ]
        ),
        outcomes=np.array(
            [0, 0, 0, 1, 0, 1, 1, 1]
        ),
    )

    with pytest.warns(
        UserWarning,
        match="both treated and control",
    ):
        plug_in_estimate, beta_mat = (
            LogitModelPlugIn().generate_parameter_estimates(
                experiment_results=[experiment]
            )
        )

    assert len(beta_mat) == 1
    assert len(plug_in_estimate) == 1

    np.testing.assert_array_equal(
        beta_mat[0],
        np.array([np.nan, np.nan, np.nan]),
    )
    assert np.isnan(plug_in_estimate[0])


#Testing suite for LogitModel PlugIn

def test_logit_generate_parameter_estimates():
    # Four observations in each treatment/covariate group:
    #
    # treatment=0, covariate=0: outcome rate 1/4
    # treatment=0, covariate=1: outcome rate 1/4
    # treatment=1, covariate=0: outcome rate 3/4
    # treatment=1, covariate=1: outcome rate 3/4

    experiment_result = ExperimentResults(
        assignments=np.array(
            [
                0, 0, 0, 0,  # Control, covariate 0
                0, 0, 0, 0,  # Control, covariate 1
                1, 1, 1, 1,  # Treated, covariate 0
                1, 1, 1, 1,  # Treated, covariate 1
            ]
        ),
        covariates=np.array(
            [
                [0], [0], [0], [0],
                [1], [1], [1], [1],
                [0], [0], [0], [0],
                [1], [1], [1], [1],
            ]
        ),
        outcomes=np.array(
            [
                0, 0, 0, 1,  # Control rate = 1/4
                0, 0, 0, 1,  # Control rate = 1/4
                0, 1, 1, 1,  # Treated rate = 3/4
                0, 1, 1, 1,  # Treated rate = 3/4
            ]
        ),
    )

    assignment_estimates, beta_mat = (
        LogitModelEstimator().generate_parameter_estimates(
            experiment_results=[experiment_result]
        )
    )

    assert len(beta_mat) == 1
    assert len(assignment_estimates) == 1

    # Parameter order:
    # [intercept, treatment, covariate]
    expected_beta = np.array(
        [
            -np.log(3),
            2 * np.log(3),
            0.0,
        ]
    )

    np.testing.assert_allclose(
        beta_mat[0],
        expected_beta,
        atol=1e-6,
        rtol=0,
    )

    # The primary estimate returned by LogitModelEstimator is the
    # fitted treatment coefficient, fitted_result.params[1].
    expected_assignment_coefficient = 2 * np.log(3)

    assert assignment_estimates[0] == pytest.approx(
        expected_assignment_coefficient,
        abs=1e-6,
    )

def test_logit_generate_parameter_estimates_when_covariate_matters():
    experiment_result = ExperimentResults(
        assignments=np.array(
            [
                0, 0, 0, 0,  # Control, covariate 0
                1, 1, 1, 1,  # Treated, covariate 0
                0, 0, 0, 0,  # Control, covariate 1
                1, 1, 1, 1,  # Treated, covariate 1
            ]
        ),
        covariates=np.array(
            [
                [0], [0], [0], [0],
                [0], [0], [0], [0],
                [1], [1], [1], [1],
                [1], [1], [1], [1],
            ]
        ),
        outcomes=np.array(
            [
                0, 0, 0, 1,  # Control, covariate 0: rate 1/4
                0, 0, 1, 1,  # Treated, covariate 0: rate 1/2
                0, 0, 1, 1,  # Control, covariate 1: rate 1/2
                0, 1, 1, 1,  # Treated, covariate 1: rate 3/4
            ]
        ),
    )

    assignment_estimates, beta_mat = (
        LogitModelEstimator().generate_parameter_estimates(
            experiment_results=[experiment_result]
        )
    )

    assert len(beta_mat) == 1
    assert len(assignment_estimates) == 1

    # Parameter order:
    # [intercept, assignment, covariate]
    expected_beta = np.array(
        [
            -np.log(3),
            np.log(3),
            np.log(3),
        ]
    )

    np.testing.assert_allclose(
        beta_mat[0],
        expected_beta,
        atol=1e-6,
        rtol=0,
    )

    # LogitModelEstimator returns the conditional treatment
    # coefficient rather than the standardized plug-in contrast.
    expected_assignment_coefficient = np.log(3)

    assert assignment_estimates[0] == pytest.approx(
        expected_assignment_coefficient,
        abs=1e-6,
    )

def test_logit_generate_parameter_estimates_all_treated_returns_nans():
    experiment = ExperimentResults(
        assignments=np.array(
            [1, 1, 1, 1, 1, 1, 1, 1]
        ),
        covariates=np.array(
            [
                [0], [0], [0], [0],
                [1], [1], [1], [1],
            ]
        ),
        outcomes=np.array(
            [0, 0, 0, 1, 0, 1, 1, 1]
        ),
    )

    with pytest.warns(
        UserWarning,
        match="both treated and control",
    ):
        assignment_estimates, beta_mat = (
            LogitModelEstimator().generate_parameter_estimates(
                experiment_results=[experiment]
            )
        )

    assert len(beta_mat) == 1
    assert len(assignment_estimates) == 1

    np.testing.assert_array_equal(
        beta_mat[0],
        np.array([np.nan, np.nan, np.nan]),
    )

    assert np.isnan(assignment_estimates[0])

def test_logit_generate_parameter_estimates_all_control_returns_nans():
    experiment = ExperimentResults(
        assignments=np.array(
            [0, 0, 0, 0, 0, 0, 0, 0]
        ),
        covariates=np.array(
            [
                [0], [0], [0], [0],
                [1], [1], [1], [1],
            ]
        ),
        outcomes=np.array(
            [0, 0, 0, 1, 0, 1, 1, 1]
        ),
    )

    with pytest.warns(
        UserWarning,
        match="both treated and control",
    ):
        assignment_estimates, beta_mat = (
            LogitModelEstimator().generate_parameter_estimates(
                experiment_results=[experiment]
            )
        )

    assert len(beta_mat) == 1
    assert len(assignment_estimates) == 1

    np.testing.assert_array_equal(
        beta_mat[0],
        np.array([np.nan, np.nan, np.nan]),
    )

    assert np.isnan(assignment_estimates[0])