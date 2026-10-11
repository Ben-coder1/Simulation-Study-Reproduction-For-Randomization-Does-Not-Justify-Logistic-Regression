import numpy as np
import statsmodels.api as sm
import pandas as pd
from DGP import ExperimentResults
from DGP import IndividualLevelConditionalOutcomes



from abc import ABC, abstractmethod


def differential_log_odds(treatment_result, control_result):
    treatment_success_rate = np.mean(treatment_result)
    control_success_rate = np.mean(control_result)

    return np.log(treatment_success_rate / (1 - treatment_success_rate)) - np.log(control_success_rate / (1 - control_success_rate))


class TrueParameterFinder(ABC):
    @abstractmethod
    def find_true_parameter(self, individuals: IndividualLevelConditionalOutcomes):
        pass

class DifferentialLogOddsFinder(TrueParameterFinder):
    def find_true_parameter(self, individuals: IndividualLevelConditionalOutcomes):
        return differential_log_odds(individuals.treatment_outcomes, individuals.control_outcomes)


class ModelParameterEstimator(ABC):
    @abstractmethod
    def generate_parameter_estimates(self, experiment_results: list[ExperimentResults]) -> pd.DataFrame:
        pass




class LogitModel(ModelParameterEstimator):
    def __init__(self):
        pass

    def generate_parameter_estimates(
        self,
        experiment_results: list[ExperimentResults] = None,
    ) -> pd.DataFrame:
        beta_mat = []
        plug_in_estimates = []

        for exp_result in experiment_results:
            covariates = np.asarray(exp_result.covariates)
            assignments = np.asarray(exp_result.assignments).reshape(-1, 1)
            outcomes = np.asarray(exp_result.outcomes).ravel()

            features = np.column_stack((assignments, covariates))
            X = sm.add_constant(features, has_constant="add")

            fitted_result = sm.Logit(outcomes, X).fit(disp=False)

            beta_mat.append(fitted_result.params)

            plug_in = LogitModel.plug_in_differential_log_odds(
                fitted_result,
                covariates,
            )
            plug_in_estimates.append(plug_in)

        combined_estimates = np.column_stack((beta_mat, plug_in_estimates))

       
        num_covariates = covariates.shape[1] if covariates.ndim > 1 else 1
        column_names = (
            ["const", "assignment"]
            + [f"cov_{i}" for i in range(num_covariates)]
            + ["plug_in_estimate"]
        )
        return pd.DataFrame(combined_estimates, columns=column_names)

    def plug_in_differential_log_odds(
        fitted_result,
        covariates: np.ndarray,
    ) -> float:
        covariates = np.asarray(covariates)
        n = covariates.shape[0]


        features_if_all_treated = np.column_stack(
            (
                np.ones(n),
                np.ones(n),
                covariates,
            )
        )

        features_if_all_control = np.column_stack(
            (
                np.ones(n),
                np.zeros(n),
                covariates,
            )
        )

        
        treatment_result =fitted_result.predict(features_if_all_treated)

        control_result = fitted_result.predict(features_if_all_control)

        differential_log_odd = differential_log_odds(treatment_result, control_result)
        return differential_log_odd


