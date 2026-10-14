import numpy as np
import statsmodels.api as sm
import warnings
from src.DGP import ExperimentResults
from src.DGP import IndividualLevelConditionalOutcomes



from abc import ABC, abstractmethod


def differential_log_odds(treatment_result, control_result) -> float:
  t = np.asarray(treatment_result)
  c = np.asarray(control_result)

  # 1. Validate inputs are non-empty
  if t.size == 0 or c.size == 0:
    raise ValueError("Input arrays must not be empty.")

  # 2. Validate inputs only contain values in {0, 1}
  if not np.isin(t, [0, 1]).all():
    raise ValueError("treatment_result must only contain values in {0, 1}.")
  if not np.isin(c, [0, 1]).all():
    raise ValueError("control_result must only contain values in {0, 1}.")

  treatment_success_rate = float(np.mean(t))
  control_success_rate = float(np.mean(c))

  # 3. Handle boundary conditions (rates of 0 or 1 result in log(0) or division by zero)
  if treatment_success_rate in (0.0, 1.0) or control_success_rate in (
      0.0,
      1.0,
  ):
    warnings.warn(
        "Success rate is 0 or 1 in treatment or control group. "
        "Log-odds calculation is undefined or infinite; returning NaN.",
        UserWarning,
        stacklevel=2,
    )
    return np.nan

  # 4. Compute log-odds differential
  t_log_odds = np.log(treatment_success_rate / (1.0 - treatment_success_rate))
  c_log_odds = np.log(control_success_rate / (1.0 - control_success_rate))

  return float(t_log_odds - c_log_odds)


class TrueParameterFinder(ABC):
    @abstractmethod
    def find_true_parameters(self, individuals: IndividualLevelConditionalOutcomes):
        pass

class DifferentialLogOddsFinder(TrueParameterFinder):
    def find_true_parameters(self, individuals: IndividualLevelConditionalOutcomes):
        return differential_log_odds(individuals.treatment_outcomes, individuals.control_outcomes)


class ModelParameterEstimator(ABC):
    @abstractmethod
    #the second element of the returned tuple is auxillary for tracking
    def generate_parameter_estimates(self, experiment_results: list[ExperimentResults]) -> tuple[np.ndarray, np.ndarray]:
        pass




class LogitModelEstimator(ModelParameterEstimator):
    def __init__(self):
        pass

    def generate_parameter_estimates(
    self,
    experiment_results: list[ExperimentResults],
) -> tuple[np.ndarray, np.ndarray]:
        beta_mat = []
        assignment_coefs = []
        

        for exp_result in experiment_results:
            covariates = np.asarray(exp_result.covariates)

            if covariates.ndim == 1:
                covariates = covariates.reshape(-1, 1)

            assignments = np.asarray(exp_result.assignments).ravel()
            outcomes = np.asarray(exp_result.outcomes).ravel()

            if not (
                covariates.shape[0] == assignments.size == outcomes.size
            ):
                raise ValueError(
                    "Covariates, assignments, and outcomes must have "
                    "the same number of observations."
                )

            if not np.isin(assignments, (0, 1)).all():
                raise ValueError(
                    "Assignments must contain only 0 and 1."
                )

            if not np.isin(outcomes, (0, 1)).all():
                raise ValueError(
                    "Outcomes must contain only 0 and 1."
                )
            outcomes = np.asarray(exp_result.outcomes).ravel()

            features = np.column_stack((assignments, covariates))
            X = sm.add_constant(features, has_constant="add")

            if np.unique(outcomes).size < 2:
                warnings.warn(
                    "Outcomes must include both 0 and 1. Returning NaN.",
                    RuntimeWarning,
                    stacklevel=2,
                )
            
                beta_mat.append(np.full(X.shape[1], np.nan))
                assignment_coefs.append(np.nan)
                continue
            # The treatment effect cannot be estimated without both groups.
            if np.unique(assignments).size < 2:
                warnings.warn(
                    "Assignments must include both treated and control "
                    "observations. Returning nans.",
                    UserWarning,
                    stacklevel=2,
                )

                beta_mat.append(np.full(X.shape[1], np.nan))
                assignment_coefs.append(np.nan)

                continue

            fitted_result = sm.Logit(outcomes, X).fit(disp=False)

            beta_mat.append(fitted_result.params)
            assignment_coefs.append(fitted_result.params[1])

        return np.array(assignment_coefs), np.array(beta_mat)

class LogitModelPlugIn(ModelParameterEstimator):
    def __init__(self):
        pass
    def generate_parameter_estimates(self, experiment_results: list[ExperimentResults]) -> tuple[np.ndarray, np.ndarray]:
        beta_mat = []
        plug_in_estimates = []
        

        for exp_result in experiment_results:
            covariates = np.asarray(exp_result.covariates)

            if covariates.ndim == 1:
                covariates = covariates.reshape(-1, 1)

            assignments = np.asarray(exp_result.assignments).ravel()
            outcomes = np.asarray(exp_result.outcomes).ravel()

            if not (
                covariates.shape[0] == assignments.size == outcomes.size
            ):
                raise ValueError(
                    "Covariates, assignments, and outcomes must have "
                    "the same number of observations."
                )

            if not np.isin(assignments, (0, 1)).all():
                raise ValueError(
                    "Assignments must contain only 0 and 1."
                )

            if not np.isin(outcomes, (0, 1)).all():
                raise ValueError(
                    "Outcomes must contain only 0 and 1."
                )

            features = np.column_stack((assignments, covariates))
            X = sm.add_constant(features, has_constant="add")
            

            if np.unique(outcomes).size < 2:
                warnings.warn(
                    "Outcomes must include both 0 and 1. Returning NaN.",
                    RuntimeWarning,
                    stacklevel=2,
                )

                beta_mat.append(np.full(X.shape[1], np.nan))
                plug_in_estimates.append(np.nan)
                continue

            # The treatment effect cannot be estimated without both groups.
            if np.unique(assignments).size < 2:
                warnings.warn(
                    "Assignments must include both treated and control "
                    "observations. Returning nans.",
                    UserWarning,
                    stacklevel=2,
                )

                beta_mat.append(np.full(X.shape[1], np.nan))
                plug_in_estimates.append(np.nan)

                continue

            fitted_result = sm.Logit(outcomes, X).fit(disp=False)
            plug_in_estimate = self.plug_in_differential_log_odds(fitted_result, covariates)

            beta_mat.append(fitted_result.params)
            plug_in_estimates.append(plug_in_estimate)

        return np.array(plug_in_estimates), np.array(beta_mat)

    def plug_in_differential_log_odds(
        self,
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

        treatment_success_rate = np.mean(treatment_result)
        control_success_rate = np.mean(control_result)

        if treatment_success_rate in (0.0, 1.0) or control_success_rate in (
              0.0,
              1.0,
          ):
            warnings.warn(
                "Success rate is 0 or 1 in treatment or control group in experiment. "
                "Log-odds calculation is undefined or infinite; returning NaN.",
                UserWarning,
                stacklevel=2,
            )
            return np.nan

        differential_log_odd = np.log(treatment_success_rate / (1.0 - treatment_success_rate)) - np.log(control_success_rate / (1.0 - control_success_rate))
        return differential_log_odd


PARAMETER_ESTIMATORS = {
    "logit": LogitModelEstimator,
    "logit_plugin": LogitModelPlugIn,
}
TRUTH_PARAMETER_FINDERS = {
    "differential_log_odds": DifferentialLogOddsFinder
}