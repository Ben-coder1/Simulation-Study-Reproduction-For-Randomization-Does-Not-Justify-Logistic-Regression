import numpy as np
import statsmodels.api as sm
import warnings
from src.dgps import ExperimentResults
from src.dgps import IndividualLevelConditionalOutcomes
from scipy.special import expit, logit



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




class LogitEstimationError(RuntimeError):
    """An experiment could not be fit reliably."""


class fast_LogitModelEstimator(ModelParameterEstimator):
    def __init__(self, max_iter: int = 35, tol: float = 1e-8):
        if max_iter < 1:
            raise ValueError("max_iter must be positive.")
        if not np.isfinite(tol) or tol <= 0:
            raise ValueError("tol must be finite and positive.")

        self.max_iter = max_iter
        self.tol = tol

    def generate_parameter_estimates(
        self,
        experiment_results: list[ExperimentResults],
    ) -> tuple[np.ndarray, np.ndarray]:
        n_experiments = len(experiment_results)
        if n_experiments == 0:
            return np.array([]), np.array([])

        # Covariates are shared across experiments.
        shared_covariates = np.asarray(
            experiment_results[0].covariates, dtype=np.float64
        )
        if shared_covariates.ndim == 1:
            shared_covariates = shared_covariates.reshape(-1, 1)
        if shared_covariates.ndim != 2:
            raise ValueError("Covariates must be a 1D or 2D array.")
        if not np.isfinite(shared_covariates).all():
            raise ValueError("Covariates must be finite.")

        n_individuals, n_covariates = shared_covariates.shape
        if n_individuals == 0:
            raise ValueError("Experiments must contain observations.")

        assignment_matrix = np.empty(
            (n_experiments, n_individuals), dtype=np.float64
        )
        outcome_matrix = np.empty(
            (n_experiments, n_individuals), dtype=np.float64
        )

        for experiment_index, experiment_result in enumerate(experiment_results):
            experiment_covariates = np.asarray(
                experiment_result.covariates, dtype=np.float64
            )
            if experiment_covariates.ndim == 1:
                experiment_covariates = experiment_covariates.reshape(-1, 1)

            if not np.array_equal(experiment_covariates, shared_covariates):
                raise ValueError(
                    f"Experiment {experiment_index}: covariates must match the "
                    "first experiment, including row ordering."
                )

            assignments = np.asarray(
                experiment_result.assignments, dtype=np.float64
            ).ravel()
            outcomes = np.asarray(
                experiment_result.outcomes, dtype=np.float64
            ).ravel()

            if assignments.size != n_individuals or outcomes.size != n_individuals:
                raise ValueError(
                    f"Experiment {experiment_index}: covariates, assignments, and "
                    "outcomes must have the same observation count."
                )

            # Assignments are assumed to be binary, as guaranteed upstream.
            if not np.isin(outcomes, (0, 1)).all():
                raise ValueError(
                    f"Experiment {experiment_index}: outcomes must contain only 0 and 1."
                )

            for group in (0, 1):
                group_outcomes = outcomes[assignments == group]
                if not (
                    np.any(group_outcomes == 0) and np.any(group_outcomes == 1)
                ):
                    raise ValueError(
                        f"Experiment {experiment_index}: assignment group {group} "
                        "must contain at least one outcome 0 and "
                        "at least one outcome 1."
                    )

            assignment_matrix[experiment_index] = assignments
            outcome_matrix[experiment_index] = outcomes

        # No padding: every experiment has the same sample size.
        n_parameters = n_covariates + 2
        design_matrices = np.empty(
            (n_experiments, n_individuals, n_parameters), dtype=np.float64
        )
        design_matrices[:, :, 0] = 1.0
        design_matrices[:, :, 1] = assignment_matrix
        design_matrices[:, :, 2:] = shared_covariates[None, :, :]

        coefficient_matrix = np.zeros(
            (n_experiments, n_parameters), dtype=np.float64
        )
        active_experiment_indices = np.arange(n_experiments)

        for _ in range(self.max_iter):
            if active_experiment_indices.size == 0:
                break

            active_design_matrices = design_matrices[active_experiment_indices]
            active_outcomes = outcome_matrix[active_experiment_indices]
            active_coefficients = coefficient_matrix[active_experiment_indices]

            linear_predictors = (
                active_design_matrices @ active_coefficients[..., None]
            )[..., 0]
            fitted_probabilities = expit(linear_predictors)

            transposed_design_matrices = active_design_matrices.transpose(0, 2, 1)
            score_vectors = (
                transposed_design_matrices
                @ (active_outcomes - fitted_probabilities)[..., None]
            )[..., 0]
            logistic_weights = fitted_probabilities * (1.0 - fitted_probabilities)
            hessian_matrices = (
                transposed_design_matrices * logistic_weights[:, None, :]
            ) @ active_design_matrices

            are_finite = (
                np.isfinite(hessian_matrices).all(axis=(1, 2))
                & np.isfinite(score_vectors).all(axis=1)
            )
            if not are_finite.all():
                raise LogitEstimationError(
                    "Nonfinite gradient or Hessian in experiments "
                    f"{active_experiment_indices[~are_finite].tolist()}."
                )

            try:
                newton_steps = np.linalg.solve(
                    hessian_matrices, score_vectors[..., None]
                )[..., 0]
            except np.linalg.LinAlgError as error:
                # Identify the failing experiments for a useful error.
                failed_experiment_indices = []
                for active_position, experiment_index in enumerate(
                    active_experiment_indices
                ):
                    try:
                        np.linalg.solve(
                            hessian_matrices[active_position],
                            score_vectors[active_position],
                        )
                    except np.linalg.LinAlgError:
                        failed_experiment_indices.append(int(experiment_index))

                raise LogitEstimationError(
                    f"Singular Hessian in experiments {failed_experiment_indices}. "
                    "Possible collinearity or separation."
                ) from error

            updated_coefficients = active_coefficients + newton_steps
            are_finite = (
                np.isfinite(newton_steps).all(axis=1)
                & np.isfinite(updated_coefficients).all(axis=1)
            )
            if not are_finite.all():
                raise LogitEstimationError(
                    "Nonfinite Newton update in experiments "
                    f"{active_experiment_indices[~are_finite].tolist()}."
                )

            coefficient_matrix[active_experiment_indices] = updated_coefficients

            has_converged = np.max(np.abs(newton_steps), axis=1) <= self.tol
            active_experiment_indices = active_experiment_indices[~has_converged]

        if active_experiment_indices.size:
            raise LogitEstimationError(
                f"Experiments {active_experiment_indices.tolist()} did not converge "
                f"within {self.max_iter} iterations. Possible separation "
                "or numerical instability."
            )

        return coefficient_matrix[:, 1].copy(), coefficient_matrix

class fast_LogitModelPlugIn(ModelParameterEstimator):
    def __init__(self, max_iter: int = 35, tol: float = 1e-8):
        if max_iter < 1:
            raise ValueError("max_iter must be positive.")
        if not np.isfinite(tol) or tol <= 0:
            raise ValueError("tol must be finite and positive.")

        self.max_iter = max_iter
        self.tol = tol

    def generate_parameter_estimates(
        self,
        experiment_results: list[ExperimentResults],
    ) -> tuple[np.ndarray, np.ndarray]:
        n_experiments = len(experiment_results)
        if n_experiments == 0:
            return np.array([]), np.array([])

        shared_covariates = np.asarray(
            experiment_results[0].covariates,
            dtype=np.float64,
        )
        if shared_covariates.ndim == 1:
            shared_covariates = shared_covariates.reshape(-1, 1)
        if shared_covariates.ndim != 2:
            raise ValueError("Covariates must be a 1D or 2D array.")
        if not np.isfinite(shared_covariates).all():
            raise ValueError("Covariates must be finite.")

        n_individuals, n_covariates = shared_covariates.shape
        n_parameters = n_covariates + 2

        assignment_matrix = np.empty(
            (n_experiments, n_individuals), dtype=np.float64
        )
        outcome_matrix = np.empty(
            (n_experiments, n_individuals), dtype=np.float64
        )
        valid_experiment_mask = np.ones(n_experiments, dtype=bool)

        # Input preparation; expensive fitting operations are batched below.
        for experiment_index, experiment_result in enumerate(experiment_results):
            experiment_covariates = np.asarray(
                experiment_result.covariates, dtype=np.float64
            )
            if experiment_covariates.ndim == 1:
                experiment_covariates = experiment_covariates.reshape(-1, 1)

            if not np.array_equal(experiment_covariates, shared_covariates):
                raise ValueError(
                    f"Experiment {experiment_index}: covariates must match the first "
                    "experiment, including row ordering."
                )

            assignments = np.asarray(
                experiment_result.assignments, dtype=np.float64
            ).ravel()
            outcomes = np.asarray(
                experiment_result.outcomes, dtype=np.float64
            ).ravel()

            if assignments.size != n_individuals or outcomes.size != n_individuals:
                raise ValueError(
                    "Covariates, assignments, and outcomes must have "
                    "the same number of observations."
                )

            if not np.isin(assignments, (0, 1)).all():
                raise ValueError("Assignments must contain only 0 and 1.")

            if not np.isin(outcomes, (0, 1)).all():
                raise ValueError("Outcomes must contain only 0 and 1.")

            assignment_matrix[experiment_index] = assignments
            outcome_matrix[experiment_index] = outcomes

            if np.unique(outcomes).size < 2:
                warnings.warn(
                    "Outcomes must include both 0 and 1. Returning NaN.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                valid_experiment_mask[experiment_index] = False

            elif np.unique(assignments).size < 2:
                warnings.warn(
                    "Assignments must include both treated and control "
                    "observations. Returning nans.",
                    UserWarning,
                    stacklevel=2,
                )
                valid_experiment_mask[experiment_index] = False

        coefficient_matrix = np.full(
            (n_experiments, n_parameters), np.nan, dtype=np.float64
        )
        plug_in_estimates = np.full(
            n_experiments, np.nan, dtype=np.float64
        )

        valid_experiment_indices = np.flatnonzero(valid_experiment_mask)
        if valid_experiment_indices.size == 0:
            return plug_in_estimates, coefficient_matrix

        # Pack only valid experiments; no padding is needed.
        n_valid_experiments = valid_experiment_indices.size
        design_matrices = np.empty(
            (n_valid_experiments, n_individuals, n_parameters),
            dtype=np.float64,
        )
        design_matrices[:, :, 0] = 1.0
        design_matrices[:, :, 1] = assignment_matrix[valid_experiment_indices]
        design_matrices[:, :, 2:] = shared_covariates[None, :, :]

        valid_outcomes = outcome_matrix[valid_experiment_indices]
        valid_coefficients = np.zeros(
            (n_valid_experiments, n_parameters), dtype=np.float64
        )
        active_valid_indices = np.arange(n_valid_experiments)

        # Batched Newton–Raphson.
        for _ in range(self.max_iter):
            if active_valid_indices.size == 0:
                break

            active_design_matrices = design_matrices[active_valid_indices]
            active_outcomes = valid_outcomes[active_valid_indices]
            active_coefficients = valid_coefficients[active_valid_indices]

            linear_predictors = (
                active_design_matrices @ active_coefficients[..., None]
            )[..., 0]
            fitted_probabilities = expit(linear_predictors)

            transposed_design_matrices = active_design_matrices.transpose(0, 2, 1)
            score_vectors = (
                transposed_design_matrices
                @ (active_outcomes - fitted_probabilities)[..., None]
            )[..., 0]
            logistic_weights = fitted_probabilities * (1.0 - fitted_probabilities)
            hessian_matrices = (
                transposed_design_matrices * logistic_weights[:, None, :]
            ) @ active_design_matrices

            are_finite = (
                np.isfinite(score_vectors).all(axis=1)
                & np.isfinite(hessian_matrices).all(axis=(1, 2))
            )
            if not are_finite.all():
                raise RuntimeError(
                    "Nonfinite gradient or Hessian in experiments "
                    f"{valid_experiment_indices[active_valid_indices[~are_finite]].tolist()}."
                )

            try:
                newton_steps = np.linalg.solve(
                    hessian_matrices, score_vectors[..., None]
                )[..., 0]
            except np.linalg.LinAlgError as error:
                failed_experiment_indices = []
                for active_position, valid_position in enumerate(
                    active_valid_indices
                ):
                    try:
                        np.linalg.solve(
                            hessian_matrices[active_position],
                            score_vectors[active_position],
                        )
                    except np.linalg.LinAlgError:
                        failed_experiment_indices.append(
                            int(valid_experiment_indices[valid_position])
                        )

                raise RuntimeError(
                    f"Singular Hessian in experiments {failed_experiment_indices}. "
                    "Possible collinearity or separation."
                ) from error

            updated_coefficients = active_coefficients + newton_steps
            are_finite = (
                np.isfinite(newton_steps).all(axis=1)
                & np.isfinite(updated_coefficients).all(axis=1)
            )
            if not are_finite.all():
                raise RuntimeError(
                    "Nonfinite Newton update in experiments "
                    f"{valid_experiment_indices[active_valid_indices[~are_finite]].tolist()}."
                )

            valid_coefficients[active_valid_indices] = updated_coefficients

            # Converged fits do not participate in subsequent iterations.
            has_converged = np.max(np.abs(newton_steps), axis=1) <= self.tol
            active_valid_indices = active_valid_indices[~has_converged]

        if active_valid_indices.size:
            raise LogitEstimationError(
                f"Experiments {valid_experiment_indices[active_valid_indices].tolist()} did not converge "
                f"within {self.max_iter} iterations. Possible separation "
                "or numerical instability."
            )

        coefficient_matrix[valid_experiment_indices] = valid_coefficients

        # Counterfactual predictions, shape (V, n).
        control_linear_predictors = (
            valid_coefficients[:, 0, None]
            + valid_coefficients[:, 2:] @ shared_covariates.T
        )
        treated_linear_predictors = (
            control_linear_predictors + valid_coefficients[:, 1, None]
        )

        treatment_success_rates = expit(treated_linear_predictors).mean(axis=1)
        control_success_rates = expit(control_linear_predictors).mean(axis=1)

        boundary_rate_mask = (
            (treatment_success_rates == 0.0)
            | (treatment_success_rates == 1.0)
            | (control_success_rates == 0.0)
            | (control_success_rates == 1.0)
        )

        for _experiment_index in valid_experiment_indices[boundary_rate_mask]:
            warnings.warn(
                "Success rate is 0 or 1 in treatment or control group "
                "in experiment. Log-odds calculation is undefined or "
                "infinite; returning NaN.",
                UserWarning,
                stacklevel=2,
            )

        interior_rate_mask = ~boundary_rate_mask
        plug_in_estimates[valid_experiment_indices[interior_rate_mask]] = (
            logit(treatment_success_rates[interior_rate_mask])
            - logit(control_success_rates[interior_rate_mask])
        )

        return plug_in_estimates, coefficient_matrix


PARAMETER_ESTIMATORS = {
    "fast_logit": fast_LogitModelEstimator,
    "fast_logit_plugin": fast_LogitModelPlugIn,
}
TRUTH_PARAMETER_FINDERS = {
    "differential_log_odds": DifferentialLogOddsFinder
}
