import numpy as np
import statsmodels.api as sm
import warnings
from src.dgps import ExperimentResults
from src.dgps import IndividualLevelConditionalOutcomes
from scipy.special import expit



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



class LogitModelEstimator(ModelParameterEstimator):
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
        E = len(experiment_results)
        if E == 0:
            return np.array([]), np.array([])

        # Covariates are shared across experiments.
        C = np.asarray(
            experiment_results[0].covariates, dtype=np.float64
        )
        if C.ndim == 1:
            C = C.reshape(-1, 1)
        if C.ndim != 2:
            raise ValueError("Covariates must be a 1D or 2D array.")
        if not np.isfinite(C).all():
            raise ValueError("Covariates must be finite.")

        n, k = C.shape
        if n == 0:
            raise ValueError("Experiments must contain observations.")

        A = np.empty((E, n), dtype=np.float64)
        Y = np.empty((E, n), dtype=np.float64)

        for idx, result in enumerate(experiment_results):
            C_i = np.asarray(result.covariates, dtype=np.float64)
            if C_i.ndim == 1:
                C_i = C_i.reshape(-1, 1)

            if not np.array_equal(C_i, C):
                raise ValueError(
                    f"Experiment {idx}: covariates must match the "
                    "first experiment, including row ordering."
                )

            a = np.asarray(
                result.assignments, dtype=np.float64
            ).ravel()
            y = np.asarray(
                result.outcomes, dtype=np.float64
            ).ravel()

            if a.size != n or y.size != n:
                raise ValueError(
                    f"Experiment {idx}: covariates, assignments, and "
                    "outcomes must have the same observation count."
                )

            # Assignments are assumed to be binary, as guaranteed upstream.
            if not np.isin(y, (0, 1)).all():
                raise ValueError(
                    f"Experiment {idx}: outcomes must contain only 0 and 1."
                )

            for group in (0, 1):
                group_y = y[a == group]
                if not (
                    np.any(group_y == 0) and np.any(group_y == 1)
                ):
                    raise ValueError(
                        f"Experiment {idx}: assignment group {group} "
                        "must contain at least one outcome 0 and "
                        "at least one outcome 1."
                    )

            A[idx] = a
            Y[idx] = y

        # No padding: every experiment has the same sample size.
        p = k + 2
        X = np.empty((E, n, p), dtype=np.float64)
        X[:, :, 0] = 1.0
        X[:, :, 1] = A
        X[:, :, 2:] = C[None, :, :]

        beta = np.zeros((E, p), dtype=np.float64)
        active = np.arange(E)

        for _ in range(self.max_iter):
            if active.size == 0:
                break

            Xa = X[active]
            ya = Y[active]
            ba = beta[active]

            eta = (Xa @ ba[..., None])[..., 0]
            mu = expit(eta)

            XT = Xa.transpose(0, 2, 1)
            grad = (XT @ (ya - mu)[..., None])[..., 0]
            weights = mu * (1.0 - mu)
            H = (XT * weights[:, None, :]) @ Xa

            finite = (
                np.isfinite(H).all(axis=(1, 2))
                & np.isfinite(grad).all(axis=1)
            )
            if not finite.all():
                raise LogitEstimationError(
                    "Nonfinite gradient or Hessian in experiments "
                    f"{active[~finite].tolist()}."
                )

            try:
                delta = np.linalg.solve(H, grad[..., None])[..., 0]
            except np.linalg.LinAlgError as exc:
                # Identify the failing experiments for a useful error.
                failed = []
                for j, idx in enumerate(active):
                    try:
                        np.linalg.solve(H[j], grad[j])
                    except np.linalg.LinAlgError:
                        failed.append(int(idx))

                raise LogitEstimationError(
                    f"Singular Hessian in experiments {failed}. "
                    "Possible collinearity or separation."
                ) from exc

            updated = ba + delta
            finite = (
                np.isfinite(delta).all(axis=1)
                & np.isfinite(updated).all(axis=1)
            )
            if not finite.all():
                raise LogitEstimationError(
                    "Nonfinite Newton update in experiments "
                    f"{active[~finite].tolist()}."
                )

            beta[active] = updated

            converged = np.max(np.abs(delta), axis=1) <= self.tol
            active = active[~converged]

        if active.size:
            raise LogitEstimationError(
                f"Experiments {active.tolist()} did not converge "
                f"within {self.max_iter} iterations. Possible separation "
                "or numerical instability."
            )

        return beta[:, 1].copy(), beta

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