import numpy as np
import statsmodels.api as sm

from DGP import ExperimentResults

class LogitModel:
    def __init__(self):
        pass
    def fit(self, experiment_results: list[ExperimentResults]):
        beta_mat = []
        plug_in_estimates = []

        for exp_result in experiment_results:
            features = np.column_stack([
                exp_result.assignments,
                exp_result.covariates 
            ])
            X = sm.add_constant(features)
            y = exp_result.outcomes
            logit_model = sm.Logit(y, X)
            model_result = logit_model.fit()
            betas = model_result.params
            beta_mat.append(betas)
            plug_in = self.plug_in_differential_log_odds(logit_model, exp_result.covariates)
            plug_in_estimates.append(plug_in)

        return beta_mat, plug_in_estimates
    def plug_in_differential_log_odds(self, logit_model, covariates):
        features_if_all_treated = np.column_stack([
            np.ones((covariates.shape[0], 1)),
            covariates
        ])
        population_level_treatment_sucess_rate =np.mean(logit_model.predict(features_if_all_treated)) 
        features_if_all_control = np.column_stack([
            np.zeros((covariates.shape[0], 1)),
            covariates
        ])
        population_level_control_sucess_rate =np.mean(logit_model.predict(features_if_all_control)) 

        differential_log_odd = np.log(population_level_treatment_sucess_rate / (1 - population_level_treatment_sucess_rate)) - np.log(population_level_control_sucess_rate / (1 - population_level_control_sucess_rate))
        return differential_log_odd

