from DGP import *
from Methods import *




class Simulation:
    def __init__(self, indivudalDGP: IndividualConditionalOutcomesDGP, ExperimentDGP: ExperimentDGP, param_estimator: ModelParameterEstimator, true_parameter_finder: TrueParameterFinder):
        self.nExperiment = ExperimentDGP.n_exp
        self.nIndivudal = indivudalDGP.n

        self.individualDGP = indivudalDGP
        
        self.ExperimentDGP = ExperimentDGP
        
        self.param_estimator = param_estimator
        self.true_parameter_finder = true_parameter_finder

    def run_simulation(
            self, 
            rng: np.random.Generator | int | None = None
            ):
        base_rng = np.random.default_rng(rng)
        indivudalsDGP_rng = base_rng.spawn(1)[0]
        individuals = self.individualDGP.generate_individuals(rng=indivudalsDGP_rng)

        self.ExperimentDGP.set_conditional_outcomes(individuals)

        experiment_rng = base_rng.spawn(1)[0]
        experiement_outcomes = self.ExperimentDGP.generate_experiments(rng=experiment_rng)

        true_parameter_estimate = self.true_parameter_finder.find_true_parameters(individuals)

        param_estimates = self.param_estimator.generate_parameter_estimates(experiement_outcomes)

        return param_estimates, true_parameter_estimate



