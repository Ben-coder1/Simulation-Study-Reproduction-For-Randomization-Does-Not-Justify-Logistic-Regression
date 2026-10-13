from logging import config

from src.DGP import *
from src.Methods import *
from src.metrics import bias, mean_and_sd
import dataframe_image as dfi
import pickle

from configuration import PICKLE_LOCATION, SEED, SIMULATIONS



class Simulation:
    def __init__(self, individualDGP: IndividualConditionalOutcomesDGP, ExperimentDGP: ExperimentDGP, param_estimator: ModelParameterEstimator, true_parameter_finder: TrueParameterFinder):
        self.nExperiment = ExperimentDGP.n_exp
        self.nindividual = individualDGP.n

        self.individualDGP = individualDGP
        
        self.ExperimentDGP = ExperimentDGP
        
        self.param_estimator = param_estimator
        self.true_parameter_finder = true_parameter_finder

    def run_simulation(
            self, 
            rng: np.random.Generator | int | None = None
            ):
        base_rng = np.random.default_rng(rng)
        individualsDGP_rng = base_rng.spawn(1)[0]
        individuals = self.individualDGP.generate_individuals(rng=individualsDGP_rng)

        self.ExperimentDGP.set_conditional_outcomes(individuals)

        experiment_rng = base_rng.spawn(1)[0]
        experiement_outcomes = self.ExperimentDGP.generate_experiments(rng=experiment_rng)

        true_parameter = self.true_parameter_finder.find_true_parameters(individuals)

        model_param_estimates, plug_in_estimates = self.param_estimator.generate_parameter_estimates(experiement_outcomes)



        return model_param_estimates, plug_in_estimates, true_parameter

class MultipleSimulations:
    def __init__(self, simulations: list[Simulation]):
        self.simulations = simulations

    def run_multiple_simulations(
        self,
        rng: np.random.Generator | int | None = None
    ):
        base_rng = np.random.default_rng(rng)
        simulation_rngs = base_rng.spawn(len(self.simulations))

        simulation_raws = []

        for simulation, simulation_rng in zip(
            self.simulations,
            simulation_rngs
        ):
            model_param_estimates, plug_in_estimates, true_parameter = (
                simulation.run_simulation(rng=simulation_rng)
            )

            # Each append creates one DataFrame row.
            simulation_raws.append(
                (
                    model_param_estimates,
                    plug_in_estimates,
                    true_parameter
                )
            )
        with open(PICKLE_LOCATION, "wb") as file:
            pickle.dump(
                simulation_raws,
                file,
                protocol=pickle.HIGHEST_PROTOCOL
            )
        return simulation_raws


def run_from_config():
    sims = [
        Simulation(
            individualDGP=INDIVIDUAL_DGPS[sim_dict["individualDGP"]](sim_dict["nIndividuals"], sim_dict["treatment_threshold"], sim_dict["control_threshold"]),
            ExperimentDGP=EXPERIMENT_DGPS[sim_dict["ExperimentDGP"]](sim_dict["nExperiments"], sim_dict["pTreatment"]),
            param_estimator=PARAMETER_ESTIMATORS[sim_dict["param_estimator"]](),
            true_parameter_finder=TRUTH_PARAMETER_FINDERS[sim_dict["true_parameter_finder"]]()
        )
        for sim_dict in SIMULATIONS
    ]

    simulations_control = MultipleSimulations(sims)
    simulations_results = simulations_control.run_multiple_simulations(rng=SEED)
    return simulations_results


simulations_results = run_from_config()
