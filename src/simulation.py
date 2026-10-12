from DGP import *
from Methods import *
from metrics import bias, mean_and_sd
import dataframe_image as dfi
import pickle

from configuration import PICKLE_LOCATION

DEFAULT_SEED = 100

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

simulation1 = Simulation(
    individualDGP=IndividualConditionalOutcomesUniformDGP(100, 3/4,1/2),
    ExperimentDGP=ExperimentDGP(1000,.75),
    param_estimator=LogitModel(),
    true_parameter_finder=DifferentialLogOddsFinder(),
)

simulation2 = Simulation(
    individualDGP=IndividualConditionalOutcomesUniformDGP(1000, 3/4,1/2),
    ExperimentDGP=ExperimentDGP(1000,.75),
    param_estimator=LogitModel(),
    true_parameter_finder=DifferentialLogOddsFinder(),
)

simulation3 = Simulation(
    individualDGP=IndividualConditionalOutcomesUniformDGP(5000, 3/4,1/2),
    ExperimentDGP=ExperimentDGP(1000,.75),
    param_estimator=LogitModel(),
    true_parameter_finder=DifferentialLogOddsFinder(),
)


start = MultipleSimulations([simulation1, simulation2, simulation3])
hold = start.run_multiple_simulations()
