import pickle
from pathlib import Path

import numpy as np

from configuration import PICKLE_LOCATION, SEED, SIMULATIONS
from src.dgps import (
    EXPERIMENT_DGPS,
    INDIVIDUAL_DGPS,
    ExperimentDGP,
    IndividualConditionalOutcomesDGP,
)
from src_optimized.optimized_methods import (
    DifferentialLogOddsFinder,
    fast_LogitModelPlugIn
)


class Simulation:
    def __init__(
        self,
        individual_dgp: IndividualConditionalOutcomesDGP,
        experiment_dgp: ExperimentDGP,
    ):
        self.n_experiments = experiment_dgp.n_exp
        self.n_individuals = individual_dgp.n

        self.individual_dgp = individual_dgp
        self.experiment_dgp = experiment_dgp

        self.param_estimator = fast_LogitModelPlugIn()
        self.true_parameter_finder = DifferentialLogOddsFinder()

    def run_simulation(
        self,
        rng: np.random.Generator | int | None = None,
    ):
        base_rng = np.random.default_rng(rng)
        individuals_rng, experiment_rng = base_rng.spawn(2)

        individuals = self.individual_dgp.generate_individuals(
            rng=individuals_rng
        )

        self.experiment_dgp.set_conditional_outcomes(individuals)

        experiment_results = self.experiment_dgp.generate_experiments(
            rng=experiment_rng
        )

        true_parameter = (
            self.true_parameter_finder.find_true_parameters(
                individuals
            )
        )

        # Fit each experiment once and calculate its plug-in estimate.
        plug_in_estimates, beta_mat = (
            self.param_estimator.generate_parameter_estimates(
                experiment_results
            )
        )

        # Columns: intercept, treatment coefficient, covariate coefficients.
        # Extract the treatment coefficient for every experiment.
        logit_estimates = beta_mat[:, 1].copy()

        # Preserve the original result names and structure.
        estimator_results = {
            "LogitModelPlugIn": {
                "estimates": plug_in_estimates,
                "auxiliary": beta_mat,
            },
            "LogitModelEstimator": {
                "estimates": logit_estimates,
                "auxiliary": beta_mat,
            },
        }

        return {
            "true_parameter": true_parameter,
            "estimator_results": estimator_results,
        }


class MultipleSimulations:
    def __init__(self, simulations: list[Simulation]):
        self.simulations = simulations

    def run_multiple_simulations(
        self,
        rng: np.random.Generator | int | None = None,
    ):
        base_rng = np.random.default_rng(rng)
        simulation_rngs = base_rng.spawn(len(self.simulations))

        simulation_results = []

        for simulation, simulation_rng in zip(
            self.simulations,
            simulation_rngs,
        ):
            result = simulation.run_simulation(
                rng=simulation_rng
            )
            simulation_results.append(result)

        pickle_path = Path(PICKLE_LOCATION)
        pickle_path.parent.mkdir(parents=True, exist_ok=True)

        with pickle_path.open("wb") as file:
            pickle.dump(
                simulation_results,
                file,
                protocol=pickle.HIGHEST_PROTOCOL,
            )

        return simulation_results


def run_from_config():
    simulations = [
        Simulation(
            individual_dgp=INDIVIDUAL_DGPS[
                sim_dict["individual_dgp"]
            ](
                sim_dict["n_individuals"],
                sim_dict["treatment_threshold"],
                sim_dict["control_threshold"],
            ),
            experiment_dgp=EXPERIMENT_DGPS[
                sim_dict["experiment_dgp"]
            ](
                sim_dict["n_experiments"],
                sim_dict["p_treatment"],
            ),
        )
        for sim_dict in SIMULATIONS
    ]

    simulation_controller = MultipleSimulations(simulations)

    return simulation_controller.run_multiple_simulations(
        rng=SEED
    )


if __name__ == "__main__":
    simulations_results = run_from_config()