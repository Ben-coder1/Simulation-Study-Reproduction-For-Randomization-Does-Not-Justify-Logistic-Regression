"""Run the vectorized simulation with experiment batches split across threads.

Each population uses the standard optimized simulation and result structure.
The number of fitting threads is configurable from the command line.
"""

import argparse
import pickle
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from configuration_optimized import PICKLE_LOCATION, SEED, SIMULATIONS
from src.dgps import EXPERIMENT_DGPS, INDIVIDUAL_DGPS, ExperimentResults
from src_optimized.optimized_methods import fast_LogitModelPlugIn
from src_optimized.optimized_simulation import Simulation


class _ParallelEstimator:
    """Fit independent experiment batches and join results in input order.

    The wrapped estimator is shared by the worker threads. Each call
    partitions its experiments into at most ``workers`` contiguous batches.
    """

    def __init__(
        self,
        estimator: fast_LogitModelPlugIn,
        executor: ThreadPoolExecutor,
        workers: int,
    ):
        """Store the estimator, executor, and requested worker count.

        Keyword arguments:
            estimator -- Batched logistic plug-in estimator to run.
            executor -- Thread pool used to fit experiment batches.
            workers -- Maximum number of batches for each call.
        """
        self.estimator = estimator
        self.executor = executor
        self.workers = workers

    def generate_parameter_estimates(
        self,
        experiment_results: list[ExperimentResults],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Fit batches and restore estimates to experiment order.

        A single batch is fitted directly without submitting a thread-pool
        task. Exceptions from the wrapped estimator propagate to the caller.

        Keyword arguments:
            experiment_results -- Experiments sharing ordered covariates.

        Returns:
            Plug-in estimates and coefficient rows in input order.
        """
        n_chunks = min(self.workers, len(experiment_results))
        estimate = self.estimator.generate_parameter_estimates
        if n_chunks <= 1:
            return estimate(experiment_results)

        chunk_size, remainder = divmod(len(experiment_results), n_chunks)
        chunks = []
        start = 0
        for chunk_index in range(n_chunks):
            stop = start + chunk_size + (chunk_index < remainder)
            chunks.append(experiment_results[start:stop])
            start = stop

        # map yields results in chunk order even if a later chunk finishes first.
        plug_in_estimates, coefficients = zip(*self.executor.map(estimate, chunks))
        return (
            np.concatenate(plug_in_estimates),
            np.concatenate(coefficients, axis=0),
        )


def run_from_config(
    workers: int = 2,
    *,
    simulation_configs: list[dict] | None = None,
    seed: int = SEED,
    pickle_path: str | Path = PICKLE_LOCATION,
):
    """Run configured populations and save the standard simulation results.

    A separate random stream is spawned for each population. The results
    use the same keys and ordering as the original simulation pipeline.

    Keyword arguments:
        workers -- Number of threads available for fitting each population.
        simulation_configs -- Population settings, or None for the optimized
            configuration's default simulations.
        seed -- Seed for the generator that spawns population streams.
        pickle_path -- Destination for the serialized result list.

    Returns:
        Simulation result dictionaries in configuration order.

    Raises:
        ValueError: If workers is not a positive integer.
    """
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise ValueError("workers must be a positive integer.")
    if simulation_configs is None:
        simulation_configs = SIMULATIONS

    simulation_rngs = np.random.default_rng(seed).spawn(len(simulation_configs))
    results = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for config, rng in zip(simulation_configs, simulation_rngs):
            simulation = Simulation(
                individual_dgp=INDIVIDUAL_DGPS[config["individual_dgp"]](
                    config["n_individuals"],
                    config["treatment_threshold"],
                    config["control_threshold"],
                ),
                experiment_dgp=EXPERIMENT_DGPS[config["experiment_dgp"]](
                    config["n_experiments"],
                    config["p_treatment"],
                ),
            )
            simulation.param_estimator = _ParallelEstimator(
                simulation.param_estimator, executor, workers
            )
            results.append(simulation.run_simulation(rng=rng))

    output_path = Path(pickle_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as output_file:
        pickle.dump(results, output_file, protocol=pickle.HIGHEST_PROTOCOL)
    return results


def main():
    """Parse the worker count and run the configured simulations."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workers", type=int, default=2, help="fitting threads (default: 2)"
    )
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be a positive integer")
    run_from_config(workers=args.workers)


if __name__ == "__main__":
    main()
