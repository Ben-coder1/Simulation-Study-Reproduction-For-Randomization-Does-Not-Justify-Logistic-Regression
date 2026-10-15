"""
Purpose-built violin plots for the current simulation output.

For each simulation in configuration.SIMULATIONS, this script creates
one subplot containing a violin for every configured estimation method.
The horizontal dashed line is the frozen-population true parameter.

This script assumes that the order of the results in the pickle matches
the order of SIMULATIONS in the configuration file.
"""

import pickle
import warnings
from math import ceil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from configuration import PICKLE_LOCATION, SIMULATIONS, VIOLIN_IMAGE_LOCATION
from src.Methods import PARAMETER_ESTIMATORS


OUTPUT_LOCATION = "simulation_violin_plots.png"

# Adjust this if you want more or fewer subplot columns.
MAX_SUBPLOT_COLUMNS = 2

# Figure resolution when saved.
OUTPUT_DPI = 300


def load_simulation_results(
    pickle_location: str | Path = PICKLE_LOCATION,
) -> list[dict]:
    """Load simulation results from the existing pickle file."""
    pickle_path = Path(pickle_location)

    if not pickle_path.exists():
        raise FileNotFoundError(
            f"Simulation pickle not found: {pickle_path.resolve()}"
        )

    with pickle_path.open("rb") as file:
        simulation_results = pickle.load(file)

    if not isinstance(simulation_results, list):
        raise TypeError(
            "Expected the pickle file to contain a list of "
            "simulation-result dictionaries."
        )

    if not simulation_results:
        raise ValueError("The simulation-results pickle is empty.")

    return simulation_results


def get_estimator_class_name(method_name: str) -> str:
    """
    Translate a configuration registry name, such as 'logit', into the
    class-name key stored in the simulation-result dictionary.
    """
    if method_name not in PARAMETER_ESTIMATORS:
        raise KeyError(
            f"Estimator {method_name!r} is not registered in "
            "PARAMETER_ESTIMATORS."
        )

    estimator_class = PARAMETER_ESTIMATORS[method_name]
    return estimator_class.__name__


def get_method_estimates(
    simulation_result: dict,
    method_name: str,
    simulation_number: int,
) -> np.ndarray:
    """
    Extract the finite estimates for one configured method.

    The simulation output is keyed by estimator class name, while the
    configuration uses registry names. This function connects the two.
    """
    if not isinstance(simulation_result, dict):
        raise TypeError(
            f"Simulation result {simulation_number} must be a dictionary."
        )

    if "estimator_results" not in simulation_result:
        raise KeyError(
            f"Simulation result {simulation_number} does not contain "
            "'estimator_results'."
        )

    estimator_results = simulation_result["estimator_results"]

    if not isinstance(estimator_results, dict):
        raise TypeError(
            f"'estimator_results' for simulation {simulation_number} "
            "must be a dictionary."
        )

    estimator_class_name = get_estimator_class_name(method_name)

    if estimator_class_name not in estimator_results:
        raise KeyError(
            f"Simulation result {simulation_number} does not contain "
            f"results for {estimator_class_name!r}, corresponding to "
            f"configured method {method_name!r}. Available estimators: "
            f"{list(estimator_results)}"
        )

    method_result = estimator_results[estimator_class_name]

    if "estimates" not in method_result:
        raise KeyError(
            f"Result for {estimator_class_name!r} in simulation "
            f"{simulation_number} does not contain 'estimates'."
        )

    estimates = np.asarray(
        method_result["estimates"],
        dtype=float,
    ).ravel()

    if estimates.size == 0:
        raise ValueError(
            f"Estimator {method_name!r} has no estimates in "
            f"simulation {simulation_number}."
        )

    finite_mask = np.isfinite(estimates)
    number_omitted = int(np.sum(~finite_mask))

    if number_omitted:
        warnings.warn(
            f"Omitting {number_omitted} nonfinite estimate(s) for "
            f"{method_name!r} in simulation {simulation_number}.",
            RuntimeWarning,
            stacklevel=2,
        )

    finite_estimates = estimates[finite_mask]

    if finite_estimates.size == 0:
        raise ValueError(
            f"Estimator {method_name!r} has no finite estimates in "
            f"simulation {simulation_number}."
        )

    return finite_estimates


def get_true_parameter(
    simulation_result: dict,
    simulation_number: int,
) -> float:
    """Extract and validate the true parameter for one simulation."""
    if "true_parameter" not in simulation_result:
        raise KeyError(
            f"Simulation result {simulation_number} does not contain "
            "'true_parameter'."
        )

    true_parameter = float(
        simulation_result["true_parameter"]
    )

    if not np.isfinite(true_parameter):
        raise ValueError(
            f"The true parameter for simulation {simulation_number} "
            f"is not finite: {true_parameter!r}."
        )

    return true_parameter


def format_method_name(method_name: str) -> str:
    """
    Format a configuration estimator name for display.

    Examples:
        'logit' -> 'logit'
        'logit_plugin' -> 'logit plugin'
    """
    return method_name.replace("_", " ")


def create_violin_plots(
    simulation_results: list[dict],
    simulations: list[dict] = SIMULATIONS,
):
    """
    Create all simulation panels side by side in one row.

    Each panel contains one violin for every configured estimator.
    The panel's y-axis is symmetric around its true parameter, placing
    the true-value line at the vertical center.

    The estimate distributions themselves are not recentered.
    """
    if len(simulation_results) != len(simulations):
        raise ValueError(
            f"The pickle contains {len(simulation_results)} simulation "
            f"results, but the configuration contains "
            f"{len(simulations)} simulations. Regenerate the pickle "
            "using the current configuration."
        )

    number_of_simulations = len(simulations)

    if number_of_simulations == 0:
        raise ValueError("SIMULATIONS cannot be empty.")

    # First extract all data. This allows us to determine a common
    # y-axis half-range while centering each panel on its own truth.
    plot_data = []
    maximum_absolute_deviation = 0.0

    for simulation_index, (
        simulation_result,
        simulation_config,
    ) in enumerate(
        zip(simulation_results, simulations),
        start=1,
    ):
        if "n_individuals" not in simulation_config:
            raise KeyError(
                f"Simulation configuration {simulation_index} does "
                "not contain 'n_individuals'."
            )

        if "param_estimators" not in simulation_config:
            raise KeyError(
                f"Simulation configuration {simulation_index} does "
                "not contain 'param_estimators'."
            )

        number_of_individuals = simulation_config[
            "n_individuals"
        ]

        method_names = list(
            simulation_config["param_estimators"]
        )

        if not method_names:
            raise ValueError(
                f"Simulation configuration {simulation_index} does "
                "not contain any parameter estimators."
            )

        estimator_class_names = [
            get_estimator_class_name(method_name)
            for method_name in method_names
        ]

        if len(estimator_class_names) != len(
            set(estimator_class_names)
        ):
            raise ValueError(
                f"Simulation configuration {simulation_index} maps "
                "multiple method names to the same estimator class. "
                "The current pickle format cannot distinguish them."
            )

        method_estimates = [
            get_method_estimates(
                simulation_result=simulation_result,
                method_name=method_name,
                simulation_number=simulation_index,
            )
            for method_name in method_names
        ]

        true_parameter = get_true_parameter(
            simulation_result,
            simulation_index,
        )

        # Find the largest distance between any estimate and its
        # simulation-specific true parameter.
        for estimates in method_estimates:
            method_maximum_deviation = float(
                np.max(
                    np.abs(
                        estimates - true_parameter
                    )
                )
            )

            maximum_absolute_deviation = max(
                maximum_absolute_deviation,
                method_maximum_deviation,
            )

        plot_data.append(
            {
                "n_individuals": number_of_individuals,
                "method_names": method_names,
                "method_estimates": method_estimates,
                "true_parameter": true_parameter,
            }
        )

    # Prevent a zero-width axis if every estimate equals its truth.
    if maximum_absolute_deviation == 0.0:
        maximum_absolute_deviation = 0.1

    # Add a small amount of space above and below all observations.
    y_axis_half_range = 1.05 * maximum_absolute_deviation

    # One row, with every simulation side by side.
    figure, axes = plt.subplots(
        nrows=1,
        ncols=number_of_simulations,
        figsize=(
            4.8 * number_of_simulations,
            5.5,
        ),
        squeeze=False,
        sharey=False,
        constrained_layout=True,
    )

    flattened_axes = axes.ravel()
    colors = plt.get_cmap("Set2").colors

    for simulation_index, (
        axis,
        simulation_data,
    ) in enumerate(
        zip(flattened_axes, plot_data),
        start=1,
    ):
        number_of_individuals = simulation_data[
            "n_individuals"
        ]
        method_names = simulation_data[
            "method_names"
        ]
        method_estimates = simulation_data[
            "method_estimates"
        ]
        true_parameter = simulation_data[
            "true_parameter"
        ]

        positions = np.arange(
            1,
            len(method_names) + 1,
        )

        violin_parts = axis.violinplot(
            dataset=method_estimates,
            positions=positions,
            widths=0.75,
            showmeans=True,
            showmedians=True,

            # Removes minima, maxima, and vertical range bars.
            showextrema=False,
        )

        # Give each method's violin a separate color.
        for method_index, violin_body in enumerate(
            violin_parts["bodies"]
        ):
            violin_body.set_facecolor(
                colors[method_index % len(colors)]
            )
            violin_body.set_edgecolor("black")
            violin_body.set_alpha(0.75)

        # Style only the mean and median lines. There are no extrema
        # or vertical range lines after showextrema=False.
        for part_name in (
            "cmeans",
            "cmedians",
        ):
            if part_name in violin_parts:
                violin_parts[part_name].set_color(
                    "black"
                )
                violin_parts[part_name].set_linewidth(
                    1.2
                )

        axis.axhline(
            y=true_parameter,
            color="crimson",
            linestyle="--",
            linewidth=2.0,
            label=(
                f"Truth = {true_parameter:.4f}"
            ),
        )

        # Center this panel vertically on its own true parameter.
        axis.set_ylim(
            true_parameter - y_axis_half_range,
            true_parameter + y_axis_half_range,
        )

        axis.set_xticks(positions)
        axis.set_xticklabels(
            [
                format_method_name(method_name)
                for method_name in method_names
            ],
            rotation=15,
            ha="right",
        )

        axis.set_title(
            f"n = {number_of_individuals:,}"
        )
        axis.set_xlabel("Estimation method")

        # Give every panel a label because their y-axis values differ.
        axis.set_ylabel("Estimated treatment effect")

        axis.grid(
            axis="y",
            linestyle=":",
            alpha=0.4,
        )

        axis.legend(
            loc="upper right",
            frameon=True,
        )

    figure.suptitle(
        "Sampling Distributions of Treatment-Effect Estimators",
        fontsize=16,
    )

    return figure


def save_violin_plots(
    figure
) -> Path:
    """Save the combined violin-plot figure."""
    figure_path = Path(VIOLIN_IMAGE_LOCATION)
    figure_path.parent.mkdir(parents=True, exist_ok=True)

    figure.savefig(
        figure_path,
        dpi=OUTPUT_DPI,
        bbox_inches="tight",
    )

    return figure_path


def main() -> None:
    simulation_results = load_simulation_results(
        PICKLE_LOCATION
    )

    figure = create_violin_plots(
        simulation_results=simulation_results,
        simulations=SIMULATIONS,
    )

    output_path = save_violin_plots(
        figure
    )

    print(
        f"Violin plots saved to: "
        f"{output_path.resolve()}"
    )




if __name__ == "__main__":
    main()