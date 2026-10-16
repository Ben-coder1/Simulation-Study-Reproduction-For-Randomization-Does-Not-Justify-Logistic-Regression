"""
Generate all figures for the differential log-odds simulations.

This file creates:

1. A combined violin-plot figure containing one panel for each
   configured simulation.

2. A combined paired-error scatterplot figure containing one panel for
   each configured simulation.

The estimand is the marginal differential log odds:

    logit(mean outcome under treatment)
    - logit(mean outcome under control).

Both figures assume that the simulation results in the pickle occur in
the same order as configuration.SIMULATIONS.

The scatterplot requires exactly two estimators. For each randomized
experiment, it plots:

    x = estimate from method 1 - true differential log odds
    y = estimate from method 2 - true differential log odds
"""

import pickle
import textwrap
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from configuration import (
    PICKLE_LOCATION,
    SCATTER_IMAGE_LOCATION,
    SIMULATIONS,
    VIOLIN_IMAGE_LOCATION,
)
from src.Methods import PARAMETER_ESTIMATORS


# ---------------------------------------------------------------------
# Shared figure settings
# ---------------------------------------------------------------------

OUTPUT_DPI = 300

FONT_FAMILY = "DejaVu Sans"
FONT_SIZE = 10


# ---------------------------------------------------------------------
# Violin-plot settings
# ---------------------------------------------------------------------

VIOLIN_WIDTH_INCHES = 6.5
VIOLIN_HEIGHT_INCHES = 5.0


# ---------------------------------------------------------------------
# Scatterplot settings
# ---------------------------------------------------------------------

SCATTER_WIDTH_INCHES = 10.0
SCATTER_HEIGHT_INCHES = 3.8

SCATTER_POINT_COLOR = "#4C72B0"
SCATTER_POINT_SIZE = 13
SCATTER_POINT_ALPHA = 0.40


# ---------------------------------------------------------------------
# Data-loading and result-extraction functions
# ---------------------------------------------------------------------

def load_simulation_results(
    pickle_location: str | Path = PICKLE_LOCATION,
) -> list[dict]:
    """Load simulation results from the existing pickle file."""
    pickle_path = Path(pickle_location)

    if not pickle_path.is_file():
        raise FileNotFoundError(
            "Simulation pickle not found at "
            f"{pickle_path.resolve()}."
        )

    with pickle_path.open("rb") as file:
        simulation_results = pickle.load(file)

    if not isinstance(simulation_results, list):
        raise TypeError(
            "Expected the pickle file to contain a list of "
            "simulation-result dictionaries."
        )

    if not simulation_results:
        raise ValueError(
            "The simulation-results pickle is empty."
        )

    return simulation_results


def get_estimator_class_name(
    method_name: str,
) -> str:
    """
    Return the estimator class name used as a key in the pickle.

    For example:

        "logit" -> "LogitModelEstimator"
    """
    if method_name not in PARAMETER_ESTIMATORS:
        raise KeyError(
            f"Estimator {method_name!r} is not registered in "
            "PARAMETER_ESTIMATORS."
        )

    return PARAMETER_ESTIMATORS[
        method_name
    ].__name__


def format_method_name(
    method_name: str,
) -> str:
    """Convert a configuration estimator name to a display label."""
    display_names = {
        "logit": "Logit",
        "logit_plugin": "Plug-in",
    }

    return display_names.get(
        method_name,
        method_name.replace("_", " ").title(),
    )


def join_method_names(
    method_names: list[str],
) -> str:
    """Join estimator names grammatically for a figure title."""
    labels = [
        format_method_name(method_name)
        for method_name in method_names
    ]

    if len(labels) == 1:
        return labels[0]

    if len(labels) == 2:
        return f"{labels[0]} and {labels[1]}"

    return (
        ", ".join(labels[:-1])
        + f", and {labels[-1]}"
    )


def get_method_colors(
    number_of_methods: int,
) -> list:
    """
    Return colors for the estimator violins.

    When exactly two methods are present:

        first method  -> red
        second method -> blue
    """
    if number_of_methods <= 0:
        raise ValueError(
            "number_of_methods must be positive."
        )

    if number_of_methods == 2:
        return [
            "#D62728",  # Red
            "#1F77B4",  # Blue
        ]

    color_map = plt.get_cmap("tab10")

    return [
        color_map(index % color_map.N)
        for index in range(number_of_methods)
    ]


def get_method_estimates(
    simulation_result: dict,
    method_name: str,
    simulation_number: int,
    remove_nonfinite: bool,
) -> np.ndarray:
    """
    Extract one estimator's results.

    Parameters
    ----------
    remove_nonfinite
        If True, nonfinite estimates are removed.

        This is appropriate for individual violin distributions.

        If False, nonfinite values are preserved so the scatterplot can
        apply a joint mask to both estimators and preserve experiment
        pairing.
    """
    if not isinstance(simulation_result, dict):
        raise TypeError(
            f"Simulation result {simulation_number} must be a "
            "dictionary."
        )

    if "estimator_results" not in simulation_result:
        raise KeyError(
            f"Simulation result {simulation_number} does not contain "
            "'estimator_results'."
        )

    estimator_results = simulation_result[
        "estimator_results"
    ]

    if not isinstance(estimator_results, dict):
        raise TypeError(
            f"'estimator_results' for simulation "
            f"{simulation_number} must be a dictionary."
        )

    estimator_class_name = get_estimator_class_name(
        method_name
    )

    if estimator_class_name not in estimator_results:
        raise KeyError(
            f"Simulation result {simulation_number} does not contain "
            f"results for {estimator_class_name!r}, corresponding to "
            f"configured method {method_name!r}. Available estimators: "
            f"{list(estimator_results)}."
        )

    method_result = estimator_results[
        estimator_class_name
    ]

    if not isinstance(method_result, dict):
        raise TypeError(
            f"Results for {estimator_class_name!r} in simulation "
            f"{simulation_number} must be a dictionary."
        )

    if "estimates" not in method_result:
        raise KeyError(
            f"Results for {estimator_class_name!r} in simulation "
            f"{simulation_number} do not contain 'estimates'."
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

    if not remove_nonfinite:
        return estimates

    finite_mask = np.isfinite(estimates)
    number_omitted = int(
        np.sum(~finite_mask)
    )

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
    """Extract the true marginal differential log-odds parameter."""
    if not isinstance(simulation_result, dict):
        raise TypeError(
            f"Simulation result {simulation_number} must be a "
            "dictionary."
        )

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
            f"The true differential log odds in simulation "
            f"{simulation_number} is not finite: "
            f"{true_parameter!r}."
        )

    return true_parameter


def validate_configurations(
    simulation_results: list[dict],
    simulations: list[dict],
) -> list[str]:
    """
    Validate result/configuration alignment and return method names.

    All simulations must contain the same methods in the same order.
    """
    if len(simulation_results) != len(simulations):
        raise ValueError(
            f"The pickle contains {len(simulation_results)} simulation "
            f"results, but the configuration contains "
            f"{len(simulations)} simulations. Regenerate the pickle "
            "using the current configuration."
        )

    if not simulations:
        raise ValueError(
            "SIMULATIONS cannot be empty."
        )

    expected_methods = None

    for simulation_number, simulation_config in enumerate(
        simulations,
        start=1,
    ):
        if not isinstance(simulation_config, dict):
            raise TypeError(
                f"Simulation configuration {simulation_number} must "
                "be a dictionary."
            )

        required_keys = {
            "n_individuals",
            "param_estimators",
        }

        missing_keys = (
            required_keys - simulation_config.keys()
        )

        if missing_keys:
            raise KeyError(
                f"Simulation configuration {simulation_number} is "
                f"missing keys: {sorted(missing_keys)}."
            )

        method_names = list(
            simulation_config["param_estimators"]
        )

        if not method_names:
            raise ValueError(
                f"Simulation configuration {simulation_number} "
                "contains no parameter estimators."
            )

        estimator_class_names = [
            get_estimator_class_name(method_name)
            for method_name in method_names
        ]

        if len(estimator_class_names) != len(
            set(estimator_class_names)
        ):
            raise ValueError(
                f"Simulation configuration {simulation_number} maps "
                "multiple method names to the same estimator class. "
                "The current pickle format cannot distinguish them."
            )

        if expected_methods is None:
            expected_methods = method_names
        elif method_names != expected_methods:
            raise ValueError(
                "Every simulation must contain the same estimators "
                "in the same order. "
                f"Simulation {simulation_number} contains "
                f"{method_names!r}; expected "
                f"{expected_methods!r}."
            )

    return expected_methods


# ---------------------------------------------------------------------
# Violin plots
# ---------------------------------------------------------------------

def create_violin_plots(
    simulation_results: list[dict],
    simulations: list[dict] = SIMULATIONS,
    caption: str | None = None,
):
    """
    Create side-by-side violin plots of differential log-odds estimates.

    Each panel represents one simulation. Its dashed horizontal line is
    the true differential log odds for that frozen finite population.

    Each panel's y-axis is centered on its own true value. The estimates
    themselves are not transformed or recentered.
    """
    method_names = validate_configurations(
        simulation_results,
        simulations,
    )

    method_colors = get_method_colors(
        len(method_names)
    )

    plot_data = []
    maximum_absolute_deviation = 0.0

    for simulation_number, (
        simulation_result,
        simulation_config,
    ) in enumerate(
        zip(simulation_results, simulations),
        start=1,
    ):
        method_estimates = [
            get_method_estimates(
                simulation_result=simulation_result,
                method_name=method_name,
                simulation_number=simulation_number,
                remove_nonfinite=True,
            )
            for method_name in method_names
        ]

        true_parameter = get_true_parameter(
            simulation_result,
            simulation_number,
        )

        for estimates in method_estimates:
            maximum_absolute_deviation = max(
                maximum_absolute_deviation,
                float(
                    np.max(
                        np.abs(
                            estimates - true_parameter
                        )
                    )
                ),
            )

        plot_data.append(
            {
                "n_individuals": simulation_config[
                    "n_individuals"
                ],
                "method_estimates": method_estimates,
                "true_parameter": true_parameter,
            }
        )

    if maximum_absolute_deviation == 0.0:
        maximum_absolute_deviation = 0.1

    y_axis_half_range = (
        1.05 * maximum_absolute_deviation
    )

    title = (
    "Simulated Distribution of Treatment Effect Estimators under "
    "non-logit randomization model: "
    f"{join_method_names(method_names)}"
    )

    wrapped_title = textwrap.fill(
        title,
        width=62,
    )

    wrapped_caption = (
        textwrap.fill(
            caption.strip(),
            width=82,
        )
        if caption is not None and caption.strip()
        else None
    )

    with plt.rc_context(
        {
            "font.family": FONT_FAMILY,
            "font.size": FONT_SIZE,
            "axes.titlesize": FONT_SIZE,
            "axes.labelsize": FONT_SIZE,
            "xtick.labelsize": FONT_SIZE,
            "ytick.labelsize": FONT_SIZE,
            "legend.fontsize": FONT_SIZE,
            "figure.titlesize": FONT_SIZE,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.transparent": True,
        }
    ):
        figure, axes = plt.subplots(
            nrows=1,
            ncols=len(plot_data),
            figsize=(
                VIOLIN_WIDTH_INCHES,
                VIOLIN_HEIGHT_INCHES,
            ),
            squeeze=False,
            sharey=False,
            facecolor="none",
        )

        for simulation_number, (
            axis,
            simulation_data,
        ) in enumerate(
            zip(axes.ravel(), plot_data),
            start=1,
        ):
            axis.set_facecolor("none")

            positions = np.arange(
                1,
                len(method_names) + 1,
            )

            violin_parts = axis.violinplot(
                dataset=simulation_data[
                    "method_estimates"
                ],
                positions=positions,
                widths=0.75,
                showmeans=True,
                showmedians=True,
                showextrema=False,
            )

            for method_index, violin_body in enumerate(
                violin_parts["bodies"]
            ):
                violin_body.set_facecolor(
                    method_colors[method_index]
                )
                violin_body.set_edgecolor("black")
                violin_body.set_linewidth(0.8)
                violin_body.set_alpha(0.75)

            for part_name in (
                "cmeans",
                "cmedians",
            ):
                if part_name in violin_parts:
                    violin_parts[
                        part_name
                    ].set_color("black")

                    violin_parts[
                        part_name
                    ].set_linewidth(1.0)

            true_parameter = simulation_data[
                "true_parameter"
            ]

            axis.axhline(
                y=true_parameter,
                color="black",
                linestyle="--",
                linewidth=1.2,
                zorder=0,
            )

            axis.set_ylim(
                true_parameter - y_axis_half_range,
                true_parameter + y_axis_half_range,
            )

            axis.set_xticks(
                positions
            )

            axis.set_xticklabels(
                [
                    textwrap.fill(
                        format_method_name(method_name),
                        width=10,
                    )
                    for method_name in method_names
                ],
                rotation=0,
                ha="center",
            )

            axis.set_title(
                "n = "
                f"{simulation_data['n_individuals']:,}"
            )

            axis.set_xlabel(
                "Estimator"
            )

            if simulation_number == 1:
                axis.set_ylabel(
                    "Differential log-odds estimate"
                )

            axis.grid(
                axis="y",
                linestyle=":",
                linewidth=0.7,
                alpha=0.4,
            )

        legend_handles = [
            Patch(
                facecolor=method_colors[index],
                edgecolor="black",
                linewidth=0.8,
                alpha=0.75,
                label=format_method_name(method_name),
            )
            for index, method_name in enumerate(
                method_names
            )
        ]

        legend_handles.append(
            Line2D(
                [0],
                [0],
                color="black",
                linestyle="--",
                linewidth=1.2,
                label="True differential log odds",
            )
        )

        figure.suptitle(
            wrapped_title,
            x=0.5,
            y=0.97,
            ha="center",
            va="top",
            fontweight="normal",
            linespacing=1.2,
        )


        figure.legend(
            handles=legend_handles,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.80),
            ncol=min(
                len(legend_handles),
                3,
            ),
            frameon=False,
            borderaxespad=0.0,
            handlelength=1.8,
            columnspacing=1.2,
        )

        if wrapped_caption:
            figure.text(
                0.5,
                0.025,
                wrapped_caption,
                ha="center",
                va="bottom",
                linespacing=1.25,
                wrap=True,
            )

        figure.subplots_adjust(
            left=0.105,
            right=0.985,
            top=0.66,
            bottom=(
                0.27
                if wrapped_caption
                else 0.18
            ),
            wspace=0.48,
        )

    return figure


def save_violin_plots(
    figure,
    output_location: str | Path = VIOLIN_IMAGE_LOCATION,
) -> Path:
    """Save the violin figure to the configured PDF location."""
    output_path = Path(output_location)

    if output_path.suffix.lower() != ".pdf":
        raise ValueError(
            "VIOLIN_IMAGE_LOCATION must use a .pdf extension."
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure.savefig(
        output_path,
        format="pdf",
        dpi=OUTPUT_DPI,
        transparent=True,
        facecolor="none",
        edgecolor="none",
        bbox_inches="tight",
        pad_inches=0.15,
    )

    return output_path


# ---------------------------------------------------------------------
# Paired-error scatterplots
# ---------------------------------------------------------------------

def create_error_scatterplots(
    simulation_results: list[dict],
    simulations: list[dict] = SIMULATIONS,
    caption: str | None = None,
):
    """
    Create side-by-side paired-error scatterplots.

    This plot requires exactly two estimators configured in this order:

        1. Logit
        2. Plug-in

    For each randomized experiment:

        x = plug-in estimate - true differential log odds
        y = logit estimate - true differential log odds

    Each point therefore represents the paired estimation errors from
    the same randomized experiment.
    """
    method_names = validate_configurations(
        simulation_results,
        simulations,
    )

    if len(method_names) != 2:
        raise ValueError(
            "The paired-error scatterplot requires exactly two "
            f"estimators, but found {len(method_names)}: "
            f"{method_names!r}."
        )

    plot_data = []

    for simulation_number, (
        simulation_result,
        simulation_config,
    ) in enumerate(
        zip(simulation_results, simulations),
        start=1,
    ):
        first_estimates = get_method_estimates(
            simulation_result=simulation_result,
            method_name=method_names[0],
            simulation_number=simulation_number,
            remove_nonfinite=False,
        )

        second_estimates = get_method_estimates(
            simulation_result=simulation_result,
            method_name=method_names[1],
            simulation_number=simulation_number,
            remove_nonfinite=False,
        )

        if first_estimates.size != second_estimates.size:
            raise ValueError(
                f"Simulation {simulation_number} contains "
                f"{first_estimates.size} estimates for method 1 and "
                f"{second_estimates.size} estimates for method 2."
            )

        true_parameter = get_true_parameter(
            simulation_result,
            simulation_number,
        )

        finite_mask = (
            np.isfinite(first_estimates)
            & np.isfinite(second_estimates)
        )

        number_omitted = int(
            np.sum(~finite_mask)
        )

        if number_omitted:
            warnings.warn(
                f"Omitting {number_omitted} randomized experiment(s) "
                f"from scatterplot simulation {simulation_number} "
                "because at least one estimate is nonfinite.",
                RuntimeWarning,
                stacklevel=2,
            )

        if not finite_mask.any():
            raise ValueError(
                f"Simulation {simulation_number} has no experiments "
                "for which both estimators are finite."
            )

        plot_data.append(
            {
                "n_individuals": simulation_config[
                    "n_individuals"
                ],
                "first_errors": (
                    first_estimates[finite_mask]
                    - true_parameter
                ),
                "second_errors": (
                    second_estimates[finite_mask]
                    - true_parameter
                ),
            }
        )

    first_method_label = format_method_name(
        method_names[0]
    )

    second_method_label = format_method_name(
        method_names[1]
    )

    maximum_absolute_error = max(
        float(
            np.max(
                np.abs(
                    np.concatenate(
                        (
                            simulation_data[
                                "first_errors"
                            ],
                            simulation_data[
                                "second_errors"
                            ],
                        )
                    )
                )
            )
        )
        for simulation_data in plot_data
    )

    if maximum_absolute_error == 0.0:
        maximum_absolute_error = 0.1

    axis_limit = (
        1.05 * maximum_absolute_error
    )

    wrapped_title = textwrap.fill(
        (
            "Paired Differential Log-Odds Estimation Errors: "
            f"{first_method_label} and {second_method_label}"
        ),
        width=85,
    )

    wrapped_caption = (
        textwrap.fill(
            caption.strip(),
            width=125,
        )
        if caption is not None and caption.strip()
        else None
    )

    with plt.rc_context(
        {
            "font.family": FONT_FAMILY,
            "font.size": FONT_SIZE,
            "axes.titlesize": FONT_SIZE,
            "axes.labelsize": FONT_SIZE,
            "xtick.labelsize": FONT_SIZE,
            "ytick.labelsize": FONT_SIZE,
            "figure.titlesize": FONT_SIZE,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.transparent": True,
        }
    ):
        figure, axes = plt.subplots(
            nrows=1,
            ncols=len(plot_data),
            figsize=(
                SCATTER_WIDTH_INCHES,
                SCATTER_HEIGHT_INCHES,
            ),
            squeeze=False,
            sharex=True,
            sharey=True,
            facecolor="none",
        )

        for axis, simulation_data in zip(
            axes.ravel(),
            plot_data,
        ):
            axis.set_facecolor("none")

            axis.scatter(
            # x-axis: Plug-in estimate minus truth
            simulation_data["second_errors"],

            # y-axis: Logit estimate minus truth
            simulation_data["first_errors"],

            s=SCATTER_POINT_SIZE,
            color=SCATTER_POINT_COLOR,
            alpha=SCATTER_POINT_ALPHA,
            edgecolors="none",
            rasterized=True,
)

            axis.axvline(
                x=0.0,
                color="black",
                linestyle="--",
                linewidth=0.9,
                zorder=0,
            )

            axis.axhline(
                y=0.0,
                color="black",
                linestyle="--",
                linewidth=0.9,
                zorder=0,
            )

            axis.set_xlim(
                -axis_limit,
                axis_limit,
            )

            axis.set_ylim(
                -axis_limit,
                axis_limit,
            )

            axis.set_aspect(
                "equal",
                adjustable="box",
            )

            axis.set_title(
                "n = "
                f"{simulation_data['n_individuals']:,}"
            )

            axis.grid(
                linestyle=":",
                linewidth=0.7,
                alpha=0.35,
            )

        figure.suptitle(
            wrapped_title,
            x=0.5,
            y=0.97,
            ha="center",
            va="top",
            fontweight="normal",
            linespacing=1.2,
        )

        figure.supxlabel(
            "Plug-in - True",
            y=(
                0.15
                if wrapped_caption
                else 0.06
            ),
        )

        figure.supylabel(
            "Logit - True",
            x=0.025,
        )

        figure.supylabel(
            f"{first_method_label}"
            "- true",
            x=0.025,
        )

        if wrapped_caption:
            figure.text(
                0.5,
                0.025,
                wrapped_caption,
                ha="center",
                va="bottom",
                linespacing=1.25,
                wrap=True,
            )

        figure.subplots_adjust(
            left=0.08,
            right=0.99,
            top=0.82,
            bottom=(
                0.24
                if wrapped_caption
                else 0.14
            ),
            wspace=0.25,
        )

    return figure


def save_error_scatterplots(
    figure,
    output_location: str | Path = SCATTER_IMAGE_LOCATION,
) -> Path:
    """Save the scatterplot figure to the configured PDF location."""
    output_path = Path(output_location)

    if output_path.suffix.lower() != ".pdf":
        raise ValueError(
            "SCATTER_IMAGE_LOCATION must use a .pdf extension."
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure.savefig(
        output_path,
        format="pdf",
        dpi=OUTPUT_DPI,
        transparent=True,
        facecolor="none",
        edgecolor="none",
        bbox_inches="tight",
        pad_inches=0.15,
    )

    return output_path


# ---------------------------------------------------------------------
# Main entry point: generate every figure
# ---------------------------------------------------------------------

def main() -> None:
    """Load one pickle and generate all configured figures."""
    simulation_results = load_simulation_results(
        PICKLE_LOCATION
    )

    violin_figure = create_violin_plots(
        simulation_results=simulation_results,
        simulations=SIMULATIONS,
        caption=(
            "The raw logit treatment-coefficient estimator has much "
            "higher bias than the plug-in estimator. The plug-in "
            "estimator also has slightly reduced variance"
        ),
    )

    violin_path = save_violin_plots(
        figure=violin_figure,
        output_location=VIOLIN_IMAGE_LOCATION,
    )

    plt.close(violin_figure)

    scatter_figure = create_error_scatterplots(
        simulation_results=simulation_results,
        simulations=SIMULATIONS,
        caption=(
            "There is a clear, nearly perfect linear relationship between "
            "the errors  of the two estimators. Under these simulations, the errors of the plug-in "
            "estimator behaves approximately like a shifted and scaled "
            "version of the errors of the logit estimator."
        ),
    )

    scatter_path = save_error_scatterplots(
        figure=scatter_figure,
        output_location=SCATTER_IMAGE_LOCATION,
    )

    plt.close(scatter_figure)

    print("Figures saved successfully:")
    print(f"  Violin plots: {violin_path.resolve()}")
    print(f"  Error scatterplots: {scatter_path.resolve()}")


if __name__ == "__main__":
    main()