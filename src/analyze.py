"""
Purpose-built reproduction script.

This module exists solely to reproduce the particular results table/chart
containing:

    beta_1, beta_2, beta_3, plug-in estimate, and true parameter

along with the empirical standard deviations of the estimated quantities.

It is intentionally tied to the current simulation output structure and
should not be used as a general-purpose simulation-analysis module.
"""

import pickle
import warnings
from pathlib import Path

import dataframe_image as dfi
import numpy as np
import pandas as pd

from configuration import PICKLE_LOCATION, ROW_NAMES
from src.metrics import bias

LOGIT_ESTIMATOR_NAME = "LogitModelEstimator"
PLUGIN_ESTIMATOR_NAME = "LogitModelPlugIn"

CSV_OUTPUT_LOCATION = "simulation_results.csv"
IMAGE_OUTPUT_LOCATION = "simulation_results.png"


def mean_and_sd(values: np.ndarray) -> tuple[float, float]:
    """
    Calculate the mean and sample standard deviation, ignoring NaNs.

    Returns (NaN, NaN) if there are no finite estimates. The standard
    deviation is NaN if only one finite estimate is available.
    """
    values = np.asarray(values, dtype=float).ravel()
    finite_values = values[np.isfinite(values)]

    if finite_values.size == 0:
        return np.nan, np.nan

    mean = float(np.mean(finite_values))

    if finite_values.size < 2:
        standard_deviation = np.nan
    else:
        standard_deviation = float(
            np.std(finite_values, ddof=1)
        )

    return mean, standard_deviation


def load_simulation_data(
    pickle_location: str | Path = PICKLE_LOCATION,
):
    """Load the labeled simulation results produced by simulation.py."""
    pickle_path = Path(pickle_location)

    if not pickle_path.exists():
        raise FileNotFoundError(
            f"Simulation output was not found at "
            f"{pickle_path.resolve()}."
        )

    with pickle_path.open("rb") as file:
        simulation_results = pickle.load(file)

    if not isinstance(simulation_results, list):
        raise ValueError(
            "Expected the pickle file to contain a list of "
            "simulation results."
        )

    if not simulation_results:
        raise ValueError("The simulation-results file is empty.")

    return simulation_results


def validate_simulation_result(
    simulation_result,
    row_number: int,
) -> None:
    """Validate the fields required for this particular table."""
    if not isinstance(simulation_result, dict):
        raise ValueError(
            f"Simulation result {row_number} must be a dictionary."
        )

    required_top_level_keys = {
        "true_parameter",
        "estimator_results",
    }

    missing_keys = (
        required_top_level_keys - simulation_result.keys()
    )

    if missing_keys:
        raise ValueError(
            f"Simulation result {row_number} is missing keys: "
            f"{sorted(missing_keys)}."
        )

    estimator_results = simulation_result["estimator_results"]

    if not isinstance(estimator_results, dict):
        raise ValueError(
            f"'estimator_results' in row {row_number} must be "
            "a dictionary."
        )

    required_estimators = {
        LOGIT_ESTIMATOR_NAME,
        PLUGIN_ESTIMATOR_NAME,
    }

    missing_estimators = (
        required_estimators - estimator_results.keys()
    )

    if missing_estimators:
        raise ValueError(
            f"Simulation result {row_number} is missing estimator "
            f"results for: {sorted(missing_estimators)}. Make sure "
            "both 'logit' and 'logit_plugin' are included in the "
            "configuration."
        )

    logit_result = estimator_results[LOGIT_ESTIMATOR_NAME]
    plugin_result = estimator_results[PLUGIN_ESTIMATOR_NAME]

    if "auxiliary" not in logit_result:
        raise ValueError(
            f"{LOGIT_ESTIMATOR_NAME} in row {row_number} does not "
            "contain 'auxiliary' coefficient results."
        )

    if "estimates" not in plugin_result:
        raise ValueError(
            f"{PLUGIN_ESTIMATOR_NAME} in row {row_number} does not "
            "contain 'estimates'."
        )


def summarize_simulation_result(
    simulation_result,
    row_number: int,
) -> dict[str, float]:
    """
    Summarize one sample-size scenario.

    The LogitModelEstimator auxiliary output is expected to be an array
    with one row per experiment and one column per fitted coefficient.

    For this specific reproduction, exactly three coefficient columns are
    required and are displayed as beta_1, beta_2, and beta_3.
    """
    validate_simulation_result(
        simulation_result,
        row_number,
    )

    estimator_results = simulation_result["estimator_results"]

    logit_coefficients = np.asarray(
        estimator_results[
            LOGIT_ESTIMATOR_NAME
        ]["auxiliary"],
        dtype=float,
    )

    plugin_estimates = np.asarray(
        estimator_results[
            PLUGIN_ESTIMATOR_NAME
        ]["estimates"],
        dtype=float,
    ).ravel()

    true_parameter = float(
        simulation_result["true_parameter"]
    )

    if logit_coefficients.ndim != 2:
        raise ValueError(
            f"The logit auxiliary output in row {row_number} must "
            "be a two-dimensional coefficient matrix. Received "
            f"shape {logit_coefficients.shape}."
        )

    if logit_coefficients.shape[1] != 3:
        raise ValueError(
            f"This purpose-built reproduction expects exactly three "
            f"logit coefficients, but row {row_number} contains "
            f"{logit_coefficients.shape[1]} columns."
        )

    if logit_coefficients.shape[0] != plugin_estimates.size:
        raise ValueError(
            f"Row {row_number} contains "
            f"{logit_coefficients.shape[0]} logit fits but "
            f"{plugin_estimates.size} plug-in estimates."
        )

    beta_1_mean, beta_1_sd = mean_and_sd(
        logit_coefficients[:, 0]
    )
    beta_2_mean, beta_2_sd = mean_and_sd(
        logit_coefficients[:, 1]
    )
    beta_3_mean, beta_3_sd = mean_and_sd(
        logit_coefficients[:, 2]
    )

    plugin_mean, plugin_sd = mean_and_sd(
        plugin_estimates
    )

    logit_failure_rate = float(
        np.mean(
            ~np.all(
                np.isfinite(logit_coefficients),
                axis=1,
            )
        )
    )

    plugin_failure_rate = float(
        np.mean(~np.isfinite(plugin_estimates))
    )

    if logit_failure_rate > 0:
        warnings.warn(
            f"Simulation row {row_number} has a logit-fit failure "
            f"rate of {logit_failure_rate:.2%}. Non-finite fits "
            "were excluded from means and standard deviations.",
            RuntimeWarning,
            stacklevel=2,
        )

    if plugin_failure_rate > 0:
        warnings.warn(
            f"Simulation row {row_number} has a plug-in failure "
            f"rate of {plugin_failure_rate:.2%}. Non-finite "
            "estimates were excluded from means and standard "
            "deviations.",
            RuntimeWarning,
            stacklevel=2,
        )

    return {
        "beta_1": beta_1_mean,
        "sd(beta_1)": beta_1_sd,
        "beta_2": beta_2_mean,
        "sd(beta_2)": beta_2_sd,
        "beta_3": beta_3_mean,
        "sd(beta_3)": beta_3_sd,
        "plug_in_estimate": plugin_mean,
        "sd(plug_in_estimate)": plugin_sd,
        "true_parameter": true_parameter,
    }


def summarize_simulation_results(
    simulation_results,
    row_names=None,
) -> pd.DataFrame:
    """
    Convert all sample-size scenarios into the reproduction table.
    """
    if not simulation_results:
        raise ValueError(
            "simulation_results cannot be empty."
        )

    if row_names is None:
        row_names = [
            f"Simulation {index}"
            for index in range(
                1,
                len(simulation_results) + 1,
            )
        ]
    else:
        row_names = list(row_names)

    if len(row_names) != len(simulation_results):
        raise ValueError(
            f"Received {len(row_names)} row names for "
            f"{len(simulation_results)} simulation results. "
            "ROW_NAMES and SIMULATIONS must use the same order."
        )

    dataframe_rows = []

    for row_number, (
        row_name,
        simulation_result,
    ) in enumerate(
        zip(row_names, simulation_results),
        start=1,
    ):
        summary = summarize_simulation_result(
            simulation_result,
            row_number,
        )

        dataframe_rows.append(
            {
                "n": row_name,
                **summary,
            }
        )

    column_order = [
        "n",
        "beta_1",
        "sd(beta_1)",
        "beta_2",
        "sd(beta_2)",
        "beta_3",
        "sd(beta_3)",
        "plug_in_estimate",
        "sd(plug_in_estimate)",
        "true_parameter",
    ]

    return pd.DataFrame(
        dataframe_rows,
        columns=column_order,
    )


def analyze_simulation_data() -> pd.DataFrame:
    """Load and summarize the current simulation output."""
    simulation_results = load_simulation_data(
        PICKLE_LOCATION
    )

    return summarize_simulation_results(
        simulation_results,
        row_names=ROW_NAMES,
    )


def export_reproduction_table(
    dataframe: pd.DataFrame,
    csv_location: str | Path = CSV_OUTPUT_LOCATION,
    image_location: str | Path = IMAGE_OUTPUT_LOCATION,
) -> None:
    """Export the purpose-built reproduction table."""
    csv_path = Path(csv_location)
    image_path = Path(image_location)

    csv_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    image_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataframe.to_csv(
        csv_path,
        index=False,
        float_format="%.4f",
    )

    styled_dataframe = (
        dataframe.style
        .hide(axis="index")
        .format(
            {
                column: "{:.4f}"
                for column in dataframe.columns
                if column != "n"
            },
            na_rep="NaN",
        )
    )

    dfi.export(
        styled_dataframe,
        image_path,
        table_conversion="matplotlib",
    )

def create_bias_dataframe(
    simulation_results,
    row_names=None,
) -> pd.DataFrame:
    """
    Create a purpose-specific bias table for the two methods used in
    the paper reproduction:

    1. Raw logistic treatment coefficient.
    2. Standardized logistic plug-in estimator.

    This is tied to the current simulation output structure and is
    not intended as a general-purpose analysis function.
    """
    if not simulation_results:
        raise ValueError(
            "simulation_results cannot be empty."
        )

    if row_names is None:
        row_names = [
            f"Simulation {index}"
            for index in range(
                1,
                len(simulation_results) + 1,
            )
        ]
    else:
        row_names = list(row_names)

    if len(row_names) != len(simulation_results):
        raise ValueError(
            f"Received {len(row_names)} row names for "
            f"{len(simulation_results)} simulation results."
        )

    rows = []

    for row_number, (
        row_name,
        simulation_result,
    ) in enumerate(
        zip(row_names, simulation_results),
        start=1,
    ):
        validate_simulation_result(
            simulation_result,
            row_number,
        )

        true_parameter = float(
            simulation_result["true_parameter"]
        )

        estimator_results = simulation_result[
            "estimator_results"
        ]

        # These are the treatment coefficients returned by
        # fitted_result.params[1].
        logit_estimates = np.asarray(
            estimator_results[
                LOGIT_ESTIMATOR_NAME
            ]["estimates"],
            dtype=float,
        ).ravel()

        plugin_estimates = np.asarray(
            estimator_results[
                PLUGIN_ESTIMATOR_NAME
            ]["estimates"],
            dtype=float,
        ).ravel()

        if logit_estimates.size != plugin_estimates.size:
            raise ValueError(
                f"Simulation row {row_number} contains "
                f"{logit_estimates.size} raw logit estimates but "
                f"{plugin_estimates.size} plug-in estimates."
            )

        logit_bias = bias(
            true_parameter,
            logit_estimates,
        )

        plugin_bias = bias(
            true_parameter,
            plugin_estimates,
        )

        rows.append(
            {
                "n": row_name,
                "true_parameter": true_parameter,
                "bias(logit)": float(logit_bias),
                "bias(logit_plugin)": float(plugin_bias),
            }
        )

    return pd.DataFrame(
        rows,
        columns=[
            "n",
            "true_parameter",
            "bias(logit)",
            "bias(logit_plugin)",
        ],
    )

def export_bias_table(
    dataframe: pd.DataFrame,
    csv_location: str | Path = "simulation_bias_results.csv",
    image_location: str | Path = "simulation_bias_results.png",
) -> None:
    """
    Export the purpose-specific bias table as CSV and PNG.
    """
    csv_path = Path(csv_location)
    image_path = Path(image_location)

    csv_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    image_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataframe.to_csv(
        csv_path,
        index=False,
        float_format="%.4f",
    )

    styled_dataframe = (
        dataframe.style
        .hide(axis="index")
        .format(
            {
                "true_parameter": "{:.4f}",
                "bias(logit)": "{:.4f}",
                "bias(logit_plugin)": "{:.4f}",
            },
            na_rep="NaN",
        )
    )

    dfi.export(
        styled_dataframe,
        image_path,
        table_conversion="matplotlib",
    )

def main() -> None:
    warnings.warn(
        "This script is purpose-built solely to reproduce the "
        "requested coefficient/plugin/truth and bias tables. "
        "It should not be used as a general simulation-analysis "
        "pipeline.",
        UserWarning,
        stacklevel=2,
    )

    simulation_results = load_simulation_data(
        PICKLE_LOCATION
    )

    results_dataframe = summarize_simulation_results(
        simulation_results,
        row_names=ROW_NAMES,
    )

    bias_dataframe = create_bias_dataframe(
        simulation_results,
        row_names=ROW_NAMES,
    )

    print("\nCoefficient, plug-in, and truth results:\n")
    print(results_dataframe.to_string(index=False))

    print("\nEstimator bias results:\n")
    print(bias_dataframe.to_string(index=False))

    export_reproduction_table(
        results_dataframe
    )

    export_bias_table(
        bias_dataframe
    )




if __name__ == "__main__":
    main()


