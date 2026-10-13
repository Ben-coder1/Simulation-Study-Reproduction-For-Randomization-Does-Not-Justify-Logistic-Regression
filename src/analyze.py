import pickle
from numbers import Real
from pathlib import Path

import dataframe_image as dfi
import numpy as np
import pandas as pd

from src.metrics import mean_and_sd
from configuration import PICKLE_LOCATION, ROW_NAMES



def analyze_simulation_data():
    with open(PICKLE_LOCATION, "rb") as file:
        simulation_raws = pickle.load(file)
    return summarize_simulation_raws(simulation_raws, row_names = ROW_NAMES)



def create_stats_dataframe(
    data,
    col_names_1=None,
    col_names_2=None,
    col_names_3=None,
    row_names=None,
    row_col_name="Row Name"
):
    """
    Convert simulation results into a pandas DataFrame.

    Each element of `data` creates exactly one DataFrame row.

    Expected structure for each element:
    [
        [(mean1, sd1), (mean2, sd2), ...],
        (plugin_mean, plugin_sd),
        true_value
    ]

    The second component may also be:
        [(plugin1_mean, plugin1_sd), (plugin2_mean, plugin2_sd), ...]

    The third component may be either one number or a sequence of numbers.

    Each estimate column is immediately followed by its SD column:
        beta_1, sd(beta_1), beta_2, sd(beta_2), ...

    Returns
    -------
    pandas.DataFrame
    """
    if data is None or len(data) == 0:
        raise ValueError("Data cannot be empty.")

    def is_numeric(value):
        return isinstance(value, Real) and not isinstance(value, bool)

    def is_estimate_pair(value):
        """Return True for a single (mean, SD) pair."""
        return (
            isinstance(value, (list, tuple, np.ndarray))
            and len(value) == 2
            and is_numeric(value[0])
            and is_numeric(value[1])
        )

    def normalize_estimates(value, component_name, row_number):
        """
        Convert either one (mean, SD) pair or a sequence of such pairs
        into a list of pairs.
        """
        if is_estimate_pair(value):
            estimates = [value]
        elif isinstance(value, (list, tuple, np.ndarray)):
            estimates = list(value)
        else:
            raise ValueError(
                f"{component_name} in row {row_number} must be a "
                "(mean, SD) pair or a sequence of (mean, SD) pairs."
            )

        for position, estimate in enumerate(estimates, start=1):
            if not is_estimate_pair(estimate):
                raise ValueError(
                    f"Item {position} of {component_name} in row "
                    f"{row_number} is not a valid (mean, SD) pair: "
                    f"{estimate!r}"
                )

        return estimates

    def normalize_true_values(value, row_number):
        """Convert one true value or a sequence of true values to a list."""
        if is_numeric(value):
            values = [value]
        elif isinstance(value, (list, tuple, np.ndarray)):
            values = list(value)
        else:
            raise ValueError(
                f"True parameters in row {row_number} must be a number "
                "or a sequence of numbers."
            )

        for position, true_value in enumerate(values, start=1):
            if not is_numeric(true_value):
                raise ValueError(
                    f"True parameter {position} in row {row_number} "
                    f"is not numeric: {true_value!r}"
                )

        return values


    normalized_data = []

    for row_index, row in enumerate(data):
        row_number = row_index + 1

        if not isinstance(row, (list, tuple)) or len(row) != 3:
            raise ValueError(
                f"Row {row_number} must contain exactly three components: "
                "model estimates, plug-in estimates, and true parameters."
            )

        model_estimates = normalize_estimates(
            row[0],
            component_name="Model estimates",
            row_number=row_number
        )

        plugin_estimates = normalize_estimates(
            row[1],
            component_name="Plug-in estimates",
            row_number=row_number
        )

        true_values = normalize_true_values(
            row[2],
            row_number=row_number
        )

        normalized_data.append(
            (model_estimates, plugin_estimates, true_values)
        )


    n_cols_1 = len(normalized_data[0][0])
    n_cols_2 = len(normalized_data[0][1])
    n_cols_3 = len(normalized_data[0][2])

    # Ensure all rows have matching dimensions.
    for row_index, (model, plugin, true) in enumerate(
        normalized_data,
        start=1
    ):
        if len(model) != n_cols_1:
            raise ValueError(
                f"Row {row_index} has {len(model)} model estimates; "
                f"expected {n_cols_1}."
            )

        if len(plugin) != n_cols_2:
            raise ValueError(
                f"Row {row_index} has {len(plugin)} plug-in estimates; "
                f"expected {n_cols_2}."
            )

        if len(true) != n_cols_3:
            raise ValueError(
                f"Row {row_index} has {len(true)} true parameters; "
                f"expected {n_cols_3}."
            )

    if col_names_1 is None:
        col_names_1 = [
            f"beta_{i + 1}"
            for i in range(n_cols_1)
        ]
    else:
        col_names_1 = list(col_names_1)

    if col_names_2 is None:
        if n_cols_2 == 1:
            col_names_2 = ["plug_in_estimate"]
        else:
            col_names_2 = [
                f"plug_in_estimate_{i + 1}"
                for i in range(n_cols_2)
            ]
    else:
        col_names_2 = list(col_names_2)

    if col_names_3 is None:
        if n_cols_3 == 1:
            col_names_3 = ["true_parameter"]
        else:
            col_names_3 = [
                f"true_parameter_{i + 1}"
                for i in range(n_cols_3)
            ]
    else:
        col_names_3 = list(col_names_3)

    if len(col_names_1) != n_cols_1:
        raise ValueError(
            f"Expected {n_cols_1} names in col_names_1, "
            f"got {len(col_names_1)}."
        )

    if len(col_names_2) != n_cols_2:
        raise ValueError(
            f"Expected {n_cols_2} names in col_names_2, "
            f"got {len(col_names_2)}."
        )

    if len(col_names_3) != n_cols_3:
        raise ValueError(
            f"Expected {n_cols_3} names in col_names_3, "
            f"got {len(col_names_3)}."
        )

    if row_names is None:
        row_names = [
            f"Row {i + 1}"
            for i in range(len(normalized_data))
        ]
    else:
        row_names = list(row_names)

    if len(row_names) != len(normalized_data):
        raise ValueError(
            f"Expected {len(normalized_data)} row names, "
            f"got {len(row_names)}."
        )

    headers = [row_col_name]

    for column_name in col_names_1:
        headers.extend([
            column_name,
            f"sd({column_name})"
        ])

    for column_name in col_names_2:
        headers.extend([
            column_name,
            f"sd({column_name})"
        ])

    headers.extend(col_names_3)


    dataframe_rows = []

    for row_name, (model, plugin, true) in zip(
        row_names,
        normalized_data
    ):
        output_row = [row_name]

        for mean, sd in model:
            output_row.extend([
                float(mean),
                float(sd)
            ])

        for mean, sd in plugin:
            output_row.extend([
                float(mean),
                float(sd)
            ])

        output_row.extend(float(value) for value in true)

        dataframe_rows.append(output_row)

    return pd.DataFrame(
        dataframe_rows,
        columns=headers
    )

def summarize_simulation_raws(
    simulation_raws,
    col_names_1=None,
    col_names_2=None,
    col_names_3=None,
    row_names=None,
):
    """
    Summarize raw simulation results.

    Each element of simulation_raws must have the form:

        (
            model_param_estimates,
            plug_in_estimates,
            true_parameter
        )

    A separate mean and SD is calculated for every column of
    model_param_estimates and plug_in_estimates.
    """
    if not simulation_raws:
        raise ValueError("simulation_raws cannot be empty.")

    summarized_results = []

    for row_number, raw_result in enumerate(simulation_raws, start=1):
        if len(raw_result) != 3:
            raise ValueError(
                f"Simulation row {row_number} must contain exactly "
                "three items."
            )

        (
            model_param_estimates,
            plug_in_estimates,
            true_parameter,
        ) = raw_result

        model_param_estimates = np.asarray(
            model_param_estimates,
            dtype=float,
        )

        plug_in_estimates = np.asarray(
            plug_in_estimates,
            dtype=float,
        )


        model_summary = mean_and_sd(model_param_estimates)



        plugin_summary = mean_and_sd(plug_in_estimates)


        summarized_results.append(
            (
                model_summary,
                plugin_summary,
                true_parameter,
            )
        )

    if row_names is None:
        row_names = [
            f"Simulation {i + 1}"
            for i in range(len(summarized_results))
        ]

    return create_stats_dataframe(
        data=summarized_results,
        col_names_1=col_names_1,
        col_names_2=col_names_2,
        col_names_3=col_names_3,
        row_names=row_names,
        row_col_name="n",
    )

if __name__ == "__main__":
    df = analyze_simulation_data()

    print(df)

    df.to_csv(
        "simulation_results.csv",
        index=False,
        float_format="%.4f",
    )


    styled_df = (
        df.style
        .hide(axis="index")
        .format(precision=4)
    )


    dfi.export(
        styled_df,
        "simulation_results.png",
        table_conversion="matplotlib",
    )