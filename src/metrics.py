"""Summarize simulation estimates and their bias."""

import numpy as np
def mean_and_sd(values: np.ndarray) -> tuple[float, float]:
    """
    Calculate the mean and sample standard deviation, ignoring NaNs.

    Returns (NaN, NaN) if there are no finite estimates. The standard
    deviation is NaN if only one finite estimate is available.

    Keyword arguments:
        values -- Estimates to summarize; nonfinite values are excluded.

    Returns:
        The mean and sample standard deviation of finite values.
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

def bias(true_param, estimated_param):
    """Return the mean estimation error relative to the true parameter.

    Keyword arguments:
        true_param -- Target parameter to subtract from each estimate.
        estimated_param -- Array of estimated parameters.
    """
    return np.mean(estimated_param - true_param, axis=0)
