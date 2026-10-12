import numpy as np
def mean_and_sd(data):
    """
    Calculate the mean and standard deviation of each column.

    Returns
    -------
    list[tuple[float, float]]
        [(mean_1, sd_1), (mean_2, sd_2), ...]
    """
    data = np.asarray(data, dtype=float)

    if data.size == 0:
        raise ValueError("data cannot be empty.")

    col_means = np.atleast_1d(np.mean(data, axis=0))
    col_sds = np.atleast_1d(np.std(data, axis=0))

    return [
        (float(mean), float(sd))
        for mean, sd in zip(col_means, col_sds)
    ]

def bias(true_param, estimated_param):
    return np.mean(estimated_param - true_param, axis=0)