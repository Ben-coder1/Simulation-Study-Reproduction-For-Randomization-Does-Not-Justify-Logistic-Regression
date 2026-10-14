PICKLE_LOCATION = "simulation_data.pkl"
SEED = 100


individual_100_simulation = {
    "individual_dgp": "individual_uniform",
    "experiment_dgp": "base",
    "n_individuals": 100,
    "n_experiments": 1000,
    "treatment_threshold": 3 / 4,
    "control_threshold": 1 / 2,
    "p_treatment": 3 / 4,
    "param_estimators": [
        "logit",
        "logit_plugin",
    ],
    "true_parameter_finder": "differential_log_odds",
}


individual_500_simulation = {
    "individual_dgp": "individual_uniform",
    "experiment_dgp": "base",
    "n_individuals": 500,
    "n_experiments": 1000,
    "treatment_threshold": 3 / 4,
    "control_threshold": 1 / 2,
    "p_treatment": 3 / 4,
    "param_estimators": [
        "logit",
        "logit_plugin",
    ],
    "true_parameter_finder": "differential_log_odds",
}


individual_1000_simulation = {
    "individual_dgp": "individual_uniform",
    "experiment_dgp": "base",
    "n_individuals": 1000,
    "n_experiments": 1000,
    "treatment_threshold": 3 / 4,
    "control_threshold": 1 / 2,
    "p_treatment": 3 / 4,
    "param_estimators": [
        "logit",
        "logit_plugin",
    ],
    "true_parameter_finder": "differential_log_odds",
}


individual_5000_simulation = {
    "individual_dgp": "individual_uniform",
    "experiment_dgp": "base",
    "n_individuals": 5000,
    "n_experiments": 1000,
    "treatment_threshold": 3 / 4,
    "control_threshold": 1 / 2,
    "p_treatment": 3 / 4,
    "param_estimators": [
        "logit",
        "logit_plugin",
    ],
    "true_parameter_finder": "differential_log_odds",
}


SIMULATIONS = [
    individual_100_simulation,
    individual_500_simulation,
    individual_1000_simulation,
    individual_5000_simulation,
]

ROW_NAMES = [
    "n=100",
    "n=500",
    "n=1000",
    "n=5000",
]