PICKLE_LOCATION = "simulation_data.pkl"
SEED = 100




individual_100_simulation = {
    "individualDGP": "individual_uniform",
    "ExperimentDGP": "base",
    "nIndividuals": 100,
    "nExperiments": 1000,
    "treatment_threshold": 3/4,
    "control_threshold": 1/2,
    "pTreatment": 3/4,
    "param_estimator": "logit",
    "true_parameter_finder": "differential_log_odds"
}


individual_1000_simulation = {
    "individualDGP": "individual_uniform",
    "ExperimentDGP": "base",
    "nIndividuals": 1000,
    "nExperiments": 1000,
    "treatment_threshold": 3/4,
    "control_threshold": 1/2,
    "pTreatment": 3/4,
    "param_estimator": "logit",
    "true_parameter_finder": "differential_log_odds"
}

individual_500_simulation = {
    "individualDGP": "individual_uniform",
    "ExperimentDGP": "base",
    "nIndividuals": 500,
    "nExperiments": 1000,
    "treatment_threshold": 3/4,
    "control_threshold": 1/2,
    "pTreatment": 3/4,
    "param_estimator": "logit",
    "true_parameter_finder": "differential_log_odds"
}

individual_5000_simulation = {
    "individualDGP": "individual_uniform",
    "ExperimentDGP": "base",
    "nIndividuals": 5000,
    "nExperiments": 1000,
    "treatment_threshold": 3/4,
    "control_threshold": 1/2,
    "pTreatment": 3/4,
    "param_estimator": "logit",
    "true_parameter_finder": "differential_log_odds"
}



ROW_NAMES = ["n=100", "n=1000", "n=500", "n=5000"]
SIMULATIONS = [individual_100_simulation, individual_1000_simulation, individual_500_simulation, individual_5000_simulation]