from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent

RESULTS_DIR = PROJECT_ROOT / "results"
RAW_RESULTS_DIR = RESULTS_DIR / "raw"
TABLES_DIR = RESULTS_DIR / "tables"
FIGURES_DIR = RESULTS_DIR / "figures"

PICKLE_LOCATION = (
    RAW_RESULTS_DIR / "simulation_data.pkl"
)

RESULTS_CSV_LOCATION = (
    TABLES_DIR / "simulation_results.csv"
)

BIAS_CSV_LOCATION = (
    TABLES_DIR / "simulation_bias_results.csv"
)

RESULTS_IMAGE_LOCATION = (
    TABLES_DIR / "simulation_results.png"
)

BIAS_IMAGE_LOCATION = (
    TABLES_DIR / "simulation_bias_results.png"
)

VIOLIN_IMAGE_LOCATION = (
    FIGURES_DIR / "simulation_violin_plots.pdf"
)

SCATTER_IMAGE_LOCATION = (
    FIGURES_DIR
    / "differential_log_odds_error_scatterplots.pdf"
)

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