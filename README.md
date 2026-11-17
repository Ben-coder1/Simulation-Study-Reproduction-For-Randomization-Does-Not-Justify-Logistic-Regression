# Description

Simulation study based on the paper *Randomization Does Not Justify Logistic Regression*, where two estimators of differential log odds are compared under a Neyman nonparametric setup with randomization. The two estimators compared are the assignment coefficient in logistic regression and the plug-in estimator introduced in the paper that uses the logistic regression model .

# Setup

Create and activate a Python virtual environment. For example, run the following commands in Git Bash if on Windows:

```bash
python -m venv env
source env/Scripts/activate
```

Run:

```bash
pip install -r requirements.txt
```

or:

```bash
pip3 install -r requirements.txt
```

# Instructions to Run Complete Analysis

Type:

```bash
make all
```

## Estimated Runtime

20 seconds.

# Key Findings Summary

Using the assignment coefficient from logistic regression is biased when the data is generated from a Neyman setting with randomization, while the plug in estimator is not. The plug in estimator also has reduced variance, and the two estimators give highly related estimates.

# Miscellaneous Commands

| Command | Description |
|---|---|
| `make all` | Runs simulation, analysis, and figures. |
| `make clean` | Removes temporary files and results folder. |
| `make test` | Runs tests using pytest. |
| `make simulate` | Runs simulations. |
| `make simulate-optimized` | Runs the optimized simulation and records its completion. |
| `make all-optimized` | Runs the optimized simulation, then analyzes its results and creates figures in `results`. |
| `make analyze` | Runs analysis. |
| `make figures` | Creates the figures. |
| `make complexity` | Measures simulation runtime as the number of individuals changes. |
| `make benchmark` | Benchmarks the original, vectorized, and non-vectorized pipelines. |
| `make timing` | Writes a full timing grid and a faceted timing plot by population size. |
| `make memory` | Writes a full peak-Python-memory grid and a matching faceted memory plot. |
| `make base-profiler` | Profiles the original simulation. |
| `make optimized-profiler` | Profiles the optimized simulation. |
| `make stability-check` | Checks numerical warnings, convergence, and finite outputs across conditions. |
| `make optimized-validation` | Runs the optimized estimator regression tests. |
| `make performance-suite` | Runs the full performance analysis and optimized regression validation in sequence. |

# AI Use

AI was used in creating this project. I used both Gemini and Sol throughout.
