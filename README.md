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

Using the assignment coefficient from logistic regression is biased when the data is generated from a newman setting with randomization, while the plug in estimator is not. The plug in estimator also has reduced variance, and the two estimators give highly related estimates.

# Miscellaneous Commands

| Command | Description |
|---|---|
| `make clean` | Removes temporary files and results folder. |
| `make test` | Runs tests using pytest. |
| `make simulate` | Runs simulations. |
| `make analyze` | Runs analysis. |
| `make figures` | Creates the figures. |