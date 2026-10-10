from dataclasses import dataclass

import numpy as np


@dataclass
class IndividualLevelConditionalOutcomes(frozen=True):
    n: int
    covariates: np.ndarray
    treatment_outcomes: np.ndarray
    control_outcomes: np.ndarray


class IndividualConditionalOutcomesDGP:
    def __init__(
        self,
        n: int,
        treatment_outcome_threshold: float,
        control_outcome_threshold: float,
    ):
        self.n = n
        self.treatment_outcome_threshold = treatment_outcome_threshold
        self.control_outcome_threshold = control_outcome_threshold

    def generate_individuals(
        self,
        rng: np.random.Generator | int | None = None,
    ) -> IndividualLevelConditionalOutcomes:
        rng = np.random.default_rng(rng)

        covariates = rng.uniform(0, 1, size=(self.n, 1))
        latents = rng.uniform(0, 1, size=(self.n, 1))

        treatment_outcomes = (
            latents + covariates > self.treatment_outcome_threshold
        ).astype(int)

        control_outcomes = (
            latents > self.control_outcome_threshold
        ).astype(int)

        return IndividualLevelConditionalOutcomes(
            n=self.n,
            covariates=covariates,
            treatment_outcomes=treatment_outcomes,
            control_outcomes=control_outcomes,
        )

@dataclass
class ExperimentResults(frozen=True):
    assignments: np.ndarray
    covariates: np.ndarray
    outcomes: np.ndarray

class ExperimentDGP:
    def __init__(
        self,
        n_exp: int,
        conditional_outcomes: IndividualLevelConditionalOutcomes,
        p_treatment: float,
    ):
        self.n_ind = conditional_outcomes.n
        self.n_exp = n_exp
        self.conditional_outcomes = conditional_outcomes
        self.p_treatment = p_treatment

    def generate_experiment(
        self,
        rng: np.random.Generator | int | None = None,
    ) -> ExperimentResults:
        rng = np.random.default_rng(rng)

        # Must be (n, 1) to match the potential-outcome arrays.
        assignments = rng.binomial(
            1,
            self.p_treatment,
            size=(self.n_ind, 1),
        )

        outcomes = (
            np.where(assignments == 1, self.conditional_outcomes.treatment_outcomes, self.conditional_outcomes.control_outcomes)
        )

        return ExperimentResults(
            assignments=assignments,
            covariates=self.conditional_outcomes.covariates,
            outcomes=outcomes,
        )

    def generate_experiments(
        self,
        rng: np.random.Generator | int | None = None,
    ) -> list[ExperimentResults]:
        base_rng = np.random.default_rng(rng)
        child_rngs = base_rng.spawn(self.n_exp)

        return [
            self.generate_experiment(child_rng)
            for child_rng in child_rngs
        ]

