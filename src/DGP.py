from abc import ABC, abstractmethod
from dataclasses import dataclass
from numbers import Real

import numpy as np
from tqdm.auto import tqdm


@dataclass(frozen=True)
class IndividualLevelConditionalOutcomes:
  n: int
  covariates: np.ndarray
  treatment_outcomes: np.ndarray
  control_outcomes: np.ndarray


class IndividualConditionalOutcomesDGP(ABC):

  @abstractmethod
  def generate_individuals(
      self,
      rng: np.random.Generator | int | None = None,
  ) -> IndividualLevelConditionalOutcomes:
    pass


class IndividualConditionalOutcomesUniformDGP(IndividualConditionalOutcomesDGP):

  def __init__(
      self,
      n: int,
      treatment_outcome_threshold: float,
      control_outcome_threshold: float,
  ):
    # Validate n
    if isinstance(n, bool) or not isinstance(n, int):
      raise TypeError(f"'n' must be an integer, got {type(n).__name__}.")
    if n <= 0:
      raise ValueError(f"'n' must be a positive integer (> 0), got {n}.")

    # Validate thresholds
    thresholds = {
        "treatment_outcome_threshold": treatment_outcome_threshold,
        "control_outcome_threshold": control_outcome_threshold,
    }
    for name, val in thresholds.items():
      if isinstance(val, bool) or not isinstance(val, (int, float, Real)):
        raise TypeError(
            f"'{name}' must be a real number, got {type(val).__name__}."
        )

    self.n = n
    self.treatment_outcome_threshold = float(treatment_outcome_threshold)
    self.control_outcome_threshold = float(control_outcome_threshold)

  def generate_individuals(
      self,
      rng: np.random.Generator | int | None = None,
  ) -> IndividualLevelConditionalOutcomes:
    rng = np.random.default_rng(rng)

    covariates = rng.uniform(0, 1, size=(self.n, 1))
    latents = rng.uniform(0, 1, size=self.n)
    covariates_1d = covariates.squeeze()

    treatment_outcomes = np.empty(self.n, dtype=int)
    control_outcomes = np.empty(self.n, dtype=int)

    # Generate binary outcomes with a progress bar over individuals.
    for i in tqdm(
        range(self.n),
        total=self.n,
        desc="Generating individuals",
        unit="individual",
    ):
      treatment_outcomes[i] = int(
          latents[i] + covariates_1d[i]
          > self.treatment_outcome_threshold
      )
      control_outcomes[i] = int(
          latents[i] > self.control_outcome_threshold
      )

    return IndividualLevelConditionalOutcomes(
        n=self.n,
        covariates=covariates,
        treatment_outcomes=treatment_outcomes,
        control_outcomes=control_outcomes,
    )


@dataclass(frozen=True)
class ExperimentResults:
  assignments: np.ndarray
  covariates: np.ndarray
  outcomes: np.ndarray


class ExperimentDGP:

  def __init__(
      self,
      n_exp: int,
      p_treatment: float,
      conditional_outcomes: IndividualLevelConditionalOutcomes | None = None,
  ):
    # Validate n_exp
    if isinstance(n_exp, bool) or not isinstance(n_exp, int):
      raise TypeError(f"'n_exp' must be an integer, got {type(n_exp).__name__}.")
    if n_exp <= 0:
      raise ValueError(f"'n_exp' must be a positive integer (> 0), got {n_exp}.")

    # Validate p_treatment
    if isinstance(p_treatment, bool) or not isinstance(
        p_treatment, (int, float, Real)
    ):
      raise TypeError(
          f"'p_treatment' must be a real number, got {type(p_treatment).__name__}."
      )
    if not (0.0 <= p_treatment <= 1.0):
      raise ValueError(
          f"'p_treatment' must be between 0.0 and 1.0, got {p_treatment}."
      )

    self.n_exp = n_exp
    self.p_treatment = float(p_treatment)
    self.conditional_outcomes = None
    self.n_ind = None

    if conditional_outcomes is not None:
      self.set_conditional_outcomes(conditional_outcomes)

  def set_conditional_outcomes(
      self, conditional_outcomes: IndividualLevelConditionalOutcomes
  ):
    if not isinstance(conditional_outcomes, IndividualLevelConditionalOutcomes):
      raise TypeError(
          "'conditional_outcomes' must be an instance of"
          " IndividualLevelConditionalOutcomes."
      )
    self.conditional_outcomes = conditional_outcomes
    self.n_ind = conditional_outcomes.n

  def generate_experiment(
      self,
      rng: np.random.Generator | int | None = None,
  ) -> ExperimentResults:
    if self.conditional_outcomes is None:
      raise ValueError("Conditional outcomes not set")

    rng = np.random.default_rng(rng)

    
    n_treated = int(round(self.n_ind * self.p_treatment))

    assignments = np.zeros(self.n_ind, dtype=int)
    assignments[rng.choice(self.n_ind, size=n_treated, replace=False)] = 1

    # Select outcomes based on 1D assignments
    outcomes = np.where(
        assignments == 1,
        self.conditional_outcomes.treatment_outcomes,
        self.conditional_outcomes.control_outcomes,
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
    if self.conditional_outcomes is None:
      raise ValueError("Conditional outcomes not set")

    base_rng = np.random.default_rng(rng)
    child_rngs = base_rng.spawn(self.n_exp)

    return [
        self.generate_experiment(child_rng)
        for child_rng in tqdm(
            child_rngs,
            total=self.n_exp,
            desc="Generating experiments",
            unit="experiment",
        )
    ]


INDIVIDUAL_DGPS = {
    "individual_uniform": IndividualConditionalOutcomesUniformDGP,
}

EXPERIMENT_DGPS = {
    "base": ExperimentDGP,
}