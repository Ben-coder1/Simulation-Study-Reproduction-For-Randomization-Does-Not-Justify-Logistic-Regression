from abc import ABC, abstractmethod
from dataclasses import dataclass
from numbers import Real

import numpy as np
from tqdm.auto import tqdm


@dataclass(frozen=True)
class IndividualLevelConditionalOutcomes:
  """Container for individual-level conditional outcomes.

  Attributes:
    n: Number of individuals.
    covariates: Array of covariate values for each individual.
    treatment_outcomes: Array of treatment outcome values for each individual.
    control_outcomes: Array of control outcome values for each individual.
  """
  n: int
  covariates: np.ndarray
  treatment_outcomes: np.ndarray
  control_outcomes: np.ndarray


class IndividualConditionalOutcomesDGP(ABC):
  """Abstract base class for generating individual-level conditional outcomes.

  Subclasses must implement the generate_individuals method to produce
  an instance of IndividualLevelConditionalOutcomes.
  """
  @abstractmethod
  def generate_individuals(
      self,
      rng: np.random.Generator | int | None = None,
  ) -> IndividualLevelConditionalOutcomes:
    pass


class IndividualConditionalOutcomesUniformDGP(IndividualConditionalOutcomesDGP):
  """DGP for generating individual-level conditional outcomes using uniform(0,1) distributions for latent and covariate values, and using thresholds to determine treatment/control success.
  
  Attributes:
    n: Number of individuals to generate.
    treatment_outcome_threshold: Threshold for determining treatment success.
    control_outcome_threshold: Threshold for determining control success.
  """

  def __init__(
      self,
      n: int,
      treatment_outcome_threshold: float,
      control_outcome_threshold: float,
  ):
    """Initialize the DGP with the specified number of individuals and outcome thresholds.

    Keyword arguments:
      n -- Number of individuals to generate.
      treatment_outcome_threshold -- Threshold for determining treatment success.
      control_outcome_threshold -- Threshold for determining control success.

    Raises:
      TypeError: If n is not an integer. If any of the thresholds are not real numbers.
      ValueError: if n is a non-positive integer.
    """
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
    """
    Generate individual-level conditional outcomes.

    Keyword arguments:
      rng -- Random number generator or seed for reproducibility.

    Returns an instance of IndividualLevelConditionalOutcomes containing the generated data.
    """
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
  """Container for the results of a causal inference experiment.

  Attributes:
    assignments: Array indicating treatment assignment for each individual.
    covariates: Array of covariate values for each individual.
    outcomes: Array of observed outcomes for each individual.
  """
  assignments: np.ndarray
  covariates: np.ndarray
  outcomes: np.ndarray


class ExperimentDGP:
  """Data-generating process for a causal inference experiment created by assigning individuals to treatment and control groups.

  Attributes:
    n_exp: Number of experiments to generate.
    p_treatment: Percentage of individuals assigned to the treatment group.
    conditional_outcomes: Instance of IndividualLevelConditionalOutcomes containing the conditional outcome data for individuals.
    n_ind: Number of individuals in the experiment.
  """
  def __init__(
      self,
      n_exp: int,
      p_treatment: float,
      conditional_outcomes: IndividualLevelConditionalOutcomes | None = None,
  ):
    """
    Initialize the ExperimentDGP with the specified number of experiments, treatment percentage, and optional conditional outcomes.

    Keyword arguments:
      n_exp -- Number of experiments to generate.
      p_treatment -- Percentage of individuals assigned to the treatment group.
      conditional_outcomes -- Instance of IndividualLevelConditionalOutcomes containing the conditional outcome data for individuals.

    Raises:
      TypeError: If n_exp is not an integer or p_treatment is not a real number.
      ValueError: If n_exp is not positive or p_treatment is not between 0.0 and 1.0.
    """
    if isinstance(n_exp, bool) or not isinstance(n_exp, int):
      raise TypeError(f"'n_exp' must be an integer, got {type(n_exp).__name__}.")
    if n_exp <= 0:
      raise ValueError(f"'n_exp' must be a positive integer (> 0), got {n_exp}.")

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
    """
    Set the conditional outcomes for the experiment.

    Keyword arguments:
      conditional_outcomes -- Instance of IndividualLevelConditionalOutcomes containing the conditional outcome data for individuals.

    Raises:
      TypeError: If conditional_outcomes is not an instance of IndividualLevelConditionalOutcomes.
    """
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
    """
    Generate a single experiment by assigning individuals to treatment and control groups and observing outcomes.

    Keyword arguments:
      rng -- Random number generator or seed for reproducibility.

    Returns:
      An instance of ExperimentResults containing the assignments, covariates, and outcomes.
    """
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
    """
    Generate multiple experiments by repeatedly calling generate_experiment.

    Keyword arguments:
      rng -- Random number generator or seed for reproducibility.

    Returns:
      A list of ExperimentResults instances.
    """
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