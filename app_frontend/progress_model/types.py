#!/usr/bin/env python3
"""
Basic data types and dataclasses for progress modeling.

Contains TimeSeriesData, AnchorConstraint, and InitialConditions.
Parameters is in a separate module due to its complexity and dependencies.
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict


@dataclass
class TimeSeriesData:
    """Input time series data"""
    time: np.ndarray  # Decimal years
    L_HUMAN: np.ndarray  # Human labor supply
    inference_compute: np.ndarray  # AI labor supply (human-equivalents)
    experiment_compute: np.ndarray  # Experiment compute budget
    training_compute_growth_rate: np.ndarray  # Training compute budget

    def __post_init__(self):
        """Precompute log-space arrays for efficient interpolation.

        Log-space interpolation is used for exponentially growing variables to
        prevent scalloping on log plots and handle exponential growth better.
        Precomputing the logs avoids redundant computation in the ODE solver inner loop.
        """
        # Precompute log arrays for variables that are always positive
        # Use np.where to handle any zeros safely
        self.log_L_HUMAN = np.where(
            self.L_HUMAN > 0,
            np.log(self.L_HUMAN),
            -np.inf
        )
        self.log_inference_compute = np.where(
            self.inference_compute > 0,
            np.log(self.inference_compute),
            -np.inf
        )
        self.log_experiment_compute = np.where(
            self.experiment_compute > 0,
            np.log(self.experiment_compute),
            -np.inf
        )

        # Flags to indicate if log interpolation is safe (all values positive)
        self.can_use_log_L_HUMAN = np.all(self.L_HUMAN > 0)
        self.can_use_log_inference_compute = np.all(self.inference_compute > 0)
        self.can_use_log_experiment_compute = np.all(self.experiment_compute > 0)


@dataclass
class AnchorConstraint:
    """Specifies a constraint for parameter estimation"""
    # Dict mapping variable names to values (can be partial)
    conditions: Dict[str, float]  # e.g., {"automation_fraction": 0.9, "inference_compute": 1e6}
    # Expected outcome
    target_variable: str  # e.g., "progress_rate"
    target_value: float   # e.g., 5.0
    weight: float = 1.0   # Weight in optimization


@dataclass
class InitialConditions:
    """Container for initial model conditions"""
    start_time: float
    initial_progress: float
    initial_automation: float
    L_HUMAN: float
    inference_compute: float
    experiment_compute: float
    training_compute_growth_rate: float
    coding_labor: float
    research_effort: float
    research_stock: float
