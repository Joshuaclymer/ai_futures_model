"""
Simulation parameters for the AI Futures Simulator.

SimulationParameters contains simulation config and nested parameter groups.
ModelParameters defines distributions for Monte Carlo sampling loaded from YAML.
"""

import numpy as np
from dataclasses import dataclass, field, fields
from typing import Optional, Dict, Any, Union, List
from pathlib import Path
import yaml

from parameters.sample_from_distribution import (
    sample_from_distribution,
    sample_from_distribution_with_quantile,
)


# =============================================================================
# PARAMETER GROUPS
# =============================================================================

@dataclass
class ComputeGrowthParameters:
    """Parameters for training compute growth dynamics."""
    constant_training_compute_growth_rate: float  # OOMs/year before slowdown
    slowdown_year: float
    post_slowdown_training_compute_growth_rate: float  # OOMs/year after slowdown


@dataclass
class SoftwareRAndDParameters:
    """
    All parameters for AI Software R&D / takeoff model.

    Parameters are organized into logical groups matching the model structure.
    """

    # =========================================================================
    # MODE FLAGS
    # =========================================================================
    human_only: bool

    # =========================================================================
    # PRODUCTION FUNCTION PARAMETERS (CES)
    # =========================================================================
    rho_coding_labor: float
    coding_labor_normalization: float

    # Experiment capacity CES
    direct_input_exp_cap_ces_params: bool
    rho_experiment_capacity: float
    alpha_experiment_capacity: float
    experiment_compute_exponent: float

    # Experiment capacity asymptotes
    inf_labor_asymptote: float
    inf_compute_asymptote: float
    labor_anchor_exp_cap: float
    compute_anchor_exp_cap: Optional[float]
    inv_compute_anchor_exp_cap: float

    # Parallel penalty
    parallel_penalty: float

    # =========================================================================
    # SOFTWARE PROGRESS PARAMETERS
    # =========================================================================
    r_software: float
    software_progress_rate_at_reference_year: float

    # =========================================================================
    # AUTOMATION SCHEDULE PARAMETERS
    # =========================================================================
    automation_fraction_at_coding_automation_anchor: float
    automation_anchors: Optional[Dict[float, float]]
    automation_interp_type: str
    automation_logistic_asymptote: float
    swe_multiplier_at_present_day: float

    # Coding labor mode
    coding_labor_mode: str
    coding_automation_efficiency_slope: float
    optimal_ces_eta_init: float
    optimal_ces_grid_size: int
    optimal_ces_frontier_tail_eps: float
    optimal_ces_frontier_cap: float
    max_serial_coding_labor_multiplier: float

    # =========================================================================
    # AI RESEARCH TASTE PARAMETERS
    # =========================================================================
    ai_research_taste_at_coding_automation_anchor_sd: float
    ai_research_taste_slope: float
    taste_schedule_type: str
    median_to_top_taste_multiplier: float
    top_percentile: float
    taste_limit: float
    taste_limit_smoothing: float

    # =========================================================================
    # HORIZON / MILESTONE PARAMETERS
    # =========================================================================
    progress_at_aa: Optional[float]
    ac_time_horizon_minutes: float
    pre_gap_ac_time_horizon: float
    horizon_extrapolation_type: str

    # Manual horizon fitting
    present_day: float
    present_horizon: float
    present_doubling_time: float
    doubling_difficulty_growth_factor: float

    # Milestone multipliers
    strat_ai_m2b: float
    ted_ai_m2b: float

    # =========================================================================
    # GAP MODE PARAMETERS
    # =========================================================================
    include_gap: Union[str, bool]
    gap_years: float


@dataclass
class SimulationSettings:
    """Settings for simulation execution."""
    simulation_start_year: int
    simulation_end_year: float
    n_eval_points: int


# =============================================================================
# SIMULATION PARAMETERS
# =============================================================================

@dataclass
class SimulationParameters:
    """
    Top-level simulation configuration for a single simulation run.

    Contains:
    - Simulation settings (start year, end year, eval points)
    - Model parameters (SoftwareRAndDParameters, ComputeGrowthParameters)

    Note: simulation_start_year must be a discrete year in the historical data (2012-2026).
    """
    settings: SimulationSettings
    software_r_and_d: SoftwareRAndDParameters
    compute_growth: ComputeGrowthParameters

    def to_dict(self) -> Dict[str, Any]:
        """Convert parameters to a dictionary for serialization."""
        return {
            "settings": self.settings.__dict__,
            "software_r_and_d": self.software_r_and_d.__dict__,
            "compute_growth": self.compute_growth.__dict__,
        }


# =============================================================================
# MODEL PARAMETERS (Distribution-based for Monte Carlo)
# =============================================================================



@dataclass
class ModelParameters:
    """
    Parameter specifications for simulation - supports both point estimates and distributions.

    Loaded from YAML config files. Supports nested structure matching parameter groups.

    Example YAML format:
        seed: 42

        settings:
          simulation_start_year: 2026
          simulation_end_year: 2040.0
          n_eval_points: 100

        software_r_and_d:
          r_software: 2.4
          rho_coding_labor:
            dist: choice
            values: [-5, -2, -1]
            p: [0.25, 0.5, 0.25]

        compute_growth:
          constant_training_compute_growth_rate:
            dist: normal
            ci80: [0.55, 0.65]
          slowdown_year: 2028.0
    """

    # Nested parameter specifications
    settings: Dict[str, Any] = field(default_factory=dict)
    software_r_and_d: Dict[str, Any] = field(default_factory=dict)
    compute_growth: Dict[str, Any] = field(default_factory=dict)

    # Correlation matrix specification (optional)
    correlation_matrix: Optional[Dict[str, Any]] = None

    # Random seed
    seed: int = 42

    # Initial progress value
    initial_progress: float = 0.0

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "ModelParameters":
        """Load model parameters from a YAML config file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path, "r") as f:
            config = yaml.safe_load(f)

        return cls(
            settings=config.get("settings", {}),
            software_r_and_d=config.get("software_r_and_d", {}),
            compute_growth=config.get("compute_growth", {}),
            correlation_matrix=config.get("correlation_matrix"),
            seed=config.get("seed", 42),
            initial_progress=config.get("initial_progress", 0.0),
        )

    def sample(self, rng: Optional[np.random.Generator] = None) -> SimulationParameters:
        """
        Sample a set of simulation parameters from the distributions.

        Args:
            rng: NumPy random generator. If None, creates one from self.seed.

        Returns:
            SimulationParameters instance with sampled values.
        """
        if rng is None:
            rng = np.random.default_rng(self.seed)

        # Sample settings
        sampled_settings = {}
        for param_name, dist_spec in self.settings.items():
            sampled_settings[param_name] = sample_from_distribution(dist_spec, rng, param_name)

        # Sample software_r_and_d parameters
        sampled_r_and_d = {}
        for param_name, dist_spec in self.software_r_and_d.items():
            sampled_r_and_d[param_name] = sample_from_distribution(dist_spec, rng, param_name)

        # Sample compute_growth parameters
        sampled_compute = {}
        for param_name, dist_spec in self.compute_growth.items():
            sampled_compute[param_name] = sample_from_distribution(dist_spec, rng, param_name)

        # Build parameter objects
        settings = SimulationSettings(**sampled_settings)
        software_r_and_d = SoftwareRAndDParameters(**sampled_r_and_d)
        compute_growth = ComputeGrowthParameters(**sampled_compute)

        return SimulationParameters(
            settings=settings,
            software_r_and_d=software_r_and_d,
            compute_growth=compute_growth,
        )

    def sample_many(
        self,
        num_samples: int,
        rng: Optional[np.random.Generator] = None
    ) -> List[SimulationParameters]:
        """
        Sample multiple parameter sets for Monte Carlo simulation.

        Args:
            num_samples: Number of parameter sets to sample
            rng: NumPy random generator

        Returns:
            List of SimulationParameters instances
        """
        if rng is None:
            rng = np.random.default_rng(self.seed)

        return [self.sample(rng) for _ in range(num_samples)]
