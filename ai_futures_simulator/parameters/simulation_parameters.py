"""
Simulation parameters for the AI Futures Simulator.

SimulationParameters contains simulation config and nested parameter groups.
ModelParameters defines distributions for Monte Carlo sampling loaded from YAML.

Parameter definitions are in separate files for clarity:
- software_r_and_d_parameters.py: AI R&D model parameters
- compute_growth_parameters.py: Compute growth dynamics
- policy_parameters.py: AI governance and slowdown policies
- black_project_parameters.py: Covert compute infrastructure
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

# Import parameter classes from dedicated files
from parameters.software_r_and_d_parameters import SoftwareRAndDParameters
from parameters.compute_parameters import ComputeParameters
from parameters.policy_parameters import PolicyParameters
from parameters.energy_consumption_parameters import EnergyConsumptionParameters
from parameters.black_project_parameters import (
    BlackProjectParameterSet,
    BlackProjectProperties,
    BlackFabParameters,
    BlackDatacenterParameters,
    DetectionParameters,
)
from parameters.perceptions_parameters import PerceptionsParameters


@dataclass
class SimulationSettings:
    """Settings for simulation execution."""
    simulation_start_year: int
    simulation_end_year: float
    n_eval_points: int
    # ODE solver settings (optional - defaults are set in ai_futures_simulator.py)
    ode_rtol: float = 1e-3  # Relative tolerance
    ode_atol: float = 1e-5  # Absolute tolerance
    ode_max_step: float = 1.0  # Maximum step size in years


# =============================================================================
# SIMULATION PARAMETERS
# =============================================================================

@dataclass
class SimulationParameters:
    """
    Top-level simulation configuration for a single simulation run.

    Contains:
    - Simulation settings (start year, end year, eval points)
    - Model parameters for each component:
      - software_r_and_d: AI R&D dynamics
      - compute_growth: Training compute growth
      - energy_consumption: Energy efficiency and consumption
      - policy: AI governance and slowdown policies
      - black_project: Covert compute infrastructure (optional)

    Note: simulation_start_year must be a discrete year in the historical data (2012-2026).
    """
    settings: SimulationSettings
    software_r_and_d: SoftwareRAndDParameters
    compute_growth: ComputeParameters
    energy_consumption: EnergyConsumptionParameters
    policy: PolicyParameters
    black_project: Optional[BlackProjectParameterSet] = None
    perceptions: Optional[PerceptionsParameters] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert parameters to a dictionary for serialization."""
        result = {
            "settings": self.settings.__dict__,
            "software_r_and_d": self.software_r_and_d.__dict__,
            "compute_growth": self.compute_growth.__dict__,
            "energy_consumption": self.energy_consumption.__dict__,
            "policy": self.policy.__dict__,
        }
        # Add black project parameters if present
        if self.black_project:
            result["black_project"] = {
                "properties": self.black_project.properties.__dict__,
                "fab": self.black_project.fab_params.__dict__,
                "datacenter": self.black_project.datacenter_params.__dict__,
                "detection": self.black_project.detection_params.__dict__,
            }
        # Add perceptions parameters if present
        if self.perceptions:
            result["perceptions"] = self.perceptions.__dict__
        return result

    @property
    def ai_slowdown_start_year(self) -> float:
        """Convenience accessor for the AI slowdown start year."""
        return self.policy.ai_slowdown_start_year


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
          us_frontier_project_compute_annual_multiplier:
            dist: normal
            ci80: [3.55, 4.47]
          slowdown_year: 2028.0

        policy:
          ai_slowdown_start_year: 2030.0

        black_project:
          properties:
            run_a_black_project: true
            ...
          fab:
            h100_sized_chips_per_wafer: 28.0
            ...
    """

    # Nested parameter specifications
    settings: Dict[str, Any] = field(default_factory=dict)
    software_r_and_d: Dict[str, Any] = field(default_factory=dict)
    compute_growth: Dict[str, Any] = field(default_factory=dict)
    energy_consumption: Dict[str, Any] = field(default_factory=dict)
    policy: Dict[str, Any] = field(default_factory=dict)
    black_project: Optional[Dict[str, Any]] = None
    perceptions: Optional[Dict[str, Any]] = None

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
            energy_consumption=config.get("energy_consumption", {}),
            policy=config.get("policy", {}),
            black_project=config.get("black_project"),
            perceptions=config.get("perceptions"),
            correlation_matrix=config.get("correlation_matrix"),
            seed=config.get("seed", 42),
            initial_progress=config.get("initial_progress", 0.0),
        )

    def _sample_nested_dict(
        self,
        config: Dict[str, Any],
        rng: np.random.Generator
    ) -> Dict[str, Any]:
        """Sample values from a nested dict, handling distributions at any level."""
        result = {}
        for key, value in config.items():
            if isinstance(value, dict):
                # Check if this dict is a distribution spec
                if "dist" in value:
                    result[key] = sample_from_distribution(value, rng, key)
                else:
                    # Recurse into nested dict
                    result[key] = self._sample_nested_dict(value, rng)
            elif isinstance(value, list):
                # Handle lists (e.g., localization curves) - pass through as-is
                # Convert to list of tuples if it's a list of lists (for localization)
                if value and isinstance(value[0], list):
                    result[key] = [tuple(item) for item in value]
                else:
                    result[key] = value
            else:
                result[key] = value
        return result

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

        # Sample energy_consumption parameters
        sampled_energy = {}
        for param_name, dist_spec in self.energy_consumption.items():
            sampled_energy[param_name] = sample_from_distribution(dist_spec, rng, param_name)

        # Sample policy parameters
        sampled_policy = self._sample_nested_dict(self.policy, rng)

        # Build parameter objects
        settings = SimulationSettings(**sampled_settings)
        software_r_and_d = SoftwareRAndDParameters(**sampled_r_and_d)
        compute_growth = ComputeParameters(**sampled_compute)
        energy_consumption = EnergyConsumptionParameters(**sampled_energy)
        policy = PolicyParameters(**sampled_policy)

        # Build black project parameters if present
        black_project = None
        if self.black_project is not None:
            sampled_bp = self._sample_nested_dict(self.black_project, rng)
            black_project = BlackProjectParameterSet(
                properties=BlackProjectProperties(**sampled_bp.get("properties", {})),
                fab_params=BlackFabParameters(**sampled_bp.get("fab", {})),
                datacenter_params=BlackDatacenterParameters(**sampled_bp.get("datacenter", {})),
                detection_params=DetectionParameters(**sampled_bp.get("detection", {})),
            )

        # Build perceptions parameters if present
        perceptions = None
        if self.perceptions is not None:
            sampled_perceptions = self._sample_nested_dict(self.perceptions, rng)
            perceptions = PerceptionsParameters(**sampled_perceptions)

        return SimulationParameters(
            settings=settings,
            software_r_and_d=software_r_and_d,
            compute_growth=compute_growth,
            energy_consumption=energy_consumption,
            policy=policy,
            black_project=black_project,
            perceptions=perceptions,
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
