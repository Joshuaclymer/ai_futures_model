"""
Simulation parameters for the AI Futures Simulator.

SimulationParameters contains simulation config and nested parameter groups.
ModelParameters defines distributions for Monte Carlo sampling loaded from YAML.

Parameter definitions are in separate files for clarity:
- software_r_and_d_parameters.py: AI R&D model parameters
- compute_parameters.py: Compute growth dynamics
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
from parameters.compute_parameters import (
    ComputeParameters,
    ExogenousComputeTrends,
    SurvivalRateParameters,
    USComputeParameters,
    PRCComputeParameters,
)
from parameters.policy_parameters import PolicyParameters
from parameters.data_center_and_energy_parameters import (
    DataCenterAndEnergyParameters,
    PRCDataCenterAndEnergyParameters,
)
from parameters.black_project_parameters import (
    BlackProjectParameters,
    BlackProjectProperties,
)
from parameters.perceptions_parameters import (
    PerceptionsParameters,
    BlackProjectPerceptionsParameters,
)
from parameters.ai_researcher_headcount_parameters import (
    AIResearcherHeadcountParameters,
    USResearcherParameters,
    PRCResearcherParameters,
)


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
      - compute: Compute growth (exogenous trends, survival rates, US/PRC compute)
      - datacenter_and_energy: Datacenter capacity and energy consumption
      - policy: AI governance and slowdown policies
      - black_project: Covert compute infrastructure (optional)
      - perceptions: Detection/perception parameters (optional)

    Note: simulation_start_year must be a discrete year in the historical data (2012-2026).
    """
    settings: SimulationSettings
    software_r_and_d: SoftwareRAndDParameters
    compute: ComputeParameters
    datacenter_and_energy: DataCenterAndEnergyParameters
    policy: PolicyParameters
    ai_researcher_headcount: Optional[AIResearcherHeadcountParameters] = None
    black_project: Optional[BlackProjectParameters] = None
    perceptions: Optional[PerceptionsParameters] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert parameters to a dictionary for serialization."""
        result = {
            "settings": self.settings.__dict__,
            "software_r_and_d": self.software_r_and_d.__dict__,
            "compute": {
                "exogenous_trends": self.compute.exogenous_trends.__dict__,
                "survival_rate_parameters": self.compute.survival_rate_parameters.__dict__,
                "us_compute": self.compute.USComputeParameters.__dict__,
                "prc_compute": self.compute.PRCComputeParameters.__dict__,
            },
            "datacenter_and_energy": {
                "prc_energy_consumption": self.datacenter_and_energy.prc_energy_consumption.__dict__,
            },
            "policy": self.policy.__dict__,
        }
        # Add black project parameters if present
        if self.black_project:
            result["black_project"] = {
                "run_a_black_project": self.black_project.run_a_black_project,
                "black_project_start_year": self.black_project.black_project_start_year,
                "properties": self.black_project.black_project_properties.__dict__,
            }
        # Add perceptions parameters if present
        if self.perceptions:
            result["perceptions"] = {
                "update_perceptions": self.perceptions.update_perceptions,
                "black_project_perception_parameters": self.perceptions.black_project_perception_parameters.__dict__,
            }
        # Add AI researcher headcount parameters if present
        if self.ai_researcher_headcount:
            result["ai_researcher_headcount"] = {
                "us_researchers": self.ai_researcher_headcount.us_researchers.__dict__,
                "prc_researchers": self.ai_researcher_headcount.prc_researchers.__dict__,
                "initial_global_ai_researcher_headcount": self.ai_researcher_headcount.initial_global_ai_researcher_headcount,
                "annual_growth_rate_of_ai_researcher_headcount": self.ai_researcher_headcount.annual_growth_rate_of_ai_researcher_headcount,
            }
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

        compute:
          exogenous_trends:
            transistor_density_scaling_exponent: 1.49
          survival_rate_parameters:
            initial_annual_hazard_rate: 0.05
          us_compute:
            us_frontier_project_compute_tpp_h100e_in_2025: 120325.0
          prc_compute:
            total_prc_compute_tpp_h100e_in_2025: 100000.0

        policy:
          ai_slowdown_start_year: 2030.0

        black_project:
          run_a_black_project: true
          black_project_start_year: 2030.0
          properties:
            total_labor: 10000
            ...
    """

    # Nested parameter specifications
    settings: Dict[str, Any] = field(default_factory=dict)
    software_r_and_d: Dict[str, Any] = field(default_factory=dict)
    compute: Dict[str, Any] = field(default_factory=dict)
    datacenter_and_energy: Dict[str, Any] = field(default_factory=dict)
    policy: Dict[str, Any] = field(default_factory=dict)
    ai_researcher_headcount: Optional[Dict[str, Any]] = None
    black_project: Optional[Dict[str, Any]] = None
    perceptions: Optional[Dict[str, Any]] = None

    # Correlation matrix specification (optional)
    correlation_matrix: Optional[Dict[str, Any]] = None

    # Random seed
    seed: int = 42

    # Initial progress value
    initial_progress: float = 0.0

    # Internal random generator for sampling (initialized on first use)
    _rng: Optional[np.random.Generator] = None
    _sample_count: int = 0

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
            compute=config.get("compute", {}),
            datacenter_and_energy=config.get("datacenter_and_energy", {}),
            policy=config.get("policy", {}),
            ai_researcher_headcount=config.get("ai_researcher_headcount"),
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
            rng: NumPy random generator. If None, uses an internal generator
                 that's initialized from self.seed on first use and advances
                 with each call, ensuring different samples each time.

        Returns:
            SimulationParameters instance with sampled values.
        """
        if rng is None:
            # Use internal random generator that advances with each call
            # This ensures different samples when sample() is called repeatedly
            if self._rng is None:
                self._rng = np.random.default_rng(self.seed)
            rng = self._rng
            self._sample_count += 1

        # Sample settings
        sampled_settings = {}
        for param_name, dist_spec in self.settings.items():
            sampled_settings[param_name] = sample_from_distribution(dist_spec, rng, param_name)

        # Sample software_r_and_d parameters
        sampled_r_and_d = {}
        for param_name, dist_spec in self.software_r_and_d.items():
            sampled_r_and_d[param_name] = sample_from_distribution(dist_spec, rng, param_name)

        # Sample compute parameters (nested structure)
        sampled_compute = self._sample_nested_dict(self.compute, rng)

        # Sample datacenter_and_energy parameters (nested structure)
        sampled_dc_energy = self._sample_nested_dict(self.datacenter_and_energy, rng)

        # Sample policy parameters
        sampled_policy = self._sample_nested_dict(self.policy, rng)

        # Build parameter objects
        settings = SimulationSettings(**sampled_settings)
        software_r_and_d = SoftwareRAndDParameters(**sampled_r_and_d)

        # Build nested compute parameters
        compute = ComputeParameters(
            exogenous_trends=ExogenousComputeTrends(**sampled_compute.get("exogenous_trends", {})),
            survival_rate_parameters=SurvivalRateParameters(**sampled_compute.get("survival_rate_parameters", {})),
            USComputeParameters=USComputeParameters(**sampled_compute.get("us_compute", {})),
            PRCComputeParameters=PRCComputeParameters(**sampled_compute.get("prc_compute", {})),
        )

        # Build nested datacenter and energy parameters
        datacenter_and_energy = DataCenterAndEnergyParameters(
            prc_energy_consumption=PRCDataCenterAndEnergyParameters(**sampled_dc_energy.get("prc_energy_consumption", {})),
        )

        policy = PolicyParameters(**sampled_policy)

        # Build black project parameters if present
        black_project = None
        if self.black_project is not None:
            sampled_bp = self._sample_nested_dict(self.black_project, rng)
            black_project = BlackProjectParameters(
                run_a_black_project=sampled_bp.get("run_a_black_project", True),
                black_project_start_year=sampled_bp.get("black_project_start_year", 2030.0),
                black_project_properties=BlackProjectProperties(**sampled_bp.get("properties", {})),
            )

        # Build perceptions parameters if present
        perceptions = None
        if self.perceptions is not None:
            sampled_perceptions = self._sample_nested_dict(self.perceptions, rng)
            perceptions = PerceptionsParameters(
                update_perceptions=sampled_perceptions.get("update_perceptions", True),
                black_project_perception_parameters=BlackProjectPerceptionsParameters(
                    **sampled_perceptions.get("black_project_perception_parameters", {})
                ),
            )

        # Build AI researcher headcount parameters if present
        ai_researcher_headcount = None
        if self.ai_researcher_headcount is not None:
            sampled_researchers = self._sample_nested_dict(self.ai_researcher_headcount, rng)
            ai_researcher_headcount = AIResearcherHeadcountParameters(
                us_researchers=USResearcherParameters(**sampled_researchers.get("us_researchers", {})),
                prc_researchers=PRCResearcherParameters(**sampled_researchers.get("prc_researchers", {})),
                initial_global_ai_researcher_headcount=sampled_researchers.get(
                    "initial_global_ai_researcher_headcount", 90000.0
                ),
                annual_growth_rate_of_ai_researcher_headcount=sampled_researchers.get(
                    "annual_growth_rate_of_ai_researcher_headcount", 1.12
                ),
                proportion_of_global_ai_researchers_in_us=sampled_researchers.get(
                    "proportion_of_global_ai_researchers_in_us", 0.55
                ),
                proportion_of_global_ai_researchers_in_prc=sampled_researchers.get(
                    "proportion_of_global_ai_researchers_in_prc", 0.44
                ),
            )

        return SimulationParameters(
            settings=settings,
            software_r_and_d=software_r_and_d,
            compute=compute,
            datacenter_and_energy=datacenter_and_energy,
            policy=policy,
            ai_researcher_headcount=ai_researcher_headcount,
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
