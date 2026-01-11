"""
Model parameters with factory/sampling methods for Monte Carlo simulations.

ModelParameters defines distributions for Monte Carlo sampling loaded from YAML.
Use ModelParameters.from_yaml() to load and .sample() to generate sampled parameters.

After sampling, parameters are calibrated (e.g., computing r_software, automation anchors)
before being returned. This ensures world updaters receive fully-determined parameters.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Union, List
from pathlib import Path
import yaml

from parameters.classes import SimulationParameters
from parameters.calibrate import calibrate_from_params


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
          ...
    """

    # All simulation parameters (nested structure)
    params: SimulationParameters = field(default_factory=SimulationParameters)

    # Correlation matrix specification (optional)
    correlation_matrix: Optional[Dict[str, Any]] = None

    # Random seed (from YAML)
    seed: Optional[int] = None

    # Initial progress value (from YAML)
    initial_progress: Optional[float] = None

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
            params=SimulationParameters.from_dict(config),
            correlation_matrix=config.get("correlation_matrix"),
            seed=config["seed"],
            initial_progress=config["initial_progress"],
        )

    def sample(self, rng: Optional[np.random.Generator] = None) -> SimulationParameters:
        """
        Sample a set of simulation parameters from the distributions.

        After sampling, runs calibration to compute derived parameters
        (r_software, automation anchors, etc.) so world updaters receive
        fully-determined parameters.

        Args:
            rng: NumPy random generator. If None, uses an internal generator
                 that's initialized from self.seed on first use and advances
                 with each call, ensuring different samples each time.

        Returns:
            SimulationParameters instance with sampled and calibrated values.
        """
        if rng is None:
            if self._rng is None:
                self._rng = np.random.default_rng(self.seed)
            rng = self._rng
            self._sample_count += 1

        params = self.params.sample(rng)

        # Run calibration to compute derived parameters
        if params.software_r_and_d is not None:
            start_year = float(params.settings.simulation_start_year) if params.settings else 2026.0
            params.software_r_and_d_calibrated = calibrate_from_params(
                params.software_r_and_d, start_year=start_year
            )

        return params

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

    def sample_modal(self) -> SimulationParameters:
        """
        Create parameters using the modal (most likely) values for all distributions.

        This produces a single "most likely" trajectory rather than a random sample.
        Useful for generating a baseline or representative simulation.

        After getting modal values, runs calibration to compute derived parameters.

        Returns:
            SimulationParameters instance with modal and calibrated values.
        """
        params = self.params.get_modal()

        # Run calibration to compute derived parameters
        if params.software_r_and_d is not None:
            start_year = float(params.settings.simulation_start_year) if params.settings else 2026.0
            params.software_r_and_d_calibrated = calibrate_from_params(
                params.software_r_and_d, start_year=start_year
            )

        return params
