"""
AI Software Progress initialization.

Initializes AISoftwareProgress state for a given year.

Uses calibration to get proper initial research_stock value so the
ODE integration doesn't fail with zero initial conditions.
"""

import torch

from classes.world.software_progress import AISoftwareProgress
from parameters.simulation_parameters import SimulationParameters
from parameters.calibrate import calibrate_from_params


def initialize_ai_software_progress(
    params: SimulationParameters,
    year: int,
    initial_progress: float = 0.0,
) -> AISoftwareProgress:
    """
    Initialize AI software progress state for a given year.

    Progress starts at initial_progress (default 0.0).
    Research stock is obtained from calibration which runs progress_model
    from the simulation start year.

    Args:
        params: SimulationParameters containing model configuration
        year: The integer year to initialize for (e.g., 2024, 2026)
        initial_progress: Starting progress value (default 0.0)

    Returns:
        Initialized AISoftwareProgress instance
    """
    # Get calibrated initial research_stock
    # Pass the year as start_year so calibration runs from the correct starting point
    start_year = float(params.settings.simulation_start_year)
    calibrated = calibrate_from_params(params.software_r_and_d, start_year=start_year)

    # Use the initial_research_stock from calibration
    # This is computed by progress_model's calculate_initial_research_stock()
    # based on RS(0) = (RS'(0))^2 / RS''(0)
    initial_research_stock = calibrated.initial_research_stock

    return AISoftwareProgress(
        # State variables
        progress=torch.tensor(initial_progress),
        research_stock=torch.tensor(initial_research_stock),
        # Metrics (initial values, will be recomputed by updaters)
        ai_coding_labor_multiplier=torch.tensor(1.0),
        ai_sw_progress_mult_ref_present_day=torch.tensor(1.0),
        progress_rate=torch.tensor(0.0),
        software_progress_rate=torch.tensor(0.0),
        research_effort=torch.tensor(0.0),
        automation_fraction=torch.tensor(0.0),
        coding_labor=torch.tensor(0.0),
        serial_coding_labor=torch.tensor(0.0),
        ai_research_taste=torch.tensor(0.0),
        ai_research_taste_sd=torch.tensor(0.0),
        aggregate_research_taste=torch.tensor(1.0),
        horizon_length=None,
    )
