"""
AI Software Developer initialization.

Initializes AISoftwareDeveloper entities for a given year using historical data.
"""

import csv
import math
import torch
from pathlib import Path

from classes.world.entities import AISoftwareDeveloper, ComputeAllocation
from classes.world.assets import Compute
from parameters.simulation_parameters import SimulationParameters
from initialize_world_history.initialize_ai_software_progress import initialize_ai_software_progress

# Load historical data from CSV
_DATA_PATH = Path(__file__).resolve().parent / "largest_ai_developer.csv"
_historical_data = {}
with open(_DATA_PATH, 'r') as f:
    for row in csv.DictReader(f):
        year = int(float(row['time']))
        _historical_data[year] = {
            'human_researchers': float(row['L_HUMAN']),
            'inference_compute': float(row['inference_compute']),
            'experiment_compute': float(row['experiment_compute']),
        }


def initialize_us_frontier_lab(params: SimulationParameters, year: int) -> AISoftwareDeveloper:
    """Initialize a US frontier AI lab for a given year using historical data."""
    data = _historical_data[year]
    inference_compute = data['inference_compute']
    experiment_compute = data['experiment_compute']
    human_researchers = int(data['human_researchers'])

    return AISoftwareDeveloper(
        id="us_frontier_lab",
        is_primarily_controlled_by_misaligned_AI=False,
        compute=Compute(
            total_tpp_h100e=inference_compute,
            total_energy_requirements_watts=inference_compute * 700.0,  # ~700W per H100e, TODO: refine
        ),
        compute_allocation=ComputeAllocation(
            fraction_for_ai_r_and_d_inference=0.1,
            fraction_for_ai_r_and_d_training=0.1,
            fraction_for_external_deployment=0.3,
            fraction_for_alignment_research=0.1,
            fraction_for_frontier_training=0.4,
        ),
        ai_software_progress=initialize_ai_software_progress(params, year),
        human_ai_capability_researchers=human_researchers,
        log_compute=torch.tensor(math.log(inference_compute)),
        log_researchers=torch.tensor(math.log(human_researchers)),
        experiment_compute=experiment_compute,
    )
