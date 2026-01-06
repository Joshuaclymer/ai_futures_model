"""
AI Software Developer initialization.

Initializes AISoftwareDeveloper entities for a given year using historical data.
This uses data from the internal CSV which is part of ai_futures_simulator.
"""

import csv
import math
from pathlib import Path

from classes.world.entities import AISoftwareDeveloper, ComputeAllocation
from classes.world.assets import Compute
from parameters.classes import SimulationParameters
from initialize_world_history.initialize_ai_software_progress import initialize_ai_software_progress

# Load historical data from CSV (internal to ai_futures_simulator)
_DATA_PATH = Path(__file__).resolve().parent / "largest_ai_developer.csv"
_historical_data = {}
with open(_DATA_PATH, 'r') as f:
    for row in csv.DictReader(f):
        year = int(float(row['time']))
        _historical_data[year] = {
            'human_researchers': float(row['L_HUMAN']),
            'inference_compute': float(row['inference_compute']),
            'experiment_compute': float(row['experiment_compute']),
            'training_compute_growth_rate': float(row['training_compute_growth_rate']),
        }


def initialize_us_frontier_lab(params: SimulationParameters, year: int) -> AISoftwareDeveloper:
    """Initialize a US frontier AI lab for a given year using historical data."""
    data = _historical_data[year]
    inference_compute = data['inference_compute']
    experiment_compute = data['experiment_compute']
    human_researchers = float(data['human_researchers'])
    training_compute_growth_rate = data['training_compute_growth_rate']

    # Get compute allocations from parameters
    allocations = params.compute.compute_allocations

    # Total compute is the sum of inference and experiment compute
    total_compute = inference_compute + experiment_compute

    # Create the compute object for operating compute
    compute = Compute(
        tpp_h100e_including_attrition=total_compute,
        functional_tpp_h100e=total_compute,  # Initially all compute is functional
        watts_per_h100e=700.0,  # ~700W per H100e
        average_functional_chip_age_years=0.0,  # New chips
    )

    # Only initialize ai_software_progress if updates are enabled
    ai_software_progress = None
    if params.software_r_and_d and params.software_r_and_d.update_software_progress:
        ai_software_progress = initialize_ai_software_progress(params, year)

    developer = AISoftwareDeveloper(
        id="us_frontier_lab",
        operating_compute=[compute],  # List of Compute objects
        compute_allocation=ComputeAllocation(
            fraction_for_ai_r_and_d_inference=allocations.fraction_for_ai_r_and_d_inference,
            fraction_for_ai_r_and_d_training=allocations.fraction_for_ai_r_and_d_training,
            fraction_for_external_deployment=allocations.fraction_for_external_deployment,
            fraction_for_alignment_research=allocations.fraction_for_alignment_research,
            fraction_for_frontier_training=allocations.fraction_for_frontier_training,
        ),
        human_ai_capability_researchers=human_researchers,
        ai_software_progress=ai_software_progress,
        training_compute_growth_rate=training_compute_growth_rate,
    )

    # Set metric attributes (init=False fields, set after construction)
    developer._set_frozen_field('ai_r_and_d_inference_compute_tpp_h100e', total_compute * allocations.fraction_for_ai_r_and_d_inference)
    developer._set_frozen_field('ai_r_and_d_training_compute_tpp_h100e', total_compute * allocations.fraction_for_ai_r_and_d_training)
    developer._set_frozen_field('external_deployment_compute_tpp_h100e', total_compute * allocations.fraction_for_external_deployment)
    developer._set_frozen_field('alignment_research_compute_tpp_h100e', total_compute * allocations.fraction_for_alignment_research)
    developer._set_frozen_field('frontier_training_compute_tpp_h100e', total_compute * allocations.fraction_for_frontier_training)

    return developer
