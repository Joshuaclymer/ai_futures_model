"""
AI Software Progress classes.

These classes track the state and metrics related to AI software R&D progress.
All values must be explicitly set during world initialization - no defaults.
"""

import torch
from torch import Tensor
from dataclasses import dataclass, field
from typing import Optional

from classes.world.tensor_dataclass import TensorDataclass


@dataclass
class AISoftwareProgress(TensorDataclass):
    """
    Tracks AI software progress state and metrics for an AI project.

    State fields have metadata={'is_state': True} and are integrated by the ODE solver.
    All other fields are metrics (derived from state, recomputed after integration).
    """

    # === State variables (integrated by ODE solver) ===
    progress: Tensor = field(metadata={'is_state': True})
    research_stock: Tensor = field(metadata={'is_state': True})

    # === Metrics (derived from state, recomputed after integration) ===
    ai_coding_labor_multiplier: Tensor
    ai_sw_progress_mult_ref_present_day: Tensor
    progress_rate: Tensor
    software_progress_rate: Tensor
    research_effort: Tensor
    automation_fraction: Tensor
    coding_labor: Tensor
    serial_coding_labor: Tensor
    ai_research_taste: Tensor
    ai_research_taste_sd: Tensor
    aggregate_research_taste: Tensor
    horizon_length: Optional[Tensor] = None

    # === Input time series (interpolated from CSV data) ===
    human_labor: Optional[Tensor] = None
    inference_compute: Optional[Tensor] = None
    experiment_compute: Optional[Tensor] = None

    # === Additional computed metrics ===
    experiment_capacity: Optional[Tensor] = None  # research_effort / aggregate_research_taste
    software_efficiency: Optional[Tensor] = None  # progress - initial_progress - training_compute
    serial_coding_labor_multiplier: Optional[Tensor] = None  # serial_coding_labor / human_only_serial_coding_labor
    training_compute: Optional[Tensor] = None  # Cumulative training compute contribution


@dataclass
class AISoftwareCapabilityCap:
    """Defines capability caps that can be imposed by policy."""
    cap_of_ai_sw_progress_mult_ref_present_day: Optional[float] = None
