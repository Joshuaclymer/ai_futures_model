"""
Shared test utilities.

This module contains common functions used across test files and generate_golden_data.py
to avoid code duplication.
"""

import csv
import json
from pathlib import Path

import numpy as np


# Standard time range used in trajectory tests
DEFAULT_TIME_RANGE = [2015.0, 2050.0]

# Golden data directory
GOLDEN_DATA_DIR = Path(__file__).parent / 'golden_data'


def load_time_series_data():
    """Load default input_data.csv into TimeSeriesData.

    Returns:
        TimeSeriesData object with data from input_data.csv
    """
    # Import here to avoid circular imports
    from progress_model import TimeSeriesData

    csv_path = Path(__file__).parent.parent / 'input_data.csv'

    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    return TimeSeriesData(
        time=np.array([float(r['time']) for r in rows]),
        L_HUMAN=np.array([float(r['L_HUMAN']) for r in rows]),
        inference_compute=np.array([float(r['inference_compute']) for r in rows]),
        experiment_compute=np.array([float(r['experiment_compute']) for r in rows]),
        training_compute_growth_rate=np.array([float(r['training_compute_growth_rate']) for r in rows])
    )


def convert_for_json(obj):
    """Convert numpy arrays and types to JSON-serializable format.

    Args:
        obj: Object to convert (can be nested dicts, lists, numpy arrays, etc.)

    Returns:
        JSON-serializable version of the object
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.floating, np.integer)):
        return float(obj)
    elif isinstance(obj, dict):
        return {k: convert_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_for_json(v) for v in obj]
    return obj


def save_golden_data(data: dict, filename: str):
    """Save data to golden data directory.

    Args:
        data: Dictionary of data to save
        filename: Name of the file to save to (relative to golden_data dir)
    """
    GOLDEN_DATA_DIR.mkdir(parents=True, exist_ok=True)
    golden_path = GOLDEN_DATA_DIR / filename

    with open(golden_path, 'w') as f:
        json.dump(convert_for_json(data), f, indent=2)

    print(f"Saved golden data to {golden_path}")


def load_golden_data(filename: str):
    """Load data from golden data directory.

    Args:
        filename: Name of the file to load (relative to golden_data dir)

    Returns:
        Loaded JSON data, or None if file doesn't exist
    """
    golden_path = GOLDEN_DATA_DIR / filename
    if not golden_path.exists():
        return None

    with open(golden_path) as f:
        return json.load(f)
