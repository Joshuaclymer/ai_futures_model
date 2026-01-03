"""
Global compute production data loading and utilities.

Loads and caches data from global_compute_production.csv for computing
global chip stock and production between years.
"""

import csv
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# Module-level cache for global compute production data
_cached_global_compute_production = None


def _parse_compute_value(s: str) -> Optional[float]:
    """Parse a compute value string with optional K/M/B/T suffix."""
    if not s:
        return None
    s = s.strip().replace(',', '')
    multipliers = {'K': 1e3, 'M': 1e6, 'B': 1e9, 'T': 1e12}
    for suffix, mult in multipliers.items():
        if s.endswith(suffix):
            try:
                return float(s[:-1]) * mult
            except ValueError:
                return None
    try:
        return float(s)
    except ValueError:
        return None


def _load_global_compute_production() -> Dict[str, List]:
    """Load the global compute production data from global_compute_production.csv.

    Returns a dict with:
        - years: list of years
        - total_stock: list of total H100e in world (no decay)
    """
    global _cached_global_compute_production
    if _cached_global_compute_production is not None:
        return _cached_global_compute_production

    csv_path = Path(__file__).resolve().parent.parent.parent.parent / "ai_futures_simulator" / "data" / "global_compute_production.csv"

    years = []
    total_stock = []

    try:
        with open(csv_path, 'r') as f:
            reader = csv.reader(f)
            # Skip header rows (first 3 rows: header + 2 metadata rows)
            for _ in range(3):
                next(reader)

            for row in reader:
                if len(row) > 0 and row[0]:
                    try:
                        year = int(row[0])
                        # Column U (index 20) is "H100e in world, no decay"
                        stock = _parse_compute_value(row[20]) if len(row) > 20 else None
                        years.append(year)
                        total_stock.append(stock)
                    except (ValueError, IndexError):
                        continue
    except FileNotFoundError:
        logger.warning(f"Global compute production CSV not found at {csv_path}")
        years = []
        total_stock = []

    _cached_global_compute_production = {
        'years': years,
        'total_stock': total_stock
    }
    return _cached_global_compute_production


def get_global_compute_stock(year: float) -> float:
    """Get the global compute stock (H100e) for a given year.

    Uses the "H100e in world, no decay" column from global_compute_production.csv.
    Linearly interpolates between years.
    """
    data = _load_global_compute_production()

    # Filter to valid entries
    valid_years = []
    valid_stocks = []
    for y, s in zip(data['years'], data['total_stock']):
        if s is not None:
            valid_years.append(y)
            valid_stocks.append(s)

    if not valid_years:
        # Fallback: use simple exponential model
        base_2025 = 500000  # ~500K H100e globally in 2025
        growth_rate = 2.5
        return base_2025 * (growth_rate ** (year - 2025))

    return float(np.interp(year, valid_years, valid_stocks))


def get_global_compute_production_between_years(start_year: float, end_year: float) -> float:
    """Calculate total global compute production between two years.

    Uses the change in global compute stock (H100e in world, no decay) between years.
    Production = Stock(end_year) - Stock(start_year)
    """
    start_stock = get_global_compute_stock(start_year)
    end_stock = get_global_compute_stock(end_year)
    return max(0.0, end_stock - start_stock)
