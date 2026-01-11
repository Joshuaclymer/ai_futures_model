"""
Build the black_datacenters section of the API response.

Contains datacenter capacity metrics, detection probabilities, and PRC capacity data.
"""

import numpy as np
from typing import Dict, List, Any

from ..percentile_helpers import get_percentiles_with_individual, compute_ccdf
from ..detection import compute_datacenter_capacity_at_detection

# Constants
LIKELIHOOD_RATIO_THRESHOLDS = [1, 2, 4]
LR_DETECTION_THRESHOLD = 5.0


def build_black_datacenters_section(
    all_data: List[Dict],
    years: List[float],
    dt: float,
    ai_slowdown_start_year: float,
    energy_by_source: List[List[float]],
    source_labels: List[str],
    prc_capacity_years: List[int],
    prc_capacity_gw: Dict[str, List[float]],
    prc_capacity_at_agreement: float,
    prc_capacity_at_agreement_samples: List[float],
) -> Dict[str, Any]:
    """
    Build the black_datacenters section of the response.

    This section contains 16 keys including datacenter metrics and PRC capacity.
    """
    # Compute combined datacenter LR for each simulation
    lr_datacenters_by_sim = []
    for d in all_data:
        bp = d.get('black_project')
        if not bp:
            lr_datacenters_by_sim.append([1.0] * len(years))
            continue

        # Get the three LR components
        lr_satellite = bp.get('lr_satellite_datacenter', 1.0)  # Fixed value
        lr_reported_energy = bp.get('lr_reported_energy', [1.0] * len(years))
        lr_worker = bp.get('lr_other_intel', [1.0] * len(years))

        # Compute combined datacenter LR
        combined_lr = []
        for i in range(len(years)):
            lr_energy = lr_reported_energy[i] if i < len(lr_reported_energy) else 1.0
            lr_work = lr_worker[i] if i < len(lr_worker) else 1.0
            combined_lr.append(lr_satellite * lr_energy * lr_work)
        lr_datacenters_by_sim.append(combined_lr)

    lr_datacenters_array = np.array(lr_datacenters_by_sim)
    datacenter_detection_prob = (np.mean(lr_datacenters_array >= LR_DETECTION_THRESHOLD, axis=0)).tolist()

    # Compute datacenter-specific detection times
    datacenter_detection_times = _compute_datacenter_detection_times(
        all_data, years, ai_slowdown_start_year, lr_datacenters_by_sim
    )

    # Fraction diverted (fixed from parameters)
    fraction_diverted = 0.01

    return {
        "years": years,
        "datacenter_capacity": get_percentiles_with_individual(
            all_data,
            lambda d: d['black_project']['datacenter_capacity_gw'] if d['black_project'] else [0.0] * len(years)
        ),
        # operational_compute: divide by 1000 to convert H100e to K H100e
        "operational_compute": get_percentiles_with_individual(
            all_data,
            lambda d: [v / 1000.0 for v in (d['black_project']['operational_compute'] if d['black_project'] else [0.0] * len(years))]
        ),
        "lr_datacenters": {
            "p25": np.percentile(lr_datacenters_array, 25, axis=0).tolist(),
            "median": np.percentile(lr_datacenters_array, 50, axis=0).tolist(),
            "p75": np.percentile(lr_datacenters_array, 75, axis=0).tolist(),
            "individual": [sim.tolist() if hasattr(sim, 'tolist') else list(sim) for sim in lr_datacenters_by_sim]
        },
        "datacenter_detection_prob": datacenter_detection_prob,
        "energy_by_source": energy_by_source,
        "source_labels": source_labels,
        "fraction_diverted": fraction_diverted,
        # Only threshold 1 for datacenter capacity CCDFs
        "capacity_ccdfs": {
            str(lr): compute_ccdf(compute_datacenter_capacity_at_detection(all_data, years, lr))
            for lr in [1]
        },
        "individual_capacity_before_detection": [
            d['black_project']['datacenter_capacity_gw'][
                min(int(datacenter_detection_times[i] / dt) if dt > 0 else 0, len(d['black_project']['datacenter_capacity_gw']) - 1)
            ] if d['black_project'] and d['black_project'].get('datacenter_capacity_gw') else 0.0
            for i, d in enumerate(all_data)
        ],
        "individual_time_before_detection": datacenter_detection_times,
        "likelihood_ratios": LIKELIHOOD_RATIO_THRESHOLDS,
        "prc_capacity_years": prc_capacity_years,
        "prc_capacity_gw": prc_capacity_gw,
        "prc_capacity_at_ai_slowdown_start_year_gw": prc_capacity_at_agreement,
        "prc_capacity_at_ai_slowdown_start_year_samples": prc_capacity_at_agreement_samples,
    }


def _compute_datacenter_detection_times(
    all_data: List[Dict],
    years: List[float],
    ai_slowdown_start_year: float,
    lr_datacenters_by_sim: List[List[float]],
) -> List[float]:
    """Compute datacenter-specific detection times using combined datacenter LR."""
    datacenter_detection_times = []
    for i, d in enumerate(all_data):
        bp = d.get('black_project')
        sim_years = d.get('years', years)

        if not bp or not sim_years:
            datacenter_detection_times.append(sim_years[-1] - ai_slowdown_start_year if sim_years else 7.0)
            continue

        lr_datacenters = lr_datacenters_by_sim[i]

        # Find first year where datacenter LR >= threshold
        detection_year = None
        for j, year in enumerate(sim_years):
            if j < len(lr_datacenters) and lr_datacenters[j] >= LR_DETECTION_THRESHOLD:
                detection_year = year
                break

        if detection_year is not None:
            datacenter_detection_times.append(detection_year - ai_slowdown_start_year)
        else:
            datacenter_detection_times.append(sim_years[-1] - ai_slowdown_start_year if sim_years else 7.0)

    return datacenter_detection_times
