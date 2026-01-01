"""
Configuration constants for the comparison module.
"""

from pathlib import Path

# API Configuration
REFERENCE_API_URL = 'https://dark-compute.onrender.com/run_simulation'
LOCAL_API_URL = 'http://127.0.0.1:5329/api/get-data-for-ai-black-projects-page'

# Cache settings
CACHE_DIR = Path(__file__).parent.parent / 'cache'

# Default simulation parameters
DEFAULT_NUM_SAMPLES = 200
DEFAULT_START_YEAR = 2029  # Reference model's start_year (prep year)
DEFAULT_AGREEMENT_YEAR = 2030  # = start_year + 1
DEFAULT_NUM_YEARS = 7  # Number of years to simulate (matches reference API output range 2030-2037)
DEFAULT_TOTAL_LABOR = 11300

# Metric comparison tolerances (percent)
METRIC_TOLERANCES = {
    'survival_rate': 15.0,
    'cumulative_lr': 25.0,
    'posterior_prob': 15.0,
    'lr_other_intel': 25.0,
    'lr_reported_energy': 25.0,
    'lr_prc_accounting': 25.0,
    'operating_compute': 30.0,
    'datacenter_capacity': 30.0,
}

# Metric mappings: (local_key, ref_key, display_name, tolerance, intentional_diff)
METRIC_MAPPINGS = [
    # Core detection metrics
    ('survival_rate', 'survival_rate', 'Survival Rate', 15.0, False),
    ('cumulative_lr', 'cumulative_lr', 'Cumulative LR', 25.0, False),
    ('posterior_prob', 'posterior_prob_project', 'Posterior Prob', 15.0, False),
    # Likelihood ratio components
    ('lr_other_intel', 'lr_other_intel', 'LR Other Intel', 25.0, False),
    ('lr_reported_energy', 'lr_reported_energy', 'LR Reported Energy', 25.0, False),
    ('lr_prc_accounting', 'lr_prc_accounting', 'LR Compute Accounting', 25.0, False),
    # Compute metrics
    ('operating_compute', 'operational_compute', 'Operational Compute', 30.0, False),
    ('datacenter_capacity', 'datacenter_capacity', 'Datacenter Capacity', 30.0, False),
]
