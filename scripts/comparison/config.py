"""
Configuration constants for the comparison module.
"""

from pathlib import Path

# API Configuration
REFERENCE_API_URL = 'http://127.0.0.1:5001/run_simulation'

# Cache settings
CACHE_DIR = Path(__file__).parent.parent / 'cache'

# Default simulation parameters
DEFAULT_NUM_SIMULATIONS = 200
DEFAULT_AGREEMENT_YEAR = 2030
DEFAULT_END_YEAR = 2037

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
