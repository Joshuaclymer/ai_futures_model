"""
Dummy endpoint for the black project page.

Provides realistic test data while the real simulation backend is being developed.
This allows frontend development to proceed independently.
"""

import math
import random
import logging
from flask import request, jsonify

logger = logging.getLogger(__name__)

# Default parameters
DEFAULT_PARAMS = {
    'agreementYear': 2027,
    'proportionOfInitialChipStockToDivert': 0.05,
    'workersInCovertProject': 11300,
    'fractionOfLaborDevotedToDatacenterConstruction': 0.885,
    'fractionOfDatacenterCapacityToDivert': 0.5,
    'maxFractionOfTotalNationalEnergyConsumption': 0.05,
    'totalPrcEnergyConsumptionGw': 1100,
    'priorOddsOfCovertProject': 0.3,
}


def register_black_project_dummy_routes(app):
    """Register dummy black project routes with the Flask app."""

    @app.route('/api/black-project-dummy', methods=['GET', 'POST'])
    def black_project_dummy():
        """
        Return dummy data for the black project page.

        Supports both GET (with query params) and POST (with JSON body).
        """
        if request.method == 'POST':
            params = request.json or {}
        else:
            params = {
                'agreementYear': int(request.args.get('agreement_year', 2027)),
                'proportionOfInitialChipStockToDivert': float(request.args.get('diversion_proportion', 0.05)),
            }

        return jsonify(generate_dummy_response(params))


def generate_dummy_response(params):
    """Generate dummy response data."""
    agreement_year = params.get('agreementYear', DEFAULT_PARAMS['agreementYear'])
    diversion_proportion = params.get('proportionOfInitialChipStockToDivert',
                                       DEFAULT_PARAMS['proportionOfInitialChipStockToDivert'])

    return {
        'success': True,
        'num_simulations': 100,

        # Initial stock section data
        'initial_stock': generate_initial_stock_data(diversion_proportion, agreement_year),

        # Rate of computation section data
        'rate_of_computation': generate_rate_of_computation_data(agreement_year),

        # Covert fab section data
        'covert_fab': generate_covert_fab_data(agreement_year),

        # Detection likelihood section data
        'detection_likelihood': generate_detection_likelihood_data(agreement_year),

        # Datacenter section data
        'black_datacenters': generate_black_datacenters_data(agreement_year),

        # Fab section data
        'black_fab': generate_black_fab_data(agreement_year),

        # Main project model data
        'black_project_model': generate_black_project_model_data(agreement_year),
    }


# Helper functions
def generate_years(start, end):
    """Generate array of years from start to end (inclusive)."""
    return list(range(start, end + 1))


def generate_quarterly_years(agreement_year, num_years=7):
    """Generate years array with quarterly intervals."""
    return [agreement_year + i * 0.25 for i in range(num_years * 4 + 1)]


def generate_random_array(length, min_val, max_val):
    """Generate sorted array of random values."""
    arr = [min_val + random.random() * (max_val - min_val) for _ in range(length)]
    return sorted(arr)


def generate_log_normal_samples(median, sigma=0.5, count=1000):
    """Generate log-normal distributed samples."""
    mu = math.log(median)
    samples = []
    for _ in range(count):
        # Box-Muller transform
        u1 = random.random()
        u2 = random.random()
        z = math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)
        samples.append(math.exp(mu + sigma * z))
    return samples


def generate_normal_samples(mean, std_dev, count=1000):
    """Generate normal distributed samples."""
    samples = []
    for _ in range(count):
        u1 = random.random()
        u2 = random.random()
        z = math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)
        samples.append(mean + std_dev * z)
    return samples


def generate_growth_data(start_year, end_year, start_value, growth_rate):
    """Generate growth data with percentile bands."""
    years = end_year - start_year + 1
    median = []
    p25 = []
    p75 = []

    for i in range(years):
        base_value = start_value * (growth_rate ** i)
        median.append(base_value)
        p25.append(base_value * 0.8)
        p75.append(base_value * 1.2)

    return {'median': median, 'p25': p25, 'p75': p75}


def generate_ccdf_data(min_x, max_x, num_points):
    """Generate CCDF data points."""
    data = []
    step = (max_x - min_x) / (num_points - 1)

    for i in range(num_points):
        x = min_x + step * i
        y = math.exp(-3 * (i / (num_points - 1)))
        data.append({'x': round(x * 100) / 100, 'y': round(y * 1000) / 1000})

    return data


def generate_exponential_growth_data(years, initial_value, growth_rate, noise=0.1):
    """Generate exponential growth with percentile bands."""
    median = [initial_value * ((1 + growth_rate) ** (i / 4)) for i in range(len(years))]
    p25 = [v * (1 - noise) for v in median]
    p75 = [v * (1 + noise) for v in median]
    return {'median': median, 'p25': p25, 'p75': p75}


def generate_decay_data(years, half_life=5):
    """Generate decay data for surviving fraction."""
    agreement_year = years[0]
    median = [(0.5 ** ((y - agreement_year) / half_life)) for y in years]
    p25 = [max(0, v - 0.05) for v in median]
    p75 = [min(1, v + 0.05) for v in median]
    return {'median': median, 'p25': p25, 'p75': p75}


def generate_linear_growth_data(years, initial_value, rate_per_year, noise=0.1):
    """Generate linear growth with percentile bands."""
    agreement_year = years[0]
    median = [initial_value + rate_per_year * (y - agreement_year) for y in years]
    p25 = [v * (1 - noise) for v in median]
    p75 = [v * (1 + noise) for v in median]
    return {'median': median, 'p25': p25, 'p75': p75}


def generate_s_curve_data(years, midpoint=2, steepness=2, noise=0.15):
    """Generate S-curve data with confidence intervals."""
    agreement_year = years[0]

    def s_curve(y, mp):
        t = y - agreement_year
        return 1 / (1 + math.exp(-steepness * (t - mp)))

    median = [s_curve(y, midpoint) for y in years]
    p25 = [s_curve(y, midpoint + noise * 2) for y in years]
    p75 = [s_curve(y, midpoint - noise * 2) for y in years]

    return {'median': median, 'p25': p25, 'p75': p75}


def generate_initial_stock_data(diversion_proportion, agreement_year):
    """Generate initial stock section data."""
    start_year = 2025
    years = list(range(start_year, agreement_year + 1))

    base_stock_2025 = 1_000_000  # 1M H100e in 2025
    annual_growth_rate = 1.5

    compute_median = []
    compute_p25 = []
    compute_p75 = []
    domestic_median = []
    proportion_domestic = []
    largest_company = []

    for i, year in enumerate(years):
        years_from_start = year - start_year
        median = base_stock_2025 * (annual_growth_rate ** years_from_start)
        compute_median.append(median)
        compute_p25.append(median * 0.6)
        compute_p75.append(median * 1.5)

        domestic_prop = min(0.5, 0.1 + 0.1 * years_from_start)
        proportion_domestic.append(domestic_prop)
        domestic_median.append(median * domestic_prop)

        largest_company.append(base_stock_2025 * 0.3 * (1.8 ** years_from_start))

    prc_stock_median = compute_median[-1]
    initial_prc_stock_samples = generate_log_normal_samples(prc_stock_median, 0.4, 1000)
    initial_compute_stock_samples = [s * diversion_proportion for s in initial_prc_stock_samples]

    lr_prc_accounting_samples = [max(1, v) for v in generate_normal_samples(1.5, 0.4, 1000)]

    years_from_h100 = agreement_year - 2023
    state_of_the_art_efficiency = 1.35 ** years_from_h100

    H100_POWER_WATTS = 700
    prc_efficiency = 0.20
    combined_efficiency = state_of_the_art_efficiency * prc_efficiency
    initial_energy_samples = [(h100e * H100_POWER_WATTS) / combined_efficiency / 1e9
                               for h100e in initial_compute_stock_samples]

    return {
        'initial_prc_stock_samples': initial_prc_stock_samples,
        'initial_compute_stock_samples': initial_compute_stock_samples,
        'initial_energy_samples': initial_energy_samples,
        'diversion_proportion': diversion_proportion,
        'lr_prc_accounting_samples': lr_prc_accounting_samples,
        'initial_black_project_detection_probs': {'1x': 0.85, '2x': 0.45, '4x': 0.12},
        'prc_compute_years': years,
        'prc_compute_over_time': {'p25': compute_p25, 'median': compute_median, 'p75': compute_p75},
        'prc_domestic_compute_over_time': {'median': domestic_median},
        'proportion_domestic_by_year': proportion_domestic,
        'largest_company_compute_over_time': largest_company,
        'state_of_the_art_energy_efficiency_relative_to_h100': state_of_the_art_efficiency,
    }


def generate_rate_of_computation_data(agreement_year):
    """Generate rate of computation section data."""
    years = generate_quarterly_years(agreement_year)

    initial_chip_stock_samples = generate_log_normal_samples(50000, 0.3, 100)
    acquired_hardware = generate_exponential_growth_data(years, 1000, 0.3, 0.2)
    surviving_fraction = generate_decay_data(years, 6)

    initial_value = 50000
    covert_chip_stock_median = [
        (initial_value + acquired_hardware['median'][i]) * surviving_fraction['median'][i]
        for i in range(len(years))
    ]
    covert_chip_stock = {
        'median': covert_chip_stock_median,
        'p25': [v * 0.7 for v in covert_chip_stock_median],
        'p75': [v * 1.3 for v in covert_chip_stock_median],
    }

    datacenter_capacity = generate_linear_growth_data(years, 5, 8, 0.15)

    energy_per_chip = 0.0005
    energy_stacked_data = [
        [initial_value * surviving_fraction['median'][i] * energy_per_chip,
         acquired_hardware['median'][i] * surviving_fraction['median'][i] * energy_per_chip]
        for i in range(len(years))
    ]

    energy_usage_median = [v * energy_per_chip for v in covert_chip_stock_median]
    energy_usage = {
        'median': energy_usage_median,
        'p25': [v * 0.8 for v in energy_usage_median],
        'p75': [v * 1.2 for v in energy_usage_median],
    }

    operating_chips_median = [
        min(covert_chip_stock_median[i], datacenter_capacity['median'][i] / energy_per_chip)
        for i in range(len(years))
    ]
    operating_chips = {
        'median': operating_chips_median,
        'p25': [v * 0.7 for v in operating_chips_median],
        'p75': [v * 1.3 for v in operating_chips_median],
    }

    cumulative = 0
    covert_computation_median = []
    for v in operating_chips_median:
        cumulative += v * 0.25
        covert_computation_median.append(cumulative)
    covert_computation = {
        'median': covert_computation_median,
        'p25': [v * 0.6 for v in covert_computation_median],
        'p75': [v * 1.4 for v in covert_computation_median],
    }

    return {
        'years': years,
        'initial_chip_stock_samples': initial_chip_stock_samples,
        'acquired_hardware': {'years': years, **acquired_hardware},
        'surviving_fraction': {'years': years, **surviving_fraction},
        'covert_chip_stock': {'years': years, **covert_chip_stock},
        'datacenter_capacity': {'years': years, **datacenter_capacity},
        'energy_usage': {'years': years, **energy_usage},
        'energy_stacked_data': energy_stacked_data,
        'energy_source_labels': ['Initial Stock', 'Fab-Produced'],
        'operating_chips': {'years': years, **operating_chips},
        'covert_computation': {'years': years, **covert_computation},
    }


def generate_covert_fab_data(agreement_year):
    """Generate covert fab section data."""
    years = generate_quarterly_years(agreement_year)

    dashboard = {
        'production': '800K H100e',
        'energy': '2.8 GW',
        'probFabBuilt': '56.2%',
        'yearsOperational': '0.8',
        'processNode': '28nm',
    }

    compute_ccdf = {}
    for threshold in [1, 2, 4]:
        shift = 0 if threshold == 4 else (0.1 if threshold == 2 else 0.3)
        points = []
        for i in range(101):
            x = 10 ** (2 + i * 0.05 + shift)
            y = max(0, 1 - (i / 100) ** 0.5)
            points.append({'x': x, 'y': y})
        compute_ccdf[str(threshold)] = points

    lr_median = [(1.5 ** (i / 4)) for i in range(len(years))]
    lr_p25 = [v * 0.7 for v in lr_median]
    lr_p75 = [v * 1.4 for v in lr_median]

    h100e_median = [0 if i / 4 < 1 else min(500000, 100000 * (i / 4 - 1)) for i in range(len(years))]
    h100e_p25 = [v * 0.6 for v in h100e_median]
    h100e_p75 = [v * 1.4 for v in h100e_median]

    is_operational = generate_s_curve_data(years, 2, 2)
    wafer_starts_samples = sorted([5000 * (0.5 + random.random() * 1.5) for _ in range(100)])

    chips_per_wafer = 28
    architecture_efficiency = 0.85
    h100_power = 700

    dennard_threshold = 0.02
    def compute_watts_per_tpp(density):
        if density < dennard_threshold:
            return density ** -0.5
        else:
            watts_at_threshold = dennard_threshold ** -0.5
            return watts_at_threshold * ((density / dennard_threshold) ** -0.15)

    transistor_density = [
        {'node': '28nm', 'density': 0.14, 'wattsPerTpp': compute_watts_per_tpp(0.14)},
        {'node': '14nm', 'density': 0.5, 'wattsPerTpp': compute_watts_per_tpp(0.5)},
        {'node': '7nm', 'density': 1.0, 'wattsPerTpp': compute_watts_per_tpp(1.0)},
    ]

    base_value = 5000 * 28 * 0.14 * 0.85
    compute_per_month_median = [is_operational['median'][i] * base_value for i in range(len(years))]
    compute_per_month = {
        'years': years,
        'median': compute_per_month_median,
        'p25': [v * 0.6 for v in compute_per_month_median],
        'p75': [v * 1.5 for v in compute_per_month_median],
    }

    watts_per_tpp_curve = {
        'densityRelative': [],
        'wattsPerTppRelative': [],
    }
    for i in range(51):
        density = 10 ** (-3 + i * 0.1)
        watts_per_tpp_curve['densityRelative'].append(density)
        watts_per_tpp_curve['wattsPerTppRelative'].append(compute_watts_per_tpp(density))

    watts_per_tpp_relative = 3
    energy_per_month_median = [(v * watts_per_tpp_relative * h100_power) / 1e9 for v in compute_per_month_median]
    energy_per_month = {
        'years': years,
        'median': energy_per_month_median,
        'p25': [v * 0.6 for v in energy_per_month_median],
        'p75': [v * 1.5 for v in energy_per_month_median],
    }

    return {
        'dashboard': dashboard,
        'compute_ccdf': compute_ccdf,
        'time_series_data': {
            'years': years,
            'lr_combined': {'years': years, 'median': lr_median, 'p25': lr_p25, 'p75': lr_p75},
            'h100e_flow': {'years': years, 'median': h100e_median, 'p25': h100e_p25, 'p75': h100e_p75},
        },
        'is_operational': {'years': years, **is_operational},
        'wafer_starts_samples': wafer_starts_samples,
        'chips_per_wafer': chips_per_wafer,
        'architecture_efficiency': architecture_efficiency,
        'h100_power': h100_power,
        'transistor_density': transistor_density,
        'compute_per_month': compute_per_month,
        'watts_per_tpp_curve': watts_per_tpp_curve,
        'energy_per_month': energy_per_month,
    }


def generate_detection_likelihood_data(agreement_year):
    """Generate detection likelihood section data."""
    years = generate_quarterly_years(agreement_year)

    chip_evidence_samples = generate_log_normal_samples(1.3, 0.6, 100)
    sme_evidence_samples = generate_log_normal_samples(1.2, 0.5, 100)
    dc_evidence_samples = generate_log_normal_samples(1.1, 0.4, 100)

    energy_evidence = generate_exponential_growth_data(years, 1.0, 0.03, 0.15)
    combined_evidence = generate_exponential_growth_data(years, 1.0, 0.15, 0.4)
    direct_evidence = generate_exponential_growth_data(years, 1.0, 0.15, 0.4)
    posterior_prob = generate_s_curve_data(years, 3, 1.5)

    return {
        'years': years,
        'chip_evidence_samples': chip_evidence_samples,
        'sme_evidence_samples': sme_evidence_samples,
        'dc_evidence_samples': dc_evidence_samples,
        'energy_evidence': {'years': years, **energy_evidence},
        'combined_evidence': {'years': years, **combined_evidence},
        'direct_evidence': {'years': years, **direct_evidence},
        'posterior_prob': {'years': years, **posterior_prob},
    }


def generate_black_datacenters_data(agreement_year):
    """Generate datacenter section data."""
    years = generate_years(agreement_year, agreement_year + 8)

    return {
        'individual_capacity_before_detection': generate_random_array(100, 30, 70),
        'individual_time_before_detection': generate_random_array(100, 0.5, 2.0),
        'years': years,
        'datacenter_capacity': {
            'median': [0, 5, 15, 30, 45, 55, 55, 55, 55],
            'p25': [0, 3, 10, 22, 35, 45, 45, 45, 45],
            'p75': [0, 7, 20, 38, 55, 65, 65, 65, 65],
        },
        'lr_datacenters': {
            'median': [1, 1.2, 1.5, 2, 3, 5, 8, 12, 20],
            'p25': [1, 1.1, 1.3, 1.6, 2.2, 3.5, 5.5, 8, 14],
            'p75': [1, 1.3, 1.8, 2.5, 4, 7, 12, 18, 30],
        },
        'capacity_ccdfs': {
            '1': [
                {'x': 5, 'y': 1.0}, {'x': 10, 'y': 0.98}, {'x': 15, 'y': 0.95},
                {'x': 20, 'y': 0.90}, {'x': 25, 'y': 0.82}, {'x': 30, 'y': 0.72},
                {'x': 35, 'y': 0.60}, {'x': 40, 'y': 0.48}, {'x': 45, 'y': 0.36},
                {'x': 50, 'y': 0.25}, {'x': 55, 'y': 0.16}, {'x': 60, 'y': 0.09},
                {'x': 65, 'y': 0.04}, {'x': 70, 'y': 0.02}, {'x': 80, 'y': 0.01},
            ],
            '4': [
                {'x': 5, 'y': 1.0}, {'x': 10, 'y': 0.96}, {'x': 15, 'y': 0.90},
                {'x': 20, 'y': 0.82}, {'x': 25, 'y': 0.70}, {'x': 30, 'y': 0.58},
                {'x': 35, 'y': 0.45}, {'x': 40, 'y': 0.34}, {'x': 45, 'y': 0.24},
                {'x': 50, 'y': 0.16}, {'x': 55, 'y': 0.10}, {'x': 60, 'y': 0.05},
                {'x': 65, 'y': 0.02}, {'x': 70, 'y': 0.01}, {'x': 80, 'y': 0.005},
            ],
        },
        'prc_capacity_years': generate_years(2020, agreement_year),
        'prc_capacity_gw': generate_growth_data(2020, agreement_year, 50, 1.15),
        'prc_capacity_at_agreement_year_gw': 132,
        'fraction_diverted': 0.01,
        'total_prc_energy_gw': 1000,
        'max_proportion_energy': 0.01,
        'construction_workers': 10000,
        'mw_per_worker_per_year': 0.2,
        'datacenter_start_year': agreement_year - 2,
    }


def generate_black_fab_data(agreement_year):
    """Generate fab section data."""
    years = generate_years(agreement_year, agreement_year + 8)

    return {
        'years': years,
        'individual_energy_before_detection': generate_random_array(100, 1.5, 4.5),
        'individual_production_before_detection': generate_random_array(100, 50000, 200000),
        'wafer_starts': {
            'median': [0, 0, 100, 500, 1200, 2000, 2500, 2800, 3000],
            'p25': [0, 0, 50, 300, 800, 1400, 1800, 2000, 2200],
            'p75': [0, 0, 150, 700, 1600, 2600, 3200, 3600, 3800],
        },
        'lr_fab': {
            'median': [1, 1.1, 1.3, 1.8, 2.5, 4, 6, 10, 15],
            'p25': [1, 1.05, 1.15, 1.4, 1.8, 2.5, 3.5, 6, 9],
            'p75': [1, 1.15, 1.5, 2.2, 3.5, 6, 10, 16, 25],
        },
        'production_ccdfs': {
            '1': generate_ccdf_data(50000, 250000, 15),
            '4': generate_ccdf_data(30000, 200000, 15),
        },
        'energy_ccdfs': {
            '1': generate_ccdf_data(1, 6, 15),
            '4': generate_ccdf_data(0.5, 5, 15),
        },
        'fab_construction_time': 2.5,
        'architecture_efficiency': 0.8,
        'wafers_per_scanner': 1000,
    }


def generate_black_project_model_data(agreement_year):
    """Generate main project model data."""
    years = generate_years(agreement_year, agreement_year + 8)

    return {
        'years': years,
        'individual_project_h100_years_before_detection': generate_random_array(100, 5000, 25000),
        'individual_project_time_before_detection': generate_random_array(100, 0.8, 3.0),
        'individual_project_h100e_before_detection': generate_random_array(100, 8000, 20000),
        'h100_years_ccdf': {
            '1': generate_ccdf_data(5000, 1000000, 20),
            '2': generate_ccdf_data(1000, 500000, 20),
            '4': generate_ccdf_data(100, 100000, 20),
        },
        'time_to_detection_ccdf': {
            '1': generate_ccdf_data(1.0, 7, 15),
            '2': generate_ccdf_data(0.5, 5, 15),
            '4': generate_ccdf_data(0.2, 3, 15),
        },
        'chip_production_reduction_ccdf': {
            'global': [
                {'x': 1, 'y': 1.0}, {'x': 0.1, 'y': 0.995}, {'x': 0.01, 'y': 0.98},
                {'x': 0.001, 'y': 0.95}, {'x': 0.0001, 'y': 0.88},
            ],
            'prc': [
                {'x': 1, 'y': 1.0}, {'x': 0.1, 'y': 0.98}, {'x': 0.01, 'y': 0.92},
                {'x': 0.001, 'y': 0.82}, {'x': 0.0001, 'y': 0.65},
            ],
        },
        'ai_rd_reduction_ccdf': {
            'largest_ai_company': [
                {'x': 1, 'y': 1.0}, {'x': 0.1, 'y': 0.995}, {'x': 0.01, 'y': 0.95},
                {'x': 0.001, 'y': 0.75}, {'x': 0.0001, 'y': 0.22},
            ],
            'prc': [
                {'x': 1, 'y': 1.0}, {'x': 0.1, 'y': 0.98}, {'x': 0.01, 'y': 0.88},
                {'x': 0.001, 'y': 0.65}, {'x': 0.0001, 'y': 0.20},
            ],
        },
        'ai_rd_reduction_median': 0.05,
    }
