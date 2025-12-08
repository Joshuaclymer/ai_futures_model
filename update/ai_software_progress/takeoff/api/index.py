from flask import Flask, request, jsonify
import numpy as np
import io
import csv
import logging
import time

from progress_model import (
    ProgressModel, Parameters, TimeSeriesData,
)
from api.response_formatters import build_time_series_payload
from time_series_generator import generate_time_series_from_dict

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

def create_default_time_series():
    """Load default time series data from input_data.csv"""
    try:
        # Try to load from input_data.csv
        import os
        csv_path = os.path.join(os.path.dirname(__file__), '../input_data.csv')
        
        if os.path.exists(csv_path):
            logger.info("Loading default time series data from input_data.csv")
            with open(csv_path, 'r') as csvfile:
                reader = csv.DictReader(csvfile)
                data = list(reader)
            
            time = np.array([float(row['time']) for row in data])
            L_HUMAN = np.array([float(row['L_HUMAN']) for row in data])
            inference_compute = np.array([float(row['inference_compute']) for row in data])
            experiment_compute = np.array([float(row['experiment_compute']) for row in data])
            training_compute_growth_rate = np.array([float(row['training_compute_growth_rate']) for row in data])
            
            logger.info(f"Loaded time series data: {len(data)} points from {time[0]} to {time[-1]}")
            return TimeSeriesData(time, L_HUMAN, inference_compute, experiment_compute, training_compute_growth_rate)
        else:
            logger.warning("input_data.csv not found, falling back to synthetic data")
            raise FileNotFoundError("input_data.csv not found")
            
    except Exception as e:
        logger.warning(f"Error loading input_data.csv: {e}, falling back to synthetic data")
        # Fallback to synthetic data if CSV loading fails
        time = np.linspace(2029, 2030, 12)
        L_HUMAN = np.ones_like(time) * 1e6
        inference_compute = np.logspace(3, 8, len(time))
        experiment_compute = np.logspace(6, 10, len(time))  # Use exponential growth as fallback
        training_compute_growth_rate = np.logspace(6, 10, len(time))
        
        logger.info("Using synthetic fallback data")
        return TimeSeriesData(time, L_HUMAN, inference_compute, experiment_compute, training_compute_growth_rate)

def create_default_parameters():
    """Create default model parameters"""
    return Parameters()

def load_time_series_from_csv_path(csv_path):
    """Load time series data from a given CSV file path"""
    with open(csv_path, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        data = list(reader)
    time = np.array([float(row['time']) for row in data])
    L_HUMAN = np.array([float(row['L_HUMAN']) for row in data])
    inference_compute = np.array([float(row['inference_compute']) for row in data])
    experiment_compute = np.array([float(row['experiment_compute']) for row in data])
    training_compute_growth_rate = np.array([float(row['training_compute_growth_rate']) for row in data])
    return TimeSeriesData(time, L_HUMAN, inference_compute, experiment_compute, training_compute_growth_rate)

def time_series_to_dict(data: TimeSeriesData):
    """Convert TimeSeriesData to dictionary for JSON serialization"""
    return {
        'time': data.time.tolist(),
        'L_HUMAN': data.L_HUMAN.tolist(),
        'inference_compute': data.inference_compute.tolist(),
        'experiment_compute': data.experiment_compute.tolist(),
        'training_compute_growth_rate': data.training_compute_growth_rate.tolist()
    }

def params_to_dict(params: Parameters):
    """Convert parameters to dictionary for JSON serialization""" 
    param_dict = {
        'rho_coding_labor': params.rho_coding_labor,
        'rho_experiment_capacity': params.rho_experiment_capacity,
        'alpha_experiment_capacity': params.alpha_experiment_capacity,
        'r_software': params.r_software,
        'automation_fraction_at_coding_automation_anchor': params.automation_fraction_at_coding_automation_anchor,
        'automation_interp_type': getattr(params, 'automation_interp_type', 'exponential'),

        'coding_labor_normalization': params.coding_labor_normalization,
        'coding_labor_mode': getattr(params, 'coding_labor_mode', 'simple_ces'),
        'coding_automation_efficiency_slope': getattr(params, 'coding_automation_efficiency_slope', 1.0),
        'optimal_ces_eta_init': getattr(params, 'optimal_ces_eta_init', 1.0),
        'optimal_ces_grid_size': getattr(params, 'optimal_ces_grid_size', 4096),
        'optimal_ces_frontier_tail_eps': getattr(params, 'optimal_ces_frontier_tail_eps', 1e-6),
        'optimal_ces_frontier_cap': getattr(params, 'optimal_ces_frontier_cap', None),
        'max_serial_coding_labor_multiplier': getattr(params, 'max_serial_coding_labor_multiplier', None),
        'experiment_compute_exponent': params.experiment_compute_exponent,
            'ai_research_taste_at_coding_automation_anchor': params.ai_research_taste_at_coding_automation_anchor,
            'ai_research_taste_at_coding_automation_anchor_sd': params.ai_research_taste_at_coding_automation_anchor_sd,
            'ai_research_taste_slope': params.ai_research_taste_slope,
        'taste_schedule_type': params.taste_schedule_type,
        'progress_at_aa': params.progress_at_aa,
        'ac_time_horizon_minutes': params.ac_time_horizon_minutes,
        'pre_gap_ac_time_horizon': getattr(params, 'pre_gap_ac_time_horizon', None),
        'horizon_extrapolation_type': params.horizon_extrapolation_type,
        # Manual horizon fitting parameters
        'present_day': params.present_day,
        'present_horizon': params.present_horizon,
        'present_doubling_time': params.present_doubling_time,
        'doubling_difficulty_growth_factor': params.doubling_difficulty_growth_factor,
        # Gap mode selection
        'include_gap': getattr(params, 'include_gap', 'no gap'),
        'gap_years': float(getattr(params, 'gap_years', 0.0)),
        # Research taste distribution parameter
        'median_to_top_taste_multiplier': getattr(params, 'median_to_top_taste_multiplier', 3.25),
        # Median-to-best multipliers for general capability milestones
        'strat_ai_m2b': getattr(params, 'strat_ai_m2b', 2.0),
        'ted_ai_m2b': getattr(params, 'ted_ai_m2b', 4.0),
    }

    return param_dict

@app.route('/api/compute', methods=['POST'])
def compute_model():
    """Compute model with given parameters"""
    # try:
    data = request.json

    # Parse parameters
    all_params_dict = data.get('parameters', {})

    # Extract time series generation parameters
    time_series_param_keys = ['constant_training_compute_growth_rate', 'slowdown_year', 'post_slowdown_training_compute_growth_rate']
    time_series_params = {k: all_params_dict[k] for k in time_series_param_keys if k in all_params_dict}

    # Filter out time series parameters from model parameters
    params_dict = {k: v for k, v in all_params_dict.items() if k not in time_series_param_keys}

    # Check if profiling is requested
    enable_profiling = data.get('enable_profiling', False)

    # Initialize profiling variables
    total_time = 0.0
    timing_breakdown = {}

    if enable_profiling:
        start_time = time.perf_counter()

    # Time: Parameters creation (includes TasteDistribution init)
    _t0 = time.perf_counter()
    params = Parameters(**params_dict)
    timing_breakdown['params_creation'] = time.perf_counter() - _t0

    # Generate time series data using parameters if provided, otherwise use default CSV
    _t0 = time.perf_counter()
    if time_series_params and len(time_series_params) == 3:
        # Use time series generator with the provided parameters
        import os
        base_csv_path = os.path.join(os.path.dirname(__file__), '../input_data.csv')
        time_series = generate_time_series_from_dict(time_series_params, base_csv_path)
        logger.info(f"Generated time series with params: {time_series_params}")
    else:
        # Fall back to default CSV
        time_series = create_default_time_series()
        logger.info("Using default time series from CSV")
    timing_breakdown['time_series_load'] = time.perf_counter() - _t0

    # Get time range
    time_range = data.get('time_range', [2029, 2030])
    initial_progress = data.get('initial_progress', 0.0)

    # Time: Model initialization
    _t0 = time.perf_counter()
    model = ProgressModel(params, time_series)
    timing_breakdown['model_init'] = time.perf_counter() - _t0

    # Time: Trajectory computation
    _t0 = time.perf_counter()
    times, progress_values, research_stock_values = model.compute_progress_trajectory(
            time_range, initial_progress
        )
    timing_breakdown['compute_trajectory'] = time.perf_counter() - _t0

    # Extract timing from model results if available
    if hasattr(model, 'results') and model.results:
        if 'timing' in model.results:
            timing_breakdown['model_internal'] = model.results['timing']

    if enable_profiling:
        end_time = time.perf_counter()
        total_time = end_time - start_time

    # All metrics are now computed by ProgressModel - use them directly
    all_metrics = model.results

    # Prepare summary focusing on SC metrics
    summary = {
        'time_range': time_range
    }
    
    # Add anchor taste slope metrics if available
    if model.results.get('ai_research_taste_slope_per_anchor_progress_year') is not None:
        summary['ai_research_taste_slope_per_anchor_progress_year'] = float(model.results['ai_research_taste_slope_per_anchor_progress_year'])
    if model.results.get('ai_research_taste_slope_per_effective_oom') is not None:
        summary['ai_research_taste_slope_per_effective_oom'] = float(model.results['ai_research_taste_slope_per_effective_oom'])

    # Add anchor progress rate if available
    if model.results.get('anchor_progress_rate') is not None:
        summary['anchor_progress_rate'] = float(model.results['anchor_progress_rate'])

    # Add r_software (calibrated value) if available
    if model.results.get('r_software') is not None:
        summary['r_software'] = float(model.results['r_software'])
    
    # Add beta_software (inverse of r_software) if available
    if model.results.get('beta_software') is not None:
        summary['beta_software'] = float(model.results['beta_software'])
    
    # Add top taste percentile metrics if available
    if model.results.get('top_taste_percentile') is not None:
        summary['top_taste_percentile'] = float(model.results['top_taste_percentile'])
    if model.results.get('top_taste_num_sds') is not None:
        summary['top_taste_num_sds'] = float(model.results['top_taste_num_sds'])
    if model.results.get('f_multiplier_per_sd') is not None:
        summary['f_multiplier_per_sd'] = float(model.results['f_multiplier_per_sd'])
    if model.results.get('slope_times_log_f') is not None:
        summary['slope_times_log_f'] = float(model.results['slope_times_log_f'])

    # Add instantaneous anchor doubling time (years) if available
    if model.results.get('instantaneous_anchor_doubling_time_years') is not None:
        summary['instantaneous_anchor_doubling_time_years'] = float(model.results['instantaneous_anchor_doubling_time_years'])

    # Add SC timing information if available
    if model.results.get('sc_progress_level') is not None and model.results.get('sc_sw_multiplier') is not None:
        summary['aa_time'] = float(model.results['aa_time'])
        summary['sc_progress_level'] = float(model.results['sc_progress_level'])
        summary['sc_sw_multiplier'] = float(model.results['sc_sw_multiplier']) 
        logger.info(f"SC time: {summary['aa_time']}, SC progress level: {summary['sc_progress_level']}, SC SW multiplier: {summary['sc_sw_multiplier']}")
    # Add AI2027 SC time if computed
    if model.results.get('ai2027_sc_time') is not None:
        try:
            summary['ai2027_sc_time'] = float(model.results.get('ai2027_sc_time'))
        except Exception:
            summary['ai2027_sc_time'] = None
    # Extract horizon params if available
    horizon_params = None
    if 'horizon_params' in model.results:
        hp = model.results['horizon_params']
        horizon_params = {
            'uses_shifted_form': bool(hp.get('uses_shifted_form', False)),
            'anchor_progress': float(hp['anchor_progress']) if hp.get('anchor_progress') is not None else None
        }

    response_data = {
        'success': True,
        'summary': summary,
        'time_series': build_time_series_payload(model.results),
        'milestones': model.results['milestones'],
        'horizon_params': horizon_params,
        # Surface computed exp capacity params so UI can display when using pseudoparameters
        'exp_capacity_params': {
            'rho': float(all_metrics.get('exp_capacity_params', {}).get('rho')) if all_metrics.get('exp_capacity_params') else None,
            'alpha': float(all_metrics.get('exp_capacity_params', {}).get('alpha')) if all_metrics.get('exp_capacity_params') else None,
            'experiment_compute_exponent': float(all_metrics.get('exp_capacity_params', {}).get('experiment_compute_exponent')) if all_metrics.get('exp_capacity_params') else None,
        }
    }

    # Add profiling data if enabled
    if enable_profiling:
        response_data['profiling'] = {
            'enabled': True,
            'total_time_seconds': total_time,
            'timing_breakdown': timing_breakdown,
            'stats': '(cProfile disabled to avoid overhead)'
        }

    return jsonify(response_data)
        
    # except Exception as e:
    #     logger.error(f"Error computing model: {e}")
    #     return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/parameter-config', methods=['GET'])
def get_parameter_config():
    """Get parameter configuration including bounds, defaults, and metadata"""
    try:
        import model_config as cfg
        
        parameter_config = {
            'bounds': cfg.PARAMETER_BOUNDS,
            'defaults': cfg.DEFAULT_PARAMETERS,
            'validation_thresholds': cfg.PARAM_VALIDATION_THRESHOLDS,
            'taste_schedule_types': cfg.TASTE_SCHEDULE_TYPES,
            'taste_slope_defaults': cfg.TASTE_SLOPE_DEFAULTS,
            'horizon_extrapolation_types': cfg.HORIZON_EXTRAPOLATION_TYPES,
            'automation_interp_types': ['exponential', 'linear', 'logistic'],
            'coding_labor_modes': ['simple_ces', 'optimal_ces'],
            'pre_gap_ac_time_horizon': cfg.DEFAULT_PARAMETERS['pre_gap_ac_time_horizon'],
            'modelConstants': {
                'baseForSoftwareLOM': cfg.BASE_FOR_SOFTWARE_LOM,
                'medianToTopGap': cfg.MEDIAN_TO_TOP_TASTE_MULTIPLIER,
                'topPercentile': cfg.TOP_PERCENTILE,
                'aggregateResearchTasteBaseline': cfg.AGGREGATE_RESEARCH_TASTE_BASELINE,
                'aiResearchTasteMaxSD': cfg.AI_RESEARCH_TASTE_MAX_SD,
                'aiResearchTasteMin': cfg.AI_RESEARCH_TASTE_MIN,
                'aiResearchTasteMax': cfg.AI_RESEARCH_TASTE_MAX,
                'automationFractionClipMin': cfg.AUTOMATION_FRACTION_CLIP_MIN,
                'trainingComputeReferenceOOMs': cfg.TRAINING_COMPUTE_REFERENCE_OOMS,
                'trainingComputeReferenceYear': cfg.TRAINING_COMPUTE_REFERENCE_YEAR,
            },
            'descriptions': {
                'rho_coding_labor': {
                    'name': 'Cognitive Elasticity (ρ_cognitive)',
                    'description': 'Elasticity of substitution between AI and human cognitive labor',
                    'units': 'dimensionless'
                },
                'rho_experiment_capacity': {
                    'name': 'Progress Elasticity (ρ_progress)', 
                    'description': 'Elasticity of substitution in progress production function',
                    'units': 'dimensionless'
                },
                'alpha_experiment_capacity': {
                    'name': 'Compute Weight (α)',
                    'description': 'Weight of compute vs cognitive output in progress production',
                    'units': 'dimensionless'
                },
                'r_software': {
                    'name': 'Software Scale',
                    'description': 'Scale factor for software progress',
                    'units': 'dimensionless'
                },
                'automation_fraction_at_coding_automation_anchor': {
                    'name': 'Max Automation',
                    'description': 'Automation fraction when AI reaches superhuman coding ability',
                    'units': 'fraction'
                },
                'automation_interp_type': {
                    'name': 'Automation Interpolation Type',
                    'description': 'Method for interpolating automation between anchors',
                    'units': 'categorical'
                },
                
                'swe_multiplier_at_present_day': {
                    'name': 'SWE Multiplier at Anchor Time',
                    'description': 'Software engineering productivity multiplier at the anchor time',
                    'units': 'dimensionless'
                },
                'coding_labor_normalization': {
                    'name': 'Cognitive Output Normalization',
                    'description': 'Normalization factor for cognitive output',
                    'units': 'dimensionless'
                },
                'coding_labor_mode': {
                    'name': 'Coding Labor Mode',
                    'description': 'Select between simple CES and optimal frontier CES',
                    'units': 'categorical'
                },
                'coding_automation_efficiency_slope': {
                    'name': 'Automation Efficiency Slope',
                    'description': 'Slope of automation multiplier above threshold (η ∝ E^θ)',
                    'units': 'dimensionless'
                },
                'optimal_ces_eta_init': {
                    'name': 'Optimal CES η_init',
                    'description': 'Automation multiplier at the threshold',
                    'units': 'dimensionless'
                },
                'optimal_ces_grid_size': {
                    'name': 'Optimal CES Grid Size',
                    'description': 'Number of grid points for frontier precompute',
                    'units': 'count'
                },
                'optimal_ces_frontier_tail_eps': {
                    'name': 'Optimal CES Frontier Tail ε',
                    'description': 'Controls how close the precomputed grid approaches full automation (smaller ε allows larger AI labor multipliers)',
                    'units': 'dimensionless'
                },
                'optimal_ces_frontier_cap': {
                    'name': 'Optimal CES Frontier Cap',
                    'description': 'Targets a specific ceiling on the AI coding labor multiplier; internally mapped to the frontier grid tail ε',
                    'units': 'multiplier'
                },
                'max_serial_coding_labor_multiplier': {
                    'name': 'Max Serial Coding Labor Multiplier',
                    'description': 'Cap on serial coding labor; the model raises this to parallel_penalty to produce the internal optimal CES frontier cap',
                    'units': 'multiplier'
                },
                'experiment_compute_exponent': {
                    'name': 'Experiment Compute Discounting (ζ)',
                    'description': 'Diminishing returns factor for experiment compute',
                    'units': 'dimensionless'
                },

                'ai_research_taste_at_coding_automation_anchor': {
                    'name': 'Max AI Research Taste',
                    'description': 'AI research taste when AI reaches superhuman coding ability',
                    'units': 'fraction'
                },
                'ai_research_taste_at_coding_automation_anchor_sd': {
                    'name': 'Max AI Research Taste (SD)',
                    'description': 'AI research taste at superhuman coder specified in human-range standard deviations',
                    'units': 'SD'
                },
                
                'ai_research_taste_slope': {
                    'name': 'AI Research Taste Slope',
                    'description': 'Steepness of AI research taste curve',
                    'units': 'dimensionless'
                },
                'taste_schedule_type': {
                    'name': 'AI Research Taste Schedule Type',
                    'description': 'Unit convention for SD-based schedule',
                    'units': 'categorical'
                },
                'progress_at_aa': {
                    'name': 'Progress at AC',
                    'description': 'Progress level where AI reaches superhuman coding ability (exponential mode)',
                    'units': 'dimensionless'
                },
                'ac_time_horizon_minutes': {
                    'name': 'Time Horizon to AC',
                    'description': 'Time horizon length corresponding to superhuman coder achievement',
                    'units': 'minutes'
                },
                'include_gap': {
                    'name': 'Include Gap',
                    'description': 'Choose whether to include a gap after pre-gap SC horizon',
                    'units': 'choice (gap | no gap)'
                },
                'gap_years': {
                    'name': 'Gap Size (YEAR-progress-years)',
                    'description': 'Additional gap after saturation horizon to reach SC, measured in YEAR-progress-years (converted using anchor_progress_rate). UI displays YEAR based on Anchor Time.',
                    'units': 'YEAR-progress-years'
                },
                'horizon_extrapolation_type': {
                    'name': 'Horizon Extrapolation Type',
                    'description': 'Method for extrapolating progress beyond the time horizon',
                    'units': 'categorical'
                },
                'parallel_penalty': {
                    'name': 'Parallel Penalty',
                    'description': 'Penalty applied to parallel coding labor contribution in experiment capacity calculations',
                    'units': 'dimensionless'
                },
                # Manual horizon fitting parameters
                'present_day': {
                    'name': 'Anchor Time',
                    'description': 'Reference time point for manual horizon fitting',
                    'units': 'year'
                },
                'present_horizon': {
                    'name': 'Anchor Horizon',
                    'description': 'Time horizon length at the anchor time (leave empty for auto-fit)',
                    'units': 'minutes'
                },
                'present_doubling_time': {
                    'name': 'Anchor Doubling Time',
                    'description': 'Doubling time parameter at the anchor point (leave empty for auto-fit)',
                    'units': 'progress units'
                },
                'doubling_difficulty_growth_factor': {
                    'name': 'Doubling Difficulty Growth Rate',
                    'description': 'Rate of growth for doubling difficulty (1 - decay rate, leave empty for auto-fit)',
                    'units': 'dimensionless'
                },
                # exp capacity pseudoparameters
                'inf_labor_asymptote': {
                    'name': 'Infinite Labor Asymptote',
                    'description': 'Asymptotic experiment capacity as labor → ∞ (holding compute finite)',
                    'units': 'capacity units'
                },
                'inf_compute_asymptote': {
                    'name': 'Infinite Compute Asymptote',
                    'description': 'Asymptotic experiment capacity as compute → ∞ (holding labor finite)',
                    'units': 'capacity units'
                },
                'labor_anchor_exp_cap': {
                    'name': 'Labor Anchor Experiment Capacity',
                    'description': 'Experiment capacity at the labor anchor used to infer CES parameters',
                    'units': 'capacity units'
                },
                'compute_anchor_exp_cap': {
                    'name': 'Compute Anchor Experiment Capacity',
                    'description': 'Experiment capacity at the compute anchor used to infer CES parameters (direct form, deprecated in UI)',
                    'units': 'capacity units'
                },
                'inv_compute_anchor_exp_cap': {
                    'name': 'Exp. Cap. Reduction with 0.1x compute',
                    'description': 'Inverse form: capacity at 0.1x compute relative to baseline (i.e., reduction factor). The model uses 1 / value internally for the direct anchor.',
                    'units': 'x relative to baseline'
                },
                'median_to_top_taste_multiplier': {
                    'name': 'Median to Top Taste Multiplier',
                    'description': 'Ratio of top percentile researcher taste to median researcher taste',
                    'units': 'ratio'
                },
                'strat_ai_m2b': {
                    'name': 'STRAT-AI M2Bs',
                    'description': 'Number of median-to-bests above SAR for STRAT-AI milestone',
                    'units': 'M2Bs'
                },
                'ted_ai_m2b': {
                    'name': 'TED-AI M2Bs',
                    'description': 'Number of median-to-bests above SAR for TED-AI milestone (ASI is always 2 M2Bs above TED-AI)',
                    'units': 'M2Bs'
                }
            }
        }

        return jsonify({
            'success': True,
            'config': parameter_config
        })
        
    except Exception as e:
        logger.error(f"Error getting parameter config: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/upload-data', methods=['POST'])
def upload_data():
    """Upload custom time series data"""
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected'}), 400
        
        # Read CSV data
        stream = io.StringIO(file.stream.read().decode("UTF8"), newline=None)
        csv_input = csv.DictReader(stream)
        data = list(csv_input)
        
        # Parse data
        time = np.array([float(row['time']) for row in data])
        L_HUMAN = np.array([float(row['L_HUMAN']) for row in data])
        inference_compute = np.array([float(row['inference_compute']) for row in data])
        experiment_compute = np.array([float(row['experiment_compute']) for row in data])
        training_compute_growth_rate = np.array([float(row['training_compute_growth_rate']) for row in data])
        
        time_series = TimeSeriesData(time, L_HUMAN, inference_compute, experiment_compute, training_compute_growth_rate)
        
        return jsonify({
            'success': True,
            'data_summary': {
                'time_range': [float(time.min()), float(time.max())],
                'data_points': len(time),
                'preview': time_series_to_dict(time_series)
            }
        })
        
    except Exception as e:
        logger.error(f"Error uploading data: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/select-data', methods=['POST'])
def select_data():
    """Select a built-in dataset by actor and update session time series"""
    try:
        data = request.json or {}
        dataset = data.get('dataset', 'default')

        import os
        base_dir = os.path.dirname(__file__)
        inputs_dir = os.path.join(base_dir, 'inputs')

        # Map datasets to absolute file paths. Note: the "default" CSV lives at project root.
        dataset_to_file = {
            'default': os.path.join(base_dir, 'input_data.csv'),
            'updated_default': os.path.join(inputs_dir, 'updated_default.csv'),
            'truncated_default': os.path.join(inputs_dir, 'truncated_default.csv'),
            'pretrain_russia': os.path.join(inputs_dir, 'pretrain_russia.csv'),
            'pretrain_us_black_site': os.path.join(inputs_dir, 'pretrain_us_black_site.csv'),
            'pretrain_cn_black_site': os.path.join(inputs_dir, 'pretrain_cn_black_site.csv'),
            'rl_russia': os.path.join(inputs_dir, 'rl_russia.csv'),
            'rl_us_black_site': os.path.join(inputs_dir, 'rl_us_black_site.csv'),
            'rl_cn_black_site': os.path.join(inputs_dir, 'rl_cn_black_site.csv')
        }

        if dataset not in dataset_to_file:
            return jsonify({'success': False, 'error': f"Unknown dataset: {dataset}"}), 400

        csv_path = dataset_to_file[dataset]

        if not os.path.exists(csv_path):
            return jsonify({'success': False, 'error': f"Dataset file not found: {os.path.basename(csv_path)}"}), 400

        time_series = load_time_series_from_csv_path(csv_path)

        time_array = time_series.time
        return jsonify({
            'success': True,
            'data_summary': {
                'time_range': [float(time_array.min()), float(time_array.max())],
                'data_points': len(time_array),
                'preview': time_series_to_dict(time_series)
            }
        })
    except Exception as e:
        logger.error(f"Error selecting dataset: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/sampling-config')
def get_sampling_config():
    """Get the sampling configuration for parameter uncertainty"""
    try:
        import yaml
        import os
        config_path = os.path.join(os.path.dirname(__file__), '../config/sampling_config.yaml')
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        return jsonify({
            'success': True,
            'config': config
        })
    except Exception as e:
        logger.error(f"Error loading sampling config: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/default-data')
def get_default_data():
    """Get default time series data and parameters"""
    time_series = create_default_time_series()
    params = create_default_parameters()
    
    return jsonify({
        'time_series': time_series_to_dict(time_series),
        'parameters': {
            'rho_coding_labor': params.rho_coding_labor,
            'rho_experiment_capacity': params.rho_experiment_capacity,
            'alpha_experiment_capacity': params.alpha_experiment_capacity,
            'r_software': params.r_software,
            'automation_fraction_at_coding_automation_anchor': params.automation_fraction_at_coding_automation_anchor,
            'coding_labor_normalization': params.coding_labor_normalization,
            'experiment_compute_exponent': params.experiment_compute_exponent,
            'ai_research_taste_at_coding_automation_anchor': params.ai_research_taste_at_coding_automation_anchor,
            'ai_research_taste_slope': params.ai_research_taste_slope,
            'taste_schedule_type': params.taste_schedule_type,
            'progress_at_aa': params.progress_at_aa,
            'ac_time_horizon_minutes': params.ac_time_horizon_minutes,
            'horizon_extrapolation_type': params.horizon_extrapolation_type,
            # Manual horizon fitting parameters
            'present_day': params.present_day,
            'present_horizon': params.present_horizon,
            'present_doubling_time': params.present_doubling_time,
            'doubling_difficulty_growth_factor': params.doubling_difficulty_growth_factor,
            # Research taste distribution parameter
            'median_to_top_taste_multiplier': params.median_to_top_taste_multiplier,
            # Median-to-best multipliers for general capability milestones
            'strat_ai_m2b': params.strat_ai_m2b,
            'ted_ai_m2b': params.ted_ai_m2b
        }
    })

import os
if __name__ == '__main__':
    # Get port from environment variable for deployment platforms
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_ENV') == 'development'
    
    app.run(debug=debug, host='0.0.0.0', port=port)
