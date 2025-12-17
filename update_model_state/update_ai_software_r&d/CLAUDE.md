# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository contains a hybrid Next.js/Python application for modeling AI progress over time using nested CES (Constant Elasticity of Substitution) production functions. The system models a feedback loop where automation fraction depends on cumulative progress, which affects the progress rate.

## Development Commands

### Frontend (Next.js/TypeScript)
- `npm run dev` - Start development server with Next.js frontend and Flask API backend
- `npm run next-dev` - Start only Next.js frontend (development mode with Turbopack)
- `npm run build` - Build Next.js app for production
- `npm run start` - Start production Next.js server
- `npm run lint` - Run ESLint on TypeScript/JavaScript files
- `npm run mermaid` - Generate Mermaid flowchart from TypeScript files

**Note**: Next.js 16.0.4 has a bug causing `routes-manifest.json` errors. If you hit this, run `npm install next@16.0.5 --no-save` to fix it locally.

### Backend (Python/Flask)
- `npm run flask-dev` - Install Python dependencies and start Flask API on port 5328
- Python dependencies are managed via `requirements.txt` (numpy, scipy, pyyaml, Flask, Flask-Cors)

### Monte Carlo Scripts
- `python scripts/batch_rollout.py` - Sample parameter distributions and run batch trajectories (automatically calls plot_rollouts.py --batch-all)
- `python scripts/plot_rollouts.py` - Generate visualizations from rollout results
- `python scripts/plot_ac_sc_sar_pdfs.py` - Plot probability density functions for AC, SC, and SAR milestone times
- `python scripts/plot_post_sar_milestone_pdfs.py` - Plot PDFs for post-SAR milestones (SIAR, TED-AI, ASI) - automatically called by batch_rollout.py
- `python scripts/run_all.py` - One-command pipeline: batch rollout → plots → sensitivity
- `python scripts/sensitivity_analysis.py` - Analyze parameter sensitivity and correlations
- `python scripts/rollout_summary.py <run_dir>` - Generate success/failure summary for a specific run
- `python scripts/scan_all_rollouts.py` - Scan all runs in outputs/ and generate statistics table

**Note for creating new plotting scripts**: See "Plotting Script Conventions" section below for recommended utilities in `scripts/plotting_utils/` and `scripts/plotting/`.

### Running the Full Stack
Use `npm run dev` to start both Next.js and Flask concurrently for full development environment.

## Core Architecture

### Frontend (Next.js App Router)
- **Components**: `/components/` - React components for progress visualization
  - `ProgressChartServer.tsx` - Server-side data fetching and chart rendering
  - `ProgressChartClient.tsx` - Client-side interactive chart component with parameter sliders
  - `CustomHorizonChart.tsx`, `CustomMetricChart.tsx`, `CustomSliderChart.tsx` - Specialized chart types
  - `ParameterSlider.tsx` - Reusable parameter control component
- **API Routes**: `/app/api/` - Next.js API routes
  - `monte-carlo-dynamic/route.ts` - Monte Carlo trajectory generation with caching
- **Styling**: Tailwind CSS with custom vivid color scheme

### Backend (Python/Flask)
- **Main API**: `/api/index.py` - Flask application with CORS support
  - `/api/parameter-config` - Parameter bounds and defaults
  - `/api/compute` - Core model computation endpoint
- **Core Model**: Root-level Python modules
  - `progress_model.py` - Main ProgressModel class with trajectory computation
  - `model_config.py` - Parameter bounds, defaults, and configuration
  - Data structures: `TimeSeriesData`, `Parameters`, `AnchorConstraint`

### Model Components
1. **Production Functions**:
   - `compute_coding_labor()` - CES combination of AI and human labor
   - `compute_software_progress_rate()` - CES combination of compute and cognitive work
   - `compute_overall_progress_rate()` - Weighted average of software and training progress

2. **Core Logic**:
   - `compute_automation_fraction()` - Log-space interpolation between anchor points
   - `progress_rate_at_time()` - Computes instantaneous progress rate
   - `integrate_progress()` - Solves differential equation for progress over time

3. **Monte Carlo System**:
   - `monte_carlo.py` - Flask blueprint for web UI and job management
   - `config/sampling_config.yaml` - Parameter distribution configurations
   - `scripts/batch_rollout.py` - CLI for batch trajectory generation
   - Results stored in `outputs/<timestamp>/` directories

4. **Sampling Config Inheritance**:
   - Configs support inheritance via `parent` field for reusability
   - Child configs inherit all parent settings and can override specific values
   - Deep merge: nested dicts merge recursively, other types replace
   - Relative paths resolved relative to child config location
   - Circular inheritance detection prevents infinite loops
   - Example: `sampling_config_daniel.yaml` inherits from `sampling_config.yaml`
   - Usage: `python scripts/batch_rollout.py --config config/sampling_config_child.yaml`

5. **Plotting Script Conventions**:
   - **Template**: See `scripts/PLOTTING_TEMPLATE.py` for a complete example of recommended patterns
   - **Recommended imports for new plotting scripts:**
     ```python
     from plotting_utils.rollouts_reader import RolloutsReader
     from plotting_utils.kde import make_gamma_kernel_kde, make_lower_bounded_kde
     from plotting_utils.helpers import load_present_day
     ```
   - **Data Loading**: Use `RolloutsReader` class to avoid duplicating JSON parsing logic
     - `reader.read_milestone_times(milestone_name)` for milestone data
     - `reader.read_transition_durations(milestone_a, milestone_b)` for transitions
     - `reader.read_trajectories(metric_name)` for trajectory data
   - **KDE/Density**: Use functions from `plotting_utils/kde.py` for consistency
     - `make_gamma_kernel_kde()` for milestone timing distributions (recommended)
     - `make_lower_bounded_kde()` for log-space KDE alternative
     - `make_gaussian_kde()` for general-purpose distributions
     - Note: 2D density for scatter plot coloring can use scipy directly
   - **High-level plotting functions**: Check `scripts/plotting/` for reusable plot types (histograms, trajectories, boxplots, scatter)
   - **Before writing new code**: Check `scripts/plotting_utils/` and `scripts/plotting/` for existing utilities to avoid duplication
   - Some older scripts predate these utilities and may use different patterns

## Adding New Parameters

When adding new parameters to the model, update these files in order:

1. **model_config.py**: Add to `PARAMETER_BOUNDS` and `DEFAULT_PARAMETERS`
2. **progress_model.py**: Add to `Parameters` dataclass with validation
3. **api/index.py**: Update parameter serialization in `build_time_series_payload()`
4. **Frontend components**: Add controls in relevant chart components

For detailed instructions, see `.cursor/rules/adding-model-parameters.mdc`

## Technology Stack

### Frontend Dependencies
- **Framework**: Next.js 15.5.3 with App Router and Turbopack
- **React**: 19.1.0 with TypeScript
- **Styling**: Tailwind CSS with PostCSS and Autoprefixer
- **Charts**: Custom chart components (no Recharts dependency)
- **YAML**: js-yaml for configuration parsing
- **Caching**: Vercel KV for production, in-memory for development

### Backend Dependencies
- **API Framework**: Flask with Flask-CORS
- **Scientific Computing**: NumPy, SciPy for numerical computations and optimization
- **Data**: YAML configuration parsing
- **Deployment**: Vercel Python runtime

## Development Setup

1. **Prerequisites**: Node.js and Python 3.7+
2. **Install dependencies**: `npm install` (also installs Python deps via requirements.txt)
3. **Development**: `npm run dev` starts both frontend (port 3000) and backend (port 5328)
4. **Production**: Deploy to Vercel with automatic Python/Node.js runtime detection

## Key Implementation Notes

1. **API Integration**: Next.js frontend communicates with Flask backend via fetch calls
2. **Caching Strategy**: Monte Carlo trajectories cached for 4 hours (Vercel KV in production, memory in dev)
3. **Numerical Stability**: CES functions handle edge cases for extreme rho values
4. **Data Flow**: Time series data → Parameter estimation → Progress trajectory computation → Visualization
5. **Error Handling**: Comprehensive error handling with fallback to synthetic data
6. **TypeScript**: Strict TypeScript configuration with Next.js plugin integration
7. **Git Workflow**: Merge (not rebase) when pulling changes

## File Structure Notes

- Frontend routes follow Next.js App Router conventions (`app/` directory)
- Python modules are in root directory for import simplicity
- Configuration files: `next.config.ts`, `tailwind.config.ts`, `tsconfig.json`, `vercel.json`
- Input data: `input_data.csv` contains default time series, individual scenario CSVs in `inputs/`
- Monte Carlo outputs: `outputs/<timestamp>/` with rollouts, plots, and sensitivity analysis

## Important Reminders

- Progress in the model is measured in OOMs of effective compute
- Run lints and typecheck before pushing to deploy (deployment will fail if they don't pass)
- Don't disable linting or typechecking
- Background the development server when starting it
- Never regenerate golden test data without explicit user permission
- Never modify tests without explicit user permission