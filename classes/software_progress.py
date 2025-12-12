class AISoftwareProgress:
    # AI multipliers
    ai_coding_labor_multiplier: float
    ai_sw_progress_mult_ref_present_day: float # Software progress multiplier ref to present_day resources

    # Time horizon (if horizon trajectory is set)
    horizon_length: float | None = None

    # Rates
    progress_rate: float
    software_progress_rate: float
    research_effort: float

    # Automation
    automation_fraction: float
    coding_labor: float
    serial_coding_labor: float

    # Taste
    ai_research_taste: float
    ai_research_taste_sd: float
    aggregate_research_taste: float

    # Core takeoff model state
    progress: float # (used internally in the takeoff model, I'm not sure what it represents exactly)
    research_stock: float

class AISoftwareCapabilityCap:
    cap_of_ai_sw_progress_mult_ref_present_day: float | None = None
    # cap_of_time_horizon: float | None = None  # in years # not implemented currently