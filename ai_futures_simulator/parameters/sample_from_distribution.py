"""
Distribution sampling functions for Monte Carlo parameter sampling.

Supports distributions matching the ai_takeoff_model sampling config format:
- fixed: Single fixed value
- uniform: Uniform between min and max
- normal: Gaussian, specified via ci80 (80% CI) or mean/sd
- lognormal: Log-normal, specified via ci80 or mu/sigma
- shifted_lognormal: shift + LogNormal
- beta: Beta distribution scaled to [min, max]
- choice: Categorical with optional weights
- metalog: 3-term semi-bounded metalog, specified via p10/p50/p90 percentiles
"""

import numpy as np
import scipy.special
from typing import Dict, Any, Optional


def _metalog_3term_semi_bounded_quantile(p: float, q_lo: float, q_mid: float, q_hi: float,
                                          prob_lo: float = 0.1, prob_hi: float = 0.9) -> float:
    """
    Compute the quantile function for a 3-term semi-bounded metalog distribution.

    The semi-bounded metalog (bounded below at 0, unbounded above) uses a log transform.
    For a 3-term metalog with percentiles at prob_lo, 0.5, prob_hi, we fit coefficients a1, a2, a3
    such that the quantile function is: Q(y) = exp(a1 + a2*logit(y) + a3*(y - 0.5)*logit(y))
    where logit(y) = ln(y / (1-y)).

    Args:
        p: Quantile (probability) in (0, 1)
        q_lo: Lower percentile value (at prob_lo)
        q_mid: 50th percentile (median) value
        q_hi: Upper percentile value (at prob_hi)
        prob_lo: Lower probability (default 0.1 for p10)
        prob_hi: Upper probability (default 0.9 for p90)

    Returns:
        The value at quantile p
    """
    # Avoid boundary issues
    p = np.clip(p, 1e-10, 1 - 1e-10)

    # Log-transform the percentile values for semi-bounded metalog
    ln_q_lo = np.log(max(q_lo, 1e-10))
    ln_q_mid = np.log(max(q_mid, 1e-10))
    ln_q_hi = np.log(max(q_hi, 1e-10))

    # Logit function: logit(y) = ln(y / (1-y))
    def logit(y):
        return np.log(y / (1 - y))

    # From the median equation: a1 = ln(q_mid) (since logit(0.5) = 0)
    a1 = ln_q_mid

    # Logit values for prob_lo and prob_hi
    logit_lo = logit(prob_lo)
    logit_hi = logit(prob_hi)

    # (y - 0.5) * logit(y) terms
    term_lo = (prob_lo - 0.5) * logit_lo
    term_hi = (prob_hi - 0.5) * logit_hi

    # System of equations:
    # ln(q_lo) = a1 + a2*logit_lo + a3*term_lo
    # ln(q_hi) = a1 + a2*logit_hi + a3*term_hi

    rhs_lo = ln_q_lo - a1
    rhs_hi = ln_q_hi - a1

    # Solve 2x2 system: [logit_lo, term_lo; logit_hi, term_hi] * [a2; a3] = [rhs_lo; rhs_hi]
    det = logit_lo * term_hi - logit_hi * term_lo
    if abs(det) > 1e-10:
        a2 = (rhs_lo * term_hi - rhs_hi * term_lo) / det
        a3 = (logit_lo * rhs_hi - logit_hi * rhs_lo) / det
    else:
        a2 = 0
        a3 = 0

    # Now compute the quantile at probability p
    logit_p = logit(p)
    term_p = (p - 0.5) * logit_p

    ln_result = a1 + a2 * logit_p + a3 * term_p
    result = np.exp(ln_result)

    return max(result, 0)


def sample_from_distribution(
    dist_spec: Any,
    rng: np.random.Generator,
    param_name: Optional[str] = None
) -> Any:
    """
    Sample from a distribution specification.

    Args:
        dist_spec: Distribution specification dict, or a plain value (point estimate)
        rng: NumPy random generator
        param_name: Parameter name for error messages

    Returns:
        Sampled value
    """
    # Generate random quantile and use quantile-based sampling
    quantile = rng.uniform(0.0, 1.0)
    return sample_from_distribution_with_quantile(dist_spec, quantile, param_name)


def sample_from_distribution_with_quantile(
    dist_spec: Any,
    quantile: float,
    param_name: Optional[str] = None
) -> Any:
    """
    Sample from a distribution using a specific quantile (for correlated sampling).

    Args:
        dist_spec: Distribution specification dict, or a plain value (point estimate)
        quantile: Quantile value in [0, 1]
        param_name: Parameter name for error messages

    Returns:
        Sampled value
    """
    # Handle point estimates: if dist_spec is not a dict, treat it as a fixed value
    if not isinstance(dist_spec, dict):
        return dist_spec

    kind = dist_spec.get("dist", "fixed")

    if kind == "fixed":
        return dist_spec.get("value")

    if kind == "uniform":
        a = float(dist_spec["min"])
        b = float(dist_spec["max"])
        return a + quantile * (b - a)

    if kind == "normal":
        # Support parameterization by ci80 (80% CI) OR mean/sd
        if "ci80" in dist_spec:
            ci = dist_spec["ci80"]
            q10, q90 = float(ci[0]), float(ci[1])
            if q10 > q90:
                q10, q90 = q90, q10
            z = 1.2815515655446004  # norm.ppf(0.9)
            mu = 0.5 * (q10 + q90)
            sigma = (q90 - q10) / (2.0 * z)
        else:
            mu = float(dist_spec.get("mean", 0.0))
            sigma = float(dist_spec.get("sd", dist_spec.get("sigma", 1.0)))

        # Use inverse normal CDF
        x = mu + sigma * np.sqrt(2) * scipy.special.erfinv(2 * quantile - 1)

        if dist_spec.get("clip_to_bounds", True) and "min" in dist_spec and "max" in dist_spec:
            x = float(np.clip(x, float(dist_spec["min"]), float(dist_spec["max"])))
        return x

    if kind == "lognormal":
        # Support parameterization by ci80 in original space OR mu/sigma in log-space
        if "ci80" in dist_spec:
            ci = dist_spec["ci80"]
            q10, q90 = float(ci[0]), float(ci[1])
            if q10 > q90:
                q10, q90 = q90, q10
            z = 1.2815515655446004
            ln_q10, ln_q90 = np.log(q10), np.log(q90)
            mu = 0.5 * (ln_q10 + ln_q90)
            sigma = (ln_q90 - ln_q10) / (2.0 * z)
        else:
            mu = float(dist_spec.get("mu", 0.0))
            sigma = float(dist_spec.get("sigma", 1.0))

        # Use inverse lognormal CDF
        x = np.exp(mu + sigma * np.sqrt(2) * scipy.special.erfinv(2 * quantile - 1))

        if dist_spec.get("clip_to_bounds", True) and "min" in dist_spec and "max" in dist_spec:
            x = float(np.clip(x, float(dist_spec["min"]), float(dist_spec["max"])))
        return x

    if kind == "shifted_lognormal":
        # x = shift + LogNormal(mu, sigma)
        if "ci80" in dist_spec:
            ci = dist_spec["ci80"]
            q10, q90 = float(ci[0]), float(ci[1])
            if q10 > q90:
                q10, q90 = q90, q10
            z = 1.2815515655446004
            ln_q10, ln_q90 = np.log(q10), np.log(q90)
            mu = 0.5 * (ln_q10 + ln_q90)
            sigma = (ln_q90 - ln_q10) / (2.0 * z)
        else:
            mu = float(dist_spec.get("mu", 0.0))
            sigma = float(dist_spec.get("sigma", 1.0))

        x_core = np.exp(mu + sigma * np.sqrt(2) * scipy.special.erfinv(2 * quantile - 1))
        shift = float(dist_spec.get("shift", 0.0))
        x = shift + x_core

        if dist_spec.get("clip_to_bounds", True) and "min" in dist_spec and "max" in dist_spec:
            x = float(np.clip(x, float(dist_spec["min"]), float(dist_spec["max"])))
        return x

    if kind == "beta":
        a = float(dist_spec["alpha"])
        b = float(dist_spec["beta"])
        lo = float(dist_spec.get("min", 0.0))
        hi = float(dist_spec.get("max", 1.0))
        # Use inverse beta CDF
        x01 = scipy.special.betaincinv(a, b, quantile)
        return lo + (hi - lo) * x01

    if kind == "choice":
        values = dist_spec["values"]
        p = dist_spec.get("p")
        if p is None:
            # Uniform choice
            idx = int(quantile * len(values))
            idx = min(idx, len(values) - 1)
            return values[idx]
        else:
            # Weighted choice - find cumulative sum
            cumsum = np.cumsum(p)
            idx = np.searchsorted(cumsum, quantile)
            idx = min(idx, len(values) - 1)
            return values[idx]

    if kind == "metalog":
        # 3-term semi-bounded metalog with configurable percentiles
        # Can use p10/p50/p90 OR p25/p50/p75 to match different reference models
        p50_val = float(dist_spec["p50"])

        # Check for p25/p75 (reference model uses this)
        if "p25" in dist_spec and "p75" in dist_spec:
            p25_val = float(dist_spec["p25"])
            p75_val = float(dist_spec["p75"])
            x = _metalog_3term_semi_bounded_quantile(quantile, p25_val, p50_val, p75_val,
                                                     prob_lo=0.25, prob_hi=0.75)
        # Fallback to p10/p90 (original format)
        elif "p10" in dist_spec and "p90" in dist_spec:
            p10_val = float(dist_spec["p10"])
            p90_val = float(dist_spec["p90"])
            x = _metalog_3term_semi_bounded_quantile(quantile, p10_val, p50_val, p90_val,
                                                     prob_lo=0.1, prob_hi=0.9)
        else:
            raise ValueError(f"metalog requires either (p10, p50, p90) or (p25, p50, p75)")

        if dist_spec.get("clip_to_bounds", True) and "min" in dist_spec and "max" in dist_spec:
            x = float(np.clip(x, float(dist_spec["min"]), float(dist_spec["max"])))
        return x

    raise ValueError(f"Unknown distribution kind: {kind} for parameter {param_name}")
