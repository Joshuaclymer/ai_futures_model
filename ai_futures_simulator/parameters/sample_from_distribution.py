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
"""

import numpy as np
import scipy.special
from typing import Dict, Any, Optional


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

    raise ValueError(f"Unknown distribution kind: {kind} for parameter {param_name}")
