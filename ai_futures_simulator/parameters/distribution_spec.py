"""
Distribution specification classes for Monte Carlo sampling.

These classes provide type-safe representations of parameter distributions
that can be sampled to produce concrete values.
"""

from dataclasses import dataclass, field
from typing import Any, List, Optional, Union
import numpy as np

from parameters.sample_from_distribution import sample_from_distribution, get_modal_value


# A parameter value can be a concrete value or a distribution spec
ParamValue = Union[float, int, str, bool, None, "DistributionSpec"]


@dataclass
class DistributionSpec:
    """
    Specification for a probability distribution.

    Supports various distribution types:
    - fixed: {dist: "fixed", value: X}
    - uniform: {dist: "uniform", min: X, max: Y}
    - normal: {dist: "normal", ci80: [low, high]} or {mean: X, sd: Y}
    - lognormal: {dist: "lognormal", ci80: [low, high]}
    - shifted_lognormal: {dist: "shifted_lognormal", ci80: [low, high], shift: X}
    - beta: {dist: "beta", alpha: X, beta: Y, min: A, max: B}
    - choice: {dist: "choice", values: [...], p: [...]}
    - metalog: {dist: "metalog", p10: X, p50: Y, p90: Z}
    """
    dist: str

    # Common parameters
    value: Optional[Any] = None  # For fixed distributions
    min: Optional[float] = None
    max: Optional[float] = None
    modal: Optional[Any] = None  # Modal (most likely) value

    # Normal/lognormal parameters
    ci80: Optional[List[float]] = None
    mean: Optional[float] = None
    sd: Optional[float] = None

    # Shifted lognormal
    shift: Optional[float] = None

    # Beta distribution
    alpha: Optional[float] = None
    beta: Optional[float] = None
    clip_to_bounds: Optional[bool] = None

    # Choice distribution
    values: Optional[List[Any]] = None
    p: Optional[List[float]] = None

    # Metalog distribution
    p10: Optional[float] = None
    p25: Optional[float] = None
    p50: Optional[float] = None
    p75: Optional[float] = None
    p90: Optional[float] = None

    def to_dict(self) -> dict:
        """Convert to dictionary format for sample_from_distribution."""
        result = {"dist": self.dist}
        for key in ["value", "min", "max", "modal", "ci80", "mean", "sd",
                    "shift", "alpha", "beta", "clip_to_bounds", "values", "p",
                    "p10", "p25", "p50", "p75", "p90"]:
            val = getattr(self, key)
            if val is not None:
                result[key] = val
        return result

    @classmethod
    def from_dict(cls, d: dict) -> "DistributionSpec":
        """Create from dictionary."""
        if not isinstance(d, dict) or "dist" not in d:
            raise ValueError(f"Invalid distribution spec: {d}")
        return cls(
            dist=d["dist"],
            value=d.get("value"),
            min=d.get("min"),
            max=d.get("max"),
            modal=d.get("modal"),
            ci80=d.get("ci80"),
            mean=d.get("mean"),
            sd=d.get("sd"),
            shift=d.get("shift"),
            alpha=d.get("alpha"),
            beta=d.get("beta"),
            clip_to_bounds=d.get("clip_to_bounds"),
            values=d.get("values"),
            p=d.get("p"),
            p10=d.get("p10"),
            p25=d.get("p25"),
            p50=d.get("p50"),
            p75=d.get("p75"),
            p90=d.get("p90"),
        )

    def sample(self, rng: np.random.Generator, name: str = "param") -> Any:
        """Sample a value from this distribution."""
        return sample_from_distribution(self.to_dict(), rng, name)

    def get_modal(self, name: str = "param") -> Any:
        """Get the modal (most likely) value."""
        return get_modal_value(self.to_dict(), name)


def parse_param_value(value: Any) -> ParamValue:
    """
    Parse a value from YAML into either a concrete value or DistributionSpec.

    Args:
        value: Raw value from YAML (could be int, float, str, dict, etc.)

    Returns:
        Either the concrete value or a DistributionSpec if it's a distribution.
    """
    if isinstance(value, dict) and "dist" in value:
        return DistributionSpec.from_dict(value)
    return value


def sample_param(value: ParamValue, rng: np.random.Generator, name: str = "param") -> Any:
    """
    Sample a concrete value from a ParamValue.

    If it's already a concrete value, returns it unchanged.
    If it's a DistributionSpec, samples from the distribution.
    """
    if isinstance(value, DistributionSpec):
        return value.sample(rng, name)
    return value


def get_modal_param(value: ParamValue, name: str = "param") -> Any:
    """
    Get the modal value from a ParamValue.

    If it's already a concrete value, returns it unchanged.
    If it's a DistributionSpec, returns the modal value.
    """
    if isinstance(value, DistributionSpec):
        return value.get_modal(name)
    return value
