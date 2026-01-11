"""
Base class for parameter specifications with sampling support.

Provides common methods for parsing from YAML, sampling distributions,
and getting modal values.
"""

from dataclasses import dataclass
from typing import Any, Dict, TypeVar, Type
import numpy as np

from parameters.distribution_spec import (
    ParamValue,
    parse_param_value,
    sample_param,
    get_modal_param,
)


T = TypeVar('T', bound='BaseSpec')


def _parse_dict(d: Dict[str, Any]) -> Dict[str, ParamValue]:
    """Parse all values in a dict to ParamValues."""
    return {k: parse_param_value(v) for k, v in d.items()}


def _sample_dict(d: Dict[str, ParamValue], rng: np.random.Generator) -> Dict[str, Any]:
    """Sample all ParamValues in a dict."""
    return {k: sample_param(v, rng, k) for k, v in d.items()}


def _modal_dict(d: Dict[str, ParamValue]) -> Dict[str, Any]:
    """Get modal values for all ParamValues in a dict."""
    return {k: get_modal_param(v, k) for k, v in d.items()}


def _get_fields(obj: Any) -> Dict[str, Any]:
    """Get fields from a dataclass, excluding private fields."""
    return {k: v for k, v in obj.__dict__.items() if not k.startswith('_')}


@dataclass
class BaseSpec:
    """Base class for parameter specifications with common methods."""

    @classmethod
    def from_dict(cls: Type[T], d: Dict[str, Any]) -> T:
        """Create instance from dictionary, parsing distribution specs."""
        parsed = _parse_dict(d)
        return cls(**{k: v for k, v in parsed.items() if hasattr(cls, k)})

    def sample(self: T, rng: np.random.Generator) -> T:
        """Sample all distributions, returning new instance with concrete values."""
        return type(self)(**_sample_dict(_get_fields(self), rng))

    def get_modal(self: T) -> T:
        """Get modal values for all distributions, returning new instance."""
        return type(self)(**_modal_dict(_get_fields(self)))
