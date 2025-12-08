from __future__ import annotations

import math
from contextlib import contextmanager
from typing import Any, Dict

import numpy as np
import yaml

Z_90 = 1.2815515655446004  # 90th percentile of the standard normal


def load_config(path: str) -> Dict[str, Any]:
    with open(path, 'r', encoding='utf-8') as handle:
        return yaml.safe_load(handle) or {}


def build_default_distribution_config() -> Dict[str, Any]:
    return {}


def _normal_params(cfg: Dict[str, Any]) -> tuple[float, float]:
    if 'mean' in cfg and ('sd' in cfg or 'sigma' in cfg):
        return float(cfg['mean']), float(cfg.get('sd', cfg.get('sigma', 1.0)))
    if 'ci80' in cfg:
        low, high = map(float, cfg['ci80'])
    elif 'ci80_low' in cfg and 'ci80_high' in cfg:
        low = float(cfg['ci80_low'])
        high = float(cfg['ci80_high'])
    else:
        raise ValueError('Normal distribution requires mean/sd or ci80 bounds')
    mean = (low + high) / 2.0
    sd = (high - low) / (2.0 * Z_90)
    return mean, max(sd, 1e-9)


def _lognormal_params(cfg: Dict[str, Any], shift: float = 0.0) -> tuple[float, float]:
    if 'mu' in cfg and ('sigma' in cfg or 'sd' in cfg):
        mu = float(cfg['mu'])
        sigma = float(cfg.get('sigma', cfg.get('sd', 1.0)))
        return mu, max(sigma, 1e-9)
    if 'ci80' in cfg:
        low, high = map(float, cfg['ci80'])
    elif 'ci80_low' in cfg and 'ci80_high' in cfg:
        low = float(cfg['ci80_low'])
        high = float(cfg['ci80_high'])
    else:
        raise ValueError('Lognormal distribution requires mu/sigma or ci80 bounds')
    low = max(low - shift, 1e-12)
    high = max(high - shift, low + 1e-12)
    mu = (math.log(low) + math.log(high)) / 2.0
    sigma = (math.log(high) - math.log(low)) / (2.0 * Z_90)
    return mu, max(sigma, 1e-9)


def _apply_bounds(value: float, cfg: Dict[str, Any]) -> float:
    if cfg.get('clip_to_bounds') and cfg.get('min') is not None and cfg.get('max') is not None:
        return float(max(cfg['min'], min(cfg['max'], value)))
    return value


def sample_parameter_dict(cfg_dict: Dict[str, Any], rng: np.random.Generator) -> Dict[str, Any]:
    sampled: Dict[str, Any] = {}
    for name, cfg in cfg_dict.items():
        dist = cfg.get('dist', 'fixed')
        if dist == 'fixed':
            sampled[name] = cfg.get('value')
        elif dist == 'choice':
            values = cfg.get('values', [])
            probs = cfg.get('p')
            if not values:
                raise ValueError(f"Choice distribution for {name} has no values")
            sampled[name] = rng.choice(values, p=probs)
        elif dist == 'uniform':
            lo = float(cfg['min'])
            hi = float(cfg['max'])
            sampled[name] = rng.uniform(lo, hi)
        elif dist == 'normal':
            mean, sd = _normal_params(cfg)
            value = float(rng.normal(mean, sd))
            sampled[name] = _apply_bounds(value, cfg)
        elif dist == 'lognormal':
            mu, sigma = _lognormal_params(cfg)
            value = float(rng.lognormal(mu, sigma))
            sampled[name] = _apply_bounds(value, cfg)
        elif dist == 'shifted_lognormal':
            shift = float(cfg.get('shift', 0.0))
            mu, sigma = _lognormal_params(cfg, shift=shift)
            value = float(rng.lognormal(mu, sigma) + shift)
            sampled[name] = _apply_bounds(value, cfg)
        elif dist == 'beta':
            alpha = float(cfg['alpha'])
            beta = float(cfg['beta'])
            lo = float(cfg.get('min', 0.0))
            hi = float(cfg.get('max', 1.0))
            raw = rng.beta(alpha, beta)
            value = lo + (hi - lo) * raw
            sampled[name] = _apply_bounds(value, cfg)
        else:
            raise ValueError(f"Unsupported distribution type: {dist}")
    return sampled


@contextmanager
def suppress_noise():
    yield
