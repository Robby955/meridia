"""Terrain layer: seeded spectral elevation with ridge structure.

Elevation is built from three seeded components on a regular grid: a power-law spectral
noise field (large-scale relief), a ridged component (absolute value of a second spectral
field, giving mountain chains), and a continental gradient that lowers elevation toward one
seeded coastline direction so the map has a coast. Everything is float64, single-threaded,
and fully determined by (seed, height, width, params); the same inputs yield byte-identical
arrays on the same platform, which the world manifest records.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TerrainParams:
    spectral_exponent: float = 1.8      # power-law slope of the base relief spectrum
    ridge_weight: float = 0.45          # contribution of the ridged component
    ridge_exponent: float = 2.2         # spectrum slope of the ridge field (smoother, longer chains)
    continental_weight: float = 0.55    # strength of the coastward gradient
    sea_level_quantile: float = 0.35    # fraction of cells below sea level


def _spectral_field(rng: np.random.Generator, height: int, width: int, exponent: float) -> np.ndarray:
    """Real random field with an isotropic power-law amplitude spectrum, unit variance."""
    ky = np.fft.fftfreq(height)[:, None]
    kx = np.fft.rfftfreq(width)[None, :]
    radial = np.sqrt(ky * ky + kx * kx)
    radial[0, 0] = 1.0
    amplitude = radial ** (-exponent)
    amplitude[0, 0] = 0.0
    phase = rng.uniform(0.0, 2.0 * np.pi, size=amplitude.shape)
    spectrum = amplitude * np.exp(1j * phase)
    field = np.fft.irfft2(spectrum, s=(height, width))
    field -= field.mean()
    std = field.std()
    return field / std if std > 0 else field


def generate_elevation(seed: int, height: int, width: int, params: TerrainParams = TerrainParams()) -> dict:
    """Return elevation (float64, arbitrary units), sea level, and the land mask."""
    rng = np.random.default_rng(np.random.SeedSequence([seed, 0x7E22A1]))
    base = _spectral_field(rng, height, width, params.spectral_exponent)
    ridge_raw = _spectral_field(rng, height, width, params.ridge_exponent)
    ridges = 1.0 - np.abs(ridge_raw) / max(np.abs(ridge_raw).max(), 1e-12)
    angle = rng.uniform(0.0, 2.0 * np.pi)
    rows = np.linspace(-0.5, 0.5, height)[:, None]
    cols = np.linspace(-0.5, 0.5, width)[None, :]
    gradient = np.cos(angle) * rows + np.sin(angle) * cols
    elevation = base + params.ridge_weight * (ridges - ridges.mean()) + params.continental_weight * gradient
    sea_level = float(np.quantile(elevation, params.sea_level_quantile))
    return {
        "elevation": elevation,
        "sea_level": sea_level,
        "land": elevation > sea_level,
        "coast_angle": float(angle),
    }
