"""World character: each world draws its own social parameters from declared ranges.

Geography already differs by seed; this layer makes societies differ too. A world's
character is a deterministic draw of the microdata and demography parameters from the
declared ranges below: how unequal income is, how urban wealth concentrates, how young
or old the population runs, how fast it grows, how dominant the largest city is. The
ranges are public; a sealed evaluation world's specific draw is not, so methods must
estimate a world's character from its data rather than assume constants.

Deterministic in the seed alone.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from .demography import DemographyParams
from .microdata import MicrodataParams
from .population import PopulationParams

# Declared ranges: (low, high) of a uniform draw per parameter.
CHARACTER_RANGES = {
    "income_sigma": (0.40, 0.85),          # income inequality (log-scale spread)
    "income_urban_bonus": (0.15, 0.55),    # urban wealth concentration
    "income_per_education": (0.25, 0.55),  # returns to education
    "education_urban_shift": (0.5, 1.4),   # urban-rural education gap
    "fertility_rate": (0.055, 0.115),      # young, growing vs old, shrinking nations
    "gompertz_a": (0.000012, 0.000040),    # mortality level (life expectancy ~70-84)
    "leave_home_rate": (0.10, 0.24),       # internal migration intensity
    "zipf_exponent": (0.75, 1.30),         # city-size primacy
    "background_share": (0.06, 0.20),      # rural share of settlement mass
}


def draw_world_character(seed: int) -> dict:
    """One deterministic draw of the world's social parameters from declared ranges."""
    rng = np.random.default_rng(np.random.SeedSequence([seed, 0xC4A2]))
    draw = {name: float(rng.uniform(lo, hi)) for name, (lo, hi) in CHARACTER_RANGES.items()}
    population = replace(PopulationParams(),
                         zipf_exponent=draw["zipf_exponent"],
                         background_share=draw["background_share"])
    microdata = replace(MicrodataParams(),
                        income_sigma=draw["income_sigma"],
                        income_urban_bonus=draw["income_urban_bonus"],
                        income_per_education=draw["income_per_education"],
                        education_urban_shift=draw["education_urban_shift"])
    demography = replace(DemographyParams(),
                         fertility_rate=draw["fertility_rate"],
                         gompertz_a=draw["gompertz_a"],
                         leave_home_rate=draw["leave_home_rate"])
    return {"draw": draw, "population": population, "microdata": microdata,
            "demography": demography}


def gini(income: np.ndarray) -> float:
    """Gini coefficient of nonnegative incomes (exact, sorted form)."""
    x = np.sort(income[income > 0].astype(np.float64))
    n = len(x)
    if n == 0:
        return 0.0
    ranks = np.arange(1, n + 1)
    return float((2.0 * (ranks * x).sum()) / (n * x.sum()) - (n + 1.0) / n)
