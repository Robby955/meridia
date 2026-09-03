"""Demography layer v0: the country ages year by year.

One call advances the population by a year: everyone ages, deaths occur by an
age-specific mortality curve (Gompertz-Makeham shape), births arrive to women of
childbearing age and join the mother's household, and a share of young adults leave home
and move, mostly toward the cities. Every event is recorded in a vital-events register,
and the accounting is exact: next year's population equals this year's plus births minus
deaths, tested to the person. Life tables are emergent, not assumed: expectancy comes
out of the simulated deaths.

Deterministic in (seed, year, inputs).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .mechanisms import newborn_frailty


@dataclass(frozen=True)
class DemographyParams:
    makeham: float = 0.0004          # age-independent hazard floor
    gompertz_a: float = 0.000022     # q(x) ~ makeham + a * exp(b x)
    gompertz_b: float = 0.105
    fertility_rate: float = 0.085    # annual birth probability per woman 18-45
    leave_home_rate: float = 0.16    # annual chance a 18-30 child leaves home
    move_city_prob: float = 0.65     # movers head to a settlement cell
    infant_extra: float = 0.003      # added first-year mortality


def mortality_probability(age: np.ndarray, params: DemographyParams) -> np.ndarray:
    q = params.makeham + params.gompertz_a * np.exp(params.gompertz_b * age.astype(np.float64))
    q = np.where(age == 0, q + params.infant_extra, q)
    return np.clip(q, 0.0, 1.0)


# The public shock family. An epidemic year moves deaths and hospital admissions
# together, because it is one event: a schedule that raised mortality and left admissions
# alone would make the liability's own systematic risk unobservable in the health source.
# Fields inside one kind share a single draw, so the two multipliers move as one.
SHOCK_FAMILY = {
    "mortality_spike": {"mortality_multiplier": (1.5, 3.0),    # epidemic or disaster year
                        "admission_multiplier": (1.4, 2.6)},
    "migration_wave": {"leave_home_multiplier": (1.8, 3.0)},   # upheaval; the young move
    "baby_bust": {"fertility_multiplier": (0.45, 0.75)},       # crisis-year fertility drop
}

# One shock year in five, the rate the packet publishes and the five-year experience file
# carries roughly one realization of. It is the systematic risk in the liability: without
# it a continuation differs from its neighbours only by demographic noise, which on a
# population this size is a fraction of a percent and far under any achievable
# reconstruction error, and the sealed tail would be a target no method could reach.
ANNUAL_SHOCK_RATE = 0.20

# Each region's published loading on that family. A shock year is one national event, but
# its bite is not the same everywhere, and a world whose regions all take the shock at
# full strength has a reserve problem that a sum of six marginal tails already answers.
# With loadings the regional liabilities are correlated through a structure a method has
# to estimate: two regions move together in proportion to the product of their loadings,
# and the aggregate tail is wider or narrower than the marginals imply. The band is
# public and every world draws its own vector inside it; the realized values are
# retained. Development worlds expose them through the experience file, where a shock
# year shows as a state-specific jump in deaths and in first qualifying events.
SHOCK_LOADING_BAND = (0.35, 1.80)


def draw_shock_loadings(seed: int, n_regions: int,
                        band: tuple[float, float] = SHOCK_LOADING_BAND) -> np.ndarray:
    """One loading per region, drawn once per world on its own stream."""
    if int(n_regions) < 1:
        raise ValueError("n_regions must be positive")
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), 0x5A0E]))
    return rng.uniform(float(band[0]), float(band[1]), size=int(n_regions))


def regional_multiplier(multiplier: float, loading: np.ndarray) -> np.ndarray:
    """A national shock multiplier as it lands where the loading is ``loading``.

    A loading of one takes the national multiplier unchanged and a loading of zero takes
    none of it, in either direction, so a fertility multiplier under one thins births
    less where the loading is low. At ``multiplier`` of one the result is exactly one
    everywhere, which is what keeps a shock-free month identical to what it was before
    the loadings existed.
    """
    return 1.0 + np.asarray(loading, dtype=np.float64) * (float(multiplier) - 1.0)


def draw_annual_shocks(rng: np.random.Generator, first_year: int, n_years: int,
                       annual_rate: float = ANNUAL_SHOCK_RATE) -> list[dict]:
    """Independent shock years from the declared family, one Bernoulli draw per year."""
    kinds = sorted(SHOCK_FAMILY)
    shocks = []
    for year in range(int(first_year), int(first_year) + int(n_years)):
        if rng.random() >= float(annual_rate):
            continue
        kind = kinds[int(rng.integers(0, len(kinds)))]
        draw = float(rng.random())
        shock = {"year": int(year), "kind": kind}
        for field, (lo, hi) in SHOCK_FAMILY[kind].items():
            shock[field] = float(lo + draw * (hi - lo))
        shocks.append(shock)
    return sorted(shocks, key=lambda s: s["year"])


def draw_world_shocks(seed: int, years: int, max_shocks: int = 2,
                      annual_rate: float | None = None) -> list[dict]:
    """Seeded shock schedule from the declared family; retained truth, sealed for eval.

    With ``annual_rate`` the schedule is one independent draw per year, which is the law a
    continuation member redraws its own future from. Without it the older bounded rule
    stands, which is what the forecast task and the standalone ledger tests use.
    """
    rng = np.random.default_rng(np.random.SeedSequence([seed, 0x5A0C]))
    if annual_rate is not None:
        return draw_annual_shocks(rng, 0, max(int(years), 1), annual_rate)
    n_shocks = int(rng.integers(0, max_shocks + 1))
    kinds = sorted(SHOCK_FAMILY)
    shocks = []
    for _ in range(n_shocks):
        kind = kinds[int(rng.integers(0, len(kinds)))]
        draw = float(rng.random())
        shock = {"year": int(rng.integers(2, max(years, 3))), "kind": kind}
        for field, (lo, hi) in SHOCK_FAMILY[kind].items():
            shock[field] = float(lo + draw * (hi - lo))
        shocks.append(shock)
    return sorted(shocks, key=lambda s: s["year"])


def _shocked_params(params: DemographyParams, shocks: list[dict], year: int) -> DemographyParams:
    from dataclasses import replace
    out = params
    for shock in shocks:
        if shock["year"] != year:
            continue
        if "mortality_multiplier" in shock:
            out = replace(out, makeham=out.makeham * shock["mortality_multiplier"],
                          gompertz_a=out.gompertz_a * shock["mortality_multiplier"])
        if "leave_home_multiplier" in shock:
            out = replace(out, leave_home_rate=min(0.9, out.leave_home_rate * shock["leave_home_multiplier"]))
        if "fertility_multiplier" in shock:
            out = replace(out, fertility_rate=out.fertility_rate * shock["fertility_multiplier"])
    return out


def step_year(person: dict, household_cell: np.ndarray, urbanity_flat: np.ndarray,
              seed: int, year: int,
              params: DemographyParams = DemographyParams()) -> tuple[dict, np.ndarray, dict]:
    """Advance one year. Returns (person, household_cell, register)."""
    rng = np.random.default_rng(np.random.SeedSequence([seed, 0xDE40, year]))
    n = len(person["age"])

    # Deaths, then aging of survivors.
    q = mortality_probability(person["age"], params)
    dies = rng.random(n) < q
    survivors = ~dies
    new = {k: v[survivors].copy() for k, v in person.items()}
    new["age"] = new["age"] + 1

    # Births: women 18-45 among survivors; newborns join the mother's household.
    mothers = np.flatnonzero((new["sex"] == 1) & (new["age"] >= 18) & (new["age"] <= 45))
    gives_birth = mothers[rng.random(len(mothers)) < params.fertility_rate]
    n_births = len(gives_birth)
    if n_births:
        babies = {
            "household": new["household"][gives_birth],
            "cell": new["cell"][gives_birth],
            "age": np.zeros(n_births, dtype=new["age"].dtype),
            "sex": (rng.random(n_births) < 0.5).astype(new["sex"].dtype),
            "role": np.full(n_births, 2, dtype=new["role"].dtype),
            "education": np.zeros(n_births, dtype=new["education"].dtype),
            "income": np.zeros(n_births, dtype=new["income"].dtype),
            "frailty": newborn_frailty(new["frailty"][gives_birth],
                                       rng.normal(0.0, 1.0, size=n_births)),
        }
        new = {k: np.concatenate([new[k], babies[k]]) for k in new}

    # Young adults leave home: new one-person household, often in a more urban cell.
    household_cell = household_cell.copy()
    at_home = np.flatnonzero((new["role"] == 2) & (new["age"] >= 18) & (new["age"] <= 30))
    movers = at_home[rng.random(len(at_home)) < params.leave_home_rate]
    n_hh = len(household_cell)
    if len(movers):
        populated = np.flatnonzero(np.bincount(household_cell, minlength=len(urbanity_flat)) > 0)
        weights = urbanity_flat[populated] + 0.02
        weights = weights / weights.sum()
        dest = np.where(rng.random(len(movers)) < params.move_city_prob,
                        rng.choice(populated, size=len(movers), p=weights),
                        new["cell"][movers])
        new_cells = []
        for j, p_idx in enumerate(movers):
            new["household"][p_idx] = n_hh + j
            new["role"][p_idx] = 0
            new["cell"][p_idx] = dest[j]
            new_cells.append(int(dest[j]))
        household_cell = np.concatenate([household_cell,
                                         np.asarray(new_cells, dtype=household_cell.dtype)])

    register = {
        "year": year,
        "deaths": int(dies.sum()),
        "death_ages": person["age"][dies].copy(),
        "births": n_births,
        "moves": int(len(movers)),
        "population_start": n,
        "population_end": len(new["age"]),
    }
    return new, household_cell, register


def run_years(person: dict, household_cell: np.ndarray, urbanity_flat: np.ndarray,
              seed: int, years: int,
              params: DemographyParams = DemographyParams(),
              shocks: list[dict] | None = None) -> tuple[dict, np.ndarray, list[dict]]:
    registers = []
    for year in range(years):
        year_params = _shocked_params(params, shocks or [], year)
        person, household_cell, register = step_year(
            person, household_cell, urbanity_flat, seed, year, year_params)
        register["shocked"] = year_params is not params
        registers.append(register)
    return person, household_cell, registers


def period_life_expectancy(params: DemographyParams = DemographyParams()) -> float:
    """Implied period expectancy of the mortality curve (cohort of 100k, ages 0-110)."""
    ages = np.arange(0, 111)
    q = mortality_probability(ages, params)
    survival = np.cumprod(1.0 - q)
    return float(0.5 + survival[:-1].sum() + 0.5 * survival[-1])
