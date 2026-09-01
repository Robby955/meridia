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


SHOCK_FAMILY = {
    "mortality_spike": {"mortality_multiplier": (1.5, 3.0)},   # epidemic or disaster year
    "migration_wave": {"leave_home_multiplier": (1.8, 3.0)},   # upheaval; the young move
    "baby_bust": {"fertility_multiplier": (0.45, 0.75)},       # crisis-year fertility drop
}


def draw_world_shocks(seed: int, years: int, max_shocks: int = 2) -> list[dict]:
    """Seeded shock schedule from the declared family; retained truth, sealed for eval."""
    rng = np.random.default_rng(np.random.SeedSequence([seed, 0x5A0C]))
    n_shocks = int(rng.integers(0, max_shocks + 1))
    kinds = sorted(SHOCK_FAMILY)
    shocks = []
    for _ in range(n_shocks):
        kind = kinds[int(rng.integers(0, len(kinds)))]
        (field, (lo, hi)), = SHOCK_FAMILY[kind].items()
        shocks.append({"year": int(rng.integers(2, max(years, 3))), "kind": kind,
                       field: float(rng.uniform(lo, hi))})
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
