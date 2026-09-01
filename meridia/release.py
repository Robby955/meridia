"""Release contract v0: the estimands a national release must publish, and their truth.

The capstone task asks an agent to run a statistical office: observed records in, a
national release out. This module fixes what "a release" is. It declares the estimand
list (what is published), the levels (nation, state, county), the flat release schema,
the additivity rules, and the detailed table that disclosure control applies to. It also
computes every estimand's exact value from the retained population, which is what a
release is scored against.

Every estimand is a plain function of the person table and the administrative
geography. Nothing here reads a survey, a register, or a world-character parameter, so
the truth side and the participant-facing schema cannot drift apart.

Units: the nation is unit 0 at level "nation"; states are 0..S-1; counties 0..C-1.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

LEVELS = ("nation", "state", "county")
RELEASE_COLUMNS = ("estimand", "level", "unit", "estimate", "lower", "upper")

AGE_BANDS = ((0, 15), (16, 24), (25, 44), (45, 64), (65, 200))   # inclusive bounds
AGE_BAND_LABELS = ("0-15", "16-24", "25-44", "45-64", "65+")
SEX_LABELS = ("male", "female")   # person["sex"]: 0 male, 1 female
LOW_INCOME_FRACTION = 0.6         # of the national median household income


@dataclass(frozen=True)
class Estimand:
    id: str
    kind: str          # "count" (additive), "mean", "median", or "proportion"
    description: str

    @property
    def additive(self) -> bool:
        return self.kind == "count"


ESTIMANDS: tuple[Estimand, ...] = (
    Estimand("persons", "count", "resident persons"),
    Estimand("households", "count", "households"),
    Estimand("children_under_16", "count", "persons aged 0 to 15"),
    Estimand("elders_65_plus", "count", "persons aged 65 and over"),
    Estimand("median_household_income", "median", "median of household income"),
    Estimand("mean_income_adults", "mean", "mean income of persons aged 16 and over"),
    Estimand("tertiary_share_25_plus", "proportion",
             "share of persons aged 25 and over with tertiary or advanced education"),
    Estimand("low_income_household_share", "proportion",
             "share of households with income below 0.6 times the national median"),
)
ESTIMAND_IDS = tuple(e.id for e in ESTIMANDS)
ESTIMAND_BY_ID = {e.id: e for e in ESTIMANDS}


def unit_membership(person: dict, household_cell: np.ndarray, admin: dict) -> dict:
    """Per-person and per-household unit labels at every level."""
    county_flat = admin["county"].flatten()
    state_flat = admin["state"].flatten()
    person_county = county_flat[person["cell"]]
    household_county = county_flat[household_cell]
    if (person_county < 0).any() or (household_county < 0).any():
        raise ValueError("a person or household sits on a cell with no county")
    return {
        "nation": (np.zeros(len(person_county), dtype=np.int64),
                   np.zeros(len(household_county), dtype=np.int64), 1),
        "state": (state_flat[person["cell"]], state_flat[household_cell], admin["n_states"]),
        "county": (person_county, household_county, admin["n_counties"]),
    }


def _group_median(values: np.ndarray, groups: np.ndarray, n_groups: int) -> np.ndarray:
    out = np.full(n_groups, np.nan)
    order = np.argsort(groups, kind="stable")
    sorted_groups = groups[order]
    starts = np.searchsorted(sorted_groups, np.arange(n_groups))
    ends = np.searchsorted(sorted_groups, np.arange(n_groups), side="right")
    sorted_values = values[order]
    for g in range(n_groups):
        if ends[g] > starts[g]:
            out[g] = np.median(sorted_values[starts[g]:ends[g]])
    return out


def compute_truth(person: dict, household_cell: np.ndarray, admin: dict) -> dict:
    """Exact estimand values: {(estimand_id, level, unit): value}.

    Units with no members get NaN for means, medians, and proportions and 0 for counts;
    the release schema still requires a row for them.
    """
    age = person["age"].astype(np.int64)
    income = person["income"].astype(np.float64)
    education = person["education"].astype(np.int64)
    n_households = len(household_cell)
    household_income = np.zeros(n_households)
    np.add.at(household_income, person["household"], income)
    national_median_hh = float(np.median(household_income))
    low_income = household_income < LOW_INCOME_FRACTION * national_median_hh
    adults = age >= 16
    over_25 = age >= 25
    tertiary = education >= 2

    membership = unit_membership(person, household_cell, admin)
    truth: dict[tuple[str, str, int], float] = {}
    for level, (p_unit, h_unit, n_units) in membership.items():
        counts = np.bincount(p_unit, minlength=n_units).astype(np.int64)
        hh_counts = np.bincount(h_unit, minlength=n_units).astype(np.int64)
        children = np.bincount(p_unit, weights=(age <= 15), minlength=n_units)
        elders = np.bincount(p_unit, weights=(age >= 65), minlength=n_units)
        median_hh = _group_median(household_income, h_unit, n_units)
        adult_n = np.bincount(p_unit, weights=adults, minlength=n_units)
        adult_income = np.bincount(p_unit, weights=income * adults, minlength=n_units)
        over_25_n = np.bincount(p_unit, weights=over_25, minlength=n_units)
        tertiary_n = np.bincount(p_unit, weights=(over_25 & tertiary), minlength=n_units)
        low_n = np.bincount(h_unit, weights=low_income, minlength=n_units)
        with np.errstate(invalid="ignore", divide="ignore"):
            mean_income = np.where(adult_n > 0, adult_income / adult_n, np.nan)
            tertiary_share = np.where(over_25_n > 0, tertiary_n / over_25_n, np.nan)
            low_share = np.where(hh_counts > 0, low_n / hh_counts, np.nan)
        for u in range(n_units):
            truth[("persons", level, u)] = float(counts[u])
            truth[("households", level, u)] = float(hh_counts[u])
            truth[("children_under_16", level, u)] = float(round(children[u]))
            truth[("elders_65_plus", level, u)] = float(round(elders[u]))
            truth[("median_household_income", level, u)] = float(median_hh[u])
            truth[("mean_income_adults", level, u)] = float(mean_income[u])
            truth[("tertiary_share_25_plus", level, u)] = float(tertiary_share[u])
            truth[("low_income_household_share", level, u)] = float(low_share[u])
    return truth


def required_rows(admin: dict) -> set[tuple[str, str, int]]:
    """Every (estimand, level, unit) a complete release must contain exactly once."""
    units = {"nation": 1, "state": admin["n_states"], "county": admin["n_counties"]}
    return {(e, level, u) for e in ESTIMAND_IDS for level in LEVELS for u in range(units[level])}


def compute_detailed_table_truth(person: dict, admin: dict) -> np.ndarray:
    """Exact county x age-band x sex person counts, shape (C, 5, 2)."""
    county = admin["county"].flatten()[person["cell"]]
    age = person["age"].astype(np.int64)
    band = np.full(len(age), -1, dtype=np.int64)
    for b, (lo, hi) in enumerate(AGE_BANDS):
        band[(age >= lo) & (age <= hi)] = b
    if (band < 0).any():
        raise ValueError("an age falls outside every band")
    sex = person["sex"].astype(np.int64)
    table = np.zeros((admin["n_counties"], len(AGE_BANDS), 2), dtype=np.int64)
    np.add.at(table, (county, band, sex), 1)
    return table
