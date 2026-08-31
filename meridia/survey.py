"""Survey instrument layer v0: designs, nonresponse, and measurement error.

This layer turns the world's microdata into survey products the way a statistical office
would produce them, while retaining the complete truth for verification. A survey is a
two-stage stratified sample: strata partition the nation geographically, cells are
sampled within strata with probability proportional to population, and households are
sampled within cells. True inclusion probabilities are recorded per responding household,
so design-based estimation is exactly checkable. On top of the clean sample sit the
planted pathologies, each with retained truth:

- unit nonresponse: logistic in age, income, and urbanity; the MNAR dial routes part of
  the mechanism through the *reported* variable itself,
- item nonresponse: per-variable missingness with its own mechanism,
- measurement error: multiplicative income misreporting and age heaping to multiples of
  five, applied to reported values only.

The participant-facing file carries reported values and design metadata; the truth
bundle carries true values, inclusion probabilities, response indicators, and error
flags. The difficulty dials are the mechanism coefficients. Deterministic in
(seed, inputs).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SurveyParams:
    n_strata_rows: int = 3               # geographic strata grid (rows x cols)
    n_strata_cols: int = 4
    cells_per_stratum: int = 24          # sampled cells (PSUs) per stratum
    households_per_cell: int = 12        # sampled households per sampled cell
    response_intercept: float = 1.4      # logistic unit-response model
    response_age: float = 0.015          # per year of head age, centered at 45
    response_income: float = -0.35       # per unit of log-income above the median (MNAR dial)
    response_urban: float = -0.5         # urban households respond less
    item_income_rate: float = 0.18       # base item-missingness for income
    item_income_mnar: float = 0.6        # extra logit toward missing for high incomes
    item_education_rate: float = 0.07
    income_error_sigma: float = 0.12     # multiplicative lognormal misreporting
    age_heaping_prob: float = 0.22       # ages reported to the nearest multiple of five


def assign_strata(height: int, width: int, params: SurveyParams) -> np.ndarray:
    """Geographic strata: a coarse grid over the map, labeled row-major."""
    rows = np.minimum(np.arange(height) * params.n_strata_rows // height, params.n_strata_rows - 1)
    cols = np.minimum(np.arange(width) * params.n_strata_cols // width, params.n_strata_cols - 1)
    return rows[:, None] * params.n_strata_cols + cols[None, :]


def draw_survey(micro: dict, population: np.ndarray, seed: int,
                params: SurveyParams = SurveyParams()) -> dict:
    """Draw one survey: sample, response, and reported values, with retained truth."""
    height, width = population.shape
    rng = np.random.default_rng(np.random.SeedSequence([seed, 0x5A11]))
    person = micro["person"]
    household_cell = micro["household_cell"]
    n_households = micro["n_households"]
    strata = assign_strata(height, width, params).flatten()
    urbanity_flat = micro["urbanity"].flatten()

    # Households indexed by cell for second-stage sampling.
    order = np.argsort(household_cell, kind="stable")
    sorted_cells = household_cell[order]
    cell_starts = np.searchsorted(sorted_cells, np.arange(height * width))
    cell_ends = np.searchsorted(sorted_cells, np.arange(height * width), side="right")

    cell_pop = population.flatten().astype(np.float64)
    sampled_households = []
    inclusion_prob = []
    for s in range(params.n_strata_rows * params.n_strata_cols):
        stratum_cells = np.flatnonzero((strata == s) & (cell_pop > 0))
        if len(stratum_cells) == 0:
            continue
        m = min(params.cells_per_stratum, len(stratum_cells))
        p_cell = cell_pop[stratum_cells] / cell_pop[stratum_cells].sum()
        chosen = rng.choice(stratum_cells, size=m, replace=False, p=p_cell)
        # PPS-without-replacement first-stage probabilities, approximated as min(1, m*p).
        first_stage = np.minimum(1.0, m * cell_pop[chosen] / cell_pop[stratum_cells].sum())
        for cell, p1 in zip(chosen, first_stage):
            hh_in_cell = order[cell_starts[cell]:cell_ends[cell]]
            k = min(params.households_per_cell, len(hh_in_cell))
            picked = rng.choice(hh_in_cell, size=k, replace=False)
            p2 = k / len(hh_in_cell)
            sampled_households.extend(int(h) for h in picked)
            inclusion_prob.extend(float(p1 * p2) for _ in range(k))
    sampled_households = np.asarray(sampled_households, dtype=np.int64)
    inclusion_prob = np.asarray(inclusion_prob, dtype=np.float64)

    # Household-level covariates for the response model.
    head_mask = person["role"] == 0
    head_age = np.zeros(n_households, dtype=np.float64)
    head_age[person["household"][head_mask]] = person["age"][head_mask]
    hh_income = np.bincount(person["household"], weights=person["income"],
                            minlength=n_households)
    hh_urb = urbanity_flat[household_cell]

    log_inc = np.log1p(hh_income[sampled_households])
    med = np.median(log_inc)
    logit = (params.response_intercept
             + params.response_age * (head_age[sampled_households] - 45.0)
             + params.response_income * (log_inc - med)
             + params.response_urban * hh_urb[sampled_households])
    responded = rng.random(len(sampled_households)) < 1.0 / (1.0 + np.exp(-logit))

    resp_hh = sampled_households[responded]
    resp_prob = inclusion_prob[responded]

    # Person-level survey file for responding households.
    in_resp = np.isin(person["household"], resp_hh)
    idx = np.flatnonzero(in_resp)
    true_age = person["age"][idx].astype(np.int16)
    true_income = person["income"][idx].astype(np.float64)
    true_education = person["education"][idx].astype(np.int8)

    reported_age = true_age.copy()
    heap = rng.random(len(idx)) < params.age_heaping_prob
    reported_age[heap] = (np.round(reported_age[heap] / 5.0) * 5).astype(np.int16)
    reported_income = np.round(
        true_income * np.exp(params.income_error_sigma * rng.normal(size=len(idx))), 2)
    reported_income[true_income == 0.0] = 0.0

    inc_logit = (np.log(1.0 / params.item_income_rate - 1.0)
                 - params.item_income_mnar * (np.log1p(true_income) - np.log1p(np.median(true_income[true_income > 0]))))
    income_missing = (rng.random(len(idx)) < 1.0 / (1.0 + np.exp(inc_logit))) & (true_age >= 16)
    education_missing = rng.random(len(idx)) < params.item_education_rate
    reported_income[income_missing] = np.nan
    reported_education = true_education.astype(np.float64)
    reported_education[education_missing] = np.nan

    hh_index = {int(h): j for j, h in enumerate(resp_hh)}
    design_weight = np.array([1.0 / resp_prob[hh_index[int(h)]]
                              for h in person["household"][idx]])

    return {
        "survey": {
            "household": person["household"][idx],
            "cell": person["cell"][idx],
            "stratum": strata[person["cell"][idx]],
            "design_weight": np.round(design_weight, 6),
            "age": reported_age,
            "sex": person["sex"][idx],
            "education": reported_education,
            "income": reported_income,
        },
        "truth": {
            "person_index": idx,
            "age": true_age,
            "income": true_income,
            "education": true_education,
            "income_missing": income_missing,
            "education_missing": education_missing,
            "age_heaped": heap,
            "sampled_households": sampled_households,
            "inclusion_prob": inclusion_prob,
            "responded": responded,
        },
        "n_sampled_households": len(sampled_households),
        "n_responding_households": int(responded.sum()),
    }
