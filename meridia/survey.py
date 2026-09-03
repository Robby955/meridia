"""Survey instrument layer v0: designs, nonresponse, and measurement error.

This layer turns the world's microdata into survey files the way real surveys are
collected, while retaining the complete truth for verification. A survey is a
two-stage stratified sample: strata partition the nation geographically, cells are
sampled within strata with probability proportional to population, and households are
sampled within cells. True inclusion probabilities are recorded per responding household,
so design-based estimation is exactly checkable. On top of the clean sample sit the
planted pathologies, each with retained truth:

- unit nonresponse: logistic in age, income, and urbanity; the MNAR dial routes part of
  the mechanism through the *reported* variable itself,
- item nonresponse: per-variable missingness with its own mechanism,
- measurement error: multiplicative income misreporting and age heaping to multiples of
  five, applied to reported values only,
- one health anchor: a recent-hospitalization item reported with a declared sensitivity
  and specificity, which is what keeps informative health-source inclusion identifiable.

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
    anchor_sensitivity: float = 0.82     # health anchor: reported given a true admission
    anchor_specificity: float = 0.93     # health anchor: not reported given no admission


# Published bands for the instrument's own mechanism, one continuous draw per world.
# Version four's first pass drew twenty source rates per world and left these nine at
# their dataclass defaults, so the whole nonresponse and measurement model was the same
# in every world and estimable exactly on the development worlds, which ship truth. The
# form is public and written into the packet contract; a world's realized values are
# retained. The anchor's sensitivity and specificity are deliberately not in this table:
# they are declared to the participant, which is what makes the anchor an anchor.
SURVEY_BANDS = {
    "response_intercept": (0.90, 1.95),
    "response_age": (0.004, 0.026),
    "response_income": (-0.62, -0.10),
    "response_urban": (-0.85, -0.15),
    "item_income_rate": (0.10, 0.27),
    "item_income_mnar": (0.25, 0.95),
    "item_education_rate": (0.030, 0.115),
    "income_error_sigma": (0.06, 0.19),
    "age_heaping_prob": (0.10, 0.35),
}

# The instrument's public plausibility envelope, wider than the development band on both
# sides of every axis. A hidden world places at most two of its nine survey axes between
# the development band and this edge, which is the same arrangement the mechanism layer
# uses for its six regime axes. Without it the whole instrument was inside one band that
# twelve worlds shipping truth covered densely, so a nonresponse and measurement model
# fitted there transferred to a world nobody had seen. Every value in the envelope keeps
# the instrument well formed: the rates stay strictly inside zero and one and the money
# error stays positive.
SURVEY_ENVELOPE = {
    "response_intercept": (0.55, 2.45),
    "response_age": (-0.004, 0.038),
    "response_income": (-0.95, 0.02),
    "response_urban": (-1.25, 0.10),
    "item_income_rate": (0.05, 0.38),
    "item_income_mnar": (0.05, 1.35),
    "item_education_rate": (0.012, 0.170),
    "income_error_sigma": (0.035, 0.270),
    "age_heaping_prob": (0.05, 0.48),
}

# How many survey axes a hidden world may take outside the development band.
N_SURVEY_OUTSIDE_AXES = 2

SURVEY_REGIMES = ("development", "hidden")

# Domain tags. The instrument draw and the sample draw are separate streams, and the
# sample stream takes the world seed and a vintage index rather than a seed shifted by
# the snapshot tick: consecutive development seeds and consecutive ticks made two worlds
# share a survey stream, which is a joint draw nobody declared.
_INSTRUMENT_DOMAIN = 0x5A12
_SAMPLE_DOMAIN = 0x5A11


def survey_stream(seed: int, vintage: int) -> np.random.SeedSequence:
    """The sample and reporting stream of one snapshot of one world."""
    return np.random.SeedSequence([int(seed), _SAMPLE_DOMAIN, int(vintage)])


def _outside_stretch(axis: str, high_side: bool) -> tuple[float, float]:
    """The gap between the development band and the envelope edge, on one side."""
    low, high = SURVEY_BANDS[axis]
    envelope_low, envelope_high = SURVEY_ENVELOPE[axis]
    return (high, envelope_high) if high_side else (envelope_low, low)


def draw_survey_instrument(seed: int, regime: str = "development",
                           params: SurveyParams = SurveyParams()
                           ) -> tuple[SurveyParams, tuple[str, ...]]:
    """One world's survey instrument, and which of its axes left the development band.

    The sample design is published and fixed; the mechanism is drawn. A development
    world draws every axis inside the published band. A hidden world draws two axes from
    the stretch between that band and the envelope edge, one side chosen per axis, and
    the remaining seven inside the band. The draw is keyed on the world's own seed, so
    no configuration can be read off this module.
    """
    from dataclasses import replace
    if regime not in SURVEY_REGIMES:
        raise ValueError(f"unknown survey regime {regime!r}")
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), _INSTRUMENT_DOMAIN]))
    names = list(SURVEY_BANDS)
    outside: tuple[str, ...] = ()
    if regime == "hidden":
        chosen = rng.choice(len(names), size=N_SURVEY_OUTSIDE_AXES, replace=False)
        outside = tuple(sorted(names[int(k)] for k in chosen))
    values = {}
    for name in names:
        band = (_outside_stretch(name, bool(rng.random() < 0.5))
                if name in outside else SURVEY_BANDS[name])
        values[name] = float(rng.uniform(*band))
    return replace(params, **values), outside


def draw_survey_params(seed: int, regime: str = "development",
                       params: SurveyParams = SurveyParams()) -> SurveyParams:
    """One world's survey instrument: the sample design fixed, the mechanism drawn."""
    return draw_survey_instrument(seed, regime, params)[0]


def assign_strata(height: int, width: int, params: SurveyParams) -> np.ndarray:
    """Geographic strata: a coarse grid over the map, labeled row-major."""
    rows = np.minimum(np.arange(height) * params.n_strata_rows // height, params.n_strata_rows - 1)
    cols = np.minimum(np.arange(width) * params.n_strata_cols // width, params.n_strata_cols - 1)
    return rows[:, None] * params.n_strata_cols + cols[None, :]


def draw_survey(micro: dict, population: np.ndarray, seed: int,
                params: SurveyParams = SurveyParams(),
                recent_admission: np.ndarray | None = None,
                vintage: int = 0) -> dict:
    """Draw one survey: sample, response, and reported values, with retained truth.

    ``seed`` is the world's own seed and ``vintage`` numbers the snapshot, so the stream
    is a function of the world and the snapshot rather than of their sum. The caller used
    to pass the seed plus the snapshot tick, which two worlds on consecutive seeds and
    consecutive ticks can agree on: those two worlds then drew the same households, the
    same nonresponse and the same measurement error.

    ``recent_admission`` is the true indicator that a person was admitted to hospital in
    the anchor window before the reference tick.  It is reported through a misclassified
    item with a declared sensitivity and specificity, both published in the packet
    contract.  That item is the independent health anchor: the sample is drawn without
    reference to health-register inclusion, so an informative inclusion rule stays
    estimable instead of being restated by the register that caused it.
    """
    height, width = population.shape
    rng = np.random.default_rng(survey_stream(seed, vintage))
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

    # Health anchor: a misclassified report of a recent hospital admission.
    if recent_admission is None:
        true_admission = np.zeros(len(idx), dtype=np.bool_)
    else:
        true_admission = np.asarray(recent_admission, dtype=np.bool_)[idx]
    flip = np.where(true_admission,
                    rng.random(len(idx)) >= params.anchor_sensitivity,
                    rng.random(len(idx)) >= params.anchor_specificity)
    reported_admission = (true_admission ^ flip).astype(np.int8)

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
            "recent_hospitalization": reported_admission,
        },
        "truth": {
            "person_index": idx,
            "recent_admission": true_admission,
            "recent_admission_misreported": flip,
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
