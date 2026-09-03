"""Strong method A: the design-based line.

Sources: deduplication keys after Fellegi and Sunter (1969); nonresponse adjustment
and raking after Deville and Sarndal (1992); synthetic and composite small-area
estimation after Rao and Molina (2015) and Fay and Herriot (1979); Rao-Wu (1988)
rescaled bootstrap; hot-deck imputation after Andridge and Little (2010);
cohort-component projection after Preston, Heuveline, and Guillot (2001). See
docs/INDEPENDENCE.md.

A classical production line that a survey methodologist would write from the packet:

1. Deduplicate the population source: one row per observed person identifier and
   reported birth month and sex, then collapse split records that share a household,
   given name, birth month, and sex. Count persons, households, children, and elders
   by county from the deduplicated records, after a closed-form correction for county
   miscoding at a rate estimated from cross-source disagreement (population against
   income, linked on unambiguous name, birth month, and sex keys, net of the moving
   rate the population source shows between its own vintages).
2. Adjust the survey's design weights for unit nonresponse within each sampling unit,
   using the public design constant of households sampled per unit, and rake them
   within each state to the register's age-band by sex, education, and county
   proportions, the county margin taken after the miscoding correction.
3. Estimate the state-level coverage of the population source as the ratio of the
   nonresponse-adjusted survey estimate to the deduplicated register count, and scale
   county register counts by their state's coverage (synthetic small-area estimation;
   the survey's direct county estimates are too noisy on this design to improve it).
   Reconcile the county-up national count with the benchmark series by inverse-variance
   weighting under the benchmark's public bias family.
4. Impute item-missing survey income by a deterministic hot deck within stratum,
   education, and age band; estimate income statistics from the survey at nation and
   state level, and at county level synthetically through the income source's
   corroborated records (linked to the population source with the same county), the
   county-to-state ratios raised to exponents fitted on the development worlds.
5. Intervals from a rescaled bootstrap over sampling units within strata; counts are
   built county-up so they add exactly.
6. Project the population to the horizon by a cohort-component step with mortality
   estimated from record disappearance between vintages and fertility from the infant
   year of the deduplicated source, each drawn around its estimate; allocate the budget
   in proportion to projected elders.
7. Publish the detailed table with primary suppression and no totals.

Every number comes from the participant files and public constants. The method never
reads the retained side.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..character import CHARACTER_RANGES
from ..demography import DemographyParams, mortality_probability
from ..release import AGE_BANDS, AGE_BAND_LABELS, ESTIMAND_IDS, LOW_INCOME_FRACTION, SEX_LABELS
from ..sources import BENCHMARK_BIAS, BENCHMARK_ITEMS, DEVELOPMENT_BAND, INCOME_ADDRESS_LAG
from ..survey import SurveyParams
from . import actuarial_reference as AR

HOUSEHOLDS_PER_PSU = SurveyParams().households_per_cell   # public design constant
MAX_AGE = 100
COVERAGE_PRIOR_UNITS = 20.0        # sampling units at which a state's own ratio gets half weight
COVERAGE_BOUNDS = (0.70, 1.05)     # from the public source mechanism ranges, after deduplication
NOMINAL = 0.90
LINK_KEYS = ["given_code", "family_code", "birth_tick", "sex"]
# Relative root-mean-square of the benchmark's national log-bias under its public
# family: |b| uniform on the magnitude range.
_lo, _hi = BENCHMARK_BIAS["nation_magnitude"]
BENCHMARK_RELATIVE_SD = float(np.sqrt((_lo ** 2 + _lo * _hi + _hi ** 2) / 3.0))
# The state series carries its own log-bias, one normal draw per state at a world-level
# spread inside the published range, so its relative error is the root mean square of that
# range rather than the national magnitude.
_slo, _shi = BENCHMARK_BIAS["state_sd"]
BENCHMARK_STATE_RELATIVE_SD = float(np.sqrt((_slo ** 2 + _slo * _shi + _shi ** 2) / 3.0))
# Coverage-model error of the register-based counts at nation and state level, which
# the survey bootstrap cannot see: the register's coverage is estimated through the
# survey, and the survey's own nonresponse bias moves by a few percent between worlds.
REGISTER_MODEL_RELATIVE_SD = 0.025
# Model error of a county count read through its state's coverage: a county can sit
# a declared 0.14 of coverage below its state (the outpost penalty in the public
# source mechanism), which no state ratio sees and the survey's direct county estimate
# is too noisy to measure county by county. The county count intervals carry this
# relative allowance.
COUNTY_MODEL_RELATIVE_SD = 0.15


@dataclass(frozen=True)
class MethodParams:
    bootstrap_replicates: int = 200
    seed: int = 20260901
    suppression_multiplier: float = 2.0    # suppress estimated cells below this x threshold
    carry_forward_width: float = 1.5       # projection interval widening for income items
    sensitivity_multiplier: float = 2.0    # income half-width += this x the raking shift
    calibration_path: str | None = None    # JSON from calibrate() on a development world
    # Version four: exposures and rates, liability tails, and the reserve file. "auto"
    # runs the actuarial layer when the packet carries the experience file and the
    # reserve block, and writes the version-three submission when it does not.
    actuarial: str = "auto"
    actuarial_params: object = None        # methods.actuarial_reference.LayerParams


# ----------------------------------------------------------------------------- inputs

def load_packet(packet_dir: Path) -> dict:
    import pandas as pd
    P = Path(packet_dir) / "participant"
    contract = json.loads((P / "contract.json").read_text())
    geography = pd.read_csv(P / "geography.csv")
    return {
        "contract": contract,
        "county_state": geography["state"].to_numpy(dtype=np.int64),
        "survey": pd.read_csv(P / "survey_revised.csv"),
        "population": pd.read_csv(P / "sources" / "population_revised.csv"),
        "population_preliminary": pd.read_csv(P / "sources" / "population_preliminary.csv"),
        "income": pd.read_csv(P / "sources" / "income_revised.csv"),
        "health": pd.read_csv(P / "sources" / "health_revised.csv"),
        "benchmark": _load_benchmark(P / "sources" / "benchmark_revised.csv"),
    }


def _load_benchmark(path: Path) -> dict | None:
    """Benchmark totals as {item: {"nation": value, "state": array}}; None if absent."""
    import pandas as pd
    if not path.exists():
        return None
    frame = pd.read_csv(path)
    out = {}
    for item in BENCHMARK_ITEMS:
        rows = frame[frame["item"] == item]
        nation = rows[rows["level"] == "nation"]["value"]
        states = rows[rows["level"] == "state"].sort_values("unit")["value"].to_numpy(dtype=np.float64)
        if len(nation) == 1:
            out[item] = {"nation": float(nation.iloc[0]), "state": states}
    return out or None


# --------------------------------------------------------------------------- register

def unambiguous_links(left, right, left_county: str, right_county: str):
    """Pairs linked on name, birth month, and sex where the key is unique on both
    sides. Ambiguous keys (true name collisions, or a reporting error that lands on
    another person's key) are dropped rather than resolved, so what remains is
    almost entirely true links."""
    a = left.drop_duplicates(subset=LINK_KEYS, keep=False)[LINK_KEYS + [left_county]]
    b = right.drop_duplicates(subset=LINK_KEYS, keep=False)[LINK_KEYS + [right_county]]
    a = a[(a["given_code"] > 0) & (a["family_code"] > 0)]
    b = b[(b["given_code"] > 0) & (b["family_code"] > 0)]
    return a.merge(b, on=LINK_KEYS, how="inner")


def estimate_county_error_rate(population_preliminary, population_revised, income,
                               tick_pre: int, tick_rev: int) -> dict:
    """County miscoding rate from cross-source disagreement, net of moving.

    The income source records the address one year back, so a linked person's two
    counties differ when either source miscoded the county or the person moved in
    the year. The moving rate over the year is read from the population source's own
    two vintages (the same record with a different county), scaled to twelve months.
    With independent miscoding at rate e in each source, the disagreement d satisfies
    1 - d = (1 - m)(1 - 2e), which gives e in closed form. Falls back to the centre of
    the public development band when the sources cannot be linked.
    """
    lo, hi = DEVELOPMENT_BAND["county_error_rate"]
    fallback = {"rate": 0.5 * (lo + hi), "moving_rate": float("nan"),
                "disagreement": float("nan"), "n_links": 0, "estimated": False}
    if income is None or "given_code" not in population_revised.columns:
        return fallback
    pre = population_preliminary.drop_duplicates("record_id")[["record_id", "county"]]
    rev = population_revised.drop_duplicates("record_id")[["record_id", "county"]]
    both = pre.merge(rev, on="record_id", suffixes=("_pre", "_rev"))
    months = max(tick_rev - tick_pre, 1)
    if len(both) < 1000:
        return fallback
    moved_share = float((both["county_pre"] != both["county_rev"]).mean())
    moving_rate = min(1.0 - (1.0 - moved_share) ** (INCOME_ADDRESS_LAG / months), 0.5)
    links = unambiguous_links(
        population_revised.drop_duplicates("person_id").rename(columns={"county": "county_pop"}),
        income.drop_duplicates("taxpayer_id").rename(columns={"county": "county_inc"}),
        "county_pop", "county_inc")
    if len(links) < 1000:
        return fallback
    disagreement = float((links["county_pop"] != links["county_inc"]).mean())
    rate = 0.5 * (1.0 - (1.0 - disagreement) / max(1.0 - moving_rate, 1e-6))
    rate = float(np.clip(rate, 0.0, 0.25))
    return {"rate": rate, "moving_rate": float(moving_rate), "disagreement": disagreement,
            "n_links": int(len(links)), "estimated": True}


def deduplicate_population(population, tick: int, income=None, health=None):
    """One row per person in the population source; age in years at ``tick``.

    Records that share an observed person identifier are the source's own duplicates,
    which carry independently reported birth months, so the identifier is collapsed
    within reported birth month and sex. A split person (two identifiers for one
    person) shares a household, given name, birth month, and sex with its twin and
    is collapsed on those. Nothing here reads another source: the income and health
    archives record the address at their own reference dates, so a county that differs
    across sources is not evidence of a miscoded population record.
    """
    frame = population.drop_duplicates(subset=["person_id", "birth_tick", "sex"]).copy()
    if "given_code" in frame.columns:
        frame = frame.drop_duplicates(subset=["household_id", "given_code", "birth_tick", "sex"])
    frame["age"] = (tick - frame["birth_tick"]) // 12
    frame = frame[frame["county"] >= 0].copy()
    return frame


def corroborate_counties(frame, income):
    """Mark register rows whose county the income source confirms.

    A row linked to the income source on an unambiguous name, birth month, and sex
    key, with the same county in both, is a corroborated resident. A misfiled record
    (county miscoded in the population source) is almost never corroborated, so in a
    small county flooded by misfiled records from everywhere else, the corroborated
    rows carry the county's own age, household, and education composition.
    """
    frame = frame.copy()
    frame["corroborated"] = False
    if income is None or "given_code" not in frame.columns:
        return frame
    links = unambiguous_links(
        frame.rename(columns={"county": "county_pop"}),
        income.drop_duplicates("taxpayer_id").rename(columns={"county": "county_inc"}),
        "county_pop", "county_inc")
    if len(links) == 0:
        return frame
    merged = frame.merge(links[LINK_KEYS + ["county_inc"]], on=LINK_KEYS, how="left")
    frame["corroborated"] = (merged["county_inc"].to_numpy() == merged["county"].to_numpy())
    return frame


def estimate_mortality(population_preliminary, population_revised, tick_pre: int, tick_rev: int) -> dict:
    """Gompertz level from the age gradient of record disappearance between snapshots.

    A record present in the preliminary source and absent from the revised source
    (the source keeps its record identifiers between vintages) either died or dropped
    out of coverage. Coverage churn does not depend on age; mortality does, so fitting
    disappearance by age to a constant plus a Gompertz-Makeham hazard over the elapsed
    months identifies the mortality level. Ages 45 to 90 carry the signal; the slope
    stays at its public default.
    """
    pre = population_preliminary.drop_duplicates(subset=["record_id"])[["record_id", "birth_tick"]]
    rev = population_revised.drop_duplicates(subset=["record_id"])[["record_id"]]
    merged = pre.merge(rev.assign(_present=1), on=["record_id"], how="left")
    gone = merged["_present"].isna().to_numpy()
    age = ((tick_pre - merged["birth_tick"].to_numpy(dtype=np.int64)) // 12)
    months = max(tick_rev - tick_pre, 1)
    ages = np.arange(45, 91)
    exposure = np.asarray([(age == x).sum() for x in ages], dtype=np.float64)
    gone_by_age = np.asarray([gone[age == x].sum() for x in ages], dtype=np.float64)
    ok = exposure >= 50
    default = DemographyParams()
    if ok.sum() < 15:
        return {"gompertz_a": default.gompertz_a, "gompertz_b": default.gompertz_b, "fitted": False}
    rate = gone_by_age[ok] / exposure[ok]
    lo_a, hi_a = CHARACTER_RANGES["gompertz_a"]
    best, best_err = default.gompertz_a, np.inf
    for a_try in np.geomspace(lo_a * 0.5, hi_a * 1.5, 60):
        q = 1.0 - (1.0 - mortality_probability(ages[ok], DemographyParams(gompertz_a=a_try, gompertz_b=default.gompertz_b))) ** (months / 12.0)
        c = float(np.clip(np.mean(rate - q), 0.0, 1.0))         # age-flat coverage churn
        err = float((exposure[ok] * (rate - q - c) ** 2).sum())
        if err < best_err:
            best, best_err = a_try, err
    return {"gompertz_a": float(np.clip(best, lo_a * 0.5, hi_a * 1.5)), "gompertz_b": default.gompertz_b, "fitted": True}


def miscoding_correction(county_counts: np.ndarray, error_rate: float) -> np.ndarray:
    """Debias county counts for uniform county miscoding at a public rate.

    Observed count in county c is (1 - e) T_c + e (T - T_c) / (K - 1) for true counts T.
    With T taken as the observed total, T_c follows in closed form. The correction is
    small for large counties and decisive for small ones, which otherwise carry a
    trickle of misfiled records from everywhere else.
    """
    counts = np.asarray(county_counts, dtype=np.float64)
    k = len(counts)
    if k < 2 or error_rate <= 0:
        return counts
    total = counts.sum()
    corrected = (counts - error_rate * total / (k - 1)) / (1.0 - error_rate - error_rate / (k - 1))
    return np.maximum(corrected, 0.0)


def register_counts(frame, n_counties: int, county_error_rate: float | None = None,
                    flooded_share: float = 0.60, min_corroborated: int = 200) -> dict:
    """County counts and composition from the deduplicated register.

    Persons per county are debiased for county miscoding in closed form at the
    estimated rate (the centre of the public development band when none is given).
    Composition (children, elders, tertiary share, the age cube, households) is
    deconvolved the same way: the expected trickle of misfiled records into a county
    carries the national composition, so that expectation is subtracted from every
    count vector before rescaling to the debiased persons. A county whose expected
    trickle exceeds ``flooded_share`` of its raw count, and that has at least
    ``min_corroborated`` rows the income source confirms, takes its composition from
    those corroborated rows instead. Households are the raw household count net of
    the trickle (one spurious household per misfiled record), rescaled for the
    county's own records filed elsewhere.
    """
    county = frame["county"].to_numpy(dtype=np.int64)
    age = frame["age"].to_numpy(dtype=np.int64)
    sex = frame["sex"].to_numpy(dtype=np.int64)
    education = frame["education"].to_numpy(dtype=np.int64)
    lo, hi = DEVELOPMENT_BAND["county_error_rate"]
    rate = 0.5 * (lo + hi) if county_error_rate is None else float(county_error_rate)
    raw = np.bincount(county, minlength=n_counties).astype(np.float64)
    persons = miscoding_correction(raw, rate)
    k = n_counties
    trickle = rate * (raw.sum() - raw) / max(k - 1, 1) if k > 1 else np.zeros(n_counties)
    corroborated = frame["corroborated"].to_numpy(dtype=bool) if "corroborated" in frame else np.zeros(len(frame), bool)
    n_corroborated = np.bincount(county, weights=corroborated, minlength=n_counties)
    flooded = (trickle > flooded_share * np.maximum(raw, 1.0)) & (n_corroborated >= min_corroborated)
    resident_rows = np.maximum(raw - trickle, 1.0)
    total = max(raw.sum(), 1.0)

    def composition(indicator: np.ndarray) -> np.ndarray:
        """Per-county count of an indicator among true residents, deconvolved."""
        all_rows = np.bincount(county, weights=indicator.astype(np.float64), minlength=n_counties)
        national = all_rows.sum() / total
        deconvolved = np.maximum(all_rows - trickle * national, 0.0)
        confirmed = np.bincount(county, weights=indicator * corroborated, minlength=n_counties)
        share_confirmed = np.where(n_corroborated > 0, confirmed / np.maximum(n_corroborated, 1.0), 0.0)
        return np.where(flooded, share_confirmed * resident_rows, deconvolved)

    scale = persons / resident_rows                       # resident rows to persons
    counts = {
        "persons": persons,
        "children_under_16": composition(age <= 15) * scale,
        "elders_65_plus": composition(age >= 65) * scale,
    }
    raw_households = frame.groupby("county")["household_id"].nunique() \
                          .reindex(range(n_counties), fill_value=0).to_numpy(dtype=np.float64)
    counts["households"] = np.maximum(raw_households - trickle, persons / 8.0) / max(1.0 - rate, 1e-6)
    over_25 = age >= 25
    known = education >= 0
    tertiary = education >= 2
    n_known = composition(over_25 & known)
    n_tert = composition(over_25 & known & tertiary)
    with np.errstate(invalid="ignore", divide="ignore"):
        counts["tertiary_share_25_plus"] = np.where(n_known > 0, n_tert / n_known, np.nan)
    counts["tertiary_n"] = n_known
    band = np.full(len(age), -1)
    for b, (lo_b, hi_b) in enumerate(AGE_BANDS):
        band[(age >= lo_b) & (age <= hi_b)] = b
    cube = np.zeros((n_counties, len(AGE_BANDS), 2))
    for b in range(len(AGE_BANDS)):
        for s_ in range(2):
            cube[:, b, s_] = composition((np.maximum(band, 0) == b) & (sex == s_))
    counts["cube"] = cube * scale[:, None, None]
    age_sex = np.zeros((n_counties, MAX_AGE + 1, 2))
    clipped_age = np.clip(age, 0, MAX_AGE)
    for s_ in range(2):
        all_rows = np.zeros((n_counties, MAX_AGE + 1))
        np.add.at(all_rows, (county, clipped_age), (sex == s_).astype(np.float64))
        national = all_rows.sum(axis=0) / total
        deconvolved = np.maximum(all_rows - trickle[:, None] * national[None, :], 0.0)
        confirmed = np.zeros((n_counties, MAX_AGE + 1))
        np.add.at(confirmed, (county, clipped_age), ((sex == s_) & corroborated).astype(np.float64))
        share_confirmed = confirmed / np.maximum(n_corroborated, 1.0)[:, None]
        age_sex[:, :, s_] = np.where(flooded[:, None], share_confirmed * resident_rows[:, None], deconvolved)
    counts["age_sex"] = age_sex * scale[:, None, None]
    counts["raw_persons"] = raw
    counts["raw_households"] = raw_households
    counts["miscoding_rate"] = rate
    counts["flooded_counties"] = flooded
    return counts


# ----------------------------------------------------------------------------- survey

def adjusted_survey(survey):
    """Nonresponse-adjusted person weights: households sampled per unit over responding."""
    frame = survey.copy()
    responding = frame.groupby("psu")["household"].nunique()
    sampled = frame.groupby("psu")["psu_sampled_households"].first()
    factor = (sampled / responding.clip(lower=1)).clip(lower=1.0)
    frame["weight"] = frame["design_weight"] * frame["psu"].map(factor).to_numpy()
    return frame


def rake_to_register(frame, register_frame, county_state: np.ndarray, iterations: int = 12,
                     county_persons: np.ndarray | None = None):
    """Rake survey weights within each state to the register's age-band x sex,
    education, and county proportions. Proportions, not totals, so register coverage
    cancels; the point is to counter response that is selective on income through its
    correlates. The county margin is taken from ``county_persons`` (the register's
    county counts after the miscoding correction) when given: the raw county
    distribution of the register overstates a small county by every record misfiled
    into it from elsewhere, and raking to it would carry that excess into the survey.
    """
    frame = frame.copy()
    state_of = lambda counties: county_state[np.asarray(counties, dtype=np.int64)]
    frame["state"] = state_of(frame["county"])
    reg_state = state_of(register_frame["county"])
    reg_age = register_frame["age"].to_numpy(dtype=np.int64)
    reg_band = np.full(len(reg_age), -1)
    sv_age = frame["age"].to_numpy(dtype=np.int64)
    sv_band = np.full(len(sv_age), -1)
    for b, (lo, hi) in enumerate(AGE_BANDS):
        reg_band[(reg_age >= lo) & (reg_age <= hi)] = b
        sv_band[(sv_age >= lo) & (sv_age <= hi)] = b
    reg_sex = register_frame["sex"].to_numpy(dtype=np.int64)
    reg_edu = register_frame["education"].to_numpy(dtype=np.int64)
    sv_edu = frame["education"].fillna(-1).to_numpy(dtype=np.int64)
    frame["cell_as"] = sv_band * 2 + frame["sex"].to_numpy(dtype=np.int64)
    frame["cell_edu"] = np.where(sv_age >= 16, sv_edu, -1)
    reg_cell_as = reg_band * 2 + reg_sex
    reg_cell_edu = np.where(reg_age >= 16, reg_edu, -1)
    frame["cell_county"] = frame["county"].to_numpy(dtype=np.int64)
    reg_cell_county = register_frame["county"].to_numpy(dtype=np.int64)
    n_cells = max(12, len(county_state))
    weight = frame["weight"].to_numpy(dtype=np.float64).copy()
    # The nonresponse-adjusted design weight survives raking as its own column: the
    # direct county estimates are design estimates of a domain total, and raking them
    # to the register's county margin would make the register its own check.
    if "weight_unraked" not in frame.columns:
        frame["weight_unraked"] = weight.copy()

    def margin_target(reg_cells: np.ndarray) -> np.ndarray:
        keep = reg_cells >= 0
        return np.bincount(reg_cells[keep], minlength=n_cells) / max(keep.sum(), 1)

    for s in range(int(county_state.max()) + 1):
        sv = np.flatnonzero(frame["state"].to_numpy() == s)
        rg = reg_state == s
        if len(sv) == 0 or rg.sum() == 0:
            continue
        if county_persons is not None:
            in_state = np.zeros(n_cells)
            in_state[: len(county_state)] = np.where(county_state == s, np.maximum(county_persons, 0.0), 0.0)
            county_target = in_state / max(in_state.sum(), 1e-9)
        else:
            county_target = margin_target(reg_cell_county[rg])
        margins = ((frame["cell_as"].to_numpy()[sv], margin_target(reg_cell_as[rg])),
                   (frame["cell_edu"].to_numpy()[sv], margin_target(reg_cell_edu[rg])),
                   (frame["cell_county"].to_numpy()[sv], county_target))
        total = weight[sv].sum()
        for _ in range(iterations):
            for sv_cells, target in margins:
                in_margin = weight[sv[sv_cells >= 0]].sum()
                for cell in np.unique(sv_cells):
                    if cell < 0 or cell >= n_cells or target[cell] <= 0 or in_margin <= 0:
                        continue
                    members = sv[sv_cells == cell]
                    current = weight[members].sum() / in_margin
                    if current > 0:
                        weight[members] *= min(max(target[cell] / current, 0.25), 4.0)
                weight[sv] *= total / weight[sv].sum()
    frame["weight"] = weight
    return frame.drop(columns=["state", "cell_as", "cell_edu", "cell_county"])


def impute_income(frame):
    """Deterministic hot deck: weighted median of donors in (stratum, education, age band)."""
    frame = frame.copy()
    frame["education"] = frame["education"].fillna(-1).astype(int)
    band = np.full(len(frame), -1)
    age = frame["age"].to_numpy()
    for b, (lo, hi) in enumerate(AGE_BANDS):
        band[(age >= lo) & (age <= hi)] = b
    frame["band"] = band
    frame.loc[frame["age"] < 16, "income"] = frame.loc[frame["age"] < 16, "income"].fillna(0.0)
    missing = frame["income"].isna()
    donors = frame[~missing]
    for keys in (["stratum", "education", "band"], ["education", "band"], ["band"], []):
        if not missing.any():
            break
        if keys:
            medians = donors.groupby(keys)["income"].median()
            fill = frame.loc[missing, keys].apply(tuple, axis=1) if len(keys) > 1 \
                else frame.loc[missing, keys[0]]
            values = fill.map(medians.to_dict() if len(keys) > 1 else medians)
        else:
            values = np.full(int(missing.sum()), float(donors["income"].median()))
        frame.loc[missing, "income"] = np.asarray(values, dtype=np.float64)
        missing = frame["income"].isna()
    return frame


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    if len(values) == 0:
        return float("nan")
    order = np.argsort(values, kind="stable")
    v, w = values[order], weights[order]
    cum = np.cumsum(w)
    return float(v[np.searchsorted(cum, 0.5 * cum[-1])])


def survey_statistics(frame, county_state: np.ndarray) -> dict:
    """Survey estimates: person totals by state, income statistics by state and nation."""
    n_states = int(county_state.max()) + 1
    state = county_state[frame["county"].to_numpy(dtype=np.int64)]
    w = frame["weight"].to_numpy()
    adults = frame["age"].to_numpy() >= 16
    income = frame["income"].to_numpy(dtype=np.float64)
    hh = frame.groupby("household").agg(income=("income", "sum"), weight=("weight", "first"),
                                        county=("county", "first"))
    hh_state = county_state[hh["county"].to_numpy(dtype=np.int64)]
    hh_income = hh["income"].to_numpy(dtype=np.float64)
    hh_w = hh["weight"].to_numpy()
    national_median = _weighted_median(hh_income, hh_w)
    low = hh_income < LOW_INCOME_FRACTION * national_median
    out = {"persons_by_state": np.bincount(state, weights=w, minlength=n_states),
           "households_by_state": np.bincount(hh_state, weights=hh_w, minlength=n_states),
           "median_household_income": {}, "mean_income_adults": {},
           "low_income_household_share": {}}
    for s in range(n_states):
        ps, hs = state == s, hh_state == s
        out["median_household_income"][s] = _weighted_median(hh_income[hs], hh_w[hs])
        out["mean_income_adults"][s] = (float((w * income * adults)[ps].sum() /
                                              max((w * adults)[ps].sum(), 1e-9)))
        out["low_income_household_share"][s] = float((hh_w * low)[hs].sum() / max(hh_w[hs].sum(), 1e-9))
    out["median_household_income"]["nation"] = national_median
    out["mean_income_adults"]["nation"] = float((w * income * adults).sum() / (w * adults).sum())
    out["low_income_household_share"]["nation"] = float((hh_w * low).sum() / hh_w.sum())
    return out


# ------------------------------------------------------------------- income source

def register_income_scale(income, survey_mean_adult_income: float) -> float:
    """Survey income unit per register cent: the register reports earnings at a
    world-specific wage level, so its cents are rescaled by the ratio of the survey's
    mean adult income to the register's mean positive earnings before any threshold
    on a survey scale is applied to them."""
    positive = income["employment_income_cents"].to_numpy(dtype=np.float64)
    positive = positive[np.isfinite(positive) & (positive > 0)]
    if len(positive) == 0 or not np.isfinite(survey_mean_adult_income) or survey_mean_adult_income <= 0:
        return 0.01
    return float(survey_mean_adult_income / positive.mean())


def corroborated_income(income, register_frame):
    """Income records whose county the population source confirms.

    A record linked to the population source on an unambiguous name, birth month, and
    sex key, with the same county in both, is a corroborated resident of that county.
    A record misfiled into a county by the income source's own miscoding is almost
    never corroborated, so the corroborated records of a small county carry the
    county's own income distribution, where the raw records carry mostly the misfiled
    trickle from everywhere else. Coverage and linkage do not select on income, and
    the county ratios below cancel any selection that is common to every county.
    """
    frame = income.drop_duplicates("taxpayer_id").copy()
    frame["corroborated"] = False
    if register_frame is None or "given_code" not in frame.columns:
        return frame
    links = unambiguous_links(
        register_frame.rename(columns={"county": "county_pop"}),
        frame.rename(columns={"county": "county_inc"}), "county_pop", "county_inc")
    agree = links[links["county_pop"] == links["county_inc"]][LINK_KEYS].assign(_agree=True)
    merged = frame.merge(agree, on=LINK_KEYS, how="left")
    frame["corroborated"] = merged["_agree"].fillna(False).to_numpy(dtype=bool)
    return frame


def income_source_ratios(income, county_state: np.ndarray, national_median_hh: float,
                         scale: float = 0.01, register_frame=None,
                         min_households: int = 30) -> dict:
    """County-to-state ratios from the income source, for synthetic county estimates.
    ``scale`` converts register cents to the survey's income unit. With
    ``register_frame`` given, a county's statistics come from the households the
    population source corroborates (any member linked with the same county), so a
    county flooded by misfiled records keeps its own income distribution; a county
    with fewer than ``min_households`` corroborated households takes its state's
    value (ratio one)."""
    n_counties = len(county_state)
    frame = income[income["county"] >= 0].copy()
    frame["employment_income_cents"] = frame["employment_income_cents"].fillna(0.0) * scale
    if register_frame is not None:
        flags = corroborated_income(income, register_frame)
        confirmed = flags[flags["corroborated"]][["county", "household_id"]].drop_duplicates()
        frame = frame.merge(confirmed.assign(_keep=True), on=["county", "household_id"], how="inner") \
                     .drop(columns=["_keep"])
    positive = frame[frame["employment_income_cents"] > 0]
    county_mean = positive.groupby("county")["employment_income_cents"].mean() \
                          .reindex(range(n_counties)).to_numpy()
    hh = frame.groupby(["county", "household_id"])["employment_income_cents"].sum().reset_index()
    hh["low"] = hh["employment_income_cents"] < LOW_INCOME_FRACTION * national_median_hh
    county_low = hh.groupby("county")["low"].mean().reindex(range(n_counties)).to_numpy()
    county_median = hh.groupby("county")["employment_income_cents"].median() \
                      .reindex(range(n_counties)).to_numpy()
    n_households = hh.groupby("county").size().reindex(range(n_counties), fill_value=0).to_numpy()
    enough = n_households >= min_households
    # The denominator is the state's own pooled statistic over the same records, so
    # that the ratio compares a county with its state as the survey sees the state,
    # rather than with an unweighted mean over the state's counties.
    n_states = int(county_state.max()) + 1
    positive_state = county_state[positive["county"].to_numpy(dtype=np.int64)]
    hh_state = county_state[hh["county"].to_numpy(dtype=np.int64)]
    state_mean = np.asarray([positive["employment_income_cents"].to_numpy()[positive_state == s].mean()
                             if (positive_state == s).any() else np.nan for s in range(n_states)])
    state_median = np.asarray([np.median(hh["employment_income_cents"].to_numpy()[hh_state == s])
                               if (hh_state == s).any() else np.nan for s in range(n_states)])
    state_low = np.asarray([hh["low"].to_numpy()[hh_state == s].mean()
                            if (hh_state == s).any() else np.nan for s in range(n_states)])
    ratios = {}
    for name, values, state_values in (("mean_income_adults", county_mean, state_mean),
                                       ("median_household_income", county_median, state_median),
                                       ("low_income_household_share", county_low, state_low)):
        values = np.where(enough, values, np.nan)
        with np.errstate(invalid="ignore", divide="ignore"):
            r = values / state_values[county_state]
        # A county-to-state ratio from an employment-income source is a proxy for a
        # total-income quantity; outside a half to double it is noise, not signal.
        r = np.where(np.isfinite(r) & (r > 0), np.clip(r, 0.5, 2.0), 1.0)
        ratios[name] = r
    return ratios


RATIO_EXPONENT_ITEMS = ("mean_income_adults", "low_income_household_share")
RATIO_EXPONENT_BOUNDS = (1.0, 3.0)


def fit_ratio_exponents(dev_packet_dirs) -> dict:
    """Exponents that map the income source's county-to-state ratios onto the
    survey's total-income quantities, fitted on the development worlds.

    Employment income is one component of total income and most households report
    little or none of it, so a county's employment-income ratio to its state is
    compressed toward one relative to the ratio of the total-income quantity the
    release asks for. On the development worlds, whose truth is shipped, the log of
    the true ratio against the log of the observed ratio has a slope near two for the
    adult mean and near one and a half for the low-income share, and the slope is
    stable across worlds; it is fitted through the origin, pooled over the worlds'
    counties, and bounded. The household median's slope is not stable across worlds
    (the employment-income median sits near zero in poorer counties), so its exponent
    stays at one.
    """
    import pandas as pd
    dev_packet_dirs = [Path(d) for d in ([dev_packet_dirs] if isinstance(dev_packet_dirs, (str, Path)) else dev_packet_dirs)]
    xs = {e: [] for e in RATIO_EXPONENT_ITEMS}
    ys = {e: [] for e in RATIO_EXPONENT_ITEMS}
    for dev in dev_packet_dirs:
        data = load_packet(dev)
        county_state = data["county_state"]
        tick = int(data["contract"]["ticks"]["revised"])
        register_frame = corroborate_counties(deduplicate_population(data["population"], tick), data["income"])
        stats = survey_statistics(adjusted_survey(data["survey"]), county_state)
        scale = register_income_scale(data["income"], stats["mean_income_adults"]["nation"])
        ratios = income_source_ratios(data["income"], county_state, stats["median_household_income"]["nation"],
                                      scale, register_frame=register_frame)
        truth = pd.read_csv(Path(dev) / "participant" / "truth" / "truth_revised.csv")
        for e in RATIO_EXPONENT_ITEMS:
            county = truth[(truth["estimand"] == e) & (truth["level"] == "county")].set_index("unit")["value"]
            state = truth[(truth["estimand"] == e) & (truth["level"] == "state")].set_index("unit")["value"]
            for c in county.index:
                observed, actual = float(ratios[e][int(c)]), float(county[c]) / float(state[county_state[int(c)]])
                if abs(np.log(observed)) > 1e-6 and actual > 0:
                    xs[e].append(np.log(observed))
                    ys[e].append(np.log(actual))
    exponents = {"median_household_income": 1.0}
    for e in RATIO_EXPONENT_ITEMS:
        x, y = np.asarray(xs[e]), np.asarray(ys[e])
        slope = float((x * y).sum() / (x ** 2).sum()) if len(x) >= 3 and (x ** 2).sum() > 0 else 1.0
        exponents[e] = float(np.clip(slope, *RATIO_EXPONENT_BOUNDS))
    return exponents


def apply_ratio_exponents(ratios: dict, exponents: dict | None) -> dict:
    """Raise each county-to-state ratio to its fitted exponent, inside the same
    half-to-double band the raw ratio was held to."""
    if not exponents:
        return ratios
    return {e: np.clip(r ** float(exponents.get(e, 1.0)), 0.5, 2.0) for e, r in ratios.items()}


# -------------------------------------------------------------------- one estimate

def estimate_once(frame, register: dict, ratios: dict, county_state: np.ndarray) -> dict:
    """County-level point estimates for all estimands from one survey replicate."""
    n_counties = len(county_state)
    n_states = int(county_state.max()) + 1
    stats = survey_statistics(frame, county_state)
    reg_state = np.bincount(county_state, weights=register["persons"], minlength=n_states)
    with np.errstate(invalid="ignore", divide="ignore"):
        coverage_state = np.where(stats["persons_by_state"] > 0, reg_state / stats["persons_by_state"], np.nan)
    # A state's coverage ratio is only as good as the survey behind it. A state with a
    # handful of sampling units gets a ratio dominated by sampling noise, so each state's
    # ratio is shrunk toward the national ratio with weight proportional to its number
    # of sampling units (empirical Bayes; twenty units count as much as the prior).
    national = float(reg_state.sum() / max(stats["persons_by_state"].sum(), 1e-9))
    n_psu_state = frame.groupby(county_state[frame["county"].to_numpy(dtype=np.int64)])["psu"].nunique() \
                       .reindex(range(n_states), fill_value=0).to_numpy(dtype=np.float64)
    weight = n_psu_state / (n_psu_state + COVERAGE_PRIOR_UNITS)
    coverage = np.where(np.isfinite(coverage_state), weight * coverage_state + (1.0 - weight) * national, national)
    # The public mechanism ranges bound plausible coverage after deduplication.
    coverage = np.clip(coverage, COVERAGE_BOUNDS[0], COVERAGE_BOUNDS[1])
    scale = 1.0 / coverage[county_state]
    county = {e: register[e] * scale for e in ("persons", "households", "children_under_16", "elders_65_plus")}
    # Households have their own coverage: a household is on the register when any
    # member is, so its coverage exceeds the person coverage and the person scale
    # would overstate household counts. The state ratio of register to survey
    # households, shrunk like the person ratio, replaces it.
    county["households"] = register["households"] * household_scale(
        stats, register, county_state, weight, national, scale, coverage)
    # County counts are synthetic: the register's county count at its state's
    # coverage. The survey's direct county estimates are not used at county level.
    # On this design (a fixed number of sampling units per stratum, drawn with
    # probability proportional to size, none of them nested in counties) a county's
    # direct estimate carries a relative error of 0.2 to 0.4 and, for a small county,
    # a conditional bias from the units that happened to land there; combining it
    # with a synthetic estimate whose error is near 0.1 would only add noise, and the
    # moment estimate of the synthetic model's variance from such residuals is not
    # identifiable. The county intervals carry the fixed relative allowance
    # ``COUNTY_MODEL_RELATIVE_SD`` instead, which a survey bootstrap alone cannot see.
    county["_model_rel_sd"] = COUNTY_MODEL_RELATIVE_SD
    county["tertiary_share_25_plus"] = register["tertiary_share_25_plus"]
    for e in ("median_household_income", "mean_income_adults", "low_income_household_share"):
        state_values = np.asarray([stats[e][s] for s in range(n_states)])
        county[e] = state_values[county_state] * ratios[e]
        if e == "low_income_household_share":
            county[e] = np.clip(county[e], 0.0, 1.0)
    # County median and mean income: the synthetic estimate (state survey value times
    # the income source's county ratio) is combined with the direct survey estimate
    # where the county has enough sampling units, as for persons above, and the
    # synthetic model error is read from the residuals net of sampling variance. The
    # income source's county ratio is noisier under dated addresses and miscoding than
    # a fixed allowance assumes, so the allowance is measured, never below ten percent.
    county["_income_model_rel_sd"] = {}
    direct_income = _direct_county_income(frame, n_counties)
    for e in ("median_household_income", "mean_income_adults"):
        direct_e, var_e = direct_income[e]
        synthetic_e = county[e].copy()
        have_e = np.isfinite(direct_e) & np.isfinite(var_e) & (var_e > 0) & (synthetic_e > 0)
        if have_e.sum() >= 3:
            residual = direct_e[have_e] - synthetic_e[have_e]
            model_var = max(float(np.mean(residual ** 2) - np.mean(var_e[have_e])), 0.0)
            gamma = np.where(have_e, model_var / np.maximum(model_var + np.nan_to_num(var_e), 1e-9), 0.0)
            county[e] = np.where(have_e, gamma * np.nan_to_num(direct_e) + (1.0 - gamma) * synthetic_e, synthetic_e)
            rel_resid = residual / synthetic_e[have_e]
            rel_sampling = var_e[have_e] / synthetic_e[have_e] ** 2
            county["_income_model_rel_sd"][e] = max(float(np.sqrt(max(np.mean(rel_resid ** 2) - np.mean(rel_sampling), 0.0))), 0.10)
        else:
            county["_income_model_rel_sd"][e] = 0.15
    return {"county": county, "state_stats": stats, "cube": register["cube"] * scale[:, None, None],
            "age_sex": register["age_sex"] * scale[:, None, None]}


def household_scale(stats: dict, register: dict, county_state: np.ndarray, weight: np.ndarray,
                    national_persons: float, person_scale: np.ndarray, person_coverage: np.ndarray) -> np.ndarray:
    """Per-county factor from register households to estimated households."""
    n_states = int(county_state.max()) + 1
    reg_hh_state = np.bincount(county_state, weights=register["households"], minlength=n_states)
    survey_hh = stats["households_by_state"]
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio_state = np.where(survey_hh > 0, reg_hh_state / survey_hh, np.nan)
    national = float(reg_hh_state.sum() / max(survey_hh.sum(), 1e-9))
    coverage_hh = np.where(np.isfinite(ratio_state), weight * ratio_state + (1.0 - weight) * national, national)
    coverage_hh = np.clip(coverage_hh, 0.60, 1.15)
    # Keep the person scale's small-county and Fay-Herriot structure, replacing only
    # the state coverage level.
    return person_scale * (person_coverage / coverage_hh)[county_state]


def _direct_county_persons(frame, n_counties: int, weight_column: str | None = None,
                           floor: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Direct survey estimate of persons by county with its design variance.

    Uses the nonresponse-adjusted design weight from before raking (``weight_unraked``)
    when the frame carries it, so that the estimate stays a design estimate rather
    than an echo of the register's county margin.

    Counties are domains of a survey stratified elsewhere: a county's total is the
    weighted sum over the sampling units that landed in it, and a unit of the same
    stratum that landed in another county contributes zero to that sum. The
    with-replacement variance over the units of each stratum therefore includes those
    zeros; a variance taken over the county's own units alone misses the randomness
    of how many units landed there, which is what dominates for a small county. The
    number of units that land in a county is close to Poisson, so the relative
    variance is floored at one over that number when ``floor`` is set; without the
    floor the design variance alone is returned, which is what a moment estimate of
    the synthetic model's variance needs (the floor would count as sampling error
    what is not there on average and drive that estimate to zero).
    """
    column = weight_column or ("weight_unraked" if "weight_unraked" in frame.columns else "weight")
    county = frame["county"].to_numpy(dtype=np.int64)
    w = frame[column].to_numpy(dtype=np.float64)
    total = np.bincount(county, weights=w, minlength=n_counties)
    per_unit = frame.groupby(["stratum", "psu", "county"])[column].sum()
    var = np.zeros(n_counties)
    units = np.zeros(n_counties)
    for _, part in per_unit.groupby(level=0):
        n = part.index.get_level_values(1).nunique()
        if n < 2:
            continue
        by_county = part.groupby(level=2)
        s1 = by_county.sum()
        s2 = (part ** 2).groupby(level=2).sum()
        idx = s1.index.to_numpy(dtype=np.int64)
        var[idx] += n / (n - 1) * np.maximum(s2.to_numpy() - s1.to_numpy() ** 2 / n, 0.0)
        units[idx] += by_county.size().to_numpy()
    if floor:
        var = np.maximum(var, total ** 2 / np.maximum(units, 1.0))
    var = np.where(units > 0, var, np.nan)
    return total, var


def _direct_county_income(frame, n_counties: int) -> dict:
    """Direct survey estimates of the county median household income and mean adult
    income, with a delete-one-unit jackknife variance over the county's sampling
    units. Counties with fewer than four units or thirty responding households are
    left undefined."""
    hh = frame.groupby("household").agg(income=("income", "sum"), weight=("weight", "first"),
                                        county=("county", "first"), psu=("psu", "first"))
    adults = frame[frame["age"] >= 16]
    out = {"median_household_income": (np.full(n_counties, np.nan), np.full(n_counties, np.nan)),
           "mean_income_adults": (np.full(n_counties, np.nan), np.full(n_counties, np.nan))}

    def median_of(part):
        return _weighted_median(part["income"].to_numpy(dtype=np.float64), part["weight"].to_numpy(dtype=np.float64))

    def mean_of(part):
        w = part["weight"].to_numpy(dtype=np.float64)
        return float((w * part["income"].to_numpy(dtype=np.float64)).sum() / max(w.sum(), 1e-9))

    for item, table, fn in (("median_household_income", hh, median_of), ("mean_income_adults", adults, mean_of)):
        for c, part in table.groupby("county"):
            units = part["psu"].unique()
            if len(units) < 4 or len(part) < 30:
                continue
            full = fn(part)
            leave = np.asarray([fn(part[part["psu"] != u]) for u in units], dtype=np.float64)
            n = len(units)
            out[item][0][int(c)] = full
            out[item][1][int(c)] = (n - 1) / n * float(((leave - leave.mean()) ** 2).sum())
    return out


def aggregate(county_values: dict, county_state: np.ndarray, stats: dict, weights: np.ndarray) -> dict:
    """County-up aggregation: counts add; shares and means are person-weighted; medians
    use the survey estimate at state and nation."""
    n_states = int(county_state.max()) + 1
    out = {}
    persons = county_values["persons"]
    for e in ESTIMAND_IDS:
        v = county_values[e]
        for c in range(len(county_state)):
            out[(e, "county", c)] = float(v[c])
        if e in ("persons", "households", "children_under_16", "elders_65_plus"):
            state_v = np.bincount(county_state, weights=v, minlength=n_states)
            nation_v = float(state_v.sum())
        elif e == "median_household_income":
            state_v = np.asarray([stats[e][s] for s in range(n_states)])
            nation_v = float(stats[e]["nation"])
        else:
            with np.errstate(invalid="ignore", divide="ignore"):
                state_v = np.bincount(county_state, weights=np.nan_to_num(v) * persons, minlength=n_states) / \
                    np.maximum(np.bincount(county_state, weights=persons, minlength=n_states), 1e-9)
            nation_v = float(np.nansum(np.nan_to_num(v) * persons) / max(persons.sum(), 1e-9))
        for s in range(n_states):
            out[(e, "state", s)] = float(state_v[s])
        out[(e, "nation", 0)] = nation_v
    return out


# ---------------------------------------------------------------------- projection

def estimate_fertility(frame, tick: int) -> dict:
    """Annual birth probability per woman aged 18 to 45, from the deduplicated
    population source: persons whose reported birth tick falls in the twenty-four
    ticks before the snapshot, halved, over women in the fertile ages. Coverage cancels
    in the ratio to the extent it does not depend on age; a two-year window absorbs the
    birth ticks that year rounding moves across the first birthday, and the projection's
    fertility draw allows for what remains. Falls back to the public range when the
    source is too thin."""
    lo, hi = CHARACTER_RANGES["fertility_rate"]
    birth = frame["birth_tick"].to_numpy(dtype=np.int64)
    months = tick - birth
    age = months // 12
    sex = frame["sex"].to_numpy()
    infants = int(((months >= 0) & (months < 24)).sum())
    women = int(((sex == 1) & (age >= 18) & (age <= 45)).sum())
    if women < 1000 or infants < 100:
        return {"fertility_rate": 0.5 * (lo + hi), "fitted": False}
    return {"fertility_rate": float(np.clip(0.5 * infants / women, 0.5 * lo, 1.5 * hi)), "fitted": True}


def project(county_values: dict, age_sex: np.ndarray, months: int, rng: np.random.Generator,
            mortality: dict | None = None, fertility: dict | None = None) -> dict:
    """Cohort-component projection on single-year ages.

    Mortality and fertility are drawn around their estimates when the source supports
    one (``estimate_mortality``, ``estimate_fertility``) and from the public ranges
    otherwise; a shock year is drawn with the frequency the public shock family
    implies. Each simulated year: survivors age by one, births arrive to women aged 18
    to 45 and survive infancy, and the open-ended top age absorbs. Income items are
    carried forward; household counts follow the person growth.
    """
    lo_a, hi_a = CHARACTER_RANGES["gompertz_a"]
    lo_f, hi_f = CHARACTER_RANGES["fertility_rate"]
    if mortality is not None and mortality.get("fitted"):
        a_draw = float(np.clip(rng.lognormal(np.log(mortality["gompertz_a"]), 0.20), lo_a * 0.5, hi_a * 1.5))
    else:
        a_draw = float(rng.uniform(lo_a, hi_a))
    if fertility is not None and fertility.get("fitted"):
        f_draw = float(np.clip(rng.lognormal(np.log(fertility["fertility_rate"]), 0.10), lo_f * 0.5, hi_f * 1.5))
    else:
        f_draw = float(rng.uniform(lo_f, hi_f))
    params = DemographyParams(gompertz_a=a_draw, fertility_rate=f_draw)
    years = int(round(months / 12.0))
    ages = np.arange(MAX_AGE + 1)
    q = mortality_probability(ages, params)
    n_shocks = int(rng.integers(0, 3))
    shock_years = set(int(v) for v in rng.integers(0, max(years, 1), size=n_shocks))
    state = age_sex.astype(np.float64).copy()
    for year in range(years):
        multiplier = float(rng.uniform(1.5, 3.0)) if year in shock_years else 1.0
        survival = 1.0 - np.clip(q * multiplier, 0.0, 1.0)
        survivors = state * survival[None, :, None]
        aged = np.zeros_like(survivors)
        aged[:, 1:, :] = survivors[:, :-1, :]
        aged[:, MAX_AGE, :] += survivors[:, MAX_AGE, :]
        women = state[:, 18:46, 1].sum(axis=1)
        rate = params.fertility_rate * (float(rng.uniform(0.45, 0.75)) if year in shock_years and rng.random() < 0.33 else 1.0)
        births = women * rate * survival[0]
        aged[:, 0, 0] += 0.5 * births
        aged[:, 0, 1] += 0.5 * births
        state = aged
    persons = state.sum(axis=(1, 2))
    growth = persons / np.maximum(age_sex.sum(axis=(1, 2)), 1e-9)
    out = dict(county_values)
    out["persons"] = persons
    out["children_under_16"] = state[:, :16, :].sum(axis=(1, 2))
    out["elders_65_plus"] = state[:, 65:, :].sum(axis=(1, 2))
    out["households"] = county_values["households"] * growth
    return out


# ------------------------------------------------------------------- benchmark

def benchmark_reconciliation(point: dict, replicates: list[dict], benchmark: dict | None,
                             model_relative_sd: float = REGISTER_MODEL_RELATIVE_SD) -> dict:
    """Per count item, the factor that moves the county-up nation to its
    inverse-variance combination with the benchmark nation. The register's variance is
    the bootstrap spread plus a coverage-model allowance, capped at that allowance; the
    benchmark's is its public bias family. Every level is scaled by the same factor, so
    additivity holds.

    The cap is what keeps the step honest for a method whose own national spread is
    wide. Without it, a line whose posterior half-width at the nation runs to a fifth of
    the estimate assigns the register almost no weight and adopts the benchmark series,
    inheriting a log-bias of 0.02 to 0.07 nearly one for one. The register's error at the
    nation after coverage correction is a model error, not a sampling error, so the
    allowance is the right ceiling on the variance the step may assume for it: no method
    can hand the benchmark more than about a fifth of the weight."""
    factors = {}
    for item in BENCHMARK_ITEMS:
        key = (item, "nation", 0)
        register = float(point.get(key, np.nan))
        if benchmark is None or item not in benchmark or not np.isfinite(register) or register <= 0:
            factors[item] = 1.0
            continue
        bench = float(benchmark[item]["nation"])
        draws = np.asarray([r[key] for r in replicates if key in r], dtype=np.float64)
        draws = draws[np.isfinite(draws)]
        rel_boot = float(np.std(draws) / register) if len(draws) >= 10 else 0.02
        var_register = min((rel_boot ** 2 + model_relative_sd ** 2),
                           model_relative_sd ** 2) * register ** 2
        var_bench = (BENCHMARK_RELATIVE_SD * bench) ** 2
        weight = var_bench / (var_bench + var_register)          # weight on the register
        combined = weight * register + (1.0 - weight) * bench
        factors[item] = float(combined / register)
    return factors


# The register's error in one state after coverage correction is a model error, not a
# sampling error, and it is far larger than the national one: coverage rides the county
# economic gradient and the outpost penalty, both declared in the public source ranges, so
# a state whose counties sit on one side of that gradient is off by a good deal more than
# the nation is. The allowance below is the county allowance damped by aggregation over
# the counties of a state, and it is what decides how much weight the benchmark's own
# state series gets.
STATE_MODEL_RELATIVE_SD = 0.10


def benchmark_state_reconciliation(point: dict, replicates: list[dict],
                                   benchmark: dict | None, county_state: np.ndarray,
                                   national: dict | None = None,
                                   model_relative_sd: float = STATE_MODEL_RELATIVE_SD) -> dict:
    """Per count item, one factor per state, from the benchmark's own state series.

    The national step above moves every level by one factor and so cannot touch the
    composition, which is where a register-based reconstruction is weakest: its national
    total is anchored, while its split across states carries the whole coverage gradient.
    The benchmark publishes a state series with a declared bias family, so each state's
    count is an inverse-variance combination of the register's and the benchmark's, and
    the combined vector is then rescaled to the national total the first step settled, so
    additivity survives.
    """
    n_states = int(np.max(county_state)) + 1 if len(county_state) else 0
    factors: dict[str, np.ndarray] = {}
    if benchmark is None or n_states < 2:
        return factors
    for item in BENCHMARK_ITEMS:
        if item not in benchmark:
            continue
        bench = np.asarray(benchmark[item].get("state", []), dtype=np.float64)
        register = np.asarray([point.get((item, "state", s), np.nan)
                               for s in range(n_states)], dtype=np.float64)
        if bench.shape != register.shape or not np.isfinite(register).all() \
                or (register <= 0).any() or not np.isfinite(bench).all() or (bench <= 0).any():
            continue
        spread = []
        for s in range(n_states):
            key = (item, "state", s)
            draws = np.asarray([r[key] for r in replicates if key in r], dtype=np.float64)
            draws = draws[np.isfinite(draws)]
            spread.append(float(np.std(draws) / register[s]) if len(draws) >= 10 else 0.03)
        rel_register = np.sqrt(np.asarray(spread) ** 2 + model_relative_sd ** 2)
        var_register = (rel_register * register) ** 2
        var_bench = (BENCHMARK_STATE_RELATIVE_SD * bench) ** 2
        weight = var_bench / (var_bench + var_register)          # weight on the register
        combined = weight * register + (1.0 - weight) * bench
        # The national factor is applied to every level before these, so the composition
        # is normalized to the register's own national total and the two steps compose
        # instead of multiplying twice.
        target = float(point.get((item, "nation", 0), np.nan))
        if np.isfinite(target) and combined.sum() > 0:
            combined = combined * (target / combined.sum())
        factors[item] = combined / register
    return factors


CHILD_MAX_AGE = 15          # the benchmark's children item is under sixteen
ELDER_MIN_AGE = 65          # and its elders item is sixty-five and over
AGE_SCALE_BOUNDS = (0.50, 2.00)


def benchmark_age_scale(cube: np.ndarray, county_state: np.ndarray, factors: dict,
                        state_factors: dict | None = None) -> np.ndarray:
    """A multiplier per county and age that rakes a county age cube to the benchmark.

    The benchmark publishes four count items, and three of them are an age structure:
    persons, children under sixteen, and people sixty-five and over. Scaling a cube by the
    persons factor alone puts the total right and leaves the shape wrong, which is the part
    the liability is priced on: the obligation pays from sixty-five, so an age cube whose
    old ages are off by a tenth prices a regional tail that is off by a tenth however good
    its headcount is.

    The children and elder blocks take their own factors, and the middle takes whatever
    factor makes the three blocks add to the state's reconciled headcount, so the raked
    cube reproduces all three published counts at once.
    """
    cube = np.asarray(cube, dtype=np.float64)
    n_counties, n_ages = cube.shape[0], cube.shape[1]
    scale = np.ones((n_counties, n_ages))
    state_factors = state_factors or {}

    def factor(item: str, state: int) -> float:
        value = float(factors.get(item, 1.0))
        per_state = state_factors.get(item)
        if per_state is not None and 0 <= state < len(per_state):
            value *= float(per_state[state])
        return value

    child = np.arange(n_ages) <= CHILD_MAX_AGE
    elder = np.arange(n_ages) >= ELDER_MIN_AGE
    middle = ~(child | elder)
    counts = cube.sum(axis=tuple(range(2, cube.ndim))) if cube.ndim > 2 else cube
    for state in range(int(np.max(county_state)) + 1 if len(county_state) else 0):
        rows = np.flatnonzero(np.asarray(county_state) == state)
        if not len(rows):
            continue
        block = counts[rows]
        c, e, m = block[:, child].sum(), block[:, elder].sum(), block[:, middle].sum()
        if min(c, e, m) <= 0:
            continue
        target = (c + e + m) * factor("persons", state)
        f_child = np.clip(factor("children_under_16", state), *AGE_SCALE_BOUNDS)
        f_elder = np.clip(factor("elders_65_plus", state), *AGE_SCALE_BOUNDS)
        f_middle = np.clip((target - c * f_child - e * f_elder) / m, *AGE_SCALE_BOUNDS)
        scale[np.ix_(rows, np.flatnonzero(child))] = f_child
        scale[np.ix_(rows, np.flatnonzero(elder))] = f_elder
        scale[np.ix_(rows, np.flatnonzero(middle))] = f_middle
    return scale


def apply_reconciliation(values: dict, factors: dict,
                         state_factors: dict | None = None,
                         county_state: np.ndarray | None = None) -> dict:
    """Scale every level by its item's national factor, then by its own state's factor."""
    out = dict(values)
    for (e, level, u), v in values.items():
        if e in factors and np.isfinite(v):
            out[(e, level, u)] = float(v * factors[e])
    if not state_factors:
        return out
    for (e, level, u), v in list(out.items()):
        per_state = state_factors.get(e)
        if per_state is None or not np.isfinite(v):
            continue
        if level == "state" and 0 <= int(u) < len(per_state):
            out[(e, level, u)] = float(v * per_state[int(u)])
        elif level == "county" and county_state is not None and 0 <= int(u) < len(county_state):
            out[(e, level, u)] = float(v * per_state[int(county_state[int(u)])])
    return out


# --------------------------------------------------------------------------- driver

def _bootstrap_frame(frame, rng: np.random.Generator):
    """Rao-Wu rescaled bootstrap: resample sampling units with replacement within strata."""
    pieces = []
    for stratum, part in frame.groupby("stratum"):
        psus = part["psu"].unique()
        n = len(psus)
        if n < 2:
            pieces.append(part)
            continue
        draw = rng.choice(psus, size=n - 1, replace=True)
        counts = dict(zip(*np.unique(draw, return_counts=True)))
        factor = part["psu"].map(lambda p: counts.get(p, 0)).to_numpy() * n / (n - 1)
        replicate = part.copy()
        replicate["weight"] = replicate["weight"] * factor
        if "weight_unraked" in replicate.columns:
            replicate["weight_unraked"] = replicate["weight_unraked"] * factor
        pieces.append(replicate[replicate["weight"] > 0])
    import pandas as pd
    return pd.concat(pieces)


INCOME_ITEMS = ("median_household_income", "mean_income_adults", "low_income_household_share")


def income_dispersion(frame) -> float:
    """Weighted standard deviation of log adult income in the survey: the observable
    proxy for the world's inequality, which drives how selective response is."""
    adults = frame[(frame["age"] >= 16) & (frame["income"] > 0)]
    x = np.log(adults["income"].to_numpy(dtype=np.float64))
    w = adults["weight"].to_numpy(dtype=np.float64)
    mean = (w * x).sum() / w.sum()
    return float(np.sqrt((w * (x - mean) ** 2).sum() / w.sum()))


def calibrate(dev_packet_dirs, calibration_path: Path,
              params: MethodParams = MethodParams()) -> dict:
    """Fit income nonresponse corrections on development worlds.

    Response is selective on income in ways raking on observables cannot remove, and
    the size of the remaining bias depends on how unequal the world is. Development
    worlds ship their truth, so the method measures its own remaining bias on each (a
    log-ratio per income item at the national level) and fits it as a linear function
    of the survey's own income dispersion. On a hidden world the correction is read off
    that line at the world's observed dispersion, held at the nearer edge of the
    development range when the world lies outside it. With one development world the
    correction is a constant. The residual across worlds is what the accuracy bar
    measures.
    """
    import pandas as pd
    dev_packet_dirs = [Path(d) for d in ([dev_packet_dirs] if isinstance(dev_packet_dirs, (str, Path)) else dev_packet_dirs)]
    rows = []
    for k, dev in enumerate(dev_packet_dirs):
        scratch = Path(calibration_path).parent / f"_calibration_run_{k}"
        result = run(dev, scratch, MethodParams(bootstrap_replicates=10, seed=params.seed))
        truth = pd.read_csv(dev / "participant" / "truth" / "truth_revised.csv")
        nation = truth[truth["level"] == "nation"].set_index("estimand")["value"]
        estimate = {r["estimand"]: r["estimate"] for r in result["release"] if r["level"] == "nation"}
        row = {"dispersion": result["dispersion"]}
        for e in INCOME_ITEMS:
            row[e] = float(nation[e] - estimate[e]) if e == "low_income_household_share" \
                else float(np.log(nation[e] / estimate[e]))
        rows.append(row)
    d = np.asarray([r["dispersion"] for r in rows])
    # The line is read only inside the development range of the dispersion and held
    # at the nearer edge beyond it: three worlds fix a slope too loosely to extrapolate.
    factors = {"dispersion_reference": float(d.mean()), "n_worlds": len(rows),
               "dispersion_range": [float(d.min()), float(d.max())]}
    for e in INCOME_ITEMS:
        y = np.asarray([r[e] for r in rows])
        if len(rows) >= 3 and d.std() > 1e-6:
            slope, intercept = np.polyfit(d, y, 1)
        else:
            slope, intercept = 0.0, float(y.mean())
        residual = y - (intercept + slope * d)
        # The correction is uncertain: its largest development-world miss, never below
        # half the spread of the raw biases, widens the income intervals (the same
        # allowance the Bayesian line carries through ``calibration_half_widths``).
        residual_sd = float(np.abs(residual).max()) if len(rows) >= 2 else float(abs(y).mean())
        residual_sd = max(residual_sd, 0.5 * float(np.std(y))) if len(rows) >= 2 else residual_sd
        factors[e] = {"intercept": float(intercept), "slope": float(slope),
                      "residual_sd": max(residual_sd, 0.01)}
    # The exact-key union control needs the same development worlds; its constants
    # ride along in this receipt so the control battery can run from calibration A.
    from .controls import fit_exact_key_union
    factors["exact_key_union"] = fit_exact_key_union(dev_packet_dirs)
    # County income ratios from the income source are raised to exponents fitted on
    # the same development worlds (``fit_ratio_exponents``); the national corrections
    # above do not depend on them, since this line reads nation and state from the
    # survey directly.
    factors["ratio_exponent"] = fit_ratio_exponents(dev_packet_dirs)
    Path(calibration_path).write_text(json.dumps(factors, indent=1, sort_keys=True) + "\n")
    return factors


def _apply_calibration(values: dict, factors: dict, dispersion: float) -> dict:
    out = dict(values)
    if "dispersion_range" in factors:
        dispersion = float(np.clip(dispersion, *factors["dispersion_range"]))
    for (e, level, u), v in values.items():
        if e not in factors or not np.isfinite(v):
            continue
        f = factors[e]
        shift = f["intercept"] + f["slope"] * dispersion if isinstance(f, dict) else float(f)
        if e == "low_income_household_share":
            out[(e, level, u)] = float(min(max(v + shift, 0.0), 1.0))
        else:
            out[(e, level, u)] = float(v * np.exp(shift))
    return out


def run(packet_dir: Path, out_dir: Path, params: MethodParams = MethodParams()) -> dict:
    import pandas as pd
    data = load_packet(packet_dir)
    contract, county_state = data["contract"], data["county_state"]
    n_counties = len(county_state)
    tick = int(contract["ticks"]["revised"])
    horizon_months = int(contract["ticks"]["horizon"]) - tick
    rng = np.random.default_rng(params.seed)

    register_frame = corroborate_counties(
        deduplicate_population(data["population"], tick, data["income"], data.get("health")), data["income"])
    miscoding = estimate_county_error_rate(data["population_preliminary"], data["population"],
                                           data["income"], int(contract["ticks"]["preliminary"]), tick)
    register = register_counts(register_frame, n_counties, miscoding["rate"])
    mortality = estimate_mortality(data["population_preliminary"], data["population"],
                                   int(contract["ticks"]["preliminary"]), tick)
    fertility = estimate_fertility(register_frame, tick)
    unraked = impute_income(adjusted_survey(data["survey"]))
    survey = impute_income(rake_to_register(adjusted_survey(data["survey"]), register_frame, county_state,
                                            county_persons=register["persons"]))
    base_stats = survey_statistics(survey, county_state)
    # Nonresponse sensitivity: response is selective on income beyond what raking on
    # observables removes. The raking shift is the visible part of that bias; the
    # income intervals are widened by it, as a sensitivity allowance the bootstrap
    # cannot see.
    unraked_stats = survey_statistics(unraked, county_state)
    sensitivity = {}
    for e in ("median_household_income", "mean_income_adults", "low_income_household_share"):
        shift = {k: abs(base_stats[e][k] - unraked_stats[e][k]) for k in base_stats[e]}
        sensitivity[e] = shift
    scale = register_income_scale(data["income"], base_stats["mean_income_adults"]["nation"])
    factors = json.loads(Path(params.calibration_path).read_text()) if params.calibration_path else {}
    ratios = apply_ratio_exponents(
        income_source_ratios(data["income"], county_state, base_stats["median_household_income"]["nation"],
                             scale, register_frame=register_frame),
        factors.get("ratio_exponent"))

    point = estimate_once(survey, register, ratios, county_state)
    model_rel_sd = float(point["county"].pop("_model_rel_sd", 0.0))
    income_model_rel_sd = dict(point["county"].pop("_income_model_rel_sd", {}))
    now = aggregate(point["county"], county_state, point["state_stats"], point["county"]["persons"])
    future_point = project(point["county"], point["age_sex"], horizon_months, np.random.default_rng(params.seed + 1), mortality, fertility)
    future = aggregate(future_point, county_state, point["state_stats"], future_point["persons"])

    dispersion = income_dispersion(survey)
    now, future = _apply_calibration(now, factors, dispersion), _apply_calibration(future, factors, dispersion)
    now_reps, future_reps = [], []
    age_sex_paths = [np.asarray(point["age_sex"], dtype=np.float64)]
    for b in range(params.bootstrap_replicates):
        replicate = _bootstrap_frame(survey, rng)
        est = estimate_once(replicate, register, ratios, county_state)
        est["county"].pop("_model_rel_sd", None)
        est["county"].pop("_income_model_rel_sd", None)
        age_sex_paths.append(np.asarray(est["age_sex"], dtype=np.float64))
        now_reps.append(_apply_calibration(
            aggregate(est["county"], county_state, est["state_stats"], est["county"]["persons"]), factors, dispersion))
        fut = project(est["county"], est["age_sex"], horizon_months, rng, mortality, fertility)
        future_reps.append(_apply_calibration(
            aggregate(fut, county_state, est["state_stats"], fut["persons"]), factors, dispersion))
    # Benchmark reconciliation of the four national counts; the same factor at every
    # level and in every replicate keeps the counts additive.
    reconciliation = benchmark_reconciliation(now, now_reps, data.get("benchmark"))
    state_reconciliation = benchmark_state_reconciliation(
        now, now_reps, data.get("benchmark"), county_state, reconciliation)
    now = apply_reconciliation(now, reconciliation, state_reconciliation, county_state)
    now_reps = [apply_reconciliation(r, reconciliation, state_reconciliation, county_state)
                for r in now_reps]
    future = apply_reconciliation(future, reconciliation, state_reconciliation, county_state)
    future_reps = [apply_reconciliation(r, reconciliation, state_reconciliation, county_state)
                   for r in future_reps]

    def rows(point_values: dict, replicates: list[dict], widen: float = 1.0) -> list[dict]:
        out = []
        for key in sorted(point_values):
            v = point_values[key]
            draws = np.asarray([r[key] for r in replicates], dtype=np.float64)
            draws = draws[np.isfinite(draws)]
            if not np.isfinite(v):
                v, lower, upper = 0.0, 0.0, 0.0
            elif len(draws) < 10:
                lower, upper = v, v
            else:
                lo, hi = np.percentile(draws, [5, 95])
                half = 0.5 * (hi - lo) * widen
                if key[1] == "county" and key[0] in ("persons", "households", "children_under_16", "elders_65_plus"):
                    half = np.sqrt(half ** 2 + (1.645 * model_rel_sd * abs(v)) ** 2)
                if key[1] != "county" and key[0] in ("persons", "households", "children_under_16", "elders_65_plus"):
                    half = np.sqrt(half ** 2 + (1.645 * REGISTER_MODEL_RELATIVE_SD * abs(v)) ** 2)
                if key[1] == "county" and key[0] in sensitivity:
                    # Synthetic county income estimates carry model error beyond the
                    # survey bootstrap: the measured relative model error for the
                    # median and the mean (never below ten percent), ten percent for
                    # the share.
                    rel = income_model_rel_sd.get(key[0], 0.10)
                    extra = 1.645 * rel * (abs(v) if key[0] != "low_income_household_share" else 0.5)
                    half = float(np.sqrt(half ** 2 + extra ** 2))
                if widen > 1.0 and key[0] in sensitivity:
                    # Carried-forward income items drift over the horizon; five percent a
                    # year in quadrature, the same allowance the Bayesian line carries.
                    drift = 1.645 * 0.05 * np.sqrt(horizon_months / 12.0)
                    half = float(np.sqrt(half ** 2 + (drift * (abs(v) if key[0] != "low_income_household_share" else 0.5)) ** 2))
                if widen > 1.0 and key[0] == "tertiary_share_25_plus":
                    half = float(np.sqrt(half ** 2 + (0.03 * horizon_months / 60.0) ** 2))
                if widen > 1.0 and key[0] in ("persons", "households", "children_under_16", "elders_65_plus"):
                    half = float(np.sqrt(half ** 2 + (1.645 * 0.03 * np.sqrt(horizon_months / 12.0) * abs(v)) ** 2))
                if key[0] == "tertiary_share_25_plus":
                    # Register share: binomial spread over the known-education base,
                    # plus an allowance for item-missing education being selective.
                    n_base = tertiary_base.get(key, 1.0)
                    half = float(np.sqrt(half ** 2 + (1.645 * np.sqrt(max(v * (1 - v), 1e-6) / max(n_base, 1.0))) ** 2 + 0.02 ** 2))
                if key[0] in factors and isinstance(factors[key[0]], dict):
                    sd = factors[key[0]].get("residual_sd", 0.0)
                    extra = 1.645 * sd if key[0] == "low_income_household_share" else 1.645 * sd * abs(v)
                    half = float(np.sqrt(half ** 2 + extra ** 2))
                if key[0] in sensitivity:
                    unit_key = "nation" if key[1] == "nation" else \
                        (int(key[2]) if key[1] == "state" else int(county_state[key[2]]))
                    half += params.sensitivity_multiplier * sensitivity[key[0]].get(unit_key, 0.0)
                center = v
                lower, upper = center - half, center + half
            kind = "proportion" if key[0].endswith("share") or key[0].startswith("tertiary") else "count"
            lower = max(lower, 0.0)
            if kind == "proportion":
                upper = min(upper, 1.0)
                v = min(max(v, lower), upper)
            out.append({"estimand": key[0], "level": key[1], "unit": int(key[2]),
                        "estimate": float(v), "lower": float(min(lower, v)), "upper": float(max(upper, v))})
        return out

    tertiary_base = {("tertiary_share_25_plus", "county", c): float(register["tertiary_n"][c]) for c in range(n_counties)}
    for s_ in range(int(county_state.max()) + 1):
        tertiary_base[("tertiary_share_25_plus", "state", s_)] = float(register["tertiary_n"][county_state == s_].sum())
    tertiary_base[("tertiary_share_25_plus", "nation", 0)] = float(register["tertiary_n"].sum())
    release_rows = rows(now, now_reps)
    projection_rows = rows(future, future_reps, widen=params.carry_forward_width)

    # Detailed table: primary suppression of small estimated cells, no totals published.
    threshold = int(contract["disclosure_threshold"])
    cube = point["cube"]
    detail = []
    for c in range(n_counties):
        for b, band in enumerate(AGE_BAND_LABELS):
            for s, sex in enumerate(SEX_LABELS):
                value = cube[c, b, s]
                suppressed = 0 < value < params.suppression_multiplier * threshold
                detail.append({"county": c, "age_band": band, "sex": sex,
                               "count": "" if suppressed else round(float(value), 3)})

    elders = np.maximum(future_point["elders_65_plus"], 0.0) * reconciliation.get("elders_65_plus", 1.0)
    budget = float(contract["allocation"]["budget"])
    allocation = elders / max(elders.sum(), 1e-9) * budget
    allocation = np.floor(allocation * 1e6) / 1e6          # never over budget by rounding

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    result = {"release": release_rows, "projection": projection_rows,
              "dispersion": dispersion, "n_bootstrap": params.bootstrap_replicates,
              "mortality": mortality, "fertility": fertility, "miscoding": miscoding,
              "register_income_scale": scale, "reconciliation": reconciliation}
    # Version four: the same reconstruction, carried on into exposures and rates, a
    # simulated liability distribution, and the reserve. The bootstrap replicates of the
    # age cube are the population draws the layer propagates, so the design-based line's
    # own sampling uncertainty reaches the liability tails.
    actuarial = None
    if params.actuarial != "off":
        # The liability is priced on the same reconstruction the release table carries, so
        # the county cube takes the national benchmark factor and its own state's factor.
        # Without the second one the reserve is priced on a population whose split across
        # states still carries the coverage gradient, which is where the register is
        # weakest and where a regional tail is decided.
        # The liability is priced on the same reconstruction the release table carries, so
        # the county age cube is raked to all three published count items, by state.
        paths = np.stack(age_sex_paths)
        scale = benchmark_age_scale(paths[0], county_state, reconciliation,
                                    state_reconciliation)
        actuarial = AR.actuarial_submission(
            Path(packet_dir), data, county_state,
            paths * scale[None, :, :, None],
            float(fertility["fertility_rate"]),
            release_rows, projection_rows, cube,
            params.suppression_multiplier * threshold, out_dir,
            AGE_BAND_LABELS, SEX_LABELS,
            params.actuarial_params or AR.LayerParams())
    if actuarial is None:
        if params.actuarial == "on":
            raise AR.MissingActuarialInputs(
                "packet carries no experience file or reserve block")
        pd.DataFrame(release_rows).to_csv(out_dir / "release.csv", index=False)
        pd.DataFrame(projection_rows).to_csv(out_dir / "projection.csv", index=False)
        pd.DataFrame(detail).to_csv(out_dir / "detailed.csv", index=False)
        pd.DataFrame({"county": np.arange(n_counties), "allocation": allocation}).to_csv(
            out_dir / "allocation.csv", index=False)
        return result
    result["release"] = actuarial["release"]
    result["reserve"] = actuarial["reserve"]
    result["actuarial"] = actuarial["diagnostics"]
    return result
