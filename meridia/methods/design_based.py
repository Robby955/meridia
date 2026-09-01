"""Strong method A: the design-based line.

A classical production line that a survey methodologist would write from the packet:

1. Deduplicate the population source on (name code, birth month, sex); count persons,
   households, children, and elders by county from the deduplicated records.
2. Adjust the survey's design weights for unit nonresponse within each sampling unit,
   using the public design constant of households sampled per unit.
3. Estimate the state-level coverage of the population source as the ratio of the
   nonresponse-adjusted survey estimate to the deduplicated register count, and scale
   county register counts by their state's coverage (synthetic small-area estimation).
4. Impute item-missing survey income by a deterministic hot deck within stratum,
   education, and age band; estimate income statistics from the survey at nation and
   state level, and at county level synthetically through the income source.
5. Intervals from a rescaled bootstrap over sampling units within strata; counts are
   built county-up so they add exactly.
6. Project the population to the horizon by a cohort-component step with mortality and
   fertility drawn from the public parameter ranges; allocate the budget in proportion
   to projected elders.
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
from ..survey import SurveyParams

HOUSEHOLDS_PER_PSU = SurveyParams().households_per_cell   # public design constant
MAX_AGE = 100
NOMINAL = 0.90


@dataclass(frozen=True)
class MethodParams:
    bootstrap_replicates: int = 200
    seed: int = 20260901
    suppression_multiplier: float = 2.0    # suppress estimated cells below this x threshold
    carry_forward_width: float = 1.5       # projection interval widening for income items
    sensitivity_multiplier: float = 2.0    # income half-width += this x the raking shift
    calibration_path: str | None = None    # JSON from calibrate() on a development world


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
        "income": pd.read_csv(P / "sources" / "income_revised.csv"),
    }


# --------------------------------------------------------------------------- register

def deduplicate_population(population, tick: int):
    """One row per (name_code, birth_tick, sex); age in years at ``tick``."""
    frame = population.drop_duplicates(subset=["name_code", "birth_tick", "sex"]).copy()
    frame["age"] = (tick - frame["birth_tick"]) // 12
    frame = frame[frame["county"] >= 0]
    return frame


def register_counts(frame, n_counties: int) -> dict:
    county = frame["county"].to_numpy(dtype=np.int64)
    age = frame["age"].to_numpy(dtype=np.int64)
    counts = {
        "persons": np.bincount(county, minlength=n_counties),
        "children_under_16": np.bincount(county, weights=(age <= 15), minlength=n_counties),
        "elders_65_plus": np.bincount(county, weights=(age >= 65), minlength=n_counties),
        "households": frame.groupby("county")["household_id"].nunique()
                           .reindex(range(n_counties), fill_value=0).to_numpy(dtype=np.float64),
    }
    over_25 = age >= 25
    known = frame["education"].to_numpy() >= 0
    tertiary = frame["education"].to_numpy() >= 2
    n_known = np.bincount(county, weights=(over_25 & known), minlength=n_counties)
    n_tert = np.bincount(county, weights=(over_25 & known & tertiary), minlength=n_counties)
    with np.errstate(invalid="ignore", divide="ignore"):
        counts["tertiary_share_25_plus"] = np.where(n_known > 0, n_tert / n_known, np.nan)
    counts["tertiary_n"] = n_known
    sex = frame["sex"].to_numpy(dtype=np.int64)
    band = np.full(len(age), -1)
    for b, (lo, hi) in enumerate(AGE_BANDS):
        band[(age >= lo) & (age <= hi)] = b
    cube = np.zeros((n_counties, len(AGE_BANDS), 2))
    np.add.at(cube, (county, np.maximum(band, 0), sex), 1)
    counts["cube"] = cube
    age_sex = np.zeros((n_counties, MAX_AGE + 1, 2))
    np.add.at(age_sex, (county, np.clip(age, 0, MAX_AGE), sex), 1)
    counts["age_sex"] = age_sex
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


def rake_to_register(frame, register_frame, county_state: np.ndarray, iterations: int = 12):
    """Rake survey weights within each state to the register's age-band x sex and
    education proportions. Proportions, not totals, so register coverage cancels; the
    point is to counter response that is selective on income through its correlates."""
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
    for s in range(int(county_state.max()) + 1):
        sv = np.flatnonzero(frame["state"].to_numpy() == s)
        rg = reg_state == s
        if len(sv) == 0 or rg.sum() == 0:
            continue
        total = weight[sv].sum()
        for _ in range(iterations):
            for sv_cells, reg_cells in ((frame["cell_as"].to_numpy()[sv], reg_cell_as[rg]),
                                        (frame["cell_edu"].to_numpy()[sv], reg_cell_edu[rg]),
                                        (frame["cell_county"].to_numpy()[sv], reg_cell_county[rg])):
                keep = reg_cells >= 0
                target = np.bincount(reg_cells[keep], minlength=n_cells) / max(keep.sum(), 1)
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

def income_source_ratios(income, county_state: np.ndarray, national_median_hh: float) -> dict:
    """County-to-state ratios from the income source, for synthetic county estimates."""
    n_counties = len(county_state)
    frame = income[income["county"] >= 0].copy()
    frame["employment_income_cents"] = frame["employment_income_cents"].fillna(0.0)
    positive = frame[frame["employment_income_cents"] > 0]
    county_mean = positive.groupby("county")["employment_income_cents"].mean() \
                          .reindex(range(n_counties)).to_numpy()
    hh = frame.groupby(["county", "household_id"])["employment_income_cents"].sum().reset_index()
    hh["low"] = hh["employment_income_cents"] < LOW_INCOME_FRACTION * national_median_hh * 100.0
    county_low = hh.groupby("county")["low"].mean().reindex(range(n_counties)).to_numpy()
    county_median = hh.groupby("county")["employment_income_cents"].median() \
                      .reindex(range(n_counties)).to_numpy()
    ratios = {}
    for name, values in (("mean_income_adults", county_mean),
                         ("median_household_income", county_median),
                         ("low_income_household_share", county_low)):
        state_values = np.asarray([np.nanmean(values[county_state == s]) if np.isfinite(values[county_state == s]).any() else np.nan
                                   for s in range(int(county_state.max()) + 1)])
        with np.errstate(invalid="ignore", divide="ignore"):
            r = values / state_values[county_state]
        ratios[name] = np.where(np.isfinite(r), r, 1.0)
    return ratios


# -------------------------------------------------------------------- one estimate

def estimate_once(frame, register: dict, ratios: dict, county_state: np.ndarray) -> dict:
    """County-level point estimates for all estimands from one survey replicate."""
    n_counties = len(county_state)
    n_states = int(county_state.max()) + 1
    stats = survey_statistics(frame, county_state)
    reg_state = np.bincount(county_state, weights=register["persons"], minlength=n_states)
    with np.errstate(invalid="ignore", divide="ignore"):
        coverage = np.where(reg_state > 0, reg_state / stats["persons_by_state"], 1.0)
    coverage = np.clip(coverage, 0.5, 1.2)
    scale = 1.0 / coverage[county_state]
    # Small counties are covered worse by the register than large ones. Pool the
    # survey's direct estimate over the smallest quartile of counties nationally and
    # estimate their coverage as a class, since no single small county has the sample
    # to estimate its own.
    direct_all, _ = _direct_county_persons(frame, n_counties)
    reg_persons = np.asarray(register["persons"], dtype=np.float64)
    small = reg_persons <= np.quantile(reg_persons[reg_persons > 0], 0.25) if (reg_persons > 0).sum() >= 8 else np.zeros(n_counties, bool)
    if small.sum() >= 2 and direct_all[small].sum() > 0:
        coverage_small = float(np.clip(reg_persons[small].sum() / direct_all[small].sum(), 0.5, 1.2))
        scale = np.where(small, 1.0 / coverage_small, scale)
    county = {e: register[e] * scale for e in ("persons", "households", "children_under_16", "elders_65_plus")}
    # County persons: combine the synthetic estimate with the direct survey estimate
    # (Fay-Herriot). The direct estimate is design-unbiased but noisy; its variance is
    # approximated from the sampling units in the county; the model variance of the
    # synthetic estimate is set by the method of moments across counties.
    direct, direct_var = _direct_county_persons(frame, n_counties)
    synthetic = county["persons"].copy()
    n_psu = frame.groupby("county")["psu"].nunique().reindex(range(n_counties), fill_value=0).to_numpy()
    have = (direct > 0) & np.isfinite(direct_var) & (direct_var > 0) & (n_psu >= 4)
    if have.sum() >= 3:
        residual = direct[have] - synthetic[have]
        model_var = max(float(np.mean(residual ** 2) - np.mean(direct_var[have])), 0.0)
        gamma = np.where(have, model_var / np.maximum(model_var + direct_var, 1e-9), 0.0)
        combined = np.where(have, gamma * direct + (1.0 - gamma) * synthetic, synthetic)
        ratio = combined / np.maximum(synthetic, 1e-9)
        for e in ("persons", "households", "children_under_16", "elders_65_plus"):
            county[e] = county[e] * ratio
        # Relative model error of the synthetic county estimate, from the residuals of
        # the well-sampled counties net of their sampling variance; carried into the
        # county intervals, which a survey bootstrap alone cannot see.
        rel_resid = residual / np.maximum(synthetic[have], 1e-9)
        rel_sampling = direct_var[have] / np.maximum(synthetic[have], 1e-9) ** 2
        county["_model_rel_sd"] = max(float(np.sqrt(max(np.mean(rel_resid ** 2) - np.mean(rel_sampling), 0.0))), 0.10)
    else:
        county["_model_rel_sd"] = 0.15
    county["tertiary_share_25_plus"] = register["tertiary_share_25_plus"]
    for e in ("median_household_income", "mean_income_adults", "low_income_household_share"):
        state_values = np.asarray([stats[e][s] for s in range(n_states)])
        county[e] = state_values[county_state] * ratios[e]
        if e == "low_income_household_share":
            county[e] = np.clip(county[e], 0.0, 1.0)
    return {"county": county, "state_stats": stats, "cube": register["cube"] * scale[:, None, None],
            "age_sex": register["age_sex"] * scale[:, None, None]}


def _direct_county_persons(frame, n_counties: int) -> tuple[np.ndarray, np.ndarray]:
    """Direct survey estimate of persons by county and a between-unit variance proxy."""
    county = frame["county"].to_numpy(dtype=np.int64)
    w = frame["weight"].to_numpy(dtype=np.float64)
    total = np.bincount(county, weights=w, minlength=n_counties)
    per_psu = frame.groupby(["county", "psu"])["weight"].sum()
    var = np.full(n_counties, np.nan)
    for c, part in per_psu.groupby(level=0):
        values = part.to_numpy()
        n = len(values)
        var[int(c)] = n * np.var(values, ddof=1) if n >= 2 else np.nan
    return total, var


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

def project(county_values: dict, age_sex: np.ndarray, months: int, rng: np.random.Generator) -> dict:
    """Cohort-component projection on single-year ages.

    Mortality and fertility are drawn from the public ranges; a shock year is drawn
    with the frequency the public shock family implies. Each simulated year: survivors
    age by one, births arrive to women aged 18 to 45 and survive infancy, and the
    open-ended top age absorbs. Income items are carried forward; household counts
    follow the person growth.
    """
    lo_a, hi_a = CHARACTER_RANGES["gompertz_a"]
    lo_f, hi_f = CHARACTER_RANGES["fertility_rate"]
    params = DemographyParams(gompertz_a=float(rng.uniform(lo_a, hi_a)),
                              fertility_rate=float(rng.uniform(lo_f, hi_f)))
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
        fertility = params.fertility_rate * (float(rng.uniform(0.45, 0.75)) if year in shock_years and rng.random() < 0.33 else 1.0)
        births = women * fertility * survival[0]
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
    that line at the world's observed dispersion. With one development world the
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
    factors = {"dispersion_reference": float(d.mean()), "n_worlds": len(rows)}
    for e in INCOME_ITEMS:
        y = np.asarray([r[e] for r in rows])
        if len(rows) >= 3 and d.std() > 1e-6:
            slope, intercept = np.polyfit(d, y, 1)
        else:
            slope, intercept = 0.0, float(y.mean())
        factors[e] = {"intercept": float(intercept), "slope": float(slope)}
    Path(calibration_path).write_text(json.dumps(factors, indent=1, sort_keys=True) + "\n")
    return factors


def _apply_calibration(values: dict, factors: dict, dispersion: float) -> dict:
    out = dict(values)
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

    register_frame = deduplicate_population(data["population"], tick)
    register = register_counts(register_frame, n_counties)
    unraked = impute_income(adjusted_survey(data["survey"]))
    survey = impute_income(rake_to_register(adjusted_survey(data["survey"]), register_frame, county_state))
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
    ratios = income_source_ratios(data["income"], county_state,
                                  base_stats["median_household_income"]["nation"])

    point = estimate_once(survey, register, ratios, county_state)
    model_rel_sd = float(point["county"].pop("_model_rel_sd", 0.0))
    now = aggregate(point["county"], county_state, point["state_stats"], point["county"]["persons"])
    future_point = project(point["county"], point["age_sex"], horizon_months, np.random.default_rng(params.seed + 1))
    future = aggregate(future_point, county_state, point["state_stats"], future_point["persons"])

    factors = json.loads(Path(params.calibration_path).read_text()) if params.calibration_path else {}
    dispersion = income_dispersion(survey)
    now, future = _apply_calibration(now, factors, dispersion), _apply_calibration(future, factors, dispersion)
    now_reps, future_reps = [], []
    for b in range(params.bootstrap_replicates):
        replicate = _bootstrap_frame(survey, rng)
        est = estimate_once(replicate, register, ratios, county_state)
        est["county"].pop("_model_rel_sd", None)
        now_reps.append(_apply_calibration(
            aggregate(est["county"], county_state, est["state_stats"], est["county"]["persons"]), factors, dispersion))
        fut = project(est["county"], est["age_sex"], horizon_months, rng)
        future_reps.append(_apply_calibration(
            aggregate(fut, county_state, est["state_stats"], fut["persons"]), factors, dispersion))

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
                if key[1] == "county" and key[0] in sensitivity:
                    # Synthetic county income estimates carry model error beyond the
                    # survey bootstrap: a ten percent relative allowance.
                    extra = 1.645 * 0.10 * (abs(v) if key[0] != "low_income_household_share" else 0.5)
                    half = float(np.sqrt(half ** 2 + extra ** 2))
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

    elders = np.maximum(future_point["elders_65_plus"], 0.0)
    budget = float(contract["allocation"]["budget"])
    allocation = elders / max(elders.sum(), 1e-9) * budget
    allocation = np.floor(allocation * 1e6) / 1e6          # never over budget by rounding

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(release_rows).to_csv(out_dir / "release.csv", index=False)
    pd.DataFrame(projection_rows).to_csv(out_dir / "projection.csv", index=False)
    pd.DataFrame(detail).to_csv(out_dir / "detailed.csv", index=False)
    pd.DataFrame({"county": np.arange(n_counties), "allocation": allocation}).to_csv(
        out_dir / "allocation.csv", index=False)
    return {"release": release_rows, "projection": projection_rows,
            "dispersion": dispersion, "n_bootstrap": params.bootstrap_replicates}
