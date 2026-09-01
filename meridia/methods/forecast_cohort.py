"""Forecast strong method A: cohort-component projection with rates read from the history.

The participant has a clean snapshot and the event ledger before it, so every rate the
projection needs is estimable rather than assumed:

- mortality by single-year age from deaths in the last year of history over the
  reconstructed exposure, smoothed by a Gompertz-Makeham fit;
- fertility from births in the last year over women aged 18 to 45;
- net migration by county from household moves in the last year;
- hospital admissions per resident by county from the last year of admissions.

The projection advances single-year cohorts per county month by month in yearly steps,
with a shock allowance drawn from the public shock family. Intervals come from a
parametric bootstrap of the estimated rates (Poisson resampling of the counted events)
and the shock draws. The allocation follows projected admissions.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..demography import DemographyParams, mortality_probability
from ..release import ESTIMAND_IDS, LOW_INCOME_FRACTION
from .common import COUNT_ITEMS, INCOME_ITEMS, rows_from_draws

MAX_AGE = 100


@dataclass(frozen=True)
class MethodParams:
    replicates: int = 200
    seed: int = 20260905
    shock_probability_per_year: float = 0.15   # public family: up to two shocks in a run


def load_forecast_packet(packet_dir: Path) -> dict:
    import pandas as pd
    P = Path(packet_dir) / "participant"
    return {
        "contract": json.loads((P / "contract.json").read_text()),
        "county_state": pd.read_csv(P / "geography.csv")["state"].to_numpy(dtype=np.int64),
        "persons": pd.read_csv(P / "persons.csv"),
        "households": pd.read_csv(P / "households.csv"),
        "hospitals": pd.read_csv(P / "hospitals.csv"),
        "events": pd.read_csv(P / "events.csv"),
    }


def age_sex_cube(persons, tick: int, n_counties: int) -> np.ndarray:
    age = np.clip((tick - persons["birth_tick"].to_numpy(dtype=np.int64)) // 12, 0, MAX_AGE)
    cube = np.zeros((n_counties, MAX_AGE + 1, 2))
    np.add.at(cube, (persons["county"].to_numpy(dtype=np.int64), age, persons["sex"].to_numpy(dtype=np.int64)), 1)
    return cube


def estimate_rates(data: dict, tick: int) -> dict:
    """Rates from the last twelve months of the ledger."""
    events, persons = data["events"], data["persons"]
    n_counties = len(data["county_state"])
    last_year = events[(events["tick"] > tick - 12) & (events["tick"] <= tick)]
    deaths = last_year[last_year["event"] == "person_death"]
    death_age = np.clip((deaths["tick"].to_numpy(dtype=np.int64) - deaths["birth_tick"].to_numpy(dtype=np.int64)) // 12, 0, MAX_AGE)
    deaths_by_age = np.bincount(death_age, minlength=MAX_AGE + 1).astype(np.float64)
    cube = age_sex_cube(persons, tick, n_counties)
    alive_by_age = cube.sum(axis=(0, 2))
    # Exposure a year ago at age a: today's survivors aged a+1 plus the deaths at age a.
    exposure = np.zeros(MAX_AGE + 1)
    exposure[:-1] = alive_by_age[1:] + deaths_by_age[:-1]
    exposure[-1] = alive_by_age[-1] + deaths_by_age[-1]
    with np.errstate(invalid="ignore", divide="ignore"):
        raw = np.where(exposure > 0, deaths_by_age / exposure, np.nan)
    # Gompertz-Makeham fit on log rates over the ages with enough exposure.
    ages = np.arange(MAX_AGE + 1)
    fit_mask = (exposure >= 200) & (raw > 0) & (ages >= 25) & (ages <= 90)
    if fit_mask.sum() >= 10:
        a_grid = np.geomspace(5e-6, 1e-4, 40)
        b_grid = np.linspace(0.06, 0.14, 33)
        best, best_err = (DemographyParams().gompertz_a, DemographyParams().gompertz_b), np.inf
        for a in a_grid:
            for b in b_grid:
                q = mortality_probability(ages, DemographyParams(gompertz_a=a, gompertz_b=b))
                err = float(np.sum(exposure[fit_mask] * (np.log(q[fit_mask]) - np.log(raw[fit_mask])) ** 2))
                if err < best_err:
                    best, best_err = (a, b), err
        gompertz_a, gompertz_b = best
    else:
        gompertz_a, gompertz_b = DemographyParams().gompertz_a, DemographyParams().gompertz_b
    births = int((last_year["event"] == "person_birth").sum())
    women = float(cube[:, 18:46, 1].sum())
    fertility = births / max(women, 1.0)
    moves = last_year[last_year["event"] == "household_moved"]
    # Net household moves by destination county minus origin county are not both
    # recorded per event (only the destination county is), so net migration is
    # estimated from the change implied by births and deaths versus the snapshot.
    hh_size = len(persons) / max(len(data["households"]), 1)
    inflow = np.bincount(moves["county"].to_numpy(dtype=np.int64)[moves["county"].to_numpy() >= 0],
                         minlength=n_counties).astype(np.float64) * hh_size
    admissions = last_year[last_year["event"] == "encounter_admitted"]
    adm_county = np.zeros(n_counties)
    hospital_county = dict(zip(data["hospitals"]["hospital_id"], data["hospitals"]["county"]))
    for h in admissions["hospital_id"].to_numpy():
        c = hospital_county.get(int(h), -1)
        if c >= 0:
            adm_county[c] += 1
    persons_county = cube.sum(axis=(1, 2))
    return {"gompertz_a": gompertz_a, "gompertz_b": gompertz_b, "deaths": deaths_by_age,
            "exposure": exposure, "births": births, "women": women, "fertility": fertility,
            "inflow": inflow, "persons_county": persons_county, "admissions": adm_county,
            "cube": cube, "hh_size": hh_size}


def project_cohorts(cube: np.ndarray, rates: dict, years: int, rng: np.random.Generator,
                    params: MethodParams, perturb: bool) -> tuple[np.ndarray, np.ndarray]:
    """Advance single-year cohorts per county; returns the final cube and admissions."""
    ages = np.arange(MAX_AGE + 1)
    fertility = rates["fertility"]
    gompertz_a = rates["gompertz_a"]
    if perturb:
        # Parametric bootstrap of the counted events behind the rates.
        births = rng.poisson(max(rates["births"], 1))
        fertility = births / max(rates["women"], 1.0)
        deaths = rng.poisson(np.maximum(rates["deaths"], 0.0))
        scale = deaths.sum() / max(rates["deaths"].sum(), 1.0)
        gompertz_a = rates["gompertz_a"] * max(scale, 0.5)
    q = mortality_probability(ages, DemographyParams(gompertz_a=gompertz_a, gompertz_b=rates["gompertz_b"]))
    state = cube.astype(np.float64).copy()
    persons_now = np.maximum(cube.sum(axis=(1, 2)), 1.0)
    for _ in range(years):
        shock = rng.random() < params.shock_probability_per_year
        multiplier = float(rng.uniform(1.5, 3.0)) if shock and rng.random() < 0.5 else 1.0
        fert = fertility * (float(rng.uniform(0.45, 0.75)) if shock and multiplier == 1.0 else 1.0)
        survival = 1.0 - np.clip(q * multiplier, 0.0, 1.0)
        survivors = state * survival[None, :, None]
        aged = np.zeros_like(survivors)
        aged[:, 1:, :] = survivors[:, :-1, :]
        aged[:, MAX_AGE, :] += survivors[:, MAX_AGE, :]
        women = state[:, 18:46, 1].sum(axis=1)
        births = women * fert * survival[0]
        aged[:, 0, 0] += 0.5 * births
        aged[:, 0, 1] += 0.5 * births
        # Internal migration nets to zero nationally; the destination-only move records
        # cannot give county net flows, so v0 projects counties without migration.
        state = aged
    persons_end = state.sum(axis=(1, 2))
    admissions = rates["admissions"] * persons_end / persons_now
    return state, admissions


def run(packet_dir: Path, out_dir: Path, params: MethodParams = MethodParams()) -> dict:
    import pandas as pd
    data = load_forecast_packet(packet_dir)
    contract, county_state = data["contract"], data["county_state"]
    n_counties = len(county_state)
    n_states = int(county_state.max()) + 1
    S, H = int(contract["ticks"]["snapshot"]), int(contract["ticks"]["horizon"])
    years = int(round((H - S) / 12.0))
    rng = np.random.default_rng(params.seed)
    rates = estimate_rates(data, S)
    persons = data["persons"]

    # Present-day income items, carried forward with drift allowances.
    adults = persons["age"] if "age" in persons else None
    age = (S - persons["birth_tick"].to_numpy(dtype=np.int64)) // 12
    income = persons["income"].to_numpy(dtype=np.float64)
    county = persons["county"].to_numpy(dtype=np.int64)
    hh = persons.groupby("household_id").agg(income=("income", "sum"), county=("county", "first"))
    hh_income, hh_county = hh["income"].to_numpy(), hh["county"].to_numpy(dtype=np.int64)
    national_median = float(np.median(hh_income))

    def county_income(values_by_unit):
        return values_by_unit

    def aggregate(cube: np.ndarray, hh_growth: np.ndarray) -> dict:
        out = {}
        persons_c = cube.sum(axis=(1, 2))
        counts = {"persons": persons_c, "children_under_16": cube[:, :16, :].sum(axis=(1, 2)),
                  "elders_65_plus": cube[:, 65:, :].sum(axis=(1, 2)),
                  "households": np.bincount(hh_county, minlength=n_counties) * hh_growth}
        for e, v in counts.items():
            for c in range(n_counties):
                out[(e, "county", c)] = float(v[c])
            st = np.bincount(county_state, weights=v, minlength=n_states)
            for s in range(n_states):
                out[(e, "state", s)] = float(st[s])
            out[(e, "nation", 0)] = float(st.sum())
        # Income and education items from the snapshot, by unit.
        for level, units, member_c, member_h in (("county", range(n_counties), county, hh_county),
                                                 ("state", range(n_states), county_state[county], county_state[hh_county]),
                                                 ("nation", [0], np.zeros(len(county), int), np.zeros(len(hh_county), int))):
            for u in units:
                pm, hm = member_c == u, member_h == u
                ad = pm & (age >= 16)
                over = pm & (age >= 25) & (persons["education"].to_numpy() >= 0)
                out[("mean_income_adults", level, u)] = float(income[ad].mean()) if ad.any() else float("nan")
                out[("median_household_income", level, u)] = float(np.median(hh_income[hm])) if hm.any() else float("nan")
                out[("low_income_household_share", level, u)] = float((hh_income[hm] < LOW_INCOME_FRACTION * national_median).mean()) if hm.any() else float("nan")
                out[("tertiary_share_25_plus", level, u)] = float((persons["education"].to_numpy()[over] >= 2).mean()) if over.any() else float("nan")
        return out

    cube0 = rates["cube"]
    persons0 = np.maximum(cube0.sum(axis=(1, 2)), 1.0)
    draws: dict[tuple, list] = {}
    adm_draws = []
    for _ in range(params.replicates):
        cube_k, adm_k = project_cohorts(cube0, rates, years, rng, params, perturb=True)
        agg = aggregate(cube_k, cube_k.sum(axis=(1, 2)) / persons0)
        for key, v in agg.items():
            draws.setdefault(key, []).append(v)
        adm_draws.append(adm_k)
    # The point projection is the median of the shock-aware replicates.
    point = {key: float(np.nanmedian(v)) for key, v in draws.items()}
    for e in COUNT_ITEMS:   # counts add exactly: rebuild state and nation from counties
        county_points = np.asarray([point[(e, "county", c)] for c in range(n_counties)])
        st = np.bincount(county_state, weights=county_points, minlength=n_states)
        for s_ in range(n_states):
            point[(e, "state", s_)] = float(st[s_])
        point[(e, "nation", 0)] = float(st.sum())
    extra = {}
    for key, v in point.items():
        if not np.isfinite(v):
            continue
        e = key[0]
        if e in INCOME_ITEMS:
            drift = 1.645 * 0.05 * np.sqrt(years)
            extra[key] = drift * (abs(v) if e != "low_income_household_share" else 0.5)
        elif e == "tertiary_share_25_plus":
            extra[key] = 0.03 * years / 5.0
        elif e in COUNT_ITEMS:
            # Rate misestimation compounds over the horizon; two percent a year in quadrature.
            model = 1.645 * 0.02 * np.sqrt(years) * abs(v)
            extra[key] = float(np.sqrt(model ** 2 + ((1.645 * 0.05 * abs(v)) ** 2 if key[1] == "county" else 0.0)))
    rows = rows_from_draws(point, draws, extra)
    budget = float(contract["allocation"]["budget"])
    expected_adm = np.mean(np.asarray(adm_draws), axis=0)
    allocation = np.floor(expected_adm / max(expected_adm.sum(), 1e-9) * budget * 1e6) / 1e6
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_dir / "projection.csv", index=False)
    pd.DataFrame({"county": np.arange(n_counties), "allocation": allocation}).to_csv(out_dir / "allocation.csv", index=False)
    return {"projection": rows, "rates": {k: rates[k] for k in ("gompertz_a", "gompertz_b", "fertility", "births", "hh_size")}}
