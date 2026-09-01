"""Forecast strong method B: Bayesian rates, posterior projection.

Sources: Poisson-gamma and grid posteriors for demographic rates after Gelman et al.
(2013); cohort-component projection after Preston, Heuveline, and Guillot (2001).
See docs/INDEPENDENCE.md.

Where method A fits rates by least squares and bootstraps the counts, this line puts a
likelihood on the ledger's counted events and projects from the posterior:

- mortality: deaths by single-year age are Poisson with mean exposure times a
  Gompertz-Makeham hazard; the (a, b) posterior is evaluated on a grid whose prior is
  uniform over the public range and sampled exactly;
- fertility: births are Poisson with mean women-years times the rate, gamma prior
  centred on the public range;
- admissions: Poisson-gamma per county;
- shocks: a Beta posterior on the annual shock hazard from how many of the observed
  history years look shocked (a death or birth count outside its Poisson band).

Every projection replicate is one joint posterior draw pushed through the same
single-year cohort engine; the point is the posterior median; intervals are posterior
quantiles plus the horizon allowance for structural drift.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..character import CHARACTER_RANGES
from ..demography import DemographyParams, mortality_probability
from .common import COUNT_ITEMS, INCOME_ITEMS, rows_from_draws
from . import forecast_cohort as FA

MAX_AGE = FA.MAX_AGE


@dataclass(frozen=True)
class MethodParams:
    draws: int = 200
    seed: int = 20260906


def posterior_rates(data: dict, tick: int, rng: np.random.Generator, n_draws: int) -> dict:
    events = data["events"]
    base = FA.estimate_rates(data, tick)
    ages = np.arange(MAX_AGE + 1)
    deaths, exposure = base["deaths"], base["exposure"]
    fit = (exposure > 0) & (ages >= 1)
    lo_a, hi_a = CHARACTER_RANGES["gompertz_a"]
    a_grid = np.linspace(lo_a * 0.6, hi_a * 1.4, 45)
    b_grid = np.linspace(0.08, 0.13, 26)
    loglik = np.zeros((len(a_grid), len(b_grid)))
    for i, a in enumerate(a_grid):
        for j, b in enumerate(b_grid):
            q = mortality_probability(ages, DemographyParams(gompertz_a=a, gompertz_b=b))
            mu = np.maximum(exposure[fit] * q[fit], 1e-12)
            loglik[i, j] = float((deaths[fit] * np.log(mu) - mu).sum())
    w = np.exp(loglik - loglik.max()).ravel()
    w /= w.sum()
    picks = rng.choice(len(w), size=n_draws, p=w)
    a_draws, b_draws = a_grid[picks // len(b_grid)], b_grid[picks % len(b_grid)]
    lo_f, hi_f = CHARACTER_RANGES["fertility_rate"]
    prior_mean, prior_strength = 0.5 * (lo_f + hi_f), 200.0     # gamma prior, weak
    fert_draws = rng.gamma(prior_strength * prior_mean + base["births"],
                           1.0 / (prior_strength + max(base["women"], 1.0)), size=n_draws)
    adm_draws = rng.gamma(base["admissions"][None, :] + 0.5, 1.0, size=(n_draws, len(base["admissions"])))
    # Shock hazard: years of history whose national death or birth count sits outside
    # its Poisson band count as shocked; Beta(1, 6) prior on the annual hazard.
    yearly = events[events["event"].isin(["person_death", "person_birth"])]
    year_index = (yearly["tick"].to_numpy(dtype=np.int64) - 1) // 12
    shocked_years, total_years = 0, 0
    for kind in ("person_death", "person_birth"):
        mask = yearly["event"].to_numpy() == kind
        counts = np.bincount(year_index[mask] - year_index.min())
        counts = counts[counts > 0]
        if len(counts) >= 2:
            median = np.median(counts)
            shocked_years += int((np.abs(counts - median) > 4 * np.sqrt(median)).sum())
            total_years = max(total_years, len(counts))
    hazard_draws = rng.beta(1.0 + shocked_years, 6.0 + max(total_years - shocked_years, 0), size=n_draws)
    return {"base": base, "gompertz_a": a_draws, "gompertz_b": b_draws, "fertility": fert_draws,
            "admissions": adm_draws, "shock_hazard": hazard_draws}


def run(packet_dir: Path, out_dir: Path, params: MethodParams = MethodParams()) -> dict:
    import pandas as pd
    data = FA.load_forecast_packet(packet_dir)
    contract, county_state = data["contract"], data["county_state"]
    n_counties = len(county_state)
    n_states = int(county_state.max()) + 1
    S, H = int(contract["ticks"]["snapshot"]), int(contract["ticks"]["horizon"])
    years = int(round((H - S) / 12.0))
    rng = np.random.default_rng(params.seed)
    post = posterior_rates(data, S, rng, params.draws)
    base = post["base"]
    cube0 = base["cube"]
    persons0 = np.maximum(cube0.sum(axis=(1, 2)), 1.0)
    persons = data["persons"]
    age = (S - persons["birth_tick"].to_numpy(dtype=np.int64)) // 12
    income = persons["income"].to_numpy(dtype=np.float64)
    county = persons["county"].to_numpy(dtype=np.int64)
    hh = persons.groupby("household_id").agg(income=("income", "sum"), county=("county", "first"))
    hh_income, hh_county = hh["income"].to_numpy(), hh["county"].to_numpy(dtype=np.int64)
    national_median = float(np.median(hh_income))
    education = persons["education"].to_numpy()

    snapshot_items = {}
    for level, units, member_c, member_h in (("county", range(n_counties), county, hh_county),
                                             ("state", range(n_states), county_state[county], county_state[hh_county]),
                                             ("nation", [0], np.zeros(len(county), int), np.zeros(len(hh_county), int))):
        for u in units:
            pm, hm = member_c == u, member_h == u
            ad = pm & (age >= 16)
            over = pm & (age >= 25) & (education >= 0)
            snapshot_items[("mean_income_adults", level, u)] = float(income[ad].mean()) if ad.any() else float("nan")
            snapshot_items[("median_household_income", level, u)] = float(np.median(hh_income[hm])) if hm.any() else float("nan")
            snapshot_items[("low_income_household_share", level, u)] = float((hh_income[hm] < 0.6 * national_median).mean()) if hm.any() else float("nan")
            snapshot_items[("tertiary_share_25_plus", level, u)] = float((education[over] >= 2).mean()) if over.any() else float("nan")
    hh_by_county = np.bincount(hh_county, minlength=n_counties).astype(np.float64)

    draws: dict[tuple, list] = {}
    adm_out = []
    ages = np.arange(MAX_AGE + 1)
    for k in range(params.draws):
        q = mortality_probability(ages, DemographyParams(gompertz_a=float(post["gompertz_a"][k]),
                                                         gompertz_b=float(post["gompertz_b"][k])))
        state = cube0.astype(np.float64).copy()
        fert = float(post["fertility"][k])
        hazard = float(post["shock_hazard"][k])
        for _ in range(years):
            shock = rng.random() < hazard
            multiplier = float(rng.uniform(1.5, 3.0)) if shock and rng.random() < 0.5 else 1.0
            f_year = fert * (float(rng.uniform(0.45, 0.75)) if shock and multiplier == 1.0 else 1.0)
            survival = 1.0 - np.clip(q * multiplier, 0.0, 1.0)
            survivors = state * survival[None, :, None]
            aged = np.zeros_like(survivors)
            aged[:, 1:, :] = survivors[:, :-1, :]
            aged[:, MAX_AGE, :] += survivors[:, MAX_AGE, :]
            births = state[:, 18:46, 1].sum(axis=1) * f_year * survival[0]
            aged[:, 0, 0] += 0.5 * births
            aged[:, 0, 1] += 0.5 * births
            state = aged
        persons_end = state.sum(axis=(1, 2))
        growth = persons_end / persons0
        values = {"persons": persons_end, "children_under_16": state[:, :16, :].sum(axis=(1, 2)),
                  "elders_65_plus": state[:, 65:, :].sum(axis=(1, 2)), "households": hh_by_county * growth}
        for e, v in values.items():
            for c in range(n_counties):
                draws.setdefault((e, "county", c), []).append(float(v[c]))
            st = np.bincount(county_state, weights=v, minlength=n_states)
            for s_ in range(n_states):
                draws.setdefault((e, "state", s_), []).append(float(st[s_]))
            draws.setdefault((e, "nation", 0), []).append(float(st.sum()))
        adm_out.append(post["admissions"][k] * growth)
    point = {key: float(np.median(v)) for key, v in draws.items()}
    for e in COUNT_ITEMS:
        county_points = np.asarray([point[(e, "county", c)] for c in range(n_counties)])
        st = np.bincount(county_state, weights=county_points, minlength=n_states)
        for s_ in range(n_states):
            point[(e, "state", s_)] = float(st[s_])
        point[(e, "nation", 0)] = float(st.sum())
    point.update(snapshot_items)
    point.update(FA.projected_tertiary_share(persons, age, county, county_state, years,
                                             float(np.median(post["gompertz_a"])), float(np.median(post["gompertz_b"]))))
    extra = {}
    for key, v in point.items():
        if not np.isfinite(v):
            continue
        e = key[0]
        if e in INCOME_ITEMS:
            extra[key] = 1.645 * 0.05 * np.sqrt(years) * (abs(v) if e != "low_income_household_share" else 0.5)
        elif e == "tertiary_share_25_plus":
            extra[key] = 0.03 * years / 5.0
        elif e in COUNT_ITEMS:
            model = 1.645 * 0.02 * np.sqrt(years) * abs(v)
            extra[key] = float(np.sqrt(model ** 2 + ((1.645 * 0.05 * abs(v)) ** 2 if key[1] == "county" else 0.0)))
    rows = rows_from_draws(point, draws, extra)
    expected_adm = np.median(np.asarray(adm_out), axis=0)
    budget = float(contract["allocation"]["budget"])
    allocation = np.floor(expected_adm / max(expected_adm.sum(), 1e-9) * budget * 1e6) / 1e6
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_dir / "projection.csv", index=False)
    pd.DataFrame({"county": np.arange(n_counties), "allocation": allocation}).to_csv(out_dir / "allocation.csv", index=False)
    return {"projection": rows, "posterior": {"gompertz_a": float(np.median(post["gompertz_a"])),
                                              "fertility": float(np.median(post["fertility"])),
                                              "shock_hazard": float(np.median(post["shock_hazard"]))}}
