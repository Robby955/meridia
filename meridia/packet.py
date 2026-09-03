"""Packet builder: one world, split into what an agent receives and what stays sealed.

A packet is a directory. ``participant/`` holds a household survey at
two snapshots carrying the health anchor, the four observed sources at two snapshots, a
five-year aggregate experience file, the county-to-state map with each county's land
area, and the contract that names the estimands, levels, snapshot ticks, projection
horizon, obligation, public reserve rule, mechanism families, and
covariate definitions. ``retained/`` holds the exact truth tables
at the revised snapshot and at the horizon and the source package's
crosswalks and mechanisms. A development packet copies the truth tables into
``participant/truth/`` so methods can be tuned on an open world; a hidden packet never
does. The manifest hashes every file and records which side it is on, and the builder
refuses to write a participant file that carries a truth column.

Everything is a deterministic function of the seed and the parameters.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .actuarial import (ACTUARIAL_AGE_BAND_LABELS, BROAD_AGE_BAND_LABELS,
                        RATE_ESTIMANDS, V4_SUBMISSION_COLUMNS, ActuarialThresholds,
                        ObligationContract, actuarial_pass, eligibility_floor,
                        regions_from_admin, reserve_total)
from .admin import build_admin
from .businesses import build_businesses
from .character import draw_world_character
from .demography import (ANNUAL_SHOCK_RATE, SHOCK_FAMILY, SHOCK_LOADING_BAND,
                         draw_world_shocks)
from .dwellings import build_dwellings
from .events import build_event_history, replay_event_history
from .hospitals import build_hospitals
from .hydrology import fill_depressions, flow_accumulation, flow_directions
from .identities import SEQUENCE_MASK, build_initial_identity_map
from .mechanisms import (QUALIFYING_DIAGNOSIS_GROUPS, build_world_mechanisms,
                         contract_block)
from .microdata import build_microdata
from .population import build_population, resource_outposts
from .projection import (DEMAND_ESTIMAND, continuation_liabilities,
                         person_table_from_state, project_truth_from_history,
                         rate_truth_from_history)
from .events import EVENT_TYPES
from .release import (AGE_BAND_LABELS, ESTIMANDS, LEVELS, SEX_LABELS,
                      compute_detailed_table_truth, compute_truth)
from .sources import (BENCHMARK_BAND_DEFINITION, BENCHMARK_BAND_LEVEL, BENCHMARK_BIAS,
                      BENCHMARK_ITEMS, BENCHMARK_ROUNDING, BENCHMARK_SUBGROUP_ITEM,
                      N_BENCHMARK_BANDS, SOURCE_REGIMES, benchmark_bands,
                      benchmark_values, build_observed_sources, draw_benchmark_bias,
                      draw_source_params, participant_source_snapshots)
from .survey import (N_SURVEY_OUTSIDE_AXES, SURVEY_BANDS, SURVEY_ENVELOPE,
                     SurveyParams, draw_survey, draw_survey_instrument)
from .terrain import generate_elevation

FORBIDDEN_COLUMN_PREFIXES = ("truth_", "mechanism", "crosswalk")


@dataclass(frozen=True)
class PacketParams:
    grid: tuple[int, int] = (288, 384)
    n_settlements: int = 24
    n_states: int = 6
    observed_months: int = 72        # ledger months before the revised snapshot
    preliminary_lag: int = 6         # revised minus preliminary, in months
    horizon_months: int = 60         # projection horizon after the revised snapshot
    disclosure_threshold: int = 10   # protected cell: 0 < true count < threshold
    budget_fraction: float = 0.9     # of persons 65+ in the revised population source
    max_shocks: int = 2
    total: int | None = None         # None draws the national total from the seed
    regime: str = "development"      # source mechanism regime: development or hidden
    design_cell: int | None = None   # row of the committed development design
    experience_years: int = 5        # years in the historical experience file
    experience_lag_months: int = 12  # publication lag of that file behind the snapshot
    ensemble_members: int = 2048     # committed continuations behind the tail truth
    reserve_weight_spread: float = 4.0   # highest regional shortfall weight over lowest
    # Provisional execution value. Qualification must replace and record it before freeze.
    reserve_rate_per_person_year: float = 4_600.0
    shock_annual_rate: float = ANNUAL_SHOCK_RATE   # published shock years per year


# Months of ledger the committed world runs before the published experience file's first
# year begins. The ledger starts from a drawn population whose frailty and age
# composition are not yet those of the process that will run it, and the first years of
# any such ledger carry a settling term: within a band, the frail die first and the band
# refills from below, so the death rate falls for a reason that has nothing to do with
# the world's mortality trend. Measured on twelve small worlds, the trend estimator's
# bias over a file at ledger months 0 to 60 was +0.084 a year against an axis whose whole
# published band is 0.058 wide, and its rank correlation with the realized intensity was
# 0.21. Over months 48 to 108 of the same worlds the bias is -0.013 and the correlation
# is 0.50, with the remaining error concentrated in the four worlds whose shock schedule
# put an epidemic year inside the window, which is a published family and visible in the
# file's own national series.
EXPERIENCE_BURN_IN_MONTHS = 48

# The committed version-four world: one size for the development set, the qualification
# worlds and the graded ones, so a bar frozen on one is read on the same object. The size
# is set by what a 2,048-member continuation ensemble costs, which is 310 seconds across
# fourteen processes here and would be hours at the version-three default grid.
#
# ``observed_months`` is the burn-in plus the five published years plus the twelve-month
# publication lag. The extra months cost 2.3 seconds of ledger per world at this size and
# nothing in the ensemble, which pays only for the horizon.
GRADING_WORLD = PacketParams(grid=(96, 128), n_settlements=8, n_states=6, total=60_000,
                             observed_months=EXPERIENCE_BURN_IN_MONTHS + 60 + 12,
                             preliminary_lag=6, horizon_months=60,
                             experience_years=5, experience_lag_months=12,
                             ensemble_members=2048)


def build_world(seed: int, params: PacketParams = PacketParams()) -> dict:
    """Every layer of one world, from terrain to observed sources."""
    if params.regime not in SOURCE_REGIMES:
        raise ValueError(f"unknown source regime {params.regime!r}")
    if params.regime == "hidden" and params.design_cell is not None:
        raise ValueError("the hidden world does not take a development design cell")
    if params.observed_months < 12 * params.experience_years + params.experience_lag_months:
        raise ValueError("the ledger is shorter than the historical experience file")
    height, width = params.grid
    character = draw_world_character(seed)
    source_params = draw_source_params(seed, params.regime, character["draw"]["payroll_level"])
    world = generate_elevation(seed, height, width)
    outlets = ~world["land"]
    outlets[0, :] = outlets[-1, :] = outlets[:, 0] = outlets[:, -1] = True
    filled = fill_depressions(world["elevation"], world["sea_level"])
    direction = flow_directions(filled, outlets)
    accumulation = flow_accumulation(direction, outlets)
    people = build_population(world, accumulation, params.total, params.n_settlements,
                              params=character["population"], seed=seed)
    micro = build_microdata(people["population"], people["habitability"],
                            people["settlements"], seed, params=character["microdata"])
    admin = build_admin(world["land"], people["settlements"], resource_outposts(world, seed),
                        n_states=params.n_states)
    identities = build_initial_identity_map(micro, seed)
    dwellings = build_dwellings(micro, seed, identities)
    businesses = build_businesses(micro, seed, identities)
    hospitals = build_hospitals(micro, seed, identities, businesses)
    mechanisms = build_world_mechanisms(
        seed, params.regime, admin, micro, businesses, params.design_cell,
        mortality_age_slope=character["demography"].gompertz_b)
    survey_params, survey_outside = draw_survey_instrument(seed, params.regime)
    months = params.observed_months + params.horizon_months
    years = max(3, months // 12 + 1)
    shocks = draw_world_shocks(seed, years, params.max_shocks,
                               annual_rate=params.shock_annual_rate)
    history = build_event_history(micro, seed, identities, dwellings, businesses, hospitals,
                                  months=months, shocks=shocks, mechanisms=mechanisms,
                                  capture_month=params.observed_months,
                                  shock_annual_rate=params.shock_annual_rate)
    snapshot = int(history["snapshot_tick"])
    revised_tick = snapshot + params.observed_months
    preliminary_tick = revised_tick - params.preliminary_lag
    sources = build_observed_sources(history, seed, admin, hospitals,
                                     preliminary_tick=preliminary_tick,
                                     revised_tick=revised_tick, params=source_params,
                                     mechanisms=mechanisms)
    benchmark_bias = draw_benchmark_bias(seed, int(admin["n_states"]))
    return {
        "seed": seed, "params": params, "character": character, "world": world,
        "people": people, "micro": micro, "admin": admin, "hospitals": hospitals,
        "history": history, "sources": sources, "shocks": shocks,
        "source_params": source_params, "benchmark_bias": benchmark_bias,
        "mechanisms": mechanisms, "survey_params": survey_params,
        "survey_outside": survey_outside,
        "ticks": {"snapshot": snapshot, "preliminary": preliminary_tick,
                  "revised": revised_tick, "horizon": snapshot + months},
    }


def _recent_admission(history: dict, state: dict, tick: int, window: int = 12) -> np.ndarray:
    """True indicator, per living person in ledger order, of an admission in the window.

    This is the quantity the survey's health anchor reports with error.  It is read off
    the ledger, never off the health source, so the anchor stays independent of the
    inclusion rule it is there to identify.
    """
    event = history["event"]
    recent = ((event["event_type"] == EVENT_TYPES["encounter_admitted"])
              & (event["tick"] > tick - window) & (event["tick"] <= tick))
    admitted = np.zeros(len(state["person"]["truth_person_id"]), dtype=np.bool_)
    position = ((event["truth_person_id"][recent] & np.uint64(SEQUENCE_MASK))
                .astype(np.int64) - 1)
    admitted[position[(position >= 0) & (position < len(admitted))]] = True
    return admitted[np.flatnonzero(state["person"]["is_alive"])]


def _survey_at(built: dict, tick: int, vintage: int) -> dict:
    state = replay_event_history(built["history"], tick)
    person, household_cell = person_table_from_state(state, tick)
    height, width = built["params"].grid
    population = np.bincount(person["cell"], minlength=height * width).reshape(height, width)
    micro = {"person": person, "household_cell": household_cell,
             "urbanity": built["micro"]["urbanity"], "n_households": len(household_cell)}
    survey = draw_survey(micro, population, built["seed"], vintage=vintage,
                         params=built["survey_params"],
                         recent_admission=_recent_admission(built["history"], state, tick))
    # Participant view: the survey carries the county, not the grid cell, and a
    # survey-local household number rather than the world's household index.
    public = dict(survey["survey"])
    county_flat = built["admin"]["county"].flatten()
    cell = public.pop("cell")
    public["county"] = county_flat[cell].astype(np.int64)
    _, public["psu"] = np.unique(cell, return_inverse=True)        # masked sampling unit
    public["psu"] = public["psu"].astype(np.int64)
    # Design documentation: households sampled in each unit (respondents are visible).
    sampled_cells = household_cell[survey["truth"]["sampled_households"]]
    sampled_per_cell = np.bincount(sampled_cells, minlength=len(county_flat))
    public["psu_sampled_households"] = sampled_per_cell[cell].astype(np.int64)
    _, public["household"] = np.unique(public["household"], return_inverse=True)
    public["household"] = public["household"].astype(np.int64)
    survey["survey"] = public
    return survey


EXPERIENCE_COLUMNS = ("year", "age_band", "sex", "state", "exposure", "deaths",
                      "qualifying_events", "net_migration")


def _experience_history(built: dict, admin: dict, obligation: ObligationContract,
                        years: int, lag_months: int) -> dict:
    """Five years of aggregate demographic experience, by year, band, sex, and state.

    Two snapshots do not identify a five-year mortality trend, so the packet ships the
    trend's own evidence: person-years of exposure, deaths, first qualifying health
    events, and net internal migration, all aggregate.  Exposure comes from the same
    person-month reading pass the actuarial truth uses, so a rate and its denominator
    can never disagree.

    The series stops ``lag_months`` before the revised snapshot, the way published
    demographic experience always lags collection.  Without that lag the most recent
    year's exposure would be a near-exact contemporaneous population count by state, and
    the scored state-level counts would come free with the anchor.

    It also starts well after the ledger does.  The committed world runs
    ``EXPERIENCE_BURN_IN_MONTHS`` before the file's first year, because a ledger's opening
    years carry a settling term in the death rate that a trend estimator reads as
    improvement.  The file is the only anchor the mortality trend has, so the window it
    covers is the difference between an axis that can be estimated and one that cannot.
    """
    history, ticks = built["history"], built["ticks"]
    county_state = np.asarray(admin["county_state"], dtype=np.int64)
    county_flat = np.asarray(admin["county"], dtype=np.int64).reshape(-1)
    n_states = int(admin["n_states"])
    region = regions_from_admin(admin)
    last = ticks["revised"] - int(lag_months)
    boundary = [last - 12 * (years - y) for y in range(years + 1)]
    states = [replay_event_history(history, tick) for tick in boundary]

    rows = {name: [] for name in EXPERIENCE_COLUMNS}
    for y in range(1, years + 1):
        start, begin, end = boundary[y - 1], states[y - 1], states[y]
        result = actuarial_pass(begin, history["event"], admin, start, 12,
                                obligation, region)
        shape = (n_states,) + result["exposure_person_months"].shape[1:]
        by_state = {}
        for name in ("exposure_person_months", "deaths", "qualifying_events"):
            cube = np.zeros(shape, dtype=np.float64)
            np.add.at(cube, county_state, result[name])
            by_state[name] = cube
        migration = _net_internal_migration(begin, end, county_flat, county_state,
                                            boundary[y], shape)
        for b, band in enumerate(ACTUARIAL_AGE_BAND_LABELS):
            for x, sex in enumerate(SEX_LABELS):
                for unit in range(n_states):
                    rows["year"].append(y)
                    rows["age_band"].append(band)
                    rows["sex"].append(sex)
                    rows["state"].append(unit)
                    rows["exposure"].append(by_state["exposure_person_months"][unit, x, b] / 12.0)
                    rows["deaths"].append(by_state["deaths"][unit, x, b])
                    rows["qualifying_events"].append(by_state["qualifying_events"][unit, x, b])
                    rows["net_migration"].append(migration[unit, x, b])
    return {"year": np.asarray(rows["year"], dtype=np.int64),
            "age_band": np.asarray(rows["age_band"]),
            "sex": np.asarray(rows["sex"]),
            "state": np.asarray(rows["state"], dtype=np.int64),
            "exposure": np.asarray(rows["exposure"], dtype=np.float64),
            "deaths": np.asarray(rows["deaths"], dtype=np.int64),
            "qualifying_events": np.asarray(rows["qualifying_events"], dtype=np.int64),
            "net_migration": np.asarray(rows["net_migration"], dtype=np.int64)}


def _net_internal_migration(begin: dict, end: dict, county_flat: np.ndarray,
                            county_state: np.ndarray, end_tick: int,
                            shape: tuple) -> np.ndarray:
    """Arrivals minus departures per state, band, and sex, over persons alive at both ends."""
    n = len(begin["person"]["truth_person_id"])
    alive = begin["person"]["is_alive"] & end["person"]["is_alive"][:n]
    from_state = county_state[np.maximum(county_flat[begin["person"]["cell"][:n]], 0)]
    to_state = county_state[np.maximum(county_flat[end["person"]["cell"][:n]], 0)]
    moved = np.flatnonzero(alive & (from_state != to_state))
    net = np.zeros(shape, dtype=np.int64)
    if not len(moved):
        return net
    age = np.maximum(0, (end_tick - end["person"]["birth_tick"][:n][moved]) // 12)
    band = np.clip(np.searchsorted(
        np.asarray([18, 45, 65, 75, 85]), age, side="right"), 0, len(ACTUARIAL_AGE_BAND_LABELS) - 1)
    sex = end["person"]["sex"][:n][moved].astype(np.int64)
    np.add.at(net, (to_state[moved], sex, band), 1)
    np.add.at(net, (from_state[moved], sex, band), -1)
    return net



RESERVE_WEIGHT_RANGE = (0.5, 2.0)


def reserve_weights(population: dict, county_state: np.ndarray, tick: int,
                    n_regions: int, spread: float) -> np.ndarray:
    """Published shortfall weights w_r, one per region, from a participant file.

    An uncovered obligation costs more where the very old are concentrated, so the
    weights ride a public ladder over the regions ranked by their share of persons 85 and
    over in the revised population source. Ranking a share rather than a count keeps the
    ladder from restating region size, which is what would make a size-proportional
    reserve optimal by construction. The register the share is read from is a participant
    file, so a method reproduces the ladder exactly; the numbers are published in the
    contract regardless, and nothing sealed enters them.
    """
    state = np.asarray(county_state, dtype=np.int64)[
        np.asarray(population["county"], dtype=np.int64)]
    age = (int(tick) - np.asarray(population["birth_tick"], dtype=np.int64)) // 12
    total = np.bincount(state, minlength=n_regions).astype(np.float64)
    oldest = np.bincount(state[age >= 85], minlength=n_regions).astype(np.float64)
    share = np.divide(oldest, total, out=np.zeros(n_regions), where=total > 0)
    rank = np.argsort(np.argsort(share, kind="stable"), kind="stable")
    low, high = RESERVE_WEIGHT_RANGE
    if n_regions < 2:
        return np.ones(n_regions)
    centre = (low * high) ** 0.5
    half = float(spread) ** 0.5
    ladder = np.exp(np.linspace(np.log(centre / half), np.log(centre * half), n_regions))
    return np.round(ladder[rank], 4)


def reserve_baseline_share(population: dict, county_state: np.ndarray, tick: int,
                           n_regions: int, min_age: int) -> np.ndarray:
    """Published regional size behind the frozen practical baseline A_B.

    The share of persons at or above the obligation's eligibility age, by region, in the
    revised population source. It is a participant file, so a submission reproduces the
    baseline exactly and knows what it has to beat. Holding a reserve in proportion to how
    many people it covers is what a practitioner does with no regional tail model.
    """
    state = np.asarray(county_state, dtype=np.int64)[
        np.asarray(population["county"], dtype=np.int64)]
    age = (int(tick) - np.asarray(population["birth_tick"], dtype=np.int64)) // 12
    eligible = np.bincount(state[age >= int(min_age)], minlength=n_regions).astype(np.float64)
    if eligible.sum() <= 0:
        return np.full(n_regions, 1.0 / max(n_regions, 1))
    return np.round(eligible / eligible.sum(), 6)


def _rate_truth_rows(truth: dict) -> dict:
    """Long rows of the retained exposure and rate truth, in a committed order."""
    keys = sorted(truth)
    return {"estimand": np.asarray([k[0] for k in keys]),
            "level": np.asarray([k[1] for k in keys]),
            "unit": np.asarray([k[2] for k in keys], dtype=np.int64),
            "sex": np.asarray([k[3] for k in keys]),
            "age_band": np.asarray([k[4] for k in keys]),
            "value": np.asarray([truth[k] for k in keys], dtype=np.float64)}


def _truth_at(built: dict, tick: int) -> tuple[dict, np.ndarray]:
    state = replay_event_history(built["history"], tick)
    person, household_cell = person_table_from_state(state, tick)
    return (compute_truth(person, household_cell, built["admin"]),
            compute_detailed_table_truth(person, built["admin"]))


def _write_table(path: Path, table: dict, forbid_truth: bool) -> None:
    import pandas as pd
    columns = {name: np.asarray(values) for name, values in table.items()}
    if forbid_truth:
        for name in columns:
            if name.startswith(FORBIDDEN_COLUMN_PREFIXES):
                raise ValueError(f"participant file {path.name} would carry column {name!r}")
    pd.DataFrame(columns).to_csv(path, index=False, float_format="%.6f")


def _truth_rows(truth: dict) -> dict:
    keys = sorted(truth)
    return {"estimand": np.asarray([k[0] for k in keys]),
            "level": np.asarray([k[1] for k in keys]),
            "unit": np.asarray([k[2] for k in keys], dtype=np.int64),
            "value": np.asarray([truth[k] for k in keys], dtype=np.float64)}


def _detailed_rows(table: np.ndarray) -> dict:
    counties, bands, sexes = np.indices(table.shape)
    return {"county": counties.ravel().astype(np.int64),
            "age_band": np.asarray(AGE_BAND_LABELS)[bands.ravel()],
            "sex": np.asarray(SEX_LABELS)[sexes.ravel()],
            "count": table.ravel().astype(np.int64)}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


ENSEMBLE_CACHE_SCHEMA = "meridia.ensemble.cache.v1"


def baseline_ledger_digest(history: dict, obligation: ObligationContract,
                           horizon_months: int, region_of_county: np.ndarray) -> str:
    """The key the continuation ensemble is cached under.

    Every member is a deterministic function of four things: the branch state the ledger
    kept at the revised snapshot, the shock law it redraws its own future from, the
    horizon it is priced over, and the obligation that prices it. This digest covers all
    four, so a cached ensemble is reused exactly when the world that produced it is
    unchanged and is rebuilt as soon as anything upstream of it moves. A verifier or a
    bar is downstream of all of it and does not enter, which is the point: refreezing a
    bar on twenty-one worlds no longer pays for their futures a second time.
    """
    digest = hashlib.sha256(ENSEMBLE_CACHE_SCHEMA.encode())
    branch = history["branch"]
    digest.update(json.dumps({
        "seed": int(branch["seed"]), "month": int(branch["month"]),
        "tick": int(branch["tick"]), "n_events": int(branch["n_events"]),
        "order": int(branch["order"]),
        "annual_shock_rate": float(branch["annual_shock_rate"]),
        "generator": int(history["generator_version"]),
        "schema": int(history["event_schema_version"]),
        "horizon_months": int(horizon_months),
        "obligation": obligation.as_public(),
        "shocks": history["shock_schedule"],
        # The mechanism record covers the coefficients and the regional shock loadings a
        # member runs under. Most of them are already in the branch state, since they
        # produced it, but a loading only shows there once a shock year has been run, and
        # a member's own future is where the rest of them go.
        "mechanisms": history["mechanism_record"],
    }, sort_keys=True, default=str).encode())
    for table in sorted(branch["state"]):
        for name, values in sorted(branch["state"][table].items()):
            digest.update(table.encode())
            digest.update(name.encode())
            digest.update(np.ascontiguousarray(values).tobytes())
    digest.update(np.ascontiguousarray(branch["household_last_move_tick"]).tobytes())
    digest.update(np.ascontiguousarray(region_of_county).tobytes())
    return digest.hexdigest()


def _cached_liability(cache_dir: Path | None, key: str, members: int) -> np.ndarray | None:
    if cache_dir is None:
        return None
    path = Path(cache_dir) / f"{key}.npz"
    if not path.is_file():
        return None
    stored = np.load(path)["liability"]
    if stored.shape[0] < members:
        return None
    return np.ascontiguousarray(stored[:members])


def _store_liability(cache_dir: Path | None, key: str, liability: np.ndarray) -> None:
    if cache_dir is None:
        return
    directory = Path(cache_dir)
    directory.mkdir(parents=True, exist_ok=True)
    # numpy appends the suffix itself, so the scratch name has to carry it: the write is
    # done under a name nothing reads and moved into place, which keeps a build that dies
    # part way from leaving a half-written ensemble for the next one to trust.
    scratch = directory / f"partial-{os.getpid()}-{key}.npz"
    np.savez_compressed(scratch, liability=liability)
    scratch.replace(directory / f"{key}.npz")


def build_packet(seed: int, out_dir: Path, params: PacketParams = PacketParams(),
                 development: bool = False, workers: int = 1,
                 cache_dir: Path | None = None) -> dict:
    """Write one packet and return its manifest.

    ``workers`` divides the continuation ensemble between processes and changes nothing
    else: every member is a deterministic function of the seed and its own index.

    ``cache_dir`` holds continuation ensembles keyed on the digest of the baseline ledger
    that produced them. The ensemble is the whole cost of a packet at the committed size,
    and it depends on nothing downstream of the ledger, so a rebuild that changes only
    what a verifier or a bar reads takes the futures back off the shelf. A cached
    ensemble with more members than the packet asks for is used from the front, since a
    member is a function of its own index.
    """
    out_dir = Path(out_dir)
    if out_dir.exists():
        raise FileExistsError(f"packet directory already exists: {out_dir}")
    if development and params.regime != "development":
        raise ValueError("a development packet must use the development source regime")
    built = build_world(seed, params)
    admin, ticks = built["admin"], built["ticks"]
    participant = out_dir / "participant"
    retained = out_dir / "retained"
    (participant / "sources").mkdir(parents=True)
    retained.mkdir(parents=True)

    # Participant side: surveys, sources, benchmark totals, geography, contract.
    for vintage, label in enumerate(("preliminary", "revised")):
        survey = _survey_at(built, ticks[label], vintage)
        _write_table(participant / f"survey_{label}.csv", survey["survey"], forbid_truth=True)
    snapshots = participant_source_snapshots(built["sources"])
    for label, snapshot in snapshots.items():
        for source, table in snapshot.items():
            if source == "snapshot_tick":
                continue
            _write_table(participant / "sources" / f"{source}_{label}.csv", table,
                         forbid_truth=True)
    county_band = benchmark_bands(built["mechanisms"].county.econ)
    for label in ("preliminary", "revised"):
        exact, _ = _truth_at(built, ticks[label])
        _write_table(participant / "sources" / f"benchmark_{label}.csv",
                     benchmark_values(exact, built["benchmark_bias"], admin["n_states"],
                                      county_band),
                     forbid_truth=True)
    county_flat = np.asarray(admin["county"], dtype=np.int64).reshape(-1)
    _write_table(participant / "geography.csv",
                 {"county": np.arange(admin["n_counties"], dtype=np.int64),
                  "state": admin["county_state"].astype(np.int64),
                  "land_cells": np.bincount(county_flat[county_flat >= 0],
                                            minlength=int(admin["n_counties"])).astype(np.int64),
                  "economic_band": county_band.astype(np.int64)},
                 forbid_truth=True)
    obligation = ObligationContract(
        horizon_months=params.horizon_months,
        qualifying_diagnosis_groups=QUALIFYING_DIAGNOSIS_GROUPS)
    experience_path = participant / "experience_history.csv"
    _write_table(experience_path,
                 _experience_history(built, admin, obligation, params.experience_years,
                                     params.experience_lag_months), forbid_truth=True)
    # Read the file back after six-decimal serialization. The public total is therefore a
    # deterministic function of the exact values a participant receives, not of a
    # higher-precision array retained only while the packet is being built.
    import pandas as pd
    public_experience = pd.read_csv(experience_path)
    latest_experience_year = int(public_experience["year"].max())
    reserve_exposure = float(public_experience.loc[
        public_experience["year"] == latest_experience_year, "exposure"].sum())
    population_revised = snapshots["revised"]["population"]
    age_years = (ticks["revised"] - population_revised["birth_tick"]) // 12
    budget = int(round(params.budget_fraction * int((age_years >= 65).sum())))

    # The tail truth: M committed continuations from the branch state the ledger kept at
    # the revised snapshot, priced into regional present values. Member zero is the
    # ledger's own future, so the realized path and the horizon truth tables are one
    # world. Nothing about the ensemble crosses to the participant side except the single
    # aggregate the protocol publishes on purpose, the reserve total.
    thresholds = ActuarialThresholds()
    region_of_county = regions_from_admin(admin)
    n_regions = int(admin["n_states"])
    weights = reserve_weights(population_revised, admin["county_state"], ticks["revised"],
                              n_regions, params.reserve_weight_spread)
    baseline_share = reserve_baseline_share(population_revised, admin["county_state"],
                                            ticks["revised"], n_regions,
                                            obligation.eligibility_min_age)
    cache_key = baseline_ledger_digest(built["history"], obligation,
                                       params.horizon_months, region_of_county)
    liability = _cached_liability(cache_dir, cache_key, params.ensemble_members)
    if liability is None:
        liability = continuation_liabilities(built["history"], admin, ticks["revised"],
                                             params.horizon_months, obligation,
                                             params.ensemble_members, region_of_county,
                                             workers=workers)
        _store_liability(cache_dir, cache_key, liability)
    rounding_unit = thresholds.reserve_rounding_unit
    total = reserve_total(reserve_exposure, params.reserve_rate_per_person_year,
                          rounding_unit)
    reserve = {"obligation": obligation.as_public(),
               "total": total,
               "total_rule": {
                   "file": "experience_history.csv",
                   "year": "maximum published year",
                   "year_column": "year",
                   "selected_year": latest_experience_year,
                   "exposure_column": "exposure",
                   "aggregation": "sum exposure over every row in the selected year",
                   "exposure_person_years": reserve_exposure,
                   "rate_per_person_year": float(params.reserve_rate_per_person_year),
                   "rounding": "up",
                   "rounding_unit": rounding_unit,
               },
               "regions": "state",
               "weights": [float(w) for w in weights],
               "baseline_share": [float(v) for v in baseline_share],
               "baseline_rule": "A_B splits the total in proportion to each region's share "
                                "of persons at or above the eligibility age in the revised "
                                "population source",
               "rounding_unit": rounding_unit,
               "weight_rule": "public ladder over regions ranked by the share of persons "
                              "85 and over in the revised population source",
               "members": int(params.ensemble_members)}
    contract = {
        "schema": "meridia.packet.v4",
        "estimands": [asdict(e) for e in ESTIMANDS],
        "levels": list(LEVELS),
        "n_states": int(admin["n_states"]),
        "n_counties": int(admin["n_counties"]),
        "interval_level": 0.90,
        "ticks": {"preliminary": ticks["preliminary"], "revised": ticks["revised"],
                  "horizon": ticks["horizon"]},
        "months_per_tick": 1,
        "allocation": {"demand": DEMAND_ESTIMAND, "level": "county", "budget": budget},
        "submission": {
            "files": {name: list(columns)
                      for name, columns in V4_SUBMISSION_COLUMNS.items()},
            "additional_entries": "forbidden",
        },
        "mechanisms": contract_block(),
        "obligation": obligation.as_public(),
        "reserve": reserve,
        "actuarial_age_bands": list(ACTUARIAL_AGE_BAND_LABELS),
        "rate_eligibility": {
            "truth_quantity": "retained person-years exposure",
            "estimands": list(RATE_ESTIMANDS),
            "bands": list(BROAD_AGE_BAND_LABELS),
            "exposure_level": "county",
            "rate_level": "state",
            "floor_person_years_by_band": {
                band: eligibility_floor(thresholds, band)
                for band in BROAD_AGE_BAND_LABELS
            },
            "reduction": "one empirical 95th percentile over all eligible cells",
        },
        "shock_family": {
            "annual_rate": params.shock_annual_rate,
            "kinds": {kind: {field: list(bounds) for field, bounds in fields.items()}
                      for kind, fields in SHOCK_FAMILY.items()},
            "note": "one draw per year, independent across years; the fields of one kind "
                    "move together on a single draw; the five-year experience file "
                    "carries the realized years",
            "regional_loading_band": list(SHOCK_LOADING_BAND),
            "regional_loading": "a shock year is national, and its mortality and "
                                "admission multipliers m land in region r as "
                                "1 + L_r * (m - 1). One loading L_r per region is drawn "
                                "once per world from the band above and held for every "
                                "year and every continuation, so regional liabilities "
                                "are correlated through the loadings rather than moving "
                                "as one. Regions are the states. The realized vector is "
                                "not published; the experience file carries it, because "
                                "a shock year shows there as a state-specific jump in "
                                "deaths and in first qualifying events. Fertility and "
                                "internal migration stay national",
        },
        "benchmark": {
            "file": "benchmark_revised.csv",
            "items": list(BENCHMARK_ITEMS),
            "levels": ["nation", "state", BENCHMARK_BAND_LEVEL],
            "reference_tick": ticks["revised"],
            "rounding": BENCHMARK_ROUNDING,
            "subgroup_item": BENCHMARK_SUBGROUP_ITEM,
            "subgroup_level": BENCHMARK_BAND_LEVEL,
            "n_economic_bands": N_BENCHMARK_BANDS,
            "subgroup_definition": BENCHMARK_BAND_DEFINITION,
            "bias_ranges": {
                name: list(bounds) for name, bounds in BENCHMARK_BIAS.items()
            },
            "bias_family": "each value is the exact count times exp(b); at nation level "
                           "b is uniform in magnitude with a fair-coin sign, at state "
                           "and economic-band level b is normal with one world-wide "
                           "standard deviation. The same b holds in both vintages",
        },
        "health_anchor": {
            "file": "survey_revised.csv",
            "item": "recent_hospitalization",
            "window_months": 12,
            "sensitivity": SurveyParams().anchor_sensitivity,
            "specificity": SurveyParams().anchor_specificity,
        },
        "survey_family": {
            "unit_response": "logit p_respond = a_0 + a_age * (head age - 45)"
                             " + a_income * (log income - median) + a_urban * urbanity",
            "item_missing": "per variable, a base rate with an extra logit in the"
                            " reported value itself for money",
            "measurement": "multiplicative lognormal money error; ages heaped to a"
                           " multiple of five with a fixed probability",
            "bands": {name: list(bounds) for name, bounds in sorted(SURVEY_BANDS.items())},
            "envelope": {name: list(bounds)
                         for name, bounds in sorted(SURVEY_ENVELOPE.items())},
            "n_outside_axes": N_SURVEY_OUTSIDE_AXES,
            "note": "one continuous draw per world. A world a method may tune on draws"
                    " every axis inside its band; an evaluation world draws"
                    " n_outside_axes of them between that band and the envelope edge."
                    " Which axes those are, and the realized values, are not published",
        },
        "experience_history": {
            "file": "experience_history.csv",
            "columns": list(EXPERIENCE_COLUMNS),
            "years": params.experience_years,
            "level": "state",
            "age_bands": list(ACTUARIAL_AGE_BAND_LABELS),
            "exposure_unit": "person-years",
            "publication_lag_months": params.experience_lag_months,
            "last_year_ends_at_tick": ticks["revised"] - params.experience_lag_months,
            "first_year_starts_at_tick": (ticks["revised"] - params.experience_lag_months
                                          - 12 * params.experience_years),
        },
        "development": development,
    }
    (participant / "contract.json").write_text(json.dumps(contract, indent=1, sort_keys=True) + "\n")

    # Retained side: exact truth now and at the horizon, the continuation ensemble, the
    # source package's sealed evidence, and the world's character draw.
    truth_revised, detailed = _truth_at(built, ticks["revised"])
    future = project_truth_from_history(built["history"], admin, ticks["horizon"])
    _write_table(retained / "truth_revised.csv", _truth_rows(truth_revised), forbid_truth=False)
    _write_table(retained / "truth_horizon.csv", _truth_rows(future["truth"]), forbid_truth=False)
    if development:
        _write_table(
            retained / "detailed_revised.csv",
            _detailed_rows(detailed),
            forbid_truth=False,
        )
    _write_table(retained / "rate_truth_horizon.csv",
                 _rate_truth_rows(rate_truth_from_history(
                     built["history"], admin, ticks["revised"], params.horizon_months,
                     obligation)), forbid_truth=False)
    np.savez_compressed(retained / "continuation_liabilities.npz", liability=liability,
                        realized_member=np.int64(0), weights=weights)
    sealed = {k: v for k, v in built["sources"].items() if k != "public_snapshots"}
    np.savez_compressed(retained / "source_evidence.npz", **_flatten(sealed))
    (retained / "world.json").write_text(json.dumps({
        "seed": seed, "character": built["character"]["draw"], "shocks": built["shocks"],
        "ticks": ticks, "params": asdict(params), "regime": params.regime,
        "source_params": asdict(built["source_params"]),
        "mechanisms": built["mechanisms"].record(),
        "survey_params": asdict(built["survey_params"]),
        "survey_outside": list(built["survey_outside"]),
        "benchmark_bias": {k: np.asarray(v).tolist() for k, v in built["benchmark_bias"].items()},
    }, indent=1, sort_keys=True, default=str) + "\n")
    if development:
        (participant / "truth").mkdir()
        for name in ("truth_revised.csv", "truth_horizon.csv", "detailed_revised.csv"):
            (participant / "truth" / name).write_bytes((retained / name).read_bytes())

    manifest = {"schema": "meridia.packet.manifest.v0", "development": development,
                "participant": {}, "retained": {}}
    for side in ("participant", "retained"):
        for path in sorted((out_dir / side).rglob("*")):
            if path.is_file():
                manifest[side][str(path.relative_to(out_dir / side))] = {
                    "sha256": _sha256(path), "bytes": path.stat().st_size}
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")
    return manifest


def _flatten(tree: dict, prefix: str = "") -> dict:
    flat = {}
    for key, value in tree.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            flat.update(_flatten(value, name + "/"))
        elif isinstance(value, (np.ndarray, np.generic, int, float, bool)):
            flat[name] = np.asarray(value)
        elif isinstance(value, (list, tuple)):
            flat[name] = np.asarray(value)
    return flat


def participant_columns(out_dir: Path) -> dict[str, list[str]]:
    """Header row of every participant CSV, for leakage checks."""
    columns = {}
    for path in sorted((Path(out_dir) / "participant").rglob("*.csv")):
        with open(path) as handle:
            columns[str(path.relative_to(Path(out_dir) / "participant"))] = \
                handle.readline().strip().split(",")
    return columns
