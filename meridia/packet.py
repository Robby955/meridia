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

import ast
import csv
import ctypes
import errno
import fcntl
import functools
import hashlib
import json
import math
import os
import re
import shutil
import stat
import struct
import sys
import tempfile
from dataclasses import asdict, dataclass, fields, is_dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path

import numpy as np

from .actuarial import (ACTUARIAL_AGE_BAND_LABELS, BROAD_AGE_BAND_LABELS,
                        CONTINUATION_DOMAIN as ACTUARIAL_CONTINUATION_DOMAIN,
                        RATE_ESTIMANDS, V4_SUBMISSION_COLUMNS, ActuarialThresholds,
                        ObligationContract, actuarial_pass, eligibility_floor,
                        regions_from_admin, reserve_total)
from .admin import build_admin
from .businesses import build_businesses
from .character import draw_world_character
from .demography import (ANNUAL_SHOCK_RATE, SHOCK_FAMILY, SHOCK_LOADING_BAND,
                         draw_world_shocks)
from .dwellings import build_dwellings
from .events import (CONTINUATION_DOMAIN as EVENT_CONTINUATION_DOMAIN, EVENT_TYPES,
                     SHOCK_SUBSTREAM, build_event_history, replay_event_history)
from .hospitals import build_hospitals
from .hydrology import fill_depressions, flow_accumulation, flow_directions
from .identities import SEQUENCE_MASK, build_initial_identity_map
from .mechanisms import (DEVELOPMENT_BAND, HIDDEN_EXTRAPOLATION_AXES,
                         HIDDEN_IN_BAND_AXES, HIDDEN_LEVEL_PATTERNS,
                         N_HIDDEN_OUTSIDE_AXES, PUBLIC_ENVELOPE,
                         QUALIFYING_DIAGNOSIS_GROUPS, build_world_mechanisms,
                         contract_block)
from .microdata import build_microdata
from .population import build_population, resource_outposts
from .projection import (DEMAND_ESTIMAND, SHOCK_REDRAW_EVIDENCE_SCHEMA,
                         continuation_liabilities,
                         person_table_from_state, project_truth_from_history,
                         rate_truth_from_history)
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


ENSEMBLE_CACHE_SCHEMA = "meridia.ensemble.cache.v4"
PACKET_MANIFEST_SCHEMA = "meridia.packet.manifest.v1"
PACKET_BUILD_PROVENANCE_SCHEMA = "meridia.packet.build-provenance.v1"
PACKET_BUILD_INTENT_SCHEMA = "meridia.packet.build-intent.v1"
PACKET_CLASSES = ("development", "qualification", "graded")
PARTICIPANT_PACKET_FILES = frozenset({
    "contract.json",
    "experience_history.csv",
    "geography.csv",
    "survey_preliminary.csv",
    "survey_revised.csv",
    "sources/benchmark_preliminary.csv",
    "sources/benchmark_revised.csv",
    "sources/business_preliminary.csv",
    "sources/business_revised.csv",
    "sources/health_preliminary.csv",
    "sources/health_revised.csv",
    "sources/income_preliminary.csv",
    "sources/income_revised.csv",
    "sources/population_preliminary.csv",
    "sources/population_revised.csv",
})
_SURVEY_COLUMNS = (
    "household", "stratum", "design_weight", "age", "sex", "education", "income",
    "recent_hospitalization", "county", "psu", "psu_sampled_households",
)
_BENCHMARK_COLUMNS = ("item", "level", "unit", "value")
_BUSINESS_COLUMNS = (
    "record_id", "business_id", "enterprise_id", "industry", "county",
    "employee_count", "annual_payroll_cents",
)
_HEALTH_COLUMNS = (
    "record_id", "encounter_id", "patient_id", "facility_id", "given_code",
    "family_code", "birth_tick", "sex", "patient_county", "facility_county",
    "admission_tick", "discharge_tick", "service", "diagnosis_group", "outcome",
    "cost_cents",
)
_INCOME_COLUMNS = (
    "record_id", "taxpayer_id", "household_id", "given_code", "family_code",
    "birth_tick", "sex", "county", "employment_income_cents", "employer_id",
)
_POPULATION_COLUMNS = (
    "record_id", "person_id", "household_id", "given_code", "family_code",
    "birth_tick", "sex", "education", "county",
)
PARTICIPANT_CSV_SCHEMAS: dict[str, tuple[str, ...]] = {
    "experience_history.csv": EXPERIENCE_COLUMNS,
    "geography.csv": ("county", "state", "land_cells", "economic_band"),
    **{f"survey_{vintage}.csv": _SURVEY_COLUMNS
       for vintage in ("preliminary", "revised")},
    **{f"sources/benchmark_{vintage}.csv": _BENCHMARK_COLUMNS
       for vintage in ("preliminary", "revised")},
    **{f"sources/business_{vintage}.csv": _BUSINESS_COLUMNS
       for vintage in ("preliminary", "revised")},
    **{f"sources/health_{vintage}.csv": _HEALTH_COLUMNS
       for vintage in ("preliminary", "revised")},
    **{f"sources/income_{vintage}.csv": _INCOME_COLUMNS
       for vintage in ("preliminary", "revised")},
    **{f"sources/population_{vintage}.csv": _POPULATION_COLUMNS
       for vintage in ("preliminary", "revised")},
}
DEVELOPMENT_TRUTH_FILES = frozenset({
    "truth/detailed_revised.csv",
    "truth/truth_horizon.csv",
    "truth/truth_revised.csv",
})
RETAINED_PACKET_FILES = frozenset({
    "continuation_liabilities.npz",
    "continuation_shock_redraw.json",
    "rate_truth_horizon.csv",
    "source_evidence.npz",
    "truth_horizon.csv",
    "truth_revised.csv",
    "world.json",
})

# These are entry points rather than a hand-maintained dependency list. The import closure
# below discovers every in-package module the event runner and liability pricer execute.
CONTINUATION_SOURCE_ROOTS = ("actuarial", "events", "projection")


def _module_source_path(directory: Path, name: str) -> tuple[Path, bool] | None:
    relative = Path(*name.split("."))
    module = (directory / relative).with_suffix(".py")
    package = directory / relative / "__init__.py"
    if module.is_file():
        return module, False
    if package.is_file():
        return package, True
    return None


def _package_imports(source: str, *, current_module: str,
                     current_is_package: bool,
                     package_name: str = "meridia") -> set[str]:
    """Return candidate in-package imports from one module's complete syntax tree."""
    found: set[str] = set()
    current_package = current_module if current_is_package \
        else current_module.rpartition(".")[0]
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            for alias in node.names:
                prefix = f"{package_name}."
                if alias.name.startswith(prefix):
                    found.add(alias.name[len(prefix):])
            continue
        if not isinstance(node, ast.ImportFrom):
            continue
        base: str | None = None
        if node.level:
            parts = current_package.split(".") if current_package else []
            climb = node.level - 1
            if climb > len(parts):
                continue
            prefix_parts = parts[:len(parts) - climb] if climb else parts
            suffix_parts = node.module.split(".") if node.module else []
            base = ".".join(prefix_parts + suffix_parts)
        elif node.module == package_name:
            base = ""
        elif node.module and node.module.startswith(f"{package_name}."):
            base = node.module[len(package_name) + 1:]
        if base is None:
            continue
        if base:
            found.add(base)
        for alias in node.names:
            if alias.name != "*":
                found.add(".".join(part for part in (base, alias.name) if part))
    return found


def continuation_source_modules(package_dir: Path | None = None,
                                roots: tuple[str, ...] = CONTINUATION_SOURCE_ROOTS,
                                package_name: str = "meridia") -> tuple[str, ...]:
    """Compute the transitive relative/absolute import closure of the pricing roots."""
    directory = Path(package_dir) if package_dir is not None \
        else Path(__file__).resolve().parent
    queue = list(roots)
    seen: set[str] = set()
    for root in roots:
        if _module_source_path(directory, root) is None:
            raise FileNotFoundError(f"continuation source root {root!r} is missing")
    while queue:
        name = queue.pop()
        if name in seen:
            continue
        resolved = _module_source_path(directory, name)
        if resolved is None:
            continue
        path, is_package = resolved
        seen.add(name)
        candidates = _package_imports(
            path.read_text(encoding="utf-8"),
            current_module=name,
            current_is_package=is_package,
            package_name=package_name,
        )
        queue.extend(candidate for candidate in candidates if candidate not in seen)
    return tuple(sorted(seen))


@functools.lru_cache(maxsize=1)
def _default_continuation_source_files() -> tuple[tuple[str, bytes], ...]:
    source_root = Path(__file__).resolve().parent
    return tuple(
        (name, hashlib.sha256(_module_source_path(source_root, name)[0].read_bytes()).digest())
        for name in continuation_source_modules(source_root)
    )


def continuation_source_law_digest(package_dir: Path | None = None) -> str:
    """Digest the constants and transitive implementation of a continuation member."""
    digest = hashlib.sha256(b"meridia.continuation.source-law.v3")
    digest.update(json.dumps({
        "actuarial_continuation_domain": int(ACTUARIAL_CONTINUATION_DOMAIN),
        "event_continuation_domain": int(EVENT_CONTINUATION_DOMAIN),
        "shock_substream": int(SHOCK_SUBSTREAM),
        "annual_shock_rate": float(ANNUAL_SHOCK_RATE),
        "shock_family": SHOCK_FAMILY,
        "shock_loading_band": SHOCK_LOADING_BAND,
        "event_types": EVENT_TYPES,
    }, sort_keys=True).encode())
    if package_dir is None:
        source_files = _default_continuation_source_files()
    else:
        source_root = Path(package_dir)
        source_files = tuple(
            (name, hashlib.sha256(
                _module_source_path(source_root, name)[0].read_bytes()
            ).digest())
            for name in continuation_source_modules(source_root)
        )
    for name, source_digest in source_files:
        digest.update(name.encode("utf-8"))
        digest.update(source_digest)
    return digest.hexdigest()


def _digest_chunk(digest, value: bytes) -> None:
    digest.update(len(value).to_bytes(8, "big"))
    digest.update(value)


def _update_structural_digest(digest, label: str, value) -> None:
    """Hash nested runtime state with explicit types, shapes and field boundaries."""
    _digest_chunk(digest, label.encode("utf-8"))
    if value is None:
        _digest_chunk(digest, b"none")
    elif isinstance(value, bool):
        _digest_chunk(digest, b"bool")
        _digest_chunk(digest, b"1" if value else b"0")
    elif isinstance(value, (int, np.integer)):
        _digest_chunk(digest, b"int")
        _digest_chunk(digest, str(int(value)).encode("ascii"))
    elif isinstance(value, (float, np.floating)):
        _digest_chunk(digest, b"float64")
        _digest_chunk(digest, struct.pack("!d", float(value)))
    elif isinstance(value, str):
        _digest_chunk(digest, b"str")
        _digest_chunk(digest, value.encode("utf-8"))
    elif isinstance(value, bytes):
        _digest_chunk(digest, b"bytes")
        _digest_chunk(digest, value)
    elif isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            raise TypeError(f"cannot digest object array at {label}")
        array = np.ascontiguousarray(value)
        _digest_chunk(digest, b"ndarray")
        dtype = value.dtype.descr if value.dtype.fields else value.dtype.str
        _digest_chunk(digest, json.dumps(dtype, separators=(",", ":")).encode("utf-8"))
        _digest_chunk(digest, json.dumps(list(value.shape)).encode("ascii"))
        _digest_chunk(digest, array.tobytes())
    elif is_dataclass(value) and not isinstance(value, type):
        _digest_chunk(digest, b"dataclass")
        _digest_chunk(
            digest,
            f"{type(value).__module__}.{type(value).__qualname__}".encode("utf-8"),
        )
        for field in fields(value):
            _update_structural_digest(digest, field.name, getattr(value, field.name))
    elif isinstance(value, dict):
        _digest_chunk(digest, b"dict")
        _digest_chunk(digest, str(len(value)).encode("ascii"))
        ordered = sorted(value.items(), key=lambda item: (type(item[0]).__name__, repr(item[0])))
        for index, (key, item) in enumerate(ordered):
            _update_structural_digest(digest, f"key[{index}]", key)
            _update_structural_digest(digest, f"value[{index}]", item)
    elif isinstance(value, (list, tuple)):
        _digest_chunk(digest, b"tuple" if isinstance(value, tuple) else b"list")
        _digest_chunk(digest, str(len(value)).encode("ascii"))
        for index, item in enumerate(value):
            _update_structural_digest(digest, str(index), item)
    else:
        raise TypeError(f"cannot deterministically digest {type(value).__name__} at {label}")


def _structural_sha256(label: str, value) -> str:
    digest = hashlib.sha256(b"meridia.structural-digest.v1")
    _update_structural_digest(digest, label, value)
    return digest.hexdigest()


def baseline_ledger_digest(history: dict, obligation: ObligationContract,
                           horizon_months: int, region_of_county: np.ndarray,
                           admin: dict) -> str:
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
    _update_structural_digest(
        digest,
        "continuation_source_law",
        continuation_source_law_digest(),
    )
    # Hash the actual runtime inputs, not a hand-maintained projection of them. In
    # particular, branch.context contains the event parameters, demography, hospital
    # layout, mechanism object and every coefficient consumed by continuation_events;
    # history and its branch state determine the common start of every redrawn member.
    _update_structural_digest(digest, "history", history)
    _update_structural_digest(digest, "admin", admin)
    _update_structural_digest(digest, "horizon_months", horizon_months)
    _update_structural_digest(digest, "obligation", obligation)
    _update_structural_digest(digest, "region_of_county", region_of_county)
    return digest.hexdigest()


def _cache_scalar(archive: np.lib.npyio.NpzFile, name: str) -> str:
    """Read one required scalar metadata field without permitting pickle payloads."""
    if name not in archive.files:
        raise ValueError(f"ensemble cache is missing {name!r} metadata")
    value = np.asarray(archive[name])
    if value.shape != ():
        raise ValueError(f"ensemble cache metadata {name!r} is not scalar")
    return str(value.item())


def _is_canonical_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and set(value) <= set("0123456789abcdef")
    )


def _validate_cached_liability(liability: np.ndarray, n_regions: int) -> np.ndarray:
    values = np.asarray(liability)
    if values.ndim != 2:
        raise ValueError("ensemble cache liability must be a two-dimensional matrix")
    if values.shape[1] != int(n_regions):
        raise ValueError("ensemble cache liability has the wrong region count")
    if (not np.issubdtype(values.dtype, np.number)
            or np.issubdtype(values.dtype, np.bool_)
            or np.issubdtype(values.dtype, np.complexfloating)):
        raise ValueError("ensemble cache liability must be numeric")
    if not np.isfinite(values).all():
        raise ValueError("ensemble cache liability contains a non-finite value")
    if (values < 0).any():
        raise ValueError("ensemble cache liability contains a negative value")
    try:
        with np.errstate(over="ignore", invalid="ignore"):
            canonical = np.ascontiguousarray(values, dtype=np.float64)
    except (OverflowError, TypeError, ValueError) as error:
        raise ValueError("ensemble cache liability cannot be represented as float64") \
            from error
    if not np.isfinite(canonical).all():
        raise ValueError("ensemble cache liability overflows float64")
    return canonical


_SHOCK_REDRAW_EVIDENCE_KEYS = {
    "schema", "continuation_source_law_sha256", "member_count",
    "redrawn_member_count", "first_future_year", "future_year_count",
    "future_year_opportunity_count", "member_schedules",
    "ordered_member_schedule_digest_sha256", "distinct_future_schedule_count",
    "future_shock_year_count", "future_mortality_spike_year_count",
}


def _canonical_json_digest(value: object) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validate_shock_redraw_evidence(
    evidence: object, *, expected_members: int | None = None
) -> dict:
    """Validate the schedules actually passed through the continuation member pricer."""

    try:
        normalized = json.loads(json.dumps(evidence, sort_keys=True, allow_nan=False))
    except (TypeError, ValueError) as error:
        raise ValueError("continuation shock evidence is not finite JSON") from error
    if not isinstance(normalized, dict) or set(normalized) != _SHOCK_REDRAW_EVIDENCE_KEYS \
            or normalized.get("schema") != SHOCK_REDRAW_EVIDENCE_SCHEMA \
            or normalized.get("continuation_source_law_sha256") \
            != continuation_source_law_digest():
        raise ValueError("continuation shock evidence schema or source law differs")
    integer_fields = (
        "member_count", "redrawn_member_count", "first_future_year",
        "future_year_count", "future_year_opportunity_count",
        "distinct_future_schedule_count", "future_shock_year_count",
        "future_mortality_spike_year_count",
    )
    if any(isinstance(normalized.get(field), bool)
           or not isinstance(normalized.get(field), int)
           or normalized[field] < 0 for field in integer_fields):
        raise ValueError("continuation shock evidence counts are invalid")
    member_count = normalized["member_count"]
    redrawn_count = normalized["redrawn_member_count"]
    future_year_count = normalized["future_year_count"]
    if member_count < 1 or future_year_count < 1 \
            or redrawn_count != member_count \
            or normalized["future_year_opportunity_count"] \
            != redrawn_count * future_year_count \
            or expected_members is not None and member_count != int(expected_members):
        raise ValueError("continuation shock evidence design counts differ")
    schedules = normalized.get("member_schedules")
    if not isinstance(schedules, list) or len(schedules) != redrawn_count \
            or [row.get("member") for row in schedules if isinstance(row, dict)] \
            != list(range(member_count)):
        raise ValueError("continuation shock member schedules are incomplete")
    first_year = normalized["first_future_year"]
    total_shocks = 0
    mortality_spikes = 0
    canonical_schedules = set()
    for row in schedules:
        if not isinstance(row, dict) or set(row) != {"member", "future_shocks"} \
                or not isinstance(row.get("future_shocks"), list):
            raise ValueError("continuation shock schedule row fields differ")
        years = []
        for shock in row["future_shocks"]:
            if not isinstance(shock, dict):
                raise ValueError("continuation shock record is not an object")
            kind = shock.get("kind")
            expected_fields = SHOCK_FAMILY.get(kind)
            if expected_fields is None or set(shock) != {"year", "kind", *expected_fields}:
                raise ValueError("continuation shock kind or fields differ from the family")
            year = shock.get("year")
            if isinstance(year, bool) or not isinstance(year, int) \
                    or not first_year <= year < first_year + future_year_count:
                raise ValueError("continuation shock year is outside the horizon")
            years.append(year)
            for field, bounds in expected_fields.items():
                value = shock.get(field)
                if isinstance(value, bool) or not isinstance(value, (int, float)) \
                        or not math.isfinite(float(value)) \
                        or not float(bounds[0]) <= float(value) <= float(bounds[1]):
                    raise ValueError("continuation shock magnitude is outside its family")
            mortality_spikes += int(kind == "mortality_spike")
        if years != sorted(set(years)):
            raise ValueError("a continuation has repeated or unsorted shock years")
        total_shocks += len(years)
        canonical_schedules.add(json.dumps(
            row["future_shocks"], sort_keys=True, separators=(",", ":"),
            allow_nan=False,
        ))
    recorded_digest = normalized.get("ordered_member_schedule_digest_sha256")
    if not _is_canonical_sha256(recorded_digest) \
            or recorded_digest != _canonical_json_digest(schedules) \
            or normalized["distinct_future_schedule_count"] != len(canonical_schedules) \
            or normalized["future_shock_year_count"] != total_shocks \
            or normalized["future_mortality_spike_year_count"] != mortality_spikes:
        raise ValueError("continuation shock evidence aggregates do not recompute")
    return normalized


def _shock_evidence_prefix(evidence: dict, members: int) -> dict:
    source = _validate_shock_redraw_evidence(evidence)
    if members > source["member_count"]:
        raise ValueError("cached continuation shock evidence has too few members")
    schedules = source["member_schedules"][:int(members)]
    values = [row["future_shocks"] for row in schedules]
    prefixed = dict(source)
    prefixed.update({
        "member_count": int(members),
        "redrawn_member_count": len(schedules),
        "future_year_opportunity_count": len(schedules) * source["future_year_count"],
        "member_schedules": schedules,
        "ordered_member_schedule_digest_sha256": _canonical_json_digest(schedules),
        "distinct_future_schedule_count": len({
            json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
            for value in values
        }),
        "future_shock_year_count": sum(len(value) for value in values),
        "future_mortality_spike_year_count": sum(
            shock.get("kind") == "mortality_spike"
            for value in values for shock in value
        ),
    })
    return _validate_shock_redraw_evidence(prefixed, expected_members=members)


def _cached_liability(cache_dir: Path | None, key: str, members: int,
                      n_regions: int) -> tuple[np.ndarray, dict] | None:
    """Load a locally trusted cache entry after integrity and semantic checks.

    The embedded digest catches corruption, partial replacement and a payload copied
    across cache keys. It is not an authenticity proof against an actor who can write the
    cache and recompute the unkeyed digest; cache directories remain trusted build state.
    """
    if cache_dir is None:
        return None
    if not _is_canonical_sha256(key):
        raise ValueError("ensemble cache key is not a canonical SHA-256 digest")
    path = Path(cache_dir) / f"{key}.npz"
    if not path.is_file():
        return None
    try:
        with np.load(path, allow_pickle=False) as archive:
            expected_fields = {
                "schema", "key", "n_regions", "liability_sha256", "liability",
                "shock_redraw_evidence_json", "shock_redraw_evidence_sha256",
            }
            if (len(archive.files) != len(expected_fields)
                    or set(archive.files) != expected_fields):
                raise ValueError("ensemble cache archive has a non-canonical field set")
            if _cache_scalar(archive, "schema") != ENSEMBLE_CACHE_SCHEMA:
                raise ValueError("ensemble cache schema does not match this builder")
            embedded_key = _cache_scalar(archive, "key")
            if not _is_canonical_sha256(embedded_key) or embedded_key != key:
                raise ValueError("ensemble cache embedded key does not match its request")
            if int(_cache_scalar(archive, "n_regions")) != int(n_regions):
                raise ValueError("ensemble cache metadata has the wrong region count")
            stored = _validate_cached_liability(archive["liability"], n_regions)
            embedded_digest = _cache_scalar(archive, "liability_sha256")
            if not _is_canonical_sha256(embedded_digest):
                raise ValueError("ensemble cache liability digest is malformed")
            if embedded_digest != _structural_sha256("liability", stored):
                raise ValueError("ensemble cache liability does not match its digest")
            shock_json = _cache_scalar(archive, "shock_redraw_evidence_json")
            shock_digest = _cache_scalar(archive, "shock_redraw_evidence_sha256")
            try:
                shock_evidence = json.loads(shock_json)
            except json.JSONDecodeError as error:
                raise ValueError("ensemble cache shock evidence is invalid") from error
            if not _is_canonical_sha256(shock_digest) \
                    or shock_digest != _canonical_json_digest(shock_evidence):
                raise ValueError("ensemble cache shock evidence does not match its digest")
            shock_evidence = _validate_shock_redraw_evidence(
                shock_evidence, expected_members=stored.shape[0]
            )
    except (OSError, ValueError, TypeError) as error:
        if isinstance(error, ValueError) and str(error).startswith("ensemble cache"):
            raise
        raise ValueError("ensemble cache archive is unreadable") from error
    if stored.shape[0] < members:
        return None
    return (
        np.ascontiguousarray(stored[:members]),
        _shock_evidence_prefix(shock_evidence, members),
    )


def _store_liability(cache_dir: Path | None, key: str, liability: np.ndarray,
                     n_regions: int, shock_redraw_evidence: dict | None = None) -> None:
    if cache_dir is None:
        return
    if not _is_canonical_sha256(key):
        raise ValueError("ensemble cache key is not a canonical SHA-256 digest")
    values = _validate_cached_liability(liability, n_regions)
    shock_evidence = _validate_shock_redraw_evidence(
        shock_redraw_evidence, expected_members=values.shape[0]
    )
    shock_json = json.dumps(
        shock_evidence, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    directory = Path(cache_dir)
    directory.mkdir(parents=True, exist_ok=True)
    # numpy appends the suffix itself, so the scratch name has to carry it: the write is
    # done under a name nothing reads and moved into place, which keeps a build that dies
    # part way from leaving a half-written ensemble for the next one to trust.
    with tempfile.NamedTemporaryFile(
            prefix="partial-", suffix=f"-{key}.npz", dir=directory, delete=False) as handle:
        scratch = Path(handle.name)
    try:
        np.savez_compressed(
            scratch,
            schema=np.asarray(ENSEMBLE_CACHE_SCHEMA),
            key=np.asarray(key),
            n_regions=np.int64(n_regions),
            liability_sha256=np.asarray(_structural_sha256("liability", values)),
            shock_redraw_evidence_json=np.asarray(shock_json),
            shock_redraw_evidence_sha256=np.asarray(
                _canonical_json_digest(shock_evidence)
            ),
            liability=values,
        )
        scratch.replace(directory / f"{key}.npz")
    finally:
        scratch.unlink(missing_ok=True)


def _packet_build_provenance(params: PacketParams | dict) -> dict:
    """Bind packet bytes to the generator, runtime, and normalized parameters."""
    # ``sealing`` imports this module, so these accessors must be loaded only after the
    # packet module has finished importing. Packet builds and validation happen after
    # that point.
    from .sealing import v4_generator_source_law_digest, v4_runtime_law_digest

    record = {
        "schema": PACKET_BUILD_PROVENANCE_SCHEMA,
        "generator_source_law_sha256": v4_generator_source_law_digest(),
        "runtime_law_sha256": v4_runtime_law_digest(),
        "packet_params_sha256": _canonical_json_digest(
            _normalise_packet_params(params)
        ),
    }
    if any(not _is_canonical_sha256(value) for name, value in record.items()
           if name.endswith("_sha256")):
        raise RuntimeError("packet build provenance contains a malformed digest")
    return record


def _packet_seed_commitment(seed: int) -> str:
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("packet seed must be an integer")
    digest = hashlib.sha256(b"meridia.packet-build-seed.v1\0")
    digest.update(str(seed).encode("ascii"))
    return digest.hexdigest()


def _packet_build_paths(out_dir: Path) -> tuple[Path, Path]:
    destination = Path(out_dir)
    name = destination.name
    if name in {"", ".", ".."}:
        raise ValueError("packet output must name one packet directory")
    return (
        destination.parent / f".{name}.build-intent.json",
        destination.parent / f".{name}.staging",
    )


def _packet_build_intent(
    out_dir: Path,
    *,
    seed: int,
    params: PacketParams | dict,
    packet_class: str,
    development: bool,
    graded_authorization: object | None,
) -> dict:
    intent_path, staging = _packet_build_paths(out_dir)
    del intent_path
    if packet_class not in PACKET_CLASSES:
        raise ValueError(f"unknown packet class {packet_class!r}")
    if development != (packet_class == "development"):
        raise ValueError("development flag and packet class disagree")
    if packet_class == "graded":
        from .sealing import V4PublicationAuthorization

        if type(graded_authorization) is not V4PublicationAuthorization:
            raise ValueError("graded packet publication requires sealed authorization")
        authorization = {
            "index": graded_authorization.index,
            "binding_sha256": graded_authorization.before.binding_sha256,
        }
    else:
        if graded_authorization is not None:
            raise ValueError("publication authorization is only valid for graded packets")
        authorization = None
    return {
        "schema": PACKET_BUILD_INTENT_SCHEMA,
        "stage": "building",
        "destination_name": Path(out_dir).name,
        "staging_name": staging.name,
        "packet_class": packet_class,
        "development": development,
        "seed_commitment_sha256": _packet_seed_commitment(seed),
        "provenance": _packet_build_provenance(params),
        "graded_authorization": authorization,
    }


def _intent_bytes(record: dict) -> bytes:
    return (json.dumps(
        record, sort_keys=True, separators=(",", ":"), allow_nan=False
    ) + "\n").encode("utf-8")


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _read_intent_descriptor(descriptor: int) -> bytes:
    metadata = os.fstat(descriptor)
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
        raise ValueError("packet build intent must be one regular file")
    if metadata.st_size > 65_536:
        raise ValueError("packet build intent is unreasonably large")
    os.lseek(descriptor, 0, os.SEEK_SET)
    chunks: list[bytes] = []
    remaining = metadata.st_size
    while remaining:
        block = os.read(descriptor, min(remaining, 65_536))
        if not block:
            raise ValueError("packet build intent is truncated")
        chunks.append(block)
        remaining -= len(block)
    return b"".join(chunks)


def _lock_intent_descriptor(descriptor: int) -> None:
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        raise RuntimeError("packet build is already active for this destination") from error


def _open_existing_build_intent(path: Path, expected: bytes) -> int:
    flags = os.O_RDWR | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ValueError("packet build intent is unreadable or linked") from error
    try:
        _lock_intent_descriptor(descriptor)
        if _read_intent_descriptor(descriptor) != expected:
            raise ValueError("packet build intent does not match this exact packet build")
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _create_build_intent(path: Path, content: bytes) -> int:
    flags = (
        os.O_RDWR
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptor = os.open(path, flags, 0o600)
    try:
        _lock_intent_descriptor(descriptor)
        written = 0
        while written < len(content):
            written += os.write(descriptor, content[written:])
        os.fsync(descriptor)
        _fsync_directory(path.parent)
        return descriptor
    except BaseException:
        try:
            metadata = path.lstat()
            opened = os.fstat(descriptor)
            if (metadata.st_dev, metadata.st_ino) == (opened.st_dev, opened.st_ino):
                path.unlink()
        finally:
            os.close(descriptor)
        raise


def _unlink_locked_build_intent(path: Path, descriptor: int, expected: bytes) -> None:
    if _read_intent_descriptor(descriptor) != expected:
        raise RuntimeError("packet build intent changed while the packet was built")
    try:
        linked = path.lstat()
    except FileNotFoundError as error:
        raise RuntimeError("packet build intent disappeared while the packet was built") \
            from error
    opened = os.fstat(descriptor)
    if not stat.S_ISREG(linked.st_mode) \
            or (linked.st_dev, linked.st_ino) != (opened.st_dev, opened.st_ino):
        raise RuntimeError("packet build intent path changed while the packet was built")
    path.unlink()
    _fsync_directory(path.parent)


def _remove_exact_staging(path: Path) -> bool:
    try:
        mode = path.lstat().st_mode
    except FileNotFoundError:
        return False
    if not stat.S_ISDIR(mode) or stat.S_ISLNK(mode):
        raise ValueError("packet staging path is not a real directory")
    shutil.rmtree(path)
    _fsync_directory(path.parent)
    return True


def _begin_packet_build(out_dir: Path, intent: dict) -> tuple[Path, Path, int, bytes]:
    intent_path, staging = _packet_build_paths(out_dir)
    expected = _intent_bytes(intent)
    created = False
    try:
        descriptor = _create_build_intent(intent_path, expected)
        created = True
    except FileExistsError:
        descriptor = _open_existing_build_intent(intent_path, expected)
    staging_created = False
    try:
        try:
            staging_mode = staging.lstat().st_mode
        except FileNotFoundError:
            staging_mode = None
        if staging_mode is not None:
            if created:
                raise ValueError(
                    "packet staging exists without the prior matching build intent"
                )
            _remove_exact_staging(staging)
        staging.mkdir(mode=0o700)
        staging_created = True
        _fsync_directory(staging.parent)
        return intent_path, staging, descriptor, expected
    except BaseException:
        if staging_created:
            try:
                _remove_exact_staging(staging)
            except (OSError, ValueError):
                pass
        if created:
            try:
                _unlink_locked_build_intent(intent_path, descriptor, expected)
            except (OSError, RuntimeError, ValueError):
                pass
        os.close(descriptor)
        raise


def _finalize_packet_build_intent(
    out_dir: Path,
    *,
    seed: int,
    params: PacketParams | dict,
    packet_class: str,
    development: bool,
    graded_authorization: object | None = None,
) -> bool:
    """Clear only a matching post-publication intent after the packet authenticates."""
    destination = Path(out_dir)
    intent_path, staging = _packet_build_paths(destination)
    try:
        intent_path.lstat()
    except FileNotFoundError:
        try:
            staging.lstat()
        except FileNotFoundError:
            return False
        raise ValueError("packet staging exists without a matching build intent")
    intent = _packet_build_intent(
        destination,
        seed=seed,
        params=params,
        packet_class=packet_class,
        development=development,
        graded_authorization=graded_authorization,
    )
    expected = _intent_bytes(intent)
    descriptor = _open_existing_build_intent(intent_path, expected)
    try:
        try:
            staging.lstat()
        except FileNotFoundError:
            pass
        else:
            raise ValueError(
                "published packet has an unexpected build staging directory"
            )
        _unlink_locked_build_intent(intent_path, descriptor, expected)
        return True
    finally:
        os.close(descriptor)


def _build_packet_into(seed: int, out_dir: Path, params: PacketParams,
                       development: bool, workers: int, cache_dir: Path | None,
                       packet_class: str, build_provenance: dict) -> dict:
    """Write one packet into an existing empty staging directory."""
    out_dir = Path(out_dir)
    if not out_dir.is_dir() or any(out_dir.iterdir()):
        raise ValueError("packet staging directory must exist and be empty")
    if development and params.regime != "development":
        raise ValueError("a development packet must use the development source regime")
    if build_provenance != _packet_build_provenance(params):
        raise ValueError("packet build provenance changed before construction")
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

    # The tail truth: M independently redrawn continuations from the branch state the
    # ledger kept at the revised snapshot, priced into regional present values. The
    # ledger's designated realized horizon is the separate point truth and is not inserted
    # into this ensemble. Nothing about the ensemble crosses to the participant side
    # except the single aggregate the protocol publishes on purpose, the reserve total.
    thresholds = ActuarialThresholds()
    region_of_county = regions_from_admin(admin)
    n_regions = int(admin["n_states"])
    weights = reserve_weights(population_revised, admin["county_state"], ticks["revised"],
                              n_regions, params.reserve_weight_spread)
    baseline_share = reserve_baseline_share(population_revised, admin["county_state"],
                                            ticks["revised"], n_regions,
                                            obligation.eligibility_min_age)
    cache_key = baseline_ledger_digest(built["history"], obligation,
                                       params.horizon_months, region_of_county, admin)
    cached_ensemble = _cached_liability(
        cache_dir, cache_key, params.ensemble_members, n_regions
    )
    if cached_ensemble is None:
        source_law_sha256 = continuation_source_law_digest()
        liability, shock_redraw_evidence = continuation_liabilities(
            built["history"], admin, ticks["revised"], params.horizon_months,
            obligation, params.ensemble_members, region_of_county, workers=workers,
            shock_source_law_sha256=source_law_sha256,
            return_shock_evidence=True,
        )
        shock_redraw_evidence = _validate_shock_redraw_evidence(
            shock_redraw_evidence, expected_members=params.ensemble_members
        )
        _store_liability(
            cache_dir, cache_key, liability, n_regions, shock_redraw_evidence
        )
    else:
        liability, shock_redraw_evidence = cached_ensemble
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
               "allocation_rule": {
                   "finite": True,
                   "minimum": 0.0,
                   "sum": "reserve.total",
                   "tolerance": float(thresholds.feasibility_tolerance),
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
        "participant_csv_schemas": {
            name: list(columns)
            for name, columns in sorted(PARTICIPANT_CSV_SCHEMAS.items())
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
            "regional_loading_formula": "1 + L_r * (m - 1)",
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
            "file": "sources/benchmark_revised.csv",
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
    np.savez_compressed(
        retained / "continuation_liabilities.npz",
        liability=liability,
        weights=weights,
    )
    (retained / "continuation_shock_redraw.json").write_text(
        json.dumps(shock_redraw_evidence, indent=1, sort_keys=True) + "\n"
    )
    sealed = {k: v for k, v in built["sources"].items() if k != "public_snapshots"}
    np.savez_compressed(retained / "source_evidence.npz", **_flatten(sealed))
    (retained / "world.json").write_text(json.dumps({
        "seed": seed, "packet_class": packet_class,
        "build_provenance": build_provenance,
        "character": built["character"]["draw"], "shocks": built["shocks"],
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

    manifest = {"schema": PACKET_MANIFEST_SCHEMA, "development": development,
                "packet_class": packet_class, "participant": {}, "retained": {}}
    for side in ("participant", "retained"):
        for path in sorted((out_dir / side).rglob("*")):
            if path.is_file():
                manifest[side][str(path.relative_to(out_dir / side))] = {
                    "sha256": _sha256(path), "bytes": path.stat().st_size}
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")
    return manifest


def build_packet(seed: int, out_dir: Path, params: PacketParams = PacketParams(),
                 development: bool = False, workers: int = 1,
                 cache_dir: Path | None = None,
                 packet_class: str | None = None,
                 graded_authorization: object | None = None) -> dict:
    """Write one packet atomically and return its authenticated manifest.

    ``workers`` divides the continuation ensemble between processes and changes nothing
    else: every member is a deterministic function of the seed and its own index.

    ``cache_dir`` holds continuation ensembles keyed on the digest of the baseline ledger
    that produced them. The ensemble is the whole cost of a packet at the committed size,
    and it depends on nothing downstream of the ledger, so a rebuild that changes only
    what a verifier or a bar reads takes the futures back off the shelf. A cached
    ensemble with more members than the packet asks for is used from the front, since a
    member is a function of its own index.

    ``packet_class`` is retained and manifested, never published to the participant. For
    compatibility with direct callers, an omitted class means ``development`` when the
    truth is published and ``qualification`` otherwise; graded builders must say so.

    A graded builder supplies the concrete publication authority issued by the V4 sealing
    layer. It receives no staging path. The packet is validated before authorization and
    again immediately afterward, before the no-replace rename makes it visible.
    """
    out_dir = Path(out_dir)
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise ValueError("workers must be a positive integer")
    workers = int(workers)
    if out_dir.exists():
        raise FileExistsError("packet output already exists")
    resolved_class = packet_class
    if resolved_class is None:
        resolved_class = "development" if development else "qualification"
    if resolved_class not in PACKET_CLASSES:
        raise ValueError(f"unknown packet class {resolved_class!r}")
    if resolved_class == "graded":
        from .sealing import V4PublicationAuthorization
        if type(graded_authorization) is not V4PublicationAuthorization:
            raise ValueError("graded packet publication requires sealed authorization")
        graded_authorization.assert_initial(seed=seed)
        initial_confirmation = graded_authorization.confirm(seed=seed, params=params)
        if initial_confirmation != graded_authorization.before:
            raise RuntimeError("graded publication authorization is not bound")
    elif graded_authorization is not None:
        raise ValueError("publication authorization is only valid for graded packets")
    if development != (resolved_class == "development"):
        raise ValueError("development flag and packet class disagree")
    if development and params.regime != "development":
        raise ValueError("a development packet must use the development source regime")

    out_dir.parent.mkdir(parents=True, exist_ok=True)
    intent = _packet_build_intent(
        out_dir,
        seed=seed,
        params=params,
        packet_class=resolved_class,
        development=development,
        graded_authorization=graded_authorization,
    )
    intent_path, staging, intent_descriptor, expected_intent = _begin_packet_build(
        out_dir, intent
    )
    published = False
    try:
        _build_packet_into(seed, staging, params, development, workers, cache_dir,
                           resolved_class, intent["provenance"])
        manifest = validate_packet_directory(
            staging,
            expected_packet_class=resolved_class,
            expected_params=params,
            expected_seed=seed,
        )
        if graded_authorization is not None:
            confirmation = graded_authorization.confirm(seed=seed, params=params)
            if confirmation != graded_authorization.before:
                raise RuntimeError("graded publication authorization is not bound")
            confirmed_manifest = validate_packet_directory(
                staging,
                expected_packet_class=resolved_class,
                expected_params=params,
                expected_seed=seed,
            )
            if confirmed_manifest != manifest:
                raise RuntimeError("graded packet changed during final authorization")
            manifest = confirmed_manifest
        if _read_intent_descriptor(intent_descriptor) != expected_intent:
            raise RuntimeError("packet build intent changed before publication")
        _publish_staging_directory(staging, out_dir)
        published = True
        _fsync_directory(out_dir.parent)
        _unlink_locked_build_intent(
            intent_path, intent_descriptor, expected_intent
        )
        return manifest
    except BaseException:
        if not published:
            try:
                _remove_exact_staging(staging)
            except (OSError, ValueError):
                pass
            try:
                _unlink_locked_build_intent(
                    intent_path, intent_descriptor, expected_intent
                )
            except (OSError, RuntimeError, ValueError):
                pass
        raise
    finally:
        os.close(intent_descriptor)


def _normalise_packet_params(params: PacketParams | dict) -> dict:
    record = asdict(params) if isinstance(params, PacketParams) else dict(params)
    return json.loads(json.dumps(record, sort_keys=True))


def _contains_metadata_key(value, forbidden: set[str]) -> bool:
    if isinstance(value, dict):
        return any(str(key).casefold() in forbidden
                   or _contains_metadata_key(item, forbidden)
                   for key, item in value.items())
    if isinstance(value, list):
        return any(_contains_metadata_key(item, forbidden) for item in value)
    return False


def _is_disclosed_seed_value(value: object, expected_seed: int) -> bool:
    """Whether one public scalar contains the canonical graded seed representation."""
    if isinstance(value, bool) or value is None:
        return False
    if isinstance(value, (int, np.integer)):
        return int(value) == expected_seed
    if isinstance(value, (float, np.floating)):
        return math.isfinite(float(value)) and Decimal(str(value)) == Decimal(expected_seed)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return False
        seed_token = re.escape(str(expected_seed))
        if re.search(rf"(?<!\d){seed_token}(?!\d)", text):
            return True
        try:
            parsed = Decimal(text)
        except InvalidOperation:
            return False
        return parsed.is_finite() and parsed == Decimal(expected_seed)
    return False


def _contains_disclosed_seed(value: object, expected_seed: int) -> bool:
    if isinstance(value, dict):
        return any(
            _is_disclosed_seed_value(key, expected_seed)
            or _contains_disclosed_seed(item, expected_seed)
            for key, item in value.items()
        )
    if isinstance(value, list):
        return any(_contains_disclosed_seed(item, expected_seed) for item in value)
    return _is_disclosed_seed_value(value, expected_seed)


def _validate_graded_world_semantics(world: object) -> None:
    """Fail closed unless retained metadata describes the registered hidden design."""
    try:
        if not isinstance(world, dict) or world.get("regime") != "hidden":
            raise ValueError("graded packet was not built under the hidden regime")
        mechanisms = world.get("mechanisms")
        design = mechanisms.get("design") if isinstance(mechanisms, dict) else None
        if not isinstance(design, dict):
            raise ValueError("graded packet mechanism design is missing")
        if design.get("regime") != "hidden" or design.get("cell") != -1:
            raise ValueError("graded packet carries a development mechanism design")

        outside = design.get("outside")
        levels = design.get("levels")
        intensity = design.get("intensity")
        if (not isinstance(outside, list)
                or len(outside) != N_HIDDEN_OUTSIDE_AXES
                or len(set(outside)) != len(outside)
                or not set(outside) <= set(HIDDEN_EXTRAPOLATION_AXES)):
            raise ValueError("graded packet has a non-canonical extrapolation set")
        if (not isinstance(levels, list)
                or any(isinstance(level, bool) or not isinstance(level, int)
                       for level in levels)
                or tuple(levels) not in HIDDEN_LEVEL_PATTERNS):
            raise ValueError("graded packet has a non-canonical hidden level pattern")
        axes = set(DEVELOPMENT_BAND)
        if not isinstance(intensity, dict) or set(intensity) != axes:
            raise ValueError("graded packet has a malformed mechanism intensity record")

        for axis in axes:
            raw_value = intensity[axis]
            if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
                raise ValueError("graded packet mechanism intensity is not numeric")
            value = float(raw_value)
            if not math.isfinite(value):
                raise ValueError("graded packet mechanism intensity is not finite")
            envelope_low, envelope_high = PUBLIC_ENVELOPE[axis]
            if not envelope_low <= value <= envelope_high:
                raise ValueError("graded packet mechanism intensity left its public envelope")
            band_low, band_high = DEVELOPMENT_BAND[axis]
            if axis in outside:
                if band_low <= value <= band_high:
                    raise ValueError(
                        "graded packet extrapolation stayed in the development band"
                    )
            elif not band_low <= value <= band_high:
                raise ValueError(
                    "graded packet unlisted intensity left the development band"
                )
        if set(outside) & set(HIDDEN_IN_BAND_AXES):
            raise ValueError("graded packet extrapolates an unidentifiable axis")
    except (KeyError, TypeError, ValueError) as error:
        if isinstance(error, ValueError) and str(error).startswith("graded packet"):
            raise
        raise ValueError("graded packet hidden-world metadata is malformed") from error


def _publish_staging_directory(staging: Path, destination: Path) -> None:
    """Atomically publish a sibling directory without replacing an existing path."""
    source_bytes = os.fsencode(staging)
    destination_bytes = os.fsencode(destination)
    libc = ctypes.CDLL(None, use_errno=True)
    if sys.platform == "darwin":
        rename_exclusive = 0x00000004
        rename = libc.renamex_np
        rename.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
        result = rename(source_bytes, destination_bytes, rename_exclusive)
    elif sys.platform.startswith("linux"):
        try:
            rename = libc.renameat2
        except AttributeError as error:
            raise RuntimeError(
                "atomic no-replace directory publication is unavailable"
            ) from error
        at_current_working_directory = -100
        rename_no_replace = 1
        rename.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int,
                           ctypes.c_char_p, ctypes.c_uint]
        result = rename(at_current_working_directory, source_bytes,
                        at_current_working_directory, destination_bytes,
                        rename_no_replace)
    else:
        raise RuntimeError("atomic no-replace directory publication is unavailable")
    if result == 0:
        return
    code = ctypes.get_errno()
    if code in {errno.EEXIST, errno.ENOTEMPTY}:
        raise FileExistsError("packet output already exists")
    raise OSError(code, os.strerror(code), os.fspath(destination))


def validate_packet_directory(path: Path, *, expected_packet_class: str,
                              expected_params: PacketParams | dict,
                              expected_seed: int) -> dict:
    """Authenticate a complete packet for resume or immediately before publication.

    The expected seed is compared to retained metadata but never interpolated into an
    error. Participant metadata is checked separately so neither the seed nor the packet
    class can cross the sealed boundary.
    """
    packet = Path(path)
    if expected_packet_class not in PACKET_CLASSES:
        raise ValueError(f"unknown packet class {expected_packet_class!r}")
    if isinstance(expected_seed, bool) or not isinstance(expected_seed, int):
        raise ValueError("expected packet seed must be an integer")
    if not packet.is_dir() or packet.is_symlink():
        raise ValueError("packet directory is missing or is not a real directory")
    root_entries = {item.name for item in packet.iterdir()}
    if root_entries != {"manifest.json", "participant", "retained"}:
        raise ValueError("packet root does not contain exactly the required entries")
    for side in ("participant", "retained"):
        directory = packet / side
        if not directory.is_dir() or directory.is_symlink():
            raise ValueError(f"packet {side} side is missing or is not a real directory")

    manifest_path = packet / "manifest.json"
    try:
        manifest_mode = manifest_path.lstat().st_mode
    except OSError as error:
        raise ValueError("packet manifest is missing") from error
    if not stat.S_ISREG(manifest_mode):
        raise ValueError("packet manifest must be a regular file")
    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("packet manifest is unreadable") from error
    manifest_fields = {"schema", "development", "packet_class", "participant", "retained"}
    if (not isinstance(manifest, dict) or set(manifest) != manifest_fields
            or manifest.get("schema") != PACKET_MANIFEST_SCHEMA):
        raise ValueError("packet manifest schema does not match this builder")
    if manifest.get("packet_class") != expected_packet_class:
        raise ValueError("packet manifest class does not match the expected class")
    expected_development = expected_packet_class == "development"
    if manifest.get("development") is not expected_development:
        raise ValueError("packet manifest development flag does not match its class")

    for side in ("participant", "retained"):
        records = manifest.get(side)
        if not isinstance(records, dict) or not records:
            raise ValueError(f"packet manifest has no {side} file inventory")
        expected_inventory = set(
            PARTICIPANT_PACKET_FILES if side == "participant" else RETAINED_PACKET_FILES
        )
        if expected_development:
            if side == "participant":
                expected_inventory |= DEVELOPMENT_TRUTH_FILES
            else:
                expected_inventory.add("detailed_revised.csv")
        if set(records) != expected_inventory:
            raise ValueError(f"packet manifest has a non-canonical {side} inventory")
        if side == "participant" and any(
                Path(name).suffix not in {".csv", ".json"} for name in records):
            raise ValueError("packet participant inventory contains an unsupported file type")
        directory = packet / side
        expected_directories: set[str] = set()
        for name in expected_inventory:
            parent = Path(name).parent
            while parent != Path("."):
                expected_directories.add(str(parent))
                parent = parent.parent
        files: dict[str, Path] = {}
        directories: set[str] = set()
        try:
            descendants = list(directory.rglob("*"))
            for item in descendants:
                relative = str(item.relative_to(directory))
                mode = item.lstat().st_mode
                if stat.S_ISLNK(mode):
                    raise ValueError(f"packet {side} side contains a symbolic link")
                if stat.S_ISDIR(mode):
                    directories.add(relative)
                elif stat.S_ISREG(mode):
                    files[relative] = item
                else:
                    raise ValueError(f"packet {side} side contains a special file")
        except OSError as error:
            raise ValueError(f"packet {side} topology is unreadable") from error
        if directories != expected_directories:
            raise ValueError(f"packet {side} directory topology is non-canonical")
        if set(files) != set(records):
            raise ValueError(f"packet {side} file set does not match its manifest")
        for name, file_path in files.items():
            record = records[name]
            if not isinstance(record, dict) or set(record) != {"sha256", "bytes"}:
                raise ValueError(f"packet {side} manifest entry is malformed")
            size = record.get("bytes")
            digest = record.get("sha256")
            if isinstance(size, bool) or not isinstance(size, int) or size < 0:
                raise ValueError(f"packet {side} manifest byte count is malformed")
            if not _is_canonical_sha256(digest):
                raise ValueError(f"packet {side} manifest digest is malformed")
            if size != file_path.stat().st_size or digest != _sha256(file_path):
                raise ValueError(f"packet {side} file does not match its manifest")

    world_path = packet / "retained" / "world.json"
    try:
        world = json.loads(world_path.read_text())
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("retained world metadata is unreadable") from error
    if world.get("packet_class") != expected_packet_class:
        raise ValueError("retained world class does not match the expected class")
    seed = world.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed != expected_seed:
        raise ValueError("retained world seed does not match the expected seed")
    if world.get("params") != _normalise_packet_params(expected_params):
        raise ValueError("retained world parameters do not match the expected parameters")
    if world.get("build_provenance") != _packet_build_provenance(expected_params):
        raise ValueError("retained packet build provenance does not match this builder")
    if expected_packet_class == "graded":
        _validate_graded_world_semantics(world)

    forbidden = {"seed", "packet_class"}
    participant = packet / "participant"
    try:
        contract = json.loads((participant / "contract.json").read_text())
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("participant contract is unreadable") from error
    for json_path in participant.rglob("*.json"):
        try:
            public_metadata = json.loads(json_path.read_text())
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise ValueError("participant JSON metadata is unreadable") from error
        if _contains_metadata_key(public_metadata, forbidden):
            raise ValueError("participant metadata exposes sealed packet identity")
        if (expected_packet_class == "graded"
                and _contains_disclosed_seed(public_metadata, expected_seed)):
            raise ValueError("participant metadata exposes the sealed packet seed")
    for csv_path in participant.rglob("*.csv"):
        try:
            with open(csv_path, encoding="utf-8", newline="") as handle:
                reader = csv.reader(handle, strict=True)
                header = next(reader, None)
                if not header or not any(name.strip() for name in header):
                    raise ValueError("participant table header is empty")
                columns = {name.strip().casefold() for name in header}
                if "" in columns:
                    raise ValueError("participant table header contains an empty column name")
                if columns & forbidden:
                    raise ValueError("participant table exposes sealed packet identity")
                if (expected_packet_class == "graded"
                        and any(_is_disclosed_seed_value(value, expected_seed)
                                for value in header)):
                    raise ValueError("participant table exposes the sealed packet seed")
                for row in reader:
                    if len(row) != len(header):
                        raise ValueError("participant table has a malformed row")
                    if (expected_packet_class == "graded"
                            and any(_is_disclosed_seed_value(value, expected_seed)
                                    for value in row)):
                        raise ValueError("participant table exposes the sealed packet seed")
        except (OSError, UnicodeError, csv.Error) as error:
            raise ValueError("participant table is unreadable") from error
    expected_csv_schemas = {
        name: list(columns)
        for name, columns in sorted(PARTICIPANT_CSV_SCHEMAS.items())
    }
    observed_csv_schemas = {
        name: columns
        for name, columns in participant_columns(packet).items()
        if name in PARTICIPANT_CSV_SCHEMAS
    }
    if contract.get("participant_csv_schemas") != expected_csv_schemas \
            or observed_csv_schemas != expected_csv_schemas:
        raise ValueError("participant CSV schemas differ from the public contract")
    benchmark_file = contract.get("benchmark", {}).get("file") \
        if isinstance(contract.get("benchmark"), dict) else None
    if benchmark_file != "sources/benchmark_revised.csv" \
            or not (participant / benchmark_file).is_file():
        raise ValueError("participant benchmark path differs from the public contract")
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
