"""Imperfect observed sources cut from the append-only institutional ledger.

The participant side of this module is four flat NumPy tables: population, business,
income, and health.  Every geography code is looked up from ``admin["county"]`` at the
recorded cell.  Preliminary and revised snapshots are materialized from exactly the
events whose ``recorded_tick`` is at or before the snapshot; truth-time replay remains a
separate operation.

Observed identifiers are source-specific random tokens drawn without using truth IDs.
The retained package contains the truth crosswalks and explicit error mechanisms needed
to score linkage, coverage, and revision decisions.  Neither is nested under a public
snapshot and neither may be exported with participant files.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np

from meridia.events import EVENT_TYPES, replay_event_history
from meridia.identities import SEQUENCE_MASK, truth_world_id

OBSERVED_SOURCES: Final = ("population", "business", "income", "health")

MECHANISM_BITS: Final = {
    "duplicate": 1,
    "split": 2,
    "merged": 4,
    "stale": 8,
    "county_error": 16,
    "linkage_error": 32,
    "item_missing": 64,
}

PUBLIC_SCHEMAS: Final = {
    "population": {
        "record_id": np.dtype(np.uint64),
        "person_id": np.dtype(np.uint64),
        "household_id": np.dtype(np.uint64),
        "name_code": np.dtype(np.uint64),
        "birth_tick": np.dtype(np.int64),
        "sex": np.dtype(np.int8),
        "education": np.dtype(np.int8),
        "county": np.dtype(np.int32),
    },
    "business": {
        "record_id": np.dtype(np.uint64),
        "business_id": np.dtype(np.uint64),
        "enterprise_id": np.dtype(np.uint64),
        "industry": np.dtype(np.int16),
        "county": np.dtype(np.int32),
        "employee_count": np.dtype(np.int32),
        "annual_payroll_cents": np.dtype(np.float64),
    },
    "income": {
        "record_id": np.dtype(np.uint64),
        "taxpayer_id": np.dtype(np.uint64),
        "household_id": np.dtype(np.uint64),
        "name_code": np.dtype(np.uint64),
        "birth_tick": np.dtype(np.int64),
        "sex": np.dtype(np.int8),
        "county": np.dtype(np.int32),
        "employment_income_cents": np.dtype(np.float64),
        "employer_id": np.dtype(np.uint64),
    },
    "health": {
        "record_id": np.dtype(np.uint64),
        "encounter_id": np.dtype(np.uint64),
        "patient_id": np.dtype(np.uint64),
        "facility_id": np.dtype(np.uint64),
        "name_code": np.dtype(np.uint64),
        "birth_tick": np.dtype(np.int64),
        "sex": np.dtype(np.int8),
        "patient_county": np.dtype(np.int32),
        "facility_county": np.dtype(np.int32),
        "admission_tick": np.dtype(np.int64),
        "discharge_tick": np.dtype(np.int64),
        "service": np.dtype(np.int8),
        "diagnosis_group": np.dtype(np.int16),
        "outcome": np.dtype(np.int8),
        "cost_cents": np.dtype(np.float64),
    },
}

_CROSSWALK_SCHEMA: Final = {
    "observed_record_id": np.dtype(np.uint64),
    "observed_entity_id": np.dtype(np.uint64),
    "truth_entity_id": np.dtype(np.uint64),
    "mechanism_code": np.dtype(np.int16),
    "valid_from_tick": np.dtype(np.int64),
    "valid_to_tick": np.dtype(np.int64),
}

_MECHANISM_SCHEMA: Final = {
    "truth_entity_id": np.dtype(np.uint64),
    "covered": np.dtype(np.bool_),
    "duplicate": np.dtype(np.bool_),
    "split": np.dtype(np.bool_),
    "merge_group": np.dtype(np.int64),
    "county_error": np.dtype(np.bool_),
    "linkage_error": np.dtype(np.bool_),
    "item_missing": np.dtype(np.bool_),
}


@dataclass(frozen=True)
class SourceParams:
    """Public mechanism rates for the four imperfect sources."""

    population_coverage: float = 0.965
    business_coverage: float = 0.940
    income_coverage: float = 0.900
    health_coverage: float = 0.925
    outpost_coverage_penalty: float = 0.140
    duplicate_rate: float = 0.025
    split_rate: float = 0.012
    merge_rate: float = 0.010
    county_error_rate: float = 0.018
    linkage_error_rate: float = 0.035
    item_missing_rate: float = 0.075


def _validate_params(params: SourceParams) -> None:
    if not isinstance(params, SourceParams):
        raise TypeError("params must be SourceParams")
    values = tuple(float(value) for value in params.__dict__.values())
    if not np.isfinite(values).all() or any(
        not 0.0 <= value <= 1.0 for value in values
    ):
        raise ValueError("source mechanism rates must be finite and in [0, 1]")
    for coverage in (
        params.population_coverage,
        params.business_coverage,
        params.income_coverage,
        params.health_coverage,
    ):
        if coverage - params.outpost_coverage_penalty <= 0.0:
            raise ValueError("outpost coverage penalty removes an entire source")


def _params_record(params: SourceParams) -> dict[str, float]:
    return {name: float(value) for name, value in params.__dict__.items()}


def _params_from_record(record: dict) -> SourceParams:
    try:
        return SourceParams(
            **{name: float(record[name]) for name in SourceParams.__dataclass_fields__}
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("source parameter record is incomplete") from exc


def _sequence_position(ids: np.ndarray) -> np.ndarray:
    return ((np.asarray(ids, dtype=np.uint64) & np.uint64(SEQUENCE_MASK)) - 1).astype(
        np.int64
    )


def _random_tokens(seed: int, domain: int, count: int) -> np.ndarray:
    """Draw compact opaque IDs in an observed-only namespace.

    The draw receives a row count, never a truth ID.  The high byte is an observed
    domain tag in 0x80..0xbf, disjoint from every sealed truth namespace in v0.  The
    lower 56 bits are pseudorandom, nonzero, and checked for collision.
    """
    if not 0 <= domain < 64:
        raise ValueError("observed token domain is outside [0, 64)")
    if count < 0:
        raise ValueError("observed token count cannot be negative")
    if count == 0:
        return np.empty(0, dtype=np.uint64)
    rng = np.random.default_rng(np.random.SeedSequence([seed, 0x0B5E7E, domain]))
    lower = rng.integers(1, 1 << 56, size=count, dtype=np.uint64)
    if len(np.unique(lower)) != count:
        # A collision is astronomically unlikely at Meridia sizes.  Fail loudly rather
        # than make collision handling depend on row positions or truth identities.
        raise RuntimeError(
            "observed identifier draw collided; use a different world seed"
        )
    return (np.uint64((0x80 + domain) << 56) | lower).astype(np.uint64, copy=False)


def _table_length(table: dict[str, np.ndarray]) -> int:
    return len(next(iter(table.values())))


def _empty_state_like_terminal(history: dict, hospitals: dict) -> dict:
    """Allocate a reporting-time state large enough for every terminal truth entity."""
    initial = history["initial_state"]
    terminal = history["terminal_state"]
    state: dict[str, dict[str, np.ndarray]] = {}
    for table_name in ("person", "household", "establishment", "job", "encounter"):
        size = _table_length(terminal[table_name])
        table: dict[str, np.ndarray] = {}
        for name, values in terminal[table_name].items():
            output = np.zeros(size, dtype=values.dtype)
            if output.dtype.kind in "iu" and name in {
                "cell",
                "birth_tick",
                "occupation",
                "employment_type",
                "scheduled_end_tick",
                "service",
                "diagnosis_group",
                "outcome",
                "bed_number",
            }:
                output[:] = -1
            n_initial = len(initial[table_name][name])
            output[:n_initial] = initial[table_name][name]
            table[name] = output
        table["exists"] = np.zeros(size, dtype=np.bool_)
        table["exists"][: _table_length(initial[table_name])] = True
        table["visible_from_tick"] = np.full(size, -1, dtype=np.int64)
        table["visible_from_tick"][: _table_length(initial[table_name])] = int(
            history["snapshot_tick"]
        )
        state[table_name] = table

    n_encounters = _table_length(terminal["encounter"])
    state["encounter"]["admission_tick"] = np.full(n_encounters, -1, dtype=np.int64)
    state["encounter"]["discharge_tick"] = np.full(n_encounters, -1, dtype=np.int64)
    n_initial_encounters = _table_length(initial["encounter"])
    state["encounter"]["admission_tick"][:n_initial_encounters] = int(
        history["snapshot_tick"]
    )
    initially_closed = ~initial["encounter"]["is_open"]
    state["encounter"]["discharge_tick"][:n_initial_encounters][initially_closed] = int(
        history["snapshot_tick"]
    )

    # Hospital cells are immutable in v0 and live outside the replay state.
    state["hospital_cell"] = np.asarray(hospitals["hospital"]["cell"], dtype=np.int64)
    return state


def _recorded_state(history: dict, hospitals: dict, snapshot_tick: int) -> dict:
    """Apply only events visible by recorded_tick, tolerating delayed ID gaps."""
    state = _empty_state_like_terminal(history, hospitals)
    event = history["event"]
    visible = event["recorded_tick"] <= snapshot_tick
    visible_corrections = event["supersedes_event_id"][visible]
    superseded = visible_corrections[visible_corrections != 0]
    if len(superseded):
        visible &= ~np.isin(event["truth_event_id"], superseded)

    for row in np.flatnonzero(visible):
        event_type = int(event["event_type"][row])
        recorded_tick = int(event["recorded_tick"][row])
        if event_type == EVENT_TYPES["person_birth"]:
            position = int(
                _sequence_position(event["truth_person_id"][row : row + 1])[0]
            )
            person = state["person"]
            person["exists"][position] = True
            person["visible_from_tick"][position] = recorded_tick
            for name in (
                "truth_person_id",
                "truth_household_id",
                "birth_tick",
                "sex",
                "role",
                "education",
                "income_cents",
            ):
                person[name][position] = event[name][row]
            person["is_alive"][position] = True
        elif event_type == EVENT_TYPES["person_death"]:
            position = int(
                _sequence_position(event["truth_person_id"][row : row + 1])[0]
            )
            state["person"]["is_alive"][position] = False
        elif event_type == EVENT_TYPES["household_formed"]:
            position = int(
                _sequence_position(event["truth_household_id"][row : row + 1])[0]
            )
            person_position = int(
                _sequence_position(event["truth_person_id"][row : row + 1])[0]
            )
            household = state["household"]
            household["exists"][position] = True
            household["visible_from_tick"][position] = recorded_tick
            household["truth_household_id"][position] = event["truth_household_id"][row]
            household["truth_dwelling_id"][position] = event["truth_dwelling_id"][row]
            household["cell"][position] = event["to_cell"][row]
            household["is_active"][position] = True
            state["person"]["truth_household_id"][person_position] = event[
                "truth_household_id"
            ][row]
            state["person"]["role"][person_position] = 0
        elif event_type == EVENT_TYPES["household_moved"]:
            position = int(
                _sequence_position(event["truth_household_id"][row : row + 1])[0]
            )
            state["household"]["truth_dwelling_id"][position] = event[
                "truth_dwelling_id"
            ][row]
            state["household"]["cell"][position] = event["to_cell"][row]
        elif event_type == EVENT_TYPES["household_closed"]:
            position = int(
                _sequence_position(event["truth_household_id"][row : row + 1])[0]
            )
            state["household"]["is_active"][position] = False
        elif event_type == EVENT_TYPES["establishment_opened"]:
            position = int(
                _sequence_position(event["truth_establishment_id"][row : row + 1])[0]
            )
            establishment = state["establishment"]
            establishment["exists"][position] = True
            establishment["visible_from_tick"][position] = recorded_tick
            for name in (
                "truth_establishment_id",
                "truth_enterprise_id",
                "industry",
            ):
                establishment[name][position] = event[name][row]
            establishment["cell"][position] = event["to_cell"][row]
            establishment["is_hospital"][position] = False
            establishment["is_active"][position] = True
        elif event_type == EVENT_TYPES["establishment_closed"]:
            position = int(
                _sequence_position(event["truth_establishment_id"][row : row + 1])[0]
            )
            state["establishment"]["is_active"][position] = False
        elif event_type == EVENT_TYPES["job_started"]:
            position = int(_sequence_position(event["truth_job_id"][row : row + 1])[0])
            job = state["job"]
            job["exists"][position] = True
            job["visible_from_tick"][position] = recorded_tick
            for name in (
                "truth_job_id",
                "truth_person_id",
                "truth_establishment_id",
                "occupation",
                "employment_type",
                "annual_hours",
                "hourly_wage_cents",
            ):
                job[name][position] = event[name][row]
            job["annual_earnings_cents"][position] = int(
                event["annual_hours"][row]
            ) * int(event["hourly_wage_cents"][row])
            job["is_active"][position] = True
        elif event_type == EVENT_TYPES["job_ended"]:
            position = int(_sequence_position(event["truth_job_id"][row : row + 1])[0])
            state["job"]["is_active"][position] = False
        elif event_type == EVENT_TYPES["encounter_admitted"]:
            position = int(
                _sequence_position(event["truth_encounter_id"][row : row + 1])[0]
            )
            encounter = state["encounter"]
            encounter["exists"][position] = True
            encounter["visible_from_tick"][position] = recorded_tick
            for name in (
                "truth_encounter_id",
                "truth_person_id",
                "truth_hospital_id",
                "scheduled_end_tick",
                "service",
                "diagnosis_group",
                "outcome",
                "cost_cents",
                "bed_number",
            ):
                encounter[name][position] = event[name][row]
            encounter["is_open"][position] = True
            encounter["admission_tick"][position] = int(event["tick"][row])
        elif event_type == EVENT_TYPES["encounter_discharged"]:
            position = int(
                _sequence_position(event["truth_encounter_id"][row : row + 1])[0]
            )
            encounter = state["encounter"]
            encounter["is_open"][position] = False
            encounter["outcome"][position] = event["outcome"][row]
            encounter["bed_number"][position] = -1
            encounter["discharge_tick"][position] = int(event["tick"][row])
        else:
            raise ValueError(f"unknown institutional event type {event_type}")

    person = state["person"]
    household_position = _sequence_position(person["truth_household_id"])
    valid_person = person["exists"] & (household_position >= 0)
    person["cell"][valid_person] = state["household"]["cell"][
        household_position[valid_person]
    ]
    return state


def _validate_inputs(
    history: dict,
    seed: int,
    admin: dict,
    hospitals: dict,
    preliminary_tick: int | None,
    revised_tick: int | None,
) -> tuple[int, int, np.ndarray, int]:
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise TypeError("seed must be an integer")
    seed = int(seed)
    try:
        generator_version = int(history["generator_version"])
        world_id = np.uint64(history["truth_world_id"])
        snapshot_tick = int(history["snapshot_tick"])
        terminal_tick = int(history["terminal_tick"])
        county = np.asarray(admin["county"])
        n_counties = int(admin["n_counties"])
        county_is_outpost = np.asarray(admin["county_is_outpost"])
        hospital_world_id = np.uint64(hospitals["truth_world_id"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "source inputs do not satisfy the retained-world schema"
        ) from exc
    if seed < 0:
        raise ValueError("seed must be nonnegative")
    expected_world_id = truth_world_id(seed, generator_version)
    if world_id != expected_world_id or hospital_world_id != expected_world_id:
        raise ValueError("seed does not match the source inputs' truth world")
    if county.ndim != 2 or county.dtype != np.int64:
        raise ValueError("admin county must be a two-dimensional int64 array")
    if n_counties < 1 or county_is_outpost.shape != (n_counties,):
        raise ValueError("admin county metadata is inconsistent")
    if county_is_outpost.dtype != np.bool_:
        raise ValueError("admin county_is_outpost must be boolean")
    on_land = county >= 0
    if not on_land.any() or int(county[on_land].max()) >= n_counties:
        raise ValueError("admin county codes are outside the declared range")
    if set(np.unique(county[on_land]).tolist()) != set(range(n_counties)):
        raise ValueError("admin county codes are not contiguous")

    if preliminary_tick is None:
        preliminary_tick = max(snapshot_tick, terminal_tick - 3)
    if revised_tick is None:
        revised_tick = terminal_tick
    for name, value in (
        ("preliminary_tick", preliminary_tick),
        ("revised_tick", revised_tick),
    ):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise TypeError(f"{name} must be an integer")
    preliminary_tick = int(preliminary_tick)
    revised_tick = int(revised_tick)
    if not snapshot_tick <= preliminary_tick < revised_tick <= terminal_tick:
        raise ValueError(
            "source snapshots must satisfy snapshot <= preliminary < revised <= terminal"
        )

    county_flat = county.reshape(-1)
    terminal = history["terminal_state"]
    cells = [
        np.asarray(terminal["person"]["cell"], dtype=np.int64),
        np.asarray(terminal["household"]["cell"], dtype=np.int64),
        np.asarray(terminal["establishment"]["cell"], dtype=np.int64),
        np.asarray(hospitals["hospital"]["cell"], dtype=np.int64),
    ]
    for values in cells:
        if values.ndim != 1 or (
            len(values) and (values.min() < 0 or values.max() >= len(county_flat))
        ):
            raise ValueError("institutional cell is outside the administrative grid")
        if len(values) and np.any(county_flat[values] < 0):
            raise ValueError("institutional cell has no county code")
    return preliminary_tick, revised_tick, county_flat, n_counties


def _employment_summary(
    state: dict, n_persons: int, n_establishments: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    job = state["job"]
    active = np.asarray(job["is_active"], dtype=np.bool_)
    if "exists" in job:
        active &= job["exists"]
    active_position = np.flatnonzero(active)
    earnings = np.zeros(n_persons, dtype=np.int64)
    employer = np.full(n_persons, -1, dtype=np.int64)
    employment = np.zeros(n_establishments, dtype=np.int32)
    payroll = np.zeros(n_establishments, dtype=np.int64)
    if len(active_position):
        person_position = _sequence_position(job["truth_person_id"][active_position])
        establishment_position = _sequence_position(
            job["truth_establishment_id"][active_position]
        )
        values = job["annual_earnings_cents"][active_position].astype(np.int64)
        np.add.at(earnings, person_position, values)
        employer[person_position] = establishment_position
        np.add.at(employment, establishment_position, 1)
        np.add.at(payroll, establishment_position, values)
    return earnings, employer, employment, payroll


def _padded_exact_state(history: dict, snapshot_tick: int) -> dict:
    exact = replay_event_history(history, through_tick=snapshot_tick)
    terminal = history["terminal_state"]
    padded: dict[str, dict[str, np.ndarray]] = {}
    for table_name in ("person", "household", "establishment", "job", "encounter"):
        terminal_size = _table_length(terminal[table_name])
        table: dict[str, np.ndarray] = {}
        for name, values in terminal[table_name].items():
            output = np.zeros(terminal_size, dtype=values.dtype)
            if output.dtype.kind in "iu" and name in {
                "cell",
                "birth_tick",
                "occupation",
                "employment_type",
                "scheduled_end_tick",
                "service",
                "diagnosis_group",
                "outcome",
                "bed_number",
            }:
                output[:] = -1
            output[: len(exact[table_name][name])] = exact[table_name][name]
            table[name] = output
        table["exists"] = np.zeros(terminal_size, dtype=np.bool_)
        table["exists"][: _table_length(exact[table_name])] = True
        padded[table_name] = table
    return padded


def _stale_flags(recorded: dict, exact: dict) -> dict[str, np.ndarray]:
    person = recorded["person"]
    exact_person = exact["person"]
    population_stale = person["exists"] & (
        ~exact_person["exists"]
        | (person["is_alive"] != exact_person["is_alive"])
        | (person["truth_household_id"] != exact_person["truth_household_id"])
        | (person["cell"] != exact_person["cell"])
    )

    establishment = recorded["establishment"]
    exact_establishment = exact["establishment"]
    business_stale = establishment["exists"] & (
        ~exact_establishment["exists"]
        | (establishment["is_active"] != exact_establishment["is_active"])
        | (establishment["cell"] != exact_establishment["cell"])
    )

    n_persons = len(person["truth_person_id"])
    n_establishments = len(establishment["truth_establishment_id"])
    (
        recorded_earnings,
        recorded_employer,
        recorded_employment,
        recorded_payroll,
    ) = _employment_summary(recorded, n_persons, n_establishments)
    exact_earnings, exact_employer, exact_employment, exact_payroll = (
        _employment_summary(exact, n_persons, n_establishments)
    )
    business_stale |= (recorded_employment != exact_employment) | (
        recorded_payroll != exact_payroll
    )
    income_stale = (
        population_stale
        | (recorded_earnings != exact_earnings)
        | (recorded_employer != exact_employer)
    )

    encounter = recorded["encounter"]
    exact_encounter = exact["encounter"]
    encounter_person_position = _sequence_position(encounter["truth_person_id"])
    valid_encounter = encounter["exists"] & (encounter_person_position >= 0)
    patient_address_stale = np.zeros(
        len(encounter["truth_encounter_id"]), dtype=np.bool_
    )
    patient_address_stale[valid_encounter] = (
        recorded["person"]["cell"][encounter_person_position[valid_encounter]]
        != exact["person"]["cell"][encounter_person_position[valid_encounter]]
    )
    health_stale = encounter["exists"] & (
        ~exact_encounter["exists"]
        | (encounter["is_open"] != exact_encounter["is_open"])
        | (encounter["outcome"] != exact_encounter["outcome"])
        | patient_address_stale
    )
    return {
        "population": population_stale,
        "business": business_stale,
        "income": income_stale,
        "health": health_stale,
    }


def _mechanism_plan(
    seed: int,
    source_index: int,
    truth_ids: np.ndarray,
    county: np.ndarray,
    county_is_outpost: np.ndarray,
    coverage: float,
    params: SourceParams,
    token_domains: tuple[int, int, int],
) -> dict:
    count = len(truth_ids)
    rng = np.random.default_rng(np.random.SeedSequence([seed, 0xAE61A7E, source_index]))
    probability = np.full(count, coverage, dtype=np.float64)
    if count:
        probability -= params.outpost_coverage_penalty * county_is_outpost[county]
    covered = rng.random(count) < probability
    split = rng.random(count) < params.split_rate
    duplicate = (rng.random(count) < params.duplicate_rate) | split
    county_error = rng.random(count) < params.county_error_rate
    if len(county_is_outpost) == 1:
        county_error[:] = False
    linkage_error = rng.random(count) < params.linkage_error_rate
    item_missing = rng.random(count) < params.item_missing_rate

    primary_id = _random_tokens(seed, token_domains[1], count)
    secondary_id = _random_tokens(seed, token_domains[2], count)
    record_id = _random_tokens(seed, token_domains[0], 2 * count).reshape(count, 2)
    merge_group = np.full(count, -1, dtype=np.int64)
    merge_candidate = np.flatnonzero(rng.random(count) < params.merge_rate)
    if len(merge_candidate) % 2:
        merge_candidate = merge_candidate[:-1]
    for group, pair in enumerate(merge_candidate.reshape(-1, 2)):
        first, second = int(pair[0]), int(pair[1])
        primary_id[second] = primary_id[first]
        merge_group[first] = merge_group[second] = group

    return {
        "truth_entity_id": np.asarray(truth_ids, dtype=np.uint64).copy(),
        "covered": covered.astype(np.bool_),
        "duplicate": duplicate.astype(np.bool_),
        "split": split.astype(np.bool_),
        "merge_group": merge_group,
        "county_error": county_error.astype(np.bool_),
        "linkage_error": linkage_error.astype(np.bool_),
        "item_missing": item_missing.astype(np.bool_),
        "primary_id": primary_id,
        "secondary_id": secondary_id,
        "record_id": record_id,
    }


def _mechanism_table(plan: dict) -> dict[str, np.ndarray]:
    return {name: np.asarray(plan[name]).copy() for name in _MECHANISM_SCHEMA}


def _expand_rows(plan: dict, active: np.ndarray, stale: np.ndarray) -> dict:
    base_position = np.flatnonzero(np.asarray(active, dtype=np.bool_) & plan["covered"])
    count = 1 + plan["duplicate"][base_position].astype(np.int64)
    position = np.repeat(base_position, count)
    if len(position):
        starts = np.repeat(np.cumsum(count) - count, count)
        slot = np.arange(len(position), dtype=np.int64) - starts
    else:
        slot = np.empty(0, dtype=np.int64)
    record_id = plan["record_id"][position, slot]
    entity_id = plan["primary_id"][position].copy()
    use_secondary = (slot == 1) & plan["split"][position]
    entity_id[use_secondary] = plan["secondary_id"][position[use_secondary]]

    mechanism_code = np.zeros(len(position), dtype=np.int16)
    for name in (
        "duplicate",
        "split",
        "county_error",
        "linkage_error",
        "item_missing",
    ):
        mechanism_code |= np.where(
            plan[name][position], MECHANISM_BITS[name], 0
        ).astype(np.int16)
    mechanism_code |= np.where(
        plan["merge_group"][position] >= 0, MECHANISM_BITS["merged"], 0
    ).astype(np.int16)
    mechanism_code |= np.where(stale[position], MECHANISM_BITS["stale"], 0).astype(
        np.int16
    )
    return {
        "position": position,
        "slot": slot,
        "record_id": record_id,
        "entity_id": entity_id,
        "mechanism_code": mechanism_code,
    }


def _county_values(
    true_county: np.ndarray, rows: dict, plan: dict, n_counties: int
) -> np.ndarray:
    position = rows["position"]
    county = np.asarray(true_county[position], dtype=np.int32).copy()
    error = plan["county_error"][position]
    if error.any():
        offset = (
            plan["primary_id"][position[error]] % np.uint64(n_counties - 1) + 1
        ).astype(np.int32)
        county[error] = (county[error] + offset) % n_counties
    return county


def _name_values(
    base_name: np.ndarray, rows: dict, plan: dict, source_index: int
) -> np.ndarray:
    position = rows["position"]
    observed = np.asarray(base_name[position], dtype=np.uint64).copy()
    error = plan["linkage_error"][position]
    if error.any():
        observed[error] ^= np.uint64((source_index + 1) * 0x9E3779)
    return observed


def _sort_table(
    table: dict[str, np.ndarray], order: np.ndarray
) -> dict[str, np.ndarray]:
    return {name: np.asarray(values)[order].copy() for name, values in table.items()}


def _crosswalk(
    rows: dict,
    plan: dict,
    visible_from_tick: np.ndarray,
) -> dict[str, np.ndarray]:
    position = rows["position"]
    table = {
        "observed_record_id": rows["record_id"].astype(np.uint64, copy=False),
        "observed_entity_id": rows["entity_id"].astype(np.uint64, copy=False),
        "truth_entity_id": plan["truth_entity_id"][position],
        "mechanism_code": rows["mechanism_code"].astype(np.int16, copy=False),
        "valid_from_tick": visible_from_tick[position].astype(np.int64, copy=False),
        "valid_to_tick": np.full(len(position), -1, dtype=np.int64),
    }
    order = np.argsort(table["observed_record_id"], kind="stable")
    return _sort_table(table, order)


def _population_source(
    state: dict,
    plan: dict,
    stale: np.ndarray,
    county_flat: np.ndarray,
    n_counties: int,
    household_id: np.ndarray,
    base_name: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    person = state["person"]
    active = person["exists"] & person["is_alive"]
    rows = _expand_rows(plan, active, stale)
    position = rows["position"]
    household_position = _sequence_position(person["truth_household_id"][position])
    true_county = county_flat[person["cell"]]
    education = person["education"][position].astype(np.int8, copy=True)
    education[plan["item_missing"][position]] = -1
    table = {
        "record_id": rows["record_id"],
        "person_id": rows["entity_id"],
        "household_id": household_id[household_position],
        "name_code": _name_values(base_name, rows, plan, 0),
        "birth_tick": person["birth_tick"][position].astype(np.int64, copy=False),
        "sex": person["sex"][position].astype(np.int8, copy=False),
        "education": education,
        "county": _county_values(true_county, rows, plan, n_counties),
    }
    order = np.argsort(table["record_id"], kind="stable")
    return _sort_table(table, order), _crosswalk(
        rows, plan, person["visible_from_tick"]
    )


def _business_source(
    state: dict,
    plan: dict,
    stale: np.ndarray,
    county_flat: np.ndarray,
    n_counties: int,
    enterprise_id: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    establishment = state["establishment"]
    active = establishment["exists"] & establishment["is_active"]
    rows = _expand_rows(plan, active, stale)
    position = rows["position"]
    _, _, employment, payroll = _employment_summary(
        state,
        len(state["person"]["truth_person_id"]),
        len(establishment["truth_establishment_id"]),
    )
    true_county = county_flat[establishment["cell"]]
    enterprise_position = _sequence_position(
        establishment["truth_enterprise_id"][position]
    )
    payroll_value = payroll[position].astype(np.float64)
    payroll_value[plan["item_missing"][position]] = np.nan
    table = {
        "record_id": rows["record_id"],
        "business_id": rows["entity_id"],
        "enterprise_id": enterprise_id[enterprise_position],
        "industry": establishment["industry"][position].astype(np.int16, copy=False),
        "county": _county_values(true_county, rows, plan, n_counties),
        "employee_count": employment[position].astype(np.int32, copy=False),
        "annual_payroll_cents": payroll_value,
    }
    order = np.argsort(table["record_id"], kind="stable")
    return _sort_table(table, order), _crosswalk(
        rows, plan, establishment["visible_from_tick"]
    )


def _income_source(
    state: dict,
    plan: dict,
    stale: np.ndarray,
    county_flat: np.ndarray,
    n_counties: int,
    household_id: np.ndarray,
    base_name: np.ndarray,
    business_plan: dict,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    person = state["person"]
    active = person["exists"] & person["is_alive"]
    rows = _expand_rows(plan, active, stale)
    position = rows["position"]
    earnings, employer_position, _, _ = _employment_summary(
        state,
        len(person["truth_person_id"]),
        len(state["establishment"]["truth_establishment_id"]),
    )
    household_position = _sequence_position(person["truth_household_id"][position])
    true_county = county_flat[person["cell"]]
    income = earnings[position].astype(np.float64)
    missing = plan["item_missing"][position]
    income[missing] = np.nan
    selected_employer_position = employer_position[position]
    employer_id = np.zeros(len(position), dtype=np.uint64)
    employed = selected_employer_position >= 0
    employer_id[employed] = business_plan["primary_id"][
        selected_employer_position[employed]
    ]
    wrong_link = employed & plan["linkage_error"][position]
    if wrong_link.any():
        employer_id[wrong_link] = business_plan["primary_id"][
            (selected_employer_position[wrong_link] + 1)
            % len(business_plan["primary_id"])
        ]
    employer_id[missing] = 0
    table = {
        "record_id": rows["record_id"],
        "taxpayer_id": rows["entity_id"],
        "household_id": household_id[household_position],
        "name_code": _name_values(base_name, rows, plan, 2),
        "birth_tick": person["birth_tick"][position].astype(np.int64, copy=False),
        "sex": person["sex"][position].astype(np.int8, copy=False),
        "county": _county_values(true_county, rows, plan, n_counties),
        "employment_income_cents": income,
        "employer_id": employer_id,
    }
    order = np.argsort(table["record_id"], kind="stable")
    return _sort_table(table, order), _crosswalk(
        rows, plan, person["visible_from_tick"]
    )


def _health_source(
    state: dict,
    plan: dict,
    stale: np.ndarray,
    county_flat: np.ndarray,
    n_counties: int,
    base_name: np.ndarray,
    patient_id: np.ndarray,
    facility_id: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    encounter = state["encounter"]
    rows = _expand_rows(plan, encounter["exists"], stale)
    position = rows["position"]
    person_position = _sequence_position(encounter["truth_person_id"][position])
    hospital_position = _sequence_position(encounter["truth_hospital_id"][position])
    person = state["person"]
    all_person_position = _sequence_position(encounter["truth_person_id"])
    patient_county = county_flat[person["cell"][all_person_position]]
    facility_county = county_flat[state["hospital_cell"]]

    observed_patient = patient_id[person_position].copy()
    wrong_link = plan["linkage_error"][position]
    if wrong_link.any():
        observed_patient[wrong_link] = patient_id[
            (person_position[wrong_link] + 1) % len(patient_id)
        ]
    observed_name = base_name[person_position].copy()
    observed_name[wrong_link] ^= np.uint64(4 * 0x9E3779)
    cost = encounter["cost_cents"][position].astype(np.float64)
    cost[plan["item_missing"][position]] = np.nan
    table = {
        "record_id": rows["record_id"],
        "encounter_id": rows["entity_id"],
        "patient_id": observed_patient,
        "facility_id": facility_id[hospital_position],
        "name_code": observed_name,
        "birth_tick": person["birth_tick"][person_position].astype(
            np.int64, copy=False
        ),
        "sex": person["sex"][person_position].astype(np.int8, copy=False),
        "patient_county": _county_values(patient_county, rows, plan, n_counties),
        "facility_county": facility_county[hospital_position].astype(
            np.int32, copy=False
        ),
        "admission_tick": encounter["admission_tick"][position].astype(
            np.int64, copy=False
        ),
        "discharge_tick": encounter["discharge_tick"][position].astype(
            np.int64, copy=False
        ),
        "service": encounter["service"][position].astype(np.int8, copy=False),
        "diagnosis_group": encounter["diagnosis_group"][position].astype(
            np.int16, copy=False
        ),
        "outcome": encounter["outcome"][position].astype(np.int8, copy=False),
        "cost_cents": cost,
    }
    order = np.argsort(table["record_id"], kind="stable")
    return _sort_table(table, order), _crosswalk(
        rows, plan, encounter["visible_from_tick"]
    )


def _build_observed_sources_unchecked(
    history: dict,
    seed: int,
    admin: dict,
    hospitals: dict,
    preliminary_tick: int,
    revised_tick: int,
    params: SourceParams,
) -> dict:
    county_flat = np.asarray(admin["county"], dtype=np.int64).reshape(-1)
    n_counties = int(admin["n_counties"])
    county_is_outpost = np.asarray(admin["county_is_outpost"], dtype=np.bool_)
    terminal = history["terminal_state"]
    person = terminal["person"]
    establishment = terminal["establishment"]
    encounter = terminal["encounter"]

    person_county = county_flat[person["cell"]]
    establishment_county = county_flat[establishment["cell"]]
    terminal_hospital_position = _sequence_position(encounter["truth_hospital_id"])
    encounter_county = county_flat[
        np.asarray(hospitals["hospital"]["cell"], dtype=np.int64)[
            terminal_hospital_position
        ]
    ]
    coverage = {
        "population": params.population_coverage,
        "business": params.business_coverage,
        "income": params.income_coverage,
        "health": params.health_coverage,
    }
    truth_ids = {
        "population": person["truth_person_id"],
        "business": establishment["truth_establishment_id"],
        "income": person["truth_person_id"],
        "health": encounter["truth_encounter_id"],
    }
    counties = {
        "population": person_county,
        "business": establishment_county,
        "income": person_county,
        "health": encounter_county,
    }
    domains = {
        "population": (0, 1, 2),
        "business": (3, 4, 5),
        "income": (6, 7, 8),
        "health": (9, 10, 11),
    }
    plans = {
        source: _mechanism_plan(
            seed,
            source_index,
            truth_ids[source],
            counties[source],
            county_is_outpost,
            coverage[source],
            params,
            domains[source],
        )
        for source_index, source in enumerate(OBSERVED_SOURCES)
    }

    n_households = _table_length(terminal["household"])
    enterprise_position = _sequence_position(establishment["truth_enterprise_id"])
    n_enterprises = int(enterprise_position.max()) + 1
    n_hospitals = len(hospitals["hospital"]["truth_hospital_id"])
    base_name = _random_tokens(seed, 12, _table_length(person))
    population_household_id = _random_tokens(seed, 13, n_households)
    income_household_id = _random_tokens(seed, 14, n_households)
    enterprise_id = _random_tokens(seed, 15, n_enterprises)
    patient_id = _random_tokens(seed, 16, _table_length(person))
    facility_id = _random_tokens(seed, 17, n_hospitals)

    public_snapshots: dict[str, dict] = {}
    crosswalks: dict[str, dict] = {}
    for label, snapshot_tick in (
        ("preliminary", preliminary_tick),
        ("revised", revised_tick),
    ):
        recorded = _recorded_state(history, hospitals, snapshot_tick)
        exact = _padded_exact_state(history, snapshot_tick)
        stale = _stale_flags(recorded, exact)
        population_table, population_crosswalk = _population_source(
            recorded,
            plans["population"],
            stale["population"],
            county_flat,
            n_counties,
            population_household_id,
            base_name,
        )
        business_table, business_crosswalk = _business_source(
            recorded,
            plans["business"],
            stale["business"],
            county_flat,
            n_counties,
            enterprise_id,
        )
        income_table, income_crosswalk = _income_source(
            recorded,
            plans["income"],
            stale["income"],
            county_flat,
            n_counties,
            income_household_id,
            base_name,
            plans["business"],
        )
        health_table, health_crosswalk = _health_source(
            recorded,
            plans["health"],
            stale["health"],
            county_flat,
            n_counties,
            base_name,
            patient_id,
            facility_id,
        )
        public_snapshots[label] = {
            "snapshot_tick": np.int64(snapshot_tick),
            "population": population_table,
            "business": business_table,
            "income": income_table,
            "health": health_table,
        }
        crosswalks[label] = {
            "population": population_crosswalk,
            "business": business_crosswalk,
            "income": income_crosswalk,
            "health": health_crosswalk,
        }

    return {
        "truth_world_id": np.uint64(history["truth_world_id"]),
        "generator_version": int(history["generator_version"]),
        "source_schema_version": 1,
        "preliminary_tick": np.int64(preliminary_tick),
        "revised_tick": np.int64(revised_tick),
        "source_params": _params_record(params),
        "public_snapshots": public_snapshots,
        "hidden": {
            "mechanisms": {
                source: _mechanism_table(plans[source]) for source in OBSERVED_SOURCES
            },
            "crosswalks": crosswalks,
        },
    }


def _validate_table(
    table: dict,
    schema: dict[str, np.dtype],
    table_name: str,
) -> int:
    if set(table) != set(schema):
        missing = sorted(set(schema) - set(table))
        extra = sorted(set(table) - set(schema))
        raise ValueError(
            f"{table_name} columns differ from schema; missing={missing}, extra={extra}"
        )
    lengths: set[int] = set()
    for name, dtype in schema.items():
        values = np.asarray(table[name])
        if values.ndim != 1:
            raise ValueError(f"{table_name}.{name} is not one-dimensional")
        if values.dtype != dtype:
            raise ValueError(
                f"{table_name}.{name} has dtype {values.dtype}, expected {dtype}"
            )
        lengths.add(len(values))
    if len(lengths) != 1:
        raise ValueError(f"{table_name} columns have different row counts")
    return next(iter(lengths))


def _validate_structure(package: dict, history: dict, admin: dict) -> None:
    expected_top = {
        "truth_world_id",
        "generator_version",
        "source_schema_version",
        "preliminary_tick",
        "revised_tick",
        "source_params",
        "public_snapshots",
        "hidden",
    }
    if set(package) != expected_top:
        raise ValueError("source package top-level fields differ from schema")
    if int(package["source_schema_version"]) != 1:
        raise ValueError("unsupported source schema version")
    if set(package["public_snapshots"]) != {"preliminary", "revised"}:
        raise ValueError("source package must contain exactly two public snapshots")
    hidden = package["hidden"]
    if set(hidden) != {"mechanisms", "crosswalks"}:
        raise ValueError("source hidden package fields differ from schema")
    if set(hidden["mechanisms"]) != set(OBSERVED_SOURCES):
        raise ValueError("source mechanism sources differ from schema")
    if set(hidden["crosswalks"]) != {"preliminary", "revised"}:
        raise ValueError("source crosswalk snapshots differ from schema")

    terminal = history["terminal_state"]
    expected_truth = {
        "population": terminal["person"]["truth_person_id"],
        "business": terminal["establishment"]["truth_establishment_id"],
        "income": terminal["person"]["truth_person_id"],
        "health": terminal["encounter"]["truth_encounter_id"],
    }
    for source in OBSERVED_SOURCES:
        mechanism = hidden["mechanisms"][source]
        _validate_table(mechanism, _MECHANISM_SCHEMA, f"hidden.mechanisms.{source}")
        if not np.array_equal(mechanism["truth_entity_id"], expected_truth[source]):
            raise ValueError(f"{source} mechanism truth identities differ from history")
        if np.any(mechanism["split"] & ~mechanism["duplicate"]):
            raise ValueError(f"{source} split identity lacks a second observed record")
        group = mechanism["merge_group"]
        nonnegative = group[group >= 0]
        if len(nonnegative):
            _, group_count = np.unique(nonnegative, return_counts=True)
            if np.any(group_count != 2):
                raise ValueError(f"{source} merge groups are not truth-entity pairs")

    n_counties = int(admin["n_counties"])
    entity_column = {
        "population": "person_id",
        "business": "business_id",
        "income": "taxpayer_id",
        "health": "encounter_id",
    }
    for label, declared_tick in (
        ("preliminary", int(package["preliminary_tick"])),
        ("revised", int(package["revised_tick"])),
    ):
        snapshot = package["public_snapshots"][label]
        if set(snapshot) != {"snapshot_tick", *OBSERVED_SOURCES}:
            raise ValueError(f"{label} public snapshot fields differ from schema")
        if int(snapshot["snapshot_tick"]) != declared_tick:
            raise ValueError(f"{label} snapshot tick differs from retained metadata")
        if set(hidden["crosswalks"][label]) != set(OBSERVED_SOURCES):
            raise ValueError(f"{label} crosswalk sources differ from schema")
        for source in OBSERVED_SOURCES:
            table = snapshot[source]
            n_rows = _validate_table(
                table, PUBLIC_SCHEMAS[source], f"public_snapshots.{label}.{source}"
            )
            record_id = table["record_id"]
            if n_rows and (
                np.any(record_id == 0)
                or len(np.unique(record_id)) != n_rows
                or not np.array_equal(record_id, np.sort(record_id, kind="stable"))
            ):
                raise ValueError(
                    f"{label} {source} record IDs are not unique and canonical"
                )
            for name in table:
                lowered = name.lower()
                if (
                    lowered.startswith("truth_")
                    or "mechanism" in lowered
                    or "seed" in lowered
                    or "regime" in lowered
                    or "crosswalk" in lowered
                ):
                    raise ValueError(f"participant-facing column {name!r} is forbidden")
            county_columns = (
                ("patient_county", "facility_county")
                if source == "health"
                else ("county",)
            )
            for county_name in county_columns:
                county = table[county_name]
                if np.any(county < 0) or np.any(county >= n_counties):
                    raise ValueError(f"{label} {source} has an invalid county code")

            crosswalk = hidden["crosswalks"][label][source]
            crosswalk_rows = _validate_table(
                crosswalk, _CROSSWALK_SCHEMA, f"hidden.crosswalks.{label}.{source}"
            )
            if crosswalk_rows != n_rows:
                raise ValueError(f"{label} {source} crosswalk row count differs")
            if not np.array_equal(crosswalk["observed_record_id"], record_id):
                raise ValueError(
                    f"{label} {source} crosswalk does not match record IDs"
                )
            if not np.array_equal(
                crosswalk["observed_entity_id"], table[entity_column[source]]
            ):
                raise ValueError(
                    f"{label} {source} crosswalk does not match entity IDs"
                )
            if np.any(crosswalk["valid_from_tick"] > declared_tick) or np.any(
                crosswalk["valid_to_tick"] != -1
            ):
                raise ValueError(f"{label} {source} crosswalk validity is impossible")
            truth_position = _sequence_position(crosswalk["truth_entity_id"])
            if np.any(truth_position < 0) or np.any(
                truth_position >= len(expected_truth[source])
            ):
                raise ValueError(f"{label} {source} crosswalk truth ID is out of range")
            if not np.array_equal(
                expected_truth[source][truth_position], crosswalk["truth_entity_id"]
            ):
                raise ValueError(f"{label} {source} crosswalk uses the wrong namespace")

    for source in OBSERVED_SOURCES:
        preliminary = package["public_snapshots"]["preliminary"][source]
        revised = package["public_snapshots"]["revised"][source]
        common, preliminary_position, revised_position = np.intersect1d(
            preliminary["record_id"],
            revised["record_id"],
            assume_unique=True,
            return_indices=True,
        )
        if len(common) and not np.array_equal(
            preliminary[entity_column[source]][preliminary_position],
            revised[entity_column[source]][revised_position],
        ):
            raise ValueError(f"{source} observed entity IDs change between snapshots")


def _assert_equal(left, right, path: str = "source_package") -> None:
    if isinstance(left, dict) and isinstance(right, dict):
        if set(left) != set(right):
            raise ValueError(f"{path} fields differ from deterministic regeneration")
        for name in left:
            _assert_equal(left[name], right[name], f"{path}.{name}")
        return
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        left_array = np.asarray(left)
        right_array = np.asarray(right)
        if (
            left_array.dtype != right_array.dtype
            or left_array.shape != right_array.shape
            or not np.array_equal(left_array, right_array, equal_nan=True)
        ):
            raise ValueError(f"{path} differs from deterministic regeneration")
        return
    if (
        isinstance(left, float)
        and isinstance(right, float)
        and np.isnan(left)
        and np.isnan(right)
    ):
        return
    if left != right:
        raise ValueError(f"{path} differs from deterministic regeneration")


def build_observed_sources(
    history: dict,
    seed: int,
    admin: dict,
    hospitals: dict,
    preliminary_tick: int | None = None,
    revised_tick: int | None = None,
    params: SourceParams = SourceParams(),
) -> dict:
    """Build two deterministic observed-source snapshots plus sealed evidence."""
    _validate_params(params)
    preliminary_tick, revised_tick, _, _ = _validate_inputs(
        history,
        seed,
        admin,
        hospitals,
        preliminary_tick,
        revised_tick,
    )
    package = _build_observed_sources_unchecked(
        history,
        int(seed),
        admin,
        hospitals,
        preliminary_tick,
        revised_tick,
        params,
    )
    _validate_structure(package, history, admin)
    return package


def validate_observed_sources(
    package: dict,
    history: dict,
    seed: int,
    admin: dict,
    hospitals: dict,
) -> None:
    """Fail unless a package is the exact seeded build from the retained inputs."""
    try:
        preliminary_tick = int(package["preliminary_tick"])
        revised_tick = int(package["revised_tick"])
        params = _params_from_record(package["source_params"])
        package_world_id = np.uint64(package["truth_world_id"])
        package_generator_version = int(package["generator_version"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("source package metadata is incomplete") from exc
    _validate_params(params)
    _validate_inputs(
        history,
        seed,
        admin,
        hospitals,
        preliminary_tick,
        revised_tick,
    )
    if package_world_id != np.uint64(history["truth_world_id"]):
        raise ValueError("source package belongs to a different truth world")
    if package_generator_version != int(history["generator_version"]):
        raise ValueError("source package uses a different generator version")
    _validate_structure(package, history, admin)
    expected = _build_observed_sources_unchecked(
        history,
        int(seed),
        admin,
        hospitals,
        preliminary_tick,
        revised_tick,
        params,
    )
    _assert_equal(package, expected)


def participant_source_snapshots(package: dict) -> dict[str, dict]:
    """Return a defensive copy of the two public bundles and no retained truth."""
    try:
        snapshots = package["public_snapshots"]
        if set(snapshots) != {"preliminary", "revised"}:
            raise ValueError
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("source package has no two-snapshot public bundle") from exc
    output: dict[str, dict] = {}
    for label in ("preliminary", "revised"):
        snapshot = snapshots[label]
        if set(snapshot) != {"snapshot_tick", *OBSERVED_SOURCES}:
            raise ValueError(f"{label} public snapshot fields differ from schema")
        output[label] = {"snapshot_tick": np.int64(snapshot["snapshot_tick"])}
        for source in OBSERVED_SOURCES:
            _validate_table(
                snapshot[source], PUBLIC_SCHEMAS[source], f"{label}.{source}"
            )
            output[label][source] = {
                name: np.asarray(values).copy()
                for name, values in snapshot[source].items()
            }
    return output
