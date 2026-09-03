"""Append-only institutional histories with exact deterministic replay.

The event ledger is retained truth.  ``tick`` records when a change took effect;
``recorded_tick`` records when that change became available to a downstream source.
The separation is what later observed-source vintages will use to create honest late
reporting and revision problems.  No observed identifier is created in this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np

from meridia.businesses import EMPLOYMENT_TYPES
from meridia.businesses import validate_business_conservation
from meridia.character import draw_world_character
from meridia.demography import (ANNUAL_SHOCK_RATE, draw_annual_shocks,
                                draw_world_shocks, mortality_probability)
from meridia.dwellings import validate_dwelling_conservation
from meridia.hospitals import ENCOUNTER_OUTCOMES, validate_hospital_conservation
from meridia.identities import ENTITY_NAMESPACE, NAMESPACE_SHIFT, SEQUENCE_MASK
from meridia.identities import entity_namespace, truth_entity_ids, truth_world_id
from meridia.mechanisms import (FRAILTY_RANGE, WorldMechanisms,
                                build_world_mechanisms,
                                death_report_late_probability, migration_age_pull,
                                newborn_frailty, quintile_band)

EVENT_TYPES: Final = {
    "person_birth": 1,
    "person_death": 2,
    "household_formed": 3,
    "household_moved": 4,
    "household_closed": 5,
    "job_started": 6,
    "job_ended": 7,
    "establishment_opened": 8,
    "establishment_closed": 9,
    "encounter_admitted": 10,
    "encounter_discharged": 11,
}

CAUSE_CODES: Final = {
    "demography": 1,
    "migration": 2,
    "labor_turnover": 3,
    "business_churn": 4,
    "health_need": 5,
    "world_shock": 6,
    "scheduled": 7,
}


@dataclass(frozen=True)
class EventHistoryParams:
    """Public mechanism ranges for the institutional timeline."""

    monthly_household_move_rate: float = 0.004
    monthly_job_turnover_rate: float = 0.012
    monthly_establishment_churn_rate: float = 0.002
    completed_encounter_share: float = 0.78
    late_report_probability: float = 0.22
    max_report_delay_months: int = 3
    minimum_work_age: int = 18
    maximum_work_age: int = 74


_EVENT_DTYPES: Final = {
    "truth_event_id": np.dtype(np.uint64),
    "tick": np.dtype(np.int64),
    "recorded_tick": np.dtype(np.int64),
    "entity_type": np.dtype(np.int8),
    "truth_entity_id": np.dtype(np.uint64),
    "event_type": np.dtype(np.int16),
    "supersedes_event_id": np.dtype(np.uint64),
    "cause_code": np.dtype(np.int16),
    "truth_person_id": np.dtype(np.uint64),
    "truth_household_id": np.dtype(np.uint64),
    "truth_prior_household_id": np.dtype(np.uint64),
    "truth_dwelling_id": np.dtype(np.uint64),
    "truth_prior_dwelling_id": np.dtype(np.uint64),
    "truth_enterprise_id": np.dtype(np.uint64),
    "truth_establishment_id": np.dtype(np.uint64),
    "truth_job_id": np.dtype(np.uint64),
    "truth_hospital_id": np.dtype(np.uint64),
    "truth_encounter_id": np.dtype(np.uint64),
    "from_cell": np.dtype(np.int64),
    "to_cell": np.dtype(np.int64),
    "birth_tick": np.dtype(np.int64),
    "sex": np.dtype(np.int8),
    "role": np.dtype(np.int8),
    "education": np.dtype(np.int8),
    "income_cents": np.dtype(np.int64),
    "industry": np.dtype(np.int16),
    "occupation": np.dtype(np.int16),
    "employment_type": np.dtype(np.int8),
    "annual_hours": np.dtype(np.int32),
    "hourly_wage_cents": np.dtype(np.int64),
    "scheduled_end_tick": np.dtype(np.int64),
    "service": np.dtype(np.int8),
    "diagnosis_group": np.dtype(np.int16),
    "outcome": np.dtype(np.int8),
    "cost_cents": np.dtype(np.int64),
    "bed_number": np.dtype(np.int32),
    "frailty_centi": np.dtype(np.int16),
}

_STATE_DTYPES: Final = {
    "person": {
        "truth_person_id": np.dtype(np.uint64),
        "truth_household_id": np.dtype(np.uint64),
        "cell": np.dtype(np.int64),
        "birth_tick": np.dtype(np.int64),
        "sex": np.dtype(np.int8),
        "role": np.dtype(np.int8),
        "education": np.dtype(np.int8),
        "income_cents": np.dtype(np.int64),
        "frailty_centi": np.dtype(np.int16),
        "is_alive": np.dtype(np.bool_),
    },
    "household": {
        "truth_household_id": np.dtype(np.uint64),
        "truth_dwelling_id": np.dtype(np.uint64),
        "cell": np.dtype(np.int64),
        "is_active": np.dtype(np.bool_),
    },
    "dwelling": {
        "truth_dwelling_id": np.dtype(np.uint64),
        "cell": np.dtype(np.int64),
        "truth_household_id": np.dtype(np.uint64),
        "resident_count": np.dtype(np.int32),
        "is_occupied": np.dtype(np.bool_),
    },
    "establishment": {
        "truth_establishment_id": np.dtype(np.uint64),
        "truth_enterprise_id": np.dtype(np.uint64),
        "cell": np.dtype(np.int64),
        "industry": np.dtype(np.int16),
        "is_hospital": np.dtype(np.bool_),
        "is_active": np.dtype(np.bool_),
    },
    "job": {
        "truth_job_id": np.dtype(np.uint64),
        "truth_person_id": np.dtype(np.uint64),
        "truth_establishment_id": np.dtype(np.uint64),
        "occupation": np.dtype(np.int16),
        "employment_type": np.dtype(np.int8),
        "annual_hours": np.dtype(np.int32),
        "hourly_wage_cents": np.dtype(np.int64),
        "annual_earnings_cents": np.dtype(np.int64),
        "is_active": np.dtype(np.bool_),
    },
    "encounter": {
        "truth_encounter_id": np.dtype(np.uint64),
        "truth_person_id": np.dtype(np.uint64),
        "truth_hospital_id": np.dtype(np.uint64),
        "scheduled_end_tick": np.dtype(np.int64),
        "service": np.dtype(np.int8),
        "diagnosis_group": np.dtype(np.int16),
        "outcome": np.dtype(np.int8),
        "cost_cents": np.dtype(np.int64),
        "bed_number": np.dtype(np.int32),
        "is_open": np.dtype(np.bool_),
    },
}


def _validate_params(params: EventHistoryParams) -> None:
    rates = (
        params.monthly_household_move_rate,
        params.monthly_job_turnover_rate,
        params.monthly_establishment_churn_rate,
        params.completed_encounter_share,
        params.late_report_probability,
    )
    if not np.isfinite(rates).all() or any(not 0.0 <= value <= 1.0 for value in rates):
        raise ValueError("event-history rates must be finite and in [0, 1]")
    if isinstance(params.max_report_delay_months, bool) or not isinstance(
        params.max_report_delay_months, (int, np.integer)
    ):
        raise TypeError("max_report_delay_months must be an integer")
    if params.max_report_delay_months < 1:
        raise ValueError("max_report_delay_months must be positive")
    if not 0 <= params.minimum_work_age <= params.maximum_work_age:
        raise ValueError("working-age bounds are invalid")


def _table_copy(table: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {name: np.asarray(values).copy() for name, values in table.items()}


def _state_copy(state: dict[str, dict[str, np.ndarray]]) -> dict:
    return {name: _table_copy(table) for name, table in state.items()}


def _table_columns(
    table: dict, expected: dict[str, np.dtype], n_rows: int, table_name: str
) -> None:
    if set(table) != set(expected):
        missing = sorted(set(expected) - set(table))
        extra = sorted(set(table) - set(expected))
        raise ValueError(
            f"{table_name} columns differ from schema; missing={missing}, extra={extra}"
        )
    for name, expected_dtype in expected.items():
        values = np.asarray(table[name])
        if values.ndim != 1 or len(values) != n_rows:
            raise ValueError(f"{table_name} column {name} has the wrong shape")
        if values.dtype != expected_dtype:
            raise ValueError(
                f"{table_name} column {name} has dtype {values.dtype}, "
                f"expected {expected_dtype}"
            )


def _assert_states_equal(left: dict, right: dict, message: str) -> None:
    if set(left) != set(right):
        raise ValueError(message)
    for table_name in left:
        if set(left[table_name]) != set(right[table_name]):
            raise ValueError(message)
        for column_name in left[table_name]:
            if not np.array_equal(
                left[table_name][column_name], right[table_name][column_name]
            ):
                raise ValueError(f"{message}: {table_name}.{column_name}")


def _initial_state(
    microdata: dict,
    identity_map: dict,
    dwellings: dict,
    businesses: dict,
    hospitals: dict,
) -> dict:
    person = microdata["person"]
    truth_person_id = identity_map["identity"]["truth_person_id"]
    truth_household_id = identity_map["identity"]["truth_household_id"]
    person_household_position = np.asarray(person["household"], dtype=np.int64)
    snapshot_tick = int(identity_map["snapshot_tick"])

    dwelling = dwellings["dwelling"]
    occupied_position = np.flatnonzero(dwelling["is_occupied"])
    occupied_household = dwelling["truth_household_id"][occupied_position]
    household_order = np.argsort(occupied_household, kind="stable")
    if not np.array_equal(occupied_household[household_order], truth_household_id):
        raise ValueError("occupied dwellings do not map one-to-one to households")
    household_dwelling_position = occupied_position[household_order]

    establishment = businesses["establishment"]
    hospital_establishment_id = hospitals["hospital"]["truth_establishment_id"]
    is_hospital = np.isin(
        establishment["truth_establishment_id"], hospital_establishment_id
    )

    encounter = hospitals["encounter"]
    open_remaining_days = np.maximum(
        encounter["discharge_tick"].astype(np.int64) - snapshot_tick, 1
    )
    open_end_month = snapshot_tick + np.maximum(
        1, np.ceil(open_remaining_days / 30.0).astype(np.int64)
    )
    scheduled_end_tick = np.where(
        encounter["is_open"], open_end_month, snapshot_tick
    ).astype(np.int64)

    return {
        "person": {
            "truth_person_id": truth_person_id.copy(),
            "truth_household_id": truth_household_id[person_household_position].copy(),
            "cell": np.asarray(person["cell"], dtype=np.int64).copy(),
            "birth_tick": (
                snapshot_tick - np.asarray(person["age"], dtype=np.int64) * 12
            ).astype(np.int64),
            "sex": np.asarray(person["sex"], dtype=np.int8).copy(),
            "role": np.asarray(person["role"], dtype=np.int8).copy(),
            "education": np.asarray(person["education"], dtype=np.int8).copy(),
            "income_cents": np.rint(
                np.asarray(person["income"], dtype=np.float64) * 100.0
            ).astype(np.int64),
            "frailty_centi": _frailty_centi(person["frailty"]),
            "is_alive": np.ones(len(truth_person_id), dtype=np.bool_),
        },
        "household": {
            "truth_household_id": truth_household_id.copy(),
            "truth_dwelling_id": dwelling["truth_dwelling_id"][
                household_dwelling_position
            ].copy(),
            "cell": np.asarray(microdata["household_cell"], dtype=np.int64).copy(),
            "is_active": np.ones(len(truth_household_id), dtype=np.bool_),
        },
        "dwelling": {
            "truth_dwelling_id": dwelling["truth_dwelling_id"].copy(),
            "cell": dwelling["cell"].copy(),
            "truth_household_id": dwelling["truth_household_id"].copy(),
            "resident_count": dwelling["resident_count"].copy(),
            "is_occupied": dwelling["is_occupied"].copy(),
        },
        "establishment": {
            "truth_establishment_id": establishment["truth_establishment_id"].copy(),
            "truth_enterprise_id": establishment["truth_enterprise_id"].copy(),
            "cell": establishment["cell"].copy(),
            "industry": establishment["industry"].copy(),
            "is_hospital": is_hospital.astype(np.bool_),
            "is_active": establishment["is_active"].copy(),
        },
        "job": {
            "truth_job_id": businesses["job"]["truth_job_id"].copy(),
            "truth_person_id": businesses["job"]["truth_person_id"].copy(),
            "truth_establishment_id": businesses["job"][
                "truth_establishment_id"
            ].copy(),
            "occupation": businesses["job"]["occupation"].copy(),
            "employment_type": businesses["job"]["employment_type"].copy(),
            "annual_hours": businesses["job"]["annual_hours"].copy(),
            "hourly_wage_cents": businesses["job"]["hourly_wage_cents"].copy(),
            "annual_earnings_cents": businesses["job"]["annual_earnings_cents"].copy(),
            "is_active": businesses["job"]["is_active"].copy(),
        },
        "encounter": {
            "truth_encounter_id": encounter["truth_encounter_id"].copy(),
            "truth_person_id": encounter["truth_person_id"].copy(),
            "truth_hospital_id": encounter["truth_hospital_id"].copy(),
            "scheduled_end_tick": scheduled_end_tick,
            "service": encounter["service"].copy(),
            "diagnosis_group": encounter["diagnosis_group"].copy(),
            "outcome": encounter["outcome"].copy(),
            "cost_cents": encounter["cost_cents"].copy(),
            "bed_number": encounter["bed_number"].copy(),
            "is_open": encounter["is_open"].copy(),
        },
    }


def _empty_event_record() -> dict[str, int]:
    record = {name: 0 for name in _EVENT_DTYPES}
    for name in (
        "from_cell",
        "to_cell",
        "birth_tick",
        "sex",
        "role",
        "education",
        "industry",
        "occupation",
        "employment_type",
        "scheduled_end_tick",
        "service",
        "diagnosis_group",
        "outcome",
        "bed_number",
    ):
        record[name] = -1
    return record


def _new_record(
    event_type: int,
    tick: int,
    recorded_tick: int,
    entity_type: int,
    truth_entity_id: int,
    cause_code: int,
    order: int,
    **payload: int,
) -> dict[str, int]:
    record = _empty_event_record()
    record.update(
        {
            "tick": tick,
            "recorded_tick": recorded_tick,
            "entity_type": entity_type,
            "truth_entity_id": truth_entity_id,
            "event_type": event_type,
            "cause_code": cause_code,
            "_order": order,
        }
    )
    record.update(payload)
    return record


def _report_tick(
    rng: np.random.Generator, tick: int, params: EventHistoryParams,
    late_probability: float | None = None
) -> int:
    """When an event reaches the register, which is the tick a source can see it at.

    ``late_probability`` overrides the ledger's published base rate for one family of
    events. Death is the family that uses it: how well a register closes a record is a
    mechanism of the world rather than a constant, and the two axes behind it are
    declared. One draw is taken either way, so the ledger's stream does not depend on
    which rate applies.
    """
    rate = params.late_report_probability if late_probability is None else float(late_probability)
    if rng.random() >= rate:
        return tick
    return tick + int(rng.integers(1, params.max_report_delay_months + 1))


def _append_rows(table: dict[str, np.ndarray], rows: dict[str, np.ndarray]) -> None:
    if not rows:
        return
    lengths = {len(values) for values in rows.values()}
    if len(lengths) != 1:
        raise RuntimeError("appended state columns have inconsistent lengths")
    for name in table:
        table[name] = np.concatenate(
            (table[name], np.asarray(rows[name], dtype=table[name].dtype))
        )


def _sequence(ids: np.ndarray) -> np.ndarray:
    return np.asarray(ids, dtype=np.uint64) & np.uint64(SEQUENCE_MASK)


def _position(identifier: int, entity: str, row_count: int) -> int:
    value = np.uint64(identifier)
    namespace = int(value >> np.uint64(NAMESPACE_SHIFT))
    if namespace != ENTITY_NAMESPACE[entity]:
        raise ValueError(f"event payload uses the wrong {entity} namespace")
    position = int(value & np.uint64(SEQUENCE_MASK)) - 1
    if not 0 <= position < row_count:
        raise ValueError(f"event references a nonexistent {entity}")
    return position


def _nearest_facility(
    grid_shape: tuple[int, int], facility_cell: np.ndarray
) -> np.ndarray:
    """Nearest facility by Chebyshev distance with stable facility-ID ties."""
    cells = np.arange(grid_shape[0] * grid_shape[1], dtype=np.int64)
    row, column = np.divmod(cells, grid_shape[1])
    facility_row, facility_column = np.divmod(
        np.asarray(facility_cell, dtype=np.int64), grid_shape[1]
    )
    owner = np.zeros(len(cells), dtype=np.int64)
    best = np.full(len(cells), np.iinfo(np.int32).max, dtype=np.int32)
    for position, (f_row, f_column) in enumerate(
        zip(facility_row, facility_column, strict=True)
    ):
        distance = np.maximum(np.abs(row - f_row), np.abs(column - f_column))
        better = distance < best
        owner[better] = position
        best[better] = distance[better].astype(np.int32)
    return owner


def _active_job_by_person(state: dict) -> np.ndarray:
    person = state["person"]
    job = state["job"]
    active_by_person = np.full(len(person["truth_person_id"]), -1, dtype=np.int64)
    active_position = np.flatnonzero(job["is_active"])
    if len(active_position):
        person_position = (
            _sequence(job["truth_person_id"][active_position]).astype(np.int64) - 1
        )
        if np.any(active_by_person[person_position] >= 0):
            raise ValueError("multiple active jobs refer to one person")
        active_by_person[person_position] = active_position
    return active_by_person


def _new_entity_ids(entity: str, existing: np.ndarray, count: int) -> np.ndarray:
    start = int(_sequence(existing).max()) + 1 if len(existing) else 1
    return truth_entity_ids(entity, count, start_sequence=start)


def _make_event_table(records: list[dict[str, int]],
                      start_sequence: int = 1) -> dict[str, np.ndarray]:
    if not records:
        return {name: np.empty(0, dtype=dtype) for name, dtype in _EVENT_DTYPES.items()}
    tick = np.fromiter((row["tick"] for row in records), dtype=np.int64)
    order = np.fromiter((row["_order"] for row in records), dtype=np.int64)
    # Reporting lag is observation metadata, never an input to truth chronology.
    canonical = np.lexsort((order, tick))
    ordered = [records[int(position)] for position in canonical]
    table = {
        name: np.asarray([row[name] for row in ordered], dtype=dtype)
        for name, dtype in _EVENT_DTYPES.items()
        if name != "truth_event_id"
    }
    table["truth_event_id"] = truth_entity_ids("event", len(ordered), start_sequence)
    return {name: table[name] for name in _EVENT_DTYPES}


def _resolved_params_record(params: EventHistoryParams, demography: object) -> dict:
    return {
        "monthly_household_move_rate": float(params.monthly_household_move_rate),
        "monthly_job_turnover_rate": float(params.monthly_job_turnover_rate),
        "monthly_establishment_churn_rate": float(
            params.monthly_establishment_churn_rate
        ),
        "completed_encounter_share": float(params.completed_encounter_share),
        "late_report_probability": float(params.late_report_probability),
        "max_report_delay_months": int(params.max_report_delay_months),
        "minimum_work_age": int(params.minimum_work_age),
        "maximum_work_age": int(params.maximum_work_age),
        "makeham": float(demography.makeham),
        "gompertz_a": float(demography.gompertz_a),
        "gompertz_b": float(demography.gompertz_b),
        "fertility_rate": float(demography.fertility_rate),
        "leave_home_rate": float(demography.leave_home_rate),
        "infant_extra": float(demography.infant_extra),
    }


def _shock_multipliers(shocks: list[dict], month: int) -> tuple[float, float, float, float]:
    year = (month - 1) // 12
    mortality = 1.0
    fertility = 1.0
    migration = 1.0
    admission = 1.0
    for shock in shocks:
        if int(shock["year"]) != year:
            continue
        mortality *= float(shock.get("mortality_multiplier", 1.0))
        fertility *= float(shock.get("fertility_multiplier", 1.0))
        migration *= float(shock.get("leave_home_multiplier", 1.0))
        admission *= float(shock.get("admission_multiplier", 1.0))
    return mortality, fertility, migration, admission


# Latent frailty travels with the person: stored in hundredths so replay conserves it
# byte for byte.  A newborn inherits part of the mother's burden, which is what makes
# baseline health burden a household-level covariate rather than white noise.
FRAILTY_CENTI_RANGE: Final = (int(FRAILTY_RANGE[0] * 100), int(FRAILTY_RANGE[1] * 100))
# Age buckets used when drawing a destination.  Representative ages only; the pull
# curve itself is public (``mechanisms.migration_age_pull``).
_DESTINATION_AGE_BUCKETS: Final = (9.0, 24.0, 37.0, 55.0, 75.0)
_DESTINATION_AGE_EDGES: Final = (18, 30, 45, 65)
_DISTANCE_HALF_LIFE_CELLS: Final = 12.0
# Substream tag for a committed continuation.  It must equal
# ``actuarial.CONTINUATION_DOMAIN``; the constant is repeated here rather than imported
# because the actuarial module reads this one's event table and importing it back would
# close a cycle.  A member's key is never arithmetic on the root seed.
CONTINUATION_DOMAIN: Final = 0xC047
LEDGER_DOMAIN: Final = 0xE7E170
# A member's own shock schedule rides the continuation tag with this substream index,
# which no month can take: months run from one to the ledger's length.
SHOCK_SUBSTREAM: Final = 0x5A0C


def _frailty_centi(frailty: np.ndarray) -> np.ndarray:
    low, high = FRAILTY_CENTI_RANGE
    return np.clip(
        np.rint(np.asarray(frailty, dtype=np.float64) * 100.0), low, high
    ).astype(np.int16)


def _gravity_destinations(
    rng: np.random.Generator,
    vacant_position: np.ndarray,
    vacant_cell: np.ndarray,
    origin_cell: np.ndarray,
    age: np.ndarray,
    urbanity_flat: np.ndarray,
    mechanisms: WorldMechanisms,
) -> tuple[np.ndarray, np.ndarray]:
    """Assign distinct vacant dwellings by a gravity draw over counties and cells.

    Destination county probability is proportional to the county's urban pull under the
    mover's own age profile, divided by one plus the seat-to-seat distance in cells.
    Within the chosen county a dwelling is drawn without replacement in proportion to
    cell urbanity.  Movers whose county runs out of vacant units do not move this month.

    Returns the positions (into ``age``) of the movers that were placed, and the row of
    ``vacant_position`` each one takes.  Both are sorted by mover position, so the draw
    is order-independent given the seed.
    """
    n_movers = len(age)
    if n_movers == 0 or len(vacant_position) == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    intensity = float(mechanisms.coefficients["migration_age_pattern"])
    pull = np.asarray(urbanity_flat, dtype=np.float64)[vacant_cell] + 0.02
    vacant_county = mechanisms.county_of_cell(vacant_cell)
    origin_county = mechanisms.county_of_cell(origin_cell)
    n_counties = mechanisms.county.n_counties
    distance = np.asarray(mechanisms.county.distance, dtype=np.float64)

    bucket = np.digitize(np.asarray(age, dtype=np.float64), _DESTINATION_AGE_EDGES)
    bucket_pull = intensity * migration_age_pull(np.asarray(_DESTINATION_AGE_BUCKETS))
    county_mass = np.zeros((len(_DESTINATION_AGE_BUCKETS), n_counties), dtype=np.float64)
    for b, exponent in enumerate(bucket_pull):
        county_mass[b] = np.bincount(
            vacant_county, weights=pull ** exponent, minlength=n_counties
        )
    decay = 1.0 / (1.0 + distance / _DISTANCE_HALF_LIFE_CELLS)

    chosen_county = np.full(n_movers, -1, dtype=np.int64)
    for b in np.unique(bucket):
        for c in np.unique(origin_county[bucket == b]):
            group = np.flatnonzero((bucket == b) & (origin_county == c))
            weight = county_mass[b] * decay[c]
            total = weight.sum()
            if total <= 0.0:
                continue
            chosen_county[group] = rng.choice(
                n_counties, size=len(group), replace=True, p=weight / total
            )

    within = pull ** (intensity * float(np.mean(migration_age_pull(np.asarray(_DESTINATION_AGE_BUCKETS)))))
    placed_mover: list[np.ndarray] = []
    placed_slot: list[np.ndarray] = []
    for county in np.unique(chosen_county[chosen_county >= 0]):
        movers = np.flatnonzero(chosen_county == county)
        slots = np.flatnonzero(vacant_county == county)
        if len(slots) == 0:
            continue
        take = min(len(movers), len(slots))
        weight = within[slots]
        total = weight.sum()
        probability = weight / total if total > 0.0 else None
        drawn = rng.choice(slots, size=take, replace=False, p=probability)
        if take < len(movers):
            # A county short of vacant units places a random subset, never the movers
            # that happen to sit first in identity order, which would tilt by age.
            movers = np.sort(rng.choice(movers, size=take, replace=False))
        placed_mover.append(movers[:take])
        placed_slot.append(np.asarray(drawn, dtype=np.int64))
    if not placed_mover:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    mover = np.concatenate(placed_mover)
    slot = np.concatenate(placed_slot)
    order = np.argsort(mover, kind="stable")
    return mover[order], slot[order]


_LEDGER_CONTEXT: tuple[str, ...] = (
    "annual_encounter_rate",
    "branch_month",
    "coefficients",
    "continuation_member",
    "death_late_probability",
    "demography",
    "hospital_beds",
    "hospital_id",
    "mechanisms",
    "nearest_hospital_by_cell",
    "params",
    "payroll_level",
    "seed",
    "shocks",
    "snapshot_tick",
    "urbanity_flat",
)


def _run_ledger_months(context: dict, loop: dict, first_month: int,
                       last_month: int) -> None:
    """Advance the ledger over one span of months, in place, over the carried state.

    ``context`` holds everything a month reads and never changes; ``loop`` holds the four
    quantities the months carry forward: the entity state, the event records, the running
    order counter, and each household's last move. The span is a separate function so a
    continuation member can start from a captured branch state and pay only for the
    months after the branch, instead of replaying the ledger that produced it.
    """
    (annual_encounter_rate,
     branch_month,
     coefficients,
     continuation_member,
     death_late_probability,
     demography,
     hospital_beds,
     hospital_id,
     mechanisms,
     nearest_hospital_by_cell,
     params,
     payroll_level,
     seed,
     shocks,
     snapshot_tick,
     urbanity_flat) = (
        context[name] for name in _LEDGER_CONTEXT)
    state = loop["state"]
    records = loop["records"]
    order = loop["order"]
    household_last_move_tick = loop["household_last_move_tick"]

    for month in range(first_month, last_month + 1):
        tick = snapshot_tick + month
        rng = np.random.default_rng(
            np.random.SeedSequence([seed, CONTINUATION_DOMAIN, continuation_member, month])
            if continuation_member is not None and month > branch_month
            else np.random.SeedSequence([seed, LEDGER_DOMAIN, month])
        )
        mortality_multiplier, fertility_multiplier, migration_multiplier, \
            admission_multiplier = _shock_multipliers(shocks, month)
        job_end_count = 0
        latest_job_end_recorded = tick
        latest_encounter_discharge_recorded = tick
        latest_household_close_recorded = tick
        opened_recorded_tick: dict[int, int] = {}
        household_count_at_month_start = len(state["household"]["truth_household_id"])

        # Close and replace a small number of non-hospital operating locations. Jobs
        # end before closure; replacements remain under the same enterprises.
        establishment = state["establishment"]
        close_candidates = np.flatnonzero(
            establishment["is_active"] & ~establishment["is_hospital"]
        )
        n_close = int(
            rng.binomial(len(close_candidates), params.monthly_establishment_churn_rate)
        )
        if n_close:
            close_position = np.sort(
                rng.choice(close_candidates, size=n_close, replace=False)
            )
            close_id = establishment["truth_establishment_id"][close_position]
            close_enterprise = establishment["truth_enterprise_id"][close_position]
            close_industry = establishment["industry"][close_position]
            close_cell = establishment["cell"][close_position]
            new_id = _new_entity_ids(
                "establishment", establishment["truth_establishment_id"], n_close
            )
            alive_cell = state["person"]["cell"][state["person"]["is_alive"]]
            new_cell = rng.choice(alive_cell, size=n_close, replace=True).astype(
                np.int64
            )
            opening_batch_recorded = _report_tick(rng, tick, params)

            for index, position in enumerate(close_position):
                recorded = _report_tick(rng, tick, params)
                linked_job = np.flatnonzero(
                    state["job"]["is_active"]
                    & (
                        state["job"]["truth_establishment_id"]
                        == establishment["truth_establishment_id"][position]
                    )
                )
                for job_position in linked_job:
                    job_identifier = int(state["job"]["truth_job_id"][job_position])
                    records.append(
                        _new_record(
                            EVENT_TYPES["job_ended"],
                            tick,
                            recorded,
                            ENTITY_NAMESPACE["job"],
                            job_identifier,
                            CAUSE_CODES["business_churn"],
                            order,
                            truth_job_id=job_identifier,
                            truth_person_id=int(
                                state["job"]["truth_person_id"][job_position]
                            ),
                            truth_establishment_id=int(close_id[index]),
                        )
                    )
                    order += 1
                state["job"]["is_active"][linked_job] = False
                job_end_count += len(linked_job)
                if len(linked_job):
                    latest_job_end_recorded = max(latest_job_end_recorded, recorded)
                records.append(
                    _new_record(
                        EVENT_TYPES["establishment_closed"],
                        tick,
                        recorded,
                        ENTITY_NAMESPACE["establishment"],
                        int(close_id[index]),
                        CAUSE_CODES["business_churn"],
                        order,
                        truth_enterprise_id=int(close_enterprise[index]),
                        truth_establishment_id=int(close_id[index]),
                        from_cell=int(close_cell[index]),
                        industry=int(close_industry[index]),
                    )
                )
                order += 1
                establishment["is_active"][position] = False

                open_recorded = opening_batch_recorded
                opened_recorded_tick[int(new_id[index])] = open_recorded
                records.append(
                    _new_record(
                        EVENT_TYPES["establishment_opened"],
                        tick,
                        open_recorded,
                        ENTITY_NAMESPACE["establishment"],
                        int(new_id[index]),
                        CAUSE_CODES["business_churn"],
                        order,
                        truth_enterprise_id=int(close_enterprise[index]),
                        truth_establishment_id=int(new_id[index]),
                        to_cell=int(new_cell[index]),
                        industry=int(close_industry[index]),
                    )
                )
                order += 1
            _append_rows(
                establishment,
                {
                    "truth_establishment_id": new_id,
                    "truth_enterprise_id": close_enterprise,
                    "cell": new_cell,
                    "industry": close_industry,
                    "is_hospital": np.zeros(n_close, dtype=np.bool_),
                    "is_active": np.ones(n_close, dtype=np.bool_),
                },
            )

        # Deaths terminate active jobs and open encounters before the person event.
        person = state["person"]
        alive_position = np.flatnonzero(person["is_alive"])
        age = np.maximum(0, (tick - person["birth_tick"][alive_position]) // 12).astype(
            np.int16
        )
        # Local mortality: the published Gompertz curve times a person's latent
        # frailty and their county's level, on a national improvement trend.  Nothing
        # here is a world constant, so a level fitted on one world does not transfer.
        death_county = mechanisms.county_of_cell(person["cell"][alive_position])
        frailty = person["frailty_centi"][alive_position].astype(np.float64) / 100.0
        improvement = float(coefficients["mortality_improvement"])
        elapsed_years = (tick - snapshot_tick) / 12.0
        mortality_level = (
            np.power(max(1.0 - improvement, 1e-6), elapsed_years)
            * np.power(frailty, float(coefficients["mortality_frailty"]))
            * np.exp(
                float(coefficients["mortality_urban"])
                * (mechanisms.covariate("urban", death_county) - 0.5)
                + float(coefficients["mortality_econ"])
                * (mechanisms.covariate("econ", death_county) - 0.5)
                + mechanisms.effect("mortality", death_county)
            )
        )
        annual_death_probability = np.clip(
            mortality_probability(age, demography) * mortality_level, 0.0, 1.0
        )
        monthly_death_probability = 1.0 - np.power(
            1.0 - annual_death_probability,
            mortality_multiplier / 12.0,
        )
        dies = alive_position[
            rng.random(len(alive_position)) < monthly_death_probability
        ]
        active_job_by_person = _active_job_by_person(state)
        for person_position in dies:
            recorded = _report_tick(rng, tick, params, death_late_probability)
            person_identifier = int(person["truth_person_id"][person_position])
            job_position = int(active_job_by_person[person_position])
            if job_position >= 0:
                job_identifier = int(state["job"]["truth_job_id"][job_position])
                records.append(
                    _new_record(
                        EVENT_TYPES["job_ended"],
                        tick,
                        recorded,
                        ENTITY_NAMESPACE["job"],
                        job_identifier,
                        CAUSE_CODES["demography"],
                        order,
                        truth_job_id=job_identifier,
                        truth_person_id=person_identifier,
                        truth_establishment_id=int(
                            state["job"]["truth_establishment_id"][job_position]
                        ),
                    )
                )
                order += 1
                state["job"]["is_active"][job_position] = False
                job_end_count += 1
                latest_job_end_recorded = max(latest_job_end_recorded, recorded)
            open_encounter = np.flatnonzero(
                state["encounter"]["is_open"]
                & (state["encounter"]["truth_person_id"] == person_identifier)
            )
            for encounter_position in open_encounter:
                encounter_identifier = int(
                    state["encounter"]["truth_encounter_id"][encounter_position]
                )
                records.append(
                    _new_record(
                        EVENT_TYPES["encounter_discharged"],
                        tick,
                        recorded,
                        ENTITY_NAMESPACE["encounter"],
                        encounter_identifier,
                        CAUSE_CODES["demography"],
                        order,
                        truth_encounter_id=encounter_identifier,
                        truth_person_id=person_identifier,
                        truth_hospital_id=int(
                            state["encounter"]["truth_hospital_id"][encounter_position]
                        ),
                        outcome=ENCOUNTER_OUTCOMES["died"],
                        bed_number=int(
                            state["encounter"]["bed_number"][encounter_position]
                        ),
                    )
                )
                order += 1
                latest_encounter_discharge_recorded = max(
                    latest_encounter_discharge_recorded, recorded
                )
            state["encounter"]["is_open"][open_encounter] = False
            state["encounter"]["bed_number"][open_encounter] = -1
            state["encounter"]["outcome"][open_encounter] = ENCOUNTER_OUTCOMES["died"]
            death_cause = (
                CAUSE_CODES["world_shock"]
                if mortality_multiplier != 1.0
                else CAUSE_CODES["demography"]
            )
            records.append(
                _new_record(
                    EVENT_TYPES["person_death"],
                    tick,
                    recorded,
                    ENTITY_NAMESPACE["person"],
                    person_identifier,
                    death_cause,
                    order,
                    truth_person_id=person_identifier,
                    truth_household_id=int(
                        person["truth_household_id"][person_position]
                    ),
                    from_cell=int(person["cell"][person_position]),
                )
            )
            order += 1
        person["is_alive"][dies] = False

        # Close households that lost their final living member and release the dwelling.
        household = state["household"]
        dwelling = state["dwelling"]
        alive_household_position = (
            _sequence(person["truth_household_id"][person["is_alive"]]).astype(np.int64)
            - 1
        )
        living_count = np.bincount(
            alive_household_position, minlength=len(household["truth_household_id"])
        )
        empty_household = np.flatnonzero(household["is_active"] & (living_count == 0))
        for household_position in empty_household:
            recorded = _report_tick(rng, tick, params)
            household_identifier = int(
                household["truth_household_id"][household_position]
            )
            dwelling_identifier = int(
                household["truth_dwelling_id"][household_position]
            )
            dwelling_position = _position(
                dwelling_identifier, "dwelling", len(dwelling["truth_dwelling_id"])
            )
            records.append(
                _new_record(
                    EVENT_TYPES["household_closed"],
                    tick,
                    recorded,
                    ENTITY_NAMESPACE["household"],
                    household_identifier,
                    CAUSE_CODES["demography"],
                    order,
                    truth_household_id=household_identifier,
                    truth_dwelling_id=dwelling_identifier,
                    from_cell=int(household["cell"][household_position]),
                )
            )
            order += 1
            latest_household_close_recorded = max(
                latest_household_close_recorded, recorded
            )
            household["is_active"][household_position] = False
            dwelling["is_occupied"][dwelling_position] = False
            dwelling["truth_household_id"][dwelling_position] = 0
            dwelling["resident_count"][dwelling_position] = 0

        # Births retain the mother's persistent household and allocate new person IDs.
        person = state["person"]
        alive_position = np.flatnonzero(person["is_alive"])
        age = np.maximum(0, (tick - person["birth_tick"][alive_position]) // 12).astype(
            np.int16
        )
        mothers = alive_position[
            (person["sex"][alive_position] == 1) & (age >= 18) & (age <= 45)
        ]
        monthly_birth_probability = 1.0 - (
            1.0 - min(1.0, demography.fertility_rate * fertility_multiplier)
        ) ** (1.0 / 12.0)
        mothers = mothers[rng.random(len(mothers)) < monthly_birth_probability]
        if len(mothers):
            new_person_id = _new_entity_ids(
                "person", person["truth_person_id"], len(mothers)
            )
            newborn_sex = (rng.random(len(mothers)) < 0.5).astype(np.int8)
            newborn_household = person["truth_household_id"][mothers]
            newborn_cell = person["cell"][mothers]
            mother_frailty = person["frailty_centi"][mothers].astype(np.float64) / 100.0
            newborn_frailty_centi = _frailty_centi(
                newborn_frailty(
                    mother_frailty, rng.normal(0.0, 1.0, size=len(mothers))
                )
            )
            birth_batch_recorded = _report_tick(rng, tick, params)
            for index, person_identifier in enumerate(new_person_id):
                recorded = birth_batch_recorded
                birth_cause = (
                    CAUSE_CODES["world_shock"]
                    if fertility_multiplier != 1.0
                    else CAUSE_CODES["demography"]
                )
                records.append(
                    _new_record(
                        EVENT_TYPES["person_birth"],
                        tick,
                        recorded,
                        ENTITY_NAMESPACE["person"],
                        int(person_identifier),
                        birth_cause,
                        order,
                        truth_person_id=int(person_identifier),
                        truth_household_id=int(newborn_household[index]),
                        to_cell=int(newborn_cell[index]),
                        birth_tick=tick,
                        sex=int(newborn_sex[index]),
                        role=2,
                        education=0,
                        income_cents=0,
                        frailty_centi=int(newborn_frailty_centi[index]),
                    )
                )
                order += 1
            _append_rows(
                person,
                {
                    "truth_person_id": new_person_id,
                    "truth_household_id": newborn_household,
                    "cell": newborn_cell,
                    "birth_tick": np.full(len(mothers), tick, dtype=np.int64),
                    "sex": newborn_sex,
                    "role": np.full(len(mothers), 2, dtype=np.int8),
                    "education": np.zeros(len(mothers), dtype=np.int8),
                    "income_cents": np.zeros(len(mothers), dtype=np.int64),
                    "frailty_centi": newborn_frailty_centi,
                    "is_alive": np.ones(len(mothers), dtype=np.bool_),
                },
            )

        # Young adults form persistent new households in existing vacant dwellings.
        person = state["person"]
        household = state["household"]
        dwelling = state["dwelling"]
        living_household_position = (
            _sequence(person["truth_household_id"][person["is_alive"]]).astype(np.int64)
            - 1
        )
        living_count = np.bincount(
            living_household_position, minlength=len(household["truth_household_id"])
        )
        age = np.maximum(0, (tick - person["birth_tick"]) // 12)
        formation_candidates = np.flatnonzero(
            person["is_alive"]
            & (person["role"] == 2)
            & (age >= 18)
            & (age <= 30)
            & (
                living_count[
                    _sequence(person["truth_household_id"]).astype(np.int64) - 1
                ]
                >= 2
            )
        )
        if len(formation_candidates):
            candidate_household = person["truth_household_id"][formation_candidates]
            _, unique_position = np.unique(candidate_household, return_index=True)
            formation_candidates = formation_candidates[np.sort(unique_position)]
        # Leaving home is a per-person hazard in the household's county, the household's
        # size, and the person's own age, not one national rate.
        base_formation = min(
            1.0, demography.leave_home_rate * migration_multiplier / 12.0
        )
        vacant_position = np.flatnonzero(~dwelling["is_occupied"])
        destination = np.empty(0, dtype=np.int64)
        if len(formation_candidates):
            candidate_county = mechanisms.county_of_cell(
                person["cell"][formation_candidates]
            )
            candidate_size = living_count[
                _sequence(person["truth_household_id"][formation_candidates]).astype(
                    np.int64
                )
                - 1
            ].astype(np.float64)
            formation_hazard = np.clip(
                base_formation
                * np.exp(
                    float(coefficients["formation_intercept_shift"])
                    + float(coefficients["formation_urban"])
                    * (mechanisms.covariate("urban", candidate_county) - 0.5)
                    + float(coefficients["formation_econ"])
                    * (mechanisms.covariate("econ", candidate_county) - 0.5)
                    + float(coefficients["formation_size"]) * (candidate_size - 3.0)
                    + float(coefficients["formation_age"])
                    * (age[formation_candidates].astype(np.float64) - 24.0)
                    / 6.0
                    + mechanisms.effect("formation", candidate_county)
                ),
                0.0,
                1.0,
            )
            movers = formation_candidates[
                rng.random(len(formation_candidates)) < formation_hazard
            ]
        else:
            movers = formation_candidates
        if len(movers):
            placed, slot = _gravity_destinations(
                rng,
                vacant_position,
                dwelling["cell"][vacant_position],
                person["cell"][movers],
                age[movers],
                urbanity_flat,
                mechanisms,
            )
            movers = movers[placed]
            destination = vacant_position[slot].astype(np.int64)
        if len(movers):
            new_household_id = _new_entity_ids(
                "household", household["truth_household_id"], len(movers)
            )
            destination_dwelling = dwelling["truth_dwelling_id"][destination]
            destination_cell = dwelling["cell"][destination]
            prior_household = person["truth_household_id"][movers].copy()
            prior_cell = person["cell"][movers].copy()
            formation_batch_recorded = _report_tick(rng, tick, params)
            formation_batch_recorded = max(
                formation_batch_recorded, latest_household_close_recorded
            )
            for index, person_position in enumerate(movers):
                recorded = formation_batch_recorded
                migration_cause = (
                    CAUSE_CODES["world_shock"]
                    if migration_multiplier != 1.0
                    else CAUSE_CODES["migration"]
                )
                records.append(
                    _new_record(
                        EVENT_TYPES["household_formed"],
                        tick,
                        recorded,
                        ENTITY_NAMESPACE["household"],
                        int(new_household_id[index]),
                        migration_cause,
                        order,
                        truth_person_id=int(person["truth_person_id"][person_position]),
                        truth_household_id=int(new_household_id[index]),
                        truth_prior_household_id=int(prior_household[index]),
                        truth_dwelling_id=int(destination_dwelling[index]),
                        from_cell=int(prior_cell[index]),
                        to_cell=int(destination_cell[index]),
                    )
                )
                order += 1
            _append_rows(
                household,
                {
                    "truth_household_id": new_household_id,
                    "truth_dwelling_id": destination_dwelling,
                    "cell": destination_cell,
                    "is_active": np.ones(len(movers), dtype=np.bool_),
                },
            )
            dwelling["is_occupied"][destination] = True
            dwelling["truth_household_id"][destination] = new_household_id
            person["truth_household_id"][movers] = new_household_id
            person["cell"][movers] = destination_cell
            person["role"][movers] = 0
            household_last_move_tick = np.concatenate(
                [household_last_move_tick, np.full(len(movers), tick, dtype=np.int64)]
            )

        # Existing households sometimes relocate, swapping an occupied unit for a
        # vacant unit. Newly formed households are eligible only in later months.
        household = state["household"]
        dwelling = state["dwelling"]
        move_candidates = np.flatnonzero(
            household["is_active"][:household_count_at_month_start]
        )
        vacant_position = np.flatnonzero(~dwelling["is_occupied"])
        move_household = np.empty(0, dtype=np.int64)
        destination = np.empty(0, dtype=np.int64)
        if len(move_candidates) and len(vacant_position):
            # Relocation is a per-household hazard in the county, the household's income
            # band, and how long it has stayed put.  The tenure term is the declared
            # migration by stale-address-linkage interaction on the ledger side.
            person = state["person"]
            alive = person["is_alive"]
            alive_household = (
                _sequence(person["truth_household_id"][alive]).astype(np.int64) - 1
            )
            n_households = len(household["truth_household_id"])
            household_income = np.bincount(
                alive_household,
                weights=person["income_cents"][alive].astype(np.float64),
                minlength=n_households,
            )
            household_members = np.bincount(alive_household, minlength=n_households)
            household_age = np.bincount(
                alive_household,
                weights=np.maximum(
                    0.0, (tick - person["birth_tick"][alive]).astype(np.float64) / 12.0
                ),
                minlength=n_households,
            ) / np.maximum(household_members, 1)
            band = quintile_band(household_income[move_candidates])
            move_county = mechanisms.county_of_cell(household["cell"][move_candidates])
            tenure = np.log1p(
                np.maximum(0, tick - household_last_move_tick[move_candidates]) / 12.0
            )
            move_hazard = np.clip(
                params.monthly_household_move_rate
                * np.exp(
                    float(coefficients["move_intercept_shift"])
                    + float(coefficients["move_urban"])
                    * (mechanisms.covariate("urban", move_county) - 0.5)
                    + float(coefficients["move_income_band"])
                    * (band.astype(np.float64) - 2.0)
                    / 2.0
                    + float(coefficients["move_tenure"]) * tenure
                    + mechanisms.effect("move", move_county)
                ),
                0.0,
                1.0,
            )
            selected = move_candidates[rng.random(len(move_candidates)) < move_hazard]
            if len(selected):
                placed, slot = _gravity_destinations(
                    rng,
                    vacant_position,
                    dwelling["cell"][vacant_position],
                    household["cell"][selected],
                    household_age[selected],
                    urbanity_flat,
                    mechanisms,
                )
                move_household = selected[placed]
                destination = vacant_position[slot].astype(np.int64)
        n_moves = len(move_household)
        if n_moves:
            for index, household_position in enumerate(move_household):
                old_dwelling_id = int(
                    household["truth_dwelling_id"][household_position]
                )
                old_dwelling_position = _position(
                    old_dwelling_id, "dwelling", len(dwelling["truth_dwelling_id"])
                )
                new_dwelling_position = int(destination[index])
                if old_dwelling_position == new_dwelling_position:
                    continue
                household_identifier = int(
                    household["truth_household_id"][household_position]
                )
                new_dwelling_id = int(
                    dwelling["truth_dwelling_id"][new_dwelling_position]
                )
                from_cell = int(household["cell"][household_position])
                to_cell = int(dwelling["cell"][new_dwelling_position])
                recorded = _report_tick(rng, tick, params)
                recorded = max(recorded, latest_household_close_recorded)
                records.append(
                    _new_record(
                        EVENT_TYPES["household_moved"],
                        tick,
                        recorded,
                        ENTITY_NAMESPACE["household"],
                        household_identifier,
                        CAUSE_CODES["migration"],
                        order,
                        truth_household_id=household_identifier,
                        truth_dwelling_id=new_dwelling_id,
                        truth_prior_dwelling_id=old_dwelling_id,
                        from_cell=from_cell,
                        to_cell=to_cell,
                    )
                )
                order += 1
                dwelling["is_occupied"][old_dwelling_position] = False
                dwelling["truth_household_id"][old_dwelling_position] = 0
                dwelling["is_occupied"][new_dwelling_position] = True
                dwelling["truth_household_id"][
                    new_dwelling_position
                ] = household_identifier
                household["truth_dwelling_id"][household_position] = new_dwelling_id
                household["cell"][household_position] = to_cell
                household_last_move_tick[household_position] = tick
            household_position = (
                _sequence(person["truth_household_id"]).astype(np.int64) - 1
            )
            person["cell"] = household["cell"][household_position].copy()

        # Scheduled encounter closures happen before new admissions.
        encounter = state["encounter"]
        scheduled_close = np.flatnonzero(
            encounter["is_open"] & (encounter["scheduled_end_tick"] <= tick)
        )
        for encounter_position in scheduled_close:
            encounter_identifier = int(
                encounter["truth_encounter_id"][encounter_position]
            )
            recorded = _report_tick(rng, tick, params)
            records.append(
                _new_record(
                    EVENT_TYPES["encounter_discharged"],
                    tick,
                    recorded,
                    ENTITY_NAMESPACE["encounter"],
                    encounter_identifier,
                    CAUSE_CODES["scheduled"],
                    order,
                    truth_encounter_id=encounter_identifier,
                    truth_person_id=int(
                        encounter["truth_person_id"][encounter_position]
                    ),
                    truth_hospital_id=int(
                        encounter["truth_hospital_id"][encounter_position]
                    ),
                    outcome=ENCOUNTER_OUTCOMES["discharged"],
                    bed_number=int(encounter["bed_number"][encounter_position]),
                )
            )
            order += 1
            latest_encounter_discharge_recorded = max(
                latest_encounter_discharge_recorded, recorded
            )
        encounter["is_open"][scheduled_close] = False
        encounter["bed_number"][scheduled_close] = -1
        encounter["outcome"][scheduled_close] = ENCOUNTER_OUTCOMES["discharged"]

        # Ordinary job turnover and replacement hiring preserve the active-job total
        # except where the living working-age pool itself becomes binding.
        job = state["job"]
        active_job_position = np.flatnonzero(job["is_active"])
        n_turnover = int(
            rng.binomial(len(active_job_position), params.monthly_job_turnover_rate)
        )
        if n_turnover:
            turnover_position = np.sort(
                rng.choice(active_job_position, size=n_turnover, replace=False)
            )
            for job_position in turnover_position:
                job_identifier = int(job["truth_job_id"][job_position])
                recorded = _report_tick(rng, tick, params)
                records.append(
                    _new_record(
                        EVENT_TYPES["job_ended"],
                        tick,
                        recorded,
                        ENTITY_NAMESPACE["job"],
                        job_identifier,
                        CAUSE_CODES["labor_turnover"],
                        order,
                        truth_job_id=job_identifier,
                        truth_person_id=int(job["truth_person_id"][job_position]),
                        truth_establishment_id=int(
                            job["truth_establishment_id"][job_position]
                        ),
                    )
                )
                order += 1
            job["is_active"][turnover_position] = False
            job_end_count += n_turnover
            latest_job_end_recorded = max(
                latest_job_end_recorded,
                max(int(record["recorded_tick"]) for record in records[-n_turnover:]),
            )

        person = state["person"]
        active_job_by_person = _active_job_by_person(state)
        age = np.maximum(0, (tick - person["birth_tick"]) // 12)
        hire_candidates = np.flatnonzero(
            person["is_alive"]
            & (age >= params.minimum_work_age)
            & (age <= params.maximum_work_age)
            & (active_job_by_person < 0)
        )
        n_hires = min(job_end_count, len(hire_candidates))
        if n_hires:
            hire_weight = 1.0 + 0.18 * person["education"][hire_candidates]
            priorities = (
                -np.log(
                    np.maximum(rng.random(len(hire_candidates)), np.finfo(float).tiny)
                )
                / hire_weight
            )
            selected = np.argpartition(priorities, n_hires - 1)[:n_hires]
            hired_person_position = np.sort(hire_candidates[selected])
            establishment = state["establishment"]
            active_establishment = np.flatnonzero(establishment["is_active"])
            active_job_position = np.flatnonzero(job["is_active"])
            job_establishment_position = (
                _sequence(job["truth_establishment_id"][active_job_position]).astype(
                    np.int64
                )
                - 1
            )
            employment = np.bincount(
                job_establishment_position, minlength=len(establishment["is_active"])
            )
            required = active_establishment[employment[active_establishment] == 0]
            if len(required) > n_hires:
                raise RuntimeError(
                    "replacement hiring cannot staff active establishments"
                )
            assigned = np.empty(n_hires, dtype=np.int64)
            assigned[: len(required)] = required
            remaining = n_hires - len(required)
            if remaining:
                weight = np.sqrt(employment[active_establishment] + 1.0)
                weight = weight / weight.sum()
                assigned[len(required) :] = rng.choice(
                    active_establishment, size=remaining, replace=True, p=weight
                )
            rng.shuffle(assigned)
            new_job_id = _new_entity_ids("job", job["truth_job_id"], n_hires)
            hired_age = age[hired_person_position]
            hired_education = person["education"][hired_person_position]
            industry = establishment["industry"][assigned]
            full_time_probability = np.clip(
                0.67
                + 0.05 * hired_education
                - 0.10 * (hired_age < 23)
                - 0.12 * (hired_age > 66),
                0.42,
                0.92,
            )
            is_full_time = rng.random(n_hires) < full_time_probability
            annual_hours = np.where(
                is_full_time,
                rng.integers(1_720, 2_201, size=n_hires),
                rng.integers(520, 1_501, size=n_hires),
            ).astype(np.int32)
            base_wage = np.asarray(
                [17.5, 22.0, 21.0, 16.5, 20.0, 27.0, 14.5, 21.5, 23.5, 22.5]
            )
            hourly_wage_cents = np.rint(
                np.clip(
                    base_wage[industry]
                    * (1.0 + 0.16 * hired_education)
                    * payroll_level
                    * np.exp(rng.normal(0.0, 0.16, n_hires))
                    * 100.0,
                    900.0,
                    25_000.0,
                )
            ).astype(np.int64)
            employment_type = np.where(
                is_full_time,
                EMPLOYMENT_TYPES["full_time"],
                EMPLOYMENT_TYPES["part_time"],
            ).astype(np.int8)
            occupation = (
                industry * 4 + np.minimum(3, hired_education + (hired_age >= 35))
            ).astype(np.int16)
            assigned_id = establishment["truth_establishment_id"][assigned]
            job_start_batch_recorded = _report_tick(rng, tick, params)
            job_start_batch_recorded = max(
                job_start_batch_recorded, latest_job_end_recorded
            )
            if opened_recorded_tick:
                job_start_batch_recorded = max(
                    job_start_batch_recorded, max(opened_recorded_tick.values())
                )
            for index, job_identifier in enumerate(new_job_id):
                recorded = job_start_batch_recorded
                records.append(
                    _new_record(
                        EVENT_TYPES["job_started"],
                        tick,
                        recorded,
                        ENTITY_NAMESPACE["job"],
                        int(job_identifier),
                        CAUSE_CODES["labor_turnover"],
                        order,
                        truth_job_id=int(job_identifier),
                        truth_person_id=int(
                            person["truth_person_id"][hired_person_position[index]]
                        ),
                        truth_establishment_id=int(assigned_id[index]),
                        occupation=int(occupation[index]),
                        employment_type=int(employment_type[index]),
                        annual_hours=int(annual_hours[index]),
                        hourly_wage_cents=int(hourly_wage_cents[index]),
                    )
                )
                order += 1
            _append_rows(
                job,
                {
                    "truth_job_id": new_job_id,
                    "truth_person_id": person["truth_person_id"][hired_person_position],
                    "truth_establishment_id": assigned_id,
                    "occupation": occupation,
                    "employment_type": employment_type,
                    "annual_hours": annual_hours,
                    "hourly_wage_cents": hourly_wage_cents,
                    "annual_earnings_cents": annual_hours.astype(np.int64)
                    * hourly_wage_cents,
                    "is_active": np.ones(n_hires, dtype=np.bool_),
                },
            )

        # Hospital admissions use current residence and exact available capacity.
        encounter = state["encounter"]
        person = state["person"]
        open_person = np.zeros(len(person["truth_person_id"]), dtype=np.bool_)
        open_encounter_position = np.flatnonzero(encounter["is_open"])
        if len(open_encounter_position):
            open_person_position = (
                _sequence(encounter["truth_person_id"][open_encounter_position]).astype(
                    np.int64
                )
                - 1
            )
            open_person[open_person_position] = True
        patient_candidates = np.flatnonzero(person["is_alive"] & ~open_person)
        target_admissions = min(
            len(patient_candidates),
            int(
                round(
                    len(np.flatnonzero(person["is_alive"]))
                    * annual_encounter_rate
                    * admission_multiplier
                    / 12_000.0
                )
            ),
        )
        if target_admissions:
            patient_age = np.maximum(
                0, (tick - person["birth_tick"][patient_candidates]) // 12
            )
            # Admission risk: the published age curve times latent frailty and the
            # county's own burden.  Incidence is therefore a local hazard, and the
            # health source's inclusion rule reads the same frailty.
            patient_county = mechanisms.county_of_cell(
                person["cell"][patient_candidates]
            )
            patient_frailty = (
                person["frailty_centi"][patient_candidates].astype(np.float64) / 100.0
            )
            risk = (
                (1.0 + 0.012 * patient_age + 0.65 * (patient_age >= 65))
                * np.power(patient_frailty, float(coefficients["incidence_frailty"]))
                * np.exp(
                    float(coefficients["incidence_urban"])
                    * (mechanisms.covariate("urban", patient_county) - 0.5)
                    + float(coefficients["incidence_elder_burden"])
                    * (mechanisms.covariate("elder", patient_county) - 0.5)
                    + mechanisms.effect("incidence", patient_county)
                )
            )
            priorities = (
                -np.log(
                    np.maximum(
                        rng.random(len(patient_candidates)), np.finfo(float).tiny
                    )
                )
                / risk
            )
            selected = np.argpartition(priorities, target_admissions - 1)[
                :target_admissions
            ]
            patient_position = np.sort(patient_candidates[selected])
            patient_hospital_position = nearest_hospital_by_cell[
                person["cell"][patient_position]
            ]
            n_new_encounters = len(patient_position)
            new_encounter_id = _new_entity_ids(
                "encounter", encounter["truth_encounter_id"], n_new_encounters
            )
            diagnosis = rng.integers(0, 8, size=n_new_encounters, dtype=np.int16)
            service = (diagnosis % 4).astype(np.int8)
            duration_months = rng.choice(
                np.asarray([0, 1, 2], dtype=np.int64),
                size=n_new_encounters,
                p=(params.completed_encounter_share, 0.20, 0.02),
            )
            scheduled_end = tick + duration_months
            cost_cents = (
                np.asarray(
                    [58_000, 72_000, 91_000, 118_000, 135_000, 83_000, 104_000, 67_000],
                    dtype=np.int64,
                )[diagnosis]
                * np.maximum(1, duration_months * 10)
                + rng.integers(5_000, 30_001, size=n_new_encounters)
            ).astype(np.int64)

            occupied: list[set[int]] = [set() for _ in range(len(hospital_id))]
            for encounter_position in open_encounter_position:
                h_position = _position(
                    int(encounter["truth_hospital_id"][encounter_position]),
                    "hospital",
                    len(hospital_id),
                )
                occupied[h_position].add(
                    int(encounter["bed_number"][encounter_position])
                )
            bed_number = np.full(n_new_encounters, -1, dtype=np.int32)
            keep = np.ones(n_new_encounters, dtype=np.bool_)
            for index, h_position in enumerate(patient_hospital_position):
                free = next(
                    (
                        bed
                        for bed in range(int(hospital_beds[h_position]))
                        if bed not in occupied[h_position]
                    ),
                    None,
                )
                if free is None:
                    keep[index] = False
                    continue
                bed_number[index] = free
                if duration_months[index] > 0:
                    occupied[h_position].add(free)
            if not keep.all():
                patient_position = patient_position[keep]
                patient_hospital_position = patient_hospital_position[keep]
                diagnosis = diagnosis[keep]
                service = service[keep]
                duration_months = duration_months[keep]
                scheduled_end = scheduled_end[keep]
                cost_cents = cost_cents[keep]
                bed_number = bed_number[keep]
                n_new_encounters = int(keep.sum())
                # Capacity filtering happens before persistent allocation so rejected
                # admissions cannot burn or gap never-reused encounter sequences.
                new_encounter_id = _new_entity_ids(
                    "encounter",
                    encounter["truth_encounter_id"],
                    n_new_encounters,
                )

            outcome = np.where(
                duration_months == 0,
                ENCOUNTER_OUTCOMES["discharged"],
                ENCOUNTER_OUTCOMES["open"],
            ).astype(np.int8)
            admission_batch_recorded = _report_tick(rng, tick, params)
            admission_batch_recorded = max(
                admission_batch_recorded, latest_encounter_discharge_recorded
            )
            for index, encounter_identifier in enumerate(new_encounter_id):
                admitted_recorded = admission_batch_recorded
                hospital_identifier = int(hospital_id[patient_hospital_position[index]])
                person_identifier = int(
                    person["truth_person_id"][patient_position[index]]
                )
                records.append(
                    _new_record(
                        EVENT_TYPES["encounter_admitted"],
                        tick,
                        admitted_recorded,
                        ENTITY_NAMESPACE["encounter"],
                        int(encounter_identifier),
                        CAUSE_CODES["health_need"],
                        order,
                        truth_encounter_id=int(encounter_identifier),
                        truth_person_id=person_identifier,
                        truth_hospital_id=hospital_identifier,
                        scheduled_end_tick=int(scheduled_end[index]),
                        service=int(service[index]),
                        diagnosis_group=int(diagnosis[index]),
                        outcome=ENCOUNTER_OUTCOMES["open"],
                        cost_cents=int(cost_cents[index]),
                        bed_number=int(bed_number[index]),
                    )
                )
                order += 1
                if duration_months[index] == 0:
                    # Same-month bed reuse requires the admission/discharge pair to
                    # remain adjacent in canonical order. Source-specific reporting
                    # asymmetry is introduced later by the observed-register layer.
                    discharged_recorded = admitted_recorded
                    records.append(
                        _new_record(
                            EVENT_TYPES["encounter_discharged"],
                            tick,
                            discharged_recorded,
                            ENTITY_NAMESPACE["encounter"],
                            int(encounter_identifier),
                            CAUSE_CODES["scheduled"],
                            order,
                            truth_encounter_id=int(encounter_identifier),
                            truth_person_id=person_identifier,
                            truth_hospital_id=hospital_identifier,
                            outcome=ENCOUNTER_OUTCOMES["discharged"],
                            bed_number=int(bed_number[index]),
                        )
                    )
                    order += 1
            _append_rows(
                encounter,
                {
                    "truth_encounter_id": new_encounter_id,
                    "truth_person_id": person["truth_person_id"][patient_position],
                    "truth_hospital_id": hospital_id[patient_hospital_position],
                    "scheduled_end_tick": scheduled_end.astype(np.int64),
                    "service": service,
                    "diagnosis_group": diagnosis,
                    "outcome": outcome,
                    "cost_cents": cost_cents,
                    "bed_number": np.where(duration_months == 0, -1, bed_number).astype(
                        np.int32
                    ),
                    "is_open": (duration_months > 0).astype(np.bool_),
                },
            )

        # Recompute exact housing counts after births, deaths, and moves.
        person = state["person"]
        household = state["household"]
        dwelling = state["dwelling"]
        active_household_position = np.flatnonzero(household["is_active"])
        resident_count = np.zeros(len(household["truth_household_id"]), dtype=np.int64)
        alive_household_position = (
            _sequence(person["truth_household_id"][person["is_alive"]]).astype(np.int64)
            - 1
        )
        np.add.at(resident_count, alive_household_position, 1)
        dwelling["resident_count"][:] = 0
        active_dwelling_position = (
            _sequence(household["truth_dwelling_id"][active_household_position]).astype(
                np.int64
            )
            - 1
        )
        dwelling["resident_count"][active_dwelling_position] = resident_count[
            active_household_position
        ].astype(np.int32)

    loop["order"] = order
    loop["household_last_move_tick"] = household_last_move_tick


def capture_branch(history: dict) -> dict:
    """The branch record a continuation member resumes from, or None if none was kept."""
    return history.get("branch")


def continuation_shocks(branch: dict, member: int, months: int) -> list[dict]:
    """The shock schedule one continuation runs under: the world's past, its own future.

    Years the branch has already lived through keep the world's realized schedule; every
    year after it is redrawn from the published family at the published annual rate, on a
    key that is a fresh tag rather than arithmetic on the root seed. This is the
    systematic risk in the sealed tail: without it the members differ only by demographic
    noise, and the liability distribution collapses to a width no reconstruction could
    resolve.
    """
    rate = float(branch.get("annual_shock_rate", ANNUAL_SHOCK_RATE))
    branch_month = int(branch["month"])
    first_future_year = branch_month // 12 if branch_month % 12 == 0 \
        else branch_month // 12 + 1
    past = [dict(shock) for shock in branch["context"]["shocks"]
            if int(shock["year"]) < first_future_year]
    rng = np.random.default_rng(np.random.SeedSequence(
        [int(branch["seed"]), CONTINUATION_DOMAIN, int(member), SHOCK_SUBSTREAM]))
    years = (branch_month + int(months) + 11) // 12 - first_future_year + 1
    return past + draw_annual_shocks(rng, first_future_year, max(years, 1), rate)


def continuation_events(branch: dict, member: int, months: int) -> dict:
    """Event table of one committed continuation, over the months after the branch.

    Every member shares the branch state exactly and draws its own months from
    ``SeedSequence([seed, CONTINUATION_DOMAIN, member, month])``, the same substream rule
    ``build_event_history`` uses with ``continuation_member``. The returned table holds
    only the months after the branch, which is all a reading pass over that window needs;
    members are never merged, since two of them hand one person identity to two different
    newborns.
    """
    member, months = int(member), int(months)
    if member < 0:
        raise ValueError("continuation member must be nonnegative")
    if months < 1:
        raise ValueError("a continuation needs at least one month")
    first = int(branch["month"]) + 1
    context = dict(branch["context"])
    context["continuation_member"] = member
    context["branch_month"] = int(branch["month"])
    context["shocks"] = continuation_shocks(branch, member, months)
    loop = {"state": _state_copy(branch["state"]), "records": [],
            "order": int(branch["order"]),
            "household_last_move_tick": np.array(branch["household_last_move_tick"],
                                                 copy=True)}
    _run_ledger_months(context, loop, first, first + months - 1)
    return _make_event_table(loop["records"], int(branch["n_events"]) + 1)



def build_event_history(
    microdata: dict,
    seed: int,
    identity_map: dict,
    dwellings: dict,
    businesses: dict,
    hospitals: dict,
    months: int = 24,
    params: EventHistoryParams = EventHistoryParams(),
    shocks: list[dict] | None = None,
    mechanisms: WorldMechanisms | None = None,
    continuation_member: int | None = None,
    branch_month: int | None = None,
    capture_month: int | None = None,
    shock_annual_rate: float = ANNUAL_SHOCK_RATE,
) -> dict:
    """Advance the institutional world in monthly, append-only truth events.

    ``mechanisms`` carries the world's local hazard coefficients and county effects.
    Without it the world is treated as a single neutral county, which is what the
    standalone ledger tests use; a packet always supplies the real one.

    ``continuation_member`` and ``branch_month`` build one committed continuation: months
    one to ``branch_month`` are the ledger's own stream, so every member shares that
    prefix byte for byte, and later months read the member's own substream.  A member is
    a complete history from the same snapshot, which is what the ledger validator
    requires, and members are never merged: two of them hand the same person identity to
    two different newborns, so any cross-member quantity keys on (member, entity).
    """
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise TypeError("seed must be an integer")
    if isinstance(months, bool) or not isinstance(months, (int, np.integer)):
        raise TypeError("months must be an integer")
    seed = int(seed)
    months = int(months)
    if seed < 0:
        raise ValueError("seed must be nonnegative")
    if months < 1:
        raise ValueError("months must be positive")
    if not isinstance(params, EventHistoryParams):
        raise TypeError("params must be EventHistoryParams")
    _validate_params(params)
    if (continuation_member is None) != (branch_month is None):
        raise ValueError("a continuation needs both a member and a branch month")
    if continuation_member is not None:
        continuation_member = int(continuation_member)
        branch_month = int(branch_month)
        if continuation_member < 0:
            raise ValueError("continuation member must be nonnegative")
        if not 0 <= branch_month < months:
            raise ValueError("branch month must leave at least one month to continue")
    if capture_month is not None:
        capture_month = int(capture_month)
        if not 0 < capture_month <= months:
            raise ValueError("capture month must fall inside the ledger")

    validate_dwelling_conservation(dwellings, microdata, identity_map)
    validate_business_conservation(businesses, microdata, identity_map, seed)
    validate_hospital_conservation(hospitals, microdata, identity_map, businesses, seed)
    generator_version = int(identity_map["generator_version"])
    identity_world_id = np.uint64(identity_map["truth_world_id"])
    if identity_world_id != truth_world_id(seed, generator_version):
        raise ValueError("seed does not match the identity map's truth world")

    demography = draw_world_character(seed)["demography"]
    if shocks is None:
        shocks = draw_world_shocks(seed, max(3, (months + 11) // 12 + 1))
    shocks = [dict(shock) for shock in shocks]
    initial_state = _initial_state(
        microdata, identity_map, dwellings, businesses, hospitals
    )
    state = _state_copy(initial_state)
    records: list[dict[str, int]] = []
    order = 0
    snapshot_tick = int(identity_map["snapshot_tick"])
    hospital = hospitals["hospital"]
    hospital_id = hospital["truth_hospital_id"]
    hospital_cell = hospital["cell"]
    hospital_beds = hospital["bed_count"].astype(np.int64)
    nearest_hospital_by_cell = _nearest_facility(
        np.asarray(microdata["urbanity"]).shape, hospital_cell
    )
    annual_encounter_rate = float(
        hospitals["hospital_params"]["annual_encounters_per_1000"]
    )
    payroll_level = float(businesses["business_params"]["payroll_level"])
    if mechanisms is None:
        mechanisms = build_world_mechanisms(seed, "development", None, microdata)
    if not isinstance(mechanisms, WorldMechanisms):
        raise TypeError("mechanisms must be a WorldMechanisms")
    coefficients = mechanisms.coefficients
    urbanity_flat = np.asarray(microdata["urbanity"], dtype=np.float64).reshape(-1)
    household_last_move_tick = np.full(
        len(state["household"]["truth_household_id"]), snapshot_tick, dtype=np.int64
    )

    context = {
        "annual_encounter_rate": annual_encounter_rate,
        "branch_month": branch_month,
        "coefficients": coefficients,
        "continuation_member": continuation_member,
        "death_late_probability": death_report_late_probability(
            coefficients, params.late_report_probability),
        "demography": demography,
        "hospital_beds": hospital_beds,
        "hospital_id": hospital_id,
        "mechanisms": mechanisms,
        "nearest_hospital_by_cell": nearest_hospital_by_cell,
        "params": params,
        "payroll_level": payroll_level,
        "seed": seed,
        "shocks": shocks,
        "snapshot_tick": snapshot_tick,
        "urbanity_flat": urbanity_flat,
    }
    loop = {"state": state, "records": records, "order": order,
            "household_last_move_tick": household_last_move_tick}
    branch: dict | None = None
    if capture_month is None:
        _run_ledger_months(context, loop, 1, months)
    else:
        _run_ledger_months(context, loop, 1, capture_month)
        branch = {"month": int(capture_month),
                  "tick": int(snapshot_tick + capture_month),
                  "n_events": len(loop["records"]),
                  "seed": int(seed),
                  "annual_shock_rate": float(shock_annual_rate),
                  "context": context,
                  "state": _state_copy(loop["state"]),
                  "order": int(loop["order"]),
                  "household_last_move_tick": np.array(
                      loop["household_last_move_tick"], copy=True)}
        _run_ledger_months(context, loop, capture_month + 1, months)
    state, records, order = loop["state"], loop["records"], loop["order"]

    event_table = _make_event_table(records)
    history_without_terminal = {
        "truth_world_id": identity_world_id,
        "generator_version": generator_version,
        "snapshot_tick": np.int64(snapshot_tick),
        "terminal_tick": np.int64(snapshot_tick + months),
        "event_schema_version": 1,
        "event_params": _resolved_params_record(params, demography),
        "mechanism_record": mechanisms.record(),
        "shock_schedule": shocks,
        "initial_state": initial_state,
        "event": event_table,
        "n_events": len(event_table["truth_event_id"]),
    }
    replayed = replay_event_history(history_without_terminal)
    _assert_states_equal(state, replayed, "generated state differs from event replay")
    history = {**history_without_terminal, "terminal_state": replayed}
    if branch is not None:
        history["branch"] = branch
    validate_event_history(
        history, microdata, identity_map, dwellings, businesses, hospitals, seed
    )
    return history


def _allocate_replay_state(initial: dict, event: dict, include: np.ndarray) -> dict:
    start_events = {
        "person": EVENT_TYPES["person_birth"],
        "household": EVENT_TYPES["household_formed"],
        "establishment": EVENT_TYPES["establishment_opened"],
        "job": EVENT_TYPES["job_started"],
        "encounter": EVENT_TYPES["encounter_admitted"],
    }
    state: dict[str, dict[str, np.ndarray]] = {}
    for table_name, columns in initial.items():
        extra = int(
            (include & (event["event_type"] == start_events.get(table_name, -1))).sum()
        )
        size = len(next(iter(columns.values()))) + extra
        table: dict[str, np.ndarray] = {}
        for name, values in columns.items():
            output = np.zeros(size, dtype=values.dtype)
            output[: len(values)] = values
            if output.dtype.kind in "iu" and name in {
                "cell",
                "birth_tick",
                "scheduled_end_tick",
                "occupation",
                "employment_type",
                "service",
                "diagnosis_group",
                "outcome",
                "bed_number",
            }:
                output[len(values) :] = -1
            table[name] = output
        state[table_name] = table
    return state


def replay_event_history(history: dict, through_tick: int | None = None) -> dict:
    """Materialize current state from the initial snapshot and effective events."""
    event = history["event"]
    initial = history["initial_state"]
    if through_tick is None:
        through_tick = int(history["terminal_tick"])
    if isinstance(through_tick, bool) or not isinstance(
        through_tick, (int, np.integer)
    ):
        raise TypeError("through_tick must be an integer")
    through_tick = int(through_tick)
    if through_tick < int(history["snapshot_tick"]):
        raise ValueError("through_tick precedes the initial snapshot")

    superseded = set(
        int(value) for value in event["supersedes_event_id"] if int(value) != 0
    )
    include = event["tick"] <= through_tick
    if superseded:
        include &= ~np.isin(
            event["truth_event_id"], np.asarray(sorted(superseded), dtype=np.uint64)
        )
    state = _allocate_replay_state(initial, event, include)
    next_row = {
        name: len(initial[name][next(iter(initial[name]))])
        for name in ("person", "household", "establishment", "job", "encounter")
    }
    active_job_by_person = np.full(
        len(state["person"]["truth_person_id"]), -1, dtype=np.int64
    )
    initial_active_job = np.flatnonzero(state["job"]["is_active"])
    initial_job_person = (
        _sequence(state["job"]["truth_person_id"][initial_active_job]).astype(np.int64)
        - 1
    )
    active_job_by_person[initial_job_person] = initial_active_job
    occupied_beds: set[tuple[int, int]] = set()
    active_encounter_by_person = np.full(
        len(state["person"]["truth_person_id"]), -1, dtype=np.int64
    )
    initial_open = np.flatnonzero(state["encounter"]["is_open"])
    for encounter_position in initial_open:
        person_position = _position(
            int(state["encounter"]["truth_person_id"][encounter_position]),
            "person",
            len(state["person"]["truth_person_id"]),
        )
        hospital_position = _position(
            int(state["encounter"]["truth_hospital_id"][encounter_position]),
            "hospital",
            np.iinfo(np.int32).max,
        )
        bed = int(state["encounter"]["bed_number"][encounter_position])
        active_encounter_by_person[person_position] = encounter_position
        occupied_beds.add((hospital_position, bed))

    for row in np.flatnonzero(include):
        event_type = int(event["event_type"][row])
        if event_type == EVENT_TYPES["person_birth"]:
            position = next_row["person"]
            identifier = int(event["truth_person_id"][row])
            if (
                _position(identifier, "person", len(state["person"]["truth_person_id"]))
                != position
            ):
                raise ValueError("birth does not allocate the next person identity")
            household_position = _position(
                int(event["truth_household_id"][row]),
                "household",
                next_row["household"],
            )
            if not state["household"]["is_active"][household_position]:
                raise ValueError("birth references an inactive household")
            state["person"]["truth_person_id"][position] = identifier
            state["person"]["truth_household_id"][position] = event[
                "truth_household_id"
            ][row]
            state["person"]["birth_tick"][position] = event["birth_tick"][row]
            state["person"]["sex"][position] = event["sex"][row]
            state["person"]["role"][position] = event["role"][row]
            state["person"]["education"][position] = event["education"][row]
            state["person"]["income_cents"][position] = event["income_cents"][row]
            state["person"]["frailty_centi"][position] = event["frailty_centi"][row]
            state["person"]["is_alive"][position] = True
            next_row["person"] += 1
        elif event_type == EVENT_TYPES["person_death"]:
            position = _position(
                int(event["truth_person_id"][row]), "person", next_row["person"]
            )
            if not state["person"]["is_alive"][position]:
                raise ValueError("death references a person who is not alive")
            if active_job_by_person[position] >= 0:
                raise ValueError("death occurs before the person's active job ends")
            if active_encounter_by_person[position] >= 0:
                raise ValueError("death occurs before the person's encounter closes")
            state["person"]["is_alive"][position] = False
        elif event_type == EVENT_TYPES["household_formed"]:
            position = next_row["household"]
            identifier = int(event["truth_household_id"][row])
            if (
                _position(
                    identifier,
                    "household",
                    len(state["household"]["truth_household_id"]),
                )
                != position
            ):
                raise ValueError(
                    "formation does not allocate the next household identity"
                )
            person_position = _position(
                int(event["truth_person_id"][row]), "person", next_row["person"]
            )
            prior_household = int(event["truth_prior_household_id"][row])
            if (
                int(state["person"]["truth_household_id"][person_position])
                != prior_household
            ):
                raise ValueError("household formation has the wrong prior household")
            dwelling_position = _position(
                int(event["truth_dwelling_id"][row]),
                "dwelling",
                len(state["dwelling"]["truth_dwelling_id"]),
            )
            if state["dwelling"]["is_occupied"][dwelling_position]:
                raise ValueError("household formation targets an occupied dwelling")
            state["household"]["truth_household_id"][position] = identifier
            state["household"]["truth_dwelling_id"][position] = event[
                "truth_dwelling_id"
            ][row]
            state["household"]["cell"][position] = state["dwelling"]["cell"][
                dwelling_position
            ]
            state["household"]["is_active"][position] = True
            state["dwelling"]["is_occupied"][dwelling_position] = True
            state["dwelling"]["truth_household_id"][dwelling_position] = identifier
            state["person"]["truth_household_id"][person_position] = identifier
            state["person"]["role"][person_position] = 0
            next_row["household"] += 1
        elif event_type == EVENT_TYPES["household_moved"]:
            household_position = _position(
                int(event["truth_household_id"][row]),
                "household",
                next_row["household"],
            )
            old_position = _position(
                int(event["truth_prior_dwelling_id"][row]),
                "dwelling",
                len(state["dwelling"]["truth_dwelling_id"]),
            )
            new_position = _position(
                int(event["truth_dwelling_id"][row]),
                "dwelling",
                len(state["dwelling"]["truth_dwelling_id"]),
            )
            if int(state["household"]["truth_dwelling_id"][household_position]) != int(
                event["truth_prior_dwelling_id"][row]
            ):
                raise ValueError("household move has the wrong prior dwelling")
            if state["dwelling"]["is_occupied"][new_position]:
                raise ValueError(
                    "household move targets an occupied dwelling "
                    f"at event row {row}, tick {int(event['tick'][row])}, "
                    f"household {int(event['truth_household_id'][row])}, "
                    f"dwelling {int(event['truth_dwelling_id'][row])}"
                )
            state["dwelling"]["is_occupied"][old_position] = False
            state["dwelling"]["truth_household_id"][old_position] = 0
            state["dwelling"]["is_occupied"][new_position] = True
            state["dwelling"]["truth_household_id"][new_position] = event[
                "truth_household_id"
            ][row]
            state["household"]["truth_dwelling_id"][household_position] = event[
                "truth_dwelling_id"
            ][row]
            state["household"]["cell"][household_position] = state["dwelling"]["cell"][
                new_position
            ]
        elif event_type == EVENT_TYPES["household_closed"]:
            household_position = _position(
                int(event["truth_household_id"][row]),
                "household",
                next_row["household"],
            )
            if not state["household"]["is_active"][household_position]:
                raise ValueError("household closes more than once")
            dwelling_position = _position(
                int(event["truth_dwelling_id"][row]),
                "dwelling",
                len(state["dwelling"]["truth_dwelling_id"]),
            )
            state["household"]["is_active"][household_position] = False
            state["dwelling"]["is_occupied"][dwelling_position] = False
            state["dwelling"]["truth_household_id"][dwelling_position] = 0
        elif event_type == EVENT_TYPES["establishment_opened"]:
            position = next_row["establishment"]
            identifier = int(event["truth_establishment_id"][row])
            if (
                _position(
                    identifier,
                    "establishment",
                    len(state["establishment"]["truth_establishment_id"]),
                )
                != position
            ):
                raise ValueError(
                    "opening does not allocate the next establishment identity"
                )
            state["establishment"]["truth_establishment_id"][position] = identifier
            state["establishment"]["truth_enterprise_id"][position] = event[
                "truth_enterprise_id"
            ][row]
            state["establishment"]["cell"][position] = event["to_cell"][row]
            state["establishment"]["industry"][position] = event["industry"][row]
            state["establishment"]["is_hospital"][position] = False
            state["establishment"]["is_active"][position] = True
            next_row["establishment"] += 1
        elif event_type == EVENT_TYPES["establishment_closed"]:
            position = _position(
                int(event["truth_establishment_id"][row]),
                "establishment",
                next_row["establishment"],
            )
            if state["establishment"]["is_hospital"][position]:
                raise ValueError(
                    "hospital establishment cannot close in event-history v0"
                )
            if not state["establishment"]["is_active"][position]:
                raise ValueError("establishment closes more than once")
            linked_active = state["job"]["is_active"][: next_row["job"]] & (
                state["job"]["truth_establishment_id"][: next_row["job"]]
                == event["truth_establishment_id"][row]
            )
            if linked_active.any():
                raise ValueError("establishment closes with active jobs")
            state["establishment"]["is_active"][position] = False
        elif event_type == EVENT_TYPES["job_started"]:
            position = next_row["job"]
            identifier = int(event["truth_job_id"][row])
            if (
                _position(identifier, "job", len(state["job"]["truth_job_id"]))
                != position
            ):
                raise ValueError("job start does not allocate the next job identity")
            person_position = _position(
                int(event["truth_person_id"][row]), "person", next_row["person"]
            )
            establishment_position = _position(
                int(event["truth_establishment_id"][row]),
                "establishment",
                next_row["establishment"],
            )
            if not state["person"]["is_alive"][person_position]:
                raise ValueError("job starts for a person who is not alive")
            if active_job_by_person[person_position] >= 0:
                raise ValueError(
                    "job starts while the person already has an active job"
                )
            if not state["establishment"]["is_active"][establishment_position]:
                raise ValueError("job starts at an inactive establishment")
            for name in (
                "truth_job_id",
                "truth_person_id",
                "truth_establishment_id",
                "occupation",
                "employment_type",
                "annual_hours",
                "hourly_wage_cents",
            ):
                state["job"][name][position] = event[name][row]
            state["job"]["annual_earnings_cents"][position] = int(
                event["annual_hours"][row]
            ) * int(event["hourly_wage_cents"][row])
            state["job"]["is_active"][position] = True
            active_job_by_person[person_position] = position
            next_row["job"] += 1
        elif event_type == EVENT_TYPES["job_ended"]:
            position = _position(
                int(event["truth_job_id"][row]), "job", next_row["job"]
            )
            if not state["job"]["is_active"][position]:
                raise ValueError("job ends more than once")
            person_position = _position(
                int(state["job"]["truth_person_id"][position]),
                "person",
                next_row["person"],
            )
            state["job"]["is_active"][position] = False
            active_job_by_person[person_position] = -1
        elif event_type == EVENT_TYPES["encounter_admitted"]:
            position = next_row["encounter"]
            identifier = int(event["truth_encounter_id"][row])
            if (
                _position(
                    identifier,
                    "encounter",
                    len(state["encounter"]["truth_encounter_id"]),
                )
                != position
            ):
                raise ValueError(
                    "admission does not allocate the next encounter identity"
                )
            person_position = _position(
                int(event["truth_person_id"][row]), "person", next_row["person"]
            )
            if not state["person"]["is_alive"][person_position]:
                raise ValueError("encounter admits a person who is not alive")
            if active_encounter_by_person[person_position] >= 0:
                raise ValueError("person has overlapping open encounters")
            hospital_position = _position(
                int(event["truth_hospital_id"][row]),
                "hospital",
                np.iinfo(np.int32).max,
            )
            bed = int(event["bed_number"][row])
            if (hospital_position, bed) in occupied_beds:
                raise ValueError("encounter admits into an occupied bed")
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
                state["encounter"][name][position] = event[name][row]
            state["encounter"]["is_open"][position] = True
            active_encounter_by_person[person_position] = position
            occupied_beds.add((hospital_position, bed))
            next_row["encounter"] += 1
        elif event_type == EVENT_TYPES["encounter_discharged"]:
            position = _position(
                int(event["truth_encounter_id"][row]),
                "encounter",
                next_row["encounter"],
            )
            if not state["encounter"]["is_open"][position]:
                raise ValueError("encounter discharges more than once")
            person_position = _position(
                int(state["encounter"]["truth_person_id"][position]),
                "person",
                next_row["person"],
            )
            hospital_position = _position(
                int(state["encounter"]["truth_hospital_id"][position]),
                "hospital",
                np.iinfo(np.int32).max,
            )
            bed = int(state["encounter"]["bed_number"][position])
            state["encounter"]["is_open"][position] = False
            state["encounter"]["bed_number"][position] = -1
            state["encounter"]["outcome"][position] = event["outcome"][row]
            active_encounter_by_person[person_position] = -1
            occupied_beds.discard((hospital_position, bed))
        else:
            raise ValueError(f"unknown event type {event_type}")

    for table_name in next_row:
        expected = len(state[table_name][next(iter(state[table_name]))])
        if next_row[table_name] != expected:
            raise ValueError(f"replay did not fill the allocated {table_name} rows")

    household_position = (
        _sequence(state["person"]["truth_household_id"]).astype(np.int64) - 1
    )
    if np.any(household_position < 0) or np.any(
        household_position >= len(state["household"]["truth_household_id"])
    ):
        raise ValueError("person references a nonexistent household after replay")
    state["person"]["cell"] = state["household"]["cell"][household_position].copy()
    state["dwelling"]["resident_count"][:] = 0
    alive_household = household_position[state["person"]["is_alive"]]
    household_residents = np.bincount(
        alive_household, minlength=len(state["household"]["truth_household_id"])
    )
    active_household = np.flatnonzero(state["household"]["is_active"])
    active_dwelling = (
        _sequence(state["household"]["truth_dwelling_id"][active_household]).astype(
            np.int64
        )
        - 1
    )
    state["dwelling"]["resident_count"][active_dwelling] = household_residents[
        active_household
    ].astype(np.int32)
    return state


def events_visible_at(history: dict, vintage_tick: int) -> dict[str, np.ndarray]:
    """Return the canonical event subsequence visible to one source vintage."""
    if isinstance(vintage_tick, bool) or not isinstance(
        vintage_tick, (int, np.integer)
    ):
        raise TypeError("vintage_tick must be an integer")
    event = history["event"]
    visible = event["recorded_tick"] <= int(vintage_tick)
    return {name: values[visible].copy() for name, values in event.items()}


def _validate_state_conservation(
    state: dict, hospitals: dict, initial: dict, event: dict
) -> None:
    for table_name, expected in _STATE_DTYPES.items():
        table = state[table_name]
        n_rows = len(next(iter(table.values())))
        _table_columns(table, expected, n_rows, f"state.{table_name}")

    person = state["person"]
    household = state["household"]
    dwelling = state["dwelling"]
    establishment = state["establishment"]
    job = state["job"]
    encounter = state["encounter"]
    for table_name, id_name, entity in (
        ("person", "truth_person_id", "person"),
        ("household", "truth_household_id", "household"),
        ("dwelling", "truth_dwelling_id", "dwelling"),
        ("establishment", "truth_establishment_id", "establishment"),
        ("job", "truth_job_id", "job"),
        ("encounter", "truth_encounter_id", "encounter"),
    ):
        identifiers = state[table_name][id_name]
        if len(np.unique(identifiers)) != len(identifiers):
            raise ValueError(f"state {table_name} identities are not unique")
        if np.any(entity_namespace(identifiers) != ENTITY_NAMESPACE[entity]):
            raise ValueError(f"state {table_name} uses the wrong identity namespace")
        if not np.array_equal(
            _sequence(identifiers), np.arange(1, len(identifiers) + 1)
        ):
            raise ValueError(f"state {table_name} identities are not contiguous")

    active_household = np.flatnonzero(household["is_active"])
    occupied_dwelling = np.flatnonzero(dwelling["is_occupied"])
    if len(active_household) != len(occupied_dwelling):
        raise ValueError("active households do not equal occupied dwellings")
    active_dwelling_id = household["truth_dwelling_id"][active_household]
    if len(np.unique(active_dwelling_id)) != len(active_dwelling_id):
        raise ValueError("active households share a dwelling")
    dwelling_position = _sequence(active_dwelling_id).astype(np.int64) - 1
    if not np.array_equal(
        dwelling["truth_household_id"][dwelling_position],
        household["truth_household_id"][active_household],
    ):
        raise ValueError("household-to-dwelling links do not reconcile")
    if not np.array_equal(
        dwelling["cell"][dwelling_position], household["cell"][active_household]
    ):
        raise ValueError("household and dwelling geography differ")
    if np.any(dwelling["truth_household_id"][~dwelling["is_occupied"]] != 0):
        raise ValueError("vacant dwelling carries a household")
    if int(dwelling["resident_count"].sum()) != int(person["is_alive"].sum()):
        raise ValueError("dwelling residents do not conserve the living population")
    person_household_position = (
        _sequence(person["truth_household_id"]).astype(np.int64) - 1
    )
    if np.any(~household["is_active"][person_household_position[person["is_alive"]]]):
        raise ValueError("living person belongs to an inactive household")
    if not np.array_equal(person["cell"], household["cell"][person_household_position]):
        raise ValueError("person and household geography differ")

    active_job = np.flatnonzero(job["is_active"])
    job_person_position = (
        _sequence(job["truth_person_id"][active_job]).astype(np.int64) - 1
    )
    job_establishment_position = (
        _sequence(job["truth_establishment_id"][active_job]).astype(np.int64) - 1
    )
    if np.any(~person["is_alive"][job_person_position]):
        raise ValueError("active job belongs to a person who is not alive")
    if np.any(~establishment["is_active"][job_establishment_position]):
        raise ValueError("active job belongs to an inactive establishment")
    if len(np.unique(job["truth_person_id"][active_job])) != len(active_job):
        raise ValueError("person has multiple active jobs")
    if not np.array_equal(
        job["annual_earnings_cents"],
        job["annual_hours"].astype(np.int64) * job["hourly_wage_cents"],
    ):
        raise ValueError("job earnings do not equal hours times wage")
    active_establishment = np.flatnonzero(establishment["is_active"])
    employment = np.bincount(
        job_establishment_position, minlength=len(establishment["is_active"])
    )
    if np.any(employment[active_establishment] < 1):
        raise ValueError("active establishment has no active job")
    if np.any(~establishment["is_active"] & establishment["is_hospital"]):
        raise ValueError("event-history v0 closed a hospital establishment")

    hospital_id = hospitals["hospital"]["truth_hospital_id"]
    bed_count = hospitals["hospital"]["bed_count"]
    open_encounter = np.flatnonzero(encounter["is_open"])
    encounter_person_position = (
        _sequence(encounter["truth_person_id"][open_encounter]).astype(np.int64) - 1
    )
    if np.any(~person["is_alive"][encounter_person_position]):
        raise ValueError("open encounter belongs to a person who is not alive")
    if len(np.unique(encounter["truth_person_id"][open_encounter])) != len(
        open_encounter
    ):
        raise ValueError("person has overlapping open encounters")
    hospital_position = (
        _sequence(encounter["truth_hospital_id"][open_encounter]).astype(np.int64) - 1
    )
    if np.any(hospital_position < 0) or np.any(hospital_position >= len(hospital_id)):
        raise ValueError("open encounter references a nonexistent hospital")
    if np.any(encounter["bed_number"][open_encounter] < 0) or np.any(
        encounter["bed_number"][open_encounter] >= bed_count[hospital_position]
    ):
        raise ValueError("open encounter bed is outside hospital capacity")
    bed_pairs = np.column_stack(
        (hospital_position, encounter["bed_number"][open_encounter])
    )
    if len(np.unique(bed_pairs, axis=0)) != len(open_encounter):
        raise ValueError("open encounters share a hospital bed")
    if np.any(encounter["bed_number"][~encounter["is_open"]] != -1):
        raise ValueError("closed encounter retains an occupied bed")

    births = int((event["event_type"] == EVENT_TYPES["person_birth"]).sum())
    deaths = int((event["event_type"] == EVENT_TYPES["person_death"]).sum())
    if (
        len(person["truth_person_id"])
        != len(initial["person"]["truth_person_id"]) + births
    ):
        raise ValueError("birth events do not reconcile to person identities")
    if int(person["is_alive"].sum()) != (
        len(initial["person"]["truth_person_id"]) + births - deaths
    ):
        raise ValueError("births and deaths do not conserve population")


def validate_event_history(
    history: dict,
    microdata: dict,
    identity_map: dict,
    dwellings: dict,
    businesses: dict,
    hospitals: dict,
    seed: int,
) -> None:
    """Fail unless the ledger is canonical, cross-world-safe, and exactly replayable."""
    validate_dwelling_conservation(dwellings, microdata, identity_map)
    validate_business_conservation(businesses, microdata, identity_map, seed)
    validate_hospital_conservation(hospitals, microdata, identity_map, businesses, seed)
    generator_version = int(identity_map["generator_version"])
    identity_world_id = truth_world_id(seed, generator_version)
    try:
        event = history["event"]
        initial = history["initial_state"]
        terminal = history["terminal_state"]
        n_events = int(history["n_events"])
        snapshot_tick = int(history["snapshot_tick"])
        terminal_tick = int(history["terminal_tick"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("event-history metadata is incomplete") from exc
    if np.uint64(history["truth_world_id"]) != identity_world_id:
        raise ValueError("event history belongs to a different truth world")
    if int(history["generator_version"]) != generator_version:
        raise ValueError("event history uses a different generator version")
    if snapshot_tick != int(identity_map["snapshot_tick"]):
        raise ValueError("event history and identity snapshot ticks differ")
    if terminal_tick <= snapshot_tick:
        raise ValueError("event-history terminal tick must follow the snapshot")
    if int(history["event_schema_version"]) != 1:
        raise ValueError("unsupported event-history schema version")
    _table_columns(event, _EVENT_DTYPES, n_events, "event")
    expected_initial = _initial_state(
        microdata, identity_map, dwellings, businesses, hospitals
    )
    _assert_states_equal(initial, expected_initial, "initial event state was altered")

    event_id = event["truth_event_id"]
    if not np.array_equal(event_id, truth_entity_ids("event", n_events)):
        raise ValueError("event identities are not canonical and contiguous")
    if np.any(event["recorded_tick"] < event["tick"]):
        raise ValueError("event is recorded before it takes effect")
    if np.any(event["tick"] <= snapshot_tick) or np.any(event["tick"] > terminal_tick):
        raise ValueError("event tick is outside the history interval")
    order = np.lexsort((event_id, event["tick"]))
    if not np.array_equal(order, np.arange(n_events)):
        raise ValueError("event ledger is not in canonical order")
    if not np.isin(event["event_type"], np.asarray(list(EVENT_TYPES.values()))).all():
        raise ValueError("event ledger contains an unknown event type")
    if not np.isin(event["cause_code"], np.asarray(list(CAUSE_CODES.values()))).all():
        raise ValueError("event ledger contains an unknown cause code")
    if np.any(event["supersedes_event_id"] != 0):
        supersedes_position = (
            _sequence(
                event["supersedes_event_id"][event["supersedes_event_id"] != 0]
            ).astype(np.int64)
            - 1
        )
        correcting_position = np.flatnonzero(event["supersedes_event_id"] != 0)
        if np.any(supersedes_position >= correcting_position):
            raise ValueError("correction does not supersede an earlier event")
        if not np.array_equal(
            event["entity_type"][supersedes_position],
            event["entity_type"][correcting_position],
        ) or not np.array_equal(
            event["truth_entity_id"][supersedes_position],
            event["truth_entity_id"][correcting_position],
        ):
            raise ValueError("correction changes the subject of an event")

    subject_field = {
        EVENT_TYPES["person_birth"]: ("truth_person_id", "person"),
        EVENT_TYPES["person_death"]: ("truth_person_id", "person"),
        EVENT_TYPES["household_formed"]: ("truth_household_id", "household"),
        EVENT_TYPES["household_moved"]: ("truth_household_id", "household"),
        EVENT_TYPES["household_closed"]: ("truth_household_id", "household"),
        EVENT_TYPES["job_started"]: ("truth_job_id", "job"),
        EVENT_TYPES["job_ended"]: ("truth_job_id", "job"),
        EVENT_TYPES["establishment_opened"]: (
            "truth_establishment_id",
            "establishment",
        ),
        EVENT_TYPES["establishment_closed"]: (
            "truth_establishment_id",
            "establishment",
        ),
        EVENT_TYPES["encounter_admitted"]: (
            "truth_encounter_id",
            "encounter",
        ),
        EVENT_TYPES["encounter_discharged"]: (
            "truth_encounter_id",
            "encounter",
        ),
    }
    for event_type, (field, entity) in subject_field.items():
        selected = event["event_type"] == event_type
        if not np.array_equal(
            event["truth_entity_id"][selected], event[field][selected]
        ):
            raise ValueError("event subject does not match its typed payload")
        if np.any(event["entity_type"][selected] != ENTITY_NAMESPACE[entity]):
            raise ValueError("event subject uses the wrong entity namespace")

    replayed = replay_event_history(history)
    _assert_states_equal(terminal, replayed, "terminal event state differs from replay")
    _validate_state_conservation(terminal, hospitals, initial, event)
