"""Hospital, staffing, and encounter truth tables for Meridia.

Hospitals are institutional identities layered over active health-sector establishments.
Every staff relationship resolves to an existing job at that establishment.  Encounters
resolve to persistent people and hospitals, and current occupied beds are represented by
unique open encounter/bed pairs.  Observed health-register identifiers belong to the
later imperfect-register layer and deliberately do not exist here.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from meridia.businesses import INDUSTRIES, validate_business_conservation
from meridia.character import draw_world_character
from meridia.identities import ENTITY_NAMESPACE, entity_namespace, truth_entity_ids
from meridia.identities import truth_world_id

HEALTH_INDUSTRY = INDUSTRIES.index("health")

HOSPITAL_TYPES = {"community": 0, "general": 1, "referral": 2}
STAFF_ROLES = {
    "support": 0,
    "technical": 1,
    "nursing_allied": 2,
    "clinical_professional": 3,
}
ENCOUNTER_OUTCOMES = {
    "open": 0,
    "discharged": 1,
    "transferred": 2,
    "died": 3,
}


@dataclass(frozen=True)
class HospitalParams:
    beds_per_1000: float = 3.5
    mean_beds_per_hospital: float = 72.0
    occupancy_rate: float = 0.76
    annual_encounters_per_1000: float = 110.0
    history_days: int = 365


def hospital_params_from_character(character: Mapping[str, float]) -> HospitalParams:
    """Translate the truth-side world-character draw into hospital parameters."""
    try:
        beds_per_1000 = float(character["hospital_beds_per_1000"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("world character is missing hospital_beds_per_1000") from exc
    return HospitalParams(beds_per_1000=beds_per_1000)


def _params_record(params: HospitalParams) -> dict[str, float | int]:
    return {
        "beds_per_1000": float(params.beds_per_1000),
        "mean_beds_per_hospital": float(params.mean_beds_per_hospital),
        "occupancy_rate": float(params.occupancy_rate),
        "annual_encounters_per_1000": float(params.annual_encounters_per_1000),
        "history_days": int(params.history_days),
    }


def _params_from_record(record: Mapping[str, float | int]) -> HospitalParams:
    try:
        return HospitalParams(
            beds_per_1000=float(record["beds_per_1000"]),
            mean_beds_per_hospital=float(record["mean_beds_per_hospital"]),
            occupancy_rate=float(record["occupancy_rate"]),
            annual_encounters_per_1000=float(record["annual_encounters_per_1000"]),
            history_days=int(record["history_days"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("hospital parameter record is incomplete") from exc


def _validate_params(params: HospitalParams) -> None:
    real_values = (
        params.beds_per_1000,
        params.mean_beds_per_hospital,
        params.occupancy_rate,
        params.annual_encounters_per_1000,
    )
    if not np.isfinite(real_values).all():
        raise ValueError("hospital parameters must be finite")
    if params.beds_per_1000 <= 0.0:
        raise ValueError("beds_per_1000 must be positive")
    if params.mean_beds_per_hospital <= 0.0:
        raise ValueError("mean_beds_per_hospital must be positive")
    if not 0.0 <= params.occupancy_rate <= 1.0:
        raise ValueError("occupancy_rate must be in [0, 1]")
    if params.annual_encounters_per_1000 < 0.0:
        raise ValueError("annual_encounters_per_1000 cannot be negative")
    if isinstance(params.history_days, bool) or not isinstance(
        params.history_days, (int, np.integer)
    ):
        raise TypeError("history_days must be an integer")
    if params.history_days < 1:
        raise ValueError("history_days must be positive")


def _validate_inputs(
    microdata: dict,
    identity_map: dict,
    businesses: dict,
    seed: int,
) -> tuple:
    validate_business_conservation(businesses, microdata, identity_map, seed)
    try:
        person = microdata["person"]
        person_cell = np.asarray(person["cell"])
        person_age = np.asarray(person["age"])
        urbanity = np.asarray(microdata["urbanity"], dtype=np.float64)
        n_persons = int(microdata["n_persons"])
        truth_person_id = np.asarray(identity_map["identity"]["truth_person_id"])
        generator_version = int(identity_map["generator_version"])
        identity_world_id = np.uint64(identity_map["truth_world_id"])
        snapshot_tick = np.int64(identity_map["snapshot_tick"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("inputs do not satisfy the Meridia hospital schema") from exc

    if person_cell.ndim != 1 or person_age.ndim != 1:
        raise ValueError("person cell and age must be one-dimensional")
    if len(person_cell) != n_persons or len(person_age) != n_persons:
        raise ValueError("person columns do not match n_persons")
    if n_persons < 1:
        raise ValueError("hospital generation requires people")
    if urbanity.ndim != 2 or not np.isfinite(urbanity).all():
        raise ValueError("urbanity must be a finite two-dimensional grid")
    if np.any((urbanity < 0.0) | (urbanity > 1.0)):
        raise ValueError("urbanity must be in [0, 1]")
    person_cell = person_cell.astype(np.int64, copy=False)
    person_age = person_age.astype(np.int16, copy=False)
    if int(person_cell.min()) < 0 or int(person_cell.max()) >= urbanity.size:
        raise ValueError("person cell is outside the urbanity grid")
    if np.any(person_age < 0):
        raise ValueError("person age cannot be negative")
    if truth_person_id.dtype != np.uint64 or truth_person_id.ndim != 1:
        raise ValueError("truth_person_id must be a one-dimensional uint64 array")
    if len(truth_person_id) != n_persons:
        raise ValueError("truth person identities do not match n_persons")
    if np.any(entity_namespace(truth_person_id) != ENTITY_NAMESPACE["person"]):
        raise ValueError("person identities use the wrong entity namespace")
    if identity_world_id != truth_world_id(seed, generator_version):
        raise ValueError("seed does not match the identity map's truth world")

    return (
        person_cell,
        person_age,
        urbanity,
        truth_person_id,
        n_persons,
        generator_version,
        identity_world_id,
        snapshot_tick,
    )


def _weighted_sample_without_replacement(
    weight: np.ndarray, count: int, rng: np.random.Generator
) -> np.ndarray:
    weight = np.asarray(weight, dtype=np.float64)
    if weight.ndim != 1 or not np.isfinite(weight).all() or np.any(weight <= 0.0):
        raise ValueError("sampling weights must be a positive finite vector")
    if not 0 <= count <= len(weight):
        raise ValueError("sample count is outside the candidate vector")
    if count == 0:
        return np.empty(0, dtype=np.int64)
    uniform = np.maximum(rng.random(len(weight)), np.finfo(np.float64).tiny)
    priority = -np.log(uniform) / weight
    selected = np.argpartition(priority, count - 1)[:count]
    return np.sort(selected, kind="stable").astype(np.int64, copy=False)


def _local_population(population: np.ndarray) -> np.ndarray:
    """Three-by-three population catchment around every grid cell."""
    padded = np.pad(population, 1, mode="constant")
    local = np.zeros(population.shape, dtype=np.int64)
    height, width = population.shape
    for row_offset in range(3):
        for column_offset in range(3):
            local += padded[
                row_offset : row_offset + height,
                column_offset : column_offset + width,
            ]
    return local


def _health_candidates(businesses: dict) -> np.ndarray:
    """One active health establishment per cell, favoring its largest workplace."""
    establishment = businesses["establishment"]
    health = np.flatnonzero(
        (establishment["industry"] == HEALTH_INDUSTRY) & establishment["is_active"]
    )
    if len(health) < 1:
        raise ValueError("hospital generation requires a health-sector establishment")
    cell = establishment["cell"][health]
    employment = establishment["employment_count"][health].astype(np.int64)
    order = np.lexsort((health, -employment, cell))
    ordered = health[order]
    _, first = np.unique(establishment["cell"][ordered], return_index=True)
    return ordered[first].astype(np.int64, copy=False)


def _bed_and_hospital_targets(
    n_persons: int, n_candidates: int, params: HospitalParams
) -> tuple[int, int]:
    total_beds = max(1, int(round(n_persons * params.beds_per_1000 / 1000.0)))
    n_hospitals = max(
        1,
        min(
            n_candidates,
            int(round(total_beds / params.mean_beds_per_hospital)),
        ),
    )
    return max(total_beds, n_hospitals), n_hospitals


def _nearest_facility(
    grid_shape: tuple[int, int], facility_cell: np.ndarray
) -> np.ndarray:
    """Map every cell to its nearest facility with stable Chebyshev-distance ties."""
    facility_cell = np.asarray(facility_cell, dtype=np.int64)
    n_cells = grid_shape[0] * grid_shape[1]
    if facility_cell.ndim != 1 or len(facility_cell) < 1:
        raise ValueError("facility_cell must be a nonempty vector")
    if np.any(facility_cell < 0) or np.any(facility_cell >= n_cells):
        raise ValueError("facility cell is outside the grid")
    if len(np.unique(facility_cell)) != len(facility_cell):
        raise ValueError("hospital facility cells must be unique")

    distance = np.full(n_cells, np.iinfo(np.int32).max, dtype=np.int32)
    owner = np.full(n_cells, -1, dtype=np.int64)
    queue: deque[int] = deque()
    for facility_position, flat in enumerate(facility_cell):
        flat = int(flat)
        distance[flat] = 0
        owner[flat] = facility_position
        queue.append(flat)

    height, width = grid_shape
    neighbors = (
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, -1),
        (0, 1),
        (1, -1),
        (1, 0),
        (1, 1),
    )
    while queue:
        flat = queue.popleft()
        row, column = divmod(flat, width)
        candidate_distance = int(distance[flat]) + 1
        candidate_owner = int(owner[flat])
        for row_delta, column_delta in neighbors:
            next_row = row + row_delta
            next_column = column + column_delta
            if not 0 <= next_row < height or not 0 <= next_column < width:
                continue
            neighbor = next_row * width + next_column
            better_distance = candidate_distance < int(distance[neighbor])
            better_tie = candidate_distance == int(
                distance[neighbor]
            ) and candidate_owner < int(owner[neighbor])
            if better_distance or better_tie:
                distance[neighbor] = candidate_distance
                owner[neighbor] = candidate_owner
                queue.append(neighbor)
    if np.any(owner < 0):
        raise RuntimeError("hospital accessibility traversal left an unassigned cell")
    return owner


def _allocate_integer_total(
    weight: np.ndarray, total: int, minimum: int = 0
) -> np.ndarray:
    weight = np.asarray(weight, dtype=np.float64)
    if weight.ndim != 1 or len(weight) < 1:
        raise ValueError("allocation weight must be a nonempty vector")
    if not np.isfinite(weight).all() or np.any(weight <= 0.0):
        raise ValueError("allocation weights must be positive and finite")
    if total < minimum * len(weight):
        raise ValueError("allocation total is below the row minimum")
    allocation = np.full(len(weight), minimum, dtype=np.int64)
    remaining = total - int(allocation.sum())
    if remaining:
        shares = weight * (remaining / weight.sum())
        floors = np.floor(shares).astype(np.int64)
        allocation += floors
        remainder = total - int(allocation.sum())
        if remainder:
            order = np.argsort(-(shares - floors), kind="stable")
            allocation[order[:remainder]] += 1
    return allocation


def _table_columns(
    table: dict, expected: dict[str, np.dtype], n_rows: int, table_name: str
) -> None:
    for name, expected_dtype in expected.items():
        if name not in table:
            raise ValueError(f"{table_name} table is missing {name}")
        values = np.asarray(table[name])
        if values.ndim != 1 or len(values) != n_rows:
            raise ValueError(f"{table_name} column {name} has the wrong shape")
        if values.dtype != expected_dtype:
            raise ValueError(
                f"{table_name} column {name} has dtype {values.dtype}, "
                f"expected {expected_dtype}"
            )


def build_hospitals(
    microdata: dict,
    seed: int,
    identity_map: dict,
    businesses: dict,
    params: HospitalParams | None = None,
) -> dict:
    """Build hospital facilities, their existing staff jobs, and recent encounters."""
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise TypeError("seed must be an integer")
    seed = int(seed)
    if seed < 0:
        raise ValueError("seed must be nonnegative")
    if params is None:
        params = hospital_params_from_character(draw_world_character(seed)["business"])
    if not isinstance(params, HospitalParams):
        raise TypeError("params must be HospitalParams or None")
    _validate_params(params)

    (
        person_cell,
        person_age,
        urbanity,
        truth_person_id,
        n_persons,
        generator_version,
        identity_world_id,
        snapshot_tick,
    ) = _validate_inputs(microdata, identity_map, businesses, seed)
    enterprise = businesses["enterprise"]
    establishment = businesses["establishment"]
    job = businesses["job"]
    rng = np.random.default_rng(np.random.SeedSequence([seed, 0xA05F17A1]))

    population = np.bincount(person_cell, minlength=urbanity.size).reshape(
        urbanity.shape
    )
    local_population = _local_population(population).reshape(-1)
    candidates = _health_candidates(businesses)
    candidate_cell = establishment["cell"][candidates]
    candidate_staff = establishment["employment_count"][candidates].astype(np.float64)
    placement_weight = (
        (local_population[candidate_cell].astype(np.float64) + 1.0) ** 0.55
        * (candidate_staff + 1.0) ** 0.25
        * (0.30 + 0.70 * urbanity.reshape(-1)[candidate_cell])
    )
    total_beds, n_hospitals = _bed_and_hospital_targets(
        n_persons, len(candidates), params
    )
    selected_candidate = _weighted_sample_without_replacement(
        placement_weight, n_hospitals, rng
    )
    hospital_establishment_index = np.sort(
        candidates[selected_candidate], kind="stable"
    )
    hospital_cell = establishment["cell"][hospital_establishment_index]
    if len(np.unique(hospital_cell)) != n_hospitals:
        raise RuntimeError("hospital selection produced duplicate facility cells")

    nearest_hospital_by_cell = _nearest_facility(urbanity.shape, hospital_cell)
    person_hospital_position = nearest_hospital_by_cell[person_cell]
    catchment_population = np.bincount(
        person_hospital_position, minlength=n_hospitals
    ).astype(np.int32)

    establishment_id = establishment["truth_establishment_id"]
    job_establishment_index = np.searchsorted(
        establishment_id, job["truth_establishment_id"]
    )
    establishment_to_hospital = np.full(
        businesses["n_establishments"], -1, dtype=np.int64
    )
    establishment_to_hospital[hospital_establishment_index] = np.arange(
        n_hospitals, dtype=np.int64
    )
    job_hospital_position = establishment_to_hospital[job_establishment_index]
    staffing_mask = job_hospital_position >= 0
    staffing_hospital_position = job_hospital_position[staffing_mask]
    staffed_position_count = np.bincount(
        staffing_hospital_position, minlength=n_hospitals
    ).astype(np.int32)
    if np.any(staffed_position_count < 1):
        raise RuntimeError("selected hospital establishment has no active staff job")

    capacity_weight = (catchment_population.astype(np.float64) + 1.0) ** 0.55 * (
        staffed_position_count.astype(np.float64) + 1.0
    ) ** 0.45
    bed_count = _allocate_integer_total(capacity_weight, total_beds, minimum=1).astype(
        np.int32
    )
    hospital_type = np.select(
        (bed_count < 50, bed_count < 150),
        (HOSPITAL_TYPES["community"], HOSPITAL_TYPES["general"]),
        default=HOSPITAL_TYPES["referral"],
    ).astype(np.int8)

    hospital_id = truth_entity_ids("hospital", n_hospitals)
    enterprise_id = enterprise["truth_enterprise_id"]
    hospital_enterprise_position = np.searchsorted(
        enterprise_id,
        establishment["truth_enterprise_id"][hospital_establishment_index],
    )
    ownership = enterprise["ownership"][hospital_enterprise_position].astype(
        np.int8, copy=False
    )

    health_risk = (
        1.0
        + 0.012 * person_age.astype(np.float64)
        + 0.65 * (person_age >= 65)
        + 0.45 * (person_age < 5)
    )
    desired_open = np.rint(bed_count * params.occupancy_rate).astype(np.int32)
    desired_open = np.minimum(desired_open, bed_count)
    open_count = np.minimum(desired_open, catchment_population).astype(np.int32)

    person_order = np.argsort(person_hospital_position, kind="stable")
    catchment_start = np.cumsum(
        np.concatenate(
            (
                np.asarray([0], dtype=np.int64),
                catchment_population[:-1].astype(np.int64),
            )
        )
    )
    open_person_parts: list[np.ndarray] = []
    open_hospital_parts: list[np.ndarray] = []
    open_bed_parts: list[np.ndarray] = []
    for hospital_position, count in enumerate(open_count):
        count = int(count)
        if count == 0:
            continue
        start = int(catchment_start[hospital_position])
        catchment = person_order[
            start : start + int(catchment_population[hospital_position])
        ]
        selected = _weighted_sample_without_replacement(
            health_risk[catchment], count, rng
        )
        open_person_parts.append(catchment[selected])
        open_hospital_parts.append(np.full(count, hospital_position, dtype=np.int64))
        open_bed_parts.append(np.arange(count, dtype=np.int32))

    if open_person_parts:
        open_person_index = np.concatenate(open_person_parts)
        open_hospital_position = np.concatenate(open_hospital_parts)
        open_bed_number = np.concatenate(open_bed_parts)
    else:
        open_person_index = np.empty(0, dtype=np.int64)
        open_hospital_position = np.empty(0, dtype=np.int64)
        open_bed_number = np.empty(0, dtype=np.int32)
    n_open = len(open_person_index)

    n_closed = int(round(n_persons * params.annual_encounters_per_1000 / 1000.0))
    if n_closed:
        encounter_probability = health_risk / health_risk.sum()
        closed_person_index = rng.choice(
            n_persons, size=n_closed, replace=True, p=encounter_probability
        ).astype(np.int64, copy=False)
        closed_hospital_position = person_hospital_position[closed_person_index]
    else:
        closed_person_index = np.empty(0, dtype=np.int64)
        closed_hospital_position = np.empty(0, dtype=np.int64)

    encounter_person_index = np.concatenate((closed_person_index, open_person_index))
    encounter_hospital_position = np.concatenate(
        (closed_hospital_position, open_hospital_position)
    )
    n_encounters = len(encounter_person_index)
    is_open = np.zeros(n_encounters, dtype=np.bool_)
    is_open[n_closed:] = True
    bed_number = np.full(n_encounters, -1, dtype=np.int32)
    bed_number[n_closed:] = open_bed_number

    closed_admission = rng.integers(
        int(snapshot_tick) - params.history_days,
        int(snapshot_tick),
        size=n_closed,
        dtype=np.int64,
    )
    closed_stay = rng.integers(1, 15, size=n_closed, dtype=np.int64)
    closed_discharge = np.minimum(closed_admission + closed_stay, snapshot_tick).astype(
        np.int64
    )
    open_admission = (
        snapshot_tick - rng.integers(0, 15, size=n_open, dtype=np.int64)
    ).astype(np.int64)
    open_discharge = (
        snapshot_tick + rng.integers(1, 15, size=n_open, dtype=np.int64)
    ).astype(np.int64)
    admission_tick = np.concatenate((closed_admission, open_admission))
    discharge_tick = np.concatenate((closed_discharge, open_discharge))

    encounter_age = person_age[encounter_person_index]
    diagnosis_group = rng.integers(0, 8, size=n_encounters, dtype=np.int16)
    diagnosis_draw = rng.random(n_encounters)
    diagnosis_group[(encounter_age >= 65) & (diagnosis_draw < 0.42)] = 4
    diagnosis_group[(encounter_age < 5) & (diagnosis_draw < 0.38)] = 1
    service = (diagnosis_group % 4).astype(np.int8)
    outcome = np.full(n_encounters, ENCOUNTER_OUTCOMES["open"], dtype=np.int8)
    if n_closed:
        outcome_draw = rng.random(n_closed)
        outcome[:n_closed] = ENCOUNTER_OUTCOMES["discharged"]
        outcome[:n_closed][outcome_draw > 0.965] = ENCOUNTER_OUTCOMES["transferred"]
        outcome[:n_closed][outcome_draw > 0.992] = ENCOUNTER_OUTCOMES["died"]

    duration = np.where(
        is_open,
        snapshot_tick - admission_tick + 1,
        discharge_tick - admission_tick,
    ).astype(np.int64)
    daily_cost = np.asarray(
        [58_000, 72_000, 91_000, 118_000, 135_000, 83_000, 104_000, 67_000],
        dtype=np.int64,
    )
    cost_cents = (
        daily_cost[diagnosis_group] * duration
        + rng.integers(5_000, 30_001, size=n_encounters, dtype=np.int64)
    ).astype(np.int64)

    state = {
        "truth_world_id": identity_world_id,
        "generator_version": generator_version,
        "snapshot_tick": snapshot_tick,
        "hospital_params": _params_record(params),
        "hospital": {
            "truth_hospital_id": hospital_id,
            "truth_establishment_id": establishment_id[hospital_establishment_index],
            "cell": hospital_cell.astype(np.int64, copy=False),
            "hospital_type": hospital_type,
            "ownership": ownership,
            "bed_count": bed_count,
            "staffed_position_count": staffed_position_count,
            "occupied_bed_count": open_count,
            "catchment_population": catchment_population,
            "opening_year": establishment["opening_year"][
                hospital_establishment_index
            ].astype(np.int16, copy=False),
            "is_active": np.ones(n_hospitals, dtype=np.bool_),
        },
        "staffing": {
            "truth_hospital_id": hospital_id[staffing_hospital_position],
            "truth_job_id": job["truth_job_id"][staffing_mask],
            "staff_role": (job["occupation"][staffing_mask] % 4).astype(np.int8),
        },
        "encounter": {
            "truth_encounter_id": truth_entity_ids("encounter", n_encounters),
            "truth_person_id": truth_person_id[encounter_person_index],
            "truth_hospital_id": hospital_id[encounter_hospital_position],
            "admission_tick": admission_tick,
            "discharge_tick": discharge_tick,
            "service": service,
            "diagnosis_group": diagnosis_group,
            "outcome": outcome,
            "cost_cents": cost_cents,
            "bed_number": bed_number,
            "is_open": is_open,
        },
        "n_hospitals": n_hospitals,
        "n_staffing": int(staffing_mask.sum()),
        "n_encounters": n_encounters,
        "n_open_encounters": n_open,
        "total_beds": int(bed_count.sum()),
    }
    validate_hospital_conservation(state, microdata, identity_map, businesses, seed)
    return state


def validate_hospital_conservation(
    state: dict,
    microdata: dict,
    identity_map: dict,
    businesses: dict,
    seed: int,
) -> None:
    """Fail unless facilities, staff jobs, catchments, and occupied beds reconcile."""
    (
        person_cell,
        _,
        urbanity,
        truth_person_id,
        n_persons,
        generator_version,
        identity_world_id,
        snapshot_tick,
    ) = _validate_inputs(microdata, identity_map, businesses, seed)
    try:
        hospital = state["hospital"]
        staffing = state["staffing"]
        encounter = state["encounter"]
        n_hospitals = int(state["n_hospitals"])
        n_staffing = int(state["n_staffing"])
        n_encounters = int(state["n_encounters"])
        n_open = int(state["n_open_encounters"])
        total_beds = int(state["total_beds"])
        params = _params_from_record(state["hospital_params"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("hospital state metadata is incomplete") from exc
    _validate_params(params)

    hospital_dtypes = {
        "truth_hospital_id": np.dtype(np.uint64),
        "truth_establishment_id": np.dtype(np.uint64),
        "cell": np.dtype(np.int64),
        "hospital_type": np.dtype(np.int8),
        "ownership": np.dtype(np.int8),
        "bed_count": np.dtype(np.int32),
        "staffed_position_count": np.dtype(np.int32),
        "occupied_bed_count": np.dtype(np.int32),
        "catchment_population": np.dtype(np.int32),
        "opening_year": np.dtype(np.int16),
        "is_active": np.dtype(np.bool_),
    }
    staffing_dtypes = {
        "truth_hospital_id": np.dtype(np.uint64),
        "truth_job_id": np.dtype(np.uint64),
        "staff_role": np.dtype(np.int8),
    }
    encounter_dtypes = {
        "truth_encounter_id": np.dtype(np.uint64),
        "truth_person_id": np.dtype(np.uint64),
        "truth_hospital_id": np.dtype(np.uint64),
        "admission_tick": np.dtype(np.int64),
        "discharge_tick": np.dtype(np.int64),
        "service": np.dtype(np.int8),
        "diagnosis_group": np.dtype(np.int16),
        "outcome": np.dtype(np.int8),
        "cost_cents": np.dtype(np.int64),
        "bed_number": np.dtype(np.int32),
        "is_open": np.dtype(np.bool_),
    }
    _table_columns(hospital, hospital_dtypes, n_hospitals, "hospital")
    _table_columns(staffing, staffing_dtypes, n_staffing, "staffing")
    _table_columns(encounter, encounter_dtypes, n_encounters, "encounter")

    if np.uint64(state["truth_world_id"]) != identity_world_id:
        raise ValueError("hospital state belongs to a different truth world")
    if int(state["generator_version"]) != generator_version:
        raise ValueError("hospital state uses a different generator version")
    if np.int64(state["snapshot_tick"]) != snapshot_tick:
        raise ValueError("hospital state and identity snapshot ticks differ")
    if min(n_hospitals, n_staffing, n_encounters, total_beds) < 1:
        raise ValueError(
            "hospital state must contain facilities, staff, beds, and encounters"
        )

    enterprise = businesses["enterprise"]
    establishment = businesses["establishment"]
    job = businesses["job"]
    candidates = _health_candidates(businesses)
    expected_beds, expected_hospitals = _bed_and_hospital_targets(
        n_persons, len(candidates), params
    )
    if n_hospitals != expected_hospitals:
        raise ValueError("hospital count does not match the capacity parameters")
    if total_beds != expected_beds or int(hospital["bed_count"].sum()) != total_beds:
        raise ValueError("hospital beds do not match the world-character capacity dial")
    if np.any(hospital["bed_count"] < 1):
        raise ValueError("every hospital must have a positive bed capacity")
    if not hospital["is_active"].all():
        raise ValueError("initial hospitals must be active")

    hospital_id = hospital["truth_hospital_id"]
    if len(np.unique(hospital_id)) != n_hospitals:
        raise ValueError("truth hospital identities are not unique")
    if np.any(entity_namespace(hospital_id) != ENTITY_NAMESPACE["hospital"]):
        raise ValueError("hospital identities use the wrong entity namespace")
    establishment_id = establishment["truth_establishment_id"]
    hospital_establishment_position = np.searchsorted(
        establishment_id, hospital["truth_establishment_id"]
    )
    valid_establishment = hospital_establishment_position < len(establishment_id)
    if not valid_establishment.all() or not np.array_equal(
        establishment_id[hospital_establishment_position],
        hospital["truth_establishment_id"],
    ):
        raise ValueError("hospital references a nonexistent establishment")
    if len(np.unique(hospital_establishment_position)) != n_hospitals:
        raise ValueError("multiple hospitals reference one establishment")
    if not np.isin(hospital_establishment_position, candidates).all():
        raise ValueError("hospital does not use the designated health establishment")
    if np.any(
        establishment["industry"][hospital_establishment_position] != HEALTH_INDUSTRY
    ):
        raise ValueError("hospital establishment is outside the health sector")
    if not np.array_equal(
        hospital["cell"], establishment["cell"][hospital_establishment_position]
    ):
        raise ValueError("hospital and establishment locations differ")
    if len(np.unique(hospital["cell"])) != n_hospitals:
        raise ValueError("hospital locations are not unique")
    if np.any(hospital["cell"] < 0) or np.any(hospital["cell"] >= urbanity.size):
        raise ValueError("hospital cell is outside the world grid")

    enterprise_id = enterprise["truth_enterprise_id"]
    hospital_enterprise_position = np.searchsorted(
        enterprise_id,
        establishment["truth_enterprise_id"][hospital_establishment_position],
    )
    if not np.array_equal(
        hospital["ownership"], enterprise["ownership"][hospital_enterprise_position]
    ):
        raise ValueError("hospital ownership does not match its enterprise")
    if not np.array_equal(
        hospital["opening_year"],
        establishment["opening_year"][hospital_establishment_position],
    ):
        raise ValueError("hospital opening year does not match its establishment")
    expected_type = np.select(
        (hospital["bed_count"] < 50, hospital["bed_count"] < 150),
        (HOSPITAL_TYPES["community"], HOSPITAL_TYPES["general"]),
        default=HOSPITAL_TYPES["referral"],
    ).astype(np.int8)
    if not np.array_equal(hospital["hospital_type"], expected_type):
        raise ValueError("hospital type does not match bed capacity")

    nearest_hospital_by_cell = _nearest_facility(urbanity.shape, hospital["cell"])
    person_hospital_position = nearest_hospital_by_cell[person_cell]
    expected_catchment = np.bincount(
        person_hospital_position, minlength=n_hospitals
    ).astype(np.int32)
    if not np.array_equal(hospital["catchment_population"], expected_catchment):
        raise ValueError("hospital catchments do not conserve the population")
    if int(hospital["catchment_population"].sum()) != n_persons:
        raise ValueError("hospital catchments do not sum to the national population")

    staffing_hospital_position = np.searchsorted(
        hospital_id, staffing["truth_hospital_id"]
    )
    valid_staff_hospital = staffing_hospital_position < n_hospitals
    if not valid_staff_hospital.all() or not np.array_equal(
        hospital_id[staffing_hospital_position], staffing["truth_hospital_id"]
    ):
        raise ValueError("staffing references a nonexistent hospital")
    job_id = job["truth_job_id"]
    staffing_job_position = np.searchsorted(job_id, staffing["truth_job_id"])
    valid_staff_job = staffing_job_position < len(job_id)
    if not valid_staff_job.all() or not np.array_equal(
        job_id[staffing_job_position], staffing["truth_job_id"]
    ):
        raise ValueError("staffing references a nonexistent job")
    if len(np.unique(staffing["truth_job_id"])) != n_staffing:
        raise ValueError("a health-sector job staffs multiple hospitals")
    job_establishment_position = np.searchsorted(
        establishment_id, job["truth_establishment_id"][staffing_job_position]
    )
    if not np.array_equal(
        job_establishment_position,
        hospital_establishment_position[staffing_hospital_position],
    ):
        raise ValueError("hospital staff job belongs to a different establishment")
    if np.any(establishment["industry"][job_establishment_position] != HEALTH_INDUSTRY):
        raise ValueError("hospital staff job is outside the health sector")
    if np.any((staffing["staff_role"] < 0) | (staffing["staff_role"] > 3)):
        raise ValueError("hospital staff role is outside its codebook")
    expected_staff_count = np.bincount(
        staffing_hospital_position, minlength=n_hospitals
    ).astype(np.int32)
    if not np.array_equal(hospital["staffed_position_count"], expected_staff_count):
        raise ValueError("hospital staffing count does not equal linked health jobs")
    if np.any(expected_staff_count < 1):
        raise ValueError("initial hospital has no linked staff job")

    encounter_id = encounter["truth_encounter_id"]
    if len(np.unique(encounter_id)) != n_encounters:
        raise ValueError("truth encounter identities are not unique")
    if np.any(entity_namespace(encounter_id) != ENTITY_NAMESPACE["encounter"]):
        raise ValueError("encounter identities use the wrong entity namespace")
    encounter_person_position = np.searchsorted(
        truth_person_id, encounter["truth_person_id"]
    )
    valid_person = encounter_person_position < n_persons
    if not valid_person.all() or not np.array_equal(
        truth_person_id[encounter_person_position], encounter["truth_person_id"]
    ):
        raise ValueError("encounter references a nonexistent person")
    encounter_hospital_position = np.searchsorted(
        hospital_id, encounter["truth_hospital_id"]
    )
    valid_hospital = encounter_hospital_position < n_hospitals
    if not valid_hospital.all() or not np.array_equal(
        hospital_id[encounter_hospital_position], encounter["truth_hospital_id"]
    ):
        raise ValueError("encounter references a nonexistent hospital")
    expected_encounter_hospital = person_hospital_position[encounter_person_position]
    if not np.array_equal(encounter_hospital_position, expected_encounter_hospital):
        raise ValueError("encounter does not use the person's accessible hospital")

    is_open = encounter["is_open"]
    if int(is_open.sum()) != n_open:
        raise ValueError("open encounter metadata does not match encounter rows")
    if n_open != int(hospital["occupied_bed_count"].sum()):
        raise ValueError("occupied beds do not equal open encounters")
    expected_occupied = np.bincount(
        encounter_hospital_position[is_open], minlength=n_hospitals
    ).astype(np.int32)
    if not np.array_equal(hospital["occupied_bed_count"], expected_occupied):
        raise ValueError("hospital occupied beds do not equal its open encounters")
    if np.any(hospital["occupied_bed_count"] > hospital["bed_count"]):
        raise ValueError("hospital occupancy exceeds bed capacity")
    if len(np.unique(encounter["truth_person_id"][is_open])) != n_open:
        raise ValueError("one person occupies multiple hospital beds")
    if np.any(encounter["bed_number"][~is_open] != -1):
        raise ValueError("closed encounter retains a current bed number")
    open_bed = encounter["bed_number"][is_open]
    open_hospital = encounter_hospital_position[is_open]
    if np.any(open_bed < 0) or np.any(open_bed >= hospital["bed_count"][open_hospital]):
        raise ValueError("open encounter bed number is outside hospital capacity")
    if n_open:
        occupied_pairs = np.column_stack((open_hospital, open_bed))
        if len(np.unique(occupied_pairs, axis=0)) != n_open:
            raise ValueError("multiple open encounters occupy one hospital bed")

    if np.any(encounter["admission_tick"] >= encounter["discharge_tick"]):
        raise ValueError("encounter admission must precede discharge")
    if np.any(encounter["admission_tick"] > snapshot_tick):
        raise ValueError("encounter admission occurs after the snapshot")
    if np.any(encounter["discharge_tick"][is_open] <= snapshot_tick):
        raise ValueError("open encounter is discharged by the snapshot")
    if np.any(encounter["discharge_tick"][~is_open] > snapshot_tick):
        raise ValueError("closed encounter discharges after the snapshot")
    if np.any(encounter["outcome"][is_open] != ENCOUNTER_OUTCOMES["open"]):
        raise ValueError("open encounter has a final outcome")
    if np.any(encounter["outcome"][~is_open] == ENCOUNTER_OUTCOMES["open"]):
        raise ValueError("closed encounter is missing a final outcome")
    if np.any((encounter["service"] < 0) | (encounter["service"] > 3)):
        raise ValueError("encounter service is outside its codebook")
    if np.any((encounter["diagnosis_group"] < 0) | (encounter["diagnosis_group"] > 7)):
        raise ValueError("encounter diagnosis group is outside its codebook")
    if np.any(encounter["cost_cents"] <= 0):
        raise ValueError("encounter cost must be positive")
    for table_name, table in (
        ("hospital", hospital),
        ("staffing", staffing),
        ("encounter", encounter),
    ):
        if any(name.startswith("observed_") for name in table):
            raise ValueError(
                f"observed register ID leaked into the {table_name} truth table"
            )
