"""Enterprise, establishment, and job truth tables for Meridia.

An enterprise is the legal/control entity, an establishment is one physical operating
location, and a job links one person to one establishment.  Observed business-register
identifiers deliberately do not exist here; they belong to a later imperfect-register
layer after event history is available.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from meridia.character import draw_world_character
from meridia.identities import ENTITY_NAMESPACE, entity_namespace, truth_entity_ids
from meridia.identities import truth_world_id

INDUSTRIES = (
    "agriculture_extraction",
    "manufacturing",
    "construction",
    "trade",
    "transport",
    "professional_finance",
    "hospitality",
    "education",
    "health",
    "public_services",
)

LEGAL_FORMS = {
    "sole_proprietor": 0,
    "partnership": 1,
    "corporation": 2,
    "cooperative": 3,
    "public": 4,
}

OWNERSHIP = {
    "domestic_private": 0,
    "foreign_private": 1,
    "cooperative": 2,
    "public": 3,
}

ESTABLISHMENT_ROLES = {"headquarters": 0, "branch": 1}
EMPLOYMENT_TYPES = {"full_time": 0, "part_time": 1}


@dataclass(frozen=True)
class BusinessParams:
    jobs_per_adult: float = 0.675
    mean_jobs_per_establishment: float = 14.0
    establishment_size_alpha: float = 1.8
    multi_establishment_rate: float = 0.20
    payroll_level: float = 1.0
    minimum_work_age: int = 18
    maximum_work_age: int = 74
    snapshot_year: int = 2026
    minimum_opening_year: int = 1880


def business_params_from_character(character: Mapping[str, float]) -> BusinessParams:
    """Translate the truth-side world-character draw into business parameters."""
    required = (
        "jobs_per_adult",
        "establishment_size_alpha",
        "multi_establishment_rate",
        "payroll_level",
    )
    try:
        values = {name: float(character[name]) for name in required}
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("world character is missing a business dial") from exc
    return BusinessParams(**values)


def _params_record(params: BusinessParams) -> dict[str, float | int]:
    return {
        "jobs_per_adult": float(params.jobs_per_adult),
        "mean_jobs_per_establishment": float(params.mean_jobs_per_establishment),
        "establishment_size_alpha": float(params.establishment_size_alpha),
        "multi_establishment_rate": float(params.multi_establishment_rate),
        "payroll_level": float(params.payroll_level),
        "minimum_work_age": int(params.minimum_work_age),
        "maximum_work_age": int(params.maximum_work_age),
        "snapshot_year": int(params.snapshot_year),
        "minimum_opening_year": int(params.minimum_opening_year),
    }


def _params_from_record(record: Mapping[str, float | int]) -> BusinessParams:
    try:
        return BusinessParams(
            jobs_per_adult=float(record["jobs_per_adult"]),
            mean_jobs_per_establishment=float(record["mean_jobs_per_establishment"]),
            establishment_size_alpha=float(record["establishment_size_alpha"]),
            multi_establishment_rate=float(record["multi_establishment_rate"]),
            payroll_level=float(record["payroll_level"]),
            minimum_work_age=int(record["minimum_work_age"]),
            maximum_work_age=int(record["maximum_work_age"]),
            snapshot_year=int(record["snapshot_year"]),
            minimum_opening_year=int(record["minimum_opening_year"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("business parameter record is incomplete") from exc


def _validate_params(params: BusinessParams) -> None:
    real_values = (
        params.jobs_per_adult,
        params.mean_jobs_per_establishment,
        params.establishment_size_alpha,
        params.multi_establishment_rate,
        params.payroll_level,
    )
    if not np.isfinite(real_values).all():
        raise ValueError("business parameters must be finite")
    if not 0.0 < params.jobs_per_adult <= 1.0:
        raise ValueError("jobs_per_adult must be in (0, 1]")
    if params.mean_jobs_per_establishment <= 0.0:
        raise ValueError("mean_jobs_per_establishment must be positive")
    if params.establishment_size_alpha <= 1.0:
        raise ValueError("establishment_size_alpha must be greater than one")
    if not 0.0 <= params.multi_establishment_rate < 1.0:
        raise ValueError("multi_establishment_rate must be in [0, 1)")
    if params.payroll_level <= 0.0:
        raise ValueError("payroll_level must be positive")
    if not 0 <= params.minimum_work_age <= params.maximum_work_age:
        raise ValueError("working-age bounds are invalid")
    if params.minimum_opening_year > params.snapshot_year:
        raise ValueError("minimum_opening_year cannot exceed snapshot_year")
    if (
        not np.iinfo(np.int16).min
        <= params.minimum_opening_year
        <= np.iinfo(np.int16).max
    ):
        raise ValueError("minimum_opening_year must fit int16")
    if not np.iinfo(np.int16).min <= params.snapshot_year <= np.iinfo(np.int16).max:
        raise ValueError("snapshot_year must fit int16")


def _validate_inputs(microdata: dict, identity_map: dict, seed: int) -> tuple:
    try:
        person = microdata["person"]
        person_cell = np.asarray(person["cell"])
        person_age = np.asarray(person["age"])
        person_education = np.asarray(person["education"])
        person_income = np.asarray(person["income"], dtype=np.float64)
        urbanity = np.asarray(microdata["urbanity"], dtype=np.float64)
        n_persons = int(microdata["n_persons"])
        truth_person_id = np.asarray(identity_map["identity"]["truth_person_id"])
        generator_version = int(identity_map["generator_version"])
        identity_world_id = np.uint64(identity_map["truth_world_id"])
        snapshot_tick = np.int64(identity_map["snapshot_tick"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("inputs do not satisfy the Meridia business schema") from exc

    columns = (person_cell, person_age, person_education, person_income)
    if any(values.ndim != 1 or len(values) != n_persons for values in columns):
        raise ValueError("person columns do not match n_persons")
    if urbanity.ndim != 2:
        raise ValueError("urbanity must be a two-dimensional grid")
    person_cell = person_cell.astype(np.int64, copy=False)
    person_age = person_age.astype(np.int16, copy=False)
    person_education = person_education.astype(np.int8, copy=False)
    if n_persons < 1:
        raise ValueError("business generation requires people")
    if int(person_cell.min()) < 0 or int(person_cell.max()) >= urbanity.size:
        raise ValueError("person cell is outside the urbanity grid")
    if np.any(person_age < 0):
        raise ValueError("person age cannot be negative")
    if np.any(person_education < 0):
        raise ValueError("person education cannot be negative")
    if not np.isfinite(person_income).all() or np.any(person_income < 0.0):
        raise ValueError("person income must be finite and nonnegative")
    if not np.isfinite(urbanity).all() or np.any((urbanity < 0.0) | (urbanity > 1.0)):
        raise ValueError("urbanity must be finite and in [0, 1]")
    if truth_person_id.dtype != np.uint64 or truth_person_id.ndim != 1:
        raise ValueError("truth_person_id must be a one-dimensional uint64 array")
    if len(truth_person_id) != n_persons:
        raise ValueError("truth person identities do not match n_persons")
    if np.any(entity_namespace(truth_person_id) != ENTITY_NAMESPACE["person"]):
        raise ValueError("person identities use the wrong entity namespace")
    if len(np.unique(truth_person_id)) != n_persons:
        raise ValueError("truth person identities are not unique")
    if identity_world_id != truth_world_id(seed, generator_version):
        raise ValueError("seed does not match the identity map's truth world")

    return (
        person_cell,
        person_age,
        person_education,
        person_income,
        urbanity,
        truth_person_id,
        n_persons,
        generator_version,
        identity_world_id,
        snapshot_tick,
    )


def _nearest_source_cell(source: np.ndarray) -> np.ndarray:
    """Map every grid cell to a nearest True cell with deterministic tie-breaking."""
    if source.ndim != 2 or not source.any():
        raise ValueError("source must be a nonempty two-dimensional mask")
    height, width = source.shape
    n_cells = height * width
    distance = np.full(n_cells, np.iinfo(np.int32).max, dtype=np.int32)
    owner = np.full(n_cells, -1, dtype=np.int64)
    queue: deque[int] = deque()
    for flat in np.flatnonzero(source.reshape(-1)):
        flat = int(flat)
        distance[flat] = 0
        owner[flat] = flat
        queue.append(flat)

    neighbors = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))
    while queue:
        flat = queue.popleft()
        row, col = divmod(flat, width)
        candidate_distance = int(distance[flat]) + 1
        candidate_owner = int(owner[flat])
        for dr, dc in neighbors:
            nr, nc = row + dr, col + dc
            if not 0 <= nr < height or not 0 <= nc < width:
                continue
            neighbor = nr * width + nc
            better_distance = candidate_distance < int(distance[neighbor])
            better_tie = candidate_distance == int(
                distance[neighbor]
            ) and candidate_owner < int(owner[neighbor])
            if better_distance or better_tie:
                distance[neighbor] = candidate_distance
                owner[neighbor] = candidate_owner
                queue.append(neighbor)
    if np.any(owner < 0):
        raise RuntimeError("nearest-workplace traversal left an unassigned cell")
    return owner


def _weighted_sample_without_replacement(
    weight: np.ndarray, count: int, rng: np.random.Generator
) -> np.ndarray:
    """Select exactly ``count`` positions with deterministic weighted priorities."""
    weight = np.asarray(weight, dtype=np.float64)
    if weight.ndim != 1 or not np.isfinite(weight).all() or np.any(weight <= 0.0):
        raise ValueError("sampling weights must be a positive finite vector")
    if not 0 <= count <= len(weight):
        raise ValueError("sample count is outside the candidate vector")
    if count == 0:
        return np.empty(0, dtype=np.int64)
    uniform = np.maximum(rng.random(len(weight)), np.finfo(np.float64).tiny)
    priority = -np.log(uniform) / weight
    chosen = np.argpartition(priority, count - 1)[:count]
    return np.sort(chosen, kind="stable").astype(np.int64, copy=False)


def _establishment_anchors(
    home_cell: np.ndarray,
    urbanity_flat: np.ndarray,
    n_establishments: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Choose unique workers as geographic anchors, then sort locations canonically."""
    weight = 0.65 + 1.35 * urbanity_flat[home_cell]
    chosen = _weighted_sample_without_replacement(weight, n_establishments, rng)
    return np.sort(home_cell[chosen], kind="stable").astype(np.int64, copy=False)


def _allocate_workers(
    establishment_cell: np.ndarray,
    worker_home_cell: np.ndarray,
    grid_shape: tuple[int, int],
    establishment_size_alpha: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Assign every worker to one establishment and every establishment at least one."""
    n_cells = grid_shape[0] * grid_shape[1]
    establishments_per_cell = np.bincount(establishment_cell, minlength=n_cells)
    source = establishments_per_cell.reshape(grid_shape) > 0
    nearest = _nearest_source_cell(source)
    work_cell = nearest[worker_home_cell]

    establishment_start = np.cumsum(
        np.concatenate(
            (
                np.asarray([0], dtype=np.int64),
                establishments_per_cell[:-1].astype(np.int64),
            )
        )
    )
    worker_order = np.argsort(work_cell, kind="stable")
    sorted_work_cell = work_cell[worker_order]
    cells, starts, worker_counts = np.unique(
        sorted_work_cell, return_index=True, return_counts=True
    )
    job_establishment_index = np.empty(len(worker_home_cell), dtype=np.int64)
    # NumPy parameterizes Pareto by the survival exponent ``a`` while the public
    # character dial uses the conventional density exponent alpha = a + 1.
    size_weight = (
        rng.pareto(establishment_size_alpha - 1.0, len(establishment_cell)) + 1.0
    )

    for cell, start, n_workers in zip(cells, starts, worker_counts):
        cell = int(cell)
        start = int(start)
        n_workers = int(n_workers)
        n_local = int(establishments_per_cell[cell])
        if n_local < 1 or n_workers < n_local:
            raise RuntimeError("workplace allocation cannot staff every establishment")
        worker_positions = worker_order[start : start + n_workers]
        worker_positions = worker_positions[rng.permutation(n_workers)]

        allocation = np.ones(n_local, dtype=np.int64)
        remaining = n_workers - n_local
        if remaining:
            first_establishment = int(establishment_start[cell])
            weights = size_weight[first_establishment : first_establishment + n_local]
            shares = weights * (remaining / weights.sum())
            allocation += np.floor(shares).astype(np.int64)
            remainder = n_workers - int(allocation.sum())
            if remainder:
                order = np.argsort(-(shares - np.floor(shares)), kind="stable")
                allocation[order[:remainder]] += 1

        cursor = 0
        first_establishment = int(establishment_start[cell])
        for local_index, count in enumerate(allocation):
            count = int(count)
            positions = worker_positions[cursor : cursor + count]
            job_establishment_index[positions] = first_establishment + local_index
            cursor += count
        if cursor != n_workers:
            raise RuntimeError("worker allocation did not conserve its cell total")
    return job_establishment_index


def _enterprise_targets(
    n_establishments: int, multi_establishment_rate: float
) -> tuple[int, int]:
    """Choose enterprise and multi-enterprise counts nearest the character dial."""
    n_enterprises = max(
        1,
        min(
            n_establishments,
            int(round(n_establishments / (1.0 + multi_establishment_rate))),
        ),
    )
    extra_establishments = n_establishments - n_enterprises
    if extra_establishments == 0:
        return n_enterprises, 0
    n_multi = max(1, int(round(multi_establishment_rate * n_enterprises)))
    return n_enterprises, min(n_enterprises, extra_establishments, n_multi)


def _assign_enterprises(
    n_establishments: int,
    n_enterprises: int,
    n_multi: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Give every enterprise a location and exactly ``n_multi`` branch operators."""
    assignment = np.empty(n_establishments, dtype=np.int64)
    establishment_order = rng.permutation(n_establishments)
    assignment[establishment_order[:n_enterprises]] = np.arange(
        n_enterprises, dtype=np.int64
    )
    remaining = establishment_order[n_enterprises:]
    if len(remaining):
        enterprise_weight = rng.lognormal(mean=0.0, sigma=1.1, size=n_enterprises)
        multi_enterprise = _weighted_sample_without_replacement(
            enterprise_weight, n_multi, rng
        )
        assignment[remaining[:n_multi]] = multi_enterprise
        extra = remaining[n_multi:]
        if len(extra):
            multi_weight = enterprise_weight[multi_enterprise]
            multi_weight /= multi_weight.sum()
            assignment[extra] = rng.choice(
                multi_enterprise, size=len(extra), replace=True, p=multi_weight
            )
    return assignment


def _draw_industries(
    head_cell: np.ndarray, urbanity_flat: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    urban = urbanity_flat[head_cell]
    base = np.asarray(
        [0.07, 0.10, 0.11, 0.18, 0.08, 0.14, 0.10, 0.07, 0.09, 0.06], dtype=np.float64
    )
    weight = np.broadcast_to(base, (len(head_cell), len(base))).copy()
    weight[:, 0] *= 1.8 - 1.6 * urban
    weight[:, 1] *= 1.25 - 0.45 * urban
    weight[:, 3] *= 0.75 + 0.65 * urban
    weight[:, 5] *= 0.45 + 1.45 * urban
    weight[:, 6] *= 0.70 + 0.60 * urban
    weight[:, 7] *= 0.80 + 0.35 * urban
    weight[:, 8] *= 0.75 + 0.55 * urban
    weight /= weight.sum(axis=1, keepdims=True)
    cumulative = np.cumsum(weight, axis=1)
    draw = rng.random(len(head_cell))
    return np.sum(draw[:, None] > cumulative, axis=1).astype(np.int16)


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


def build_businesses(
    microdata: dict,
    seed: int,
    identity_map: dict,
    params: BusinessParams | None = None,
) -> dict:
    """Build initial enterprise, establishment, and job truth tables."""
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise TypeError("seed must be an integer")
    seed = int(seed)
    if seed < 0:
        raise ValueError("seed must be nonnegative")
    if params is None:
        params = business_params_from_character(draw_world_character(seed)["business"])
    if not isinstance(params, BusinessParams):
        raise TypeError("params must be BusinessParams or None")
    _validate_params(params)

    (
        person_cell,
        person_age,
        person_education,
        _,
        urbanity,
        truth_person_id,
        _,
        generator_version,
        identity_world_id,
        snapshot_tick,
    ) = _validate_inputs(microdata, identity_map, seed)
    urbanity_flat = urbanity.reshape(-1)
    person_urbanity = urbanity_flat[person_cell]
    rng = np.random.default_rng(np.random.SeedSequence([seed, 0xB051E55]))

    working_age = (person_age >= params.minimum_work_age) & (
        person_age <= params.maximum_work_age
    )
    working_person_index = np.flatnonzero(working_age)
    if len(working_person_index) < 1:
        raise ValueError("business generation requires working-age people")
    age_profile = np.exp(-(((person_age.astype(np.float64) - 43.0) / 25.0) ** 4))
    employment_weight = np.clip(
        0.07 + 0.68 * age_profile + 0.035 * person_education + 0.055 * person_urbanity,
        0.02,
        0.94,
    )
    n_jobs = max(
        1,
        min(
            len(working_person_index),
            int(round(params.jobs_per_adult * len(working_person_index))),
        ),
    )
    selected_worker = _weighted_sample_without_replacement(
        employment_weight[working_person_index], n_jobs, rng
    )
    employed_person_index = np.sort(
        working_person_index[selected_worker], kind="stable"
    )
    worker_home_cell = person_cell[employed_person_index]

    n_establishments = max(
        1, min(n_jobs, int(round(n_jobs / params.mean_jobs_per_establishment)))
    )
    establishment_cell = _establishment_anchors(
        worker_home_cell, urbanity_flat, n_establishments, rng
    )
    job_establishment_index = _allocate_workers(
        establishment_cell,
        worker_home_cell,
        urbanity.shape,
        params.establishment_size_alpha,
        rng,
    )

    n_enterprises, n_multi = _enterprise_targets(
        n_establishments, params.multi_establishment_rate
    )
    establishment_enterprise_index = _assign_enterprises(
        n_establishments, n_enterprises, n_multi, rng
    )

    enterprise_id = truth_entity_ids("enterprise", n_enterprises)
    establishment_id = truth_entity_ids("establishment", n_establishments)
    job_id = truth_entity_ids("job", n_jobs)

    first_establishment = np.full(n_enterprises, n_establishments, dtype=np.int64)
    np.minimum.at(
        first_establishment,
        establishment_enterprise_index,
        np.arange(n_establishments, dtype=np.int64),
    )
    enterprise_industry = _draw_industries(
        establishment_cell[first_establishment], urbanity_flat, rng
    )
    establishment_industry = enterprise_industry[establishment_enterprise_index]
    job_industry = establishment_industry[job_establishment_index]

    worker_age = person_age[employed_person_index]
    worker_education = person_education[employed_person_index]
    worker_urbanity = person_urbanity[employed_person_index]
    full_time_probability = np.clip(
        0.70
        + 0.045 * worker_education
        - 0.10 * (worker_age < 23)
        - 0.12 * (worker_age > 66),
        0.45,
        0.91,
    )
    is_full_time = rng.random(n_jobs) < full_time_probability
    full_hours = rng.integers(1_720, 2_201, size=n_jobs, dtype=np.int32)
    part_hours = rng.integers(520, 1_501, size=n_jobs, dtype=np.int32)
    annual_hours = np.where(is_full_time, full_hours, part_hours).astype(np.int32)

    industry_wage = np.asarray(
        [17.5, 22.0, 21.0, 16.5, 20.0, 27.0, 14.5, 21.5, 23.5, 22.5], dtype=np.float64
    )
    experience = 1.0 + 0.008 * np.clip(worker_age - 24, 0, 28)
    hourly_wage = (
        industry_wage[job_industry]
        * (1.0 + 0.16 * worker_education)
        * experience
        * (0.92 + 0.18 * worker_urbanity)
        * params.payroll_level
        * np.exp(rng.normal(0.0, 0.16, n_jobs))
    )
    hourly_wage_cents = np.rint(np.clip(hourly_wage * 100.0, 900.0, 25_000.0)).astype(
        np.int64
    )
    annual_earnings_cents = annual_hours.astype(np.int64) * hourly_wage_cents
    employment_type = np.where(
        is_full_time, EMPLOYMENT_TYPES["full_time"], EMPLOYMENT_TYPES["part_time"]
    ).astype(np.int8)
    skill_group = np.minimum(
        3, worker_education.astype(np.int16) + (worker_age >= 35)
    ).astype(np.int16)
    occupation = (job_industry * 4 + skill_group).astype(np.int16)
    maximum_job_tenure = np.minimum(
        worker_age.astype(np.int64) - params.minimum_work_age,
        params.snapshot_year - params.minimum_opening_year,
    )
    job_tenure = np.minimum(
        np.rint(rng.exponential(scale=5.0, size=n_jobs)).astype(np.int64),
        maximum_job_tenure,
    )
    job_start_year = (params.snapshot_year - job_tenure).astype(np.int16)

    establishment_employment = np.zeros(n_establishments, dtype=np.int32)
    np.add.at(establishment_employment, job_establishment_index, 1)
    establishment_payroll = np.zeros(n_establishments, dtype=np.int64)
    np.add.at(establishment_payroll, job_establishment_index, annual_earnings_cents)
    minimum_job_start = np.full(n_establishments, params.snapshot_year, dtype=np.int16)
    np.minimum.at(minimum_job_start, job_establishment_index, job_start_year)

    maximum_age = params.snapshot_year - params.minimum_opening_year
    establishment_age = np.rint(
        np.clip(rng.gamma(2.0, 12.0, n_establishments), 0, maximum_age)
    ).astype(np.int64)
    establishment_opening_year = (params.snapshot_year - establishment_age).astype(
        np.int16
    )
    establishment_opening_year = np.minimum(
        establishment_opening_year, minimum_job_start
    ).astype(np.int16)
    revenue_multiplier = np.asarray(
        [2.4, 3.2, 2.7, 2.9, 2.6, 2.2, 2.5, 1.8, 1.9, 1.7], dtype=np.float64
    )
    establishment_revenue = np.rint(
        establishment_payroll.astype(np.float64)
        * revenue_multiplier[establishment_industry]
        * np.exp(rng.normal(0.0, 0.14, n_establishments))
    ).astype(np.int64)
    establishment_revenue = np.maximum(establishment_revenue, establishment_payroll)
    area_per_job = np.asarray(
        [55.0, 42.0, 32.0, 24.0, 30.0, 20.0, 28.0, 26.0, 36.0, 25.0], dtype=np.float64
    )
    establishment_floor_area = np.round(
        np.maximum(
            20.0,
            area_per_job[establishment_industry]
            * establishment_employment
            * np.exp(rng.normal(0.0, 0.10, n_establishments)),
        ),
        1,
    )

    headquarters_index = np.full(n_enterprises, -1, dtype=np.int64)
    order = np.lexsort(
        (
            np.arange(n_establishments, dtype=np.int64),
            -establishment_employment.astype(np.int64),
            establishment_enterprise_index,
        )
    )
    ordered_enterprise = establishment_enterprise_index[order]
    _, first = np.unique(ordered_enterprise, return_index=True)
    headquarters_index[ordered_enterprise[first]] = order[first]
    if np.any(headquarters_index < 0):
        raise RuntimeError("an enterprise has no headquarters")
    establishment_role = np.full(
        n_establishments, ESTABLISHMENT_ROLES["branch"], dtype=np.int8
    )
    establishment_role[headquarters_index] = ESTABLISHMENT_ROLES["headquarters"]

    enterprise_establishments = np.zeros(n_enterprises, dtype=np.int32)
    np.add.at(enterprise_establishments, establishment_enterprise_index, 1)
    enterprise_employment = np.zeros(n_enterprises, dtype=np.int32)
    np.add.at(
        enterprise_employment, establishment_enterprise_index, establishment_employment
    )
    enterprise_payroll = np.zeros(n_enterprises, dtype=np.int64)
    np.add.at(enterprise_payroll, establishment_enterprise_index, establishment_payroll)
    enterprise_revenue = np.zeros(n_enterprises, dtype=np.int64)
    np.add.at(enterprise_revenue, establishment_enterprise_index, establishment_revenue)
    enterprise_opening_year = np.full(
        n_enterprises, params.snapshot_year, dtype=np.int16
    )
    np.minimum.at(
        enterprise_opening_year,
        establishment_enterprise_index,
        establishment_opening_year,
    )
    enterprise_size_class = np.digitize(
        enterprise_employment, [1, 5, 20, 100, 500], right=True
    ).astype(np.int8)

    legal_draw = rng.random(n_enterprises)
    legal_form = np.full(n_enterprises, LEGAL_FORMS["corporation"], dtype=np.int8)
    small = enterprise_employment < 10
    legal_form[small & (legal_draw < 0.54)] = LEGAL_FORMS["sole_proprietor"]
    legal_form[small & (legal_draw >= 0.54) & (legal_draw < 0.76)] = LEGAL_FORMS[
        "partnership"
    ]
    cooperative = (legal_draw > 0.96) & (enterprise_industry != 9)
    legal_form[cooperative] = LEGAL_FORMS["cooperative"]
    public_industry = np.isin(enterprise_industry, [7, 8, 9])
    public_entity = public_industry & (rng.random(n_enterprises) < 0.36)
    legal_form[public_entity] = LEGAL_FORMS["public"]

    ownership_draw = rng.random(n_enterprises)
    ownership = np.full(n_enterprises, OWNERSHIP["domestic_private"], dtype=np.int8)
    ownership[ownership_draw < 0.07] = OWNERSHIP["foreign_private"]
    ownership[legal_form == LEGAL_FORMS["cooperative"]] = OWNERSHIP["cooperative"]
    ownership[legal_form == LEGAL_FORMS["public"]] = OWNERSHIP["public"]

    state = {
        "truth_world_id": identity_world_id,
        "generator_version": generator_version,
        "snapshot_tick": snapshot_tick,
        "business_params": _params_record(params),
        "enterprise": {
            "truth_enterprise_id": enterprise_id,
            "headquarters_establishment_id": establishment_id[headquarters_index],
            "headquarters_cell": establishment_cell[headquarters_index],
            "industry": enterprise_industry,
            "legal_form": legal_form,
            "ownership": ownership,
            "establishment_count": enterprise_establishments,
            "employment_count": enterprise_employment,
            "annual_payroll_cents": enterprise_payroll,
            "annual_revenue_cents": enterprise_revenue,
            "opening_year": enterprise_opening_year,
            "size_class": enterprise_size_class,
            "is_active": np.ones(n_enterprises, dtype=np.bool_),
        },
        "establishment": {
            "truth_establishment_id": establishment_id,
            "truth_enterprise_id": enterprise_id[establishment_enterprise_index],
            "cell": establishment_cell,
            "industry": establishment_industry,
            "establishment_role": establishment_role,
            "employment_count": establishment_employment,
            "annual_payroll_cents": establishment_payroll,
            "annual_revenue_cents": establishment_revenue,
            "floor_area_m2": establishment_floor_area,
            "opening_year": establishment_opening_year,
            "is_active": np.ones(n_establishments, dtype=np.bool_),
        },
        "job": {
            "truth_job_id": job_id,
            "truth_person_id": truth_person_id[employed_person_index],
            "truth_establishment_id": establishment_id[job_establishment_index],
            "occupation": occupation,
            "employment_type": employment_type,
            "annual_hours": annual_hours,
            "hourly_wage_cents": hourly_wage_cents,
            "annual_earnings_cents": annual_earnings_cents,
            "start_year": job_start_year,
            "is_active": np.ones(n_jobs, dtype=np.bool_),
        },
        "n_enterprises": n_enterprises,
        "n_establishments": n_establishments,
        "n_jobs": n_jobs,
    }
    validate_business_conservation(state, microdata, identity_map, seed)
    return state


def validate_business_conservation(
    state: dict, microdata: dict, identity_map: dict, seed: int
) -> None:
    """Fail unless all current-state business identities reconcile exactly."""
    (
        _,
        person_age,
        _,
        _,
        urbanity,
        truth_person_id,
        _,
        generator_version,
        identity_world_id,
        snapshot_tick,
    ) = _validate_inputs(microdata, identity_map, seed)
    try:
        enterprise = state["enterprise"]
        establishment = state["establishment"]
        job = state["job"]
        n_enterprises = int(state["n_enterprises"])
        n_establishments = int(state["n_establishments"])
        n_jobs = int(state["n_jobs"])
        params = _params_from_record(state["business_params"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("business state metadata is incomplete") from exc
    _validate_params(params)

    enterprise_dtypes = {
        "truth_enterprise_id": np.dtype(np.uint64),
        "headquarters_establishment_id": np.dtype(np.uint64),
        "headquarters_cell": np.dtype(np.int64),
        "industry": np.dtype(np.int16),
        "legal_form": np.dtype(np.int8),
        "ownership": np.dtype(np.int8),
        "establishment_count": np.dtype(np.int32),
        "employment_count": np.dtype(np.int32),
        "annual_payroll_cents": np.dtype(np.int64),
        "annual_revenue_cents": np.dtype(np.int64),
        "opening_year": np.dtype(np.int16),
        "size_class": np.dtype(np.int8),
        "is_active": np.dtype(np.bool_),
    }
    establishment_dtypes = {
        "truth_establishment_id": np.dtype(np.uint64),
        "truth_enterprise_id": np.dtype(np.uint64),
        "cell": np.dtype(np.int64),
        "industry": np.dtype(np.int16),
        "establishment_role": np.dtype(np.int8),
        "employment_count": np.dtype(np.int32),
        "annual_payroll_cents": np.dtype(np.int64),
        "annual_revenue_cents": np.dtype(np.int64),
        "floor_area_m2": np.dtype(np.float64),
        "opening_year": np.dtype(np.int16),
        "is_active": np.dtype(np.bool_),
    }
    job_dtypes = {
        "truth_job_id": np.dtype(np.uint64),
        "truth_person_id": np.dtype(np.uint64),
        "truth_establishment_id": np.dtype(np.uint64),
        "occupation": np.dtype(np.int16),
        "employment_type": np.dtype(np.int8),
        "annual_hours": np.dtype(np.int32),
        "hourly_wage_cents": np.dtype(np.int64),
        "annual_earnings_cents": np.dtype(np.int64),
        "start_year": np.dtype(np.int16),
        "is_active": np.dtype(np.bool_),
    }
    _table_columns(enterprise, enterprise_dtypes, n_enterprises, "enterprise")
    _table_columns(
        establishment, establishment_dtypes, n_establishments, "establishment"
    )
    _table_columns(job, job_dtypes, n_jobs, "job")
    for table_name, table in (
        ("enterprise", enterprise),
        ("establishment", establishment),
        ("job", job),
    ):
        if "observed_business_register_id" in table:
            raise ValueError(
                f"observed register ID leaked into the {table_name} truth table"
            )

    if np.uint64(state["truth_world_id"]) != identity_world_id:
        raise ValueError("business state belongs to a different truth world")
    if int(state["generator_version"]) != generator_version:
        raise ValueError("business state uses a different generator version")
    if np.int64(state["snapshot_tick"]) != snapshot_tick:
        raise ValueError("business state and identity snapshot ticks differ")
    if min(n_enterprises, n_establishments, n_jobs) < 1:
        raise ValueError(
            "business state must contain enterprises, establishments, and jobs"
        )
    working_age_count = int(
        (
            (person_age >= params.minimum_work_age)
            & (person_age <= params.maximum_work_age)
        ).sum()
    )
    expected_jobs = max(
        1,
        min(working_age_count, int(round(params.jobs_per_adult * working_age_count))),
    )
    if n_jobs != expected_jobs:
        raise ValueError("job count does not match the world-character employment dial")
    expected_establishments = max(
        1,
        min(n_jobs, int(round(n_jobs / params.mean_jobs_per_establishment))),
    )
    if n_establishments != expected_establishments:
        raise ValueError("establishment count does not match business parameters")
    expected_enterprises, expected_multi = _enterprise_targets(
        n_establishments, params.multi_establishment_rate
    )
    if n_enterprises != expected_enterprises:
        raise ValueError("enterprise count does not match the world-character dial")

    enterprise_id = enterprise["truth_enterprise_id"]
    establishment_id = establishment["truth_establishment_id"]
    job_id = job["truth_job_id"]
    for name, ids, namespace in (
        ("enterprise", enterprise_id, ENTITY_NAMESPACE["enterprise"]),
        ("establishment", establishment_id, ENTITY_NAMESPACE["establishment"]),
        ("job", job_id, ENTITY_NAMESPACE["job"]),
    ):
        if len(np.unique(ids)) != len(ids):
            raise ValueError(f"truth {name} identities are not unique")
        if np.any(entity_namespace(ids) != namespace):
            raise ValueError(f"{name} identities use the wrong entity namespace")

    job_person_id = job["truth_person_id"]
    person_position = np.searchsorted(truth_person_id, job_person_id)
    person_valid = person_position < len(truth_person_id)
    if not person_valid.all() or not np.array_equal(
        truth_person_id[person_position], job_person_id
    ):
        raise ValueError("job references a nonexistent person")
    if len(np.unique(job_person_id)) != n_jobs:
        raise ValueError("v0 assigns multiple active jobs to one person")
    worker_age = person_age[person_position]
    if np.any(worker_age < params.minimum_work_age) or np.any(
        worker_age > params.maximum_work_age
    ):
        raise ValueError("job references a person outside the working-age bounds")

    job_establishment_id = job["truth_establishment_id"]
    job_establishment_index = np.searchsorted(establishment_id, job_establishment_id)
    establishment_valid = job_establishment_index < n_establishments
    if not establishment_valid.all() or not np.array_equal(
        establishment_id[job_establishment_index], job_establishment_id
    ):
        raise ValueError("job references a nonexistent establishment")
    establishment_enterprise_id = establishment["truth_enterprise_id"]
    establishment_enterprise_index = np.searchsorted(
        enterprise_id, establishment_enterprise_id
    )
    enterprise_valid = establishment_enterprise_index < n_enterprises
    if not enterprise_valid.all() or not np.array_equal(
        enterprise_id[establishment_enterprise_index], establishment_enterprise_id
    ):
        raise ValueError("establishment references a nonexistent enterprise")

    if not job["is_active"].all() or not establishment["is_active"].all():
        raise ValueError("initial jobs and establishments must be active")
    if not enterprise["is_active"].all():
        raise ValueError("initial enterprises must be active")
    actual_multi = int((enterprise["establishment_count"] > 1).sum())
    if actual_multi != expected_multi:
        raise ValueError(
            "multi-establishment count does not match the world-character dial"
        )
    expected_earnings = job["annual_hours"].astype(np.int64) * job["hourly_wage_cents"]
    if not np.array_equal(job["annual_earnings_cents"], expected_earnings):
        raise ValueError("job earnings do not equal hours times wage")
    if np.any(job["annual_hours"] <= 0) or np.any(job["hourly_wage_cents"] <= 0):
        raise ValueError("job hours and wages must be positive")

    expected_establishment_employment = np.zeros(n_establishments, dtype=np.int32)
    np.add.at(expected_establishment_employment, job_establishment_index, 1)
    if not np.array_equal(
        establishment["employment_count"], expected_establishment_employment
    ):
        raise ValueError("establishment employment does not equal linked jobs")
    if np.any(expected_establishment_employment < 1):
        raise ValueError("initial establishment has no linked job")
    expected_establishment_payroll = np.zeros(n_establishments, dtype=np.int64)
    np.add.at(
        expected_establishment_payroll,
        job_establishment_index,
        job["annual_earnings_cents"],
    )
    if not np.array_equal(
        establishment["annual_payroll_cents"], expected_establishment_payroll
    ):
        raise ValueError("establishment payroll does not equal linked job earnings")
    if np.any(
        establishment["annual_revenue_cents"] < establishment["annual_payroll_cents"]
    ):
        raise ValueError("establishment revenue is below payroll")
    if np.any(establishment["cell"] < 0) or np.any(
        establishment["cell"] >= urbanity.size
    ):
        raise ValueError("establishment cell is outside the world grid")
    if np.any(establishment["floor_area_m2"] <= 0.0):
        raise ValueError("establishment floor area must be positive")

    expected_enterprise_establishments = np.zeros(n_enterprises, dtype=np.int32)
    np.add.at(expected_enterprise_establishments, establishment_enterprise_index, 1)
    expected_enterprise_employment = np.zeros(n_enterprises, dtype=np.int32)
    np.add.at(
        expected_enterprise_employment,
        establishment_enterprise_index,
        establishment["employment_count"],
    )
    expected_enterprise_payroll = np.zeros(n_enterprises, dtype=np.int64)
    np.add.at(
        expected_enterprise_payroll,
        establishment_enterprise_index,
        establishment["annual_payroll_cents"],
    )
    expected_enterprise_revenue = np.zeros(n_enterprises, dtype=np.int64)
    np.add.at(
        expected_enterprise_revenue,
        establishment_enterprise_index,
        establishment["annual_revenue_cents"],
    )
    if not np.array_equal(
        enterprise["establishment_count"], expected_enterprise_establishments
    ):
        raise ValueError("enterprise establishment counts do not reconcile")
    if not np.array_equal(
        enterprise["employment_count"], expected_enterprise_employment
    ):
        raise ValueError("enterprise employment does not reconcile")
    if not np.array_equal(
        enterprise["annual_payroll_cents"], expected_enterprise_payroll
    ):
        raise ValueError("enterprise payroll does not reconcile")
    if not np.array_equal(
        enterprise["annual_revenue_cents"], expected_enterprise_revenue
    ):
        raise ValueError("enterprise revenue does not reconcile")

    headquarters_index = np.searchsorted(
        establishment_id, enterprise["headquarters_establishment_id"]
    )
    headquarters_valid = headquarters_index < n_establishments
    if not headquarters_valid.all() or not np.array_equal(
        establishment_id[headquarters_index],
        enterprise["headquarters_establishment_id"],
    ):
        raise ValueError("enterprise references a nonexistent headquarters")
    if not np.array_equal(
        establishment_enterprise_index[headquarters_index],
        np.arange(n_enterprises, dtype=np.int64),
    ):
        raise ValueError("headquarters is owned by a different enterprise")
    if not np.array_equal(
        establishment["cell"][headquarters_index], enterprise["headquarters_cell"]
    ):
        raise ValueError("enterprise headquarters cell does not reconcile")
    if np.any(
        establishment["establishment_role"][headquarters_index]
        != ESTABLISHMENT_ROLES["headquarters"]
    ):
        raise ValueError("headquarters establishment has the wrong role")
    if (
        int(
            (
                establishment["establishment_role"]
                == ESTABLISHMENT_ROLES["headquarters"]
            ).sum()
        )
        != n_enterprises
    ):
        raise ValueError("enterprise does not have exactly one headquarters")

    expected_enterprise_opening = np.full(
        n_enterprises, np.iinfo(np.int16).max, dtype=np.int16
    )
    np.minimum.at(
        expected_enterprise_opening,
        establishment_enterprise_index,
        establishment["opening_year"],
    )
    if not np.array_equal(enterprise["opening_year"], expected_enterprise_opening):
        raise ValueError("enterprise opening year does not match its establishments")
    if not np.array_equal(
        establishment["industry"],
        enterprise["industry"][establishment_enterprise_index],
    ):
        raise ValueError("establishment industry differs from its enterprise")
