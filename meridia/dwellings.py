"""Initial dwelling stock linked exactly to Meridia households and persons.

V0 creates one occupied dwelling per household plus an explicit vacant stock.  The
physical and economic attributes are seeded, while occupancy, household location, and
resident accounting are exact identities checked at the build boundary.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from meridia.identities import ENTITY_NAMESPACE, build_initial_identity_map
from meridia.identities import entity_namespace, truth_entity_ids, truth_world_id

DWELLING_TYPES = {
    "detached": 0,
    "attached": 1,
    "low_rise": 2,
    "high_rise": 3,
}

TENURES = {
    "owner": 0,
    "mortgage": 1,
    "private_rent": 2,
    "social_rent": 3,
    "vacant": 4,
}


@dataclass(frozen=True)
class DwellingParams:
    vacancy_rate: float = 0.08
    snapshot_year: int = 2026
    minimum_year_built: int = 1880


def vacant_stock_target(n_households: int, vacancy_rate: float) -> int:
    """Nearest integer vacant stock for a requested share of all dwellings."""
    if isinstance(n_households, bool) or not isinstance(
        n_households, (int, np.integer)
    ):
        raise TypeError("n_households must be an integer")
    n_households = int(n_households)
    if n_households < 0:
        raise ValueError("n_households must be nonnegative")
    if not np.isfinite(vacancy_rate) or not 0.0 <= vacancy_rate < 1.0:
        raise ValueError("vacancy_rate must be finite and in [0, 1)")
    if n_households == 0 or vacancy_rate == 0.0:
        return 0
    return int(round(n_households * vacancy_rate / (1.0 - vacancy_rate)))


def _allocate_vacant_cells(
    household_cell: np.ndarray, n_cells: int, n_vacant: int
) -> np.ndarray:
    """Largest-remainder allocation of vacancies proportional to cell households."""
    if n_vacant == 0:
        return np.empty(0, dtype=np.int64)
    household_counts = np.bincount(household_cell, minlength=n_cells)
    total = int(household_counts.sum())
    if total == 0:
        raise ValueError("vacant stock cannot be allocated without households")
    shares = household_counts.astype(np.float64) * (n_vacant / total)
    allocation = np.floor(shares).astype(np.int64)
    remainder = n_vacant - int(allocation.sum())
    if remainder:
        order = np.argsort(-(shares - allocation), kind="stable")
        allocation[order[:remainder]] += 1
    if int(allocation.sum()) != n_vacant:
        raise RuntimeError(
            "vacant-stock allocation did not conserve its national target"
        )
    return np.repeat(np.arange(n_cells, dtype=np.int64), allocation)


def _validate_inputs(microdata: dict, identity_map: dict) -> tuple:
    try:
        person = microdata["person"]
        person_household = np.asarray(person["household"])
        person_income = np.asarray(person["income"], dtype=np.float64)
        household_cell = np.asarray(microdata["household_cell"])
        urbanity = np.asarray(microdata["urbanity"], dtype=np.float64)
        n_persons = int(microdata["n_persons"])
        n_households = int(microdata["n_households"])
        truth_person_id = np.asarray(identity_map["identity"]["truth_person_id"])
        truth_household_id = np.asarray(identity_map["identity"]["truth_household_id"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "inputs do not satisfy the Meridia identity/dwelling schema"
        ) from exc

    if person_household.ndim != 1 or person_income.ndim != 1:
        raise ValueError("person columns must be one-dimensional")
    if household_cell.ndim != 1 or urbanity.ndim != 2:
        raise ValueError("household_cell must be 1D and urbanity must be a 2D grid")
    if len(person_household) != n_persons or len(person_income) != n_persons:
        raise ValueError("person columns do not match n_persons")
    if len(household_cell) != n_households:
        raise ValueError("household_cell does not match n_households")
    if truth_person_id.dtype != np.uint64 or truth_person_id.ndim != 1:
        raise ValueError("truth_person_id must be a one-dimensional uint64 array")
    if len(truth_person_id) != n_persons:
        raise ValueError("truth person identities do not match n_persons")
    if truth_household_id.dtype != np.uint64 or truth_household_id.ndim != 1:
        raise ValueError("truth_household_id must be a one-dimensional uint64 array")
    if len(truth_household_id) != n_households:
        raise ValueError("truth household identities do not match n_households")
    if n_persons < 1 or n_households < 1:
        raise ValueError("dwelling generation requires persons and households")
    person_household = person_household.astype(np.int64, copy=False)
    household_cell = household_cell.astype(np.int64, copy=False)
    if int(person_household.min()) < 0 or int(person_household.max()) >= n_households:
        raise ValueError("person household import key is outside the household table")
    if int(household_cell.min()) < 0 or int(household_cell.max()) >= urbanity.size:
        raise ValueError("household cell is outside the urbanity grid")
    if not np.isfinite(person_income).all() or np.any(person_income < 0.0):
        raise ValueError("person income must be finite and nonnegative")
    if not np.isfinite(urbanity).all() or np.any((urbanity < 0.0) | (urbanity > 1.0)):
        raise ValueError("urbanity must be finite and in [0, 1]")
    if np.any(entity_namespace(truth_person_id) != ENTITY_NAMESPACE["person"]):
        raise ValueError("person identities use the wrong entity namespace")
    if len(np.unique(truth_person_id)) != n_persons:
        raise ValueError("truth person identities are not unique")
    if np.any(entity_namespace(truth_household_id) != ENTITY_NAMESPACE["household"]):
        raise ValueError("household identities use the wrong entity namespace")
    if len(np.unique(truth_household_id)) != n_households:
        raise ValueError("truth household identities are not unique")

    return (
        person_household,
        person_income,
        household_cell,
        urbanity,
        truth_household_id,
        n_persons,
        n_households,
    )


def build_dwellings(
    microdata: dict,
    seed: int,
    identity_map: dict | None = None,
    params: DwellingParams = DwellingParams(),
) -> dict:
    """Build the deterministic initial dwelling stock for a microdata snapshot."""
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise TypeError("seed must be an integer")
    seed = int(seed)
    if identity_map is None:
        identity_map = build_initial_identity_map(microdata, seed)
    (
        person_household,
        person_income,
        household_cell,
        urbanity,
        truth_household_id,
        n_persons,
        n_households,
    ) = _validate_inputs(microdata, identity_map)
    try:
        generator_version = int(identity_map["generator_version"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("identity map is missing generator_version") from exc
    if np.uint64(identity_map["truth_world_id"]) != truth_world_id(
        seed, generator_version
    ):
        raise ValueError("seed does not match the identity map's truth world")

    if params.minimum_year_built > params.snapshot_year:
        raise ValueError("minimum_year_built cannot exceed snapshot_year")
    if (
        not np.iinfo(np.int16).min
        <= params.minimum_year_built
        <= np.iinfo(np.int16).max
    ):
        raise ValueError("minimum_year_built must fit int16")
    if not np.iinfo(np.int16).min <= params.snapshot_year <= np.iinfo(np.int16).max:
        raise ValueError("snapshot_year must fit int16")

    n_vacant = vacant_stock_target(n_households, params.vacancy_rate)
    vacant_cell = _allocate_vacant_cells(household_cell, urbanity.size, n_vacant)
    cell = np.concatenate((household_cell, vacant_cell)).astype(np.int64, copy=False)
    n_dwellings = len(cell)
    is_occupied = np.zeros(n_dwellings, dtype=np.bool_)
    is_occupied[:n_households] = True

    occupant_id = np.zeros(n_dwellings, dtype=np.uint64)
    occupant_id[:n_households] = truth_household_id
    household_sizes = np.bincount(person_household, minlength=n_households)
    if int(household_sizes.max()) > np.iinfo(np.int32).max:
        raise ValueError("household size exceeds the resident_count capacity")
    resident_count = np.zeros(n_dwellings, dtype=np.int32)
    resident_count[:n_households] = household_sizes.astype(np.int32)

    rng = np.random.default_rng(np.random.SeedSequence([seed, 0xD7E11]))
    urban = urbanity.reshape(-1)[cell]

    # Dense cells contain more attached and multi-unit dwellings; all probabilities
    # remain positive, so different seeds retain architectural variation.
    p_high_rise = 0.02 + 0.28 * urban**2
    p_low_rise = 0.10 + 0.25 * urban
    p_attached = 0.18 + 0.12 * urban
    p_detached = 1.0 - p_high_rise - p_low_rise - p_attached
    type_draw = rng.random(n_dwellings)
    dwelling_type = np.full(n_dwellings, DWELLING_TYPES["high_rise"], dtype=np.int8)
    dwelling_type[type_draw < p_detached] = DWELLING_TYPES["detached"]
    attached_cut = p_detached + p_attached
    low_rise_cut = attached_cut + p_low_rise
    dwelling_type[(type_draw >= p_detached) & (type_draw < attached_cut)] = (
        DWELLING_TYPES["attached"]
    )
    dwelling_type[(type_draw >= attached_cut) & (type_draw < low_rise_cut)] = (
        DWELLING_TYPES["low_rise"]
    )

    vacant_bedrooms = 1 + rng.binomial(3, 0.38 - 0.10 * urban)
    occupied_bedrooms = np.maximum(1, (resident_count + 1) // 2)
    bedrooms = np.where(is_occupied, occupied_bedrooms, vacant_bedrooms)
    bedrooms += rng.random(n_dwellings) < (0.20 + 0.18 * (1.0 - urban))
    bedrooms = np.clip(bedrooms, 1, 7)
    bedrooms = np.where(
        dwelling_type == DWELLING_TYPES["high_rise"], np.minimum(bedrooms, 3), bedrooms
    )
    bedrooms = np.where(
        dwelling_type == DWELLING_TYPES["low_rise"], np.minimum(bedrooms, 4), bedrooms
    ).astype(np.int8)

    type_base = np.asarray([62.0, 56.0, 43.0, 35.0], dtype=np.float64)
    type_per_bedroom = np.asarray([24.0, 21.0, 18.0, 15.0], dtype=np.float64)
    floor_area = (
        type_base[dwelling_type]
        + type_per_bedroom[dwelling_type] * bedrooms
        + rng.normal(0.0, 7.0, n_dwellings)
    )
    floor_area = np.round(np.clip(floor_area, 24.0, 350.0), 1)

    maximum_age = params.snapshot_year - params.minimum_year_built
    age_years = rng.gamma(shape=2.1, scale=17.0, size=n_dwellings)
    age_years *= 1.0 - 0.18 * urban
    age_years = np.rint(np.clip(age_years, 0, maximum_age)).astype(np.int64)
    year_built = (params.snapshot_year - age_years).astype(np.int16)

    household_income = np.bincount(
        person_household, weights=person_income, minlength=n_households
    )
    log_income = np.log1p(household_income)
    income_scale = max(float(log_income.std()), 1e-12)
    income_z = np.clip(
        (log_income - float(np.median(log_income))) / income_scale, -3.0, 3.0
    )
    household_urban = urban[:n_households]
    p_owner = np.clip(0.18 + 0.06 * income_z - 0.08 * household_urban, 0.05, 0.32)
    p_private = np.clip(0.25 + 0.20 * household_urban - 0.05 * income_z, 0.10, 0.55)
    p_social = np.clip(
        0.07 - 0.02 * income_z + 0.03 * (1.0 - household_urban), 0.02, 0.16
    )
    p_mortgage = 1.0 - p_owner - p_private - p_social
    tenure_draw = rng.random(n_households)
    tenure = np.full(n_dwellings, TENURES["vacant"], dtype=np.int8)
    tenure[:n_households] = TENURES["social_rent"]
    tenure[:n_households][tenure_draw < p_owner] = TENURES["owner"]
    mortgage_cut = p_owner + p_mortgage
    private_cut = mortgage_cut + p_private
    occupied_tenure = tenure[:n_households]
    occupied_tenure[(tenure_draw >= p_owner) & (tenure_draw < mortgage_cut)] = TENURES[
        "mortgage"
    ]
    occupied_tenure[(tenure_draw >= mortgage_cut) & (tenure_draw < private_cut)] = (
        TENURES["private_rent"]
    )

    value_noise = np.exp(rng.normal(0.0, 0.12, n_dwellings))
    assessed_value = (
        (45_000.0 + floor_area * 1_750.0) * (0.72 + 0.90 * urban) * value_noise
    )
    assessed_value = np.round(np.clip(assessed_value, 20_000.0, None), -2)

    monthly_rent = np.zeros(n_dwellings, dtype=np.float64)
    rental = (tenure == TENURES["private_rent"]) | (tenure == TENURES["social_rent"])
    monthly_rent[rental] = (
        floor_area[rental]
        * (8.5 + 7.5 * urban[rental])
        * np.exp(rng.normal(0.0, 0.08, int(rental.sum())))
    )
    social = tenure == TENURES["social_rent"]
    monthly_rent[social] *= 0.62
    monthly_rent = np.round(monthly_rent, 2)

    stock = {
        "truth_world_id": np.uint64(identity_map["truth_world_id"]),
        "generator_version": generator_version,
        "snapshot_tick": np.int64(identity_map["snapshot_tick"]),
        "dwelling": {
            "truth_dwelling_id": truth_entity_ids("dwelling", n_dwellings),
            "cell": cell,
            "dwelling_type": dwelling_type,
            "tenure": tenure,
            "bedrooms": bedrooms,
            "floor_area_m2": floor_area,
            "year_built": year_built,
            "assessed_value": assessed_value,
            "monthly_rent": monthly_rent,
            "is_occupied": is_occupied,
            "truth_household_id": occupant_id,
            "resident_count": resident_count,
        },
        "n_dwellings": n_dwellings,
        "n_occupied": n_households,
        "n_vacant": n_vacant,
    }
    validate_dwelling_conservation(stock, microdata, identity_map)
    return stock


def validate_dwelling_conservation(
    stock: dict, microdata: dict, identity_map: dict
) -> None:
    """Fail loudly unless every v0 dwelling identity holds exactly."""
    (
        person_household,
        _,
        household_cell,
        urbanity,
        truth_household_id,
        n_persons,
        n_households,
    ) = _validate_inputs(microdata, identity_map)
    try:
        table = stock["dwelling"]
        n_dwellings = int(stock["n_dwellings"])
        n_occupied = int(stock["n_occupied"])
        n_vacant = int(stock["n_vacant"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("dwelling stock metadata is incomplete") from exc

    required_dtypes = {
        "truth_dwelling_id": np.dtype(np.uint64),
        "cell": np.dtype(np.int64),
        "dwelling_type": np.dtype(np.int8),
        "tenure": np.dtype(np.int8),
        "bedrooms": np.dtype(np.int8),
        "floor_area_m2": np.dtype(np.float64),
        "year_built": np.dtype(np.int16),
        "assessed_value": np.dtype(np.float64),
        "monthly_rent": np.dtype(np.float64),
        "is_occupied": np.dtype(np.bool_),
        "truth_household_id": np.dtype(np.uint64),
        "resident_count": np.dtype(np.int32),
    }
    for name, expected_dtype in required_dtypes.items():
        if name not in table:
            raise ValueError(f"dwelling table is missing {name}")
        values = np.asarray(table[name])
        if values.ndim != 1 or len(values) != n_dwellings:
            raise ValueError(f"dwelling column {name} has the wrong shape")
        if values.dtype != expected_dtype:
            raise ValueError(
                f"dwelling column {name} has dtype {values.dtype}, "
                f"expected {expected_dtype}"
            )

    if np.uint64(stock["truth_world_id"]) != np.uint64(identity_map["truth_world_id"]):
        raise ValueError("dwelling stock belongs to a different truth world")
    if np.int64(stock["snapshot_tick"]) != np.int64(identity_map["snapshot_tick"]):
        raise ValueError("dwelling stock and identity snapshot ticks differ")
    occupied = table["is_occupied"]
    vacant = ~occupied
    if n_occupied != n_households or int(occupied.sum()) != n_households:
        raise ValueError("occupied dwellings do not equal households")
    if n_vacant != int(vacant.sum()) or n_dwellings != n_occupied + n_vacant:
        raise ValueError("occupied and vacant dwellings do not conserve dwelling stock")
    if np.any(table["truth_household_id"][vacant] != 0):
        raise ValueError("vacant dwelling carries a household truth ID")
    if np.any(table["resident_count"][vacant] != 0):
        raise ValueError("vacant dwelling carries residents")
    if np.any(table["truth_household_id"][occupied] == 0):
        raise ValueError("occupied dwelling is missing a household truth ID")

    dwelling_id = table["truth_dwelling_id"]
    if len(np.unique(dwelling_id)) != n_dwellings:
        raise ValueError("truth dwelling identities are not unique")
    if np.any(entity_namespace(dwelling_id) != ENTITY_NAMESPACE["dwelling"]):
        raise ValueError("dwelling identities use the wrong entity namespace")

    occupied_household_id = table["truth_household_id"][occupied]
    if len(np.unique(occupied_household_id)) != n_households:
        raise ValueError("a household occupies zero or multiple dwellings")
    occupied_order = np.argsort(occupied_household_id, kind="stable")
    identity_order = np.argsort(truth_household_id, kind="stable")
    if not np.array_equal(
        occupied_household_id[occupied_order], truth_household_id[identity_order]
    ):
        raise ValueError("occupied household identities do not match the identity map")

    expected_sizes = np.bincount(person_household, minlength=n_households).astype(
        np.int32
    )
    if not np.array_equal(
        table["resident_count"][occupied][occupied_order],
        expected_sizes[identity_order],
    ):
        raise ValueError("dwelling resident counts do not equal household sizes")
    if int(table["resident_count"].sum()) != n_persons:
        raise ValueError("dwelling resident counts do not conserve persons")
    if not np.array_equal(
        table["cell"][occupied][occupied_order], household_cell[identity_order]
    ):
        raise ValueError("occupied dwelling and household cells differ")

    if np.any(table["cell"] < 0) or np.any(table["cell"] >= urbanity.size):
        raise ValueError("dwelling cell is outside the world grid")
    if np.any(table["bedrooms"] < 1) or np.any(table["floor_area_m2"] <= 0.0):
        raise ValueError("dwelling physical attributes are invalid")
    if np.any(table["assessed_value"] <= 0.0) or np.any(table["monthly_rent"] < 0.0):
        raise ValueError("dwelling economic attributes are invalid")
    if np.any(table["tenure"][vacant] != TENURES["vacant"]):
        raise ValueError("vacant dwelling has a non-vacant tenure")
    rental = (table["tenure"] == TENURES["private_rent"]) | (
        table["tenure"] == TENURES["social_rent"]
    )
    if np.any(table["monthly_rent"][rental] <= 0.0):
        raise ValueError("rental dwelling has no positive rent")
    if np.any(table["monthly_rent"][~rental] != 0.0):
        raise ValueError("non-rental dwelling carries rent")
