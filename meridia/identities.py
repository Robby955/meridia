"""Persistent sealed identities for Meridia's institutional layers.

Core modules use compact array indices.  This module turns those import keys into
persistent truth identities before institutional relationships are created.  Truth IDs
remain inside sealed world state; observed registers will receive independently generated
identifiers and a hidden crosswalk in a later layer.
"""

from __future__ import annotations

import numpy as np

ENTITY_NAMESPACE = {
    "person": 1,
    "household": 2,
    "dwelling": 3,
    "business": 4,
    "enterprise": 4,
    "hospital": 5,
    "job": 6,
    "encounter": 7,
    "event": 8,
    "observed_record_source": 9,
    "establishment": 10,
}

NAMESPACE_SHIFT = 56
SEQUENCE_MASK = (1 << NAMESPACE_SHIFT) - 1
UINT64_MASK = (1 << 64) - 1


def _as_integer(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    return int(value)


def _splitmix64(value: int) -> int:
    """One deterministic 64-bit mix, implemented with Python integer arithmetic."""
    z = (value + 0x9E3779B97F4A7C15) & UINT64_MASK
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & UINT64_MASK
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & UINT64_MASK
    return (z ^ (z >> 31)) & UINT64_MASK


def truth_world_id(seed: int, generator_version: int = 0) -> np.uint64:
    """Return the sealed world namespace for a seed and generator version."""
    seed_int = _as_integer("seed", seed)
    version_int = _as_integer("generator_version", generator_version)
    if version_int < 0:
        raise ValueError("generator_version must be nonnegative")
    mixed_input = (
        (seed_int & UINT64_MASK)
        ^ ((version_int * 0xD6E8FEB86659FD93) & UINT64_MASK)
        ^ 0x4D45524944494130
    )  # ASCII-like domain separator: MERIDIA0
    value = _splitmix64(mixed_input)
    # Zero is reserved as the null foreign-key sentinel throughout the truth schema.
    return np.uint64(value if value != 0 else 1)


def truth_entity_ids(entity: str, count: int, start_sequence: int = 1) -> np.ndarray:
    """Allocate a contiguous, namespace-separated block of persistent truth IDs.

    Sequence zero is not allocated.  IDs are unique by construction within a world,
    including across entity types because the namespace occupies the high eight bits.
    """
    if entity not in ENTITY_NAMESPACE:
        choices = ", ".join(sorted(ENTITY_NAMESPACE))
        raise ValueError(
            f"unknown entity namespace {entity!r}; expected one of {choices}"
        )
    count_int = _as_integer("count", count)
    start_int = _as_integer("start_sequence", start_sequence)
    if count_int < 0:
        raise ValueError("count must be nonnegative")
    if start_int < 1:
        raise ValueError("start_sequence must be at least one")
    if count_int and start_int + count_int - 1 > SEQUENCE_MASK:
        raise ValueError("entity sequence exceeds the 56-bit namespace capacity")

    stop = start_int + count_int
    sequence = np.arange(start_int, stop, dtype=np.uint64)
    prefix = np.uint64(ENTITY_NAMESPACE[entity] << NAMESPACE_SHIFT)
    return prefix | sequence


def entity_namespace(ids: np.ndarray) -> np.ndarray:
    """Return the namespace code encoded in each truth entity ID."""
    values = np.asarray(ids)
    if values.dtype != np.uint64 or values.ndim != 1:
        raise TypeError("ids must be a one-dimensional uint64 array")
    return (values >> np.uint64(NAMESPACE_SHIFT)).astype(np.uint8)


def build_initial_identity_map(
    microdata: dict, seed: int, snapshot_tick: int = 0, generator_version: int = 0
) -> dict:
    """Assign persistent truth IDs to an existing deterministic microdata snapshot."""
    try:
        person = microdata["person"]
        person_household = np.asarray(person["household"])
        household_cell = np.asarray(microdata["household_cell"])
        n_persons = int(microdata["n_persons"])
        n_households = int(microdata["n_households"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "microdata does not satisfy the Meridia core snapshot schema"
        ) from exc

    tick = _as_integer("snapshot_tick", snapshot_tick)
    if person_household.ndim != 1 or household_cell.ndim != 1:
        raise ValueError(
            "person household keys and household cells must be one-dimensional"
        )
    if len(person_household) != n_persons:
        raise ValueError("n_persons does not match the person table")
    if len(household_cell) != n_households:
        raise ValueError("n_households does not match household_cell")
    if n_persons < 1 or n_households < 1:
        raise ValueError(
            "the initial identity snapshot must contain persons and households"
        )
    if int(person_household.min()) < 0 or int(person_household.max()) >= n_households:
        raise ValueError("person household import key is outside the household table")
    household_sizes = np.bincount(
        person_household.astype(np.int64), minlength=n_households
    )
    if len(household_sizes) != n_households or np.any(household_sizes == 0):
        raise ValueError("every initial household must contain at least one person")

    return {
        "truth_world_id": truth_world_id(seed, generator_version),
        "generator_version": generator_version,
        "snapshot_tick": np.int64(tick),
        "identity": {
            "truth_person_id": truth_entity_ids("person", n_persons),
            "truth_household_id": truth_entity_ids("household", n_households),
        },
        "n_persons": n_persons,
        "n_households": n_households,
    }
