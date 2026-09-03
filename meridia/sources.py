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

No perfect cross-source person key is shipped.  Each person source reports a name as
two tokens (given and family) drawn from finite vocabularies with a heavy-tailed
frequency law, so distinct persons share a name pair at a rate the development worlds
reveal only in aggregate.  Every source re-reports the pair, the birth tick, and the sex
with its own error process, and every source records the address at its own reference
date: the population source at the snapshot, the income source one year earlier, the
health source at admission.  A mover therefore carries different counties across the
archives for a legitimate reason.

The mechanism rates are one draw per world from published ranges; the hidden world's
draw comes from a published shift family that lies outside the development band.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np

from meridia.events import EVENT_TYPES, replay_event_history
from meridia.identities import SEQUENCE_MASK, truth_world_id
from meridia.mechanisms import (WorldMechanisms, build_world_mechanisms, expit,
                                logit, quintile_band)

OBSERVED_SOURCES: Final = ("population", "business", "income", "health")

# How far the target-dependence axis moves the health source's inclusion logit per log
# unit of latent burden, and the largest shift any one record's burden may produce.
HEALTH_FRAILTY_LOGIT_SCALE: Final = 2.0
HEALTH_FRAILTY_LOGIT_CAP: Final = 2.0

# Observed-token namespaces.  Each source draws a record, a primary entity, and a
# secondary entity pool per vintage: the second entry of each pair is the replacement
# pool an identifier moves into when it does not persist across vintages.
TOKEN_DOMAINS: Final = {
    "population": {"record": (0, 18), "primary": (1, 19), "secondary": (2, 20)},
    "business": {"record": (3, 21), "primary": (4, 22), "secondary": (5, 23)},
    "income": {"record": (6, 24), "primary": (7, 25), "secondary": (8, 26)},
    "health": {"record": (9, 27), "primary": (10, 28), "secondary": (11, 29)},
}
IDENTIFIER_RECENT_MOVE_MONTHS: Final = 12

MECHANISM_BITS: Final = {
    "duplicate": 1,
    "split": 2,
    "merged": 4,
    "stale": 8,
    "county_error": 16,
    "linkage_error": 32,
    "item_missing": 64,
    "address_lag": 128,
    "birth_error": 256,
    "name_error": 512,
}

# Name vocabularies: token counts and the Zipf exponent of the frequency law.  The
# vocabulary sizes and the exponent are public; the tokens themselves are random per
# world, and a variant token exists for every family name (a second spelling).
NAME_VOCABULARY: Final = {"given": 1500, "family": 8000, "zipf": 0.9}
INCOME_ADDRESS_LAG: Final = 12   # the income source records the address one year back

PUBLIC_SCHEMAS: Final = {
    "population": {
        "record_id": np.dtype(np.uint64),
        "person_id": np.dtype(np.uint64),
        "household_id": np.dtype(np.uint64),
        "given_code": np.dtype(np.uint64),
        "family_code": np.dtype(np.uint64),
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
        "given_code": np.dtype(np.uint64),
        "family_code": np.dtype(np.uint64),
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
        "given_code": np.dtype(np.uint64),
        "family_code": np.dtype(np.uint64),
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
    "birth_error": np.dtype(np.bool_),
    "name_error": np.dtype(np.bool_),
}


@dataclass(frozen=True)
class SourceParams:
    """National base rates for the four imperfect sources.

    Every field is a per-world continuous draw from the published development band, and
    none of them is the rate any single record actually experiences: the base rate is
    the intercept of a published family whose slopes are the world's hidden mechanism
    coefficients and whose county effects are drawn per world.  Version three froze
    fifteen of these nineteen numbers across every world, so a participant could measure
    them once on development and carry them unchanged to the hidden world.
    """

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
    name_given_alternate_rate: float = 0.040   # another given name is on file
    name_family_variant_rate: float = 0.040    # the family name's second spelling
    name_transposed_rate: float = 0.010        # given and family entered swapped
    name_missing_rate: float = 0.015           # given name not recorded
    birth_month_slip_rate: float = 0.040       # birth month off by one to three
    birth_year_round_rate: float = 0.030       # birth month rounded to the year
    birth_year_shift_rate: float = 0.010       # birth year off by one
    sex_miscode_rate: float = 0.006
    register_income_scale: float = 1.0         # register earnings unit relative to truth


# Per-world mechanism draw.  The development band is the support every development
# world is drawn from.  The hidden family places each hidden value outside that band
# by at least the stated margin; the direction of the income-scale shift is a fair
# coin.  Published as ranges; a world's realized draw is retained metadata only.
SOURCE_REGIMES: Final = ("development", "hidden")
DEVELOPMENT_BAND: Final = {
    "population_coverage": (0.940, 0.985),
    "business_coverage": (0.900, 0.970),
    "income_coverage": (0.855, 0.940),
    "health_coverage": (0.900, 0.950),
    "outpost_coverage_penalty": (0.090, 0.185),
    "duplicate_rate": (0.014, 0.038),
    "split_rate": (0.006, 0.020),
    "merge_rate": (0.005, 0.017),
    "county_error_rate": (0.012, 0.024),
    "linkage_error_rate": (0.020, 0.052),
    "item_missing_rate": (0.048, 0.105),
    "name_given_alternate_rate": (0.024, 0.058),
    "name_family_variant_rate": (0.024, 0.058),
    "name_transposed_rate": (0.005, 0.017),
    "name_missing_rate": (0.008, 0.024),
    "birth_month_slip_rate": (0.024, 0.058),
    "birth_year_round_rate": (0.017, 0.045),
    "birth_year_shift_rate": (0.005, 0.017),
    "sex_miscode_rate": (0.003, 0.010),
    "register_income_scale": (0.94, 1.06),
}
HIDDEN_SHIFT: Final = {
    "population_coverage_below": (0.02, 0.08),   # subtracted from the band's low edge
    "health_coverage_below": (0.06, 0.20),       # subtracted from the band's low edge
    "county_error_multiplier": (1.5, 3.0),       # times the band's high edge
    "income_level_low": (0.50, 0.63),            # effective register wage level, low side
    "income_level_high": (1.52, 1.90),           # effective register wage level, high side
}
# Effective register wage level = world payroll level x register income scale.  The
# development band on that product follows from the public payroll range (0.75, 1.30)
# times the scale band: (0.705, 1.378).  The hidden family sits outside it on one side.


def draw_source_params(
    seed: int, regime: str = "development", payroll_level: float = 1.0
) -> SourceParams:
    """One deterministic per-world draw of the varying mechanism rates.

    Uses its own seed sequence key so the geography and society draws, and the sealed
    digests built from them, are unchanged.
    """
    if regime not in SOURCE_REGIMES:
        raise ValueError(f"unknown source regime {regime!r}")
    if not np.isfinite(payroll_level) or payroll_level <= 0.0:
        raise ValueError("payroll_level must be positive")
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), 0xC4A3]))
    values = {
        name: float(rng.uniform(*band)) for name, band in sorted(DEVELOPMENT_BAND.items())
    }
    if regime == "hidden":
        values["population_coverage"] = DEVELOPMENT_BAND["population_coverage"][0] - rng.uniform(
            *HIDDEN_SHIFT["population_coverage_below"]
        )
        values["health_coverage"] = DEVELOPMENT_BAND["health_coverage"][0] - rng.uniform(
            *HIDDEN_SHIFT["health_coverage_below"]
        )
        values["county_error_rate"] = DEVELOPMENT_BAND["county_error_rate"][1] * rng.uniform(
            *HIDDEN_SHIFT["county_error_multiplier"]
        )
        side = "income_level_high" if rng.random() < 0.5 else "income_level_low"
        values["register_income_scale"] = rng.uniform(*HIDDEN_SHIFT[side]) / float(payroll_level)
    params = SourceParams(**{name: float(value) for name, value in values.items()})
    _validate_params(params)
    return params


# Benchmark totals: a separately produced aggregate series for the four counts at
# nation and state level.  Each value is the exact count at the snapshot times exp(b):
# at nation level |b| is uniform in the magnitude range with a fair-coin sign, at state
# level b is normal with a world-specific standard deviation from the sd range.  The
# bias is persistent across vintages (the same b in both snapshots) and independent
# across items and units.  Values are rounded to the nearest hundred.  Ranges are
# public; a world's draws are retained metadata only.
BENCHMARK_ITEMS: Final = ("persons", "households", "children_under_16", "elders_65_plus")
BENCHMARK_BIAS: Final = {"nation_magnitude": (0.02, 0.07), "state_sd": (0.03, 0.08),
                         "economic_band_sd": (0.004, 0.015)}
BENCHMARK_ROUNDING: Final = 100

# The benchmark also publishes a count for one defined subgroup, which protocol section 3
# names among the imperfect aggregates the agent receives. The subgroup is a band of
# counties: the benchmark producer classifies every county by its own establishment
# payroll per resident adult and publishes the resident person count of each band, with
# the same bias family and the same reference tick as the rest of the series. The band a
# county sits in is published in ``geography.csv``, so the grouping is reproducible and
# is not a quantity the participant has to estimate.
#
# It exists because the completeness axis had no anchor. Register coverage rides the
# county economic gradient, the covariate that reports that gradient is itself thinned by
# it, and the state series pools counties from both ends of the gradient, so neither the
# register against the survey nor the register against the state benchmark tracked the
# axis: the two statistics read a signed rank correlation of -0.150 and -0.057 over
# eighteen worlds, with the sign reversing between regimes. Against a benchmark published
# on the gradient itself, the register's coverage per band is the gradient.
N_BENCHMARK_BANDS: Final = 4
BENCHMARK_SUBGROUP_ITEM: Final = "persons"
BENCHMARK_BAND_LEVEL: Final = "economic_band"
BENCHMARK_BAND_DEFINITION: Final = (
    "counties in ascending quartiles of establishment payroll per resident adult, as the"
    " benchmark producer measures it; the band of each county is published in"
    " geography.csv as economic_band, and the count is of persons resident at the"
    " snapshot tick, the same reference tick and the same bias family as the nation and"
    " state rows"
)


def benchmark_bands(econ_rank: np.ndarray,
                    n_bands: int = N_BENCHMARK_BANDS) -> np.ndarray:
    """The published economic band of each county, 0 for the lowest quartile.

    ``econ_rank`` is the county's payroll-per-adult rank in [0, 1], the quantity the
    coverage family keys off, so the bands cut the gradient the axis runs along.
    """
    rank = np.asarray(econ_rank, dtype=np.float64)
    return np.clip((rank * int(n_bands)).astype(np.int64), 0, int(n_bands) - 1)


def draw_benchmark_bias(seed: int, n_states: int,
                        n_bands: int = N_BENCHMARK_BANDS) -> dict[str, np.ndarray]:
    """Per-world log-bias of the benchmark series: nation (per item), state (per item
    and state), and economic band.  Own seed sequence key; never derivable from public
    files.  The band draw comes last so the nation and state values of a world do not
    move when the subgroup series is added."""
    if n_states < 1:
        raise ValueError("n_states must be positive")
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), 0xBE4C]))
    n_items = len(BENCHMARK_ITEMS)
    magnitude = rng.uniform(*BENCHMARK_BIAS["nation_magnitude"], size=n_items)
    sign = np.where(rng.random(n_items) < 0.5, -1.0, 1.0)
    state_sd = float(rng.uniform(*BENCHMARK_BIAS["state_sd"]))
    state = rng.normal(0.0, state_sd, size=(n_items, int(n_states)))
    # The subgroup series carries a much smaller unit-level bias than the state series,
    # and its range is published beside it. The state figures come from six separate
    # collections; the subgroup count is one national operation on a classification the
    # producer publishes. It has to be the smaller of the two for the series to be an
    # anchor at all: register coverage moves by about two percent per band across the
    # gradient, so a per-band bias at the state series' spread would be the whole signal.
    band_sd = float(rng.uniform(*BENCHMARK_BIAS["economic_band_sd"]))
    band = rng.normal(0.0, band_sd, size=int(n_bands))
    return {"nation": magnitude * sign, "state": state, "band": band,
            "state_sd": np.float64(state_sd), "band_sd": np.float64(band_sd)}


def benchmark_values(truth: dict, bias: dict, n_states: int,
                     county_band: np.ndarray | None = None,
                     n_bands: int = N_BENCHMARK_BANDS) -> dict[str, np.ndarray]:
    """Benchmark table rows (item, level, unit, value) from exact truth and the bias.

    With ``county_band`` the table also carries the subgroup series: the resident person
    count of each economic band, summed from the county truth and biased on the band's
    own draw.
    """
    items, levels, units, values = [], [], [], []
    for k, item in enumerate(BENCHMARK_ITEMS):
        exact = float(truth[(item, "nation", 0)])
        items.append(item); levels.append("nation"); units.append(0)
        values.append(exact * float(np.exp(bias["nation"][k])))
        for s in range(int(n_states)):
            exact = float(truth[(item, "state", s)])
            items.append(item); levels.append("state"); units.append(s)
            values.append(exact * float(np.exp(bias["state"][k, s])))
    if county_band is not None:
        band = np.asarray(county_band, dtype=np.int64)
        for b in range(int(n_bands)):
            exact = float(sum(truth[(BENCHMARK_SUBGROUP_ITEM, "county", int(c))]
                              for c in np.flatnonzero(band == b)))
            items.append(BENCHMARK_SUBGROUP_ITEM)
            levels.append(BENCHMARK_BAND_LEVEL)
            units.append(b)
            values.append(exact * float(np.exp(bias["band"][b])))
    rounded = np.rint(np.asarray(values, dtype=np.float64) / BENCHMARK_ROUNDING) * BENCHMARK_ROUNDING
    return {
        "item": np.asarray(items),
        "level": np.asarray(levels),
        "unit": np.asarray(units, dtype=np.int64),
        "value": rounded.astype(np.int64),
    }


def _validate_params(params: SourceParams) -> None:
    if not isinstance(params, SourceParams):
        raise TypeError("params must be SourceParams")
    values = {name: float(value) for name, value in params.__dict__.items()}
    scale = values.pop("register_income_scale")
    if not np.isfinite(scale) or not 0.05 <= scale <= 20.0:
        raise ValueError("register_income_scale must be finite and in [0.05, 20]")
    if not np.isfinite(tuple(values.values())).all() or any(
        not 0.0 <= value <= 1.0 for value in values.values()
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


def _zipf_weights(size: int, exponent: float) -> np.ndarray:
    weights = np.arange(1, size + 1, dtype=np.float64) ** (-float(exponent))
    return weights / weights.sum()


def _name_vocabulary(seed: int) -> dict[str, np.ndarray]:
    """Random tokens for every given name, family name, and family-name variant.

    All three sets come from one token draw, so nothing in a token's bits says which
    set it belongs to.  Frequency ranks are never emitted.
    """
    n_given = int(NAME_VOCABULARY["given"])
    n_family = int(NAME_VOCABULARY["family"])
    tokens = _random_tokens(seed, 12, n_given + 2 * n_family)
    return {
        "given": tokens[:n_given],
        "family": tokens[n_given : n_given + n_family],
        "variant": tokens[n_given + n_family :],
        "given_weights": _zipf_weights(n_given, NAME_VOCABULARY["zipf"]),
        "family_weights": _zipf_weights(n_family, NAME_VOCABULARY["zipf"]),
    }


def _true_names(seed: int, vocabulary: dict, n_persons: int) -> tuple[np.ndarray, np.ndarray]:
    """Each truth person's (given, family) vocabulary indices, drawn from the frequency
    law.  Receives a row count only, never a truth ID."""
    rng = np.random.default_rng(np.random.SeedSequence([seed, 0x4E414D, 0]))
    given = rng.choice(len(vocabulary["given"]), size=n_persons, p=vocabulary["given_weights"])
    family = rng.choice(len(vocabulary["family"]), size=n_persons, p=vocabulary["family_weights"])
    return given.astype(np.int64), family.astype(np.int64)


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


_TICK_SHIFT: Final = 1 << 31
_TICK_SPAN: Final = 1 << 32


def _household_cells_at(
    history: dict,
    snapshot_tick: int,
    household_position: np.ndarray,
    at_tick: np.ndarray,
    fallback_cell: np.ndarray,
) -> np.ndarray:
    """Cell of each queried household in force at each queried tick.

    Uses only the address events recorded by ``snapshot_tick`` (the same visibility
    rule as ``_recorded_state``).  A household with no visible address event keeps
    its recorded cell; before its first visible move it sits at that move's origin.
    """
    household_position = np.asarray(household_position, dtype=np.int64)
    at_tick = np.broadcast_to(np.asarray(at_tick, dtype=np.int64), household_position.shape)
    fallback = np.asarray(fallback_cell, dtype=np.int64)
    event = history["event"]
    visible = event["recorded_tick"] <= snapshot_tick
    corrections = event["supersedes_event_id"][visible]
    superseded = corrections[corrections != 0]
    if len(superseded):
        visible &= ~np.isin(event["truth_event_id"], superseded)
    address = visible & np.isin(
        event["event_type"],
        (EVENT_TYPES["household_formed"], EVENT_TYPES["household_moved"]),
    )
    rows = np.flatnonzero(address)
    if len(rows) == 0 or len(household_position) == 0:
        return fallback.copy()
    hh = _sequence_position(event["truth_household_id"][rows])
    tick = np.asarray(event["tick"][rows], dtype=np.int64)
    to_cell = np.asarray(event["to_cell"][rows], dtype=np.int64)
    from_cell = np.asarray(event["from_cell"][rows], dtype=np.int64)
    order = np.lexsort((rows, tick, hh))
    hh, tick, to_cell, from_cell = hh[order], tick[order], to_cell[order], from_cell[order]
    first = np.ones(len(hh), dtype=np.bool_)
    first[1:] = hh[1:] != hh[:-1]
    origin_cell = np.where(from_cell[first] >= 0, from_cell[first], to_cell[first])
    all_hh = np.concatenate([hh[first], hh])
    all_tick = np.concatenate([np.full(int(first.sum()), -_TICK_SHIFT + 1, dtype=np.int64), tick])
    all_cell = np.concatenate([origin_cell, to_cell])
    order = np.lexsort((np.arange(len(all_hh)), all_tick, all_hh))
    all_hh, all_tick, all_cell = all_hh[order], all_tick[order], all_cell[order]
    keys = all_hh * _TICK_SPAN + (all_tick + _TICK_SHIFT)
    query = household_position * _TICK_SPAN + (at_tick + _TICK_SHIFT)
    index = np.searchsorted(keys, query, side="right") - 1
    valid = (index >= 0) & (household_position >= 0)
    valid[valid] &= all_hh[index[valid]] == household_position[valid]
    result = fallback.copy()
    result[valid] = all_cell[index[valid]]
    return result


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


def _reported_identity(
    rng: np.random.Generator,
    count: int,
    identity: dict,
    rates: dict[str, np.ndarray],
    linkage_error: np.ndarray,
    split: np.ndarray,
    merge_pairs: np.ndarray,
) -> dict[str, np.ndarray]:
    """Per-row, per-slot reported name, birth tick, and sex under a reporting-error
    process.  Slot 1 (the duplicate record) draws independently, so duplicates are
    near-duplicates.  A linkage error reports another same-county person's name: a
    true confusable, not a token that matches nothing.  The second member of a merge
    pair reports the first member's name.  A split's second record carries the other
    spelling of the family name."""
    person_position = np.asarray(identity["person_position"], dtype=np.int64)
    given = np.asarray(identity["given"], dtype=np.int64)
    family = np.asarray(identity["family"], dtype=np.int64)
    n_given, n_family = len(identity["given_weights"]), len(identity["family_weights"])
    true_birth = np.asarray(identity["birth_tick"], dtype=np.int64)
    true_sex = np.asarray(identity["sex"], dtype=np.int8)
    person_county = np.asarray(identity["person_county"], dtype=np.int64)

    # Same-county confusables: a random other person from the row person's county.
    order = np.argsort(person_county, kind="stable")
    county_size = np.bincount(person_county, minlength=int(person_county.max()) + 1 if len(person_county) else 1)
    county_start = np.cumsum(county_size) - county_size
    row_county = person_county[person_position]
    confusable = order[
        county_start[row_county]
        + np.minimum(
            (rng.random(count) * county_size[row_county]).astype(np.int64),
            np.maximum(county_size[row_county] - 1, 0),
        )
    ] if count else np.empty(0, dtype=np.int64)
    swapped = np.where(linkage_error, confusable, person_position)

    given_index = np.zeros((count, 2), dtype=np.int64)
    family_index = np.zeros((count, 2), dtype=np.int64)
    family_variant = np.zeros((count, 2), dtype=np.bool_)
    transposed = np.zeros((count, 2), dtype=np.bool_)
    given_missing = np.zeros((count, 2), dtype=np.bool_)
    birth = np.zeros((count, 2), dtype=np.int64)
    sex = np.zeros((count, 2), dtype=np.int8)
    name_error = np.zeros((count, 2), dtype=np.bool_)
    birth_error = np.zeros((count, 2), dtype=np.bool_)
    for slot in range(2):
        alternate = rng.random(count) < rates["name_given_alternate"]
        alternate_given = rng.choice(n_given, size=count, p=identity["given_weights"]) if count else np.empty(0, dtype=np.int64)
        variant = rng.random(count) < rates["name_family_variant"]
        swap = rng.random(count) < rates["name_transposed"]
        missing = rng.random(count) < rates["name_missing"]
        slip = rng.random(count) < rates["birth_month_slip"]
        slip_size = rng.integers(1, 4, size=count) * np.where(rng.random(count) < 0.5, -1, 1)
        rounding = rng.random(count) < rates["birth_year_round"]
        year_shift = rng.random(count) < rates["birth_year_shift"]
        year_sign = np.where(rng.random(count) < 0.5, -12, 12)
        miscode = rng.random(count) < rates["sex_miscode"]
        if slot == 1:
            variant = np.where(split, ~family_variant[:, 0], variant)

        given_index[:, slot] = np.where(alternate, alternate_given, given[swapped])
        family_index[:, slot] = family[swapped]
        family_variant[:, slot] = variant
        transposed[:, slot] = swap
        given_missing[:, slot] = missing
        reported_birth = true_birth[person_position].copy()
        reported_birth[slip] += slip_size[slip]
        reported_birth[rounding] = (reported_birth[rounding] // 12) * 12
        reported_birth[year_shift] += year_sign[year_shift]
        birth[:, slot] = reported_birth
        reported_sex = true_sex[person_position].copy()
        reported_sex[miscode] = 1 - reported_sex[miscode]
        sex[:, slot] = reported_sex
        name_error[:, slot] = alternate | variant | swap | missing | linkage_error
        birth_error[:, slot] = (reported_birth != true_birth[person_position]) | miscode

    for first, second in merge_pairs:
        given_index[second, :] = given_index[first, 0]
        family_index[second, :] = family_index[first, 0]
        family_variant[second, :] = family_variant[first, 0]
        transposed[second, :] = transposed[first, 0]
        given_missing[second, :] = given_missing[first, 0]
        name_error[second, :] = True
    return {
        "given_index": given_index,
        "family_index": family_index,
        "family_variant": family_variant,
        "transposed": transposed,
        "given_missing": given_missing,
        "reported_birth_tick": birth,
        "reported_sex": sex,
        "name_error_slot": name_error,
        "birth_error_slot": birth_error,
    }


def _record_mechanism_rates(
    params: SourceParams,
    mechanisms: WorldMechanisms,
    source: str,
    county: np.ndarray,
    band: np.ndarray,
    frailty: np.ndarray,
    age_years: np.ndarray,
    coverage_base: np.ndarray,
) -> dict[str, np.ndarray]:
    """Per-record probability of every declared defect, from the published families.

    The forms are in ``mechanisms.contract_block``; the slopes and the county effects
    are the world's hidden coefficients.  Rurality drives name, address, and linkage
    error; county economic condition drives coverage and item missingness; latent
    frailty drives health-source inclusion, at a slope the completeness axis modulates;
    the record's own money band drives missingness on the value being estimated.  None of these is a world constant, so a
    rate measured on one world does not transfer to another.
    """
    coefficients = mechanisms.coefficients
    urban = mechanisms.covariate("urban", county)
    econ = mechanisms.covariate("econ", county)
    elder = mechanisms.covariate("elder", county)
    # The rural excess in name, address and linkage error is itself scaled by the world's
    # migration intensity. That product of two axes is the first of the three the protocol
    # predeclares: a world that moves people harder loses more of them at the rural end of
    # the gradient, so linkage quality cannot be read off urbanity alone and the two axes
    # cannot be fitted one at a time on the development design.
    rural_gradient = float(coefficients["linkage_urban_gradient"]) * (
        1.0
        + float(coefficients["linkage_gradient_by_migration"])
        * (float(coefficients["migration_age_pattern"]) - 1.0)
    )
    linkage_shift = (
        rural_gradient * (0.5 - urban)
        + float(coefficients["linkage_intercept_shift"])
        + mechanisms.effect("linkage", county)
    )
    coverage_shift = (
        float(coefficients["administrative_completeness"]) * (econ - 0.5)
        + float(coefficients["coverage_elder_slope"]) * (elder - 0.5)
        + float(coefficients["coverage_intercept_shift"])
        + mechanisms.effect("coverage", county)
    )
    if source == "health":
        # The target-dependence axis carries exactly one mechanism: how strongly health
        # inclusion reads latent morbidity. Its declared interaction with administrative
        # completeness is the product of two axes, so a method that fits the two
        # separately on the design does not transfer to a configuration it has not seen.
        frailty_slope = float(coefficients["missingness_target_dependence"]) * (
            1.0
            + float(coefficients["health_inclusion_completeness_by_target"])
            * (float(coefficients["administrative_completeness"]) - 1.0)
        )
        # The slope is scaled and then capped. Latent burden's own mean moves by about
        # half a log unit between a child and a person of eighty, so at the raw slope an
        # axis at the middle of its band moved the inclusion share by six points across
        # the whole age range, under a survey anchor whose false positives are of the
        # same order. The axis had a mechanism and no readable trace. The cap keeps the
        # far tail of the burden distribution from driving an inclusion probability to
        # zero or one, which protocol section 10 refuses as underidentified.
        coverage_shift = coverage_shift + np.clip(
            HEALTH_FRAILTY_LOGIT_SCALE
            * frailty_slope
            * np.log(np.clip(frailty, 0.15, 6.0)),
            -HEALTH_FRAILTY_LOGIT_CAP,
            HEALTH_FRAILTY_LOGIT_CAP,
        )
    # Item missingness on the money value has its own money-band slope. In version four's
    # first pass this was the target-dependence axis again, which loaded one coefficient
    # onto two mechanisms with different targets and left neither identified. Its county
    # effect is its own draw as well: reusing the coverage effect verbatim made the two
    # county patterns identical, so measuring where a register is thin also measured
    # where its values go missing, and one estimate did for both.
    missing_shift = (
        float(coefficients["item_missing_econ_slope"]) * (econ - 0.5)
        + float(coefficients["item_missing_band_slope"])
        * (np.asarray(band, dtype=np.float64) - 2.0)
        / 2.0
        + mechanisms.effect("item_missing", county)
    )
    age_multiplier = (
        float(coefficients["age_reporting_error"])
        * float(coefficients.get("age_error_mortality_scale", 1.0))
        * (
            1.0
            + float(coefficients["age_error_age_slope"])
            * (np.asarray(age_years, dtype=np.float64) - 45.0)
            / 40.0
        )
    )
    age_multiplier = np.clip(age_multiplier, 0.05, 6.0)

    def shifted(base: float, shift: np.ndarray, multiplier=1.0) -> np.ndarray:
        level = np.clip(np.asarray(base, dtype=np.float64) * multiplier, 1e-6, 0.60)
        return np.clip(expit(logit(level) + shift), 1e-6, 0.90)

    return {
        "covered": np.clip(
            expit(logit(np.clip(coverage_base, 1e-4, 0.9999)) + coverage_shift),
            1e-4,
            0.9999,
        ),
        "duplicate": shifted(params.duplicate_rate, linkage_shift),
        "split": shifted(params.split_rate, linkage_shift),
        "merge": shifted(params.merge_rate, linkage_shift),
        "county_error": shifted(params.county_error_rate, linkage_shift),
        "linkage_error": shifted(params.linkage_error_rate, linkage_shift),
        "item_missing": shifted(params.item_missing_rate, missing_shift),
        "name_given_alternate": shifted(params.name_given_alternate_rate, linkage_shift),
        "name_family_variant": shifted(params.name_family_variant_rate, linkage_shift),
        "name_transposed": shifted(params.name_transposed_rate, linkage_shift),
        "name_missing": shifted(params.name_missing_rate, linkage_shift),
        "birth_month_slip": shifted(params.birth_month_slip_rate, linkage_shift, age_multiplier),
        "birth_year_round": shifted(params.birth_year_round_rate, linkage_shift, age_multiplier),
        "birth_year_shift": shifted(params.birth_year_shift_rate, linkage_shift, age_multiplier),
        "sex_miscode": shifted(params.sex_miscode_rate, linkage_shift),
    }


def _identifier_persistence(
    mechanisms: WorldMechanisms, county: np.ndarray, recent_move: np.ndarray
) -> np.ndarray:
    """Probability that an observed identifier survives from one vintage to the next.

    Version three drew one identifier per truth entity and reused it in both vintages,
    so a preliminary-to-revised join was exact and longitudinal matching was free.  Here
    persistence is a declared family in the county's urbanity and whether the entity
    moved recently, and the stale-address term is itself scaled by the world's migration
    intensity: that product of two axes is the migration by stale-address-linkage
    interaction the protocol predeclares.  A world that moves people harder also loses
    their identifiers faster, so linkage quality cannot be read off urbanity alone.
    """
    coefficients = mechanisms.coefficients
    urban = mechanisms.covariate("urban", county)
    move_term = float(coefficients["id_persist_recent_move"]) * (
        1.0
        + float(coefficients["id_persist_move_by_migration"])
        * (float(coefficients["migration_age_pattern"]) - 1.0)
    )
    return expit(
        float(coefficients["id_persist_intercept"])
        + float(coefficients["id_persist_urban"]) * (urban - 0.5)
        + move_term * np.asarray(recent_move, dtype=np.float64)
        + mechanisms.effect("id_persist", county)
    )


def _vintage_identifiers(
    seed: int,
    source_index: int,
    vintage: int,
    count: int,
    domains: dict[str, tuple[int, int]],
    persistence: np.ndarray,
    reissue_rate: float,
    merge_pairs: np.ndarray,
) -> dict[str, np.ndarray]:
    """Record and entity identifiers for one vintage of one source.

    Record identifiers are always fresh: a file row is not the same row next vintage.
    An entity identifier survives with the declared probability, is replaced from a
    second pool when it does not, and is sometimes reissued to a different entity.
    """
    rng = np.random.default_rng(
        np.random.SeedSequence([int(seed), 0x1DC8, int(source_index), int(vintage)])
    )
    record_id = _random_tokens(seed, domains["record"][vintage], 2 * count).reshape(count, 2)
    primary_id = _random_tokens(seed, domains["primary"][0], count)
    secondary_id = _random_tokens(seed, domains["secondary"][0], count)
    if vintage > 0 and count:
        alternate_primary = _random_tokens(seed, domains["primary"][1], count)
        alternate_secondary = _random_tokens(seed, domains["secondary"][1], count)
        persists = rng.random(count) < np.asarray(persistence, dtype=np.float64)
        released = np.flatnonzero(~persists)
        primary_id = np.where(persists, primary_id, alternate_primary)
        secondary_id = np.where(persists, secondary_id, alternate_secondary)
        reissued = released[rng.random(len(released)) < float(reissue_rate)]
        if len(reissued):
            donor = _random_tokens(seed, domains["primary"][0], count)
            primary_id[reissued] = donor[(reissued + 1) % count]
    for pair in merge_pairs:
        primary_id[int(pair[1])] = primary_id[int(pair[0])]
    return {
        "record_id": record_id,
        "primary_id": primary_id.astype(np.uint64, copy=False),
        "secondary_id": secondary_id.astype(np.uint64, copy=False),
    }


def _mechanism_plan(
    seed: int,
    source_index: int,
    source: str,
    truth_ids: np.ndarray,
    county: np.ndarray,
    county_is_outpost: np.ndarray,
    coverage: float,
    params: SourceParams,
    mechanisms: WorldMechanisms,
    band: np.ndarray,
    frailty: np.ndarray,
    age_years: np.ndarray,
    identity: dict | None = None,
) -> dict:
    """Everything about a source's defects that belongs to the truth entity itself.

    Identifiers are not here: they are drawn per vintage by ``_vintage_identifiers``, so
    the preliminary and revised files no longer share a record key.
    """
    count = len(truth_ids)
    rng = np.random.default_rng(np.random.SeedSequence([seed, 0xAE61A7E, source_index]))
    probability = np.full(count, coverage, dtype=np.float64)
    n_counties = len(county_is_outpost)
    if count:
        probability -= params.outpost_coverage_penalty * county_is_outpost[county]
    rates = _record_mechanism_rates(
        params, mechanisms, source, county, band, frailty, age_years, probability
    )
    covered = rng.random(count) < rates["covered"]
    split = rng.random(count) < rates["split"]
    duplicate = (rng.random(count) < rates["duplicate"]) | split
    county_error = rng.random(count) < rates["county_error"]
    if n_counties == 1:
        county_error[:] = False
    linkage_error = rng.random(count) < rates["linkage_error"]
    item_missing = rng.random(count) < rates["item_missing"]
    county_error_offset = (
        rng.integers(1, n_counties, size=count).astype(np.int32)
        if n_counties > 1 and count
        else np.zeros(count, dtype=np.int32)
    )

    merge_group = np.full(count, -1, dtype=np.int64)
    merge_candidate = np.flatnonzero(rng.random(count) < rates["merge"])
    if len(merge_candidate) % 2:
        merge_candidate = merge_candidate[:-1]
    merge_pairs = merge_candidate.reshape(-1, 2)
    for group, pair in enumerate(merge_pairs):
        merge_group[int(pair[0])] = merge_group[int(pair[1])] = group

    plan = {
        "truth_entity_id": np.asarray(truth_ids, dtype=np.uint64).copy(),
        "covered": covered.astype(np.bool_),
        "duplicate": duplicate.astype(np.bool_),
        "split": split.astype(np.bool_),
        "merge_group": merge_group,
        "county_error": county_error.astype(np.bool_),
        "linkage_error": linkage_error.astype(np.bool_),
        "item_missing": item_missing.astype(np.bool_),
        "county_error_offset": county_error_offset,
        "merge_pairs": merge_pairs,
        "rates": rates,
    }
    if identity is not None:
        plan.update(
            _reported_identity(rng, count, identity, rates, plan["linkage_error"], plan["split"], merge_pairs)
        )
    else:
        plan["name_error_slot"] = np.zeros((count, 2), dtype=np.bool_)
        plan["birth_error_slot"] = np.zeros((count, 2), dtype=np.bool_)
    plan["name_error"] = plan["name_error_slot"][:, 0].copy()
    plan["birth_error"] = plan["birth_error_slot"][:, 0].copy()
    return plan


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
    for name in ("name_error", "birth_error"):
        mechanism_code |= np.where(
            plan[f"{name}_slot"][position, slot], MECHANISM_BITS[name], 0
        ).astype(np.int16)
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
        offset = plan["county_error_offset"][position[error]].astype(np.int32)
        county[error] = (county[error] + offset) % n_counties
    return county


def _name_codes(
    vocabulary: dict, rows: dict, plan: dict
) -> tuple[np.ndarray, np.ndarray]:
    """Reported (given_code, family_code) tokens for each public row."""
    position, slot = rows["position"], rows["slot"]
    given_index = plan["given_index"][position, slot]
    family_index = plan["family_index"][position, slot]
    variant = plan["family_variant"][position, slot]
    given_code = np.asarray(vocabulary["given"][given_index], dtype=np.uint64).copy()
    family_code = np.where(
        variant, vocabulary["variant"][family_index], vocabulary["family"][family_index]
    ).astype(np.uint64)
    transposed = plan["transposed"][position, slot]
    given_code, family_code = (
        np.where(transposed, family_code, given_code).astype(np.uint64),
        np.where(transposed, given_code, family_code).astype(np.uint64),
    )
    given_code[plan["given_missing"][position, slot]] = np.uint64(0)
    return given_code, family_code


def _reported_birth_and_sex(rows: dict, plan: dict) -> tuple[np.ndarray, np.ndarray]:
    position, slot = rows["position"], rows["slot"]
    return (
        plan["reported_birth_tick"][position, slot].astype(np.int64, copy=True),
        plan["reported_sex"][position, slot].astype(np.int8, copy=True),
    )


def _flag_address_lag(rows: dict, reported_county: np.ndarray, snapshot_county: np.ndarray) -> None:
    """Mark rows whose reference-date address differs from the snapshot address."""
    position = rows["position"]
    lagged = reported_county != snapshot_county[position]
    rows["mechanism_code"] = (
        rows["mechanism_code"]
        | np.where(lagged, MECHANISM_BITS["address_lag"], 0).astype(np.int16)
    ).astype(np.int16)


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


def _recorded_county(cell: np.ndarray, county_flat: np.ndarray) -> np.ndarray:
    """County of a recorded cell; -1 where no address has been recorded yet.

    A person whose household formed or moved in a late-reported event carries a cell
    of -1 in the recorded state. Indexing the county grid with -1 would silently read
    the last grid cell, which at some seeds is sea. Such a person has no observable
    address in that vintage and must not appear in an address-bearing source.
    """
    cell = np.asarray(cell, dtype=np.int64)
    return np.where(cell >= 0, county_flat[np.maximum(cell, 0)], -1).astype(np.int64)


def _population_source(
    state: dict,
    plan: dict,
    stale: np.ndarray,
    county_flat: np.ndarray,
    n_counties: int,
    household_id: np.ndarray,
    vocabulary: dict,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    person = state["person"]
    true_county = _recorded_county(person["cell"], county_flat)
    active = person["exists"] & person["is_alive"] & (true_county >= 0)
    rows = _expand_rows(plan, active, stale)
    position = rows["position"]
    household_position = _sequence_position(person["truth_household_id"][position])
    education = person["education"][position].astype(np.int8, copy=True)
    education[plan["item_missing"][position]] = -1
    given_code, family_code = _name_codes(vocabulary, rows, plan)
    birth_tick, sex = _reported_birth_and_sex(rows, plan)
    table = {
        "record_id": rows["record_id"],
        "person_id": rows["entity_id"],
        "household_id": household_id[household_position],
        "given_code": given_code,
        "family_code": family_code,
        "birth_tick": birth_tick,
        "sex": sex,
        "education": education,
        "county": _county_values(true_county, rows, plan, n_counties),
    }
    order = np.argsort(table["record_id"], kind="stable")
    return _sort_table(table, order), _crosswalk(
        rows, plan, person["visible_from_tick"]
    )


def _local_money_scale(
    mechanisms: WorldMechanisms,
    income_scale: float,
    county: np.ndarray,
    band: np.ndarray,
) -> np.ndarray:
    """Register money unit for one record: s_cr = s_0 * exp(b1 urban_c + b2 band_r + u_c).

    Version three multiplied every money value in both registers by one world-global
    float, so a single national ratio of register earnings to survey income recovered
    the unit exactly.  Here the unit varies by county and by income band, and the
    survey still reports true-scale income in every county, so b1, b2 and the county
    spread stay identifiable while the level does not fall out of one ratio.
    """
    coefficients = mechanisms.coefficients
    urban = mechanisms.covariate("urban", county)
    return float(income_scale) * np.exp(
        float(coefficients["income_scale_urban"]) * (urban - 0.5)
        + float(coefficients["income_scale_band"])
        * (np.asarray(band, dtype=np.float64) - 2.0)
        / 2.0
        + mechanisms.effect("income_scale", county)
    )


def _business_source(
    state: dict,
    plan: dict,
    stale: np.ndarray,
    county_flat: np.ndarray,
    n_counties: int,
    enterprise_id: np.ndarray,
    income_scale: float,
    mechanisms: WorldMechanisms,
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
    payroll_value = np.rint(
        payroll[position].astype(np.float64)
        * _local_money_scale(
            mechanisms,
            income_scale,
            np.maximum(true_county[position], 0),
            quintile_band(payroll)[position],
        )
    )
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
    vocabulary: dict,
    business_plan: dict,
    history: dict,
    snapshot_tick: int,
    income_scale: float,
    mechanisms: WorldMechanisms,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    person = state["person"]
    snapshot_county = _recorded_county(person["cell"], county_flat)
    active = person["exists"] & person["is_alive"] & (snapshot_county >= 0)
    rows = _expand_rows(plan, active, stale)
    position = rows["position"]
    earnings, employer_position, _, _ = _employment_summary(
        state,
        len(person["truth_person_id"]),
        len(state["establishment"]["truth_establishment_id"]),
    )
    household_position = _sequence_position(person["truth_household_id"][position])
    # Address on file is the household's address one year before the snapshot.
    lagged_cell = _household_cells_at(
        history,
        snapshot_tick,
        household_position,
        snapshot_tick - INCOME_ADDRESS_LAG,
        person["cell"][position],
    )
    true_county = snapshot_county.copy()
    true_county[position] = _recorded_county(lagged_cell, county_flat)
    income = np.rint(
        earnings[position].astype(np.float64)
        * _local_money_scale(
            mechanisms,
            income_scale,
            np.maximum(true_county[position], 0),
            quintile_band(earnings)[position],
        )
    )
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
    _flag_address_lag(rows, true_county[position], snapshot_county)
    given_code, family_code = _name_codes(vocabulary, rows, plan)
    birth_tick, sex = _reported_birth_and_sex(rows, plan)
    table = {
        "record_id": rows["record_id"],
        "taxpayer_id": rows["entity_id"],
        "household_id": household_id[household_position],
        "given_code": given_code,
        "family_code": family_code,
        "birth_tick": birth_tick,
        "sex": sex,
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
    vocabulary: dict,
    patient_id: np.ndarray,
    facility_id: np.ndarray,
    history: dict,
    snapshot_tick: int,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    encounter = state["encounter"]
    person = state["person"]
    all_person_position = _sequence_position(encounter["truth_person_id"])
    snapshot_patient_county = _recorded_county(
        person["cell"][all_person_position], county_flat
    )
    # An encounter whose patient has no recorded address yet is not observable in
    # this vintage.
    rows = _expand_rows(plan, encounter["exists"] & (snapshot_patient_county >= 0), stale)
    position = rows["position"]
    person_position = _sequence_position(encounter["truth_person_id"][position])
    hospital_position = _sequence_position(encounter["truth_hospital_id"][position])
    facility_county = county_flat[state["hospital_cell"]]
    # Patient address as recorded at admission: the household's cell in force then.
    admission_cell = _household_cells_at(
        history,
        snapshot_tick,
        _sequence_position(person["truth_household_id"][person_position]),
        encounter["admission_tick"][position],
        person["cell"][person_position],
    )
    patient_county = snapshot_patient_county.copy()
    patient_county[position] = _recorded_county(admission_cell, county_flat)
    _flag_address_lag(rows, patient_county[position], snapshot_patient_county)

    observed_patient = patient_id[person_position].copy()
    wrong_link = plan["linkage_error"][position]
    if wrong_link.any():
        observed_patient[wrong_link] = patient_id[
            (person_position[wrong_link] + 1) % len(patient_id)
        ]
    given_code, family_code = _name_codes(vocabulary, rows, plan)
    birth_tick, sex = _reported_birth_and_sex(rows, plan)
    cost = encounter["cost_cents"][position].astype(np.float64)
    cost[plan["item_missing"][position]] = np.nan
    table = {
        "record_id": rows["record_id"],
        "encounter_id": rows["entity_id"],
        "patient_id": observed_patient,
        "facility_id": facility_id[hospital_position],
        "given_code": given_code,
        "family_code": family_code,
        "birth_tick": birth_tick,
        "sex": sex,
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


def _recent_move_flags(
    history: dict,
    truth_household_id: np.ndarray,
    at_tick: int,
    window: int = IDENTIFIER_RECENT_MOVE_MONTHS,
) -> np.ndarray:
    """One where the entity's household changed address inside the last ``window`` months.

    This is the observable side of the migration by stale-address-linkage interaction:
    a recent mover's register identifier is the one most likely to be reissued.
    """
    event = history["event"]
    address_change = np.isin(
        event["event_type"],
        np.asarray(
            [EVENT_TYPES["household_moved"], EVENT_TYPES["household_formed"]],
            dtype=event["event_type"].dtype,
        ),
    )
    recent = address_change & (event["tick"] > at_tick - window) & (event["tick"] <= at_tick)
    moved_households = np.unique(event["truth_household_id"][recent])
    return np.isin(
        np.asarray(truth_household_id, dtype=np.uint64), moved_households
    ).astype(np.float64)


def _build_observed_sources_unchecked(
    history: dict,
    seed: int,
    admin: dict,
    hospitals: dict,
    preliminary_tick: int,
    revised_tick: int,
    params: SourceParams,
    mechanisms: WorldMechanisms,
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
    domains = TOKEN_DOMAINS
    n_persons = _table_length(person)
    vocabulary = _name_vocabulary(seed)
    true_given, true_family = _true_names(seed, vocabulary, n_persons)
    person_identity = {
        "given": true_given,
        "family": true_family,
        "given_weights": vocabulary["given_weights"],
        "family_weights": vocabulary["family_weights"],
        "birth_tick": np.asarray(person["birth_tick"], dtype=np.int64),
        "sex": np.asarray(person["sex"], dtype=np.int8),
        "person_county": person_county,
    }
    identities = {
        "population": {**person_identity, "person_position": np.arange(n_persons)},
        "business": None,
        "income": {**person_identity, "person_position": np.arange(n_persons)},
        "health": {
            **person_identity,
            "person_position": _sequence_position(encounter["truth_person_id"]),
        },
    }
    # Covariates every local mechanism keys off, one value per truth entity.
    earnings_all, _, _, payroll_all = _employment_summary(
        terminal, n_persons, _table_length(establishment)
    )
    person_frailty = np.asarray(person["frailty_centi"], dtype=np.float64) / 100.0
    person_age_years = np.maximum(
        0.0, (revised_tick - np.asarray(person["birth_tick"], dtype=np.float64)) / 12.0
    )
    encounter_person_position = _sequence_position(encounter["truth_person_id"])
    person_band = quintile_band(earnings_all)
    recent_move = _recent_move_flags(
        history, np.asarray(person["truth_household_id"], dtype=np.uint64), revised_tick
    )
    covariates = {
        "population": {
            "band": person_band,
            "frailty": person_frailty,
            "age_years": person_age_years,
            "recent_move": recent_move,
        },
        "business": {
            "band": quintile_band(payroll_all),
            "frailty": np.ones(_table_length(establishment)),
            "age_years": np.full(_table_length(establishment), 45.0),
            "recent_move": np.zeros(_table_length(establishment), dtype=np.float64),
        },
        "income": {
            "band": person_band,
            "frailty": person_frailty,
            "age_years": person_age_years,
            "recent_move": recent_move,
        },
        "health": {
            "band": quintile_band(np.asarray(encounter["cost_cents"], dtype=np.float64)),
            "frailty": person_frailty[encounter_person_position],
            "age_years": person_age_years[encounter_person_position],
            "recent_move": recent_move[encounter_person_position],
        },
    }
    plans = {
        source: _mechanism_plan(
            seed,
            source_index,
            source,
            truth_ids[source],
            counties[source],
            county_is_outpost,
            coverage[source],
            params,
            mechanisms,
            covariates[source]["band"],
            covariates[source]["frailty"],
            covariates[source]["age_years"],
            identities[source],
        )
        for source_index, source in enumerate(OBSERVED_SOURCES)
    }
    persistence = {
        source: _identifier_persistence(
            mechanisms,
            np.maximum(counties[source], 0),
            covariates[source]["recent_move"],
        )
        for source in OBSERVED_SOURCES
    }

    n_households = _table_length(terminal["household"])
    enterprise_position = _sequence_position(establishment["truth_enterprise_id"])
    n_enterprises = int(enterprise_position.max()) + 1
    n_hospitals = len(hospitals["hospital"]["truth_hospital_id"])
    income_scale = float(params.register_income_scale)
    params_reissue_rate = float(mechanisms.coefficients["id_reissue_rate"])
    population_household_id = _random_tokens(seed, 13, n_households)
    income_household_id = _random_tokens(seed, 14, n_households)
    enterprise_id = _random_tokens(seed, 15, n_enterprises)
    patient_id = _random_tokens(seed, 16, _table_length(person))
    facility_id = _random_tokens(seed, 17, n_hospitals)

    public_snapshots: dict[str, dict] = {}
    crosswalks: dict[str, dict] = {}
    for vintage, (label, snapshot_tick) in enumerate(
        (("preliminary", preliminary_tick), ("revised", revised_tick))
    ):
        recorded = _recorded_state(history, hospitals, snapshot_tick)
        exact = _padded_exact_state(history, snapshot_tick)
        stale = _stale_flags(recorded, exact)
        # Identifiers are redrawn for this vintage: record keys are always new, and an
        # entity key survives only with its declared probability.
        vintage_plans = {
            source: {
                **plans[source],
                **_vintage_identifiers(
                    seed,
                    source_index,
                    vintage,
                    len(truth_ids[source]),
                    domains[source],
                    persistence[source],
                    params_reissue_rate,
                    plans[source]["merge_pairs"],
                ),
            }
            for source_index, source in enumerate(OBSERVED_SOURCES)
        }
        population_table, population_crosswalk = _population_source(
            recorded,
            vintage_plans["population"],
            stale["population"],
            county_flat,
            n_counties,
            population_household_id,
            vocabulary,
        )
        business_table, business_crosswalk = _business_source(
            recorded,
            vintage_plans["business"],
            stale["business"],
            county_flat,
            n_counties,
            enterprise_id,
            income_scale,
            mechanisms,
        )
        income_table, income_crosswalk = _income_source(
            recorded,
            vintage_plans["income"],
            stale["income"],
            county_flat,
            n_counties,
            income_household_id,
            vocabulary,
            vintage_plans["business"],
            history,
            snapshot_tick,
            income_scale,
            mechanisms,
        )
        health_table, health_crosswalk = _health_source(
            recorded,
            vintage_plans["health"],
            stale["health"],
            county_flat,
            n_counties,
            vocabulary,
            patient_id,
            facility_id,
            history,
            snapshot_tick,
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
        "source_schema_version": 2,
        "preliminary_tick": np.int64(preliminary_tick),
        "revised_tick": np.int64(revised_tick),
        "source_params": _params_record(params),
        "mechanism_record": mechanisms.record(),
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
        "mechanism_record",
        "public_snapshots",
        "hidden",
    }
    if set(package) != expected_top:
        raise ValueError("source package top-level fields differ from schema")
    if int(package["source_schema_version"]) != 2:
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
    mechanisms: WorldMechanisms | None = None,
) -> dict:
    """Build two deterministic observed-source snapshots plus sealed evidence.

    ``mechanisms`` carries the world's local defect coefficients and county effects.
    Without it the world is treated as a single neutral county, which is what the
    standalone source tests use; a packet always supplies the real one.
    """
    _validate_params(params)
    if mechanisms is None:
        mechanisms = build_world_mechanisms(int(seed), "development", admin)
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
        mechanisms,
    )
    _validate_structure(package, history, admin)
    return package


def validate_observed_sources(
    package: dict,
    history: dict,
    seed: int,
    admin: dict,
    hospitals: dict,
    mechanisms: WorldMechanisms | None = None,
) -> None:
    """Fail unless a package is the exact seeded build from the retained inputs."""
    if mechanisms is None:
        mechanisms = build_world_mechanisms(int(seed), "development", admin)
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
        mechanisms,
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
