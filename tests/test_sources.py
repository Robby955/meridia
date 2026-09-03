"""Imperfect sources: recorded-time cuts, county provenance, and sealed errors."""

from __future__ import annotations

import hashlib
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from meridia.admin import build_admin
from meridia.businesses import build_businesses
from meridia.character import draw_world_character
from meridia.dwellings import build_dwellings
from meridia.events import EVENT_TYPES, build_event_history
from meridia.hospitals import build_hospitals
from meridia.hydrology import fill_depressions, flow_accumulation, flow_directions
from meridia.identities import build_initial_identity_map
from meridia.microdata import build_microdata
from meridia.population import build_population, resource_outposts
from meridia.sources import MECHANISM_BITS, PUBLIC_SCHEMAS, OBSERVED_SOURCES
from meridia.sources import SourceParams, build_observed_sources
from meridia.sources import participant_source_snapshots, validate_observed_sources
from meridia.terrain import generate_elevation

SEED = 20260831
H, W = 60, 72
TOTAL = 40_000


@lru_cache(maxsize=1)
def _setup():
    character = draw_world_character(SEED)
    world = generate_elevation(SEED, H, W)
    outlets = ~world["land"]
    outlets[0, :] = outlets[-1, :] = outlets[:, 0] = outlets[:, -1] = True
    filled = fill_depressions(world["elevation"], world["sea_level"])
    direction = flow_directions(filled, outlets)
    accumulation = flow_accumulation(direction, outlets)
    people = build_population(
        world,
        accumulation,
        TOTAL,
        6,
        params=character["population"],
        seed=SEED,
    )
    micro = build_microdata(
        people["population"],
        people["habitability"],
        people["settlements"],
        SEED,
        params=character["microdata"],
    )
    identities = build_initial_identity_map(micro, SEED)
    dwellings = build_dwellings(micro, SEED, identities)
    businesses = build_businesses(micro, SEED, identities)
    hospitals = build_hospitals(micro, SEED, identities, businesses)
    history = build_event_history(
        micro,
        SEED,
        identities,
        dwellings,
        businesses,
        hospitals,
        months=18,
    )
    admin = build_admin(
        world["land"],
        people["settlements"],
        resource_outposts(world, SEED),
        n_states=6,
    )
    package = build_observed_sources(history, SEED, admin, hospitals)
    return {
        "world": world,
        "people": people,
        "micro": micro,
        "identities": identities,
        "dwellings": dwellings,
        "businesses": businesses,
        "hospitals": hospitals,
        "history": history,
        "admin": admin,
        "package": package,
    }


def _digest(value, digest: hashlib._Hash | None = None) -> str:
    if digest is None:
        digest = hashlib.sha256()
    if isinstance(value, dict):
        for name in sorted(value):
            digest.update(name.encode())
            _digest(value[name], digest)
    elif isinstance(value, np.ndarray):
        digest.update(value.dtype.str.encode())
        digest.update(str(value.shape).encode())
        digest.update(np.ascontiguousarray(value).tobytes())
    else:
        digest.update(repr(value).encode())
    return digest.hexdigest()


def _bit(crosswalk: dict, name: str) -> np.ndarray:
    return (crosswalk["mechanism_code"] & MECHANISM_BITS[name]) != 0


def test_participant_snapshots_are_four_flat_observed_tables_only():
    package = _setup()["package"]
    public = participant_source_snapshots(package)

    assert set(public) == {"preliminary", "revised"}
    for label, snapshot in public.items():
        assert set(snapshot) == {"snapshot_tick", *OBSERVED_SOURCES}
        for source in OBSERVED_SOURCES:
            table = snapshot[source]
            assert set(table) == set(PUBLIC_SCHEMAS[source])
            assert len({len(values) for values in table.values()}) == 1
            for name, values in table.items():
                assert values.ndim == 1
                assert values.dtype == PUBLIC_SCHEMAS[source][name]
                lowered = name.lower()
                assert not lowered.startswith("truth_")
                assert all(
                    forbidden not in lowered
                    for forbidden in ("mechanism", "seed", "regime", "crosswalk")
                )
    assert "hidden" not in public


def test_observed_identifiers_are_disjoint_from_truth_and_stable_between_snapshots():
    package = _setup()["package"]
    entity_column = {
        "population": "person_id",
        "business": "business_id",
        "income": "taxpayer_id",
        "health": "encounter_id",
    }
    for source in OBSERVED_SOURCES:
        preliminary = package["public_snapshots"]["preliminary"][source]
        revised = package["public_snapshots"]["revised"][source]
        crosswalk = package["hidden"]["crosswalks"]["revised"][source]
        assert ((revised["record_id"] >> np.uint64(56)) >= 0x80).all()
        assert ((revised[entity_column[source]] >> np.uint64(56)) >= 0x80).all()
        assert not np.isin(
            revised[entity_column[source]], crosswalk["truth_entity_id"]
        ).any()
        common, pre_position, rev_position = np.intersect1d(
            preliminary["record_id"],
            revised["record_id"],
            assume_unique=True,
            return_indices=True,
        )
        assert len(common) > 0
        assert np.array_equal(
            preliminary[entity_column[source]][pre_position],
            revised[entity_column[source]][rev_position],
        )


def test_snapshots_are_recorded_tick_cuts_with_real_late_reporting():
    setup = _setup()
    package = setup["package"]
    history = setup["history"]
    preliminary_tick = int(package["preliminary_tick"])
    revised_tick = int(package["revised_tick"])
    event = history["event"]
    newly_visible = (event["recorded_tick"] > preliminary_tick) & (
        event["recorded_tick"] <= revised_tick
    )
    source_event = {
        "population": (EVENT_TYPES["person_birth"], "truth_person_id"),
        "business": (EVENT_TYPES["establishment_opened"], "truth_establishment_id"),
        "income": (EVENT_TYPES["person_birth"], "truth_person_id"),
        "health": (EVENT_TYPES["encounter_admitted"], "truth_encounter_id"),
    }
    for source, (event_type, field) in source_event.items():
        pre_truth = set(
            package["hidden"]["crosswalks"]["preliminary"][source][
                "truth_entity_id"
            ].tolist()
        )
        revised_truth = set(
            package["hidden"]["crosswalks"]["revised"][source][
                "truth_entity_id"
            ].tolist()
        )
        candidates = set(
            event[field][newly_visible & (event["event_type"] == event_type)].tolist()
        )
        assert (revised_truth - pre_truth) & candidates

    effective_but_late = (event["tick"] <= preliminary_tick) & (
        event["recorded_tick"] > preliminary_tick
    )
    assert effective_but_late.any()
    for source in OBSERVED_SOURCES:
        assert _bit(
            package["hidden"]["crosswalks"]["preliminary"][source], "stale"
        ).any()


def test_one_month_history_uses_snapshot_and_terminal_as_two_valid_cuts():
    setup = _setup()
    one_month = build_event_history(
        setup["micro"],
        SEED,
        setup["identities"],
        setup["dwellings"],
        setup["businesses"],
        setup["hospitals"],
        months=1,
    )
    package = build_observed_sources(
        one_month, SEED, setup["admin"], setup["hospitals"]
    )
    assert int(package["preliminary_tick"]) == int(one_month["snapshot_tick"])
    assert int(package["revised_tick"]) == int(one_month["terminal_tick"])


def test_all_planted_mechanisms_are_exercised_and_outposts_have_lower_coverage():
    setup = _setup()
    package = setup["package"]
    for source in OBSERVED_SOURCES:
        mechanism = package["hidden"]["mechanisms"][source]
        assert (~mechanism["covered"]).any()
        assert mechanism["duplicate"].any()
        assert mechanism["split"].any()
        assert (mechanism["merge_group"] >= 0).any()
        assert mechanism["county_error"].any()
        assert mechanism["linkage_error"].any()
        assert mechanism["item_missing"].any()

    terminal_person = setup["history"]["terminal_state"]["person"]
    county = setup["admin"]["county"].reshape(-1)[terminal_person["cell"]]
    outpost = setup["admin"]["county_is_outpost"][county]
    covered = package["hidden"]["mechanisms"]["population"]["covered"]
    assert outpost.any() and (~outpost).any()
    assert covered[outpost].mean() + 0.05 < covered[~outpost].mean()


def test_split_merge_duplicate_and_missingness_evidence_matches_public_rows():
    package = _setup()["package"]
    for source in OBSERVED_SOURCES:
        crosswalk = package["hidden"]["crosswalks"]["revised"][source]
        duplicate_truth = crosswalk["truth_entity_id"][_bit(crosswalk, "duplicate")]
        _, duplicate_count = np.unique(duplicate_truth, return_counts=True)
        assert (duplicate_count >= 2).any()

    population_crosswalk = package["hidden"]["crosswalks"]["revised"]["population"]
    split = _bit(population_crosswalk, "split")
    split_truth = population_crosswalk["truth_entity_id"][split]
    split_entity = population_crosswalk["observed_entity_id"][split]
    assert any(
        len(np.unique(split_entity[split_truth == truth_id])) > 1
        for truth_id in np.unique(split_truth)
    )
    merged = _bit(population_crosswalk, "merged")
    merged_truth = population_crosswalk["truth_entity_id"][merged]
    merged_entity = population_crosswalk["observed_entity_id"][merged]
    assert any(
        len(np.unique(merged_truth[merged_entity == observed_id])) > 1
        for observed_id in np.unique(merged_entity)
    )

    revised = package["public_snapshots"]["revised"]
    crosswalks = package["hidden"]["crosswalks"]["revised"]
    assert np.array_equal(
        revised["population"]["education"] == -1,
        _bit(crosswalks["population"], "item_missing"),
    )
    assert np.array_equal(
        np.isnan(revised["business"]["annual_payroll_cents"]),
        _bit(crosswalks["business"], "item_missing"),
    )
    assert np.array_equal(
        np.isnan(revised["income"]["employment_income_cents"]),
        _bit(crosswalks["income"], "item_missing"),
    )
    assert np.array_equal(
        np.isnan(revised["health"]["cost_cents"]),
        _bit(crosswalks["health"], "item_missing"),
    )


def test_uncorrupted_counties_are_exact_admin_lookups_from_cells():
    setup = _setup()
    package = setup["package"]
    revised = package["public_snapshots"]["revised"]
    crosswalks = package["hidden"]["crosswalks"]["revised"]
    terminal = setup["history"]["terminal_state"]
    county_flat = setup["admin"]["county"].reshape(-1)

    for source, truth_table, county_column in (
        ("population", "person", "county"),
        ("income", "person", "county"),
        ("business", "establishment", "county"),
    ):
        crosswalk = crosswalks[source]
        # The income source records the address one year back; rows where that
        # differs from the snapshot address carry the address_lag bit.
        clean = ~(
            _bit(crosswalk, "stale")
            | _bit(crosswalk, "county_error")
            | _bit(crosswalk, "address_lag")
        )
        position = (crosswalk["truth_entity_id"] & np.uint64((1 << 56) - 1)).astype(
            np.int64
        ) - 1
        expected = county_flat[terminal[truth_table]["cell"][position]]
        assert np.array_equal(revised[source][county_column][clean], expected[clean])
        if source == "income":
            assert _bit(crosswalk, "address_lag").any()

    health_crosswalk = crosswalks["health"]
    health_clean = ~(
        _bit(health_crosswalk, "stale")
        | _bit(health_crosswalk, "county_error")
        | _bit(health_crosswalk, "address_lag")
    )
    encounter_position = (
        health_crosswalk["truth_entity_id"] & np.uint64((1 << 56) - 1)
    ).astype(np.int64) - 1
    encounter = terminal["encounter"]
    person_position = (
        encounter["truth_person_id"][encounter_position] & np.uint64((1 << 56) - 1)
    ).astype(np.int64) - 1
    hospital_position = (
        encounter["truth_hospital_id"][encounter_position] & np.uint64((1 << 56) - 1)
    ).astype(np.int64) - 1
    expected_patient = county_flat[terminal["person"]["cell"][person_position]]
    expected_facility = county_flat[
        setup["hospitals"]["hospital"]["cell"][hospital_position]
    ]
    assert np.array_equal(
        revised["health"]["patient_county"][health_clean],
        expected_patient[health_clean],
    )
    assert np.array_equal(revised["health"]["facility_county"], expected_facility)


def test_population_coverage_dial_changes_only_the_population_source():
    setup = _setup()
    baseline = setup["package"]
    lower = build_observed_sources(
        setup["history"],
        SEED,
        setup["admin"],
        setup["hospitals"],
        params=SourceParams(population_coverage=0.800),
    )
    assert len(lower["public_snapshots"]["revised"]["population"]["record_id"]) < len(
        baseline["public_snapshots"]["revised"]["population"]["record_id"]
    )
    for source in ("business", "income", "health"):
        for name in baseline["public_snapshots"]["revised"][source]:
            assert np.array_equal(
                lower["public_snapshots"]["revised"][source][name],
                baseline["public_snapshots"]["revised"][source][name],
                equal_nan=True,
            )


def test_source_build_is_byte_deterministic_and_validates_end_to_end():
    setup = _setup()
    second = build_observed_sources(
        setup["history"], SEED, setup["admin"], setup["hospitals"]
    )
    assert _digest(setup["package"]) == _digest(second)
    validate_observed_sources(
        second,
        setup["history"],
        SEED,
        setup["admin"],
        setup["hospitals"],
    )


def test_public_and_hidden_tampering_are_rejected():
    setup = _setup()
    package = setup["package"]
    changed_public = {
        **package,
        "public_snapshots": {
            **package["public_snapshots"],
            "revised": {**package["public_snapshots"]["revised"]},
        },
    }
    changed_public["public_snapshots"]["revised"]["population"] = {
        **package["public_snapshots"]["revised"]["population"]
    }
    changed_county = package["public_snapshots"]["revised"]["population"][
        "county"
    ].copy()
    changed_county[0] = (changed_county[0] + 1) % setup["admin"]["n_counties"]
    changed_public["public_snapshots"]["revised"]["population"][
        "county"
    ] = changed_county
    with pytest.raises(ValueError, match="deterministic regeneration"):
        validate_observed_sources(
            changed_public,
            setup["history"],
            SEED,
            setup["admin"],
            setup["hospitals"],
        )

    changed_hidden = {
        **package,
        "hidden": {
            **package["hidden"],
            "crosswalks": {
                **package["hidden"]["crosswalks"],
                "revised": {**package["hidden"]["crosswalks"]["revised"]},
            },
        },
    }
    changed_hidden["hidden"]["crosswalks"]["revised"]["income"] = {
        **package["hidden"]["crosswalks"]["revised"]["income"]
    }
    changed_truth = package["hidden"]["crosswalks"]["revised"]["income"][
        "truth_entity_id"
    ].copy()
    changed_truth[0] = setup["history"]["terminal_state"]["person"]["truth_person_id"][
        1
    ]
    changed_hidden["hidden"]["crosswalks"]["revised"]["income"][
        "truth_entity_id"
    ] = changed_truth
    with pytest.raises(ValueError, match="deterministic regeneration"):
        validate_observed_sources(
            changed_hidden,
            setup["history"],
            SEED,
            setup["admin"],
            setup["hospitals"],
        )


def test_source_builder_rejects_another_truth_world():
    setup = _setup()
    with pytest.raises(ValueError, match="truth world"):
        build_observed_sources(
            setup["history"],
            SEED + 1,
            setup["admin"],
            setup["hospitals"],
        )


def _person_position(truth_ids: np.ndarray) -> np.ndarray:
    return (truth_ids & np.uint64((1 << 56) - 1)).astype(np.int64) - 1


def _linked_pairs(package: dict, history: dict, left: str, right: str):
    """Rows of two person sources that belong to the same truth person, by the sealed
    crosswalk (health rows map through the encounter's patient)."""
    import pandas as pd

    frames = {}
    encounter = history["terminal_state"]["encounter"]
    for source in (left, right):
        table = package["public_snapshots"]["revised"][source]
        crosswalk = package["hidden"]["crosswalks"]["revised"][source]
        frame = pd.DataFrame({name: values for name, values in table.items()})
        truth = crosswalk["truth_entity_id"]
        if source == "health":
            truth = encounter["truth_person_id"][_person_position(truth)]
        frame["truth_person"] = truth
        frames[source] = frame.drop_duplicates("truth_person")
    return frames[left].merge(frames[right], on="truth_person", suffixes=("_l", "_r"))


def test_no_exact_cross_source_person_key_is_shipped():
    setup = _setup()
    package = setup["package"]
    pairs = _linked_pairs(package, setup["history"], "population", "income")
    assert len(pairs) > 1000
    same_name = (pairs["given_code_l"] == pairs["given_code_r"]) & (
        pairs["family_code_l"] == pairs["family_code_r"]
    )
    same_birth = pairs["birth_tick_l"] == pairs["birth_tick_r"]
    same_sex = pairs["sex_l"] == pairs["sex_r"]
    # Each field disagrees for a material share of true links; none is exact.
    assert 0.05 < 1.0 - same_name.mean() < 0.50
    assert 0.03 < 1.0 - same_birth.mean() < 0.30
    assert 0.002 < 1.0 - same_sex.mean() < 0.05
    assert (same_name & same_birth & same_sex).mean() < 0.85
    # Movers legitimately carry different counties across archives.
    crosswalk = package["hidden"]["crosswalks"]["revised"]["income"]
    assert _bit(crosswalk, "address_lag").mean() > 0.005
    for source in ("population", "income", "health"):
        crosswalk = package["hidden"]["crosswalks"]["revised"][source]
        assert _bit(crosswalk, "name_error").any()
        assert _bit(crosswalk, "birth_error").any()
        table = package["public_snapshots"]["revised"][source]
        assert "name_code" not in table
        assert (table["given_code"] == 0).any()          # missing given names exist
    health_pairs = _linked_pairs(package, setup["history"], "population", "health")
    assert len(health_pairs) > 100
    assert (health_pairs["county"] != health_pairs["patient_county"]).any()


def test_true_name_collisions_near_duplicates_and_merged_names():
    import pandas as pd

    setup = _setup()
    package = setup["package"]
    table = package["public_snapshots"]["revised"]["population"]
    crosswalk = package["hidden"]["crosswalks"]["revised"]["population"]
    frame = pd.DataFrame({name: values for name, values in table.items()})
    frame["truth"] = crosswalk["truth_entity_id"]
    frame["code"] = crosswalk["mechanism_code"]
    # Distinct truth persons share a full name pair.
    persons_per_pair = frame.drop_duplicates("truth").groupby(["given_code", "family_code"])["truth"].nunique()
    assert (persons_per_pair > 1).sum() > 100
    # Duplicate records of one person are near-duplicates: some differ in a reported field.
    duplicates = frame[(frame["code"] & MECHANISM_BITS["duplicate"]) != 0]
    spread = duplicates.groupby("truth").agg(
        given=("given_code", "nunique"), family=("family_code", "nunique"),
        birth=("birth_tick", "nunique"), n=("record_id", "size"))
    spread = spread[spread["n"] >= 2]
    assert len(spread) > 50
    assert ((spread["given"] > 1) | (spread["family"] > 1) | (spread["birth"] > 1)).mean() > 0.10
    assert ((spread["given"] == 1) & (spread["family"] == 1) & (spread["birth"] == 1)).mean() > 0.30
    # The two persons of a merge pair report one name pair.
    merged = frame[(frame["code"] & MECHANISM_BITS["merged"]) != 0]
    by_id = merged.groupby("person_id").agg(
        persons=("truth", "nunique"), pairs=("family_code", lambda v: len(set(zip(merged.loc[v.index, "given_code"], v)))))
    both = by_id[by_id["persons"] == 2]
    assert len(both) > 5 and (both["pairs"] == 1).mean() > 0.5


def test_source_regime_draw_places_the_hidden_world_outside_the_development_band():
    from meridia.sources import DEVELOPMENT_BAND, draw_source_params

    for seed in (1, 7, 20260915, 20260916):
        for payroll in (0.75, 1.0, 1.30):
            development = draw_source_params(seed, "development", payroll)
            hidden = draw_source_params(seed, "hidden", payroll)
            assert development == draw_source_params(seed, "development", payroll)
            assert hidden == draw_source_params(seed, "hidden", payroll)
            for name, (lo, hi) in DEVELOPMENT_BAND.items():
                assert lo <= getattr(development, name) <= hi
            assert hidden.population_coverage < DEVELOPMENT_BAND["population_coverage"][0] - 0.02 + 1e-12
            assert hidden.health_coverage < DEVELOPMENT_BAND["health_coverage"][0] - 0.06 + 1e-12
            assert hidden.county_error_rate > DEVELOPMENT_BAND["county_error_rate"][1] * 1.5 - 1e-12
            level = payroll * hidden.register_income_scale
            assert level < 0.705 - 0.07 or level > 1.378 + 0.07
            # The fixed reporting-error rates do not move with the regime.
            assert hidden.name_family_variant_rate == development.name_family_variant_rate
            assert hidden.birth_month_slip_rate == development.birth_month_slip_rate
    with pytest.raises(ValueError, match="regime"):
        draw_source_params(1, "other")


def test_hidden_regime_changes_only_the_retained_rates_and_shifts_the_sources():
    from meridia.sources import draw_source_params

    setup = _setup()
    baseline = setup["package"]
    hidden_params = draw_source_params(SEED, "hidden", 1.0)
    hidden = build_observed_sources(
        setup["history"], SEED, setup["admin"], setup["hospitals"], params=hidden_params
    )
    assert hidden["source_params"] != baseline["source_params"]
    for label in ("preliminary", "revised"):
        for source in OBSERVED_SOURCES:
            assert set(hidden["public_snapshots"][label][source]) == set(PUBLIC_SCHEMAS[source])
            for name in hidden["public_snapshots"][label][source]:
                assert "rate" not in name and "coverage" not in name and "scale" not in name
    revised_h = hidden["public_snapshots"]["revised"]
    revised_b = baseline["public_snapshots"]["revised"]
    assert len(revised_h["health"]["record_id"]) < len(revised_b["health"]["record_id"])
    assert len(revised_h["population"]["record_id"]) < len(revised_b["population"]["record_id"])
    assert hidden["hidden"]["mechanisms"]["population"]["county_error"].mean() > \
        1.4 * baseline["hidden"]["mechanisms"]["population"]["county_error"].mean()
    ratio = np.nanmean(revised_h["income"]["employment_income_cents"]) / \
        np.nanmean(revised_b["income"]["employment_income_cents"])
    assert abs(np.log(ratio) - np.log(hidden_params.register_income_scale)) < 0.05
    validate_observed_sources(hidden, setup["history"], SEED, setup["admin"], setup["hospitals"])
