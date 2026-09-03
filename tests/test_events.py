"""Persistent monthly histories, reporting lag, and exact replay."""

import hashlib
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from meridia.businesses import build_businesses
from meridia.character import draw_world_character
from meridia.dwellings import build_dwellings
from meridia.events import CAUSE_CODES, EVENT_TYPES, EventHistoryParams
from meridia.events import build_event_history, continuation_events
from meridia.events import continuation_shocks, events_visible_at
from meridia.events import replay_event_history, validate_event_history
from meridia.hospitals import HospitalParams, build_hospitals
from meridia.hydrology import fill_depressions, flow_accumulation, flow_directions
from meridia.admin import build_admin
from meridia.demography import SHOCK_LOADING_BAND, regional_multiplier
from meridia.identities import ENTITY_NAMESPACE, SEQUENCE_MASK, build_initial_identity_map
from meridia.identities import entity_namespace
from meridia.mechanisms import build_world_mechanisms
from meridia.microdata import build_microdata
from meridia.population import build_population, resource_outposts
from meridia.terrain import generate_elevation

SEED = 20260831
H, W = 60, 72
TOTAL = 40_000


@lru_cache(maxsize=2)
def _start(seed: int = SEED):
    return _start_full(seed)[2:]


@lru_cache(maxsize=2)
def _start_full(seed: int = SEED):
    character = draw_world_character(seed)
    world = generate_elevation(seed, H, W)
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
        seed=seed,
    )
    micro = build_microdata(
        people["population"],
        people["habitability"],
        people["settlements"],
        seed,
        params=character["microdata"],
    )
    identities = build_initial_identity_map(micro, seed)
    dwellings = build_dwellings(micro, seed, identities)
    businesses = build_businesses(micro, seed, identities)
    hospitals = build_hospitals(micro, seed, identities, businesses)
    return world, people, micro, identities, dwellings, businesses, hospitals


@lru_cache(maxsize=8)
def _history(months: int = 18):
    micro, identities, dwellings, businesses, hospitals = _start()
    return build_event_history(
        micro,
        SEED,
        identities,
        dwellings,
        businesses,
        hospitals,
        months=months,
    )


def _history_digest(history: dict) -> str:
    digest = hashlib.sha256()
    for table_name in sorted(history["initial_state"]):
        for name, values in history["initial_state"][table_name].items():
            digest.update(table_name.encode())
            digest.update(name.encode())
            digest.update(np.ascontiguousarray(values).tobytes())
    for name, values in history["event"].items():
        digest.update(name.encode())
        digest.update(np.ascontiguousarray(values).tobytes())
    for table_name in sorted(history["terminal_state"]):
        for name, values in history["terminal_state"][table_name].items():
            digest.update(table_name.encode())
            digest.update(name.encode())
            digest.update(np.ascontiguousarray(values).tobytes())
    return digest.hexdigest()


def test_history_contains_every_required_change_family_with_persistent_ids():
    history = _history()
    event = history["event"]

    for event_type in EVENT_TYPES.values():
        assert (event["event_type"] == event_type).any()
    assert (
        entity_namespace(event["truth_event_id"]) == ENTITY_NAMESPACE["event"]
    ).all()
    assert np.array_equal(
        event["truth_event_id"],
        np.sort(event["truth_event_id"], kind="stable"),
    )
    assert (event["recorded_tick"] >= event["tick"]).all()
    assert (event["recorded_tick"] > event["tick"]).any()
    assert not any(name.startswith("observed_") for name in event)


def test_replay_conserves_population_housing_jobs_and_beds_exactly():
    micro, identities, dwellings, businesses, hospitals = _start()
    history = _history()
    replayed = replay_event_history(history)

    for table_name in replayed:
        for name in replayed[table_name]:
            assert np.array_equal(
                replayed[table_name][name],
                history["terminal_state"][table_name][name],
            )
    births = int((history["event"]["event_type"] == EVENT_TYPES["person_birth"]).sum())
    deaths = int((history["event"]["event_type"] == EVENT_TYPES["person_death"]).sum())
    assert int(replayed["person"]["is_alive"].sum()) == TOTAL + births - deaths
    assert int(replayed["dwelling"]["resident_count"].sum()) == int(
        replayed["person"]["is_alive"].sum()
    )
    active_jobs = replayed["job"]["is_active"]
    assert len(np.unique(replayed["job"]["truth_person_id"][active_jobs])) == int(
        active_jobs.sum()
    )
    open_encounters = replayed["encounter"]["is_open"]
    assert len(
        np.unique(replayed["encounter"]["truth_person_id"][open_encounters])
    ) == int(open_encounters.sum())
    validate_event_history(
        history,
        micro,
        identities,
        dwellings,
        businesses,
        hospitals,
        SEED,
    )


def test_longer_generation_only_appends_and_vintages_hide_late_reports():
    six_months = _history(6)
    twelve_months = _history(12)
    prefix_length = six_months["n_events"]

    for name in six_months["event"]:
        assert np.array_equal(
            six_months["event"][name],
            twelve_months["event"][name][:prefix_length],
        )
    six_from_long_history = replay_event_history(twelve_months, through_tick=6)
    for table_name in six_from_long_history:
        for name in six_from_long_history[table_name]:
            assert np.array_equal(
                six_from_long_history[table_name][name],
                six_months["terminal_state"][table_name][name],
            )
    preliminary = events_visible_at(twelve_months, 6)
    effective_by_six = int((twelve_months["event"]["tick"] <= 6).sum())
    assert len(preliminary["truth_event_id"]) < effective_by_six
    assert (preliminary["recorded_tick"] <= 6).all()


def _late_dependency_history():
    """A one-month ledger that actually contains a job end recorded after its death.

    The pair the test needs is a coincidence of a churn closure and a death in the same
    month for the same person, reported in the wrong order. The fixture raises the
    closure rate until the coincidence happens rather than resting on one draw: any
    change to a coefficient upstream moves which draw produces it, and a test that
    asserts a coincidence at one draw is testing the draw.
    """
    micro, identities, dwellings, businesses, hospitals = _start()
    for churn in (0.05, 0.09, 0.14, 0.20, 0.28):
        history = build_event_history(
            micro,
            SEED,
            identities,
            dwellings,
            businesses,
            hospitals,
            months=1,
            params=EventHistoryParams(
                monthly_establishment_churn_rate=churn,
                late_report_probability=1.0,
                max_report_delay_months=3,
            ),
            shocks=[
                {
                    "year": 0,
                    "kind": "compound_intervention",
                    "mortality_multiplier": 20.0,
                }
            ],
        )
        event = history["event"]
        ends = np.flatnonzero(
            (event["event_type"] == EVENT_TYPES["job_ended"])
            & (event["cause_code"] == CAUSE_CODES["business_churn"])
        )
        death_of = {int(event["truth_person_id"][position]): int(position)
                    for position in np.flatnonzero(
                        event["event_type"] == EVENT_TYPES["person_death"])}
        if any(int(event["truth_person_id"][position]) in death_of
               and event["recorded_tick"][position]
               > event["recorded_tick"][death_of[int(event["truth_person_id"][position])]]
               for position in ends):
            return history
    raise AssertionError("no churn closure met a death in the same month at any rate")


def test_recording_lag_never_reorders_same_tick_dependencies():
    history = _late_dependency_history()
    event = history["event"]
    closure_job_ends = np.flatnonzero(
        (event["event_type"] == EVENT_TYPES["job_ended"])
        & (event["cause_code"] == CAUSE_CODES["business_churn"])
    )
    death_by_person = {
        int(event["truth_person_id"][position]): int(position)
        for position in np.flatnonzero(
            event["event_type"] == EVENT_TYPES["person_death"]
        )
    }
    late_dependency_pairs = [
        (
            int(end_position),
            death_by_person[int(event["truth_person_id"][end_position])],
        )
        for end_position in closure_job_ends
        if int(event["truth_person_id"][end_position]) in death_by_person
        and event["recorded_tick"][end_position]
        > event["recorded_tick"][
            death_by_person[int(event["truth_person_id"][end_position])]
        ]
    ]

    assert late_dependency_pairs
    end_position, death_position = late_dependency_pairs[0]
    establishment_id = event["truth_establishment_id"][end_position]
    closure_position = np.flatnonzero(
        (event["event_type"] == EVENT_TYPES["establishment_closed"])
        & (event["truth_establishment_id"] == establishment_id)
    )
    assert len(closure_position) == 1
    closure_position = int(closure_position[0])
    assert end_position < closure_position < death_position
    assert event["tick"][end_position] == event["tick"][death_position]
    assert event["recorded_tick"][end_position] > event["recorded_tick"][death_position]

    replayed = replay_event_history(history)
    for table_name in replayed:
        for name in replayed[table_name]:
            assert np.array_equal(
                replayed[table_name][name],
                history["terminal_state"][table_name][name],
            )

    visible_at_death = events_visible_at(
        history, int(event["recorded_tick"][death_position])
    )
    visible_ids = set(map(int, visible_at_death["truth_event_id"]))
    assert int(event["truth_event_id"][death_position]) in visible_ids
    assert int(event["truth_event_id"][end_position]) not in visible_ids


def test_continuations_share_the_prefix_and_diverge_after_the_branch():
    """A committed continuation is the same ledger to the branch month and its own
    substream after it, so the ensemble's members agree on the past and disagree on the
    future. The member key is never arithmetic on the root seed."""
    from meridia.events import CONTINUATION_DOMAIN, LEDGER_DOMAIN

    micro, identities, dwellings, businesses, hospitals = _start()
    branch = 6
    base = _history(12)
    members = [
        build_event_history(micro, SEED, identities, dwellings, businesses, hospitals,
                            months=12, continuation_member=m, branch_month=branch)
        for m in range(3)
    ]
    assert CONTINUATION_DOMAIN != LEDGER_DOMAIN
    for member in members:
        early = member["event"]["tick"] <= member["snapshot_tick"] + branch
        base_early = base["event"]["tick"] <= base["snapshot_tick"] + branch
        for column in ("truth_event_id", "tick", "event_type", "truth_person_id"):
            assert np.array_equal(member["event"][column][early],
                                  base["event"][column][base_early]), column
    # Measured on this world: base 18,117 events against member counts 17,905, 18,247
    # and 18,244, with different survivor sets, so the futures are genuinely independent.
    assert len({int(member["n_events"]) for member in members}) == len(members)
    survivors = {int(member["terminal_state"]["person"]["is_alive"].sum()) for member in members}
    assert len(survivors) > 1
    repeat = build_event_history(micro, SEED, identities, dwellings, businesses, hospitals,
                                 months=12, continuation_member=1, branch_month=branch)
    assert np.array_equal(repeat["event"]["event_type"], members[1]["event"]["event_type"])
    with pytest.raises(ValueError, match="continuation"):
        build_event_history(micro, SEED, identities, dwellings, businesses, hospitals,
                            months=12, continuation_member=0)


def test_a_resumed_continuation_equals_the_one_that_replayed_its_prefix():
    """The branch capture is what makes an ensemble affordable, so it has to be exact.

    A member resumed from the state the ledger kept at the branch month must agree, event
    for event and identifier for identifier, with the member that re-ran every month
    before the branch. Without that equality the ensemble would be a different object
    from the one the substream rule defines.
    """
    micro, identities, dwellings, businesses, hospitals = _start()
    branch = 6
    captured = build_event_history(micro, SEED, identities, dwellings, businesses,
                                   hospitals, months=12, capture_month=branch)
    assert int(captured["branch"]["tick"]) == int(captured["snapshot_tick"]) + branch
    for member in (0, 1, 7):
        full = build_event_history(micro, SEED, identities, dwellings, businesses,
                                   hospitals, months=12, continuation_member=member,
                                   branch_month=branch)
        after = full["event"]["tick"] > captured["branch"]["tick"]
        suffix = {name: values[after] for name, values in full["event"].items()}
        resumed = continuation_events(captured["branch"], member, 12 - branch)
        assert sorted(resumed) == sorted(suffix)
        for column in suffix:
            assert np.array_equal(suffix[column], resumed[column]), (member, column)
    assert not np.array_equal(
        continuation_events(captured["branch"], 1, 6)["event_type"],
        continuation_events(captured["branch"], 2, 6)["event_type"])
    with pytest.raises(ValueError, match="capture month"):
        build_event_history(micro, SEED, identities, dwellings, businesses, hospitals,
                            months=12, capture_month=0)


def test_a_continuation_draws_its_own_future_shock_years():
    """Systematic risk is what gives the sealed tail a width worth predicting.

    A member keeps the world's realized shock years up to the branch and draws every year
    after it from the published family at the published rate. Members that share one
    frozen future schedule differ only by demographic noise, which on a population of any
    size is a fraction of a percent, far under the error any reconstruction of the same
    regions carries.
    """
    micro, identities, dwellings, businesses, hospitals = _start()
    branch = 24
    captured = build_event_history(micro, SEED, identities, dwellings, businesses,
                                   hospitals, months=48, capture_month=branch,
                                   shocks=[{"year": 0, "kind": "baby_bust",
                                            "fertility_multiplier": 0.5},
                                           {"year": 3, "kind": "mortality_spike",
                                            "mortality_multiplier": 2.0,
                                            "admission_multiplier": 2.0}])
    schedules = [continuation_shocks(captured["branch"], m, 24) for m in range(64)]
    for schedule in schedules:
        past = [s for s in schedule if s["year"] < 2]
        assert past == [{"year": 0, "kind": "baby_bust", "fertility_multiplier": 0.5}]
        assert all(s["year"] >= 2 for s in schedule if s not in past)
    futures = {tuple(sorted((s["year"], s["kind"]) for s in schedule if s["year"] >= 2))
               for schedule in schedules}
    assert len(futures) > 1, "every member drew the same future"
    drawn = sum(len([s for s in schedule if s["year"] >= 2]) for schedule in schedules)
    years = sum(len({s["year"] for s in schedule if s["year"] >= 2}) or 0
                for schedule in schedules)
    assert years == drawn                      # at most one shock a year
    assert 0.05 < drawn / (len(schedules) * 3) < 0.45   # around the published rate
    assert continuation_shocks(captured["branch"], 5, 24) == \
        continuation_shocks(captured["branch"], 5, 24)


def test_history_is_byte_deterministic():
    micro, identities, dwellings, businesses, hospitals = _start()
    first = build_event_history(
        micro,
        SEED,
        identities,
        dwellings,
        businesses,
        hospitals,
        months=6,
    )
    second = build_event_history(
        micro,
        SEED,
        identities,
        dwellings,
        businesses,
        hospitals,
        months=6,
    )
    assert _history_digest(first) == _history_digest(second)


def test_capacity_filtered_admissions_do_not_gap_encounter_identities():
    micro, identities, dwellings, businesses, _ = _start()
    scarce_hospitals = build_hospitals(
        micro,
        SEED,
        identities,
        businesses,
        HospitalParams(beds_per_1000=0.10),
    )
    history = build_event_history(
        micro,
        SEED,
        identities,
        dwellings,
        businesses,
        scarce_hospitals,
        months=1,
    )
    encounter_ids = history["terminal_state"]["encounter"]["truth_encounter_id"]
    sequence = encounter_ids & np.uint64((1 << 56) - 1)

    assert np.array_equal(sequence, np.arange(1, len(sequence) + 1))


def test_terminal_state_tamper_is_rejected():
    micro, identities, dwellings, businesses, hospitals = _start()
    history = _history(6)
    changed = {
        **history,
        "terminal_state": {
            **history["terminal_state"],
            "dwelling": {**history["terminal_state"]["dwelling"]},
        },
    }
    changed["terminal_state"]["dwelling"]["resident_count"] = history["terminal_state"][
        "dwelling"
    ]["resident_count"].copy()
    changed["terminal_state"]["dwelling"]["resident_count"][0] += 1

    with pytest.raises(ValueError, match="terminal event state differs from replay"):
        validate_event_history(
            changed,
            micro,
            identities,
            dwellings,
            businesses,
            hospitals,
            SEED,
        )


def test_move_history_tamper_is_rejected():
    micro, identities, dwellings, businesses, hospitals = _start()
    history = _history(6)
    changed = {**history, "event": {**history["event"]}}
    changed["event"]["truth_prior_dwelling_id"] = history["event"][
        "truth_prior_dwelling_id"
    ].copy()
    move_position = np.flatnonzero(
        changed["event"]["event_type"] == EVENT_TYPES["household_moved"]
    )[0]
    changed["event"]["truth_prior_dwelling_id"][move_position] = changed["event"][
        "truth_dwelling_id"
    ][move_position]

    with pytest.raises(ValueError, match="wrong prior dwelling"):
        validate_event_history(
            changed,
            micro,
            identities,
            dwellings,
            businesses,
            hospitals,
            SEED,
        )


def test_shock_schedule_changes_demographic_event_counts():
    micro, identities, dwellings, businesses, hospitals = _start()
    baseline = build_event_history(
        micro,
        SEED,
        identities,
        dwellings,
        businesses,
        hospitals,
        months=12,
        shocks=[],
    )
    shocked = build_event_history(
        micro,
        SEED,
        identities,
        dwellings,
        businesses,
        hospitals,
        months=12,
        shocks=[
            {
                "year": 0,
                "kind": "compound_intervention",
                "mortality_multiplier": 4.0,
                "fertility_multiplier": 0.25,
                "leave_home_multiplier": 3.0,
            }
        ],
    )

    baseline_deaths = int(
        (baseline["event"]["event_type"] == EVENT_TYPES["person_death"]).sum()
    )
    shocked_deaths = int(
        (shocked["event"]["event_type"] == EVENT_TYPES["person_death"]).sum()
    )
    baseline_births = int(
        (baseline["event"]["event_type"] == EVENT_TYPES["person_birth"]).sum()
    )
    shocked_births = int(
        (shocked["event"]["event_type"] == EVENT_TYPES["person_birth"]).sum()
    )
    assert shocked_deaths > baseline_deaths * 2
    assert shocked_births < baseline_births * 0.5


def test_event_builder_rejects_a_seed_from_another_truth_world():
    micro, identities, dwellings, businesses, hospitals = _start()
    with pytest.raises(ValueError, match="seed does not match"):
        build_event_history(
            micro,
            SEED + 1,
            identities,
            dwellings,
            businesses,
            hospitals,
            months=2,
        )


def test_a_shock_year_lands_in_proportion_to_the_region_loading():
    """A shock is one national event; the loadings decide how hard it lands where.

    Without them every region takes the whole multiplier, the regional liabilities move
    as one, and the aggregate tail is what the six marginals already say. With them the
    correlation structure is a thing a method has to estimate. The loadings are a per
    world draw from a published band and are held for every year.
    """
    world, people, micro, identities, dwellings, businesses, hospitals = _start_full()
    admin = build_admin(world["land"], people["settlements"],
                        resource_outposts(world, SEED), n_states=4)
    mechanisms = build_world_mechanisms(SEED, "development", admin, micro, businesses)
    loading = mechanisms.region_shock_loading
    assert len(loading) == 4
    assert (loading >= SHOCK_LOADING_BAND[0]).all()
    assert (loading <= SHOCK_LOADING_BAND[1]).all()
    assert len(set(np.round(loading, 9))) == 4

    common = dict(months=12, mechanisms=mechanisms)
    spike = [{"year": 0, "kind": "mortality_spike", "mortality_multiplier": 3.0,
              "admission_multiplier": 2.6}]
    shocked = build_event_history(micro, SEED, identities, dwellings, businesses,
                                  hospitals, shocks=spike, **common)
    quiet = build_event_history(micro, SEED, identities, dwellings, businesses,
                                hospitals, shocks=[], **common)

    county_flat = np.asarray(admin["county"], dtype=np.int64).reshape(-1)
    state_of_county = np.asarray(admin["county_state"], dtype=np.int64)
    initial = shocked["initial_state"]["person"]
    person_state = state_of_county[np.maximum(county_flat[initial["cell"]], 0)]

    def deaths_by_state(history):
        event = history["event"]
        died = event["event_type"] == EVENT_TYPES["person_death"]
        position = ((event["truth_person_id"][died] & np.uint64(SEQUENCE_MASK))
                    .astype(np.int64) - 1)
        position = position[(position >= 0) & (position < len(person_state))]
        return np.bincount(person_state[position], minlength=4).astype(np.float64)

    excess = deaths_by_state(shocked) / np.maximum(deaths_by_state(quiet), 1.0)
    assert excess.min() > 1.0
    order = np.argsort(loading)
    assert excess[order[-1]] > excess[order[0]]
    assert float(np.corrcoef(loading, excess)[0, 1]) > 0.5


def test_a_quiet_month_is_untouched_by_the_loadings():
    """A multiplier of one is one in every region, so a shock-free world does not move."""
    assert regional_multiplier(1.0, np.asarray([0.35, 1.0, 1.8])).tolist() == [1.0, 1.0, 1.0]
    world, people, micro, identities, dwellings, businesses, hospitals = _start_full()
    admin = build_admin(world["land"], people["settlements"],
                        resource_outposts(world, SEED), n_states=4)
    mechanisms = build_world_mechanisms(SEED, "development", admin, micro, businesses)
    quiet = build_event_history(micro, SEED, identities, dwellings, businesses,
                                hospitals, months=6, shocks=[], mechanisms=mechanisms)
    flat = build_world_mechanisms(SEED, "development", admin, micro, businesses)
    object.__setattr__(flat, "county_shock_loading",
                       np.ones_like(flat.county_shock_loading))
    same = build_event_history(micro, SEED, identities, dwellings, businesses,
                               hospitals, months=6, shocks=[], mechanisms=flat)
    assert _history_digest(quiet) == _history_digest(same)
