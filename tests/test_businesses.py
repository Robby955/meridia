"""Business identity separation and exact employment/payroll accounting."""

import hashlib
import sys
from dataclasses import replace
from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from meridia.businesses import ESTABLISHMENT_ROLES, build_businesses
from meridia.businesses import business_params_from_character
from meridia.businesses import validate_business_conservation
from meridia.character import draw_world_character
from meridia.hydrology import fill_depressions, flow_accumulation, flow_directions
from meridia.identities import ENTITY_NAMESPACE, build_initial_identity_map
from meridia.identities import entity_namespace
from meridia.microdata import build_microdata
from meridia.population import build_population
from meridia.terrain import generate_elevation

SEED = 20260831
H, W = 60, 72
TOTAL = 40_000


@lru_cache(maxsize=2)
def _start(seed: int = SEED):
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
    return micro, identities


def _state_digest(state: dict) -> str:
    digest = hashlib.sha256()
    for name, value in state["business_params"].items():
        digest.update(name.encode("utf-8"))
        digest.update(repr(value).encode("ascii"))
    for table_name in ("enterprise", "establishment", "job"):
        digest.update(table_name.encode("utf-8"))
        for name, values in state[table_name].items():
            digest.update(name.encode("utf-8"))
            digest.update(np.ascontiguousarray(values).tobytes())
    return digest.hexdigest()


def test_three_business_identities_are_kept_separate():
    micro, identities = _start()
    state = build_businesses(micro, SEED, identities)
    enterprise_id = state["enterprise"]["truth_enterprise_id"]
    establishment_id = state["establishment"]["truth_establishment_id"]

    assert (entity_namespace(enterprise_id) == ENTITY_NAMESPACE["enterprise"]).all()
    assert (
        entity_namespace(establishment_id) == ENTITY_NAMESPACE["establishment"]
    ).all()
    assert np.intersect1d(enterprise_id, establishment_id).size == 0
    assert "observed_business_register_id" not in state["enterprise"]
    assert "observed_business_register_id" not in state["establishment"]
    assert "observed_business_register_id" not in state["job"]


def test_every_job_links_one_person_to_one_establishment():
    micro, identities = _start()
    state = build_businesses(micro, SEED, identities)
    job = state["job"]
    establishment = state["establishment"]
    person_id = identities["identity"]["truth_person_id"]

    assert state["n_jobs"] == len(job["truth_person_id"])
    assert len(np.unique(job["truth_person_id"])) == state["n_jobs"]
    assert np.isin(job["truth_person_id"], person_id).all()
    assert np.isin(
        job["truth_establishment_id"], establishment["truth_establishment_id"]
    ).all()
    assert np.array_equal(
        job["annual_earnings_cents"],
        job["annual_hours"].astype(np.int64) * job["hourly_wage_cents"],
    )


def test_payroll_and_counts_reconcile_at_both_business_levels():
    micro, identities = _start()
    state = build_businesses(micro, SEED, identities)
    enterprise = state["enterprise"]
    establishment = state["establishment"]
    job = state["job"]

    job_establishment_index = np.searchsorted(
        establishment["truth_establishment_id"], job["truth_establishment_id"]
    )
    establishment_jobs = np.zeros(state["n_establishments"], dtype=np.int32)
    establishment_payroll = np.zeros(state["n_establishments"], dtype=np.int64)
    np.add.at(establishment_jobs, job_establishment_index, 1)
    np.add.at(
        establishment_payroll, job_establishment_index, job["annual_earnings_cents"]
    )
    assert np.array_equal(establishment["employment_count"], establishment_jobs)
    assert np.array_equal(establishment["annual_payroll_cents"], establishment_payroll)

    establishment_enterprise_index = np.searchsorted(
        enterprise["truth_enterprise_id"], establishment["truth_enterprise_id"]
    )
    enterprise_establishments = np.zeros(state["n_enterprises"], dtype=np.int32)
    enterprise_jobs = np.zeros(state["n_enterprises"], dtype=np.int32)
    enterprise_payroll = np.zeros(state["n_enterprises"], dtype=np.int64)
    np.add.at(enterprise_establishments, establishment_enterprise_index, 1)
    np.add.at(
        enterprise_jobs,
        establishment_enterprise_index,
        establishment["employment_count"],
    )
    np.add.at(
        enterprise_payroll,
        establishment_enterprise_index,
        establishment["annual_payroll_cents"],
    )
    assert np.array_equal(enterprise["establishment_count"], enterprise_establishments)
    assert np.array_equal(enterprise["employment_count"], enterprise_jobs)
    assert np.array_equal(enterprise["annual_payroll_cents"], enterprise_payroll)
    assert int(enterprise["employment_count"].sum()) == state["n_jobs"]
    assert (establishment["employment_count"] >= 1).all()
    assert (
        int(
            (
                establishment["establishment_role"]
                == ESTABLISHMENT_ROLES["headquarters"]
            ).sum()
        )
        == state["n_enterprises"]
    )
    validate_business_conservation(state, micro, identities, SEED)


def test_business_state_is_byte_deterministic():
    micro, identities = _start()
    first = build_businesses(micro, SEED, identities)
    second = build_businesses(micro, SEED, identities)
    assert _state_digest(first) == _state_digest(second)


def test_default_generation_uses_the_world_character_employment_dial():
    micro, identities = _start()
    state = build_businesses(micro, SEED, identities)
    character = draw_world_character(SEED)["business"]

    for name in (
        "jobs_per_adult",
        "establishment_size_alpha",
        "multi_establishment_rate",
        "payroll_level",
    ):
        assert state["business_params"][name] == character[name]

    params = business_params_from_character(character)
    working_age = (micro["person"]["age"] >= params.minimum_work_age) & (
        micro["person"]["age"] <= params.maximum_work_age
    )
    assert state["n_jobs"] == round(params.jobs_per_adult * int(working_age.sum()))


def test_all_four_world_character_business_dials_are_load_bearing():
    micro, identities = _start()
    base = business_params_from_character(draw_world_character(SEED)["business"])

    low_jobs = build_businesses(
        micro, SEED, identities, replace(base, jobs_per_adult=0.56)
    )
    high_jobs = build_businesses(
        micro, SEED, identities, replace(base, jobs_per_adult=0.79)
    )
    assert high_jobs["n_jobs"] > low_jobs["n_jobs"]

    heavy_tail = build_businesses(
        micro, SEED, identities, replace(base, establishment_size_alpha=1.4)
    )
    light_tail = build_businesses(
        micro, SEED, identities, replace(base, establishment_size_alpha=2.2)
    )
    assert np.std(heavy_tail["establishment"]["employment_count"]) > np.std(
        light_tail["establishment"]["employment_count"]
    )

    low_multi = build_businesses(
        micro, SEED, identities, replace(base, multi_establishment_rate=0.08)
    )
    high_multi = build_businesses(
        micro, SEED, identities, replace(base, multi_establishment_rate=0.30)
    )
    low_share = np.mean(low_multi["enterprise"]["establishment_count"] > 1)
    high_share = np.mean(high_multi["enterprise"]["establishment_count"] > 1)
    assert high_share > low_share + 0.20

    low_payroll = build_businesses(
        micro, SEED, identities, replace(base, payroll_level=0.75)
    )
    high_payroll = build_businesses(
        micro, SEED, identities, replace(base, payroll_level=1.30)
    )
    assert int(high_payroll["job"]["annual_earnings_cents"].sum()) > int(
        low_payroll["job"]["annual_earnings_cents"].sum()
    )


def test_payroll_tamper_is_rejected():
    micro, identities = _start()
    state = build_businesses(micro, SEED, identities)
    changed = {**state, "establishment": {**state["establishment"]}}
    changed["establishment"]["annual_payroll_cents"] = state["establishment"][
        "annual_payroll_cents"
    ].copy()
    changed["establishment"]["annual_payroll_cents"][0] += 1

    with pytest.raises(ValueError, match="establishment payroll"):
        validate_business_conservation(changed, micro, identities, SEED)


def test_business_builder_rejects_a_seed_from_another_truth_world():
    micro, identities = _start()
    with pytest.raises(ValueError, match="seed does not match"):
        build_businesses(micro, SEED + 1, identities)
