"""Stage ten: future truth is the advanced population, exactly; allocation loss is
feasibility-gated, oracle-anchored, and cannot be hedged."""

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from meridia.admin import build_admin
from meridia.demography import draw_world_shocks
from meridia.hydrology import fill_depressions, flow_accumulation, flow_directions
from meridia.microdata import build_microdata
from meridia.population import build_population, resource_outposts
from meridia.projection import demand_from_truth, project_truth, score_allocation
from meridia.release import compute_truth, required_rows
from meridia.scoring import rows_from_values, score_release, validate_release
from meridia.terrain import generate_elevation

SEED = 777
H, W = 96, 128
TOTAL = 120_000
YEARS = 5
_CACHE = {}


def _setup():
    if "admin" not in _CACHE:
        world = generate_elevation(SEED, H, W)
        outlets = ~world["land"]
        outlets[0, :] = outlets[-1, :] = outlets[:, 0] = outlets[:, -1] = True
        filled = fill_depressions(world["elevation"], world["sea_level"])
        direction = flow_directions(filled, outlets)
        accumulation = flow_accumulation(direction, outlets)
        people = build_population(world, accumulation, TOTAL, 8, seed=SEED)
        micro = build_microdata(people["population"], people["habitability"],
                                people["settlements"], SEED)
        admin = build_admin(world["land"], people["settlements"],
                            resource_outposts(world, SEED), n_states=3)
        shocks = draw_world_shocks(SEED, YEARS)
        future = project_truth(micro["person"], micro["household_cell"],
                               micro["urbanity"].flatten(), admin, SEED, YEARS, shocks=shocks)
        _CACHE.update(micro=micro, admin=admin, future=future,
                      now=compute_truth(micro["person"], micro["household_cell"], admin))
    return _CACHE


def test_future_truth_is_complete_and_conserved():
    s = _setup()
    future, admin = s["future"], s["admin"]
    assert set(future["truth"]) == required_rows(admin)
    assert future["truth"][("persons", "nation", 0)] == future["n_persons"]
    assert future["registers"][-1]["population_end"] == future["n_persons"]
    counties = sum(future["truth"][("persons", "county", c)] for c in range(admin["n_counties"]))
    assert counties == future["n_persons"]


def test_the_country_changed():
    s = _setup()
    assert s["future"]["truth"][("persons", "nation", 0)] != s["now"][("persons", "nation", 0)]
    assert s["future"]["truth"][("elders_65_plus", "nation", 0)] != s["now"][("elders_65_plus", "nation", 0)]


def test_projection_scored_like_a_release():
    s = _setup()
    naive = rows_from_values(s["now"], lambda e, v: 0.02 * max(abs(v), 1.0))   # "nothing changes"
    assert validate_release(naive, s["admin"]) == []
    metrics = score_release(naive, s["future"]["truth"], s["admin"])
    assert metrics["persons/nation"]["worst_error"] > 0.0
    oracle = rows_from_values(s["future"]["truth"], lambda e, v: 0.01 * max(abs(v), 1.0))
    assert score_release(oracle, s["future"]["truth"], s["admin"])["persons/county"]["coverage"] == 1.0


def test_allocation_oracle_and_proportional_have_zero_regret():
    s = _setup()
    demand = demand_from_truth(s["future"]["truth"], s["admin"])
    assert demand.sum() > 0
    generous = score_allocation(demand, demand, budget=demand.sum())
    assert generous["feasible"] and generous["loss"] == 0.0 and generous["regret"] == 0.0
    budget = 0.7 * demand.sum()
    proportional = score_allocation(demand * budget / demand.sum(), demand, budget)
    assert proportional["feasible"]
    assert proportional["loss"] == pytest.approx(0.3)
    assert proportional["regret"] == pytest.approx(0.0, abs=1e-12)


def test_uniform_allocation_has_regret_and_hedging_cannot_remove_it():
    s = _setup()
    demand = demand_from_truth(s["future"]["truth"], s["admin"])
    budget = 0.7 * demand.sum()
    uniform = score_allocation(np.full_like(demand, budget / len(demand)), demand, budget)
    assert uniform["feasible"] and uniform["regret"] > 0.05
    assert uniform["waste"] > 0.0


def test_infeasible_allocations_fail():
    s = _setup()
    demand = demand_from_truth(s["future"]["truth"], s["admin"])
    over = score_allocation(demand, demand, budget=0.5 * demand.sum())
    negative = score_allocation(-demand, demand, budget=demand.sum())
    assert not over["feasible"] and math.isnan(over["loss"])
    assert not negative["feasible"]
    with pytest.raises(ValueError):
        score_allocation(demand[:-1], demand, budget=demand.sum())


def test_projection_deterministic():
    s = _setup()
    again = project_truth(s["micro"]["person"], s["micro"]["household_cell"],
                          s["micro"]["urbanity"].flatten(), s["admin"], SEED, YEARS,
                          shocks=draw_world_shocks(SEED, YEARS))
    assert again["truth"] == s["future"]["truth"]
