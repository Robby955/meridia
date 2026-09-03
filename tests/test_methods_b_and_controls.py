"""Method B clears the hard gates from participant files alone; every control runs and
writes a complete submission."""

import shutil
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from meridia.methods import bayesian, controls, design_based
from meridia.packet import PacketParams, build_packet
from meridia.verify import verify_submission

SEED = 4711
PARAMS = PacketParams(grid=(72, 96), n_settlements=6, n_states=2, observed_months=24,
                      preliminary_lag=3, horizon_months=12, total=40_000,
                      experience_years=1, ensemble_members=32)


@pytest.fixture(scope="module")
def packet(tmp_path_factory):
    out = tmp_path_factory.mktemp("b") / "hidden"
    build_packet(SEED, out, PARAMS, development=False)
    return out


@pytest.fixture(scope="module")
def dev_packet(tmp_path_factory):
    out = tmp_path_factory.mktemp("b") / "development"
    build_packet(SEED, out, PARAMS, development=True)
    return out


def test_method_b_clears_hard_gates_from_participant_files(packet, tmp_path):
    blind = tmp_path / "packet"
    blind.mkdir()
    shutil.copytree(packet / "participant", blind / "participant")
    out = tmp_path / "B"
    bayesian.run(blind, out, bayesian.MethodParams(sweeps=120, burn_in=40))
    report = verify_submission(packet, out)
    assert report["schema_errors"] == [] and report["additivity_errors"] == []
    assert report["metrics"]["persons/nation"]["worst_error"] < 0.06
    assert report["metrics"]["persons/all"]["coverage"] > 0.5
    assert report["reserve"]["feasible"]
    families = {reason.split(":")[0] for reason in report["reasons"]}
    assert families <= {"tail", "reserve", "exposure", "rate", "coverage"}, report["reasons"]


@pytest.mark.parametrize("name", controls.CONTROLS)
def test_every_control_writes_a_complete_submission(packet, dev_packet, tmp_path, name):
    out = tmp_path / name
    calibration = None
    if name == "exact_key_union":
        calibration = str(tmp_path / "calibration_A.json")
        design_based.calibrate([dev_packet], calibration)
    controls.run(name, packet, out, calibration_path=calibration)
    for file in ("release.csv", "projection.csv", "detailed.csv", "reserve.csv"):
        assert (out / file).exists(), (name, file)
    report = verify_submission(packet, out)
    assert report["schema_errors"] == [], (name, report["schema_errors"][:3])


def test_exact_key_union_control_reproduces_the_version_two_recipe(dev_packet, tmp_path):
    packet = dev_packet
    calibration = tmp_path / "calibration_A.json"
    fit = design_based.calibrate([packet], calibration)["exact_key_union"]
    assert set(fit) == {"ratio", "ratio_spread", "county_q90"}
    assert all(fit["ratio"][item] > 0 for item in fit["ratio"])
    with pytest.raises(ValueError, match="calibration A"):
        controls.run("exact_key_union", packet, tmp_path / "bare")
    out = tmp_path / "exact_key_union"
    controls.run("exact_key_union", packet, out, calibration_path=str(calibration))
    release = pd.read_csv(out / "release.csv")
    truth = pd.read_csv(packet / "participant" / "truth" / "truth_revised.csv")
    nation_truth = truth[truth["level"] == "nation"].set_index("estimand")["value"]
    nation = release[release["level"] == "nation"].set_index("estimand")
    # one fitting world: the constant reproduces that world's own nation exactly
    for item in ("persons", "households", "children_under_16", "elders_65_plus"):
        assert abs(nation.loc[item, "estimate"] / nation_truth[item] - 1.0) < 1e-9
        assert nation.loc[item, "lower"] < nation.loc[item, "estimate"] < nation.loc[item, "upper"]
    counties = release[(release["level"] == "county") & (release["estimand"] == "persons")]
    assert abs(counties["estimate"].sum() - nation.loc["persons", "estimate"]) < 1e-6


def test_both_calibrations_carry_the_fitted_ratio_exponents(dev_packet, tmp_path):
    import json
    from meridia.methods.design_based import RATIO_EXPONENT_BOUNDS, fit_ratio_exponents
    exponents = fit_ratio_exponents([dev_packet])
    assert exponents["median_household_income"] == 1.0
    for item in ("mean_income_adults", "low_income_household_share"):
        assert RATIO_EXPONENT_BOUNDS[0] <= exponents[item] <= RATIO_EXPONENT_BOUNDS[1]
    calibration_b = tmp_path / "calibration_B.json"
    bayesian.calibrate([dev_packet], calibration_b)
    stored = json.loads(calibration_b.read_text())
    assert stored["ratio_exponent"] == exponents
    assert "mean_income_adults" in stored and "n_worlds" in stored


def test_the_state_benchmark_step_moves_the_composition_and_keeps_the_total():
    """The national factor fixes the level; this step fixes the split across states."""
    import numpy as np
    county_state = np.asarray([0, 0, 1, 1, 2, 2])
    point = {("persons", "nation", 0): 1000.0,
             ("persons", "state", 0): 600.0,
             ("persons", "state", 1): 300.0,
             ("persons", "state", 2): 100.0}
    benchmark = {"persons": {"nation": 1000.0, "state": np.asarray([500.0, 350.0, 150.0])}}
    factors = design_based.benchmark_state_reconciliation(point, [], benchmark, county_state)
    moved = np.asarray([point[("persons", "state", s)] for s in range(3)]) * factors["persons"]
    assert moved.sum() == pytest.approx(1000.0)
    # Every state moves toward the benchmark and none moves past it.
    for s, (before, after) in enumerate(zip([600.0, 300.0, 100.0], moved)):
        target = benchmark["persons"]["state"][s]
        assert abs(after - target) < abs(before - target)
        assert (after - before) * (target - before) > 0

    # Composed with the national factor, the two steps scale once, not twice.
    national = {"persons": 1.10}
    scaled = design_based.apply_reconciliation(point, national, factors, county_state)
    states = sum(scaled[("persons", "state", s)] for s in range(3))
    assert states == pytest.approx(scaled[("persons", "nation", 0)])


def test_the_age_rake_reproduces_all_three_published_count_items():
    """A cube scaled by the persons factor alone has the right total and the wrong shape."""
    import numpy as np
    from meridia.methods.design_based import (CHILD_MAX_AGE, ELDER_MIN_AGE,
                                              benchmark_age_scale)
    n_ages = 101
    cube = np.zeros((2, n_ages, 2))
    cube[:, : CHILD_MAX_AGE + 1, :] = 5.0        # 2 counties x 16 ages x 2 sexes
    cube[:, CHILD_MAX_AGE + 1: ELDER_MIN_AGE, :] = 10.0
    cube[:, ELDER_MIN_AGE:, :] = 2.0
    county_state = np.asarray([0, 0])
    factors = {"persons": 1.0, "children_under_16": 1.2, "elders_65_plus": 0.8}
    scale = benchmark_age_scale(cube, county_state, factors)
    scaled = cube * scale[:, :, None]
    child = slice(0, CHILD_MAX_AGE + 1)
    elder = slice(ELDER_MIN_AGE, n_ages)
    assert scaled[:, child].sum() == pytest.approx(1.2 * cube[:, child].sum())
    assert scaled[:, elder].sum() == pytest.approx(0.8 * cube[:, elder].sum())
    assert scaled.sum() == pytest.approx(cube.sum())
