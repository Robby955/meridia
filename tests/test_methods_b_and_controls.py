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
PARAMS = PacketParams(grid=(72, 96), n_settlements=6, n_states=2, observed_months=6,
                      preliminary_lag=3, horizon_months=12, total=40_000)


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
    assert report["pass"], report["reasons"]
    assert report["metrics"]["persons/nation"]["worst_error"] < 0.06
    assert report["metrics"]["persons/all"]["coverage"] > 0.5
    assert report["allocation"]["feasible"]


@pytest.mark.parametrize("name", controls.CONTROLS)
def test_every_control_writes_a_complete_submission(packet, dev_packet, tmp_path, name):
    out = tmp_path / name
    calibration = None
    if name == "exact_key_union":
        calibration = str(tmp_path / "calibration_A.json")
        design_based.calibrate([dev_packet], calibration)
    controls.run(name, packet, out, calibration_path=calibration)
    for file in ("release.csv", "projection.csv", "detailed.csv", "allocation.csv"):
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
