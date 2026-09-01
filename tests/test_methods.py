"""Reference method A runs from participant files alone and clears the hard gates."""

import shutil
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from meridia.methods import design_based
from meridia.packet import PacketParams, build_packet
from meridia.verify import verify_submission

SEED = 31337
PARAMS = PacketParams(grid=(72, 96), n_settlements=6, n_states=2, observed_months=6,
                      preliminary_lag=3, horizon_months=12, total=40_000)


@pytest.fixture(scope="module")
def packet(tmp_path_factory):
    out = tmp_path_factory.mktemp("method") / "dev"
    build_packet(SEED, out, PARAMS, development=False)
    return out


@pytest.fixture(scope="module")
def submission(packet, tmp_path_factory):
    # The method sees a directory holding only the participant side.
    blind = tmp_path_factory.mktemp("blind") / "packet"
    blind.mkdir()
    shutil.copytree(packet / "participant", blind / "participant")
    out = tmp_path_factory.mktemp("sub") / "A"
    design_based.run(blind, out, design_based.MethodParams(bootstrap_replicates=25))
    return out


def test_method_a_writes_all_four_files(submission):
    for name in ("release.csv", "projection.csv", "detailed.csv", "allocation.csv"):
        assert (submission / name).exists()
    release = pd.read_csv(submission / "release.csv")
    assert (release["lower"] <= release["estimate"]).all() and (release["estimate"] <= release["upper"]).all()


def test_method_a_clears_hard_gates_and_is_close_on_persons(packet, submission):
    report = verify_submission(packet, submission)
    assert report["schema_errors"] == [] and report["additivity_errors"] == []
    assert report["disclosure"]["pass"] and report["allocation"]["feasible"]
    assert report["pass"], report["reasons"]
    assert report["metrics"]["persons/nation"]["worst_error"] < 0.05
    assert report["metrics"]["persons/county"]["worst_error"] < 0.5
    assert report["allocation"]["regret"] < 0.15


def test_method_a_is_deterministic(packet, tmp_path):
    first = design_based.run(packet, tmp_path / "a", design_based.MethodParams(bootstrap_replicates=10))
    second = design_based.run(packet, tmp_path / "b", design_based.MethodParams(bootstrap_replicates=10))
    assert first["release"] == second["release"] and first["projection"] == second["projection"]
