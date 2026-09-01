"""Method B clears the hard gates from participant files alone; every control runs and
writes a complete submission."""

import shutil
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from meridia.methods import bayesian, controls
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
def test_every_control_writes_a_complete_submission(packet, tmp_path, name):
    out = tmp_path / name
    controls.run(name, packet, out)
    for file in ("release.csv", "projection.csv", "detailed.csv", "allocation.csv"):
        assert (out / file).exists(), (name, file)
    report = verify_submission(packet, out)
    assert report["schema_errors"] == [], (name, report["schema_errors"][:3])
