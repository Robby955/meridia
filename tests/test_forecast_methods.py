"""Forecast strong method A reads its rates from the ledger and clears the hard gates."""

import shutil
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from meridia.forecast import ForecastParams, build_forecast_packet, verify_forecast
from meridia.methods import forecast_cohort

SEED = 6161
PARAMS = ForecastParams(grid=(72, 96), n_settlements=6, n_states=2, history_months=24,
                        horizon_months=36, demand_window_months=12, total=60_000)


@pytest.fixture(scope="module")
def packet(tmp_path_factory):
    out = tmp_path_factory.mktemp("fc") / "hidden"
    build_forecast_packet(SEED, out, PARAMS, development=False)
    return out


def test_rates_are_estimated_from_the_ledger(packet):
    data = forecast_cohort.load_forecast_packet(packet)
    rates = forecast_cohort.estimate_rates(data, int(data["contract"]["ticks"]["snapshot"]))
    assert 0.03 < rates["fertility"] < 0.2
    assert 5e-6 <= rates["gompertz_a"] <= 1e-4 and rates["births"] > 0
    assert rates["deaths"].sum() > 0 and rates["admissions"].sum() > 0


def test_method_clears_hard_gates_from_participant_files(packet, tmp_path):
    blind = tmp_path / "packet"
    blind.mkdir()
    shutil.copytree(packet / "participant", blind / "participant")
    out = tmp_path / "FA"
    forecast_cohort.run(blind, out, forecast_cohort.MethodParams(replicates=40))
    report = verify_forecast(packet, out)
    assert report["pass"], report["reasons"]
    assert report["metrics"]["persons/nation"]["worst_error"] < 0.06
    assert report["metrics"]["persons/all"]["coverage"] > 0.5
    assert report["allocation"]["feasible"] and report["allocation"]["regret"] < 0.2


def test_method_is_deterministic(packet, tmp_path):
    a = forecast_cohort.run(packet, tmp_path / "a", forecast_cohort.MethodParams(replicates=10))
    b = forecast_cohort.run(packet, tmp_path / "b", forecast_cohort.MethodParams(replicates=10))
    assert a["projection"] == b["projection"]
