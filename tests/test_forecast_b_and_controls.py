"""Forecast method B clears the hard gates; every forecast control writes a submission."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from meridia.forecast import ForecastParams, build_forecast_packet, verify_forecast
from meridia.methods import forecast_bayes, forecast_controls

SEED = 7272
PARAMS = ForecastParams(grid=(72, 96), n_settlements=6, n_states=2, history_months=24,
                        horizon_months=36, demand_window_months=12, total=60_000)


@pytest.fixture(scope="module")
def packet(tmp_path_factory):
    out = tmp_path_factory.mktemp("fb") / "hidden"
    build_forecast_packet(SEED, out, PARAMS, development=False)
    return out


def test_method_b_clears_hard_gates(packet, tmp_path):
    out = tmp_path / "FB"
    result = forecast_bayes.run(packet, out, forecast_bayes.MethodParams(draws=60))
    report = verify_forecast(packet, out)
    assert report["pass"], report["reasons"]
    assert report["metrics"]["persons/nation"]["worst_error"] < 0.06
    assert 0.0 < result["posterior"]["fertility"] < 0.2
    assert report["allocation"]["feasible"]


@pytest.mark.parametrize("name", forecast_controls.CONTROLS)
def test_every_forecast_control_writes_a_submission(packet, tmp_path, name):
    out = tmp_path / name
    forecast_controls.run(name, packet, out)
    assert (out / "projection.csv").exists() and (out / "allocation.csv").exists()
    report = verify_forecast(packet, out)
    assert report["schema_errors"] == [], (name, report["schema_errors"][:3])
