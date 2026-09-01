"""Forecast task: clean opaque history on the participant side, sealed future on the
retained side, demand that actually arrives, and a verifier that prices the allocation."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from meridia.forecast import ForecastParams, build_forecast_packet, verify_forecast
from meridia.release import required_rows
from meridia.scoring import rows_from_values
from meridia.verify import admin_from_packet, load_truth

SEED = 8080
PARAMS = ForecastParams(grid=(72, 96), n_settlements=6, n_states=2, history_months=12,
                        horizon_months=24, demand_window_months=6, total=40_000)


@pytest.fixture(scope="module")
def packet(tmp_path_factory):
    out = tmp_path_factory.mktemp("forecast") / "hidden"
    manifest = build_forecast_packet(SEED, out, PARAMS, development=False)
    return out, manifest


def test_participant_side_is_clean_and_opaque(packet):
    out, manifest = packet
    persons = pd.read_csv(out / "participant" / "persons.csv")
    events = pd.read_csv(out / "participant" / "events.csv")
    hospitals = pd.read_csv(out / "participant" / "hospitals.csv")
    contract = json.loads((out / "participant" / "contract.json").read_text())
    for frame in (persons, events, hospitals):
        assert not any(c.startswith("truth_") for c in frame.columns)
    assert persons["person_id"].is_unique and (persons["person_id"] > 0).all()
    assert events["tick"].max() <= contract["ticks"]["snapshot"]
    assert set(events["event"]) >= {"person_birth", "person_death", "encounter_admitted"}
    assert (hospitals["bed_count"] > 0).all() and hospitals["county"].between(0, contract["n_counties"] - 1).all()
    assert not (out / "participant" / "truth").exists()
    assert contract["allocation"]["budget"] > 0


def test_snapshot_persons_match_participant_file(packet):
    out, _ = packet
    persons = pd.read_csv(out / "participant" / "persons.csv")
    truth_now = load_truth(out / "retained" / "truth_snapshot.csv")
    assert len(persons) == truth_now[("persons", "nation", 0)]
    assert persons["household_id"].nunique() == truth_now[("households", "nation", 0)]
    admin = admin_from_packet(out)
    for c in range(admin["n_counties"]):
        assert int((persons["county"] == c).sum()) == truth_now[("persons", "county", c)]


def test_future_differs_and_demand_exists(packet):
    out, _ = packet
    now = load_truth(out / "retained" / "truth_snapshot.csv")
    future = load_truth(out / "retained" / "truth_horizon.csv")
    admin = admin_from_packet(out)
    assert set(future) == required_rows(admin)
    assert future[("persons", "nation", 0)] != now[("persons", "nation", 0)]
    demand = pd.read_csv(out / "retained" / "demand_horizon.csv")["admissions"].to_numpy()
    assert demand.sum() > 0 and len(demand) == admin["n_counties"]


def test_verifier_prices_the_allocation(packet, tmp_path):
    out, _ = packet
    future = load_truth(out / "retained" / "truth_horizon.csv")
    demand = pd.read_csv(out / "retained" / "demand_horizon.csv")["admissions"].to_numpy(dtype=float)
    contract = json.loads((out / "participant" / "contract.json").read_text())
    budget = float(contract["allocation"]["budget"])
    sub = tmp_path / "oracle"
    sub.mkdir()
    pd.DataFrame(rows_from_values(future, lambda e, v: 0.02 * max(abs(v), 1.0))).to_csv(sub / "projection.csv", index=False)
    pd.DataFrame({"county": np.arange(len(demand)),
                  "allocation": np.floor(demand / demand.sum() * budget * 1e6) / 1e6}).to_csv(sub / "allocation.csv", index=False)
    report = verify_forecast(out, sub)
    assert report["pass"], report["reasons"]
    assert report["metrics"]["persons/all"]["worst_error"] == 0.0
    assert abs(report["allocation"]["regret"]) < 1e-9
    uniform = tmp_path / "uniform"
    uniform.mkdir()
    (uniform / "projection.csv").write_bytes((sub / "projection.csv").read_bytes())
    pd.DataFrame({"county": np.arange(len(demand)),
                  "allocation": np.full(len(demand), np.floor(budget / len(demand) * 1e6) / 1e6)}).to_csv(uniform / "allocation.csv", index=False)
    report = verify_forecast(out, uniform, {"allocation_regret_ceiling": 0.02})
    assert not report["pass"] and report["allocation"]["regret"] > 0.02


def test_forecast_packet_deterministic(tmp_path):
    a = build_forecast_packet(SEED, tmp_path / "a", PARAMS)
    b = build_forecast_packet(SEED, tmp_path / "b", PARAMS)
    assert a["participant"] == b["participant"] and a["retained"] == b["retained"]
