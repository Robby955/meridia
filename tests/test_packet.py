"""Packets: flat participant files with no truth, sealed retained truth that matches the
engine exactly, deterministic manifests, and a development packet that ships its truth."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from meridia.packet import (FORBIDDEN_COLUMN_PREFIXES, PacketParams, build_packet,
                            build_world, participant_columns)
from meridia.projection import project_truth_from_history
from meridia.release import ESTIMAND_IDS, required_rows

SEED = 9001
PARAMS = PacketParams(grid=(72, 96), n_settlements=6, n_states=2, observed_months=8,
                      preliminary_lag=3, horizon_months=12, total=30_000)


@pytest.fixture(scope="module")
def packet(tmp_path_factory):
    out = tmp_path_factory.mktemp("packets") / "hidden"
    manifest = build_packet(SEED, out, PARAMS, development=False)
    return out, manifest


def _read_truth(path: Path) -> dict:
    import pandas as pd
    frame = pd.read_csv(path)
    return {(r.estimand, r.level, int(r.unit)): float(r.value) for r in frame.itertuples()}


def test_participant_side_carries_no_truth_columns(packet):
    out, manifest = packet
    for name, columns in participant_columns(out).items():
        for column in columns:
            assert not column.startswith(FORBIDDEN_COLUMN_PREFIXES), (name, column)
    assert "truth/truth_revised.csv" not in manifest["participant"]
    assert not (out / "participant" / "truth").exists()
    assert set(manifest["participant"]) >= {
        "survey_preliminary.csv", "survey_revised.csv", "geography.csv", "contract.json",
        "sources/population_preliminary.csv", "sources/health_revised.csv"}


def test_contract_names_what_the_scorer_needs(packet):
    out, _ = packet
    contract = json.loads((out / "participant" / "contract.json").read_text())
    assert [e["id"] for e in contract["estimands"]] == list(ESTIMAND_IDS)
    assert contract["levels"] == ["nation", "state", "county"]
    assert contract["ticks"]["preliminary"] < contract["ticks"]["revised"] < contract["ticks"]["horizon"]
    assert contract["allocation"]["budget"] > 0
    assert contract["development"] is False


def test_retained_truth_matches_the_engine_exactly(packet):
    out, _ = packet
    built = build_world(SEED, PARAMS)
    admin = built["admin"]
    revised = _read_truth(out / "retained" / "truth_revised.csv")
    horizon = _read_truth(out / "retained" / "truth_horizon.csv")
    assert set(revised) == required_rows(admin) and set(horizon) == required_rows(admin)
    engine = project_truth_from_history(built["history"], admin, built["ticks"]["horizon"])["truth"]
    for key, value in engine.items():
        if np.isnan(value):
            assert np.isnan(horizon[key])
        else:
            assert horizon[key] == pytest.approx(value, abs=1e-6), key
    assert revised[("persons", "nation", 0)] != horizon[("persons", "nation", 0)]


def test_survey_snapshots_differ_and_look_like_surveys(packet):
    out, _ = packet
    import pandas as pd
    a = pd.read_csv(out / "participant" / "survey_preliminary.csv")
    b = pd.read_csv(out / "participant" / "survey_revised.csv")
    assert set(a.columns) == {"household", "county", "stratum", "design_weight", "age", "sex",
                              "education", "income"}
    assert a["household"].min() == 0 and a["household"].max() == a["household"].nunique() - 1
    assert len(a) > 100 and len(b) > 100 and len(a) != len(b)
    assert a["income"].isna().any() and (a["design_weight"] >= 1).all()


def test_geography_map_matches_admin(packet):
    out, _ = packet
    import pandas as pd
    geography = pd.read_csv(out / "participant" / "geography.csv")
    admin = build_world(SEED, PARAMS)["admin"]
    assert list(geography["county"]) == list(range(admin["n_counties"]))
    assert list(geography["state"]) == list(admin["county_state"])


def test_packet_is_byte_deterministic(tmp_path):
    first = build_packet(SEED, tmp_path / "a", PARAMS)
    second = build_packet(SEED, tmp_path / "b", PARAMS)
    assert first["participant"] == second["participant"]
    assert first["retained"] == second["retained"]
    with pytest.raises(FileExistsError):
        build_packet(SEED, tmp_path / "a", PARAMS)


def test_development_packet_ships_truth_and_hidden_does_not(tmp_path, packet):
    out, hidden = packet
    dev = build_packet(SEED, tmp_path / "dev", PARAMS, development=True)
    assert "truth/truth_horizon.csv" in dev["participant"]
    assert dev["participant"]["truth/truth_horizon.csv"]["sha256"] == \
        hidden["retained"]["truth_horizon.csv"]["sha256"]
    public = {k: v for k, v in dev["participant"].items()
              if not k.startswith("truth/") and k != "contract.json"}
    assert public == {k: v for k, v in hidden["participant"].items() if k != "contract.json"}
