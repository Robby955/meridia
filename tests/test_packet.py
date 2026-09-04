"""Packets: flat participant files with no truth, sealed retained truth that matches the
engine exactly, deterministic manifests, and a development packet that ships its truth."""

import hashlib
import json
import math
import os
import shutil
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from meridia.actuarial import ObligationContract
from meridia.packet import (ENSEMBLE_CACHE_SCHEMA, EXPERIENCE_BURN_IN_MONTHS,
                            FORBIDDEN_COLUMN_PREFIXES, GRADING_WORLD,
                            PACKET_MANIFEST_SCHEMA, PARTICIPANT_CSV_SCHEMAS,
                            PARTICIPANT_PACKET_FILES, PacketParams, _cached_liability,
                            _experience_history, _packet_build_provenance,
                            _publish_staging_directory,
                            _store_liability, _structural_sha256,
                            _validate_shock_redraw_evidence,
                            baseline_ledger_digest, build_packet, build_world,
                            continuation_source_law_digest,
                            continuation_source_modules, participant_columns,
                            reserve_weights, validate_packet_directory)
from meridia.projection import _shock_redraw_evidence, project_truth_from_history
from meridia.survey import (N_SURVEY_OUTSIDE_AXES, SURVEY_BANDS, SURVEY_ENVELOPE,
                            SurveyParams)
from meridia.release import ESTIMAND_IDS, required_rows

SEED = 9001
PARAMS = PacketParams(grid=(72, 96), n_settlements=6, n_states=2, observed_months=36,
                      preliminary_lag=3, horizon_months=12, total=30_000,
                      experience_years=2, ensemble_members=32)


@pytest.fixture(scope="module")
def packet(tmp_path_factory):
    out = tmp_path_factory.mktemp("packets") / "hidden"
    manifest = build_packet(SEED, out, PARAMS, development=False)
    return out, manifest


def _read_truth(path: Path) -> dict:
    import pandas as pd
    frame = pd.read_csv(path)
    return {(r.estimand, r.level, int(r.unit)): float(r.value) for r in frame.itertuples()}


def _cache_shock_evidence(members: int) -> dict:
    schedules = [(member, []) for member in range(members)]
    if schedules:
        schedules[0][1].append({
            "year": 3,
            "kind": "mortality_spike",
            "mortality_multiplier": 2.0,
            "admission_multiplier": 2.0,
        })
    return _shock_redraw_evidence(
        {"month": 36}, 12, members, schedules, continuation_source_law_digest()
    )


def _refresh_manifest_record(packet: Path, side: str, name: str) -> None:
    manifest_path = packet / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    file_path = packet / side / name
    manifest[side][name] = {
        "bytes": file_path.stat().st_size,
        "sha256": hashlib.sha256(file_path.read_bytes()).hexdigest(),
    }
    manifest_path.write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")


def _synthetic_graded_copy(source: Path, destination: Path) -> tuple[Path, PacketParams]:
    from meridia.mechanisms import (DEVELOPMENT_BAND, HIDDEN_EXTRAPOLATION_AXES,
                                    HIDDEN_LEVEL_PATTERNS, PUBLIC_ENVELOPE)

    shutil.copytree(source, destination)
    hidden_params = PacketParams(**{**PARAMS.__dict__, "regime": "hidden"})
    world_path = destination / "retained" / "world.json"
    world = json.loads(world_path.read_text())
    outside = list(HIDDEN_EXTRAPOLATION_AXES[:2])
    intensity = {
        axis: (sum(bounds) / 2.0) for axis, bounds in DEVELOPMENT_BAND.items()
    }
    for axis in outside:
        intensity[axis] = PUBLIC_ENVELOPE[axis][0]
    world.update({
        "packet_class": "graded",
        "regime": "hidden",
        "params": json.loads(json.dumps(asdict(hidden_params))),
        "build_provenance": _packet_build_provenance(hidden_params),
    })
    world["mechanisms"]["design"] = {
        "regime": "hidden",
        "cell": -1,
        "outside": outside,
        "levels": list(HIDDEN_LEVEL_PATTERNS[0]),
        "intensity": intensity,
    }
    world_path.write_text(json.dumps(world, indent=1, sort_keys=True) + "\n")
    manifest_path = destination / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["packet_class"] = "graded"
    manifest_path.write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")
    _refresh_manifest_record(destination, "retained", "world.json")
    return destination, hidden_params


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
    # The participant surface gained a file and three contract blocks, so the tag moves.
    assert contract["schema"] == "meridia.packet.v4"


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
    assert set(a.columns) == {"household", "county", "psu", "psu_sampled_households", "stratum",
                              "design_weight", "age", "sex", "education", "income",
                              "recent_hospitalization"}
    assert set(a["recent_hospitalization"].unique()) <= {0, 1}
    assert 0 < a["recent_hospitalization"].mean() < 0.5
    responding = a.groupby("psu")["household"].nunique()
    sampled = a.groupby("psu")["psu_sampled_households"].first()
    assert (responding <= sampled).all() and (sampled <= 12).all()
    assert a["household"].min() == 0 and a["household"].max() == a["household"].nunique() - 1
    assert len(a) > 100 and len(b) > 100 and len(a) != len(b)
    assert a["income"].isna().any() and (a["design_weight"] >= 1).all()


def test_experience_history_is_shipped_and_adds_up(packet):
    """Two snapshots do not identify a five-year trend, so the packet ships the trend's
    own aggregate evidence: exposure, deaths, qualifying events, and net migration."""
    import pandas as pd
    out, manifest = packet
    assert "experience_history.csv" in manifest["participant"]
    frame = pd.read_csv(out / "participant" / "experience_history.csv")
    assert list(frame.columns) == ["year", "age_band", "sex", "state", "exposure",
                                   "deaths", "qualifying_events", "net_migration"]
    contract = json.loads((out / "participant" / "contract.json").read_text())
    block = contract["experience_history"]
    years, bands = block["years"], block["age_bands"]
    assert bands == ["0-17", "18-44", "45-64", "65-74", "75-84", "85+"]
    assert sorted(frame["year"].unique()) == list(range(1, years + 1))
    assert set(frame["age_band"]) == set(bands)
    assert len(frame) == years * len(bands) * 2 * contract["n_states"]
    assert (frame["exposure"] >= 0).all() and frame["exposure"].sum() > 0
    assert (frame["deaths"] >= 0).all() and frame["deaths"].sum() > 0
    # Internal migration moves people between states and never creates them.
    for year, rows in frame.groupby("year"):
        assert rows["net_migration"].sum() == 0, year
    # Exposure is person-years, so a year of it is close to the mean living population.
    truth = _read_truth(out / "retained" / "truth_revised.csv")
    last = frame[frame["year"] == years]["exposure"].sum()
    assert 0.5 * truth[("persons", "nation", 0)] < last < 1.5 * truth[("persons", "nation", 0)]
    # The series lags the snapshot, so it never hands over a contemporaneous state count.
    assert block["publication_lag_months"] > 0
    assert block["last_year_ends_at_tick"] == contract["ticks"]["revised"] - block["publication_lag_months"]
    assert block["last_year_ends_at_tick"] < contract["ticks"]["preliminary"]
    # Deaths rise with age band, which is what makes the file a mortality anchor.
    by_band = frame.groupby("age_band")[["deaths", "exposure"]].sum()
    rate = by_band["deaths"] / by_band["exposure"].clip(lower=1e-9)
    assert rate["85+"] > rate["45-64"] > rate["18-44"]


def test_health_anchor_and_covariates_are_declared_in_the_contract(packet):
    out, _ = packet
    contract = json.loads((out / "participant" / "contract.json").read_text())
    anchor = contract["health_anchor"]
    assert anchor["item"] == "recent_hospitalization" and anchor["window_months"] == 12
    assert 0.5 < anchor["sensitivity"] < 1.0 and 0.5 < anchor["specificity"] < 1.0
    covariates = contract["mechanisms"]["covariates"]
    assert set(covariates) >= {"urban_c", "econ_c", "elder_c", "band_r"}
    import pandas as pd
    geography = pd.read_csv(out / "participant" / "geography.csv")
    assert "land_cells" in geography.columns and (geography["land_cells"] > 0).all()


def test_the_reserve_block_is_published_and_carries_no_sealed_quantity(packet):
    """The contract publishes the obligation, one aggregate scalar, and the weights.

    Protocol section 9 reveals R by design and nothing else about the ensemble. The
    weights are a public ladder a method can rebuild from a participant file, so the
    reserve problem is fully specified from the files the agent receives.
    """
    out, _ = packet
    contract = json.loads((out / "participant" / "contract.json").read_text())
    reserve = contract["reserve"]
    assert set(reserve) >= {
        "obligation", "total", "total_rule", "allocation_rule", "regions", "weights"
    }
    assert reserve["allocation_rule"] == {
        "finite": True,
        "minimum": 0.0,
        "sum": "reserve.total",
        "tolerance": pytest.approx(1e-6),
    }
    assert reserve["regions"] == "state"
    assert reserve["total"] > 0 and reserve["total"] % reserve["rounding_unit"] == 0
    weights = np.asarray(reserve["weights"], dtype=np.float64)
    assert len(weights) == contract["n_states"]
    assert weights.min() > 0 and len(set(np.round(weights, 6))) == len(weights)
    obligation = reserve["obligation"]
    assert obligation["horizon_months"] == PARAMS.horizon_months
    assert obligation["eligibility_min_age"] == 65
    # Nothing regional and sealed rides along: no per-region liability, quantile or
    # shortfall appears anywhere in the participant contract.
    text = json.dumps(reserve)
    for forbidden in ("q95", "es95", "liability", "exceedance", "member"):
        assert f'"{forbidden}"' not in text


def test_contract_lists_every_core_participant_csv_header(packet):
    out, _ = packet
    contract = json.loads((out / "participant" / "contract.json").read_text())
    observed = participant_columns(out)
    assert contract["participant_csv_schemas"] == observed
    assert contract["participant_csv_schemas"] == {
        name: list(columns) for name, columns in sorted(PARTICIPANT_CSV_SCHEMAS.items())
    }
    assert set(observed) == PARTICIPANT_PACKET_FILES - {"contract.json"}
    assert contract["benchmark"]["file"] == "sources/benchmark_revised.csv"
    assert (out / "participant" / contract["benchmark"]["file"]).is_file()


def test_contract_publishes_the_exact_three_file_schema(packet):
    out, _ = packet
    contract = json.loads((out / "participant" / "contract.json").read_text())
    assert contract["submission"] == {
        "files": {
            "release.csv": ["estimand", "level", "unit", "sex", "age_band",
                            "estimate", "lower", "upper"],
            "projection.csv": ["estimand", "level", "unit", "estimate", "lower", "upper"],
            "reserve.csv": ["region", "liability_mean", "q95", "es95", "allocation"],
        },
        "additional_entries": "forbidden",
    }


def test_contract_publishes_the_exposure_only_rate_eligibility_rule(packet):
    out, _ = packet
    contract = json.loads((out / "participant" / "contract.json").read_text())
    rule = contract["rate_eligibility"]
    assert rule["truth_quantity"] == "retained person-years exposure"
    assert rule["bands"] == ["0-17", "18-64", "65+"]
    assert rule["floor_person_years_by_band"] == {
        "0-17": 600.0, "18-64": 600.0, "65+": 500.0,
    }
    assert "reference_rate" not in rule
    assert "minimum_expected_events" not in rule


def test_the_published_weights_rebuild_from_the_population_source(packet):
    """A method reproduces the ladder, so the weights are information, not a secret."""
    import pandas as pd
    out, _ = packet
    contract = json.loads((out / "participant" / "contract.json").read_text())
    geography = pd.read_csv(out / "participant" / "geography.csv")
    population = pd.read_csv(out / "participant" / "sources" / "population_revised.csv")
    county_state = geography.set_index("county")["state"]
    tick = int(contract["ticks"]["revised"])
    rebuilt = reserve_weights(
        {"county": population["county"].to_numpy(),
         "birth_tick": population["birth_tick"].to_numpy()},
        county_state.to_numpy(), tick, int(contract["n_states"]),
        PARAMS.reserve_weight_spread)
    assert np.allclose(rebuilt, np.asarray(contract["reserve"]["weights"]))


def test_retained_tail_truth_and_rate_truth_are_written(packet):
    """The three artifacts the version-four verifier reads, in the shapes it reads."""
    import pandas as pd
    out, manifest = packet
    assert "continuation_liabilities.npz" in manifest["retained"]
    assert "rate_truth_horizon.csv" in manifest["retained"]
    contract = json.loads((out / "participant" / "contract.json").read_text())
    with np.load(out / "retained" / "continuation_liabilities.npz") as archive:
        liability = archive["liability"]
        assert "realized_member" not in archive
        assert np.allclose(archive["weights"], contract["reserve"]["weights"])
    assert liability.shape == (PARAMS.ensemble_members, contract["n_states"])
    assert (liability > 0).all() and len(np.unique(liability[:, 0])) > 1
    rates = pd.read_csv(out / "retained" / "rate_truth_horizon.csv")
    assert list(rates.columns) == ["estimand", "level", "unit", "sex", "age_band", "value"]
    assert set(rates["estimand"]) == {"person_years_exposure", "mortality_rate",
                                      "qualifying_event_rate"}
    assert set(rates["level"]) == {"state", "county"}
    exposure = rates[(rates["estimand"] == "person_years_exposure")
                     & (rates["level"] == "state")]
    for band in ("18-64", "65+"):
        assert (exposure["age_band"] == band).any()


def test_the_reserve_total_recomputes_from_the_public_experience_file(packet):
    import pandas as pd
    out, _ = packet
    contract = json.loads((out / "participant" / "contract.json").read_text())
    frame = pd.read_csv(out / "participant" / "experience_history.csv")
    rule = contract["reserve"]["total_rule"]
    latest = int(frame["year"].max())
    exposure = float(frame.loc[frame["year"] == latest, "exposure"].sum())
    raw = exposure * float(rule["rate_per_person_year"])
    expected = math.ceil(raw / float(rule["rounding_unit"])) * float(rule["rounding_unit"])
    total = float(contract["reserve"]["total"])
    assert rule["selected_year"] == latest
    assert rule["exposure_person_years"] == pytest.approx(exposure)
    assert total == pytest.approx(expected)
    assert 0 <= total - raw < float(rule["rounding_unit"])


def test_the_shock_family_is_published_with_its_rate(packet):
    """The tail's systematic risk is a declared family, not a hidden mechanism."""
    out, _ = packet
    contract = json.loads((out / "participant" / "contract.json").read_text())
    family = contract["shock_family"]
    assert 0.0 < family["annual_rate"] <= 0.5
    assert set(family["kinds"]) == {"mortality_spike", "migration_wave", "baby_bust"}
    epidemic = family["kinds"]["mortality_spike"]
    assert set(epidemic) == {"mortality_multiplier", "admission_multiplier"}
    for bounds in epidemic.values():
        assert len(bounds) == 2 and bounds[0] < bounds[1]


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


def test_packet_manifest_binds_class_params_and_retained_seed(packet):
    out, manifest = packet
    assert manifest["schema"] == PACKET_MANIFEST_SCHEMA
    assert manifest["packet_class"] == "qualification"
    world = json.loads((out / "retained" / "world.json").read_text())
    assert world["packet_class"] == "qualification"
    assert world["seed"] == SEED
    assert world["params"] == json.loads(json.dumps(asdict(PARAMS)))
    assert world["build_provenance"] == _packet_build_provenance(PARAMS)

    contract = json.loads((out / "participant" / "contract.json").read_text())
    assert "seed" not in contract and "packet_class" not in contract
    assert '"seed"' not in json.dumps(manifest)
    assert '"packet_class"' not in json.dumps(contract)
    assert validate_packet_directory(
        out,
        expected_packet_class="qualification",
        expected_params=PARAMS,
        expected_seed=SEED,
    ) == manifest


def test_packet_validation_rejects_generator_provenance_drift(packet, monkeypatch):
    import meridia.sealing as sealing

    out, _ = packet
    monkeypatch.setattr(sealing, "v4_generator_source_law_digest", lambda: "f" * 64)
    with pytest.raises(ValueError, match="build provenance"):
        validate_packet_directory(
            out,
            expected_packet_class="qualification",
            expected_params=PARAMS,
            expected_seed=SEED,
        )


def test_packet_validation_rejects_runtime_provenance_drift(packet, monkeypatch):
    """The interpreter and numerical-library law is bound as tightly as the source."""
    import meridia.sealing as sealing

    out, _ = packet
    monkeypatch.setattr(sealing, "v4_runtime_law_digest", lambda: "e" * 64)
    with pytest.raises(ValueError, match="build provenance"):
        validate_packet_directory(
            out,
            expected_packet_class="qualification",
            expected_params=PARAMS,
            expected_seed=SEED,
        )


def test_build_provenance_binds_the_normalized_parameters_and_rejects_a_bad_digest():
    record = _packet_build_provenance(PARAMS)
    assert set(record) == {"schema", "generator_source_law_sha256",
                           "runtime_law_sha256", "packet_params_sha256"}
    assert record == _packet_build_provenance(asdict(PARAMS))
    moved = PacketParams(**(asdict(PARAMS) | {"ensemble_members": 64}))
    assert _packet_build_provenance(moved)["packet_params_sha256"] \
        != record["packet_params_sha256"]
    assert _packet_build_provenance(moved)["generator_source_law_sha256"] \
        == record["generator_source_law_sha256"]


def test_build_provenance_refuses_a_malformed_digest(monkeypatch):
    import meridia.sealing as sealing

    monkeypatch.setattr(sealing, "v4_generator_source_law_digest", lambda: "not a digest")
    with pytest.raises(RuntimeError, match="malformed digest"):
        _packet_build_provenance(PARAMS)


def test_construction_refuses_provenance_that_moved_after_the_intent(tmp_path):
    """The staging writer recomputes the record rather than trusting what it is handed."""
    import meridia.packet as packet_module

    staging = tmp_path / "staging"
    staging.mkdir()
    stale = dict(_packet_build_provenance(PARAMS), generator_source_law_sha256="a" * 64)
    with pytest.raises(ValueError, match="provenance changed before construction"):
        packet_module._build_packet_into(
            SEED, staging, PARAMS, False, 1, None, "qualification", stale
        )
    assert not list(staging.iterdir())


def test_a_restart_refuses_an_intent_minted_under_a_different_generator(tmp_path,
                                                                       monkeypatch):
    """Crash recovery adopts staging only when the provenance is the current one."""
    import meridia.packet as packet_module
    import meridia.sealing as sealing

    out = tmp_path / "drifted"
    monkeypatch.setattr(sealing, "v4_generator_source_law_digest", lambda: "b" * 64)
    intent = packet_module._packet_build_intent(
        out,
        seed=SEED,
        params=PARAMS,
        packet_class="qualification",
        development=False,
        graded_authorization=None,
    )
    monkeypatch.undo()
    intent_path = tmp_path / ".drifted.build-intent.json"
    intent_path.write_bytes(packet_module._intent_bytes(intent))
    intent_path.chmod(0o600)
    staging = tmp_path / ".drifted.staging"
    staging.mkdir()
    marker = staging / "preserve"
    marker.write_text("for diagnosis")

    with pytest.raises(ValueError, match="does not match this exact packet build"):
        build_packet(SEED, out, PARAMS)
    assert marker.read_text() == "for diagnosis"
    assert intent_path.exists()
    assert not out.exists()


def test_packet_validation_fails_closed_without_disclosing_the_expected_seed(packet):
    out, _ = packet
    with pytest.raises(ValueError, match="seed does not match") as caught:
        validate_packet_directory(
            out,
            expected_packet_class="qualification",
            expected_params=PARAMS,
            expected_seed=SEED + 1,
        )
    assert str(SEED) not in str(caught.value)
    assert str(SEED + 1) not in str(caught.value)


def test_packet_validation_rejects_file_tampering(packet, tmp_path):
    out, _ = packet
    copied = tmp_path / "copied"
    shutil.copytree(out, copied)
    contract = copied / "participant" / "contract.json"
    original = contract.read_text()
    contract.write_text(original + " ")
    with pytest.raises(ValueError, match="does not match its manifest"):
        validate_packet_directory(
            copied,
            expected_packet_class="qualification",
            expected_params=PARAMS,
            expected_seed=SEED,
        )
    contract.write_text(original)
    (copied / "participant" / "unlisted.csv").write_text("value\n1\n")
    with pytest.raises(ValueError, match="file set does not match"):
        validate_packet_directory(
            copied,
            expected_packet_class="qualification",
            expected_params=PARAMS,
            expected_seed=SEED,
        )
    (copied / "participant" / "unlisted.csv").unlink()

    manifest_path = copied / "manifest.json"
    copied_manifest = json.loads(manifest_path.read_text())
    extra = copied / "participant" / "unlisted.csv"
    extra.write_text("value\n1\n")
    copied_manifest["participant"]["unlisted.csv"] = {
        "bytes": extra.stat().st_size,
        "sha256": hashlib.sha256(extra.read_bytes()).hexdigest(),
    }
    manifest_path.write_text(json.dumps(copied_manifest))
    with pytest.raises(ValueError, match="non-canonical participant inventory"):
        validate_packet_directory(
            copied,
            expected_packet_class="qualification",
            expected_params=PARAMS,
            expected_seed=SEED,
        )
    extra.unlink()
    copied_manifest["participant"].pop("unlisted.csv")

    population_record = copied_manifest["participant"]["sources/population_revised.csv"]
    population_record["sha256"] = population_record["sha256"].upper()
    manifest_path.write_text(json.dumps(copied_manifest))
    with pytest.raises(ValueError, match="digest is malformed"):
        validate_packet_directory(
            copied,
            expected_packet_class="qualification",
            expected_params=PARAMS,
            expected_seed=SEED,
        )

    empty = copied / "participant" / "survey_revised.csv"
    population_record["sha256"] = population_record["sha256"].lower()
    empty.write_bytes(b"")
    empty_record = copied_manifest["participant"]["survey_revised.csv"]
    empty_record["bytes"] = 0
    empty_record["sha256"] = hashlib.sha256(b"").hexdigest()
    manifest_path.write_text(json.dumps(copied_manifest))
    with pytest.raises(ValueError, match="header is empty"):
        validate_packet_directory(
            copied,
            expected_packet_class="qualification",
            expected_params=PARAMS,
            expected_seed=SEED,
        )


def test_packet_validation_rejects_the_wrong_class_or_params(packet):
    out, _ = packet
    with pytest.raises(ValueError, match="class does not match"):
        validate_packet_directory(
            out,
            expected_packet_class="graded",
            expected_params=PARAMS,
            expected_seed=SEED,
        )
    changed = PacketParams(**{**PARAMS.__dict__, "horizon_months": 24})
    with pytest.raises(ValueError, match="parameters do not match"):
        validate_packet_directory(
            out,
            expected_packet_class="qualification",
            expected_params=changed,
            expected_seed=SEED,
        )


def test_packet_validation_rejects_extra_directories_and_special_nodes(packet, tmp_path):
    source, _ = packet
    extra_directory = tmp_path / "extra-directory"
    shutil.copytree(source, extra_directory)
    (extra_directory / "participant" / "unused").mkdir()
    with pytest.raises(ValueError, match="directory topology is non-canonical"):
        validate_packet_directory(
            extra_directory,
            expected_packet_class="qualification",
            expected_params=PARAMS,
            expected_seed=SEED,
        )

    special_node = tmp_path / "special-node"
    shutil.copytree(source, special_node)
    os.mkfifo(special_node / "retained" / "unexpected.fifo")
    with pytest.raises(ValueError, match="contains a special file"):
        validate_packet_directory(
            special_node,
            expected_packet_class="qualification",
            expected_params=PARAMS,
            expected_seed=SEED,
        )


def test_graded_validation_rejects_seed_values_in_canonical_participant_files(
    packet, tmp_path
):
    source, _ = packet
    copied, hidden_params = _synthetic_graded_copy(source, tmp_path / "graded-copy")
    assert validate_packet_directory(
        copied,
        expected_packet_class="graded",
        expected_params=hidden_params,
        expected_seed=SEED,
    )["packet_class"] == "graded"

    contract_path = copied / "participant" / "contract.json"
    contract = json.loads(contract_path.read_text())
    contract["opaque_build_identifier"] = f"world-{SEED}"
    contract_path.write_text(json.dumps(contract, indent=1, sort_keys=True) + "\n")
    _refresh_manifest_record(copied, "participant", "contract.json")
    with pytest.raises(ValueError, match="exposes the sealed packet seed"):
        validate_packet_directory(
            copied,
            expected_packet_class="graded",
            expected_params=hidden_params,
            expected_seed=SEED,
        )

    contract.pop("opaque_build_identifier")
    disclosed_key = f"world-{SEED}"
    contract[disclosed_key] = "opaque"
    contract_path.write_text(json.dumps(contract, indent=1, sort_keys=True) + "\n")
    _refresh_manifest_record(copied, "participant", "contract.json")
    with pytest.raises(ValueError, match="exposes the sealed packet seed"):
        validate_packet_directory(
            copied,
            expected_packet_class="graded",
            expected_params=hidden_params,
            expected_seed=SEED,
        )

    contract.pop(disclosed_key)
    contract_path.write_text(json.dumps(contract, indent=1, sort_keys=True) + "\n")
    _refresh_manifest_record(copied, "participant", "contract.json")
    geography_path = copied / "participant" / "geography.csv"
    original_lines = geography_path.read_text().splitlines()
    lines = list(original_lines)
    lines[0] += f",world-{SEED}"
    for index in range(1, len(lines)):
        lines[index] += ","
    geography_path.write_text("\n".join(lines) + "\n")
    _refresh_manifest_record(copied, "participant", "geography.csv")
    with pytest.raises(ValueError, match="exposes the sealed packet seed"):
        validate_packet_directory(
            copied,
            expected_packet_class="graded",
            expected_params=hidden_params,
            expected_seed=SEED,
        )

    lines = list(original_lines)
    lines[0] += ",opaque_build_identifier"
    lines[1] += f",world-{SEED}"
    for index in range(2, len(lines)):
        lines[index] += ","
    geography_path.write_text("\n".join(lines) + "\n")
    _refresh_manifest_record(copied, "participant", "geography.csv")
    with pytest.raises(ValueError, match="exposes the sealed packet seed"):
        validate_packet_directory(
            copied,
            expected_packet_class="graded",
            expected_params=hidden_params,
            expected_seed=SEED,
        )


@pytest.mark.parametrize("workers", [True, False, 0, -1, 1.5, "2"])
def test_packet_rejects_invalid_worker_counts_before_writing(tmp_path, workers):
    out = tmp_path / f"invalid-{workers!s}"
    with pytest.raises(ValueError, match="positive integer"):
        build_packet(SEED, out, PARAMS, workers=workers)
    assert not out.exists()


def test_packet_build_is_atomic_and_removes_its_failed_staging_directory(tmp_path,
                                                                         monkeypatch):
    import meridia.packet as packet_module

    def fail_in_staging(seed, out_dir, params, development, workers, cache_dir,
                        packet_class, build_provenance):
        del seed, params, development, workers, cache_dir, packet_class
        del build_provenance
        (out_dir / "partial.txt").write_text("incomplete")
        raise RuntimeError("simulated interrupted build")

    monkeypatch.setattr(packet_module, "_build_packet_into", fail_in_staging)
    out = tmp_path / "atomic"
    with pytest.raises(RuntimeError, match="interrupted"):
        build_packet(SEED, out, PARAMS)
    assert not out.exists()
    assert not (tmp_path / ".atomic.staging").exists()
    assert not (tmp_path / ".atomic.build-intent.json").exists()


def test_packet_records_locked_intent_before_build_materializes_data(tmp_path, monkeypatch):
    import meridia.packet as packet_module

    out = tmp_path / "bound"

    def inspect_intent(seed, out_dir, params, development, workers, cache_dir,
                       packet_class, build_provenance):
        del params, development, workers, cache_dir, packet_class
        intent_path = tmp_path / ".bound.build-intent.json"
        record = json.loads(intent_path.read_text())
        assert out_dir == tmp_path / ".bound.staging"
        assert not list(out_dir.iterdir())
        assert record["destination_name"] == "bound"
        assert record["staging_name"] == ".bound.staging"
        assert record["seed_commitment_sha256"] != str(seed)
        assert "seed" not in record
        assert record["provenance"] == build_provenance
        assert intent_path.stat().st_mode & 0o077 == 0
        raise RuntimeError("stop before world build")

    monkeypatch.setattr(packet_module, "_build_packet_into", inspect_intent)
    with pytest.raises(RuntimeError, match="before world build"):
        build_packet(SEED, out, PARAMS)
    assert not out.exists()
    assert not (tmp_path / ".bound.staging").exists()
    assert not (tmp_path / ".bound.build-intent.json").exists()


def test_packet_restart_recovers_only_the_exact_bound_staging(tmp_path, monkeypatch):
    import meridia.packet as packet_module

    out = tmp_path / "recover"
    intent = packet_module._packet_build_intent(
        out,
        seed=SEED,
        params=PARAMS,
        packet_class="qualification",
        development=False,
        graded_authorization=None,
    )
    intent_path = tmp_path / ".recover.build-intent.json"
    intent_path.write_bytes(packet_module._intent_bytes(intent))
    intent_path.chmod(0o600)
    staging = tmp_path / ".recover.staging"
    staging.mkdir()
    (staging / "stale-retained-data").write_text("remove")
    unrelated = tmp_path / ".another.staging"
    unrelated.mkdir()
    (unrelated / "keep").write_text("unrelated")

    def inspect_recovery(seed, out_dir, params, development, workers, cache_dir,
                         packet_class, build_provenance):
        del seed, params, development, workers, cache_dir, packet_class
        del build_provenance
        assert out_dir == staging
        assert not list(out_dir.iterdir())
        assert (unrelated / "keep").read_text() == "unrelated"
        raise RuntimeError("recovery inspected")

    monkeypatch.setattr(packet_module, "_build_packet_into", inspect_recovery)
    with pytest.raises(RuntimeError, match="recovery inspected"):
        build_packet(SEED, out, PARAMS)
    assert (unrelated / "keep").read_text() == "unrelated"
    assert not staging.exists()
    assert not intent_path.exists()


def test_packet_restart_preserves_unbound_staging_and_mismatched_intent(tmp_path):
    import meridia.packet as packet_module

    out = tmp_path / "mismatch"
    intent = packet_module._packet_build_intent(
        out,
        seed=SEED + 1,
        params=PARAMS,
        packet_class="qualification",
        development=False,
        graded_authorization=None,
    )
    intent_path = tmp_path / ".mismatch.build-intent.json"
    intent_path.write_bytes(packet_module._intent_bytes(intent))
    intent_path.chmod(0o600)
    staging = tmp_path / ".mismatch.staging"
    staging.mkdir()
    marker = staging / "preserve"
    marker.write_text("for diagnosis")

    with pytest.raises(ValueError, match="does not match this exact packet build"):
        build_packet(SEED, out, PARAMS)
    assert marker.read_text() == "for diagnosis"
    assert intent_path.exists()

    intent_path.unlink()
    with pytest.raises(ValueError, match="staging exists without the prior matching"):
        build_packet(SEED, out, PARAMS)
    assert marker.read_text() == "for diagnosis"
    assert not intent_path.exists()


def test_packet_resume_clears_only_matching_post_publication_intent(tmp_path):
    import meridia.packet as packet_module

    out = tmp_path / "published"
    out.mkdir()
    intent = packet_module._packet_build_intent(
        out,
        seed=SEED,
        params=PARAMS,
        packet_class="qualification",
        development=False,
        graded_authorization=None,
    )
    intent_path = tmp_path / ".published.build-intent.json"
    intent_path.write_bytes(packet_module._intent_bytes(intent))
    intent_path.chmod(0o600)

    assert packet_module._finalize_packet_build_intent(
        out,
        seed=SEED,
        params=PARAMS,
        packet_class="qualification",
        development=False,
    ) is True
    assert not intent_path.exists()
    assert packet_module._finalize_packet_build_intent(
        out,
        seed=SEED,
        params=PARAMS,
        packet_class="qualification",
        development=False,
    ) is False


def test_graded_packet_requires_final_authorization_before_staging(tmp_path):
    from meridia.sealing import V4PublicationAuthorization, V4WorldAuthorization

    class ForgedPublicationAuthorization(V4PublicationAuthorization):
        def confirm(self, *, seed, params):
            del seed, params
            return self.before

    hidden = PacketParams(**{**PARAMS.__dict__, "regime": "hidden"})
    out = tmp_path / "graded"
    with pytest.raises(ValueError, match="requires sealed authorization"):
        build_packet(SEED, out, hidden, packet_class="graded")
    with pytest.raises(ValueError, match="requires sealed authorization"):
        build_packet(
            SEED,
            out,
            hidden,
            packet_class="graded",
            graded_authorization=lambda: None,
        )
    forged = ForgedPublicationAuthorization(
        before=V4WorldAuthorization(seed=SEED, binding_sha256="a" * 64),
        index=0,
        seal_manifest_path=tmp_path / "seal.json",
        key_path=tmp_path / "key",
        bars_path=tmp_path / "bars.json",
        reserve_calibration_path=tmp_path / "reserve.json",
    )
    with pytest.raises(ValueError, match="requires sealed authorization"):
        build_packet(
            SEED,
            out,
            hidden,
            packet_class="graded",
            graded_authorization=forged,
        )
    assert not out.exists()


def test_failed_final_authorization_never_publishes_a_staged_packet(tmp_path, monkeypatch):
    import meridia.packet as packet_module
    from meridia.sealing import V4PublicationAuthorization, V4WorldAuthorization

    calls = []

    def fake_build(seed, out_dir, params, development, workers, cache_dir, packet_class,
                   build_provenance):
        del seed, params, development, workers, cache_dir, packet_class
        del build_provenance
        calls.append("build")
        (out_dir / "staged.txt").write_text("complete")

    def fake_validate(*args, **kwargs):
        calls.append("semantic-validation")
        return {"packet_class": "graded"}

    def reject(self, *, seed, params):
        del seed, params
        calls.append("authorization")
        if calls.count("authorization") == 2:
            raise RuntimeError("authorization drift")
        return self.before

    monkeypatch.setattr(packet_module, "_build_packet_into", fake_build)
    monkeypatch.setattr(packet_module, "validate_packet_directory", fake_validate)
    monkeypatch.setattr(V4PublicationAuthorization, "confirm", reject)
    out = tmp_path / "graded"
    hidden = PacketParams(**{**PARAMS.__dict__, "regime": "hidden"})
    authority = V4PublicationAuthorization(
        before=V4WorldAuthorization(seed=SEED, binding_sha256="a" * 64),
        index=0,
        seal_manifest_path=tmp_path / "seal.json",
        key_path=tmp_path / "key",
        bars_path=tmp_path / "bars.json",
        reserve_calibration_path=tmp_path / "reserve.json",
    )
    with pytest.raises(RuntimeError, match="authorization drift"):
        build_packet(
            SEED,
            out,
            hidden,
            packet_class="graded",
            graded_authorization=authority,
        )
    assert calls == [
        "authorization", "build", "semantic-validation", "authorization"
    ]
    assert not out.exists()
    assert not (tmp_path / ".graded.staging").exists()
    assert not (tmp_path / ".graded.build-intent.json").exists()


def test_staging_is_revalidated_after_authorization_before_publish(tmp_path, monkeypatch):
    import meridia.packet as packet_module
    from meridia.sealing import V4PublicationAuthorization, V4WorldAuthorization

    staging_path = None
    calls = []

    def fake_build(seed, out_dir, params, development, workers, cache_dir, packet_class,
                   build_provenance):
        nonlocal staging_path
        del seed, params, development, workers, cache_dir, packet_class
        del build_provenance
        staging_path = out_dir
        (out_dir / "content").write_text("validated")
        calls.append("build")

    def fake_validate(path, **kwargs):
        del kwargs
        calls.append("validate")
        return {"content": (path / "content").read_text()}

    def mutate(self, *, seed, params):
        del seed, params
        calls.append("authorize")
        if calls.count("authorize") == 2:
            (staging_path / "content").write_text("mutated")
        return self.before

    monkeypatch.setattr(packet_module, "_build_packet_into", fake_build)
    monkeypatch.setattr(packet_module, "validate_packet_directory", fake_validate)
    monkeypatch.setattr(V4PublicationAuthorization, "confirm", mutate)
    hidden = PacketParams(**{**PARAMS.__dict__, "regime": "hidden"})
    authority = V4PublicationAuthorization(
        before=V4WorldAuthorization(seed=SEED, binding_sha256="a" * 64),
        index=0,
        seal_manifest_path=tmp_path / "seal.json",
        key_path=tmp_path / "key",
        bars_path=tmp_path / "bars.json",
        reserve_calibration_path=tmp_path / "reserve.json",
    )
    out = tmp_path / "graded"
    with pytest.raises(RuntimeError, match="changed during final authorization"):
        build_packet(
            SEED,
            out,
            hidden,
            packet_class="graded",
            graded_authorization=authority,
        )
    assert calls == ["authorize", "build", "validate", "authorize", "validate"]
    assert not out.exists()


def test_atomic_publish_never_replaces_an_existing_output(tmp_path):
    staging = tmp_path / "staging"
    staging.mkdir()
    (staging / "new.txt").write_text("new")
    existing = tmp_path / "existing"
    existing.mkdir()
    with pytest.raises(FileExistsError, match="already exists"):
        _publish_staging_directory(staging, existing)
    assert list(existing.iterdir()) == []
    assert (staging / "new.txt").read_text() == "new"


def test_development_packet_ships_truth_and_hidden_does_not(tmp_path, packet):
    out, hidden = packet
    dev = build_packet(SEED, tmp_path / "dev", PARAMS, development=True)
    assert dev["packet_class"] == "development"
    assert "truth/truth_horizon.csv" in dev["participant"]
    assert dev["participant"]["truth/truth_horizon.csv"]["sha256"] == \
        hidden["retained"]["truth_horizon.csv"]["sha256"]
    public = {k: v for k, v in dev["participant"].items()
              if not k.startswith("truth/") and k != "contract.json"}
    assert public == {k: v for k, v in hidden["participant"].items() if k != "contract.json"}


def test_benchmark_series_is_shipped_with_its_own_bias(packet):
    import pandas as pd
    out, manifest = packet
    for label in ("preliminary", "revised"):
        assert f"sources/benchmark_{label}.csv" in manifest["participant"]
    bench = pd.read_csv(out / "participant" / "sources" / "benchmark_revised.csv")
    assert list(bench.columns) == ["item", "level", "unit", "value"]
    assert set(bench["item"]) == {"persons", "households", "children_under_16", "elders_65_plus"}
    assert set(bench["level"]) == {"nation", "state", "economic_band"}
    assert (bench["value"] % 100 == 0).all()
    truth = _read_truth(out / "retained" / "truth_revised.csv")
    world = json.loads((out / "retained" / "world.json").read_text())
    bias = world["benchmark_bias"]
    for k, item in enumerate(("persons", "households", "children_under_16", "elders_65_plus")):
        value = float(bench[(bench["item"] == item) & (bench["level"] == "nation")]["value"].iloc[0])
        exact = truth[(item, "nation", 0)]
        assert 0.02 <= abs(bias["nation"][k]) <= 0.07
        rounding = 60.0 / exact                                            # half of 100, with slack
        assert abs(np.log(value / exact) - bias["nation"][k]) < rounding + 1e-9
        assert abs(np.log(value / exact)) > 0.02 - rounding                # never the exact count
    assert world["regime"] == "development"
    assert set(world["source_params"]) >= {"population_coverage", "health_coverage", "county_error_rate", "register_income_scale"}
    contract = json.loads((out / "participant" / "contract.json").read_text())
    text = json.dumps(contract)
    assert "regime" not in text and "coverage" not in text
    # The mechanism families are published in form; no realized coefficient is.
    mechanism_record = world["mechanisms"]
    assert set(contract["mechanisms"]) >= {"axes", "public_envelope", "development_band",
                                           "declared_interactions", "development_design",
                                           "covariates", "families"}
    for name, value in mechanism_record["coefficients"].items():
        assert f"{value!r}" not in text, name
    assert "cell" not in contract["mechanisms"]


def test_hidden_regime_packet_is_retained_only(tmp_path):
    hidden_params = PacketParams(**{**PARAMS.__dict__, "regime": "hidden"})
    hidden = build_packet(SEED, tmp_path / "hidden", hidden_params, development=False,
                          packet_class="qualification")
    dev = build_packet(SEED, tmp_path / "dev", PARAMS, development=False)
    world_h = json.loads((tmp_path / "hidden" / "retained" / "world.json").read_text())
    world_d = json.loads((tmp_path / "dev" / "retained" / "world.json").read_text())
    assert hidden["packet_class"] == "qualification"
    assert world_h["packet_class"] == "qualification"
    assert world_h["regime"] == "hidden" and world_d["regime"] == "development"
    assert world_h["source_params"] != world_d["source_params"]
    assert world_h["character"] == world_d["character"]
    assert world_h["benchmark_bias"] == world_d["benchmark_bias"]
    # The hidden regime moves demographic mechanisms as well as reporting ones, so the
    # ledger and every file cut from it differ while the baseline microdata does not.
    assert world_h["mechanisms"]["design"]["regime"] == "hidden"
    assert world_h["mechanisms"]["design"]["cell"] == -1
    assert world_d["mechanisms"]["design"]["cell"] >= 0
    assert world_h["mechanisms"]["coefficients"] != world_d["mechanisms"]["coefficients"]
    # The survey instrument moves with the source rule as well: an evaluation world puts
    # two of its nine survey axes outside the band the open worlds are drawn from.
    assert len(world_h["survey_outside"]) == N_SURVEY_OUTSIDE_AXES
    assert world_d["survey_outside"] == []
    assert world_h["survey_params"] != world_d["survey_params"]
    for axis in world_h["survey_outside"]:
        low, high = SURVEY_BANDS[axis]
        assert not low <= world_h["survey_params"][axis] <= high
    assert hidden["retained"]["truth_revised.csv"] != dev["retained"]["truth_revised.csv"]
    assert hidden["participant"]["sources/population_revised.csv"] != dev["participant"]["sources/population_revised.csv"]
    assert hidden["participant"]["survey_revised.csv"] != dev["participant"]["survey_revised.csv"]
    for name, columns in participant_columns(tmp_path / "hidden").items():
        for column in columns:
            assert "regime" not in column and "seed" not in column, (name, column)
    with pytest.raises(ValueError, match="development"):
        build_packet(SEED, tmp_path / "bad", hidden_params, development=True)


def test_the_contract_publishes_the_survey_family_and_the_baseline_share(packet):
    """The instrument's bands and the baseline rule are public; the draw is not."""
    packet_dir, _ = packet
    contract = json.loads((packet_dir / "participant" / "contract.json").read_text())
    survey = contract["survey_family"]
    assert set(survey["bands"]) == set(SURVEY_BANDS)
    for name, (low, high) in SURVEY_BANDS.items():
        assert survey["bands"][name] == [low, high]
        envelope_low, envelope_high = SURVEY_ENVELOPE[name]
        assert survey["envelope"][name] == [envelope_low, envelope_high]
        assert envelope_low < low < high < envelope_high
    assert survey["n_outside_axes"] == N_SURVEY_OUTSIDE_AXES
    reserve = contract["reserve"]
    share = np.asarray(reserve["baseline_share"], dtype=np.float64)
    assert len(share) == int(contract["n_states"])
    assert share.sum() == pytest.approx(1.0, abs=1e-4)
    assert (share >= 0).all()
    assert "eligibility age" in reserve["baseline_rule"]

    # The realized instrument is retained, never published, and differs from the
    # dataclass defaults that every world shared before it was drawn.
    world = json.loads((packet_dir / "retained" / "world.json").read_text())
    drawn = world["survey_params"]
    default = asdict(SurveyParams())
    assert any(drawn[name] != default[name] for name in SURVEY_BANDS)
    text = (packet_dir / "participant" / "contract.json").read_text()
    for name in SURVEY_BANDS:
        assert f'"{name}": {drawn[name]}' not in text


def test_the_published_baseline_share_is_a_function_of_participant_files_only(packet):
    """Rebuild the contract's baseline share from the participant tree and nothing else.

    The share is the split the frozen practical baseline A_B spends the reserve total on,
    so a participant has to be able to reproduce it exactly, and it must carry no sealed
    regional composition. Both are the same statement: the value is a function of two
    participant files, and the register those files hold has its own coverage error.
    """
    import pandas as pd

    packet_dir, _ = packet
    participant = packet_dir / "participant"
    contract = json.loads((participant / "contract.json").read_text())
    reserve = contract["reserve"]
    rule = reserve["baseline_share_rule"]
    assert rule["file"] == "sources/population_revised.csv"
    assert rule["geography_file"] == "geography.csv"
    assert rule["as_of_tick"] == contract["ticks"]["revised"]
    assert rule["minimum_age"] == contract["obligation"]["eligibility_min_age"]
    assert rule["file"] in PARTICIPANT_PACKET_FILES
    assert rule["geography_file"] in PARTICIPANT_PACKET_FILES

    register = pd.read_csv(participant / rule["file"])
    geography = pd.read_csv(participant / rule["geography_file"])
    county_state = dict(zip(geography[rule["geography_county_column"]],
                            geography[rule["geography_state_column"]]))
    age = (rule["as_of_tick"] - register[rule["birth_tick_column"]]) // 12
    elders = register.loc[age >= rule["minimum_age"], rule["county_column"]]
    counts = np.zeros(int(contract["n_states"]), dtype=np.float64)
    for state, count in elders.map(county_state).value_counts().items():
        counts[int(state)] = float(count)
    assert counts.sum() > 0
    recomputed = np.round(counts / counts.sum(), rule["decimals"])
    published = np.asarray(reserve["baseline_share"], dtype=np.float64)
    assert recomputed.tolist() == published.tolist()

    # The register is a reported source, so its elder composition is not the sealed one.
    # A share that matched the retained counts would be publishing them.
    truth = pd.read_csv(packet_dir / "retained" / "truth_revised.csv")
    sealed_rows = truth[(truth["estimand"] == "elders_65_plus")
                        & (truth["level"] == "state")].sort_values("unit")
    sealed = sealed_rows["value"].to_numpy(dtype=np.float64)
    assert len(sealed) == len(published)
    assert not np.allclose(sealed / sealed.sum(), published, atol=1e-6)


# ------------------------------------ the mortality trend's only anchor is the file

TREND_WORLD = PacketParams(grid=(64, 80), n_settlements=5, n_states=3, total=40_000,
                           observed_months=120, preliminary_lag=6, horizon_months=12,
                           experience_years=5, experience_lag_months=12,
                           ensemble_members=4, design_cell=4)



def _experience_drift(rows) -> float:
    """Count-weighted log mortality drift, each band by sex by state cell centred first.

    The published statistic from the identifiability report. Cells are centred on their
    own count-weighted means because the bands sit orders of magnitude apart, so a slope
    taken across them would read the composition of the deaths rather than the trend.
    """
    keep = (rows["exposure"] > 0) & (rows["deaths"] > 0)
    label = np.asarray([f"{b}|{s}|{u}" for b, s, u in
                        zip(rows["age_band"], rows["sex"], rows["state"])])
    numerator = denominator = 0.0
    for name in np.unique(label[keep]):
        block = keep & (label == name)
        if block.sum() < 3:
            continue
        weight = rows["deaths"][block].astype(np.float64)
        year = rows["year"][block].astype(np.float64)
        year = year - np.average(year, weights=weight)
        rate = np.log(rows["deaths"][block] / rows["exposure"][block])
        rate = rate - np.average(rate, weights=weight)
        numerator += float((weight * year * rate).sum())
        denominator += float((weight * year ** 2).sum())
    return -numerator / denominator if denominator else float("nan")


def test_the_committed_world_runs_a_burn_in_before_the_published_experience_file():
    """The file is the mortality trend's only anchor, so where its window sits matters."""
    world = GRADING_WORLD
    first_year_starts = (world.observed_months - world.experience_lag_months
                         - 12 * world.experience_years)
    assert first_year_starts >= EXPERIENCE_BURN_IN_MONTHS
    assert EXPERIENCE_BURN_IN_MONTHS >= 36


def test_the_experience_file_reads_the_trend_once_the_ledger_has_settled():
    """A ledger's opening years carry a settling term a trend estimator reads as improvement.

    The frail die first and each band refills from below, so the death rate inside a cell
    falls for reasons that have nothing to do with the world's mortality axis. Measured
    over twelve small worlds, a file at ledger months 0 to 60 had a bias of +0.084 a year
    against a published band 0.058 wide; the same worlds read at months 48 to 108 had a
    bias of -0.013. This checks one of those worlds, so the committed window is a
    measured choice rather than a stated one.
    """
    built = build_world(1105, TREND_WORLD)
    truth = built["mechanisms"].design.intensity["mortality_improvement"]
    obligation = ObligationContract()
    early = _experience_drift(_experience_history(built, built["admin"], obligation, 5, 60))
    late = _experience_drift(_experience_history(built, built["admin"], obligation, 5, 12))
    assert abs(early - truth) > 0.05
    assert abs(late - truth) < 0.5 * abs(early - truth)


def test_the_benchmark_publishes_a_subgroup_count_on_the_economic_gradient(packet):
    """The completeness axis gets an anchor: a benchmark count for a defined subgroup.

    Register coverage rides the county economic gradient, and the covariate that reports
    that gradient is thinned by the same mechanism, so neither the register against the
    survey nor the register against the state benchmark tracked the axis. The benchmark
    now publishes the resident person count of each economic band of counties, with the
    band of every county in ``geography.csv``, so the register's shortfall against it is
    the gradient itself.
    """
    import pandas as pd
    out, _ = packet
    geography = pd.read_csv(out / "participant" / "geography.csv")
    assert "economic_band" in geography.columns
    assert sorted(set(geography["economic_band"])) == [0, 1, 2, 3]

    contract = json.loads((out / "participant" / "contract.json").read_text())
    block = contract["benchmark"]
    assert block["subgroup_level"] == "economic_band"
    assert block["subgroup_item"] == "persons"
    assert block["n_economic_bands"] == 4
    assert block["reference_tick"] == contract["ticks"]["revised"]
    assert block["bias_ranges"] == {
        "nation_magnitude": [0.02, 0.07],
        "state_sd": [0.03, 0.08],
        "economic_band_sd": [0.004, 0.015],
    }
    assert "payroll per resident adult" in block["subgroup_definition"]

    bench = pd.read_csv(out / "participant" / "sources" / "benchmark_revised.csv")
    subgroup = bench[bench["level"] == "economic_band"].sort_values("unit")
    assert list(subgroup["item"]) == ["persons"] * 4
    assert list(subgroup["unit"]) == [0, 1, 2, 3]
    assert (subgroup["value"] > 0).all()

    # The anchor works: register persons over benchmark persons falls or rises with the
    # band, and the slope is positive because the axis is positive on every world.
    population = pd.read_csv(out / "participant" / "sources" / "population_revised.csv")
    band_of_county = geography.set_index("county")["economic_band"]
    band = band_of_county.reindex(population["county"]).to_numpy()
    register = np.bincount(band[band >= 0].astype(np.int64), minlength=4).astype(float)
    published = subgroup["value"].to_numpy(dtype=float)
    coverage = np.log(register / published)
    slope = float(np.polyfit(np.arange(4.0), coverage, 1)[0])
    assert slope > 0.0


def test_the_shock_family_publishes_its_regional_loadings(packet):
    """A shock year is national; how hard it lands is not."""
    out, _ = packet
    contract = json.loads((out / "participant" / "contract.json").read_text())
    band = contract["shock_family"]["regional_loading_band"]
    assert band == [0.35, 1.80]
    assert "1 + L_r * (m - 1)" in contract["shock_family"]["regional_loading"]
    world = json.loads((out / "retained" / "world.json").read_text())
    loading = world["mechanisms"]["region_shock_loading"]
    assert len(loading) == PARAMS.n_states
    assert all(band[0] <= float(v) <= band[1] for v in loading)
    assert len(set(round(float(v), 9) for v in loading)) == len(loading)


def test_the_continuation_ensemble_is_cached_on_the_baseline_ledger(tmp_path,
                                                                    monkeypatch):
    """A rebuild that changes nothing upstream of the ledger does not pay for futures.

    The ensemble is what a packet costs at the committed size, and it is a function of
    the branch state, the shock law, the horizon and the obligation. None of those is
    downstream of a verifier or a bar, so refreezing does not have to rebuild them.
    """
    import meridia.packet as packet_module
    cache = tmp_path / "ensembles"
    first_dir = tmp_path / "first"
    first = build_packet(SEED, first_dir, PARAMS, development=False, cache_dir=cache)
    stored = sorted(cache.glob("*.npz"))
    assert len(stored) == 1
    assert len(stored[0].stem) == 64
    with np.load(stored[0], allow_pickle=False) as archive:
        assert str(archive["schema"].item()) == ENSEMBLE_CACHE_SCHEMA
        assert str(archive["key"].item()) == stored[0].stem
        assert int(archive["n_regions"]) == PARAMS.n_states
        assert str(archive["liability_sha256"].item()) == _structural_sha256(
            "liability", archive["liability"]
        )

    def refuse(*args, **kwargs):
        raise AssertionError("the cached ensemble was rebuilt")

    monkeypatch.setattr(packet_module, "continuation_liabilities", refuse)
    second_dir = tmp_path / "second"
    second = build_packet(SEED, second_dir, PARAMS, development=False, cache_dir=cache)
    assert second["retained"]["continuation_liabilities.npz"]["sha256"] == \
        first["retained"]["continuation_liabilities.npz"]["sha256"]
    assert second["participant"]["contract.json"]["sha256"] == \
        first["participant"]["contract.json"]["sha256"]

    # A world the cache has not seen is built, not read: the key covers the ledger.
    third_dir = tmp_path / "third"
    with pytest.raises(AssertionError, match="rebuilt"):
        build_packet(SEED + 1, third_dir, PARAMS, development=False, cache_dir=cache)


def test_cached_liabilities_validate_schema_key_shape_and_finiteness(tmp_path):
    cache = tmp_path / "cache"
    key = "a" * 64
    values = np.arange(24, dtype=np.float64).reshape(8, 3)
    shock = _cache_shock_evidence(len(values))
    _store_liability(cache, key, values, n_regions=3, shock_redraw_evidence=shock)
    cached = _cached_liability(cache, key, members=5, n_regions=3)
    assert cached is not None
    assert np.array_equal(cached[0], values[:5])
    assert cached[1]["member_count"] == 5
    assert _cached_liability(cache, key, members=9, n_regions=3) is None

    shock_json = json.dumps(
        shock, sort_keys=True, separators=(",", ":"), allow_nan=False
    )

    def payload(embedded_key, liability=values, regions=3):
        return {
            "schema": np.asarray(ENSEMBLE_CACHE_SCHEMA),
            "key": np.asarray(embedded_key),
            "n_regions": np.int64(regions),
            "liability_sha256": np.asarray(
                _structural_sha256("liability", np.asarray(liability, dtype=np.float64))
            ),
            "shock_redraw_evidence_json": np.asarray(shock_json),
            "shock_redraw_evidence_sha256": np.asarray(
                hashlib.sha256(shock_json.encode()).hexdigest()
            ),
            "liability": liability,
        }

    invalid = {
        "missing-schema": {
            "key": np.asarray("b" * 64), "n_regions": np.int64(3),
            "liability_sha256": np.asarray(_structural_sha256("liability", values)),
            "shock_redraw_evidence_json": np.asarray(shock_json),
            "shock_redraw_evidence_sha256": np.asarray(
                hashlib.sha256(shock_json.encode()).hexdigest()
            ),
            "liability": values,
        },
        "wrong-key": payload("0" * 64),
        "wrong-regions": payload("d" * 64, regions=2),
        "wrong-region-shape": payload("e" * 64, np.ones((8, 4))),
        "one-dimensional": payload("f" * 64, values.ravel()),
        "non-finite": payload(
            "g" * 64, np.where(values == values[-1, -1], np.nan, values)
        ),
        "negative": payload("h" * 64, np.where(values == 1, -1.0, values)),
    }
    for index, (name, payload) in enumerate(invalid.items()):
        bad_key = chr(ord("b") + index) * 64
        if name != "wrong-key":
            payload["key"] = np.asarray(bad_key)
        np.savez_compressed(cache / f"{bad_key}.npz", **payload)
        with pytest.raises(ValueError, match="ensemble cache"):
            _cached_liability(cache, bad_key, members=4, n_regions=3)

    tampered_key = "9" * 64
    _store_liability(
        cache, tampered_key, values, n_regions=3, shock_redraw_evidence=shock
    )
    with np.load(cache / f"{tampered_key}.npz", allow_pickle=False) as archive:
        tampered = {name: archive[name] for name in archive.files}
    tampered["liability"] = values + 1.0
    np.savez_compressed(cache / f"{tampered_key}.npz", **tampered)
    with pytest.raises(ValueError, match="does not match its digest"):
        _cached_liability(cache, tampered_key, members=4, n_regions=3)


def test_shock_redraw_evidence_rejects_removed_and_reused_member_schedules():
    evidence = _cache_shock_evidence(8)
    assert _validate_shock_redraw_evidence(evidence, expected_members=8) == evidence

    missing = json.loads(json.dumps(evidence))
    missing["member_schedules"].pop()
    with pytest.raises(ValueError, match="member schedules"):
        _validate_shock_redraw_evidence(missing, expected_members=8)

    reused = json.loads(json.dumps(evidence))
    reused["member_schedules"][1]["member"] = 0
    with pytest.raises(ValueError, match="member schedules"):
        _validate_shock_redraw_evidence(reused, expected_members=8)


def test_continuation_source_law_digest_binds_rng_shocks_and_implementation(monkeypatch):
    import meridia.packet as packet_module

    digest = continuation_source_law_digest()
    assert len(digest) == 64
    closure = set(continuation_source_modules())
    assert {"actuarial", "events", "projection", "demography", "mechanisms"} <= closure
    assert {"packet", "sources", "survey", "verify"}.isdisjoint(closure)
    with monkeypatch.context() as patch:
        patch.setattr(packet_module, "SHOCK_SUBSTREAM", packet_module.SHOCK_SUBSTREAM + 1)
        assert continuation_source_law_digest() != digest
    with monkeypatch.context() as patch:
        changed = {**packet_module.SHOCK_FAMILY, "test-shock": {"factor": (1.0, 2.0)}}
        patch.setattr(packet_module, "SHOCK_FAMILY", changed)
        assert continuation_source_law_digest() != digest


def test_continuation_source_law_digest_is_runtime_portable(monkeypatch):
    import meridia.packet as packet_module

    digest = continuation_source_law_digest()
    with monkeypatch.context() as patch:
        patch.setattr(packet_module.np, "__version__", "999.0-test")
        patch.setattr(packet_module.sys, "version_info", (9, 9, 9))
        assert continuation_source_law_digest() == digest


def test_continuation_source_closure_follows_relative_and_absolute_imports(tmp_path):
    package = tmp_path / "meridia"
    package.mkdir()
    (package / "root.py").write_text(
        "from .relative import VALUE\n"
        "import meridia.absolute\n"
        "def late():\n"
        "    from .nested import VALUE\n"
    )
    (package / "relative.py").write_text("from meridia.transitive import VALUE\n")
    (package / "absolute.py").write_text("VALUE = 1\n")
    (package / "nested.py").write_text("VALUE = 2\n")
    (package / "transitive.py").write_text("VALUE = 3\n")
    (package / "unrelated.py").write_text("VALUE = 4\n")

    closure = continuation_source_modules(package, roots=("root",))
    assert closure == ("absolute", "nested", "relative", "root", "transitive")


def test_continuation_source_digest_moves_with_a_transitive_dependency(tmp_path):
    import meridia.packet as packet_module

    source = Path(packet_module.__file__).resolve().parent
    copied = tmp_path / "meridia"
    shutil.copytree(source, copied, ignore=shutil.ignore_patterns("__pycache__"))
    original = continuation_source_law_digest(source)
    assert continuation_source_law_digest(copied) == original

    businesses = copied / "businesses.py"
    business_source = businesses.read_text()
    businesses.write_text(business_source + "\n# continuation-law change\n")
    assert continuation_source_law_digest(copied) != original

    businesses.write_text(business_source)
    verifier = copied / "verify.py"
    verifier.write_text(verifier.read_text() + "\n# downstream-only change\n")
    assert continuation_source_law_digest(copied) == original


def test_the_cache_key_moves_when_the_priced_world_moves(tmp_path, monkeypatch):
    """The digest covers the branch, the shock law, the horizon and the obligation."""
    from meridia.actuarial import ObligationContract, regions_from_admin
    from meridia.mechanisms import QUALIFYING_DIAGNOSIS_GROUPS

    built = build_world(SEED, PARAMS)
    region = regions_from_admin(built["admin"])
    obligation = ObligationContract(
        horizon_months=PARAMS.horizon_months,
        qualifying_diagnosis_groups=QUALIFYING_DIAGNOSIS_GROUPS)
    admin = built["admin"]
    key = baseline_ledger_digest(built["history"], obligation, PARAMS.horizon_months,
                                 region, admin)
    assert key == baseline_ledger_digest(built["history"], obligation,
                                         PARAMS.horizon_months, region, admin)
    assert key != baseline_ledger_digest(built["history"], obligation,
                                         PARAMS.horizon_months + 1, region, admin)
    dearer = ObligationContract(
        horizon_months=PARAMS.horizon_months,
        qualifying_diagnosis_groups=QUALIFYING_DIAGNOSIS_GROUPS,
        death_benefit=ObligationContract().death_benefit + 100.0)
    assert key != baseline_ledger_digest(built["history"], dearer,
                                         PARAMS.horizon_months, region, admin)
    other = build_world(SEED + 1, PARAMS)
    assert key != baseline_ledger_digest(other["history"], obligation,
                                         PARAMS.horizon_months,
                                         regions_from_admin(other["admin"]),
                                         other["admin"])

    context = built["history"]["branch"]["context"]
    encounter_rate = context["annual_encounter_rate"]
    context["annual_encounter_rate"] = encounter_rate + 0.01
    try:
        assert key != baseline_ledger_digest(
            built["history"], obligation, PARAMS.horizon_months, region, admin
        )
    finally:
        context["annual_encounter_rate"] = encounter_rate

    changed_admin = dict(admin)
    changed_admin["county"] = np.array(admin["county"], copy=True)
    changed_admin["county"].flat[0] = (changed_admin["county"].flat[0] + 1) \
        % int(admin["n_counties"])
    assert key != baseline_ledger_digest(
        built["history"], obligation, PARAMS.horizon_months, region, changed_admin
    )
    with monkeypatch.context() as patch:
        patch.setattr("meridia.packet.continuation_source_law_digest", lambda: "0" * 64)
        assert key != baseline_ledger_digest(built["history"], obligation,
                                             PARAMS.horizon_months, region, admin)
