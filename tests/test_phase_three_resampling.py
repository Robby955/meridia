import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from meridia.methods import resampling


def _participant_packet(path: Path) -> Path:
    participant = path / "participant"
    (participant / "sources").mkdir(parents=True)
    survey = pd.DataFrame(
        {
            "household": np.arange(16),
            "county": np.repeat([0, 1], 8),
            "psu": np.repeat(np.arange(8), 2),
            "psu_sampled_households": 2,
            "stratum": np.repeat([0, 1], 8),
            "design_weight": np.linspace(10.0, 25.0, 16),
            "age": np.arange(30, 46),
            "sex": np.tile([0, 1], 8),
            "education": 2,
            "income": np.linspace(20_000.0, 50_000.0, 16),
            "recent_hospitalization": np.tile([0, 0, 0, 1], 4),
        }
    )
    survey.to_csv(participant / "survey_preliminary.csv", index=False)
    survey.to_csv(participant / "survey_revised.csv", index=False)
    experience = pd.DataFrame(
        {
            "year": [1, 1, 2, 2, 3, 3],
            "age_band": ["65-74", "75-84"] * 3,
            "sex": ["female", "male"] * 3,
            "state": [0, 1] * 3,
            "exposure": [1000.0, 800.0, 990.0, 790.0, 980.0, 780.0],
            "deaths": [20, 30, 22, 33, 24, 36],
            "qualifying_events": [40, 45, 42, 49, 46, 52],
            "net_migration": [2, -2, 1, -1, 0, 0],
        }
    )
    experience.to_csv(participant / "experience_history.csv", index=False)
    (participant / "sources" / "population_revised.csv").write_text(
        "person_id,county\n1,0\n"
    )
    (participant / "contract.json").write_text(
        json.dumps(
            {"experience_history": {"file": "experience_history.csv"}},
            sort_keys=True,
        )
        + "\n"
    )
    return path


def test_rao_wu_resample_uses_rescaled_psu_weight_multipliers():
    frame = pd.DataFrame(
        {
            "stratum": np.repeat([0, 1], 8),
            "psu": np.repeat(np.arange(8), 2),
            "design_weight": np.ones(16),
        }
    )
    sampled, evidence = resampling.rao_wu_resample(
        frame, np.random.default_rng(771)
    )
    ratio = sampled["design_weight"].to_numpy()
    for stratum in (0, 1):
        psus = frame.loc[frame["stratum"] == stratum, "psu"].unique()
        factors = [ratio[np.flatnonzero(frame["psu"].to_numpy() == psu)[0]] for psu in psus]
        assert np.mean(factors) == pytest.approx(1.0)
        assert all(factor in {0.0, 4.0 / 3.0, 8.0 / 3.0, 4.0} for factor in factors)
    assert evidence["design"].startswith("Rao-Wu")
    assert evidence["sampled_psus"] == 8


def test_paired_outer_resamples_are_actual_shared_and_digest_bound(tmp_path):
    packet = _participant_packet(tmp_path / "packet")
    original_survey = pd.read_csv(packet / "participant" / "survey_revised.csv")
    original_experience = pd.read_csv(
        packet / "participant" / "experience_history.csv"
    )
    root = tmp_path / "outer"
    manifest = resampling.materialize_paired_outer_resamples(
        packet, root, replicates=3, seed=9917
    )

    pd.testing.assert_frame_equal(
        pd.read_csv(packet / "participant" / "survey_revised.csv"),
        original_survey,
    )
    pd.testing.assert_frame_equal(
        pd.read_csv(packet / "participant" / "experience_history.csv"),
        original_experience,
    )
    assert manifest["schema"] == resampling.OUTER_RESAMPLE_SCHEMA
    assert manifest["reference_lines"] == ["A", "B", "C"]
    assert manifest["method_seeds_fixed_across_outer_resamples"] is True
    inputs = resampling.paired_reference_inputs(root)
    assert len(inputs) == 3
    assert all(row["method_seeds"] == resampling.REFERENCE_METHOD_SEEDS for row in inputs)
    assert len({row["participant_digest_sha256"] for row in inputs}) == 3

    survey_changed = []
    experience_changed = []
    for row in inputs:
        participant = row["packet"] / "participant"
        survey = pd.read_csv(participant / "survey_revised.csv")
        experience = pd.read_csv(participant / "experience_history.csv")
        survey_changed.append(
            not np.array_equal(
                survey["design_weight"].to_numpy(),
                original_survey["design_weight"].to_numpy(),
            )
        )
        experience_changed.append(
            not np.array_equal(
                experience[list(resampling.EXPERIENCE_COUNT_COLUMNS)].to_numpy(),
                original_experience[
                    list(resampling.EXPERIENCE_COUNT_COLUMNS)
                ].to_numpy(),
            )
        )
        assert np.array_equal(
            experience["exposure"].to_numpy(),
            original_experience["exposure"].to_numpy(),
        )
        assert np.array_equal(
            experience["net_migration"].to_numpy(),
            original_experience["net_migration"].to_numpy(),
        )
    assert all(survey_changed)
    assert all(experience_changed)

    changed = inputs[0]["packet"] / "participant" / "survey_revised.csv"
    changed.write_text(changed.read_text() + "\n")
    with pytest.raises(ValueError, match="resampling bytes changed"):
        resampling.verify_paired_outer_resamples(root)


def test_outer_resamples_replay_exactly_and_exclude_oracle_diagnostics(tmp_path):
    packet = _participant_packet(tmp_path / "packet")
    first = resampling.materialize_paired_outer_resamples(
        packet, tmp_path / "first", replicates=2, seed=1234
    )
    second = resampling.materialize_paired_outer_resamples(
        packet, tmp_path / "second", replicates=2, seed=1234
    )
    assert [row["participant_digest_sha256"] for row in first["resamples"]] == [
        row["participant_digest_sha256"] for row in second["resamples"]
    ]
    assert set(resampling.ORACLE_DIAGNOSTICS).isdisjoint(
        first["reference_lines"]
    )
    assert first["oracle_diagnostics"] == {
        "included": False,
        "names": list(resampling.ORACLE_DIAGNOSTICS),
        "reason": "development-only oracle diagnostics are not reference lines",
    }
