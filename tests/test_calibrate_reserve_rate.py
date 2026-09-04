import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from scripts.calibrate_reserve_rate import (
    EXPERIENCE_COLUMNS,
    MANIFEST_SCHEMA,
    PENDING_BLOCKERS,
    TARGET_RULE,
    calibrate,
    main,
)


# The fixture world: two regions, a fixed lognormal continuation ensemble per world, and
# a public exposure large enough that the candidate totals sit above the mean liability.
# That is the regime the qualification worlds are in, where the skill denominator falls
# as the published total rises, so a margin threshold selects a lower rate.
FIXTURE_MEDIANS = (55.0, 110.0)
FIXTURE_SIGMAS = (0.35, 0.5)
FIXTURE_MEMBERS = 64
LINE_SCALES = {"A": 1.00, "B": 1.12, "C": 1.06}


def _write_packet(tmp_path: Path, world: str, exposure: float) -> Path:
    packet = tmp_path / "packets" / world
    participant = packet / "participant"
    retained = packet / "retained"
    participant.mkdir(parents=True, exist_ok=True)
    retained.mkdir(parents=True, exist_ok=True)
    index = int(world.rsplit("-", 1)[1])
    rng = np.random.default_rng(7919 + index)
    liability = rng.lognormal(
        mean=np.log(FIXTURE_MEDIANS), sigma=FIXTURE_SIGMAS,
        size=(FIXTURE_MEMBERS, len(FIXTURE_MEDIANS)))
    np.savez(retained / "continuation_liabilities.npz", liability=liability)
    contract = {
        "schema": "meridia.packet.v4",
        "n_states": 2,
        "reserve": {
            "rounding_unit": 10.0,
            "baseline_share": [0.5, 0.5],
            "weights": [1.0, 2.0],
        },
        "experience_history": {
            "file": "experience_history.csv",
            "columns": list(EXPERIENCE_COLUMNS),
        },
    }
    (participant / "contract.json").write_text(json.dumps(contract))
    with (participant / "experience_history.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=EXPERIENCE_COLUMNS)
        writer.writeheader()
        for year, value in ((1, exposure * 0.8), (2, exposure)):
            writer.writerow({
                "year": year, "age_band": "65-74", "sex": "female", "state": 0,
                "exposure": value, "deaths": 1, "qualifying_events": 2,
                "net_migration": 0,
            })
    return packet


def _entry(tmp_path: Path, line: str, world: str, exposure: float,
           means: tuple[float, float], tail_multiple: float = 1.5) -> dict:
    packet = _write_packet(tmp_path, world, exposure)
    submission = tmp_path / "submissions" / line / world
    submission.mkdir(parents=True, exist_ok=True)
    with (submission / "reserve.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("region", "liability_mean", "q95", "es95", "allocation"),
        )
        writer.writeheader()
        for region, mean in enumerate(means):
            writer.writerow({
                "region": region, "liability_mean": mean,
                "q95": mean * tail_multiple, "es95": mean * tail_multiple * 1.2,
                "allocation": mean,
            })
    return {
        "reference_line": line,
        "world": world,
        "evidence_id": f"{line}-{world}",
        "deterministic": True,
        "packet_dir": str(packet),
        "submission_dir": str(submission),
    }


def _evidence(tmp_path: Path, *, tail_multiple: float = 1.5,
              worlds: int = 2) -> list[dict]:
    entries = []
    for line, scale in LINE_SCALES.items():
        for index in range(worlds):
            entries.append(_entry(
                tmp_path, line, f"qual-{index}", 100.0 + index,
                (120.0 * scale, 240.0 * scale), tail_multiple=tail_multiple,
            ))
    return entries


def _calibrate(entries, **kwargs):
    options = {"expected_worlds": ("qual-0", "qual-1"), "rate_grid": 0.01}
    options.update(kwargs)
    return calibrate(entries, **options)


def test_candidate_is_the_largest_identified_rate_from_submitted_means(tmp_path):
    entries = _evidence(tmp_path)
    result = _calibrate(entries, margin_share=0.02)
    assert result["candidate"] is True
    assert result["target_rule"] == TARGET_RULE
    # Line B on qual-0 files the largest mean sum, 403.2 against an exposure of 100.
    assert result["rate_per_person_year"] == pytest.approx(4.04)
    assert result["binding_reference"] == "B-qual-0"
    binding = next(row for row in result["evidence"] if row["evidence_id"] == "B-qual-0")
    assert binding["submitted_liability_mean_sum"] == pytest.approx(403.2)
    assert binding["required_rate"] == pytest.approx(4.032)
    chosen = result["identification"]["chosen"]
    assert chosen["rate_per_person_year"] == pytest.approx(4.04)
    assert chosen["worst_margin_share"] >= 0.02
    assert set(chosen["worlds"]) == {"qual-0", "qual-1"}
    for reading in chosen["worlds"].values():
        assert reading["skill_denominator"] > 0.0
        assert reading["j_baseline"] > reading["j_oracle"]
    assert result["accepted"] is False
    assert result["blockers"] == list(PENDING_BLOCKERS)


def test_a_wider_margin_selects_a_lower_candidate_rate(tmp_path):
    entries = _evidence(tmp_path)
    result = _calibrate(entries, margin_share=0.05)
    assert result["candidate"] is True
    assert result["rate_per_person_year"] == pytest.approx(3.78)
    chosen = result["identification"]["chosen"]
    assert min(reading["margin_share"] for reading in chosen["worlds"].values()) >= 0.05
    candidates = result["identification"]["candidates"]
    assert [row["rate_per_person_year"] for row in candidates] \
        == sorted((row["rate_per_person_year"] for row in candidates), reverse=True)
    rejected = [row for row in candidates
                if row["rate_per_person_year"] > result["rate_per_person_year"]]
    assert rejected and all(row["identified"] is False for row in rejected)
    assert all(row["worst_margin_share"] < 0.05 for row in rejected)


def test_no_identified_candidate_fails_closed(tmp_path):
    entries = _evidence(tmp_path)
    result = _calibrate(entries, margin_share=0.9)
    assert result["candidate"] is False
    assert "keeps the reserve decision identified" in result["blockers"][0]
    assert "qual-" in result["blockers"][0]


def test_the_rate_no_longer_reads_the_submitted_tails(tmp_path):
    narrow = _calibrate(_evidence(tmp_path / "narrow", tail_multiple=1.5),
                        margin_share=0.02)
    wide = _calibrate(_evidence(tmp_path / "wide", tail_multiple=6.0),
                      margin_share=0.02)
    assert narrow["candidate"] is True and wide["candidate"] is True
    assert wide["rate_per_person_year"] == narrow["rate_per_person_year"]
    narrow_row = next(row for row in narrow["evidence"] if row["evidence_id"] == "B-qual-0")
    wide_row = next(row for row in wide["evidence"] if row["evidence_id"] == "B-qual-0")
    assert wide_row["submitted_q95_sum"] > 3.0 * narrow_row["submitted_q95_sum"]
    assert wide_row["required_rate"] == pytest.approx(narrow_row["required_rate"])


def test_a_missing_continuation_ensemble_fails_closed(tmp_path):
    entries = _evidence(tmp_path)
    (tmp_path / "packets" / "qual-1" / "retained"
     / "continuation_liabilities.npz").unlink()
    result = _calibrate(entries, margin_share=0.02)
    assert result["candidate"] is False
    assert "continuation_liabilities.npz" in result["blockers"][0]


def test_missing_pair_and_forbidden_path_fail_closed(tmp_path):
    entries = _evidence(tmp_path)
    missing = _calibrate(entries[:-1], margin_share=0.02)
    assert missing["candidate"] is False
    assert "exactly one entry" in missing["blockers"][0]

    bad = list(entries)
    bad[0] = dict(bad[0], packet_dir=str(tmp_path / "graded-packets" / "qual-0"))
    unsafe = _calibrate(bad, margin_share=0.02)
    assert unsafe["candidate"] is False
    assert "forbidden component" in unsafe["blockers"][0]


def test_cli_writes_the_candidate_record(tmp_path, capsys):
    entries = _evidence(tmp_path, worlds=6)
    manifest = tmp_path / "evidence.json"
    manifest.write_text(json.dumps({"schema": MANIFEST_SCHEMA, "entries": entries}))
    out = tmp_path / "candidate.json"
    assert main(["--evidence", str(manifest), "--out", str(out)]) == 0
    result = json.loads(out.read_text())
    assert result["candidate"] is True
    assert result["qualification_worlds"] == [f"qual-{index}" for index in range(6)]
    assert result["identification_margin_share"] == pytest.approx(0.01)
    assert set(result["identification"]["chosen"]["worlds"]) \
        == {f"qual-{index}" for index in range(6)}
    assert json.loads(capsys.readouterr().out)["rate_per_person_year"] \
        == result["rate_per_person_year"]


def test_script_help_runs_from_outside_the_repository(tmp_path):
    script = Path(__file__).resolve().parents[1] / "scripts" / "calibrate_reserve_rate.py"
    completed = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "--evidence" in completed.stdout and "--out" in completed.stdout


def test_the_freeze_contract_matches_what_this_calibrator_emits(tmp_path):
    """The freezer and the verifier accept exactly this candidate, field for field."""
    import importlib.util

    from meridia import verify

    from scripts.calibrate_reserve_rate import (IDENTIFICATION_MARGIN_SHARE,
                                                IDENTIFICATION_RULE, RATE_GRID, SCHEMA)

    path = Path(__file__).resolve().parents[1] / "scripts" / "freeze_v4_bars.py"
    spec = importlib.util.spec_from_file_location("freeze_v4_bars_contract", path)
    freeze = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(freeze)

    entries = _evidence(tmp_path, worlds=6)
    emitted = calibrate(entries)
    assert emitted["candidate"] is True

    assert set(emitted) == freeze.RESERVE_CALIBRATION_CANDIDATE_KEYS
    assert all(set(row) == freeze.RESERVE_CALIBRATION_EVIDENCE_KEYS
               for row in emitted["evidence"])
    assert set(emitted["identification"]) \
        == freeze.RESERVE_CALIBRATION_IDENTIFICATION_KEYS
    assert set(emitted["identification"]["chosen"]) \
        == freeze.RESERVE_CALIBRATION_CHOSEN_KEYS
    assert all(set(reading) == freeze.RESERVE_CALIBRATION_CHOSEN_WORLD_KEYS
               for reading in emitted["identification"]["chosen"]["worlds"].values())
    assert all(set(rung) == freeze.RESERVE_CALIBRATION_LADDER_KEYS
               for rung in emitted["identification"]["candidates"])

    assert SCHEMA == freeze.RESERVE_CALIBRATION_SCHEMA \
        == verify.RESERVE_CALIBRATION_SCHEMA
    assert TARGET_RULE == freeze.RESERVE_CALIBRATION_TARGET_RULE \
        == verify.RESERVE_CALIBRATION_TARGET_RULE
    assert IDENTIFICATION_RULE == freeze.RESERVE_CALIBRATION_IDENTIFICATION_RULE \
        == verify.RESERVE_CALIBRATION_IDENTIFICATION_RULE
    assert IDENTIFICATION_MARGIN_SHARE == freeze.RESERVE_IDENTIFICATION_MARGIN_SHARE \
        == verify.RESERVE_IDENTIFICATION_MARGIN_SHARE
    assert RATE_GRID == 1.0
    assert list(PENDING_BLOCKERS) == list(freeze.RESERVE_CALIBRATION_PENDING_BLOCKERS)
