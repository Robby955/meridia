import csv
import json
import subprocess
import sys
from pathlib import Path

from scripts.calibrate_reserve_rate import (
    EXPERIENCE_COLUMNS,
    MANIFEST_SCHEMA,
    calibrate,
    main,
)


def _entry(tmp_path: Path, line: str, world: str, exposure: float,
           q95: tuple[float, float], es95: tuple[float, float]) -> dict:
    packet = tmp_path / "packets" / world
    participant = packet / "participant"
    participant.mkdir(parents=True, exist_ok=True)
    contract = {
        "schema": "meridia.packet.v4",
        "n_states": 2,
        "reserve": {"rounding_unit": 10.0},
        "experience_history": {
            "file": "experience_history.csv",
            "columns": list(EXPERIENCE_COLUMNS),
        },
    }
    (participant / "contract.json").write_text(json.dumps(contract))
    with (participant / "experience_history.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=EXPERIENCE_COLUMNS)
        writer.writeheader()
        writer.writerow({
            "year": 1, "age_band": "65-74", "sex": "female", "state": 0,
            "exposure": exposure * 0.8, "deaths": 1, "qualifying_events": 2,
            "net_migration": 0,
        })
        writer.writerow({
            "year": 2, "age_band": "65-74", "sex": "female", "state": 0,
            "exposure": exposure, "deaths": 1, "qualifying_events": 2,
            "net_migration": 0,
        })
    submission = tmp_path / "submissions" / line / world
    submission.mkdir(parents=True)
    with (submission / "reserve.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("region", "liability_mean", "q95", "es95", "allocation"),
        )
        writer.writeheader()
        for region in range(2):
            writer.writerow({
                "region": region, "liability_mean": q95[region] - 1.0,
                "q95": q95[region], "es95": es95[region],
                "allocation": q95[region] + 1.0,
            })
    return {
        "reference_line": line,
        "world": world,
        "evidence_id": f"{line}-{world}",
        "deterministic": True,
        "packet_dir": str(packet),
        "submission_dir": str(submission),
    }


def _evidence(tmp_path: Path) -> list[dict]:
    entries = []
    for line, bump in (("A", 0.0), ("B", 10.0), ("C", 5.0)):
        for index in range(2):
            entries.append(_entry(
                tmp_path, line, f"qual-{index}", 100.0 + index,
                (100.0 + bump, 200.0 + bump),
                (120.0 + bump, 240.0 + bump),
            ))
    return entries


def test_candidate_is_the_smallest_registered_rate_covering_every_reference(tmp_path):
    entries = _evidence(tmp_path)
    result = calibrate(entries, expected_worlds=("qual-0", "qual-1"), rate_grid=1.0)
    assert result["candidate"] is True
    # Line B, qual-0 binds: (320 + .25 * 60) / 100 = 3.35, rounded to the rate grid.
    assert result["rate_per_person_year"] == 4.0
    assert min(row["candidate_margin"] for row in result["evidence"]) >= 0.0
    assert result["accepted"] is False
    assert len(result["blockers"]) == 3


def test_missing_pair_and_forbidden_path_fail_closed(tmp_path):
    entries = _evidence(tmp_path)
    missing = calibrate(entries[:-1], expected_worlds=("qual-0", "qual-1"))
    assert missing["candidate"] is False
    assert "exactly one entry" in missing["blockers"][0]

    bad = list(entries)
    bad[0] = dict(bad[0], packet_dir=str(tmp_path / "graded-packets" / "qual-0"))
    unsafe = calibrate(bad, expected_worlds=("qual-0", "qual-1"))
    assert unsafe["candidate"] is False
    assert "forbidden component" in unsafe["blockers"][0]


def test_cli_writes_the_candidate_record(tmp_path, capsys):
    entries = []
    for line in ("A", "B", "C"):
        for index in range(6):
            entries.append(_entry(
                tmp_path, line, f"qual-{index}", 100.0 + index,
                (100.0, 200.0), (120.0, 240.0),
            ))
    manifest = tmp_path / "evidence.json"
    manifest.write_text(json.dumps({"schema": MANIFEST_SCHEMA, "entries": entries}))
    out = tmp_path / "candidate.json"
    assert main(["--evidence", str(manifest), "--out", str(out)]) == 0
    result = json.loads(out.read_text())
    assert result["candidate"] is True
    assert result["qualification_worlds"] == [f"qual-{index}" for index in range(6)]
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
