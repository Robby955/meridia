import csv
import json
from pathlib import Path

import numpy as np
import pytest

from scripts.red_team_reserve_total import (
    EXPERIENCE_COLUMNS,
    MeasurementError,
    empirical_tail,
    main,
    run_measurement,
)


def _liability(q95: float, es95: float, regions: int = 2) -> np.ndarray:
    assert es95 > q95 > 2.0
    column = np.asarray(
        [q95 - 2.0] * 18 + [q95, 2.0 * es95 - q95], dtype=np.float64
    )
    return np.repeat(column[:, None], regions, axis=1)


def _write_world(root: Path, name: str, reserve: float, exposure: float) -> None:
    participant = root / name / "participant"
    retained = root / name / "retained"
    participant.mkdir(parents=True)
    retained.mkdir()
    contract = {
        "n_states": 2,
        "reserve": {
            "total": reserve,
            "total_rule": {
                "file": "experience_history.csv",
                "year": "maximum published year",
                "year_column": "year",
                "selected_year": 2,
                "exposure_column": "exposure",
                "aggregation": "sum exposure over every row in the selected year",
                "exposure_person_years": exposure,
                "rate_per_person_year": (reserve - 0.5) / exposure,
                "rounding": "up",
                "rounding_unit": 1.0,
            },
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
        for year, multiplier in ((1, 0.75), (2, 1.0)):
            writer.writerow(
                {
                    "year": year,
                    "age_band": "65-74",
                    "sex": "F",
                    "state": 0,
                    "exposure": exposure * multiplier,
                    "deaths": 1,
                    "qualifying_events": 2,
                    "net_migration": 0,
                }
            )
    q95 = 2.0 * reserve + 5.0
    es95 = 3.0 * reserve + 7.0
    np.savez_compressed(
        retained / "continuation_liabilities.npz",
        liability=_liability(q95, es95),
        realized_member=np.int64(0),
        weights=np.ones(2),
    )


def _packet_roots(tmp_path: Path) -> tuple[Path, Path]:
    development = tmp_path / "development"
    qualification = tmp_path / "qualification"
    for index in range(12):
        _write_world(
            development,
            f"dev-{index:02d}",
            reserve=11.0 + index,
            exposure=1_001.0 + index,
        )
    for index in range(6):
        _write_world(
            qualification,
            f"qual-{index}",
            reserve=31.0 + index,
            exposure=2_001.0 + index,
        )
    return development, qualification


def test_empirical_tail_uses_ceiling_rank_and_includes_all_ties():
    liability = np.asarray([0.0] * 17 + [5.0, 5.0, 10.0])[:, None]
    q95, es95 = empirical_tail(liability)
    assert q95.tolist() == [5.0]
    assert es95.tolist() == pytest.approx([20.0 / 3.0])


def test_development_fit_predicts_qualification_and_reports_public_inputs(tmp_path):
    development, qualification = _packet_roots(tmp_path)
    result = run_measurement(development, qualification)
    assert result["world_counts"] == {"development": 12, "qualification": 6, "total": 18}
    assert result["independent_unit"] == "world"
    predictive = result["qualification_predictive_regional_r2"]
    assert {key: predictive[key] for key in ("q95", "es95")} \
        == pytest.approx({"q95": 1.0, "es95": 1.0})
    assert result["qualification_incremental_regional_r2_over_region_means"] \
        == pytest.approx({"q95": 1.0, "es95": 1.0, "headline_max": 1.0})
    assert predictive["per_region"]["q95"] == pytest.approx([1.0, 1.0])
    assert result["descriptive_pooled_regional_r2"]["q95"] == pytest.approx(1.0)
    assert result["world_aggregate_tail_r2"]["qualification_predictive"][
        "es95"
    ] == pytest.approx(1.0)
    public = result["public_quantities"]
    assert public["development"][0] == {
        "world": "dev-00",
        "latest_year_total_exposure": 1001.0,
        "reserve_total": 11.0,
    }
    assert result["tail_definition"]["quantile_rank"].startswith("ceil")
    assert "world.json" not in " ".join(result["files_read_per_world"])
    assert result["reserve_total_public_rule_verified"] is True


def test_json_cli_output_contains_the_headline(tmp_path, capsys):
    development, qualification = _packet_roots(tmp_path)
    assert main(
        [
            "--development-root",
            str(development),
            "--qualification-root",
            str(qualification),
        ]
    ) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["qualification_incremental_regional_r2_over_region_means"][
        "headline_max"
    ] == pytest.approx(1.0)
    assert output["primary_measure"].startswith("qualification incremental")


def test_forbidden_path_component_is_rejected_before_packet_read(tmp_path):
    development, qualification = _packet_roots(tmp_path)
    unsafe = tmp_path / "hidden-packets"
    development.rename(unsafe)
    with pytest.raises(MeasurementError, match="forbidden component"):
        run_measurement(unsafe, qualification)


def test_world_roots_require_the_exact_frozen_names(tmp_path):
    development, qualification = _packet_roots(tmp_path)
    (development / "dev-11").rename(development / "dev-13")
    with pytest.raises(MeasurementError, match="must contain exactly"):
        run_measurement(development, qualification)


def test_nonfinite_liability_fails_closed(tmp_path):
    development, qualification = _packet_roots(tmp_path)
    path = development / "dev-00" / "retained" / "continuation_liabilities.npz"
    liability = _liability(30.0, 40.0)
    liability[0, 0] = np.nan
    np.savez_compressed(path, liability=liability)
    with pytest.raises(MeasurementError, match="nonnegative and finite"):
        run_measurement(development, qualification)


def test_reserve_total_must_recompute_from_the_public_rule(tmp_path):
    development, qualification = _packet_roots(tmp_path)
    path = development / "dev-00" / "participant" / "contract.json"
    contract = json.loads(path.read_text())
    contract["reserve"]["total"] += 1.0
    path.write_text(json.dumps(contract))
    with pytest.raises(MeasurementError, match="does not follow its public rule"):
        run_measurement(development, qualification)
