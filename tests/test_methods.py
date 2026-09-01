"""Reference method A runs from participant files alone and clears the hard gates."""

import json
import shutil
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from meridia.methods import design_based
from meridia.packet import PacketParams, build_packet
from meridia.release import AGE_BAND_LABELS, SEX_LABELS
from meridia.verify import admin_from_packet, load_detailed, verify_submission

SEED = 31337
PARAMS = PacketParams(
    grid=(72, 96),
    n_settlements=6,
    n_states=2,
    observed_months=6,
    preliminary_lag=3,
    horizon_months=12,
    total=40_000,
)


@pytest.fixture(scope="module")
def packet(tmp_path_factory):
    out = tmp_path_factory.mktemp("method") / "dev"
    build_packet(SEED, out, PARAMS, development=False)
    return out


@pytest.fixture(scope="module")
def submission(packet, tmp_path_factory):
    # The method sees a directory holding only the participant side.
    blind = tmp_path_factory.mktemp("blind") / "packet"
    blind.mkdir()
    shutil.copytree(packet / "participant", blind / "participant")
    out = tmp_path_factory.mktemp("sub") / "A"
    design_based.run(blind, out, design_based.MethodParams(bootstrap_replicates=25))
    return out


def test_method_a_writes_all_four_files(submission):
    for name in ("release.csv", "projection.csv", "detailed.csv", "allocation.csv"):
        assert (submission / name).exists()
    release = pd.read_csv(submission / "release.csv")
    assert (release["lower"] <= release["estimate"]).all() and (
        release["estimate"] <= release["upper"]
    ).all()


def test_method_a_clears_hard_gates_and_is_close_on_persons(packet, submission):
    report = verify_submission(packet, submission)
    assert report["schema_errors"] == [] and report["additivity_errors"] == []
    assert report["disclosure"]["pass"] and report["allocation"]["feasible"]
    assert report["pass"], report["reasons"]
    assert report["metrics"]["persons/nation"]["worst_error"] < 0.05
    assert report["metrics"]["persons/county"]["worst_error"] < 0.5
    assert report["allocation"]["regret"] < 0.15


def test_method_a_is_deterministic(packet, tmp_path):
    first = design_based.run(
        packet, tmp_path / "a", design_based.MethodParams(bootstrap_replicates=10)
    )
    second = design_based.run(
        packet, tmp_path / "b", design_based.MethodParams(bootstrap_replicates=10)
    )
    assert (
        first["release"] == second["release"]
        and first["projection"] == second["projection"]
    )


def test_verifier_rejects_extra_files_and_audits_published_totals(
    packet, submission, tmp_path
):
    extra = tmp_path / "extra"
    shutil.copytree(submission, extra)
    (extra / "notes.csv").write_text("anything\n")
    report = verify_submission(packet, extra)
    assert not report["pass"] and report["reasons"][0].startswith("file set")

    # Make one protected cell in a packet copy, suppress it in an otherwise exact
    # detailed table, and publish the corresponding county-age total. This isolates
    # the verifier's recoverability branch even when the generated packet has no
    # naturally protected cells.
    protected_packet = tmp_path / "protected-packet"
    shutil.copytree(packet, protected_packet)
    contract = json.loads(
        (protected_packet / "participant" / "contract.json").read_text()
    )
    threshold = int(contract["disclosure_threshold"])
    truth_path = protected_packet / "retained" / "detailed_revised.csv"
    truth = pd.read_csv(truth_path)
    truth.loc[(truth["count"] > 0) & (truth["count"] < threshold), "count"] = threshold
    target = truth.index[0]
    truth.loc[target, "count"] = max(1, threshold - 1)
    truth.to_csv(truth_path, index=False)

    leaky = tmp_path / "leaky"
    shutil.copytree(submission, leaky)
    published = truth.copy()
    published["count"] = published["count"].astype("Float64")
    published.loc[target, "count"] = pd.NA
    published.to_csv(leaky / "detailed.csv", index=False)

    target_row = truth.loc[target]
    same_line = (truth["county"] == target_row["county"]) & (
        truth["age_band"] == target_row["age_band"]
    )
    pd.DataFrame(
        [
            {
                "kind": "county_age",
                "county": int(target_row["county"]),
                "age_band": target_row["age_band"],
                "sex": "",
                "count": float(truth.loc[same_line, "count"].sum()),
            }
        ]
    ).to_csv(leaky / "totals.csv", index=False)

    expected_cell = (
        int(target_row["county"]),
        AGE_BAND_LABELS.index(str(target_row["age_band"])),
        SEX_LABELS.index(str(target_row["sex"])),
    )
    report = verify_submission(protected_packet, leaky)
    assert not report["pass"]
    assert report["reasons"] == [
        "disclosure: protected cell published, recoverable, or inconsistent"
    ]
    assert report["disclosure"] == {
        "pass": False,
        "n_protected": 1,
        "n_suppressed": 1,
        "published_protected": [],
        "recoverable": [expected_cell],
    }


def test_verifier_audits_generated_packet_totals(packet, submission, tmp_path):
    # The generated-packet check remains useful for coverage of the full totals schema.
    admin = admin_from_packet(packet)
    truth = load_detailed(
        packet / "retained" / "detailed_revised.csv", admin["n_counties"]
    )
    leaky = tmp_path / "leaky"
    shutil.copytree(submission, leaky)
    rows = []
    for c in range(admin["n_counties"]):
        for b, band in enumerate(("0-15", "16-24", "25-44", "45-64", "65+")):
            rows.append(
                {
                    "kind": "county_age",
                    "county": c,
                    "age_band": band,
                    "sex": "",
                    "count": float(truth[c, b, :].sum()),
                }
            )
        for s_, sex in enumerate(("male", "female")):
            rows.append(
                {
                    "kind": "county_sex",
                    "county": c,
                    "age_band": "",
                    "sex": sex,
                    "count": float(truth[c, :, s_].sum()),
                }
            )
    pd.DataFrame(rows).to_csv(leaky / "totals.csv", index=False)
    report = verify_submission(packet, leaky)
    if (
        report["disclosure"]["n_suppressed"] > 0
        and report["disclosure"]["n_protected"] > 0
    ):
        assert not report["disclosure"]["pass"]
