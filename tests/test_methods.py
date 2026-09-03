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
from meridia.verify import (
    admin_from_packet,
    load_detailed,
    verify_release_projection_allocation,
    verify_submission,
)

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


def test_three_file_task_surface_passes_and_fails_closed(packet, submission, tmp_path):
    three_files = tmp_path / "three-files"
    shutil.copytree(submission, three_files)
    (three_files / "detailed.csv").unlink()
    report = verify_release_projection_allocation(packet, three_files)
    assert report["pass"], report["reasons"]
    assert "disclosure" not in report

    (three_files / "detailed.csv").write_text(
        "county,age_band,sex,count\n0,0-15,male,\n"
    )
    report = verify_release_projection_allocation(packet, three_files)
    assert not report["pass"]
    assert report["reasons"][0].startswith("file set: unexpected ['detailed.csv']")


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


def test_income_calibration_is_held_at_the_development_edge():
    from meridia.methods.common import apply_calibration
    from meridia.methods.design_based import _apply_calibration
    factors = {"dispersion_range": [0.89, 0.94],
               "mean_income_adults": {"intercept": -1.15, "slope": 1.41, "residual_sd": 0.02}}
    values = {("mean_income_adults", "nation", 0): 100.0}
    inside = apply_calibration(values, factors, 0.90)[("mean_income_adults", "nation", 0)]
    below = apply_calibration(values, factors, 0.63)[("mean_income_adults", "nation", 0)]
    edge = apply_calibration(values, factors, 0.89)[("mean_income_adults", "nation", 0)]
    assert below == edge and below < inside
    assert _apply_calibration(values, factors, 0.63) == _apply_calibration(values, factors, 0.89)
    # without a stored range the line is read as fitted
    bare = {"mean_income_adults": factors["mean_income_adults"]}
    key = ("mean_income_adults", "nation", 0)
    assert apply_calibration(values, bare, 0.63)[key] < apply_calibration(values, bare, 0.89)[key]


def test_fertility_estimate_reads_the_infant_years():
    import numpy as np
    import pandas as pd
    from meridia.methods.design_based import estimate_fertility
    rng = np.random.default_rng(3)
    tick = 24
    women = 20_000
    birth_women = tick - rng.integers(18 * 12, 45 * 12 + 11, size=women)
    infants = 0.5 * 2 * 0.08 * women          # two years at 0.08 per woman per year
    birth_infants = tick - rng.integers(0, 24, size=int(2 * 0.08 * women))
    frame = pd.DataFrame({"birth_tick": np.concatenate([birth_women, birth_infants]),
                          "sex": np.concatenate([np.ones(women, dtype=np.int8), rng.integers(0, 2, size=len(birth_infants)).astype(np.int8)])})
    est = estimate_fertility(frame, tick)
    assert est["fitted"] and abs(est["fertility_rate"] - 0.08) < 0.002
    thin = estimate_fertility(frame.iloc[:200], tick)
    assert not thin["fitted"]


def test_direct_county_variance_counts_the_units_that_landed_elsewhere():
    import numpy as np
    import pandas as pd
    from meridia.methods.design_based import _direct_county_persons
    # One stratum, ten units; county 0 holds two of them, county 1 the other eight.
    rows = []
    for psu in range(10):
        county = 0 if psu < 2 else 1
        for _ in range(3):
            rows.append({"stratum": 0, "psu": psu, "county": county, "weight": 100.0})
    frame = pd.DataFrame(rows)
    total, var = _direct_county_persons(frame, 3)
    assert total[0] == 600.0 and total[1] == 2400.0 and total[2] == 0.0
    # Design variance of the domain total with the eight zero units: n/(n-1) * sum (z - zbar)^2.
    z = np.array([300.0] * 2 + [0.0] * 8)
    expected = 10 / 9 * ((z - z.mean()) ** 2).sum()
    assert abs(var[0] - max(expected, total[0] ** 2 / 2)) < 1e-6
    # A county no unit landed in has no variance; the floor is one over the units.
    assert np.isnan(var[2])
    assert var[1] >= total[1] ** 2 / 8 - 1e-6
    _, design = _direct_county_persons(frame, 3, floor=False)
    assert abs(design[0] - expected) < 1e-6 and design[1] < total[1] ** 2 / 8


def test_raking_county_margin_follows_the_corrected_counts():
    import numpy as np
    import pandas as pd
    from meridia.methods.design_based import rake_to_register
    rng = np.random.default_rng(5)
    county_state = np.array([0, 0])
    # Register: county 1 is a small county flooded by misfiled records (raw share 0.5).
    register = pd.DataFrame({"county": np.repeat([0, 1], 500), "age": rng.integers(0, 80, 1000),
                             "sex": rng.integers(0, 2, 1000), "education": rng.integers(0, 3, 1000)})
    survey = pd.DataFrame({"county": np.repeat([0, 1], [900, 100]), "age": rng.integers(0, 80, 1000),
                           "sex": rng.integers(0, 2, 1000), "education": rng.integers(0, 3, 1000).astype(float),
                           "weight": 1.0, "psu": np.arange(1000) // 10, "stratum": 0, "household": np.arange(1000)})
    raw = rake_to_register(survey, register, county_state)
    corrected = rake_to_register(survey, register, county_state, county_persons=np.array([900.0, 100.0]))
    share_raw = raw.loc[raw["county"] == 1, "weight"].sum() / raw["weight"].sum()
    share_corrected = corrected.loc[corrected["county"] == 1, "weight"].sum() / corrected["weight"].sum()
    assert share_raw > 0.35 and abs(share_corrected - 0.10) < 0.02


def test_corroborated_income_ratios_ignore_the_misfiled_trickle():
    import numpy as np
    import pandas as pd
    from meridia.methods.design_based import corroborated_income, income_source_ratios
    rng = np.random.default_rng(11)
    n = 4000
    keys = pd.DataFrame({"given_code": np.arange(1, n + 1), "family_code": np.arange(1, n + 1) * 7,
                         "birth_tick": rng.integers(-800, -200, n), "sex": rng.integers(0, 2, n)})
    true_county = np.where(np.arange(n) < 3600, 0, 1)          # county 1: 400 true residents
    register = keys.assign(county=true_county, age=40, education=1, household_id=np.arange(n))
    # Income source: county 1's residents earn half of county 0's; a fifth of
    # county 0's records are misfiled into county 1.
    misfiled = (true_county == 0) & (rng.random(n) < 0.2)
    income = keys.assign(county=np.where(misfiled, 1, true_county), household_id=np.arange(n),
                         taxpayer_id=np.arange(n), record_id=np.arange(n),
                         employment_income_cents=np.where(true_county == 1, 2_000_000.0, 4_000_000.0))
    flags = corroborated_income(income, register)
    assert flags["corroborated"].sum() == (~misfiled).sum()
    raw = income_source_ratios(income, np.array([0, 0]), 30_000.0)
    confirmed = income_source_ratios(income, np.array([0, 0]), 30_000.0, register_frame=register)
    assert raw["mean_income_adults"][1] > 0.75
    pooled = (2880 * 4.0 + 400 * 2.0) / 3280          # corroborated records of the state
    assert abs(confirmed["mean_income_adults"][1] - 2.0 / pooled) < 0.02


def test_ratio_exponents_are_bounded_and_applied_inside_the_band():
    import numpy as np
    from meridia.methods.design_based import apply_ratio_exponents
    ratios = {"mean_income_adults": np.array([0.8, 1.0, 1.3, 2.0]),
              "median_household_income": np.array([0.8, 1.2]),
              "low_income_household_share": np.array([1.5])}
    out = apply_ratio_exponents(ratios, {"mean_income_adults": 2.0, "median_household_income": 1.0,
                                         "low_income_household_share": 1.66})
    assert np.allclose(out["mean_income_adults"], [0.64, 1.0, 1.69, 2.0])
    assert np.allclose(out["median_household_income"], [0.8, 1.2])
    assert abs(out["low_income_household_share"][0] - 1.5 ** 1.66) < 1e-9
    assert apply_ratio_exponents(ratios, None) is ratios
