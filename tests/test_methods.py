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
    verify_submission,
)

SEED = 31337
PARAMS = PacketParams(
    grid=(72, 96),
    n_settlements=6,
    n_states=2,
    observed_months=24,
    experience_years=1,
    preliminary_lag=3,
    horizon_months=12,
    total=40_000,
    ensemble_members=32,
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
    for name in ("release.csv", "projection.csv", "detailed.csv", "reserve.csv"):
        assert (submission / name).exists()
    release = pd.read_csv(submission / "release.csv")
    assert (release["lower"] <= release["estimate"]).all() and (
        release["estimate"] <= release["upper"]
    ).all()


def test_the_four_file_surface_fails_closed_on_the_file_set(packet, submission, tmp_path):
    # Version four scores four files. A short submission and a long one both fail on the
    # file set, before any truth is read.
    short = tmp_path / "three-files"
    shutil.copytree(submission, short)
    (short / "detailed.csv").unlink()
    report = verify_submission(packet, short)
    assert not report["pass"]
    assert report["reasons"][0].startswith("file set: unexpected [], missing")

    long = tmp_path / "five-files"
    shutil.copytree(submission, long)
    (long / "allocation.csv").write_text("county,allocation\n0,1\n")
    report = verify_submission(packet, long)
    assert not report["pass"]
    assert report["reasons"][0].startswith("file set: unexpected ['allocation.csv']")


def test_method_a_clears_hard_gates_and_is_close_on_persons(packet, submission):
    report = verify_submission(packet, submission)
    assert report["schema_errors"] == [] and report["additivity_errors"] == []
    assert report["rate_errors"] == [] and report["reserve_errors"] == []
    assert report["disclosure"]["pass"] and report["reserve"]["feasible"]
    assert report["metrics"]["persons/nation"]["worst_error"] < 0.05
    assert report["metrics"]["persons/county"]["worst_error"] < 0.5
    # The tail and reserve ceilings are placeholders until they are frozen on
    # qualification worlds, so this world gates only the version-three families.
    families = {reason.split(":")[0] for reason in report["reasons"]}
    assert families <= {"tail", "reserve", "exposure", "rate", "coverage"}, report["reasons"]


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
    assert "disclosure: protected cell published, recoverable, or inconsistent" \
        in report["reasons"]
    audit = report["disclosure"]
    assert {k: audit[k] for k in ("pass", "n_protected", "n_suppressed",
                                 "published_protected", "recoverable")} == {
        "pass": False,
        "n_protected": 1,
        "n_suppressed": 1,
        "published_protected": [],
        "recoverable": [expected_cell],
    }
    # The audit also carries the utility share the disclosure gate reads: this table
    # suppresses one protected cell and publishes everything a rule would release.
    assert audit["n_published_releasable"] == audit["n_releasable"]
    assert audit["utility"] == 1.0


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


# ------------------------------------------------------- the shared actuarial layer
#
# The estimators the version-four reference adds, each on a case where the right answer
# is known by construction. They are unit tests on synthetic input, so they say what the
# estimator does rather than what one world happened to produce.

def _experience(n_years=5, n_states=4, drift=-0.03, level=0.01, shock_year=None,
                shock=2.0, slope=0.10, exposure=20_000.0):
    """A five-year experience file with a planted drift, slope and optional shock."""
    import numpy as np
    from meridia.methods.actuarial_reference import ACTUARIAL_AGE_BANDS
    n_bands = len(ACTUARIAL_AGE_BANDS)
    midpoint = np.array([0.5 * (lo + min(hi, 100)) for lo, hi in ACTUARIAL_AGE_BANDS])
    base = level * np.exp(slope * (midpoint - 45.0))
    out = {k: np.zeros((n_years, n_states, n_bands, 2))
           for k in ("exposure", "deaths", "qualifying_events", "net_migration")}
    rng = np.random.default_rng(11)
    for y in range(n_years):
        factor = np.exp(drift * y) * (shock if y == shock_year else 1.0)
        for s in range(n_states):
            for x in range(2):
                out["exposure"][y, s, :, x] = exposure
                out["deaths"][y, s, :, x] = rng.poisson(base * factor * exposure)
                out["qualifying_events"][y, s, :, x] = rng.poisson(
                    4.0 * base * factor * exposure)
                out["net_migration"][y, s, :, x] = 20.0 * np.exp(-0.05 * midpoint)
    out["years"] = np.arange(1, n_years + 1)
    return out


def test_the_published_shock_family_is_read_and_its_multipliers_move_together():
    import numpy as np
    from meridia.methods import actuarial_reference as AR
    contract = {"shock_family": {"annual_rate": 0.2, "kinds": {
        "mortality_spike": {"mortality_multiplier": [1.5, 3.0],
                            "admission_multiplier": [1.4, 2.6]},
        "baby_bust": {"fertility_multiplier": [0.45, 0.75]},
        "migration_wave": {"leave_home_multiplier": [1.8, 3.0]}}}}
    family = AR.read_shock_family(contract)
    assert family["annual_rate"] == 0.2 and len(family["kinds"]) == 3
    assert AR.shock_range_for(family, "incidence") == (1.4, 2.6)
    draw = AR.draw_shock_year(np.random.default_rng(3), 4000, family, 0.0, (1.0, 1.0))
    hit = draw["mortality"] > 1.0
    # One kind a year out of three, at the published annual rate.
    assert 0.04 < hit.mean() < 0.10
    # An epidemic year raises deaths and admissions on the same draw, never one alone.
    assert np.array_equal(hit, draw["incidence"] > 1.0)
    position = (draw["mortality"][hit] - 1.5) / 1.5
    assert np.allclose(draw["incidence"][hit], 1.4 + position * 1.2)
    assert np.all(draw["fertility"][hit] == 1.0)
    ordinary = ~hit
    assert np.all(draw["mortality"][ordinary] == 1.0)


def test_the_shock_loading_a_family_carries_is_what_an_average_year_already_holds():
    from meridia.methods import actuarial_reference as AR
    family = {"annual_rate": 0.3, "kinds": [{"mortality": (1.5, 3.0)},
                                            {"fertility": (0.5, 0.7)},
                                            {"migration": (2.0, 2.0)}]}
    assert abs(AR.expected_shock_loading(family, "mortality") - 0.1 * 1.25) < 1e-12
    assert abs(AR.expected_shock_loading(family, "fertility") + 0.1 * 0.4) < 1e-12
    assert AR.expected_shock_loading(None, "mortality") == 0.0


def test_the_drift_estimator_is_not_dragged_by_one_published_shock_year():
    from meridia.methods import actuarial_reference as AR
    family = {"annual_rate": 0.2, "kinds": [{"mortality": (1.5, 3.0)}]}
    clean = _experience(drift=-0.03)
    fit = AR.estimate_improvement(clean["exposure"], clean["deaths"], shock_family=family)
    assert abs(fit["drift"] + 0.03) < 0.01 and fit["fitted"]
    shocked = _experience(drift=-0.03, shock_year=1, shock=2.0)
    naive = AR.estimate_improvement(shocked["exposure"], shocked["deaths"])
    shock_aware = AR.estimate_improvement(shocked["exposure"], shocked["deaths"],
                                          shock_family=family)
    assert shock_aware["shock_posterior"][1] > 0.9
    assert abs(shock_aware["drift"] + 0.03) < abs(naive["drift"] + 0.03)
    assert abs(shock_aware["drift"] + 0.03) < 0.02


def test_the_gompertz_slope_comes_back_off_the_experience_file():
    from meridia.methods import actuarial_reference as AR
    fit = AR.gompertz_slope(_experience(slope=0.09))
    assert fit["fitted"] and abs(fit["slope"] - 0.09) < 0.01
    assert 0.0 < fit["slope_se"] < 0.01


def test_age_heaping_is_measured_and_moved_back_over_its_neighbours():
    import numpy as np
    from meridia.methods import actuarial_reference as AR
    rng = np.random.default_rng(5)
    age = rng.integers(20, 90, 40_000)
    heaped = np.where(rng.random(len(age)) < 0.3, 5 * np.round(age / 5).astype(int), age)
    measured = AR.age_heaping_intensity(heaped)
    assert measured["fitted"] and 0.05 < measured["excess"] < 0.35
    assert AR.age_heaping_intensity(age)["excess"] < 0.02
    cube = np.zeros((2, 101, 2))
    for a in heaped:
        cube[0, a, 0] += 1.0
    fixed = AR.deheap_age_cube(cube, measured["excess"])
    assert abs(fixed.sum() - cube.sum()) < 1e-6
    on_five = [a for a in range(25, 90) if a % 5 == 0]
    assert fixed[0, on_five, 0].sum() < cube[0, on_five, 0].sum()


def test_the_response_model_reads_the_gradient_the_sampling_units_show():
    import numpy as np
    import pandas as pd
    from meridia.methods import actuarial_reference as AR
    rng = np.random.default_rng(9)
    urbanity = np.linspace(0.0, 1.0, 12)
    rows = []
    for county in range(12):
        rate = 1.0 / (1.0 + np.exp(-(1.2 - 1.6 * urbanity[county])))
        sampled = 60
        responded = int(rng.binomial(sampled, rate))
        for household in range(responded):
            rows.append({"county": county, "household": f"{county}-{household}",
                         "psu": county, "psu_sampled_households": sampled,
                         "design_weight": 10.0, "age": 45.0, "income": 30_000.0})
    survey = pd.DataFrame(rows)
    fit = AR.fit_survey_response(survey, np.zeros(12, dtype=int), urbanity)
    assert fit["fitted"] and -2.4 < fit["urban"] < -0.9
    weights = AR.nonresponse_weights(survey, fit)
    county = survey["county"].to_numpy()
    # The county totals stay where the design put them; only the composition moves.
    for c in range(12):
        assert abs(weights[county == c].sum() -
                   survey["design_weight"].to_numpy()[county == c].sum()) < 1e-6
    assert weights[county == 11].max() > weights[county == 0].max()


def test_the_churn_fit_keeps_the_deaths_a_flat_rate_would_remove_from_the_old():
    import numpy as np
    from meridia.methods import actuarial_reference as AR
    n_counties, n_bands = 6, 6
    experience = _experience(n_states=2)
    mobility = AR.mobility_profile(experience)
    age_error = np.ones(n_bands)
    at_risk = np.full((n_counties, n_bands, 2), 4000.0)
    truth = np.zeros((n_counties, n_bands, 2))
    truth[:, 3:, :] = 0.05          # deaths concentrate in the three oldest bands
    truth[:, :3, :] = 0.001
    churn = 0.10 * np.broadcast_to(mobility, truth.shape[1:])[None]
    gone = at_risk * (truth + churn)
    fit = AR.fit_churn(gone, at_risk, 12, truth, mobility, age_error,
                       np.zeros(n_counties, dtype=int))
    assert fit["fitted"]
    recovered = (gone - fit["churn"] * at_risk) / at_risk
    flat = gone[:, :2, :].sum(axis=(1, 2)) / at_risk[:, :2, :].sum(axis=(1, 2))
    flat_recovered = (gone - flat[:, None, None] * at_risk) / at_risk
    old = np.s_[:, 3:, :]
    assert np.abs(recovered[old] - truth[old]).mean() < \
        np.abs(flat_recovered[old] - truth[old]).mean()
    assert np.abs(recovered[old] / truth[old] - 1.0).max() < 0.25


def test_a_deviation_collapses_to_one_when_its_own_measurement_is_the_noise():
    import numpy as np
    from meridia.methods.actuarial_reference import shrink_deviation
    rng = np.random.default_rng(2)
    noisy = shrink_deviation(1.0 + rng.normal(0.0, 0.5, 40), np.full(40, 0.25))
    assert noisy["tau2"] < 0.05 and np.abs(noisy["deviation"] - 1.0).max() < 0.2
    signal = np.repeat([0.6, 1.4], 20)
    clean = shrink_deviation(signal, np.full(40, 1e-4))
    assert clean["tau2"] > 0.1 and np.abs(clean["deviation"] - signal).max() < 0.05


def test_two_measurements_of_one_level_combine_by_their_own_precision():
    import numpy as np
    from meridia.methods.actuarial_reference import blend_levels, level_disagreement
    first = np.array([1.0, 2.0])
    second = np.array([2.0, 4.0])
    tight = blend_levels(first, np.array([1e-6, 1e-6]), second, np.array([1.0, 1.0]))
    assert np.allclose(tight["rate"], first, rtol=1e-3) and tight["weight"].max() < 1e-5
    even = blend_levels(first, np.array([0.1, 0.1]), second, np.array([0.1, 0.1]))
    assert np.allclose(even["rate"], np.sqrt(first * second))
    spread = level_disagreement(first, second)
    assert abs(spread["national"] - np.log(2.0) / np.sqrt(2.0)) < 1e-9
    assert spread["regional"] < 1e-9


def test_the_priced_composition_is_raked_to_the_file_and_the_nation_is_untouched():
    import numpy as np
    from meridia.methods import actuarial_reference as AR
    county_state = np.array([0, 0, 1, 1])
    paths = np.full((3, 4, 101, 2), 5.0)
    experience = _experience(n_states=2)
    experience["exposure"][-1, 0] *= 3.0            # state 0 holds three quarters
    shares = AR.experience_state_shares(experience)
    assert abs(shares[0, 3, 0] - 0.75) < 1e-9
    raked = AR.rake_to_state_shares(paths, county_state, shares)
    collapse = AR.band_matrix(100)
    banded = np.einsum("ba,cas->cbs", collapse, raked.mean(axis=0))
    state = np.stack([banded[:2].sum(axis=0), banded[2:].sum(axis=0)])
    assert np.allclose(state / state.sum(axis=0), shares, atol=1e-9)
    before = np.einsum("ba,cas->cbs", collapse, paths.mean(axis=0)).sum(axis=0)
    assert np.allclose(banded.sum(axis=0), before)


def test_the_third_line_advances_lagged_state_elder_shares():
    import numpy as np
    from meridia.methods import actuarial_reference as AR

    experience = _experience(n_states=2)
    experience["deaths"][:, 0, 3:, :] *= 4.0
    published = AR.experience_state_shares(experience)
    advanced = AR.advanced_experience_state_shares(experience, 1.5)
    assert np.allclose(advanced.sum(axis=0), 1.0)
    assert not np.allclose(advanced[:, 3:, :], published[:, 3:, :])


def test_the_third_line_reconciles_absolute_elder_cohorts_at_state_level():
    import pandas as pd
    import numpy as np
    from meridia.methods import actuarial_reference as AR

    county_state = np.array([0, 0, 1, 1])
    paths = np.full((3, 4, 101, 2), 5.0)
    experience = _experience(n_states=2)
    experience["exposure"][-1, :, 3:, :] *= 1.2
    profile = np.zeros((2, 101, 2))
    np.add.at(profile, county_state, paths.mean(axis=0))
    target = AR.advanced_experience_state_exposure(
        experience, 1.5, age_profile=profile
    )
    survey = pd.DataFrame(
        {
            "county": [0, 1, 2, 3] * 4,
            "age": [70] * 8 + [80] * 4 + [90] * 4,
            "sex": [0, 1] * 8,
            "design_weight": [10.0] * 16,
        }
    )
    reconciled, diagnostics = AR.rake_to_cohort_component(
        paths, county_state, target, survey
    )
    collapse = AR.band_matrix(100)
    banded = np.einsum("ba,cas->cbs", collapse, reconciled.mean(axis=0))
    state = np.zeros_like(target)
    np.add.at(state, county_state, banded)
    assert np.allclose(state[:, 3:, :], target[:, 3:, :])
    before = np.einsum("ba,cas->cbs", collapse, paths.mean(axis=0)).sum(axis=0)
    assert np.allclose(banded[:, :3, :].sum(axis=0), before[:3, :])
    assert diagnostics["elder_after"] == pytest.approx(diagnostics["elder_target"])

    distorted = target.copy()
    distorted[:, :3, 0] *= 9.0
    distorted[:, :3, 1] *= 0.1
    preserved, _ = AR.rake_to_cohort_component(
        paths, county_state, distorted, survey
    )
    preserved_banded = np.einsum(
        "ba,cas->cbs", collapse, preserved.mean(axis=0)
    )
    assert np.allclose(preserved_banded[:, :3, :].sum(axis=0), before[:3, :])

    with pytest.raises(ValueError, match="one value per survey row"):
        AR.rake_to_cohort_component(
            paths,
            county_state,
            target,
            survey,
            survey_weights=np.ones(len(survey) - 1),
        )


def test_a_wider_level_uncertainty_widens_the_tail_without_lifting_the_mean():
    import numpy as np
    from meridia.methods import actuarial_reference as AR
    from meridia.actuarial import ObligationContract
    obligation = ObligationContract(monthly_benefit=150.0, qualifying_event_cost=15_000.0,
                                    death_benefit=7_500.0, monthly_discount_rate=0.002,
                                    eligibility_min_age=65, horizon_months=24)
    ac = AR.ActuarialContract(
        obligation=obligation, region_of_county=np.array([0, 1]), n_regions=2,
        reserve_total=0.0, reserve_weights=np.ones(2), gamma=0.25,
        anchor_item="recent_hospitalization", anchor_sensitivity=1.0,
        anchor_specificity=1.0, anchor_window_months=12, experience_years=5,
        experience_file="experience_history.csv", experience_last_tick=0)
    paths = np.zeros((4096, 2, 101, 2))
    paths[:, :, 70, :] = 500.0
    rates = {"mortality": np.full((2, 101, 2), 0.02),
             "incidence": np.full((2, 101, 2), 0.03),
             "migration": np.zeros((2, 101, 2)), "migration_se": np.zeros((2, 101, 2)),
             "not_yet": np.ones((2, 101, 2)), "fertility": 0.0,
             "mortality_drift": 0.0, "mortality_drift_se": 0.0,
             "incidence_drift": 0.0, "incidence_drift_se": 0.0}
    params = AR.SimulationParams(path_chunk=512, shock_probability=0.0)
    narrow = AR.simulate_liabilities(paths, dict(rates, mortality_log_sd=0.0,
                                                 incidence_log_sd=0.0), ac, params)
    wide = AR.simulate_liabilities(paths, dict(rates, mortality_log_sd=0.30,
                                               incidence_log_sd=0.30,
                                               mortality_log_sd_region=np.full(2, 0.30),
                                               incidence_log_sd_region=np.full(2, 0.30)),
                                   ac, params)
    narrow_mean = narrow["liability"].mean(axis=0)
    wide_mean = wide["liability"].mean(axis=0)
    # Mean one in expectation: what is left at four thousand paths is Monte Carlo error
    # on a level whose own spread is a third, not a loading.
    assert np.abs(wide_mean / narrow_mean - 1.0).max() < 0.02
    assert (wide["liability"].std(axis=0) > 2.0 * narrow["liability"].std(axis=0)).all()


def test_tail_summary_uses_the_declared_order_statistic_and_includes_ties():
    import numpy as np
    from meridia.methods import actuarial_reference as AR

    liability = np.asarray([[1.0, 9.0], [2.0, 4.0], [2.0, 4.0], [3.0, 1.0]])
    summary = AR.tail_summary(liability, alpha=0.50)
    assert np.allclose(summary["q"], [2.0, 4.0])
    assert np.allclose(summary["es"], [7.0 / 3.0, 17.0 / 3.0])
    with pytest.raises(ValueError, match="nonempty"):
        AR.tail_summary(np.zeros((0, 2)))
    for alpha in (-0.01, 1.01, float("nan")):
        with pytest.raises(ValueError, match="alpha"):
            AR.tail_summary(liability, alpha=alpha)


def test_tail_summary_boundary_is_not_numpy_higher_interpolation():
    import numpy as np
    from meridia.methods import actuarial_reference as AR

    liability = np.asarray([[1.0], [2.0], [3.0], [4.0]])
    # ceil(0.5 * 4) is observation two in one-based indexing.
    assert AR.tail_summary(liability, alpha=0.5)["q"].item() == 2.0


def test_each_simulated_member_redraws_the_published_shock_process():
    import numpy as np
    from meridia.methods import actuarial_reference as AR

    family = {
        "annual_rate": 0.20,
        "kinds": [
            {"mortality": (1.5, 3.0), "incidence": (1.4, 2.6)}
        ],
    }
    first = AR.draw_shock_year(np.random.default_rng(1), 20_000, family, 0.0, (1, 1))
    second = AR.draw_shock_year(np.random.default_rng(2), 20_000, family, 0.0, (1, 1))
    first_hit = first["mortality"] > 1.0
    second_hit = second["mortality"] > 1.0
    assert 0.19 < first_hit.mean() < 0.21
    assert not np.array_equal(first_hit, second_hit)
    assert np.array_equal(first_hit, first["incidence"] > 1.0)
    common_draw_mortality = (first["mortality"][first_hit] - 1.5) / 1.5
    common_draw_incidence = (first["incidence"][first_hit] - 1.4) / 1.2
    assert np.allclose(common_draw_mortality, common_draw_incidence)


def test_phase_three_reports_state_elder_exposure_and_survival(packet, submission):
    from meridia.methods.phase_three import (
        elder_state_exposure_survival,
        participant_elder_identifiability,
        third_elder_comparison,
    )

    audit = elder_state_exposure_survival(packet, submission)
    assert audit["thresholds"] == {
        "aggregate_exposure_relative_error_ceiling_copied_from_bar": None,
        "aggregate_mortality_relative_error_ceiling_copied_from_bar": None,
        "criterion": "absolute aggregate relative error, not the verifier cell percentile",
    }
    assert len(audit["states"]) == PARAMS.n_states
    for row in audit["states"]:
        assert row["estimated_person_years"] > 0.0
        assert row["sealed_person_years"] > 0.0
        assert 0.0 < row["estimated_survival"] <= 1.0
        assert 0.0 < row["sealed_survival"] <= 1.0
    participant = participant_elder_identifiability(packet)
    assert len(participant["states"]) == PARAMS.n_states
    assert all(
        abs(row["public_experience_state_share_error"]) < 1.0
        for row in participant["states"]
    )
    comparison = third_elder_comparison(participant, audit)
    assert set(comparison) >= {"public_experience", "third_line"}

    from meridia.methods.phase_three import (
        elder_eligibility_audit,
        mortality_gap_decomposition,
    )

    eligibility = elder_eligibility_audit(packet)
    assert eligibility["scored"]["age_band"] == "65+"
    assert eligibility["scored"]["floor_person_years"] == 500.0
    assert [row["age_band"] for row in eligibility["report_only"]] == [
        "65-74",
        "75-84",
        "85+",
    ]
    decomposition = mortality_gap_decomposition(packet)
    assert decomposition["trend_active_during_public_experience_window"] is True
    assert decomposition["trend_starts_only_after_public_window"] is False
    assert decomposition["continuation_shocks_redrawn_per_member"] is True


def test_elder_eligibility_rejects_a_missing_final_state(packet, tmp_path):
    import json
    import shutil

    import pandas as pd

    from meridia.methods.phase_three import elder_eligibility_audit

    broken = tmp_path / "broken-eligibility"
    (broken / "participant").mkdir(parents=True)
    (broken / "retained").mkdir()
    shutil.copy2(
        packet / "participant" / "contract.json",
        broken / "participant" / "contract.json",
    )
    contract = json.loads((broken / "participant" / "contract.json").read_text())
    truth = pd.read_csv(packet / "retained" / "rate_truth_horizon.csv")
    last_state = int(contract["n_states"]) - 1
    truth = truth[
        ~(
            (truth["level"] == "state")
            & (truth["unit"] == last_state)
            & (truth["estimand"] == "person_years_exposure")
            & (truth["age_band"] == "85+")
        )
    ]
    truth.to_csv(broken / "retained" / "rate_truth_horizon.csv", index=False)

    with pytest.raises(ValueError, match="retained eligibility exposure"):
        elder_eligibility_audit(broken)
