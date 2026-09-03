"""Packets: flat participant files with no truth, sealed retained truth that matches the
engine exactly, deterministic manifests, and a development packet that ships its truth."""

import json
import math
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from meridia.actuarial import ObligationContract
from meridia.packet import (EXPERIENCE_BURN_IN_MONTHS, FORBIDDEN_COLUMN_PREFIXES,
                            GRADING_WORLD, PacketParams, _experience_history,
                            build_packet, build_world, participant_columns,
                            reserve_weights)
from meridia.projection import project_truth_from_history
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
    assert set(reserve) >= {"obligation", "total", "total_rule", "regions", "weights"}
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
        assert int(archive["realized_member"]) == 0
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


def test_development_packet_ships_truth_and_hidden_does_not(tmp_path, packet):
    out, hidden = packet
    dev = build_packet(SEED, tmp_path / "dev", PARAMS, development=True)
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
    hidden = build_packet(SEED, tmp_path / "hidden", hidden_params, development=False)
    dev = build_packet(SEED, tmp_path / "dev", PARAMS, development=False)
    world_h = json.loads((tmp_path / "hidden" / "retained" / "world.json").read_text())
    world_d = json.loads((tmp_path / "dev" / "retained" / "world.json").read_text())
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


def test_the_cache_key_moves_when_the_priced_world_moves(tmp_path):
    """The digest covers the branch, the shock law, the horizon and the obligation."""
    from meridia.actuarial import ObligationContract, regions_from_admin
    from meridia.mechanisms import QUALIFYING_DIAGNOSIS_GROUPS
    from meridia.packet import baseline_ledger_digest

    built = build_world(SEED, PARAMS)
    region = regions_from_admin(built["admin"])
    obligation = ObligationContract(
        horizon_months=PARAMS.horizon_months,
        qualifying_diagnosis_groups=QUALIFYING_DIAGNOSIS_GROUPS)
    key = baseline_ledger_digest(built["history"], obligation, PARAMS.horizon_months,
                                 region)
    assert key == baseline_ledger_digest(built["history"], obligation,
                                         PARAMS.horizon_months, region)
    assert key != baseline_ledger_digest(built["history"], obligation,
                                         PARAMS.horizon_months + 1, region)
    dearer = ObligationContract(
        horizon_months=PARAMS.horizon_months,
        qualifying_diagnosis_groups=QUALIFYING_DIAGNOSIS_GROUPS,
        death_benefit=ObligationContract().death_benefit + 100.0)
    assert key != baseline_ledger_digest(built["history"], dearer,
                                         PARAMS.horizon_months, region)
    other = build_world(SEED + 1, PARAMS)
    assert key != baseline_ledger_digest(other["history"], obligation,
                                         PARAMS.horizon_months,
                                         regions_from_admin(other["admin"]))
