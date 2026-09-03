"""Packets: flat participant files with no truth, sealed retained truth that matches the
engine exactly, deterministic manifests, and a development packet that ships its truth."""

import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from meridia.actuarial import ensemble_truth
from meridia.packet import (FORBIDDEN_COLUMN_PREFIXES, PacketParams, build_packet,
                            reserve_weights,
                            build_world, participant_columns)
from meridia.projection import project_truth_from_history
from meridia.survey import SURVEY_BANDS, SurveyParams
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
    assert set(reserve) >= {"obligation", "total", "gamma", "regions", "weights"}
    assert reserve["regions"] == "state" and 0.20 <= reserve["gamma"] <= 0.30
    assert reserve["total"] > 0 and reserve["total"] % reserve["rounding_unit"] == 0
    weights = np.asarray(reserve["weights"], dtype=np.float64)
    assert len(weights) == contract["n_states"]
    assert weights.min() > 0 and len(set(np.round(weights, 6))) == len(weights)
    obligation = reserve["obligation"]
    assert obligation["horizon_months"] == PARAMS.horizon_months
    assert obligation["eligibility_min_age"] == 65
    # Nothing regional and sealed rides along: no per-region liability, quantile or
    # shortfall appears anywhere in the participant contract.
    text = json.dumps(contract)
    for forbidden in ("q95", "es95", "liability", "exceedance", "member"):
        assert f'"{forbidden}"' not in text


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


def test_the_reserve_total_sits_above_the_sealed_quantiles(packet):
    """R = sum q* + gamma sum (es* - q*): feasible for a perfect submission, and no more."""
    out, _ = packet
    contract = json.loads((out / "participant" / "contract.json").read_text())
    with np.load(out / "retained" / "continuation_liabilities.npz") as archive:
        liability = archive["liability"]
    truth = ensemble_truth(liability)
    total = float(contract["reserve"]["total"])
    assert total >= float(truth["q"].sum())
    raw = float(truth["q"].sum()
                + contract["reserve"]["gamma"] * float((truth["es"] - truth["q"]).sum()))
    assert 0 <= total - raw < contract["reserve"]["rounding_unit"]


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
    assert set(bench["level"]) == {"nation", "state"}
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
