"""Version-four mechanisms: the committed design, local heterogeneity, and the anchors.

Each test here names one of the four unintended sufficient statistics the version-three
falsification found, or one of the anchors that keeps the replacement identifiable:

1. stable within-source identifiers solved longitudinal matching,
2. one near-constant household growth factor solved projection,
3. a discrete set of register money scales made unit recovery a decoding problem,
4. fifteen of nineteen source rates were the same constant in every world.

A test that only checks a mechanism exists is not enough: each one also checks that the
quantity a participant would have measured on a development world no longer transfers,
and that what replaced it is estimable from files the participant receives.
"""

import itertools
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from meridia.events import EVENT_TYPES, replay_event_history
from meridia.mechanisms import (COEFFICIENT_RANGES, DEVELOPMENT_BAND, DEVELOPMENT_DESIGN,
                                HIDDEN_EXTRAPOLATION_AXES, HIDDEN_IN_BAND_AXES,
                                HIDDEN_LEVEL_PATTERNS, N_HIDDEN_OUTSIDE_AXES, MECHANISM_AXES,
                                N_DEVELOPMENT_CELLS, PAIRWISE_AXIS_INTERACTIONS,
                                PUBLIC_ENVELOPE, build_world_mechanisms,
                                contract_block, draw_mechanism_coefficients,
                                draw_mechanism_design, migration_age_pull, quintile_band,
                                rank_uniform)
from meridia.mechanisms import (WorldMechanisms, death_report_late_probability,
                                draw_county_effects)
from meridia.packet import PacketParams, build_world
from meridia.sources import (SourceParams, _employment_summary,
                             _record_mechanism_rates, _sequence_position)

SEED = 20260902
WORLD = PacketParams(grid=(96, 128), n_settlements=8, n_states=3, observed_months=24,
                     preliminary_lag=6, horizon_months=12, total=120_000,
                     experience_years=1)


@pytest.fixture(scope="module")
def worlds():
    return {seed: build_world(seed, WORLD) for seed in (3, 7, 11)}


@pytest.fixture(scope="module")
def hidden_world():
    return build_world(7, PacketParams(**{**WORLD.__dict__, "regime": "hidden"}))


def _county_of(built, cell):
    flat = np.asarray(built["admin"]["county"], dtype=np.int64).reshape(-1)
    return np.maximum(flat[np.asarray(cell, dtype=np.int64)], 0)


# --------------------------------------------------------------- the committed design

def test_development_design_is_balanced_and_leaves_main_effects_clean():
    design = DEVELOPMENT_DESIGN.astype(np.int64)
    assert design.shape == (N_DEVELOPMENT_CELLS, len(MECHANISM_AXES))
    assert 8 <= N_DEVELOPMENT_CELLS <= 12
    assert set(np.unique(design)) == {-1, 1}
    assert (design.sum(axis=0) == 0).all()
    gram = design.T @ design
    assert np.array_equal(gram, N_DEVELOPMENT_CELLS * np.eye(len(MECHANISM_AXES), dtype=np.int64))
    worst = 0.0
    for i, j in itertools.combinations(range(len(MECHANISM_AXES)), 2):
        interaction = design[:, i] * design[:, j]
        for k in range(len(MECHANISM_AXES)):
            worst = max(worst, abs(int(interaction @ design[:, k])) / N_DEVELOPMENT_CELLS)
    assert worst < 1.0, "a two-factor interaction is fully aliased with a main effect"
    assert worst == pytest.approx(1.0 / 3.0)


def test_every_development_cell_is_reachable_and_intensities_stay_in_band():
    seen = set()
    for cell in range(N_DEVELOPMENT_CELLS):
        design = draw_mechanism_design(SEED + cell, "development", cell)
        assert design.cell == cell
        assert design.outside == ()
        seen.add(design.levels)
        for axis, value in design.intensity.items():
            low, high = DEVELOPMENT_BAND[axis]
            assert low <= value <= high, (axis, value)
    assert len(seen) == N_DEVELOPMENT_CELLS


def test_hidden_world_is_a_new_joint_configuration_with_two_intensities_outside():
    design_rows = {tuple(int(v) for v in row) for row in DEVELOPMENT_DESIGN}
    assert len(HIDDEN_LEVEL_PATTERNS) == 2 ** len(MECHANISM_AXES) - len(design_rows)
    assert not design_rows & set(HIDDEN_LEVEL_PATTERNS)
    seen_levels, seen_outside = set(), set()
    for seed in (1, 7, 4242, 2101, 2102, 8101, 8102, 8103):
        design = draw_mechanism_design(seed, "hidden")
        assert design.cell == -1
        assert design.levels in HIDDEN_LEVEL_PATTERNS
        assert len(design.outside) == N_HIDDEN_OUTSIDE_AXES
        assert set(design.outside) <= set(HIDDEN_EXTRAPOLATION_AXES)
        assert not set(design.outside) & set(HIDDEN_IN_BAND_AXES)
        seen_levels.add(design.levels)
        seen_outside.add(design.outside)
        outside = 0
        for axis, value in design.intensity.items():
            envelope_low, envelope_high = PUBLIC_ENVELOPE[axis]
            band_low, band_high = DEVELOPMENT_BAND[axis]
            assert envelope_low <= value <= envelope_high, (axis, value)
            if not band_low <= value <= band_high:
                outside += 1
                assert axis in design.outside
        assert outside == N_HIDDEN_OUTSIDE_AXES
        for axis in HIDDEN_IN_BAND_AXES:
            low, high = DEVELOPMENT_BAND[axis]
            assert low <= design.intensity[axis] <= high
    # The hidden configuration is a draw, not a constant of this repository.
    assert len(seen_levels) > 1
    assert len(seen_outside) > 1
    policy = contract_block()["hidden_axis_policy"]
    assert policy == {
        "outside_axis_count": 2,
        "eligible_for_outside_development_band": list(HIDDEN_EXTRAPOLATION_AXES),
        "held_inside_development_band": list(HIDDEN_IN_BAND_AXES),
        "anchor_correlation_required_for_extrapolation": 0.4,
    }
    with pytest.raises(ValueError, match="design cell"):
        draw_mechanism_design(1, "hidden", 0)


def test_coefficients_are_deterministic_and_never_shared_between_worlds():
    first = build_world_mechanisms(SEED, "development", cell=4)
    again = build_world_mechanisms(SEED, "development", cell=4)
    assert first.coefficients == again.coefficients
    vectors = [
        tuple(build_world_mechanisms(seed, "development").coefficients[name]
              for name in sorted(COEFFICIENT_RANGES))
        for seed in range(1, 25)
    ]
    assert len(set(vectors)) == len(vectors)
    for name, (low, high) in COEFFICIENT_RANGES.items():
        values = [build_world_mechanisms(seed, "development").coefficients[name]
                  for seed in range(1, 25)]
        assert all(low <= value <= high for value in values)
        assert len(set(values)) == len(values)


def test_contract_publishes_the_families_and_the_design_but_no_realized_value():
    block = contract_block()
    assert block["axes"] == list(MECHANISM_AXES)
    assert block["development_design"] == DEVELOPMENT_DESIGN.tolist()
    assert set(block["declared_interactions"]) == {
        "linkage_gradient_by_migration",
        "health_completeness_by_latent_frailty",
        "death_capture_by_age_error",
        "migration_by_stale_address_linkage",
        "rurality_by_name_and_address_error",
        "age_error_by_age_slope_of_mortality",
        "income_scale_by_income_dependent_migration",
    }
    # Three of them are a product of two axes at one site, and the contract says which.
    assert set(block["pairwise_axis_interactions"]) == set(PAIRWISE_AXIS_INTERACTIONS)
    assert len(block["pairwise_axis_interactions"]) >= 3
    for description in block["declared_interactions"].values():
        coefficient = description.split(":")[0].strip()
        assert coefficient in COEFFICIENT_RANGES or coefficient in MECHANISM_AXES
    assert set(block["covariates"]) >= {"urban_c", "econ_c", "elder_c", "band_r"}


def test_migration_age_pull_is_a_published_curve_that_turns_over_with_age():
    young, middle, old = migration_age_pull(np.asarray([24.0, 50.0, 78.0]))
    assert young > middle > 0.0 > old


# ------------------------------------------------- 1. identifiers no longer stay put

def test_identifier_persistence_is_partial_and_keys_are_sometimes_reissued(worlds):
    for seed, built in worlds.items():
        for source, column in (("population", "person_id"), ("health", "encounter_id")):
            preliminary = built["sources"]["public_snapshots"]["preliminary"][source]
            revised = built["sources"]["public_snapshots"]["revised"][source]
            assert len(np.intersect1d(preliminary["record_id"], revised["record_id"])) == 0
            before = built["sources"]["hidden"]["crosswalks"]["preliminary"][source]
            after = built["sources"]["hidden"]["crosswalks"]["revised"][source]
            first = dict(zip(before["observed_entity_id"].tolist(),
                             before["truth_entity_id"].tolist()))
            second = dict(zip(after["observed_entity_id"].tolist(),
                              after["truth_entity_id"].tolist()))
            shared = set(first) & set(second)
            assert 0.20 < len(shared) / len(second) < 0.98, (seed, source)
            assert sum(1 for key in shared if first[key] != second[key]) > 0
            assert revised[column].dtype == np.uint64


# --------------------------------------- 2. household growth is structured, not global

def test_county_household_growth_follows_the_county_covariates(worlds):
    correlations = []
    for built in worlds.values():
        ticks = built["ticks"]
        county_flat = np.asarray(built["admin"]["county"], dtype=np.int64).reshape(-1)
        n_counties = int(built["admin"]["n_counties"])

        def households(tick):
            state = replay_event_history(built["history"], tick)
            active = state["household"]["is_active"]
            county = np.maximum(county_flat[state["household"]["cell"][active]], 0)
            return np.bincount(county, minlength=n_counties).astype(np.float64)

        start, end = households(ticks["revised"]), households(ticks["horizon"])
        dense = start >= 40
        growth = end[dense] / start[dense] - 1.0
        urban = built["mechanisms"].county.urban[dense]
        assert dense.sum() >= 8
        correlations.append(float(np.corrcoef(growth, urban)[0, 1]))
    # Version three's growth was one national scalar, so county growth carried no signal.
    # The floor is a floor on a drawn quantity, not the quantity itself: the three worlds
    # of this fixture read 0.35, 0.48 and 0.79.
    assert min(correlations) > 0.30, correlations


def test_destinations_are_age_patterned_rather_than_uniform_over_vacant_units(worlds):
    for built in worlds.values():
        event = built["history"]["event"]
        urbanity = np.asarray(built["micro"]["urbanity"], dtype=np.float64).reshape(-1)
        formed = event["event_type"] == EVENT_TYPES["household_formed"]
        assert formed.sum() > 200
        origin = urbanity[event["from_cell"][formed]]
        destination = urbanity[event["to_cell"][formed]]
        # A uniform draw over vacant dwellings would put destination urbanity at the
        # national vacancy average, uncorrelated with where the mover started.
        assert abs(float(np.corrcoef(origin, destination)[0, 1])) > 0.05


# ---------------------------------------- 3. the register money unit is local, not one float

def test_register_money_unit_varies_by_county_so_one_national_ratio_cannot_decode_it(worlds):
    spreads = []
    for built in worlds.values():
        terminal = built["history"]["terminal_state"]
        earnings, _, _, _ = _employment_summary(
            terminal,
            len(terminal["person"]["truth_person_id"]),
            len(terminal["establishment"]["truth_establishment_id"]),
        )
        revised = built["sources"]["public_snapshots"]["revised"]["income"]
        crosswalk = built["sources"]["hidden"]["crosswalks"]["revised"]["income"]
        order = np.argsort(crosswalk["observed_record_id"])
        assert np.array_equal(crosswalk["observed_record_id"][order], revised["record_id"])
        truth = earnings[_sequence_position(crosswalk["truth_entity_id"][order])].astype(np.float64)
        observed = revised["employment_income_cents"]
        usable = np.isfinite(observed) & (truth > 0)
        realized = observed[usable] / truth[usable]
        county = revised["county"][usable]
        medians = np.asarray([
            np.median(realized[county == c])
            for c in np.unique(county)
            if (county == c).sum() > 30
        ])
        assert len(medians) >= 8
        spreads.append(float(medians.max() / medians.min()))
    assert min(spreads) > 1.10, spreads


def test_money_band_coefficient_moves_the_unit_across_worlds():
    """The band slope is a per-world draw, so it is material in most worlds and small in
    a few. What must never happen is a world where every coefficient is the same."""
    implied = np.asarray([
        np.exp(2.0 * abs(build_world_mechanisms(seed, "development").coefficients["income_scale_band"]))
        for seed in range(1, 41)
    ])
    assert np.median(implied) > 1.10
    assert len(set(implied.tolist())) == len(implied)


# ------------------------------------------ 4. no source rate is a constant across worlds

def test_no_source_rate_is_the_same_number_in_every_world():
    from meridia.sources import DEVELOPMENT_BAND as SOURCE_BAND
    from meridia.sources import draw_source_params
    seeds = (1, 7, 11, 4242, 20260902)
    # Every field of SourceParams now has a published band and a per-world draw.
    assert len(SOURCE_BAND) == 20
    for name in SOURCE_BAND:
        values = {getattr(draw_source_params(seed, "development", 1.0), name) for seed in seeds}
        assert len(values) == len(seeds), name


def test_defect_rates_carry_a_rural_gradient_inside_one_world(worlds):
    """The declared rurality by name and address error interaction, measured."""
    gradients = []
    for built in worlds.values():
        mechanism = built["sources"]["hidden"]["mechanisms"]["population"]
        person = built["history"]["terminal_state"]["person"]
        county = _county_of(built, person["cell"])
        urban = built["mechanisms"].county.urban[county]
        rural = urban <= np.quantile(urban, 0.3)
        city = urban >= np.quantile(urban, 0.7)
        gradients.append(float(mechanism["linkage_error"][rural].mean()
                               - mechanism["linkage_error"][city].mean()))
    assert min(gradients) > 0.0, gradients


# --------------------------------------------------- health selection and its anchor

def test_health_source_inclusion_rises_with_latent_frailty(worlds, hidden_world):
    def gap(built):
        terminal = built["history"]["terminal_state"]
        position = _sequence_position(terminal["encounter"]["truth_person_id"])
        frailty = terminal["person"]["frailty_centi"][position].astype(np.float64) / 100.0
        covered = built["sources"]["hidden"]["mechanisms"]["health"]["covered"]
        low, high = np.quantile(frailty, [0.2, 0.8])
        return float(covered[frailty >= high].mean() - covered[frailty <= low].mean())

    development = [gap(built) for built in worlds.values()]
    assert min(development) > 0.01, development
    # The hidden world's intensity sits outside the development band, so a method that
    # ignores informative inclusion pays much more there.
    assert gap(hidden_world) > max(development)


def test_the_survey_anchor_tracks_true_admission_without_reading_the_register(worlds):
    for built in worlds.values():
        from meridia.packet import _recent_admission, _survey_at
        tick = built["ticks"]["revised"]
        state = replay_event_history(built["history"], tick)
        truth = _recent_admission(built["history"], state, tick)
        assert 0.0 < truth.mean() < 0.5
        survey = _survey_at(built, tick, 1)
        reported = survey["survey"]["recent_hospitalization"]
        anchored = survey["truth"]["recent_admission"]
        assert set(np.unique(reported)) <= {0, 1}
        assert reported.mean() > 0.0
        assert float(reported[anchored].mean()) > float(reported[~anchored].mean())


def test_latent_frailty_is_conserved_by_replay_and_inherited_by_newborns(worlds):
    for built in worlds.values():
        history = built["history"]
        replayed = replay_event_history(history, int(history["terminal_tick"]))
        assert np.array_equal(replayed["person"]["frailty_centi"],
                              history["terminal_state"]["person"]["frailty_centi"])
        born = history["event"]["event_type"] == EVENT_TYPES["person_birth"]
        assert born.sum() > 50
        assert (history["event"]["frailty_centi"][born] > 0).all()
        assert len(np.unique(history["event"]["frailty_centi"][born])) > 10


# ------------------------------------------------- covariates a participant can rebuild

def test_county_covariates_are_recoverable_from_the_participant_files(worlds):
    agreement = []
    for built in worlds.values():
        county_flat = np.asarray(built["admin"]["county"], dtype=np.int64).reshape(-1)
        n_counties = int(built["admin"]["n_counties"])
        land = np.bincount(county_flat[county_flat >= 0], minlength=n_counties).astype(np.float64)
        revised = built["sources"]["public_snapshots"]["revised"]["population"]
        registered = np.bincount(revised["county"], minlength=n_counties).astype(np.float64)
        estimate = rank_uniform(registered / np.maximum(land, 1.0))
        truth = built["mechanisms"].county.urban
        assert float(np.corrcoef(estimate, truth)[0, 1]) > 0.85

        # econ_c is the payroll-per-adult definition, rebuilt from the business and
        # population sources.  It is only partially recovered, which is the protocol's
        # "partially learnable" case: the register that reports it is itself thinned by
        # the completeness gradient econ_c drives.  Measured on three worlds: 0.25, 0.80,
        # 0.54.  What matters is that the trace is real and positive in every world.
        business = built["sources"]["public_snapshots"]["revised"]["business"]
        revised_tick = int(built["ticks"]["revised"])
        adults = np.bincount(
            revised["county"][(revised_tick - revised["birth_tick"]) // 12 >= 16],
            minlength=n_counties).astype(np.float64)
        priced = np.isfinite(business["annual_payroll_cents"])
        payroll = np.bincount(business["county"][priced],
                              weights=business["annual_payroll_cents"][priced],
                              minlength=n_counties)
        keep = adults > 50
        assert keep.sum() >= 8
        econ_estimate = rank_uniform((payroll / np.maximum(adults, 1.0))[keep])
        econ_truth = rank_uniform(built["mechanisms"].county.econ[keep])
        agreement.append(float(np.corrcoef(econ_estimate, econ_truth)[0, 1]))
    assert min(agreement) > 0.15, agreement
    assert float(np.mean(agreement)) > 0.35, agreement


def test_quintile_band_is_invariant_to_the_unknown_money_scale():
    rng = np.random.default_rng(0)
    values = np.exp(rng.normal(size=5_000))
    for scale in (0.55, 1.0, 1.87):
        assert np.array_equal(quintile_band(values), quintile_band(values * scale))


# ------------------------------------------------------------------ determinism

def test_the_mechanism_layer_does_not_disturb_ledger_determinism():
    first = build_world(3, WORLD)
    second = build_world(3, WORLD)
    assert np.array_equal(first["history"]["event"]["truth_event_id"],
                          second["history"]["event"]["truth_event_id"])
    assert np.array_equal(first["history"]["terminal_state"]["person"]["frailty_centi"],
                          second["history"]["terminal_state"]["person"]["frailty_centi"])
    assert first["mechanisms"].record() == second["mechanisms"].record()


# ------------------------------------------------- the hidden draw is a draw, not a line

def test_two_hidden_seeds_draw_different_level_patterns():
    """The hidden corner is estimated from data or it is read off this module.

    Version four's first pass wrote the level pattern and the pair of axes that leave the
    development band as module constants, so every hidden world sat in one corner and
    moved one pair. Proof obligation 7 was then tested inside a single configuration
    rather than across the family.
    """
    import meridia.mechanisms as module
    assert not hasattr(module, "HIDDEN_LEVELS")
    assert not hasattr(module, "HIDDEN_OUTSIDE_AXES")

    first = draw_mechanism_design(2101, "hidden")
    second = draw_mechanism_design(2102, "hidden")
    assert first.levels != second.levels
    assert first.outside != second.outside
    assert draw_mechanism_design(2101, "hidden").levels == first.levels

    # Nine hidden seeds, and the patterns are not one value. The seeds here are the
    # qualification set and three arbitrary others: an evaluation seed belongs in the
    # sealed file the build script reads, not in a test.
    hidden = [draw_mechanism_design(seed, "hidden")
              for seed in (2101, 2102, 2103, 2104, 2105, 2106, 9001, 9002, 9003)]
    assert len({design.levels for design in hidden}) >= 8
    assert len({design.outside for design in hidden}) >= 4

    # The design draw and the coefficient draw are separate streams, so neither reads the
    # other's position and a world's corner is not a function of its coefficient vector.
    axes = set(MECHANISM_AXES)
    one = draw_mechanism_coefficients(2101, draw_mechanism_design(2101, "hidden"))
    other = draw_mechanism_coefficients(2101, draw_mechanism_design(2101, "development"))
    assert {k: v for k, v in one.items() if k not in axes} == \
           {k: v for k, v in other.items() if k not in axes}


# ------------------------------------------------- products of two axes at one site

def _rates(mechanisms, source="population"):
    n = mechanisms.county.n_counties
    county = np.arange(n, dtype=np.int64)
    return _record_mechanism_rates(
        SourceParams(), mechanisms, source, county,
        np.full(n, 2, dtype=np.int8), np.ones(n), np.full(n, 45.0), np.full(n, 0.90))


def _with(mechanisms, coefficients=None, effects=None):
    return WorldMechanisms(mechanisms.design,
                           dict(mechanisms.coefficients, **(coefficients or {})),
                           mechanisms.county,
                           dict(mechanisms.effects, **(effects or {})))


def test_the_rural_linkage_gradient_is_scaled_by_the_migration_axis(worlds):
    """First of the three products of two axes: linkage_urban_gradient x migration."""
    base = worlds[3]["mechanisms"]
    urban = base.county.urban
    rural = urban <= np.quantile(urban, 0.3)
    city = urban >= np.quantile(urban, 0.7)

    def gradient(migration):
        rates = _rates(_with(base, {"migration_age_pattern": float(migration)}))
        return float(rates["linkage_error"][rural].mean()
                     - rates["linkage_error"][city].mean())

    assert gradient(2.30) > gradient(0.25) > 0.0
    # The axis enters only through the product, so a world at the neutral migration
    # intensity gets the gradient the linkage axis alone implies.
    neutral = _with(base, {"migration_age_pattern": 1.0})
    alone = _with(base, {"migration_age_pattern": 1.0, "linkage_gradient_by_migration": 0.0})
    assert _rates(neutral)["linkage_error"] == pytest.approx(_rates(alone)["linkage_error"])


def test_register_death_capture_is_a_product_of_the_trend_and_the_age_error(worlds):
    """Third of the three: mortality_improvement x age_reporting_error."""
    base = dict(worlds[3]["mechanisms"].coefficients, death_report_by_age_error=5.0)
    published = 0.22

    def late(improvement, age_error):
        return death_report_late_probability(
            dict(base, mortality_improvement=improvement, age_reporting_error=age_error),
            published)

    assert late(0.070, 3.0) > published > late(-0.028, 3.0)
    assert late(0.070, 1.0) == pytest.approx(published)
    assert late(-0.028, 1.0) == pytest.approx(published)
    # Neither axis alone moves it: the site reads the product.
    assert late(0.070, 3.0) > late(0.070, 1.6) > published

    # The ledger reads the same number the family publishes.
    built = worlds[3]
    context = built["history"]["branch"]["context"]
    assert context["death_late_probability"] == pytest.approx(
        death_report_late_probability(built["mechanisms"].coefficients, published))


def test_item_missingness_has_its_own_county_effect(worlds):
    """The coverage county effect was reused verbatim, so one estimate did for both."""
    base = worlds[3]["mechanisms"]
    assert "item_missing" in base.effects
    assert not np.allclose(base.effects["item_missing"], base.effects["coverage"])

    plain = _rates(base)
    moved_missing = _rates(_with(base, effects={
        "item_missing": base.effects["item_missing"] + 0.5}))
    moved_coverage = _rates(_with(base, effects={
        "coverage": base.effects["coverage"] + 0.5}))
    assert (moved_missing["item_missing"] > plain["item_missing"]).all()
    assert moved_missing["covered"] == pytest.approx(plain["covered"])
    assert (moved_coverage["covered"] > plain["covered"]).all()
    assert moved_coverage["item_missing"] == pytest.approx(plain["item_missing"])

    # The two effects are drawn from their own published spreads, not one vector twice.
    effects = draw_county_effects(11, 24, base.coefficients)
    assert not np.allclose(effects["item_missing"], effects["coverage"])


def test_health_inclusion_slope_is_scaled_by_administrative_completeness(worlds):
    """Second of the three: administrative_completeness x missingness_target_dependence."""
    base = worlds[3]["mechanisms"]
    n = base.county.n_counties
    county = np.arange(n, dtype=np.int64)

    def frailty_slope(completeness):
        mechanisms = _with(base, {"administrative_completeness": float(completeness)})

        def covered(frailty):
            return _record_mechanism_rates(
                SourceParams(), mechanisms, "health", county,
                np.full(n, 2, dtype=np.int8), np.full(n, frailty),
                np.full(n, 45.0), np.full(n, 0.90))["covered"]

        return float((covered(3.0) - covered(0.4)).mean())

    assert frailty_slope(2.5) > frailty_slope(0.4) > 0.0


def test_the_survey_draw_is_keyed_on_the_world_and_not_on_the_snapshot_tick(worlds,
                                                                           monkeypatch):
    """Development seeds run consecutively and so do the two snapshot ticks.

    The packet used to key the survey on the seed plus the tick, so worlds whose seeds
    and ticks summed to the same number drew the same households, the same nonresponse
    and the same reported error.
    """
    import meridia.packet as packet_module
    seen = []
    real = packet_module.draw_survey

    def spy(micro, population, seed, **kwargs):
        seen.append((int(seed), int(kwargs.get("vintage", 0))))
        return real(micro, population, seed, **kwargs)

    monkeypatch.setattr(packet_module, "draw_survey", spy)
    built = worlds[3]
    packet_module._survey_at(built, built["ticks"]["preliminary"], 0)
    packet_module._survey_at(built, built["ticks"]["revised"], 1)
    assert seen == [(built["seed"], 0), (built["seed"], 1)]
