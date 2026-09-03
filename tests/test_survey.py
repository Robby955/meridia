"""Survey instrument: design validity, mechanism direction, no leakage, determinism."""

import hashlib
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from meridia.hydrology import fill_depressions, flow_accumulation, flow_directions
from meridia.microdata import build_microdata
from meridia.population import build_population
from meridia.survey import (SURVEY_BANDS, SurveyParams, draw_survey,
                            draw_survey_params)
from meridia.terrain import generate_elevation

SEED = 777
H, W = 96, 128
TOTAL = 250_000


def _setup():
    world = generate_elevation(SEED, H, W)
    outlets = ~world["land"]
    outlets[0, :] = outlets[-1, :] = outlets[:, 0] = outlets[:, -1] = True
    filled = fill_depressions(world["elevation"], world["sea_level"])
    direction = flow_directions(filled, outlets)
    accumulation = flow_accumulation(direction, outlets)
    people = build_population(world, accumulation, TOTAL, 8)
    micro = build_microdata(people["population"], people["habitability"],
                            people["settlements"], SEED)
    survey = draw_survey(micro, people["population"], SEED)
    return people, micro, survey


def test_response_between_zero_and_full():
    _, _, survey = _setup()
    assert 0 < survey["n_responding_households"] < survey["n_sampled_households"]


def test_design_weights_positive_and_finite():
    _, _, survey = _setup()
    w = survey["survey"]["design_weight"]
    assert np.isfinite(w).all() and (w >= 1.0).all()


def test_nonresponse_is_income_selective():
    _, micro, survey = _setup()
    truth = survey["truth"]
    hh_income = np.bincount(micro["person"]["household"],
                            weights=micro["person"]["income"],
                            minlength=micro["n_households"])
    sampled = truth["sampled_households"]
    responded = truth["responded"]
    inc = np.log1p(hh_income[sampled])
    assert inc[responded].mean() < inc[~responded].mean()


def test_item_missingness_is_mnar_in_income():
    _, _, survey = _setup()
    truth = survey["truth"]
    adult = truth["age"] >= 16
    missing = truth["income_missing"] & adult
    observed = ~truth["income_missing"] & adult & (truth["income"] > 0)
    assert truth["income"][missing].mean() > truth["income"][observed].mean()


def test_survey_file_has_no_truth_columns():
    _, _, survey = _setup()
    assert set(survey["survey"].keys()) == {
        "household", "cell", "stratum", "design_weight", "age", "sex",
        "education", "income", "recent_hospitalization"}
    missing_income = np.isnan(survey["survey"]["income"]).sum()
    assert missing_income > 0  # pathologies actually present in the reported file


def test_health_anchor_is_misclassified_at_the_declared_rates():
    """The anchor is the only external handle on latent frailty, so its error rates are
    published and its reported value must not equal the truth."""
    people, micro, _ = _setup()
    rng = np.random.default_rng(0)
    admission = rng.random(len(micro["person"]["age"])) < 0.18
    survey = draw_survey(micro, people["population"], SEED,
                         recent_admission=admission)
    truth = survey["truth"]["recent_admission"]
    reported = survey["survey"]["recent_hospitalization"].astype(bool)
    assert truth.any() and (~truth).any()
    sensitivity = reported[truth].mean()
    specificity = (~reported[~truth]).mean()
    assert abs(sensitivity - SurveyParams().anchor_sensitivity) < 0.05
    assert abs(specificity - SurveyParams().anchor_specificity) < 0.05
    assert not np.array_equal(reported, truth)


def test_reported_income_differs_from_truth():
    _, _, survey = _setup()
    truth = survey["truth"]
    reported = survey["survey"]["income"]
    observed = ~np.isnan(reported) & (truth["income"] > 0)
    assert not np.allclose(reported[observed], truth["income"][observed])


def test_ht_estimate_reasonable():
    """Design-weighted person count should land near the true population."""
    people, _, survey = _setup()
    estimate = survey["survey"]["design_weight"].sum()
    # nonresponse biases this downward; it must still be the right order and below 3x
    assert 0.25 * TOTAL < estimate < 3.0 * TOTAL


def test_survey_deterministic():
    digests = []
    for _ in range(2):
        _, _, survey = _setup()
        blob = b"".join(np.ascontiguousarray(v).tobytes() for v in survey["survey"].values())
        digests.append(hashlib.sha256(blob).hexdigest())
    assert digests[0] == digests[1]


def test_the_survey_instrument_is_a_per_world_draw_inside_its_published_bands():
    """A world constant here is estimable once on a world that ships truth."""
    seen = {}
    for seed in (1101, 1105, 2101, 2103, 3101):
        drawn = draw_survey_params(seed)
        again = draw_survey_params(seed)
        for name, (low, high) in SURVEY_BANDS.items():
            value = getattr(drawn, name)
            assert low <= value <= high, (seed, name, value)
            assert value == getattr(again, name)
            seen.setdefault(name, set()).add(round(value, 9))
    for name, values in seen.items():
        assert len(values) == 5, name
    # The sample design and the anchor's declared error are not drawn: the design is
    # published and the anchor's error is what makes the anchor an anchor.
    fixed = draw_survey_params(1101)
    default = SurveyParams()
    for name in ("n_strata_rows", "n_strata_cols", "cells_per_stratum",
                 "households_per_cell", "anchor_sensitivity", "anchor_specificity"):
        assert getattr(fixed, name) == getattr(default, name)
