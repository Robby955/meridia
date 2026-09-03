"""Release contract and scoring: exact truth, schema, additivity, worst-group accuracy,
coverage with a proper interval score, and a disclosure audit by linear recovery."""

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from meridia.admin import build_admin, county_totals
from meridia.hydrology import fill_depressions, flow_accumulation, flow_directions
from meridia.microdata import build_microdata
from meridia.population import build_population, resource_outposts
from meridia.release import (AGE_BANDS, ESTIMAND_IDS, LEVELS, compute_detailed_table_truth,
                             compute_truth, required_rows)
from meridia.scoring import (check_additivity, disclosure_audit, evaluate_gates,
                             rows_from_values, score_release, validate_release)
from meridia.terrain import generate_elevation

SEED = 777
H, W = 96, 128
TOTAL = 250_000
SETTLEMENTS = 8

_CACHE = {}


def _setup():
    if "world" not in _CACHE:
        world = generate_elevation(SEED, H, W)
        outlets = ~world["land"]
        outlets[0, :] = outlets[-1, :] = outlets[:, 0] = outlets[:, -1] = True
        filled = fill_depressions(world["elevation"], world["sea_level"])
        direction = flow_directions(filled, outlets)
        accumulation = flow_accumulation(direction, outlets)
        people = build_population(world, accumulation, TOTAL, SETTLEMENTS, seed=SEED)
        micro = build_microdata(people["population"], people["habitability"],
                                people["settlements"], SEED)
        admin = build_admin(world["land"], people["settlements"],
                            resource_outposts(world, SEED), n_states=3)
        truth = compute_truth(micro["person"], micro["household_cell"], admin)
        _CACHE.update(world=world, people=people, micro=micro, admin=admin, truth=truth)
    return _CACHE


def _oracle_rows(truth, rel=0.02):
    return rows_from_values(truth, lambda e, v: rel if e.endswith("share") or
                            e.startswith("tertiary") else rel * max(abs(v), 1.0))


def test_truth_counts_match_the_population_grid_and_hierarchy():
    s = _setup()
    truth, admin, people = s["truth"], s["admin"], s["people"]
    assert truth[("persons", "nation", 0)] == TOTAL
    assert truth[("households", "nation", 0)] == s["micro"]["n_households"]
    by_county = county_totals(people["population"], admin["county"].flatten(), admin["n_counties"])
    for c in range(admin["n_counties"]):
        assert truth[("persons", "county", c)] == by_county[c]
    for e in ("persons", "households", "children_under_16", "elders_65_plus"):
        assert sum(truth[(e, "county", c)] for c in range(admin["n_counties"])) == truth[(e, "nation", 0)]
        assert sum(truth[(e, "state", st)] for st in range(admin["n_states"])) == truth[(e, "nation", 0)]
    assert truth[("children_under_16", "nation", 0)] + truth[("elders_65_plus", "nation", 0)] < TOTAL


def test_truth_covers_exactly_the_required_rows():
    s = _setup()
    assert set(s["truth"]) == required_rows(s["admin"])
    assert len(s["truth"]) == len(ESTIMAND_IDS) * (1 + s["admin"]["n_states"] + s["admin"]["n_counties"])
    for (e, level, u), v in s["truth"].items():
        if e.endswith("share") or e.startswith("tertiary"):
            assert math.isnan(v) or 0.0 <= v <= 1.0


def test_national_low_income_share_is_below_half():
    s = _setup()
    v = s["truth"][("low_income_household_share", "nation", 0)]
    assert 0.0 < v < 0.5


def test_oracle_release_passes_everything():
    s = _setup()
    rows = _oracle_rows(s["truth"])
    assert validate_release(rows, s["admin"]) == []
    assert check_additivity(rows, s["admin"]) == []
    metrics = score_release(rows, s["truth"], s["admin"])
    assert all(m["worst_error"] == 0.0 and m["coverage"] == 1.0 for m in metrics.values())
    bars = {"worst_error": {k: 0.01 for k in metrics}, "coverage_floor": 0.85}
    verdict = evaluate_gates([], [], metrics, None, bars)
    assert verdict["pass"], verdict["reasons"]


def test_schema_catches_missing_duplicate_and_malformed_rows():
    s = _setup()
    rows = _oracle_rows(s["truth"])
    errors = validate_release(rows[1:], s["admin"])
    assert any(e.startswith("missing row") for e in errors)
    errors = validate_release(rows + [rows[0]], s["admin"])
    assert any(e.startswith("duplicate row") for e in errors)
    bad = dict(rows[0], lower=rows[0]["estimate"] + 1.0)
    errors = validate_release([bad] + rows[1:], s["admin"])
    assert any("does not contain the estimate" in e for e in errors)
    bad = dict(rows[0], estimate=float("nan"))
    errors = validate_release([bad] + rows[1:], s["admin"])
    assert any("not a finite number" in e for e in errors)


def test_one_bad_county_fails_worst_group_but_not_the_mean():
    s = _setup()
    rows = _oracle_rows(s["truth"])
    victim = next(r for r in rows if r["estimand"] == "persons" and r["level"] == "county"
                  and r["unit"] == 0)
    victim["estimate"] *= 1.5
    victim["upper"] = victim["estimate"] * 1.02
    victim["lower"] = victim["estimate"] * 0.98
    metrics = score_release(rows, s["truth"], s["admin"])
    m = metrics["persons/county"]
    assert m["worst_error"] > 0.49 and m["worst_unit"] == 0
    assert m["mean_error"] < 0.1
    assert m["coverage"] < 1.0
    verdict = evaluate_gates([], [], metrics, None, {"worst_error": {"persons/county": 0.05}})
    assert not verdict["pass"] and verdict["reasons"][0].startswith("accuracy")
    assert check_additivity(rows, s["admin"])   # the state total no longer adds up


def test_inflated_intervals_keep_coverage_but_lose_the_interval_score():
    s = _setup()
    tight = score_release(_oracle_rows(s["truth"], rel=0.02), s["truth"], s["admin"])
    wide = score_release(_oracle_rows(s["truth"], rel=0.50), s["truth"], s["admin"])
    for key in tight:
        assert wide[key]["coverage"] == 1.0
        assert wide[key]["mean_interval_score"] > tight[key]["mean_interval_score"]


def test_missed_truth_is_penalised_more_than_width():
    s = _setup()
    rows = _oracle_rows(s["truth"], rel=0.02)
    shifted = [dict(r) for r in rows]
    for r in shifted:
        if r["estimand"] == "persons":
            r["estimate"] *= 1.10
            r["lower"] *= 1.10
            r["upper"] *= 1.10
    honest = score_release(rows, s["truth"], s["admin"])["persons/county"]
    off = score_release(shifted, s["truth"], s["admin"])["persons/county"]
    assert off["coverage"] == 0.0
    assert off["mean_interval_score"] > 5 * honest["mean_interval_score"]


def test_detailed_table_truth_is_exact():
    s = _setup()
    table = compute_detailed_table_truth(s["micro"]["person"], s["admin"])
    assert table.shape == (s["admin"]["n_counties"], len(AGE_BANDS), 2)
    assert table.sum() == TOTAL
    for c in range(s["admin"]["n_counties"]):
        assert table[c].sum() == s["truth"][("persons", "county", c)]


def test_disclosure_audit_detects_published_protected_and_single_subtraction():
    truth = np.array([[[3, 40], [50, 60]],
                      [[70, 80], [90, 100]]], dtype=np.int64)   # cell (0,0,0) is protected
    threshold = 5
    published_all = truth.astype(float)
    audit = disclosure_audit(published_all, truth, threshold)
    assert not audit["pass"] and audit["published_protected"] == [(0, 0, 0)]

    suppressed = published_all.copy()
    suppressed[0, 0, 0] = np.nan
    audit = disclosure_audit(suppressed, truth, threshold)
    assert audit["pass"]   # no totals published, nothing to difference against
    marginals = {"county_age": truth.sum(axis=2).astype(float)}
    audit = disclosure_audit(suppressed, truth, threshold, marginals)
    assert not audit["pass"] and audit["recoverable"] == [(0, 0, 0)]


def test_disclosure_audit_accepts_complementary_suppression_and_catches_linear_recovery():
    truth = np.array([[[3, 40], [50, 60]],
                      [[70, 80], [90, 100]]], dtype=np.int64)
    threshold = 5
    marginals = {"county_age": truth.sum(axis=2).astype(float),
                 "county_sex": truth.sum(axis=1).astype(float),
                 "county": truth.sum(axis=(1, 2)).astype(float),
                 "age_sex": truth.sum(axis=0).astype(float)}
    published = truth.astype(float)
    # Suppress the protected cell plus a 2x2 complement inside county 0: no line has a
    # single hole, and no linear combination of the county-0 totals isolates the cell.
    for cell in [(0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1)]:
        published[cell] = np.nan
    audit = disclosure_audit(published, truth, threshold, marginals)
    # The national age x sex totals give the county-0 cells directly (county 1 is
    # fully published): linear recovery that no single subtraction within a line shows.
    assert not audit["pass"] and audit["recoverable"] == [(0, 0, 0)]
    # Withhold the age x sex national totals and the pattern is safe.
    safe = {k: v for k, v in marginals.items() if k != "age_sex"}
    audit = disclosure_audit(published, truth, threshold, safe)
    assert audit["pass"], audit
    # An inconsistent published total is a failure too.
    wrong = dict(safe)
    wrong["county"] = safe["county"].copy()
    wrong["county"][1] += 1.0
    audit = disclosure_audit(published, truth, threshold, wrong)
    assert not audit["pass"] and audit["inconsistent"]


def test_levels_and_estimands_are_frozen_names():
    assert LEVELS == ("nation", "state", "county")
    assert "persons" in ESTIMAND_IDS and len(set(ESTIMAND_IDS)) == len(ESTIMAND_IDS)


def test_a_suppress_everything_table_protects_everything_and_fails_the_utility_gate():
    """Disclosure protection is one-sided, so the gate needs a utility requirement."""
    truth = np.array([[[12, 3], [40, 9]], [[7, 60], [2, 25]]])
    blank = np.full(truth.shape, np.nan)
    audit = disclosure_audit(blank, truth, threshold=10)
    assert audit["pass"] is True
    assert audit["n_releasable"] == 4
    assert audit["n_published_releasable"] == 0
    assert audit["utility"] == 0.0
    assert evaluate_gates([], [], {}, audit, None)["pass"] is True
    gated = evaluate_gates([], [], {}, audit, {"disclosure_utility_floor": 0.8})
    assert gated["pass"] is False
    assert any(r.startswith("disclosure utility") for r in gated["reasons"])

    published = truth.astype(float)
    published[truth < 10] = np.nan
    honest = disclosure_audit(published, truth, threshold=10)
    assert honest["utility"] == 1.0
    assert evaluate_gates([], [], {}, honest,
                          {"disclosure_utility_floor": 0.8})["pass"] is True


def test_the_released_detailed_cells_carry_an_accuracy_bar_of_their_own():
    """A share alone is met by publishing every releasable cell as any number at all."""
    truth = np.array([[[12, 3], [40, 9]], [[7, 60], [2, 25]]])
    bars = {"disclosure_utility_floor": 0.8, "detailed_accuracy_ceiling": 0.25}

    honest = truth.astype(float)
    honest[truth < 10] = np.nan
    audit = disclosure_audit(honest, truth, threshold=10)
    assert audit["n_scored"] == 4
    assert audit["detailed_error"] == 0.0
    assert evaluate_gates([], [], {}, audit, bars)["pass"] is True

    # Every releasable cell published, none of them near its own count. The utility share
    # reads one, and only the accuracy bar refuses it.
    garbage = np.where(truth >= 10, 1.0, np.nan)
    audit = disclosure_audit(garbage, truth, threshold=10)
    assert audit["pass"] is True and audit["utility"] == 1.0
    assert audit["detailed_error"] > 0.25
    assert evaluate_gates([], [], {}, audit, {"disclosure_utility_floor": 0.8})["pass"] \
        is True
    verdict = evaluate_gates([], [], {}, audit, bars)
    assert verdict["pass"] is False
    assert any(r.startswith("detailed accuracy") for r in verdict["reasons"])

    # A blank table has nothing to score, and the utility floor is what refuses it.
    blank = disclosure_audit(np.full(truth.shape, np.nan), truth, threshold=10)
    assert blank["n_scored"] == 0 and math.isnan(blank["detailed_error"])
    assert evaluate_gates([], [], {}, blank, bars)["pass"] is False
