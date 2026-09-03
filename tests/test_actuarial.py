"""Every version-four actuarial formula checked against a hand-computed answer.

The ledger here is three persons in two households across two counties over three months,
small enough that exposure, deaths, incidence, and the discounted obligation can all be
written out by hand in the test itself. The ensemble, tail, and reserve formulas are then
checked on a four-by-two liability matrix whose quantiles and shortfalls are exact.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from meridia.actuarial import (ACTUARIAL_AGE_BAND_LABELS, EXPOSURE_ESTIMAND,
                               INCIDENCE_ESTIMAND, MORTALITY_ESTIMAND,
                               ActuarialThresholds, ContinuationEnsemble,
                               ObligationContract, actuarial_pass,
                               build_continuation_ensemble, calibration_criteria,
                               check_rate_additivity, continuation_member_key,
                               eligibility_floor, ensemble_truth,
                               evaluate_actuarial_gates,
                               exceedance_probabilities, expected_uncovered,
                               exposure_and_rate_truth, interval_score,
                               liabilities_from_pass, parse_rate_rows, parse_reserve_rows,
                               perfect_information_allocation,
                               proportional_baseline_allocation, quantile_score,
                               relative_error, reserve_total, score_rates, score_reserve,
                               shortfall_error, skill_score)
from meridia.events import EVENT_TYPES, _EVENT_DTYPES

MALE, FEMALE = 0, 1


# ------------------------------------------------------------------ the tiny ledger

def tiny_admin() -> dict:
    """Two counties on two cells, one county per state, so a region is a county."""
    return {"n_counties": 2, "n_states": 2,
            "county_state": np.asarray([0, 1], dtype=np.int64),
            "county": np.asarray([[0, 1]], dtype=np.int64)}


def tiny_start_state() -> dict:
    """Tick zero: an elder and an adult in county 0, an older elder in county 1."""
    return {
        "person": {
            "truth_person_id": np.asarray([101, 102, 103], dtype=np.uint64),
            "truth_household_id": np.asarray([201, 201, 202], dtype=np.uint64),
            "birth_tick": np.asarray([-840, -360, -960], dtype=np.int64),
            "sex": np.asarray([MALE, FEMALE, MALE], dtype=np.int8),
            "is_alive": np.asarray([True, True, True]),
        },
        "household": {
            "truth_household_id": np.asarray([201, 202], dtype=np.uint64),
            "cell": np.asarray([0, 1], dtype=np.int64),
        },
    }


def _event_table(rows: list[dict]) -> dict:
    table = {name: np.zeros(len(rows), dtype=dtype) for name, dtype in _EVENT_DTYPES.items()}
    for i, row in enumerate(rows):
        table["truth_event_id"][i] = i + 1
        for name, value in row.items():
            table[name][i] = value
    return table


def tiny_events() -> dict:
    """Tick 2: the county-1 elder dies. Tick 3: household 201 moves to county 1, a child
    is born into it, and the adult has a first qualifying admission there."""
    return _event_table([
        {"tick": 1, "event_type": EVENT_TYPES["encounter_admitted"],
         "truth_person_id": 102, "diagnosis_group": 3},          # not a qualifying group
        {"tick": 2, "event_type": EVENT_TYPES["person_death"],
         "truth_person_id": 103, "truth_household_id": 202, "from_cell": 1},
        {"tick": 3, "event_type": EVENT_TYPES["household_moved"],
         "truth_household_id": 201, "from_cell": 0, "to_cell": 1},
        {"tick": 3, "event_type": EVENT_TYPES["person_birth"], "truth_person_id": 104,
         "truth_household_id": 201, "to_cell": 1, "birth_tick": 3, "sex": FEMALE},
        {"tick": 3, "event_type": EVENT_TYPES["encounter_admitted"],
         "truth_person_id": 102, "diagnosis_group": 7},
        {"tick": 3, "event_type": EVENT_TYPES["encounter_admitted"],
         "truth_person_id": 102, "diagnosis_group": 7},          # not the first, ignored
    ])


UNDISCOUNTED = ObligationContract(monthly_benefit=100.0, eligibility_min_age=65,
                                  qualifying_event_cost=1000.0, death_benefit=500.0,
                                  monthly_discount_rate=0.0, horizon_months=3,
                                  qualifying_diagnosis_groups=(7,))


def tiny_pass(contract: ObligationContract = UNDISCOUNTED) -> dict:
    return actuarial_pass(tiny_start_state(), tiny_events(), tiny_admin(),
                          start_tick=0, months=3, contract=contract)


# ----------------------------------------------------------- the person-month ledger

def test_person_months_follow_residence_age_and_survival():
    band = {label: i for i, label in enumerate(ACTUARIAL_AGE_BAND_LABELS)}
    exposure = tiny_pass()["exposure_person_months"]
    # County 0: the elder and the adult for months 1 and 2, then they move away.
    assert exposure[0, MALE, band["65-74"]] == 2.0
    assert exposure[0, FEMALE, band["18-44"]] == 2.0
    # County 1: the older elder for month 1 only, then the arrivals in month 3.
    assert exposure[1, MALE, band["75-84"]] == 1.0
    assert exposure[1, MALE, band["65-74"]] == 1.0
    assert exposure[1, FEMALE, band["18-44"]] == 1.0
    assert exposure[1, FEMALE, band["0-17"]] == 1.0
    assert exposure.sum() == 8.0          # 3 + 2 + 3 living person-months


def test_deaths_and_first_qualifying_events_land_in_the_right_cell():
    band = {label: i for i, label in enumerate(ACTUARIAL_AGE_BAND_LABELS)}
    result = tiny_pass()
    assert result["deaths"].sum() == 1.0
    assert result["deaths"][1, MALE, band["75-84"]] == 1.0
    # One qualifying admission is counted: the tick-1 one is the wrong diagnosis group
    # and the second tick-3 one is not the person's first.
    assert result["qualifying_events"].sum() == 1.0
    assert result["qualifying_events"][1, FEMALE, band["18-44"]] == 1.0


def test_cash_flows_are_charged_to_the_region_of_residence_that_month():
    result = tiny_pass()
    assert result["benefit"].tolist() == [[100.0, 100.0], [100.0, 0.0], [0.0, 100.0]]
    assert result["death_benefit"].tolist() == [[0.0, 0.0], [0.0, 500.0], [0.0, 0.0]]
    assert result["event_cost"].tolist() == [[0.0, 0.0], [0.0, 0.0], [0.0, 1000.0]]


def test_undiscounted_liability_is_the_sum_of_the_three_cash_flows():
    liability = liabilities_from_pass(tiny_pass(), UNDISCOUNTED)
    assert liability.tolist() == [200.0, 1700.0]


def test_discount_factors_price_each_month_separately():
    contract = ObligationContract(monthly_benefit=100.0, eligibility_min_age=65,
                                  qualifying_event_cost=1000.0, death_benefit=500.0,
                                  monthly_discount_rate=0.01, horizon_months=3,
                                  qualifying_diagnosis_groups=(7,))
    v1, v2, v3 = 1 / 1.01, 1 / 1.01 ** 2, 1 / 1.01 ** 3
    liability = liabilities_from_pass(tiny_pass(contract), contract)
    assert liability[0] == pytest.approx(100 * v1 + 100 * v2)
    assert liability[1] == pytest.approx(100 * v1 + 500 * v2 + 1100 * v3)


def test_exposure_and_rates_are_occurrence_over_exposure_in_person_years():
    truth = exposure_and_rate_truth(tiny_pass(), tiny_admin())
    assert truth[(EXPOSURE_ESTIMAND, "county", 0, "male", "65-74")] == pytest.approx(2 / 12)
    assert truth[(EXPOSURE_ESTIMAND, "county", 1, "male", "75-84")] == pytest.approx(1 / 12)
    # One death against one person-month of exposure is twelve per person-year.
    assert truth[(MORTALITY_ESTIMAND, "county", 1, "male", "75-84")] == pytest.approx(12.0)
    assert truth[(INCIDENCE_ESTIMAND, "county", 1, "female", "18-44")] == pytest.approx(12.0)
    # A cell with no exposure has no rate.
    assert math.isnan(truth[(MORTALITY_ESTIMAND, "county", 0, "male", "0-17")])
    # Broad bands are unions of actuarial bands, and one county is one state here.
    assert truth[(EXPOSURE_ESTIMAND, "county", 0, "male", "65+")] == pytest.approx(2 / 12)
    assert truth[(EXPOSURE_ESTIMAND, "county", 0, "female", "18-64")] == pytest.approx(2 / 12)
    assert truth[(EXPOSURE_ESTIMAND, "state", 1, "female", "0-17")] == pytest.approx(1 / 12)


def test_a_superseded_event_never_enters_the_pass():
    events = tiny_events()
    events["supersedes_event_id"][5] = 2       # the tick-2 death is superseded
    result = actuarial_pass(tiny_start_state(), events, tiny_admin(), 0, 3, UNDISCOUNTED)
    assert result["deaths"].sum() == 0.0
    assert result["death_benefit"].sum() == 0.0
    assert result["exposure_person_months"].sum() == 10.0    # the elder lives all three


# --------------------------------------------------------------- continuation ensemble

LIABILITY = np.asarray([[10.0, 100.0], [20.0, 200.0], [30.0, 300.0], [40.0, 400.0]])


def test_member_keys_are_distinct_across_members_and_months():
    seed = 12345
    a = continuation_member_key(seed, 0, 1).generate_state(4)
    b = continuation_member_key(seed, 1, 1).generate_state(4)
    c = continuation_member_key(seed, 0, 2).generate_state(4)
    assert not np.array_equal(a, b) and not np.array_equal(a, c)
    # The rule is a pure function of the arguments, so a member replays byte for byte.
    assert np.array_equal(a, continuation_member_key(seed, 0, 1).generate_state(4))
    with pytest.raises(ValueError):
        continuation_member_key(seed, -1, 1)


def test_ensemble_truth_is_the_mean_the_quantile_and_the_mean_above_it():
    truth = ensemble_truth(LIABILITY, 0.95)
    assert truth["mean"].tolist() == [25.0, 250.0]
    assert truth["q"].tolist() == [40.0, 400.0]
    assert truth["es"].tolist() == [40.0, 400.0]


def test_the_ensemble_is_assembled_member_by_member_with_one_realized_future():
    ensemble = build_continuation_ensemble(lambda m: LIABILITY[m], n_members=4,
                                           realized_member=2)
    assert ensemble.n_members == 4 and ensemble.n_regions == 2
    assert ensemble.realized().tolist() == [30.0, 300.0]
    assert ensemble.truth()["q"].tolist() == [40.0, 400.0]
    with pytest.raises(ValueError):
        build_continuation_ensemble(lambda m: LIABILITY[m], n_members=1)


def test_a_continuation_that_changes_width_is_rejected():
    def ragged(m: int) -> np.ndarray:
        return LIABILITY[m] if m else LIABILITY[m][:1]
    with pytest.raises(ValueError):
        build_continuation_ensemble(ragged, n_members=3)


# --------------------------------------------------------------------------- tail gates

def test_exceedance_probability_counts_continuations_above_the_submitted_quantile():
    p = exceedance_probabilities(np.asarray([25.0, 250.0]), LIABILITY)
    assert p.tolist() == [0.5, 0.5]
    assert exceedance_probabilities(np.asarray([40.0, 400.0]), LIABILITY).tolist() == [0.0, 0.0]


def test_calibration_criteria_measure_distance_from_the_nominal_five_percent():
    criteria = calibration_criteria(np.asarray([0.5, 0.5]), 0.95, 0.90)
    assert criteria["target"] == pytest.approx(0.05)
    assert criteria["pooled"] == pytest.approx(0.45)
    assert criteria["worst"] == pytest.approx(0.45)


def test_quantile_score_punishes_padding_as_well_as_shortfall():
    padded = quantile_score(np.asarray([40.0, 400.0]), LIABILITY, np.asarray([1.0, 1.0]))
    # Every continuation sits at or below the quantile: 0.05 * (q - y), averaged.
    assert padded.tolist() == pytest.approx([0.75, 7.5])
    tight = quantile_score(np.asarray([30.0, 300.0]), LIABILITY, np.asarray([1.0, 1.0]))
    # 0.05 * (20 + 10 + 0) / 4 for the three at or below, 0.95 * 10 / 4 for the one above.
    assert tight[0] == pytest.approx((0.05 * 30 + 0.95 * 10) / 4)
    assert quantile_score(np.asarray([30.0, 300.0]), LIABILITY,
                          np.asarray([2.0, 2.0]))[0] == pytest.approx(tight[0] / 2)


def test_shortfall_error_is_normalized_against_the_ensemble_tail_mean():
    truth = ensemble_truth(LIABILITY, 0.95)
    error = shortfall_error(np.asarray([50.0, 380.0]), truth["es"], truth["q"])
    assert error.tolist() == pytest.approx([10.0 / 40.0, 20.0 / 400.0])


# ----------------------------------------------------------- reserve and the decision

RESERVE_THRESHOLDS = ActuarialThresholds(reserve_rounding_unit=10.0)


def test_reserve_total_adds_a_public_share_of_the_tail_and_rounds_up():
    q = np.asarray([40.0, 400.0])
    assert reserve_total(q, q, RESERVE_THRESHOLDS) == pytest.approx(440.0)
    # 440 + 0.25 * (10 + 100) = 467.5, rounded up to the next multiple of ten.
    assert reserve_total(q, np.asarray([50.0, 500.0]), RESERVE_THRESHOLDS) == pytest.approx(470.0)


def test_the_practical_baseline_covers_every_quantile_and_spreads_the_slack():
    baseline = proportional_baseline_allocation(np.asarray([40.0, 400.0]), 550.0)
    assert baseline.tolist() == pytest.approx([50.0, 500.0])
    assert float(baseline.sum()) == pytest.approx(550.0)


def test_expected_uncovered_obligation_is_the_weighted_mean_positive_part():
    j = expected_uncovered(np.asarray([25.0, 250.0]), LIABILITY)
    assert j == pytest.approx((0 + 0 + 5 + 15) / 4 + (0 + 0 + 50 + 150) / 4)
    weighted = expected_uncovered(np.asarray([25.0, 250.0]), LIABILITY,
                                  np.asarray([2.0, 0.0]))
    assert weighted == pytest.approx(2 * 5.0)


def test_the_perfect_information_allocation_beats_every_split_of_the_same_total():
    symmetric = np.asarray([[10.0, 10.0], [20.0, 20.0], [30.0, 30.0], [40.0, 40.0]])
    oracle = perfect_information_allocation(symmetric, 60.0)
    assert float(oracle.sum()) == pytest.approx(60.0)
    # Every slope-one segment first, then every slope-three-quarters segment, and so on.
    assert oracle.tolist() == pytest.approx([30.0, 30.0])
    j_oracle = expected_uncovered(oracle, symmetric)
    assert j_oracle == pytest.approx(2.5 + 2.5)
    for split in ([40.0, 20.0], [20.0, 40.0], [50.0, 10.0]):
        assert expected_uncovered(np.asarray(split), symmetric) >= j_oracle


def test_the_oracle_stands_on_the_submission_own_floors():
    """Skill has to measure the allocation, not the floors the tail gates already score.

    An oracle free of the submitted quantiles would beat any feasible allocation by the
    distance between the floors and the truth, so every submission would score near zero
    and the decision gate would carry nothing. Constrained to the same floors, the oracle
    spends only what the submission has left to spend.
    """
    liability = np.array([[10.0, 100.0], [20.0, 200.0], [30.0, 300.0], [40.0, 400.0]])
    floors = np.array([25.0, 250.0])
    total = 300.0
    free = perfect_information_allocation(liability, total)
    floored = perfect_information_allocation(liability, total, floors=floors)
    assert (floored >= floors - 1e-9).all()
    assert abs(floored.sum() - total) < 1e-9 and abs(free.sum() - total) < 1e-9
    assert expected_uncovered(floored, liability) >= expected_uncovered(free, liability)
    zero = perfect_information_allocation(liability, total, floors=np.zeros(2))
    assert np.allclose(zero, free)
    # Floors that already spend the total leave no decision, and the oracle says so.
    tight = perfect_information_allocation(liability, total,
                                           floors=np.array([100.0, 250.0]))
    assert abs(tight.sum() - total) < 1e-9


def test_the_oracle_spends_the_total_where_the_exceedance_is_widest():
    # Region one is ten times region zero, so its slope-one segment is ten times as long
    # and the whole slack after region zero's first ten units belongs to it.
    oracle = perfect_information_allocation(LIABILITY, 60.0)
    assert oracle.tolist() == pytest.approx([10.0, 50.0])
    assert expected_uncovered(oracle, LIABILITY) == pytest.approx(15.0 + 200.0)
    assert expected_uncovered(np.asarray([30.0, 30.0]), LIABILITY) == pytest.approx(222.5)


def test_skill_is_one_at_the_oracle_and_zero_at_the_baseline():
    assert skill_score(5.0, 7.5, 5.0) == pytest.approx(1.0)
    assert skill_score(7.5, 7.5, 5.0) == pytest.approx(0.0)
    assert skill_score(6.25, 7.5, 5.0) == pytest.approx(0.5)
    assert math.isnan(skill_score(5.0, 5.0, 5.0))


def test_score_reserve_rejects_an_allocation_that_misses_the_fixed_total():
    report = score_reserve(np.asarray([25.0, 250.0]), np.asarray([25.0, 250.0]),
                           np.asarray([40.0, 400.0]), np.asarray([25.0, 250.0]),
                           LIABILITY, total=550.0, thresholds=RESERVE_THRESHOLDS)
    assert not report["feasible"]
    assert any("sum to" in reason for reason in report["feasibility_reasons"])


def test_score_reserve_rejects_an_allocation_below_the_submitted_quantile():
    report = score_reserve(np.asarray([10.0, 540.0]), np.asarray([40.0, 400.0]),
                           np.asarray([40.0, 400.0]), np.asarray([25.0, 250.0]),
                           LIABILITY, total=550.0, thresholds=RESERVE_THRESHOLDS)
    assert not report["feasible"]
    assert any("below the region's own submitted q95" in r
               for r in report["feasibility_reasons"])


def test_score_reserve_reports_skill_against_baseline_and_oracle():
    q_hat = np.asarray([10.0, 10.0])
    baseline = proportional_baseline_allocation(q_hat, 60.0)
    assert baseline.tolist() == pytest.approx([30.0, 30.0])
    report = score_reserve(baseline, q_hat, np.asarray([40.0, 400.0]),
                           np.asarray([25.0, 250.0]), LIABILITY, total=60.0,
                           thresholds=RESERVE_THRESHOLDS)
    assert report["feasible"]
    assert report["J"] == pytest.approx(222.5)
    assert report["J_baseline"] == pytest.approx(222.5)
    assert report["J_oracle"] == pytest.approx(215.0)
    assert report["skill"] == pytest.approx(0.0)
    oracle = score_reserve(np.asarray([10.0, 50.0]), q_hat, np.asarray([40.0, 400.0]),
                           np.asarray([25.0, 250.0]), LIABILITY, total=60.0,
                           thresholds=RESERVE_THRESHOLDS)
    assert oracle["skill"] == pytest.approx(1.0)


# ------------------------------------------------------------------ exposure and rates

def test_relative_error_uses_the_frozen_stabilizer_not_a_count_scale():
    assert relative_error(11.0, 10.0, 1.0) == pytest.approx(1.0 / 11.0)
    # A rate of order 1e-3 stays measurable: the count scaler of one would flatten it.
    assert relative_error(2.0e-3, 1.0e-3, 5.0e-4) == pytest.approx(1.0e-3 / 1.5e-3)
    with pytest.raises(ValueError):
        relative_error(1.0, 1.0, 0.0)


def test_interval_score_charges_width_and_misses():
    assert interval_score(8.0, 12.0, 10.0, 1.0) == pytest.approx(4.0)
    assert interval_score(8.0, 12.0, 15.0, 1.0) == pytest.approx(4.0 + 20.0 * 3.0)
    assert interval_score(8.0, 12.0, 15.0, 2.0) == pytest.approx((4.0 + 60.0) / 2.0)


def _rate_truth_and_submission(exposure: float = 10_000.0, rate: float = 0.01):
    truth = {(EXPOSURE_ESTIMAND, "state", 0, "male", "65-74"): exposure,
             (MORTALITY_ESTIMAND, "state", 0, "male", "65-74"): rate}
    parsed = {(EXPOSURE_ESTIMAND, "state", 0, "male", "65-74"): (exposure, exposure, exposure),
              (MORTALITY_ESTIMAND, "state", 0, "male", "65-74"): (rate, rate * 0.9, rate * 1.1)}
    return truth, parsed


def test_score_rates_reads_only_cells_with_enough_exposure():
    thresholds = ActuarialThresholds()
    truth, parsed = _rate_truth_and_submission(exposure=10_000.0)
    metrics = score_rates(parsed, truth, thresholds)
    assert metrics[f"{MORTALITY_ESTIMAND}/state"]["n_cells"] == 1
    assert metrics[f"{MORTALITY_ESTIMAND}/state"]["gated"] is True
    assert metrics[f"{MORTALITY_ESTIMAND}/state"]["percentile_error"] == pytest.approx(0.0)
    assert metrics[f"{MORTALITY_ESTIMAND}/state"]["coverage"] == pytest.approx(1.0)
    thin_truth, thin_parsed = _rate_truth_and_submission(exposure=100.0)
    assert score_rates(thin_parsed, thin_truth, thresholds) == {}


def test_score_rates_gates_exposure_at_county_and_rates_at_state():
    thresholds = ActuarialThresholds(exposure_eligibility_person_years=0.0)
    truth = {(EXPOSURE_ESTIMAND, "county", 0, "male", "65+"): 100.0,
             (EXPOSURE_ESTIMAND, "county", 0, "male", "65-74"): 100.0,
             (EXPOSURE_ESTIMAND, "state", 0, "male", "65-74"): 100.0,
             (MORTALITY_ESTIMAND, "county", 0, "male", "65-74"): 0.01,
             (MORTALITY_ESTIMAND, "state", 0, "male", "65-74"): 0.01}
    parsed = {k: (v, v, v) for k, v in truth.items()}
    metrics = score_rates(parsed, truth, thresholds)
    assert metrics[f"{EXPOSURE_ESTIMAND}/county"]["gated"] is True
    assert metrics[f"{EXPOSURE_ESTIMAND}/state"]["gated"] is False
    assert metrics[f"{MORTALITY_ESTIMAND}/state"]["gated"] is True
    assert metrics[f"{MORTALITY_ESTIMAND}/county"]["gated"] is False


def test_exposure_adds_over_bands_and_counties_and_rates_are_exempt():
    admin = tiny_admin()
    parsed = {}
    for level, units in (("county", 2), ("state", 2)):
        for u in range(units):
            for sex in ("male", "female"):
                for band in ACTUARIAL_AGE_BAND_LABELS:
                    parsed[(EXPOSURE_ESTIMAND, level, u, sex, band)] = (10.0, 10.0, 10.0)
                parsed[(EXPOSURE_ESTIMAND, level, u, sex, "18-64")] = (20.0, 20.0, 20.0)
                parsed[(EXPOSURE_ESTIMAND, level, u, sex, "65+")] = (30.0, 30.0, 30.0)
    assert check_rate_additivity(parsed, admin) == []
    parsed[(EXPOSURE_ESTIMAND, "county", 0, "male", "65+")] = (31.0, 31.0, 31.0)
    errors = check_rate_additivity(parsed, admin)
    assert len(errors) == 1 and "65+" in errors[0]


# ------------------------------------------------------------------ submission parsing

def _reserve_rows(values: list[tuple[float, float, float, float]]) -> list[dict]:
    return [{"region": r, "liability_mean": m, "q95": q, "es95": e, "allocation": a}
            for r, (m, q, e, a) in enumerate(values)]


def test_reserve_rows_parse_and_report_their_own_violations():
    parsed, errors = parse_reserve_rows(_reserve_rows([(25.0, 40.0, 40.0, 50.0),
                                                       (250.0, 400.0, 400.0, 500.0)]), 2)
    assert errors == []
    assert parsed["q95"].tolist() == [40.0, 400.0]
    assert parsed["allocation"].tolist() == [50.0, 500.0]
    _, missing = parse_reserve_rows(_reserve_rows([(25.0, 40.0, 40.0, 50.0)]), 2)
    assert missing == ["missing reserve row for region 1"]
    _, ordered = parse_reserve_rows(_reserve_rows([(25.0, 20.0, 40.0, 50.0)]), 1)
    assert any("q95 below the liability mean" in e for e in ordered)
    _, negative = parse_reserve_rows(_reserve_rows([(25.0, 40.0, 40.0, -1.0)]), 1)
    assert any("allocation is not a finite non-negative number" in e for e in negative)


def test_rate_rows_report_missing_unexpected_and_malformed_rows():
    admin = {"n_counties": 1, "n_states": 1, "county_state": np.asarray([0])}
    rows = [{"estimand": EXPOSURE_ESTIMAND, "level": "county", "unit": 0, "sex": "male",
             "age_band": "65+", "estimate": 1.0, "lower": 2.0, "upper": 3.0}]
    parsed, errors = parse_rate_rows(rows, admin)
    assert parsed == {}
    assert any("interval does not contain the estimate" in e for e in errors)
    assert any("missing rate row" in e for e in errors)
    bad = [{"estimand": "not_an_estimand", "level": "county", "unit": 0, "sex": "male",
            "age_band": "65+", "estimate": 1.0, "lower": 1.0, "upper": 1.0}]
    _, unknown = parse_rate_rows(bad, admin)
    assert any("unknown estimand" in e for e in unknown)


# ------------------------------------------------------------------------------ gates

def _reserve_report(q_hat: np.ndarray, allocation: np.ndarray, total: float,
                    es_hat: np.ndarray | None = None) -> dict:
    truth = ensemble_truth(LIABILITY, 0.95)
    return score_reserve(allocation, q_hat,
                         truth["es"] if es_hat is None else es_hat,
                         truth["mean"], LIABILITY, total=total,
                         thresholds=RESERVE_THRESHOLDS)


def test_a_too_low_quantile_fails_on_exceedances():
    q_hat = np.asarray([20.0, 200.0])
    report = _reserve_report(q_hat, proportional_baseline_allocation(q_hat, 550.0), 550.0)
    verdict = evaluate_actuarial_gates([], {}, [], report, RESERVE_THRESHOLDS)
    assert not verdict["pass"]
    assert any(r.startswith("tail: pooled exceedance deviation") for r in verdict["reasons"])


def test_a_padded_quantile_fails_on_the_proper_score():
    q_hat = np.asarray([400.0, 4000.0])
    report = _reserve_report(q_hat, proportional_baseline_allocation(q_hat, 4400.0), 4400.0)
    verdict = evaluate_actuarial_gates([], {}, [], report, RESERVE_THRESHOLDS)
    assert not verdict["pass"]
    assert any(r.startswith("tail: quantile score") for r in verdict["reasons"])


def test_a_good_forecast_with_the_baseline_allocation_fails_the_decision_gate():
    q_hat = np.asarray([10.0, 10.0])
    report = _reserve_report(q_hat, proportional_baseline_allocation(q_hat, 60.0), 60.0)
    verdict = evaluate_actuarial_gates([], {}, [], report, RESERVE_THRESHOLDS)
    assert report["skill"] == pytest.approx(0.0)
    assert any(r.startswith("reserve: skill") for r in verdict["reasons"])


def test_a_world_where_the_oracle_ties_the_baseline_reports_no_attainable_value():
    q_hat = np.asarray([40.0, 400.0])
    report = _reserve_report(q_hat, proportional_baseline_allocation(q_hat, 550.0), 550.0)
    assert report["J_baseline"] == pytest.approx(report["J_oracle"])
    verdict = evaluate_actuarial_gates([], {}, [], report, RESERVE_THRESHOLDS)
    assert any("skill is undefined" in r for r in verdict["reasons"])


def test_schema_violations_surface_as_named_families():
    verdict = evaluate_actuarial_gates(["missing rate row"], {}, ["missing reserve row"],
                                       None, RESERVE_THRESHOLDS)
    assert verdict["reasons"] == ["rate schema: 1 violation(s)", "reserve schema: 1 violation(s)"]


def test_a_rate_block_over_its_ceiling_fails_with_its_own_family_name():
    metrics = {f"{MORTALITY_ESTIMAND}/state": {
        "n_cells": 4, "percentile_error": 0.9, "max_error": 0.9, "worst_cell": (0, "male", "65-74"),
        "coverage": 1.0, "mean_interval_score": 0.1, "gated": True}}
    verdict = evaluate_actuarial_gates([], metrics, [], None, ActuarialThresholds())
    assert any(r.startswith("rate: mortality_rate/state percentile error")
               for r in verdict["reasons"])
    exposure_metrics = {f"{EXPOSURE_ESTIMAND}/county": {
        "n_cells": 4, "percentile_error": 0.0, "max_error": 0.0, "worst_cell": (0, "male", "65+"),
        "coverage": 0.1, "mean_interval_score": 0.1, "gated": True}}
    verdict = evaluate_actuarial_gates([], exposure_metrics, [], None, ActuarialThresholds())
    assert any(r.startswith("coverage: person_years_exposure/county") for r in verdict["reasons"])


def test_an_ungated_block_is_reported_and_never_decides_a_pass():
    metrics = {f"{MORTALITY_ESTIMAND}/county": {
        "n_cells": 4, "percentile_error": 9.0, "max_error": 9.0, "worst_cell": (0, "male", "65-74"),
        "coverage": 0.0, "mean_interval_score": 9.0, "gated": False}}
    assert evaluate_actuarial_gates([], metrics, [], None, ActuarialThresholds())["pass"]


def test_the_ensemble_container_refuses_a_single_path():
    with pytest.raises(ValueError):
        ContinuationEnsemble(LIABILITY[:1]).truth()


# ------------------------------------------------------- the version-four verifier

def _write(path, table):
    import pandas as pd
    pd.DataFrame(table).to_csv(path, index=False)


def _packet(tmp_path):
    """A minimal version-four packet: geography, contract, retained truth, ensemble."""
    from meridia.actuarial import ensemble_truth, reserve_total
    from meridia.release import (AGE_BAND_LABELS, SEX_LABELS,
                                 compute_detailed_table_truth, compute_truth)
    admin = tiny_admin() | {"state": np.asarray([[0, 1]], dtype=np.int64)}
    person = {"household": np.asarray([0, 0, 1, 1]), "cell": np.asarray([0, 0, 1, 1]),
              "age": np.asarray([70, 30, 80, 10]), "sex": np.asarray([0, 1, 0, 1]),
              "role": np.asarray([0, 1, 0, 1]), "education": np.asarray([2, 1, 2, 0]),
              "income": np.asarray([100.0, 200.0, 300.0, 400.0])}
    household_cell = np.asarray([0, 1])
    truth = compute_truth(person, household_cell, admin)
    detailed = compute_detailed_table_truth(person, admin)
    rate_truth = exposure_and_rate_truth(tiny_pass(), admin)

    liability = np.column_stack([1000.0 + 10.0 * np.arange(40),
                                 np.append(10000.0 + 100.0 * np.arange(39), 50000.0)])
    sealed = ensemble_truth(liability, 0.95)
    total = reserve_total(sealed["q"], sealed["es"])

    packet = tmp_path / "packet"
    (packet / "participant").mkdir(parents=True)
    (packet / "retained").mkdir()
    _write(packet / "participant" / "geography.csv", {"county": [0, 1], "state": [0, 1]})
    (packet / "participant" / "contract.json").write_text(json.dumps({
        "schema": "meridia.packet.v4", "disclosure_threshold": 0,
        "reserve": {"total": total, "gamma": 0.25,
                    "obligation": UNDISCOUNTED.as_public()}}, indent=1) + "\n")
    keys = sorted(truth)
    for name in ("truth_revised.csv", "truth_horizon.csv"):
        _write(packet / "retained" / name,
               {"estimand": [k[0] for k in keys], "level": [k[1] for k in keys],
                "unit": [k[2] for k in keys], "value": [truth[k] for k in keys]})
    counties, bands, sexes = np.indices(detailed.shape)
    _write(packet / "retained" / "detailed_revised.csv", {
        "county": counties.ravel(), "age_band": np.asarray(AGE_BAND_LABELS)[bands.ravel()],
        "sex": np.asarray(SEX_LABELS)[sexes.ravel()], "count": detailed.ravel()})
    rate_keys = sorted(rate_truth)
    _write(packet / "retained" / "rate_truth_horizon.csv", {
        "estimand": [k[0] for k in rate_keys], "level": [k[1] for k in rate_keys],
        "unit": [k[2] for k in rate_keys], "sex": [k[3] for k in rate_keys],
        "age_band": [k[4] for k in rate_keys],
        "value": [rate_truth[k] for k in rate_keys]})
    np.savez(packet / "retained" / "continuation_liabilities.npz", liability=liability,
             realized_member=np.asarray(0))
    return packet, admin, truth, detailed, rate_truth, liability, sealed, total


def _oracle_submission(directory, admin, truth, detailed, rate_truth, sealed, total,
                       allocation, q_hat):
    from meridia.release import AGE_BAND_LABELS, SEX_LABELS
    from meridia.scoring import rows_from_values
    directory.mkdir(parents=True)
    rows = rows_from_values(truth, 0.0)
    block = {"estimand": [], "level": [], "unit": [], "sex": [], "age_band": [],
             "estimate": [], "lower": [], "upper": []}
    for row in rows:
        for column in ("estimand", "level", "unit", "estimate", "lower", "upper"):
            block[column].append(row[column])
        block["sex"].append("")
        block["age_band"].append("")
    for key in sorted(rate_truth):
        value = rate_truth[key] if math.isfinite(rate_truth[key]) else 0.0
        for column, v in zip(("estimand", "level", "unit", "sex", "age_band"), key):
            block[column].append(v)
        for column in ("estimate", "lower", "upper"):
            block[column].append(value)
    for name in ("release.csv", "projection.csv"):
        _write(directory / name, block)
    counties, bands, sexes = np.indices(detailed.shape)
    _write(directory / "detailed.csv", {
        "county": counties.ravel(), "age_band": np.asarray(AGE_BAND_LABELS)[bands.ravel()],
        "sex": np.asarray(SEX_LABELS)[sexes.ravel()], "count": detailed.ravel()})
    _write(directory / "reserve.csv", {
        "region": [0, 1], "liability_mean": sealed["mean"], "q95": q_hat,
        "es95": sealed["es"], "allocation": allocation})


def test_the_version_four_verifier_scores_all_four_files(tmp_path):
    from meridia.verify import verify_submission
    packet, admin, truth, detailed, rate_truth, liability, sealed, total = _packet(tmp_path)
    # Two continuations of forty exceed this quantile in each region, so the submission
    # is exactly calibrated at five percent.
    q_hat = np.asarray([1370.0, 13700.0])
    oracle = perfect_information_allocation(liability, total)
    _oracle_submission(tmp_path / "submission", admin, truth, detailed, rate_truth,
                       sealed, total, oracle, q_hat)
    report = verify_submission(packet, tmp_path / "submission")
    assert report["schema_errors"] == [] and report["additivity_errors"] == []
    assert report["rate_errors"] == [] and report["reserve_errors"] == []
    assert report["disclosure"]["pass"]
    # No cell of a three-month toy ledger reaches the structural exposure floor, so the
    # rate gate reads nothing and decides nothing.
    assert report["rate_metrics"] == {}
    reserve = report["reserve"]
    assert reserve["feasible"]
    assert reserve["exceedance"].tolist() == pytest.approx([0.05, 0.05])
    assert reserve["calibration"]["pooled"] == pytest.approx(0.0)
    assert reserve["skill"] == pytest.approx(1.0)
    assert float(reserve["reserve_total"]) == pytest.approx(total)


def test_the_version_four_verifier_names_an_infeasible_reserve(tmp_path):
    from meridia.verify import verify_submission
    packet, admin, truth, detailed, rate_truth, liability, sealed, total = _packet(tmp_path)
    q_hat = np.asarray([1370.0, 13700.0])
    oracle = perfect_information_allocation(liability, total)
    _oracle_submission(tmp_path / "submission", admin, truth, detailed, rate_truth,
                       sealed, total, oracle * 0.5, q_hat)
    report = verify_submission(packet, tmp_path / "submission")
    assert not report["pass"]
    assert any(r.startswith("reserve: infeasible") for r in report["reasons"])


def test_the_version_four_verifier_fails_an_unexpected_file(tmp_path):
    from meridia.verify import verify_submission
    packet, admin, truth, detailed, rate_truth, liability, sealed, total = _packet(tmp_path)
    directory = tmp_path / "submission"
    _oracle_submission(directory, admin, truth, detailed, rate_truth, sealed, total,
                       perfect_information_allocation(liability, total),
                       np.asarray([1370.0, 13700.0]))
    (directory / "allocation.csv").write_text("county,allocation\n0,1\n")
    report = verify_submission(packet, directory)
    assert not report["pass"]
    assert report["reasons"][0].startswith("file set: unexpected ['allocation.csv']")


def test_the_eligibility_floor_falls_with_age_so_the_oldest_bands_are_still_gated():
    """A flat floor removed every band the obligation is made of from every rate gate."""
    thresholds = ActuarialThresholds()
    floors = {band: eligibility_floor(thresholds, band)
              for band in ACTUARIAL_AGE_BAND_LABELS}
    assert floors["85+"] < floors["75-84"] < floors["65-74"] < floors["45-64"]
    # The exposures a state by sex cell carries at the committed world size, measured on
    # qualification seed 2101: the three oldest bands clear their own floors and would
    # have failed one flat floor of five thousand person-years.
    measured = {"0-17": 5_719.0, "18-44": 9_090.0, "45-64": 6_532.0,
                "65-74": 1_910.0, "75-84": 1_030.0, "85+": 720.0}
    for band, exposure in measured.items():
        assert exposure >= eligibility_floor(thresholds, band), band
        if band in ("65-74", "75-84", "85+"):
            assert exposure < 5_000.0


def test_the_frozen_baseline_is_public_and_never_reads_the_submission():
    """A_B is the published size share of R, so a padded submission cannot move it."""
    share = np.asarray([0.5, 0.3, 0.2])
    total = 1_000.0
    baseline = proportional_baseline_allocation(share, total)
    assert baseline == pytest.approx([500.0, 300.0, 200.0])
    assert float(baseline.sum()) == pytest.approx(total)

    liability = np.asarray([[100.0, 400.0], [140.0, 420.0], [180.0, 440.0], [260.0, 460.0]])
    q_hat = np.asarray([200.0, 450.0])
    honest = score_reserve(np.asarray([300.0, 350.0]), q_hat, q_hat, q_hat.copy(),
                           liability, 650.0, baseline_share=np.asarray([0.5, 0.5]))
    padded = score_reserve(np.asarray([300.0, 350.0]), q_hat * 2.0, q_hat * 2.0,
                           q_hat.copy(), liability, 650.0,
                           baseline_share=np.asarray([0.5, 0.5]))
    assert honest["J_baseline"] == pytest.approx(padded["J_baseline"])
    assert honest["baseline_allocation"] == pytest.approx([325.0, 325.0])


def test_an_unfrozen_bar_set_refuses_to_gate_a_version_four_submission(tmp_path):
    """A bar file written before its own freeze verdict must not be read as frozen."""
    from meridia.verify import verify_submission
    packet, admin, truth, detailed, rate_truth, liability, sealed, total = _packet(tmp_path)
    q_hat = np.asarray([1370.0, 13700.0])
    oracle = perfect_information_allocation(liability, total)
    _oracle_submission(tmp_path / "submission", admin, truth, detailed, rate_truth,
                       sealed, total, oracle, q_hat)
    bars = {"actuarial": {"tau_mean": 0.5}}
    unfrozen = verify_submission(packet, tmp_path / "submission", bars)
    assert unfrozen["pass"] is False
    assert unfrozen["reasons"] == ["bars: this bar set does not record a completed freeze"]
    during_freeze = verify_submission(packet, tmp_path / "submission", bars,
                                      allow_unfrozen=True)
    assert during_freeze["pass"] is True
    frozen = verify_submission(packet, tmp_path / "submission", dict(bars, frozen=True))
    assert frozen["pass"] is True
