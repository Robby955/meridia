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
                               eligibility_floor, empirical_tail, ensemble_truth,
                               evaluate_actuarial_gates,
                               exceedance_probabilities, expected_uncovered,
                               exposure_and_rate_truth, interval_score,
                               liabilities_from_pass, parse_rate_rows, parse_reserve_rows,
                               perfect_information_allocation,
                               proportional_baseline_allocation, quantile_score,
                               relative_error, required_rate_rows, reserve_total,
                               score_rates, score_reserve,
                               shortfall_error, skill_score, width_relative_error)
from meridia.events import EVENT_TYPES, _EVENT_DTYPES
from meridia.packet import PARTICIPANT_CSV_SCHEMAS

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


def test_empirical_tail_uses_ceiling_rank_and_includes_every_tie():
    values = np.arange(1.0, 21.0)[:, None]
    q, es = empirical_tail(values, 0.95)
    assert q.tolist() == [19.0]
    assert es.tolist() == [19.5]
    tied = np.asarray([1.0] * 18 + [5.0, 5.0])[:, None]
    q, es = empirical_tail(tied, 0.95)
    assert q.tolist() == [5.0]
    assert es.tolist() == [5.0]


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


def test_reserve_total_uses_only_public_exposure_rate_and_rounding():
    assert reserve_total(100.0, 4.4, 10.0) == pytest.approx(440.0)
    assert reserve_total(101.0, 4.4, 10.0) == pytest.approx(450.0)
    with pytest.raises(ValueError):
        reserve_total(100.0, 0.0, 10.0)


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


def test_score_reserve_allows_an_allocation_below_the_submitted_quantile():
    report = score_reserve(np.asarray([10.0, 540.0]), np.asarray([40.0, 400.0]),
                           np.asarray([40.0, 400.0]), np.asarray([25.0, 250.0]),
                           LIABILITY, total=550.0, thresholds=RESERVE_THRESHOLDS)
    assert report["feasible"]
    assert report["feasibility_reasons"] == []


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_score_reserve_rejects_a_nonfinite_allocation(bad):
    report = score_reserve(np.asarray([bad, 550.0]), np.asarray([40.0, 400.0]),
                           np.asarray([40.0, 400.0]), np.asarray([25.0, 250.0]),
                           LIABILITY, total=550.0, thresholds=RESERVE_THRESHOLDS)
    assert not report["feasible"]
    assert "non-finite allocation" in report["feasibility_reasons"]


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
    truth, parsed = {}, {}
    for band in ("65-74", "75-84", "85+"):
        cell_exposure = exposure / 3.0
        truth[(EXPOSURE_ESTIMAND, "state", 0, "male", band)] = cell_exposure
        truth[(MORTALITY_ESTIMAND, "state", 0, "male", band)] = rate
        parsed[(EXPOSURE_ESTIMAND, "state", 0, "male", band)] = \
            (cell_exposure, cell_exposure, cell_exposure)
        parsed[(MORTALITY_ESTIMAND, "state", 0, "male", band)] = \
            (rate, rate * 0.9, rate * 1.1)
    return truth, parsed


def test_score_rates_reads_only_cells_with_enough_exposure():
    thresholds = ActuarialThresholds()
    truth, parsed = _rate_truth_and_submission(exposure=10_000.0)
    metrics = score_rates(parsed, truth, thresholds)
    assert metrics[f"{MORTALITY_ESTIMAND}/state"]["n_cells"] == 1
    assert metrics[f"{MORTALITY_ESTIMAND}/state"]["gated"] is True
    assert metrics[f"{MORTALITY_ESTIMAND}/state"]["percentile_error"] == pytest.approx(0.0)
    assert metrics[f"{MORTALITY_ESTIMAND}/state"]["coverage"] == pytest.approx(1.0)
    assert metrics[f"{MORTALITY_ESTIMAND}/state"]["cells"] == [(0, "male", "65+")]
    thin_truth, thin_parsed = _rate_truth_and_submission(exposure=100.0)
    thin = score_rates(thin_parsed, thin_truth, thresholds)
    assert all(block["n_cells"] == 0 for block in thin.values())


def test_composite_eligible_cells_are_truth_defined_even_when_a_row_is_missing():
    thresholds = ActuarialThresholds()
    truth, parsed = _rate_truth_and_submission(exposure=10_000.0)
    complete = score_rates(parsed, truth, thresholds)["composite"]
    missing = dict(parsed)
    missing.pop((MORTALITY_ESTIMAND, "state", 0, "male", "85+"))
    incomplete = score_rates(missing, truth, thresholds)["composite"]
    assert incomplete["eligible_cells"] == complete["eligible_cells"]
    assert incomplete["n_eligible"] == complete["n_eligible"]
    assert incomplete["n_cells"] < complete["n_cells"]
    assert [MORTALITY_ESTIMAND, "state", 0, "male", "65+"] \
        in incomplete["eligible_cells"]


def test_interval_composite_does_not_cancel_opposite_subgroup_miscalibration():
    from meridia.verify import _interval_quality
    low = {"first/all": {"n_units": 10, "coverage": 0.8,
                         "mean_interval_score": 0.2}}
    high = {"second/all": {"n_units": 10, "coverage": 1.0,
                            "mean_interval_score": 0.2}}
    deviation, score = _interval_quality(low, high, {}, alpha=0.10)
    assert deviation == pytest.approx(0.10)
    assert score == pytest.approx(0.20)


def test_score_rates_gates_exposure_at_county_and_rates_at_state():
    thresholds = ActuarialThresholds(exposure_eligibility_person_years=1.0)
    truth = {(EXPOSURE_ESTIMAND, "county", 0, "male", "65+"): 100.0,
             (EXPOSURE_ESTIMAND, "county", 0, "male", "65-74"): 100.0,
             (EXPOSURE_ESTIMAND, "state", 0, "male", "65-74"): 34.0,
             (EXPOSURE_ESTIMAND, "state", 0, "male", "75-84"): 33.0,
             (EXPOSURE_ESTIMAND, "state", 0, "male", "85+"): 33.0,
             (MORTALITY_ESTIMAND, "county", 0, "male", "65-74"): 0.01,
             (MORTALITY_ESTIMAND, "state", 0, "male", "65-74"): 0.01,
             (MORTALITY_ESTIMAND, "state", 0, "male", "75-84"): 0.01,
             (MORTALITY_ESTIMAND, "state", 0, "male", "85+"): 0.01}
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


def test_submission_indices_reject_fractional_values_instead_of_truncating():
    reserve_rows = _reserve_rows([(25.0, 40.0, 40.0, 50.0)])
    reserve_rows[0]["region"] = 0.5
    _, reserve_errors = parse_reserve_rows(reserve_rows, 1)
    assert any("region is not a known region index" in e for e in reserve_errors)

    admin = {"n_counties": 1, "n_states": 1, "county_state": np.asarray([0])}
    rate_rows = [{"estimand": EXPOSURE_ESTIMAND, "level": "county", "unit": 0.5,
                  "sex": "male", "age_band": "65+", "estimate": 1.0,
                  "lower": 1.0, "upper": 1.0}]
    _, rate_errors = parse_rate_rows(rate_rows, admin)
    assert any("unit is not a known county integer" in e for e in rate_errors)


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
                    es_hat: np.ndarray | None = None,
                    baseline_share: np.ndarray | None = None) -> dict:
    """One scored reserve. The baseline share is published, so these tests state it."""
    truth = ensemble_truth(LIABILITY, 0.95)
    share = q_hat / q_hat.sum() if baseline_share is None else baseline_share
    return score_reserve(allocation, q_hat,
                         truth["es"] if es_hat is None else es_hat,
                         truth["mean"], LIABILITY, total=total,
                         thresholds=RESERVE_THRESHOLDS, baseline_share=share)


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


TOY_EXPOSURE_SCALE = 4_000.0


def _write_participant_inputs(packet):
    """Header-only stand-ins for every participant CSV the contract has to publish.

    The verifier compares the published column map against the files that are actually
    on disk, so a toy packet has to carry all fourteen of them. Only the experience file
    and the geography map are read for a number here, and both are written afterwards
    with their own rows.
    """
    participant = packet / "participant"
    for relative, columns in sorted(PARTICIPANT_CSV_SCHEMAS.items()):
        path = participant / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(",".join(columns) + "\n")


def _packet(tmp_path):
    """A minimal version-four packet: geography, contract, retained truth, ensemble."""
    from meridia.actuarial import ensemble_truth, reserve_total
    from meridia.packet import continuation_source_law_digest
    from meridia.projection import _shock_redraw_evidence
    from meridia.release import compute_detailed_table_truth, compute_truth
    admin = tiny_admin() | {"state": np.asarray([[0, 1]], dtype=np.int64)}
    person = {"household": np.asarray([0, 0, 1, 1]), "cell": np.asarray([0, 0, 1, 1]),
              "age": np.asarray([70, 30, 80, 10]), "sex": np.asarray([0, 1, 0, 1]),
              "role": np.asarray([0, 1, 0, 1]), "education": np.asarray([2, 1, 2, 0]),
              "income": np.asarray([100.0, 200.0, 300.0, 400.0])}
    household_cell = np.asarray([0, 1])
    truth = compute_truth(person, household_cell, admin)
    detailed = compute_detailed_table_truth(person, admin)
    rate_truth = exposure_and_rate_truth(tiny_pass(), admin)
    # The toy ledger is three months over four people, so every cell sits far under the
    # published eligibility floors and no gated block would have a cell to read. Exposure
    # is lifted by one factor, which leaves every rate, every band sum and every
    # county-to-state sum exactly as the reading pass wrote them.
    rate_truth = {key: (value * TOY_EXPOSURE_SCALE if key[0] == EXPOSURE_ESTIMAND else value)
                  for key, value in rate_truth.items()}

    liability = np.column_stack([1000.0 + 10.0 * np.arange(40),
                                 np.append(10000.0 + 100.0 * np.arange(39), 50000.0)])
    sealed = ensemble_truth(liability, 0.95)
    total = reserve_total(1_000.0, 16.0, 1_000.0)

    packet = tmp_path / "packet"
    (packet / "participant").mkdir(parents=True)
    (packet / "retained").mkdir()
    _write_participant_inputs(packet)
    _write(packet / "participant" / "geography.csv",
           {"county": [0, 1], "state": [0, 1], "land_cells": [1, 1],
            "economic_band": [0, 1]})
    (packet / "participant" / "contract.json").write_text(json.dumps({
        "schema": "meridia.packet.v4",
        "experience_history": {
            "file": "experience_history.csv",
            "columns": ["year", "age_band", "sex", "state", "exposure", "deaths",
                        "qualifying_events", "net_migration"],
        },
        "participant_csv_schemas": {
            name: list(columns)
            for name, columns in sorted(PARTICIPANT_CSV_SCHEMAS.items())
        },
        "benchmark": {"file": "sources/benchmark_revised.csv"},
        "submission": {
            "files": {
                "release.csv": ["estimand", "level", "unit", "sex", "age_band",
                                "estimate", "lower", "upper"],
                "projection.csv": ["estimand", "level", "unit", "estimate", "lower",
                                   "upper"],
                "reserve.csv": ["region", "liability_mean", "q95", "es95", "allocation"],
            },
            "additional_entries": "forbidden",
        },
        "reserve": {
            "total": total,
            "members": len(liability),
            "allocation_rule": {
                "finite": True,
                "minimum": 0.0,
                "sum": "reserve.total",
                "tolerance": ActuarialThresholds().feasibility_tolerance,
            },
            "total_rule": {
                "file": "experience_history.csv",
                "year": "maximum published year",
                "year_column": "year",
                "selected_year": 2,
                "exposure_column": "exposure",
                "aggregation": "sum exposure over every row in the selected year",
                "exposure_person_years": 1_000.0,
                "rate_per_person_year": 16.0,
                "rounding": "up",
                "rounding_unit": 1_000.0,
            },
            "obligation": UNDISCOUNTED.as_public(),
        }}, indent=1) + "\n")
    _write(packet / "participant" / "experience_history.csv", {
        "year": [2], "age_band": ["65-74"], "sex": ["female"], "state": [0],
        "exposure": [1_000.0], "deaths": [10], "qualifying_events": [20],
        "net_migration": [0],
    })
    keys = sorted(truth)
    for name in ("truth_revised.csv", "truth_horizon.csv"):
        _write(packet / "retained" / name,
               {"estimand": [k[0] for k in keys], "level": [k[1] for k in keys],
                "unit": [k[2] for k in keys], "value": [truth[k] for k in keys]})
    rate_keys = sorted(rate_truth)
    _write(packet / "retained" / "rate_truth_horizon.csv", {
        "estimand": [k[0] for k in rate_keys], "level": [k[1] for k in rate_keys],
        "unit": [k[2] for k in rate_keys], "sex": [k[3] for k in rate_keys],
        "age_band": [k[4] for k in rate_keys],
        "value": [rate_truth[k] for k in rate_keys]})
    np.savez(packet / "retained" / "continuation_liabilities.npz", liability=liability,
             realized_member=np.asarray(0))
    schedules = [(member, []) for member in range(len(liability))]
    schedules[0][1].append({
        "year": 2,
        "kind": "mortality_spike",
        "mortality_multiplier": 2.0,
        "admission_multiplier": 2.0,
    })
    shock_evidence = _shock_redraw_evidence(
        {"month": 24}, 12, len(liability), schedules,
        continuation_source_law_digest(),
    )
    (packet / "retained" / "continuation_shock_redraw.json").write_text(
        json.dumps(shock_evidence, indent=1, sort_keys=True) + "\n"
    )
    return packet, admin, truth, detailed, rate_truth, liability, sealed, total


def _oracle_submission(directory, admin, truth, detailed, rate_truth, sealed, total,
                       allocation, q_hat):
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
    for key in sorted(required_rate_rows(admin)):
        value = rate_truth[key] if math.isfinite(rate_truth[key]) else 0.0
        for column, v in zip(("estimand", "level", "unit", "sex", "age_band"), key):
            block[column].append(v)
        for column in ("estimate", "lower", "upper"):
            block[column].append(value)
    _write(directory / "release.csv", block)
    _write(directory / "projection.csv", {
        column: [row[column] for row in rows]
        for column in ("estimand", "level", "unit", "estimate", "lower", "upper")})
    _write(directory / "reserve.csv", {
        "region": [0, 1], "liability_mean": sealed["mean"], "q95": q_hat,
        "es95": sealed["es"], "allocation": allocation})


def test_the_version_four_verifier_scores_five_composites_from_three_files(tmp_path):
    from meridia.verify import verify_submission
    packet, admin, truth, detailed, rate_truth, liability, sealed, total = _packet(tmp_path)
    # Two continuations of forty exceed this quantile in each region, so the submission
    # is exactly calibrated at five percent.
    q_hat = np.asarray([1370.0, 13700.0])
    oracle = perfect_information_allocation(liability, total)
    participant_input = packet / "participant" / "sources" / "observed.csv"
    participant_input.parent.mkdir(exist_ok=True)
    participant_input.write_text("value\n1\n")
    _oracle_submission(tmp_path / "submission", admin, truth, detailed, rate_truth,
                       sealed, total, oracle, q_hat)
    report = verify_submission(packet, tmp_path / "submission")
    assert report["schema_errors"] == [] and report["additivity_errors"] == []
    assert report["rate_errors"] == [] and report["reserve_errors"] == []
    # Every gated block reads at least one eligible cell, and the submission publishes the
    # retained truth, so each of them scores an error of zero on cells it names.
    for key in (f"{EXPOSURE_ESTIMAND}/county", f"{MORTALITY_ESTIMAND}/state",
                f"{INCIDENCE_ESTIMAND}/state"):
        block = report["rate_metrics"][key]
        assert block["gated"] is True and block["n_cells"] > 0
        assert block["percentile_error"] == pytest.approx(0.0)
        assert len(block["cells"]) == block["n_cells"]
    assert report["hard_pass"] is True
    assert report["pass"] is False
    assert report["reasons"] == [
        "bars: no frozen composite bar receipt was supplied"
    ]
    assert set(report["composite_metrics"]) == {
        "exposures_and_rates", "release_accuracy", "interval_quality",
        "tail_calibration", "reserve_skill"}
    assert set(report["gate_results"]) == set(report["composite_metrics"])
    assert all(not result["evaluated"] for result in report["gate_results"].values())
    assert report["evidence"]["schema"] == "meridia.v4.verifier-evidence.v1"
    assert all(len(report["evidence"][field]) == 64 for field in (
        "packet_digest_sha256", "contract_digest_sha256",
        "submission_digest_sha256", "verifier_digest_sha256",
    ))
    q95_feasibility = report["reserve_q95_feasibility"]
    assert q95_feasibility["q95_sum"] == pytest.approx(float(q_hat.sum()))
    assert q95_feasibility["allocation_sum"] == pytest.approx(float(oracle.sum()))
    assert q95_feasibility["reserve_total"] == pytest.approx(float(total))
    assert q95_feasibility["total_minus_q95_sum"] \
        == pytest.approx(float(total - q_hat.sum()))
    assert q95_feasibility["all_regions_at_or_above_q95"] is True
    assert q95_feasibility["allocation_sums_to_total"] is True
    assert q95_feasibility["feasible"] is True
    tail_evidence = report["reserve_tail_evidence"]
    assert tail_evidence["schema"] == "meridia.v4.reserve-tail-evidence.v1"
    assert tail_evidence["valid"] is True
    assert tail_evidence["q95_sum"] == pytest.approx(float(q_hat.sum()))
    assert tail_evidence["es95_sum"] == pytest.approx(float(sealed["es"].sum()))
    assert tail_evidence["reserve_submission_sha256"] \
        == report["evidence"]["submission_file_sha256"]["reserve.csv"]
    first_packet_digest = report["evidence"]["packet_digest_sha256"]
    participant_input.write_text("value\n2\n")
    rebound = verify_submission(packet, tmp_path / "submission")
    assert rebound["evidence"]["packet_digest_sha256"] != first_packet_digest
    eligibility = report["eligibility_evidence"]["bands"]
    assert eligibility["65+"]["status"] == "scored"
    assert eligibility["65+"]["floor_person_years"] == 500.0
    assert eligibility["65-74"]["status"] == "report-only"
    assert eligibility["75-84"]["status"] == "report-only"
    assert eligibility["85+"]["status"] == "report-only"
    for record in eligibility.values():
        assert len(record["cells"]) == record["cell_count"] == 4
        assert record["eligible_count"] == sum(
            cell["eligible"] for cell in record["cells"])
        assert record["minimum_exposure_person_years"] == min(
            cell["exposure_person_years"] for cell in record["cells"])
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


def test_verifier_rejects_a_relabelled_continuation_source_receipt(tmp_path):
    from meridia.verify import verify_submission

    packet, admin, truth, detailed, rate_truth, liability, sealed, total = _packet(
        tmp_path
    )
    runtime_path = packet / "retained" / "continuation_shock_redraw.json"
    runtime = json.loads(runtime_path.read_text())
    runtime["continuation_source_law_sha256"] = "0" * 64
    runtime_path.write_text(json.dumps(runtime, indent=1, sort_keys=True) + "\n")
    _oracle_submission(
        tmp_path / "submission",
        admin,
        truth,
        detailed,
        rate_truth,
        sealed,
        total,
        perfect_information_allocation(liability, total),
        np.asarray([1370.0, 13700.0]),
    )

    report = verify_submission(packet, tmp_path / "submission")
    assert report["pass"] is False
    assert any("source law differs" in error for error in report["schema_errors"])


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
    assert "unexpected ['allocation.csv']" in report["reasons"][0]


def test_the_version_four_verifier_fails_an_unexpected_directory(tmp_path):
    from meridia.verify import verify_submission
    packet, admin, truth, detailed, rate_truth, liability, sealed, total = _packet(tmp_path)
    directory = tmp_path / "submission"
    _oracle_submission(directory, admin, truth, detailed, rate_truth, sealed, total,
                       perfect_information_allocation(liability, total),
                       np.asarray([1370.0, 13700.0]))
    (directory / "artifacts").mkdir()
    report = verify_submission(packet, directory)
    assert not report["pass"]
    assert "unexpected ['artifacts']" in report["reasons"][0]


def test_eligibility_depends_only_on_exposure_and_maps_fine_old_ages_to_65_plus():
    thresholds = ActuarialThresholds()
    for band in ACTUARIAL_AGE_BAND_LABELS:
        for estimand in (EXPOSURE_ESTIMAND, MORTALITY_ESTIMAND, INCIDENCE_ESTIMAND):
            expected = 500.0 if band in ("65-74", "75-84", "85+") else 600.0
            assert eligibility_floor(thresholds, band, estimand) == pytest.approx(expected)
    assert eligibility_floor(thresholds, "65+") == pytest.approx(500.0)
    moved = ActuarialThresholds(exposure_eligibility_person_years=750.0)
    assert eligibility_floor(moved, "85+", INCIDENCE_ESTIMAND) == pytest.approx(625.0)


def test_a_gated_block_with_no_eligible_cell_is_named_rather_than_dropped():
    """A block that scores nothing used to vanish, and a shorter report read as a pass."""
    thresholds = ActuarialThresholds()
    truth, parsed = _rate_truth_and_submission(exposure=100.0)
    metrics = score_rates(parsed, truth, thresholds)
    block = metrics[f"{MORTALITY_ESTIMAND}/state"]
    assert block["n_cells"] == 0 and block["gated"] is True
    assert block["n_eligible"] == 0
    assert "fixed eligible set" in block["reason"]
    verdict = evaluate_actuarial_gates([], metrics, [], None, thresholds)
    assert not verdict["pass"]
    assert any("has no eligible cell" in r for r in verdict["reasons"])


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


def _synthetic_ensemble(members: int = 2048):
    """A 2,048-member ensemble whose regional tail width matches the sealed worlds.

    The committed qualification worlds carry a regional (q95 - mean) / mean of about 0.099
    at 2,048 continuations, and this draw reproduces it: a normal tail of that width puts
    q95 at 1.645 standard deviations, so a standard deviation of 0.0602 of the mean.
    """
    rng = np.random.default_rng(20260903)
    means = np.asarray([1.0e6, 1.4e6, 0.8e6, 2.0e6, 1.1e6, 1.6e6])
    return np.random.default_rng(rng.bit_generator.state["state"]["state"]).normal(
        means, 0.0602 * means, size=(members, means.size))


def test_the_tail_bars_refuse_a_mean_only_and_a_doubled_tail():
    """A tail bar on the level lets a tail that is out by its whole width through."""
    liability = _synthetic_ensemble()
    truth = ensemble_truth(liability, 0.95)
    width = (truth["q"] - truth["mean"]) / truth["mean"]
    assert float(width.mean()) == pytest.approx(0.099, abs=0.005)

    scale = np.maximum(truth["q"], 1.0)
    oracle_score = float(quantile_score(truth["q"], liability, scale).mean())
    padded_q = truth["mean"] + 2.0 * (truth["q"] - truth["mean"])
    padded_es = truth["mean"] + 2.0 * (truth["es"] - truth["mean"])
    total = float(padded_q.sum()) * 1.05
    share = np.full(liability.shape[1], 1.0 / liability.shape[1])

    # The bars of the first freeze that sit inside the range their criterion can take.
    level_bars = {"tau_mean": 0.15, "tau_worst": 0.30,
                  "quantile_score_ceiling": round(1.75 * oracle_score, 6),
                  "es_error_ceiling": 0.10,
                  "q95_width_error_ceiling": None, "es95_width_error_ceiling": None,
                  "skill_minimum": None, "regional_shortfall_ceiling": None,
                  "catastrophic_tail_ceiling": None}
    width_bars = dict(level_bars, q95_width_error_ceiling=0.50,
                      es95_width_error_ceiling=0.50)

    def tail_reasons(q_hat, es_hat, bars):
        allocation = q_hat + (total - q_hat.sum()) / liability.shape[1]
        report = score_reserve(allocation, q_hat, es_hat, truth["mean"], liability, total,
                               baseline_share=share)
        assert report["feasible"]
        verdict = evaluate_actuarial_gates([], {}, [], report, ActuarialThresholds(), bars)
        return [r for r in verdict["reasons"] if r.startswith("tail:")], report

    honest, report = tail_reasons(truth["q"], truth["es"], width_bars)
    assert honest == []
    assert report["mean_q95_width_error"] == pytest.approx(0.0, abs=1e-9)

    # Out by one whole width in either direction, which is what the criterion measures.
    for q_hat, es_hat in ((truth["mean"], truth["mean"]), (padded_q, padded_es)):
        _, report = tail_reasons(q_hat, es_hat, width_bars)
        assert report["mean_q95_width_error"] == pytest.approx(1.0, abs=0.02)
        assert report["mean_es95_width_error"] == pytest.approx(1.0, abs=0.02)

    loose, _ = tail_reasons(padded_q, truth["es"], level_bars)
    assert loose == [], "a doubled tail cleared every bar frozen on the level"
    gated, _ = tail_reasons(padded_q, truth["es"], width_bars)
    assert any(r.startswith("tail: q95 error") for r in gated)
    gated, _ = tail_reasons(truth["q"], padded_es, width_bars)
    assert any(r.startswith("tail: ES95 error") for r in gated)

    mean_only, _ = tail_reasons(truth["mean"], truth["mean"], width_bars)
    assert any(r.startswith("tail: q95 error") for r in mean_only)
    assert any(r.startswith("tail: ES95 error") for r in mean_only)


def test_the_width_criterion_holds_its_divisor_above_a_published_minimum():
    """A region whose ensemble barely moves cannot divide the criterion by zero."""
    mean = np.asarray([1000.0, 1000.0])
    tail = np.asarray([1000.0, 1100.0])
    error = width_relative_error(np.asarray([1005.0, 1100.0]), tail, mean, 0.010)
    assert error[0] == pytest.approx(5.0 / 10.0)
    assert error[1] == pytest.approx(0.0)


def test_the_baseline_stays_public_when_the_contract_names_no_share():
    """Both arms of the skill denominator used to read the submission's own q95."""
    liability = np.asarray([[100.0, 400.0], [140.0, 420.0], [180.0, 440.0], [260.0, 460.0]])
    total = 650.0
    allocation = np.asarray([300.0, 350.0])
    honest = score_reserve(allocation, np.asarray([200.0, 300.0]), np.asarray([220.0, 320.0]),
                           np.asarray([170.0, 290.0]), liability, total)
    padded = score_reserve(allocation, np.asarray([260.0, 340.0]), np.asarray([280.0, 360.0]),
                           np.asarray([170.0, 290.0]), liability, total)
    assert honest["baseline_source"] == "even split"
    assert honest["baseline_allocation"] == pytest.approx([325.0, 325.0])
    assert honest["J_baseline"] == pytest.approx(padded["J_baseline"])
    assert honest["J_oracle"] == pytest.approx(padded["J_oracle"])
    assert honest["skill"] == pytest.approx(padded["skill"])


def test_a_state_rate_its_own_counties_contradict_is_named():
    """The docstring promised a ratio-consistency check that the code did not run."""
    admin = {"n_states": 1, "n_counties": 2,
             "county_state": np.asarray([0, 0], dtype=np.int64)}
    band = "65-74"
    parsed = {}
    for unit, level, exposure, rate in ((0, "county", 1_000.0, 0.05),
                                        (1, "county", 3_000.0, 0.09),
                                        (0, "state", 4_000.0, 0.08)):
        parsed[(EXPOSURE_ESTIMAND, level, unit, "male", band)] = (exposure,) * 3
        parsed[(MORTALITY_ESTIMAND, level, unit, "male", band)] = (rate,) * 3
    # 1,000 * 0.05 + 3,000 * 0.09 is 320 deaths, and the state files 4,000 * 0.08 = 320.
    assert check_rate_additivity(parsed, admin) == []
    parsed[(MORTALITY_ESTIMAND, "state", 0, "male", band)] = (0.10, 0.10, 0.10)
    errors = check_rate_additivity(parsed, admin)
    assert len(errors) == 1
    assert errors[0].startswith(f"{MORTALITY_ESTIMAND}: state 0 male 65-74 implies")


def test_an_unfrozen_bar_set_refuses_to_gate_a_version_four_submission(tmp_path):
    """A bar file written before its own freeze verdict must not be read as frozen."""
    from meridia.verify import verify_submission
    packet, admin, truth, detailed, rate_truth, liability, sealed, total = _packet(tmp_path)
    q_hat = np.asarray([1370.0, 13700.0])
    oracle = perfect_information_allocation(liability, total)
    _oracle_submission(tmp_path / "submission", admin, truth, detailed, rate_truth,
                       sealed, total, oracle, q_hat)
    bars = {
        "schema": "meridia.v4.composite-bars.v1",
        "gates": {
            "exposures_and_rates": {"components": {
                "p95_relative_error": {"direction": "ceiling", "value": 1e9}}},
            "release_accuracy": {"components": {
                "p95_relative_error": {"direction": "ceiling", "value": 1e9}}},
            "interval_quality": {"components": {
                "coverage_deviation": {"direction": "ceiling", "value": 1e9},
                "mean_interval_score": {"direction": "ceiling", "value": 1e9}}},
            "tail_calibration": {"components": {
                "pooled_exceedance_deviation": {"direction": "ceiling", "value": 1e9},
                "q95_width_relative_error": {"direction": "ceiling", "value": 1e9},
                "es95_width_relative_error": {"direction": "ceiling", "value": 1e9}}},
            "reserve_skill": {"components": {
                "skill_loss": {"direction": "ceiling", "value": 1e9},
                "worst_regional_shortfall_probability": {
                    "direction": "ceiling", "value": 1.0,
                }}},
        },
    }
    unfrozen = verify_submission(packet, tmp_path / "submission", bars)
    assert unfrozen["pass"] is False
    assert unfrozen["reasons"] == ["bars: this bar set does not record a completed freeze"]
    during_freeze = verify_submission(packet, tmp_path / "submission", bars,
                                      allow_unfrozen=True)
    assert during_freeze["pass"] is False
    assert during_freeze["reasons"] == unfrozen["reasons"]
    forged = verify_submission(packet, tmp_path / "submission", dict(bars, frozen=True))
    assert forged["pass"] is False
    assert any(reason.startswith("bars:") for reason in forged["reasons"])
    assert forged["bar_schema_errors"]
