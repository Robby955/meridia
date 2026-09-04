"""An undefined reserve skill is a recorded gate failure, never a stopped run.

Skill is (J(A_B) - J(A)) / (J(A_B) - J(A*)). Where the published total is large enough
that the proportional baseline and the perfect-information oracle both cover every
continuation, the denominator is zero and the score does not exist. That is a real state
of a world and a rate, and every layer downstream of it has to record it rather than fail
on it.
"""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pytest

from meridia.actuarial import score_reserve
from meridia.methods.phase_three import _json_safe
from meridia.verify import (COMPOSITE_GATE_COMPONENTS, build_composite_metrics,
                            component_value, evaluate_composite_gates)


def _freeze():
    path = Path(__file__).resolve().parents[1] / "scripts" / "freeze_v4_bars.py"
    spec = importlib.util.spec_from_file_location("freeze_v4_bars_undefined", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _covered_reserve() -> dict:
    """Score a reserve whose total covers every continuation in every region."""
    liability = np.array([[10.0, 20.0], [12.0, 18.0], [11.0, 22.0], [9.0, 19.0]])
    total = 1_000.0
    allocation = np.array([500.0, 500.0])
    q_hat = np.array([12.0, 22.0])
    es_hat = np.array([12.0, 22.0])
    mean_hat = liability.mean(axis=0)
    return score_reserve(allocation, q_hat, es_hat, mean_hat, liability, total,
                         baseline_share=np.array([0.5, 0.5]))


def _bars() -> dict:
    return {"gates": {gate: {"components": {component: {"value": 0.5}
                                            for component in components}}
                      for gate, components in COMPOSITE_GATE_COMPONENTS.items()}}


def _metrics(skill_loss: object) -> dict:
    return {
        "exposures_and_rates": {"p95_relative_error": 0.1},
        "release_accuracy": {"p95_relative_error": 0.1},
        "interval_quality": {"coverage_deviation": 0.1, "mean_interval_score": 0.1},
        "tail_calibration": {"pooled_exceedance_deviation": 0.1,
                             "q95_width_relative_error": 0.1,
                             "es95_width_relative_error": 0.1},
        "reserve_skill": {"skill_loss": skill_loss,
                          "worst_regional_shortfall_probability": 0.0},
    }


def test_a_covered_total_leaves_the_skill_undefined():
    reserve = _covered_reserve()
    assert reserve["feasible"] is True
    assert reserve["J_baseline"] == pytest.approx(0.0)
    assert reserve["J_oracle"] == pytest.approx(0.0)
    assert math.isnan(reserve["skill"])
    composite = build_composite_metrics({}, {}, {}, reserve, 0.10)
    assert math.isnan(composite["reserve_skill"]["skill_loss"])


def test_the_record_reaches_disk_as_null_instead_of_stopping_the_run():
    composite = build_composite_metrics({}, {}, {}, _covered_reserve(), 0.10)
    safe = _json_safe({"composite_metrics": composite})
    assert safe["composite_metrics"]["reserve_skill"]["skill_loss"] is None
    encoded = json.dumps(safe, sort_keys=True, allow_nan=False)
    assert '"skill_loss":null' in encoded.replace(" ", "")


def test_infinities_and_numpy_scalars_are_recorded_the_same_way():
    safe = _json_safe({
        "positive": float("inf"),
        "negative": float("-inf"),
        "numpy": np.float64("nan"),
        "array": np.array([1.0, np.inf]),
        "finite": 0.25,
    })
    assert safe["positive"] is None and safe["negative"] is None
    assert safe["numpy"] is None and safe["array"] == [1.0, None]
    assert safe["finite"] == 0.25
    json.dumps(safe, sort_keys=True, allow_nan=False)


@pytest.mark.parametrize("value", [None, float("nan"), float("inf"), "", "x", True,
                                   {}, {"value": None}, [1.0]])
def test_component_value_never_raises_on_an_undefined_component(value):
    assert math.isnan(component_value(value))


@pytest.mark.parametrize("value", [0.25, 1, np.float64(0.5), {"value": 0.75}])
def test_component_value_reads_a_defined_component(value):
    assert math.isfinite(component_value(value))


@pytest.mark.parametrize("profile", ["full", "lite"])
@pytest.mark.parametrize("undefined", [float("nan"), None])
def test_an_undefined_skill_fails_the_gate_with_a_stated_reason(profile, undefined):
    results = evaluate_composite_gates(_metrics(undefined), _bars(), True, profile)
    reserve = results["reserve_skill"]
    assert reserve["gated"] is True
    assert reserve["evaluated"] is True
    assert reserve["pass"] is False
    assert len(reserve["reasons"]) == 1
    reason = reserve["reasons"][0]
    assert "non-finite components ['skill_loss']" in reason
    assert "denominator J(A_B) - J(A*) is not positive" in reason
    assert all(results[gate]["pass"] is not False
               for gate in results if gate != "reserve_skill"
               and results[gate]["gated"])


def test_a_defined_skill_still_decides_normally():
    results = evaluate_composite_gates(_metrics(0.25), _bars(), True, "full")
    assert results["reserve_skill"]["pass"] is True
    worse = evaluate_composite_gates(_metrics(0.75), _bars(), True, "full")
    assert worse["reserve_skill"]["pass"] is False
    assert worse["reserve_skill"]["reasons"] == ["skill_loss 0.75 > 0.5"]


def test_the_freeze_refuses_to_calibrate_a_bar_from_an_undefined_skill():
    freeze = _freeze()
    report = {"composite_metrics": _json_safe(
        {**_metrics(float("nan")), "reserve_skill": {
            "skill_loss": float("nan"), "worst_regional_shortfall_probability": 0.0}})}
    with pytest.raises(freeze.EvidenceError) as excinfo:
        freeze.extract_composite_metrics(report)
    assert "reserve_skill/skill_loss" in str(excinfo.value)
