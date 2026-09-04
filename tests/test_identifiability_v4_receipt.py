import json
from itertools import product
from pathlib import Path

import pandas as pd
import pytest

from meridia.character import CHARACTER_RANGES
from meridia.mechanisms import (
    COEFFICIENT_RANGES,
    DEVELOPMENT_BAND,
    PUBLIC_ENVELOPE,
    build_world_mechanisms,
)
from scripts import build_v4_worlds as builder
from scripts import identifiability_v4 as audit


def _interaction_envelope(
    primary: tuple[float, float],
    modifier: tuple[float, float],
    coefficient: tuple[float, float],
) -> tuple[float, float]:
    values = [
        axis * (1.0 + interaction * (other - 1.0))
        for axis, other, interaction in product(primary, modifier, coefficient)
    ]
    return min(values), max(values)


def _age_reporting_envelope(axis: tuple[float, float]) -> tuple[float, float]:
    values = [
        intensity * (
            1.0
            + coefficient
            * (gompertz_b / audit.REFERENCE_MORTALITY_AGE_SLOPE - 1.0)
        )
        for intensity, coefficient, gompertz_b in product(
            axis,
            COEFFICIENT_RANGES["age_error_by_mortality_slope"],
            CHARACTER_RANGES["gompertz_b"],
        )
    ]
    return min(values), max(values)


def _coefficient_record(
    intensity: dict[str, float],
    *,
    gompertz_b: float = 0.105,
    health_interaction: float = 0.6,
) -> dict[str, float]:
    age_interaction = 0.7
    return {
        **intensity,
        "linkage_gradient_by_migration": 0.4,
        "health_inclusion_completeness_by_target": health_interaction,
        "age_error_by_mortality_slope": age_interaction,
        "age_error_mortality_scale": 1.0 + age_interaction * (
            gompertz_b / audit.REFERENCE_MORTALITY_AGE_SLOPE - 1.0
        ),
    }


def test_registered_realized_envelopes_match_published_generator_law() -> None:
    expected = {
        axis: {
            "development": DEVELOPMENT_BAND[axis],
            "public": PUBLIC_ENVELOPE[axis],
        }
        for axis in audit.AXES
    }
    expected["linkage_urban_gradient"] = {
        "development": _interaction_envelope(
            DEVELOPMENT_BAND["linkage_urban_gradient"],
            DEVELOPMENT_BAND["migration_age_pattern"],
            COEFFICIENT_RANGES["linkage_gradient_by_migration"],
        ),
        "public": _interaction_envelope(
            PUBLIC_ENVELOPE["linkage_urban_gradient"],
            PUBLIC_ENVELOPE["migration_age_pattern"],
            COEFFICIENT_RANGES["linkage_gradient_by_migration"],
        ),
    }
    expected["missingness_target_dependence"] = {
        "development": _interaction_envelope(
            DEVELOPMENT_BAND["missingness_target_dependence"],
            DEVELOPMENT_BAND["administrative_completeness"],
            COEFFICIENT_RANGES["health_inclusion_completeness_by_target"],
        ),
        "public": _interaction_envelope(
            PUBLIC_ENVELOPE["missingness_target_dependence"],
            PUBLIC_ENVELOPE["administrative_completeness"],
            COEFFICIENT_RANGES["health_inclusion_completeness_by_target"],
        ),
    }
    expected["age_reporting_error"] = {
        "development": _age_reporting_envelope(
            DEVELOPMENT_BAND["age_reporting_error"]
        ),
        "public": _age_reporting_envelope(
            PUBLIC_ENVELOPE["age_reporting_error"]
        ),
    }

    assert set(audit.REGISTERED_REALIZED_MECHANISM_ENVELOPES) == set(audit.AXES)
    for axis in audit.AXES:
        for family in ("development", "public"):
            assert audit.REGISTERED_REALIZED_MECHANISM_ENVELOPES[axis][family] == \
                pytest.approx(expected[axis][family])


def test_interacted_mechanism_is_not_conflated_with_raw_axis_policy() -> None:
    intensity = {
        axis: sum(DEVELOPMENT_BAND[axis]) / 2.0 for axis in audit.AXES
    }
    intensity.update({
        "administrative_completeness": 1.3701720989710344,
        "missingness_target_dependence": 1.016875136531417,
    })
    coefficient = 0.8377418938547305
    realized = audit._realized_mechanisms(
        intensity,
        _coefficient_record(intensity, health_interaction=coefficient),
        {"gompertz_b": 0.105},
    )

    raw = intensity["missingness_target_dependence"]
    assert DEVELOPMENT_BAND["missingness_target_dependence"][0] <= raw <= \
        DEVELOPMENT_BAND["missingness_target_dependence"][1]
    assert realized["missingness_target_dependence"] == \
        pytest.approx(1.3322169380099145)
    assert realized["missingness_target_dependence"] > \
        DEVELOPMENT_BAND["missingness_target_dependence"][1]
    assert realized["missingness_target_dependence"] <= \
        audit.REGISTERED_REALIZED_MECHANISM_ENVELOPES[
            "missingness_target_dependence"
        ]["development"][1]


def test_realized_mechanisms_require_complete_consistent_coefficients() -> None:
    intensity = {
        axis: sum(DEVELOPMENT_BAND[axis]) / 2.0 for axis in audit.AXES
    }
    coefficients = _coefficient_record(intensity, gompertz_b=0.121)
    realized = audit._realized_mechanisms(
        intensity,
        coefficients,
        {"gompertz_b": 0.121},
    )
    assert realized["age_reporting_error"] == pytest.approx(
        intensity["age_reporting_error"]
        * coefficients["age_error_mortality_scale"]
    )

    missing = dict(coefficients)
    del missing["linkage_gradient_by_migration"]
    with pytest.raises(ValueError, match="missing required values"):
        audit._realized_mechanisms(intensity, missing, {"gompertz_b": 0.121})

    inconsistent = dict(coefficients)
    inconsistent["administrative_completeness"] += 0.01
    with pytest.raises(ValueError, match="differs from the design intensity"):
        audit._realized_mechanisms(
            intensity,
            inconsistent,
            {"gompertz_b": 0.121},
        )


def test_axis_range_record_applies_raw_and_realized_envelopes_separately() -> None:
    frame = pd.DataFrame({
        "regime": ["development", "development", "hidden", "hidden"],
        "axis_intensity_missingness_target_dependence": [0.2, 1.3, 0.25, 1.25],
        "realized_mechanism_missingness_target_dependence": [
            0.074,
            1.3322169380099145,
            0.386,
            1.1947,
        ],
    })

    record = audit._axis_range_record(frame, "missingness_target_dependence")

    assert record["correlation_target"] == "realized_mechanism"
    assert record["axis_intensity_range_observed"]["hidden"] == [0.25, 1.25]
    assert record["realized_mechanism_range_observed"]["pooled"][1] == \
        pytest.approx(1.3322169380099145)
    assert record["registered_realized_mechanism_envelopes"]["development"] == \
        [0.074, 2.119]

    outside_raw_policy = frame.copy()
    outside_raw_policy.loc[outside_raw_policy["regime"] == "hidden",
                           "axis_intensity_missingness_target_dependence"] = 1.31
    with pytest.raises(ValueError, match="hidden raw axis intensity"):
        audit._axis_range_record(
            outside_raw_policy,
            "missingness_target_dependence",
        )


def test_identifiability_receipt_requires_exact_registered_world_family() -> None:
    frame = pd.DataFrame([
        {"world": world, "regime": regime}
        for world, regime in audit.EXPECTED_WORLD_REGIMES.items()
    ])
    audit._validate_world_family(frame)

    wrong = frame.copy()
    wrong.loc[wrong["world"] == "qual-5", "world"] = "qual-6"
    with pytest.raises(ValueError, match=r"registered 12\+6 audit"):
        audit._validate_world_family(wrong)

    assert audit.ANCHOR_CORRELATION_THRESHOLD == 0.4
    assert audit.RECEIPT_SCHEMA == "meridia.v4.regime-identifiability-audit.v3"
    assert [audit.EXPECTED_WORLD_SEEDS[f"dev-{index:02d}"] for index in range(12)] \
        == list(builder.DEVELOPMENT_SEEDS)
    assert [audit.EXPECTED_WORLD_SEEDS[f"qual-{index}"] for index in range(6)] \
        == list(builder.QUALIFICATION_SEEDS)


def test_packet_preflight_rejects_wrong_family_before_retained_reads(tmp_path) -> None:
    packets = []
    for world, packet_class in audit.EXPECTED_WORLD_PACKET_CLASSES.items():
        packet = tmp_path / world
        packet.mkdir()
        (packet / "manifest.json").write_text(json.dumps({
            "schema": audit.PACKET_MANIFEST_SCHEMA,
            "development": world.startswith("dev-"),
            "packet_class": packet_class,
            "participant": {},
            "retained": {},
        }))
        packets.append(packet)

    ordered = audit._preflight_packets([str(packet) for packet in reversed(packets)])
    assert [packet.name for packet in ordered] == list(audit.EXPECTED_WORLD_REGIMES)

    graded = tmp_path / "graded-0"
    graded.mkdir()
    wrong_family = packets[:-1] + [graded]
    with pytest.raises(ValueError, match=r"registered 12\+6 audit"):
        audit._preflight_packets([str(packet) for packet in wrong_family])

    (packets[0] / "manifest.json").write_text(json.dumps({
        "schema": audit.PACKET_MANIFEST_SCHEMA,
        "development": True,
        "packet_class": "qualification",
        "participant": {},
        "retained": {},
    }))
    with pytest.raises(ValueError, match="manifest class differs"):
        audit._preflight_packets([str(packet) for packet in packets])


def test_retained_world_validation_binds_seed_cell_and_hidden_policy() -> None:
    development = build_world_mechanisms(1101, "development", cell=0)
    dev_world = {
        "seed": 1101,
        "packet_class": "development",
        "regime": "development",
        "params": {"regime": "development", "design_cell": 0},
        "character": {"gompertz_b": 0.105},
        "mechanisms": development.record(),
    }
    _, realized = audit._validate_world_record(
        Path("dev-00"),
        dev_world,
    )
    assert set(realized) == set(audit.AXES)

    wrong_cell = json.loads(json.dumps(dev_world))
    wrong_cell["mechanisms"]["design"]["cell"] = 1
    with pytest.raises(ValueError, match="development design cell differs"):
        audit._validate_world_record(Path("dev-00"), wrong_cell)

    wrong_seed = json.loads(json.dumps(dev_world))
    wrong_seed["seed"] = 1102
    with pytest.raises(ValueError, match="seed differs"):
        audit._validate_world_record(Path("dev-00"), wrong_seed)

    hidden = build_world_mechanisms(2101, "hidden")
    hidden_world = {
        "seed": 2101,
        "packet_class": "qualification",
        "regime": "hidden",
        "params": {"regime": "hidden", "design_cell": None},
        "character": {"gompertz_b": 0.105},
        "mechanisms": hidden.record(),
    }
    audit._validate_world_record(Path("qual-0"), hidden_world)

    bad_outside = json.loads(json.dumps(hidden_world))
    bad_outside["mechanisms"]["design"]["outside"][0] = \
        "administrative_completeness"
    bad_outside["mechanisms"]["design"]["outside"].sort()
    with pytest.raises(ValueError, match="hidden outside-axis policy differs"):
        audit._validate_world_record(Path("qual-0"), bad_outside)
