from dataclasses import replace

import numpy as np
import pytest

from meridia.mechanisms import _vector_digest, build_world_mechanisms


def test_mechanism_record_binds_county_effect_order() -> None:
    mechanisms = build_world_mechanisms(1105, "development", cell=4)
    effects = {
        "coverage": np.asarray([0.2, -0.4, 0.7, -0.1], dtype=np.float64),
        "linkage": np.asarray([-0.3, 0.1, 0.8, -0.6], dtype=np.float64),
    }
    original = replace(mechanisms, effects=effects)
    permuted = replace(
        original,
        effects={family: values[::-1].copy() for family, values in effects.items()},
    )

    original_record = original.record()
    permuted_record = permuted.record()
    assert original_record["county_effect_sd"] == permuted_record["county_effect_sd"]
    assert original_record["county_effect_digest"] != \
        permuted_record["county_effect_digest"]


def test_mechanism_record_binds_applied_county_shock_order() -> None:
    mechanisms = build_world_mechanisms(1105, "development", cell=4)
    original = replace(
        mechanisms,
        region_shock_loading=np.asarray([0.8, 1.2], dtype=np.float64),
        county_shock_loading=np.asarray([0.8, 1.2, 0.8, 1.2], dtype=np.float64),
    )
    permuted = replace(
        original,
        county_shock_loading=original.county_shock_loading[::-1].copy(),
    )

    original_record = original.record()
    permuted_record = permuted.record()
    assert original_record["region_shock_loading"] == \
        permuted_record["region_shock_loading"]
    assert original_record["county_shock_loading_digest"] != \
        permuted_record["county_shock_loading_digest"]


def test_vector_digest_is_canonical_across_byte_order() -> None:
    little = np.asarray([0.25, -1.5, 3.0], dtype="<f8")
    big = np.asarray([0.25, -1.5, 3.0], dtype=">f8")

    assert _vector_digest(little) == _vector_digest(big)
    assert len(_vector_digest(little)) == 64

    with pytest.raises(ValueError, match="one-dimensional"):
        _vector_digest(np.zeros((2, 2)))
    with pytest.raises(ValueError, match="finite"):
        _vector_digest(np.asarray([0.0, np.nan]))
