"""Participant-side regional shock-loading identification and propagation."""

from __future__ import annotations

from dataclasses import fields

import numpy as np
import pytest

from meridia.actuarial import ObligationContract
from meridia.methods import actuarial_reference as AR
from meridia.methods import bayesian, controls, design_based, third_reference


def _family_contract() -> dict:
    return {
        "shock_family": {
            "annual_rate": 0.20,
            "kinds": {
                "mortality_spike": {
                    "mortality_multiplier": [1.5, 3.0],
                    "admission_multiplier": [1.4, 2.6],
                },
                "migration_wave": {"leave_home_multiplier": [1.8, 3.0]},
                "baby_bust": {"fertility_multiplier": [0.45, 0.75]},
            },
            "regional_loading_band": [0.35, 1.80],
            "regional_loading": (
                "mortality and admission multiplier m lands in region r as "
                "1 + L_r * (m - 1); the realized vector is not published"
            ),
        }
    }


def _experience(shock_year: int | None = None) -> tuple[dict, np.ndarray]:
    years = np.arange(2020, 2025)
    states = 4
    bands = len(AR.ACTUARIAL_AGE_BANDS)
    exposure = np.full((len(years), states, bands, 2), 20_000.0)
    deaths = np.zeros_like(exposure)
    events = np.zeros_like(exposure)
    loading = np.asarray([0.40, 0.80, 1.20, 1.70])
    rng = np.random.default_rng(120)
    for year_index in range(len(years)):
        trend = np.exp(-0.02 * year_index)
        mortality_multiplier = np.ones(states)
        incidence_multiplier = np.ones(states)
        if year_index == shock_year:
            mortality_multiplier = 1.0 + loading * (2.40 - 1.0)
            incidence_multiplier = 1.0 + loading * (2.10 - 1.0)
        deaths[year_index] = rng.poisson(
            exposure[year_index]
            * 0.008
            * trend
            * mortality_multiplier[:, None, None]
        )
        events[year_index] = rng.poisson(
            exposure[year_index]
            * 0.025
            * trend
            * incidence_multiplier[:, None, None]
        )
    return {
        "years": years,
        "exposure": exposure,
        "deaths": deaths,
        "qualifying_events": events,
    }, loading


def _contract(family: dict, horizon_months: int = 24) -> AR.ActuarialContract:
    return AR.ActuarialContract(
        obligation=ObligationContract(
            monthly_benefit=150.0,
            qualifying_event_cost=15_000.0,
            death_benefit=7_500.0,
            monthly_discount_rate=0.002,
            eligibility_min_age=65,
            horizon_months=horizon_months,
        ),
        region_of_county=np.asarray([0, 1]),
        n_regions=2,
        reserve_total=0.0,
        reserve_weights=np.ones(2),
        gamma=0.25,
        anchor_item="recent_hospitalization",
        anchor_sensitivity=1.0,
        anchor_specificity=1.0,
        anchor_window_months=12,
        experience_years=5,
        experience_file="experience_history.csv",
        experience_last_tick=0,
        shock_family=family,
    )


def _rates() -> dict:
    shape = (2, 101, 2)
    return {
        "mortality": np.full(shape, 0.02),
        "incidence": np.full(shape, 0.03),
        "migration": np.zeros(shape),
        "migration_se": np.zeros(shape),
        "not_yet": np.ones(shape),
        "fertility": 0.0,
        "mortality_drift": 0.0,
        "mortality_drift_se": 0.0,
        "incidence_drift": 0.0,
        "incidence_drift_se": 0.0,
        "mortality_log_sd": 0.0,
        "incidence_log_sd": 0.0,
    }


def test_public_regional_loading_band_and_formula_are_parsed_not_executed():
    family = AR.read_shock_family(_family_contract())
    assert family["regional_loading_band"] == (0.35, 1.80)
    assert family["regional_loading_formula"] == AR.REGIONAL_LOADING_FORMULA

    incomplete = _family_contract()
    incomplete["shock_family"].pop("regional_loading")
    with pytest.raises(AR.MissingActuarialInputs, match="both a band and a formula"):
        AR.read_shock_family(incomplete)

    unsupported = _family_contract()
    unsupported["shock_family"]["regional_loading"] = "loadings come from retained truth"
    with pytest.raises(AR.MissingActuarialInputs, match="unsupported regional-loading formula"):
        AR.read_shock_family(unsupported)


def test_clean_history_marginalizes_over_the_public_band_per_outer_path():
    family = AR.read_shock_family(_family_contract())
    experience, _ = _experience()
    evidence = AR.infer_regional_shock_loadings(experience, family)

    assert evidence["mode"] == "public_band_marginalization"
    assert evidence["identified_year"] is None
    assert evidence["uses_retained_realized_loadings"] is False
    assert evidence["target_shock_annual_probability"] == pytest.approx(0.20 / 3.0)
    draws = AR.sample_regional_loading_paths(
        np.random.default_rng(44), 512, 4, family, evidence
    )
    assert draws.shape == (512, 4)
    assert draws.min() >= 0.35 and draws.max() <= 1.80
    assert np.unique(draws, axis=0).shape[0] == 512
    assert np.array_equal(
        draws,
        AR.sample_regional_loading_paths(
            np.random.default_rng(44), 512, 4, family, evidence
        ),
    )


def test_identifiable_history_uses_only_public_experience_for_a_clipped_posterior():
    family = AR.read_shock_family(_family_contract())
    raw, truth_loading = _experience(shock_year=2)

    class ParticipantExperience(dict):
        def __getitem__(self, key):
            if key in {"retained", "realized_loading", "world"}:
                raise AssertionError("retained input was requested")
            return super().__getitem__(key)

    experience = ParticipantExperience(raw)
    evidence = AR.infer_regional_shock_loadings(experience, family)
    assert evidence["mode"] == "participant_experience_posterior"
    assert evidence["identified_year"] == 2022
    assert evidence["uses_retained_realized_loadings"] is False
    assert set(evidence["input_fields"]) == {
        "years",
        "exposure",
        "deaths",
        "qualifying_events",
    }
    estimate = np.asarray(evidence["loading_mean"])
    assert np.corrcoef(estimate, truth_loading)[0, 1] > 0.98
    assert np.all((estimate >= 0.35) & (estimate <= 1.80))
    draws = AR.sample_regional_loading_paths(
        np.random.default_rng(81), 1000, 4, family, evidence
    )
    assert draws.min() >= 0.35 and draws.max() <= 1.80
    assert np.corrcoef(draws.mean(axis=0), truth_loading)[0, 1] > 0.98


def test_one_loading_vector_is_fixed_within_a_path_and_redrawn_across_paths():
    family = AR.read_shock_family(_family_contract())
    experience, _ = _experience()
    evidence = AR.infer_regional_shock_loadings(experience, family)
    loading = AR.sample_regional_loading_paths(
        np.random.default_rng(3), 128, 3, family, evidence
    )
    first_year = AR.regionalize_shock_multiplier(np.full(128, 1.6), loading)
    later_year = AR.regionalize_shock_multiplier(np.full(128, 2.4), loading)
    assert np.allclose((first_year - 1.0) / 0.6, loading)
    assert np.allclose((later_year - 1.0) / 1.4, loading)
    assert np.unique(loading, axis=0).shape[0] == len(loading)


def test_simulation_materializes_one_deterministic_loading_vector_per_path():
    family = AR.read_shock_family(_family_contract())
    experience, _ = _experience()
    evidence = AR.infer_regional_shock_loadings(experience, family)
    paths = np.zeros((64, 2, 101, 2))
    paths[:, :, 70, :] = 100.0
    params = AR.SimulationParams(n_paths=64, path_chunk=16, seed=913)
    first = AR.simulate_liabilities(paths, _rates(), _contract(family), params, evidence)
    second = AR.simulate_liabilities(paths, _rates(), _contract(family), params, evidence)

    assert np.array_equal(first["regional_loading"], second["regional_loading"])
    assert np.array_equal(first["liability"], second["liability"])
    assert first["regional_loading_draws_per_path"] == 1
    assert first["regional_loading_held_years"] == 2
    assert np.unique(first["regional_loading"], axis=0).shape[0] == 64
    diagnostics = AR.regional_loading_diagnostics(
        evidence, first["regional_loading"], first["regional_loading_held_years"]
    )
    assert diagnostics["mode"] == "public_band_marginalization"
    assert diagnostics["one_vector_per_outer_path"] is True
    assert diagnostics["held_across_horizon"] is True
    assert diagnostics["uses_retained_realized_loadings"] is False
    assert len(diagnostics["predictive_draw_digest"]) == 64

    different_chunk = AR.simulate_liabilities(
        paths,
        _rates(),
        _contract(family),
        AR.SimulationParams(n_paths=64, path_chunk=32, seed=913),
        evidence,
    )
    assert np.array_equal(first["regional_loading"], different_chunk["regional_loading"])


def test_all_reference_lines_share_the_participant_only_logic_and_controls_get_no_override():
    assert design_based.AR is AR
    assert bayesian.AR is AR
    assert third_reference.AR is AR
    assert controls.AR is AR
    parameter_names = {field.name for field in fields(AR.LayerParams)}
    assert "regional_loading" not in parameter_names
    assert "realized_regional_loading" not in parameter_names
    assert all(
        not any("loading" in key for key in switches)
        for switches in controls.ACTUARIAL_SWITCHES.values()
    )
