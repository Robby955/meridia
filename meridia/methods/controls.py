"""The control battery: plausible shortcuts that must each fail a named gate.

Each control is a deliberate omission a rushed analyst might make. Bars are frozen only
if every control fails at least one gate on every hidden world; a control that passes
means the gate it targets is too loose.

- ``register_only``: trust the deduplicated register as the population; no coverage
  correction, survey used only for incomes with design weights, tight intervals.
  Targets the accuracy bar on counts and the coverage floor.
- ``survey_only``: design-weighted survey estimates alone, no nonresponse adjustment,
  no register. Targets accuracy on counts at every level.
- ``no_dedup``: method A without deduplication of the population source. Targets the
  accuracy bar on counts (over-count from duplicates and splits).
- ``inflated_intervals``: method A's points with intervals of plus or minus 40 percent.
  Targets the interval-score ceiling.
- ``static_projection``: method A now, projection equal to the present. Targets the
  projection accuracy bar on elders and children.
- ``uniform_allocation``: method A with reserve slack above the submitted regional
  q95 floors spread equally across regions. Targets the reserve-skill ceiling.
- ``benchmark_only``: the benchmark nation total spread over counties in proportion
  to raw register counts, with tight intervals. Targets the accuracy bar on national
  counts (the benchmark carries its own bias) and the county bars.
- ``exact_key_union``: the count recipe that cleared version two, on the version-three
  surface. One row per exact name, birth-tick, and sex key in each source; the nation
  is the union of keys across the population, income, and health sources times a
  constant fitted on the development worlds; counties are the nation times the
  population source's reported county shares; widths come from the between-world
  spread of the constant. Targets the accuracy bar on national and county counts and
  the pooled count coverage floor (the fitted constant does not transfer to a world
  whose coverage sits outside the development band, and the exact key no longer
  identifies a person).

The version-four battery, one per targeted ablation of protocol section 11. Each keeps
the strong line intact and removes exactly one step, so a control that clears its gate
says the gate is loose rather than that the control was subtle.

- ``deterministic_linkage``: exact-key linkage between the register vintages instead of
  a probabilistic one, and rates read straight off the archive with raw register counts
  as exposure. Ablation 3. Targets the mortality and incidence rate gates: the exact key
  misses every record whose name, birth month or sex was reported differently in the two
  vintages and over-links whenever a key repeats, and unadjusted archive counts carry
  the coverage churn into the death rate.
- ``ignore_health_selection``: the whole line with the inclusion probability held at one,
  so the survey anchor is never used. Ablation 4. Targets the incidence gate and, through
  the projected first events, the tail and reserve gates.
- ``development_average_regime``: mortality improvement and its uncertainty fixed at the
  development-world average instead of read from this world's experience file. The
  average is the one measured on the development worlds, in calibration A; with no
  calibration to hand it is the midpoint and the spread of the development band the
  contract publishes, which is what that average estimates. Ablation 5. Targets the
  projection and the tails.
- ``experience_history_only``: the aggregate experience file extrapolated on its own,
  with no microdata at all. Counts come from the last published year's exposure spread
  over counties by land area and aged forward; rates and the liability come from the
  file's own state levels; households, money and education have no source in that file
  and are filed as zero rather than borrowed from a register the control says it does not
  need. Targets the county and state count gates, the rate gates through a level that is a
  year and a half stale, and the tail gates through a distribution with no linkage,
  selection or reconstruction uncertainty in it.
- ``mean_only_tail``: the reference's own liability paths, with the mean submitted as the
  95th percentile. Ablation 6. Targets the exceedance criterion.
- ``normal_tail``: the reference's paths summarised by mean plus 1.645 standard
  deviations. Ablation 6. Targets the quantile score, to the extent the regional
  liability distribution is skewed.
- ``padded_tail``: the reference's quantile plus a cushion of six tenths of the expected
  cost. Ablation 7. Targets the lower calibration bound and the quantile score. The
  public reserve total bounds how far a padded tail can travel, since the submission
  must satisfy sum A = R.
- ``proportional_reserve``: the reference forecast with the reserve split in proportion
  to projected eligible exposure rather than by marginal expected shortfall.
  Ablation 8. Targets the decision skill gate.
- ``version_three_recipe``: the recipe that solved version three, on the
  version-four surface.
  Longitudinal matching on the within-source identifier, one national growth factor for
  every county, one decoded global income scale, counts built from an equal blend of the
  income and population source county shares, deterministic register-vintage mortality,
  archive incidence with no selection correction, and point tails. Proof obligation 2.
  Targets the county count and rate gates, then the tail and reserve gates.
"""

from __future__ import annotations

import hashlib
import json
import tempfile
from dataclasses import replace
from pathlib import Path

import numpy as np

from ..release import AGE_BANDS, AGE_BAND_LABELS, ESTIMAND_IDS, SEX_LABELS
from . import actuarial_reference as AR
from . import design_based as A
from .common import COUNT_ITEMS, load_packet

CONTROLS = (
    "register_only",
    "survey_only",
    "no_dedup",
    "inflated_intervals",
    "static_projection",
    "uniform_allocation",
    "benchmark_only",
    "exact_key_union",
)
ACTUARIAL_CONTROLS = (
    "deterministic_linkage",
    "ignore_health_selection",
    "development_average_regime",
    "mean_only_tail",
    "normal_tail",
    "padded_tail",
    "proportional_reserve",
    "version_three_recipe",
    "experience_history_only",
)
ALL_CONTROLS = CONTROLS + ACTUARIAL_CONTROLS

# One layer switch per version-four control. version_three_recipe also rebuilds the release
# and projection tables, and experience_history_only builds its own population, so both are
# handled apart from this table. development_average_regime's override is not a constant
# either: it is the development-world average, built at run time.
ACTUARIAL_SWITCHES = {
    "deterministic_linkage": {
        "deterministic_linkage": True,
        "archive_only_rates": True,
    },
    "ignore_health_selection": {"ignore_health_selection": True},
    "development_average_regime": {},
    "mean_only_tail": {"tail": "mean"},
    "normal_tail": {"tail": "normal"},
    "padded_tail": {"tail": "padded", "padding": 1.6},
    "proportional_reserve": {"allocation": "proportional"},
    "version_three_recipe": {
        "deterministic_linkage": True,
        "archive_only_rates": True,
        "ignore_health_selection": True,
        "tail": "mean",
        "allocation": "proportional",
        "reconstruction_uncertainty": False,
        "rake_to_experience": False,
        "simulation": AR.SimulationParams(process_noise=False, parameter_noise=False),
    },
}

DELETION_CONTROLS = {
    "reconstruction_uncertainty": {"reconstruction_uncertainty": False},
    "informative_selection": {"ignore_health_selection": True},
    "regime_recombination": {},
    "predictive_tails": {"tail": "mean"},
    "reserve_allocation": {"allocation": "proportional"},
}
DECOMPOSITION_CONTROLS = (
    "design_reconstruction_oracle_tail",
    "true_population_normal_tail",
)

CONTROL_TARGET_COMPOSITES = {
    "register_only": "release_accuracy",
    "survey_only": "release_accuracy",
    "no_dedup": "release_accuracy",
    "inflated_intervals": "interval_quality",
    "static_projection": "release_accuracy",
    "uniform_allocation": "reserve_skill",
    "benchmark_only": "release_accuracy",
    "exact_key_union": "release_accuracy",
    "deterministic_linkage": "exposures_and_rates",
    "ignore_health_selection": "exposures_and_rates",
    "development_average_regime": "tail_calibration",
    "mean_only_tail": "tail_calibration",
    "normal_tail": "tail_calibration",
    "padded_tail": "tail_calibration",
    "proportional_reserve": "reserve_skill",
    "version_three_recipe": "release_accuracy",
    "experience_history_only": "release_accuracy",
    "reconstruction_uncertainty": "interval_quality",
    "informative_selection": "exposures_and_rates",
    "regime_recombination": "tail_calibration",
    "predictive_tails": "tail_calibration",
    "reserve_allocation": "reserve_skill",
}
QUALIFICATION_CONTROLS = ALL_CONTROLS + tuple(DELETION_CONTROLS)
if set(CONTROL_TARGET_COMPOSITES) != set(QUALIFICATION_CONTROLS):
    raise RuntimeError("every qualification control needs exactly one target composite")

VERSION_THREE_DISCRETE_INCOME_SCALES = (0.55, 0.75, 1.0)
VERSION_THREE_RELEASE_WIDTH = {
    "count": 0.30,
    "median_household_income": 0.12,
    "mean_income_adults": 0.10,
    "tertiary_share_25_plus": 0.008,
    "low_income_household_share": 0.025,
}
VERSION_THREE_PROJECTION_WIDTH = {
    "count": 0.32,
    "median_household_income": 0.20,
    "mean_income_adults": 0.15,
    "tertiary_share_25_plus": 0.030,
    "low_income_household_share": 0.050,
}

EXACT_KEY = ["given_code", "family_code", "birth_tick", "sex"]
UNION_WIDTH = {
    "nation": 2.5,
    "state": 2.2,
    "county": 2.0,
}  # multiples of the development spread
UNION_MIN_HALF = 0.004


def exact_key_union_raw(data: dict, tick: int, n_counties: int) -> tuple[dict, dict]:
    """Raw counts of the exact-key recipe: the union of exact keys across the three
    person sources for persons and children, the population source alone for elders
    and households, and county shares from the population source's reported county."""
    import pandas as pd
    population = data["population"].drop_duplicates(EXACT_KEY)
    union = pd.concat([population[EXACT_KEY], data["income"].drop_duplicates(EXACT_KEY)[EXACT_KEY],
                       data["health"].drop_duplicates(EXACT_KEY)[EXACT_KEY]]).drop_duplicates()
    union_age = (tick - union["birth_tick"].to_numpy(dtype=np.int64)) // 12
    located = population[population["county"] >= 0]
    age = (tick - located["birth_tick"].to_numpy(dtype=np.int64)) // 12
    county = located["county"].to_numpy(dtype=np.int64)
    households = located.drop_duplicates("household_id")
    raw = {"persons": float(len(union)), "children_under_16": float((union_age <= 15).sum()),
           "elders_65_plus": float((age >= 65).sum()), "households": float(population["household_id"].nunique())}
    shares = {"persons": np.bincount(county, minlength=n_counties).astype(np.float64),
              "children_under_16": np.bincount(county, weights=(age <= 15), minlength=n_counties),
              "elders_65_plus": np.bincount(county, weights=(age >= 65), minlength=n_counties),
              "households": np.bincount(households["county"].to_numpy(dtype=np.int64), minlength=n_counties).astype(np.float64)}
    for item in shares:
        shares[item] = np.maximum(shares[item], 1e-9) / max(shares[item].sum(), 1e-9)
    return raw, shares


def fit_exact_key_union(dev_packet_dirs) -> dict:
    """The recipe's development-world constants: truth over raw per count item, the
    between-world spread of that ratio, and the 90th percentile of the county relative
    error of nation times shares. Stored in calibration A for the control."""
    import pandas as pd
    ratios = {item: [] for item in COUNT_ITEMS}
    county_errors = {item: [] for item in COUNT_ITEMS}
    for dev in dev_packet_dirs:
        dev = Path(dev)
        data = load_packet(dev)
        tick = int(data["contract"]["ticks"]["revised"])
        raw, shares = exact_key_union_raw(data, tick, len(data["county_state"]))
        truth = pd.read_csv(dev / "participant" / "truth" / "truth_revised.csv")
        nation = truth[truth["level"] == "nation"].set_index("estimand")["value"]
        county = truth[truth["level"] == "county"]
        for item in COUNT_ITEMS:
            ratios[item].append(float(nation[item]) / raw[item])
            actual = county[county["estimand"] == item].set_index("unit")["value"]
            units = actual.index.to_numpy(dtype=np.int64)
            values = actual.to_numpy(dtype=np.float64)
            keep = values > 0
            estimate = float(nation[item]) * shares[item][units[keep]]
            county_errors[item] += list(np.abs(estimate - values[keep]) / values[keep])
    fit = {"ratio": {}, "ratio_spread": {}, "county_q90": {}}
    for item in COUNT_ITEMS:
        r = np.asarray(ratios[item], dtype=np.float64)
        fit["ratio"][item] = float(r.mean())
        fit["ratio_spread"][item] = float(np.max(np.abs(r - r.mean())) / r.mean()) if len(r) > 1 else 0.0
        fit["county_q90"][item] = float(np.quantile(county_errors[item], 0.9)) if county_errors[item] else 0.0
    return fit


def _exact_key_union_rows(data: dict, tick: int, county_state: np.ndarray, fit: dict) -> list[dict]:
    n_counties = len(county_state)
    n_states = int(county_state.max()) + 1
    raw, shares = exact_key_union_raw(data, tick, n_counties)
    rows = []
    for item in COUNT_ITEMS:
        nation = raw[item] * fit["ratio"][item]
        county = nation * shares[item]
        state = np.bincount(county_state, weights=county, minlength=n_states)
        half = {"nation": max(UNION_WIDTH["nation"] * fit["ratio_spread"][item], UNION_MIN_HALF),
                "state": max(UNION_WIDTH["state"] * fit["county_q90"][item], UNION_MIN_HALF),
                "county": max(UNION_WIDTH["county"] * fit["county_q90"][item], UNION_MIN_HALF)}
        values = [("nation", 0, nation)] + [("state", s, float(state[s])) for s in range(n_states)] +                  [("county", c, float(county[c])) for c in range(n_counties)]
        for level, unit, value in values:
            rows.append({"estimand": item, "level": level, "unit": int(unit), "estimate": float(value),
                         "lower": float(value * (1.0 - half[level])), "upper": float(value * (1.0 + half[level]))})
    return rows


def _rows_with_relative_half(point: dict, rel: float) -> list[dict]:
    rows = []
    for key in sorted(point):
        v = point[key]
        if not np.isfinite(v):
            v = 0.0
        proportion = key[0].endswith("share") or key[0].startswith("tertiary")
        half = rel if proportion else rel * abs(v)
        lower, upper = max(v - half, 0.0), v + half
        if proportion:
            upper = min(upper, 1.0)
            v = min(max(v, lower), upper)
        rows.append({"estimand": key[0], "level": key[1], "unit": int(key[2]),
                     "estimate": float(v), "lower": float(lower), "upper": float(upper)})
    return rows



def deterministic_reserve_rows(contract: dict, county_state: np.ndarray,
                               county_weight: np.ndarray) -> list[dict] | None:
    """The reserve a recipe with no tail model can honestly file.

    Every count control treats the future as a point: it has a projection and nothing
    that says how wide the distribution around it is. The reserve that follows treats the
    regional liability as deterministic, so the mean, the quantile and the shortfall are
    one number and the total is split on the region's projected share. It fails the
    exceedance criterion at once, which is the point of a control that carries no tail.
    """
    block = contract.get("reserve")
    if not block or "total" not in block:
        return None
    total = float(block["total"])
    n_regions = int(county_state.max()) + 1
    weight = np.bincount(county_state, weights=np.maximum(county_weight, 0.0),
                         minlength=n_regions)
    share = weight / weight.sum() if weight.sum() > 0 else np.full(n_regions, 1.0 / n_regions)
    value = share * total
    value[-1] = total - float(value[:-1].sum())        # the total holds exactly
    return [{"region": r, "liability_mean": float(value[r]), "q95": float(value[r]),
             "es95": float(value[r]), "allocation": float(value[r])}
            for r in range(n_regions)]


def fit_development_regime(dev_packet_dirs) -> dict:
    """The regime a method carries when it does not read this world's own file.

    Each development world's experience file is read with the same estimator the strong
    line uses, and the four numbers stored here are the averages of those readings: the
    mortality and incidence drift, and the standard error each reading reported. That is
    what "the development-world average" means, and it is the quantity ablation 5 fixes
    in place of the world in front of it. Stored in calibration A beside the other
    development-world constants.
    """
    drifts = {"mortality": [], "incidence": []}
    errors = {"mortality": [], "incidence": []}
    for dev in dev_packet_dirs:
        dev = Path(dev)
        contract = json.loads((dev / "participant" / "contract.json").read_text())
        experience = AR.load_experience(dev, contract)
        if experience is None:
            continue
        n_states = int(np.asarray(contract.get("n_states", 1), dtype=np.int64)) or 1
        arrays = AR.experience_arrays(experience, n_states)
        family = AR.read_shock_family(contract)
        for kind, counts in (("mortality", arrays["deaths"]),
                             ("incidence", arrays["qualifying_events"])):
            fit = AR.estimate_improvement(
                arrays["exposure"], counts, shock_family=family,
                shock_range=AR.shock_range_for(family, kind))
            drifts[kind].append(float(fit["drift"]))
            errors[kind].append(float(fit["drift_se"]))
    out = {"n_worlds": len(drifts["mortality"])}
    for kind in ("mortality", "incidence"):
        if drifts[kind]:
            out[f"{kind}_drift"] = float(np.mean(drifts[kind]))
            out[f"{kind}_drift_se"] = float(np.mean(errors[kind]))
    return out


def development_regime_override(contract: dict, calibration: dict | None) -> dict:
    """The override ablation 5 runs under, from the calibration or from the contract.

    The measured average over the development worlds is the right quantity and it is what
    calibration A carries. Without one, the published development band of the mortality
    improvement axis says the same thing in closed form: the design is balanced, so the
    average intensity over the twelve development worlds is the band's midpoint, and the
    spread a method would carry from that average is the band's own standard deviation.
    The axis is a proportional decline, so the drift it implies is the log of one minus it.
    """
    fitted = (calibration or {}).get("development_regime")
    if fitted and "mortality_drift" in fitted:
        return {
            "mortality_drift": float(fitted["mortality_drift"]),
            "mortality_drift_se": float(fitted["mortality_drift_se"]),
            "incidence_drift": float(fitted.get("incidence_drift", 0.0)),
            "incidence_drift_se": float(
                fitted.get("incidence_drift_se", fitted["mortality_drift_se"])
            ),
        }
    band = ((contract.get("mechanisms") or {}).get("development_band") or {}).get(
        "mortality_improvement"
    )
    if not band:
        raise ValueError(
            "development_average_regime needs either the development-world "
            "average in calibration A or the published development band"
        )
    low, high = float(band[0]), float(band[1])
    drift = float(np.log(max(1.0 - 0.5 * (low + high), 1e-6)))
    spread = float((high - low) / np.sqrt(12.0))
    return {
        "mortality_drift": drift,
        "mortality_drift_se": spread,
        "incidence_drift": 0.0,
        "incidence_drift_se": spread,
    }


def _read_rows(path: Path) -> list[dict]:
    import pandas as pd

    return pd.read_csv(path).to_dict("records")


def _version_three_count_vectors(
    frame, id_column: str, tick: int, n_counties: int
) -> dict[str, np.ndarray]:
    """One record per within-source identifier, as used by the version-three recipe."""
    person = frame.drop_duplicates(id_column).copy()
    person = person[(person["county"] >= 0) & (person["county"] < n_counties)]
    county = person["county"].to_numpy(dtype=np.int64)
    age = (tick - person["birth_tick"].to_numpy(dtype=np.int64)) // 12
    household = person.drop_duplicates("household_id")
    return {
        "persons": np.bincount(county, minlength=n_counties).astype(np.float64),
        "households": np.bincount(
            household["county"].to_numpy(dtype=np.int64), minlength=n_counties
        ).astype(np.float64),
        "children_under_16": np.bincount(
            county, weights=age <= 15, minlength=n_counties
        ).astype(np.float64),
        "elders_65_plus": np.bincount(
            county, weights=age >= 65, minlength=n_counties
        ).astype(np.float64),
    }


def _version_three_transition_ratios(
    preliminary, revised, tick_pre: int, tick_rev: int, horizon_months: int
) -> dict:
    """Repeat the within-source cohort transition that solved version three.

    Retention and arrivals are measured by the source's own identifier in one observed
    interval, smoothed over neighbouring age bins, and repeated to the horizon. Version
    four reissues that identifier, so the same arithmetic is intentionally brittle here.
    """
    id_column = "taxpayer_id"
    pre = preliminary.drop_duplicates(id_column)
    rev = revised.drop_duplicates(id_column)
    step_months = max(tick_rev - tick_pre, 1)
    steps = max(int(round(horizon_months / step_months)), 1)
    n_bins = max(int(np.ceil((AR.MAX_AGE + 10) * 12 / step_months)), 32)
    pre_bin = np.clip(
        ((tick_pre - pre["birth_tick"].to_numpy(dtype=np.int64)) / step_months).astype(
            np.int64
        ),
        0,
        n_bins - 1,
    )
    rev_bin = np.clip(
        ((tick_rev - rev["birth_tick"].to_numpy(dtype=np.int64)) / step_months).astype(
            np.int64
        ),
        0,
        n_bins - 1,
    )
    pre_ids = set(pre[id_column].to_numpy())
    rev_ids = set(rev[id_column].to_numpy())
    pre_count = np.bincount(pre_bin, minlength=n_bins).astype(np.float64)
    current = np.bincount(rev_bin, minlength=n_bins).astype(np.float64)
    retained = pre[id_column].isin(rev_ids).to_numpy()
    retained_count = np.bincount(pre_bin[retained], minlength=n_bins).astype(np.float64)
    arrived = ~rev[id_column].isin(pre_ids).to_numpy()
    arrivals = np.bincount(rev_bin[arrived], minlength=n_bins).astype(np.float64)
    kernel = np.ones(5)
    retention = np.convolve(retained_count, kernel, mode="same") / np.maximum(
        np.convolve(pre_count, kernel, mode="same"), 1.0
    )
    retention = np.clip(retention, 0.0, 1.0)
    forecast = current.copy()
    for _ in range(steps):
        moved = np.zeros_like(forecast)
        moved[1:] = forecast[:-1] * retention[:-1]
        moved[-1] += forecast[-1] * retention[-1]
        forecast = moved + arrivals
    child_end = int(np.ceil(16 * 12 / step_months))
    elder_start = int(np.floor(65 * 12 / step_months))
    current_values = {
        "persons": float(current.sum()),
        "children_under_16": float(current[:child_end].sum()),
        "elders_65_plus": float(current[elder_start:].sum()),
    }
    future_values = {
        "persons": float(forecast.sum()),
        "children_under_16": float(forecast[:child_end].sum()),
        "elders_65_plus": float(forecast[elder_start:].sum()),
    }
    return {
        item: float(
            np.clip(future_values[item] / max(current_values[item], 1.0), 0.5, 2.0)
        )
        for item in current_values
    }


def _version_three_income(
    data: dict, tick: int, county_state: np.ndarray, survey_stats: dict | None = None
) -> tuple[dict, dict, float]:
    """Decode one discrete money unit and compute source-only income summaries."""
    frame = data["income"].drop_duplicates("taxpayer_id").copy()
    frame = frame[(frame["county"] >= 0) & (frame["county"] < len(county_state))]
    frame["age"] = (tick - frame["birth_tick"]) // 12
    raw = (
        frame.loc[
            (frame["age"] >= 16) & frame["employment_income_cents"].notna(),
            "employment_income_cents",
        ].to_numpy(dtype=np.float64)
        / 100.0
    )
    raw_mean = float(raw.mean()) if len(raw) else 1.0
    survey_mean = (
        float(survey_stats["mean_income_adults"]["nation"])
        if survey_stats is not None
        else raw_mean
    )
    observed = raw_mean / max(survey_mean, 1e-9)
    scale = min(
        VERSION_THREE_DISCRETE_INCOME_SCALES,
        key=lambda candidate: abs(np.log(max(observed, 1e-9) / candidate)),
    )
    frame["decoded_income"] = frame["employment_income_cents"].fillna(0.0) / (
        100.0 * scale
    )
    n_counties = len(county_state)
    adult = frame[frame["age"] >= 16]
    mean = (
        adult.groupby("county")["decoded_income"]
        .mean()
        .reindex(range(n_counties))
        .to_numpy(dtype=np.float64)
    )
    household = frame.groupby("household_id", sort=False).agg(
        county=("county", "first"), income=("decoded_income", "sum")
    )
    national_median = float(household["income"].median()) if len(household) else 0.0
    median = (
        household.groupby("county")["income"]
        .median()
        .reindex(range(n_counties))
        .to_numpy(dtype=np.float64)
    )
    low = (
        household.assign(low=household["income"] < 0.6 * national_median)
        .groupby("county")["low"]
        .mean()
        .reindex(range(n_counties))
        .to_numpy(dtype=np.float64)
    )
    mean = np.nan_to_num(mean, nan=raw_mean / scale)
    median = np.nan_to_num(median, nan=national_median)
    low = np.nan_to_num(
        low, nan=float(np.nanmean(low)) if np.isfinite(low).any() else 0.0
    )

    n_states = int(county_state.max()) + 1
    median_level = {"nation": national_median}
    for state in range(n_states):
        members = set(np.flatnonzero(county_state == state))
        values = household.loc[household["county"].isin(members), "income"]
        median_level[state] = float(values.median()) if len(values) else national_median
    return (
        {
            "median_household_income": median,
            "mean_income_adults": mean,
            "low_income_household_share": np.clip(low, 0.0, 1.0),
        },
        median_level,
        float(scale),
    )


def _version_three_tertiary(population, tick: int, n_counties: int) -> np.ndarray:
    frame = population.drop_duplicates("person_id").copy()
    frame["age"] = (tick - frame["birth_tick"]) // 12
    frame = frame[
        (frame["county"] >= 0)
        & (frame["county"] < n_counties)
        & (frame["age"] >= 25)
        & (frame["education"] >= 0)
    ]
    share = (
        frame.assign(tertiary=frame["education"] >= 2)
        .groupby("county")["tertiary"]
        .mean()
        .reindex(range(n_counties))
        .to_numpy(dtype=np.float64)
    )
    return np.nan_to_num(
        share, nan=float(np.nanmean(share)) if np.isfinite(share).any() else 0.0
    )


def _version_three_aggregate(
    county: dict, county_state: np.ndarray, median_level: dict
) -> dict:
    n_states = int(county_state.max()) + 1
    persons = np.maximum(np.asarray(county["persons"], dtype=np.float64), 0.0)
    out = {}
    for estimand in ESTIMAND_IDS:
        values = np.asarray(county[estimand], dtype=np.float64)
        for c, value in enumerate(values):
            out[(estimand, "county", c)] = float(value)
        if estimand in COUNT_ITEMS:
            state = np.bincount(county_state, weights=values, minlength=n_states)
            nation = float(state.sum())
        elif estimand == "median_household_income":
            state = np.asarray([median_level[s] for s in range(n_states)])
            nation = float(median_level["nation"])
        else:
            denominator = np.bincount(county_state, weights=persons, minlength=n_states)
            state = np.bincount(
                county_state, weights=values * persons, minlength=n_states
            ) / np.maximum(denominator, 1e-9)
            nation = float((values * persons).sum() / max(persons.sum(), 1e-9))
        for s, value in enumerate(state):
            out[(estimand, "state", s)] = float(value)
        out[(estimand, "nation", 0)] = nation
    return out


def _version_three_rows(point: dict, widths: dict) -> list[dict]:
    rows = []
    for key in sorted(point):
        estimand, level, unit = key
        value = float(np.nan_to_num(point[key]))
        width = widths["count"] if estimand in COUNT_ITEMS else widths[estimand]
        half = (
            width * abs(value)
            if estimand not in ("tertiary_share_25_plus", "low_income_household_share")
            else width
        )
        lower, upper = max(value - half, 0.0), value + half
        if estimand.endswith("share") or estimand.startswith("tertiary"):
            upper = min(upper, 1.0)
        rows.append(
            {
                "estimand": estimand,
                "level": level,
                "unit": int(unit),
                "estimate": value,
                "lower": float(lower),
                "upper": float(upper),
            }
        )
    return rows


def fit_version_three_recipe(dev_packet_dirs) -> dict:
    """Fit only the constants and discrete choices used by the passed version-three line."""
    import pandas as pd

    samples = []
    for packet in map(Path, dev_packet_dirs):
        data = load_packet(packet)
        tick = int(data["contract"]["ticks"]["revised"])
        tick_pre = int(data["contract"]["ticks"]["preliminary"])
        horizon_months = int(data["contract"]["ticks"]["horizon"]) - tick
        income_count = _version_three_count_vectors(
            data["income"], "taxpayer_id", tick, len(data["county_state"])
        )
        truth_now = pd.read_csv(packet / "participant" / "truth" / "truth_revised.csv")
        truth_future = pd.read_csv(
            packet / "participant" / "truth" / "truth_horizon.csv"
        )
        now = truth_now[truth_now["level"] == "nation"].set_index("estimand")["value"]
        future = truth_future[truth_future["level"] == "nation"].set_index("estimand")[
            "value"
        ]
        survey = A.impute_income(
            data["survey"].assign(weight=data["survey"]["design_weight"])
        )
        survey_stats = A.survey_statistics(survey, data["county_state"])
        income, median_level, decoded = _version_three_income(
            data, tick, data["county_state"], survey_stats
        )
        tertiary = _version_three_tertiary(
            data["population"], tick, len(data["county_state"])
        )
        county = (
            {item: income_count[item] for item in COUNT_ITEMS}
            | income
            | {"tertiary_share_25_plus": tertiary}
        )
        estimate = _version_three_aggregate(county, data["county_state"], median_level)
        transition = _version_three_transition_ratios(
            data["income_preliminary"], data["income"], tick_pre, tick, horizon_months
        )
        row = {"decoded_scale": decoded}
        for item in COUNT_ITEMS:
            row[f"current/{item}"] = float(
                now[item] / max(income_count[item].sum(), 1.0)
            )
        for item in ("persons", "children_under_16", "elders_65_plus"):
            row[f"transition/{item}"] = float(
                (future[item] / max(now[item], 1.0)) / max(transition[item], 1e-9)
            )
        row["household_growth"] = float(
            future["households"] / max(now["households"], 1.0)
        )
        for item in ("median_household_income", "mean_income_adults"):
            raw = estimate[(item, "nation", 0)]
            row[f"income/{item}"] = float(now[item] / max(raw, 1e-9))
            row[f"forecast/{item}"] = float(future[item] / max(now[item], 1e-9))
        for item in ("low_income_household_share", "tertiary_share_25_plus"):
            row[f"income/{item}"] = float(now[item] - estimate[(item, "nation", 0)])
            row[f"forecast/{item}"] = float(future[item] - now[item])
        samples.append(row)
    keys = sorted({key for row in samples for key in row if key != "decoded_scale"})
    return {
        "n_worlds": len(samples),
        "discrete_income_scales": list(VERSION_THREE_DISCRETE_INCOME_SCALES),
        **{key: float(np.mean([row[key] for row in samples])) for key in keys},
    }


def _version_three_release(
    data: dict, tick: int, county_state: np.ndarray, horizon_months: int, fit: dict
) -> tuple[list[dict], list[dict]]:
    """The registered version-three recipe, reproduced from its transcript evidence."""
    n_counties = len(county_state)
    tick_pre = int(data["contract"]["ticks"]["preliminary"])
    income_counts = _version_three_count_vectors(
        data["income"], "taxpayer_id", tick, n_counties
    )
    population_counts = _version_three_count_vectors(
        data["population"], "person_id", tick, n_counties
    )
    survey = A.impute_income(
        data["survey"].assign(weight=data["survey"]["design_weight"])
    )
    survey_stats = A.survey_statistics(survey, county_state)
    income, median_level, _ = _version_three_income(
        data, tick, county_state, survey_stats
    )
    transition = _version_three_transition_ratios(
        data["income_preliminary"], data["income"], tick_pre, tick, horizon_months
    )

    now, future = {}, {}
    for item in COUNT_ITEMS:
        inc = np.maximum(income_counts[item], 0.0)
        pop = np.maximum(population_counts[item], 0.0)
        share = 0.5 * inc / max(inc.sum(), 1.0) + 0.5 * pop / max(pop.sum(), 1.0)
        share /= max(share.sum(), 1e-9)
        nation = float(inc.sum() * fit[f"current/{item}"])
        now[item] = nation * share
        if item == "households":
            ratio = fit["household_growth"]
        else:
            ratio = transition[item] * fit[f"transition/{item}"]
        future[item] = now[item] * float(np.clip(ratio, 0.5, 2.0))

    for item in ("median_household_income", "mean_income_adults"):
        now[item] = income[item] * fit[f"income/{item}"]
        future[item] = now[item] * fit[f"forecast/{item}"]
        median_level = dict(median_level)
        if item == "median_household_income":
            for level in list(median_level):
                median_level[level] *= fit[f"income/{item}"]
    for item in ("low_income_household_share", "tertiary_share_25_plus"):
        source = (
            income[item]
            if item in income
            else _version_three_tertiary(data["population"], tick, n_counties)
        )
        now[item] = np.clip(source + fit[f"income/{item}"], 0.0, 1.0)
        future[item] = np.clip(now[item] + fit[f"forecast/{item}"], 0.0, 1.0)

    point_now = _version_three_aggregate(now, county_state, median_level)
    future_median = {
        key: value * fit["forecast/median_household_income"]
        for key, value in median_level.items()
    }
    point_future = _version_three_aggregate(future, county_state, future_median)
    return (
        _version_three_rows(point_now, VERSION_THREE_RELEASE_WIDTH),
        _version_three_rows(point_future, VERSION_THREE_PROJECTION_WIDTH),
    )


def _development_control_inputs(packet_dir: Path) -> tuple[Path, Path]:
    """Return the two retained inputs permitted for development controls only.

    Qualification and graded packets do not expose participant truth. Requiring that
    development-only marker before opening a retained file makes these controls fail
    closed when they are pointed at any other packet class.
    """
    packet_dir = Path(packet_dir).resolve()
    if any(part.lower().startswith("graded") for part in packet_dir.parts):
        raise ValueError("decomposition controls refuse graded packet paths")
    manifest_path = packet_dir / "manifest.json"
    if manifest_path.is_symlink() or manifest_path.resolve().parent != packet_dir:
        raise ValueError("decomposition controls refuse a linked packet manifest")
    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, ValueError) as error:
        raise ValueError(
            "decomposition controls require a valid development packet manifest"
        ) from error
    if manifest.get("development") is not True:
        raise ValueError("decomposition controls require a development packet")
    for side in ("participant", "retained"):
        side_path = packet_dir / side
        if side_path.is_symlink() or side_path.resolve().parent != packet_dir:
            raise ValueError(
                f"decomposition controls refuse a linked {side} directory"
            )
        linked = [
            str(path.relative_to(side_path))
            for path in side_path.rglob("*")
            if path.is_symlink()
        ]
        if linked:
            raise ValueError(
                f"decomposition controls refuse linked {side} paths: {sorted(linked)}"
            )
    truth = packet_dir / "participant" / "truth"
    ensemble = packet_dir / "retained" / "continuation_liabilities.npz"
    participant_root = (packet_dir / "participant").resolve()
    retained_root = (packet_dir / "retained").resolve()
    if truth.is_symlink() or truth.resolve().parent != participant_root:
        raise ValueError("decomposition controls refuse a linked truth directory")
    required = (truth / "truth_revised.csv", truth / "detailed_revised.csv", ensemble)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise ValueError(
            "decomposition controls require a development packet with "
            "participant truth and a retained continuation ensemble"
        )
    for path in required:
        participant_file = path.is_relative_to(packet_dir / "participant")
        side = "participant" if participant_file else "retained"
        expected_root = truth.resolve() if participant_file else retained_root
        if path.is_symlink() or path.resolve().parent != expected_root:
            raise ValueError(
                f"decomposition controls refuse a linked {side} input: {path.name}"
            )
        name = str(path.relative_to(packet_dir / side))
        claim = (manifest.get(side) or {}).get(name)
        if not isinstance(claim, dict):
            raise ValueError(f"development manifest does not bind {side}/{name}")
        with path.open("rb") as handle:
            digest = hashlib.file_digest(handle, "sha256").hexdigest()
        if claim.get("bytes") != path.stat().st_size or claim.get("sha256") != digest:
            raise ValueError(f"development manifest hash mismatch for {side}/{name}")
    return truth, ensemble


def _truth_release_rows(truth_dir: Path) -> list[dict]:
    """Exact current release rows supplied to the true-population control."""
    import pandas as pd

    truth = pd.read_csv(Path(truth_dir) / "truth_revised.csv")
    return [
        {
            "estimand": str(row.estimand),
            "level": str(row.level),
            "unit": int(row.unit),
            "estimate": float(row.value),
            "lower": float(row.value),
            "upper": float(row.value),
        }
        for row in truth.itertuples()
    ]


def _truth_population_cubes(
    truth_dir: Path, n_counties: int
) -> tuple[np.ndarray, np.ndarray]:
    """Expand development detailed truth into a single-age cube.

    Development truth identifies the published age bands, not single years inside a
    band. The single-age cube therefore spreads each true band count evenly across its
    included ages. The second return value preserves the exact submitted band counts.
    """
    import pandas as pd

    frame = pd.read_csv(Path(truth_dir) / "detailed_revised.csv")
    bands = {name: index for index, name in enumerate(AGE_BAND_LABELS)}
    sexes = {name: index for index, name in enumerate(SEX_LABELS)}
    band_cube = np.zeros((n_counties, len(AGE_BAND_LABELS), len(SEX_LABELS)))
    seen = np.zeros_like(band_cube, dtype=bool)
    for row in frame.itertuples():
        county = int(row.county)
        band = bands[str(row.age_band)]
        sex = sexes[str(row.sex)]
        if not 0 <= county < n_counties:
            raise ValueError("development detailed truth has an invalid county")
        band_cube[county, band, sex] = float(row.count)
        seen[county, band, sex] = True
    if not seen.all():
        raise ValueError("development detailed truth does not contain every cell")
    single_age = np.zeros((n_counties, AR.MAX_AGE + 1, len(SEX_LABELS)))
    for band, (low, high) in enumerate(AGE_BANDS):
        ages = np.arange(low, min(high, AR.MAX_AGE) + 1)
        single_age[:, ages, :] = band_cube[:, band, None, :] / len(ages)
    return single_age, band_cube


def run_decomposition(
    name: str,
    packet_dir: Path,
    out_dir: Path,
    calibration_path: str | None = None,
    bootstrap_replicates: int = 60,
    simulation_paths: int = 2048,
) -> dict:
    """Run one development-only reconstruction and tail decomposition control."""
    if name not in DECOMPOSITION_CONTROLS:
        raise ValueError(f"unknown decomposition control {name!r}")
    packet_dir, out_dir = Path(packet_dir), Path(out_dir)
    truth_dir, ensemble_path = _development_control_inputs(packet_dir)
    common_layer = AR.LayerParams(
        simulation=AR.SimulationParams(n_paths=simulation_paths),
    )

    if name == "design_reconstruction_oracle_tail":
        result = A.run(
            packet_dir,
            out_dir,
            A.MethodParams(
                bootstrap_replicates=bootstrap_replicates,
                calibration_path=calibration_path,
                actuarial="on",
                actuarial_params=common_layer,
            ),
        )
        with np.load(ensemble_path) as archive:
            liability = np.asarray(archive["liability"], dtype=np.float64)
            archive_weights = (
                np.asarray(archive["weights"], dtype=np.float64)
                if "weights" in archive
                else None
            )
        data = load_packet(packet_dir)
        reserve = data["contract"]["reserve"]
        weights = archive_weights
        if weights is None and reserve.get("weights"):
            weights = np.asarray(reserve["weights"], dtype=np.float64)
        design_rows = sorted(result["reserve"], key=lambda row: int(row["region"]))
        design_mean = np.asarray(
            [row["liability_mean"] for row in design_rows], dtype=np.float64
        )
        sealed_summary = AR.tail_summary(liability)
        oracle_residual_paths = (
            liability - sealed_summary["mean"][None, :] + design_mean[None, :]
        )
        summary = AR.tail_summary(oracle_residual_paths)
        allocation_detail = AR.allocate_reserve(
            oracle_residual_paths, summary["q"], float(reserve["total"]), weights
        )
        rows = AR.reserve_rows(summary, allocation_detail["allocation"])
        import pandas as pd

        pd.DataFrame(rows).to_csv(out_dir / "reserve.csv", index=False)
        result["reserve"] = rows
        result["reconstruction_actuarial_diagnostics"] = result.pop("actuarial", {})
        result["decomposition"] = {
            "name": name,
            "oracle_tail_members": int(liability.shape[0]),
            "oracle_component": "sealed centered regional tail residuals",
            "level_component": "design reconstruction regional liability mean",
            "design_liability_mean": design_mean.tolist(),
            "sealed_liability_mean": sealed_summary["mean"].tolist(),
            "submitted_liability_mean": summary["mean"].tolist(),
            "reserve_feasible": bool(allocation_detail["feasible"]),
        }
        return result

    data = load_packet(packet_dir)
    county_state = np.asarray(data["county_state"], dtype=np.int64)
    single_age, band_cube = _truth_population_cubes(truth_dir, len(county_state))
    with tempfile.TemporaryDirectory(prefix="meridia-true-population-") as temporary:
        base = A.run(
            packet_dir,
            Path(temporary),
            A.MethodParams(
                bootstrap_replicates=bootstrap_replicates,
                calibration_path=calibration_path,
                actuarial="off",
            ),
        )
    contract = data["contract"]
    layer = replace(
        common_layer,
        tail="normal",
        reconstruction_uncertainty=False,
        rake_to_experience=False,
    )
    result = AR.actuarial_submission(
        packet_dir,
        data,
        county_state,
        single_age[None],
        float(base["fertility"]["fertility_rate"]),
        _truth_release_rows(truth_dir),
        base["projection"],
        band_cube,
        2.0 * int(contract.get("disclosure_threshold", 10)),
        out_dir,
        AGE_BAND_LABELS,
        SEX_LABELS,
        layer,
    )
    if result is None:
        raise AR.MissingActuarialInputs(
            "true_population_normal_tail needs version-four actuarial inputs"
        )
    result["decomposition"] = {
        "name": name,
        "truth_source": "development participant truth",
        "single_age_rule": "uniform within each true published age band",
    }
    return result


def run_deletion(
    name: str,
    packet_dir: Path,
    out_dir: Path,
    calibration_path: str | None = None,
    bootstrap_replicates: int = 60,
    simulation_paths: int = 2048,
) -> dict:
    """Run the design reference with exactly one named layer removed."""
    if name not in DELETION_CONTROLS:
        raise ValueError(f"unknown deletion control {name!r}")
    switches = dict(DELETION_CONTROLS[name])
    if name == "regime_recombination":
        calibration = (
            json.loads(Path(calibration_path).read_text()) if calibration_path else None
        )
        contract = json.loads(
            (Path(packet_dir) / "participant" / "contract.json").read_text()
        )
        switches["regime_override"] = development_regime_override(contract, calibration)
    layer = AR.LayerParams(
        simulation=AR.SimulationParams(n_paths=simulation_paths),
        **switches,
    )
    result = A.run(
        packet_dir,
        out_dir,
        A.MethodParams(
            bootstrap_replicates=bootstrap_replicates,
            calibration_path=calibration_path,
            actuarial="on",
            actuarial_params=layer,
        ),
    )
    result["deletion"] = name
    return result


def experience_only_cube(
    arrays: dict, county_state: np.ndarray, land: np.ndarray, years_ahead: float
) -> np.ndarray:
    """A county by age by sex population built from the experience file alone.

    The last published year's person-years are the state's stock at the middle of that
    year. They are split across the counties of the state in proportion to land area,
    which is the only county-level quantity a participant holds that does not come from a
    register, spread evenly over the single years inside each band, and then aged forward
    to the snapshot under the file's own survival and net migration. Every step is a
    shortcut, and each one is the shortcut a file with no microdata behind it forces.
    """
    n_counties = len(county_state)
    n_states = int(county_state.max()) + 1
    exposure = np.asarray(arrays["exposure"], dtype=np.float64)[-1]
    deaths = np.asarray(arrays["deaths"], dtype=np.float64)[-1]
    migration = np.asarray(arrays["net_migration"], dtype=np.float64)[-1]
    land = np.maximum(np.asarray(land, dtype=np.float64), 1.0)
    share = np.zeros(n_counties)
    for s in range(n_states):
        members = county_state == s
        share[members] = land[members] / max(land[members].sum(), 1e-9)
    cube = np.zeros((n_counties, AR.MAX_AGE + 1, 2))
    for b, (lo, hi) in enumerate(AR.ACTUARIAL_AGE_BANDS):
        ages = np.arange(lo, min(hi, AR.MAX_AGE) + 1)
        with np.errstate(invalid="ignore", divide="ignore"):
            rate = np.where(exposure[:, b, :] > 0,
                            (migration[:, b, :] - deaths[:, b, :]) /
                            np.maximum(exposure[:, b, :], 1e-9), 0.0)
        level = exposure[:, b, :] * (1.0 + np.clip(rate, -0.5, 0.5)) ** years_ahead
        per_age = level / len(ages)
        for age in ages:
            cube[:, age, :] = share[:, None] * per_age[county_state]
    shifted = np.zeros_like(cube)
    steps = int(round(years_ahead))
    if steps <= 0:
        return cube
    shifted[:, steps:, :] = cube[:, :-steps, :]
    shifted[:, -1, :] += cube[:, -1, :]
    return shifted


def experience_only_rows(cube: np.ndarray, county_state: np.ndarray,
                         relative_half: float) -> list[dict]:
    """The eight release estimands from that population, with zeros where the file is
    silent. Households, money and education have no source in an aggregate demographic
    file, and a control that says it needs no microdata files them as what it knows."""
    counties = np.arange(len(county_state))
    ages = np.arange(cube.shape[1])
    point = {}
    for estimand, mask in (("persons", np.ones(len(ages), dtype=bool)),
                           ("children_under_16", ages <= 15),
                           ("elders_65_plus", ages >= 65)):
        county_value = cube[:, mask, :].sum(axis=(1, 2))
        for c in counties:
            point[(estimand, "county", int(c))] = float(county_value[c])
        for s in range(int(county_state.max()) + 1):
            point[(estimand, "state", s)] = float(county_value[county_state == s].sum())
        point[(estimand, "nation", 0)] = float(county_value.sum())
    for estimand in ("households", "median_household_income", "mean_income_adults",
                     "low_income_household_share", "tertiary_share_25_plus"):
        for c in counties:
            point[(estimand, "county", int(c))] = 0.0
        for s in range(int(county_state.max()) + 1):
            point[(estimand, "state", s)] = 0.0
        point[(estimand, "nation", 0)] = 0.0
    return _rows_with_relative_half(point, relative_half)


def _experience_history_only(packet_dir: Path, out_dir: Path,
                             layer: "AR.LayerParams | None" = None) -> None:
    """Everything from ``experience_history.csv``, ``geography.csv`` and the contract."""
    import pandas as pd
    participant = Path(packet_dir) / "participant"
    contract = json.loads((participant / "contract.json").read_text())
    geography = pd.read_csv(participant / "geography.csv")
    county_state = geography["state"].to_numpy(dtype=np.int64)
    land = geography["land_cells"].to_numpy(dtype=np.float64) \
        if "land_cells" in geography.columns else np.ones(len(county_state))
    experience = AR.load_experience(packet_dir, contract)
    if experience is None:
        raise AR.MissingActuarialInputs(
            "experience_history_only needs a version-four packet with the experience file")
    n_states = int(county_state.max()) + 1
    arrays = AR.experience_arrays(experience, n_states)
    tick = int(contract["ticks"]["revised"])
    horizon_months = int(contract["ticks"]["horizon"]) - tick
    lag = float((contract.get("experience_history") or {}).get("publication_lag_months", 12))
    now = experience_only_cube(arrays, county_state, land, lag / 12.0 + 0.5)
    future = experience_only_cube(arrays, county_state, land,
                                  lag / 12.0 + 0.5 + horizon_months / 12.0)
    # Births are not in the file, so the birth rate is read out of the stock it left: the
    # person-years under eighteen spread over eighteen years, over the person-years of
    # women of childbearing age.
    exposure = arrays["exposure"].sum(axis=0)
    women = float(exposure[:, 1, 1].sum())
    children = float(exposure[:, 0, :].sum() / 18.0)
    fertility = children / max(women, 1e-9)
    release = experience_only_rows(now, county_state, 0.02)
    projection = experience_only_rows(future, county_state, 0.04)
    detail = np.zeros((len(county_state), len(AGE_BAND_LABELS), len(SEX_LABELS)))
    ages = np.arange(now.shape[1])
    for b, (lo, hi) in enumerate(((0, 15), (16, 24), (25, 44), (45, 64), (65, 200))):
        inside = (ages >= lo) & (ages <= min(hi, AR.MAX_AGE))
        detail[:, b, :] = now[:, inside, :].sum(axis=1)
    data = {"contract": contract, "county_state": county_state, "land_cells": land}
    layer = (
        replace(layer, experience_only=True)
        if layer is not None
        else AR.LayerParams(experience_only=True)
    )
    result = AR.actuarial_submission(
        Path(packet_dir),
        data,
        county_state,
        now[None],
        fertility,
        release,
        projection,
        detail,
        2.0 * int(contract.get("disclosure_threshold", 10)),
        Path(out_dir),
        AGE_BAND_LABELS,
        SEX_LABELS,
        layer,
    )
    if result is None:
        raise AR.MissingActuarialInputs(
            "experience_history_only needs the reserve block in the contract"
        )


def _actuarial_control(
    name: str,
    packet_dir: Path,
    out_dir: Path,
    calibration_path: str | None,
    simulation_paths: int | None,
) -> None:
    """Run the strong design-based line with exactly one step removed."""
    if name == "experience_history_only":
        simulation = AR.SimulationParams()
        if simulation_paths is not None:
            simulation = replace(simulation, n_paths=simulation_paths)
        layer = AR.LayerParams(simulation=simulation)
        _experience_history_only(Path(packet_dir), Path(out_dir), layer)
        return
    switches = dict(ACTUARIAL_SWITCHES[name])
    if simulation_paths is not None:
        simulation = switches.get("simulation", AR.SimulationParams())
        switches["simulation"] = replace(simulation, n_paths=simulation_paths)
    if name == "development_average_regime":
        calibration = (
            json.loads(Path(calibration_path).read_text()) if calibration_path else None
        )
        contract = json.loads(
            (Path(packet_dir) / "participant" / "contract.json").read_text()
        )
        switches["regime_override"] = development_regime_override(contract, calibration)
    layer = AR.LayerParams(**switches)
    if name != "version_three_recipe":
        A.run(
            packet_dir,
            out_dir,
            A.MethodParams(
                bootstrap_replicates=60,
                calibration_path=calibration_path,
                actuarial="on",
                actuarial_params=layer,
            ),
        )
        return
    data = load_packet(packet_dir)
    contract, county_state = data["contract"], data["county_state"]
    tick = int(contract["ticks"]["revised"])
    horizon_months = int(contract["ticks"]["horizon"]) - tick
    calibration = (
        json.loads(Path(calibration_path).read_text()) if calibration_path else {}
    )
    fit = calibration.get("version_three_recipe")
    if fit is None:
        raise ValueError(
            "version_three_recipe needs its development fit in calibration A"
        )
    register_frame = data["population"].drop_duplicates("person_id").copy()
    register_frame = register_frame[
        (register_frame["county"] >= 0)
        & (register_frame["county"] < len(county_state))
    ]
    register_frame["age"] = (
        tick - register_frame["birth_tick"].to_numpy(dtype=np.int64)
    ) // 12
    register = A.register_counts(register_frame, len(county_state), 0.0)
    release, projection = _version_three_release(
        data, tick, county_state, horizon_months, fit
    )
    age_sex = np.asarray(register["age_sex"], dtype=np.float64)
    submitted_persons = np.asarray(
        [
            next(
                r["estimate"]
                for r in release
                if r["estimand"] == "persons"
                and r["level"] == "county"
                and r["unit"] == c
            )
            for c in range(len(county_state))
        ],
        dtype=np.float64,
    )
    county_persons = age_sex.sum(axis=(1, 2))
    age_sex *= (submitted_persons / np.maximum(county_persons, 1.0))[:, None, None]
    result = AR.actuarial_submission(
        Path(packet_dir),
        data,
        county_state,
        age_sex[None],
        0.055,
        release,
        projection,
        np.asarray(register["cube"], dtype=np.float64),
        2.0 * int(contract.get("disclosure_threshold", 10)),
        Path(out_dir),
        AGE_BAND_LABELS,
        SEX_LABELS,
        layer,
    )
    if result is None:
        raise AR.MissingActuarialInputs(
            "version_three_recipe needs a version-four packet with the experience file"
        )


def _run_structural_base(
    packet_dir: Path,
    out_dir: Path,
    calibration_path: str | None,
    simulation_paths: int | None,
) -> dict:
    """Write the strong V4 rate and reserve blocks without public-total tail fitting."""
    simulation = AR.SimulationParams()
    if simulation_paths is not None:
        simulation = replace(simulation, n_paths=simulation_paths)
    return A.run(
        packet_dir,
        out_dir,
        A.MethodParams(
            bootstrap_replicates=60,
            calibration_path=calibration_path,
            actuarial="on",
            actuarial_params=AR.LayerParams(simulation=simulation),
        ),
    )


def _overlay_core_rows(
    out_dir: Path, release_rows: list[dict], projection_rows: list[dict]
) -> None:
    """Replace only core rows and retain the structural base's V4 rate block."""
    import pandas as pd

    out_dir = Path(out_dir)
    submitted_release = pd.read_csv(out_dir / "release.csv")
    submitted_projection = pd.read_csv(out_dir / "projection.csv")
    rate_rows = submitted_release[
        submitted_release["estimand"].isin(AR.RATE_ESTIMANDS)
    ]
    core = pd.DataFrame(release_rows)
    for column in submitted_release.columns:
        if column not in core:
            core[column] = "" if column in AR.RATE_EXTRA_COLUMNS else np.nan
    combined = pd.concat(
        [core[list(submitted_release.columns)], rate_rows], ignore_index=True
    )
    combined.to_csv(out_dir / "release.csv", index=False)

    projection = pd.DataFrame(projection_rows)
    for column in submitted_projection.columns:
        if column not in projection:
            projection[column] = ""
    projection[list(submitted_projection.columns)].to_csv(
        out_dir / "projection.csv", index=False
    )


def run(
    name: str,
    packet_dir: Path,
    out_dir: Path,
    calibration_path: str | None = None,
    simulation_paths: int | None = None,
) -> None:
    if name in ACTUARIAL_CONTROLS:
        _actuarial_control(
            name,
            Path(packet_dir),
            Path(out_dir),
            calibration_path,
            simulation_paths,
        )
        return
    if name not in CONTROLS:
        raise ValueError(f"unknown control {name!r}")
    data = load_packet(packet_dir)
    contract, county_state = data["contract"], data["county_state"]
    n_counties = len(county_state)
    tick = int(contract["ticks"]["revised"])
    horizon_months = int(contract["ticks"]["horizon"]) - tick
    out_dir = Path(out_dir)

    if name in ("inflated_intervals", "static_projection", "uniform_allocation", "exact_key_union"):
        base = _run_structural_base(
            Path(packet_dir), out_dir, calibration_path, simulation_paths
        )
        import pandas as pd
        if name == "exact_key_union":
            fit = None
            if calibration_path is not None:
                import json
                fit = json.loads(Path(calibration_path).read_text()).get("exact_key_union")
            if fit is None:
                raise ValueError("exact_key_union needs the development fit stored in calibration A")
            rows = [r for r in base["release"] if r["estimand"] not in COUNT_ITEMS]
            rows += _exact_key_union_rows(data, tick, county_state, fit)
            submitted = pd.read_csv(out_dir / "release.csv")
            frame = pd.DataFrame(rows)
            for column in submitted.columns:
                if column not in frame:
                    frame[column] = "" if column in AR.RATE_EXTRA_COLUMNS else np.nan
            frame[list(submitted.columns)].to_csv(out_dir / "release.csv", index=False)
        elif name == "inflated_intervals":
            rows = pd.read_csv(out_dir / "release.csv")
            core = ~rows["estimand"].isin(AR.RATE_ESTIMANDS)
            estimate = rows.loc[core, "estimate"].to_numpy(dtype=np.float64)
            proportion = rows.loc[core, "estimand"].isin(
                ("tertiary_share_25_plus", "low_income_household_share")
            ).to_numpy()
            half = np.where(proportion, 0.40, 0.40 * np.abs(estimate))
            rows.loc[core, "lower"] = np.maximum(
                estimate - half, 0.0
            )
            upper = estimate + half
            upper[proportion] = np.minimum(upper[proportion], 1.0)
            rows.loc[core, "upper"] = upper
            rows.to_csv(out_dir / "release.csv", index=False)
        elif name == "static_projection":
            release = pd.read_csv(out_dir / "release.csv")
            projection = pd.read_csv(out_dir / "projection.csv")
            core = release[~release["estimand"].isin(AR.RATE_ESTIMANDS)].copy()
            core[list(projection.columns)].to_csv(
                out_dir / "projection.csv", index=False
            )
        else:   # equal reserve slack above every submitted regional floor
            reserve_path = out_dir / "reserve.csv"
            if reserve_path.exists():
                rows = pd.read_csv(reserve_path)
                total = float(data["contract"]["reserve"]["total"])
                floor = rows["q95"].to_numpy(dtype=np.float64)
                slack = total - float(floor.sum())
                if slack < -1e-6:
                    # The legacy contract built R from sealed q95 and ES. Once the
                    # prohibited tail-to-total fit is removed, a participant forecast can
                    # therefore file floors whose sum exceeds R. There is no feasible
                    # uniform allocation in that state. Keep the complete filing and let
                    # the hard feasibility check report the contract obstruction. New
                    # exposure-rule packets must instead demonstrate non-negative slack.
                    allocation = floor
                else:
                    allocation = floor + max(slack, 0.0) / len(rows)
                    allocation[-1] = total - float(allocation[:-1].sum())
                rows["allocation"] = allocation
                rows.to_csv(reserve_path, index=False)
            else:
                raise RuntimeError(
                    "uniform_allocation requires the structural V4 reserve.csv"
                )
        return

    survey = A.impute_income(data["survey"].assign(weight=data["survey"]["design_weight"]))
    stats = A.survey_statistics(survey, county_state)
    if name == "no_dedup":
        frame = data["population"].copy()
        frame["age"] = (tick - frame["birth_tick"]) // 12
        frame = frame[frame["county"] >= 0]
        register = A.register_counts(frame, n_counties)
    else:
        register = A.register_counts(A.corroborate_counties(
            A.deduplicate_population(data["population"], tick, data["income"], data.get("health")), data["income"]), n_counties)
    ratios = A.income_source_ratios(data["income"], county_state, stats["median_household_income"]["nation"],
                                    A.register_income_scale(data["income"], stats["mean_income_adults"]["nation"]))

    if name == "survey_only":
        county = {}
        w = survey["weight"].to_numpy(dtype=np.float64)
        c = survey["county"].to_numpy(dtype=np.int64)
        age = survey["age"].to_numpy()
        hh = survey.groupby("household").agg(weight=("weight", "first"), county=("county", "first"))
        county = {
            "persons": np.bincount(c, weights=w, minlength=n_counties),
            "children_under_16": np.bincount(c, weights=w * (age <= 15), minlength=n_counties),
            "elders_65_plus": np.bincount(c, weights=w * (age >= 65), minlength=n_counties),
            "households": np.bincount(hh["county"].to_numpy(dtype=np.int64), weights=hh["weight"].to_numpy(), minlength=n_counties),
        }
        edu = survey["education"].fillna(-1).to_numpy()
        over = (age >= 25) & (edu >= 0)
        with np.errstate(invalid="ignore", divide="ignore"):
            county["tertiary_share_25_plus"] = np.bincount(c, weights=w * over * (edu >= 2), minlength=n_counties) / \
                np.bincount(c, weights=w * over, minlength=n_counties)
    elif name == "benchmark_only":
        county = {}
        benchmark = data.get("benchmark") or {}
        for e in COUNT_ITEMS:
            raw = np.asarray(register[e], dtype=np.float64)
            total = float(benchmark[e]["nation"]) if e in benchmark else float(raw.sum())
            county[e] = raw / max(raw.sum(), 1e-9) * total
        county["tertiary_share_25_plus"] = register["tertiary_share_25_plus"]
    else:  # register_only, no_dedup
        county = {e: np.asarray(register[e], dtype=np.float64) for e in COUNT_ITEMS}
        county["tertiary_share_25_plus"] = register["tertiary_share_25_plus"]
    n_states = int(county_state.max()) + 1
    for e in ("median_household_income", "mean_income_adults", "low_income_household_share"):
        state_values = np.asarray([stats[e][s] for s in range(n_states)])
        county[e] = np.clip(state_values[county_state] * ratios[e], 0.0, 1.0) if e.endswith("share") \
            else state_values[county_state] * ratios[e]
    now = A.aggregate(county, county_state, stats, county["persons"])
    future = A.aggregate(A.project(county, register["age_sex"], horizon_months, np.random.default_rng(1)),
                         county_state, stats, county["persons"])
    _run_structural_base(
        Path(packet_dir), out_dir, calibration_path, simulation_paths
    )
    _overlay_core_rows(
        out_dir,
        _rows_with_relative_half(now, 0.01),
        _rows_with_relative_half(future, 0.02),
    )
