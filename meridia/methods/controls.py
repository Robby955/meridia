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
- ``uniform_allocation``: method A with the budget spread equally over counties.
  Targets the allocation regret ceiling.
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
  development-world average instead of read from this world's experience file.
  Ablation 5. Targets the projection and the tails.
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
  every county, one global register income scale from a single national ratio, counts
  built as a national total times register county shares, archive rates with no
  selection correction, and a normal tail. Proof obligation 2. Targets the county count
  and rate gates, then the tail and reserve gates.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np

from ..release import AGE_BAND_LABELS, ESTIMAND_IDS, SEX_LABELS
from . import actuarial_reference as AR
from . import design_based as A
from .common import COUNT_ITEMS, load_packet, rows_from_draws, write_submission

CONTROLS = ("register_only", "survey_only", "no_dedup", "inflated_intervals",
            "static_projection", "uniform_allocation", "benchmark_only", "exact_key_union")
ACTUARIAL_CONTROLS = ("deterministic_linkage", "ignore_health_selection",
                      "development_average_regime", "mean_only_tail", "normal_tail",
                      "padded_tail", "proportional_reserve", "version_three_recipe",
                      "suppress_all_detail")
ALL_CONTROLS = CONTROLS + ACTUARIAL_CONTROLS

# One layer switch per version-four control. version_three_recipe also rebuilds the release
# and projection tables, so it is handled apart from this table.
ACTUARIAL_SWITCHES = {
    "deterministic_linkage": {"deterministic_linkage": True, "archive_only_rates": True},
    "ignore_health_selection": {"ignore_health_selection": True},
    "development_average_regime": {"regime_override": {"mortality_drift": 0.0,
                                                       "mortality_drift_se": 0.002,
                                                       "incidence_drift": 0.0,
                                                       "incidence_drift_se": 0.002}},
    "mean_only_tail": {"tail": "mean"},
    "normal_tail": {"tail": "normal"},
    "padded_tail": {"tail": "padded", "padding": 1.6},
    "proportional_reserve": {"allocation": "proportional"},
    "version_three_recipe": {"deterministic_linkage": True, "archive_only_rates": True,
                        "ignore_health_selection": True, "tail": "normal",
                        "allocation": "proportional"},
}

EXACT_KEY = ["given_code", "family_code", "birth_tick", "sex"]
UNION_WIDTH = {"nation": 2.5, "state": 2.2, "county": 2.0}   # multiples of the development spread
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


def _read_rows(path: Path) -> list[dict]:
    import pandas as pd
    return pd.read_csv(path).to_dict("records")


def _version_three_release(data: dict, tick: int, county_state: np.ndarray, register: dict,
                    stats: dict, horizon_months: int) -> tuple[list[dict], list[dict]]:
    """The version-three recipe: one national total split on register county shares, one
    global income scale, one growth factor for every county.

    Longitudinal matching is the within-source identifier, which version four no longer
    keeps across vintages, so the growth factor is read off a join that mostly fails; one
    income scale is a single national ratio, so a scale that varies by county and income
    band collapses to its average; and every county moves by the same factor, so the
    projection carries no structure at all. The counts add exactly by construction, which
    is the point: arithmetic additivity was free in version three and stays free here.
    """
    n_states = int(county_state.max()) + 1
    shares = {}
    for item in COUNT_ITEMS:
        raw = np.asarray(register[item], dtype=np.float64)
        shares[item] = np.maximum(raw, 1e-9) / max(raw.sum(), 1e-9)
    nation = {item: float(np.asarray(register[item], dtype=np.float64).sum())
              for item in COUNT_ITEMS}
    pre = data["population_preliminary"].drop_duplicates("person_id")
    rev = data["population"].drop_duplicates("person_id")
    matched = len(rev.merge(pre[["person_id"]], on="person_id", how="inner"))
    months = max(tick - int(data["contract"]["ticks"]["preliminary"]), 1)
    ratio = len(rev) / max(matched, 1) if matched else 1.0
    growth = float(np.clip(ratio ** (horizon_months / months), 0.5, 2.0))
    scale = A.register_income_scale(data["income"], stats["mean_income_adults"]["nation"])
    now, future = {}, {}
    for item in COUNT_ITEMS:
        now[item] = nation[item] * shares[item]
        future[item] = now[item] * growth
    for target in (now, future):
        target["tertiary_share_25_plus"] = register["tertiary_share_25_plus"]
        for e in ("median_household_income", "mean_income_adults",
                  "low_income_household_share"):
            level = np.asarray([stats[e][s] for s in range(n_states)])[county_state]
            target[e] = np.clip(level * scale, 0.0, 1.0) if e.endswith("share") \
                else level * scale
    release = _rows_with_relative_half(
        A.aggregate(now, county_state, stats, now["persons"]), 0.01)
    projection = _rows_with_relative_half(
        A.aggregate(future, county_state, stats, future["persons"]), 0.02)
    return release, projection


def _actuarial_control(name: str, packet_dir: Path, out_dir: Path,
                       calibration_path: str | None) -> None:
    """Run the strong design-based line with exactly one step removed."""
    if name == "suppress_all_detail":
        # A strong submission whose protected table publishes nothing. Disclosure
        # protection is one-sided and a blank table meets it; the utility floor is what
        # refuses one, and this control is what shows the floor firing.
        import pandas as pd
        A.run(packet_dir, out_dir,
              A.MethodParams(bootstrap_replicates=60, calibration_path=calibration_path,
                             actuarial="on"))
        detail = pd.read_csv(out_dir / "detailed.csv")
        detail["count"] = float("nan")
        detail.to_csv(out_dir / "detailed.csv", index=False)
        return
    layer = AR.LayerParams(**ACTUARIAL_SWITCHES[name])
    if name != "version_three_recipe":
        A.run(packet_dir, out_dir,
              A.MethodParams(bootstrap_replicates=60, calibration_path=calibration_path,
                             actuarial="on", actuarial_params=layer))
        return
    data = load_packet(packet_dir)
    contract, county_state = data["contract"], data["county_state"]
    tick = int(contract["ticks"]["revised"])
    horizon_months = int(contract["ticks"]["horizon"]) - tick
    register_frame = A.deduplicate_population(data["population"], tick)
    register = A.register_counts(register_frame, len(county_state), 0.0)
    survey = A.impute_income(data["survey"].assign(weight=data["survey"]["design_weight"]))
    stats = A.survey_statistics(survey, county_state)
    release, projection = _version_three_release(data, tick, county_state, register, stats,
                                          horizon_months)
    age_sex = np.asarray(register["age_sex"], dtype=np.float64)
    result = AR.actuarial_submission(
        Path(packet_dir), data, county_state, age_sex[None], 0.055, release, projection,
        np.asarray(register["cube"], dtype=np.float64),
        2.0 * int(contract["disclosure_threshold"]), Path(out_dir),
        AGE_BAND_LABELS, SEX_LABELS, layer)
    if result is None:
        raise AR.MissingActuarialInputs(
            "version_three_recipe needs a version-four packet with the experience file")


def run(name: str, packet_dir: Path, out_dir: Path, calibration_path: str | None = None) -> None:
    if name in ACTUARIAL_CONTROLS:
        _actuarial_control(name, Path(packet_dir), Path(out_dir), calibration_path)
        return
    if name not in CONTROLS:
        raise ValueError(f"unknown control {name!r}")
    data = load_packet(packet_dir)
    contract, county_state = data["contract"], data["county_state"]
    n_counties = len(county_state)
    tick = int(contract["ticks"]["revised"])
    horizon_months = int(contract["ticks"]["horizon"]) - tick
    budget = float(contract["allocation"]["budget"])
    threshold = int(contract["disclosure_threshold"])
    out_dir = Path(out_dir)

    if name in ("inflated_intervals", "static_projection", "uniform_allocation", "exact_key_union"):
        base = A.run(packet_dir, out_dir, A.MethodParams(bootstrap_replicates=60, calibration_path=calibration_path))
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
            pd.DataFrame(rows).to_csv(out_dir / "release.csv", index=False)
        elif name == "inflated_intervals":
            point = {(r["estimand"], r["level"], r["unit"]): r["estimate"] for r in base["release"]}
            pd.DataFrame(_rows_with_relative_half(point, 0.40)).to_csv(out_dir / "release.csv", index=False)
        elif name == "static_projection":
            pd.DataFrame(base["release"]).to_csv(out_dir / "projection.csv", index=False)
        else:   # uniform_allocation: a strong forecast with the reserve split evenly
            reserve_path = out_dir / "reserve.csv"
            if reserve_path.exists():
                rows = pd.read_csv(reserve_path)
                total = float(data["contract"]["reserve"]["total"])
                even = np.full(len(rows), total / len(rows))
                even[-1] = total - float(even[:-1].sum())
                rows["allocation"] = even
                rows.to_csv(reserve_path, index=False)
            else:
                pd.DataFrame({"county": np.arange(n_counties),
                              "allocation": np.full(n_counties, np.floor(budget / n_counties * 1e6) / 1e6)}
                             ).to_csv(out_dir / "allocation.csv", index=False)
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
        county = frame_counts = {}
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
    elders = np.maximum(A.project(county, register["age_sex"], horizon_months, np.random.default_rng(1))["elders_65_plus"], 0.0)
    allocation = np.floor(elders / max(elders.sum(), 1e-9) * budget * 1e6) / 1e6
    write_submission(out_dir, _rows_with_relative_half(now, 0.01), _rows_with_relative_half(future, 0.02),
                     register["cube"], 2.0 * threshold, allocation,
                     deterministic_reserve_rows(contract, county_state, elders))
