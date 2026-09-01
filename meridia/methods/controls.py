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
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np

from ..release import ESTIMAND_IDS
from . import design_based as A
from .common import COUNT_ITEMS, load_packet, rows_from_draws, write_submission

CONTROLS = ("register_only", "survey_only", "no_dedup", "inflated_intervals",
            "static_projection", "uniform_allocation")


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


def _read_rows(path: Path) -> list[dict]:
    import pandas as pd
    return pd.read_csv(path).to_dict("records")


def run(name: str, packet_dir: Path, out_dir: Path, calibration_path: str | None = None) -> None:
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

    if name in ("inflated_intervals", "static_projection", "uniform_allocation"):
        base = A.run(packet_dir, out_dir, A.MethodParams(bootstrap_replicates=60, calibration_path=calibration_path))
        import pandas as pd
        if name == "inflated_intervals":
            point = {(r["estimand"], r["level"], r["unit"]): r["estimate"] for r in base["release"]}
            pd.DataFrame(_rows_with_relative_half(point, 0.40)).to_csv(out_dir / "release.csv", index=False)
        elif name == "static_projection":
            pd.DataFrame(base["release"]).to_csv(out_dir / "projection.csv", index=False)
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
        register = A.register_counts(A.deduplicate_population(data["population"], tick, data["income"], data.get("health")), n_counties)
    ratios = A.income_source_ratios(data["income"], county_state, stats["median_household_income"]["nation"])

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
                     register["cube"], 2.0 * threshold, allocation)
