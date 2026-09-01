"""Controls for the forecast task: shortcuts that must each fail a named gate.

- ``constant_population``: the horizon equals the snapshot, tight intervals. Targets
  projection accuracy and coverage on counts.
- ``public_midpoint_rates``: ignore the ledger; project with the midpoints of the public
  parameter ranges. Targets accuracy on persons and children.
- ``no_age_structure``: scale every county count by one national exponential growth rate
  fitted to the history. Targets accuracy on elders and children.
- ``inflated_intervals``: method A's points with plus or minus 40 percent intervals.
  Targets the interval-score ceiling.
- ``uniform_allocation``: method A with the budget spread equally. Targets the regret
  ceiling.
- ``current_shares_allocation``: allocate by today's admission shares with no growth.
  Targets the regret ceiling where county growth differs.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..character import CHARACTER_RANGES
from ..demography import DemographyParams
from .common import COUNT_ITEMS
from . import forecast_cohort as FA

CONTROLS = ("constant_population", "public_midpoint_rates", "no_age_structure",
            "inflated_intervals", "uniform_allocation", "current_shares_allocation")


def _rows(point: dict, rel: float) -> list[dict]:
    rows = []
    for key in sorted(point):
        v = point[key]
        v = 0.0 if not np.isfinite(v) else v
        proportion = key[0].endswith("share") or key[0].startswith("tertiary")
        half = rel if proportion else rel * abs(v)
        lower, upper = max(v - half, 0.0), (min(v + half, 1.0) if proportion else v + half)
        v = min(max(v, lower), upper)
        rows.append({"estimand": key[0], "level": key[1], "unit": int(key[2]),
                     "estimate": float(v), "lower": float(lower), "upper": float(upper)})
    return rows


def run(name: str, packet_dir: Path, out_dir: Path) -> None:
    import pandas as pd
    if name not in CONTROLS:
        raise ValueError(f"unknown control {name!r}")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data = FA.load_forecast_packet(packet_dir)
    contract, county_state = data["contract"], data["county_state"]
    n_counties = len(county_state)
    S, H = int(contract["ticks"]["snapshot"]), int(contract["ticks"]["horizon"])
    years = int(round((H - S) / 12.0))
    budget = float(contract["allocation"]["budget"])
    base = FA.run(packet_dir, out_dir, FA.MethodParams(replicates=40))
    point = {(r["estimand"], r["level"], r["unit"]): r["estimate"] for r in base["projection"]}
    rates = FA.estimate_rates(data, S)

    if name == "inflated_intervals":
        pd.DataFrame(_rows(point, 0.40)).to_csv(out_dir / "projection.csv", index=False)
        return
    if name == "uniform_allocation":
        pd.DataFrame({"county": np.arange(n_counties),
                      "allocation": np.full(n_counties, np.floor(budget / n_counties * 1e6) / 1e6)}).to_csv(out_dir / "allocation.csv", index=False)
        return
    if name == "current_shares_allocation":
        shares = rates["admissions"] / max(rates["admissions"].sum(), 1e-9)
        pd.DataFrame({"county": np.arange(n_counties),
                      "allocation": np.floor(shares * budget * 1e6) / 1e6}).to_csv(out_dir / "allocation.csv", index=False)
        return

    cube0 = rates["cube"]
    persons0 = np.maximum(cube0.sum(axis=(1, 2)), 1.0)
    if name == "constant_population":
        end = cube0.copy()
    elif name == "public_midpoint_rates":
        mid = {k: 0.5 * (lo + hi) for k, (lo, hi) in CHARACTER_RANGES.items()}
        fake = dict(rates, gompertz_a=mid["gompertz_a"], gompertz_b=DemographyParams().gompertz_b,
                    fertility=mid["fertility_rate"])
        end, _ = FA.project_cohorts(cube0, fake, years, np.random.default_rng(1),
                                    FA.MethodParams(shock_probability_per_year=0.0), perturb=False)
    else:  # no_age_structure
        events = data["events"]
        births = int((events["event"] == "person_birth").sum())
        deaths = int((events["event"] == "person_death").sum())
        months = int(S - events["tick"].min()) or 1
        growth = (1.0 + (births - deaths) / max(persons0.sum(), 1.0) * 12.0 / months) ** years
        end = cube0 * growth
    persons_end = end.sum(axis=(1, 2))
    for e, v in (("persons", persons_end), ("children_under_16", end[:, :16, :].sum(axis=(1, 2))),
                 ("elders_65_plus", end[:, 65:, :].sum(axis=(1, 2)))):
        for c in range(n_counties):
            point[(e, "county", c)] = float(v[c])
    hh_now = np.asarray([point[("households", "county", c)] for c in range(n_counties)])
    for c in range(n_counties):
        point[("households", "county", c)] = float(hh_now[c] * persons_end[c] / max(base["projection"][0]["estimate"], 1.0)) if False else point[("households", "county", c)]
    n_states = int(county_state.max()) + 1
    for e in COUNT_ITEMS:
        cp = np.asarray([point[(e, "county", c)] for c in range(n_counties)])
        st = np.bincount(county_state, weights=cp, minlength=n_states)
        for s_ in range(n_states):
            point[(e, "state", s_)] = float(st[s_])
        point[(e, "nation", 0)] = float(st.sum())
    pd.DataFrame(_rows(point, 0.02)).to_csv(out_dir / "projection.csv", index=False)
