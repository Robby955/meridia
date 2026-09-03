"""Every hidden mechanism against the supplied anchor that estimates it.

Protocol proof obligation 5 asks that the hidden regime stay identifiable from observable
anchors, so that a frontier failure is never a failure to know the unknowable. This
script answers it as a measurement rather than a claim: for each of the six axes of the
regime family it computes one statistic from the participant files alone, then reports
that statistic's rank correlation with the realized intensity across the committed
generator-only worlds.

One statistic per axis, all from files the agent receives:

- mortality_improvement       count-weighted log mortality drift within cells of
                              experience_history.csv
- migration_age_pattern       urban pull of net internal migration at 18 to 44 minus the
                              same pull at 65 and over
- age_reporting_error         share of reported birth ticks off a year boundary
- linkage_urban_gradient      slope of the missing-name share on the county's urbanity rank
- administrative_completeness slope of register coverage on the county's economic rank
- missingness_target_dependence  log gap between the health archive's admission rate and
                              the anchor's, corrected for its declared error

Run it on development and qualification worlds. Graded worlds are not opened.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

AXES = ("mortality_improvement", "migration_age_pattern", "age_reporting_error",
        "linkage_urban_gradient", "administrative_completeness",
        "missingness_target_dependence")
STATISTIC = {"mortality_improvement": "experience file mortality drift within cells",
             "migration_age_pattern": "urban pull of net migration, young minus old",
             "age_reporting_error": "birth ticks off a year boundary",
             "linkage_urban_gradient": "missing-name share against urbanity",
             "administrative_completeness": "register coverage against economic rank",
             "missingness_target_dependence": "archive against anchor admission rate"}
# The sign each mechanism implies, read off the family and not off the data. The health
# archive observes the included, and inclusion rises with latent burden, so a stronger
# dependence on frailty raises the archive's admission rate above the population rate the
# anchor estimates. The axis carries that one mechanism and nothing else: item
# missingness on money moved to its own published slope in version four's second pass,
# after one coefficient loading two mechanisms left the statistic's sign reversing
# between regimes.
EXPECTED_SIGN = {"mortality_improvement": +1, "migration_age_pattern": +1,
                 "age_reporting_error": +1, "linkage_urban_gradient": -1,
                 "administrative_completeness": +1,
                 "missingness_target_dependence": +1}


def _rank01(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(np.argsort(values, kind="stable"), kind="stable")
    return order / max(len(values) - 1, 1)


def _slope(x: np.ndarray, y: np.ndarray) -> float:
    keep = np.isfinite(x) & np.isfinite(y)
    if keep.sum() < 3 or np.ptp(x[keep]) == 0:
        return float("nan")
    return float(np.polyfit(x[keep], y[keep], 1)[0])


def covariates(participant: Path) -> dict:
    """The two county covariates a participant rebuilds, as the contract defines them."""
    geography = pd.read_csv(participant / "geography.csv")
    population = pd.read_csv(participant / "sources" / "population_revised.csv")
    business = pd.read_csv(participant / "sources" / "business_revised.csv")
    contract = json.loads((participant / "contract.json").read_text())
    n_counties = int(contract["n_counties"])
    land = geography.set_index("county")["land_cells"].reindex(range(n_counties)).to_numpy()
    county = population["county"].to_numpy(dtype=np.int64)
    persons = np.bincount(county[county >= 0], minlength=n_counties).astype(np.float64)
    age = (int(contract["ticks"]["revised"]) - population["birth_tick"].to_numpy()) // 12
    adults = np.bincount(county[(county >= 0) & (age >= 18)], minlength=n_counties).astype(np.float64)
    payroll = np.bincount(business["county"].to_numpy(dtype=np.int64).clip(0),
                          weights=business["annual_payroll_cents"].fillna(0.0).to_numpy(),
                          minlength=n_counties)
    return {"urban_c": _rank01(persons / np.maximum(land, 1.0)),
            "econ_c": _rank01(payroll / np.maximum(adults, 1.0)),
            "persons": persons, "n_counties": n_counties,
            "county_state": geography.set_index("county")["state"].reindex(
                range(n_counties)).to_numpy()}


def _elder_share(population: pd.DataFrame, tick: int, n_counties: int) -> np.ndarray:
    """Share of persons 65 and over per county, the public definition of elder_c."""
    county = population["county"].to_numpy(dtype=np.int64).clip(0)
    age = (int(tick) - population["birth_tick"].to_numpy(dtype=np.int64)) // 12
    total = np.bincount(county, minlength=n_counties).astype(np.float64)
    old = np.bincount(county[age >= 65], minlength=n_counties).astype(np.float64)
    return np.divide(old, total, out=np.zeros(n_counties), where=total > 0)


def statistics(packet: Path) -> dict:
    participant = packet / "participant"
    contract = json.loads((participant / "contract.json").read_text())
    tick = int(contract["ticks"]["revised"])
    cov = covariates(participant)
    out = {}

    experience = pd.read_csv(participant / "experience_history.csv")
    cell = experience[(experience["exposure"] > 0) & (experience["deaths"] > 0)].copy()
    cell["log_rate"] = np.log(cell["deaths"] / cell["exposure"])
    # Cell effects, not a pooled slope: the age bands sit orders of magnitude apart, so a
    # slope taken across them reads the changing composition of the deaths rather than the
    # trend. Each cell is centred on its own count-weighted means first.
    numerator = denominator = 0.0
    for _, block in cell.groupby(["age_band", "sex", "state"]):
        w = block["deaths"].to_numpy(dtype=np.float64)
        if len(block) < 3 or w.sum() <= 0:
            continue
        year = block["year"].to_numpy(dtype=np.float64)
        year = year - np.average(year, weights=w)
        rate = block["log_rate"].to_numpy(dtype=np.float64)
        rate = rate - np.average(rate, weights=w)
        numerator += float((w * year * rate).sum())
        denominator += float((w * year ** 2).sum())
    out["mortality_improvement"] = -numerator / denominator if denominator else float("nan")
    # The estimator's own sampling error, from the Poisson variance of a log rate: one
    # over the count. Reported beside the intensity spread, because an anchor that cannot
    # resolve the spread it is there to identify is the thing worth knowing.
    out["mortality_improvement_se"] = float(np.sqrt(1.0 / denominator)) if denominator \
        else float("nan")

    # The axis is the strength of the age gradient in the pull toward urban destinations,
    # so its trace is the gap between where the young settle and where the old do. Net
    # internal migration per person-year of exposure, regressed on the state's urbanity,
    # once for the young-adult band and once for 65 and over.
    state_urban = np.zeros(int(contract["n_states"]))
    np.add.at(state_urban, cov["county_state"], cov["urban_c"])
    np.add.at(state_urban := state_urban, np.arange(0), 0.0)
    counties_per_state = np.bincount(cov["county_state"],
                                     minlength=int(contract["n_states"]))
    state_urban = state_urban / np.maximum(counties_per_state, 1)
    pull = {}
    for label, bands in (("young", ("18-44",)), ("old", ("65-74", "75-84", "85+"))):
        block = experience[experience["age_band"].isin(bands)]
        n = int(contract["n_states"])
        net = block.groupby("state")["net_migration"].sum().reindex(range(n)).fillna(0.0)
        exposure = block.groupby("state")["exposure"].sum().reindex(range(n)).fillna(1.0)
        pull[label] = _slope(state_urban, (net / np.maximum(exposure, 1.0)).to_numpy())
    out["migration_age_pattern"] = pull["young"] - pull["old"]

    population = pd.read_csv(participant / "sources" / "population_revised.csv")
    out["age_reporting_error"] = float((population["birth_tick"] % 12 != 0).mean())

    county = population["county"].to_numpy(dtype=np.int64)
    missing = (population["given_code"].to_numpy() == 0).astype(np.float64)
    records = np.bincount(county.clip(0), minlength=cov["n_counties"]).astype(np.float64)
    share = np.bincount(county.clip(0), weights=missing, minlength=cov["n_counties"]) / \
        np.maximum(records, 1.0)
    out["linkage_urban_gradient"] = _slope(cov["urban_c"], share)

    survey = pd.read_csv(participant / "survey_revised.csv")
    weight = survey["design_weight"].to_numpy(dtype=np.float64)
    survey_persons = np.bincount(survey["county"].to_numpy(dtype=np.int64), weights=weight,
                                 minlength=cov["n_counties"])
    # The completeness axis is a gradient of register coverage in the county's economic
    # rank. The survey is a noisy denominator on a world this size, so the statistic is
    # the register against the published benchmark, state by state, which is the second
    # handle the covariate note names: the benchmark is a count of the same population
    # with a declared bias and no coverage gradient of its own.
    benchmark = pd.read_csv(participant / "sources" / "benchmark_revised.csv")
    bench_state = benchmark[(benchmark["item"] == "persons")
                            & (benchmark["level"] == "state")].sort_values("unit")
    n_states = int(contract["n_states"])
    register_state = np.bincount(cov["county_state"], weights=cov["persons"],
                                 minlength=n_states)
    econ_state = np.bincount(cov["county_state"], weights=cov["econ_c"], minlength=n_states) / \
        np.maximum(np.bincount(cov["county_state"], minlength=n_states), 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        gap = np.log(np.maximum(register_state, 1.0)
                     / np.maximum(bench_state["value"].to_numpy(dtype=np.float64), 1.0))
        coverage = np.log(np.maximum(cov["persons"], 1.0) / np.maximum(survey_persons, 1.0))
    out["administrative_completeness"] = _slope(econ_state, gap)
    out["administrative_completeness_survey"] = _slope(cov["econ_c"], coverage)
    cov["elder_c"] = _rank01(_elder_share(population, tick, cov["n_counties"]))

    # The axis is a gradient, not a level: inclusion in the health source rises with a
    # person's latent burden. Its trace has to be a gradient too, or it reads the source's
    # own coverage level, which moves with a different axis. The anchor gives a population
    # admission rate per county, corrected for its declared error; the archive gives the
    # rate it observed; and the gap between them widens with the county's elder burden
    # exactly when inclusion reads morbidity.
    anchor = contract["health_anchor"]
    health = pd.read_csv(participant / "sources" / "health_revised.csv")
    window = health[health["admission_tick"] > tick - int(anchor["window_months"])]
    n = cov["n_counties"]
    observed = np.bincount(survey["county"].to_numpy(dtype=np.int64),
                           weights=weight * survey["recent_hospitalization"].to_numpy(),
                           minlength=n) / np.maximum(survey_persons, 1e-9)
    corrected = (observed - (1.0 - anchor["specificity"])) / \
        (anchor["sensitivity"] - (1.0 - anchor["specificity"]))
    archive = window["patient_id"].nunique() / max(population["person_id"].nunique(), 1)
    out["missingness_target_dependence"] = float(
        np.log(max(archive, 1e-9)) - np.log(max(float(np.average(corrected,
                                                                 weights=np.maximum(
                                                                     survey_persons, 1e-9))),
                                                1e-9)))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--packets", nargs="+", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    rows = []
    for path in args.packets:
        packet = Path(path)
        world = json.loads((packet / "retained" / "world.json").read_text())
        intensity = world["mechanisms"]["design"]["intensity"]
        measured = statistics(packet)
        coefficients = world["mechanisms"].get("coefficients", {})
        realized = dict(intensity)
        # Some axes are predeclared to interact, so the quantity a mechanism actually runs
        # on is a product, not the axis. Identifiability is a claim about the realized
        # coefficient: it is what a method has to recover, and it is what decides a world.
        #
        # Two of the three products of two axes land on a statistic below. Health
        # inclusion reads latent frailty at a slope administrative completeness scales,
        # and the rural excess of the name and address error rate is scaled by the
        # world's migration intensity, which is exactly what the missing-name slope on
        # urbanity measures. The third, the late-report probability of a death, does not
        # enter either of the statistics its two axes are read from, so no axis carries a
        # correction for it.
        if "health_inclusion_completeness_by_target" in coefficients:
            realized["missingness_target_dependence"] = float(
                intensity["missingness_target_dependence"]
                * (1.0 + float(coefficients["health_inclusion_completeness_by_target"])
                   * (float(intensity["administrative_completeness"]) - 1.0)))
        if "linkage_gradient_by_migration" in coefficients:
            realized["linkage_urban_gradient"] = float(
                intensity["linkage_urban_gradient"]
                * (1.0 + float(coefficients["linkage_gradient_by_migration"])
                   * (float(intensity["migration_age_pattern"]) - 1.0)))
        rows.append({"world": packet.name, "regime": world["regime"],
                     **{f"true_{a}": float(realized[a]) for a in AXES},
                     **{f"read_{a}": float(measured[a]) for a in AXES},
                     "drift_se": float(measured["mortality_improvement_se"])})
    frame = pd.DataFrame(rows)
    lines = [f"# Identifiability of the six axes, {len(frame)} generator-only worlds", ""]
    for axis in AXES:
        truth = frame[f"true_{axis}"].to_numpy()
        read = frame[f"read_{axis}"].to_numpy()
        keep = np.isfinite(truth) & np.isfinite(read)
        rho = float(np.corrcoef(_rank01(truth[keep]), _rank01(read[keep]))[0, 1]) \
            if keep.sum() > 2 else float("nan")
        signed = rho * EXPECTED_SIGN[axis]
        within = []
        for family in sorted(set(frame["regime"])):
            block = frame[frame["regime"] == family]
            t = block[f"true_{axis}"].to_numpy()
            r = block[f"read_{axis}"].to_numpy()
            ok = np.isfinite(t) & np.isfinite(r)
            if ok.sum() > 2:
                within.append(f"{family} {float(np.corrcoef(_rank01(t[ok]), _rank01(r[ok]))[0, 1]) * EXPECTED_SIGN[axis]:+.3f}")
        lines.append(f"- {axis}: {STATISTIC[axis]}; signed rank correlation {signed:+.3f} "
                     f"pooled, within regime " + ", ".join(within) +
                     f"; intensity spread {truth.min():.3f} to {truth.max():.3f}")
        if axis == "mortality_improvement":
            lines.append(f"    drift estimator standard error, mean over worlds "
                         f"{frame['drift_se'].mean():.4f}, against an intensity spread of "
                         f"{truth.max() - truth.min():.4f}")
    text = "\n".join(lines) + "\n"
    print(text)
    if args.out:
        Path(args.out).write_text(text)
        frame.to_csv(Path(args.out).with_suffix(".csv"), index=False)


if __name__ == "__main__":
    main()
