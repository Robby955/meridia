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
- administrative_completeness slope of the register's shortfall against the benchmark's
                              subgroup count, over the published economic bands
- missingness_target_dependence  old minus young contrast of the log gap between the
                              health archive's admission rate and the anchor's,
                              corrected for its declared error

Run it on development and hidden measurement worlds. Graded worlds are not opened.

``--record`` writes the reading as a file the tree keeps: the seed set it was taken on,
the world size, the six signed rank correlations, and a resampled interval for each. The
seeds are the point. A correlation over thirty worlds is a claim about thirty definite
worlds, every figure moves with which hidden worlds were drawn, and a set that lives only
in one run is a claim no reader can rebuild. ``scripts/build_v4_worlds.py --family
identifiability`` builds exactly the set the record names.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

# Draws in the resample, and the ends of the interval it reports. The seed is fixed so
# the record is a function of the worlds and not of the run.
RESAMPLE_DRAWS = 2000
RESAMPLE_ENDS = (5.0, 95.0)
RESAMPLE_SEED = 20260903
RECORD_SCHEMA = "meridia.identifiability.v1"


def builder():
    """The world-set builder, which is where the measurement's seeds are committed."""
    path = Path(__file__).resolve().parent / "build_v4_worlds.py"
    spec = importlib.util.spec_from_file_location("build_v4_worlds", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

AXES = ("mortality_improvement", "migration_age_pattern", "age_reporting_error",
        "linkage_urban_gradient", "administrative_completeness",
        "missingness_target_dependence")
STATISTIC = {"mortality_improvement": "experience file mortality drift within cells",
             "migration_age_pattern": "urban pull of net migration, young minus old",
             "age_reporting_error": "birth ticks off a year boundary",
             "linkage_urban_gradient": "missing-name share against urbanity",
             "administrative_completeness":
                 "register against the benchmark subgroup count, over economic bands",
             "missingness_target_dependence":
                 "archive against anchor admission rate, oldest minus youngest"}
BROAD_BAND_EDGES = (45, 65)
N_BROAD_BANDS = len(BROAD_BAND_EDGES) + 1
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


def _broad_band(age: np.ndarray) -> np.ndarray:
    """Under 45, 45 to 64, 65 and over: the groups the anchor can place a rate in."""
    return np.clip(np.searchsorted(np.asarray(BROAD_BAND_EDGES),
                                   np.asarray(age, dtype=np.int64), side="right"),
                   0, N_BROAD_BANDS - 1)


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
    # rank, and until the benchmark published a count on that gradient it had no anchor.
    # The register against the survey and the register against the state benchmark both
    # read the axis at about zero and reversed sign between regimes: the survey is a thin
    # denominator at this world size, the state series pools counties from both ends of
    # the gradient, and the covariate that reports the gradient is itself thinned by the
    # mechanism. The benchmark now publishes the resident person count of each economic
    # band of counties, and the band of every county sits in geography.csv, so the
    # register's shortfall band by band is the gradient with nothing in between.
    benchmark = pd.read_csv(participant / "sources" / "benchmark_revised.csv")
    geography = pd.read_csv(participant / "geography.csv")
    bands = benchmark[(benchmark["item"] == contract["benchmark"]["subgroup_item"])
                      & (benchmark["level"] == contract["benchmark"]["subgroup_level"])] \
        .sort_values("unit")
    n_bands = int(contract["benchmark"]["n_economic_bands"])
    band_of_county = geography.set_index("county")["economic_band"].reindex(
        range(cov["n_counties"])).to_numpy(dtype=np.int64)
    register_band = np.bincount(band_of_county, weights=cov["persons"], minlength=n_bands)
    with np.errstate(invalid="ignore", divide="ignore"):
        shortfall = np.log(np.maximum(register_band, 1.0)
                           / np.maximum(bands["value"].to_numpy(dtype=np.float64), 1.0))
    out["administrative_completeness"] = _slope(np.arange(float(n_bands)), shortfall)
    cov["elder_c"] = _rank01(_elder_share(population, tick, cov["n_counties"]))

    # The axis is a gradient, not a level: inclusion in the health source rises with a
    # person's latent burden. Its trace has to be a gradient too, or it reads how much of
    # the archive the source keeps at all, which is a different axis. The national gap
    # read the axis at -0.02 over twenty-one worlds for exactly that reason.
    #
    # Latent burden rises with age, so the anchor and the archive are compared on three
    # broad age groups: under 45, 45 to 64, and 65 and over. Both count the same event
    # over the same anchor window. The statistic is the contrast between the oldest group
    # and the youngest, averaged over the two snapshots, and broad groups are what the
    # anchor's own sampling error allows: at six bands the rate of the thinnest of them
    # is less certain than the whole gradient.
    anchor = contract["health_anchor"]
    contrasts = []
    for label in ("preliminary", "revised"):
        snapshot = int(contract["ticks"][label])
        register = pd.read_csv(participant / "sources" / f"population_{label}.csv")
        snapshot_survey = pd.read_csv(participant / f"survey_{label}.csv")
        archive = pd.read_csv(participant / "sources" / f"health_{label}.csv")
        window = archive[
            (archive["admission_tick"] > snapshot - int(anchor["window_months"]))
            & (archive["admission_tick"] <= snapshot)].drop_duplicates("patient_id")
        kept = np.bincount(
            _broad_band((snapshot - window["birth_tick"].to_numpy(dtype=np.int64)) // 12),
            minlength=N_BROAD_BANDS).astype(np.float64)
        persons = np.bincount(
            _broad_band((snapshot - register["birth_tick"].to_numpy(dtype=np.int64)) // 12),
            minlength=N_BROAD_BANDS).astype(np.float64)
        group = _broad_band(snapshot_survey["age"].to_numpy(dtype=np.int64))
        design = snapshot_survey["design_weight"].to_numpy(dtype=np.float64)
        responders = np.bincount(group, weights=design, minlength=N_BROAD_BANDS)
        reported = np.bincount(
            group, weights=design * snapshot_survey["recent_hospitalization"].to_numpy(),
            minlength=N_BROAD_BANDS)
        with np.errstate(invalid="ignore", divide="ignore"):
            observed = reported / np.maximum(responders, 1e-9)
            corrected = (observed - (1.0 - anchor["specificity"])) / \
                (anchor["sensitivity"] - (1.0 - anchor["specificity"]))
            archive_rate = kept / np.maximum(persons, 1.0)
            gap = np.log(np.where(archive_rate > 0, archive_rate, np.nan)) - \
                np.log(np.where(corrected > 0, corrected, np.nan))
        contrasts.append(gap[N_BROAD_BANDS - 1] - gap[0])
    out["missingness_target_dependence"] = float(np.nanmean(contrasts))
    return out


def signed_correlation(truth: np.ndarray, read: np.ndarray, axis: str) -> float:
    """Rank correlation of one statistic with the intensity it reads, in the sign the
    mechanism implies."""
    truth, read = np.asarray(truth, dtype=np.float64), np.asarray(read, dtype=np.float64)
    keep = np.isfinite(truth) & np.isfinite(read)
    if keep.sum() <= 2:
        return float("nan")
    rho = float(np.corrcoef(_rank01(truth[keep]), _rank01(read[keep]))[0, 1])
    return rho * EXPECTED_SIGN[axis]


def resampled_interval(truth: np.ndarray, read: np.ndarray, axis: str,
                       draws: int = RESAMPLE_DRAWS) -> tuple[float, float]:
    """Where the reading would have landed had the worlds been drawn again.

    The worlds are resampled with replacement and the correlation is retaken on each
    draw. This is the spread the reading carries as a reading of the family, and it is
    the quantity a decision about an axis has to clear, because a point estimate from
    thirty worlds says nothing about how far the next thirty would move it. A resample
    repeats worlds, so its ranks carry ties the original does not, and the interval it
    gives is a little wider than the family's own.
    """
    truth, read = np.asarray(truth, dtype=np.float64), np.asarray(read, dtype=np.float64)
    keep = np.isfinite(truth) & np.isfinite(read)
    truth, read = truth[keep], read[keep]
    if len(truth) <= 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(RESAMPLE_SEED)
    values = []
    for _ in range(int(draws)):
        index = rng.integers(0, len(truth), len(truth))
        if np.ptp(truth[index]) == 0 or np.ptp(read[index]) == 0:
            continue
        values.append(signed_correlation(truth[index], read[index], axis))
    low, high = np.percentile(np.asarray(values, dtype=np.float64), list(RESAMPLE_ENDS))
    return float(low), float(high)


def measurement_record(frame: pd.DataFrame) -> dict:
    """The reading as a file: the worlds it was taken on and what each axis read."""
    build = builder()
    world = build.WORLD
    axes = {}
    for axis in AXES:
        truth = frame[f"true_{axis}"].to_numpy()
        read = frame[f"read_{axis}"].to_numpy()
        hidden = frame["regime"] == "hidden"
        low, high = resampled_interval(truth, read, axis)
        hidden_low, hidden_high = resampled_interval(truth[hidden], read[hidden], axis)
        axes[axis] = {
            "statistic": STATISTIC[axis],
            "expected_sign": int(EXPECTED_SIGN[axis]),
            "pooled": round(signed_correlation(truth, read, axis), 3),
            "within_development": round(
                signed_correlation(truth[~hidden], read[~hidden], axis), 3),
            "within_hidden": round(signed_correlation(truth[hidden], read[hidden], axis), 3),
            "pooled_interval": [round(low, 3), round(high, 3)],
            "hidden_interval": [round(hidden_low, 3), round(hidden_high, 3)],
            "intensity_spread": [round(float(np.nanmin(truth)), 3),
                                 round(float(np.nanmax(truth)), 3)],
        }
    return {
        "schema": RECORD_SCHEMA,
        "worlds": {
            "development_seeds": list(build.DEVELOPMENT_SEEDS),
            "hidden_seeds": list(build.IDENTIFIABILITY_HIDDEN_SEEDS),
            "grid": list(world.grid),
            "total": int(world.total),
            "n_states": int(world.n_states),
            "observed_months": int(world.observed_months),
            "horizon_months": int(world.horizon_months),
            "ensemble_members": int(build.IDENTIFIABILITY_MEMBERS),
            "build": "scripts/build_v4_worlds.py --family identifiability",
        },
        "resample": {"draws": RESAMPLE_DRAWS, "ends": list(RESAMPLE_ENDS),
                     "seed": RESAMPLE_SEED, "unit": "world, with replacement"},
        "axes": axes,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--packets", nargs="+", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--record", default=None,
                    help="write the reading, and the seed set it was taken on, as JSON")
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
    if args.record:
        Path(args.record).write_text(
            json.dumps(measurement_record(frame), indent=1, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
