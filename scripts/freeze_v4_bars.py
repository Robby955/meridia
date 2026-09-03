"""Freeze the version-four bars on qualification worlds, then prove the battery fails.

    python scripts/freeze_v4_bars.py --dev DEV ... --qualification QUAL ... --out bars/national-v8

Steps, in this order and no other: calibrate both strong lines on the development
worlds; run both on every qualification world; set every bar from the worse of the two
witnesses with a margin declared in this file before any value was measured; check that
both witnesses clear the frozen bars on every qualification world; run the whole control
battery and record the named gate each control fails; and report the reserve decision's
attainable value, world by world.

The graded worlds are never opened here. Protocol section 12 freezes from generator-only
calibration worlds, and the external review adds that the graded world is minted
afterwards and never used to confirm or revise a bar.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from meridia.methods import bayesian, controls, design_based
from meridia.methods import actuarial_reference as AR
from meridia.actuarial import ensemble_truth as AR_ensemble_truth
from meridia.actuarial import quantile_score
from meridia.release import ESTIMAND_BY_ID
from meridia.verify import verify_submission

# ------------------------------------------------------------------ declared margins
# Fixed before the first qualification world was scored. A bar is the worse witness
# times its margin, never below its floor; a floor bar is the worse witness minus its
# slack, never below its own floor. Nothing here is a function of a measured value.
ACCURACY_MARGIN = 1.25
COVERAGE_SLACK = 0.10
SCORE_MARGIN = 1.5
RATE_MARGIN = 1.5            # exposure, mortality and incidence percentile errors
TAIL_MARGIN = 1.5            # exceedance deviations, quantile score, shortfall error
SHORTFALL_MARGIN = 1.5       # worst regional shortfall probability
SKILL_SLACK = 0.15           # skill minimum sits this far under the worse witness

# Floors. A bar taken only from the in-sample worst miss leaves no room where the two
# witnesses happened to be accurate, and a fresh world then fails on noise. Each floor
# states the irreducible between-world variation of its own quantity under the public
# mechanism ranges, and each is fixed before any world is seen.
RATE_FLOOR = {"person_years_exposure": 0.10,   # linkage and coverage churn on exposure
              "mortality_rate": 0.25,          # thin death counts in the oldest bands
              "qualifying_event_rate": 0.25}   # health inclusion is anchored, not known
RATE_COVERAGE_CAP = 0.60     # never demand more coverage than this, or than attained
TAU_MEAN_FLOOR = 0.020
TAU_WORST_FLOOR = 0.050
QUANTILE_SCORE_FLOOR = 0.050
ES_ERROR_FLOOR = 0.100
SHORTFALL_FLOOR = 0.150
CATASTROPHIC_FLOOR = 0.500
SKILL_CAP = 0.60             # never demand more skill than this, or than attained

# Declared attainability targets. A bar is a gate only inside the range its own criterion
# can take, and only where it is tight enough to refuse a method that ignores the
# mechanism it scores. Version four's first freeze clamped nothing and emitted three bars
# that could never fire: a tau_worst of 1.425 against a deviation whose maximum is 0.95, a
# regional shortfall ceiling of 1.5 against a probability, and a coverage floor of -0.06.
#
# Each entry below is the loosest value that still gates, fixed here before any world was
# scored. When the worse witness needs more room than its cap allows, the bar is written at
# the cap, the witness fails it, and the run reports NOT FROZEN for that gate. A cap is
# never widened to admit a witness.
CEILING_CAP = {
    "tau_mean": 0.150,                    # mean |p_r - 0.05|; mean-only tails sit near 0.45
    "tau_worst": 0.300,                   # the 0.90 quantile of the same deviation
    "es_error_ceiling": 0.100,            # |ES_hat - ES*| on the regional scale
    "regional_shortfall_ceiling": 0.350,  # a probability, so never above one
    "catastrophic_tail_ceiling": 0.500,
    "exposure_error_ceiling": 0.500,
    "mortality_error_ceiling": 1.000,
    "incidence_error_ceiling": 1.000,
}
# The quantile score has no natural scale, so its cap is a declared multiple of the score
# the sealed truth itself attains on the same ensemble: a submission may pay this much more
# than perfect information does, and no more.
QUANTILE_SCORE_ORACLE_MULTIPLE = 1.75
# Floor bars: a bar below its minimum is not a gate, it is a formality.
# The rate coverage bar is deliberately absent: the protocol's own review says the coverage
# gate is an empirical tolerance frozen on generator-only worlds, not a nominal level, so it
# takes the witnesses' own attainment minus the declared slack and is only clamped to the
# range a coverage can take.
FLOOR_MINIMUM = {
    "skill_minimum": 0.050,          # never accept an allocation worse than the public baseline
    "disclosure_utility_floor": 0.500,
}
DISCLOSURE_UTILITY_CAP = 0.90
DISCLOSURE_UTILITY_SLACK = 0.10

ACCURACY_FLOOR = {
    "nation": {"count": 0.05, "mean": 0.08, "median": 0.08, "proportion": 0.010},
    "state": {"count": 0.08, "mean": 0.12, "median": 0.12, "proportion": 0.020},
    "county": {"count": 0.15, "mean": 0.15, "median": 0.15, "proportion": 0.030},
}
ACCURACY_FLOOR["all"] = ACCURACY_FLOOR["county"]

RATE_KEY_ESTIMANDS = tuple(RATE_FLOOR)


def _floor_for(key: str) -> float:
    estimand, level = key.split("/")
    return ACCURACY_FLOOR[level][ESTIMAND_BY_ID[estimand].kind]


def _rate_bars(reports: dict) -> dict:
    """One ceiling per rate estimand, from the worse witness over every gated cell."""
    bars = {}
    for estimand in RATE_KEY_ESTIMANDS:
        worst = [m["percentile_error"] for r in reports.values()
                 for k, m in r["rate_metrics"].items()
                 if m["gated"] and k.split("/")[0] == estimand]
        value = max(worst) if worst else 0.0
        name = {"person_years_exposure": "exposure_error_ceiling",
                "mortality_rate": "mortality_error_ceiling",
                "qualifying_event_rate": "incidence_error_ceiling"}[estimand]
        bars[name] = round(max(RATE_MARGIN * value, RATE_FLOOR[estimand]), 6)
    coverage = [m["coverage"] for r in reports.values()
                for m in r["rate_metrics"].values() if m["gated"]]
    # A coverage bar is a floor on the submission, so its constant is a cap on the bar,
    # never a floor under it. A constant that sits above what both witnesses attain is
    # not a bar, it is a world neither of them can pass.
    bars["rate_coverage_floor"] = round(
        max(min(min(coverage) - COVERAGE_SLACK, RATE_COVERAGE_CAP), 0.0), 3) if coverage \
        else RATE_COVERAGE_CAP
    return bars


def _oracle_quantile_score(packets: list) -> float:
    """The quantile score perfect information itself pays, worst over the worlds.

    Read from the sealed ensemble alone, with no submission in it, which is the oracle
    distribution protocol section 12 says a bar is preregistered against.
    """
    worst = 0.0
    for packet in packets:
        ensemble = np.load(Path(packet) / "retained" / "continuation_liabilities.npz")
        liability = np.asarray(ensemble["liability"], dtype=np.float64)
        truth = AR_ensemble_truth(liability)
        scale = np.maximum(truth["q"], 1.0)
        worst = max(worst, float(quantile_score(truth["q"], liability, scale).mean()))
    return worst


def _tail_bars(reports: dict, oracle_quantile_score: float) -> dict:
    feasible = [r["reserve"] for r in reports.values() if r["reserve"].get("feasible")]
    if not feasible:
        raise SystemExit("no witness filed a feasible reserve; nothing can be frozen")
    pooled = max(float(r["calibration"]["pooled"]) for r in feasible)
    worst = max(float(r["calibration"]["worst"]) for r in feasible)
    score = max(float(r["mean_quantile_score"]) for r in feasible)
    es = max(float(r["mean_shortfall_error"]) for r in feasible)
    shortfall = max(float(np.max(r["regional_shortfall_probability"])) for r in feasible)
    tail = max(float(np.max(r["regional_tail"])) for r in feasible)
    skill = min(float(r["skill"]) for r in feasible if np.isfinite(r["skill"]))
    utility = min(float(r["disclosure"].get("utility", 1.0)) for r in reports.values())
    score_cap = QUANTILE_SCORE_ORACLE_MULTIPLE * max(oracle_quantile_score, 1e-12)
    bars = {
        "tau_mean": round(max(TAIL_MARGIN * pooled, TAU_MEAN_FLOOR), 4),
        "tau_worst": round(max(TAIL_MARGIN * worst, TAU_WORST_FLOOR), 4),
        "quantile_score_ceiling": round(
            min(max(TAIL_MARGIN * score, QUANTILE_SCORE_FLOOR), score_cap), 6),
        "es_error_ceiling": round(max(TAIL_MARGIN * es, ES_ERROR_FLOOR), 4),
        "regional_shortfall_ceiling": round(
            max(SHORTFALL_MARGIN * shortfall, SHORTFALL_FLOOR), 4),
        "catastrophic_tail_ceiling": round(max(TAIL_MARGIN * tail, CATASTROPHIC_FLOOR), 4),
        "skill_minimum": round(float(min(skill - SKILL_SLACK, SKILL_CAP)), 4),
    }
    bars["disclosure_utility_floor"] = round(
        float(min(utility - DISCLOSURE_UTILITY_SLACK, DISCLOSURE_UTILITY_CAP)), 3)
    return bars


def _clamp_to_attainability(bars: dict, oracle_quantile_score: float) -> list[str]:
    """Hold every bar inside its declared range and say which ones the witnesses miss."""
    notes: list[str] = []
    caps = dict(CEILING_CAP)
    caps["quantile_score_ceiling"] = round(
        QUANTILE_SCORE_ORACLE_MULTIPLE * oracle_quantile_score, 6)
    for name, cap in caps.items():
        value = bars["actuarial"].get(name)
        if value is None:
            continue
        if value > cap:
            bars["actuarial"][name] = cap
            notes.append(f"{name}: the worse witness needed {value:.4f}, the declared "
                         f"attainability cap is {cap:.4f}; written at the cap, so the "
                         "witness does not clear it")
    for name, minimum in FLOOR_MINIMUM.items():
        holder = bars if name == "disclosure_utility_floor" else bars["actuarial"]
        value = holder.get(name)
        if value is None:
            continue
        if value < minimum:
            holder[name] = minimum
            notes.append(f"{name}: the worse witness reached only {value:.4f}, the declared "
                         f"minimum for a gate is {minimum:.4f}; written at the minimum, so "
                         "the witness does not clear it")
    coverage = bars["actuarial"].get("rate_coverage_floor")
    if coverage is not None and coverage <= 0.0:
        notes.append("rate_coverage_floor: the witnesses' own interval coverage on the "
                     "gated rate cells minus the declared slack is at or below zero, so "
                     "this bar cannot refuse anything. The protocol makes the coverage "
                     "gate an empirical tolerance rather than a nominal level, so it is "
                     "recorded at zero and reported as not gating rather than invented")
    return notes


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev", nargs="+", required=True)
    ap.add_argument("--qualification", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--bootstrap", type=int, default=100)
    ap.add_argument("--sweeps", type=int, default=400)
    ap.add_argument("--controls", default="all")
    args = ap.parse_args()
    out = Path(args.out)
    for packet in args.qualification:
        world = json.loads((Path(packet) / "retained" / "world.json").read_text())
        if world.get("regime") != "hidden":
            raise SystemExit(f"{packet} is not a hidden-regime qualification world")
    for packet in args.dev:
        world = json.loads((Path(packet) / "retained" / "world.json").read_text())
        if world.get("regime") != "development":
            raise SystemExit(f"{packet} is not a development world")
    out.mkdir(parents=True, exist_ok=True)

    cal_a, cal_b = out / "calibration_A.json", out / "calibration_B.json"
    design_based.calibrate(args.dev, cal_a)
    bayesian.calibrate(args.dev, cal_b)
    layer = AR.LayerParams()
    witnesses = {
        "A": lambda p, o: design_based.run(p, o, design_based.MethodParams(
            bootstrap_replicates=args.bootstrap, calibration_path=str(cal_a),
            actuarial="on", actuarial_params=layer)),
        "B": lambda p, o: bayesian.run(p, o, bayesian.MethodParams(
            sweeps=args.sweeps, burn_in=args.sweeps // 4, calibration_path=str(cal_b),
            actuarial="on", actuarial_params=layer)),
    }
    reports = {}
    for name, run in witnesses.items():
        for packet in args.qualification:
            packet = Path(packet)
            sub = out / "runs" / packet.name / name
            if not sub.exists():
                run(packet, sub)
            reports[(name, packet.name)] = verify_submission(packet, sub)
            print(f"witness {name} on {packet.name}: scored", flush=True)

    bars = {"worst_error": {}, "interval_score_ceiling": {},
            "projection": {"worst_error": {}, "interval_score_ceiling": {}}}
    coverages, pcoverages = [], []
    for key in sorted({k for r in reports.values() for k in r["metrics"]}):
        worst = max(r["metrics"][key]["worst_error"] for r in reports.values())
        score = max(r["metrics"][key]["mean_interval_score"] for r in reports.values())
        bars["worst_error"][key] = round(max(ACCURACY_MARGIN * worst, _floor_for(key)), 6)
        bars["interval_score_ceiling"][key] = round(
            max(SCORE_MARGIN * score, 3.0 * _floor_for(key)), 6)
        if key.endswith("/all"):
            coverages += [r["metrics"][key]["coverage"] for r in reports.values()]
    for key in sorted({k for r in reports.values() for k in r["projection_metrics"]}):
        worst = max(r["projection_metrics"][key]["worst_error"] for r in reports.values())
        score = max(r["projection_metrics"][key]["mean_interval_score"]
                    for r in reports.values())
        bars["projection"]["worst_error"][key] = round(
            max(ACCURACY_MARGIN * worst, 1.5 * _floor_for(key)), 6)
        bars["projection"]["interval_score_ceiling"][key] = round(
            max(SCORE_MARGIN * score, 4.5 * _floor_for(key)), 6)
        if key.endswith("/all"):
            pcoverages += [r["projection_metrics"][key]["coverage"] for r in reports.values()]
    bars["coverage_floor"] = round(min(min(coverages) - COVERAGE_SLACK, 0.70), 3)
    bars["projection"]["coverage_floor"] = round(
        min(min(pcoverages) - COVERAGE_SLACK, 0.70), 3)
    oracle_quantile_score = _oracle_quantile_score(args.qualification)
    tail = _tail_bars(reports, oracle_quantile_score)
    bars["disclosure_utility_floor"] = tail.pop("disclosure_utility_floor")
    bars["actuarial"] = _rate_bars(reports) | tail
    attainability = _clamp_to_attainability(bars, oracle_quantile_score)
    bars["frozen_from"] = {
        "dev": [Path(p).name for p in args.dev],
        "qualification": [Path(p).name for p in args.qualification],
        "witnesses": sorted(witnesses),
        "margins": {"accuracy": ACCURACY_MARGIN, "coverage_slack": COVERAGE_SLACK,
                    "score": SCORE_MARGIN, "rate": RATE_MARGIN, "tail": TAIL_MARGIN,
                    "shortfall": SHORTFALL_MARGIN, "skill_slack": SKILL_SLACK},
        "floors": {"accuracy": ACCURACY_FLOOR, "rate": RATE_FLOOR,
                   "rate_coverage_cap": RATE_COVERAGE_CAP, "tau_mean": TAU_MEAN_FLOOR,
                   "tau_worst": TAU_WORST_FLOOR, "quantile_score": QUANTILE_SCORE_FLOOR,
                   "es_error": ES_ERROR_FLOOR, "shortfall": SHORTFALL_FLOOR,
                   "catastrophic": CATASTROPHIC_FLOOR,
                   "skill_cap": SKILL_CAP},
        "attainability_caps": CEILING_CAP,
        "attainability_floors": FLOOR_MINIMUM,
        "oracle_quantile_score": round(oracle_quantile_score, 6),
    }

    lines = ["# Version-four bar freeze report", ""]
    ok = True
    if attainability:
        lines.append("## Bars written at their declared attainability limit")
        lines.append("")
        lines += [f"- {note}" for note in attainability]
        lines.append("")
    lines.append("## Witnesses under the frozen bars")
    for (name, world), _ in sorted(reports.items()):
        packet = Path(next(p for p in args.qualification if Path(p).name == world))
        gated = verify_submission(packet, out / "runs" / world / name, bars,
                                  allow_unfrozen=True)
        verdict = "PASS" if gated["pass"] else "FAIL " + "; ".join(gated["reasons"])
        lines.append(f"- witness {name} on {world}: {verdict}")
        ok &= gated["pass"]

    lines.append("")
    lines.append("## Reserve decision value (proof obligation 4)")
    for (name, world), report in sorted(reports.items()):
        reserve = report["reserve"]
        if not reserve.get("feasible"):
            lines.append(f"- {name} on {world}: infeasible reserve")
            continue
        gain = (reserve["J_baseline"] - reserve["J_oracle"]) / max(reserve["J_baseline"], 1e-12)
        lines.append(f"- {name} on {world}: J(A_B) {reserve['J_baseline']:,.1f}  "
                     f"J(A*) {reserve['J_oracle']:,.1f}  oracle gain {gain:.2%}  "
                     f"submitted skill {reserve['skill']:.4f}")

    lines.append("")
    lines.append("## Control battery")
    battery = controls.ALL_CONTROLS if args.controls == "all" else tuple(args.controls.split(","))
    for control in battery:
        for packet in args.qualification:
            packet = Path(packet)
            sub = out / "runs" / packet.name / f"control_{control}"
            if not sub.exists():
                controls.run(control, packet, sub, calibration_path=str(cal_a))
            gated = verify_submission(packet, sub, bars, allow_unfrozen=True)
            if gated["pass"]:
                status = "PASSES (bar too loose)"
            else:
                families = sorted({r.split(":")[0] for r in gated["reasons"]})
                status = ("FAILS " + ", ".join(families) + "; "
                          + "; ".join(gated["reasons"][:2]))
            lines.append(f"- {control} on {packet.name}: {status}")
            ok &= not gated["pass"]
            print(f"control {control} on {packet.name}: {'PASS' if gated['pass'] else 'fails'}",
                  flush=True)
    lines.append("")
    lines.append("RESULT: " + ("bars frozen; every control fails a named gate"
                               if ok else "NOT FROZEN"))
    # A bar file says whether its own run reached a verdict. verify_submission refuses to
    # gate on a set that did not, so an unfinished freeze cannot be read later as a frozen
    # one. Version four's first freeze wrote its bars before the control battery ran and
    # ended NOT FROZEN, and the file it left behind carried no trace of that.
    bars["frozen"] = bool(ok)
    (out / "bars.json").write_text(json.dumps(bars, indent=1, sort_keys=True) + "\n")
    (out / "freeze_report.txt").write_text("\n".join(lines) + "\n")
    (out / "PROVENANCE.md").write_text(_provenance(args, bars, witnesses))
    print("\n".join(lines))
    return 0 if ok else 1


def _provenance(args, bars: dict, witnesses: dict) -> str:
    """Where every number in bars.json came from, in the order the freeze produced it."""
    frozen = bars["frozen_from"]
    return "\n".join([
        "# Provenance of the version-four bars", "",
        "Every threshold here was set by `scripts/freeze_v4_bars.py` from the worse of",
        "two methodologically different witnesses on the qualification worlds named",
        "below, times a margin declared in that script before any world was scored. No",
        "graded world was opened, and no submission was looked at.", "",
        "## Worlds", "",
        "- development, used only to calibrate the two witnesses: "
        + ", ".join(frozen["dev"]),
        "- qualification, hidden source regime, the only worlds any bar reads: "
        + ", ".join(frozen["qualification"]),
        "- graded: not opened by this script.", "",
        "## Witnesses", "",
        f"- A, the design-based line, {args.bootstrap} bootstrap replicates, "
        "with the shared actuarial layer at its reference settings.",
        f"- B, the Bayesian line, {args.sweeps} sweeps with a quarter burned in, "
        "through the same actuarial layer on its own posterior draws.",
        "",
        "The two share the actuarial chain and differ in their reconstruction, which is",
        "recorded in docs/INDEPENDENCE.md. A bar set from the worse of two lines that",
        "agree on the chain is a weaker guarantee than one set from two independent",
        "chains, and is read as such.", "",
        "## Rule", "",
        "- a ceiling is the worst witness value times its margin, never under its own",
        "  floor, which only ever loosens it;",
        "- a coverage or skill bar is the worst witness value minus its slack, and its",
        "  constant is a cap on the bar rather than a floor under it: a constant above",
        "  what both witnesses attain would not be a bar but a world neither can pass;",
        "- margins: " + json.dumps(frozen["margins"], sort_keys=True) + ";",
        "- floors: " + json.dumps(frozen["floors"], sort_keys=True) + ";",
        "- a bar is then held inside the range its own criterion can take, at the",
        "  declared attainability caps "
        + json.dumps(frozen["attainability_caps"], sort_keys=True) + " and the declared",
        "  floor minima " + json.dumps(frozen["attainability_floors"], sort_keys=True)
        + ". A bar that would be written outside that range is written at the limit",
        "  and the witness then fails it, rather than inheriting a number that can never",
        "  fire. The quantile score cap is "
        + str(QUANTILE_SCORE_ORACLE_MULTIPLE) + " times the score the sealed truth itself",
        "  pays on the same ensemble, which is "
        + str(frozen["oracle_quantile_score"]) + " here.", "",
        "## Frozen actuarial bars", "",
        *[f"- {name}: {value}" for name, value in sorted(bars["actuarial"].items())],
        "",
        "## Whether this set froze", "",
        f"`bars.json` records `frozen`: {json.dumps(bars.get('frozen'))}. It is written",
        "after the control battery, not before it, and `verify_submission` refuses to",
        "gate a version-four submission with a set that does not record a completed",
        "freeze. A freeze completes when both witnesses clear every bar on every",
        "qualification world and every control fails at least one named gate. The freeze",
        "report names what blocked it when it does not.", "",
        "## Reading them", "",
        "`verify_submission(packet, submission, bars)` reads `bars[\"actuarial\"]` for the",
        "version-four gates and falls back to the placeholders in `ActuarialThresholds`",
        "for any key that is absent. A metric with no threshold is reported and never",
        "gates.", ""])


if __name__ == "__main__":
    raise SystemExit(main())
