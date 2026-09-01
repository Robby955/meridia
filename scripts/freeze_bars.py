"""Freeze the pass bars from the two strong methods, then prove every control fails.

    python scripts/freeze_bars.py --dev DEV_PACKET [DEV_PACKET ...] \\
        --hidden HIDDEN_PACKET [HIDDEN_PACKET ...] --out bars/

Steps: calibrate both strong methods on the development packets; run both on every
hidden packet; set each bar from the worst strong-method result with a margin; check
that both strong methods pass the frozen bars on every hidden packet; run the control
battery on every hidden packet and record which named gate each control fails. Writes
``bars.json`` and ``freeze_report.txt``. Exits nonzero if a strong method fails its own
bars or a control passes.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from meridia.methods import bayesian, controls, design_based
from meridia.verify import verify_submission

ACCURACY_MARGIN = 1.25       # bar = 1.25 x worst strong-method error
COVERAGE_SLACK = 0.10        # floor = min strong coverage - 0.10, never below 0.70
SCORE_MARGIN = 1.5           # ceiling = 1.5 x worst strong interval score
REGRET_MARGIN = 2.0          # ceiling = 2 x worst strong regret, at least 0.02


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev", nargs="+", required=True)
    ap.add_argument("--hidden", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--bootstrap", type=int, default=100)
    ap.add_argument("--sweeps", type=int, default=400)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    cal_a, cal_b = out / "calibration_A.json", out / "calibration_B.json"
    design_based.calibrate(args.dev, cal_a)
    bayesian.calibrate(args.dev, cal_b)
    strong = {
        "A": lambda p, o: design_based.run(p, o, design_based.MethodParams(
            bootstrap_replicates=args.bootstrap, calibration_path=str(cal_a))),
        "B": lambda p, o: bayesian.run(p, o, bayesian.MethodParams(
            sweeps=args.sweeps, burn_in=args.sweeps // 4, calibration_path=str(cal_b))),
    }
    reports = {}
    for name, fn in strong.items():
        for packet in args.hidden:
            packet = Path(packet)
            sub = out / "runs" / packet.name / name
            fn(packet, sub)
            reports[(name, packet.name)] = verify_submission(packet, sub)

    bars = {"worst_error": {}, "interval_score_ceiling": {}, "projection": {"worst_error": {}, "interval_score_ceiling": {}}}
    keys = sorted({k for r in reports.values() for k in r["metrics"]})
    coverages, pcoverages, regrets = [], [], []
    for key in keys:
        worst = max(r["metrics"][key]["worst_error"] for r in reports.values() if key in r["metrics"])
        score = max(r["metrics"][key]["mean_interval_score"] for r in reports.values() if key in r["metrics"])
        bars["worst_error"][key] = round(ACCURACY_MARGIN * worst, 6)
        bars["interval_score_ceiling"][key] = round(SCORE_MARGIN * score, 6)
        if key.endswith("/all"):
            coverages += [r["metrics"][key]["coverage"] for r in reports.values() if key in r["metrics"]]
    for key in sorted({k for r in reports.values() for k in r["projection_metrics"]}):
        worst = max(r["projection_metrics"][key]["worst_error"] for r in reports.values() if key in r["projection_metrics"])
        score = max(r["projection_metrics"][key]["mean_interval_score"] for r in reports.values() if key in r["projection_metrics"])
        bars["projection"]["worst_error"][key] = round(ACCURACY_MARGIN * worst, 6)
        bars["projection"]["interval_score_ceiling"][key] = round(SCORE_MARGIN * score, 6)
        if key.endswith("/all"):
            pcoverages += [r["projection_metrics"][key]["coverage"] for r in reports.values() if key in r["projection_metrics"]]
    regrets = [r["allocation"]["regret"] for r in reports.values() if r["allocation"]["feasible"]]
    bars["coverage_floor"] = round(max(min(coverages) - COVERAGE_SLACK, 0.70), 3)
    bars["projection"]["coverage_floor"] = round(max(min(pcoverages) - COVERAGE_SLACK, 0.70), 3)
    bars["allocation_regret_ceiling"] = round(max(REGRET_MARGIN * max(regrets), 0.02), 6)
    bars["frozen_from"] = {"dev": [Path(p).name for p in args.dev], "hidden": [Path(p).name for p in args.hidden],
                           "margins": {"accuracy": ACCURACY_MARGIN, "coverage_slack": COVERAGE_SLACK,
                                       "score": SCORE_MARGIN, "regret": REGRET_MARGIN}}
    (out / "bars.json").write_text(json.dumps(bars, indent=1, sort_keys=True) + "\n")

    lines = ["# Bar freeze report", ""]
    ok = True
    for (name, packet), report in sorted(reports.items()):
        gated = verify_submission(Path(next(p for p in args.hidden if Path(p).name == packet)),
                                  out / "runs" / packet / name, bars)
        lines.append(f"- strong {name} on {packet}: {'PASS' if gated['pass'] else 'FAIL ' + '; '.join(gated['reasons'])}")
        ok &= gated["pass"]
    for control in controls.CONTROLS:
        for packet in args.hidden:
            packet = Path(packet)
            sub = out / "runs" / packet.name / f"control_{control}"
            controls.run(control, packet, sub, calibration_path=str(cal_a))
            gated = verify_submission(packet, sub, bars)
            families = sorted({r.split(":")[0] for r in gated["reasons"]})
            status = "FAILS " + ", ".join(families) + "; e.g. " + "; ".join(gated["reasons"][:2]) if not gated["pass"] else "PASSES (bar too loose)"
            lines.append(f"- control {control} on {packet.name}: {status}")
            ok &= not gated["pass"]
    lines.append("")
    lines.append("RESULT: " + ("bars frozen; every control fails a named gate" if ok else "NOT FROZEN"))
    (out / "freeze_report.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
