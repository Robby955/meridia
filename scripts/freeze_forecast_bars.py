"""Freeze the forecast task's bars from its two strong methods; prove the controls fail.

    python scripts/freeze_forecast_bars.py --hidden PACKET [PACKET ...] --out bars/

No calibration step: the forecast task ships a clean history, so nothing is tuned on
development worlds. Bars: 1.25 times the worse strong worst-unit error; pooled coverage
floor the worse strong coverage minus 0.10, never below 0.70; interval-score ceiling 1.5
times; regret ceiling 2 times, at least 0.02.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from meridia.forecast import verify_forecast
from meridia.methods import forecast_bayes, forecast_cohort, forecast_controls


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hidden", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--replicates", type=int, default=150)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    strong = {"A": lambda p, o: forecast_cohort.run(p, o, forecast_cohort.MethodParams(replicates=args.replicates)),
              "B": lambda p, o: forecast_bayes.run(p, o, forecast_bayes.MethodParams(draws=args.replicates))}
    reports = {}
    for name, fn in strong.items():
        for packet in args.hidden:
            packet = Path(packet)
            sub = out / "runs" / packet.name / name
            fn(packet, sub)
            reports[(name, packet.name)] = verify_forecast(packet, sub)
    bars = {"worst_error": {}, "interval_score_ceiling": {}}
    coverages, regrets = [], []
    for key in sorted({k for r in reports.values() for k in r["metrics"]}):
        worst = max(r["metrics"][key]["worst_error"] for r in reports.values() if key in r["metrics"])
        score = max(r["metrics"][key]["mean_interval_score"] for r in reports.values() if key in r["metrics"])
        bars["worst_error"][key] = round(1.25 * worst, 6)
        bars["interval_score_ceiling"][key] = round(1.5 * score, 6)
        if key.endswith("/all"):
            coverages += [r["metrics"][key]["coverage"] for r in reports.values() if key in r["metrics"]]
    regrets = [r["allocation"]["regret"] for r in reports.values() if r["allocation"]["feasible"]]
    bars["coverage_floor"] = round(max(min(coverages) - 0.10, 0.70), 3)
    bars["allocation_regret_ceiling"] = round(max(2.0 * max(regrets), 0.02), 6)
    bars["frozen_from"] = {"hidden": [Path(p).name for p in args.hidden]}
    (out / "bars.json").write_text(json.dumps(bars, indent=1, sort_keys=True) + "\n")
    lines, ok = ["# Forecast bar freeze report", ""], True
    for (name, packet), _ in sorted(reports.items()):
        gated = verify_forecast(Path(next(p for p in args.hidden if Path(p).name == packet)), out / "runs" / packet / name, bars)
        lines.append(f"- strong {name} on {packet}: {'PASS' if gated['pass'] else 'FAIL ' + '; '.join(gated['reasons'])}")
        ok &= gated["pass"]
    for control in forecast_controls.CONTROLS:
        for packet in args.hidden:
            packet = Path(packet)
            sub = out / "runs" / packet.name / f"control_{control}"
            forecast_controls.run(control, packet, sub)
            gated = verify_forecast(packet, sub, bars)
            families = sorted({r.split(":")[0] for r in gated["reasons"]})
            status = ("FAILS " + ", ".join(families) + "; e.g. " + "; ".join(gated["reasons"][:2])) if not gated["pass"] else "PASSES (bar too loose)"
            lines.append(f"- control {control} on {packet.name}: {status}")
            ok &= not gated["pass"]
    lines += ["", "RESULT: " + ("bars frozen; every control fails a named gate" if ok else "NOT FROZEN")]
    (out / "freeze_report.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
