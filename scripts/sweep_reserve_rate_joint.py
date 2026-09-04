"""Sweep the reserve rate under the joint identification and shortfall condition.

The earlier rate rule asked one question of a candidate rate: does the published
proportional baseline still carry a measurably larger expected uncovered obligation than
a perfect-information allocation of the same total, on every qualification world. A rate
that fails it leaves the reserve skill score undefined. It selected 3769, and at 3769 the
second reading of the same reserve block saturates: the worst-region shortfall
probability is one on every reference report, the top of that component's attainable
range, so the calibrated bar cannot land below the range's ceiling and the freeze refuses
it as a bar that nothing can fail.

This script asks both questions of every rate on the ladder. A rate is admissible when

  (a) on every qualification world, the baseline-minus-oracle expected uncovered
      obligation is at least ``--margin-share`` of that world's sealed mean total
      liability, and
  (b) on every qualification world and every reference line, the worst-region shortfall
      probability of that line's allocation, read against the sealed continuation
      ensemble, is at or below ``--shortfall-ceiling``, the registered
      ``regional_shortfall_ceiling``. That ceiling lies strictly inside the component's
      attainable range of zero to one, which is what the freeze's attainable-range rule
      asks of a published bar.

Both readings are freeze-side. Neither is available to a participant, and neither moves
any published value on its own: the output is a table and a verdict, and the rate a
packet compiles stays where ``meridia/packet.py`` puts it.

The three reference lines simulate their own liability paths and then allocate the
published total against those paths. The paths do not read the total, so one run of each
line on each world supports every candidate on the ladder. The run is instrumented at the
allocation call, which records the paths it is handed and then defers to the original.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from meridia.actuarial import (PLACEHOLDER_THRESHOLDS, ensemble_truth,
                               proportional_baseline_allocation, reserve_total)
from meridia.methods import actuarial_reference as AR
from meridia.methods import phase_three

import build_v4_freeze_evidence as evidence

REFERENCE_LINES = ("A", "B", "C")
QUALIFICATION_WORLDS = tuple(f"qual-{index}" for index in range(6))
DEFAULT_MARGIN_SHARE = 0.01
DEFAULT_SHORTFALL_CEILING = PLACEHOLDER_THRESHOLDS.regional_shortfall_ceiling
JOINT_RULE = (
    "largest candidate rate at which, on every qualification world, the "
    "baseline-minus-oracle expected uncovered obligation is at least "
    "margin_share of the sealed mean total liability and every reference line's "
    "worst-region shortfall probability is at or below the registered "
    "regional_shortfall_ceiling"
)


class SweepError(ValueError):
    """A world, a calibration input, or a reference run is missing or unusable."""


class Fill:
    """The greedy fill of ``perfect_information_allocation``, evaluated at many totals.

    The optimizer raises each region through the intervals between its ordered
    continuations, taking the highest marginal value first. The order of those intervals
    does not depend on the total, so sorting them once answers every candidate rate, and
    the answers agree with the optimizer to floating-point rounding.
    """

    def __init__(self, liability: np.ndarray, weights: np.ndarray | None) -> None:
        values = np.asarray(liability, dtype=np.float64)
        if values.ndim != 2 or min(values.shape, default=0) <= 0:
            raise SweepError("liability must be a nonempty members-by-regions array")
        members, regions = values.shape
        weight = (np.ones(regions) if weights is None
                  else np.asarray(weights, dtype=np.float64))
        order = np.sort(values, axis=0)
        widths = np.diff(np.vstack([np.zeros((1, regions)), order]), axis=0)
        slopes = weight[None, :] * ((members - np.arange(members))[:, None] / members)
        rank = np.repeat(np.arange(members)[:, None], regions, axis=1)
        region = np.repeat(np.arange(regions)[None, :], members, axis=0)
        keep = (widths > 0) & (weight[None, :] > 0)
        sequence = np.lexsort((rank[keep], region[keep], -slopes[keep]))
        self._width = widths[keep][sequence]
        self._region = region[keep][sequence]
        self._cumulative = np.cumsum(self._width)
        filled = np.zeros((len(self._width) + 1, regions))
        running = np.zeros(regions)
        for index, (region_index, width) in enumerate(zip(self._region, self._width)):
            running[region_index] += width
            filled[index + 1] = running
        self._filled = filled
        self._order = order
        self._above = np.cumsum(order[::-1], axis=0)[::-1]
        self._members = members
        self._regions = regions
        self._weight = weight

    def allocate(self, total: float) -> np.ndarray:
        total = float(total)
        index = int(np.searchsorted(self._cumulative, total, side="left"))
        if index >= len(self._cumulative):
            allocation = self._filled[-1].copy()
            allocation[0] += total - float(self._cumulative[-1])
            return allocation
        allocation = self._filled[index].copy()
        spent = 0.0 if index == 0 else float(self._cumulative[index - 1])
        allocation[int(self._region[index])] += total - spent
        return allocation

    def expected_uncovered(self, allocation: np.ndarray) -> float:
        total = 0.0
        for region in range(self._regions):
            column = self._order[:, region]
            cut = int(np.searchsorted(column, allocation[region], side="right"))
            above = float(self._above[cut, region]) if cut < self._members else 0.0
            mean = (above - allocation[region] * (self._members - cut)) / self._members
            total += float(self._weight[region]) * mean
        return total

    def worst_exceedance(self, allocation: np.ndarray) -> float:
        worst = 0.0
        for region in range(self._regions):
            cut = int(np.searchsorted(self._order[:, region], allocation[region],
                                      side="right"))
            worst = max(worst, (self._members - cut) / self._members)
        return worst


def _world_inputs(packet: Path) -> dict[str, Any]:
    import calibrate_reserve_rate as calibration

    exposure, rounding_unit, _, _ = calibration._public_exposure(packet)
    geometry = calibration._identification_inputs(packet)
    return {
        "exposure_person_years": exposure,
        "rounding_unit": rounding_unit,
        "liability": geometry["liability"],
        "share": geometry["share"],
        "weights": geometry["weights"],
        "sealed_mean_total_liability": geometry["sealed_mean_total_liability"],
    }


def _reference_paths(line: str, packet: Path, stage: Path, calibration_a: Path,
                     calibration_b: Path,
                     params: phase_three.MeasurementParams) -> np.ndarray:
    """Run one reference line on one world and return the paths it allocated against."""
    recorded: dict[str, np.ndarray] = {}
    original = AR.allocate_reserve

    def recording(liability, total, weights=None):
        recorded["liability"] = np.asarray(liability, dtype=np.float64).copy()
        return original(liability, total, weights)

    AR.allocate_reserve = recording
    try:
        evidence._run_reference_line(line, packet, stage, calibration_a, calibration_b,
                                     params)
    finally:
        AR.allocate_reserve = original
    if "liability" not in recorded:
        raise SweepError(f"{line} on {packet.name} allocated no reserve")
    return recorded["liability"]


def sweep(qualification_root: Path, calibration_a: Path, calibration_b: Path,
          stage_root: Path, low: int, high: int, margin_share: float,
          shortfall_ceiling: float,
          params: phase_three.MeasurementParams) -> dict[str, Any]:
    if high < low:
        raise SweepError("the ladder's upper endpoint is below its lower endpoint")
    worlds = {}
    for name in QUALIFICATION_WORLDS:
        packet = qualification_root / name
        if not (packet / "manifest.json").is_file():
            raise SweepError(f"{name}: qualification packet is missing")
        worlds[name] = _world_inputs(packet)
    sealed = {name: Fill(world["liability"], world["weights"])
              for name, world in worlds.items()}
    lines: dict[tuple[str, str], Fill] = {}
    for name in QUALIFICATION_WORLDS:
        for line in REFERENCE_LINES:
            stage = stage_root / f"{line}_{name}"
            paths = _reference_paths(line, qualification_root / name, stage,
                                     calibration_a, calibration_b, params)
            lines[(line, name)] = Fill(paths, worlds[name]["weights"])

    records = []
    for rate in range(int(low), int(high) + 1):
        margins: dict[str, float] = {}
        shortfalls: dict[str, dict[str, float]] = {}
        for name, world in worlds.items():
            total = reserve_total(world["exposure_person_years"], float(rate),
                                  world["rounding_unit"])
            fill = sealed[name]
            baseline = proportional_baseline_allocation(world["share"], total)
            margins[name] = ((fill.expected_uncovered(baseline)
                              - fill.expected_uncovered(fill.allocate(total)))
                             / world["sealed_mean_total_liability"])
            shortfalls[name] = {
                line: fill.worst_exceedance(lines[(line, name)].allocate(total))
                for line in REFERENCE_LINES
            }
        worst_margin = min(margins.values())
        worst_shortfall = max(max(row.values()) for row in shortfalls.values())
        records.append({
            "rate_per_person_year": rate,
            "worst_margin_share": worst_margin,
            "identified": bool(worst_margin >= margin_share),
            "worst_reference_shortfall_probability": worst_shortfall,
            "under_registered_ceiling": bool(worst_shortfall <= shortfall_ceiling),
            "margin_share_by_world": margins,
            "shortfall_probability_by_world_and_line": shortfalls,
        })

    identified = [row["rate_per_person_year"] for row in records if row["identified"]]
    under = [row["rate_per_person_year"] for row in records
             if row["under_registered_ceiling"]]
    joint = [row["rate_per_person_year"] for row in records
             if row["identified"] and row["under_registered_ceiling"]]
    result: dict[str, Any] = {
        "schema": "meridia.reserve-rate-joint-sweep.v1",
        "ladder": {"low": int(low), "high": int(high), "grid": 1},
        "margin_share": float(margin_share),
        "registered_shortfall_ceiling": float(shortfall_ceiling),
        "joint_rule": JOINT_RULE,
        "identified_rates": {"low": identified[0] if identified else None,
                             "high": identified[-1] if identified else None,
                             "count": len(identified)},
        "under_ceiling_rates": {"low": under[0] if under else None,
                                "high": under[-1] if under else None,
                                "count": len(under)},
        "chosen": max(joint) if joint else None,
        "records": records,
    }
    if not joint:
        result["nearest"] = {
            "highest_identified_rate": identified[-1] if identified else None,
            "lowest_rate_under_ceiling": under[0] if under else None,
            "gap_rate_points": (under[0] - identified[-1]
                                if identified and under else None),
        }
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qualification-root", type=Path, required=True)
    parser.add_argument("--calibration-a", type=Path, required=True)
    parser.add_argument("--calibration-b", type=Path, required=True)
    parser.add_argument("--stage", type=Path, required=True,
                        help="scratch directory for the eighteen reference runs")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--low", type=int, required=True)
    parser.add_argument("--high", type=int, required=True)
    parser.add_argument("--margin-share", type=float, default=DEFAULT_MARGIN_SHARE)
    parser.add_argument("--shortfall-ceiling", type=float,
                        default=DEFAULT_SHORTFALL_CEILING)
    parser.add_argument("--bootstrap", type=int, default=100)
    parser.add_argument("--sweeps", type=int, default=400)
    parser.add_argument("--simulation-paths", type=int, default=2048)
    parser.add_argument("--linkage-bootstraps", type=int, default=12)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = sweep(
            args.qualification_root, args.calibration_a, args.calibration_b,
            args.stage, args.low, args.high, args.margin_share, args.shortfall_ceiling,
            phase_three.MeasurementParams(
                bootstrap_replicates=args.bootstrap,
                bayesian_sweeps=args.sweeps,
                simulation_paths=args.simulation_paths,
                linkage_bootstraps=args.linkage_bootstraps,
            ),
        )
    except (SweepError, OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1, sort_keys=True))
    print(json.dumps({key: result[key] for key in
                      ("chosen", "identified_rates", "under_ceiling_rates")
                      if key in result}, indent=2, sort_keys=True))
    if result["chosen"] is None:
        print(json.dumps(result["nearest"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
