"""Proof obligation 4: does the reserve decision have attainable value, world by world?

The question is a property of a world and its published contract, not of any submission,
so this script asks it with the truth in hand: give the submission the exact regional
quantiles, and measure what a sealed-information allocation of the same published total
saves over the frozen practical baseline. The baseline is the public rule the contract
publishes, the total split in proportion to each region's share of persons at or above
the eligibility age, and the oracle spends the same total under non-negativity alone.

Reported per world: J at the baseline, J at the oracle, the gain, and the slack the
published total leaves above the sum of the true quantiles, which is what the decision
has to work with.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from meridia.actuarial import (ensemble_truth, expected_uncovered,
                               perfect_information_allocation,
                               proportional_baseline_allocation)


def value_of_one_world(packet: Path) -> dict:
    contract = json.loads((packet / "participant" / "contract.json").read_text())
    reserve = contract["reserve"]
    with np.load(packet / "retained" / "continuation_liabilities.npz") as archive:
        liability = np.asarray(archive["liability"], dtype=np.float64)
    weights = np.asarray(reserve["weights"], dtype=np.float64) \
        if reserve.get("weights") else None
    total = float(reserve["total"])
    truth = ensemble_truth(liability)
    q = truth["q"]
    share = np.asarray(reserve["baseline_share"], dtype=np.float64) \
        if reserve.get("baseline_share") else q
    baseline = proportional_baseline_allocation(share, total)
    oracle = perfect_information_allocation(liability, total, weights)
    j_baseline = expected_uncovered(baseline, liability, weights)
    j_oracle = expected_uncovered(oracle, liability, weights)
    # A held-out oracle: fitted on half the ensemble, paid on the other half. It is what
    # separates a real decision from an allocation fitted to Monte Carlo noise.
    half = liability.shape[0] // 2
    fitted = perfect_information_allocation(liability[:half], total, weights)
    j_held_baseline = expected_uncovered(baseline, liability[half:], weights)
    j_held_fitted = expected_uncovered(fitted, liability[half:], weights)
    return {
        "world": packet.name, "members": int(liability.shape[0]),
        "regions": int(liability.shape[1]), "total": total,
        "slack_over_sum_q": float((total - q.sum()) / q.sum()),
        "tail_width_min": float(((q - truth["mean"]) / truth["mean"]).min()),
        "tail_width_max": float(((q - truth["mean"]) / truth["mean"]).max()),
        "J_baseline": j_baseline, "J_oracle": j_oracle,
        "gain": (j_baseline - j_oracle) / max(j_baseline, 1e-12),
        "held_out_gain": (j_held_baseline - j_held_fitted) / max(j_held_baseline, 1e-12),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--packets", nargs="+", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    rows = [value_of_one_world(Path(p)) for p in args.packets]
    lines = ["# Reserve decision value on the qualification worlds", ""]
    for row in rows:
        lines.append(
            f"- {row['world']}: R {row['total']:,.0f}, {row['regions']} regions, "
            f"{row['members']} continuations; slack {row['slack_over_sum_q']:.4f}; "
            f"regional tail width {row['tail_width_min']:.3f} to {row['tail_width_max']:.3f}; "
            f"J(A_B) {row['J_baseline']:,.0f}, J(A*) {row['J_oracle']:,.0f}, "
            f"gain {row['gain']:.2%}, held out {row['held_out_gain']:.2%}")
    gains = np.asarray([row["gain"] for row in rows])
    held = np.asarray([row["held_out_gain"] for row in rows])
    lines += ["", f"gain: min {gains.min():.2%}, median {np.median(gains):.2%}, "
                  f"max {gains.max():.2%}",
              f"held out: min {held.min():.2%}, median {np.median(held):.2%}, "
              f"max {held.max():.2%}"]
    text = "\n".join(lines) + "\n"
    print(text)
    if args.out:
        Path(args.out).write_text(text)


if __name__ == "__main__":
    main()
