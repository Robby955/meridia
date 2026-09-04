"""Draw the control separation matrix of a frozen bar set, straight from bars.json.

Every cell is the number of qualification worlds on which one registered control fails
one composite gate. Nothing is typed in and nothing outside the repository is read: the
only input is the freeze directory already tracked under bars/.

    python scripts/render_control_separation.py \
        --freeze bars/national-v14-standard \
        --out renders/control-separation-v14-standard.svg
"""

import argparse
import json
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--freeze", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    bars = json.loads((args.freeze / "bars.json").read_text())
    support = bars["control_support"]
    matrix = support["matrix"]
    worlds = list(bars["qualification_worlds"])
    gates = list(bars["gates"].keys())
    reported_only = set(bars.get("reported_only_gates", []))
    by_gate = support["registered_controls_by_gate"]
    home = {c: g for g, cs in by_gate.items() for c in cs}
    controls = sorted(matrix, key=lambda c: (gates.index(home.get(c, gates[0])), c))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    grid = np.full((len(controls), len(gates)), np.nan)
    for i, control in enumerate(controls):
        for j, gate in enumerate(gates):
            cell = matrix[control]["gates"].get(gate)
            if cell is None:
                continue
            failed = set(cell["failed_worlds"]) | set(cell["hard_invalid_worlds"])
            grid[i, j] = len(failed & set(worlds))

    fig, ax = plt.subplots(figsize=(9.0, 0.34 * len(controls) + 2.4), dpi=200)
    ax.imshow(grid, cmap="YlOrRd", aspect="auto", vmin=0, vmax=len(worlds))
    for i in range(len(controls)):
        for j in range(len(gates)):
            value = grid[i, j]
            if np.isnan(value):
                continue
            ax.text(j, i, f"{int(value)}", ha="center", va="center", fontsize=8,
                    color="black" if value < len(worlds) * 0.6 else "white")
    for i, control in enumerate(controls):
        j = gates.index(home[control])
        ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                   edgecolor="black", linewidth=1.6))

    ax.set_xticks(range(len(gates)))
    ax.set_xticklabels([g + ("\n(reported only)" if g in reported_only else "")
                        for g in gates], rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(len(controls)))
    ax.set_yticklabels(controls, fontsize=8)
    ax.set_title(
        f"Control separation, {args.freeze.name}: worlds failed out of {len(worlds)}\n"
        f"outline = the gate the control is registered against; "
        f"reference lines {', '.join(bars['reference_lines'])} fail none",
        fontsize=9, loc="left")
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight")
    print(f"wrote {args.out} ({len(controls)} controls x {len(gates)} gates, "
          f"{len(worlds)} qualification worlds)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
