"""Draw the gate matrix of a bar freeze from its stored submissions: every method and
control against every gate family on every qualification world. Nothing is typed in;
each cell is re-verified from the submission files and the retained truth.

    python scripts/render_gate_matrix.py --freeze PATH/bars-national-N --out renders/gate-matrix.png
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from meridia.verify import verify_submission

FAMILIES = ("schema", "additivity", "accuracy", "coverage", "interval score", "disclosure",
            "projection accuracy", "projection coverage", "projection interval score", "allocation")


def family_of(reason: str) -> str:
    head = reason.split(":")[0].strip()
    return head if head in FAMILIES else head


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--freeze", required=True)
    ap.add_argument("--packets", default=str(Path.home() / "Projects" / "meridia-packets"))
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    freeze = Path(args.freeze)
    bars = json.loads((freeze / "bars.json").read_text())
    packets = Path(args.packets)
    rows = []          # (world, submission name, {family: failed})
    for world_dir in sorted((freeze / "runs").iterdir()):
        packet = packets / world_dir.name
        for sub in sorted(world_dir.iterdir()):
            report = verify_submission(packet, sub, bars)
            failed = {family_of(r) for r in report["reasons"]}
            rows.append((world_dir.name, sub.name, failed))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    labels = [f"{w.replace('hidden-', '')}  {s.replace('control_', '')}" for w, s, _ in rows]
    grid = np.zeros((len(rows), len(FAMILIES)))
    for i, (_, _, failed) in enumerate(rows):
        for j, fam in enumerate(FAMILIES):
            grid[i, j] = 1.0 if fam in failed else 0.0
    fig, ax = plt.subplots(figsize=(11, 0.42 * len(rows) + 1.6), dpi=180)
    ax.imshow(grid, cmap=matplotlib.colors.ListedColormap([[0.80, 0.90, 0.80], [0.85, 0.45, 0.40]]),
              aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(len(FAMILIES)))
    ax.set_xticklabels(FAMILIES, rotation=35, ha="right", fontsize=8)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(labels, fontsize=8)
    for i, (_, s, failed) in enumerate(rows):
        strong = s in ("A", "B")
        verdict = ("PASS" if not failed else "FAIL") if strong else ("fails" if failed else "PASSES")
        ax.text(len(FAMILIES) - 0.4, i, verdict, va="center", ha="left", fontsize=8,
                color=("black" if (strong and not failed) or (not strong and failed) else "red"))
    ax.set_xlim(-0.5, len(FAMILIES) + 1.2)
    ax.set_title("Gate matrix: strong methods must pass every gate, controls must fail one "
                 "(red = gate failed)", fontsize=9, loc="left")
    fig.tight_layout()
    fig.savefig(args.out, bbox_inches="tight")
    print(f"wrote {args.out} ({len(rows)} submissions re-verified)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
