"""Build one small world, run reference line A on its participant files, verify the
result, and print the report. This is the shortest end to end path through the
repository and it needs nothing that is not in the tree.

    python scripts/demo_verify.py

It writes into a temporary directory and leaves nothing behind. Expect about a minute
on four cores. The world is a miniature, forty thousand people rather than the
2,400,000 of a graded world, so the stochastic composites are not expected to clear
bars calibrated on full worlds; the deterministic file, schema, additivity, and
reserve feasibility checks are, and the report prints every composite either way.
"""

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from meridia.methods import actuarial_reference as AR
from meridia.methods import design_based
from meridia.packet import PacketParams, build_packet
from meridia.verify import summary_table, verify_submission

SEED = 4711
PARAMS = PacketParams(grid=(72, 96), n_settlements=6, n_states=2, observed_months=24,
                      preliminary_lag=3, horizon_months=12, total=40_000,
                      experience_years=1, ensemble_members=32)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bars", type=Path, default=Path("bars/national-v14-standard/bars.json"))
    ap.add_argument("--gate-profile", default="standard")
    args = ap.parse_args()

    bars = json.loads(args.bars.read_text())
    with tempfile.TemporaryDirectory(prefix="meridia-demo-") as tmp:
        root = Path(tmp)
        world = root / "world"
        print("building one miniature development world ...", flush=True)
        build_packet(SEED, world, PARAMS, development=True)

        blind = root / "packet"
        blind.mkdir()
        shutil.copytree(world / "participant", blind / "participant")

        submission = root / "submission"
        print("running reference line A on the participant files ...", flush=True)
        design_based.run(
            blind,
            submission,
            design_based.MethodParams(
                bootstrap_replicates=10,
                actuarial="on",
                actuarial_params=AR.LayerParams(
                    simulation=AR.SimulationParams(n_paths=32, path_chunk=16)),
            ),
        )
        print("submission files:",
              ", ".join(sorted(p.name for p in submission.iterdir())), flush=True)

        report = verify_submission(world, submission, bars=bars,
                                   gate_profile=args.gate_profile)
        print()
        print(summary_table(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
