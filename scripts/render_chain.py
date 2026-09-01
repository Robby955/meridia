"""Draw the chain on one qualification world from stored state and stored submissions:
county reconstruction error, five-year projection against the sealed future, and the
committed allocation against the demand that arrived. Every number is read back from
the packet's retained truth and a method's submission files; nothing is typed in.

    python scripts/render_chain.py --packet PATH/hidden-2026xxxx --submission PATH/runs/hidden-2026xxxx/A --out renders/chain.png
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from meridia.verify import admin_from_packet, load_rows, load_truth


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--packet", required=True)
    ap.add_argument("--submission", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--label", default="strong method")
    args = ap.parse_args()
    packet, sub = Path(args.packet), Path(args.submission)
    admin = admin_from_packet(packet)
    n = admin["n_counties"]
    truth_now = load_truth(packet / "retained" / "truth_revised.csv")
    truth_future = load_truth(packet / "retained" / "truth_horizon.csv")
    release = {(r["estimand"], r["level"], r["unit"]): r for r in load_rows(sub / "release.csv")}
    projection = {(r["estimand"], r["level"], r["unit"]): r for r in load_rows(sub / "projection.csv")}
    import pandas as pd
    allocation = pd.read_csv(sub / "allocation.csv").sort_values("county")["allocation"].to_numpy()
    contract = json.loads((packet / "participant" / "contract.json").read_text())
    demand = np.asarray([truth_future[(contract["allocation"]["demand"], "county", c)] for c in range(n)])
    order = np.argsort([truth_now[("persons", "county", c)] for c in range(n)])

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), dpi=180)

    ax = axes[0]
    t = np.asarray([truth_now[("persons", "county", c)] for c in order])
    e = np.asarray([release[("persons", "county", c)]["estimate"] for c in order])
    lo = np.asarray([release[("persons", "county", c)]["lower"] for c in order])
    hi = np.asarray([release[("persons", "county", c)]["upper"] for c in order])
    x = np.arange(n)
    ax.errorbar(x, e / t, yerr=[(e - lo) / t, (hi - e) / t], fmt="o", ms=3, lw=0.8, color="#3b5b8c", label=f"{args.label}: estimate / truth")
    ax.axhline(1.0, color="black", lw=0.8)
    ax.set_xticks(x[::max(1, n // 8)])
    ax.set_xticklabels([f"{int(v/1000)}k" for v in t[::max(1, n // 8)]], fontsize=7)
    ax.set_xlabel("counties, smallest to largest (true persons)")
    ax.set_ylabel("estimate relative to truth")
    ax.set_title("Reconstruction: county persons with 90% intervals", fontsize=9, loc="left")
    ax.legend(fontsize=7, loc="upper right")

    ax = axes[1]
    tf = np.asarray([truth_future[("elders_65_plus", "county", c)] for c in order])
    pf = np.asarray([projection[("elders_65_plus", "county", c)]["estimate"] for c in order])
    plo = np.asarray([projection[("elders_65_plus", "county", c)]["lower"] for c in order])
    phi = np.asarray([projection[("elders_65_plus", "county", c)]["upper"] for c in order])
    ax.errorbar(x, pf / tf, yerr=[(pf - plo) / tf, (phi - pf) / tf], fmt="s", ms=3, lw=0.8, color="#8c5b3b", label="projected / realized, elders 65+")
    ax.axhline(1.0, color="black", lw=0.8)
    ax.set_xticks(x[::max(1, n // 8)])
    ax.set_xticklabels([f"{int(v/1000)}k" for v in t[::max(1, n // 8)]], fontsize=7)
    ax.set_xlabel("same counties")
    ax.set_title("Projection: elders at the horizon against the sealed future", fontsize=9, loc="left")
    ax.legend(fontsize=7, loc="upper right")

    ax = axes[2]
    d = demand[order]
    a = allocation[order]
    ax.bar(x - 0.2, d / d.sum(), width=0.4, color="#666666", label="share of realized demand")
    ax.bar(x + 0.2, a / max(a.sum(), 1e-9), width=0.4, color="#3b8c5b", label="share of committed allocation")
    ax.set_xticks(x[::max(1, n // 8)])
    ax.set_xticklabels([f"{int(v/1000)}k" for v in t[::max(1, n // 8)]], fontsize=7)
    ax.set_xlabel("same counties")
    unmet = float(np.maximum(demand - allocation, 0).sum() / demand.sum())
    ax.set_title(f"Allocation: committed before the future, unmet demand {unmet:.1%}", fontsize=9, loc="left")
    ax.legend(fontsize=7, loc="upper left")
    fig.suptitle(f"One qualification world ({packet.name}), {args.label}, drawn from retained truth and the stored submission",
                 fontsize=10, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(args.out, bbox_inches="tight")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
