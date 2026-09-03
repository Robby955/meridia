"""Shared plumbing for reference methods: packet loading, submission writing, the
development-world calibration of income nonresponse. Each strong line keeps its own
estimation; what is shared here is the tuning channel every participant also has."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..release import AGE_BAND_LABELS, SEX_LABELS

INCOME_ITEMS = ("median_household_income", "mean_income_adults", "low_income_household_share")
COUNT_ITEMS = ("persons", "households", "children_under_16", "elders_65_plus")


def load_packet(packet_dir: Path) -> dict:
    import pandas as pd
    P = Path(packet_dir) / "participant"
    contract = json.loads((P / "contract.json").read_text())
    geography = pd.read_csv(P / "geography.csv")
    return {
        "contract": contract,
        "county_state": geography["state"].to_numpy(dtype=np.int64),
        "survey": pd.read_csv(P / "survey_revised.csv"),
        "population": pd.read_csv(P / "sources" / "population_revised.csv"),
        "population_preliminary": pd.read_csv(P / "sources" / "population_preliminary.csv"),
        "income": pd.read_csv(P / "sources" / "income_revised.csv"),
        "health": pd.read_csv(P / "sources" / "health_revised.csv"),
        "benchmark": _load_benchmark(P / "sources" / "benchmark_revised.csv"),
    }


def _load_benchmark(path: Path) -> dict | None:
    from .design_based import _load_benchmark as load
    return load(path)


def income_dispersion(frame) -> float:
    """Weighted standard deviation of log adult income in the survey: the observable
    proxy for the world's inequality, which drives how selective response is."""
    adults = frame[(frame["age"] >= 16) & (frame["income"] > 0)]
    x = np.log(adults["income"].to_numpy(dtype=np.float64))
    w = adults["weight"].to_numpy(dtype=np.float64)
    mean = (w * x).sum() / w.sum()
    return float(np.sqrt((w * (x - mean) ** 2).sum() / w.sum()))


def calibrate_income(run_fn, dev_packet_dirs, calibration_path: Path) -> dict:
    """Fit income nonresponse corrections for one method on development worlds.

    ``run_fn(packet_dir, out_dir)`` must return a dict with ``release`` rows and the
    survey ``dispersion``. The method's remaining national bias per income item (a
    log-ratio, or a difference for the share) is fitted as a linear function of the
    dispersion across worlds; with fewer than three worlds it is a constant. On a hidden
    world the correction is read off at that world's observed dispersion, held at the
    nearer edge of the development range when the world lies outside it.
    """
    import pandas as pd
    dev_packet_dirs = [Path(d) for d in ([dev_packet_dirs] if isinstance(dev_packet_dirs, (str, Path)) else dev_packet_dirs)]
    rows = []
    for k, dev in enumerate(dev_packet_dirs):
        result = run_fn(dev, Path(calibration_path).parent / f"_calibration_run_{k}")
        truth = pd.read_csv(dev / "participant" / "truth" / "truth_revised.csv")
        nation = truth[truth["level"] == "nation"].set_index("estimand")["value"]
        estimate = {r["estimand"]: r["estimate"] for r in result["release"] if r["level"] == "nation"}
        row = {"dispersion": float(result["dispersion"])}
        for e in INCOME_ITEMS:
            row[e] = float(nation[e] - estimate[e]) if e == "low_income_household_share" \
                else float(np.log(nation[e] / estimate[e]))
        rows.append(row)
    d = np.asarray([r["dispersion"] for r in rows])
    # Read inside the development range of the dispersion only; held at the nearer
    # edge beyond it, since three worlds fix a slope too loosely to extrapolate.
    factors = {"dispersion_reference": float(d.mean()), "n_worlds": len(rows),
               "dispersion_range": [float(d.min()), float(d.max())]}
    for e in INCOME_ITEMS:
        y = np.asarray([r[e] for r in rows])
        if len(rows) >= 3 and d.std() > 1e-6:
            slope, intercept = np.polyfit(d, y, 1)
        else:
            slope, intercept = 0.0, float(y.mean())
        residual = y - (intercept + slope * d)
        # Conservative: the largest miss across development worlds, not the average,
        # since the fit has few points and the hidden world may sit at an edge.
        residual_sd = float(np.abs(residual).max()) if len(rows) >= 2 else float(abs(y).mean())
        residual_sd = max(residual_sd, 0.5 * float(np.std(y))) if len(rows) >= 2 else residual_sd
        factors[e] = {"intercept": float(intercept), "slope": float(slope),
                      "residual_sd": max(residual_sd, 0.01)}
    Path(calibration_path).write_text(json.dumps(factors, indent=1, sort_keys=True) + "\n")
    return factors


def apply_calibration(values: dict, factors: dict, dispersion: float) -> dict:
    out = dict(values)
    if "dispersion_range" in factors:
        dispersion = float(np.clip(dispersion, *factors["dispersion_range"]))
    for (e, level, u), v in values.items():
        if e not in factors or not np.isfinite(v):
            continue
        f = factors[e]
        shift = f["intercept"] + f["slope"] * dispersion if isinstance(f, dict) else float(f)
        shift = float(np.clip(shift, -0.25, 0.25))   # a correction beyond this is a model failure, not a fix
        if e == "low_income_household_share":
            out[(e, level, u)] = float(min(max(v + shift, 0.0), 1.0))
        else:
            out[(e, level, u)] = float(v * np.exp(shift))
    return out


def calibration_half_widths(point: dict, factors: dict, z: float = 1.645) -> dict:
    """Extra half-width per income item from the calibration's residual spread across
    development worlds: the correction is uncertain, and honest intervals say so."""
    extra = {}
    for (e, level, u), v in point.items():
        f = factors.get(e)
        if not isinstance(f, dict) or not np.isfinite(v):
            continue
        sd = f.get("residual_sd", 0.0)
        extra[(e, level, u)] = z * sd if e == "low_income_household_share" else z * sd * abs(v)
    return extra


def load_factors(path: str | None) -> dict:
    return json.loads(Path(path).read_text()) if path else {}


def rows_from_draws(point: dict, draws: dict, extra_half: dict | None = None) -> list[dict]:
    """Release rows: point estimate with a symmetric 90 percent half-width from draws,
    optionally widened per key; empty units publish zeros."""
    rows = []
    for key in sorted(point):
        v = point[key]
        sample = np.asarray(draws.get(key, []), dtype=np.float64)
        sample = sample[np.isfinite(sample)]
        if not np.isfinite(v):
            v, lower, upper = 0.0, 0.0, 0.0
        else:
            half = 0.0
            if len(sample) >= 10:
                lo, hi = np.percentile(sample, [5, 95])
                half = 0.5 * (hi - lo)
            if extra_half and key in extra_half:
                half = float(np.sqrt(half ** 2 + extra_half[key] ** 2))
            lower, upper = v - half, v + half
        proportion = key[0].endswith("share") or key[0].startswith("tertiary")
        lower = max(lower, 0.0)
        if proportion:
            upper = min(upper, 1.0)
            v = min(max(v, lower), upper)
        rows.append({"estimand": key[0], "level": key[1], "unit": int(key[2]),
                     "estimate": float(v), "lower": float(min(lower, v)), "upper": float(max(upper, v))})
    return rows


def write_submission(out_dir: Path, release_rows, projection_rows, cube: np.ndarray,
                     suppress_below: float, allocation: np.ndarray) -> None:
    import pandas as pd
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    n_counties = cube.shape[0]
    detail = []
    for c in range(n_counties):
        for b, band in enumerate(AGE_BAND_LABELS):
            for s, sex in enumerate(SEX_LABELS):
                value = float(cube[c, b, s])
                detail.append({"county": c, "age_band": band, "sex": sex,
                               "count": "" if 0 < value < suppress_below else round(value, 3)})
    pd.DataFrame(release_rows).to_csv(out_dir / "release.csv", index=False)
    pd.DataFrame(projection_rows).to_csv(out_dir / "projection.csv", index=False)
    pd.DataFrame(detail).to_csv(out_dir / "detailed.csv", index=False)
    pd.DataFrame({"county": np.arange(n_counties), "allocation": allocation}).to_csv(
        out_dir / "allocation.csv", index=False)
