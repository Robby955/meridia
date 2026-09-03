"""Independent elder exposure, linkage, and mortality choices for the third line.

The base population reconstruction remains design-based. An absolute elder
cohort-component advances the lagged experience stock with its deaths and net migration,
then reconciles linked-register and survey county shares. Vintage linkage uses unique
exact matches followed by a clerical comparison-field bootstrap. Mortality trend uses
cellwise log-linear slopes from the experience file and a weighted median across cells.
It does not use the shared fitted mixture or shock-posterior trend.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import actuarial_reference as AR
from . import design_based


@dataclass(frozen=True)
class ThirdReferenceParams:
    bootstrap_replicates: int = 80
    linkage_bootstraps: int = 12
    simulation_paths: int = 2048
    seed: int = 20260905
    calibration_path: str | None = None


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values)
    values = np.asarray(values, dtype=np.float64)[order]
    weights = np.maximum(np.asarray(weights, dtype=np.float64)[order], 0.0)
    if len(values) == 0 or weights.sum() <= 0:
        return 0.0
    index = int(np.searchsorted(np.cumsum(weights), 0.5 * weights.sum(), side="left"))
    return float(values[min(index, len(values) - 1)])


def stratified_mortality_trend(experience, n_states: int) -> dict:
    """Cellwise Poisson log-rate slopes combined by their information.

    Each state, age band, and sex supplies one five-year slope. The weighted median makes
    one unusual year local to its cells instead of fitting a common year effect. Sampling
    error and the disagreement between cell slopes set the reported trend uncertainty.
    """
    arrays = AR.experience_arrays(experience, n_states)
    exposure = np.asarray(arrays["exposure"], dtype=np.float64)
    deaths = np.asarray(arrays["deaths"], dtype=np.float64)
    year = np.arange(exposure.shape[0], dtype=np.float64)
    slopes, information, variances = [], [], []
    for s in range(exposure.shape[1]):
        for b in range(exposure.shape[2]):
            for x in range(exposure.shape[3]):
                e = exposure[:, s, b, x]
                d = deaths[:, s, b, x]
                keep = (e >= 500.0) & np.isfinite(e) & np.isfinite(d)
                if keep.sum() < 3:
                    continue
                t = year[keep]
                w = np.maximum(d[keep] + 0.5, 0.5)
                y = np.log((d[keep] + 0.5) / np.maximum(e[keep], 1.0))
                centre = float((w * t).sum() / w.sum())
                denominator = float((w * (t - centre) ** 2).sum())
                if denominator <= 0:
                    continue
                level = float((w * y).sum() / w.sum())
                slope = float((w * (t - centre) * (y - level)).sum() / denominator)
                slopes.append(float(np.clip(slope, -0.15, 0.15)))
                information.append(denominator)
                variances.append(1.0 / denominator)
    if len(slopes) < 4:
        return {
            "mortality_drift": 0.0,
            "mortality_drift_se": 0.05,
            "n_cells": len(slopes),
            "fitted": False,
            "strategy": "cellwise_weighted_median",
        }
    slope = np.asarray(slopes, dtype=np.float64)
    weight = np.asarray(information, dtype=np.float64)
    point = _weighted_median(slope, weight)
    mad = _weighted_median(np.abs(slope - point), weight)
    sampling = float(np.average(np.asarray(variances), weights=weight))
    effective = float(weight.sum() ** 2 / np.maximum((weight**2).sum(), 1e-12))
    se = np.sqrt(sampling + (1.4826 * mad) ** 2 / max(effective, 1.0))
    return {
        "mortality_drift": float(point),
        "mortality_drift_se": float(np.clip(se, 1e-4, 0.05)),
        "n_cells": len(slopes),
        "fitted": True,
        "strategy": "cellwise_weighted_median",
    }


def run(
    packet_dir: Path,
    out_dir: Path,
    params: ThirdReferenceParams = ThirdReferenceParams(),
) -> dict:
    """Write the third reference submission from participant files only."""
    packet_dir = Path(packet_dir)
    data = design_based.load_packet(packet_dir)
    experience = AR.load_experience(packet_dir, data["contract"])
    if experience is None:
        raise AR.MissingActuarialInputs(
            "third reference needs the published experience file"
        )
    n_states = int(np.asarray(data["county_state"]).max()) + 1
    trend = stratified_mortality_trend(experience, n_states)
    layer = AR.LayerParams(
        simulation=AR.SimulationParams(
            n_paths=params.simulation_paths, seed=params.seed + 1
        ),
        linkage_strategy="clerical_bootstrap",
        experience_share_strategy="cohort_component",
        regime_override={
            "mortality_drift": trend["mortality_drift"],
            "mortality_drift_se": trend["mortality_drift_se"],
        },
        mortality_improvement=trend,
        n_link_imputations=params.linkage_bootstraps,
        seed=params.seed + 2,
    )
    result = design_based.run(
        packet_dir,
        Path(out_dir),
        design_based.MethodParams(
            bootstrap_replicates=params.bootstrap_replicates,
            seed=params.seed,
            calibration_path=params.calibration_path,
            actuarial="on",
            actuarial_params=layer,
        ),
    )
    result["third_reference"] = {
        "linkage": "deterministic exact plus clerical comparison-field bootstrap",
        "elder_exposure": (
            "absolute experience cohort component over the publication lag, with "
            "linked-register and survey county reconciliation"
        ),
        "mortality": trend,
        "tail_calibrated_to_total": False,
    }
    return result
