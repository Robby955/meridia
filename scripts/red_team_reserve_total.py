"""Measure how much regional tail information the public reserve total carries.

The development worlds fit two deliberately small regressions, one for regional q95
and one for regional ES95.  Each regression has only an intercept and the public scalar
``contract["reserve"]["total"]``.  The qualification worlds are used once for predictive
R2.  A pooled eighteen-world fit and fits to the world-aggregate tails are descriptive
companions to that held-out result.

Only three packet artifacts are opened: ``participant/contract.json``,
``participant/experience_history.csv``, and
``retained/continuation_liabilities.npz``.  The script rejects any path whose components
contain ``graded``, ``sealed``, or ``hidden`` and never opens ``world.json``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from decimal import Decimal, ROUND_CEILING
from pathlib import Path
from typing import Any, Sequence

import numpy as np


FORBIDDEN_PATH_FRAGMENTS = ("graded", "sealed", "hidden")
EXPERIENCE_COLUMNS = (
    "year",
    "age_band",
    "sex",
    "state",
    "exposure",
    "deaths",
    "qualifying_events",
    "net_migration",
)
TAIL_LEVEL = 0.95
DEVELOPMENT_WORLD_NAMES = tuple(f"dev-{index:02d}" for index in range(12))
QUALIFICATION_WORLD_NAMES = tuple(f"qual-{index}" for index in range(6))


class MeasurementError(ValueError):
    """A fail-closed input or measurement error safe to show without a packet path."""


@dataclass(frozen=True)
class WorldMeasurement:
    name: str
    reserve_total: float
    latest_year_exposure: float
    q95: np.ndarray
    es95: np.ndarray


def _safe_path(path: Path) -> Path:
    """Return an absolute path after rejecting unsafe raw and resolved components."""
    candidate = Path(path).expanduser()
    raw_parts = candidate.absolute().parts
    resolved = candidate.resolve(strict=False)
    for part in (*raw_parts, *resolved.parts):
        lowered = part.casefold()
        if any(fragment in lowered for fragment in FORBIDDEN_PATH_FRAGMENTS):
            raise MeasurementError("a packet path contains a forbidden component")
    return resolved


def _required_file(world: Path, relative: str) -> Path:
    path = world / relative
    if path.is_symlink():
        raise MeasurementError(f"{world.name}: {relative} must not be a symbolic link")
    safe = _safe_path(path)
    try:
        safe.relative_to(world)
    except ValueError as exc:
        raise MeasurementError(f"{world.name}: {relative} leaves the world directory") from exc
    if not safe.is_file():
        raise MeasurementError(f"{world.name}: missing required {relative}")
    return safe


def _worlds(root: Path, names: Sequence[str], label: str) -> list[Path]:
    root = _safe_path(root)
    if root.is_symlink() or not root.is_dir():
        raise MeasurementError(f"the {label} root must be a real directory")
    expected = list(names)
    try:
        entries = list(root.iterdir())
    except OSError as exc:
        raise MeasurementError(f"the {label} root cannot be read") from exc
    if sorted(entry.name for entry in entries) != expected:
        raise MeasurementError(
            f"the {label} root must contain exactly {', '.join(expected)}"
        )
    worlds: list[Path] = []
    for name in expected:
        world = root / name
        if world.is_symlink() or not world.is_dir():
            raise MeasurementError(f"{name} must be a real directory")
        worlds.append(_safe_path(world))
    return worlds


def _json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in pairs:
        if key in out:
            raise MeasurementError("contract.json contains a duplicate key")
        out[key] = value
    return out


def _finite_number(value: Any, label: str, world: str, *, positive: bool = False) -> float:
    if isinstance(value, bool):
        raise MeasurementError(f"{world}: {label} must be numeric")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise MeasurementError(f"{world}: {label} must be numeric") from exc
    if not math.isfinite(result) or (positive and result <= 0.0):
        condition = "positive and finite" if positive else "finite"
        raise MeasurementError(f"{world}: {label} must be {condition}")
    return result


def _read_contract(world: Path) -> tuple[dict[str, Any], float, int]:
    path = _required_file(world, "participant/contract.json")
    try:
        payload = json.loads(path.read_text(), object_pairs_hook=_json_object)
    except MeasurementError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise MeasurementError(f"{world.name}: contract.json is not valid JSON") from exc
    if not isinstance(payload, dict):
        raise MeasurementError(f"{world.name}: contract.json must contain an object")
    reserve = payload.get("reserve")
    if not isinstance(reserve, dict) or "total" not in reserve:
        raise MeasurementError(f"{world.name}: contract reserve.total is missing")
    reserve_total = _finite_number(
        reserve["total"], "contract reserve.total", world.name, positive=True
    )
    n_states_value = payload.get("n_states")
    if isinstance(n_states_value, bool):
        raise MeasurementError(f"{world.name}: contract n_states must be a positive integer")
    try:
        n_states = int(n_states_value)
    except (TypeError, ValueError) as exc:
        raise MeasurementError(
            f"{world.name}: contract n_states must be a positive integer"
        ) from exc
    if n_states <= 0 or n_states_value != n_states:
        raise MeasurementError(f"{world.name}: contract n_states must be a positive integer")
    experience = payload.get("experience_history")
    if not isinstance(experience, dict):
        raise MeasurementError(f"{world.name}: contract experience_history is missing")
    if experience.get("file") != "experience_history.csv":
        raise MeasurementError(f"{world.name}: experience_history file declaration differs")
    if tuple(experience.get("columns", ())) != EXPERIENCE_COLUMNS:
        raise MeasurementError(f"{world.name}: experience_history columns differ")
    return payload, reserve_total, n_states


def _as_integer(value: str, label: str, world: str, *, nonnegative: bool = False) -> int:
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise MeasurementError(f"{world}: {label} must be an integer") from exc
    if not math.isfinite(numeric) or not numeric.is_integer():
        raise MeasurementError(f"{world}: {label} must be an integer")
    result = int(numeric)
    if nonnegative and result < 0:
        raise MeasurementError(f"{world}: {label} must be nonnegative")
    return result


def _latest_year_exposure(world: Path, n_states: int) -> tuple[int, float]:
    path = _required_file(world, "participant/experience_history.csv")
    observations: list[tuple[int, float]] = []
    try:
        with path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            if tuple(reader.fieldnames or ()) != EXPERIENCE_COLUMNS:
                raise MeasurementError(
                    f"{world.name}: experience_history.csv has the wrong columns"
                )
            for row in reader:
                year = _as_integer(row["year"], "experience year", world.name)
                if not row["age_band"] or not row["sex"]:
                    raise MeasurementError(
                        f"{world.name}: experience age_band and sex must be nonempty"
                    )
                state = _as_integer(
                    row["state"], "experience state", world.name, nonnegative=True
                )
                if state >= n_states:
                    raise MeasurementError(f"{world.name}: experience state is out of range")
                exposure = _finite_number(
                    row["exposure"], "experience exposure", world.name
                )
                if exposure < 0.0:
                    raise MeasurementError(
                        f"{world.name}: experience exposure must be nonnegative"
                    )
                for column in ("deaths", "qualifying_events"):
                    value = _finite_number(row[column], f"experience {column}", world.name)
                    if value < 0.0:
                        raise MeasurementError(
                            f"{world.name}: experience {column} must be nonnegative"
                        )
                _finite_number(
                    row["net_migration"], "experience net_migration", world.name
                )
                observations.append((year, exposure))
    except MeasurementError:
        raise
    except (OSError, UnicodeError, csv.Error) as exc:
        raise MeasurementError(
            f"{world.name}: experience_history.csv cannot be read"
        ) from exc
    if not observations:
        raise MeasurementError(f"{world.name}: experience_history.csv is empty")
    latest_year = max(year for year, _ in observations)
    total = float(sum(exposure for year, exposure in observations if year == latest_year))
    if not math.isfinite(total) or total <= 0.0:
        raise MeasurementError(f"{world.name}: latest-year exposure must be positive and finite")
    return latest_year, total


def _validate_total_rule(contract: dict[str, Any], reserve_total: float,
                         latest_year: int, exposure: float, world: str) -> None:
    reserve = contract.get("reserve")
    rule = reserve.get("total_rule") if isinstance(reserve, dict) else None
    if not isinstance(rule, dict):
        raise MeasurementError(f"{world}: contract reserve.total_rule is missing")
    exact_fields = {
        "file": "experience_history.csv",
        "year": "maximum published year",
        "year_column": "year",
        "exposure_column": "exposure",
        "rounding": "up",
    }
    for field, expected in exact_fields.items():
        if rule.get(field) != expected:
            raise MeasurementError(f"{world}: reserve total rule {field} differs")
    selected_year = _as_integer(
        rule.get("selected_year"), "reserve total rule selected_year", world
    )
    if selected_year != latest_year:
        raise MeasurementError(f"{world}: reserve total rule selected_year differs")
    declared_exposure = _finite_number(
        rule.get("exposure_person_years"),
        "reserve total rule exposure_person_years",
        world,
        positive=True,
    )
    if not math.isclose(declared_exposure, exposure, rel_tol=1e-12, abs_tol=1e-9):
        raise MeasurementError(f"{world}: reserve total rule exposure differs from the file")
    rate = _finite_number(
        rule.get("rate_per_person_year"),
        "reserve total rule rate_per_person_year",
        world,
        positive=True,
    )
    unit = _finite_number(
        rule.get("rounding_unit"), "reserve total rule rounding_unit", world,
        positive=True,
    )
    exposure_decimal = Decimal(str(exposure))
    rate_decimal = Decimal(str(rate))
    unit_decimal = Decimal(str(unit))
    recomputed = float(
        (exposure_decimal * rate_decimal / unit_decimal).to_integral_value(
            rounding=ROUND_CEILING
        ) * unit_decimal
    )
    if not math.isclose(reserve_total, recomputed, rel_tol=1e-12,
                        abs_tol=max(1e-9, unit * 1e-12)):
        raise MeasurementError(f"{world}: reserve.total does not follow its public rule")


def empirical_tail(liability: np.ndarray, level: float = TAIL_LEVEL) -> tuple[np.ndarray, np.ndarray]:
    """Return the ceiling-rank quantile and tied-inclusive expected shortfall by region."""
    values = np.asarray(liability, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
        raise MeasurementError("liability must be a nonempty members-by-regions matrix")
    if not np.isfinite(values).all() or (values < 0.0).any():
        raise MeasurementError("liability entries must be nonnegative and finite")
    if not math.isfinite(level) or not 0.0 < level <= 1.0:
        raise MeasurementError("tail level must lie in (0, 1]")
    rank = int(math.ceil(level * values.shape[0]))
    q95 = np.partition(values, rank - 1, axis=0)[rank - 1].astype(np.float64)
    es95 = np.empty(values.shape[1], dtype=np.float64)
    for region in range(values.shape[1]):
        tail = values[values[:, region] >= q95[region], region]
        if tail.size == 0:
            raise MeasurementError("the tied-inclusive tail is empty")
        es95[region] = float(tail.mean())
    if not np.isfinite(q95).all() or not np.isfinite(es95).all():
        raise MeasurementError("empirical tail calculation produced a nonfinite value")
    return q95, es95


def _liability_tail(world: Path, n_states: int) -> tuple[np.ndarray, np.ndarray]:
    path = _required_file(world, "retained/continuation_liabilities.npz")
    try:
        with np.load(path, allow_pickle=False) as archive:
            if "liability" not in archive.files:
                raise MeasurementError(f"{world.name}: liability archive lacks liability")
            liability = np.asarray(archive["liability"], dtype=np.float64)
    except MeasurementError:
        raise
    except (OSError, ValueError, EOFError) as exc:
        raise MeasurementError(f"{world.name}: liability archive cannot be read") from exc
    if liability.ndim != 2 or liability.shape[1] != n_states:
        raise MeasurementError(
            f"{world.name}: liability shape does not match contract n_states"
        )
    try:
        return empirical_tail(liability)
    except MeasurementError as exc:
        raise MeasurementError(f"{world.name}: {exc}") from exc


def read_world(world: Path) -> WorldMeasurement:
    """Read one world's three permitted artifacts and compute its two tail targets."""
    world = _safe_path(world)
    contract, reserve_total, n_states = _read_contract(world)
    latest_year, exposure = _latest_year_exposure(world, n_states)
    _validate_total_rule(contract, reserve_total, latest_year, exposure, world.name)
    q95, es95 = _liability_tail(world, n_states)
    return WorldMeasurement(world.name, reserve_total, exposure, q95, es95)


def _stack(worlds: Sequence[WorldMeasurement], outcome: str) -> np.ndarray:
    values = np.stack([getattr(world, outcome) for world in worlds])
    if values.ndim != 2 or not np.isfinite(values).all():
        raise MeasurementError(f"{outcome} targets cannot be stacked")
    return values


def _fit_line(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if x.shape != y.shape or len(x) < 2 or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise MeasurementError("regression inputs must be aligned, finite, and nonempty")
    design = np.column_stack((np.ones(len(x), dtype=np.float64), x))
    if np.linalg.matrix_rank(design) != 2:
        raise MeasurementError("reserve.total has no usable variation")
    coefficients, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
    if not np.isfinite(coefficients).all():
        raise MeasurementError("regression fit produced a nonfinite coefficient")
    return float(coefficients[0]), float(coefficients[1])


def _predict(model: tuple[float, float], x: np.ndarray) -> np.ndarray:
    return model[0] + model[1] * np.asarray(x, dtype=np.float64)


def _r2(truth: np.ndarray, prediction: np.ndarray) -> float:
    truth = np.asarray(truth, dtype=np.float64).reshape(-1)
    prediction = np.asarray(prediction, dtype=np.float64).reshape(-1)
    if truth.shape != prediction.shape or not np.isfinite(truth).all() \
            or not np.isfinite(prediction).all():
        raise MeasurementError("R2 inputs must be aligned and finite")
    denominator = float(np.square(truth - truth.mean()).sum())
    if denominator <= 0.0:
        raise MeasurementError("R2 is undefined because the target has no variance")
    numerator = float(np.square(truth - prediction).sum())
    value = 1.0 - numerator / denominator
    if not math.isfinite(value):
        raise MeasurementError("R2 calculation produced a nonfinite value")
    return float(value)


def _regional_fit(train: Sequence[WorldMeasurement], outcome: str) \
        -> tuple[tuple[float, float], ...]:
    targets = _stack(train, outcome)
    reserves = np.asarray([world.reserve_total for world in train], dtype=np.float64)
    return tuple(_fit_line(reserves, targets[:, region])
                 for region in range(targets.shape[1]))


def _regional_prediction(models: Sequence[tuple[float, float]],
                         worlds: Sequence[WorldMeasurement]) -> np.ndarray:
    reserves = np.asarray([world.reserve_total for world in worlds], dtype=np.float64)
    return np.column_stack([_predict(model, reserves) for model in models])


def _regional_r2(
    models: Sequence[tuple[float, float]], worlds: Sequence[WorldMeasurement], outcome: str
) -> float:
    targets = _stack(worlds, outcome)
    return _r2(targets, _regional_prediction(models, worlds))


def _regional_incremental_r2(models: Sequence[tuple[float, float]],
                             train: Sequence[WorldMeasurement],
                             test: Sequence[WorldMeasurement], outcome: str) -> float:
    """Predictive improvement over development region means, without pooling regions."""
    truth = _stack(test, outcome)
    prediction = _regional_prediction(models, test)
    baseline = np.broadcast_to(_stack(train, outcome).mean(axis=0), truth.shape)
    denominator = float(np.square(truth - baseline).sum())
    if denominator <= 0.0:
        raise MeasurementError("incremental R2 is undefined because the baseline is exact")
    return float(1.0 - np.square(truth - prediction).sum() / denominator)


def _per_region_r2(models: Sequence[tuple[float, float]],
                   worlds: Sequence[WorldMeasurement], outcome: str) -> list[float | None]:
    truth = _stack(worlds, outcome)
    prediction = _regional_prediction(models, worlds)
    result: list[float | None] = []
    for region in range(truth.shape[1]):
        try:
            result.append(_r2(truth[:, region], prediction[:, region]))
        except MeasurementError:
            result.append(None)
    return result


def _aggregate_fit(train: Sequence[WorldMeasurement], outcome: str) -> tuple[float, float]:
    reserves = np.asarray([world.reserve_total for world in train], dtype=np.float64)
    targets = _stack(train, outcome).sum(axis=1)
    return _fit_line(reserves, targets)


def _aggregate_r2(
    model: tuple[float, float], worlds: Sequence[WorldMeasurement], outcome: str
) -> float:
    reserves = np.asarray([world.reserve_total for world in worlds], dtype=np.float64)
    targets = _stack(worlds, outcome).sum(axis=1)
    return _r2(targets, _predict(model, reserves))


def _model_json(model: tuple[float, float]) -> dict[str, float]:
    return {"intercept": model[0], "reserve_total_coefficient": model[1]}


def _regional_models_json(models: Sequence[tuple[float, float]]) -> list[dict[str, float | int]]:
    return [dict(region=region, **_model_json(model))
            for region, model in enumerate(models)]


def _public_rows(worlds: Sequence[WorldMeasurement]) -> list[dict[str, float | str]]:
    return [
        {
            "world": world.name,
            "latest_year_total_exposure": world.latest_year_exposure,
            "reserve_total": world.reserve_total,
        }
        for world in worlds
    ]


def run_measurement(development_root: Path, qualification_root: Path) -> dict[str, Any]:
    """Run the preregistered development fit and qualification evaluation."""
    development = [read_world(path) for path in _worlds(
        development_root, DEVELOPMENT_WORLD_NAMES, "development")]
    qualification = [read_world(path) for path in _worlds(
        qualification_root, QUALIFICATION_WORLD_NAMES, "qualification")]
    region_counts = {len(world.q95) for world in (*development, *qualification)}
    if len(region_counts) != 1:
        raise MeasurementError("all worlds must have the same number of regions")

    dev_models = {
        outcome: _regional_fit(development, outcome) for outcome in ("q95", "es95")
    }
    qualification_r2 = {
        outcome: _regional_r2(dev_models[outcome], qualification, outcome)
        for outcome in ("q95", "es95")
    }
    qualification_incremental_r2 = {
        outcome: _regional_incremental_r2(
            dev_models[outcome], development, qualification, outcome)
        for outcome in ("q95", "es95")
    }
    qualification_per_region_r2 = {
        outcome: _per_region_r2(dev_models[outcome], qualification, outcome)
        for outcome in ("q95", "es95")
    }

    all_worlds = [*development, *qualification]
    pooled_models = {
        outcome: _regional_fit(all_worlds, outcome) for outcome in ("q95", "es95")
    }
    pooled_r2 = {
        outcome: _regional_r2(pooled_models[outcome], all_worlds, outcome)
        for outcome in ("q95", "es95")
    }

    aggregate_dev_models = {
        outcome: _aggregate_fit(development, outcome) for outcome in ("q95", "es95")
    }
    aggregate_pooled_models = {
        outcome: _aggregate_fit(all_worlds, outcome) for outcome in ("q95", "es95")
    }
    aggregate_qualification_r2 = {
        outcome: _aggregate_r2(aggregate_dev_models[outcome], qualification, outcome)
        for outcome in ("q95", "es95")
    }
    aggregate_pooled_r2 = {
        outcome: _aggregate_r2(aggregate_pooled_models[outcome], all_worlds, outcome)
        for outcome in ("q95", "es95")
    }

    return {
        "schema": "meridia.reserve-total-red-team.v1",
        "independent_unit": "world",
        "world_counts": {"development": 12, "qualification": 6, "total": 18},
        "regions_per_world": region_counts.pop(),
        "files_read_per_world": [
            "participant/contract.json",
            "participant/experience_history.csv",
            "retained/continuation_liabilities.npz:liability",
        ],
        "reserve_total_public_rule_verified": True,
        "tail_definition": {
            "level": TAIL_LEVEL,
            "quantile_rank": "ceil(level * members), one-indexed",
            "expected_shortfall": "mean of all members at or above the quantile, ties included",
        },
        "public_quantities": {
            "development": _public_rows(development),
            "qualification": _public_rows(qualification),
        },
        "development_regional_models": {
            outcome: _regional_models_json(models)
            for outcome, models in dev_models.items()
        },
        "qualification_predictive_regional_r2": {
            **qualification_r2,
            "per_region": qualification_per_region_r2,
        },
        "qualification_incremental_regional_r2_over_region_means": {
            **qualification_incremental_r2,
            "headline_max": max(qualification_incremental_r2.values()),
        },
        "primary_measure": (
            "qualification incremental regional R2 over development region means"
        ),
        "descriptive_pooled_regional_r2": {
            **pooled_r2,
            "headline_max": max(pooled_r2.values()),
            "models": {
                outcome: _regional_models_json(models)
                for outcome, models in pooled_models.items()
            },
        },
        "world_aggregate_tail_r2": {
            "qualification_predictive": {
                **aggregate_qualification_r2,
                "headline_max": max(aggregate_qualification_r2.values()),
            },
            "descriptive_pooled": {
                **aggregate_pooled_r2,
                "headline_max": max(aggregate_pooled_r2.values()),
            },
        },
        "interpretation": (
            "The headline is held-out incremental regional R2 over development region "
            "means. It asks whether reserve.total adds predictive information within a "
            "region after its ordinary level is known. Raw regional R2 is diagnostic "
            "because fixed region effects can otherwise dominate it. "
            "The reserve total is a checked deterministic function of already published "
            "exposure, so it supplies no world-specific information conditional on that "
            "exposure. Pooled and aggregate values are descriptive. Worlds, not regions "
            "or continuation members, are the independent units. No p-values are computed."
        ),
    }


def _human_text(result: dict[str, Any]) -> str:
    predictive = result["qualification_predictive_regional_r2"]
    pooled = result["descriptive_pooled_regional_r2"]
    incremental = result["qualification_incremental_regional_r2_over_region_means"]
    aggregate = result["world_aggregate_tail_r2"]
    return "\n".join(
        (
            "Reserve-total red-team measurement",
            "Independent worlds: 12 development, 6 qualification",
            "Headline qualification incremental regional R2 over region means: "
            f"q95 {incremental['q95']:.6f}, ES95 {incremental['es95']:.6f}, "
            f"headline max {incremental['headline_max']:.6f}",
            "Diagnostic qualification raw regional R2: "
            f"q95 {predictive['q95']:.6f}, ES95 {predictive['es95']:.6f}",
            "Descriptive pooled regional R2: "
            f"q95 {pooled['q95']:.6f}, ES95 {pooled['es95']:.6f}",
            "Qualification world-aggregate predictive R2: "
            f"q95 {aggregate['qualification_predictive']['q95']:.6f}, "
            f"ES95 {aggregate['qualification_predictive']['es95']:.6f}",
            "Descriptive pooled world-aggregate R2: "
            f"q95 {aggregate['descriptive_pooled']['q95']:.6f}, "
            f"ES95 {aggregate['descriptive_pooled']['es95']:.6f}",
            "No p-values. The independent unit is the world.",
        )
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development-root", type=Path, required=True)
    parser.add_argument("--qualification-root", type=Path, required=True)
    parser.add_argument(
        "--human", action="store_true", help="print a concise report instead of JSON"
    )
    args = parser.parse_args(argv)
    try:
        result = run_measurement(args.development_root, args.qualification_root)
    except MeasurementError as exc:
        parser.error(str(exc))
    if args.human:
        print(_human_text(result))
    else:
        print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
