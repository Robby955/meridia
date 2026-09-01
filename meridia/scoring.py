"""Scoring a published estimate table against exact truth.

Sources: interval score after Gneiting and Raftery (2007); disclosure by linear
recovery after Cox (1980) and Hundepool et al. (2012). See docs/INDEPENDENCE.md.

Everything a release is judged on lives here: the schema check (every required row once,
intervals well formed), additivity of counts across the geography, accuracy judged on
the worst unit rather than the average, interval coverage together with a proper
interval score so wide intervals cannot buy safety, and a disclosure audit that treats a
protected cell as compromised when the published numbers determine it, by any linear
combination of published totals, not only by a single subtraction.

Bars are supplied, not invented here: they are frozen from executed strong pipelines and
passed in as a dictionary. With no bars the functions report metrics only.
"""

from __future__ import annotations

import math

import numpy as np

from .release import (AGE_BANDS, ESTIMAND_BY_ID, ESTIMAND_IDS, LEVELS, RELEASE_COLUMNS,
                      required_rows)

NOMINAL_ALPHA = 0.10          # released intervals are 90 percent intervals
ADDITIVITY_TOLERANCE = 1e-6   # relative, for published counts


# ---------------------------------------------------------------- schema and additivity

def validate_release(rows: list[dict], admin: dict) -> list[str]:
    """Return a list of schema violations; an empty list means the schema is exact."""
    errors: list[str] = []
    seen: dict[tuple[str, str, int], int] = {}
    for i, row in enumerate(rows):
        if set(row) != set(RELEASE_COLUMNS):
            errors.append(f"row {i}: columns {sorted(row)} differ from {list(RELEASE_COLUMNS)}")
            continue
        est_id, level, unit = row["estimand"], row["level"], row["unit"]
        if est_id not in ESTIMAND_BY_ID:
            errors.append(f"row {i}: unknown estimand {est_id!r}")
            continue
        if level not in LEVELS:
            errors.append(f"row {i}: unknown level {level!r}")
            continue
        if not isinstance(unit, (int, np.integer)) or isinstance(unit, bool):
            errors.append(f"row {i}: unit must be an integer")
            continue
        key = (est_id, level, int(unit))
        seen[key] = seen.get(key, 0) + 1
        values = []
        for column in ("estimate", "lower", "upper"):
            v = row[column]
            if isinstance(v, bool) or not isinstance(v, (int, float, np.integer, np.floating)) \
                    or not math.isfinite(float(v)):
                errors.append(f"row {i}: {column} is not a finite number")
                break
            values.append(float(v))
        if len(values) < 3:
            continue
        estimate, lower, upper = values
        if not lower <= estimate <= upper:
            errors.append(f"row {i}: interval does not contain the estimate")
        kind = ESTIMAND_BY_ID[est_id].kind
        if kind in ("count", "mean", "median") and lower < 0:
            errors.append(f"row {i}: negative lower bound for a {kind}")
        if kind == "proportion" and not (0.0 <= lower and upper <= 1.0):
            errors.append(f"row {i}: proportion interval outside [0, 1]")
    required = required_rows(admin)
    for key, n in seen.items():
        if n > 1:
            errors.append(f"duplicate row {key} ({n} copies)")
        if key not in required:
            errors.append(f"unexpected row {key}")
    for key in sorted(required - set(seen)):
        errors.append(f"missing row {key}")
    return errors


def _estimates(rows: list[dict]) -> dict[tuple[str, str, int], tuple[float, float, float]]:
    return {(r["estimand"], r["level"], int(r["unit"])):
            (float(r["estimate"]), float(r["lower"]), float(r["upper"])) for r in rows}


def check_additivity(rows: list[dict], admin: dict) -> list[str]:
    """Published counts must add up: counties to their state, states to the nation."""
    est = _estimates(rows)
    errors: list[str] = []
    county_state = admin["county_state"]
    for e in ESTIMAND_IDS:
        if not ESTIMAND_BY_ID[e].additive:
            continue
        for s in range(admin["n_states"]):
            members = np.flatnonzero(county_state == s)
            total = sum(est[(e, "county", int(c))][0] for c in members)
            stated = est[(e, "state", s)][0]
            if abs(total - stated) > ADDITIVITY_TOLERANCE * max(1.0, abs(stated)):
                errors.append(f"{e}: counties of state {s} sum to {total}, state says {stated}")
        total = sum(est[(e, "state", s)][0] for s in range(admin["n_states"]))
        stated = est[(e, "nation", 0)][0]
        if abs(total - stated) > ADDITIVITY_TOLERANCE * max(1.0, abs(stated)):
            errors.append(f"{e}: states sum to {total}, nation says {stated}")
    return errors


# ------------------------------------------------------------------ accuracy and coverage

def _scaled_error(kind: str, estimate: float, truth: float) -> float:
    if kind == "proportion":
        return abs(estimate - truth)
    return abs(estimate - truth) / max(abs(truth), 1.0)


def _interval_score(kind: str, lower: float, upper: float, truth: float, alpha: float) -> float:
    """Gneiting-Raftery interval score on the same scale as the error."""
    score = (upper - lower) + (2.0 / alpha) * max(lower - truth, 0.0) \
        + (2.0 / alpha) * max(truth - upper, 0.0)
    if kind == "proportion":
        return score
    return score / max(abs(truth), 1.0)


def score_release(rows: list[dict], truth: dict, admin: dict,
                  alpha: float = NOMINAL_ALPHA) -> dict:
    """Metrics per (estimand, level): worst and mean error, coverage, mean interval score.

    Units whose truth is undefined (no members) are skipped. Missing rows are reported by
    ``validate_release``; here they simply do not contribute.
    """
    est = _estimates(rows)
    metrics: dict[str, dict] = {}
    for e in ESTIMAND_IDS:
        kind = ESTIMAND_BY_ID[e].kind
        pooled_errors, pooled_covered, pooled_scores = [], [], []
        for level in LEVELS:
            errors, covered, scores, worst_unit = [], [], [], -1
            for (te, tl, u), t in truth.items():
                if te != e or tl != level or not math.isfinite(t) or (e, level, u) not in est:
                    continue
                estimate, lower, upper = est[(e, level, u)]
                err = _scaled_error(kind, estimate, t)
                if not errors or err > max(errors):
                    worst_unit = u
                errors.append(err)
                covered.append(lower <= t <= upper)
                scores.append(_interval_score(kind, lower, upper, t, alpha))
            if not errors:
                continue
            pooled_errors += errors
            pooled_covered += covered
            pooled_scores += scores
            metrics[f"{e}/{level}"] = {
                "n_units": len(errors),
                "worst_error": float(max(errors)),
                "worst_unit": int(worst_unit),
                "mean_error": float(np.mean(errors)),
                "coverage": float(np.mean(covered)),
                "mean_interval_score": float(np.mean(scores)),
            }
        if pooled_errors:
            # Coverage is only meaningful pooled over enough units: one nation and a
            # handful of states cannot show a rate. The coverage gate binds here.
            metrics[f"{e}/all"] = {
                "n_units": len(pooled_errors),
                "worst_error": float(max(pooled_errors)),
                "worst_unit": -1,
                "mean_error": float(np.mean(pooled_errors)),
                "coverage": float(np.mean(pooled_covered)),
                "mean_interval_score": float(np.mean(pooled_scores)),
            }
    return metrics


# ------------------------------------------------------------------------- disclosure

def _lines(shape: tuple[int, ...]):
    """Every one-dimensional line of a cube: (axis, fixed index tuple) pairs."""
    for axis in range(len(shape)):
        others = [range(n) for k, n in enumerate(shape) if k != axis]
        for fixed in np.ndindex(*[len(r) for r in others]):
            yield axis, fixed


def disclosure_audit(published: np.ndarray, truth_table: np.ndarray, threshold: int,
                     marginals: dict[str, np.ndarray] | None = None,
                     tolerance: float = 1e-6) -> dict:
    """Audit a published county x age-band x sex table against the retained truth.

    ``published`` is the cube of published estimates with NaN where a cell is suppressed.
    ``marginals`` may carry published totals: ``county_age`` (C, A), ``county_sex`` (C, 2),
    ``county`` (C,), ``age_sex`` (A, 2), each NaN where suppressed. A protected cell is a
    true count strictly between zero and ``threshold``. The audit fails when a protected
    cell is published, or when the published cells and totals determine a suppressed
    protected cell exactly (its indicator lies in the row space of the published linear
    constraints), or when published cells and totals are not internally consistent.
    """
    published = np.asarray(published, dtype=np.float64)
    truth_table = np.asarray(truth_table)
    if published.shape != truth_table.shape:
        raise ValueError("published table shape differs from the truth table")
    marginals = marginals or {}
    protected = (truth_table > 0) & (truth_table < threshold)
    suppressed = np.isnan(published)
    findings = {"published_protected": [], "recoverable": [], "inconsistent": []}
    for idx in zip(*np.nonzero(protected & ~suppressed)):
        findings["published_protected"].append(tuple(int(i) for i in idx))

    # Linear constraints from every published total over the suppressed cells.
    suppressed_index = {tuple(int(i) for i in idx): k
                        for k, idx in enumerate(zip(*np.nonzero(suppressed)))}
    n_unknown = len(suppressed_index)
    equations: list[np.ndarray] = []

    def add_constraint(cells: list[tuple[int, ...]], total: float) -> None:
        row = np.zeros(n_unknown)
        known = 0.0
        for cell in cells:
            if cell in suppressed_index:
                row[suppressed_index[cell]] = 1.0
            else:
                known += float(published[cell])
        if row.any():
            equations.append(row)
        elif abs(known - total) > tolerance * max(1.0, abs(total)):
            findings["inconsistent"].append((cells[0], known, total))

    shape = published.shape
    named = {"county_age": (2,), "county_sex": (1,), "county": (1, 2), "age_sex": (0,)}
    for name, summed_axes in named.items():
        if name not in marginals:
            continue
        table = np.asarray(marginals[name], dtype=np.float64)
        kept_axes = [a for a in range(3) if a not in summed_axes]
        if table.shape != tuple(shape[a] for a in kept_axes):
            raise ValueError(f"marginal {name} has the wrong shape")
        for fixed in np.ndindex(*table.shape):
            total = float(table[fixed])
            if math.isnan(total):
                continue
            cells = []
            for free in np.ndindex(*[shape[a] for a in summed_axes]):
                idx = [0, 0, 0]
                for a, v in zip(kept_axes, fixed):
                    idx[a] = int(v)
                for a, v in zip(summed_axes, free):
                    idx[a] = int(v)
                cells.append(tuple(idx))
            add_constraint(cells, total)

    if n_unknown and equations:
        a = np.vstack(equations)
        rank = np.linalg.matrix_rank(a)
        for cell, k in suppressed_index.items():
            if not protected[cell]:
                continue
            unit = np.zeros((1, n_unknown))
            unit[0, k] = 1.0
            if np.linalg.matrix_rank(np.vstack([a, unit])) == rank:
                findings["recoverable"].append(cell)
    findings["pass"] = not (findings["published_protected"] or findings["recoverable"]
                            or findings["inconsistent"])
    findings["n_protected"] = int(protected.sum())
    findings["n_suppressed"] = int(suppressed.sum())
    return findings


# ------------------------------------------------------------------------------ gates

def evaluate_gates(schema_errors: list[str], additivity_errors: list[str], metrics: dict,
                   disclosure: dict | None, bars: dict | None) -> dict:
    """Combine the checks into a pass verdict with named reasons.

    ``bars`` keys: ``worst_error`` and ``interval_score_ceiling`` map "estimand/level" to a
    ceiling; ``coverage_floor`` is one number. A metric with no bar is reported, not gated.
    """
    reasons: list[str] = []
    if schema_errors:
        reasons.append(f"schema: {len(schema_errors)} violation(s)")
    if additivity_errors:
        reasons.append(f"additivity: {len(additivity_errors)} violation(s)")
    if disclosure is not None and not disclosure["pass"]:
        reasons.append("disclosure: protected cell published, recoverable, or inconsistent")
    bars = bars or {}
    for key, m in metrics.items():
        ceiling = bars.get("worst_error", {}).get(key)
        if ceiling is not None and m["worst_error"] > ceiling:
            reasons.append(f"accuracy: {key} worst error {m['worst_error']:.4f} > {ceiling}")
        floor = bars.get("coverage_floor")
        if floor is not None and key.endswith("/all") and m["coverage"] < floor:
            reasons.append(f"coverage: {key} {m['coverage']:.3f} < {floor}")
        score_ceiling = bars.get("interval_score_ceiling", {}).get(key)
        if score_ceiling is not None and m["mean_interval_score"] > score_ceiling:
            reasons.append(f"interval score: {key} {m['mean_interval_score']:.4f} > {score_ceiling}")
    return {"pass": not reasons, "reasons": reasons}


def rows_from_values(values: dict, half_width) -> list[dict]:
    """Release rows from a value map, with intervals of a given half width.

    ``half_width`` is a number or a function of (estimand_id, value). Undefined values
    (NaN, empty units) are published as zero with a zero-width interval so the schema
    stays complete; such units are never scored.
    """
    rows = []
    for (e, level, u), v in sorted(values.items()):
        kind = ESTIMAND_BY_ID[e].kind
        if not math.isfinite(v):
            rows.append({"estimand": e, "level": level, "unit": u,
                         "estimate": 0.0, "lower": 0.0, "upper": 0.0})
            continue
        h = half_width(e, v) if callable(half_width) else float(half_width)
        lower, upper = v - h, v + h
        if kind != "proportion":
            lower = max(lower, 0.0)
        else:
            lower, upper = max(lower, 0.0), min(upper, 1.0)
        rows.append({"estimand": e, "level": level, "unit": u,
                     "estimate": float(v), "lower": float(lower), "upper": float(upper)})
    return rows


def age_band_count() -> int:
    return len(AGE_BANDS)
