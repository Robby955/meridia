"""Every hidden mechanism against the supplied anchor that estimates it.

Protocol proof obligation 5 asks that the hidden regime stay identifiable from observable
anchors, so that a frontier failure is never a failure to know the unknowable. This
script answers it as a measurement rather than a claim: for each of the six axes of the
regime family it computes one statistic from the participant files alone, then reports
that statistic's rank correlation with the realized mechanism coefficient across the
committed generator-only worlds.  The receipt separately records the raw axis intensity
that the hidden-world policy constrains.

One statistic per axis, all from files the agent receives:

- mortality_improvement       count-weighted log mortality drift within cells of
                              experience_history.csv
- migration_age_pattern       urban pull of net internal migration at 18 to 44 minus the
                              same pull at 65 and over
- age_reporting_error         share of reported birth ticks off a year boundary
- linkage_urban_gradient      slope of the missing-name share on the county's urbanity rank
- administrative_completeness slope of the register's shortfall against the benchmark's
                              subgroup count, over the published economic bands
- missingness_target_dependence  log gap between the health archive's admission rate and
                              the anchor's, corrected for its declared error

Run it on development and qualification worlds. Graded worlds are not opened.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from meridia.mechanisms import (
    DEVELOPMENT_BAND,
    DEVELOPMENT_DESIGN,
    HIDDEN_EXTRAPOLATION_AXES,
    HIDDEN_IN_BAND_AXES,
    N_HIDDEN_OUTSIDE_AXES,
    PUBLIC_ENVELOPE,
)
from scripts import build_v4_worlds as builder

AXES = ("mortality_improvement", "migration_age_pattern", "age_reporting_error",
        "linkage_urban_gradient", "administrative_completeness",
        "missingness_target_dependence")
STATISTIC = {"mortality_improvement": "experience file mortality drift within cells",
             "migration_age_pattern": "urban pull of net migration, young minus old",
             "age_reporting_error": "birth ticks off a year boundary",
             "linkage_urban_gradient": "missing-name share against urbanity",
             "administrative_completeness":
                 "register against the benchmark subgroup count, over economic bands",
             "missingness_target_dependence":
                 "archive against anchor admission rate"}
# The sign each mechanism implies, read off the family and not off the data. The health
# archive observes the included, and inclusion rises with latent burden, so a stronger
# dependence on frailty raises the archive's admission rate above the population rate the
# anchor estimates. The axis carries that one mechanism and nothing else: item
# missingness on money carries its own published slope, because one coefficient loading
# two mechanisms leaves the statistic's sign reversing between regimes.
EXPECTED_SIGN = {"mortality_improvement": +1, "migration_age_pattern": +1,
                 "age_reporting_error": +1, "linkage_urban_gradient": -1,
                 "administrative_completeness": +1,
                 "missingness_target_dependence": +1}
ANCHOR_CORRELATION_THRESHOLD = 0.4
RECEIPT_SCHEMA = "meridia.v4.regime-identifiability-audit.v3"
PACKET_MANIFEST_SCHEMA = "meridia.packet.manifest.v1"
REFERENCE_MORTALITY_AGE_SLOPE = 0.105
REGISTERED_GOMPERTZ_B_RANGE = (0.092, 0.121)
REGISTERED_INTERACTION_RANGES = {
    "linkage_gradient_by_migration": (0.15, 0.75),
    "health_inclusion_completeness_by_target": (0.25, 0.90),
    "age_error_by_mortality_slope": (0.30, 1.20),
}

EXPECTED_WORLD_REGIMES = {
    **{f"dev-{index:02d}": "development" for index in range(12)},
    **{f"qual-{index}": "hidden" for index in range(6)},
}
EXPECTED_WORLD_PACKET_CLASSES = {
    world: ("development" if regime == "development" else "qualification")
    for world, regime in EXPECTED_WORLD_REGIMES.items()
}
EXPECTED_DEVELOPMENT_WORLD_SEEDS = {
    f"dev-{index:02d}": 1101 + index for index in range(12)
}


def expected_world_seed(name: str) -> int:
    """The registered seed of one audited world.

    Development seeds are committed, so they are registered here and a builder change
    that moved them would fail this audit rather than pass quietly. Qualification seeds
    are sealed outside the repository, so the qualification half is read from the same
    file the builder reads, through the builder, at the point the audit needs a value
    rather than at import. No caller of this function puts its result in a message.
    """
    if name in EXPECTED_DEVELOPMENT_WORLD_SEEDS:
        return EXPECTED_DEVELOPMENT_WORLD_SEEDS[name]
    if name not in EXPECTED_WORLD_REGIMES:
        raise KeyError(name)
    return builder.qualification_seeds()[int(name.rsplit("-", 1)[1])]

# These envelopes are registered independently of the generator's coefficient table.
# The focused tests derive the same extrema from the published raw-axis and interaction
# ranges.  Keeping literal values here makes a generator-law change fail review instead
# of silently changing the meaning of an already registered receipt.
REGISTERED_REALIZED_MECHANISM_ENVELOPES = {
    "mortality_improvement": {
        "development": (-0.010, 0.048),
        "public": (-0.030, 0.075),
    },
    "migration_age_pattern": {
        "development": (0.25, 1.55),
        "public": (0.00, 2.40),
    },
    "age_reporting_error": {
        "development": (0.596, 2.4248571428571424),
        "public": (0.298, 4.021714285714285),
    },
    "linkage_urban_gradient": {
        "development": (0.13125, 2.189375),
        "public": (0.0, 5.33),
    },
    "administrative_completeness": {
        "development": (0.30, 1.70),
        "public": (0.00, 2.80),
    },
    "missingness_target_dependence": {
        "development": (0.074, 2.119),
        "public": (0.0, 5.764),
    },
}

REALIZED_MECHANISM_DEFINITIONS = {
    axis: "axis_intensity" for axis in AXES
}
REALIZED_MECHANISM_DEFINITIONS.update({
    "age_reporting_error": (
        "age_reporting_error * age_error_mortality_scale"
    ),
    "linkage_urban_gradient": (
        "linkage_urban_gradient * (1 + linkage_gradient_by_migration * "
        "(migration_age_pattern - 1))"
    ),
    "missingness_target_dependence": (
        "missingness_target_dependence * "
        "(1 + health_inclusion_completeness_by_target * "
        "(administrative_completeness - 1))"
    ),
})


def _canonical_digest(value: object) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _participant_digest(packet: Path) -> str:
    participant = packet / "participant"
    manifest = json.loads((packet / "manifest.json").read_text())
    expected = manifest["participant"]
    records = []
    for path in sorted(participant.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"linked participant path is not valid evidence: {path}")
        if path.is_file():
            records.append({
                "path": str(path.relative_to(participant)),
                "sha256": _file_digest(path),
                "bytes": path.stat().st_size,
            })
    if not records:
        raise ValueError(f"participant packet has no files: {packet}")
    actual = {
        record["path"]: {
            "sha256": record["sha256"],
            "bytes": record["bytes"],
        }
        for record in records
    }
    if actual != expected:
        raise ValueError(f"participant inventory differs from packet manifest: {packet}")
    return _canonical_digest(records)


def _finite_number(value: object, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a finite number")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a finite number") from exc
    if not math.isfinite(number):
        raise ValueError(f"{label} must be a finite number")
    return number


def _realized_mechanisms(
    intensity: Mapping[str, object],
    coefficients: Mapping[str, object],
    character: Mapping[str, object],
) -> dict[str, float]:
    """Return the coefficients the measured mechanisms actually consume.

    Raw axis intensity is the hidden-world policy variable.  Two participant-file
    traces, however, observe predeclared interacted coefficients.  They must be kept
    distinct: using an interacted coefficient to audit the raw policy can falsely call
    an in-band world an extrapolation.
    """
    if set(intensity) != set(AXES):
        raise ValueError("mechanism design must contain exactly the six registered axes")
    required = set(AXES) | set(REGISTERED_INTERACTION_RANGES) | {
        "age_error_mortality_scale",
    }
    missing = sorted(required - set(coefficients))
    if missing:
        raise ValueError(
            "mechanism coefficient record is missing required values: "
            + ", ".join(missing)
        )
    raw = {
        axis: _finite_number(intensity[axis], f"{axis} raw intensity")
        for axis in AXES
    }
    for axis in AXES:
        duplicated = _finite_number(coefficients[axis], f"{axis} coefficient")
        if duplicated != raw[axis]:
            raise ValueError(f"{axis} coefficient differs from the design intensity")
    interactions = {
        name: _finite_number(coefficients[name], f"{name} coefficient")
        for name in REGISTERED_INTERACTION_RANGES
    }
    for name, value in interactions.items():
        low, high = REGISTERED_INTERACTION_RANGES[name]
        if not low <= value <= high:
            raise ValueError(f"{name} coefficient lies outside its registered range")
    gompertz_b = _finite_number(character.get("gompertz_b"), "gompertz_b")
    if not REGISTERED_GOMPERTZ_B_RANGE[0] <= gompertz_b \
            <= REGISTERED_GOMPERTZ_B_RANGE[1]:
        raise ValueError("gompertz_b lies outside its registered range")
    expected_age_scale = 1.0 + interactions["age_error_by_mortality_slope"] * (
        gompertz_b / REFERENCE_MORTALITY_AGE_SLOPE - 1.0
    )
    age_scale = _finite_number(
        coefficients["age_error_mortality_scale"],
        "age_error_mortality_scale coefficient",
    )
    if not math.isclose(age_scale, expected_age_scale, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError(
            "age_error_mortality_scale differs from its registered interaction"
        )
    realized = dict(raw)
    realized["age_reporting_error"] = raw["age_reporting_error"] * age_scale
    realized["missingness_target_dependence"] = (
        raw["missingness_target_dependence"]
        * (
            1.0
            + interactions["health_inclusion_completeness_by_target"]
            * (raw["administrative_completeness"] - 1.0)
        )
    )
    realized["linkage_urban_gradient"] = (
        raw["linkage_urban_gradient"]
        * (
            1.0
            + interactions["linkage_gradient_by_migration"]
            * (raw["migration_age_pattern"] - 1.0)
        )
    )
    return realized


def _finite_range(values: object, label: str) -> list[float]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0 or not np.isfinite(array).all():
        raise ValueError(f"{label} must contain finite values")
    return [float(array.min()), float(array.max())]


def _observed_ranges(frame: pd.DataFrame, column: str) -> dict[str, list[float]]:
    return {
        "pooled": _finite_range(frame[column].to_numpy(), f"{column} pooled"),
        "development": _finite_range(
            frame.loc[frame["regime"] == "development", column].to_numpy(),
            f"{column} development",
        ),
        "hidden": _finite_range(
            frame.loc[frame["regime"] == "hidden", column].to_numpy(),
            f"{column} hidden",
        ),
    }


def _range_is_inside(observed: list[float], envelope: tuple[float, float]) -> bool:
    tolerance = 1e-12
    return (
        envelope[0] - tolerance <= observed[0] <= observed[1]
        <= envelope[1] + tolerance
    )


def _validate_axis_ranges(
    axis: str,
    axis_intensity_ranges: Mapping[str, list[float]],
    realized_mechanism_ranges: Mapping[str, list[float]],
) -> None:
    """Validate raw policy scope and interacted measurement scope separately."""
    constrained = axis in HIDDEN_IN_BAND_AXES
    raw_hidden_envelope = (
        DEVELOPMENT_BAND[axis] if constrained else PUBLIC_ENVELOPE[axis]
    )
    realized_envelopes = REGISTERED_REALIZED_MECHANISM_ENVELOPES[axis]
    realized_hidden_envelope = (
        realized_envelopes["development"]
        if constrained
        else realized_envelopes["public"]
    )
    checks = (
        (
            axis_intensity_ranges["development"],
            DEVELOPMENT_BAND[axis],
            "development raw axis intensity",
        ),
        (
            axis_intensity_ranges["hidden"],
            raw_hidden_envelope,
            "hidden raw axis intensity",
        ),
        (
            axis_intensity_ranges["pooled"],
            PUBLIC_ENVELOPE[axis],
            "pooled raw axis intensity",
        ),
        (
            realized_mechanism_ranges["development"],
            realized_envelopes["development"],
            "development realized mechanism",
        ),
        (
            realized_mechanism_ranges["hidden"],
            realized_hidden_envelope,
            "hidden realized mechanism",
        ),
        (
            realized_mechanism_ranges["pooled"],
            realized_envelopes["public"],
            "pooled realized mechanism",
        ),
    )
    for observed, envelope, label in checks:
        if not _range_is_inside(observed, envelope):
            raise ValueError(
                f"{axis}: {label} {observed} lies outside registered envelope "
                f"{list(envelope)}"
            )


def _axis_range_record(frame: pd.DataFrame, axis: str) -> dict[str, object]:
    axis_intensity_ranges = _observed_ranges(
        frame, f"axis_intensity_{axis}"
    )
    realized_mechanism_ranges = _observed_ranges(
        frame, f"realized_mechanism_{axis}"
    )
    _validate_axis_ranges(
        axis,
        axis_intensity_ranges,
        realized_mechanism_ranges,
    )
    return {
        "correlation_target": "realized_mechanism",
        "realized_mechanism_definition": REALIZED_MECHANISM_DEFINITIONS[axis],
        "axis_intensity_range_observed": axis_intensity_ranges,
        "realized_mechanism_range_observed": realized_mechanism_ranges,
        "registered_realized_mechanism_envelopes": {
            family: list(bounds)
            for family, bounds in
            REGISTERED_REALIZED_MECHANISM_ENVELOPES[axis].items()
        },
    }


def _validate_world_family(frame: pd.DataFrame) -> None:
    """Require the registered twelve-development, six-qualification audit."""
    if len(frame) != len(EXPECTED_WORLD_REGIMES) or frame["world"].duplicated().any():
        raise ValueError(
            "identifiability receipt requires twelve distinct development and six "
            "qualification worlds"
        )
    observed = dict(zip(frame["world"], frame["regime"], strict=True))
    if observed != EXPECTED_WORLD_REGIMES:
        raise ValueError(
            "identifiability world names or regimes differ from the registered 12+6 audit"
        )


def _preflight_packets(paths: list[str]) -> list[Path]:
    """Resolve the registered packet family before opening any retained truth."""
    if len(paths) != len(EXPECTED_WORLD_REGIMES):
        raise ValueError(
            "identifiability receipt requires exactly twelve development and six "
            "qualification packets"
        )
    unresolved = [Path(path) for path in paths]
    names = [path.name for path in unresolved]
    if len(set(names)) != len(names) or set(names) != set(EXPECTED_WORLD_REGIMES):
        raise ValueError(
            "identifiability packet names differ from the registered 12+6 audit"
        )
    resolved_by_name: dict[str, Path] = {}
    for packet in unresolved:
        if packet.is_symlink():
            raise ValueError(f"linked packet directory is not valid evidence: {packet}")
        resolved = packet.resolve(strict=True)
        if not resolved.is_dir():
            raise ValueError(f"identifiability packet is not a directory: {packet}")
        if resolved in resolved_by_name.values():
            raise ValueError("identifiability packets must resolve to distinct directories")
        manifest_path = resolved / "manifest.json"
        if manifest_path.is_symlink() or not manifest_path.is_file():
            raise ValueError(f"{packet.name}: packet manifest is missing or linked")
        manifest = json.loads(manifest_path.read_text())
        expected_development = packet.name.startswith("dev-")
        if not isinstance(manifest, Mapping) \
                or set(manifest) != {
                    "schema", "development", "packet_class", "participant", "retained"
                } \
                or manifest.get("schema") != PACKET_MANIFEST_SCHEMA \
                or manifest.get("packet_class") \
                != EXPECTED_WORLD_PACKET_CLASSES[packet.name] \
                or manifest.get("development") is not expected_development \
                or not isinstance(manifest.get("participant"), Mapping) \
                or not isinstance(manifest.get("retained"), Mapping):
            raise ValueError(f"{packet.name}: packet manifest class differs from the audit")
        resolved_by_name[packet.name] = resolved
    return [resolved_by_name[name] for name in EXPECTED_WORLD_REGIMES]


def _validate_world_record(packet: Path, world: object) -> tuple[dict, dict[str, float]]:
    """Validate one retained design before any participant statistic is read."""
    name = packet.name
    expected_regime = EXPECTED_WORLD_REGIMES[name]
    expected_class = EXPECTED_WORLD_PACKET_CLASSES[name]
    if not isinstance(world, dict) \
            or world.get("packet_class") != expected_class \
            or world.get("regime") != expected_regime:
        raise ValueError(f"{name}: retained world family differs from the audit")
    seed = world.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int) \
            or seed != expected_world_seed(name):
        raise ValueError(f"{name}: retained seed differs from the registered plan")
    params = world.get("params")
    mechanisms = world.get("mechanisms")
    character = world.get("character")
    if not isinstance(params, Mapping) \
            or not isinstance(mechanisms, Mapping) \
            or not isinstance(character, Mapping):
        raise ValueError(f"{name}: retained world metadata is incomplete")
    design = mechanisms.get("design")
    coefficients = mechanisms.get("coefficients")
    if not isinstance(design, Mapping) or not isinstance(coefficients, Mapping):
        raise ValueError(f"{name}: retained mechanism design is incomplete")
    if design.get("regime") != expected_regime \
            or params.get("regime") != expected_regime:
        raise ValueError(f"{name}: retained regime labels disagree")
    levels = design.get("levels")
    outside = design.get("outside")
    intensity = design.get("intensity")
    if not isinstance(levels, list) or len(levels) != len(AXES) \
            or any(isinstance(level, bool) or level not in (-1, 1) for level in levels) \
            or not isinstance(outside, list) \
            or not isinstance(intensity, Mapping) or set(intensity) != set(AXES):
        raise ValueError(f"{name}: retained mechanism design has invalid fields")
    raw = {
        axis: _finite_number(intensity[axis], f"{name}: {axis} intensity")
        for axis in AXES
    }
    if expected_regime == "development":
        cell = int(name.removeprefix("dev-"))
        expected_levels = [int(value) for value in DEVELOPMENT_DESIGN[cell]]
        if design.get("cell") != cell or params.get("design_cell") != cell \
                or levels != expected_levels or outside:
            raise ValueError(f"{name}: retained development design cell differs")
        for index, axis in enumerate(AXES):
            low, high = DEVELOPMENT_BAND[axis]
            midpoint = 0.5 * (low + high)
            half = (low, midpoint) if levels[index] < 0 else (midpoint, high)
            if not _range_is_inside([raw[axis], raw[axis]], half):
                raise ValueError(f"{name}: {axis} intensity differs from its design half")
    else:
        if design.get("cell") != -1 or params.get("design_cell") is not None \
                or outside != sorted(outside) \
                or len(outside) != N_HIDDEN_OUTSIDE_AXES \
                or len(set(outside)) != len(outside) \
                or not set(outside) <= set(HIDDEN_EXTRAPOLATION_AXES):
            raise ValueError(f"{name}: hidden outside-axis policy differs")
        if any(levels == [int(value) for value in row] for row in DEVELOPMENT_DESIGN):
            raise ValueError(f"{name}: hidden design repeats a development cell")
        for index, axis in enumerate(AXES):
            low, high = DEVELOPMENT_BAND[axis]
            value = raw[axis]
            if axis not in outside:
                if not _range_is_inside([value, value], (low, high)):
                    raise ValueError(f"{name}: in-band {axis} intensity is outside its band")
                continue
            public_low, public_high = PUBLIC_ENVELOPE[axis]
            expected = (
                (public_low, low) if levels[index] < 0 else (high, public_high)
            )
            if not _range_is_inside([value, value], expected) \
                    or low <= value <= high:
                raise ValueError(f"{name}: out-of-band {axis} intensity is invalid")
    realized = _realized_mechanisms(raw, coefficients, character)
    return dict(world), realized


def _rank01(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(np.argsort(values, kind="stable"), kind="stable")
    return order / max(len(values) - 1, 1)


def _slope(x: np.ndarray, y: np.ndarray) -> float:
    keep = np.isfinite(x) & np.isfinite(y)
    if keep.sum() < 3 or np.ptp(x[keep]) == 0:
        return float("nan")
    return float(np.polyfit(x[keep], y[keep], 1)[0])


def covariates(participant: Path) -> dict:
    """The two county covariates a participant rebuilds, as the contract defines them."""
    geography = pd.read_csv(participant / "geography.csv")
    population = pd.read_csv(participant / "sources" / "population_revised.csv")
    business = pd.read_csv(participant / "sources" / "business_revised.csv")
    contract = json.loads((participant / "contract.json").read_text())
    n_counties = int(contract["n_counties"])
    land = geography.set_index("county")["land_cells"].reindex(range(n_counties)).to_numpy()
    county = population["county"].to_numpy(dtype=np.int64)
    persons = np.bincount(county[county >= 0], minlength=n_counties).astype(np.float64)
    age = (int(contract["ticks"]["revised"]) - population["birth_tick"].to_numpy()) // 12
    adults = np.bincount(county[(county >= 0) & (age >= 18)], minlength=n_counties).astype(np.float64)
    payroll = np.bincount(business["county"].to_numpy(dtype=np.int64).clip(0),
                          weights=business["annual_payroll_cents"].fillna(0.0).to_numpy(),
                          minlength=n_counties)
    return {"urban_c": _rank01(persons / np.maximum(land, 1.0)),
            "econ_c": _rank01(payroll / np.maximum(adults, 1.0)),
            "persons": persons, "n_counties": n_counties,
            "county_state": geography.set_index("county")["state"].reindex(
                range(n_counties)).to_numpy()}


def _elder_share(population: pd.DataFrame, tick: int, n_counties: int) -> np.ndarray:
    """Share of persons 65 and over per county, the public definition of elder_c."""
    county = population["county"].to_numpy(dtype=np.int64).clip(0)
    age = (int(tick) - population["birth_tick"].to_numpy(dtype=np.int64)) // 12
    total = np.bincount(county, minlength=n_counties).astype(np.float64)
    old = np.bincount(county[age >= 65], minlength=n_counties).astype(np.float64)
    return np.divide(old, total, out=np.zeros(n_counties), where=total > 0)


def statistics(packet: Path) -> dict:
    participant = packet / "participant"
    contract = json.loads((participant / "contract.json").read_text())
    tick = int(contract["ticks"]["revised"])
    cov = covariates(participant)
    out = {}

    experience = pd.read_csv(participant / "experience_history.csv")
    cell = experience[(experience["exposure"] > 0) & (experience["deaths"] > 0)].copy()
    cell["log_rate"] = np.log(cell["deaths"] / cell["exposure"])
    # Cell effects, not a pooled slope: the age bands sit orders of magnitude apart, so a
    # slope taken across them reads the changing composition of the deaths rather than the
    # trend. Each cell is centred on its own count-weighted means first.
    numerator = denominator = 0.0
    for _, block in cell.groupby(["age_band", "sex", "state"]):
        w = block["deaths"].to_numpy(dtype=np.float64)
        if len(block) < 3 or w.sum() <= 0:
            continue
        year = block["year"].to_numpy(dtype=np.float64)
        year = year - np.average(year, weights=w)
        rate = block["log_rate"].to_numpy(dtype=np.float64)
        rate = rate - np.average(rate, weights=w)
        numerator += float((w * year * rate).sum())
        denominator += float((w * year ** 2).sum())
    out["mortality_improvement"] = -numerator / denominator if denominator else float("nan")
    # The estimator's own sampling error, from the Poisson variance of a log rate: one
    # over the count. Reported beside the intensity spread, because an anchor that cannot
    # resolve the spread it is there to identify is the thing worth knowing.
    out["mortality_improvement_se"] = float(np.sqrt(1.0 / denominator)) if denominator \
        else float("nan")

    # The axis is the strength of the age gradient in the pull toward urban destinations,
    # so its trace is the gap between where the young settle and where the old do. Net
    # internal migration per person-year of exposure, regressed on the state's urbanity,
    # once for the young-adult band and once for 65 and over.
    state_urban = np.zeros(int(contract["n_states"]))
    np.add.at(state_urban, cov["county_state"], cov["urban_c"])
    np.add.at(state_urban := state_urban, np.arange(0), 0.0)
    counties_per_state = np.bincount(cov["county_state"],
                                     minlength=int(contract["n_states"]))
    state_urban = state_urban / np.maximum(counties_per_state, 1)
    pull = {}
    for label, bands in (("young", ("18-44",)), ("old", ("65-74", "75-84", "85+"))):
        block = experience[experience["age_band"].isin(bands)]
        n = int(contract["n_states"])
        net = block.groupby("state")["net_migration"].sum().reindex(range(n)).fillna(0.0)
        exposure = block.groupby("state")["exposure"].sum().reindex(range(n)).fillna(1.0)
        pull[label] = _slope(state_urban, (net / np.maximum(exposure, 1.0)).to_numpy())
    out["migration_age_pattern"] = pull["young"] - pull["old"]

    population = pd.read_csv(participant / "sources" / "population_revised.csv")
    out["age_reporting_error"] = float((population["birth_tick"] % 12 != 0).mean())

    county = population["county"].to_numpy(dtype=np.int64)
    missing = (population["given_code"].to_numpy() == 0).astype(np.float64)
    records = np.bincount(county.clip(0), minlength=cov["n_counties"]).astype(np.float64)
    share = np.bincount(county.clip(0), weights=missing, minlength=cov["n_counties"]) / \
        np.maximum(records, 1.0)
    out["linkage_urban_gradient"] = _slope(cov["urban_c"], share)

    survey = pd.read_csv(participant / "survey_revised.csv")
    weight = survey["design_weight"].to_numpy(dtype=np.float64)
    survey_persons = np.bincount(survey["county"].to_numpy(dtype=np.int64), weights=weight,
                                 minlength=cov["n_counties"])
    # The completeness axis is a gradient of register coverage in the county's economic
    # rank, and until the benchmark published a count on that gradient it had no anchor.
    # The register against the survey and the register against the state benchmark both
    # read the axis at about zero and reversed sign between regimes: the survey is a thin
    # denominator at this world size, the state series pools counties from both ends of
    # the gradient, and the covariate that reports the gradient is itself thinned by the
    # mechanism. The benchmark now publishes the resident person count of each economic
    # band of counties, and the band of every county sits in geography.csv, so the
    # register's shortfall band by band is the gradient with nothing in between.
    benchmark = pd.read_csv(participant / "sources" / "benchmark_revised.csv")
    geography = pd.read_csv(participant / "geography.csv")
    bands = benchmark[(benchmark["item"] == contract["benchmark"]["subgroup_item"])
                      & (benchmark["level"] == contract["benchmark"]["subgroup_level"])] \
        .sort_values("unit")
    n_bands = int(contract["benchmark"]["n_economic_bands"])
    band_of_county = geography.set_index("county")["economic_band"].reindex(
        range(cov["n_counties"])).to_numpy(dtype=np.int64)
    register_band = np.bincount(band_of_county, weights=cov["persons"], minlength=n_bands)
    with np.errstate(invalid="ignore", divide="ignore"):
        shortfall = np.log(np.maximum(register_band, 1.0)
                           / np.maximum(bands["value"].to_numpy(dtype=np.float64), 1.0))
    out["administrative_completeness"] = _slope(np.arange(float(n_bands)), shortfall)
    cov["elder_c"] = _rank01(_elder_share(population, tick, cov["n_counties"]))

    # This statistic remains a diagnostic only. Its two preflight correlations (+0.020
    # and +0.139) did not clear the 0.4 identifiability threshold, so the attempted
    # age-gradient anchor was removed and the hidden axis is constrained to the public
    # development band. Keep the original public archive-versus-survey comparison here
    # so the failed experiment cannot silently become a grading dependency.
    anchor = contract["health_anchor"]
    health = pd.read_csv(participant / "sources" / "health_revised.csv")
    window = health[health["admission_tick"] > tick - int(anchor["window_months"])]
    n = cov["n_counties"]
    observed = np.bincount(
        survey["county"].to_numpy(dtype=np.int64),
        weights=weight * survey["recent_hospitalization"].to_numpy(),
        minlength=n,
    ) / np.maximum(survey_persons, 1e-9)
    corrected = (observed - (1.0 - anchor["specificity"])) / \
        (anchor["sensitivity"] - (1.0 - anchor["specificity"]))
    archive = window["patient_id"].nunique() / max(population["person_id"].nunique(), 1)
    out["missingness_target_dependence"] = float(
        np.log(max(archive, 1e-9))
        - np.log(max(float(np.average(corrected,
                                      weights=np.maximum(survey_persons, 1e-9))),
                         1e-9))
    )
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--packets", nargs="+", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument(
        "--receipt",
        default=None,
        help="write the machine-readable freeze receipt without per-world hidden values",
    )
    args = ap.parse_args()
    packets = _preflight_packets(args.packets)
    retained_worlds = []
    for packet in packets:
        world_path = packet / "retained" / "world.json"
        if world_path.is_symlink() or not world_path.is_file():
            raise ValueError(f"{packet.name}: retained world record is missing or linked")
        world, realized = _validate_world_record(
            packet,
            json.loads(world_path.read_text()),
        )
        retained_worlds.append((packet, world, realized))
    rows = []
    for packet, world, realized in retained_worlds:
        intensity = world["mechanisms"]["design"]["intensity"]
        measured = statistics(packet)
        rows.append({"world": packet.name, "regime": world["regime"],
                     **{f"axis_intensity_{a}": float(intensity[a]) for a in AXES},
                     **{f"realized_mechanism_{a}": realized[a] for a in AXES},
                     **{f"read_{a}": float(measured[a]) for a in AXES},
                     "drift_se": float(measured["mortality_improvement_se"])})
    frame = pd.DataFrame(rows)
    _validate_world_family(frame)
    frame = frame.set_index("world").loc[list(EXPECTED_WORLD_REGIMES)].reset_index()
    lines = [f"# Identifiability of the six axes, {len(frame)} generator-only worlds", ""]
    axis_receipts = {}
    for axis in AXES:
        truth = frame[f"realized_mechanism_{axis}"].to_numpy()
        read = frame[f"read_{axis}"].to_numpy()
        keep = np.isfinite(truth) & np.isfinite(read)
        rho = float(np.corrcoef(_rank01(truth[keep]), _rank01(read[keep]))[0, 1]) \
            if keep.sum() > 2 else float("nan")
        signed = rho * EXPECTED_SIGN[axis]
        within = []
        within_values = {}
        for family in sorted(set(frame["regime"])):
            block = frame[frame["regime"] == family]
            t = block[f"realized_mechanism_{axis}"].to_numpy()
            r = block[f"read_{axis}"].to_numpy()
            ok = np.isfinite(t) & np.isfinite(r)
            if ok.sum() > 2:
                within_signed = float(
                    np.corrcoef(_rank01(t[ok]), _rank01(r[ok]))[0, 1]
                ) * EXPECTED_SIGN[axis]
                within_values[family] = within_signed
                within.append(f"{family} {within_signed:+.3f}")
        range_record = _axis_range_record(frame, axis)
        axis_intensity_ranges = range_record["axis_intensity_range_observed"]
        lines.append(f"- {axis}: {STATISTIC[axis]}; signed rank correlation {signed:+.3f} "
                     f"pooled, within regime " + ", ".join(within) +
                     f"; raw axis spread "
                     f"{axis_intensity_ranges['pooled'][0]:.3f} to "
                     f"{axis_intensity_ranges['pooled'][1]:.3f}; realized mechanism "
                     f"spread {truth.min():.3f} to {truth.max():.3f}")
        if axis == "mortality_improvement":
            lines.append(f"    drift estimator standard error, mean over worlds "
                         f"{frame['drift_se'].mean():.4f}, against a realized spread of "
                         f"{truth.max() - truth.min():.4f}")
        constrained = axis in HIDDEN_IN_BAND_AXES
        qualified = signed > ANCHOR_CORRELATION_THRESHOLD
        # The pooled correlation is taken over twelve development worlds and six hidden
        # ones, so a trace can be carried largely by the development block while the six
        # worlds a submission is actually scored on hold less of it. The hidden
        # within-regime correlation is measured against the same number and recorded, and
        # it is reported rather than decided on: six worlds is too few points for a rank
        # correlation to carry a threshold. The pooled reading is the registered gate.
        hidden_signed = within_values.get("hidden", float("nan"))
        hidden_qualified = bool(
            math.isfinite(hidden_signed) and hidden_signed > ANCHOR_CORRELATION_THRESHOLD
        )
        axis_receipts[axis] = {
            "statistic": STATISTIC[axis],
            "expected_sign": EXPECTED_SIGN[axis],
            "signed_rank_correlation": signed,
            "within_regime_signed_rank_correlation": within_values,
            **range_record,
            "anchor_correlation_qualified": qualified,
            "hidden_regime_correlation_qualified": hidden_qualified,
            "disposition": (
                "constrained_to_development_range" if constrained
                else "participant_anchor"
            ),
            "development_range": list(DEVELOPMENT_BAND[axis]),
            "hidden_generation_range": list(
                DEVELOPMENT_BAND[axis] if constrained else PUBLIC_ENVELOPE[axis]
            ),
            "hidden_out_of_band_allowed": not constrained,
        }
    binding_axis = min(
        AXES, key=lambda axis: axis_receipts[axis]["signed_rank_correlation"])
    binding_value = axis_receipts[binding_axis]["signed_rank_correlation"]
    shortfalls = sorted(
        axis for axis in HIDDEN_EXTRAPOLATION_AXES
        if not axis_receipts[axis]["hidden_regime_correlation_qualified"]
    )
    lines.append("")
    lines.append(
        f"- binding axis on the pooled rule: {binding_axis} at {binding_value:+.3f} "
        f"against a threshold of {ANCHOR_CORRELATION_THRESHOLD}"
    )
    if shortfalls:
        detail = ", ".join(
            f"{axis} {axis_receipts[axis]['within_regime_signed_rank_correlation']['hidden']:+.3f}"
            for axis in shortfalls
        )
        lines.append(
            f"- HIDDEN-REGIME READING: {len(shortfalls)} anchored axis or axes read below "
            f"{ANCHOR_CORRELATION_THRESHOLD} within the six hidden worlds: {detail}"
        )
        lines.append(
            "  reported, not decided. A rank correlation over six worlds moves by more "
            "than the margin the threshold asks for when one world changes rank, so the "
            "pooled eighteen-world reading is what an anchor is held to."
        )
    else:
        lines.append(
            "- every anchored axis reaches the threshold within the six hidden worlds"
        )
    text = "\n".join(lines) + "\n"
    print(text)
    if args.out:
        Path(args.out).write_text(text)
        frame.to_csv(Path(args.out).with_suffix(".csv"), index=False)
    if args.receipt:
        bindings = []
        for packet, world, _ in retained_worlds:
            manifest = packet / "manifest.json"
            bindings.append({
                "world": packet.name,
                "regime": world["regime"],
                "participant_digest_sha256": _participant_digest(packet),
                "packet_manifest_digest_sha256": _file_digest(manifest),
            })
        source_paths = [
            Path(__file__),
            Path(__file__).resolve().parents[1] / "meridia" / "character.py",
            Path(__file__).resolve().parents[1] / "meridia" / "events.py",
            Path(__file__).resolve().parents[1] / "meridia" / "mechanisms.py",
            Path(__file__).resolve().parents[1] / "meridia" / "packet.py",
            Path(__file__).resolve().parents[1] / "meridia" / "sources.py",
            Path(__file__).resolve().parent / "build_sealed_reconstruction_packet.py",
            Path(__file__).resolve().parent / "build_v4_worlds.py",
        ]
        source_digest = _canonical_digest([
            {
                "path": str(path.relative_to(Path(__file__).resolve().parents[1])),
                "sha256": _file_digest(path),
            }
            for path in source_paths
        ])
        measurement_rows = frame.to_dict(orient="records")
        receipt = {
            "schema": RECEIPT_SCHEMA,
            "anchor_correlation_threshold": ANCHOR_CORRELATION_THRESHOLD,
            "binding_axis": {
                "axis": binding_axis,
                "signed_rank_correlation": binding_value,
            },
            "hidden_regime_correlation_shortfalls": shortfalls,
            "world_count": len(frame),
            "world_bindings": sorted(bindings, key=lambda row: row["world"]),
            "measurement_rows_digest_sha256": _canonical_digest(measurement_rows),
            "generator_source_digest_sha256": source_digest,
            "generator_policy": {
                "outside_axis_count": N_HIDDEN_OUTSIDE_AXES,
                "eligible_for_outside_development_band": list(
                    HIDDEN_EXTRAPOLATION_AXES
                ),
                "held_inside_development_band": list(HIDDEN_IN_BAND_AXES),
            },
            "axes": axis_receipts,
        }
        receipt["digest_sha256"] = _canonical_digest(receipt)
        Path(args.receipt).write_text(
            json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n"
        )


if __name__ == "__main__":
    main()
