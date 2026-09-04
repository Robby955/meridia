"""Public mechanism families with hidden coefficients: the version-four heterogeneity layer.

A separate module because three generator layers need the same three objects and none of
them can own it: the event ledger (``events``), the observed sources (``sources``), and
the packet builder (``packet``). Those objects are the county covariate definitions that
every local coefficient keys off, the committed development design that assigns a world
its mechanism configuration, and the coefficient draw that turns that configuration into
the numbers the ledger and the sources actually use.

Version three had none of this. Fifteen of its nineteen source rates were the same
constant in every world, household growth was one global scalar, and register money was
one global multiplier, so those quantities could be measured once on a development world
and carried unchanged to the hidden world. Here every rate is a per-record or per-person
probability from a published family with hidden coefficients and a county effect, and no
two worlds share a coefficient vector.

Form is public, value is hidden. The families and the covariate definitions are written
into the packet contract; a world's realized coefficients are retained metadata. Every
covariate a coefficient keys off is defined so a participant can evaluate it from the
files they receive:

- ``urban_c``  rank of the county's persons per land cell, scaled to [0, 1]
- ``econ_c``   rank of the county's establishment payroll per resident adult, scaled to [0, 1]
- ``elder_c``  rank of the county's share of persons 65 and over, scaled to [0, 1]
- ``band_r``   the record's own quintile of the national income distribution, 0 to 4

All four are rank statistics, so a participant recovers them from a noisy or
differently scaled estimate of the same underlying quantity. That is what keeps the
local measurement scale identifiable while its level is unknown.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Final

import numpy as np

# ---------------------------------------------------------------------------
# Axes of the regime family (protocol section 10), one continuous intensity each.
# ---------------------------------------------------------------------------

MECHANISM_AXES: Final = (
    "mortality_improvement",          # annual proportional decline in the mortality level
    "migration_age_pattern",          # strength of the age gradient in destination pull
    "age_reporting_error",            # multiplier on the birth and age reporting family
    "linkage_urban_gradient",         # rural excess in name, address, and linkage error
    "administrative_completeness",    # county economic gradient in source coverage
    "missingness_target_dependence",  # dependence of inclusion and item missing on frailty
)

# The public plausibility envelope, and the strictly narrower band every development
# world is drawn from. A hidden world may place at most two intensities between the
# development band and the envelope edge.
PUBLIC_ENVELOPE: Final = {
    "mortality_improvement": (-0.030, 0.075),
    "migration_age_pattern": (0.00, 2.40),
    "age_reporting_error": (0.35, 3.40),
    "linkage_urban_gradient": (0.00, 2.60),
    "administrative_completeness": (0.00, 2.80),
    "missingness_target_dependence": (0.00, 2.20),
}
DEVELOPMENT_BAND: Final = {
    "mortality_improvement": (-0.010, 0.048),
    "migration_age_pattern": (0.25, 1.55),
    "age_reporting_error": (0.70, 2.05),
    "linkage_urban_gradient": (0.30, 1.55),
    "administrative_completeness": (0.30, 1.70),
    "missingness_target_dependence": (0.20, 1.30),
}

# Predeclared interactions (protocol section 10, last paragraph). Nothing outside this
# list is generated, and each entry names the coefficient that carries it.
#
# Three of them are a product of two axes at one site, listed first. Until version four's
# second pass only one was, so every other axis entered additively at a single place and a
# method that fitted the six axes separately on the twelve development worlds transferred
# exactly to any recombination of them. The hidden corner then carried no difficulty
# beyond its six marginals, which is the whole point of putting the hidden world in a
# corner the design does not spend.
PAIRWISE_AXIS_INTERACTIONS: Final = (
    "linkage_gradient_by_migration",
    "health_completeness_by_latent_frailty",
    "death_capture_by_age_error",
)

DECLARED_INTERACTIONS: Final = {
    "linkage_gradient_by_migration":
        "linkage_gradient_by_migration: linkage_urban_gradient x"
        " migration_age_pattern,"
        " scaling the rural excess of the name, address and linkage error rates, so a"
        " world that moves people harder also loses them across the urban gradient",
    "health_completeness_by_latent_frailty":
        "health_inclusion_completeness_by_target: administrative_completeness x"
        " missingness_target_dependence x log frailty_i, in the health inclusion logit",
    "death_capture_by_age_error":
        "death_report_by_age_error: mortality_improvement x age_reporting_error, in the"
        " probability that a death reaches the register after the snapshot and the"
        " register therefore still carries the person",
    "migration_by_stale_address_linkage":
        "id_persist_move_by_migration: migration_age_pattern x recent_move_i, inside the"
        " identifier persistence logit that linkage_urban_gradient also enters",
    "rurality_by_name_and_address_error":
        "linkage_urban_gradient: axis x urban_c",
    "age_error_by_age_slope_of_mortality":
        "age_error_by_mortality_slope: age_reporting_error x the world's Gompertz age"
        " slope, which is itself a per-world draw",
    "income_scale_by_income_dependent_migration":
        "move_income_band: axis-free money band of the mover x the world's income scale"
        " band gradient",
}

# ---------------------------------------------------------------------------
# Committed development design: a twelve-run Plackett-Burman layout in six factors.
#
# Rows 0..10 are the cyclic shifts of the generating vector; row 11 is all low. The
# first six columns are used, one per axis. Main effects are mutually orthogonal and
# orthogonal to every two-factor interaction column, so each axis is estimable from
# twelve worlds and no interaction is confounded with a main effect. The hidden world
# takes a joint configuration that appears in no row.
# ---------------------------------------------------------------------------

_PB12_GENERATOR: Final = (1, 1, -1, 1, 1, 1, -1, -1, -1, 1, -1)


def _plackett_burman_12() -> np.ndarray:
    rows = [np.roll(np.asarray(_PB12_GENERATOR, dtype=np.int8), shift) for shift in range(11)]
    rows.append(np.full(11, -1, dtype=np.int8))
    return np.asarray(rows, dtype=np.int8)


DEVELOPMENT_DESIGN: Final = _plackett_burman_12()[:, : len(MECHANISM_AXES)]
N_DEVELOPMENT_CELLS: Final = int(DEVELOPMENT_DESIGN.shape[0])

# The hidden world draws its level pattern from the sixty-four sign patterns the six axes
# admit, minus the twelve the development design already spends. Only axes with a measured
# participant-file trace may leave the development band. The administrative benchmark
# anchor measured +0.715, while two health/survey preflights measured only +0.020 and
# +0.139. The freeze policy nevertheless keeps both axes in band: anchor availability and
# hidden-axis scope are separate fail-closed decisions. They still vary continuously and
# enter the joint hidden level pattern.
HIDDEN_IN_BAND_AXES: Final = (
    "administrative_completeness",
    "missingness_target_dependence",
)
HIDDEN_EXTRAPOLATION_AXES: Final = tuple(
    axis for axis in MECHANISM_AXES if axis not in HIDDEN_IN_BAND_AXES
)
N_HIDDEN_OUTSIDE_AXES: Final = 2

def _absent_level_patterns() -> tuple[tuple[int, ...], ...]:
    """Every sign pattern over the six axes that no development design row takes."""
    rows = {tuple(int(v) for v in row) for row in DEVELOPMENT_DESIGN}
    patterns = []
    for mask in range(1 << len(MECHANISM_AXES)):
        pattern = tuple(1 if mask >> k & 1 else -1 for k in range(len(MECHANISM_AXES)))
        if pattern not in rows:
            patterns.append(pattern)
    return tuple(patterns)


HIDDEN_LEVEL_PATTERNS: Final = _absent_level_patterns()

# ---------------------------------------------------------------------------
# Coefficients that are not axis intensities: continuous draws from public ranges.
# ---------------------------------------------------------------------------

COEFFICIENT_RANGES: Final = {
    # Register and archive identifiers.
    "id_persist_intercept": (1.20, 2.70),      # logit of vintage-to-vintage id persistence
    "id_persist_urban": (-1.10, 1.10),
    "id_persist_recent_move": (-2.30, -0.70),  # a recent mover keeps an id less often
    "id_persist_move_by_migration": (0.25, 1.05),  # migration intensity x stale address
    "id_reissue_rate": (0.004, 0.022),         # a released id handed to another entity
    # Coverage and item missingness.
    "coverage_intercept_shift": (-0.30, 0.30),
    "coverage_elder_slope": (-0.65, 0.65),
    "coverage_county_sd": (0.10, 0.38),
    "item_missing_econ_slope": (-0.80, 0.80),
    "item_missing_band_slope": (-1.30, 1.30),   # money band of the record, not the target axis
    "item_missing_county_sd": (0.09, 0.36),     # its own county effect, not coverage's
    # Register death capture.
    "death_report_by_age_error": (6.0, 16.0),  # mortality trend x age reporting error
    # Name, birth, and linkage error.
    "linkage_intercept_shift": (-0.35, 0.35),
    "linkage_gradient_by_migration": (0.15, 0.75),  # rural gradient x migration intensity
    "linkage_county_sd": (0.14, 0.48),
    "age_error_age_slope": (0.20, 0.95),       # older ages reported more coarsely
    "age_error_by_mortality_slope": (0.30, 1.20),  # age error x the world's mortality age slope
    # Local money scale.
    "income_scale_urban": (-0.30, 0.30),
    "income_scale_band": (-0.24, 0.24),
    "income_scale_county_sd": (0.04, 0.15),
    # Household dynamics.
    "formation_intercept_shift": (-0.35, 0.35),
    "formation_urban": (-0.95, 0.95),
    "formation_econ": (-0.85, 0.85),
    "formation_size": (0.05, 0.38),
    "formation_age": (-0.30, 0.02),
    "formation_county_sd": (0.10, 0.42),
    "move_intercept_shift": (-0.40, 0.40),
    "move_urban": (-0.85, 0.85),
    "move_income_band": (-0.45, 0.45),
    "move_tenure": (-0.55, -0.10),             # months since the last move damp the hazard
    "move_county_sd": (0.10, 0.42),
    # Mortality and incidence.
    "mortality_urban": (-0.32, 0.10),
    "mortality_econ": (-0.34, 0.06),
    "mortality_frailty": (0.55, 1.10),
    "mortality_county_sd": (0.05, 0.24),
    "incidence_frailty": (0.60, 1.45),
    "incidence_urban": (-0.38, 0.38),
    "incidence_elder_burden": (0.00, 0.55),
    "health_inclusion_completeness_by_target": (0.25, 0.90),  # completeness x target dependence
    "incidence_county_sd": (0.06, 0.26),
}

COUNTY_EFFECT_FAMILIES: Final = (
    "coverage",
    "item_missing",
    "linkage",
    "income_scale",
    "formation",
    "move",
    "mortality",
    "incidence",
    "id_persist",
)

# Public definitions written into the packet contract.
COVARIATE_DEFINITIONS: Final = {
    "urban_c": "rank of the county's persons per land cell, scaled to [0, 1]",
    "econ_c": "rank of the county's establishment payroll per resident adult, scaled to [0, 1];"
               " payroll from the business source, adults from the population source",
    "elder_c": "rank of the county's share of persons 65 and over, scaled to [0, 1]",
    "band_r": "the record's quintile of the national money distribution of its own source, 0 to 4",
    "recent_move_i": "1 when the entity's address changed within the last twelve months",
    "frailty_i": "latent per-person health burden, mean one; observed only through the health anchor",
}

# A newborn inherits part of the mother's latent burden, so health burden clusters by
# household rather than being white noise. Both constants are public.
# The published reference mortality age slope. The world's own slope is a character draw;
# the interaction below reads the ratio, so a world with a steeper age gradient in deaths
# also reports age more coarsely at the old ages.
REFERENCE_MORTALITY_AGE_SLOPE: Final = 0.105

# The neutral value of the age-reporting axis, where its declared products vanish.
REFERENCE_AGE_REPORTING_ERROR: Final = 1.0

NEWBORN_FRAILTY_INHERITANCE: Final = 0.35
NEWBORN_FRAILTY_SIGMA: Final = 0.42
FRAILTY_RANGE: Final = (0.15, 6.00)


def newborn_frailty(mother_frailty: np.ndarray, standard_normal: np.ndarray) -> np.ndarray:
    """Latent frailty of a newborn, given the mother's and a standard normal draw."""
    mother = np.clip(np.asarray(mother_frailty, dtype=np.float64), *FRAILTY_RANGE)
    drawn = np.exp(
        NEWBORN_FRAILTY_INHERITANCE * np.log(mother)
        + NEWBORN_FRAILTY_SIGMA * np.asarray(standard_normal, dtype=np.float64)
        - 0.5 * NEWBORN_FRAILTY_SIGMA ** 2
    )
    return np.clip(drawn, *FRAILTY_RANGE)


# A first qualifying health event is an admission in one of these diagnosis groups. The
# subset is public so first-event incidence is a hazard a participant can define.
QUALIFYING_DIAGNOSIS_GROUPS: Final = (0, 3, 5)


# ---------------------------------------------------------------------------
# Small numeric helpers.
# ---------------------------------------------------------------------------


def expit(x: np.ndarray | float) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=np.float64)))


def logit(p: np.ndarray | float) -> np.ndarray:
    q = np.clip(np.asarray(p, dtype=np.float64), 1e-9, 1.0 - 1e-9)
    return np.log(q / (1.0 - q))


def death_report_late_probability(coefficients: dict[str, float], base: float) -> float:
    """Chance that a death reaches the register after the snapshot, for one world.

    The second of the three products of two axes. A register whose ages are reported
    coarsely closes its records slowly, and how slowly depends on which way the mortality
    level is moving: a falling death count leaves a register with less pressure to
    reconcile, a rising one with more. The observable trace is the gap between the deaths
    the experience file publishes and the disappearances a vintage-to-vintage link
    measures, which is a quantity a method already computes for the mortality rate.

    Both axes are centred where the development band centres them, so a world at the
    middle of the design gets the published base rate and the product is a departure
    from it rather than a level shift.
    """
    low, high = DEVELOPMENT_BAND["mortality_improvement"]
    trend = float(coefficients["mortality_improvement"]) - 0.5 * (low + high)
    coarse = float(coefficients["age_reporting_error"]) - REFERENCE_AGE_REPORTING_ERROR
    shift = float(coefficients["death_report_by_age_error"]) * trend * coarse
    return float(np.clip(expit(logit(float(base)) + shift), 0.02, 0.85))


def rank_uniform(values: np.ndarray) -> np.ndarray:
    """Ranks of ``values`` mapped to [0, 1]; ties broken by position, constant input 0.5."""
    x = np.asarray(values, dtype=np.float64)
    if x.size == 0:
        return x.copy()
    if x.size == 1 or np.ptp(x) == 0.0:
        return np.full(x.shape, 0.5)
    order = np.argsort(x, kind="stable")
    ranks = np.empty(x.size, dtype=np.float64)
    ranks[order] = np.arange(x.size, dtype=np.float64)
    return ranks / (x.size - 1.0)


def migration_age_pull(age: np.ndarray) -> np.ndarray:
    """Public age profile of the pull toward urban destinations.

    Positive for young adults, mildly positive for children moving with a parent, and
    negative past retirement, so migration is age-patterned rather than uniform. The
    world's own intensity multiplies this curve; the curve itself is published.
    """
    x = np.asarray(age, dtype=np.float64)
    return 1.20 * np.exp(-(((x - 26.0) / 16.0) ** 2)) - 0.50 * expit((x - 68.0) / 6.0)


def quintile_band(values: np.ndarray) -> np.ndarray:
    """Rank quintile 0..4 of finite positive values; non-positive and missing map to 0."""
    x = np.asarray(values, dtype=np.float64)
    band = np.zeros(x.shape, dtype=np.int8)
    usable = np.isfinite(x) & (x > 0.0)
    if usable.sum() > 1:
        band[usable] = np.minimum(4, (rank_uniform(x[usable]) * 5.0).astype(np.int8))
    return band


# ---------------------------------------------------------------------------
# The design draw.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MechanismDesign:
    """One world's position in the mechanism family."""

    regime: str
    cell: int                       # design row for a development world, -1 for hidden
    levels: tuple[int, ...]         # +1 or -1 per axis, in MECHANISM_AXES order
    intensity: dict[str, float]     # realized continuous value per axis
    outside: tuple[str, ...]        # axes drawn beyond the development band

    def record(self) -> dict:
        return {
            "regime": self.regime,
            "cell": int(self.cell),
            "levels": [int(v) for v in self.levels],
            "intensity": {k: float(v) for k, v in self.intensity.items()},
            "outside": list(self.outside),
        }


def _half_band(axis: str, level: int) -> tuple[float, float]:
    lo, hi = DEVELOPMENT_BAND[axis]
    middle = 0.5 * (lo + hi)
    return (lo, middle) if level < 0 else (middle, hi)


def _outside_band(axis: str, level: int) -> tuple[float, float]:
    """The stretch between the development band and the envelope edge, on the level's side."""
    lo, hi = DEVELOPMENT_BAND[axis]
    envelope_lo, envelope_hi = PUBLIC_ENVELOPE[axis]
    return (envelope_lo, lo) if level < 0 else (hi, envelope_hi)


def draw_mechanism_design(seed: int, regime: str, cell: int | None = None) -> MechanismDesign:
    """The world's axis levels and continuous intensities.

    A development world takes a committed design row: ``cell`` when given, otherwise the
    row the seed selects. Every axis lands in the half of the development band its level
    names, at a continuous position inside that half. The hidden world takes a level
    pattern that appears in no design row and pushes two identifiable intensities past the
    development band, staying inside the public envelope. Axes in
    ``HIDDEN_IN_BAND_AXES`` always remain within their development ranges.
    """
    if regime not in ("development", "hidden"):
        raise ValueError(f"unknown mechanism regime {regime!r}")
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), 0x4DE5]))
    if regime == "development":
        row = int(rng.integers(0, N_DEVELOPMENT_CELLS)) if cell is None else int(cell)
        if not 0 <= row < N_DEVELOPMENT_CELLS:
            raise ValueError(f"design cell {cell!r} is outside the committed design")
        levels = tuple(int(v) for v in DEVELOPMENT_DESIGN[row])
        intensity = {
            axis: float(rng.uniform(*_half_band(axis, levels[k])))
            for k, axis in enumerate(MECHANISM_AXES)
        }
        return MechanismDesign("development", row, levels, intensity, ())
    if cell is not None:
        raise ValueError("the hidden world does not take a development design cell")
    levels = HIDDEN_LEVEL_PATTERNS[int(rng.integers(0, len(HIDDEN_LEVEL_PATTERNS)))]
    if (DEVELOPMENT_DESIGN == np.asarray(levels, dtype=np.int8)).all(axis=1).any():
        raise RuntimeError("the hidden level pattern repeats a development design row")
    outside = tuple(sorted(
        HIDDEN_EXTRAPOLATION_AXES[k]
        for k in rng.choice(
            len(HIDDEN_EXTRAPOLATION_AXES),
            size=N_HIDDEN_OUTSIDE_AXES,
            replace=False,
        )
    ))
    intensity = {}
    for k, axis in enumerate(MECHANISM_AXES):
        band = (
            _outside_band(axis, levels[k])
            if axis in outside
            else _half_band(axis, levels[k])
        )
        intensity[axis] = float(rng.uniform(*band))
    return MechanismDesign("hidden", -1, levels, intensity, outside)


def draw_mechanism_coefficients(seed: int, design: MechanismDesign) -> dict[str, float]:
    """Every hidden coefficient of the published families, for one world.

    The six axis intensities enter unchanged; the rest are continuous draws from
    ``COEFFICIENT_RANGES``. Two worlds never share a vector, so a coefficient measured on
    a development world does not transfer.
    """
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), 0x4DE6]))
    coefficients = {name: float(rng.uniform(lo, hi)) for name, (lo, hi) in COEFFICIENT_RANGES.items()}
    coefficients.update({axis: float(design.intensity[axis]) for axis in MECHANISM_AXES})
    return coefficients


# ---------------------------------------------------------------------------
# County covariates and county effects.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CountyCovariates:
    """The three public county covariates, plus the land area their definitions use."""

    urban: np.ndarray
    econ: np.ndarray
    elder: np.ndarray
    land_cells: np.ndarray
    county_flat: np.ndarray
    is_outpost: np.ndarray
    distance: np.ndarray          # Chebyshev cell distance between county seats

    @property
    def n_counties(self) -> int:
        return int(len(self.urban))


def build_county_covariates(admin: dict | None, micro: dict | None = None,
                            businesses: dict | None = None) -> CountyCovariates:
    """Rank covariates per county from the snapshot microdata.

    With no administrative geography the world is one neutral county, which is what the
    standalone ledger tests use.  With geography but no microdata every covariate sits
    at the neutral 0.5 while the county effects still apply, so a caller that has not
    built a full world still gets a well-formed, correctly sized layer.
    """
    if admin is None:
        cell = (
            np.asarray(micro["person"]["cell"], dtype=np.int64)
            if micro is not None
            else np.zeros(1, dtype=np.int64)
        )
        neutral = np.full(1, 0.5)
        return CountyCovariates(
            urban=neutral,
            econ=neutral.copy(),
            elder=neutral.copy(),
            land_cells=np.array([max(1, int(cell.max()) + 1)], dtype=np.int64),
            county_flat=np.zeros(int(cell.max()) + 1, dtype=np.int64),
            is_outpost=np.zeros(1, dtype=np.bool_),
            distance=np.zeros((1, 1), dtype=np.float64),
        )
    county_grid = np.asarray(admin["county"], dtype=np.int64)
    county_flat = county_grid.reshape(-1)
    n_counties = int(admin["n_counties"])
    land_cells = np.bincount(county_flat[county_flat >= 0], minlength=n_counties).astype(np.int64)
    width = int(county_grid.shape[1])
    seat = np.asarray(admin["county_seat"], dtype=np.int64)
    seat_row, seat_col = seat // width, seat % width
    distance = np.maximum(
        np.abs(seat_row[:, None] - seat_row[None, :]),
        np.abs(seat_col[:, None] - seat_col[None, :]),
    ).astype(np.float64)
    is_outpost = np.asarray(admin["county_is_outpost"], dtype=np.bool_)

    if micro is None:
        neutral = np.full(n_counties, 0.5)
        return CountyCovariates(
            urban=neutral,
            econ=neutral.copy(),
            elder=neutral.copy(),
            land_cells=land_cells,
            county_flat=county_flat,
            is_outpost=is_outpost,
            distance=distance,
        )

    person = micro["person"]
    cell = np.asarray(person["cell"], dtype=np.int64)
    county = county_flat[cell]
    valid = county >= 0
    persons = np.bincount(county[valid], minlength=n_counties).astype(np.float64)
    density = persons / np.maximum(land_cells, 1)

    age = np.asarray(person["age"], dtype=np.int64)
    adult = valid & (age >= 16)
    adult_count = np.bincount(county[adult], minlength=n_counties).astype(np.float64)
    if businesses is None:
        raise ValueError("county covariates need the business layer for econ_c")
    establishment = businesses["establishment"]
    open_now = np.asarray(establishment["is_active"], dtype=np.bool_)
    establishment_county = county_flat[np.asarray(establishment["cell"], dtype=np.int64)]
    payroll = np.bincount(
        np.maximum(establishment_county[open_now], 0),
        weights=np.asarray(establishment["annual_payroll_cents"], dtype=np.float64)[open_now],
        minlength=n_counties,
    )
    payroll_per_adult = payroll / np.maximum(adult_count, 1.0)
    elder = valid & (age >= 65)
    elder_share = np.bincount(county[elder], minlength=n_counties).astype(np.float64) / np.maximum(
        persons, 1.0
    )
    return CountyCovariates(
        urban=rank_uniform(density),
        econ=rank_uniform(payroll_per_adult),
        elder=rank_uniform(elder_share),
        land_cells=land_cells,
        county_flat=county_flat,
        is_outpost=is_outpost,
        distance=distance,
    )


def draw_county_effects(seed: int, n_counties: int, coefficients: dict[str, float]) -> dict[str, np.ndarray]:
    """One partially pooled county effect per mechanism family, mean removed exactly.

    Removing the mean keeps the family's intercept interpretable as the national level,
    so a county effect is what a hierarchical method has to pool toward, not a hidden
    shift of the whole world.
    """
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), 0x4DE7]))
    effects: dict[str, np.ndarray] = {}
    for family in COUNTY_EFFECT_FAMILIES:
        sd = float(coefficients.get(f"{family}_county_sd", 0.20))
        draw = rng.normal(0.0, sd, size=int(n_counties))
        effects[family] = draw - draw.mean() if n_counties > 1 else np.zeros(int(n_counties))
    return effects


def _vector_digest(values: np.ndarray) -> str:
    """Hash one floating-point vector in application order.

    The retained mechanism record is JSON, so it cannot carry every county-level
    random effect without becoming another copy of the generator state.  A canonical
    little-endian float64 digest preserves the part summaries lose: which county
    received which value.  The length prefix also keeps differently sized byte streams
    in distinct domains.
    """
    vector = np.asarray(values, dtype="<f8")
    if vector.ndim != 1:
        raise ValueError("mechanism vector digest requires one-dimensional values")
    if not np.isfinite(vector).all():
        raise ValueError("mechanism vector digest requires finite values")
    vector = np.ascontiguousarray(vector)
    digest = hashlib.sha256(b"meridia.mechanism-vector.v1\0")
    digest.update(int(vector.size).to_bytes(8, "little", signed=False))
    digest.update(vector.tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class WorldMechanisms:
    """Everything the ledger and the sources need to evaluate a local mechanism."""

    design: MechanismDesign
    coefficients: dict[str, float]
    county: CountyCovariates
    effects: dict[str, np.ndarray] = field(default_factory=dict)
    # One loading per region on the shared shock family, and the same vector spread over
    # the counties so the ledger can read it from a cell. A world with no administrative
    # geography carries a single loading of one, which leaves the standalone ledger where
    # it was.
    region_shock_loading: np.ndarray = field(
        default_factory=lambda: np.ones(1, dtype=np.float64))
    county_shock_loading: np.ndarray = field(
        default_factory=lambda: np.ones(1, dtype=np.float64))

    def county_of_cell(self, cell: np.ndarray) -> np.ndarray:
        index = np.asarray(cell, dtype=np.int64)
        flat = self.county.county_flat
        if len(flat) <= 1:
            return np.zeros(index.shape, dtype=np.int64)
        safe = np.clip(index, 0, len(flat) - 1)
        return np.maximum(flat[safe], 0)

    def covariate(self, name: str, county: np.ndarray) -> np.ndarray:
        values = getattr(self.county, name)
        return values[np.asarray(county, dtype=np.int64)]

    def effect(self, family: str, county: np.ndarray) -> np.ndarray:
        values = self.effects.get(family)
        if values is None:
            return np.zeros(np.shape(county), dtype=np.float64)
        return values[np.asarray(county, dtype=np.int64)]

    def shock_loading(self, county: np.ndarray) -> np.ndarray:
        """The shock loading of the region each county sits in."""
        values = self.county_shock_loading
        if len(values) <= 1:
            return np.full(np.shape(county), float(values[0]) if len(values) else 1.0)
        return values[np.asarray(county, dtype=np.int64)]

    def record(self) -> dict:
        """Return a compact record that still binds order-sensitive mechanisms."""
        return {
            "design": self.design.record(),
            "coefficients": {k: float(v) for k, v in self.coefficients.items()},
            "county_effect_sd": {
                family: float(np.std(values)) for family, values in sorted(self.effects.items())
            },
            "county_effect_digest": {
                family: _vector_digest(values)
                for family, values in sorted(self.effects.items())
            },
            "region_shock_loading": [float(v) for v in self.region_shock_loading],
            "county_shock_loading_digest": _vector_digest(self.county_shock_loading),
        }


def build_world_mechanisms(
    seed: int,
    regime: str = "development",
    admin: dict | None = None,
    micro: dict | None = None,
    businesses: dict | None = None,
    cell: int | None = None,
    mortality_age_slope: float = REFERENCE_MORTALITY_AGE_SLOPE,
) -> WorldMechanisms:
    """Assemble one world's mechanism layer. Deterministic in (seed, regime, cell).

    ``mortality_age_slope`` is the world's own Gompertz slope, which the caller has
    already drawn. It enters one derived coefficient, the age-reporting interaction the
    protocol predeclares, and nothing else.
    """
    design = draw_mechanism_design(seed, regime, cell)
    coefficients = draw_mechanism_coefficients(seed, design)
    coefficients["age_error_mortality_scale"] = float(
        1.0
        + coefficients["age_error_by_mortality_slope"]
        * (float(mortality_age_slope) / REFERENCE_MORTALITY_AGE_SLOPE - 1.0)
    )
    covariates = build_county_covariates(admin, micro, businesses)
    effects = draw_county_effects(seed, covariates.n_counties, coefficients)
    # The shock family lives in ``demography``, which imports this module for newborn
    # frailty, so the loading draw is reached here rather than at import time.
    from .demography import draw_shock_loadings
    if admin is None:
        region_loading = np.ones(1, dtype=np.float64)
        county_loading = np.ones(covariates.n_counties, dtype=np.float64)
    else:
        county_state = np.asarray(admin["county_state"], dtype=np.int64)
        region_loading = draw_shock_loadings(seed, int(admin["n_states"]))
        county_loading = region_loading[county_state]
    return WorldMechanisms(design, coefficients, covariates, effects,
                           region_loading, county_loading)


def contract_block() -> dict:
    """The public description of the mechanism layer, written into every packet."""
    return {
        "axes": list(MECHANISM_AXES),
        "public_envelope": {k: list(v) for k, v in PUBLIC_ENVELOPE.items()},
        "development_band": {k: list(v) for k, v in DEVELOPMENT_BAND.items()},
        "declared_interactions": dict(DECLARED_INTERACTIONS),
        "pairwise_axis_interactions": list(PAIRWISE_AXIS_INTERACTIONS),
        "development_design": DEVELOPMENT_DESIGN.tolist(),
        "n_development_cells": N_DEVELOPMENT_CELLS,
        "hidden_axis_policy": {
            "outside_axis_count": N_HIDDEN_OUTSIDE_AXES,
            "eligible_for_outside_development_band": list(HIDDEN_EXTRAPOLATION_AXES),
            "held_inside_development_band": list(HIDDEN_IN_BAND_AXES),
            "anchor_correlation_required_for_extrapolation": 0.4,
        },
        "covariates": dict(COVARIATE_DEFINITIONS),
        "families": {
            "source_inclusion": "logit p_include = logit(base) + a_completeness * (econ_c - 0.5)"
                        " + a_elder * (elder_c - 0.5) + u_c ; the health source adds"
                        " clip(2 * a_frailty * (1 + a_completeness_by_target *"
                        " (a_completeness - 1)) * log(frailty_i), -2, 2)",
            "item_missing": "logit p_missing = logit(base) + b_econ * (econ_c - 0.5)"
                            " + b_band * (band_r - 2) / 2 + v_c ; b_band is the record's"
                            " own money band and is not the health selection slope, and"
                            " v_c is item missingness's own county effect, a separate"
                            " draw from the u_c of the inclusion family above",
            "death_capture": "logit p_late = logit(base) + a_capture * (improvement -"
                             " midpoint of the published improvement band) *"
                             " (age_reporting_error - 1); a death reported late is a"
                             " person the register still carries at the snapshot",
            "reporting_error": "logit p_error = logit(base) + a_rural * (0.5 - urban_c)"
                               " * (1 + a_rural_by_migration * (migration_age_pattern"
                               " - 1)) + u_c ; birth and age error scale with"
                               " (1 + a_age_slope * (age - 45) / 40) and with"
                               " (1 + a_mortality_slope * (b / 0.105 - 1)) for the"
                               " world's own Gompertz age slope b",
            "money_scale": "s_cr = s_0 * exp(b_urban * (urban_c - 0.5)"
                           " + b_band * (band_r - 2) / 2 + u_c)",
            "identifier_persistence": "logit p_persist = a_0 + a_urban * (urban_c - 0.5)"
                                      " + a_move * (1 + a_move_by_migration *"
                                      " migration_age_pattern) * recent_move_i + u_c",
            "household_formation": "log h_i = log(base) + b_urban * (urban_c - 0.5)"
                                   " + b_econ * (econ_c - 0.5) + b_size * (size_i - 3)"
                                   " + b_age * (age_i - 24) / 6 + u_c",
            "household_move": "log h_h = log(base) + b_urban * (urban_c - 0.5)"
                              " + b_band * (band_r - 2) / 2 + b_tenure * log1p(months_since_move / 12)"
                              " + u_c",
            "mortality": "q_i = q_gompertz(age) * frailty_i ** b_frailty"
                         " * exp(b_urban * (urban_c - 0.5) + b_econ * (econ_c - 0.5) + u_c)"
                         " * (1 - improvement) ** (years since the snapshot)",
            "incidence": "risk_i proportional to frailty_i ** b_frailty"
                         " * exp(b_urban * (urban_c - 0.5) + b_elder * (elder_c - 0.5) + u_c)"
                         " times the published age curve",
            "destination": "gravity weight proportional to urbanity ** (a_age_pattern * g(age))"
                           " divided by (1 + distance / 12)",
        },
        "qualifying_diagnosis_groups": list(QUALIFYING_DIAGNOSIS_GROUPS),
    }
