"""Version four actuarial truth and gates: exposures, rates, liabilities, tails, reserve.

Version three scored a point allocation against one realized future, and its regret was
zero for any allocation that sat under every county's true demand: the decision gate
rewarded restraint rather than forecasting. This module carries the replacement described
in sections 4 to 9 of the version-four protocol, and nothing else. It holds four things
that the release, projection, and verifier modules then call:

1. One reading pass over the monthly ledger that reconstructs person-months of residence
   between two ticks. Exposure by attained age and the obligation cash flows come out of
   the same pass, so a rate and the liability it prices can never disagree.
2. The obligation contract: a monthly benefit while a person is alive, eligible, and
   resident; a fixed cost at a person's first qualifying health event; an optional
   death-contingent amount; and public monthly discount factors. The formula is public
   and simple. The difficulty is survival, migration, and incidence.
3. The continuation ensemble: M committed future substreams derived from the sealed root
   seed, one of them designated the realized future for reporting. Regional truth is the
   ensemble mean, its 0.95 quantile, and the mean above that quantile. Binary tail and
   reserve gates read the ensemble, never one path.
4. The scored surface: one pooled exposure-and-rate gate, one tail gate, the public
   exposure-based reserve total, the sealed expected uncovered obligation, and the skill
   score against a frozen practical baseline and a perfect-information oracle.

Every threshold is a named field of ``ActuarialThresholds`` carrying a placeholder value.
Placeholders are frozen on qualification worlds before the hidden world exists, following
protocol section 12; none of them may be tuned to a submission.

Age band vocabulary: the population release bands stay as they are. The actuarial bands
are a separate constant because attained-age rate estimation needs a finer old-age split.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from decimal import Decimal, ROUND_CEILING
from typing import Callable, Iterable, Sequence

import numpy as np

from .events import EVENT_TYPES
from .release import RELEASE_COLUMNS, SEX_LABELS

# ------------------------------------------------------------------- frozen vocabulary

ACTUARIAL_AGE_BANDS: tuple[tuple[int, int], ...] = (
    (0, 17), (18, 44), (45, 64), (65, 74), (75, 84), (85, 200))
ACTUARIAL_AGE_BAND_LABELS: tuple[str, ...] = (
    "0-17", "18-44", "45-64", "65-74", "75-84", "85+")

# Broad bands are the dense cells the county-level exposure gate reads. Each is a union
# of actuarial bands, so one accumulation serves both vocabularies.
BROAD_AGE_BAND_LABELS: tuple[str, ...] = ("0-17", "18-64", "65+")
BROAD_BAND_MEMBERS: tuple[tuple[int, ...], ...] = ((0,), (1, 2), (3, 4, 5))

EXPOSURE_ESTIMAND = "person_years_exposure"
MORTALITY_ESTIMAND = "mortality_rate"
INCIDENCE_ESTIMAND = "qualifying_event_rate"
RATE_ESTIMANDS: tuple[str, ...] = (EXPOSURE_ESTIMAND, MORTALITY_ESTIMAND, INCIDENCE_ESTIMAND)

# The exposure and rate fields ride the release table (protocol section 4 item 1). They
# carry two columns the eight version-three estimands leave blank.
RATE_EXTRA_COLUMNS: tuple[str, ...] = ("sex", "age_band")
RESERVE_COLUMNS: tuple[str, ...] = ("region", "liability_mean", "q95", "es95", "allocation")
V4_RELEASE_COLUMNS: tuple[str, ...] = (
    RELEASE_COLUMNS[:3] + RATE_EXTRA_COLUMNS + RELEASE_COLUMNS[3:])
V4_PROJECTION_COLUMNS: tuple[str, ...] = RELEASE_COLUMNS
V4_SUBMISSION_COLUMNS: dict[str, tuple[str, ...]] = {
    "release.csv": V4_RELEASE_COLUMNS,
    "projection.csv": V4_PROJECTION_COLUMNS,
    "reserve.csv": RESERVE_COLUMNS,
}

MONTHS_PER_YEAR = 12
TAIL_LEVEL = 0.95
DEFAULT_ENSEMBLE_SIZE = 2048
CONTINUATION_DOMAIN = 0xC047      # substream tag; distinct from the ledger's own tag


# ---------------------------------------------------------------------- the thresholds

@dataclass(frozen=True)
class ActuarialThresholds:
    """Named scoring parameters and legacy diagnostic thresholds.

    Stabilizers, the eligibility floor, the tail-width divisor, rounding, and numerical
    tolerance remain inputs to version-four measurements. The old individual ceilings and
    floors remain for legacy diagnostic reports only. The version-four pass decision reads
    the separate exact composite bar document and never falls back to these values.
    """

    # Section 7: stabilizers c_x in e = |x_hat - x*| / (x* + c_x).
    exposure_stabilizer: float = 500.0            # person-years; to be frozen
    mortality_stabilizer: float = 5.0e-4          # per person-year; to be frozen
    incidence_stabilizer: float = 5.0e-4          # per person-year; to be frozen
    # Section 7: which cells are gated, and how the cell errors are reduced.
    exposure_eligibility_person_years: float = 600.0     # the denominator rule, below
    rate_gate_percentile: float = 0.95                   # to be frozen
    exposure_error_ceiling: float = 0.15                 # to be frozen
    mortality_error_ceiling: float = 0.35                # to be frozen
    incidence_error_ceiling: float = 0.35                # to be frozen
    rate_coverage_floor: float = 0.70                    # to be frozen
    # Section 8: tail calibration and the proper score.
    tau_mean: float = 0.020                              # to be frozen
    tau_worst: float = 0.050                             # to be frozen
    worst_region_quantile: float = 0.90                  # fixed by the protocol
    quantile_score_ceiling: float = 0.050                # to be frozen
    es_error_ceiling: float = 0.150                      # to be frozen
    q95_width_error_ceiling: float = 0.500               # to be frozen
    es95_width_error_ceiling: float = 0.500              # to be frozen
    min_tail_width_fraction: float = 0.010               # fixed, guards the divisor
    # Section 9: the reserve construction and the decision gate.
    reserve_rounding_unit: float = 1_000.0               # public rounding rule
    skill_minimum: float = 0.35                          # to be frozen
    regional_shortfall_ceiling: float = 0.20             # to be frozen
    catastrophic_tail_ceiling: float = 0.50              # to be frozen
    feasibility_tolerance: float = 1e-6


PLACEHOLDER_THRESHOLDS = ActuarialThresholds()


# Structural eligibility reads sealed exposure only. It does not use an expected-event
# schedule. Rates are pooled to the three broad bands before this rule is applied, so the
# 65-and-over quantity the obligation prices is one scored state cell rather than three
# thin cells whose inclusion depends on a reference rate.
EXPOSURE_ELIGIBILITY_BY_BAND: dict[str, float] = {
    "0-17": 600.0,
    "18-64": 600.0,
    "65+": 500.0,
}
RATE_GATE_BANDS: tuple[str, ...] = BROAD_AGE_BAND_LABELS

_BROAD_BAND_FOR_ACTUARIAL = {
    band: broad
    for broad, members in zip(BROAD_AGE_BAND_LABELS, BROAD_BAND_MEMBERS)
    for member in members
    for band in (ACTUARIAL_AGE_BAND_LABELS[member],)
}


def eligibility_floor(thresholds: ActuarialThresholds, band: str,
                      estimand: str = EXPOSURE_ESTIMAND) -> float:
    """Sealed person-years a broad cell needs before the composite rate gate reads it.

    ``estimand`` remains in the signature for callers that label diagnostics by quantity;
    it cannot change the floor. That is the phase-three rule: eligibility is a property of
    exposure, not of a reference event-rate schedule.
    """
    del estimand
    default = float(thresholds.exposure_eligibility_person_years)
    if not math.isfinite(default) or default <= 0.0:
        raise ValueError("the exposure eligibility floor must be positive and finite")
    broad = _BROAD_BAND_FOR_ACTUARIAL.get(band, band)
    base = float(EXPOSURE_ELIGIBILITY_BY_BAND.get(broad, 600.0))
    return base * default / 600.0


# ----------------------------------------------------------------- obligation contract

@dataclass(frozen=True)
class ObligationContract:
    """The public, deterministic obligation the reserve has to cover.

    ``b`` is paid every month a person is alive, at or above ``eligibility_min_age``, and
    resident in the region. ``c`` is paid once, at a person's first qualifying health event
    inside the window. ``d`` is paid at death. Discount factors are v_t = (1 + i)^(-t) for
    t = 1..horizon_months. Eligibility reads attained age, sex, and residence only, all of
    which a method can estimate from the files it receives.
    """

    # The weights are public and were set from the sealed ensemble's own dispersion, not
    # from any submission. A liability that is mostly a monthly annuity over a large
    # elderly stock is very nearly deterministic: measured on a qualification world, its
    # regional tail (q95 minus mean, over the mean) ran 0.032 to 0.089, under the three to
    # nine percent a reconstruction of the same regions misses by, so the sealed tail
    # would have been a target no method could reach. Shifting weight onto the
    # first-event cost and the death benefit, both counts rather than stocks, widens that
    # tail to 0.056 to 0.142 while leaving all three terms of section 5 in force.
    monthly_benefit: float = 150.0
    eligibility_min_age: int = 65
    qualifying_event_cost: float = 15_000.0
    death_benefit: float = 7_500.0
    monthly_discount_rate: float = 0.002
    horizon_months: int = 60
    qualifying_diagnosis_groups: tuple[int, ...] = ()   # empty means every admission

    def discount_factors(self) -> np.ndarray:
        t = np.arange(1, self.horizon_months + 1, dtype=np.float64)
        return (1.0 + self.monthly_discount_rate) ** (-t)

    def qualifies(self, diagnosis_group: int) -> bool:
        if not self.qualifying_diagnosis_groups:
            return True
        return int(diagnosis_group) in self.qualifying_diagnosis_groups

    def as_public(self) -> dict:
        return asdict(self) | {"qualifying_diagnosis_groups":
                               list(self.qualifying_diagnosis_groups)}

    @staticmethod
    def from_public(payload: dict) -> "ObligationContract":
        groups = tuple(int(g) for g in payload.get("qualifying_diagnosis_groups", ()))
        known = {f: payload[f] for f in
                 ("monthly_benefit", "eligibility_min_age", "qualifying_event_cost",
                  "death_benefit", "monthly_discount_rate", "horizon_months")
                 if f in payload}
        return ObligationContract(qualifying_diagnosis_groups=groups, **known)


def regions_from_admin(admin: dict) -> np.ndarray:
    """Region of every county. Regions are the states, the level the reserve is held at."""
    return np.asarray(admin["county_state"], dtype=np.int64)


# ------------------------------------------------------- the person-month ledger pass

def _band_of_age(age: np.ndarray,
                 bands: Sequence[tuple[int, int]] = ACTUARIAL_AGE_BANDS) -> np.ndarray:
    age = np.asarray(age, dtype=np.int64)
    band = np.full(len(age), -1, dtype=np.int64)
    for b, (lo, hi) in enumerate(bands):
        band[(age >= lo) & (age <= hi)] = b
    if (band < 0).any():
        raise ValueError("an attained age falls outside every actuarial band")
    return band


def _positions(ids: Iterable[int]) -> dict[int, int]:
    table: dict[int, int] = {}
    for value in ids:
        v = int(value)
        if v and v not in table:
            table[v] = len(table)
    return table


def actuarial_pass(start_state: dict, event: dict, admin: dict, start_tick: int,
                   months: int, contract: ObligationContract,
                   region_of_county: np.ndarray | None = None) -> dict:
    """One reading pass over the ledger: person-months, deaths, events, cash flows.

    ``start_state`` is a replayed ledger state at ``start_tick``: it needs a ``person``
    block with truth_person_id, truth_household_id, birth_tick, sex, is_alive, and a
    ``household`` block with truth_household_id and cell. ``event`` is the ledger's event
    table. Months are t = 1..``months``, and month t is the state after every event with
    tick at or below ``start_tick + t`` has been applied: a birth at that tick counts, a
    death at that tick does not, and a household that moved at that tick is charged to its
    new region. Attained age is recomputed every month, so a person who crosses a band
    boundary inside the window contributes to both bands, which is what makes the mortality
    figure a rate rather than a ratio of stocks.
    """
    if months <= 0:
        raise ValueError("months must be positive")
    county_flat = np.asarray(admin["county"]).flatten() if "county" in admin else None
    n_counties = int(admin["n_counties"])
    region_of_county = regions_from_admin(admin) if region_of_county is None else \
        np.asarray(region_of_county, dtype=np.int64)
    n_regions = int(region_of_county.max()) + 1 if len(region_of_county) else 0
    n_bands = len(ACTUARIAL_AGE_BANDS)

    person, household = start_state["person"], start_state["household"]
    person_ids = list(int(v) for v in person["truth_person_id"])
    household_ids = list(int(v) for v in household["truth_household_id"])
    born = event["truth_person_id"][event["event_type"] == EVENT_TYPES["person_birth"]]
    formed = event["truth_household_id"][event["event_type"] == EVENT_TYPES["household_formed"]]
    p_at = _positions(person_ids + [int(v) for v in born])
    h_at = _positions(household_ids + [int(v) for v in formed])

    n_p, n_h = len(p_at), len(h_at)
    p_household = np.full(n_p, -1, dtype=np.int64)
    p_alive = np.zeros(n_p, dtype=bool)
    p_birth_tick = np.zeros(n_p, dtype=np.int64)
    p_sex = np.zeros(n_p, dtype=np.int64)
    p_qualified = np.zeros(n_p, dtype=bool)
    h_cell = np.full(n_h, -1, dtype=np.int64)

    for k, identifier in enumerate(person_ids):
        pos = p_at[int(identifier)]
        p_household[pos] = h_at[int(person["truth_household_id"][k])]
        p_alive[pos] = bool(person["is_alive"][k])
        p_birth_tick[pos] = int(person["birth_tick"][k])
        p_sex[pos] = int(person["sex"][k])
    for k, identifier in enumerate(household_ids):
        h_cell[h_at[int(identifier)]] = int(household["cell"][k])

    def county_of_cell(cell: np.ndarray) -> np.ndarray:
        return cell if county_flat is None else county_flat[cell]

    exposure = np.zeros((n_counties, len(SEX_LABELS), n_bands), dtype=np.float64)
    deaths = np.zeros_like(exposure)
    qualifying = np.zeros_like(exposure)
    benefit = np.zeros((months, n_regions), dtype=np.float64)
    event_cost = np.zeros((months, n_regions), dtype=np.float64)
    death_benefit = np.zeros((months, n_regions), dtype=np.float64)

    superseded = {int(v) for v in event["supersedes_event_id"] if int(v)}
    tick_column = np.asarray(event["tick"], dtype=np.int64)
    keep = (tick_column > int(start_tick)) & (tick_column <= int(start_tick) + months)
    if superseded:
        keep &= ~np.isin(np.asarray(event["truth_event_id"], dtype=np.uint64),
                         np.asarray(sorted(superseded), dtype=np.uint64))
    rows_by_tick: dict[int, list[int]] = {}
    for row in np.flatnonzero(keep):
        rows_by_tick.setdefault(int(tick_column[row]), []).append(int(row))

    def charge(cell_value: int, sex: int, band: int, table: np.ndarray) -> int:
        county = int(county_of_cell(np.asarray([cell_value]))[0])
        table[county, sex, band] += 1.0
        return county

    for t in range(1, months + 1):
        tick = int(start_tick) + t
        for row in rows_by_tick.get(tick, []):
            kind = int(event["event_type"][row])
            if kind == EVENT_TYPES["person_birth"]:
                pos = p_at[int(event["truth_person_id"][row])]
                p_household[pos] = h_at[int(event["truth_household_id"][row])]
                p_birth_tick[pos] = int(event["birth_tick"][row])
                p_sex[pos] = int(event["sex"][row])
                p_alive[pos] = True
            elif kind == EVENT_TYPES["person_death"]:
                pos = p_at[int(event["truth_person_id"][row])]
                p_alive[pos] = False
                age = (tick - int(p_birth_tick[pos])) // MONTHS_PER_YEAR
                band = int(_band_of_age(np.asarray([age]))[0])
                county = charge(int(event["from_cell"][row]), int(p_sex[pos]), band, deaths)
                death_benefit[t - 1, region_of_county[county]] += contract.death_benefit
            elif kind == EVENT_TYPES["household_formed"]:
                new_h = h_at[int(event["truth_household_id"][row])]
                h_cell[new_h] = int(event["to_cell"][row])
                p_household[p_at[int(event["truth_person_id"][row])]] = new_h
            elif kind == EVENT_TYPES["household_moved"]:
                h_cell[h_at[int(event["truth_household_id"][row])]] = int(event["to_cell"][row])
            elif kind == EVENT_TYPES["encounter_admitted"]:
                identifier = int(event["truth_person_id"][row])
                if identifier not in p_at:
                    continue
                pos = p_at[identifier]
                if p_qualified[pos] or not p_alive[pos]:
                    continue
                if not contract.qualifies(int(event["diagnosis_group"][row])):
                    continue
                p_qualified[pos] = True
                age = (tick - int(p_birth_tick[pos])) // MONTHS_PER_YEAR
                band = int(_band_of_age(np.asarray([age]))[0])
                county = charge(int(h_cell[p_household[pos]]), int(p_sex[pos]), band,
                                qualifying)
                event_cost[t - 1, region_of_county[county]] += contract.qualifying_event_cost

        alive = np.flatnonzero(p_alive)
        if not len(alive):
            continue
        county = county_of_cell(h_cell[p_household[alive]])
        sex = p_sex[alive]
        age = (tick - p_birth_tick[alive]) // MONTHS_PER_YEAR
        band = _band_of_age(age)
        np.add.at(exposure, (county, sex, band), 1.0)
        eligible = age >= contract.eligibility_min_age
        if eligible.any():
            np.add.at(benefit, (t - 1, region_of_county[county[eligible]]),
                      contract.monthly_benefit)

    return {"start_tick": int(start_tick), "months": int(months), "n_regions": n_regions,
            "exposure_person_months": exposure, "deaths": deaths,
            "qualifying_events": qualifying, "benefit": benefit,
            "event_cost": event_cost, "death_benefit": death_benefit}


def actuarial_pass_from_history(history: dict, admin: dict, start_tick: int, months: int,
                                contract: ObligationContract,
                                region_of_county: np.ndarray | None = None) -> dict:
    """``actuarial_pass`` on a full ledger, replaying the start state once."""
    from .events import replay_event_history
    state = replay_event_history(history, int(start_tick))
    return actuarial_pass(state, history["event"], admin, start_tick, months, contract,
                          region_of_county)


def liabilities_from_pass(result: dict, contract: ObligationContract) -> np.ndarray:
    """Regional present values L_r from one pass, discounted at the public factors."""
    v = contract.discount_factors()[:result["months"]]
    flow = result["benefit"] + result["event_cost"] + result["death_benefit"]
    return v @ flow


# --------------------------------------------------------------- continuation ensemble

def continuation_member_key(seed: int, member: int, month: int) -> np.random.SeedSequence:
    """Substream key for month ``month`` of continuation ``member``.

    A fresh domain tag keeps every continuation disjoint from the ledger's own monthly
    stream and from every other member. Member seeds are never derived by arithmetic on the
    root seed, because sums collide across members and months.
    """
    if member < 0 or month < 1:
        raise ValueError("member must be non-negative and month must be positive")
    return np.random.SeedSequence([int(seed), CONTINUATION_DOMAIN, int(member), int(month)])


@dataclass(frozen=True)
class ContinuationEnsemble:
    """Regional liabilities on every committed continuation.

    ``liability`` has shape (M, R). ``realized_member`` names the single continuation that
    is the world's realized future for reporting; the tail and reserve gates read the whole
    matrix, never that one row.
    """

    liability: np.ndarray
    realized_member: int = 0
    level: float = TAIL_LEVEL

    @property
    def n_members(self) -> int:
        return int(self.liability.shape[0])

    @property
    def n_regions(self) -> int:
        return int(self.liability.shape[1])

    def truth(self) -> dict:
        return ensemble_truth(self.liability, self.level)

    def realized(self) -> np.ndarray:
        return np.asarray(self.liability[self.realized_member], dtype=np.float64)


def empirical_tail(liability: np.ndarray, level: float = TAIL_LEVEL) -> tuple[np.ndarray,
                                                                              np.ndarray]:
    """Exact empirical quantile and expected shortfall, column by column.

    For ``M`` observations the quantile is order statistic ``ceil(level * M)`` using
    one-based ranks. Expected shortfall is the mean of every observation at or above that
    value, including all ties. This definition is shared by truth, references, and audits;
    no interpolated quantile convention is allowed to change it.
    """
    liability = np.asarray(liability, dtype=np.float64)
    if liability.ndim != 2 or liability.shape[0] < 2:
        raise ValueError("liability must be a (members, regions) matrix of two or more rows")
    if not math.isfinite(float(level)) or not 0.0 < float(level) <= 1.0:
        raise ValueError("the empirical tail level must lie in (0, 1]")
    if not np.isfinite(liability).all():
        raise ValueError("liability must contain only finite values")
    rank = int(math.ceil(float(level) * liability.shape[0]))
    q = np.sort(liability, axis=0)[rank - 1]
    tail_mean = np.empty(liability.shape[1], dtype=np.float64)
    for r in range(liability.shape[1]):
        above = liability[:, r][liability[:, r] >= q[r]]
        tail_mean[r] = float(above.mean())
    return q, tail_mean


def ensemble_truth(liability: np.ndarray, level: float = TAIL_LEVEL) -> dict:
    """Regional mean, exact empirical quantile, and tied-tail expected shortfall."""
    liability = np.asarray(liability, dtype=np.float64)
    q, tail_mean = empirical_tail(liability, level)
    return {"mean": liability.mean(axis=0), "q": q, "es": tail_mean, "level": float(level)}


def build_continuation_ensemble(continue_member: Callable[[int], np.ndarray],
                                n_members: int = DEFAULT_ENSEMBLE_SIZE,
                                realized_member: int = 0,
                                level: float = TAIL_LEVEL) -> ContinuationEnsemble:
    """Assemble the ensemble from a member builder that returns regional liabilities.

    ``continue_member(m)`` runs continuation ``m`` from the baseline truth under the same
    hidden regime and prices it, returning one vector of length R. Members are separate
    histories and are never merged into one ledger: two continuations that branch at the
    same tick hand the same person identity to two different newborns, so every
    cross-member quantity is keyed on (member, region) and nothing else.
    """
    if n_members < 2:
        raise ValueError("an ensemble needs at least two continuations")
    rows = [np.asarray(continue_member(m), dtype=np.float64) for m in range(n_members)]
    widths = {row.shape for row in rows}
    if len(widths) != 1:
        raise ValueError("continuations disagree on the number of regions")
    return ContinuationEnsemble(np.vstack(rows), int(realized_member), float(level))


# ------------------------------------------------------- exposure and rate truth table

def exposure_and_rate_truth(result: dict, admin: dict) -> dict:
    """Exposure and rate truth keyed (estimand, level, unit, sex, band).

    Exposure is in person-years at county and state on the actuarial and broad bands.
    Mortality and incidence are occurrence-exposure rates per person-year at both levels
    and both band vocabularies. Fine-band rates remain diagnostics. The composite gate
    reads county broad-band exposure and state broad-band rates, all selected by sealed
    exposure alone.
    """
    county_state = np.asarray(admin["county_state"], dtype=np.int64)
    n_states = int(admin["n_states"])
    exposure_months = result["exposure_person_months"]
    deaths, events = result["deaths"], result["qualifying_events"]

    def by_state(cube: np.ndarray) -> np.ndarray:
        out = np.zeros((n_states,) + cube.shape[1:], dtype=np.float64)
        np.add.at(out, county_state, cube)
        return out

    levels = {"county": (exposure_months, deaths, events),
              "state": (by_state(exposure_months), by_state(deaths), by_state(events))}
    truth: dict[tuple[str, str, int, str, str], float] = {}
    for level, (months_cube, death_cube, event_cube) in levels.items():
        years = months_cube / MONTHS_PER_YEAR
        for u in range(years.shape[0]):
            for s, sex_label in enumerate(SEX_LABELS):
                for b, band in enumerate(ACTUARIAL_AGE_BAND_LABELS):
                    e = float(years[u, s, b])
                    truth[(EXPOSURE_ESTIMAND, level, u, sex_label, band)] = e
                    truth[(MORTALITY_ESTIMAND, level, u, sex_label, band)] = \
                        float(death_cube[u, s, b]) / e if e > 0 else float("nan")
                    truth[(INCIDENCE_ESTIMAND, level, u, sex_label, band)] = \
                        float(event_cube[u, s, b]) / e if e > 0 else float("nan")
                for broad, members in zip(BROAD_AGE_BAND_LABELS, BROAD_BAND_MEMBERS):
                    exposure = float(sum(years[u, s, b] for b in members))
                    truth[(EXPOSURE_ESTIMAND, level, u, sex_label, broad)] = exposure
                    truth[(MORTALITY_ESTIMAND, level, u, sex_label, broad)] = \
                        float(sum(death_cube[u, s, b] for b in members)) / exposure \
                        if exposure > 0 else float("nan")
                    truth[(INCIDENCE_ESTIMAND, level, u, sex_label, broad)] = \
                        float(sum(event_cube[u, s, b] for b in members)) / exposure \
                        if exposure > 0 else float("nan")
    return truth


GATED_RATE_CELLS = {
    EXPOSURE_ESTIMAND: ("county", BROAD_AGE_BAND_LABELS),
    MORTALITY_ESTIMAND: ("state", BROAD_AGE_BAND_LABELS),
    INCIDENCE_ESTIMAND: ("state", BROAD_AGE_BAND_LABELS),
}


# Exposure is published on both vocabularies, because the broad bands are the dense cells
# the county gate reads and the actuarial bands are the denominators of the rates.
EXPOSURE_BAND_LABELS: tuple[str, ...] = tuple(
    ACTUARIAL_AGE_BAND_LABELS + tuple(b for b in BROAD_AGE_BAND_LABELS
                                      if b not in ACTUARIAL_AGE_BAND_LABELS))
RATE_LEVELS: tuple[str, ...] = ("state", "county")


def bands_for(estimand: str, level: str) -> tuple[str, ...]:
    """Band vocabulary of one published block, and of the cells its gate reads."""
    if estimand != EXPOSURE_ESTIMAND:
        return ACTUARIAL_AGE_BAND_LABELS
    return BROAD_AGE_BAND_LABELS if level == "county" else ACTUARIAL_AGE_BAND_LABELS


def required_rate_rows(admin: dict) -> set[tuple[str, str, int, str, str]]:
    """Every (estimand, level, unit, sex, band) a complete version-four release carries."""
    units = {"state": int(admin["n_states"]), "county": int(admin["n_counties"])}
    required: set[tuple[str, str, int, str, str]] = set()
    for estimand in RATE_ESTIMANDS:
        bands = (EXPOSURE_BAND_LABELS if estimand == EXPOSURE_ESTIMAND
                 else ACTUARIAL_AGE_BAND_LABELS)
        for level in RATE_LEVELS:
            for u in range(units[level]):
                for sex in SEX_LABELS:
                    for band in bands:
                        required.add((estimand, level, u, sex, band))
    return required


def check_rate_additivity(parsed: dict, admin: dict,
                          tolerance: float = 1e-6) -> list[str]:
    """Exposure adds; rates and quantiles do not, and are exempt by kind.

    Three requirements. Two are on the quantity that adds: a broad band equals the sum of
    the actuarial bands inside it, and a state equals the sum of its counties. The third is
    the ratio-consistency check that mortality and incidence take instead, because a rate
    is a ratio and reconciling it arithmetically would be wrong: a published rate times its
    own published exposure is an event count, and a state's event count has to equal the
    sum of its counties' event counts. A submission that files a state rate its own county
    rates and exposures contradict is stating two different numbers of deaths.
    """
    errors: list[str] = []
    county_state = np.asarray(admin["county_state"], dtype=np.int64)
    for level in RATE_LEVELS:
        units = int(admin["n_states"]) if level == "state" else int(admin["n_counties"])
        for u in range(units):
            for sex in SEX_LABELS:
                for broad, members in zip(BROAD_AGE_BAND_LABELS, BROAD_BAND_MEMBERS):
                    if len(members) == 1:
                        continue
                    key = (EXPOSURE_ESTIMAND, level, u, sex, broad)
                    parts = [(EXPOSURE_ESTIMAND, level, u, sex,
                              ACTUARIAL_AGE_BAND_LABELS[m]) for m in members]
                    if key not in parsed or any(k not in parsed for k in parts):
                        continue
                    total = sum(parsed[k][0] for k in parts)
                    stated = parsed[key][0]
                    if abs(total - stated) > tolerance * max(1.0, abs(stated)):
                        errors.append(f"{EXPOSURE_ESTIMAND}: {level} {u} {sex} {broad} "
                                      f"states {stated}, its bands sum to {total}")
    for s in range(int(admin["n_states"])):
        members = np.flatnonzero(county_state == s)
        for sex in SEX_LABELS:
            for band in ACTUARIAL_AGE_BAND_LABELS:
                key = (EXPOSURE_ESTIMAND, "state", s, sex, band)
                parts = [(EXPOSURE_ESTIMAND, "county", int(c), sex, band) for c in members]
                if key not in parsed or any(k not in parsed for k in parts):
                    continue
                total = sum(parsed[k][0] for k in parts)
                stated = parsed[key][0]
                if abs(total - stated) > tolerance * max(1.0, abs(stated)):
                    errors.append(f"{EXPOSURE_ESTIMAND}: counties of state {s} {sex} {band} "
                                  f"sum to {total}, state says {stated}")
    for estimand in (MORTALITY_ESTIMAND, INCIDENCE_ESTIMAND):
        for s in range(int(admin["n_states"])):
            members = np.flatnonzero(county_state == s)
            for sex in SEX_LABELS:
                for band in ACTUARIAL_AGE_BAND_LABELS:
                    rate_key = (estimand, "state", s, sex, band)
                    exposure_key = (EXPOSURE_ESTIMAND, "state", s, sex, band)
                    parts = [((estimand, "county", int(c), sex, band),
                              (EXPOSURE_ESTIMAND, "county", int(c), sex, band))
                             for c in members]
                    keys = [rate_key, exposure_key] + [k for pair in parts for k in pair]
                    if any(k not in parsed for k in keys):
                        continue
                    stated = parsed[rate_key][0] * parsed[exposure_key][0]
                    total = sum(parsed[r][0] * parsed[e][0] for r, e in parts)
                    if abs(total - stated) > tolerance * max(1.0, abs(stated)):
                        errors.append(
                            f"{estimand}: state {s} {sex} {band} implies {stated} events "
                            f"on its own exposure, its counties imply {total}")
    return errors


# ------------------------------------------------------------------ submission parsing

def parse_rate_rows(rows: list[dict], admin: dict) -> tuple[dict, list[str]]:
    """Parse and check the exposure and rate block of a release or projection table."""
    errors: list[str] = []
    parsed: dict[tuple[str, str, int, str, str], tuple[float, float, float]] = {}
    counts: dict[tuple[str, str, int, str, str], int] = {}
    bands = set(ACTUARIAL_AGE_BAND_LABELS) | set(BROAD_AGE_BAND_LABELS)
    for i, row in enumerate(rows):
        estimand, level = str(row["estimand"]), str(row["level"])
        if estimand not in RATE_ESTIMANDS:
            errors.append(f"rate row {i}: unknown estimand {estimand!r}")
            continue
        if level not in ("state", "county"):
            errors.append(f"rate row {i}: rates are published at state and county only")
            continue
        sex, band = str(row["sex"]), str(row["age_band"])
        if sex not in SEX_LABELS or band not in bands:
            errors.append(f"rate row {i}: unknown sex {sex!r} or band {band!r}")
            continue
        values = []
        for column in ("estimate", "lower", "upper"):
            v = row[column]
            if isinstance(v, bool) or not isinstance(v, (int, float, np.integer, np.floating)) \
                    or not math.isfinite(float(v)):
                errors.append(f"rate row {i}: {column} is not a finite number")
                break
            values.append(float(v))
        if len(values) < 3:
            continue
        estimate, lower, upper = values
        if not lower <= estimate <= upper:
            errors.append(f"rate row {i}: interval does not contain the estimate")
            continue
        if lower < 0.0:
            errors.append(f"rate row {i}: negative lower bound")
            continue
        unit = row["unit"]
        limit = int(admin["n_states"] if level == "state" else admin["n_counties"])
        if isinstance(unit, bool) or not isinstance(unit, (int, np.integer)) \
                or not 0 <= int(unit) < limit:
            errors.append(f"rate row {i}: unit is not a known {level} integer")
            continue
        key = (estimand, level, int(unit), sex, band)
        counts[key] = counts.get(key, 0) + 1
        parsed[key] = (estimate, lower, upper)
    required = required_rate_rows(admin)
    for key, n in counts.items():
        if n > 1:
            errors.append(f"rate row {key} appears {n} times")
        if key not in required:
            errors.append(f"unexpected rate row {key}")
    for key in sorted(required - set(counts)):
        errors.append(f"missing rate row {key}")
    return parsed, errors


def parse_reserve_rows(rows: list[dict], n_regions: int) -> tuple[dict, list[str]]:
    """Parse the reserve file: region, liability mean, q95, ES95, allocated reserve."""
    errors: list[str] = []
    mean = np.full(n_regions, np.nan)
    q = np.full(n_regions, np.nan)
    es = np.full(n_regions, np.nan)
    allocation = np.full(n_regions, np.nan)
    seen: set[int] = set()
    for i, row in enumerate(rows):
        if set(row) != set(RESERVE_COLUMNS):
            errors.append(f"reserve row {i}: columns {sorted(row)} differ from "
                          f"{list(RESERVE_COLUMNS)}")
            continue
        region = row["region"]
        if isinstance(region, bool) or not isinstance(region, (int, np.integer)) \
                or not 0 <= int(region) < n_regions:
            errors.append(f"reserve row {i}: region is not a known region index")
            continue
        region = int(region)
        if region in seen:
            errors.append(f"reserve row {i}: region {region} appears more than once")
            continue
        values = []
        for column in ("liability_mean", "q95", "es95", "allocation"):
            v = row[column]
            if isinstance(v, bool) or not isinstance(v, (int, float, np.integer, np.floating)) \
                    or not math.isfinite(float(v)) or float(v) < 0.0:
                errors.append(f"reserve row {i}: {column} is not a finite non-negative number")
                break
            values.append(float(v))
        if len(values) < 4:
            continue
        seen.add(region)
        mean[region], q[region], es[region], allocation[region] = values
        if q[region] < mean[region]:
            errors.append(f"reserve row {i}: q95 below the liability mean")
        if es[region] < q[region]:
            errors.append(f"reserve row {i}: expected shortfall below q95")
    for region in sorted(set(range(n_regions)) - seen):
        errors.append(f"missing reserve row for region {region}")
    return {"liability_mean": mean, "q95": q, "es95": es, "allocation": allocation}, errors


# ---------------------------------------------------------------- exposure and rates

def relative_error(estimate: float, truth: float, stabilizer: float) -> float:
    """e = |x_hat - x*| / (x* + c_x) with a frozen, quantity-specific stabilizer.

    The release's count stabilizer of one is useless on a rate of order 1e-3, which is why
    this is a separate function and not a reuse of the release scorer.
    """
    if stabilizer <= 0:
        raise ValueError("the stabilizer must be positive")
    return abs(float(estimate) - float(truth)) / (float(truth) + stabilizer)


def interval_score(lower: float, upper: float, truth: float, scale: float,
                   alpha: float = 0.10) -> float:
    """Gneiting and Raftery interval score, normalized by a frozen scale."""
    score = (upper - lower) + (2.0 / alpha) * max(lower - truth, 0.0) \
        + (2.0 / alpha) * max(truth - upper, 0.0)
    return float(score) / float(scale)


def _empirical_quantile_1d(values: Sequence[float], level: float) -> float:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not len(array) or not np.isfinite(array).all():
        raise ValueError("an empirical quantile needs a non-empty finite vector")
    if not 0.0 < float(level) <= 1.0:
        raise ValueError("an empirical quantile level must lie in (0, 1]")
    rank = int(math.ceil(float(level) * len(array)))
    return float(np.sort(array)[rank - 1])


def _broad_exposure(truth: dict, level: str, unit: int, sex: str,
                    band: str) -> float | None:
    key = (EXPOSURE_ESTIMAND, level, unit, sex, band)
    if key in truth:
        return float(truth[key])
    members = BROAD_BAND_MEMBERS[BROAD_AGE_BAND_LABELS.index(band)]
    values = [truth.get((EXPOSURE_ESTIMAND, level, unit, sex,
                         ACTUARIAL_AGE_BAND_LABELS[m])) for m in members]
    if any(value is None for value in values):
        return None
    return float(sum(float(value) for value in values))


def _broad_truth_rate(truth: dict, estimand: str, level: str, unit: int,
                      sex: str, band: str) -> float | None:
    key = (estimand, level, unit, sex, band)
    if key in truth and math.isfinite(float(truth[key])):
        return float(truth[key])
    members = BROAD_BAND_MEMBERS[BROAD_AGE_BAND_LABELS.index(band)]
    numerator, denominator = 0.0, 0.0
    for member in members:
        fine = ACTUARIAL_AGE_BAND_LABELS[member]
        exposure = truth.get((EXPOSURE_ESTIMAND, level, unit, sex, fine))
        rate = truth.get((estimand, level, unit, sex, fine))
        if exposure is None or rate is None or not math.isfinite(float(rate)):
            continue
        numerator += float(exposure) * float(rate)
        denominator += float(exposure)
    return numerator / denominator if denominator > 0.0 else None


def _broad_submitted_rate(parsed: dict, estimand: str, level: str, unit: int,
                          sex: str, band: str) -> tuple[float, float, float] | None:
    members = BROAD_BAND_MEMBERS[BROAD_AGE_BAND_LABELS.index(band)]
    numerator = np.zeros(3, dtype=np.float64)
    denominator = 0.0
    for member in members:
        fine = ACTUARIAL_AGE_BAND_LABELS[member]
        exposure_row = parsed.get((EXPOSURE_ESTIMAND, level, unit, sex, fine))
        rate_row = parsed.get((estimand, level, unit, sex, fine))
        if exposure_row is None or rate_row is None:
            return None
        exposure = float(exposure_row[0])
        if exposure < 0.0 or not math.isfinite(exposure):
            return None
        numerator += exposure * np.asarray(rate_row, dtype=np.float64)
        denominator += exposure
    if denominator <= 0.0:
        return None
    values = numerator / denominator
    return float(values[0]), float(values[1]), float(values[2])


def score_rates(parsed: dict, truth: dict, thresholds: ActuarialThresholds,
                alpha: float = 0.10) -> dict:
    """Score the exposure-and-rate surface and form its single pooled gate statistic.

    Eligibility is fixed from retained exposure before a submission is read. The scored
    cells are broad-band county exposures and broad-band state mortality and incidence.
    Submitted broad rates are reconstructed from the submitted fine rates and exposures;
    the fine blocks remain available as diagnostics but do not create more pass events.
    """
    stabilizer = {EXPOSURE_ESTIMAND: thresholds.exposure_stabilizer,
                  MORTALITY_ESTIMAND: thresholds.mortality_stabilizer,
                  INCIDENCE_ESTIMAND: thresholds.incidence_stabilizer}
    metrics: dict[str, dict] = {}
    pooled: list[tuple[float, bool, float, tuple[str, str, int, str, str]]] = []
    pooled_eligible: list[tuple[str, str, int, str, str]] = []
    for estimand in RATE_ESTIMANDS:
        gated_level, gated_bands = GATED_RATE_CELLS[estimand]
        for level in RATE_LEVELS:
            gated = level == gated_level
            cells: list[tuple[float, bool, float, tuple[int, str, str]]] = []
            eligible: list[tuple[int, str, str]] = []
            if gated:
                units = sorted({int(key[2]) for key in truth
                                if key[0] == EXPOSURE_ESTIMAND and key[1] == level})
                for unit in units:
                    for sex in SEX_LABELS:
                        for band in gated_bands:
                            exposure = _broad_exposure(truth, level, unit, sex, band)
                            if exposure is None or not math.isfinite(exposure) \
                                    or exposure < eligibility_floor(thresholds, band):
                                continue
                            eligible.append((unit, sex, band))
                            pooled_eligible.append((estimand, level, unit, sex, band))
                            if estimand == EXPOSURE_ESTIMAND:
                                value = exposure
                                submitted = parsed.get((estimand, level, unit, sex, band))
                            else:
                                value = _broad_truth_rate(truth, estimand, level, unit, sex, band)
                                submitted = _broad_submitted_rate(
                                    parsed, estimand, level, unit, sex, band)
                            if value is None or not math.isfinite(float(value)) or submitted is None:
                                continue
                            estimate, lower, upper = submitted
                            record = (
                                relative_error(estimate, float(value), stabilizer[estimand]),
                                bool(lower <= float(value) <= upper),
                                interval_score(lower, upper, float(value),
                                               float(value) + stabilizer[estimand], alpha),
                                (unit, sex, band),
                            )
                            cells.append(record)
                            pooled.append(record[:3] + ((estimand, level, unit, sex, band),))
            else:
                # Diagnostics use the directly published fine rows. They never set a bar.
                for key, value in truth.items():
                    if key[0] != estimand or key[1] != level or key not in parsed \
                            or key[4] not in ACTUARIAL_AGE_BAND_LABELS \
                            or not math.isfinite(float(value)):
                        continue
                    _, _, unit, sex, band = key
                    exposure = truth.get((EXPOSURE_ESTIMAND, level, unit, sex, band))
                    if exposure is None or float(exposure) < eligibility_floor(thresholds, band):
                        continue
                    estimate, lower, upper = parsed[key]
                    cells.append((relative_error(estimate, value, stabilizer[estimand]),
                                  bool(lower <= value <= upper),
                                  interval_score(lower, upper, value,
                                                 float(value) + stabilizer[estimand], alpha),
                                  (unit, sex, band)))
            name = f"{estimand}/{level}"
            if not cells:
                metrics[name] = {
                    "n_cells": 0, "n_eligible": len(eligible),
                    "percentile_error": float("nan"), "max_error": float("nan"),
                    "worst_cell": None, "coverage": float("nan"),
                    "mean_interval_score": float("nan"), "gated": gated,
                    "cells": [], "eligible_cells": sorted(eligible),
                    "raw_errors": [], "raw_covered": [], "raw_interval_scores": [],
                    "reason": "no submitted cell was scored from the fixed eligible set",
                }
                continue
            errors = [float(cell[0]) for cell in cells]
            worst = int(np.argmax(errors))
            metrics[name] = {
                "n_cells": len(cells), "n_eligible": len(eligible) if gated else len(cells),
                "cells": sorted(cell[3] for cell in cells),
                "eligible_cells": sorted(eligible) if gated else [], "reason": "",
                "percentile_error": _empirical_quantile_1d(
                    errors, thresholds.rate_gate_percentile),
                "max_error": float(max(errors)), "worst_cell": cells[worst][3],
                "coverage": float(np.mean([cell[1] for cell in cells])),
                "mean_interval_score": float(np.mean([cell[2] for cell in cells])),
                "raw_errors": errors,
                "raw_covered": [bool(cell[1]) for cell in cells],
                "raw_interval_scores": [float(cell[2]) for cell in cells],
                "gated": gated,
            }
    pooled_errors = [float(cell[0]) for cell in pooled]
    metrics["composite"] = {
        "gated": False,
        "n_cells": len(pooled),
        "n_eligible": len(pooled_eligible),
        "cells": [list(cell[3]) for cell in pooled],
        # This list is derived from retained exposure before submitted rows are looked up.
        # It therefore remains identical for a complete submission and one that omits rows.
        "eligible_cells": [list(cell) for cell in sorted(pooled_eligible)],
        "p95_relative_error": _empirical_quantile_1d(
            pooled_errors, thresholds.rate_gate_percentile) if pooled_errors else float("nan"),
        "coverage": float(np.mean([cell[1] for cell in pooled])) if pooled else float("nan"),
        "mean_interval_score": float(np.mean([cell[2] for cell in pooled]))
        if pooled else float("nan"),
        "raw_errors": pooled_errors,
        "raw_covered": [bool(cell[1]) for cell in pooled],
        "raw_interval_scores": [float(cell[2]) for cell in pooled],
        "reason": "" if pooled else "the fixed eligible set contains no scored cell",
    }
    return metrics


# -------------------------------------------------------------------------- tail gates

def exceedance_probabilities(q_hat: np.ndarray, liability: np.ndarray) -> np.ndarray:
    """p_r = mean_m 1{L_rm > q_hat_r}, the sealed exceedance rate of a submitted q95."""
    q_hat = np.asarray(q_hat, dtype=np.float64)
    liability = np.asarray(liability, dtype=np.float64)
    if q_hat.shape != (liability.shape[1],):
        raise ValueError("one submitted quantile per region is required")
    return (liability > q_hat[None, :]).mean(axis=0)


def calibration_criteria(p: np.ndarray, level: float = TAIL_LEVEL,
                         worst_quantile: float = 0.90) -> dict:
    """Pooled and worst-region deviation of the exceedance rate from its nominal 1 - level."""
    deviation = np.abs(np.asarray(p, dtype=np.float64) - (1.0 - level))
    return {"target": 1.0 - level, "pooled": float(deviation.mean()),
            "worst": float(np.quantile(deviation, worst_quantile, method="higher")),
            "deviation": deviation}


def quantile_score(q_hat: np.ndarray, liability: np.ndarray, scale: np.ndarray,
                   level: float = TAIL_LEVEL) -> np.ndarray:
    """QS(q, y) = (level - 1{y <= q})(y - q), averaged over continuations, normalized.

    ``scale`` is the preregistered regional scale, frozen on qualification worlds. A
    too-low q95 is punished through the exceedance term and a padded q95 through the
    (level - 1) branch, so neither direction buys safety.
    """
    q_hat = np.asarray(q_hat, dtype=np.float64)
    liability = np.asarray(liability, dtype=np.float64)
    scale = np.broadcast_to(np.asarray(scale, dtype=np.float64), (liability.shape[1],))
    if (scale <= 0).any():
        raise ValueError("the regional scale must be positive")
    indicator = (liability <= q_hat[None, :]).astype(np.float64)
    per_member = (level - indicator) * (liability - q_hat[None, :])
    return per_member.mean(axis=0) / scale


def width_relative_error(hat: np.ndarray, tail_truth: np.ndarray, mean_truth: np.ndarray,
                         min_width_fraction: float = 0.010) -> np.ndarray:
    """|hat - truth| per region, in units of the ensemble's own tail width.

    The width is the sealed tail statistic's distance above the sealed mean, which is the
    quantity a tail gate exists to ask for. Scoring the same error against the level
    instead lets a tail that is out by its entire width read as a fraction of a percent:
    on a regional width of about ten percent of the mean, a two times padded q95 and a
    mean-only q95 both sat inside version four's first tail bars. Both are out by one
    width, so both read 1.0 here.

    The divisor is held at a published minimum fraction of the regional mean, so a region
    whose ensemble is nearly degenerate cannot divide by zero.
    """
    hat = np.asarray(hat, dtype=np.float64)
    tail_truth = np.asarray(tail_truth, dtype=np.float64)
    mean_truth = np.asarray(mean_truth, dtype=np.float64)
    width = np.maximum(tail_truth - mean_truth,
                       float(min_width_fraction) * np.maximum(np.abs(mean_truth), 1.0))
    return np.abs(hat - tail_truth) / width


def shortfall_error(es_hat: np.ndarray, es_truth: np.ndarray,
                    scale: np.ndarray) -> np.ndarray:
    """Normalized error of a submitted expected shortfall against the ensemble truth."""
    es_hat = np.asarray(es_hat, dtype=np.float64)
    es_truth = np.asarray(es_truth, dtype=np.float64)
    scale = np.broadcast_to(np.asarray(scale, dtype=np.float64), es_truth.shape)
    return np.abs(es_hat - es_truth) / scale


# ------------------------------------------------------------ reserve and the decision

def reserve_total(exposure_person_years: float, rate_per_person_year: float,
                  rounding_unit: float = PLACEHOLDER_THRESHOLDS.reserve_rounding_unit) -> float:
    """Public reserve total from published exposure, a frozen rate, and round-up.

    No continuation quantile or expected shortfall enters this function. Packet creation
    recomputes ``exposure_person_years`` from the six-decimal participant CSV, so a reader
    can reproduce the total byte for byte from the files they receive.
    """
    exposure = float(exposure_person_years)
    rate = float(rate_per_person_year)
    unit = float(rounding_unit)
    if not all(math.isfinite(value) for value in (exposure, rate, unit)):
        raise ValueError("reserve-total inputs must be finite")
    if exposure < 0.0 or rate <= 0.0 or unit <= 0.0:
        raise ValueError("reserve exposure must be non-negative and rate and unit positive")
    # Decimal-from-string makes an exact multiple stay an exact multiple. Binary floating
    # point can represent 100 * 4.4 as 440.00000000000006 and incorrectly round it up.
    raw_units = (Decimal(str(exposure)) * Decimal(str(rate)) / Decimal(str(unit)))
    return float(raw_units.to_integral_value(rounding=ROUND_CEILING) * Decimal(str(unit)))


def proportional_baseline_allocation(share: np.ndarray, total: float) -> np.ndarray:
    """The frozen practical baseline A_B: the total split in proportion to ``share``.

    ``share`` is a public regional size, published in the packet contract as
    ``reserve.baseline_share``: the region's share of persons at or above the obligation's
    eligibility age in the revised population source. Holding a reserve in proportion to
    how many people it covers is what a practitioner does with no regional tail model, and
    it is the version-three proportional heuristic in its new place.

    The baseline is a fixed, public rule and never reads the submission. An earlier pass
    spread the slack above the submitted quantiles in proportion to those quantiles, which
    made A_B a function of the submission and left the skill scale measuring only how the
    sliver between sum q_hat and R was spread. On these worlds that sliver is a fraction of
    a percent of R, so the decision gate carried almost no information.
    """
    share = np.asarray(share, dtype=np.float64)
    share = np.where(np.isfinite(share) & (share > 0.0), share, 0.0)
    base = float(share.sum())
    if base <= 0.0:
        return np.full(share.shape, float(total) / len(share))
    return share * (float(total) / base)


def expected_uncovered(allocation: np.ndarray, liability: np.ndarray,
                       weights: np.ndarray | None = None) -> float:
    """J(A) = sum_r w_r mean_m (L_rm - A_r)_+, the sealed expected uncovered obligation."""
    allocation = np.asarray(allocation, dtype=np.float64)
    liability = np.asarray(liability, dtype=np.float64)
    w = np.ones(liability.shape[1]) if weights is None else \
        np.asarray(weights, dtype=np.float64)
    short = np.maximum(liability - allocation[None, :], 0.0).mean(axis=0)
    return float((w * short).sum())


def perfect_information_allocation(liability: np.ndarray, total: float,
                                   weights: np.ndarray | None = None) -> np.ndarray:
    """A*: the allocation of a fixed total that minimizes J against the sealed ensemble.

    J is separable, convex, and piecewise linear in A: raising A_r through the interval
    between the k-th and the (k+1)-th smallest continuation of region r buys
    w_r (M - k) / M per unit and no more. Sorting every such segment by its slope and
    filling greedily until the total is spent is therefore exact, not a search. Ties are
    broken by region index, which is neutral: two segments of equal slope trade one for
    one. Anything left after every paying segment is spent goes to region zero, where it
    changes nothing.

    The feasible set is the same one published to participants: finite nonnegative
    allocations whose sum is ``total``. Submitted quantiles do not enter either the
    oracle or the allocation constraint; the tail gate scores those forecasts directly.
    """
    liability = np.asarray(liability, dtype=np.float64)
    n_members, n_regions = liability.shape
    w = np.ones(n_regions) if weights is None else np.asarray(weights, dtype=np.float64)
    base = np.zeros(n_regions, dtype=np.float64)
    allocation = base.copy()
    remaining = float(total) - float(base.sum())
    if remaining <= 0:
        return allocation
    segments: list[tuple[float, int, int, float]] = []
    for r in range(n_regions):
        if w[r] <= 0:
            continue
        order = np.sort(liability[:, r])
        edge = float(base[r])
        for k, value in enumerate(order):
            width = float(value) - edge
            if width > 0:
                segments.append((w[r] * (n_members - k) / n_members, r, k, width))
                edge = float(value)
    segments.sort(key=lambda seg: (-seg[0], seg[1], seg[2]))
    for slope, r, _, width in segments:
        if remaining <= 0 or slope <= 0:
            break
        step = min(width, remaining)
        allocation[r] += step
        remaining -= step
    if remaining > 0:
        allocation[0] += remaining
    return allocation


def skill_score(j_submitted: float, j_baseline: float, j_oracle: float) -> float:
    """Skill(A) = (J(A_B) - J(A)) / (J(A_B) - J(A*)); one at the oracle, zero at A_B."""
    spread = float(j_baseline) - float(j_oracle)
    if spread <= 0:
        return float("nan")
    return (float(j_baseline) - float(j_submitted)) / spread


def score_reserve(allocation: np.ndarray, q_hat: np.ndarray, es_hat: np.ndarray,
                  mean_hat: np.ndarray, liability: np.ndarray, total: float,
                  thresholds: ActuarialThresholds = PLACEHOLDER_THRESHOLDS,
                  weights: np.ndarray | None = None,
                  scale: np.ndarray | None = None,
                  baseline_share: np.ndarray | None = None) -> dict:
    """Feasibility, tail diagnostics, and decision value for one reserve file.

    Allocation feasibility is independent of the submitted tail estimates. A legal
    decision is finite, nonnegative, and spends the public total. The tail gate scores
    q95 and ES95 directly against the continuation ensemble, so making q95 an allocation
    floor would count the same forecast twice and could make a truthful allocation
    impossible on a new world whose stand-alone regional quantiles sum above the fixed
    national reserve.
    """
    allocation = np.asarray(allocation, dtype=np.float64)
    q_hat = np.asarray(q_hat, dtype=np.float64)
    liability = np.asarray(liability, dtype=np.float64)
    truth = ensemble_truth(liability, TAIL_LEVEL)
    if scale is None:
        scale = np.maximum(truth["q"], 1.0)
    tolerance = thresholds.feasibility_tolerance

    reasons: list[str] = []
    if not np.isfinite(allocation).all():
        reasons.append("non-finite allocation")
    elif (allocation < -tolerance).any():
        reasons.append("negative allocation")
    if not math.isfinite(float(total)):
        reasons.append("non-finite reserve total")
    elif np.isfinite(allocation).all() and (
        abs(float(allocation.sum()) - float(total))
        > tolerance * max(1.0, abs(float(total)))
    ):
        reasons.append(f"allocations sum to {float(allocation.sum()):.6f}, not {float(total):.6f}")
    feasible = not reasons

    p = exceedance_probabilities(q_hat, liability)
    calibration = calibration_criteria(p, TAIL_LEVEL, thresholds.worst_region_quantile)
    qs = quantile_score(q_hat, liability, scale, TAIL_LEVEL)
    es_err = shortfall_error(es_hat, truth["es"], scale)
    mean_err = shortfall_error(mean_hat, truth["mean"], scale)
    fraction = thresholds.min_tail_width_fraction
    q_width_err = width_relative_error(q_hat, truth["q"], truth["mean"], fraction)
    es_width_err = width_relative_error(es_hat, truth["es"], truth["mean"], fraction)
    relative_width = (truth["q"] - truth["mean"]) / np.maximum(np.abs(truth["mean"]), 1.0)

    # A_B is the public size-proportional split of R and never reads the submission; A* is
    # the perfect-information allocation of the same total under the same constraints the
    # oracle faces, which is non-negativity and the total, not the submission's quantiles.
    #
    # The share comes from the contract, where it is published as a share of persons at or
    # above the eligibility age. With no share published the baseline splits R evenly,
    # which is still a published rule that no submission can move. Standing the baseline on
    # the submission's own q95, as the first pass did when the key was absent, put the
    # submission on both arms of the skill denominator: a padded tail then moved the
    # baseline it was scored against.
    baseline_source = "contract share" if baseline_share is not None else "even split"
    share = np.full(liability.shape[1], 1.0 / liability.shape[1]) \
        if baseline_share is None else np.asarray(baseline_share, dtype=np.float64)
    baseline = proportional_baseline_allocation(share, total)
    oracle = perfect_information_allocation(liability, total, weights)
    j = expected_uncovered(allocation, liability, weights) if feasible else float("nan")
    j_baseline = expected_uncovered(baseline, liability, weights)
    j_oracle = expected_uncovered(oracle, liability, weights)
    shortfall_probability = (liability > allocation[None, :]).mean(axis=0) if feasible \
        else np.full(liability.shape[1], np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        tail = np.maximum(liability - allocation[None, :], 0.0).mean(axis=0) / \
            np.maximum(truth["q"], 1.0) if feasible else np.full(liability.shape[1], np.nan)
    return {
        "feasible": feasible, "feasibility_reasons": reasons,
        "exceedance": p, "calibration": calibration,
        "quantile_score": qs, "mean_quantile_score": float(qs.mean()),
        "shortfall_error": es_err, "mean_shortfall_error": float(es_err.mean()),
        "mean_liability_error": float(mean_err.mean()),
        "q95_width_error": q_width_err,
        "mean_q95_width_error": float(q_width_err.mean()),
        "es95_width_error": es_width_err,
        "mean_es95_width_error": float(es_width_err.mean()),
        "ensemble_tail_width": relative_width,
        "J": j, "J_baseline": j_baseline, "J_oracle": j_oracle,
        "baseline_allocation": baseline, "baseline_source": baseline_source,
        "skill": skill_score(j, j_baseline, j_oracle) if feasible else float("nan"),
        "regional_shortfall_probability": shortfall_probability,
        "regional_tail": tail, "reserve_total": float(total),
    }


# ------------------------------------------------------------------------------ gates

def evaluate_actuarial_gates(rate_errors: list[str], rate_metrics: dict,
                             reserve_errors: list[str], reserve: dict | None,
                             thresholds: ActuarialThresholds = PLACEHOLDER_THRESHOLDS,
                             bars: dict | None = None) -> dict:
    """Combine the version-four checks into a verdict with ``family: detail`` reasons.

    Reason families are exposure, rate, coverage, tail, and reserve, matching the shape the
    freeze report splits on. A metric with no threshold is reported, never gated.
    """
    bars = bars or {}
    ceilings = {EXPOSURE_ESTIMAND: bars.get("exposure_error_ceiling",
                                            thresholds.exposure_error_ceiling),
                MORTALITY_ESTIMAND: bars.get("mortality_error_ceiling",
                                             thresholds.mortality_error_ceiling),
                INCIDENCE_ESTIMAND: bars.get("incidence_error_ceiling",
                                             thresholds.incidence_error_ceiling)}
    reasons: list[str] = []
    if rate_errors:
        reasons.append(f"rate schema: {len(rate_errors)} violation(s)")
    for key, m in sorted(rate_metrics.items()):
        if not m["gated"]:
            continue
        estimand = key.split("/")[0]
        family = "exposure" if estimand == EXPOSURE_ESTIMAND else "rate"
        # A gated block with no eligible cell decides nothing, and a verdict that reads a
        # shorter dictionary cannot tell that apart from a block that passed.
        if not int(m.get("n_cells", 0)):
            reasons.append(f"{family}: {key} has no eligible cell, so the gate reads "
                           "nothing")
            continue
        ceiling = ceilings[estimand]
        if ceiling is not None and m["percentile_error"] > ceiling:
            reasons.append(f"{family}: {key} percentile error "
                           f"{m['percentile_error']:.4f} > {ceiling}")
        floor = bars.get("rate_coverage_floor", thresholds.rate_coverage_floor)
        if floor is not None and m["coverage"] < floor:
            reasons.append(f"coverage: {key} {m['coverage']:.3f} < {floor}")
    if reserve_errors:
        reasons.append(f"reserve schema: {len(reserve_errors)} violation(s)")
    if reserve is not None:
        if not reserve["feasible"]:
            detail = "; ".join(reserve["feasibility_reasons"])
            reasons.append(f"reserve: infeasible ({detail})")
        else:
            calibration = reserve["calibration"]
            tau_mean = bars.get("tau_mean", thresholds.tau_mean)
            tau_worst = bars.get("tau_worst", thresholds.tau_worst)
            if tau_mean is not None and calibration["pooled"] > tau_mean:
                reasons.append(f"tail: pooled exceedance deviation "
                               f"{calibration['pooled']:.4f} > {tau_mean}")
            if tau_worst is not None and calibration["worst"] > tau_worst:
                reasons.append(f"tail: worst-region exceedance deviation "
                               f"{calibration['worst']:.4f} > {tau_worst}")
            score_ceiling = bars.get("quantile_score_ceiling", thresholds.quantile_score_ceiling)
            if score_ceiling is not None and reserve["mean_quantile_score"] > score_ceiling:
                reasons.append(f"tail: quantile score {reserve['mean_quantile_score']:.4f} "
                               f"> {score_ceiling}")
            es_ceiling = bars.get("es_error_ceiling", thresholds.es_error_ceiling)
            if es_ceiling is not None and reserve["mean_shortfall_error"] > es_ceiling:
                reasons.append(f"tail: expected shortfall error "
                               f"{reserve['mean_shortfall_error']:.4f} > {es_ceiling}")
            # The two width bars are what separate a tail from a level. Both criteria are
            # the error in units of the ensemble's own tail width, so a bar under one
            # refuses a submission that is out by a whole width in either direction.
            q_width_ceiling = bars.get("q95_width_error_ceiling",
                                       thresholds.q95_width_error_ceiling)
            if q_width_ceiling is not None \
                    and reserve["mean_q95_width_error"] > q_width_ceiling:
                reasons.append(f"tail: q95 error {reserve['mean_q95_width_error']:.4f} "
                               f"of the ensemble tail width > {q_width_ceiling}")
            es_width_ceiling = bars.get("es95_width_error_ceiling",
                                        thresholds.es95_width_error_ceiling)
            if es_width_ceiling is not None \
                    and reserve["mean_es95_width_error"] > es_width_ceiling:
                reasons.append(f"tail: ES95 error {reserve['mean_es95_width_error']:.4f} "
                               f"of the ensemble tail width > {es_width_ceiling}")
            minimum = bars.get("skill_minimum", thresholds.skill_minimum)
            if not math.isfinite(reserve["skill"]):
                reasons.append("reserve: skill is undefined because the oracle and the "
                               "frozen baseline carry the same uncovered obligation")
            elif minimum is not None and reserve["skill"] < minimum:
                reasons.append(f"reserve: skill {reserve['skill']:.4f} < {minimum}")
            shortfall_ceiling = bars.get("regional_shortfall_ceiling",
                                         thresholds.regional_shortfall_ceiling)
            worst = float(np.max(reserve["regional_shortfall_probability"]))
            if shortfall_ceiling is not None and worst > shortfall_ceiling:
                reasons.append(f"reserve: worst regional shortfall probability "
                               f"{worst:.4f} > {shortfall_ceiling}")
            tail_ceiling = bars.get("catastrophic_tail_ceiling",
                                    thresholds.catastrophic_tail_ceiling)
            worst_tail = float(np.max(reserve["regional_tail"]))
            if tail_ceiling is not None and worst_tail > tail_ceiling:
                reasons.append(f"reserve: worst regional tail {worst_tail:.4f} > {tail_ceiling}")
    return {"pass": not reasons, "reasons": reasons}
