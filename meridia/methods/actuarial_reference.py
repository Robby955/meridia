"""The actuarial layer both strong lines run.

Why this file exists: version four adds an inferential chain that neither strong line
contains today and that both must run the same way to stay comparable. Probabilistic
linkage without a shared key, exposure and event rates with a selection adjustment for
health-source inclusion, regime estimation from the historical experience file, a
simulated five-year liability distribution priced under the public contract, and a
reserve allocation that targets the regional tails. Folding it into design_based.py
would make the design-based line the owner of machinery the Bayesian witness also
needs; a copy in each file would let the two drift apart between releases. What stays
with each line is its own population reconstruction and its own uncertainty draws,
which is where the two statistical philosophies actually differ. What is shared here is
the actuarial arithmetic and the reading of the public contract.

Sources: Fellegi and Sunter (1969) and Sadinle (2017) for linkage; Rogan and Gladen
(1978) for prevalence under an imperfect test; Buhlmann and Straub (1970) credibility
for the partially pooled rates; Lee and Carter (1992) for the mortality drift; Gneiting
and Raftery (2007) for the scoring the tails are aimed at; Rockafellar and Uryasev
(2000) for the shortfall allocation. See docs/INDEPENDENCE.md.

Everything here reads participant files and public contract fields only.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

MAX_AGE = 100

# The frozen vocabulary is owned by meridia/actuarial.py: estimand names, band labels,
# the reserve columns, and the public obligation. It is imported, never restated, so a
# reference submission and the verifier cannot disagree about what a column is called.
from ..actuarial import (ACTUARIAL_AGE_BANDS, ACTUARIAL_AGE_BAND_LABELS,  # noqa: E402,F401
                         BROAD_AGE_BAND_LABELS, BROAD_BAND_MEMBERS,
                         EXPOSURE_BAND_LABELS, EXPOSURE_ESTIMAND, INCIDENCE_ESTIMAND,
                         MORTALITY_ESTIMAND, ObligationContract, RATE_ESTIMANDS,
                         RATE_EXTRA_COLUMNS, RATE_LEVELS, RESERVE_COLUMNS)
# Re-exported so a caller writing a version-four submission needs one import, not two.
SUBMISSION_VOCABULARY = (RATE_ESTIMANDS, RATE_EXTRA_COLUMNS, RATE_LEVELS,
                         EXPOSURE_BAND_LABELS, RESERVE_COLUMNS)
from ..release import SEX_LABELS                                          # noqa: E402

EXPERIENCE_COLUMNS = ("year", "age_band", "sex", "state", "exposure", "deaths",
                      "qualifying_events", "net_migration")
EXPERIENCE_FILENAMES = ("experience_history.csv", "experience.csv")

DEFAULT_ELIGIBLE_MIN_AGE = 65
LINK_FIELDS = ("given_code", "family_code", "birth_tick", "sex")

# Inclusion probability of a person with a recent admission in the health archive is
# never taken outside this range: below the floor the cell carries no usable signal and
# the anchor is used on its own, above one the comparison is noise.
INCLUSION_BOUNDS = (0.25, 1.0)
CREDIBILITY_FLOOR = 1e-9


class MissingActuarialInputs(RuntimeError):
    """Raised when a packet does not carry the version-four anchors."""


# ------------------------------------------------------------------ public contract

@dataclass(frozen=True)
class ActuarialContract:
    """What a participant is told about the obligation, the regions, and the reserve.

    Read from participant/contract.json only. ``reserve_total`` is protocol section 9's
    R, published because the submission must satisfy sum_r A_r = R exactly; it is the
    one field of this block the packet does not yet write, and the reader says so by
    name rather than guessing a value.
    """
    obligation: ObligationContract
    region_of_county: np.ndarray
    n_regions: int
    reserve_total: float
    reserve_weights: np.ndarray
    gamma: float
    anchor_item: str
    anchor_sensitivity: float
    anchor_specificity: float
    anchor_window_months: int
    experience_years: int
    experience_file: str
    experience_last_tick: int

    @property
    def horizon_months(self) -> int:
        return int(self.obligation.horizon_months)

    @property
    def n_years(self) -> int:
        return int(round(self.horizon_months / 12.0))

    @property
    def discount(self) -> np.ndarray:
        return self.obligation.discount_factors()

    @property
    def monthly_benefit(self) -> float:
        return float(self.obligation.monthly_benefit)

    @property
    def eligible_min_age(self) -> int:
        return int(self.obligation.eligibility_min_age)

    @property
    def first_event_cost(self) -> float:
        return float(self.obligation.qualifying_event_cost)

    @property
    def death_benefit(self) -> float:
        return float(self.obligation.death_benefit)

    @property
    def qualifying_groups(self) -> tuple[int, ...] | None:
        groups = self.obligation.qualifying_diagnosis_groups
        return tuple(groups) if groups else None


def _first(mapping: dict, names, default=None):
    for name in names:
        if isinstance(mapping, dict) and name in mapping and mapping[name] is not None:
            return mapping[name]
    return default


def read_actuarial_contract(contract: dict, county_state: np.ndarray) -> ActuarialContract:
    """Read the obligation, the health anchor, the experience block and the reserve.

    Regions are the states, which is what ``actuarial.regions_from_admin`` returns and
    what docs/V4_DECISIONS.md records as the default; a published county-level map in the
    contract overrides it.
    """
    obligation_block = _first(contract, ("obligation",), None)
    anchor = _first(contract, ("health_anchor", "anchor"), {}) or {}
    history = _first(contract, ("experience_history",), {}) or {}
    reserve = _first(contract, ("reserve",), {}) or {}
    missing = []
    if obligation_block is None:
        missing.append('contract["obligation"]')
    if not history:
        missing.append('contract["experience_history"]')
    total = _first(reserve, ("total", "R", "reserve_total"))
    if total is None:
        missing.append('contract["reserve"]["total"] (protocol section 9 R)')
    if missing:
        raise MissingActuarialInputs(
            "contract.json carries no complete actuarial block; missing "
            + "; ".join(missing))
    obligation = ObligationContract.from_public(obligation_block)
    region_spec = _first(reserve, ("regions", "region"), "state")
    if isinstance(region_spec, dict):
        region_spec = _first(region_spec, ("level", "of_county", "map"), "state")
    if isinstance(region_spec, (list, tuple, np.ndarray)):
        region_of_county = np.asarray(region_spec, dtype=np.int64)
    elif isinstance(region_spec, str) and region_spec.startswith("count"):
        region_of_county = np.arange(len(county_state), dtype=np.int64)
    else:
        region_of_county = np.asarray(county_state, dtype=np.int64)
    n_regions = int(region_of_county.max()) + 1
    weights = _first(reserve, ("weights", "w"), None)
    weights = np.ones(n_regions) if weights is None else np.asarray(weights, dtype=np.float64)
    return ActuarialContract(
        obligation=obligation,
        region_of_county=region_of_county,
        n_regions=n_regions,
        reserve_total=float(total),
        reserve_weights=weights,
        gamma=float(_first(reserve, ("gamma",), 0.25)),
        anchor_item=str(_first(anchor, ("item", "column"), "recent_hospitalization")),
        anchor_sensitivity=float(_first(anchor, ("sensitivity", "se"), 1.0)),
        anchor_specificity=float(_first(anchor, ("specificity", "sp"), 1.0)),
        anchor_window_months=int(_first(anchor, ("window_months", "window"), 12)),
        experience_years=int(_first(history, ("years",), 5)),
        experience_file=str(_first(history, ("file",), EXPERIENCE_FILENAMES[0])),
        experience_last_tick=int(_first(history, ("last_year_ends_at_tick",),
                                        contract["ticks"]["revised"])),
    )


def load_experience(packet_dir: Path, contract: dict | None = None):
    """The aggregate experience file, or None when the packet predates it."""
    import pandas as pd
    base = Path(packet_dir) / "participant"
    names = list(EXPERIENCE_FILENAMES)
    if contract:
        named = (contract.get("experience_history") or {}).get("file")
        if named:
            names = [named] + [n for n in names if n != named]
    for name in names:
        path = base / name
        if path.exists():
            frame = pd.read_csv(path)
            missing = [c for c in EXPERIENCE_COLUMNS if c not in frame.columns]
            if missing:
                raise MissingActuarialInputs(
                    f"{path.name} is missing columns {missing}; "
                    f"expected {list(EXPERIENCE_COLUMNS)}")
            return frame
    return None


def has_actuarial_inputs(packet_dir: Path) -> bool:
    """True when the packet carries the obligation, the reserve total, and the file."""
    try:
        contract = json.loads(
            (Path(packet_dir) / "participant" / "contract.json").read_text())
    except (OSError, ValueError):
        return False
    if load_experience(packet_dir, contract) is None:
        return False
    try:
        read_actuarial_contract(contract, np.zeros(1, dtype=np.int64))
    except MissingActuarialInputs:
        return False
    return True


# ------------------------------------------------------------- age-band arithmetic

def band_of_age(age: np.ndarray) -> np.ndarray:
    """Index into ACTUARIAL_AGE_BANDS for each attained age; -1 outside every band."""
    age = np.asarray(age)
    band = np.full(age.shape, -1, dtype=np.int64)
    for b, (lo, hi) in enumerate(ACTUARIAL_AGE_BANDS):
        band[(age >= lo) & (age <= hi)] = b
    return band


def band_matrix(max_age: int = MAX_AGE) -> np.ndarray:
    """(n_bands, max_age + 1) indicator, for collapsing single-year age arrays."""
    ages = np.arange(max_age + 1)
    band = band_of_age(ages)
    m = np.zeros((len(ACTUARIAL_AGE_BANDS), max_age + 1))
    for b in range(len(ACTUARIAL_AGE_BANDS)):
        m[b, band == b] = 1.0
    return m


# ------------------------------------------------------------ probabilistic linkage

def _candidate_pairs(left, right, left_key: str, right_key: str):
    """Candidate pairs blocked on sex and birth year, kept when at least one name code
    agrees. Two records that share neither name code are not worth comparing: the pair
    count would grow with the product of the files and the posterior for such a pair is
    far below any threshold the one-to-one reduction would ever reach.
    """
    import pandas as pd
    cols = ["given_code", "family_code", "birth_tick", "sex", "_county", "_id", "_block"]
    a = left[cols]
    b = right[cols]
    named = ["given_code", "family_code", "birth_tick", "sex", "_county", "_id"]
    pieces = []
    for on in ("given_code", "family_code"):
        merged = a.merge(b, on=["_block", on], how="inner", suffixes=("_l", "_r"))
        # The join column itself is not suffixed by the merge; restore both sides so
        # every piece carries the same column names before they are stacked.
        merged[f"{on}_l"] = merged[on]
        merged[f"{on}_r"] = merged[on]
        pieces.append(merged[[f"{c}_l" for c in named] + [f"{c}_r" for c in named]])
    pairs = pd.concat(pieces, ignore_index=True)
    return pairs.drop_duplicates(subset=["_id_l", "_id_r"], ignore_index=True)


def _agreement(pairs) -> np.ndarray:
    """Binary agreement on given name, family name, exact birth month, and county."""
    given = (pairs["given_code_l"].to_numpy() == pairs["given_code_r"].to_numpy())
    family = (pairs["family_code_l"].to_numpy() == pairs["family_code_r"].to_numpy())
    birth = (pairs["birth_tick_l"].to_numpy() == pairs["birth_tick_r"].to_numpy())
    county = (pairs["_county_l"].to_numpy() == pairs["_county_r"].to_numpy())
    return np.column_stack([given, family, birth, county]).astype(np.float64)


def fit_fellegi_sunter(agreement: np.ndarray, iterations: int = 80) -> dict:
    """Two-class mixture over agreement patterns, fitted by expectation maximisation.

    Conditional independence across the four comparisons, as in Fellegi and Sunter
    (1969). The u probabilities are estimated on the blocked candidate set rather than
    the full cross product, so they read as "agreement among plausible pairs" and the
    resulting weights are conservative: a genuine non-match inside a block agrees on
    one name by construction. That is the intended reading here, since only blocked
    pairs are ever scored.
    """
    n, k = agreement.shape
    if n == 0:
        return {"m": np.full(4, 0.9), "u": np.full(4, 0.1), "p": 0.1, "n_pairs": 0,
                "converged": False}
    m = np.clip(agreement.mean(axis=0) + 0.30, 0.05, 0.98)
    u = np.clip(agreement.mean(axis=0) - 0.20, 0.02, 0.95)
    p = 0.2
    posterior = np.full(n, p)
    for _ in range(iterations):
        log_m = agreement @ np.log(m) + (1.0 - agreement) @ np.log(1.0 - m)
        log_u = agreement @ np.log(u) + (1.0 - agreement) @ np.log(1.0 - u)
        top = np.log(max(p, 1e-9)) + log_m
        bottom = np.log(max(1.0 - p, 1e-9)) + log_u
        high = np.maximum(top, bottom)
        posterior = np.exp(top - high) / (np.exp(top - high) + np.exp(bottom - high))
        weight = posterior.sum()
        if weight < 1.0 or (n - weight) < 1.0:
            break
        new_m = np.clip((posterior[:, None] * agreement).sum(axis=0) / weight, 0.02, 0.995)
        new_u = np.clip(((1.0 - posterior)[:, None] * agreement).sum(axis=0) / (n - weight),
                        0.005, 0.98)
        new_p = float(np.clip(weight / n, 1e-4, 0.999))
        shift = float(np.abs(new_m - m).max() + np.abs(new_u - u).max() + abs(new_p - p))
        m, u, p = new_m, new_u, new_p
        if shift < 1e-8:
            return {"m": m, "u": u, "p": p, "n_pairs": int(n), "converged": True,
                    "posterior": posterior}
    return {"m": m, "u": u, "p": float(p), "n_pairs": int(n), "converged": False,
            "posterior": posterior}


def _one_to_one(pairs, posterior: np.ndarray):
    """Greedy one-to-one reduction: highest posterior first, each record used once."""
    order = np.argsort(-posterior, kind="stable")
    left_id = pairs["_id_l"].to_numpy()[order]
    right_id = pairs["_id_r"].to_numpy()[order]
    seen_left, seen_right = set(), set()
    keep = np.zeros(len(order), dtype=bool)
    for j in range(len(order)):
        a, b = left_id[j], right_id[j]
        if a in seen_left or b in seen_right:
            continue
        seen_left.add(a)
        seen_right.add(b)
        keep[j] = True
    return order[keep]


def probabilistic_links(left, right, left_id: str, right_id: str,
                        left_county: str = "county", right_county: str = "county",
                        max_pairs: int = 4_000_000) -> dict:
    """Link two sources that share no identifier, and keep the match probability.

    Returns the one-to-one reduced link set with a posterior probability on each pair.
    Downstream estimates draw a Bernoulli link indicator from that probability rather
    than treating the link set as certain, which is what carries linkage uncertainty
    into the rates and, through them, into the liability distribution.
    """
    import pandas as pd
    def prepare(frame, id_column, county_column):
        out = frame.copy()
        out["_id"] = out[id_column].to_numpy()
        out["_county"] = out[county_column].to_numpy() if county_column in out.columns \
            else np.full(len(out), -1, dtype=np.int64)
        out["_block"] = (out["birth_tick"].to_numpy(dtype=np.int64) // 12) * 2 + \
            out["sex"].to_numpy(dtype=np.int64)
        named = (out["given_code"].to_numpy() > 0) & (out["family_code"].to_numpy() > 0)
        return out[named].drop_duplicates(subset=["_id"])

    a = prepare(left, left_id, left_county)
    b = prepare(right, right_id, right_county)
    empty = {"links": pd.DataFrame(columns=["_id_l", "_id_r", "p_match"]),
             "fit": None, "n_candidates": 0}
    if len(a) == 0 or len(b) == 0:
        return empty
    pairs = _candidate_pairs(a, b, left_id, right_id)
    if pairs is None or len(pairs) == 0:
        return empty
    if len(pairs) > max_pairs:
        # A block this dense carries no discriminating information; keeping the pairs
        # would cost memory for weights that are all near the prior.
        pairs = pairs.iloc[:max_pairs]
    agreement = _agreement(pairs)
    fit = fit_fellegi_sunter(agreement)
    posterior = fit.get("posterior")
    if posterior is None:
        posterior = np.full(len(pairs), fit["p"])
    keep = _one_to_one(pairs, posterior)
    links = pairs.iloc[keep][["_id_l", "_id_r"]].copy()
    links["p_match"] = posterior[keep]
    return {"links": links.reset_index(drop=True), "fit": fit, "n_candidates": int(len(pairs))}


def sample_link_indicator(p_match: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """One imputation of the link set: keep each pair with its match probability."""
    p = np.asarray(p_match, dtype=np.float64)
    return rng.random(len(p)) < p


# ------------------------------------------------- health-source selection adjustment

def rogan_gladen(observed: np.ndarray, sensitivity: float, specificity: float) -> np.ndarray:
    """True prevalence behind an item with declared sensitivity and specificity.

    p_true = (p_obs + specificity - 1) / (sensitivity + specificity - 1). Undefined when
    the item carries no information (sensitivity + specificity = 1), where the observed
    share is returned unchanged.
    """
    observed = np.asarray(observed, dtype=np.float64)
    youden = sensitivity + specificity - 1.0
    if abs(youden) < 1e-6:
        return observed
    return np.clip((observed + specificity - 1.0) / youden, 0.0, 1.0)


def anchor_prevalence(survey, ac: ActuarialContract, county_state: np.ndarray) -> dict:
    """Recent-admission prevalence from the independent survey item, by state, sex and
    band, corrected for the item's declared error rates.

    The survey sample is drawn without reference to health-source inclusion, so this is
    an external anchor rather than a restatement of the archive. Its sampling variance
    comes from the design weights through the effective sample size, so a thin cell is
    shrunk toward its state margin rather than believed.
    """
    n_states = int(county_state.max()) + 1
    n_bands = len(ACTUARIAL_AGE_BANDS)
    shape = (n_states, n_bands, 2)
    prevalence = np.full(shape, np.nan)
    effective_n = np.zeros(shape)
    if ac.anchor_item not in survey.columns:
        return {"prevalence": prevalence, "effective_n": effective_n, "available": False}
    frame = survey[np.isfinite(survey[ac.anchor_item].to_numpy(dtype=np.float64))]
    if len(frame) == 0:
        return {"prevalence": prevalence, "effective_n": effective_n, "available": False}
    weight_column = "weight" if "weight" in frame.columns else "design_weight"
    w = frame[weight_column].to_numpy(dtype=np.float64)
    y = frame[ac.anchor_item].to_numpy(dtype=np.float64)
    state = county_state[frame["county"].to_numpy(dtype=np.int64)]
    band = band_of_age(frame["age"].to_numpy(dtype=np.int64))
    sex = frame["sex"].to_numpy(dtype=np.int64)
    ok = band >= 0
    flat = (state[ok] * n_bands + band[ok]) * 2 + sex[ok]
    size = n_states * n_bands * 2
    total_w = np.bincount(flat, weights=w[ok], minlength=size)
    total_w2 = np.bincount(flat, weights=w[ok] ** 2, minlength=size)
    total_wy = np.bincount(flat, weights=w[ok] * y[ok], minlength=size)
    with np.errstate(invalid="ignore", divide="ignore"):
        observed = np.where(total_w > 0, total_wy / np.maximum(total_w, 1e-9), np.nan)
        kish = np.where(total_w2 > 0, total_w ** 2 / np.maximum(total_w2, 1e-9), 0.0)
    prevalence = rogan_gladen(np.nan_to_num(observed, nan=0.0),
                              ac.anchor_sensitivity, ac.anchor_specificity).reshape(shape)
    prevalence[np.isnan(observed).reshape(shape)] = np.nan
    return {"prevalence": prevalence, "effective_n": kish.reshape(shape),
            "observed": observed.reshape(shape), "available": True}


def archive_recent_counts(health, county_state: np.ndarray, tick: int,
                          window_months: int, qualifying_groups=None) -> np.ndarray:
    """Distinct patients the health archive shows admitted inside the window, by county,
    band and sex.

    One row per patient inside the window, which is also the archive's view of a first
    qualifying event under the window convention of docs/V4_DECISIONS.md: the event
    scored is a person's first inside the window, not their first ever, because pre-window
    health history is not among the files a participant receives.
    """
    n_counties = len(county_state)
    n_bands = len(ACTUARIAL_AGE_BANDS)
    frame = health
    if qualifying_groups is not None and "diagnosis_group" in frame.columns:
        frame = frame[frame["diagnosis_group"].isin(list(qualifying_groups))]
    admission = frame["admission_tick"].to_numpy(dtype=np.int64)
    inside = (admission > tick - window_months) & (admission <= tick)
    frame = frame[inside].drop_duplicates("patient_id")
    county = frame["patient_county"].to_numpy(dtype=np.int64)
    band = band_of_age((tick - frame["birth_tick"].to_numpy(dtype=np.int64)) // 12)
    sex = frame["sex"].to_numpy(dtype=np.int64)
    ok = (county >= 0) & (county < n_counties) & (band >= 0) & (sex >= 0) & (sex < 2)
    flat = (county[ok] * n_bands + band[ok]) * 2 + sex[ok]
    return np.bincount(flat, minlength=n_counties * n_bands * 2).reshape(
        (n_counties, n_bands, 2)).astype(float)


def inclusion_probability(archive_count: np.ndarray, anchor: dict,
                          population: np.ndarray) -> dict:
    """Health-source inclusion probability for a person with a recent admission.

    The anchor says how many people in a cell were admitted; the archive says how many
    of them it holds. Their ratio is the inclusion probability. The pooling is a ratio
    of sums rather than a mean of ratios: cell ratios have a small random denominator
    and are right-skewed, so averaging them would bias the pooled inclusion upward and
    understate the correction the incidence rates need. Each cell is then shrunk toward
    that pooled value by its own expected count, so a band with four expected admissions
    does not carry its own selection rate.
    """
    expected = np.asarray(anchor["prevalence"], dtype=np.float64) * \
        np.asarray(population, dtype=np.float64)
    archive_count = np.asarray(archive_count, dtype=np.float64)
    usable = np.isfinite(expected) & (expected > 0)
    if not usable.any():
        return {"pi": np.ones_like(archive_count), "pooled": 1.0, "available": False,
                "expected": expected}
    pooled = float(np.clip(archive_count[usable].sum() / max(expected[usable].sum(), 1e-9),
                           *INCLUSION_BOUNDS))
    with np.errstate(invalid="ignore", divide="ignore"):
        raw = np.where(usable, archive_count / np.maximum(expected, 1e-9), np.nan)
    # Half weight on a cell's own ratio at twenty-five expected admissions: below that
    # the Poisson noise in the numerator dominates the between-cell spread.
    z = np.where(usable, expected / (expected + 25.0), 0.0)
    pi = z * np.nan_to_num(raw, nan=pooled) + (1.0 - z) * pooled
    return {"pi": np.clip(pi, *INCLUSION_BOUNDS), "pooled": pooled, "available": True,
            "raw": raw, "expected": expected}


# ---------------------------------------------------- historical experience and regime

def experience_arrays(frame, n_states: int) -> dict:
    """The experience file as arrays indexed [year, state, band, sex].

    Years are indexed from the earliest in the file. A band or state absent from the
    file is left at zero exposure and is skipped by every estimator downstream.
    """
    n_bands = len(ACTUARIAL_AGE_BANDS)
    band_index = {label: b for b, label in enumerate(ACTUARIAL_AGE_BAND_LABELS)}
    sex_index = {label: s for s, label in enumerate(SEX_LABELS)}
    years = np.sort(frame["year"].unique())
    year_index = {int(y): i for i, y in enumerate(years)}
    shape = (len(years), n_states, n_bands, 2)
    out = {k: np.zeros(shape) for k in ("exposure", "deaths", "qualifying_events",
                                        "net_migration")}
    raw_band = frame["age_band"].to_numpy()
    raw_sex = frame["sex"].to_numpy()
    for i in range(len(frame)):
        b = band_index.get(str(raw_band[i]))
        s = sex_index.get(str(raw_sex[i]))
        if b is None or s is None:
            continue
        y = year_index[int(frame["year"].iat[i])]
        st = int(frame["state"].iat[i])
        if not (0 <= st < n_states):
            continue
        for key in out:
            out[key][y, st, b, s] += float(frame[key].iat[i])
    out["years"] = years
    return out


def estimate_improvement(exposure: np.ndarray, counts: np.ndarray,
                         min_exposure: float = 500.0) -> dict:
    """Annual log drift in a rate, pooled over states, bands and sexes.

    A single drift with a free intercept per band, sex and state, which is the Lee and
    Carter (1992) reduction when the age response b_x is held flat. Five annual points
    fix a level and a slope; they do not fix an age-varying response, and pretending
    otherwise would put a spurious age pattern into the projection.

    Cells are weighted by their event count, not their exposure. The sampling variance
    of a log rate is one over the count, so a young band with large exposure and twenty
    deaths carries the noisiest observation in the file; weighting it by exposure would
    hand the drift to exactly the cells that cannot measure it.
    """
    n_years = exposure.shape[0]
    if n_years < 3:
        return {"drift": 0.0, "drift_se": 0.02, "fitted": False, "n_cells": 0}
    ok = (exposure >= min_exposure) & (counts > 0)
    if ok.sum() < 8:
        return {"drift": 0.0, "drift_se": 0.02, "fitted": False, "n_cells": int(ok.sum())}
    log_rate = np.full(exposure.shape, np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        log_rate[ok] = np.log(counts[ok] / exposure[ok])
    year = np.arange(n_years, dtype=np.float64)
    year = year - year.mean()
    numerator = 0.0
    denominator = 0.0
    residual = []
    weights = []
    cells = 0
    for index in np.ndindex(exposure.shape[1:]):
        column = log_rate[(slice(None),) + index]
        w = np.where(ok[(slice(None),) + index], counts[(slice(None),) + index], 0.0)
        if (w > 0).sum() < 3:
            continue
        cells += 1
        mean_y = float((w * np.nan_to_num(column)).sum() / w.sum())
        mean_x = float((w * year).sum() / w.sum())
        numerator += float((w * (year - mean_x) * (np.nan_to_num(column) - mean_y)).sum())
        denominator += float((w * (year - mean_x) ** 2).sum())
        residual.append((np.nan_to_num(column) - mean_y, year - mean_x, w))
        weights.append(w.sum())
    if denominator <= 0 or cells == 0:
        return {"drift": 0.0, "drift_se": 0.02, "fitted": False, "n_cells": cells}
    drift = numerator / denominator
    sse = sum(float((w * (r - drift * x) ** 2).sum()) for r, x, w in residual)
    dof = max(sum(int((w > 0).sum()) for _, _, w in residual) - 2 * cells, 1)
    sigma2 = sse / dof
    drift_se = float(np.sqrt(max(sigma2 / denominator, 1e-12)))
    return {"drift": float(np.clip(drift, -0.15, 0.15)),
            "drift_se": float(min(max(drift_se, 1e-4), 0.05)),
            "fitted": True, "n_cells": cells}


def estimate_migration_profile(experience: dict) -> dict:
    """Net migration rate by state, band and sex, averaged over the file's years, with
    the between-year spread as its standard error. This is the age-patterned migration
    axis of the regime family, read where it is identified."""
    exposure = experience["exposure"]
    net = experience["net_migration"]
    with np.errstate(invalid="ignore", divide="ignore"):
        rate = np.where(exposure > 0, net / np.maximum(exposure, 1e-9), 0.0)
    mean = rate.mean(axis=0)
    spread = rate.std(axis=0, ddof=1) if rate.shape[0] > 1 else np.zeros_like(mean)
    se = spread / np.sqrt(max(rate.shape[0], 1))
    return {"rate": mean, "se": np.maximum(se, 1e-5),
            "national": float(np.average(mean, weights=np.maximum(exposure.mean(axis=0), 1e-9)))}


# ----------------------------------------------------------- partially pooled rates

def buhlmann_straub(counts: np.ndarray, exposure: np.ndarray) -> dict:
    """Credibility weights for Poisson counts over cells of unequal exposure.

    Expected process variance is the overall rate itself (one Poisson count per unit of
    exposure); the variance of hypothetical means is what is left of the between-cell
    spread after that. The credibility factor Z = E / (E + k) with k = EPV / VHM is the
    partial pooling: a county with a hundred person-years keeps almost none of its own
    rate, a state with a hundred thousand keeps almost all of it.
    """
    counts = np.asarray(counts, dtype=np.float64).ravel()
    exposure = np.asarray(exposure, dtype=np.float64).ravel()
    ok = exposure > 0
    total_exposure = float(exposure[ok].sum())
    if total_exposure <= 0 or ok.sum() < 2:
        overall = float(counts.sum() / max(total_exposure, 1e-9))
        return {"overall": overall, "k": np.inf, "vhm": 0.0, "z": np.zeros(len(counts))}
    overall = float(counts[ok].sum() / total_exposure)
    rate = np.zeros(len(counts))
    rate[ok] = counts[ok] / exposure[ok]
    n_cells = int(ok.sum())
    spread = float((exposure[ok] * (rate[ok] - overall) ** 2).sum())
    correction = overall * (n_cells - 1)
    denominator = total_exposure - float((exposure[ok] ** 2).sum()) / total_exposure
    vhm = (spread - correction) / max(denominator, 1e-9)
    if not np.isfinite(vhm) or vhm <= CREDIBILITY_FLOOR:
        return {"overall": overall, "k": np.inf, "vhm": 0.0, "z": np.zeros(len(counts))}
    k = overall / vhm
    z = exposure / (exposure + k)
    return {"overall": overall, "k": float(k), "vhm": float(vhm), "z": z}


def credibility_rate(counts: np.ndarray, exposure: np.ndarray, prior: float | None = None) -> dict:
    """Partially pooled rate per cell with a 90 percent interval.

    The posterior for a Poisson count with a gamma prior of shape k*prior and rate k is
    gamma(count + k*prior, exposure + k), whose mean is exactly the credibility estimate
    and whose quantiles give the interval. When the between-cell spread does not survive
    the process variance, every cell collapses onto the pooled rate and the interval is
    the pooled one, which is the honest statement in that case.
    """
    counts = np.asarray(counts, dtype=np.float64)
    exposure = np.asarray(exposure, dtype=np.float64)
    shape_in = counts.shape
    fit = buhlmann_straub(counts, exposure)
    overall = fit["overall"] if prior is None else float(prior)
    k = fit["k"]
    flat_counts = counts.ravel()
    flat_exposure = exposure.ravel()
    if not np.isfinite(k):
        point = np.full(flat_counts.shape, overall)
        alpha = np.full(flat_counts.shape, max(flat_counts.sum(), 1.0))
        beta = np.full(flat_counts.shape, max(flat_exposure.sum(), 1e-9))
    else:
        alpha = flat_counts + k * overall
        beta = flat_exposure + k
        point = alpha / np.maximum(beta, 1e-12)
    lower, upper = _gamma_interval(alpha, beta)
    return {"rate": point.reshape(shape_in), "lower": lower.reshape(shape_in),
            "upper": upper.reshape(shape_in), "k": k, "overall": overall,
            "z": fit["z"].reshape(shape_in) if np.ndim(fit["z"]) else fit["z"]}


def _gamma_interval(alpha: np.ndarray, beta: np.ndarray, level: float = 0.90):
    """Equal-tailed gamma interval, by the Wilson-Hilferty cube-root approximation so
    the module keeps to numpy. The approximation is within a percent of the exact gamma
    quantile for shapes above one, which every cell that clears the exposure floor has.
    """
    tail = 0.5 * (1.0 - level)
    z_lo, z_hi = -1.6448536269514722, 1.6448536269514722
    alpha = np.maximum(np.asarray(alpha, dtype=np.float64), 1e-6)
    beta = np.maximum(np.asarray(beta, dtype=np.float64), 1e-12)
    def quantile(z):
        w = 1.0 - 1.0 / (9.0 * alpha) + z / np.sqrt(9.0 * alpha)
        return alpha * np.maximum(w, 0.0) ** 3 / beta
    del tail
    return quantile(z_lo), quantile(z_hi)


# ------------------------------------------------------ exposure and rates by cell

def current_band_exposure(age_sex: np.ndarray, window_months: int) -> np.ndarray:
    """Person-years by county, band and sex over a window ending at the reconstruction.

    Used only as a denominator for the current-period signals the projection is fitted
    on: the register's disappearance counts and the archive's first events. The exposure
    that is published is the projected one, over the sixty-month horizon window, which
    the simulation accumulates directly.
    """
    cube = np.asarray(age_sex, dtype=np.float64)
    m = band_matrix(cube.shape[1] - 1)
    banded = np.einsum("ba,cas->cbs", m, cube)
    return banded * (window_months / 12.0)


def _percentile_pair(sample: np.ndarray, point: float) -> tuple[float, float]:
    """A 90 percent interval from simulation paths, always containing the point."""
    sample = np.asarray(sample, dtype=np.float64)
    sample = sample[np.isfinite(sample)]
    if len(sample) < 10:
        return point, point
    lo, hi = np.percentile(sample, [5.0, 95.0])
    return float(min(lo, point)), float(max(hi, point))


def rate_release_rows(exposure: np.ndarray, deaths: np.ndarray, events: np.ndarray,
                      county_state: np.ndarray) -> list[dict]:
    """The exposure and rate block of the release table, protocol section 4 item 1.

    Inputs are the simulation's per-path cubes over the horizon window, each of shape
    (paths, counties, sexes, actuarial bands). Point estimates are means over paths and
    intervals are the 5th and 95th path percentiles, so the interval carries the same
    reconstruction, parameter and process uncertainty the liability tails carry.

    Additivity is built in rather than checked afterwards: a state's exposure is the sum
    of its counties' published exposures and a broad band is the sum of the actuarial
    bands inside it, both on the point estimates. Rates are ratios of the same summed
    quantities at the level they are published at, never sums of rates.
    """
    exposure = np.asarray(exposure, dtype=np.float64)
    deaths = np.asarray(deaths, dtype=np.float64)
    events = np.asarray(events, dtype=np.float64)
    county_state = np.asarray(county_state, dtype=np.int64)
    n_states = int(county_state.max()) + 1
    rows: list[dict] = []

    def to_state(cube: np.ndarray) -> np.ndarray:
        out = np.zeros((cube.shape[0], n_states) + cube.shape[2:])
        np.add.at(out, (slice(None), county_state), cube)
        return out

    cubes = {"county": (exposure, deaths, events),
             "state": (to_state(exposure), to_state(deaths), to_state(events))}
    national_rate = {}
    for estimand, index in ((MORTALITY_ESTIMAND, 1), (INCIDENCE_ESTIMAND, 2)):
        numerator = cubes["state"][index].sum(axis=(1,))
        denominator = cubes["state"][0].sum(axis=(1,))
        with np.errstate(invalid="ignore", divide="ignore"):
            national_rate[estimand] = np.where(denominator.mean(axis=0) > 0,
                                               numerator.mean(axis=0) /
                                               np.maximum(denominator.mean(axis=0), 1e-12),
                                               0.0)

    def emit(estimand, level, unit, sex, band, estimate, lower, upper):
        estimate = float(estimate) if np.isfinite(estimate) else 0.0
        lower = float(lower) if np.isfinite(lower) else estimate
        upper = float(upper) if np.isfinite(upper) else estimate
        rows.append({"estimand": estimand, "level": level, "unit": int(unit),
                     "sex": SEX_LABELS[sex], "age_band": band,
                     "estimate": estimate,
                     "lower": max(min(lower, estimate), 0.0),
                     "upper": max(upper, estimate)})

    for level, (e_cube, d_cube, n_cube) in cubes.items():
        n_units = e_cube.shape[1]
        e_point = e_cube.mean(axis=0)
        for u in range(n_units):
            for x in range(len(SEX_LABELS)):
                for b, band in enumerate(ACTUARIAL_AGE_BAND_LABELS):
                    lo, hi = _percentile_pair(e_cube[:, u, x, b], e_point[u, x, b])
                    emit(EXPOSURE_ESTIMAND, level, u, x, band, e_point[u, x, b], lo, hi)
                for broad, members in zip(BROAD_AGE_BAND_LABELS, BROAD_BAND_MEMBERS):
                    if len(members) == 1:
                        continue
                    total = float(sum(e_point[u, x, m] for m in members))
                    path_total = e_cube[:, u, x, list(members)].sum(axis=1)
                    lo, hi = _percentile_pair(path_total, total)
                    emit(EXPOSURE_ESTIMAND, level, u, x, broad, total, lo, hi)
                for estimand, count_cube in ((MORTALITY_ESTIMAND, d_cube),
                                             (INCIDENCE_ESTIMAND, n_cube)):
                    for b, band in enumerate(ACTUARIAL_AGE_BAND_LABELS):
                        e = float(e_point[u, x, b])
                        if e <= 0.0:
                            fallback = float(national_rate[estimand][x, b])
                            emit(estimand, level, u, x, band, fallback, 0.0,
                                 max(2.0 * fallback, 1e-6))
                            continue
                        point = float(count_cube[:, u, x, b].mean()) / e
                        with np.errstate(invalid="ignore", divide="ignore"):
                            per_path = count_cube[:, u, x, b] / np.maximum(
                                e_cube[:, u, x, b], 1e-9)
                        lo, hi = _percentile_pair(per_path, point)
                        emit(estimand, level, u, x, band, point, lo, hi)
    return rows


def state_rates_from_experience(experience: dict, ac: ActuarialContract,
                                years_ahead: float, kind: str) -> dict:
    """State by band by sex rate for the release window, from the experience file.

    All five years contribute. Each year's exposure is put on the last year's level by
    the estimated drift, so the pooled ratio of total events to drift-adjusted total
    exposure is the last year's rate with five years of information behind it; using the
    last year alone would throw away four fifths of the file and hand the thin cells to
    Poisson noise. Partial pooling across states then follows, and ``years_ahead``
    carries the level to a release window that sits after the file's last year.
    """
    counts = experience["deaths"] if kind == "mortality" else experience["qualifying_events"]
    exposure = experience["exposure"]
    improvement = estimate_improvement(exposure, counts)
    n_years = exposure.shape[0]
    offset = np.arange(n_years) - (n_years - 1)
    adjusted = exposure * np.exp(improvement["drift"] * offset)[:, None, None, None]
    pooled_counts = counts.sum(axis=0)
    pooled_exposure = adjusted.sum(axis=0)
    n_bands, n_sexes = pooled_counts.shape[1], pooled_counts.shape[2]
    rate = np.zeros_like(pooled_counts)
    lower = np.zeros_like(pooled_counts)
    upper = np.zeros_like(pooled_counts)
    for b in range(n_bands):
        for s in range(n_sexes):
            fitted = credibility_rate(pooled_counts[:, b, s], pooled_exposure[:, b, s])
            rate[:, b, s] = fitted["rate"]
            lower[:, b, s] = fitted["lower"]
            upper[:, b, s] = fitted["upper"]
    factor = float(np.exp(improvement["drift"] * years_ahead))
    # The drift itself is estimated, so carrying the level forward widens the interval
    # by the drift's own standard error over the distance carried.
    drift_half = float(1.6448536269514722 * improvement["drift_se"] * abs(years_ahead))
    return {"rate": rate * factor,
            "lower": lower * factor * np.exp(-drift_half),
            "upper": upper * factor * np.exp(drift_half),
            "improvement": improvement, "base_rate": rate}


def county_rate_shape(counts: np.ndarray, exposure: np.ndarray, county_state: np.ndarray,
                      state_rate: np.ndarray) -> dict:
    """County by band by sex rates: the state level times a partially pooled county
    deviation. The deviation is fitted on the county's own counts against its state's
    expected counts, so a county with little exposure sits on its state's rate."""
    counts = np.asarray(counts, dtype=np.float64)
    exposure = np.asarray(exposure, dtype=np.float64)
    expected = state_rate[county_state] * exposure
    ratio_fit = buhlmann_straub(counts.sum(axis=(1, 2)), expected.sum(axis=(1, 2)))
    z = np.asarray(ratio_fit["z"], dtype=np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        raw = np.where(expected.sum(axis=(1, 2)) > 0,
                       counts.sum(axis=(1, 2)) / np.maximum(expected.sum(axis=(1, 2)), 1e-9),
                       1.0)
    deviation = z * np.nan_to_num(raw, nan=1.0) + (1.0 - z) * ratio_fit["overall"]
    deviation = np.clip(deviation / max(ratio_fit["overall"], 1e-9), 0.4, 2.5)
    return {"rate": state_rate[county_state] * deviation[:, None, None],
            "deviation": deviation, "k": ratio_fit["k"]}


# ------------------------------------------------------------- liability simulation

@dataclass(frozen=True)
class SimulationParams:
    # Two thousand paths, the lower of the two ensemble sizes protocol section 6 names.
    # The Monte Carlo error of an empirical 95th percentile is about 0.16 standard
    # deviations at 180 paths and about 0.05 at 2,048, and a submitted quantile that
    # noisy scores worse than a normal approximation to the same distribution, which
    # would make ablation 6 pass for the wrong reason.
    n_paths: int = 2048
    path_chunk: int = 128
    seed: int = 20260903
    process_noise: bool = True       # Poisson counts rather than expected counts
    parameter_noise: bool = True     # draw the rate level and drift per path
    shock_probability: float = 0.10  # per projected year, from the public shock family
    shock_range: tuple[float, float] = (1.5, 3.0)


def discount_weights(v: np.ndarray, n_years: int) -> dict:
    """Yearly aggregates of the monthly discount vector.

    stock: a benefit paid every month to a headcount moving linearly from the year's
    start to its end contributes N_start * stock_level + (N_end - N_start) * stock_ramp.
    flow: an event spread uniformly over the year is discounted at the mean factor.
    """
    v = np.asarray(v, dtype=np.float64)
    level = np.zeros(n_years)
    ramp = np.zeros(n_years)
    flow = np.zeros(n_years)
    for y in range(n_years):
        window = v[12 * y: 12 * (y + 1)]
        if len(window) == 0:
            continue
        months = np.arange(1, len(window) + 1) / 12.0
        level[y] = window.sum()
        ramp[y] = float((window * months).sum())
        flow[y] = float(window.mean())
    return {"level": level, "ramp": ramp, "flow": flow}


def _region_matrix(region_of_county: np.ndarray, n_regions: int) -> np.ndarray:
    m = np.zeros((len(region_of_county), n_regions))
    m[np.arange(len(region_of_county)), np.asarray(region_of_county, dtype=np.int64)] = 1.0
    return m


def simulate_liabilities(age_sex_paths: np.ndarray, rates: dict, ac: ActuarialContract,
                         params: SimulationParams = SimulationParams()) -> dict:
    """Simulated present value of the region's obligations over the horizon.

    One path per population draw, so the spread carries reconstruction uncertainty; on
    top of it the rate level and the mortality drift are drawn per path (parameter
    uncertainty) and the yearly counts of deaths, first events, births and net movers
    are Poisson draws (process uncertainty). A tail quantile taken from a set of point
    projections would carry only the first of the three, which is what the mean-only and
    normal-approximation controls are built to expose.
    """
    paths = np.asarray(age_sex_paths, dtype=np.float64)
    n_paths, n_counties, n_ages, n_sexes = paths.shape
    n_years = ac.n_years
    weights = discount_weights(ac.discount, n_years)
    region = _region_matrix(ac.region_of_county, ac.n_regions)
    ages = np.arange(n_ages)
    eligible = (ages >= ac.eligible_min_age).astype(np.float64)[None, None, :, None]
    liability = np.zeros((n_paths, ac.n_regions))
    n_bands = len(ACTUARIAL_AGE_BANDS)
    # Per-path exposure, deaths and first events by county, sex and band: the release
    # table's rate block is this window, so it is accumulated in the same pass.
    exposure_acc = np.zeros((n_paths, n_counties, n_sexes, n_bands))
    death_acc = np.zeros((n_paths, n_counties, n_sexes, n_bands))
    event_acc = np.zeros((n_paths, n_counties, n_sexes, n_bands))
    q_base = np.asarray(rates["mortality"], dtype=np.float64)   # (counties, ages, sexes)
    lam_base = np.asarray(rates["incidence"], dtype=np.float64)
    mig_base = np.asarray(rates["migration"], dtype=np.float64)
    not_yet = np.asarray(rates["not_yet"], dtype=np.float64)
    for start in range(0, n_paths, params.path_chunk):
        stop = min(start + params.path_chunk, n_paths)
        chunk = stop - start
        rng = np.random.default_rng([params.seed, start])
        state = paths[start:stop].copy()
        pending = state * not_yet[None]
        if params.parameter_noise:
            level_q = np.exp(rng.normal(0.0, rates.get("mortality_log_sd", 0.05), size=chunk))
            level_l = np.exp(rng.normal(0.0, rates.get("incidence_log_sd", 0.08), size=chunk))
            drift_q = rng.normal(rates.get("mortality_drift", 0.0),
                                 rates.get("mortality_drift_se", 0.01), size=chunk)
            drift_l = rng.normal(rates.get("incidence_drift", 0.0),
                                 rates.get("incidence_drift_se", 0.01), size=chunk)
            mig_noise = rng.normal(0.0, 1.0, size=(chunk, 1, 1, 1)) * \
                np.asarray(rates.get("migration_se", 0.0), dtype=np.float64)[None]
        else:
            level_q = np.ones(chunk)
            level_l = np.ones(chunk)
            drift_q = np.full(chunk, rates.get("mortality_drift", 0.0))
            drift_l = np.full(chunk, rates.get("incidence_drift", 0.0))
            mig_noise = np.zeros((chunk, 1, 1, 1))
        migration = mig_base[None] + mig_noise
        for year in range(n_years):
            shock = np.ones(chunk)
            if params.process_noise and params.shock_probability > 0:
                hit = rng.random(chunk) < params.shock_probability
                shock[hit] = rng.uniform(*params.shock_range, size=int(hit.sum()))
            elapsed = year + 0.5
            q = np.clip(q_base[None] * np.exp(drift_q * elapsed)[:, None, None, None] *
                        (level_q * shock)[:, None, None, None], 0.0, 0.98)
            lam = np.clip(lam_base[None] * np.exp(drift_l * elapsed)[:, None, None, None] *
                          level_l[:, None, None, None], 0.0, 1.0)
            if params.process_noise:
                deaths = np.minimum(rng.poisson(np.maximum(state * q, 0.0)), state)
                events = np.minimum(rng.poisson(np.maximum(pending * lam, 0.0)), pending)
            else:
                deaths = state * q
                events = pending * lam
            survivors = np.maximum(state - deaths, 0.0)
            with np.errstate(invalid="ignore", divide="ignore"):
                keep = np.where(state > 0, survivors / np.maximum(state, 1e-12), 0.0)
            pending = np.maximum(pending - events, 0.0) * keep
            flow = survivors * migration
            if params.process_noise:
                flow = flow + rng.normal(0.0, np.sqrt(np.abs(flow) + 1e-9))
            after = np.maximum(survivors + flow, 0.0)
            with np.errstate(invalid="ignore", divide="ignore"):
                pending = pending * np.where(survivors > 0,
                                             after / np.maximum(survivors, 1e-12), 0.0)
            # Attained age is constant inside the year, so a year's person-years, deaths
            # and first events all land in the band of the age held during that year.
            mid = 0.5 * (state + after)
            for b, (lo_b, hi_b) in enumerate(ACTUARIAL_AGE_BANDS):
                sl = slice(lo_b, min(hi_b, n_ages - 1) + 1)
                exposure_acc[start:stop, :, :, b] += mid[:, :, sl, :].sum(axis=2)
                death_acc[start:stop, :, :, b] += deaths[:, :, sl, :].sum(axis=2)
                event_acc[start:stop, :, :, b] += events[:, :, sl, :].sum(axis=2)
            women = after[:, :, 18:46, 1].sum(axis=2)
            births = women * rates.get("fertility", 0.0)
            if params.process_noise:
                births = rng.poisson(np.maximum(births, 0.0)).astype(np.float64)
            aged = np.zeros_like(after)
            aged[:, :, 1:, :] = after[:, :, :-1, :]
            aged[:, :, -1, :] += after[:, :, -1, :]
            aged[:, :, 0, 0] += 0.5 * births
            aged[:, :, 0, 1] += 0.5 * births
            aged_pending = np.zeros_like(pending)
            aged_pending[:, :, 1:, :] = pending[:, :, :-1, :]
            aged_pending[:, :, -1, :] += pending[:, :, -1, :]
            aged_pending[:, :, 0, 0] += 0.5 * births
            aged_pending[:, :, 0, 1] += 0.5 * births
            start_elig = (state * eligible).sum(axis=(2, 3))
            end_elig = (aged * eligible).sum(axis=(2, 3))
            stock = ac.monthly_benefit * (start_elig * weights["level"][year] +
                                          (end_elig - start_elig) * weights["ramp"][year])
            event_flow = ac.first_event_cost * events.sum(axis=(2, 3)) * weights["flow"][year]
            death_flow = ac.death_benefit * deaths.sum(axis=(2, 3)) * weights["flow"][year]
            liability[start:stop] += (stock + event_flow + death_flow) @ region
            state, pending = aged, aged_pending
    return {"liability": liability, "n_paths": n_paths, "exposure": exposure_acc,
            "deaths": death_acc, "events": event_acc}
    return {"liability": liability, "n_paths": n_paths}


def tail_summary(liability: np.ndarray, alpha: float = 0.95) -> dict:
    """Region liability mean, value at risk and expected shortfall at ``alpha``."""
    liability = np.asarray(liability, dtype=np.float64)
    mean = liability.mean(axis=0)
    q = np.quantile(liability, alpha, axis=0)
    es = np.zeros_like(q)
    for r in range(liability.shape[1]):
        tail = liability[:, r][liability[:, r] >= q[r]]
        es[r] = float(tail.mean()) if len(tail) else float(q[r])
    return {"mean": mean, "q": q, "es": es}


def exceedance_probability(q_hat: np.ndarray, liability: np.ndarray) -> np.ndarray:
    """Share of paths above the submitted quantile, per region. The verifier computes
    this against the sealed ensemble; the reference computes it against its own paths
    as an internal check that the submitted vector is self-consistent."""
    return (np.asarray(liability) > np.asarray(q_hat)[None, :]).mean(axis=0)


def quantile_score(q_hat: np.ndarray, liability: np.ndarray, alpha: float = 0.95) -> np.ndarray:
    """Pinball loss of the submitted quantile, averaged over paths, per region."""
    q = np.asarray(q_hat, dtype=np.float64)[None, :]
    y = np.asarray(liability, dtype=np.float64)
    return ((alpha - (y <= q).astype(np.float64)) * (y - q)).mean(axis=0)


# --------------------------------------------------------------- reserve allocation

def shortfall_objective(allocation: np.ndarray, liability: np.ndarray,
                        weights: np.ndarray) -> float:
    """J(A) = sum_r w_r mean_m (L_rm - A_r)_+, the sealed expected uncovered
    obligation of section 9, computed here against the method's own paths."""
    a = np.asarray(allocation, dtype=np.float64)[None, :]
    short = np.maximum(np.asarray(liability, dtype=np.float64) - a, 0.0)
    return float((np.asarray(weights, dtype=np.float64) * short.mean(axis=0)).sum())


# The tail calibration may widen a tail as well as narrow it, but not without limit: a
# factor this far from one says the method's own tail is not the thing that is wrong, and
# a submission is better served filing what it estimated than a shape rescaled past
# recognition.
MAX_TAIL_CALIBRATION = 3.0


def calibrate_quantiles_to_total(mean: np.ndarray, q: np.ndarray, es: np.ndarray, total: float,
                                 gamma: float) -> tuple[np.ndarray, np.ndarray, np.ndarray,
                                                        float, float]:
    """Bring the tail the method believes into agreement with the public reserve total.

    R is not the sum of the regional quantiles: it is sum_r q_r* + gamma sum_r (e_r* -
    q_r*), so a method that scales its own tail until the quantiles alone sum to R has
    forced sum_r q_hat_r = R and left itself no allocation to make. The coherent target is
    the total R was built from. Pulling the quantile and the shortfall toward the mean by
    one common factor and solving for the factor that puts the method's own implied
    reserve at R keeps the ordering across regions, leaves the means untouched, and leaves
    exactly the slack the construction implies.

    The factor moves the tail in both directions. An earlier pass applied it only when it
    shrank, on the argument that a method should not inflate its own tail to meet a public
    number. That argument is wrong about what R is: R is built from the sealed regional
    quantiles, so a method whose implied reserve sits under R has been told, by a published
    quantity, that its tail is too low. Refusing to use it left the reference filing
    quantiles about five percent under the truth on a tail five to fourteen percent wide,
    which put its sealed exceedance rate at three times nominal and made the tail gate
    unattainable by the very reference that has to attain it.
    """
    mean = np.asarray(mean, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    es = np.asarray(es, dtype=np.float64)
    implied = mean.sum() + (1.0 - gamma) * (q.sum() - mean.sum()) \
        + gamma * (es.sum() - mean.sum())
    if mean.sum() >= total:
        # The public total is under this method's own expected liability, so what is too
        # high is the level, not the width. Everything scales by the one factor that puts
        # the implied reserve at R. Shrinking only the tail here would file a 95th
        # percentile under the mean beside it, which is not a distribution.
        scale = float(total / max(implied, 1e-9))
        return mean * scale, q * scale, es * scale, 1.0, scale
    spread = (1.0 - gamma) * (q.sum() - mean.sum()) + gamma * (es.sum() - mean.sum())
    if spread <= 0:
        return mean, q, es, 1.0, 1.0
    theta = float(np.clip((total - mean.sum()) / spread, 0.0, MAX_TAIL_CALIBRATION))
    return mean, mean + theta * (q - mean), mean + theta * (es - mean), theta, 1.0


def _quantile_at(sorted_liability: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Per-region empirical quantile at a per-region probability."""
    n = sorted_liability.shape[0]
    out = np.empty(sorted_liability.shape[1])
    for r in range(sorted_liability.shape[1]):
        out[r] = float(np.interp(np.clip(p[r], 0.0, 1.0), np.linspace(0.0, 1.0, n),
                                 sorted_liability[:, r]))
    return out


def allocate_reserve(liability: np.ndarray, q_hat: np.ndarray, total: float,
                     weights: np.ndarray | None = None, iterations: int = 80) -> dict:
    """Allocate the fixed reserve to minimise the weighted expected shortfall.

    J(A) is separable and convex in A, so at the optimum every region not held at its
    own floor shares one marginal cost: w_r P(L_r > A_r) = nu. Inverting that gives
    A_r(nu) = max(q_hat_r, F_r inverse of 1 - nu / w_r), and the sum is monotone in nu,
    so one bisection on the single multiplier solves the constrained problem exactly on
    the empirical distribution. This is the step a proportional split cannot imitate:
    money goes where the marginal probability of a shortfall is highest, not where the
    exposure is largest.
    """
    liability = np.asarray(liability, dtype=np.float64)
    n_regions = liability.shape[1]
    weights = np.ones(n_regions) if weights is None else np.asarray(weights, dtype=np.float64)
    floor = np.maximum(np.asarray(q_hat, dtype=np.float64), 0.0)
    ordered = np.sort(liability, axis=0)
    if floor.sum() > total + 1e-6:
        return {"allocation": floor, "feasible": False, "nu": float("nan"),
                "reason": "submitted quantiles already sum above the reserve total"}

    def at(nu: float) -> np.ndarray:
        p = 1.0 - nu / np.maximum(weights, 1e-12)
        return np.maximum(_quantile_at(ordered, p), floor)

    hi = float(weights.max())
    if at(0.0).sum() <= total:
        allocation = at(0.0)
        residual = total - allocation.sum()
        allocation = allocation + residual / n_regions
        return {"allocation": allocation, "feasible": True, "nu": 0.0,
                "reason": "reserve covers every simulated path"}
    lo = 0.0
    for _ in range(iterations):
        mid = 0.5 * (lo + hi)
        if at(mid).sum() > total:
            lo = mid
        else:
            hi = mid
    allocation = at(hi)
    allocation = _repair_to_total(allocation, floor, total)
    return {"allocation": allocation, "feasible": True, "nu": float(hi), "reason": ""}


def _repair_to_total(allocation: np.ndarray, floor: np.ndarray, total: float) -> np.ndarray:
    """Close the residual the bisection leaves, without breaking either constraint."""
    allocation = np.asarray(allocation, dtype=np.float64).copy()
    for _ in range(50):
        residual = total - allocation.sum()
        if abs(residual) <= 1e-9 * max(abs(total), 1.0):
            break
        if residual > 0:
            allocation += residual / len(allocation)
        else:
            slack = np.maximum(allocation - floor, 0.0)
            if slack.sum() <= 1e-12:
                break
            allocation -= (-residual) * slack / slack.sum()
            allocation = np.maximum(allocation, floor)
    return allocation


def proportional_reserve(q_hat: np.ndarray, share: np.ndarray, total: float) -> np.ndarray:
    """The practical baseline and the ablation: the floor plus the remaining reserve
    split in proportion to a size measure, with no reference to the regional tails."""
    floor = np.maximum(np.asarray(q_hat, dtype=np.float64), 0.0)
    share = np.maximum(np.asarray(share, dtype=np.float64), 0.0)
    share = share / max(share.sum(), 1e-12)
    remainder = total - floor.sum()
    if remainder < 0:
        return floor * (total / max(floor.sum(), 1e-9))
    return floor + remainder * share


# ------------------------------------------------------------------------- writers

def reserve_rows(summary: dict, allocation: np.ndarray) -> list[dict]:
    """The section 4 item 4 file: one row per region."""
    rows = []
    for r in range(len(allocation)):
        rows.append({"region": int(r), "liability_mean": float(summary["mean"][r]),
                     "q95": float(summary["q"][r]), "es95": float(summary["es"][r]),
                     "allocation": float(allocation[r])})
    return rows


def widen_release_rows(rows: list[dict]) -> list[dict]:
    """Give the version-three release rows the two key columns the rate rows carry, so
    one file holds both row shapes."""
    out = []
    for row in rows:
        widened = dict(row)
        widened.setdefault("sex", "")
        widened.setdefault("age_band", "")
        out.append(widened)
    return out


def write_actuarial_submission(out_dir: Path, release_rows, projection_rows, cube,
                               suppress_below: float, reserve, band_labels,
                               sex_labels) -> None:
    """The four submitted files of section 4, with the reserve file in place of the
    hospital allocation."""
    import pandas as pd
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cube = np.asarray(cube, dtype=np.float64)
    detail = []
    for c in range(cube.shape[0]):
        for b, band in enumerate(band_labels):
            for s, sex in enumerate(sex_labels):
                value = float(cube[c, b, s])
                detail.append({"county": c, "age_band": band, "sex": sex,
                               "count": "" if 0 < value < suppress_below else round(value, 3)})
    columns = ["estimand", "level", "unit"] + list(RATE_EXTRA_COLUMNS) + \
        ["estimate", "lower", "upper"]
    pd.DataFrame(release_rows)[columns].to_csv(out_dir / "release.csv", index=False)
    pd.DataFrame(projection_rows).to_csv(out_dir / "projection.csv", index=False)
    pd.DataFrame(detail).to_csv(out_dir / "detailed.csv", index=False)
    pd.DataFrame(reserve)[list(RESERVE_COLUMNS)].to_csv(out_dir / "reserve.csv", index=False)


# ------------------------------------------------------- band to single-year age fill

def expand_band_rates(band_rate: np.ndarray, band_exposure: np.ndarray,
                      max_age: int = MAX_AGE) -> np.ndarray:
    """Single-year age schedule consistent with a set of band rates.

    Above thirty the log rate is close to linear in age, so a weighted straight line
    through the band midpoints gives a schedule whose band averages reproduce the input
    after one rescaling pass. Below thirty the age gradient is not log-linear and the
    band rate itself is used flat.
    """
    band_rate = np.asarray(band_rate, dtype=np.float64)
    band_exposure = np.asarray(band_exposure, dtype=np.float64)
    ages = np.arange(max_age + 1)
    band = band_of_age(ages)
    schedule = np.zeros(max_age + 1)
    for b in range(len(ACTUARIAL_AGE_BANDS)):
        schedule[band == b] = max(band_rate[b], 0.0)
    midpoint = np.array([0.5 * (lo + min(hi, max_age)) for lo, hi in ACTUARIAL_AGE_BANDS])
    use = (midpoint >= 30) & (band_rate > 0) & (band_exposure > 0)
    if use.sum() >= 3:
        w = band_exposure[use]
        x = midpoint[use]
        y = np.log(band_rate[use])
        mx = float((w * x).sum() / w.sum())
        my = float((w * y).sum() / w.sum())
        slope = float((w * (x - mx) * (y - my)).sum() / max((w * (x - mx) ** 2).sum(), 1e-12))
        fitted = np.exp(my + slope * (ages - mx))
        above = ages >= 30
        schedule[above] = fitted[above]
        # One rescaling pass so each band's exposure-weighted mean returns its input.
        for b in range(len(ACTUARIAL_AGE_BANDS)):
            cells = (band == b) & above
            if cells.sum() == 0 or band_rate[b] <= 0:
                continue
            current = float(schedule[cells].mean())
            if current > 0:
                schedule[cells] *= band_rate[b] / current
    return np.clip(schedule, 0.0, 1.0)


def _band_to_ages(values: np.ndarray) -> np.ndarray:
    """Broadcast a per-band vector over single-year ages."""
    ages = np.arange(MAX_AGE + 1)
    band = band_of_age(ages)
    out = np.zeros(MAX_AGE + 1)
    for b in range(len(ACTUARIAL_AGE_BANDS)):
        out[band == b] = values[b]
    return out


# --------------------------------------------------------------- vintage linkage

def vintage_death_counts(preliminary, revised, tick_pre: int, tick_rev: int,
                         county_state: np.ndarray, rng: np.random.Generator,
                         n_imputations: int = 6, deterministic: bool = False) -> dict:
    """Disappearance counts between the two register vintages, by county, band and sex.

    Version four does not carry a record identifier across vintages, so the two files
    are linked probabilistically and the disappearance count is averaged over
    imputations of the link set. The spread across imputations is the linkage
    contribution to the mortality estimate and is reported so the caller can widen its
    intervals by it. With ``deterministic`` the link set is the exact-key join, which is
    the ablation: exact keys over-link when names repeat and under-link whenever a name,
    birth month or sex is misreported, and the disappearance count inherits both.
    """
    n_counties = len(county_state)
    n_bands = len(ACTUARIAL_AGE_BANDS)
    shape = (n_counties, n_bands, 2)

    def cells(frame, tick):
        county = frame["county"].to_numpy(dtype=np.int64)
        band = band_of_age((tick - frame["birth_tick"].to_numpy(dtype=np.int64)) // 12)
        sex = frame["sex"].to_numpy(dtype=np.int64)
        ok = (county >= 0) & (county < n_counties) & (band >= 0) & (sex >= 0) & (sex < 2)
        return county, band, sex, ok

    pre = preliminary.drop_duplicates(subset=["person_id"]).reset_index(drop=True)
    rev = revised.drop_duplicates(subset=["person_id"]).reset_index(drop=True)
    pre = pre.assign(_row=np.arange(len(pre)))
    rev = rev.assign(_row=np.arange(len(rev)))
    county, band, sex, ok = cells(pre, tick_pre)
    at_risk = np.zeros(shape)
    flat = (county[ok] * n_bands + band[ok]) * 2 + sex[ok]
    at_risk = np.bincount(flat, minlength=n_counties * n_bands * 2).reshape(shape).astype(float)
    if deterministic:
        key = ["given_code", "family_code", "birth_tick", "sex"]
        matched = pre.merge(rev[key].drop_duplicates().assign(_seen=1), on=key, how="left")
        survived = np.zeros(len(pre), dtype=bool)
        survived[:] = matched["_seen"].to_numpy() == 1
        draws = [survived]
    else:
        result = probabilistic_links(pre, rev, "_row", "_row")
        links = result["links"]
        p = links["p_match"].to_numpy(dtype=np.float64) if len(links) else np.zeros(0)
        left_row = links["_id_l"].to_numpy(dtype=np.int64) if len(links) else np.zeros(0, np.int64)
        draws = []
        for _ in range(max(n_imputations, 1)):
            survived = np.zeros(len(pre), dtype=bool)
            if len(p):
                survived[left_row[sample_link_indicator(p, rng)]] = True
            draws.append(survived)
    gone_draws = []
    for survived in draws:
        gone = (~survived) & ok
        flat_gone = (county[gone] * n_bands + band[gone]) * 2 + sex[gone]
        gone_draws.append(np.bincount(flat_gone, minlength=n_counties * n_bands * 2)
                          .reshape(shape).astype(float))
    stacked = np.stack(gone_draws)
    return {"gone": stacked.mean(axis=0), "at_risk": at_risk,
            "linkage_spread": stacked.std(axis=0) if len(stacked) > 1 else np.zeros(shape),
            "months": max(tick_rev - tick_pre, 1), "n_imputations": len(draws)}


# ------------------------------------------------------------------- the whole layer

@dataclass(frozen=True)
class LayerParams:
    """One switch per targeted ablation of protocol section 11, all false for the
    reference. Each switch removes exactly one step and nothing else, so a control that
    passes its gate says the gate is loose rather than that the control is subtle."""
    simulation: SimulationParams = SimulationParams()
    deterministic_linkage: bool = False        # ablation 3, first half
    archive_only_rates: bool = False           # ablation 3, second half
    ignore_health_selection: bool = False      # ablation 4
    regime_override: dict | None = None        # ablation 5: development-average regime
    tail: str = "simulated"                    # ablations 6 and 7
    padding: float = 1.6
    allocation: str = "shortfall"              # ablation 8 uses "proportional"
    n_link_imputations: int = 6
    seed: int = 20260904


def _rate_interval(counts: np.ndarray, exposure: np.ndarray, point: np.ndarray,
                   k: float) -> tuple[np.ndarray, np.ndarray]:
    """Gamma interval around a partially pooled rate."""
    if not np.isfinite(k):
        k = max(float(np.nanmax(exposure)), 1.0) * 10.0
    alpha = np.asarray(counts, dtype=np.float64) + k * np.asarray(point, dtype=np.float64)
    beta = np.asarray(exposure, dtype=np.float64) + k
    return _gamma_interval(alpha, beta)


def _state_sum(values: np.ndarray, county_state: np.ndarray, n_states: int) -> np.ndarray:
    out = np.zeros((n_states,) + values.shape[1:])
    for c in range(values.shape[0]):
        out[int(county_state[c])] += values[c]
    return out


def actuarial_layer(data: dict, county_state: np.ndarray, age_sex_paths: np.ndarray,
                    ac: ActuarialContract, experience, fertility_rate: float,
                    params: LayerParams = LayerParams()) -> dict:
    """Exposures, rates, liability tails and a reserve, from one line's population draws.

    The caller supplies ``age_sex_paths`` of shape (paths, counties, ages, sexes): the
    design-based line passes its bootstrap replicates, the Bayesian line its posterior
    draws. Everything after that point is shared, so the two submissions differ where
    the two reconstructions differ and nowhere else.
    """
    contract = data["contract"]
    tick = int(contract["ticks"]["revised"])
    tick_pre = int(contract["ticks"]["preliminary"])
    n_counties = len(county_state)
    n_states = int(county_state.max()) + 1
    n_bands = len(ACTUARIAL_AGE_BANDS)
    rng = np.random.default_rng(params.seed)
    paths = np.asarray(age_sex_paths, dtype=np.float64)
    if paths.ndim != 4:
        raise ValueError("age_sex_paths must be (paths, counties, ages, sexes)")
    mean_cube = paths.mean(axis=0)

    # 1. Current-period exposure. This is a denominator for the signals the projection
    #    is fitted on, not a published quantity: what the release carries is the
    #    projected exposure over the horizon window, which the simulation accumulates.
    current_exposure = current_band_exposure(mean_cube, 12)
    state_exposure = _state_sum(current_exposure, county_state, n_states)
    state_population = state_exposure

    # 2. Health-source selection, anchored on the survey item.
    anchor = anchor_prevalence(data["survey"], ac, county_state)
    archive_county = archive_recent_counts(data["health"], county_state, tick,
                                           ac.anchor_window_months, ac.qualifying_groups)
    archive_recent = _state_sum(archive_county, county_state, n_states)
    if params.ignore_health_selection or not anchor["available"]:
        inclusion = {"pi": np.ones((n_states, n_bands, 2)), "pooled": 1.0,
                     "available": False}
    else:
        inclusion = inclusion_probability(archive_recent, anchor, state_population)
    pi = np.asarray(inclusion["pi"], dtype=np.float64)

    # 3. Regime from the historical experience file. The file's last year is centred half
    #    a year before the revised snapshot, so its level is carried that far forward and
    #    the simulation carries the drift from there.
    exp_arrays = experience_arrays(experience, n_states)
    mortality_state = state_rates_from_experience(exp_arrays, ac, 0.5, "mortality")
    incidence_state = state_rates_from_experience(exp_arrays, ac, 0.5, "incidence")
    migration = estimate_migration_profile(exp_arrays)
    override = params.regime_override or {}
    mortality_drift = float(override.get("mortality_drift",
                                         mortality_state["improvement"]["drift"]))
    mortality_drift_se = float(override.get("mortality_drift_se",
                                            mortality_state["improvement"]["drift_se"]))
    incidence_drift = float(override.get("incidence_drift",
                                         incidence_state["improvement"]["drift"]))
    incidence_drift_se = float(override.get("incidence_drift_se",
                                            incidence_state["improvement"]["drift_se"]))
    if "migration_scale" in override:
        migration = {"rate": migration["rate"] * float(override["migration_scale"]),
                     "se": migration["se"], "national": migration["national"]}

    # 4. County shape of mortality: register disappearance between the two vintages, net
    #    of an age-flat coverage churn read off the bands where deaths are rare. The two
    #    vintages share no record identifier in version four, so the join is the
    #    probabilistic one and the estimate is averaged over imputations of the link set.
    vintages = vintage_death_counts(data["population_preliminary"], data["population"],
                                    tick_pre, tick, county_state, rng,
                                    n_imputations=params.n_link_imputations,
                                    deterministic=params.deterministic_linkage)
    months = vintages["months"]
    young = [b for b, (lo, hi) in enumerate(ACTUARIAL_AGE_BANDS) if hi <= 44]
    at_risk = vintages["at_risk"]
    with np.errstate(invalid="ignore", divide="ignore"):
        churn = np.where(at_risk[:, young, :].sum(axis=(1, 2)) > 0,
                         vintages["gone"][:, young, :].sum(axis=(1, 2)) /
                         np.maximum(at_risk[:, young, :].sum(axis=(1, 2)), 1e-9), 0.0)
    if params.archive_only_rates:
        churn = np.zeros_like(churn)
    death_counts = np.maximum(vintages["gone"] - churn[:, None, None] * at_risk, 0.0)
    death_exposure = at_risk * (months / 12.0)
    if params.archive_only_rates:
        pooled = credibility_rate(_state_sum(death_counts, county_state, n_states),
                                  _state_sum(death_exposure, county_state, n_states))
        mortality_state = {"rate": pooled["rate"], "lower": pooled["lower"],
                           "upper": pooled["upper"], "base_rate": pooled["rate"],
                           "improvement": {"drift": 0.0, "drift_se": 0.02,
                                           "fitted": False}}
    mortality_county = county_rate_shape(death_counts, death_exposure, county_state,
                                         mortality_state["rate"])

    # 5. County shape of incidence: first qualifying events in the archive, divided by
    #    the estimated inclusion probability of an admitted person in that cell.
    events = archive_recent_counts(data["health"], county_state, tick, 12,
                                   ac.qualifying_groups)
    events_adjusted = events / np.maximum(pi[county_state], INCLUSION_BOUNDS[0])
    if params.archive_only_rates:
        pooled = credibility_rate(_state_sum(events, county_state, n_states),
                                  np.maximum(state_exposure, 1e-9))
        incidence_state = {"rate": pooled["rate"], "lower": pooled["lower"],
                           "upper": pooled["upper"], "base_rate": pooled["rate"],
                           "improvement": {"drift": 0.0, "drift_se": 0.02,
                                           "fitted": False}}
    incidence_county = county_rate_shape(events_adjusted,
                                         np.maximum(current_exposure, 1e-9),
                                         county_state, incidence_state["rate"])

    # 6. Single-year schedules for the projection.
    national_band_mortality = np.zeros((n_bands, 2))
    national_band_incidence = np.zeros((n_bands, 2))
    with np.errstate(invalid="ignore", divide="ignore"):
        national_band_mortality = (mortality_state["rate"] * state_exposure).sum(axis=0) / \
            np.maximum(state_exposure.sum(axis=0), 1e-9)
        national_band_incidence = (incidence_state["rate"] * state_exposure).sum(axis=0) / \
            np.maximum(state_exposure.sum(axis=0), 1e-9)
    national_band_exposure = state_exposure.sum(axis=0)
    q_ages = np.stack([expand_band_rates(national_band_mortality[:, s],
                                         national_band_exposure[:, s]) for s in range(2)], axis=1)
    lam_ages = np.stack([expand_band_rates(national_band_incidence[:, s],
                                           national_band_exposure[:, s])
                         for s in range(2)], axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        county_mortality_level = np.where(national_band_mortality[None] > 0,
                                          mortality_county["rate"] /
                                          np.maximum(national_band_mortality[None], 1e-12), 1.0)
        county_incidence_level = np.where(national_band_incidence[None] > 0,
                                          incidence_county["rate"] /
                                          np.maximum(national_band_incidence[None], 1e-12), 1.0)
    weight = np.maximum(current_exposure, 1e-9)
    level_m = (county_mortality_level * weight).sum(axis=(1, 2)) / weight.sum(axis=(1, 2))
    level_i = (county_incidence_level * weight).sum(axis=(1, 2)) / weight.sum(axis=(1, 2))
    mortality_full = q_ages[None] * np.clip(level_m, 0.4, 2.5)[:, None, None]
    incidence_full = lam_ages[None] * np.clip(level_i, 0.4, 2.5)[:, None, None]
    migration_full = np.stack([np.stack([_band_to_ages(migration["rate"][county_state[c], :, s])
                                         for s in range(2)], axis=1)
                               for c in range(n_counties)])
    migration_se_full = np.stack([np.stack([_band_to_ages(migration["se"][county_state[c], :, s])
                                            for s in range(2)], axis=1)
                                  for c in range(n_counties)])

    # 7. Everyone enters the window without a qualifying event: the scored event is a
    #    person's first inside the sixty months, which is the convention the truth uses
    #    and the only one the participant's files can support.
    not_yet = np.ones((n_counties, MAX_AGE + 1, 2))

    rates = {
        "mortality": mortality_full, "incidence": incidence_full,
        "migration": migration_full, "migration_se": migration_se_full,
        "not_yet": not_yet, "fertility": float(fertility_rate),
        "mortality_drift": mortality_drift, "mortality_drift_se": mortality_drift_se,
        "incidence_drift": incidence_drift, "incidence_drift_se": incidence_drift_se,
        "mortality_log_sd": float(override.get("mortality_log_sd", 0.06)),
        "incidence_log_sd": float(override.get("incidence_log_sd", 0.10)),
    }

    # 8. One simulation produces three things: the projected exposure and rates the
    #    release table carries, the liability paths, and the reserve the tails decide.
    sim = params.simulation
    n_paths = sim.n_paths
    index = np.arange(n_paths) % paths.shape[0]
    simulated = simulate_liabilities(paths[index], rates, ac, sim)
    liability = simulated["liability"]
    rate_rows = rate_release_rows(simulated["exposure"], simulated["deaths"],
                                  simulated["events"], county_state)
    summary = tail_summary(liability)
    if params.tail == "mean":
        q_hat = summary["mean"].copy()
        es_hat = summary["mean"].copy()
    elif params.tail == "normal":
        # The textbook normal approximation on the simulated first two moments: right
        # mean, right variance, no skewness. It separates from the reference exactly to
        # the extent the liability distribution is not normal.
        sd = liability.std(axis=0)
        q_hat = summary["mean"] + 1.6448536269514722 * sd
        es_hat = summary["mean"] + 2.0627128 * sd
    elif params.tail == "padded":
        # A cushion of a fixed share of the expected cost, the shape an over-cautious
        # analyst adds. A multiplicative pad would be undone exactly by the calibration
        # to the public reserve total, since that step shrinks along the same ray.
        cushion = (params.padding - 1.0) * summary["mean"]
        q_hat = summary["q"] + cushion
        es_hat = summary["es"] + cushion
    else:
        q_hat = summary["q"].copy()
        es_hat = summary["es"].copy()
    mean_hat, q_hat, es_hat, theta, scale = calibrate_quantiles_to_total(
        summary["mean"], q_hat, es_hat, ac.reserve_total, ac.gamma)
    q_hat = np.maximum(q_hat, mean_hat)
    es_hat = np.maximum(es_hat, q_hat)
    # The public total is a statement about the whole predictive distribution, not about
    # two summaries of it, so the paths the allocation is decided on move toward or away
    # from their own means by the same factor the summaries did.
    # Allocating against an uncalibrated spread while filing calibrated floors would leave
    # the optimiser believing every region is under-covered, and it would then spend the
    # slack on the largest weight rather than on the thickest residual tail.
    if scale != 1.0:
        liability = liability * scale
    elif theta != 1.0:
        liability = liability.mean(axis=0) + theta * (liability - liability.mean(axis=0))
    # The practical baseline splits the reserve above the floors in proportion to
    # projected eligible exposure, which is the size measure a reserving office would
    # reach for and the analogue of the version-three elders-proportional allocation.
    eligible_bands = [b for b, (lo, hi) in enumerate(ACTUARIAL_AGE_BANDS)
                      if lo >= ac.eligible_min_age]
    county_eligible = simulated["exposure"].mean(axis=0)[:, :, eligible_bands].sum(axis=(1, 2))
    share = np.zeros(ac.n_regions)
    np.add.at(share, ac.region_of_county, county_eligible)
    baseline = proportional_reserve(q_hat, share, ac.reserve_total)
    if params.allocation == "proportional":
        allocation = baseline
        allocation_detail = {"feasible": True, "nu": float("nan"),
                             "reason": "proportional to projected eligible exposure"}
    else:
        allocation_detail = allocate_reserve(liability, q_hat, ac.reserve_total,
                                             ac.reserve_weights)
        allocation = allocation_detail["allocation"]
    summary_out = {"mean": mean_hat, "q": q_hat, "es": es_hat}
    # The uncalibrated tail is kept beside the submitted one: the freeze needs the
    # method's own q and es before the public total is allowed to move them.
    rows = reserve_rows(summary_out, allocation)
    diagnostics = {
        "inclusion_pooled": float(inclusion["pooled"]),
        "selection_adjusted": bool(inclusion["available"]),
        "mortality_drift": mortality_drift, "incidence_drift": incidence_drift,
        "linkage_imputations": int(vintages["n_imputations"]),
        "linkage_spread": float(np.mean(vintages["linkage_spread"])),
        "objective": shortfall_objective(allocation, liability, ac.reserve_weights),
        "objective_baseline": shortfall_objective(baseline, liability, ac.reserve_weights),
        "exceedance": exceedance_probability(q_hat, liability).tolist(),
        "quantile_score": quantile_score(q_hat, liability).tolist(),
        "reserve_feasible": bool(allocation_detail.get("feasible", True)),
        "n_paths": int(liability.shape[0]),
    }
    return {"rate_rows": rate_rows, "reserve": rows, "liability": liability,
            "summary": summary_out, "raw_summary": summary, "allocation": allocation,
            "exposure": simulated["exposure"], "diagnostics": diagnostics,
            "rates": rates}


def actuarial_submission(packet_dir: Path, data: dict, county_state: np.ndarray,
                         age_sex_paths: np.ndarray, fertility_rate: float,
                         release_rows: list[dict], projection_rows: list[dict],
                         cube: np.ndarray, suppress_below: float, out_dir: Path,
                         band_labels, sex_labels,
                         params: LayerParams = LayerParams()) -> dict | None:
    """Run the layer and write the four version-four files, or return None.

    None means the packet carries no experience file and no reserve block, which is a
    version-three packet; the caller then writes its version-three submission unchanged.
    Both strong lines and every control go through this one entry point, so a schema
    change on the generator side is read in a single place.
    """
    experience = load_experience(packet_dir)
    if experience is None:
        return None
    try:
        ac = read_actuarial_contract(data["contract"], county_state)
    except MissingActuarialInputs:
        return None
    result = actuarial_layer(data, county_state, age_sex_paths, ac, experience,
                             fertility_rate, params)
    rows = widen_release_rows(release_rows) + result["rate_rows"]
    write_actuarial_submission(out_dir, rows, projection_rows, cube, suppress_below,
                               result["reserve"], band_labels, sex_labels)
    result["contract"] = ac
    result["release"] = rows
    return result
