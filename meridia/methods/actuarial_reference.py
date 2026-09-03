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

import hashlib
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
                         RATE_EXTRA_COLUMNS, RATE_LEVELS, RESERVE_COLUMNS,
                         V4_PROJECTION_COLUMNS, V4_RELEASE_COLUMNS, empirical_tail,
                         perfect_information_allocation,
                         proportional_baseline_allocation)
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
    R and ``reserve_baseline_share`` fixes the comparison allocation before any
    submission is seen. The submitted allocation must be finite, nonnegative, and sum
    to R; its tail forecasts are separate scored quantities.
    """
    obligation: ObligationContract
    region_of_county: np.ndarray
    n_regions: int
    reserve_total: float
    reserve_weights: np.ndarray
    anchor_item: str
    anchor_sensitivity: float
    anchor_specificity: float
    anchor_window_months: int
    experience_years: int
    experience_file: str
    experience_last_tick: int
    shock_family: dict | None = None
    reserve_baseline_share: np.ndarray | None = None

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


# Which simulated quantity each published multiplier of the shock family drives. The
# contract names the multipliers; this table says where each one enters the continuation,
# and a multiplier whose name is not here is carried in the parsed kind and left unused
# rather than silently mapped onto the wrong quantity.
SHOCK_TARGETS = {"mortality_multiplier": "mortality",
                 "admission_multiplier": "incidence",
                 "leave_home_multiplier": "migration",
                 "fertility_multiplier": "fertility"}

# The generator publishes this expression as prose rather than executable code.  The
# reader recognizes the one supported expression instead of evaluating packet text.
# Keeping the canonical spelling here also gives the evidence receipt an unambiguous
# statement of the model it used.
REGIONAL_LOADING_FORMULA = "1 + L_r * (m - 1)"
REGIONAL_LOADING_TARGETS = ("mortality", "incidence")
REGIONAL_LOADING_IDENTIFICATION_THRESHOLD = 0.80


def _parse_regional_loading(block: dict) -> dict:
    """Parse the public regional-loading band and its declared formula.

    Older V4 packets have neither field and remain valid with a national shock.  A
    packet with only half of the declaration is rejected: silently assuming a formula
    for a published band would make the participant contract incomplete.  Packet prose
    is never executed; it must contain the canonical affine rule the generator uses.
    """
    raw_band = block.get("regional_loading_band")
    raw_rule = block.get("regional_loading_formula", block.get("regional_loading"))
    if raw_band is None and raw_rule is None:
        return {}
    if raw_band is None or raw_rule is None:
        raise MissingActuarialInputs(
            "shock_family regional loading needs both a band and a formula"
        )
    if not isinstance(raw_band, (list, tuple)) or len(raw_band) != 2:
        raise MissingActuarialInputs(
            "shock_family.regional_loading_band must contain two bounds"
        )
    band = (float(raw_band[0]), float(raw_band[1]))
    if not np.isfinite(band).all() or band[0] < 0.0 or band[1] <= band[0]:
        raise MissingActuarialInputs(
            "shock_family.regional_loading_band must be finite, nonnegative, and ordered"
        )
    if isinstance(raw_rule, dict):
        raw_rule = raw_rule.get("formula")
    if not isinstance(raw_rule, str):
        raise MissingActuarialInputs(
            "shock_family regional-loading formula must be public text"
        )
    normalized = "".join(raw_rule.split())
    expected = "".join(REGIONAL_LOADING_FORMULA.split())
    if expected not in normalized:
        raise MissingActuarialInputs(
            "unsupported regional-loading formula; expected "
            f"{REGIONAL_LOADING_FORMULA!r}"
        )
    return {
        "regional_loading_band": band,
        "regional_loading_formula": REGIONAL_LOADING_FORMULA,
    }


def read_shock_family(contract: dict) -> dict | None:
    """The published shock family, parsed into the form the continuation draws from.

    Protocol section 6 gives the truth ensemble independent futures, and the generator
    publishes the family those futures are drawn from: an annual rate and, per kind, the
    multipliers that move together on one draw. A continuation that ignores it prices a
    world with no systematic risk, so its tail is the demographic noise of a large stock
    and nothing else. Reading the family is the difference between a predictive
    distribution and a point projection with a spread around it.
    """
    block = contract.get("shock_family")
    if not isinstance(block, dict):
        return None
    kinds = block.get("kinds")
    if not isinstance(kinds, dict) or not kinds:
        return None
    parsed = []
    for name, fields in kinds.items():
        if not isinstance(fields, dict):
            continue
        entry = {"name": str(name)}
        for key, value in fields.items():
            target = SHOCK_TARGETS.get(str(key))
            if target is None or not isinstance(value, (list, tuple)) or len(value) != 2:
                continue
            entry[target] = (float(value[0]), float(value[1]))
        if len(entry) > 1:
            parsed.append(entry)
    if not parsed:
        return None
    family = {"annual_rate": float(block.get("annual_rate", 0.0)), "kinds": parsed}
    family.update(_parse_regional_loading(block))
    return family


def shock_range_for(shock_family: dict | None, target: str = "mortality"):
    """The published multiplier range of the kind that moves one simulated quantity."""
    if not shock_family:
        return None
    for kind in shock_family["kinds"]:
        if target in kind:
            return kind[target]
    return None


def expected_shock_loading(shock_family: dict | None, target: str = "mortality") -> float:
    """How much a shock family lifts a quantity on average, over ordinary and shock years.

    One Bernoulli a year at the published rate, then one kind drawn uniformly, then a
    single uniform draw shared by that kind's multipliers. The expected multiplier is
    therefore one plus the chance of drawing a kind that moves this quantity times the
    average lift of that kind. It is what an estimate read off several years already
    contains, and what the continuation adds on top: charging both would price the same
    epidemic twice.
    """
    if not shock_family or not shock_family["kinds"]:
        return 0.0
    kinds = shock_family["kinds"]
    rate = float(shock_family["annual_rate"]) / len(kinds)
    loading = 0.0
    for kind in kinds:
        if target in kind:
            lo, hi = kind[target]
            loading += rate * (0.5 * (lo + hi) - 1.0)
    return float(loading)


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
    baseline_share = _first(reserve, ("baseline_share",), None)
    if baseline_share is None:
        missing.append('contract["reserve"]["baseline_share"]')
    if missing:
        raise MissingActuarialInputs(
            "contract.json carries no complete actuarial block; missing "
            + "; ".join(missing))
    total = float(total)
    if not np.isfinite(total) or total < 0.0:
        raise MissingActuarialInputs(
            "contract reserve total must be finite and nonnegative"
        )
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
    if weights.shape != (n_regions,):
        raise MissingActuarialInputs(
            "contract reserve weights must contain one value per region"
        )
    if not np.isfinite(weights).all() or (weights < 0.0).any():
        raise MissingActuarialInputs(
            "contract reserve weights must be finite and nonnegative"
        )
    baseline_share = np.asarray(baseline_share, dtype=np.float64)
    if baseline_share.shape != (n_regions,):
        raise MissingActuarialInputs(
            "contract reserve baseline_share must contain one value per region"
        )
    if not np.isfinite(baseline_share).all() or (baseline_share < 0.0).any():
        raise MissingActuarialInputs(
            "contract reserve baseline_share must be finite and nonnegative"
        )
    if float(baseline_share.sum()) <= 0.0:
        raise MissingActuarialInputs(
            "contract reserve baseline_share must have positive total weight"
        )
    return ActuarialContract(
        obligation=obligation,
        region_of_county=region_of_county,
        n_regions=n_regions,
        reserve_total=total,
        reserve_weights=weights,
        reserve_baseline_share=baseline_share,
        anchor_item=str(_first(anchor, ("item", "column"), "recent_hospitalization")),
        anchor_sensitivity=float(_first(anchor, ("sensitivity", "se"), 1.0)),
        anchor_specificity=float(_first(anchor, ("specificity", "sp"), 1.0)),
        anchor_window_months=int(_first(anchor, ("window_months", "window"), 12)),
        experience_years=int(_first(history, ("years",), 5)),
        experience_file=str(_first(history, ("file",), EXPERIENCE_FILENAMES[0])),
        experience_last_tick=int(_first(history, ("last_year_ends_at_tick",),
                                        contract["ticks"]["revised"])),
        shock_family=read_shock_family(contract),
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
    reserve = contract.get("reserve") or {}
    region_spec = reserve.get("regions", "state")
    county_level = isinstance(region_spec, str) and region_spec.startswith("count")
    n_probe = int(contract.get("n_counties" if county_level else "n_states", 1))
    try:
        read_actuarial_contract(contract, np.arange(max(n_probe, 1), dtype=np.int64))
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


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-4, 1.0 - 1e-4)
    return np.log(p / (1.0 - p))


def fit_survey_response(survey, county_state: np.ndarray, urbanity: np.ndarray,
                        register_ages: np.ndarray | None = None) -> dict:
    """The world's own unit-response model, fitted on what the survey itself publishes.

    The contract declares the form: logit p = a_0 + a_age (head age - 45) + a_income
    (log income - median) + a_urban urbanity. Version four draws the four coefficients per
    world, so a response model fitted once on a development world is a model of a
    different world. All three coefficients are estimable from participant files:

    - the survey names the households each sampling unit drew, so the response rate is
      observed per unit and its regression on the county's urbanity gives a_0 and a_urban;
    - a logistic response tilts the mean of a covariate among responders away from the
      population mean by about the coefficient times the covariate's variance times the
      non-response share, which inverts to a_age against the register's own age
      distribution.

    The money coefficient is left at zero and is not fitted. The same inversion needs the
    population's money distribution in the units the survey reports money in, and the two
    money sources sit on scales that differ by a factor no participant file states. A
    county-level comparison does not recover it either: what would identify it is the
    difference between a county's response rate and its money level, and both are driven
    by that county's urbanity, so the two regressors are the same one. It is recorded as
    not identified rather than filled with a statistic that answers a different question.

    The propensities are returned per responding household so a weight can be divided by
    one. Nothing here reads a truth file, and the health anchor is where the corrected
    weights are spent: response is selective on age, which carries frailty, so an anchor
    read on design weights alone reports the admissions of the people who answered.
    """
    out = {"fitted": False, "intercept": 0.0, "age": 0.0, "income": 0.0, "urban": 0.0,
           "response_rate": float("nan")}
    needed = {"household", "county", "age", "design_weight"}
    if not needed.issubset(set(survey.columns)):
        return out
    frame = survey
    heads = frame.groupby(["county", "household"], sort=False).agg(
        age=("age", "max"),
        income=("income", "sum") if "income" in frame.columns else ("age", "size"),
        weight=("design_weight", "first"),
        psu=("psu", "first") if "psu" in frame.columns else ("county", "first"),
        sampled=("psu_sampled_households", "first")
        if "psu_sampled_households" in frame.columns else ("county", "size"))
    heads = heads.reset_index()
    if "psu_sampled_households" not in frame.columns:
        return out
    unit = heads.groupby(["county", "psu"], sort=False).agg(
        responded=("household", "size"), sampled=("sampled", "first")).reset_index()
    unit = unit[unit["sampled"] > 0]
    if len(unit) < 8:
        return out
    rate = np.clip(unit["responded"].to_numpy(dtype=np.float64) /
                   unit["sampled"].to_numpy(dtype=np.float64), 1e-3, 0.999)
    x = np.asarray(urbanity, dtype=np.float64)[unit["county"].to_numpy(dtype=np.int64)]
    w = unit["sampled"].to_numpy(dtype=np.float64) * rate * (1.0 - rate)
    mx = float((w * x).sum() / max(w.sum(), 1e-9))
    my = float((w * _logit(rate)).sum() / max(w.sum(), 1e-9))
    denominator = float((w * (x - mx) ** 2).sum())
    a_urban = float((w * (x - mx) * (_logit(rate) - my)).sum() / denominator) \
        if denominator > 0 else 0.0
    a_urban = float(np.clip(a_urban, -3.0, 3.0))
    mean_rate = float(unit["responded"].sum() / max(unit["sampled"].sum(), 1))
    missing = max(1.0 - mean_rate, 1e-3)
    a_age = 0.0
    age = heads["age"].to_numpy(dtype=np.float64)
    if register_ages is not None and len(register_ages) > 100:
        population = np.asarray(register_ages, dtype=np.float64)
        variance = float(population.var())
        if variance > 1e-6:
            a_age = float(np.clip((age.mean() - population.mean()) / (variance * missing),
                                  -0.2, 0.2))
    centre = a_urban * float(np.average(np.asarray(urbanity, dtype=np.float64)))
    intercept = float(_logit(np.array([mean_rate]))[0]) - centre
    propensity = intercept + a_urban * np.asarray(urbanity, dtype=np.float64)[
        heads["county"].to_numpy(dtype=np.int64)] + a_age * (age - 45.0)
    p = 1.0 / (1.0 + np.exp(-np.clip(propensity, -8.0, 8.0)))
    household = heads[["county", "household"]].copy()
    household["propensity"] = np.clip(p, 0.05, 1.0)
    return {"fitted": True, "intercept": intercept, "age": a_age, "income": 0.0,
            "income_identified": False, "urban": a_urban, "response_rate": mean_rate,
            "household": household, "n_units": int(len(unit))}


def nonresponse_weights(survey, response: dict) -> np.ndarray:
    """Design weights divided by the fitted propensity, renormalised inside each county.

    The level of the design weights is what the sampling design already fixed; what
    non-response moves is the composition inside a county, so the correction is applied
    to the composition and the county total is left where the design put it.
    """
    weight_column = "weight" if "weight" in survey.columns else "design_weight"
    base = survey[weight_column].to_numpy(dtype=np.float64)
    if not response.get("fitted") or "household" not in response:
        return base
    table = response["household"].set_index(["county", "household"])["propensity"]
    index = list(zip(survey["county"].to_numpy(dtype=np.int64),
                     survey["household"].to_numpy()))
    propensity = table.reindex(index).to_numpy(dtype=np.float64)
    propensity = np.where(np.isfinite(propensity), propensity, 1.0)
    adjusted = base / np.maximum(propensity, 0.05)
    county = survey["county"].to_numpy(dtype=np.int64)
    n = int(county.max()) + 1 if len(county) else 1
    before = np.bincount(county, weights=base, minlength=n)
    after = np.bincount(county, weights=adjusted, minlength=n)
    factor = np.where(after > 0, before / np.maximum(after, 1e-9), 1.0)
    return adjusted * factor[county]


def anchor_prevalence(survey, ac: ActuarialContract, county_state: np.ndarray,
                      heaping: float = 0.0) -> dict:
    """Recent-admission prevalence from the independent survey item, by state, sex and
    band, corrected for the item's declared error rates.

    The survey sample is drawn without reference to health-source inclusion, so this is
    an external anchor rather than a restatement of the archive. Its sampling variance
    comes from the design weights through the effective sample size, so a thin cell is
    shrunk toward its state margin rather than believed.

    ``heaping`` is the measured share of reported ages sitting on a multiple of five; at
    a positive value the weights are spread back over the neighbouring ages before the
    bands are cut.
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
    if heaping and heaping > 0.0:
        # Reported ages heap onto multiples of five, and two band boundaries the
        # obligation is priced across sit on one. The weight and the item are moved back
        # over the neighbouring ages before the bands are formed, so a heaped respondent
        # is not read as a member of the band their reported age landed in.
        raw_age = np.clip(frame["age"].to_numpy(dtype=np.int64), 0, MAX_AGE)
        cell = state * (MAX_AGE + 1) * 2 + raw_age * 2 + sex
        size_age = n_states * (MAX_AGE + 1) * 2
        shape_age = (n_states, MAX_AGE + 1, 2)
        cubes = {}
        for name, values in (("w", w), ("w2", w ** 2), ("wy", w * y)):
            cubes[name] = deheap_age_cube(
                np.bincount(cell, weights=values, minlength=size_age).reshape(shape_age),
                heaping)
        collapse = band_matrix(MAX_AGE)
        total_w = np.einsum("ba,sax->sbx", collapse, cubes["w"]).reshape(-1)
        total_w2 = np.einsum("ba,sax->sbx", collapse, cubes["w2"]).reshape(-1)
        total_wy = np.einsum("ba,sax->sbx", collapse, cubes["wy"]).reshape(-1)
        with np.errstate(invalid="ignore", divide="ignore"):
            observed = np.where(total_w > 0, total_wy / np.maximum(total_w, 1e-9), np.nan)
            kish = np.where(total_w2 > 0, total_w ** 2 / np.maximum(total_w2, 1e-9), 0.0)
        prevalence = rogan_gladen(np.nan_to_num(observed, nan=0.0),
                                  ac.anchor_sensitivity,
                                  ac.anchor_specificity).reshape(shape)
        prevalence[np.isnan(observed).reshape(shape)] = np.nan
        return {"prevalence": prevalence, "effective_n": kish.reshape(shape),
                "observed": observed.reshape(shape), "available": True}
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


def inclusion_surface(raw: np.ndarray, expected: np.ndarray, prevalence: np.ndarray,
                      completeness: np.ndarray | None, pooled: float) -> dict:
    """Fit the inclusion probability on the two quantities the contract says move it.

    The declared interaction is administrative completeness times the dependence of
    missingness on the target, entering the health source's inclusion logit as a slope on
    log frailty. Frailty is latent and the anchor is its only handle, so the cell's own
    anchored prevalence stands in for it, and the register's shortfall against the
    published benchmark stands in for completeness. Fitting

        log pi = a + b completeness + c log prevalence + d completeness log prevalence

    by expected count gives a surface each thin cell is shrunk toward, in place of one
    pooled number that says the selection is the same everywhere. The interaction term is
    the one the contract declares; the two main effects are what identify it.
    """
    usable = np.isfinite(raw) & np.isfinite(expected) & (expected > 0) & (raw > 0)
    flat_target = np.full(raw.shape, pooled)
    if usable.sum() < 8:
        return {"target": flat_target, "fitted": False, "coefficients": np.zeros(4)}
    prevalence = np.asarray(prevalence, dtype=np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        log_prevalence = np.log(np.maximum(prevalence, 1e-6))
    centre = float(np.average(log_prevalence[usable], weights=expected[usable]))
    log_prevalence = log_prevalence - centre
    if completeness is None:
        gap = np.zeros(raw.shape)
    else:
        completeness = np.asarray(completeness, dtype=np.float64)
        gap = np.broadcast_to(completeness[:, None, None], raw.shape) - \
            float(np.mean(completeness))
    columns = [np.ones(raw.shape), gap, log_prevalence, gap * log_prevalence]
    design = np.stack([c[usable] for c in columns], axis=1)
    weight = np.sqrt(expected[usable])
    response = np.log(np.maximum(raw[usable], 1e-6))
    solution, *_ = np.linalg.lstsq(design * weight[:, None], response * weight, rcond=None)
    fitted = np.exp(sum(solution[i] * columns[i] for i in range(4)))
    target = np.clip(np.where(np.isfinite(fitted), fitted, pooled), *INCLUSION_BOUNDS)
    return {"target": target, "fitted": True, "coefficients": solution}


def inclusion_probability(archive_count: np.ndarray, anchor: dict,
                          population: np.ndarray,
                          completeness: np.ndarray | None = None) -> dict:
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
    surface = inclusion_surface(raw, expected, np.asarray(anchor["prevalence"]),
                                completeness, pooled)
    target = surface["target"]
    pi = z * np.nan_to_num(raw, nan=pooled) + (1.0 - z) * target
    return {"pi": np.clip(pi, *INCLUSION_BOUNDS), "pooled": pooled, "available": True,
            "raw": raw, "expected": expected, "surface": surface}


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


def year_effects(exposure: np.ndarray, counts: np.ndarray,
                 min_exposure: float = 500.0) -> dict:
    """The file's common year effect, free of the cell composition.

    Each cell contributes its log rate net of its own five-year mean, and the year's
    effect is the count-weighted mean of those deviations. This is the Lee and Carter
    (1992) reduction with a flat age response: a level per band, sex and state, and one
    series over time. Counts are the weights because the sampling variance of a log rate
    is one over the count, so exposure weighting would hand the series to the young bands
    that carry the most person-years and the fewest deaths.
    """
    ok = (exposure >= min_exposure) & (counts > 0)
    shape = exposure.shape
    n_years = shape[0]
    if ok.sum() < 8:
        return {"effect": np.zeros(n_years), "weight": np.zeros(n_years),
                "n_cells": int(ok.sum()), "fitted": False}
    log_rate = np.zeros(shape)
    with np.errstate(invalid="ignore", divide="ignore"):
        log_rate[ok] = np.log(counts[ok] / exposure[ok])
    weight = np.where(ok, counts, 0.0)
    complete = (weight > 0).sum(axis=0) >= 3      # a cell needs three years to fix a level
    weight = weight * complete[None]
    total = weight.sum(axis=0)
    cell_mean = np.where(total > 0, (weight * log_rate).sum(axis=0) / np.maximum(total, 1e-9), 0.0)
    deviation = np.where(weight > 0, log_rate - cell_mean[None], 0.0)
    per_year = weight.reshape(n_years, -1).sum(axis=1)
    effect = np.where(per_year > 0,
                      (weight * deviation).reshape(n_years, -1).sum(axis=1) /
                      np.maximum(per_year, 1e-9), 0.0)
    return {"effect": effect, "weight": per_year, "n_cells": int(complete.sum()),
            "fitted": True}


def _theil_sen(y: np.ndarray) -> float:
    """Median pairwise slope: the initial value the shock weighting refines."""
    n = len(y)
    slopes = [(y[j] - y[i]) / (j - i) for i in range(n) for j in range(i + 1, n)]
    return float(np.median(slopes)) if slopes else 0.0


def estimate_improvement(exposure: np.ndarray, counts: np.ndarray,
                         min_exposure: float = 500.0,
                         shock_family: dict | None = None,
                         shock_range: tuple[float, float] | None = None) -> dict:
    """Annual log drift in a rate, with the published shock family taken out of it.

    Five annual points fix a level and a slope. They do not fix an age-varying response,
    and pretending otherwise would put a spurious age pattern into the projection.

    What they also do not fix is a year that carries one of the published shocks. The
    contract states the family: a shock arrives at an annual rate and multiplies the year
    by a factor inside a published range. One such year inside five moves an ordinary
    least-squares slope by more than the whole width of the regime axis the drift is
    trying to measure. Each year therefore carries a posterior probability of being a
    shock year, computed under the published rate and range against the year's own
    sampling noise, and the slope is fitted on the years net of their expected shock and
    weighted by one minus that posterior.

    Measured on the twelve development worlds against the realized intensity: the plain
    weighted fit has a bias of -0.0239 and a root mean square error of 0.0411, and this
    estimator has a bias of -0.0035 and a root mean square error of 0.0212, against an
    axis whose development band is 0.058 wide.
    """
    n_years = exposure.shape[0]
    if n_years < 3:
        return {"drift": 0.0, "drift_se": 0.02, "fitted": False, "n_cells": 0,
                "shock_posterior": np.zeros(max(n_years, 0))}
    series = year_effects(exposure, counts, min_exposure)
    if not series["fitted"]:
        return {"drift": 0.0, "drift_se": 0.02, "fitted": False,
                "n_cells": series["n_cells"], "shock_posterior": np.zeros(n_years)}
    effect = series["effect"]
    count = np.maximum(series["weight"], 1.0)
    # A year effect is a weighted mean of log rates, so its sampling standard deviation
    # is one over the root of the deaths behind it. The floor is the year to year process
    # spread an ordinary year still carries.
    sd = np.sqrt(1.0 / count + 0.02 ** 2)
    year = np.arange(n_years, dtype=np.float64)
    year = year - year.mean()
    if shock_range is None:
        shock_range = shock_range_for(shock_family)
    rate = float(shock_family["annual_rate"]) if shock_family else 0.0
    posterior = np.zeros(n_years)
    slope = _theil_sen(effect)
    expected_shock = 0.0
    if shock_range is not None and 0.0 < rate < 1.0 and shock_range[1] > shock_range[0] > 0:
        grid = np.linspace(np.log(shock_range[0]), np.log(shock_range[1]), 40)
        expected_shock = float(grid.mean())
        for _ in range(30):
            residual = effect - slope * year
            residual = residual - np.median(residual)
            base = np.exp(-0.5 * (residual / sd) ** 2) / sd
            shocked = np.mean(np.exp(-0.5 * ((residual[:, None] - grid[None, :]) /
                                             sd[:, None]) ** 2) / sd[:, None], axis=1)
            posterior = rate * shocked / np.maximum(rate * shocked + (1.0 - rate) * base,
                                                    1e-300)
            cleaned = effect - posterior * expected_shock
            w = (1.0 - posterior) * count
            if w.sum() <= 0:
                break
            mx = float((w * year).sum() / w.sum())
            my = float((w * cleaned).sum() / w.sum())
            denominator = float((w * (year - mx) ** 2).sum())
            if denominator <= 0:
                break
            new = float((w * (year - mx) * (cleaned - my)).sum() / denominator)
            if abs(new - slope) < 1e-10:
                slope = new
                break
            slope = new
    else:
        w = count
        mx = float((w * year).sum() / w.sum())
        my = float((w * effect).sum() / w.sum())
        denominator = float((w * (year - mx) ** 2).sum())
        if denominator > 0:
            slope = float((w * (year - mx) * (effect - my)).sum() / denominator)
    # The slope is a linear combination of the year effects, so its variance is the sum
    # of the squared coefficients times each year's own variance: the sampling variance
    # of that year effect plus whatever process spread is left after the fit. A shock
    # year keeps its weight in the variance even though it lost it in the fit, which is
    # what makes a file with a shock in it report a wider drift than a clean one.
    cleaned = effect - posterior * expected_shock
    w = (1.0 - posterior) * count
    if w.sum() <= 0:
        w = count.copy()
    mx = float((w * year).sum() / w.sum())
    my = float((w * cleaned).sum() / w.sum())
    denominator = float((w * (year - mx) ** 2).sum())
    residual = cleaned - (my + slope * (year - mx))
    extra = max(float((w * residual ** 2).sum() / max(w.sum(), 1e-9)) -
                float((w * sd ** 2).sum() / max(w.sum(), 1e-9)), 0.0)
    variance = sd ** 2 + extra
    if denominator > 0:
        coefficient = w * (year - mx) / denominator
        drift_se = float(np.sqrt(max(float((coefficient ** 2 * variance).sum()), 1e-12)))
    else:
        drift_se = 0.05
    return {"drift": float(np.clip(slope, -0.15, 0.15)),
            "drift_se": float(min(max(drift_se, 1e-4), 0.05)),
            "fitted": True, "n_cells": series["n_cells"],
            "year_effect": effect, "shock_posterior": posterior}


def _participant_experience_digest(experience: dict) -> str:
    """Digest the public arrays that identify the regional-loading fit."""
    digest = hashlib.sha256()
    for name in ("years", "exposure", "deaths", "qualifying_events"):
        value = np.ascontiguousarray(np.asarray(experience[name]))
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(str(value.dtype).encode("ascii") + b"\0")
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.tobytes())
    return digest.hexdigest()


def _joint_shock_kind(shock_family: dict | None) -> tuple[dict | None, float]:
    """Return the published kind visible in both priced experience signals.

    ``annual_rate`` is the chance of drawing any kind.  The identifiable prior is the
    chance of drawing one that moves mortality and admissions together, so a family
    with three uniformly selected kinds contributes one third of that annual rate.
    """
    if not shock_family or not shock_family.get("kinds"):
        return None, 0.0
    kinds = list(shock_family["kinds"])
    joint = [kind for kind in kinds if all(target in kind for target in REGIONAL_LOADING_TARGETS)]
    if not joint:
        return None, 0.0
    probability = float(shock_family.get("annual_rate", 0.0)) * len(joint) / len(kinds)
    return joint[0], float(np.clip(probability, 0.0, 1.0))


def _combine_shock_posteriors(
    mortality: np.ndarray, incidence: np.ndarray, prior: float
) -> np.ndarray:
    """Combine conditionally independent target posteriors without counting the prior twice."""
    mortality = np.asarray(mortality, dtype=np.float64)
    incidence = np.asarray(incidence, dtype=np.float64)
    if mortality.shape != incidence.shape:
        raise ValueError("mortality and incidence shock posteriors must align")
    if not 0.0 < prior < 1.0:
        return np.zeros_like(mortality)
    epsilon = 1e-12
    p_m = np.clip(mortality, epsilon, 1.0 - epsilon)
    p_i = np.clip(incidence, epsilon, 1.0 - epsilon)
    prior_odds = prior / (1.0 - prior)
    odds = (p_m / (1.0 - p_m)) * (p_i / (1.0 - p_i)) / prior_odds
    return odds / (1.0 + odds)


def _counterfactual_state_lift(
    exposure: np.ndarray, counts: np.ndarray, shock_year_index: int
) -> tuple[np.ndarray, np.ndarray]:
    """State rate lift in one year against a leave-that-year-out trend.

    Counts and exposures are collapsed over public age bands and sex.  The fitted
    counterfactual uses every other public year.  The returned precision includes a
    small ordinary-year process floor, so large states do not claim that a five-point
    trend is known to machine precision merely because its Poisson count is large.
    """
    exposure = np.asarray(exposure, dtype=np.float64).sum(axis=(2, 3))
    counts = np.asarray(counts, dtype=np.float64).sum(axis=(2, 3))
    if exposure.shape != counts.shape or exposure.ndim != 2:
        raise ValueError("experience counts and exposure must be [year, state, band, sex]")
    n_years, n_states = exposure.shape
    if not 0 <= int(shock_year_index) < n_years:
        raise ValueError("shock year index is outside the experience window")
    year = np.arange(n_years, dtype=np.float64)
    lift = np.zeros(n_states)
    precision = np.zeros(n_states)
    for state in range(n_states):
        valid = (exposure[:, state] > 0.0) & np.isfinite(exposure[:, state]) & \
            np.isfinite(counts[:, state])
        fit = valid.copy()
        fit[int(shock_year_index)] = False
        if fit.sum() < 2 or not valid[int(shock_year_index)]:
            continue
        t = year[fit]
        observed_count = np.maximum(counts[fit, state], 0.0)
        y = np.log((observed_count + 0.5) / np.maximum(exposure[fit, state], 1.0))
        weight = observed_count + 0.5
        centre = float(np.average(t, weights=weight))
        denominator = float((weight * (t - centre) ** 2).sum())
        slope = 0.0 if denominator <= 0.0 else float(
            (weight * (t - centre) * (y - np.average(y, weights=weight))).sum()
            / denominator
        )
        slope = float(np.clip(slope, -0.15, 0.15))
        intercept = float(np.average(y - slope * (t - centre), weights=weight))
        expected_rate = float(np.exp(intercept + slope * (year[shock_year_index] - centre)))
        expected_count = max(expected_rate * exposure[shock_year_index, state], 0.0)
        actual_count = max(counts[shock_year_index, state], 0.0)
        ratio = (actual_count + 0.5) / (expected_count + 0.5)
        lift[state] = ratio - 1.0
        log_variance = 1.0 / (actual_count + 0.5) + 1.0 / (expected_count + 0.5)
        precision[state] = 1.0 / (log_variance + 0.03 ** 2)
    return lift, precision


def _loading_posterior_grid(
    experience: dict, shock_year_index: int, shock_kind: dict,
    band: tuple[float, float]
) -> dict:
    """Fit the public affine regional loading over the shared shock-magnitude grid."""
    mortality_lift, mortality_precision = _counterfactual_state_lift(
        experience["exposure"], experience["deaths"], shock_year_index
    )
    incidence_lift, incidence_precision = _counterfactual_state_lift(
        experience["exposure"], experience["qualifying_events"], shock_year_index
    )
    u = np.linspace(0.005, 0.995, 100)
    mortality_range = shock_kind["mortality"]
    incidence_range = shock_kind["incidence"]
    mortality_national_lift = (
        mortality_range[0] + u * (mortality_range[1] - mortality_range[0]) - 1.0
    )
    incidence_national_lift = (
        incidence_range[0] + u * (incidence_range[1] - incidence_range[0]) - 1.0
    )
    n_states = mortality_lift.shape[0]
    loading = np.zeros((len(u), n_states))
    loading_se = np.zeros_like(loading)
    score = np.zeros(len(u))
    for index, (mortality_x, incidence_x) in enumerate(
        zip(mortality_national_lift, incidence_national_lift)
    ):
        denominator = (
            mortality_precision * mortality_x ** 2
            + incidence_precision * incidence_x ** 2
        )
        numerator = (
            mortality_precision * mortality_x * mortality_lift
            + incidence_precision * incidence_x * incidence_lift
        )
        raw = np.where(denominator > 0.0, numerator / np.maximum(denominator, 1e-12),
                       0.5 * (band[0] + band[1]))
        loading[index] = np.clip(raw, *band)
        loading_se[index] = np.where(
            denominator > 0.0,
            np.sqrt(1.0 / np.maximum(denominator, 1e-12)),
            (band[1] - band[0]) / np.sqrt(12.0),
        )
        score[index] = float(
            (mortality_precision * (mortality_lift - loading[index] * mortality_x) ** 2).sum()
            + (incidence_precision * (incidence_lift - loading[index] * incidence_x) ** 2).sum()
        )
    relative = -0.5 * (score - float(np.nanmin(score)))
    probability = np.exp(np.clip(relative, -700.0, 0.0))
    probability = probability / max(float(probability.sum()), 1e-300)
    mean = (probability[:, None] * loading).sum(axis=0)
    second = (probability[:, None] * (loading ** 2 + loading_se ** 2)).sum(axis=0)
    standard_deviation = np.sqrt(np.maximum(second - mean ** 2, 0.0))
    return {
        "magnitude_grid": u,
        "grid_probability": probability,
        "loading_grid": loading,
        "loading_se_grid": loading_se,
        "loading_mean": np.clip(mean, *band),
        "loading_sd": np.clip(standard_deviation, 1e-6, band[1] - band[0]),
        "mortality_observed_lift": mortality_lift,
        "incidence_observed_lift": incidence_lift,
    }


def infer_regional_shock_loadings(
    experience: dict,
    shock_family: dict | None,
    identification_threshold: float = REGIONAL_LOADING_IDENTIFICATION_THRESHOLD,
) -> dict:
    """Infer the predictive regional-loading law from participant-visible evidence.

    With no identifiable mortality/admission shock, the realized world loading is not
    identified.  The honest predictive distribution therefore integrates over the
    uniform public band separately for every outer path.  With a jointly identifiable
    shock year, a magnitude/loading grid is fitted to state death and qualifying-event
    lifts and paths sample that participant-data posterior, clipped to the same band.
    No path or parameter accepts a retained realized loading.
    """
    required = ("years", "exposure", "deaths", "qualifying_events")
    missing = [name for name in required if name not in experience]
    if missing:
        raise MissingActuarialInputs(
            f"regional shock loading needs participant experience fields {missing}"
        )
    digest = _participant_experience_digest(experience)
    band = None if not shock_family else shock_family.get("regional_loading_band")
    shock_kind, target_probability = _joint_shock_kind(shock_family)
    base = {
        "band": None if band is None else tuple(float(v) for v in band),
        "formula": None if band is None else shock_family.get("regional_loading_formula"),
        "evidence_source": "participant experience_history and public shock_family",
        "experience_digest": digest,
        "input_fields": list(required),
        "uses_retained_realized_loadings": False,
        "target_shock_annual_probability": target_probability,
        "identification_threshold": float(identification_threshold),
    }
    exposure = np.asarray(experience["exposure"], dtype=np.float64)
    if exposure.ndim != 4:
        raise ValueError("experience exposure must be [year, state, band, sex]")
    n_years, n_states = exposure.shape[:2]
    if band is None or shock_kind is None:
        return {
            **base,
            "mode": "national_only",
            "shock_year_posterior": np.zeros(n_years),
            "identified_year_index": None,
            "identified_year": None,
            "loading_mean": np.ones(n_states),
            "loading_sd": np.zeros(n_states),
        }
    band = tuple(float(v) for v in band)
    if not 0.0 <= float(identification_threshold) <= 1.0:
        raise ValueError("identification threshold must be between zero and one")
    target_family = {"annual_rate": target_probability, "kinds": [shock_kind]}
    mortality = estimate_improvement(
        exposure,
        np.asarray(experience["deaths"], dtype=np.float64),
        shock_family=target_family,
        shock_range=shock_kind["mortality"],
    )
    incidence = estimate_improvement(
        exposure,
        np.asarray(experience["qualifying_events"], dtype=np.float64),
        shock_family=target_family,
        shock_range=shock_kind["incidence"],
    )
    joint = _combine_shock_posteriors(
        mortality.get("shock_posterior", np.zeros(n_years)),
        incidence.get("shock_posterior", np.zeros(n_years)),
        target_probability,
    )
    selected = int(np.argmax(joint)) if len(joint) else None
    identified = selected is not None and float(joint[selected]) >= identification_threshold
    years = np.asarray(experience["years"])
    midpoint = 0.5 * (band[0] + band[1])
    if not identified:
        return {
            **base,
            "mode": "public_band_marginalization",
            "shock_year_posterior": joint,
            "mortality_shock_posterior": np.asarray(mortality["shock_posterior"]),
            "incidence_shock_posterior": np.asarray(incidence["shock_posterior"]),
            "identified_year_index": None,
            "identified_year": None,
            "loading_mean": np.full(n_states, midpoint),
            "loading_sd": np.full(n_states, (band[1] - band[0]) / np.sqrt(12.0)),
        }
    posterior = _loading_posterior_grid(experience, selected, shock_kind, band)
    return {
        **base,
        **posterior,
        "mode": "participant_experience_posterior",
        "shock_year_posterior": joint,
        "mortality_shock_posterior": np.asarray(mortality["shock_posterior"]),
        "incidence_shock_posterior": np.asarray(incidence["shock_posterior"]),
        "identified_year_index": selected,
        "identified_year": int(years[selected]),
    }


def experience_state_shares(experience: dict) -> np.ndarray:
    """How the last published year splits each band and sex across the states.

    The file's exposure is person-years read off the same pass the truth uses, so its
    composition is exact as of a year and a half before the snapshot. What it cannot say
    is the level now: the publication lag is there precisely so the most recent year is
    not a contemporaneous headcount, and every state's stock has moved since. The shares
    are the part the lag does not spoil, because what moves them is the difference in
    ageing between states over eighteen months and not the growth they share.

    Measured against the retained truth on six worlds, the state shares of the population
    at sixty-five and over are within about three percent, where a register-based
    reconstruction of the same shares is within about thirteen.
    """
    last = np.asarray(experience["exposure"], dtype=np.float64)[-1]
    total = last.sum(axis=0)
    return np.where(
        total[None] > 0,
        last / np.maximum(total[None], 1e-9),
        1.0 / max(last.shape[0], 1),
    )


def advanced_experience_state_exposure(
    experience: dict,
    years_ahead: float,
    age_profile: np.ndarray | None = None,
    mortality_drift: float = 0.0,
) -> np.ndarray:
    """Advance the last published exposure level to the reconstruction date.

    The annual exposure is an average stock centred six months before the end of its
    year. ``years_ahead`` therefore includes that half year as well as the publication
    lag. The stock advances one month at a time with the last observed deaths and net
    migration. A participant-side state by single-age by sex profile can split each
    broad experience band before ageing; without one, the split is uniform.

    This is a cohort-component level, not a share. Keeping that distinction matters for
    the elder obligation: a state share can be accurate while every state's 65-plus
    population is low by the same amount.
    """
    exposure = np.asarray(experience["exposure"], dtype=np.float64)
    deaths = np.asarray(experience["deaths"], dtype=np.float64)
    migration = np.asarray(experience["net_migration"], dtype=np.float64)
    stock = np.zeros((exposure.shape[1], MAX_AGE + 1, exposure.shape[3]))
    mortality = np.zeros_like(stock)
    migration_rate = np.zeros_like(stock)
    with np.errstate(invalid="ignore", divide="ignore"):
        band_mortality = deaths[-1] / np.maximum(exposure[-1], 1e-9)
        band_migration = migration[-1] / np.maximum(exposure[-1], 1e-9)
    profile = None if age_profile is None else np.asarray(age_profile, dtype=np.float64)
    if profile is not None and profile.shape != stock.shape:
        raise ValueError("age_profile must be (states, ages, sexes)")
    for band, (low, high) in enumerate(ACTUARIAL_AGE_BANDS):
        ages = np.arange(low, min(high, MAX_AGE) + 1)
        if profile is None:
            share = np.full((stock.shape[0], len(ages), stock.shape[2]), 1.0 / len(ages))
        else:
            raw = np.maximum(profile[:, ages, :], 0.0)
            total = raw.sum(axis=1, keepdims=True)
            share = np.where(
                total > 0,
                raw / np.maximum(total, 1e-9),
                1.0 / len(ages),
            )
        stock[:, ages, :] = exposure[-1, :, band, None, :] * share
        mortality[:, ages, :] = band_mortality[:, band, None, :]
        migration_rate[:, ages, :] = band_migration[:, band, None, :]
    for month in range(max(int(round(12.0 * years_ahead)), 0)):
        q = mortality * np.exp(float(mortality_drift) * ((month + 0.5) / 12.0))
        after = stock * np.maximum(1.0 + (migration_rate - q) / 12.0, 0.0)
        aged = after * (11.0 / 12.0)
        aged[:, 1:, :] += after[:, :-1, :] / 12.0
        aged[:, -1, :] += after[:, -1, :] / 12.0
        stock = aged
    collapse = band_matrix()
    return np.einsum("ba,sax->sbx", collapse, stock)


def advanced_experience_state_shares(
    experience: dict, years_ahead: float
) -> np.ndarray:
    """Advance the experience stock and return its state shares.

    This compatibility path is the former third-line strategy. It retains only the
    state composition; ``advanced_experience_state_exposure`` retains the absolute
    cohort-component level used by the revised third line.
    """
    banded = advanced_experience_state_exposure(experience, years_ahead)
    total = banded.sum(axis=0)
    return np.where(
        total[None] > 0,
        banded / np.maximum(total[None], 1e-9),
        1.0 / max(banded.shape[0], 1),
    )


def rake_to_cohort_component(
    paths: np.ndarray,
    county_state: np.ndarray,
    state_exposure: np.ndarray,
    survey,
    survey_weights: np.ndarray | None = None,
    minimum_level_age: int = 65,
) -> tuple[np.ndarray, dict]:
    """Reconcile cohort-component state levels with register and survey county shares.

    The experience file supplies absolute levels for bands beginning at
    ``minimum_level_age``. Younger bands use its state composition while keeping the
    reconstructed national level. Inside a state, the reconstructed county share is
    corrected toward the survey share only when the county differences exceed the
    survey's own design variance. This is an empirical small-area shrinkage step: thin
    survey cells leave the linked-register reconstruction in place.
    """
    paths = np.asarray(paths, dtype=np.float64)
    county_state = np.asarray(county_state, dtype=np.int64)
    target = np.asarray(state_exposure, dtype=np.float64).copy()
    if paths.ndim != 4:
        raise ValueError("paths must be (draws, counties, ages, sexes)")
    n_states = int(county_state.max()) + 1
    n_bands = len(ACTUARIAL_AGE_BANDS)
    if target.shape != (n_states, n_bands, paths.shape[3]):
        raise ValueError("state_exposure must be (states, bands, sexes)")

    collapse = band_matrix(paths.shape[2] - 1)
    mean_cube = paths.mean(axis=0)
    register = np.einsum("ba,cas->cbs", collapse, mean_cube)
    current_state = np.zeros_like(target)
    np.add.at(current_state, county_state, register)
    for band, (low, _) in enumerate(ACTUARIAL_AGE_BANDS):
        if low >= minimum_level_age:
            continue
        for sex_index in range(paths.shape[3]):
            total = float(current_state[:, band, sex_index].sum())
            proposed = float(target[:, band, sex_index].sum())
            if proposed > 0:
                target[:, band, sex_index] *= total / proposed
            else:
                target[:, band, sex_index] = current_state[:, band, sex_index]

    survey_count = np.zeros_like(register)
    survey_w2 = np.zeros_like(register)
    county = survey["county"].to_numpy(dtype=np.int64)
    band = band_of_age(survey["age"].to_numpy(dtype=np.int64))
    sex = survey["sex"].to_numpy(dtype=np.int64)
    if survey_weights is None:
        column = "weight" if "weight" in survey.columns else "design_weight"
        weight = survey[column].to_numpy(dtype=np.float64)
    else:
        weight = np.asarray(survey_weights, dtype=np.float64)
        if weight.shape != (len(survey),):
            raise ValueError("survey_weights must have one value per survey row")
    valid = (
        (county >= 0)
        & (county < paths.shape[1])
        & (band >= 0)
        & (sex >= 0)
        & (sex < paths.shape[3])
        & np.isfinite(weight)
        & (weight > 0)
    )
    flat = (county[valid] * n_bands + band[valid]) * paths.shape[3] + sex[valid]
    size = paths.shape[1] * n_bands * paths.shape[3]
    survey_count = np.bincount(flat, weights=weight[valid], minlength=size).reshape(
        survey_count.shape
    )
    survey_w2 = np.bincount(
        flat, weights=weight[valid] ** 2, minlength=size
    ).reshape(survey_w2.shape)

    county_share = np.zeros_like(register)
    survey_influence = np.zeros_like(register)
    for state in range(n_states):
        members = np.flatnonzero(county_state == state)
        for band_index in range(n_bands):
            for sex_index in range(paths.shape[3]):
                r = np.maximum(register[members, band_index, sex_index], 0.0)
                sw = np.maximum(survey_count[members, band_index, sex_index], 0.0)
                sw2 = np.maximum(survey_w2[members, band_index, sex_index], 0.0)
                if r.sum() <= 0:
                    continue
                r_share = r / r.sum()
                if sw.sum() <= 0 or len(members) < 2:
                    county_share[members, band_index, sex_index] = r_share
                    continue
                s_share = sw / sw.sum()
                state_neff = sw.sum() ** 2 / max(float(sw2.sum()), 1e-9)
                usable = (r_share > 0) & (s_share > 0)
                sampling = np.full(len(members), np.inf)
                sampling[usable] = (1.0 - s_share[usable]) / np.maximum(
                    s_share[usable] * state_neff, 1e-9
                )
                log_ratio = np.zeros(len(members))
                log_ratio[usable] = np.log(s_share[usable] / r_share[usable])
                finite = usable & np.isfinite(sampling)
                if finite.sum() >= 2:
                    observed = float(np.var(log_ratio[finite], ddof=1))
                    sampling_mean = float(np.mean(sampling[finite]))
                    between = max(observed - sampling_mean, 0.0)
                else:
                    between = 0.0
                influence = np.zeros(len(members))
                influence[finite] = between / np.maximum(
                    between + sampling[finite], 1e-9
                )
                adjusted = r_share * np.exp(influence * log_ratio)
                adjusted[r <= 0] = 0.0
                county_share[members, band_index, sex_index] = (
                    adjusted / adjusted.sum() if adjusted.sum() > 0 else r_share
                )
                survey_influence[members, band_index, sex_index] = influence

    county_target = target[county_state] * county_share
    factor = np.ones_like(register)
    usable = register > 0
    factor[usable] = county_target[usable] / register[usable]
    per_age = np.einsum("ba,cbs->cas", collapse, factor)
    result = paths * per_age[None]
    after_register = np.einsum("ba,cas->cbs", collapse, result.mean(axis=0))
    after_state = np.zeros_like(target)
    np.add.at(after_state, county_state, after_register)
    elder = np.asarray(
        [low >= minimum_level_age for low, _ in ACTUARIAL_AGE_BANDS], dtype=bool
    )
    return result, {
        "minimum_level_age": int(minimum_level_age),
        "state_target": target.tolist(),
        "state_before": current_state.tolist(),
        "state_after": after_state.tolist(),
        "elder_before": current_state[:, elder, :].sum(axis=(1, 2)).tolist(),
        "elder_target": target[:, elder, :].sum(axis=(1, 2)).tolist(),
        "elder_after": after_state[:, elder, :].sum(axis=(1, 2)).tolist(),
        "mean_survey_influence": (
            float(np.mean(survey_influence[usable])) if np.any(usable) else 0.0
        ),
    }


def rake_to_state_shares(
    paths: np.ndarray,
    county_state: np.ndarray,
    shares: np.ndarray,
    bound: tuple[float, float] = (0.5, 2.0),
) -> np.ndarray:
    """Move each state's share of a band and sex onto the experience file's, in place of
    the register's, and leave every national band total where the reconstruction put it.

    One factor per state, band and sex, computed on the mean of the draws and applied to
    every draw, so the spread between draws is the reconstruction's own and only the
    composition moves. The register's coverage rides the county economic gradient and the
    outpost penalty, which is what puts a state's headcount out by more than the nation's;
    this is the one published quantity that measures that composition without going
    through the register at all.
    """
    paths = np.asarray(paths, dtype=np.float64)
    county_state = np.asarray(county_state, dtype=np.int64)
    n_states = int(county_state.max()) + 1
    collapse = band_matrix(paths.shape[2] - 1)
    mean_cube = paths.mean(axis=0)
    banded = np.einsum("ba,cas->cbs", collapse, mean_cube)
    current = np.zeros((n_states,) + banded.shape[1:])
    np.add.at(current, county_state, banded)
    total = current.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        have = np.where(total[None] > 0, current / np.maximum(total[None], 1e-9), 0.0)
        factor = np.where(have > 0, np.asarray(shares, dtype=np.float64) /
                          np.maximum(have, 1e-9), 1.0)
    factor = np.clip(np.nan_to_num(factor, nan=1.0), *bound)
    per_age = np.einsum("ba,sbx->sax", collapse, factor)
    return paths * per_age[county_state][None]


def gompertz_slope(experience: dict, min_age: float = 30.0) -> dict:
    """The world's own age slope of adult mortality, from the experience file.

    log q(x) is close to linear in age above thirty, and the slope of that line is the
    Gompertz b. It is a per-world draw in version four, and it is the parameter a
    projection is most sensitive to: it sets how much of the obligation the oldest bands
    carry, and it is the second half of the declared interaction between age reporting
    error and the age slope of mortality. Five years of state by band by sex counts fix
    it far better than one snapshot does, so it is read here and not from the register.

    The fit is a count-weighted straight line through the band midpoints of the pooled
    five-year log rate, with the sampling variance of each point taken as one over its
    deaths.
    """
    exposure = experience["exposure"].sum(axis=0)
    deaths = experience["deaths"].sum(axis=0)
    midpoint = np.array([0.5 * (lo + min(hi, MAX_AGE)) for lo, hi in ACTUARIAL_AGE_BANDS])
    band_exposure = exposure.sum(axis=(0, 2))
    band_deaths = deaths.sum(axis=(0, 2))
    use = (midpoint >= min_age) & (band_deaths > 0) & (band_exposure > 0)
    if use.sum() < 3:
        return {"slope": 0.0, "slope_se": 0.0, "fitted": False}
    x = midpoint[use]
    y = np.log(band_deaths[use] / band_exposure[use])
    w = band_deaths[use]
    mx = float((w * x).sum() / w.sum())
    my = float((w * y).sum() / w.sum())
    denominator = float((w * (x - mx) ** 2).sum())
    if denominator <= 0:
        return {"slope": 0.0, "slope_se": 0.0, "fitted": False}
    slope = float((w * (x - mx) * (y - my)).sum() / denominator)
    coefficient = w * (x - mx) / denominator
    variance = 1.0 / np.maximum(w, 1.0)
    return {"slope": float(np.clip(slope, 0.0, 0.25)),
            "slope_se": float(np.sqrt(max(float((coefficient ** 2 * variance).sum()), 1e-12))),
            "fitted": True, "n_bands": int(use.sum())}


def age_heaping_intensity(ages: np.ndarray) -> dict:
    """How much of the reported age distribution sits on a multiple of five.

    Age reporting error is one of the six regime axes and it is declared to interact with
    the age slope of mortality: an age moved to the nearest multiple of five crosses a
    band boundary at 65 and at 85, and the rate it carries across is the slope times the
    distance moved. The excess mass at multiples of five, over a five-point moving
    average of the neighbouring ages, measures the displacement without any truth file.
    """
    ages = np.asarray(ages, dtype=np.int64)
    ages = ages[(ages >= 20) & (ages <= 95)]
    if len(ages) < 200:
        return {"excess": 0.0, "fitted": False}
    counts = np.bincount(ages, minlength=MAX_AGE + 1).astype(np.float64)
    grid = np.arange(MAX_AGE + 1)
    inside = (grid >= 22) & (grid <= 93)
    smooth = np.zeros_like(counts)
    for a in grid[inside]:
        window = counts[a - 2:a + 3]
        smooth[a] = window.sum() / len(window)
    on_five = inside & (grid % 5 == 0)
    total = counts[inside].sum()
    if total <= 0 or smooth[on_five].sum() <= 0:
        return {"excess": 0.0, "fitted": False}
    excess = float((counts[on_five] - smooth[on_five]).sum() / total)
    return {"excess": float(np.clip(excess, 0.0, 0.6)), "fitted": True}


def deheap_age_cube(cube: np.ndarray, excess: float) -> np.ndarray:
    """Move the excess mass at multiples of five back over the ages it came from.

    The heaped share is returned to the four neighbouring ages in proportion to their own
    smoothed mass, which leaves the total and the sex and county margins untouched and
    only repairs the single-year shape. Bands whose boundary is a multiple of five are
    the ones that gain or lose by it, which is where the obligation is priced.
    """
    cube = np.asarray(cube, dtype=np.float64)
    if excess <= 0.0 or cube.shape[-2] < 30:
        return cube
    out = cube.copy()
    n_ages = cube.shape[-2]
    ages = np.arange(n_ages)
    for a in ages[(ages % 5 == 0) & (ages >= 25) & (ages <= n_ages - 3)]:
        neighbours = [a - 2, a - 1, a + 1, a + 2]
        neighbours = [n for n in neighbours if 0 <= n < n_ages]
        take = out[..., a, :] * excess
        weight = np.stack([out[..., n, :] for n in neighbours], axis=0)
        total = weight.sum(axis=0)
        share = np.where(total > 0, weight / np.maximum(total, 1e-12),
                         1.0 / len(neighbours))
        out[..., a, :] -= take
        for i, n in enumerate(neighbours):
            out[..., n, :] += take * share[i]
    return np.maximum(out, 0.0)


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
                for estimand, count_cube in (
                    (MORTALITY_ESTIMAND, d_cube),
                    (INCIDENCE_ESTIMAND, n_cube),
                ):
                    for b, band in enumerate(ACTUARIAL_AGE_BAND_LABELS):
                        e = float(e_point[u, x, b])
                        if e <= 0.0:
                            fallback = float(national_rate[estimand][x, b])
                            emit(
                                estimand,
                                level,
                                u,
                                x,
                                band,
                                fallback,
                                0.0,
                                max(2.0 * fallback, 1e-6),
                            )
                            continue
                        point = float(count_cube[:, u, x, b].mean()) / e
                        with np.errstate(invalid="ignore", divide="ignore"):
                            per_path = count_cube[:, u, x, b] / np.maximum(
                                e_cube[:, u, x, b], 1e-9
                            )
                        lo, hi = _percentile_pair(per_path, point)
                        emit(estimand, level, u, x, band, point, lo, hi)
    return rows


def state_rates_from_experience(
    experience: dict,
    ac: ActuarialContract,
    years_ahead: float,
    kind: str,
    shock_family: dict | None = None,
    improvement_override: dict | None = None,
) -> dict:
    """State by band by sex rate for the release window, from the experience file.

    All five years contribute. Each year's exposure is put on the last year's level by
    the estimated drift, so the pooled ratio of total events to drift-adjusted total
    exposure is the last year's rate with five years of information behind it; using the
    last year alone would throw away four fifths of the file and hand the thin cells to
    Poisson noise. Partial pooling across states then follows, and ``years_ahead``
    carries the level to a release window that sits after the file's last year.
    """
    counts = (
        experience["deaths"] if kind == "mortality" else experience["qualifying_events"]
    )
    exposure = experience["exposure"]
    target = "mortality" if kind == "mortality" else "incidence"
    detected = estimate_improvement(
        exposure,
        counts,
        shock_family=shock_family,
        shock_range=shock_range_for(shock_family, target),
    )
    if improvement_override is None:
        improvement = detected
    else:
        improvement = {
            "drift": float(improvement_override["mortality_drift"]),
            "drift_se": float(improvement_override["mortality_drift_se"]),
            # The third line keeps its independent trend estimator.  It still removes
            # an identifiable published-family shock from the starting level before
            # the continuation redraws future shocks; otherwise that line would charge
            # the same observed event twice.
            "shock_posterior": np.asarray(detected["shock_posterior"]),
            "fitted": bool(improvement_override.get("fitted", True)),
            "strategy": str(improvement_override.get("strategy", "external")),
        }
    n_years = exposure.shape[0]
    offset = np.arange(n_years) - (n_years - 1)
    adjusted = exposure * np.exp(improvement["drift"] * offset)[:, None, None, None]
    # A year the published family says was a shock year is not the level the projection
    # starts from: the continuation draws its own shocks on top of that level, so leaving
    # this one in it would charge the same epidemic twice. The year's counts are divided
    # by the expected multiplier it carries, weighted by how sure the file is.
    posterior = np.asarray(improvement.get("shock_posterior", np.zeros(n_years)),
                           dtype=np.float64)
    shock_range = shock_range_for(shock_family, target)
    removed = 0.0
    if shock_range is not None and posterior.shape == (n_years,) and posterior.any():
        expected = float(np.mean([np.log(shock_range[0]), np.log(shock_range[1])]))
        factor = np.exp(posterior * expected)
        counts = counts / factor[:, None, None, None]
        removed = float(np.mean(factor - 1.0))
    # A shock the detector could not resolve is still in the average of five years. The
    # published family says how much loading those years carry in expectation, so what
    # the detector did not take out is removed here, and neither step can remove it
    # twice.
    residual = max(expected_shock_loading(shock_family, target) - removed, 0.0)
    if residual > 0:
        counts = counts / (1.0 + residual)
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
                      state_rate: np.ndarray, count_variance: np.ndarray | None = None) -> dict:
    """County by band by sex rates: the state level times a partially pooled county
    deviation.

    The deviation is fitted on the county's own counts against its state's expected
    counts. ``count_variance`` is the variance of that county's count as an estimate of
    its deaths or events, which is not the count itself when the count is what is left
    after a correction. Given one, the deviation is shrunk by how much between-county
    spread survives that error, so a source whose correction dominates it contributes a
    vector of ones rather than the correction's noise.
    """
    counts = np.asarray(counts, dtype=np.float64)
    exposure = np.asarray(exposure, dtype=np.float64)
    expected = state_rate[county_state] * exposure
    total_expected = expected.sum(axis=(1, 2))
    total_counts = counts.sum(axis=(1, 2))
    with np.errstate(invalid="ignore", divide="ignore"):
        raw = np.where(total_expected > 0, total_counts / np.maximum(total_expected, 1e-9), 1.0)
    raw = np.nan_to_num(raw, nan=1.0)
    if count_variance is None:
        ratio_fit = buhlmann_straub(total_counts, total_expected)
        z = np.asarray(ratio_fit["z"], dtype=np.float64)
        centre = ratio_fit["overall"]
        deviation = z * raw + (1.0 - z) * centre
        k = ratio_fit["k"]
    else:
        variance = np.asarray(count_variance, dtype=np.float64) / \
            np.maximum(total_expected ** 2, 1e-12)
        fit = shrink_deviation(raw, variance, total_expected)
        deviation = fit["deviation"]
        centre = fit["centre"]
        z = fit["z"]
        k = float(fit["tau2"])
    deviation = np.clip(deviation / max(float(np.mean(centre)), 1e-9), 0.4, 2.5)
    return {"rate": state_rate[county_state] * deviation[:, None, None],
            "deviation": deviation, "k": k, "z": z}


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
    # Used only when the contract publishes no shock family. When it does, the rate and
    # the multipliers of every kind are read from it rather than assumed here.
    shock_probability: float = 0.10
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


def draw_shock_year(rng: np.random.Generator, n: int, family: dict | None,
                    fallback_probability: float,
                    fallback_range: tuple[float, float]) -> dict:
    """One year of the published shock family, drawn independently for each path.

    A shock arrives at the published annual rate, one kind is drawn, and every
    multiplier of that kind moves on a single draw, which is what the contract states:
    an epidemic year raises deaths and admissions together, and a migration year moves
    only the flows. Drawing the multipliers independently would break exactly the
    dependence that decides how heavy the regional tail is.
    """
    out = {name: np.ones(n) for name in ("mortality", "incidence", "migration", "fertility")}
    if not family:
        hit = rng.random(n) < fallback_probability
        if hit.any():
            out["mortality"][hit] = rng.uniform(*fallback_range, size=int(hit.sum()))
        return out
    kinds = family["kinds"]
    hit = rng.random(n) < float(family["annual_rate"])
    if not hit.any():
        return out
    which = rng.integers(0, len(kinds), size=int(hit.sum()))
    u = rng.random(int(hit.sum()))
    index = np.flatnonzero(hit)
    for k, kind in enumerate(kinds):
        rows = index[which == k]
        if len(rows) == 0:
            continue
        share = u[which == k]
        for target in ("mortality", "incidence", "migration", "fertility"):
            if target in kind:
                lo, hi = kind[target]
                out[target][rows] = lo + share * (hi - lo)
    return out


def sample_regional_loading_paths(
    rng: np.random.Generator,
    n_paths: int,
    n_regions: int,
    shock_family: dict | None,
    evidence: dict | None = None,
) -> np.ndarray:
    """One participant-identifiable regional-loading vector per predictive path.

    A clean history cannot identify the realized world vector, so every path draws a
    fresh vector from the uniform public band.  An identified history instead draws
    from the magnitude/loading posterior fitted above.  There is deliberately no mode
    that accepts a retained or caller-supplied realized vector.
    """
    if int(n_paths) < 1 or int(n_regions) < 1:
        raise ValueError("regional-loading draws need positive path and region counts")
    band = None if not shock_family else shock_family.get("regional_loading_band")
    if band is None:
        return np.ones((int(n_paths), int(n_regions)))
    low, high = (float(band[0]), float(band[1]))
    mode = "public_band_marginalization" if evidence is None else evidence.get("mode")
    if mode == "national_only":
        return np.ones((int(n_paths), int(n_regions)))
    if mode == "public_band_marginalization":
        return rng.uniform(low, high, size=(int(n_paths), int(n_regions)))
    if mode != "participant_experience_posterior":
        raise ValueError(f"unknown regional-loading evidence mode {mode!r}")
    probability = np.asarray(evidence.get("grid_probability"), dtype=np.float64)
    loading = np.asarray(evidence.get("loading_grid"), dtype=np.float64)
    loading_se = np.asarray(evidence.get("loading_se_grid"), dtype=np.float64)
    if probability.ndim != 1 or len(probability) == 0:
        raise ValueError("participant regional-loading posterior is malformed")
    expected_shape = (len(probability), int(n_regions))
    if (
        loading.shape != expected_shape
        or loading_se.shape != expected_shape
        or not np.isfinite(probability).all()
        or not np.isfinite(loading).all()
        or not np.isfinite(loading_se).all()
        or float(probability.sum()) <= 0.0
    ):
        raise ValueError("participant regional-loading posterior is malformed")
    probability = probability / probability.sum()
    grid_index = rng.choice(len(probability), size=int(n_paths), p=probability)
    draws = rng.normal(loading[grid_index], np.maximum(loading_se[grid_index], 1e-9))
    return np.clip(draws, low, high)


def regionalize_shock_multiplier(
    national_multiplier: np.ndarray, loading: np.ndarray
) -> np.ndarray:
    """Apply the public ``1 + L_r * (m - 1)`` rule to path-region pairs."""
    multiplier = np.asarray(national_multiplier, dtype=np.float64)
    loading = np.asarray(loading, dtype=np.float64)
    if multiplier.ndim != 1 or loading.ndim != 2 or loading.shape[0] != len(multiplier):
        raise ValueError("shock multiplier must align with path by region loadings")
    return 1.0 + loading * (multiplier[:, None] - 1.0)


def regional_loading_diagnostics(
    evidence: dict, draws: np.ndarray, held_years: int
) -> dict:
    """Serializable receipt for the loading evidence and predictive draws."""
    draws = np.ascontiguousarray(np.asarray(draws, dtype=np.float64))
    if draws.ndim != 2 or len(draws) == 0:
        raise ValueError("regional-loading diagnostics need path by region draws")
    posterior = np.asarray(evidence.get("shock_year_posterior", []), dtype=np.float64)
    mortality = np.asarray(
        evidence.get("mortality_shock_posterior", []), dtype=np.float64
    )
    incidence = np.asarray(
        evidence.get("incidence_shock_posterior", []), dtype=np.float64
    )
    return {
        "mode": evidence["mode"],
        "evidence_source": evidence["evidence_source"],
        "experience_digest": evidence["experience_digest"],
        "input_fields": list(evidence["input_fields"]),
        "uses_retained_realized_loadings": bool(
            evidence["uses_retained_realized_loadings"]
        ),
        "band": None if evidence["band"] is None else list(evidence["band"]),
        "formula": evidence["formula"],
        "target_shock_annual_probability": float(
            evidence["target_shock_annual_probability"]
        ),
        "identification_threshold": float(evidence["identification_threshold"]),
        "shock_year_posterior": posterior.tolist(),
        "mortality_shock_posterior": mortality.tolist(),
        "incidence_shock_posterior": incidence.tolist(),
        "identified_year_index": evidence.get("identified_year_index"),
        "identified_year": evidence.get("identified_year"),
        "loading_mean": np.asarray(evidence["loading_mean"], dtype=np.float64).tolist(),
        "loading_sd": np.asarray(evidence["loading_sd"], dtype=np.float64).tolist(),
        "predictive_draw_min": draws.min(axis=0).tolist(),
        "predictive_draw_max": draws.max(axis=0).tolist(),
        "predictive_draw_mean": draws.mean(axis=0).tolist(),
        "predictive_draw_digest": hashlib.sha256(draws.tobytes()).hexdigest(),
        "predictive_paths": int(draws.shape[0]),
        "regions": int(draws.shape[1]),
        "one_vector_per_outer_path": True,
        "held_across_horizon": True,
        "held_years": int(held_years),
        "distinct_loading_vectors": int(np.unique(draws, axis=0).shape[0]),
    }


def simulate_liabilities(age_sex_paths: np.ndarray, rates: dict, ac: ActuarialContract,
                         params: SimulationParams = SimulationParams(),
                         regional_loading_evidence: dict | None = None) -> dict:
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
    region_of_county = np.asarray(ac.region_of_county, dtype=np.int64)
    # Loading uncertainty is outside the year process: one vector belongs to one whole
    # predictive path.  Its separate stream makes the draw reproducible and invariant
    # to ``path_chunk`` while the vector is held unchanged through every horizon year.
    loading_rng = np.random.default_rng(
        np.random.SeedSequence([int(params.seed), 0x10AD])
    )
    regional_loading = sample_regional_loading_paths(
        loading_rng,
        n_paths,
        ac.n_regions,
        ac.shock_family,
        regional_loading_evidence,
    )

    def level_draw(rng, chunk, national_sd, regional_sd):
        """A common level error and one per region, both on the log scale.

        The rates are estimated, and the error in them is not independent across
        counties: a national estimate is one number for every county, and a region's own
        counts move its whole regional estimate together. A tail read off paths that
        differ only by demographic noise around a rate believed exactly would be far too
        thin, which is the shortcut that leaves an exceedance rate nowhere near nominal.
        """
        national_sd = max(float(national_sd), 0.0)
        regional = np.asarray(regional_sd, dtype=np.float64)
        if regional.ndim == 0:
            regional = np.full(ac.n_regions, float(regional))
        national = rng.normal(0.0, national_sd, size=(chunk, 1))
        draws = rng.normal(0.0, 1.0, size=(chunk, ac.n_regions)) * regional[None, :]
        # A rate estimate is unbiased on its own scale, not on the log scale, so the
        # level multiplier has to average one. Without the half-variance term a wider
        # uncertainty would raise the expected liability as well as its spread, and the
        # honest widening of the tail would arrive as a quiet loading on the mean.
        centre = 0.5 * (national_sd ** 2 + regional[region_of_county] ** 2)
        return np.exp(national + draws[:, region_of_county] - centre[None, :])

    for start in range(0, n_paths, params.path_chunk):
        stop = min(start + params.path_chunk, n_paths)
        chunk = stop - start
        rng = np.random.default_rng([params.seed, start])
        state = paths[start:stop].copy()
        pending = state * not_yet[None]
        path_loading = regional_loading[start:stop]
        if params.parameter_noise:
            level_q = level_draw(rng, chunk, rates.get("mortality_log_sd", 0.05),
                                 rates.get("mortality_log_sd_region", 0.0))
            level_l = level_draw(rng, chunk, rates.get("incidence_log_sd", 0.08),
                                 rates.get("incidence_log_sd_region", 0.0))
            drift_q = rng.normal(rates.get("mortality_drift", 0.0),
                                 rates.get("mortality_drift_se", 0.01), size=chunk)
            drift_l = rng.normal(rates.get("incidence_drift", 0.0),
                                 rates.get("incidence_drift_se", 0.01), size=chunk)
            mig_noise = rng.normal(0.0, 1.0, size=(chunk, 1, 1, 1)) * \
                np.asarray(rates.get("migration_se", 0.0), dtype=np.float64)[None]
        else:
            level_q = np.ones((chunk, n_counties))
            level_l = np.ones((chunk, n_counties))
            drift_q = np.full(chunk, rates.get("mortality_drift", 0.0))
            drift_l = np.full(chunk, rates.get("incidence_drift", 0.0))
            mig_noise = np.zeros((chunk, 1, 1, 1))
        migration = mig_base[None] + mig_noise
        for year in range(n_years):
            if params.process_noise:
                shock = draw_shock_year(rng, chunk, ac.shock_family,
                                        params.shock_probability, params.shock_range)
            else:
                shock = {name: np.ones(chunk) for name in
                         ("mortality", "incidence", "migration", "fertility")}
            mortality_shock = regionalize_shock_multiplier(
                shock["mortality"], path_loading
            )[:, region_of_county]
            incidence_shock = regionalize_shock_multiplier(
                shock["incidence"], path_loading
            )[:, region_of_county]
            elapsed = year + 0.5
            q = np.clip(q_base[None] * np.exp(drift_q * elapsed)[:, None, None, None] *
                        (level_q * mortality_shock)[:, :, None, None],
                        0.0, 0.98)
            lam = np.clip(lam_base[None] * np.exp(drift_l * elapsed)[:, None, None, None] *
                          (level_l * incidence_shock)[:, :, None, None],
                          0.0, 1.0)
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
            flow = survivors * migration * shock["migration"][:, None, None, None]
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
            births = women * rates.get("fertility", 0.0) * shock["fertility"][:, None]
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
            "deaths": death_acc, "events": event_acc,
            "regional_loading": regional_loading,
            "regional_loading_draws_per_path": 1,
            "regional_loading_held_years": n_years}


def tail_summary(liability: np.ndarray, alpha: float = 0.95) -> dict:
    """Empirical mean, order-statistic quantile, and tied-tail expected shortfall.

    The quantile is observation ``ceil(alpha * M)`` in one-based indexing. Expected
    shortfall is the mean of every member at or above that value, including ties.
    """
    liability = np.asarray(liability, dtype=np.float64)
    if liability.ndim != 2 or liability.shape[0] == 0:
        raise ValueError("liability must be a nonempty (members, regions) array")
    if not np.isfinite(alpha) or not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be finite and between zero and one")
    mean = liability.mean(axis=0)
    q, es = empirical_tail(liability, alpha)
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


def allocate_reserve(liability: np.ndarray, total: float,
                     weights: np.ndarray | None = None) -> dict:
    """Allocate the fixed reserve to minimise the weighted expected shortfall.

    The feasible set is exactly the public one: every allocation is finite and
    nonnegative and the regional values sum to ``total``. Submitted q95 and ES95 are
    forecasts scored by the tail gate, not capital floors. On an empirical distribution,
    expected shortfall is piecewise linear: each interval between ordered liabilities has
    slope ``w_r P(L_r > A_r)``. The shared verifier routine fills those intervals from
    highest to lowest marginal value, which is the exact optimum against this method's
    own simulated paths.
    """
    liability = np.asarray(liability, dtype=np.float64)
    if liability.ndim != 2 or min(liability.shape, default=0) <= 0:
        raise ValueError("liability must be a nonempty (members, regions) array")
    if not np.isfinite(liability).all() or (liability < 0.0).any():
        raise ValueError("liability must be finite and nonnegative")
    n_regions = liability.shape[1]
    total = float(total)
    if not np.isfinite(total) or total < 0.0:
        raise ValueError("reserve total must be finite and nonnegative")
    weights = (
        np.ones(n_regions, dtype=np.float64)
        if weights is None
        else np.asarray(weights, dtype=np.float64)
    )
    if weights.shape != (n_regions,):
        raise ValueError("weights must contain one value per region")
    if not np.isfinite(weights).all() or (weights < 0.0).any():
        raise ValueError("weights must be finite and nonnegative")
    allocation = np.asarray(
        perfect_information_allocation(liability, total, weights), dtype=np.float64
    )
    tolerance = 1e-10 * max(total, 1.0)
    if (
        allocation.shape != (n_regions,)
        or not np.isfinite(allocation).all()
        or (allocation < -tolerance).any()
        or abs(float(allocation.sum()) - total) > tolerance
    ):
        raise RuntimeError("reserve optimizer returned an allocation outside the public feasible set")
    allocation = np.maximum(allocation, 0.0)
    return {
        "allocation": allocation,
        "feasible": True,
        "nu": float("nan"),
        "reason": (
            "weighted expected-shortfall optimum over finite nonnegative allocations "
            "summing to the public reserve total; q95 and es95 are forecasts only"
        ),
    }


def proportional_reserve(share: np.ndarray, total: float) -> np.ndarray:
    """Return the verifier's frozen public ``baseline_share`` allocation exactly."""
    share = np.asarray(share, dtype=np.float64)
    total = float(total)
    if share.ndim != 1 or len(share) == 0:
        raise ValueError("baseline share must be a nonempty one-dimensional array")
    if not np.isfinite(total) or total < 0.0:
        raise ValueError("reserve total must be finite and nonnegative")
    return proportional_baseline_allocation(share, total)


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


ADDITIVE_RELEASE_ESTIMANDS = (
    "persons", "households", "children_under_16", "elders_65_plus")


def reconcile_additive_release_rows(rows: list[dict], county_state: np.ndarray) -> list[dict]:
    """Make additive point estimates and interval endpoints agree across geography."""
    out = [dict(row) for row in rows]
    indexed = {(str(row["estimand"]), str(row["level"]), int(row["unit"])): row
               for row in out}
    county_state = np.asarray(county_state, dtype=np.int64)
    n_states = int(county_state.max()) + 1
    for estimand in ADDITIVE_RELEASE_ESTIMANDS:
        if any((estimand, "county", county) not in indexed
               for county in range(len(county_state))):
            continue
        for field in ("estimate", "lower", "upper"):
            for state in range(n_states):
                members = np.flatnonzero(county_state == state)
                indexed[(estimand, "state", state)][field] = float(sum(
                    indexed[(estimand, "county", int(county))][field]
                    for county in members))
            indexed[(estimand, "nation", 0)][field] = float(sum(
                indexed[(estimand, "state", state)][field] for state in range(n_states)))
    return out


def write_actuarial_submission(out_dir: Path, release_rows, projection_rows, cube,
                               suppress_below: float, reserve, band_labels,
                               sex_labels) -> None:
    """Write the exact three-file version-four submission."""
    import pandas as pd
    del cube, suppress_below, band_labels, sex_labels
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(release_rows)[list(V4_RELEASE_COLUMNS)].to_csv(
        out_dir / "release.csv", index=False)
    pd.DataFrame(projection_rows)[list(V4_PROJECTION_COLUMNS)].to_csv(
        out_dir / "projection.csv", index=False)
    pd.DataFrame(reserve)[list(RESERVE_COLUMNS)].to_csv(out_dir / "reserve.csv", index=False)


# ------------------------------------------------------- band to single-year age fill

def expand_band_rates(band_rate: np.ndarray, band_exposure: np.ndarray,
                      max_age: int = MAX_AGE, slope: float | None = None) -> np.ndarray:
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
        # The age gradient of the world's own experience file, when one was fitted:
        # five years of counts identify it better than six band rates read off one
        # snapshot, and it is a per-world draw rather than a constant of the family.
        if slope is None or not np.isfinite(slope) or slope <= 0:
            slope = float((w * (x - mx) * (y - my)).sum() /
                          max((w * (x - mx) ** 2).sum(), 1e-12))
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


def _clerical_bootstrap_survival(
    pre, rev, rng: np.random.Generator, n_imputations: int
) -> list[np.ndarray]:
    """Deterministic exact links plus a comparison-field bootstrap for the rest.

    Unique exact keys are clerical matches. Remaining records are blocked on sex and
    birth year, require at least one name agreement, and are reduced one to one after a
    weighted clerical score. Each imputation resamples the four comparison fields before
    that reduction. The spread therefore measures whether the linkage conclusion depends
    on a particular name, birth-month, or county comparison, without fitting the mixture
    model used by the main reference line.
    """
    key = list(LINK_FIELDS)
    left = pre[key + ["county"]].copy()
    right = rev[key + ["county"]].copy()
    left["_left_row"] = np.arange(len(left), dtype=np.int64)
    right["_right_row"] = np.arange(len(right), dtype=np.int64)

    left["_left_n"] = left.groupby(key, dropna=False)["_left_row"].transform("size")
    right["_right_n"] = right.groupby(key, dropna=False)["_right_row"].transform("size")
    left_unique = left
    right_unique = right
    left_unique = left_unique[
        (left_unique["_left_n"] == 1)
        & (left_unique["given_code"] > 0)
        & (left_unique["family_code"] > 0)
    ]
    right_unique = right_unique[
        (right_unique["_right_n"] == 1)
        & (right_unique["given_code"] > 0)
        & (right_unique["family_code"] > 0)
    ]
    exact = left_unique[key + ["_left_row"]].merge(
        right_unique[key + ["_right_row"]], on=key
    )
    fixed_left = set(exact["_left_row"].to_numpy(dtype=np.int64))
    fixed_right = set(exact["_right_row"].to_numpy(dtype=np.int64))

    a = left[~left["_left_row"].isin(fixed_left)].copy()
    b = right[~right["_right_row"].isin(fixed_right)].copy()
    a["_id"], b["_id"] = a["_left_row"], b["_right_row"]
    a["_county"] = a["county"]
    b["_county"] = b["county"]
    a["_block"] = (a["birth_tick"].to_numpy(dtype=np.int64) // 12) * 2 + a[
        "sex"
    ].to_numpy(dtype=np.int64)
    b["_block"] = (b["birth_tick"].to_numpy(dtype=np.int64) // 12) * 2 + b[
        "sex"
    ].to_numpy(dtype=np.int64)
    a = a[(a["given_code"] > 0) & (a["family_code"] > 0)]
    b = b[(b["given_code"] > 0) & (b["family_code"] > 0)]
    pairs = _candidate_pairs(a, b, "_id", "_id") if len(a) and len(b) else None

    fixed = (
        np.fromiter(fixed_left, dtype=np.int64, count=len(fixed_left))
        if fixed_left
        else np.zeros(0, dtype=np.int64)
    )
    if pairs is None or len(pairs) == 0:
        survived = np.zeros(len(pre), dtype=bool)
        survived[fixed] = True
        return [survived]

    agreement = _agreement(pairs)
    weights = np.asarray([2.0, 2.5, 2.0, 0.5], dtype=np.float64)
    draws = []
    for _ in range(max(int(n_imputations), 1)):
        picked = rng.integers(0, agreement.shape[1], size=agreement.shape[1])
        sampled_weight = weights[picked]
        score = (agreement[:, picked] * sampled_weight[None]).sum(axis=1)
        score *= weights.sum() / max(float(sampled_weight.sum()), 1e-9)
        eligible = score >= 4.0
        survived = np.zeros(len(pre), dtype=bool)
        survived[fixed] = True
        if eligible.any():
            candidate = pairs.loc[eligible].reset_index(drop=True)
            candidate_score = score[eligible]
            keep = _one_to_one(candidate, candidate_score)
            survived[candidate.iloc[keep]["_id_l"].to_numpy(dtype=np.int64)] = True
        draws.append(survived)
    return draws


def vintage_death_counts(
    preliminary,
    revised,
    tick_pre: int,
    tick_rev: int,
    county_state: np.ndarray,
    rng: np.random.Generator,
    n_imputations: int = 6,
    deterministic: bool = False,
    linkage_strategy: str = "fellegi_sunter",
) -> dict:
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
    elif linkage_strategy == "clerical_bootstrap":
        draws = _clerical_bootstrap_survival(pre, rev, rng, n_imputations)
    elif linkage_strategy != "fellegi_sunter":
        raise ValueError(f"unknown linkage strategy {linkage_strategy!r}")
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


def _pooled_by_state(rate: np.ndarray, weight: np.ndarray) -> np.ndarray:
    """One rate per state, weighted by the exposure behind each band and sex."""
    rate = np.asarray(rate, dtype=np.float64)
    weight = np.maximum(np.asarray(weight, dtype=np.float64), 0.0)
    total = weight.sum(axis=(1, 2))
    return np.where(total > 0, (rate * weight).sum(axis=(1, 2)) / np.maximum(total, 1e-9), 0.0)


def level_disagreement(first: np.ndarray, second: np.ndarray) -> dict:
    """How far apart two independent estimates of the same level sit, by region.

    The experience file and the register vintages measure the same mortality, and the
    experience file and the anchored archive measure the same incidence. Neither pair
    shares a source, so the log ratio between them is a measurement of the error in both:
    if the two errors are independent and comparable in size, each carries about the
    spread of the difference over root two. That is where the level uncertainty the
    continuation draws from comes from, instead of a constant chosen to look reasonable.
    """
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    usable = (first > 0) & (second > 0)
    if usable.sum() < 2:
        return {"national": 0.0, "regional": 0.0, "n": int(usable.sum())}
    difference = np.log(second[usable]) - np.log(first[usable])
    return {"national": float(abs(difference.mean()) / np.sqrt(2.0)),
            "regional": float(np.std(difference, ddof=1) / np.sqrt(2.0)),
            "n": int(usable.sum())}


def blend_levels(first: np.ndarray, first_variance: np.ndarray,
                 second: np.ndarray, second_variance: np.ndarray) -> dict:
    """Combine two independent estimates of the same rate by their own precisions.

    The experience file measures a level over five years ending a year before the
    snapshot. The register vintages and the health archive measure the same level in the
    months around it. Neither dominates by construction: the file has the counts, the
    current sources are closer to the window being priced, and each carries a correction
    the other does not.

    What decides the weight is measured, not assumed. Each side supplies the variance of
    its own log rate, so a current source whose correction is most of what it measures
    weighs almost nothing, and the same code puts real weight on one whose correction is
    small. Measured on six worlds, the median absolute log error of a state mortality
    level is 0.24 from the experience file and 1.6 to 5.0 from the register vintages,
    because the register loses records to identifier churn at many times the death rate;
    the variance the vintage side reports carries exactly that, and its weight collapses
    without a constant anywhere saying so.
    """
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    v1 = np.maximum(np.asarray(first_variance, dtype=np.float64), 1e-12)
    v2 = np.maximum(np.asarray(second_variance, dtype=np.float64), 1e-12)
    usable = (first > 0) & (second > 0) & np.isfinite(v1) & np.isfinite(v2)
    weight = np.where(usable, (1.0 / v2) / (1.0 / v1 + 1.0 / v2), 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        combined = np.exp((1.0 - weight) * np.log(np.maximum(first, 1e-12)) +
                          weight * np.log(np.maximum(second, 1e-12)))
    blended = np.where(usable, combined, np.where(first > 0, first, second))
    return {"rate": blended, "weight": weight}


def shrink_deviation(raw: np.ndarray, variance: np.ndarray,
                     weight: np.ndarray | None = None) -> dict:
    """Pull a county's own multiplier toward one by how noisy that multiplier is.

    A county deviation read off a source whose correction is most of its signal is not a
    county deviation, it is the correction's noise. The between-county spread that
    survives the measurement error is estimated by moments, and each county keeps the
    share of its own reading that spread supports. Where the source is noisy the whole
    vector collapses to one, which is the honest statement that the county detail is not
    in the data, and where it is clean nothing is shrunk.
    """
    raw = np.asarray(raw, dtype=np.float64)
    variance = np.maximum(np.asarray(variance, dtype=np.float64), 0.0)
    weight = np.ones_like(raw) if weight is None else np.maximum(
        np.asarray(weight, dtype=np.float64), 0.0)
    if weight.sum() <= 0:
        return {"deviation": np.ones_like(raw), "tau2": 0.0, "z": np.zeros_like(raw),
                "centre": 1.0}
    centre = float((weight * raw).sum() / weight.sum())
    spread = float((weight * (raw - centre) ** 2).sum() / weight.sum())
    mean_variance = float((weight * variance).sum() / weight.sum())
    tau2 = max(spread - mean_variance, 0.0)
    z = tau2 / np.maximum(tau2 + variance, 1e-12)
    return {"deviation": z * raw + (1.0 - z) * centre, "tau2": tau2, "z": z,
            "centre": centre}


def anchor_log_variance(anchor: dict, sensitivity: float, specificity: float) -> np.ndarray:
    """Variance of the log anchored prevalence, cell by cell.

    The item is imperfect, so the corrected prevalence is (p + sp - 1) / (se + sp - 1) and
    its sampling variance is the observed one divided by the square of that denominator.
    At a Youden index of three quarters, an anchor read on forty effective respondents
    carries a relative error of about a sixth, which is what decides how much weight the
    archive side of the incidence level is allowed to carry.
    """
    observed = np.asarray(anchor.get("observed", anchor["prevalence"]), dtype=np.float64)
    corrected = np.asarray(anchor["prevalence"], dtype=np.float64)
    effective = np.maximum(np.asarray(anchor["effective_n"], dtype=np.float64), 1.0)
    youden = max(abs(sensitivity + specificity - 1.0), 1e-6)
    with np.errstate(invalid="ignore", divide="ignore"):
        variance = observed * (1.0 - observed) / effective / youden ** 2
        relative = variance / np.maximum(corrected ** 2, 1e-12)
    return np.where(np.isfinite(relative), np.clip(relative, 0.0, 4.0), 4.0)


def rate_uncertainty(experience: dict, vintage_mortality: np.ndarray,
                     archive_incidence: np.ndarray, state_rates: dict,
                     linkage_relative: float = 0.0) -> dict:
    """The width of the level errors the continuation propagates, per region.

    Three sources, all measured: the counts behind each level, the disagreement between
    the two independent estimates of that level, and the spread across the imputations of
    the link set that produced the death counts. A continuation that draws only process
    noise around a level believed exactly reports a tail that is the demographic noise of
    a large stock, which is far thinner than the distribution being scored.
    """
    exposure = experience["exposure"].sum(axis=0)
    deaths = experience["deaths"].sum(axis=0)
    events = experience["qualifying_events"].sum(axis=0)
    n_states = exposure.shape[0]
    sampling_m = 1.0 / np.sqrt(np.maximum(deaths.sum(axis=(1, 2)), 1.0))
    sampling_i = 1.0 / np.sqrt(np.maximum(events.sum(axis=(1, 2)), 1.0))
    mortality = level_disagreement(_pooled_by_state(state_rates["mortality"], exposure),
                                   _pooled_by_state(vintage_mortality, exposure))
    incidence = level_disagreement(_pooled_by_state(state_rates["incidence"], exposure),
                                   _pooled_by_state(archive_incidence, exposure))
    national_m = float(np.sqrt(mortality["national"] ** 2 +
                               1.0 / max(deaths.sum(), 1.0) + linkage_relative ** 2))
    national_i = float(np.sqrt(incidence["national"] ** 2 + 1.0 / max(events.sum(), 1.0)))
    region_m = np.sqrt(sampling_m ** 2 + mortality["regional"] ** 2)
    region_i = np.sqrt(sampling_i ** 2 + incidence["regional"] ** 2)
    return {"mortality_national": min(national_m, 0.5),
            "incidence_national": min(national_i, 0.5),
            "mortality_region": np.minimum(region_m, 0.5),
            "incidence_region": np.minimum(region_i, 0.5),
            "mortality_disagreement": mortality, "incidence_disagreement": incidence,
            "n_states": n_states}


def estimate_age_error(preliminary, revised, tick: int) -> dict:
    """How often a reported birth date moves between the two register vintages, by band.

    Records whose given and family codes are unique in both files and agree across them
    are the same person with near certainty, so the disagreement of their birth dates
    measures the reported-age error directly. The declared interaction between age
    reporting error and the age slope of mortality means the rate rises with age, and a
    record whose birth year moved is a record the blocked linkage cannot recover: it
    leaves the file as a disappearance and is counted as a death unless the estimate of
    the death count says otherwise. The profile returned here is that age pattern, which
    the churn model uses as one of its two shapes.
    """
    n_bands = len(ACTUARIAL_AGE_BANDS)
    empty = {"rate": np.zeros(n_bands), "share": 0.0, "sd_years": 0.0, "fitted": False}
    keys = ["given_code", "family_code", "sex"]
    if any(k not in preliminary.columns or k not in revised.columns for k in keys):
        return empty
    left = preliminary.drop_duplicates(subset=["person_id"])
    right = revised.drop_duplicates(subset=["person_id"])
    left = left[(left["given_code"] > 0) & (left["family_code"] > 0)]
    right = right[(right["given_code"] > 0) & (right["family_code"] > 0)]
    left = left[~left.duplicated(subset=keys, keep=False)]
    right = right[~right.duplicated(subset=keys, keep=False)]
    if len(left) < 200 or len(right) < 200:
        return empty
    pairs = left.merge(right, on=keys, suffixes=("_l", "_r"))
    if len(pairs) < 200:
        return empty
    shift = (pairs["birth_tick_r"].to_numpy(dtype=np.float64) -
             pairs["birth_tick_l"].to_numpy(dtype=np.float64)) / 12.0
    moved = np.abs(shift) > 1e-9
    band = band_of_age((tick - pairs["birth_tick_l"].to_numpy(dtype=np.int64)) // 12)
    rate = np.zeros(n_bands)
    for b in range(n_bands):
        cell = band == b
        if cell.sum() >= 50:
            rate[b] = float(moved[cell].mean())
    if not moved.any():
        return {"rate": rate, "share": 0.0, "sd_years": 0.0, "fitted": True}
    return {"rate": rate, "share": float(moved.mean()),
            "sd_years": float(np.std(shift[moved])), "fitted": True,
            "n_pairs": int(len(pairs))}


def mobility_profile(experience: dict) -> np.ndarray:
    """The age and sex shape of movement, from the experience file's net migration.

    Net migration understates gross movement, but its age and sex shape is the shape of
    the movement that breaks an address and with it an identifier, and the level it
    carries is the world's own migration intensity. That is the first half of the
    declared interaction between migration and stale-address linkage; the county scale
    fitted against this shape is the second.
    """
    exposure = experience["exposure"].sum(axis=0)
    net = np.abs(experience["net_migration"]).sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        rate = np.where(exposure > 0, net / np.maximum(exposure, 1e-9), 0.0)
    profile = rate.sum(axis=0) / max(len(rate), 1)
    total = float(np.average(profile, weights=np.maximum(exposure.sum(axis=0), 1e-9)))
    if total <= 0:
        return np.ones_like(profile)
    return profile / total


def fit_churn(gone: np.ndarray, at_risk: np.ndarray, months: int,
              expected_rate: np.ndarray, mobility: np.ndarray, age_error: np.ndarray,
              county_state: np.ndarray) -> dict:
    """False disappearances between vintages, with the age shape the mechanisms imply.

    A record leaves the register between two vintages because the person died, because
    they moved and their identifier did not survive the move, or because their reported
    birth date changed and no blocked comparison can find them again. The last two are
    churn, and both have an age pattern: movement is concentrated in early adult life and
    reported-age error rises with age. Subtracting one flat rate read off the young bands
    therefore removes too much at the ages the obligation is priced on, which is where
    the reference was weakest.

    Two shapes are fitted rather than assumed: the mobility profile from the experience
    file and the age-error profile from the cross-vintage probe. Their national mix is
    fitted on the residual of the disappearance rate over the mortality the experience
    file already implies, and each county then carries one non-negative scale on the
    mixed shape, shrunk toward the national scale by its own numbers.
    """
    gone = np.asarray(gone, dtype=np.float64)
    at_risk = np.asarray(at_risk, dtype=np.float64)
    years = max(months, 1) / 12.0
    with np.errstate(invalid="ignore", divide="ignore"):
        rate = np.where(at_risk > 0, gone / np.maximum(at_risk, 1e-9), 0.0)
    expected = np.asarray(expected_rate, dtype=np.float64) * years
    residual = np.maximum(rate - expected, 0.0)
    shapes = np.stack([np.broadcast_to(mobility, residual.shape[1:]),
                       np.broadcast_to(age_error[:, None], residual.shape[1:])])
    national = (at_risk * residual).sum(axis=0) / np.maximum(at_risk.sum(axis=0), 1e-9)
    weight = at_risk.sum(axis=0)
    mix = np.zeros(2)
    design = np.stack([(shapes[i] * np.sqrt(weight)).ravel() for i in range(2)], axis=1)
    target = (national * np.sqrt(weight)).ravel()
    for _ in range(60):     # projected gradient, two non-negative coefficients
        gradient = design.T @ (design @ mix - target)
        step = 1.0 / max(float(np.linalg.norm(design.T @ design, 2)), 1e-9)
        mix = np.maximum(mix - step * gradient, 0.0)
    shape = mix[0] * shapes[0] + mix[1] * shapes[1]
    scale = float(np.average(shape, weights=np.maximum(weight, 1e-9)))
    if scale <= 0:
        flat = (at_risk * residual).sum(axis=(1, 2)) / np.maximum(at_risk.sum(axis=(1, 2)), 1e-9)
        return {"churn": flat[:, None, None] * np.ones_like(residual), "mix": mix,
                "kappa": flat, "fitted": False}
    shape = shape / scale
    numerator = (at_risk * residual * shape[None]).sum(axis=(1, 2))
    denominator = (at_risk * shape[None] ** 2).sum(axis=(1, 2))
    raw = np.where(denominator > 0, numerator / np.maximum(denominator, 1e-9), 0.0)
    national_kappa = float((at_risk * residual * shape[None]).sum() /
                           max((at_risk * shape[None] ** 2).sum(), 1e-9))
    credibility = buhlmann_straub(numerator, denominator)
    z = np.asarray(credibility["z"], dtype=np.float64)
    kappa = np.maximum(z * raw + (1.0 - z) * national_kappa, 0.0)
    churn = np.minimum(kappa[:, None, None] * shape[None], rate)
    # What the fit did not explain is the error the correction carries into the death
    # count, and it is large: the disappearance rate runs many times the death rate, so a
    # correction good to a few percent of itself is still comparable with the whole
    # signal. Reporting it is what lets the county deviation be shrunk by evidence.
    left = residual - kappa[:, None, None] * shape[None]
    residual_sd = np.sqrt(
        np.maximum(
            (at_risk * left**2).sum(axis=(1, 2))
            / np.maximum(at_risk.sum(axis=(1, 2)), 1e-9),
            0.0,
        )
    )
    return {
        "churn": churn,
        "mix": mix,
        "kappa": kappa,
        "shape": shape,
        "national_kappa": national_kappa,
        "fitted": True,
        "residual_sd": residual_sd,
    }


# ------------------------------------------------------------------- the whole layer


@dataclass(frozen=True)
class LayerParams:
    """One switch per targeted ablation of protocol section 11, all false for the
    reference. Each switch removes exactly one step and nothing else, so a control that
    passes its gate says the gate is loose rather than that the control is subtle."""

    simulation: SimulationParams = SimulationParams()
    deterministic_linkage: bool = False  # ablation 3, first half
    linkage_strategy: str = (
        "fellegi_sunter"  # the independent third line uses clerical bootstrap
    )
    archive_only_rates: bool = False  # ablation 3, second half
    ignore_health_selection: bool = False  # ablation 4
    regime_override: dict | None = None  # ablation 5: development-average regime
    mortality_improvement: dict | None = None  # third line's independent history fit
    reconstruction_uncertainty: bool = True  # deletion test: one fixed population path
    rake_to_experience: bool = True  # false only for the true-population control
    experience_share_strategy: str = (
        "published"  # third line advances an absolute elder cohort component
    )
    # The experience file on its own: no register, no archive, no survey. The control
    # that files this is the one that says what the microdata is worth.
    experience_only: bool = False
    tail: str = "simulated"  # ablations 6 and 7
    padding: float = 1.6
    allocation: str = "shortfall"  # ablation 8 uses "proportional"
    n_link_imputations: int = 6
    seed: int = 20260904


def _rate_interval(
    counts: np.ndarray, exposure: np.ndarray, point: np.ndarray, k: float
) -> tuple[np.ndarray, np.ndarray]:
    """Gamma interval around a partially pooled rate."""
    if not np.isfinite(k):
        k = max(float(np.nanmax(exposure)), 1.0) * 10.0
    alpha = np.asarray(counts, dtype=np.float64) + k * np.asarray(
        point, dtype=np.float64
    )
    beta = np.asarray(exposure, dtype=np.float64) + k
    return _gamma_interval(alpha, beta)


def _urbanity(data: dict, mean_cube: np.ndarray, n_counties: int) -> np.ndarray:
    """The published urbanity covariate: the rank of persons per land cell, in [0, 1].

    Land cells come from ``geography.csv`` and the persons from the reconstruction the
    caller passed in, which is the definition ``contract.json`` gives. With no land
    column the county's own size stands in, and the response model then reads a size
    gradient rather than a density one.
    """
    persons = np.asarray(mean_cube, dtype=np.float64).sum(axis=(1, 2))
    land = data.get("land_cells")
    if land is None or len(np.asarray(land)) != n_counties:
        density = persons
    else:
        density = persons / np.maximum(np.asarray(land, dtype=np.float64), 1.0)
    order = np.argsort(np.argsort(density))
    return order / max(n_counties - 1, 1)


def _register_head_ages(population, tick: int) -> np.ndarray | None:
    """Ages of the oldest person in each register household, the survey's head age."""
    if population is None or "household_id" not in population.columns:
        return None
    frame = population.drop_duplicates(subset=["person_id"]) \
        if "person_id" in population.columns else population
    age = (tick - frame["birth_tick"].to_numpy(dtype=np.int64)) // 12
    household = frame["household_id"].to_numpy()
    import pandas as pd
    return pd.Series(age).groupby(household).max().to_numpy(dtype=np.float64)


def _register_completeness(data: dict, county_state: np.ndarray,
                           mean_cube: np.ndarray) -> np.ndarray | None:
    """Each state's register shortfall against the published benchmark.

    The completeness axis rides on the county economic gradient, and the register that
    reports that gradient is itself thinned by it, so a covariate built from the register
    measures the mechanism through the mechanism. The benchmark's own state series does
    not go through it at all, which is why the ratio of the two is the completeness proxy
    the inclusion surface reads.
    """
    benchmark = data.get("benchmark") or {}
    item = benchmark.get("persons") or {}
    series = item.get("state") if isinstance(item, dict) else None
    if series is None:
        return None
    series = np.asarray(series, dtype=np.float64)
    n_states = int(np.asarray(county_state).max()) + 1
    if series.shape != (n_states,) or not np.all(series > 0):
        return None
    persons = np.asarray(mean_cube, dtype=np.float64).sum(axis=(1, 2))
    register = np.bincount(np.asarray(county_state, dtype=np.int64), weights=persons,
                           minlength=n_states)
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = np.where(series > 0, register / np.maximum(series, 1e-9), 1.0)
    return np.clip(ratio, 0.5, 2.0)


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
    exp_arrays = experience_arrays(experience, n_states)
    regional_loading_evidence = infer_regional_shock_loadings(
        exp_arrays, ac.shock_family
    )
    if (
        regional_loading_evidence["band"] is not None
        and ac.n_regions != n_states
    ):
        raise MissingActuarialInputs(
            "regional shock loadings are identified at state level, so reserve regions "
            "must be states"
        )
    experience_last_tick = int(
        (contract.get("experience_history") or {}).get(
            "last_year_ends_at_tick", tick - 12
        )
    )
    experience_level_years_ahead = max(
        (tick - experience_last_tick) / 12.0 + 0.5, 0.0
    )

    # 0. The composition of the priced population is raked to the experience file's own
    #    state evidence. The shared lines use state shares. The third line advances the
    #    absolute elder stock over the publication lag, then reconciles its county split
    #    between the linked-register reconstruction and the survey.
    if not params.reconstruction_uncertainty:
        paths = paths.mean(axis=0, keepdims=True)
    response = None
    survey = None
    cohort_component = None
    if params.rake_to_experience:
        if params.experience_share_strategy == "published":
            shares = experience_state_shares(exp_arrays)
        elif params.experience_share_strategy == "advanced":
            shares = advanced_experience_state_shares(
                exp_arrays, experience_level_years_ahead
            )
        elif params.experience_share_strategy == "cohort_component":
            initial_mean = paths.mean(axis=0)
            state_age_profile = _state_sum(initial_mean, county_state, n_states)
            improvement = params.mortality_improvement or {}
            state_target = advanced_experience_state_exposure(
                exp_arrays,
                experience_level_years_ahead,
                age_profile=state_age_profile,
                mortality_drift=float(improvement.get("mortality_drift", 0.0)),
            )
            urbanity = _urbanity(data, initial_mean, n_counties)
            register_ages = _register_head_ages(data.get("population"), tick)
            response = fit_survey_response(
                data["survey"], county_state, urbanity, register_ages
            )
            survey = data["survey"]
            survey_weights = (
                nonresponse_weights(survey, response)
                if response.get("fitted")
                else None
            )
            paths, cohort_component = rake_to_cohort_component(
                paths,
                county_state,
                state_target,
                survey,
                survey_weights=survey_weights,
                minimum_level_age=DEFAULT_ELIGIBLE_MIN_AGE,
            )
            shares = None
        else:
            raise ValueError(
                f"unknown experience share strategy {params.experience_share_strategy!r}"
            )
        if shares is not None:
            paths = rake_to_state_shares(paths, county_state, shares)
    mean_cube = paths.mean(axis=0)

    # 1. Current-period exposure. This is a denominator for the signals the projection
    #    is fitted on, not a published quantity: what the release carries is the
    #    projected exposure over the horizon window, which the simulation accumulates.
    current_exposure = current_band_exposure(mean_cube, 12)
    state_exposure = _state_sum(current_exposure, county_state, n_states)
    state_population = state_exposure

    # 2. The survey's own response model, and the anchor read through it. Version four
    #    draws the response coefficients per world, so they are fitted here rather than
    #    carried from a development world, and the anchor is where they are spent.
    if params.experience_only:
        # The control that files the experience file on its own opens no microdata here,
        # which is the whole content of its claim.
        response = {"fitted": False}
        survey = None
        heaping = {"excess": 0.0, "fitted": False}
    else:
        if response is None:
            urbanity = _urbanity(data, mean_cube, n_counties)
            register_ages = _register_head_ages(data.get("population"), tick)
            response = fit_survey_response(data["survey"], county_state, urbanity,
                                           register_ages)
            survey = data["survey"]
        if response.get("fitted") and "weight" not in survey.columns:
            survey = survey.assign(weight=nonresponse_weights(survey, response))
        heaping = age_heaping_intensity(survey["age"].to_numpy(dtype=np.int64)) \
            if "age" in survey.columns else {"excess": 0.0, "fitted": False}

    # 3. Health-source selection, anchored on the survey item. The anchor asks about any
    #    admission inside the window, so the archive count it is compared against is
    #    every recent patient, not only the ones whose diagnosis qualifies. Comparing the
    #    anchor with the qualifying subset alone reads the share of admissions that
    #    qualify as if it were the share of patients the archive holds.
    empty_cells = (n_states, n_bands, 2)
    if params.experience_only:
        anchor = {"prevalence": np.full(empty_cells, np.nan),
                  "effective_n": np.zeros(empty_cells), "available": False}
        archive_recent = np.zeros(empty_cells)
        completeness = None
    else:
        anchor = anchor_prevalence(survey, ac, county_state, heaping["excess"])
        archive_county = archive_recent_counts(data["health"], county_state, tick,
                                               ac.anchor_window_months, None)
        archive_recent = _state_sum(archive_county, county_state, n_states)
        completeness = _register_completeness(data, county_state, mean_cube)
    if params.ignore_health_selection or not anchor["available"]:
        inclusion = {"pi": np.ones((n_states, n_bands, 2)), "pooled": 1.0,
                     "available": False}
    else:
        inclusion = inclusion_probability(archive_recent, anchor, state_population,
                                          completeness)
    pi = np.asarray(inclusion["pi"], dtype=np.float64)

    # 4. Regime from the historical experience file. Its last annual average is centred
    #    six months before that year ends, and the publication lag sits between that end
    #    and the revised snapshot. The level is carried across both intervals before the
    #    simulation carries the drift through the projection.
    slope = gompertz_slope(exp_arrays)
    mortality_state = state_rates_from_experience(
        exp_arrays,
        ac,
        experience_level_years_ahead,
        "mortality",
        shock_family=ac.shock_family,
        improvement_override=params.mortality_improvement,
    )
    incidence_state = state_rates_from_experience(
        exp_arrays,
        ac,
        experience_level_years_ahead,
        "incidence",
        shock_family=ac.shock_family,
    )
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

    # 5. County shape of mortality: register disappearance between the two vintages, net
    #    of the records that left for a reason other than death. The two vintages share
    #    no record identifier in version four, so the join is the probabilistic one and
    #    the estimate is averaged over imputations of the link set. What is subtracted is
    #    not one flat rate: the two declared mechanisms that break a link, a move that
    #    outlives an address and a reported birth date that changed, have opposite age
    #    patterns, and both are fitted here.
    if params.experience_only:
        shape = (n_counties, n_bands, 2)
        vintages = {
            "gone": np.zeros(shape),
            "at_risk": np.zeros(shape),
            "linkage_spread": np.zeros(shape),
            "months": 12,
            "n_imputations": 0,
        }
        age_error = {
            "rate": np.zeros(n_bands),
            "share": 0.0,
            "sd_years": 0.0,
            "fitted": False,
        }
    else:
        vintages = vintage_death_counts(
            data["population_preliminary"],
            data["population"],
            tick_pre,
            tick,
            county_state,
            rng,
            n_imputations=params.n_link_imputations,
            deterministic=params.deterministic_linkage,
            linkage_strategy=params.linkage_strategy,
        )
        age_error = estimate_age_error(
            data["population_preliminary"], data["population"], tick_pre
        )
    months = vintages["months"]
    at_risk = vintages["at_risk"]
    mobility = mobility_profile(exp_arrays)
    error_profile = np.asarray(age_error["rate"], dtype=np.float64)
    if error_profile.sum() > 0:
        error_profile = error_profile / error_profile.mean()
    else:
        error_profile = np.ones(n_bands)
    churn_fit = fit_churn(vintages["gone"], at_risk, months,
                          mortality_state["rate"][county_state], mobility, error_profile,
                          county_state)
    churn = churn_fit["churn"]
    if params.archive_only_rates:
        churn = np.zeros_like(churn)
    death_counts = np.maximum(vintages["gone"] - churn * at_risk, 0.0)
    death_exposure = at_risk * (months / 12.0)
    if params.archive_only_rates:
        pooled = credibility_rate(_state_sum(death_counts, county_state, n_states),
                                  _state_sum(death_exposure, county_state, n_states))
        mortality_state = {"rate": pooled["rate"], "lower": pooled["lower"],
                           "upper": pooled["upper"], "base_rate": pooled["rate"],
                           "improvement": {"drift": 0.0, "drift_se": 0.02,
                                           "fitted": False}}
    #    The file and the vintages measure the same state level from sources that share
    #    nothing, so the level is the precision-weighted average of the two. The variance
    #    the vintage side reports carries its churn correction, which is most of what it
    #    measures, so in practice it takes almost none of the weight and the code says why
    #    rather than assuming it.
    churn_count_sd = np.asarray(churn_fit.get("residual_sd", np.zeros(n_counties)),
                                dtype=np.float64)[:, None, None] * at_risk
    death_count_variance = vintages["gone"] + churn_count_sd ** 2 + \
        vintages["linkage_spread"] ** 2
    state_death_counts = _state_sum(death_counts, county_state, n_states)
    state_death_exposure = _state_sum(death_exposure, county_state, n_states)
    vintage_state_rate = credibility_rate(state_death_counts,
                                          np.maximum(state_death_exposure, 1e-9))["rate"]
    experience_deaths = exp_arrays["deaths"].sum(axis=0)
    if not params.archive_only_rates:
        mortality_blend = blend_levels(
            mortality_state["rate"], 1.0 / np.maximum(experience_deaths, 0.5),
            vintage_state_rate,
            _state_sum(death_count_variance, county_state, n_states) /
            np.maximum(state_death_counts ** 2, 1e-9))
        mortality_state = dict(mortality_state)
        mortality_state["rate"] = mortality_blend["rate"]
    else:
        mortality_blend = {"weight": np.ones(1)}
    mortality_county = county_rate_shape(
        death_counts, death_exposure, county_state, mortality_state["rate"],
        None if params.archive_only_rates
        else death_count_variance.sum(axis=(1, 2)))

    # 6. County shape of incidence: first qualifying events in the archive, divided by
    #    the estimated inclusion probability of an admitted person in that cell.
    events = np.zeros((n_counties, n_bands, 2)) if params.experience_only else \
        archive_recent_counts(data["health"], county_state, tick, 12, ac.qualifying_groups)
    events_adjusted = events / np.maximum(pi[county_state], INCLUSION_BOUNDS[0])
    if params.archive_only_rates:
        pooled = credibility_rate(_state_sum(events, county_state, n_states),
                                  np.maximum(state_exposure, 1e-9))
        incidence_state = {"rate": pooled["rate"], "lower": pooled["lower"],
                           "upper": pooled["upper"], "base_rate": pooled["rate"],
                           "improvement": {"drift": 0.0, "drift_se": 0.02,
                                           "fitted": False}}
    #    The anchored archive is the second independent measurement of the incidence
    #    level: dividing the archive's qualifying events by the inclusion probability is
    #    the same arithmetic as multiplying the anchored prevalence by the population and
    #    by the archive's own qualifying share, so what it costs is the anchor's sampling
    #    error. That error is measured here and it is what sets the weight.
    state_events = _state_sum(events, county_state, n_states)
    state_events_adjusted = _state_sum(events_adjusted, county_state, n_states)
    # The archive's window is one year, and a year carries the family's expected loading
    # with no way to say whether this one did. The unconditional expectation is taken out
    # here for the same reason it is taken out of the file: the continuation puts it back.
    archive_loading = 1.0 + expected_shock_loading(ac.shock_family, "incidence")
    archive_state_rate = credibility_rate(state_events_adjusted / archive_loading,
                                          np.maximum(state_exposure, 1e-9))["rate"]
    experience_events = exp_arrays["qualifying_events"].sum(axis=0)
    anchor_variance = anchor_log_variance(anchor, ac.anchor_sensitivity,
                                          ac.anchor_specificity) \
        if anchor["available"] and not params.ignore_health_selection \
        else np.zeros((n_states, n_bands, 2))
    archive_variance = 1.0 / np.maximum(state_events, 0.5) + anchor_variance
    if not params.archive_only_rates:
        incidence_blend = blend_levels(incidence_state["rate"],
                                       1.0 / np.maximum(experience_events, 0.5),
                                       archive_state_rate, archive_variance)
        incidence_state = dict(incidence_state)
        incidence_state["rate"] = incidence_blend["rate"]
    else:
        incidence_blend = {"weight": np.ones(1)}
    event_count_variance = events * (1.0 + np.mean(anchor_variance))
    incidence_county = county_rate_shape(
        events_adjusted, np.maximum(current_exposure, 1e-9), county_state,
        incidence_state["rate"],
        None if params.archive_only_rates else event_count_variance.sum(axis=(1, 2)))

    # 7. Single-year schedules for the projection, one per state rather than one for the
    #    nation. The experience file is published by state, and a state's own level is
    #    what prices that region's obligation; collapsing it to a national schedule with
    #    one scalar per county throws away exactly the between-region variation the
    #    regional tails are scored on. The age gradient inside a band is the world's own
    #    Gompertz slope, fitted on the same file.
    schedule_slope = slope["slope"] if slope["fitted"] else None
    q_ages = np.stack([np.stack([expand_band_rates(mortality_state["rate"][s, :, x],
                                                   state_exposure[s, :, x],
                                                   slope=schedule_slope)
                                 for x in range(2)], axis=1) for s in range(n_states)])
    lam_ages = np.stack([np.stack([expand_band_rates(incidence_state["rate"][s, :, x],
                                                     state_exposure[s, :, x])
                                   for x in range(2)], axis=1) for s in range(n_states)])
    level_m = np.clip(mortality_county["deviation"], 0.4, 2.5)
    level_i = np.clip(incidence_county["deviation"], 0.4, 2.5)
    mortality_full = q_ages[county_state] * level_m[:, None, None]
    incidence_full = lam_ages[county_state] * level_i[:, None, None]
    migration_full = np.stack([np.stack([_band_to_ages(migration["rate"][county_state[c], :, s])
                                         for s in range(2)], axis=1)
                               for c in range(n_counties)])
    migration_se_full = np.stack([np.stack([_band_to_ages(migration["se"][county_state[c], :, s])
                                            for s in range(2)], axis=1)
                                  for c in range(n_counties)])

    # 8. Everyone enters the window without a qualifying event: the scored event is a
    #    person's first inside the sixty months, which is the convention the truth uses
    #    and the only one the participant's files can support.
    not_yet = np.ones((n_counties, MAX_AGE + 1, 2))

    # 9. How well the two levels are known, region by region. These are the widths the
    #    continuation draws its level errors from, and they are estimated rather than
    #    assumed: the counts behind each level, the spread across the linkage imputations
    #    that produced the death counts, and the sampling error of the anchor the
    #    inclusion probability divides by.
    with np.errstate(invalid="ignore", divide="ignore"):
        vintage_state = np.where(
            _state_sum(death_exposure, county_state, n_states) > 0,
            _state_sum(death_counts, county_state, n_states) /
            np.maximum(_state_sum(death_exposure, county_state, n_states), 1e-9), 0.0)
        archive_state = np.where(
            state_exposure > 0,
            _state_sum(events_adjusted, county_state, n_states) /
            np.maximum(state_exposure, 1e-9), 0.0)
    linkage_relative = float(vintages["linkage_spread"].sum() /
                             max(vintages["gone"].sum(), 1.0))
    uncertainty = rate_uncertainty(exp_arrays, vintage_state, archive_state,
                                   {"mortality": mortality_state["rate"],
                                    "incidence": incidence_state["rate"]},
                                   linkage_relative)
    rates = {
        "mortality": mortality_full, "incidence": incidence_full,
        "migration": migration_full, "migration_se": migration_se_full,
        "not_yet": not_yet, "fertility": float(fertility_rate),
        "mortality_drift": mortality_drift, "mortality_drift_se": mortality_drift_se,
        "incidence_drift": incidence_drift, "incidence_drift_se": incidence_drift_se,
        "mortality_log_sd": float(override.get("mortality_log_sd",
                                               uncertainty["mortality_national"])),
        "incidence_log_sd": float(override.get("incidence_log_sd",
                                               uncertainty["incidence_national"])),
        "mortality_log_sd_region": uncertainty["mortality_region"],
        "incidence_log_sd_region": uncertainty["incidence_region"],
    }

    # 10. One simulation produces three things: the projected exposure and rates the
    #     release table carries, the liability paths, and the reserve the tails decide.
    sim = params.simulation
    n_paths = sim.n_paths
    index = np.arange(n_paths) % paths.shape[0]
    simulated = simulate_liabilities(
        paths[index], rates, ac, sim, regional_loading_evidence
    )
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
        # A cushion of a fixed share of expected cost, the shape an over-cautious
        # analyst adds. It changes only the tail forecasts; allocation is scored under
        # the separate public-total constraint below.
        cushion = (params.padding - 1.0) * summary["mean"]
        q_hat = summary["q"] + cushion
        es_hat = summary["es"] + cushion
    else:
        q_hat = summary["q"].copy()
        es_hat = summary["es"].copy()
    mean_hat = summary["mean"].copy()
    q_hat = np.maximum(q_hat, mean_hat)
    es_hat = np.maximum(es_hat, q_hat)
    # A_B is fixed before the submission: the verifier and every reference line read the
    # same participant-visible baseline_share from the contract. Forecast q95 and ES95 do
    # not change either this baseline or the feasible allocation set.
    if ac.reserve_baseline_share is None:
        raise MissingActuarialInputs(
            "contract reserve baseline_share is required for the frozen reserve baseline"
        )
    baseline = proportional_reserve(ac.reserve_baseline_share, ac.reserve_total)
    if params.allocation == "proportional":
        allocation = baseline
        allocation_detail = {"feasible": True, "nu": float("nan"),
                             "reason": (
                                 "public contract baseline_share proportional split over "
                                 "finite nonnegative allocations summing to the public "
                                 "reserve total; q95 and es95 are forecasts only"
                             )}
    else:
        allocation_detail = allocate_reserve(
            liability, ac.reserve_total, ac.reserve_weights
        )
        allocation = allocation_detail["allocation"]
    summary_out = {"mean": mean_hat, "q": q_hat, "es": es_hat}
    rows = reserve_rows(summary_out, allocation)
    diagnostics = {
        "inclusion_pooled": float(inclusion["pooled"]),
        "selection_adjusted": bool(inclusion["available"]),
        "mortality_drift": mortality_drift,
        "incidence_drift": incidence_drift,
        "linkage_imputations": int(vintages["n_imputations"]),
        "linkage_strategy": params.linkage_strategy,
        "linkage_spread": float(np.mean(vintages["linkage_spread"])),
        "reconstruction_uncertainty": bool(params.reconstruction_uncertainty),
        "experience_share_strategy": params.experience_share_strategy,
        "experience_level_years_ahead": experience_level_years_ahead,
        "cohort_component": cohort_component,
        "mortality_history_strategy": str(
            mortality_state["improvement"].get("strategy", "shared")
        ),
        "tail_calibrated_to_total": False,
        "objective": shortfall_objective(allocation, liability, ac.reserve_weights),
        "objective_baseline": shortfall_objective(
            baseline, liability, ac.reserve_weights
        ),
        "exceedance": exceedance_probability(q_hat, liability).tolist(),
        "quantile_score": quantile_score(q_hat, liability).tolist(),
        "reserve_feasible": bool(allocation_detail.get("feasible", True)),
        "reserve_allocation_rule": str(allocation_detail["reason"]),
        "reserve_baseline_rule": (
            "public contract baseline_share proportional split; q95 and es95 are "
            "forecast quantities, not allocation floors"
        ),
        "n_paths": int(liability.shape[0]),
        # What this world was read to be. Every one of these is a per-world draw the
        # generator makes, so a freeze report that carries them can say whether the
        # reference read the world or carried a constant into it.
        "mortality_drift_se": mortality_drift_se,
        "shock_posterior": np.asarray(
            mortality_state["improvement"].get("shock_posterior", [])).tolist(),
        "gompertz_slope": float(slope["slope"]) if slope["fitted"] else None,
        "age_heaping": float(heaping["excess"]),
        "response_urban": float(response.get("urban", 0.0)),
        "response_age": float(response.get("age", 0.0)),
        "response_rate": float(response.get("response_rate", float("nan"))),
        "age_error_share": float(age_error.get("share", 0.0)),
        "churn_mix": np.asarray(churn_fit.get("mix", [])).tolist(),
        "inclusion_surface": np.asarray(
            inclusion.get("surface", {}).get("coefficients", [])).tolist()
        if isinstance(inclusion.get("surface"), dict) else [],
        "mortality_level_weight": float(np.mean(mortality_blend["weight"])),
        "incidence_level_weight": float(np.mean(incidence_blend["weight"])),
        "mortality_log_sd": float(rates["mortality_log_sd"]),
        "incidence_log_sd": float(rates["incidence_log_sd"]),
        "mortality_log_sd_region": np.asarray(
            uncertainty["mortality_region"]).tolist(),
        "regional_shock_loading": regional_loading_diagnostics(
            regional_loading_evidence,
            simulated["regional_loading"],
            simulated["regional_loading_held_years"],
        ),
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
    """Run the layer and write the three version-four files, or return None.

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
    release_rows = reconcile_additive_release_rows(release_rows, county_state)
    projection_rows = reconcile_additive_release_rows(projection_rows, county_state)
    rows = widen_release_rows(release_rows) + result["rate_rows"]
    write_actuarial_submission(out_dir, rows, projection_rows, cube, suppress_below,
                               result["reserve"], band_labels, sex_labels)
    result["contract"] = ac
    result["release"] = rows
    return result
