"""Stage ten: the forward projection and the allocation committed against it.

The release's last stage looks ahead. The agent publishes the next vintage's key figures
with intervals and commits a bounded allocation of a resource across counties. The world
then runs forward under its real dynamics, shocks included, and the sealed future decides
two things: whether the projected intervals covered what happened (scored with the same
error, coverage, and interval score as the release), and the realized loss of the
allocation against true future demand, which no interval hedging can soften.

The future is the monthly institutional ledger replayed through the horizon tick: the
same engine that produced the observed sources, so the world an agent forecasts is the
world its records came from. ``project_truth`` keeps the annual demography loop for
development-scale experiments that never touch the ledger; the capstone uses
``project_truth_from_history``.
"""

from __future__ import annotations

import hashlib
import json

import numpy as np

from .actuarial import (ObligationContract, actuarial_pass, actuarial_pass_from_history,
                        exposure_and_rate_truth, liabilities_from_pass)
from .demography import DemographyParams, run_years
from .events import (continuation_events, continuation_redraw_year_window,
                     replay_event_history)
from .release import compute_truth

DEMAND_ESTIMAND = "elders_65_plus"   # v0 demand proxy: care demand follows the old
SHOCK_REDRAW_EVIDENCE_SCHEMA = "meridia.v4.continuation-shock-redraw.v1"


def _canonical_digest(value: object) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _shock_redraw_evidence(
    branch: dict,
    months: int,
    n_members: int,
    schedules: list[tuple[int, list[dict]]],
    source_law_sha256: str,
) -> dict:
    branch_month = int(branch["month"])
    first_future_year, future_year_count = continuation_redraw_year_window(
        branch_month, months
    )
    member_schedules = []
    for member, schedule in schedules:
        future = [
            dict(shock) for shock in schedule
            if int(shock["year"]) >= first_future_year
        ]
        member_schedules.append({"member": int(member), "future_shocks": future})
    schedule_values = [row["future_shocks"] for row in member_schedules]
    evidence = {
        "schema": SHOCK_REDRAW_EVIDENCE_SCHEMA,
        "continuation_source_law_sha256": str(source_law_sha256),
        "member_count": int(n_members),
        "redrawn_member_count": len(member_schedules),
        "first_future_year": first_future_year,
        "future_year_count": future_year_count,
        "future_year_opportunity_count": len(member_schedules) * future_year_count,
        "member_schedules": member_schedules,
        "ordered_member_schedule_digest_sha256": _canonical_digest(member_schedules),
        "distinct_future_schedule_count": len({
            json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
            for value in schedule_values
        }),
        "future_shock_year_count": sum(len(value) for value in schedule_values),
        "future_mortality_spike_year_count": sum(
            shock.get("kind") == "mortality_spike"
            for value in schedule_values for shock in value
        ),
    }
    return evidence


def person_table_from_state(state: dict, tick: int) -> tuple[dict, np.ndarray]:
    """Core person table and household cells from a replayed ledger state.

    Living persons only; ages in whole years at ``tick``; incomes in currency units;
    household indices point into the active households, in ledger order.
    """
    person = state["person"]
    household = state["household"]
    alive = np.flatnonzero(person["is_alive"])
    active = np.flatnonzero(household["is_active"])
    household_index = {int(h): i for i, h in enumerate(household["truth_household_id"][active])}
    person_household = np.asarray(
        [household_index[int(h)] for h in person["truth_household_id"][alive]], dtype=np.int64)
    core = {
        "household": person_household,
        "cell": person["cell"][alive].astype(np.int64),
        "age": ((tick - person["birth_tick"][alive]) // 12).astype(np.int16),
        "sex": person["sex"][alive].astype(np.int8),
        "role": person["role"][alive].astype(np.int8),
        "education": person["education"][alive].astype(np.int8),
        "income": person["income_cents"][alive].astype(np.float64) / 100.0,
    }
    return core, household["cell"][active].astype(np.int64)


def project_truth_from_history(history: dict, admin: dict,
                               through_tick: int | None = None) -> dict:
    """Exact future estimand table by replaying the ledger through ``through_tick``."""
    tick = int(history["terminal_tick"] if through_tick is None else through_tick)
    state = replay_event_history(history, tick)
    person, household_cell = person_table_from_state(state, tick)
    truth = compute_truth(person, household_cell, admin)
    return {"truth": truth, "tick": tick, "n_persons": len(person["age"]),
            "n_households": len(household_cell)}


def project_truth(person: dict, household_cell: np.ndarray, urbanity_flat: np.ndarray,
                  admin: dict, seed: int, years: int,
                  params: DemographyParams = DemographyParams(),
                  shocks: list[dict] | None = None) -> dict:
    """Run the world forward and return the exact future estimand table and registers."""
    future_person, future_cells, registers = run_years(
        person, household_cell, urbanity_flat, seed, years, params, shocks)
    truth = compute_truth(future_person, future_cells, admin)
    return {"truth": truth, "registers": registers, "years": years,
            "n_persons": len(future_person["age"])}


def demand_from_truth(truth: dict, admin: dict, estimand: str = DEMAND_ESTIMAND) -> np.ndarray:
    """True future demand by county, from the future estimand table."""
    return np.asarray([truth[(estimand, "county", c)] for c in range(admin["n_counties"])],
                      dtype=np.float64)


def liabilities_from_history(history: dict, admin: dict, start_tick: int, months: int,
                             contract: ObligationContract,
                             region_of_county: np.ndarray | None = None) -> np.ndarray:
    """Regional present values of the obligation over one continuation of the ledger.

    This is the member pricer the continuation ensemble is assembled from: one reading
    pass over the events after ``start_tick``, then the public discount factors.
    """
    result = actuarial_pass_from_history(history, admin, start_tick, months, contract,
                                         region_of_county)
    return liabilities_from_pass(result, contract)


def rate_truth_from_history(history: dict, admin: dict, start_tick: int, months: int,
                            contract: ObligationContract) -> dict:
    """Exposure and rate truth over the same window, from the same reading pass."""
    return exposure_and_rate_truth(
        actuarial_pass_from_history(history, admin, start_tick, months, contract), admin)


# What a pooled member pricer reads. It is module state rather than an argument because
# a forked worker inherits the parent's memory and a pickled branch would cost more than
# the member it prices.
_ENSEMBLE_JOB: tuple | None = None


def continuation_liabilities(history: dict, admin: dict, start_tick: int, months: int,
                            contract: ObligationContract, n_members: int,
                            region_of_county: np.ndarray | None = None,
                            workers: int = 1, *,
                            shock_source_law_sha256: str | None = None,
                            return_shock_evidence: bool = False,
                            ) -> np.ndarray | tuple[np.ndarray, dict]:
    """Regional liabilities on every committed continuation, shape (members, regions).

    Every member resumes from the branch state the ledger captured at ``start_tick`` and
    independently redraws the public continuation law. The ledger's designated realized
    horizon remains the point truth outside this ensemble; it is not substituted for one
    of the tail members. A member therefore costs the horizon window rather than the whole
    ledger.

    ``workers`` changes only how the members are divided between processes. Each member is
    a deterministic function of the seed and its own index, so the matrix does not depend
    on it.
    """
    if isinstance(n_members, bool) or not isinstance(n_members, int) or n_members < 1:
        raise ValueError("the ensemble member count must be a positive integer")
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise ValueError("workers must be a positive integer")
    branch = history.get("branch")
    if branch is None or int(branch["tick"]) != int(start_tick):
        raise ValueError("the ledger kept no branch state at the continuation start tick")
    start_state = replay_event_history(history, start_tick)
    rows = []
    schedules: list[tuple[int, list[dict]]] = []
    members = range(int(n_members))
    if workers > 1:
        import multiprocessing as mp
        context = mp.get_context("fork")
        globals()["_ENSEMBLE_JOB"] = (branch, start_state, admin, start_tick, months,
                                      contract, region_of_county)
        try:
            with context.Pool(int(workers)) as pool:
                priced = pool.map(_price_member, members, chunksize=8)
        finally:
            globals().pop("_ENSEMBLE_JOB", None)
    else:
        globals()["_ENSEMBLE_JOB"] = (branch, start_state, admin, start_tick, months,
                                      contract, region_of_county)
        try:
            priced = [_price_member(m) for m in members]
        finally:
            globals().pop("_ENSEMBLE_JOB", None)
    for member, (liability, schedule) in zip(members, priced, strict=True):
        rows.append(liability)
        schedules.append((member, schedule))
    liability = np.asarray(rows, dtype=np.float64)
    if not return_shock_evidence:
        return liability
    if not isinstance(shock_source_law_sha256, str) \
            or len(shock_source_law_sha256) != 64:
        raise ValueError("shock evidence requires a continuation source-law digest")
    return liability, _shock_redraw_evidence(
        branch, months, n_members, schedules, shock_source_law_sha256
    )


def _price_member(member: int) -> tuple[np.ndarray, list[dict]]:
    """One continuation, priced. Reads the job the pool's parent process set up."""
    branch, start_state, admin, start_tick, months, contract, region = _ENSEMBLE_JOB
    event, schedule = continuation_events(
        branch, member, months, return_shock_schedule=True
    )
    liability = liabilities_from_pass(
        actuarial_pass(start_state, event, admin, start_tick, months, contract, region),
        contract)
    return liability, schedule


def score_allocation(allocation: np.ndarray, demand: np.ndarray, budget: float,
                     tolerance: float = 1e-9) -> dict:
    """Realized loss of a committed county allocation against true demand.

    Retired on the version-four reconstruction surface, and kept only for the version-three
    packets already frozen and for the forecast task. Its regret is zero for any allocation
    that sits under every county's true demand, so it scores restraint rather than
    forecasting. ``meridia.actuarial.score_reserve`` is the replacement: it reads the
    committed continuation ensemble rather than one realized path, and its skill score is
    measured against a frozen practical baseline and a perfect-information allocation.

    Loss is the share of demand left unmet; the oracle loss is what a perfect forecast
    could achieve with the same budget, so regret is the part attributable to the agent.
    An allocation that is negative, non-finite, or over budget is infeasible and fails.
    """
    allocation = np.asarray(allocation, dtype=np.float64)
    demand = np.asarray(demand, dtype=np.float64)
    if allocation.shape != demand.shape:
        raise ValueError("allocation and demand must have one entry per county")
    total_demand = float(demand.sum())
    feasible = bool(np.isfinite(allocation).all() and (allocation >= -tolerance).all()
                    and allocation.sum() <= budget * (1.0 + tolerance) + tolerance)
    if not feasible or total_demand <= 0:
        return {"feasible": feasible, "loss": float("nan"), "oracle_loss": float("nan"),
                "regret": float("nan"), "unmet": float("nan"), "waste": float("nan")}
    unmet = float(np.maximum(demand - allocation, 0.0).sum())
    waste = float(np.maximum(allocation - demand, 0.0).sum())
    loss = unmet / total_demand
    oracle_loss = max(total_demand - budget, 0.0) / total_demand
    return {"feasible": True, "loss": loss, "oracle_loss": oracle_loss,
            "regret": loss - oracle_loss, "unmet": unmet, "waste": waste,
            "spent": float(allocation.sum()), "budget": float(budget)}
