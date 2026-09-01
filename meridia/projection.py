"""Stage ten: the forward projection and the allocation committed against it.

The release's last stage looks ahead. The agent publishes the next vintage's key figures
with intervals and commits a bounded allocation of a resource across counties. The world
then runs forward under its real dynamics, shocks included, and the sealed future decides
two things: whether the projected intervals covered what happened (scored with the same
error, coverage, and interval score as the release), and the realized loss of the
allocation against true future demand, which no interval hedging can soften.

Truth here is the same population advanced by the demography layer; the projection
estimands are the release estimands evaluated on the future table.
"""

from __future__ import annotations

import numpy as np

from .demography import DemographyParams, run_years
from .release import compute_truth

DEMAND_ESTIMAND = "elders_65_plus"   # v0 demand proxy: care demand follows the old


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


def score_allocation(allocation: np.ndarray, demand: np.ndarray, budget: float,
                     tolerance: float = 1e-9) -> dict:
    """Realized loss of a committed county allocation against true demand.

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
