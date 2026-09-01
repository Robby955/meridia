"""The forecast task: a clean history in, a sealed future out, capacity committed.

Distinct from the reconstruction task and never sharing a world with it. The participant
receives the world's own records as of a snapshot, complete and consistent: every living
person with age, sex, education, income, household, and county; every household; every
hospital with its beds; and the full event history up to the snapshot (births, deaths,
household moves and closures, job starts and ends, hospital admissions and discharges),
with opaque identifiers. Nothing is corrupted. The difficulty is the future: the world
runs forward for years under its real dynamics and sealed shocks, and the participant is
scored on a projection table at the horizon and on the realized loss of a hospital-bed
allocation across counties against the admissions that actually arrive.

Everything is a deterministic function of the seed and the parameters.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .admin import build_admin
from .businesses import build_businesses
from .character import draw_world_character
from .demography import draw_world_shocks
from .dwellings import build_dwellings
from .events import EVENT_TYPES, build_event_history, replay_event_history
from .hospitals import build_hospitals
from .hydrology import fill_depressions, flow_accumulation, flow_directions
from .identities import build_initial_identity_map
from .microdata import build_microdata
from .population import build_population, resource_outposts
from .projection import person_table_from_state, project_truth_from_history, score_allocation
from .release import ESTIMANDS, LEVELS, compute_truth
from .scoring import evaluate_gates, score_release, validate_release
from .terrain import generate_elevation

EVENT_NAMES = {code: name for name, code in EVENT_TYPES.items()}


@dataclass(frozen=True)
class ForecastParams:
    grid: tuple[int, int] = (288, 384)
    n_settlements: int = 24
    n_states: int = 6
    history_months: int = 36        # ledger months the participant sees
    horizon_months: int = 120       # sealed months the world runs forward
    demand_window_months: int = 12  # admissions counted over the last year before the horizon
    budget_fraction: float = 0.9    # of admissions in the last year before the snapshot
    max_shocks: int = 3
    total: int | None = None


def build_forecast_world(seed: int, params: ForecastParams = ForecastParams()) -> dict:
    height, width = params.grid
    character = draw_world_character(seed)
    world = generate_elevation(seed, height, width)
    outlets = ~world["land"]
    outlets[0, :] = outlets[-1, :] = outlets[:, 0] = outlets[:, -1] = True
    filled = fill_depressions(world["elevation"], world["sea_level"])
    direction = flow_directions(filled, outlets)
    accumulation = flow_accumulation(direction, outlets)
    people = build_population(world, accumulation, params.total, params.n_settlements,
                              params=character["population"], seed=seed)
    micro = build_microdata(people["population"], people["habitability"],
                            people["settlements"], seed, params=character["microdata"])
    admin = build_admin(world["land"], people["settlements"], resource_outposts(world, seed),
                        n_states=params.n_states)
    identities = build_initial_identity_map(micro, seed)
    dwellings = build_dwellings(micro, seed, identities)
    businesses = build_businesses(micro, seed, identities)
    hospitals = build_hospitals(micro, seed, identities, businesses)
    months = params.history_months + params.horizon_months
    shocks = draw_world_shocks(seed, max(3, months // 12 + 1), params.max_shocks)
    history = build_event_history(micro, seed, identities, dwellings, businesses, hospitals,
                                  months=months, shocks=shocks)
    snapshot = int(history["snapshot_tick"])
    return {"seed": seed, "params": params, "character": character, "admin": admin,
            "hospitals": hospitals, "history": history, "shocks": shocks,
            "ticks": {"start": snapshot, "snapshot": snapshot + params.history_months,
                      "horizon": snapshot + months}}


def _opaque(ids: np.ndarray, rng: np.random.Generator, table: dict) -> np.ndarray:
    """Opaque tokens for truth ids, consistent within a packet; zero stays zero."""
    out = np.zeros(len(ids), dtype=np.uint64)
    for i, value in enumerate(np.asarray(ids, dtype=np.uint64)):
        v = int(value)
        if v == 0:
            continue
        if v not in table:
            table[v] = int(rng.integers(1 << 40, 1 << 62))
        out[i] = table[v]
    return out


def admissions_by_county(history: dict, hospitals: dict, admin: dict,
                         start_tick: int, end_tick: int) -> np.ndarray:
    """Admissions with start_tick <= tick < end_tick, counted by the hospital's county."""
    event = history["event"]
    admitted = event["event_type"] == EVENT_TYPES["encounter_admitted"]
    window = admitted & (event["tick"] >= start_tick) & (event["tick"] < end_tick)
    county_flat = admin["county"].flatten()
    hospital_county = {int(h): int(county_flat[int(c)]) for h, c in
                       zip(hospitals["hospital"]["truth_hospital_id"], hospitals["hospital"]["cell"])}
    counties = np.asarray([hospital_county.get(int(h), -1) for h in event["truth_hospital_id"][window]])
    counts = np.zeros(admin["n_counties"], dtype=np.float64)
    np.add.at(counts, counties[counties >= 0], 1.0)
    return counts


def _write(path: Path, table: dict) -> None:
    import pandas as pd
    pd.DataFrame({k: np.asarray(v) for k, v in table.items()}).to_csv(path, index=False, float_format="%.6f")


def build_forecast_packet(seed: int, out_dir: Path, params: ForecastParams = ForecastParams(),
                          development: bool = False) -> dict:
    out_dir = Path(out_dir)
    if out_dir.exists():
        raise FileExistsError(f"packet directory already exists: {out_dir}")
    built = build_forecast_world(seed, params)
    admin, history, hospitals, ticks = built["admin"], built["history"], built["hospitals"], built["ticks"]
    participant, retained = out_dir / "participant", out_dir / "retained"
    participant.mkdir(parents=True)
    retained.mkdir()
    rng = np.random.default_rng(np.random.SeedSequence([seed, 0x0F0C]))
    tokens: dict[int, int] = {}
    county_flat = admin["county"].flatten()
    S = ticks["snapshot"]

    # Participant: the world as of the snapshot, clean, with opaque identifiers.
    state = replay_event_history(history, S)
    person, household = state["person"], state["household"]
    alive = np.flatnonzero(person["is_alive"])
    active = np.flatnonzero(household["is_active"])
    _write(participant / "persons.csv", {
        "person_id": _opaque(person["truth_person_id"][alive], rng, tokens),
        "household_id": _opaque(person["truth_household_id"][alive], rng, tokens),
        "birth_tick": person["birth_tick"][alive], "sex": person["sex"][alive],
        "education": person["education"][alive],
        "income": person["income_cents"][alive].astype(np.float64) / 100.0,
        "county": county_flat[person["cell"][alive]]})
    _write(participant / "households.csv", {
        "household_id": _opaque(household["truth_household_id"][active], rng, tokens),
        "county": county_flat[household["cell"][active]]})
    hosp = hospitals["hospital"]
    _write(participant / "hospitals.csv", {
        "hospital_id": _opaque(hosp["truth_hospital_id"], rng, tokens),
        "county": county_flat[hosp["cell"]], "bed_count": hosp["bed_count"],
        "opening_year": hosp["opening_year"]})
    event = history["event"]
    visible = event["tick"] <= S
    county_of_event = np.full(int(visible.sum()), -1, dtype=np.int64)
    to_cell = event["to_cell"][visible]
    from_cell = event["from_cell"][visible]
    county_of_event[to_cell >= 0] = county_flat[to_cell[to_cell >= 0]]
    county_of_event[(to_cell < 0) & (from_cell >= 0)] = county_flat[from_cell[(to_cell < 0) & (from_cell >= 0)]]
    # Person events carry the person's birth month and sex, as a civil ledger would.
    all_persons = history["terminal_state"]["person"]
    birth_of = dict(zip(all_persons["truth_person_id"].tolist(), all_persons["birth_tick"].tolist()))
    sex_of = dict(zip(all_persons["truth_person_id"].tolist(), all_persons["sex"].tolist()))
    pid = event["truth_person_id"][visible]
    birth_tick_col = np.asarray([birth_of.get(int(i), -1) if int(i) else -1 for i in pid], dtype=np.int64)
    sex_col = np.asarray([sex_of.get(int(i), -1) if int(i) else -1 for i in pid], dtype=np.int64)
    _write(participant / "events.csv", {
        "tick": event["tick"][visible],
        "event": np.asarray([EVENT_NAMES[int(t)] for t in event["event_type"][visible]]),
        "person_id": _opaque(event["truth_person_id"][visible], rng, tokens),
        "household_id": _opaque(event["truth_household_id"][visible], rng, tokens),
        "hospital_id": _opaque(event["truth_hospital_id"][visible], rng, tokens),
        "encounter_id": _opaque(event["truth_encounter_id"][visible], rng, tokens),
        "county": county_of_event, "sex": sex_col, "birth_tick": birth_tick_col,
        "service": event["service"][visible], "diagnosis_group": event["diagnosis_group"][visible],
        "outcome": event["outcome"][visible], "cause_code": event["cause_code"][visible]})
    _write(participant / "geography.csv", {"county": np.arange(admin["n_counties"], dtype=np.int64),
                                           "state": admin["county_state"].astype(np.int64)})
    recent = admissions_by_county(history, hospitals, admin, S - params.demand_window_months, S)
    budget = int(round(params.budget_fraction * recent.sum()))
    contract = {
        "schema": "meridia.forecast-packet.v0",
        "estimands": [asdict(e) for e in ESTIMANDS], "levels": list(LEVELS),
        "n_states": int(admin["n_states"]), "n_counties": int(admin["n_counties"]),
        "interval_level": 0.90, "months_per_tick": 1,
        "ticks": {"start": ticks["start"], "snapshot": S, "horizon": ticks["horizon"]},
        "allocation": {"demand": "hospital admissions in the final twelve months before the horizon, by the hospital's county",
                       "level": "county", "budget": budget, "unit": "admissions"},
        "development": development,
    }
    (participant / "contract.json").write_text(json.dumps(contract, indent=1, sort_keys=True) + "\n")

    # Retained: exact truth at the horizon and the demand that arrives.
    future = project_truth_from_history(history, admin, ticks["horizon"])
    keys = sorted(future["truth"])
    _write(retained / "truth_horizon.csv", {"estimand": [k[0] for k in keys], "level": [k[1] for k in keys],
                                            "unit": [k[2] for k in keys], "value": [future["truth"][k] for k in keys]})
    now_person, now_cells = person_table_from_state(state, S)
    truth_now = compute_truth(now_person, now_cells, admin)
    keys = sorted(truth_now)
    _write(retained / "truth_snapshot.csv", {"estimand": [k[0] for k in keys], "level": [k[1] for k in keys],
                                             "unit": [k[2] for k in keys], "value": [truth_now[k] for k in keys]})
    demand = admissions_by_county(history, hospitals, admin,
                                  ticks["horizon"] - params.demand_window_months, ticks["horizon"])
    _write(retained / "demand_horizon.csv", {"county": np.arange(admin["n_counties"]), "admissions": demand})
    (retained / "world.json").write_text(json.dumps({
        "seed": seed, "character": built["character"]["draw"], "shocks": built["shocks"],
        "ticks": ticks, "params": asdict(params)}, indent=1, sort_keys=True, default=str) + "\n")
    if development:
        (participant / "truth").mkdir()
        for name in ("truth_horizon.csv", "demand_horizon.csv"):
            (participant / "truth" / name).write_bytes((retained / name).read_bytes())

    manifest = {"schema": "meridia.forecast-packet.manifest.v0", "development": development,
                "participant": {}, "retained": {}}
    for side in ("participant", "retained"):
        for path in sorted((out_dir / side).rglob("*")):
            if path.is_file():
                manifest[side][str(path.relative_to(out_dir / side))] = {
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "bytes": path.stat().st_size}
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")
    return manifest


def verify_forecast(packet_dir: Path, submission_dir: Path, bars: dict | None = None,
                    alpha: float = 0.10) -> dict:
    """Score a forecast submission: projection.csv in the release schema, allocation.csv."""
    import pandas as pd
    from .verify import admin_from_packet, load_rows, load_truth
    packet_dir, submission_dir = Path(packet_dir), Path(submission_dir)
    contract = json.loads((packet_dir / "participant" / "contract.json").read_text())
    admin = admin_from_packet(packet_dir)
    truth = load_truth(packet_dir / "retained" / "truth_horizon.csv")
    projection = load_rows(submission_dir / "projection.csv")
    schema_errors = validate_release(projection, admin)
    metrics = score_release(projection, truth, admin, alpha)
    demand_frame = pd.read_csv(packet_dir / "retained" / "demand_horizon.csv").sort_values("county")
    demand = demand_frame["admissions"].to_numpy(dtype=np.float64)
    alloc_frame = pd.read_csv(submission_dir / "allocation.csv").sort_values("county")
    allocation = np.zeros(admin["n_counties"])
    allocation[alloc_frame["county"].to_numpy(dtype=np.int64)] = alloc_frame["allocation"].to_numpy(dtype=np.float64)
    allocation_score = score_allocation(allocation, demand, float(contract["allocation"]["budget"]))
    gates = evaluate_gates(schema_errors, [], metrics, None, bars)
    reasons = list(gates["reasons"])
    if not allocation_score["feasible"]:
        reasons.append("allocation: infeasible")
    ceiling = (bars or {}).get("allocation_regret_ceiling")
    if ceiling is not None and allocation_score["feasible"] and allocation_score["regret"] > ceiling:
        reasons.append(f"allocation: regret {allocation_score['regret']:.4f} > {ceiling}")
    return {"pass": not reasons, "reasons": reasons, "schema_errors": schema_errors,
            "metrics": metrics, "allocation": allocation_score}
