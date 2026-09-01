"""Packet builder: one world, split into what an agent receives and what stays sealed.

A packet is a directory. ``participant/`` holds flat files only: a household survey at
two snapshots, the four observed sources at two snapshots, the county-to-state map, and
the contract that names the estimands, levels, snapshot ticks, projection horizon,
disclosure threshold, and allocation budget. ``retained/`` holds the exact truth tables
at the revised snapshot and at the horizon, the detailed table, and the source package's
crosswalks and mechanisms. A development packet copies the truth tables into
``participant/truth/`` so methods can be tuned on an open world; a hidden packet never
does. The manifest hashes every file and records which side it is on, and the builder
refuses to write a participant file that carries a truth column.

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
from .events import build_event_history, replay_event_history
from .hospitals import build_hospitals
from .hydrology import fill_depressions, flow_accumulation, flow_directions
from .identities import build_initial_identity_map
from .microdata import build_microdata
from .population import build_population, resource_outposts
from .projection import DEMAND_ESTIMAND, person_table_from_state, project_truth_from_history
from .release import (AGE_BAND_LABELS, ESTIMANDS, LEVELS, SEX_LABELS,
                      compute_detailed_table_truth, compute_truth)
from .sources import build_observed_sources, participant_source_snapshots
from .survey import draw_survey
from .terrain import generate_elevation

FORBIDDEN_COLUMN_PREFIXES = ("truth_", "mechanism", "crosswalk")


@dataclass(frozen=True)
class PacketParams:
    grid: tuple[int, int] = (288, 384)
    n_settlements: int = 24
    n_states: int = 6
    observed_months: int = 24        # ledger months before the revised snapshot
    preliminary_lag: int = 6         # revised minus preliminary, in months
    horizon_months: int = 60         # projection horizon after the revised snapshot
    disclosure_threshold: int = 10   # protected cell: 0 < true count < threshold
    budget_fraction: float = 0.9     # of persons 65+ in the revised population source
    max_shocks: int = 2
    total: int | None = None         # None draws the national total from the seed


def build_world(seed: int, params: PacketParams = PacketParams()) -> dict:
    """Every layer of one world, from terrain to observed sources."""
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
    months = params.observed_months + params.horizon_months
    years = max(3, months // 12 + 1)
    shocks = draw_world_shocks(seed, years, params.max_shocks)
    history = build_event_history(micro, seed, identities, dwellings, businesses, hospitals,
                                  months=months, shocks=shocks)
    snapshot = int(history["snapshot_tick"])
    revised_tick = snapshot + params.observed_months
    preliminary_tick = revised_tick - params.preliminary_lag
    sources = build_observed_sources(history, seed, admin, hospitals,
                                     preliminary_tick=preliminary_tick,
                                     revised_tick=revised_tick)
    return {
        "seed": seed, "params": params, "character": character, "world": world,
        "people": people, "micro": micro, "admin": admin, "hospitals": hospitals,
        "history": history, "sources": sources, "shocks": shocks,
        "ticks": {"snapshot": snapshot, "preliminary": preliminary_tick,
                  "revised": revised_tick, "horizon": snapshot + months},
    }


def _survey_at(built: dict, tick: int) -> dict:
    state = replay_event_history(built["history"], tick)
    person, household_cell = person_table_from_state(state, tick)
    height, width = built["params"].grid
    population = np.bincount(person["cell"], minlength=height * width).reshape(height, width)
    micro = {"person": person, "household_cell": household_cell,
             "urbanity": built["micro"]["urbanity"], "n_households": len(household_cell)}
    return draw_survey(micro, population, built["seed"] + tick)


def _truth_at(built: dict, tick: int) -> tuple[dict, np.ndarray]:
    state = replay_event_history(built["history"], tick)
    person, household_cell = person_table_from_state(state, tick)
    return (compute_truth(person, household_cell, built["admin"]),
            compute_detailed_table_truth(person, built["admin"]))


def _write_table(path: Path, table: dict, forbid_truth: bool) -> None:
    import pandas as pd
    columns = {name: np.asarray(values) for name, values in table.items()}
    if forbid_truth:
        for name in columns:
            if name.startswith(FORBIDDEN_COLUMN_PREFIXES):
                raise ValueError(f"participant file {path.name} would carry column {name!r}")
    pd.DataFrame(columns).to_csv(path, index=False, float_format="%.6f")


def _truth_rows(truth: dict) -> dict:
    keys = sorted(truth)
    return {"estimand": np.asarray([k[0] for k in keys]),
            "level": np.asarray([k[1] for k in keys]),
            "unit": np.asarray([k[2] for k in keys], dtype=np.int64),
            "value": np.asarray([truth[k] for k in keys], dtype=np.float64)}


def _detailed_rows(table: np.ndarray) -> dict:
    counties, bands, sexes = np.indices(table.shape)
    return {"county": counties.ravel().astype(np.int64),
            "age_band": np.asarray(AGE_BAND_LABELS)[bands.ravel()],
            "sex": np.asarray(SEX_LABELS)[sexes.ravel()],
            "count": table.ravel().astype(np.int64)}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_packet(seed: int, out_dir: Path, params: PacketParams = PacketParams(),
                 development: bool = False) -> dict:
    """Write one packet and return its manifest."""
    out_dir = Path(out_dir)
    if out_dir.exists():
        raise FileExistsError(f"packet directory already exists: {out_dir}")
    built = build_world(seed, params)
    admin, ticks = built["admin"], built["ticks"]
    participant = out_dir / "participant"
    retained = out_dir / "retained"
    (participant / "sources").mkdir(parents=True)
    retained.mkdir(parents=True)

    # Participant side: surveys, sources, geography, contract.
    for label in ("preliminary", "revised"):
        survey = _survey_at(built, ticks[label])
        _write_table(participant / f"survey_{label}.csv", survey["survey"], forbid_truth=True)
    snapshots = participant_source_snapshots(built["sources"])
    for label, snapshot in snapshots.items():
        for source, table in snapshot.items():
            if source == "snapshot_tick":
                continue
            _write_table(participant / "sources" / f"{source}_{label}.csv", table,
                         forbid_truth=True)
    _write_table(participant / "geography.csv",
                 {"county": np.arange(admin["n_counties"], dtype=np.int64),
                  "state": admin["county_state"].astype(np.int64)}, forbid_truth=True)
    population_revised = snapshots["revised"]["population"]
    age_years = (ticks["revised"] - population_revised["birth_tick"]) // 12
    budget = int(round(params.budget_fraction * int((age_years >= 65).sum())))
    contract = {
        "schema": "meridia.packet.v0",
        "estimands": [asdict(e) for e in ESTIMANDS],
        "levels": list(LEVELS),
        "n_states": int(admin["n_states"]),
        "n_counties": int(admin["n_counties"]),
        "interval_level": 0.90,
        "ticks": {"preliminary": ticks["preliminary"], "revised": ticks["revised"],
                  "horizon": ticks["horizon"]},
        "months_per_tick": 1,
        "disclosure_threshold": params.disclosure_threshold,
        "allocation": {"demand": DEMAND_ESTIMAND, "level": "county", "budget": budget},
        "development": development,
    }
    (participant / "contract.json").write_text(json.dumps(contract, indent=1, sort_keys=True) + "\n")

    # Retained side: exact truth now and at the horizon, the detailed table, the
    # source package's sealed evidence, and the world's character draw.
    truth_revised, detailed = _truth_at(built, ticks["revised"])
    future = project_truth_from_history(built["history"], admin, ticks["horizon"])
    _write_table(retained / "truth_revised.csv", _truth_rows(truth_revised), forbid_truth=False)
    _write_table(retained / "truth_horizon.csv", _truth_rows(future["truth"]), forbid_truth=False)
    _write_table(retained / "detailed_revised.csv", _detailed_rows(detailed), forbid_truth=False)
    sealed = {k: v for k, v in built["sources"].items() if k != "public_snapshots"}
    np.savez_compressed(retained / "source_evidence.npz", **_flatten(sealed))
    (retained / "world.json").write_text(json.dumps({
        "seed": seed, "character": built["character"]["draw"], "shocks": built["shocks"],
        "ticks": ticks, "params": asdict(params)}, indent=1, sort_keys=True, default=str) + "\n")
    if development:
        (participant / "truth").mkdir()
        for name in ("truth_revised.csv", "truth_horizon.csv", "detailed_revised.csv"):
            (participant / "truth" / name).write_bytes((retained / name).read_bytes())

    manifest = {"schema": "meridia.packet.manifest.v0", "development": development,
                "participant": {}, "retained": {}}
    for side in ("participant", "retained"):
        for path in sorted((out_dir / side).rglob("*")):
            if path.is_file():
                manifest[side][str(path.relative_to(out_dir / side))] = {
                    "sha256": _sha256(path), "bytes": path.stat().st_size}
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")
    return manifest


def _flatten(tree: dict, prefix: str = "") -> dict:
    flat = {}
    for key, value in tree.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            flat.update(_flatten(value, name + "/"))
        elif isinstance(value, (np.ndarray, np.generic, int, float, bool)):
            flat[name] = np.asarray(value)
        elif isinstance(value, (list, tuple)):
            flat[name] = np.asarray(value)
    return flat


def participant_columns(out_dir: Path) -> dict[str, list[str]]:
    """Header row of every participant CSV, for leakage checks."""
    columns = {}
    for path in sorted((Path(out_dir) / "participant").rglob("*.csv")):
        with open(path) as handle:
            columns[str(path.relative_to(Path(out_dir) / "participant"))] = \
                handle.readline().strip().split(",")
    return columns
