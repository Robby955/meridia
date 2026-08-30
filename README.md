# Meridia

A seeded synthetic planet serving as the shared world for a family of verification-grade
statistical environments: one population, one survey program, retained truth at every
layer. Design documents (the source of authority for scope and constraints):

- `theorempath/HQ/STATPROD_ONE_WORLD_DESIGN_2026-08-31.md` (the world: layers, geography,
  time, engine decisions, demonstration strategy)
- `theorempath/HQ/STATPROD_TASK_FAMILY_DESIGN_2026-08-30.md` (the task family, gates,
  oracle principle)
- `theorempath/HQ/STATPROD_STAGE_PLAYBOOK_2026-08-31.md` (per-stage generation,
  presentation, oracle, advanced bars)

Engine rules: Python with NumPy and SciPy only; seeded and deterministic end to end;
conservation laws hold exactly and are tested; versions freeze with manifests and
byte-identity regeneration; no third-party simulators; synthetic data and public
methodology only.

Status: layers `terrain` (spectral elevation with ridges and a coast) and `hydrology`
(priority-flood depression filling, D8 directions, flow accumulation with exact
conservation) implemented and tested. Next, in order: climate, weather, chemistry,
populations, program.

Run tests: `python3 -m pytest tests/ -q`
