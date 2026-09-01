# Meridia identity and institutional schema v0

Status: implementation contract for `codex/institutions-v0`.

This document defines identity, table, and history rules for Meridia's social and
institutional layers. It sits above the existing geography, weather, population,
microdata, survey, and demography modules. Those modules remain the source of terrain,
cells, persons, households, and dynamics; this layer imports their outputs without
changing them.

The first implementation governed by this contract is the dwelling stock. Businesses,
hospitals, jobs, encounters, and imperfect registers will use the same identity and
history rules in later additive modules.

## 1. Two identity domains

Meridia has two deliberately separate identity domains.

### Sealed truth identity

Every real entity in a generated world receives a persistent truth identity. Truth IDs
exist only inside the retained world state and verifier-side crosswalks. They are never
written into a participant-facing survey, register, release, or task packet.

A truth identity is the composite:

```
(truth_world_id, truth_entity_id)
```

`truth_world_id` is a deterministic `uint64` derived from the world seed and generator
identity. `truth_entity_id` is a `uint64` with an 8-bit entity namespace followed by a
56-bit never-reused sequence number. The composite is globally scoped; the entity ID
alone is scoped to one world.

The v0 namespace codes are frozen:

| Code | Entity |
| ---: | --- |
| 1 | person |
| 2 | household |
| 3 | dwelling |
| 4 | enterprise (the legal/control entity; originally reserved as `business`) |
| 5 | hospital |
| 6 | job |
| 7 | encounter |
| 8 | event |
| 9 | observed-record source |
| 10 | establishment (one physical operating location) |

Sequence numbers are allocation order, not row numbers. Initial persons and households
inherit their allocation order from the deterministic microdata snapshot. New entities
take the next unused sequence in their namespace. IDs are never reassigned after death,
closure, demolition, merger, or record correction. Sorting, filtering, and snapshot
materialization may change row positions but cannot change IDs.

Core array indices are import keys only. In particular, `person[17]` and household index
`17` are not persistent identities. An identity map is created when a core snapshot
enters this layer, and all institutional relationships use its truth IDs.

### Observed identity

Registers and surveys receive source-specific observed identifiers generated independently
of truth IDs. An observed identifier may be:

- missing;
- stale after a move or status change;
- duplicated across two observed records;
- split, so one truth entity has multiple observed IDs;
- merged, so records for different truth entities share an observed ID;
- mistyped or transposed according to a recorded error mechanism.

Observed IDs are not hashes, encodings, prefixes, suffixes, or arithmetic transforms of
truth IDs. A verifier-side crosswalk records the source, observed record ID, truth entity
ID, mechanism, and validity interval. The crosswalk is never exported. Participant files
may contain the observed ID attached to its observed record, but never the corresponding
truth ID or hidden mechanism annotation.

Participant-facing schema checks must reject any column named `truth_*`, any truth-world
identifier, and any hidden crosswalk or mechanism field not explicitly declared visible.

## 2. Columnar table convention

Engine tables are dictionaries of one-dimensional NumPy arrays. Every column in a table
has the same row count. Tables carry scalar metadata outside the column dictionary:

```
{
    "truth_world_id": uint64,
    "generator_version": int,
    "snapshot_tick": int64,
    "<table_name>": {"column": ndarray, ...},
    "n_<entities>": int,
}
```

The `truth_world_id` is retained truth metadata and is never copied into an observed
table. Integer codes are used for finite categories, with codebooks declared beside the
module. Required foreign keys are `uint64`. Nullable truth foreign keys use `0` only
when a separate boolean state column makes the absence explicit; no allocated truth ID
is zero. Time is an integer world tick. Floats are `float64`, categorical codes are
`int8` or `int16`, counts are sized to their supported range, and grid cells use `int64`
flat indices compatible with the core modules.

Current-state tables are materialized views. They are replaceable outputs of a
deterministic replay, not the historical authority.

## 3. Append-only history

Once event modules land, institutional changes are recorded in append-only event tables.
Existing event rows are never mutated or deleted. Corrections append a new event that
supersedes an earlier event by ID and gives the reason.

Every event will contain at least:

| Column | Type | Meaning |
| --- | --- | --- |
| `truth_event_id` | `uint64` | persistent event identity |
| `tick` | `int64` | effective world tick |
| `recorded_tick` | `int64` | tick at which the event entered the ledger |
| `entity_type` | `int8` | namespace code of the subject |
| `truth_entity_id` | `uint64` | subject identity |
| `event_type` | `int16` | module-specific event code |
| `supersedes_event_id` | `uint64` | zero or an earlier event ID |
| `cause_code` | `int16` | explicit generative mechanism |

Event order is canonical: `(tick, recorded_tick, truth_event_id)`. Replaying the same
initial state and ordered ledger must reconstruct the current-state tables byte for byte.
Late reporting is represented by `recorded_tick > tick`; history is never back-edited.

## 4. Initial identity snapshot

For the initial microdata snapshot:

- there is exactly one truth person ID for each person row;
- there is exactly one truth household ID for each household index;
- every person's household import key resolves to exactly one truth household ID;
- the composite IDs are deterministic in the world seed and core snapshot counts;
- person and household namespaces are disjoint by construction.

Births, household formation, and future migrations will be integrated by an additive
identity-aware dynamics wrapper. It will retain survivor IDs and allocate new IDs for
births and newly formed households rather than treating post-filter array positions as
identity.

## 5. Dwelling current-state table v0

The initial dwelling stock has one occupied dwelling for every household and an explicit
vacant stock allocated across inhabited cells. V0 does not yet model multiple households
sharing one dwelling or institutional residences. Those are future event/schema changes,
not silent reinterpretations of v0 rows.

Required columns:

| Column | Type | Meaning |
| --- | --- | --- |
| `truth_dwelling_id` | `uint64` | persistent sealed dwelling identity |
| `cell` | `int64` | flat geography-cell index |
| `dwelling_type` | `int8` | detached, attached, low-rise, or high-rise |
| `tenure` | `int8` | owner, mortgage, private rent, social rent, or vacant |
| `bedrooms` | `int8` | bedroom count |
| `floor_area_m2` | `float64` | current usable floor area |
| `year_built` | `int16` | construction year in the synthetic calendar |
| `assessed_value` | `float64` | synthetic current value |
| `monthly_rent` | `float64` | zero for non-rental and vacant units |
| `is_occupied` | `bool` | whether a household occupies the dwelling |
| `truth_household_id` | `uint64` | occupying truth household, or zero if vacant |
| `resident_count` | `int32` | residents linked through the occupying household |

The table is generated from the imported household cells, person-to-household mapping,
urbanity field, seed, and declared parameters. Occupied dwellings remain in the same cell
as their household. Vacancies are assigned by an exact largest-remainder allocation of a
declared national vacant-stock target over cells in proportion to household counts.

The initial stock must satisfy all of these identities exactly:

```
occupied_dwellings = households
occupied_dwellings + vacant_dwellings = dwellings
sum(resident_count) = persons
resident_count[dwelling_of(h)] = household_size[h]  for every household h
cell[dwelling_of(h)] = household_cell[h]            for every household h
unique(nonzero truth_household_id) = households
unique(truth_dwelling_id) = dwellings
```

No tolerance is permitted for these checks. Economic and physical attributes may be
stochastic, but the seeded replay must be byte-identical.

## 6. Business identities and current-state tables

Business data has three identities that must never be collapsed:

1. `truth_enterprise_id` identifies the sealed legal or controlling organization. One
   enterprise may operate multiple establishments.
2. `truth_establishment_id` identifies one sealed physical operating location. Jobs link
   to establishments because employment, payroll, production, and geography occur there.
3. `observed_business_register_id` identifies a source record, not a true business. It is
   generated independently for a later imperfect register and may be duplicated, stale,
   split, merged, or absent. It is never a foreign key in a truth table.

The current-state business layer implements only the first two truth identities and jobs.
It does not generate the observed register ID before event history exists: otherwise the
register could not represent openings, closures, mergers, moves, or reporting lag honestly.

The `enterprise` table contains:

| Column | Type | Meaning |
| --- | --- | --- |
| `truth_enterprise_id` | `uint64` | persistent sealed enterprise identity |
| `headquarters_establishment_id` | `uint64` | one establishment owned by the enterprise |
| `headquarters_cell` | `int64` | headquarters geography cell |
| `industry` | `int16` | synthetic industry section |
| `legal_form` | `int8` | sole proprietor, partnership, corporation, cooperative, or public |
| `ownership` | `int8` | domestic private, foreign private, cooperative, or public |
| `establishment_count` | `int32` | active physical locations |
| `employment_count` | `int32` | active jobs across those locations |
| `annual_payroll_cents` | `int64` | exact sum of establishment payroll |
| `annual_revenue_cents` | `int64` | exact sum of establishment revenue |
| `opening_year` | `int16` | earliest opening among current establishments |
| `size_class` | `int8` | class derived from employment count |
| `is_active` | `bool` | current enterprise state |

The `establishment` table contains:

| Column | Type | Meaning |
| --- | --- | --- |
| `truth_establishment_id` | `uint64` | persistent sealed location identity |
| `truth_enterprise_id` | `uint64` | owning enterprise |
| `cell` | `int64` | physical operating cell |
| `industry` | `int16` | inherited enterprise industry in v0 |
| `establishment_role` | `int8` | headquarters or branch |
| `employment_count` | `int32` | jobs linked to this location |
| `annual_payroll_cents` | `int64` | exact sum of linked job earnings |
| `annual_revenue_cents` | `int64` | synthetic location revenue, not below payroll |
| `floor_area_m2` | `float64` | operating floor area |
| `opening_year` | `int16` | opening year in the synthetic calendar |
| `is_active` | `bool` | current establishment state |

The `job` table contains:

| Column | Type | Meaning |
| --- | --- | --- |
| `truth_job_id` | `uint64` | persistent sealed job identity |
| `truth_person_id` | `uint64` | worker identity |
| `truth_establishment_id` | `uint64` | physical workplace identity |
| `occupation` | `int16` | synthetic occupation group |
| `employment_type` | `int8` | full-time or part-time |
| `annual_hours` | `int32` | paid annual hours |
| `hourly_wage_cents` | `int64` | integer hourly wage |
| `annual_earnings_cents` | `int64` | exactly hours times hourly wage |
| `start_year` | `int16` | start year in the synthetic calendar |
| `is_active` | `bool` | current job state |

V0 assigns at most one active job to a person. Multiple-job holding arrives through event
history without changing the identity model. Business generation follows population
geography: establishments are anchored to employed residents' cells with an urbanity
effect, and workers in cells without a workplace are assigned to the nearest workplace
cell under a deterministic grid traversal.

The default generator consumes four truth-side values from the world's character draw.
`jobs_per_adult` sets the exact national working-age employment count;
`establishment_size_alpha` sets the Pareto density exponent used to allocate jobs among
locations;
`multi_establishment_rate` sets the number of enterprises operating more than one
location; and `payroll_level` scales the wage schedule before integer-cent earnings are
formed. The values and their public ranges come from `meridia.character`, while a sealed
world's realized draw remains in truth-side state only. These are structural inputs: the
validator recomputes the employment, establishment, and multi-location counts from the
stored parameter record, and tests intervene on each dial while holding the source world
fixed.

The current state must satisfy these identities exactly:

```
each job -> one existing person and one existing establishment
job annual_earnings_cents = annual_hours * hourly_wage_cents
establishment employment_count = count(linked active jobs)
establishment annual_payroll_cents = sum(linked job annual_earnings_cents)
enterprise establishment_count = count(linked active establishments)
enterprise employment_count = sum(linked establishment employment_count)
enterprise annual_payroll_cents = sum(linked establishment annual_payroll_cents)
enterprise annual_revenue_cents = sum(linked establishment annual_revenue_cents)
```

All sums use integer counts or cents. No floating tolerance is admitted. Initial active
establishments each carry at least one job, every headquarters belongs to its enterprise,
and all truth foreign keys remain inside one `truth_world_id`.

## 7. Reserved current-state relationships

Later modules will add the following tables without changing the dwelling contract:

- `hospital`: location, capacity, services, staffing, ownership, and operational state;
- `encounter`: person-hospital relationship, admission and discharge ticks, service,
  diagnosis group, outcome, and cost.

Required truth foreign keys always refer to entities in the same `truth_world_id`. A job
cannot reference a closed business at its start tick; an encounter cannot reference a
hospital before opening; a household move changes its dwelling relationship through an
event and does not change either persistent identity.

Each layer must introduce at least one exact identity or conservation test. Planned examples
are payroll equals the sum of active job earnings, staffed hospital positions reconcile to
health-sector jobs, occupied beds reconcile to open encounters, and register error counts
reconcile exactly to their recorded mechanisms.

## 8. Determinism and versioning

Generation uses only the Python standard library plus NumPy/SciPy already allowed by the
engine. Each additive module receives an explicit seed and uses a module-specific
`SeedSequence` component so adding draws in one module does not perturb another module's
stream. No global random state, clock time, process ID, filesystem order, or network state
may affect an output.

Schema version, generator version, parameters, source table digests, and output table
digests belong in the future world manifest. A schema change increments the schema version.
An algorithm change increments the generator version. Existing canonical timelines remain
replayable; a new version extends or forks them and never rewrites their past.

## 9. Delivery order

1. This identity-and-schema contract.
2. Initial identity mapping and dwelling current-state table with exact tests.
3. Business and hospital current-state tables, then jobs and encounters.
4. Append-only institutional event histories and deterministic replay.
5. Imperfect source registers and sealed truth crosswalks.
6. The capstone production contract consuming observed records only.

Sealing protocol, shock-dial implementation, geography, weather, population, microdata,
survey, demography, and rendering remain outside this branch's ownership. This branch imports
their public outputs and does not modify their code.
