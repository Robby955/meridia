# Meridia identity and institutional schema v0

Status: implementation contract for the institutions layer, version 0.

This document defines identity, table, and history rules for Meridia's social and
institutional layers. It sits above the existing geography, weather, population,
microdata, survey, and demography modules. Those modules remain the source of terrain,
cells, persons, households, and dynamics; this layer imports their outputs without
changing them.

The implemented strata governed by this contract are the dwelling stock; enterprises,
establishments, and jobs; hospitals and encounters; the append-only institutional
history; and four imperfect observed sources with sealed crosswalks.

## 1. Two identity domains

Meridia has two deliberately separate identity domains.

### Sealed truth identity

Every real entity in a generated world receives a persistent truth identity. Truth IDs
exist only inside the retained world state and verifier-side crosswalks. They are never
written into a participant-facing survey, source file, published table, or task packet.

A truth identity combines `truth_world_id` with `truth_entity_id`. The world component is
a deterministic `uint64` derived from the world seed and generator
identity. `truth_entity_id` is a `uint64` with an 8-bit entity namespace followed by a
56-bit never-reused sequence number. The composite is globally scoped; the entity ID
alone is scoped to one world.

The v0 namespace codes are frozen: `1` person, `2` household, `3` dwelling, `4`
enterprise, `5` hospital, `6` job, `7` encounter, `8` event, `9` observed-record source,
and `10` establishment. An enterprise is the legal or controlling organization; an
establishment is one physical operating location.

Sequence numbers are allocation order, not row numbers. Initial persons and households
inherit their allocation order from the deterministic microdata snapshot. New entities
take the next unused sequence in their namespace. IDs are never reassigned after death,
closure, demolition, merger, or record correction. Sorting, filtering, and snapshot
materialization may change row positions but cannot change IDs.

Core array indices are import keys only. In particular, `person[17]` and household index
`17` are not persistent identities. An identity map is created when a core snapshot
enters this layer, and all institutional relationships use its truth IDs.

### Observed identity

Source archives and surveys receive observed identifiers generated independently of truth
IDs. An observed identifier may be:

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
metadata = {
    "truth_world_id": uint64,
    "generator_version": int,
    "snapshot_tick": int64,
    "person": {"column": ndarray, ...},
    "n_persons": int,
}
```

Retained metadata includes `truth_world_id`, which is never copied into an observed
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

Every event contains `truth_event_id` (`uint64`), `tick` (`int64`), `recorded_tick`
(`int64`), `entity_type` (`int8`), `truth_entity_id` (`uint64`), `event_type` (`int16`),
`supersedes_event_id` (`uint64`), and `cause_code` (`int16`). These fields identify the
event and its subject, separate effective time from recorded time, name the change and
its mechanism, and link a correction to an earlier event when needed.

Event order is canonical: `(tick, allocation order)`. `truth_event_id` is assigned in
that order and is the persisted tie-break. Replaying the same initial state and ordered
ledger must reconstruct the current-state tables byte for byte. `recorded_tick` never
affects ledger or replay order; it is used only to select the events visible in an
observed-source snapshot. Late reporting is represented by `recorded_tick > tick`;
history is never back-edited.

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

The dwelling schema is `truth_dwelling_id: uint64`, `cell: int64`, `dwelling_type: int8`,
`tenure: int8`, `bedrooms: int8`, `floor_area_m2: float64`, `year_built: int16`,
`assessed_value: float64`, `monthly_rent: float64`, `is_occupied: bool`,
`truth_household_id: uint64`, and `resident_count: int32`. The attributes describe the
physical unit, tenure, value, and rent. The final three fields state whether it is
occupied, identify the occupying household or zero, and give the linked resident count.

The table is generated from the imported household cells, person-to-household mapping,
urbanity field, seed, and declared parameters. Occupied dwellings remain in the same cell
as their household. Vacancies are assigned by an exact largest-remainder allocation of a
declared world-level vacant-stock target over cells in proportion to household counts.

The initial stock obeys these equations with no tolerance:

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
3. `observed_business_source_id` identifies a source record, not a true business. It is
   generated independently for a later imperfect archive and may be duplicated, stale,
   split, merged, or absent. It is never a foreign key in a truth table.

The current-state business layer implements only the first two truth identities and jobs.
It does not generate the observed source ID before event history exists. Waiting for the
ledger lets the archive represent openings, closures, mergers, moves, and reporting lag.

The enterprise schema is `truth_enterprise_id: uint64`,
`headquarters_establishment_id: uint64`, `headquarters_cell: int64`, `industry: int16`,
`legal_form: int8`, `ownership: int8`, `establishment_count: int32`,
`employment_count: int32`, `annual_payroll_cents: int64`, `annual_revenue_cents: int64`,
`opening_year: int16`, `size_class: int8`, and `is_active: bool`. It records the sealed
organization and headquarters, its classification and ownership, its active location and
job counts, exact payroll and revenue sums, earliest opening, derived size, and state.

The establishment schema is `truth_establishment_id: uint64`,
`truth_enterprise_id: uint64`, `cell: int64`, `industry: int16`,
`establishment_role: int8`, `employment_count: int32`, `annual_payroll_cents: int64`,
`annual_revenue_cents: int64`, `floor_area_m2: float64`, `opening_year: int16`, and
`is_active: bool`. It identifies the location and owner, records geography and role, and
holds exact linked-job counts and payroll alongside synthetic revenue and floor space.

The job schema is `truth_job_id: uint64`, `truth_person_id: uint64`,
`truth_establishment_id: uint64`, `occupation: int16`, `employment_type: int8`,
`annual_hours: int32`, `hourly_wage_cents: int64`, `annual_earnings_cents: int64`,
`start_year: int16`, and `is_active: bool`. It links one worker to one physical workplace,
classifies the work, and records hours, wage, exact earnings, start year, and state.

V0 assigns at most one active job to a person. Multiple-job holding arrives through event
history without changing the identity model. Business generation follows population
geography: establishments are anchored to employed residents' cells with an urbanity
effect, and workers in cells without a workplace are assigned to the nearest workplace
cell under a deterministic grid traversal.

The default generator consumes four truth-side values from the world's character draw.
`jobs_per_adult` sets the exact world-level working-age employment count;
`establishment_size_alpha` sets the Pareto density exponent used to allocate jobs among
locations;
`multi_establishment_rate` sets the number of enterprises operating more than one
location; and `payroll_level` scales the wage schedule before integer-cent earnings are
formed. The values and their public ranges come from `meridia.character`, while a sealed
world's realized draw remains in truth-side state only. These are structural inputs: the
validator recomputes the employment, establishment, and multi-location counts from the
stored parameter record, and tests intervene on each dial while holding the source world
fixed.

Business conservation is exact:

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

## 7. Hospital and encounter current-state tables

A hospital is a persistent institutional identity layered over one active health-sector
establishment. The establishment remains the workplace and payroll unit; the hospital
adds care capacity and patient relationships. One establishment can carry at most one
hospital in v0, and every hospital location is therefore an existing populated workplace
cell. Candidate locations are scored from local population, urban accessibility, and
existing staff. Every cell is assigned to its nearest selected hospital with stable
Chebyshev-distance ties, producing exact population catchments.

The hospital schema is `truth_hospital_id: uint64`, `truth_establishment_id: uint64`,
`cell: int64`, `hospital_type: int8`, `ownership: int8`, `bed_count: int32`,
`staffed_position_count: int32`, `occupied_bed_count: int32`,
`catchment_population: int32`, `opening_year: int16`, and `is_active: bool`. It binds the
facility to its workplace and cell, classifies capacity and ownership, and records exact
bed, staffing, occupancy, and catchment counts.

The `staffing` relationship has one row for every active job at a selected hospital. Its
fields are `truth_hospital_id: uint64`, `truth_job_id: uint64`, and `staff_role: int8`.
The role codes distinguish support, technical, nursing or allied, and clinical work.

The encounter schema is `truth_encounter_id: uint64`, `truth_person_id: uint64`,
`truth_hospital_id: uint64`, `admission_tick: int64`, `discharge_tick: int64`,
`service: int8`, `diagnosis_group: int16`, `outcome: int8`, `cost_cents: int64`,
`bed_number: int32`, and `is_open: bool`. It links patient and facility, records the
interval and clinical categories, and retains accrued cost and current bed state.

`hospital_beds_per_1000` comes from the truth-side world-character draw and fixes the
world-level integer bed total. Beds are allocated by an exact largest-remainder rule over
facility catchment population and staffing. The realized draw is never copied into an
observed health source. Observed facility, staff, and encounter identifiers arrive only
after event history exists and are never derived from these truth IDs.

Hospital conservation is exact:

```
sum(hospital catchment_population) = persons
sum(hospital bed_count) = round(persons * hospital_beds_per_1000 / 1000)
hospital staffed_position_count = count(linked active health-sector jobs)
each staffing job belongs to that hospital's establishment
hospital occupied_bed_count = count(linked open encounters)
sum(hospital occupied_bed_count) = open encounters
each open encounter -> one unique person and one unique (hospital, bed_number)
each encounter -> one existing person and that person's nearest hospital
```

All required truth foreign keys remain within one `truth_world_id`. Encounter intervals
straddle the snapshot exactly when open; completed encounters discharge no later than the
snapshot and retain no current bed number. Later event modules will append openings,
closures, staffing changes, admissions, discharges, and corrections without changing
these identities.

## 8. Monthly institutional event history

`meridia.events` advances the truth world in monthly ticks without rewriting any prior
row. It consumes the persistent identity, dwelling, business, hospital, world-character,
and declared shock layers. It does not consume an employer file or any downstream
observed source. The ledger covers:

- births and deaths;
- household formation, relocation, and closure;
- job starts and ends;
- establishment openings and closures;
- encounter admissions and discharges.

The module retains an exact initial operational state, the canonical event ledger, and a
materialized terminal state. Replaying the initial state through any effective tick
reconstructs the corresponding state. A longer run can only append: for the same world,
the event bytes through month `m` are identical whether the requested horizon is `m` or
longer.

`tick` and `recorded_tick` are intentionally different clocks. `tick` is the effective
month in sealed truth. `recorded_tick` is when a later source is allowed to observe the
event and is always at least `tick`. Preliminary and revised source snapshots are cuts on
`recorded_tick`; truth replay remains a cut on `tick`. Late reporting therefore
creates a real cross-snapshot discrepancy without changing the underlying event.

The replayed operational state includes person-to-household residence, household-to-
dwelling occupancy, establishment activity, job relationships and integer-cent earnings,
and encounter occupancy. At every replayed date:

```
living persons = initial persons + births - deaths
sum(dwelling resident_count) = living persons
active households = occupied dwellings
each living person -> one active household -> one occupied dwelling
each active job -> one living person and one active establishment
job annual_earnings_cents = annual_hours * hourly_wage_cents
each active establishment has at least one active job
each open encounter -> one living person and one unique hospital bed
open bed_number < hospital bed_count
```

New person, household, establishment, job, encounter, and event IDs take the next unused
sequence in their namespace. Closed or dead entities remain in state with their identity;
none is reassigned. Hospital establishments are excluded from v0 establishment closure.
The declared shock schedule can alter mortality, fertility, and household-formation
mechanisms, and its realized schedule remains truth-side metadata.

## 9. Imperfect observed sources

`meridia.sources` materializes four source-distinct, flat files at preliminary and revised
snapshots. The public half of each snapshot contains exactly `snapshot_tick`,
`population`, `business`, `income`, and `health`. Observed record and entity identifiers
remain stable across snapshots when the underlying source record persists. Each snapshot
applies exactly the ledger rows satisfying `recorded_tick <= snapshot_tick`; it is never
created by perturbing terminal truth. Effective truth at the same date comes from the
independent `tick` replay, allowing the retained evidence to identify stale reporting.

All observed IDs are random `uint64` tokens in source-specific namespaces disjoint from
the truth-ID namespaces. The random draw receives a source and row count, never a truth
ID. It is therefore not a hash, encoding, prefix, suffix, or arithmetic transform of a
truth ID. Names are represented by two synthetic tokens, `given_code` and
`family_code`, rather than real names. Each truth person has one true pair drawn from
finite vocabularies (1500 given names, 8000 family names) under a Zipf frequency law
with exponent 0.9, so distinct persons share a pair at a rate the development worlds
reveal only in aggregate. Every person source re-reports the pair, the birth tick, and
the sex with its own error process at public constant rates: another given name on file
(0.040), the family name's second spelling (0.040), given and family entered swapped
(0.010), given name missing and reported as `0` (0.015), birth month off by one to three
(0.040), birth month rounded to the year (0.030), birth year off by one (0.010), and sex
miscoded (0.006). A linkage error reports the name pair of another person in the same
county. The second record of a duplicate draws its own errors, so duplicates are
near-duplicates; the second record of a split carries the other spelling of the family
name; the second member of a merge pair reports the first member's name. Each source
records the address at its own reference date: the population source at the snapshot,
the income source twelve ticks earlier, the health source at admission. Error-prone
names, birth ticks, sex, household IDs, employer IDs, and dated county codes create a
probabilistic linkage problem; no perfect cross-source person key is shipped.

Population fields are `record_id: uint64`, `person_id: uint64`, `household_id: uint64`,
`given_code: uint64`, `family_code: uint64`, `birth_tick: int64`, `sex: int8`,
`education: int8`, and `county: int32`. They provide imperfect person and household
identities, synthetic linkage fields, and reported demographics. Missing education
uses `-1`.

Business fields are `record_id: uint64`, `business_id: uint64`, `enterprise_id: uint64`,
`industry: int16`, `county: int32`, `employee_count: int32`, and
`annual_payroll_cents: float64`. Counts and payroll reflect only changes visible to that
source; missing payroll uses `NaN`.

Income fields are `record_id: uint64`, `taxpayer_id: uint64`, `household_id: uint64`,
`given_code: uint64`, `family_code: uint64`, `birth_tick: int64`, `sex: int8`,
`county: int32`, `employment_income_cents: float64`, and `employer_id: uint64`. A planted
mechanism can make the employer identifier wrong or zero, while missing income uses
`NaN`. Earnings and business payroll are reported in the register's own wage unit: truth
cents times the world's `register_income_scale`, a per-world draw (section 9a).

Health fields are `record_id: uint64`, `encounter_id: uint64`, `patient_id: uint64`,
`facility_id: uint64`, `given_code: uint64`, `family_code: uint64`, `birth_tick: int64`, `sex: int8`,
`patient_county: int32`, `facility_county: int32`, `admission_tick: int64`,
`discharge_tick: int64`, `service: int8`, `diagnosis_group: int16`, `outcome: int8`, and
`cost_cents: float64`. Open encounters use `-1` for discharge, and missing cost uses
`NaN`.

Every public county value originates from the authoritative administrative partition:

```
county = admin["county"].reshape(-1)[recorded_cell]
```

The source layer never derives or repartitions counties. A planted county-error
mechanism may replace that value by another valid code; the retained crosswalk identifies
the affected row. Facility county is not corrupted in v0.

The sealed mechanism table has one row per possible truth entity and records `covered`,
`duplicate`, `split`, `merge_group`, `county_error`, `linkage_error`, `item_missing`,
`birth_error`, and `name_error` (the last two for the primary record; the crosswalk
carries the per-record bits, plus `address_lag` where a source's dated address differs
from the snapshot address).
Coverage is lower by a declared penalty in outpost counties, creating structural thin-
county undercoverage. Split entities necessarily receive a second record and observed
ID; merge groups contain exactly two truth entities. Reporting staleness is not sampled:
it is measured by comparing the `recorded_tick` view with truth replay at the same
effective tick.

### 9a. Per-world mechanism draw and the hidden shift family

Four mechanism quantities are one draw per world from `draw_source_params` on its own
seed sequence key, so geography and society draws and the sealed digests are unchanged.
Development worlds draw from the development band: population coverage in
(0.940, 0.985), health coverage in (0.900, 0.950), county miscoding rate in
(0.012, 0.024), register income scale in (0.94, 1.06). The hidden world draws from the
published shift family, which lies outside that band: population coverage below the
band's low edge by 0.02 to 0.08; health coverage below it by 0.06 to 0.20; county
miscoding rate at 1.5 to 3.0 times the band's high edge; the effective register wage
level (payroll level times register income scale) in (0.50, 0.63) or in (1.52, 1.90),
side by a fair coin, against a development band of (0.705, 1.378). The realized values
live only in `retained/world.json` and the retained source evidence.

### 9b. Benchmark totals

`participant/sources/benchmark_{preliminary,revised}.csv` (`item, level, unit, value`)
carries the four counts at nation and state level from a separately produced series.
Each value is the exact count at the snapshot times `exp(b)`: at nation level `|b|` is
uniform in (0.02, 0.07) with a fair-coin sign per item; at state level `b` is normal
with a world-specific standard deviation uniform in (0.03, 0.08). The bias is persistent
across vintages and independent across items and units; values are rounded to the
nearest hundred. The draws live only in `retained/world.json`.

Each public row has one sealed crosswalk row. Its schema is
`observed_record_id: uint64`, `observed_entity_id: uint64`, `truth_entity_id: uint64`,
`mechanism_code: int16`, `valid_from_tick: int64`, and `valid_to_tick: int64`. The final
field is `-1` for a row current at that snapshot.

The crosswalk, mechanism tables, source parameters, truth-world metadata, and event ledger
are retained verifier evidence. `participant_source_snapshots` returns a defensive copy
containing none of them. Validation regenerates the complete retained package from the
seeded inputs and requires byte-equivalent arrays, including `NaN` placement.

## 10. Presentation and independence boundary

The eventual participant packet contains flat observed-record files only, plus an
estimand list, published-table schema, and disclosure rules. It contains no truth ID, crosswalk,
mechanism label, world-character draw, full-population statistic, terrain, or generator
code. Event visibility at two source snapshots is determined downstream from
`recorded_tick`; the truth ledger itself is never exported.

Geographic vocabulary is fixed everywhere in this lane:

```
world > state > county > settlement
```

Schemas, tables, docs, and future exports use no alternate or country-specific
administrative terms.

The independence firewall is binding: no employer data, code, material, narratives, or
references are used. All external work is independent research. The engine uses entirely
synthetic countries and public scientific literature; it uses no government data,
branding, or implied endorsement.

## 11. Determinism and versioning

Generation uses only the Python standard library plus NumPy/SciPy already allowed by the
engine. Each additive module receives an explicit seed and uses a module-specific
`SeedSequence` component so adding draws in one module does not perturb another module's
stream. No global random state, clock time, process ID, filesystem order, or network state
may affect an output.

Schema version, generator version, parameters, source table digests, and output table
digests belong in the future world manifest. A schema change increments the schema version.
An algorithm change increments the generator version. Existing canonical timelines remain
replayable; a new version extends or forks them and never rewrites their past.

## 12. Delivery order

1. This identity-and-schema contract.
2. Initial identity mapping and dwelling current-state table with exact tests.
3. Business and hospital current-state tables, then jobs and encounters.
4. Append-only institutional event histories and deterministic replay (implemented).
5. Imperfect observed sources and sealed truth crosswalks (implemented).
6. The capstone production contract consuming observed records only (next).

Sealing protocol, shock-dial implementation, geography, weather, population, microdata,
survey, demography, and rendering remain outside this branch's ownership. This branch imports
their public outputs and does not modify their code.
