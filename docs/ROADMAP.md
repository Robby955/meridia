# Meridia roadmap

Where the project is going, in build order. Everything below follows the same rules as
what exists: plain Python with NumPy and SciPy, seeded determinism end to end, an exact
conservation or identity test for every layer, and retained truth underneath every
observed record.

## Built

- Terrain, rivers (exact runoff accounting), population grid, persons and households, survey
  instrument with planted defects, year-by-year demography with a vital-events register,
  weather with orographic rain and lagged river response.
- World character: each seed draws its society's inequality, demography, wealth
  gradients, urban structure, and institutional densities from declared ranges.
- Administrative geography (states and counties, exact partition) and the release
  contract v0 with its scorer: estimands, schema, additivity, worst-unit accuracy,
  coverage with an interval score, and a linear-recovery disclosure audit.
- Persistent truth identities, dwelling stock, and a business layer of enterprises,
  establishments, and jobs, with payroll conserved to the cent (`meridia/identities.py`,
  `meridia/dwellings.py`, `meridia/businesses.py`).
- Hospitals and encounters: capacity from the world's character draw, patient encounters
  linking people to facilities (`meridia/hospitals.py`).
- Event histories: append-only monthly event tables of births, deaths, moves, job changes,
  and business openings and closures, from which any date's snapshot reconstructs exactly
  (`meridia/events.py`).
- Sealing protocol: evaluation worlds generated at registered seeds and never inspected,
  with hash-sealed manifests (`meridia/sealing.py`, `seals/`).
- Observed registers: imperfect population, business, income, and health registers over the
  retained truth, with duplicates, stale addresses, split and merged identifiers, and
  reporting delays, each by an explicit recorded mechanism (`meridia/sources.py`).
- The reconstruction chain: surveys drawn from imperfect frames, then editing, imputation,
  weighting, estimation, variance, and the scored release, projection, and reserve tables
  (`meridia/methods/`, `meridia/verify.py`).

## Next

1. **Shock dial.** Break years: a mortality spike, a migration wave, an economic break,
   each a parameter change with retained truth, so methods can be tested across
   structural change with a counterfactual that exists. The dial exists in
   `meridia/demography.py`; what is not here is a world family built around it.
2. **Epidemics.** SEIR on household and workplace contact structure rather than well mixed.
3. **Commuting.** A journey-to-work layer over the existing settlement and job structure.
4. **Climate beyond weather.** Latitude temperature structure and seasons above the
   existing orographic precipitation.

## Further out

Multiple planets with independent
geographies and institutions; a persistent universe where sealed worlds serve blinded
evaluation and open worlds serve method research.

## Realism backlog, with the literature each upgrade will cite

Mechanisms below are adopted with their citation when their layer becomes load-bearing:

- Income top tail: lognormal body with a Pareto upper tail, replacing the pure
  lognormal.
- Firm sizes: Zipf's law for firm sizes (Axtell, Science 2001); the establishment-size
  dial already brackets it.
- Migration and commuting: the radiation model (Simini, Gonzalez, Maritan, Barabasi,
  Nature 2012), replacing ad-hoc urbanity pull.
- Mortality over time: Lee-Carter (JASA 1992) once time-varying mortality matters;
  Gompertz-Makeham remains the static curve.
- Climate: latitude temperature structure and seasons above the existing orographic
  precipitation (which already follows the standard uplift mechanism).
- Epidemics: SEIR on household/workplace contact structure, not well-mixed.
- Settlement spacing: central-place structure as an alternative to greedy suppression.
- Terrain hydrology already cites Barnes, Lehman, Mulla (Computers & Geosciences 2014)
  for priority-flood depression filling.

## Standing invariants

- Truth identifiers never appear in any participant-facing file.
- Observed identifiers are never derivable from truth identifiers.
- The parameter ranges of a world's character are public; a sealed world's draw is not.
- A picture of the world is drawn from stored state, or it does not ship.
