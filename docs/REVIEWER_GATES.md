# What an independent reviewer can check, and where

Two failure patterns recur in computational science tasks built on synthetic data:
verifiers co-designed with the oracle, which fail valid methods the author did not
anticipate, and authors who use privileged knowledge of the generative function to build
an oracle nobody else could build. This document lists, for each pattern, the mechanism in
this repository that closes it and the artifact a reviewer can read to see that the
mechanism ran. Every line names a file.

## Pattern 1: the verifier must not know the method

- The verifier reads three output files and the retained truth, nothing else
  (`meridia/verify.py`). It recomputes error, coverage, the interval score, additivity,
  projection error, and reserve skill from the submitted numbers. There is no method
  inspection, no expected intermediate, no required library.
- Three reference lines from different statistical philosophies clear the same bars: a
  design-based line (`meridia/methods/design_based.py`: deduplication, nonresponse
  adjustment, raking, synthetic small-area estimation, bootstrap), a Bayesian line
  (`meridia/methods/bayesian.py`: grid posterior on coverage, hierarchical county incomes,
  posterior projection), and a third with its own elder exposure, linkage, and mortality
  choices (`meridia/methods/third_reference.py`). Their county estimates disagree while all
  three pass, which is what a method-open bar looks like.
- Each bar is frozen from the worst of the three lines, never the best
  (`scripts/freeze_v4_bars.py`, `bars/national-v14-standard/PROVENANCE.md`). A method that
  is merely different from all three has room by construction.
- Every restriction a participant must satisfy is in the instruction and the contract, not
  discovered at grading. `docs/SUBMISSION_FORMAT.md` fixes the file set, the headers, and
  the column rules; `docs/history/RELEASE_CONTRACT_V0.md` carries the estimand definitions
  and the release schema those files still use.
- The gates were red-teamed with five attacks. One succeeded: a fifth file of published
  totals the verifier ignored, which let a protected cell be recovered. The verifier now
  fails closed on the file set and audits every published total as a linear constraint, and
  the fix carries a test in `tests/test_methods.py`. The attack log and its resolution are
  authoring evidence held with the packaging repository, not here.

## Pattern 2: the author must not have privileged knowledge

- Every reference line runs on a directory that holds only the participant side of a
  packet; the tests copy `participant/` alone into a fresh directory and run the method
  there (`tests/test_methods.py`, `tests/test_methods_b_and_controls.py`). Nothing on the
  retained side is importable from a method.
- What is public is exactly what a participant gets: the mechanism families and the
  parameter ranges of a world's character (`meridia/character.py`, `meridia/sources.py`,
  `meridia/survey.py`) and twelve development worlds shipped with their truth. What is
  sealed is a world's realized draw. The reference lines use the public ranges only as
  bounds and priors and estimate the rest from the records: mortality from record
  disappearance by age between the two snapshots, coverage from the survey, income
  nonresponse from the development worlds, which any participant can also do.
- The proof that the oracle is not privileged is that it failed. Bars frozen on one hidden
  world did not hold on a second; the strong lines missed a five-year elders projection
  because they took mortality from the public range instead of estimating it. A fresh third
  world exposed a second failure, a one-county state with few sampling units, where the
  lines trusted a noisy survey ratio. Each failure was fixed by a general statistical
  improvement, never by moving a bar or reading the generator, and each fix is a commit
  with a test in this repository's history.
- The shipping bars are qualified on six qualification worlds, `qualification_worlds` in
  `bars/national-v14-standard/bars.json`, and three graded worlds, `graded_world_count` in
  the same file, are sealed and were never seen by the freeze. The bars do not change after
  a graded world is read. If a reference line fails there, the protocol is stop and report.
- The version-four battery is the twenty-two registered controls in
  `meridia/methods/controls.py`, the count carried as
  `control_support.registered_controls` in `bars/national-v14-standard/bars.json`. Each is
  a plausible shortcut with a registered primary gate. Under the shipping profile the tail
  block is the only deciding block a registered wrong method fails on every qualification
  world. The other three deciding blocks are registered validity gates: they reject an
  empty or broken submission and, at six worlds, do not separate the registered wrong
  methods from the reference. `bars/national-v14-standard/freeze_report.txt` records, per
  deciding block, which controls separate and which do not.

## Why the difficulty is structural

Only the final tables are scored and the stages compound: identity resolution sets
coverage, coverage sets every county count, counts set the projection, the projection sets
a committed allocation whose loss is realized when the sealed world runs forward under its
own shocks. The world has dynamics, weather, societies with different inequality and age
structure, and a monthly institutional ledger. The difficulty is structural rather than a
hidden convention, and `docs/DESIGN.md` states how much of it the current bars actually
measure.
