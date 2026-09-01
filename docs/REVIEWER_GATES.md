# How this benchmark answers the reviewer's two patterns

A Terminal-Bench Science release reviewer named two failure patterns in computational
science tasks: verifiers co-designed with the oracle that fail valid methods the author
did not anticipate, and authors of synthetic tasks using privileged knowledge of the
generative function to build oracle solutions nobody else could build. This document
lists, for each pattern, the mechanism in this repository that closes it and the
artifact that proves the mechanism ran. Every line names a file or a commit.

## Pattern 1: the verifier must not know the method

- The verifier reads four output files and the retained truth, nothing else
  (`meridia/verify.py`). It recomputes error, coverage, the interval score, additivity,
  disclosure recoverability, projection error, and allocation regret from the submitted
  numbers. There is no method inspection, no expected intermediate, no required library.
- Two strong methods from different statistical philosophies both pass the same bars:
  a design-based line (`meridia/methods/design_based.py`: deduplication, nonresponse
  adjustment, raking, synthetic small-area estimation, bootstrap) and a Bayesian line
  (`meridia/methods/bayesian.py`: grid posterior on coverage, hierarchical county
  incomes, posterior projection). Their county estimates disagree by up to twenty
  percent on thin counties and both pass, which is what a method-open bar looks like.
- Bars are frozen from the worse of the two methods with a margin, never from the
  better (`scripts/freeze_bars.py`, `bars/`). A method that is merely different from
  both reference lines has room by construction.
- Every restriction a participant must satisfy is in the instruction and the contract
  (`docs/RELEASE_CONTRACT_V0.md`), not discovered at grading: file set, schema, interval
  level, disclosure threshold, budget.
- An independent agent red-teamed the gates with five attacks. One succeeded: a fifth
  file of published totals the verifier ignored, which let a protected cell be
  recovered. The verifier now fails closed on the file set and audits every published
  total as a linear constraint (commit 1555ea1, test in `tests/test_methods.py`). The
  red-team report ships as authoring evidence.

## Pattern 2: the author must not have privileged knowledge

- Both strong methods run on a directory that holds only the participant side of a
  packet; the tests copy `participant/` alone into a fresh directory and run the method
  there (`tests/test_methods.py`, `tests/test_methods_b_and_controls.py`). Nothing on
  the retained side is importable from a method.
- What is public is exactly what a participant gets: the mechanism families and the
  parameter ranges of a world's character (`meridia/character.py`, `meridia/sources.py`,
  `meridia/survey.py`) and three development worlds shipped with their truth. What is
  sealed is a world's realized draw. The strong methods use the public ranges only as
  bounds and priors, and estimate the rest from the records: mortality from record
  disappearance by age between the two snapshots, coverage from the survey, income
  nonresponse from the development worlds, which any participant can also do.
- The proof that the oracle is not privileged is that it failed. Bars frozen on one
  hidden world did not hold on a second; both strong methods missed a five-year elders
  projection by up to thirty percent because they took mortality from the public range
  instead of estimating it. A fresh third world exposed a second failure: a one-county
  state with six sampling units, where both methods trusted a noisy survey ratio. Each
  failure was fixed by a general statistical improvement, never by moving a bar or
  reading the generator, and each fix is a commit with a test (eb32a5f, c34f8b6).
- Bars are qualified on three hidden worlds and the task ships a fourth that neither
  the freeze nor the red-team ever saw. The bars do not change after that world is seen.
  If a strong method fails there, the protocol is stop and report.
- Six controls, each a plausible shortcut, must fail a named gate on every qualification
  world: register-only, survey-only, no deduplication, inflated intervals, static
  projection, uniform allocation (`meridia/methods/controls.py`).

## Why this is not the earlier tasks

The two merged tasks were single-stage with their own answer keys: a small-area
estimation equivalence and a cell-suppression problem. Here only the final tables are
scored and the stages compound: identity resolution sets coverage, coverage sets every
county count, counts set the projection, the projection sets a committed allocation
whose loss is realized when the sealed world runs forward under its own shocks. The
world has dynamics, weather, societies with different inequality and age structure, and
a monthly institutional ledger; the difficulty is structural rather than a hidden
convention.

## What remains before the proposal

Frontier trials inside the participant container on the shipped hidden world, with the
measured pass rate reported as is. The pre-registered escalation ladder (degrade the
sources along measured dials; lengthen the horizon; add a second evidence function on the
same world) is climbed one rung per measurement only if trials land above the band, with
both strong methods required to keep passing at every rung.
