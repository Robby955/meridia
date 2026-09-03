# Rubric map: the 39 implementation criteria, the proposal rubric, the trial analysis

The benchmark's review rubric (`rubrics/task-implementation.toml` upstream) has 39 named
criteria; the proposal rubric has seven; the trial-analysis rubric has eight. Each line
below names the criterion, the artifact that satisfies it, and who owns it: S for the
science on Meridia main, P for the packaging in the task directory. Nothing is marked
done without the artifact.

## Implementation criteria

- verifiable: S, `meridia/verify.py` recomputes every gate from the four output files
  and retained truth; deterministic, under a minute.
- well_specified: S and P, `docs/RELEASE_CONTRACT_V0.md` and the instruction name every
  scored file, schema, interval level, threshold, budget.
- solvable: S, two strong methods pass the frozen bars on seven qualification worlds and
  the confirmation world; P, witness receipts from inside the container.
- difficult: S, compounding stages with only final tables scored; measured by the frontier
  trials of `docs/FRONTIER_TRIALS.md`.
- scientifically_grounded: S, `docs/INDEPENDENCE.md` cites every method at its use.
- scope: P, mathematical sciences, statistics.
- outcome_verified: S, the verifier reads outputs only; the instruction describes the end
  state.
- anti_cheat_robustness: S, red-team with five attacks and the fixed disclosure hole;
  strict file set; eight controls that must fail, one of them the count recipe that
  cleared version two.
- task_security: P, no network needed, no credentials, no execution of agent code by the
  verifier beyond reading files.
- functional_verification: S, numbers recomputed, never pattern-matched.
- ctrf_reporting: P, per-gate CTRF report to `/logs/verifier/ctrf.json`.
- ground_truth_provenance: S, retained truth is the generator's exact population; stated
  in the contract and `bars/national-v7/PROVENANCE.md`.
- graded_instances_discriminate: S, both strong methods pass every bar and all eight
  controls fail a named gate on every one of the nine qualification worlds and on the
  sealed world (`bars/national-v7/freeze_report.txt`,
  `seals/meridia-reconstruction-v3-confirmation.md`).
- deterministic_reproducible: S, same seed same world byte for byte; pinned dependencies.
- essential_difficulty: S, no formatting traps; schema is exact but trivial to meet.
- test_instruction_alignment: S and P, contract to verifier one to one; audit item.
- do_not_modify_enforced: P, participant files read-only in the container; the
  verifier uses the retained copy, never the participant copy.
- novel: S, sealed synthetic worlds; nothing memorizable.
- agentic: S, multi-stage work over hundreds of megabytes of files.
- reviewable: S, `docs/REVIEWER_GATES.md`, the chain and gate-matrix figures, plain
  contract text.
- instruction_clarity: P, under 700 words, outcome only, absolute paths, no hints.
- solution_quality: S, both reference solutions compute everything from the files.
- separate_verifier_configured: P, verifier image with Meridia pinned and retained truth
  baked in.
- verifier_execution_isolation: P, the verifier never executes agent code.
- artifact_efficiency: P, only the four output files collected; data baked into images.
- environment_hygiene: P, no tests or solution in the participant image.
- structured_data_schema: S, exact schemas in the contract and `contract.json`.
- typos: P, spellcheck pass on instruction and file names.
- difficulty_explanation_quality: P, from `docs/REVIEWER_GATES.md` section on compounding.
- solution_explanation_quality: P, from the two method docstrings and citations.
- verification_explanation_quality: P, from the verifier docstring and the contract.
- category_and_tags: P.
- task_name: P, `meridia-reconstruction`.
- resource_configuration: P, timeout fixed before trials and never changed between batches.
- task_readme: P, development context for reviewers, including the red-team record.
- task_authoring_dir: P, evidence, freeze reports, red-team, and blockers under `authoring/`.
- expert_time_estimate: P, two to four weeks for one world with the reusable program as the
  substantive part.
- task_toml_schema: P, valid fields only.
- no_extraneous_files: P, audited at sealing.

## Proposal rubric, seven parts

Verifiable, well-specified, solvable, difficult, scientifically grounded and interesting,
scope, outcome-verified. The form draft answers each in its own field; the GitHub
proposal follows the upstream template headings in that order.

## Trial-analysis rubric, eight criteria

A failed frontier trial is attributed under the upstream taxonomy before it is counted:
specification_completeness, solution_discoverability, verifier_correctness,
boundary_fairness, policy_refusal, execution_blocked, unearned_credit,
meaningful_difficulty. Only a failure attributed to meaningful_difficulty counts toward
the difficulty measurement; the others are task defects to fix and rerun once, with both
records kept.

## Author-fit review

The upstream review includes a conflict-of-interest section. Recusal from reviewing own
submissions is stated in the proposal.
