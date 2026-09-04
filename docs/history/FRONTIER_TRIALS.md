# Frontier trial protocol, registered before any trial

Written 2026-09-01, before the task is sealed and before any frontier agent has seen a
packet. Nothing here changes after the first trial starts; a change is a new protocol
with a new date.

## What is measured

The pass rate of frontier agents on the sealed hidden world, under the participant
container, against the frozen bars of the current `bars/national-vN`. A trial passes only
if the verifier returns reward 1: every gate on every scored file. Partial credit does
not exist.

## Who, how many, how much

- Systems: the strongest solvers available through the participant container at trial
  time, named with their exact identifiers in the sealed prediction; three trials each in
  the first batch.
- A per-trial cost ceiling and a batch cap are registered with the packaging repository,
  which is where the trials are run and paid for. A trial that exceeds its ceiling is
  stopped and counted as a failure with the reason recorded.
- Time limit per trial: fixed in the task configuration before the batch and never
  changed between batches, since a changed limit voids earlier evidence.

## What the number means

- At or below the registered band, which is 20 percent and is registered here and nowhere
  else: the task sits in the intended band; report it as measured and proceed.
- Above that band: climb the pre-registered escalation ladder one rung (degrade the
  sources along measured dials; then lengthen the horizon; then add a second evidence
  function on the same world), re-qualify the bars on fresh worlds with both strong
  methods, and run a new batch. Never tighten a bar to manufacture failure.
- Every failed trial is attributed before it is believed, under the benchmark's own
  trial-analysis taxonomy: specification_completeness, solution_discoverability,
  verifier_correctness, boundary_fairness, policy_refusal, execution_blocked,
  unearned_credit, meaningful_difficulty. Only meaningful_difficulty counts toward the
  measured pass rate, with the scientific stage named (linkage, coverage, calibration,
  projection, allocation). Any other attribution is a task defect: fixed, rerun once, both
  records kept.

## What is retained

Every trial's submission files, verifier report, transcript, token and dollar cost, and
the attribution note, under authoring evidence. Passing solutions are kept in full: a
method that beats the strong lines is a finding to publish, not a reason to move a bar.

## Sealed prediction

A prediction on the first batch's pass rate is sealed in the research ledger before the
batch, with its resolution criterion fixed: the fraction of first-batch trials with
reward 1, resolved from the retained verifier reports.
