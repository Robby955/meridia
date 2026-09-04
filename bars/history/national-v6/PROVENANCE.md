# Frozen bars, national v6 (confirmed)

Frozen 2026-09-01 by `scripts/freeze_bars.py` at Meridia main 79bc89e, under the rule with
per-kind, per-level floors and interval-score ceilings bound on pooled keys. Development
worlds (calibration only): seeds 20260831, 20260903, 20260904. Qualification worlds
(bars): seeds 20260902, 20260908, 20260909, 20260910, 20260911, 20260912, 20260913. Both
strong methods pass every bar on all seven; all six controls fail a named gate on all
seven.

Confirmation on the fresh world 20260914, generated after these bars were fixed and
never inspected before the run: both strong methods PASS every bar; all six controls
FAIL their targeted gates (register-only, survey-only, and no-deduplication on
accuracy; inflated intervals on the interval score; static projection on projection
accuracy and coverage; uniform allocation on regret). Bars unchanged after the
confirmation. That date-derived world was later burned by an adversarial seed scan and
is retained only as qualification evidence; it is not the task's hidden world.

For task seal v2, a new hidden world was registered at keyed world index 0 through
`meridia.sealing.sealed_seed` before its full packet was generated. The one permitted
confirmation again passed both strong methods and failed all six controls, with the v6
bars unchanged. The registration and seed-free evidence are in `seals/`.

The fresh-world loop that produced this took six iterations; each earlier fresh world
became a qualification world and each miss was closed by a general improvement, never by
moving a bar: coverage-ratio shrinkage by sampling support, mortality estimated from
record disappearance, stated floors in the freeze rule, pooled score ceilings, and a
cross-source county vote with a public-rate miscoding debias. The full sequence is in
the earlier `bars/national-v*/PROVENANCE.md` files.
