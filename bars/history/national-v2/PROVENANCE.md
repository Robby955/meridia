# Frozen bars, national v2

Frozen 2026-09-01 by `scripts/freeze_bars.py` at Meridia main e4768f4. Development
worlds (calibration only): seeds 20260831, 20260903, 20260904. Qualification worlds
(bars): seeds 20260902, 20260908, 20260909. Both strong methods pass every frozen bar
on all three; all six controls fail at least one named gate on all three. Supersedes
national-v1 (two worlds), which did not hold on 20260909 until both methods shrank
state coverage ratios toward the national ratio by sampling support and the
design-based line carried a horizon drift allowance on income items. These bars are
fixed before the task's hidden world (seed 20260910) is generated and do not change
after it is seen. Margins unchanged from v0.
