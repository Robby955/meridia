# Frozen bars, national v5

Frozen 2026-09-01 by `scripts/freeze_bars.py` at Meridia main 79fac7f, under the rule
with per-kind, per-level floors and interval-score ceilings that bind on pooled keys.
Development worlds (calibration only): seeds 20260831, 20260903, 20260904. Qualification
worlds (bars): seeds 20260902, 20260908, 20260909, 20260910, 20260911, 20260912. Both
strong methods pass every bar on all six; all six controls fail a named gate on all six.
These bars are fixed before the fresh world 20260913 is generated and do not change
after it is seen.
