# Frozen bars, national v3

Frozen 2026-09-01 by `scripts/freeze_bars.py` at Meridia main b1af721, the first freeze
under the amended rule with stated per-kind, per-level floors. Development worlds
(calibration only): seeds 20260831, 20260903, 20260904. Qualification worlds (bars):
seeds 20260902, 20260908, 20260909, 20260910. Both strong methods pass every bar on all
four; all six controls fail a named gate on all four. Confirmation on the fresh world
20260911: the Bayesian line passed every bar, all six controls failed, the design-based
line exceeded two state-level interval-score ceilings by about one percent. Per protocol
these bars were not moved; 20260911 joins the qualification set for v4.
