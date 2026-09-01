# Frozen bars, national v1

Frozen 2026-09-01 by `scripts/freeze_bars.py` at Meridia main cf2589d. Development
worlds (calibration only): seeds 20260831, 20260903, 20260904. Hidden worlds (bars):
seeds 20260902 and 20260908. Both strong methods (design-based line A, Bayesian line B)
pass every frozen bar on both hidden worlds; all six controls fail at least one named
gate on both. Supersedes national-v0, which was frozen on one hidden world and did not
hold on the second until the methods estimated mortality from the two register
snapshots. Margins unchanged: accuracy 1.25 times the worse strong worst-unit error;
coverage floor the worse strong pooled coverage minus 0.10, never below 0.70;
interval-score ceiling 1.5 times; regret ceiling 2 times. Packets regenerate
byte-identically from `meridia.packet.build_packet(seed, out, PacketParams(), development)`.
