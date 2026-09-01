# Frozen bars, national v0

Frozen 2026-09-01 by `scripts/freeze_bars.py` at Meridia main ce5fb5a. Development
worlds (calibration only): seeds 20260831, 20260903, 20260904. Hidden world (bars):
seed 20260902. Both strong methods (design-based line A, Bayesian line B) pass every
frozen bar on the hidden world; all six controls fail at least one named gate. Margins:
accuracy 1.25 times the worse strong worst-unit error; coverage floor the worse strong
pooled coverage minus 0.10, never below 0.70; interval-score ceiling 1.5 times; regret
ceiling 2 times. Packets are not in the repository (1.2 GB each); they regenerate
byte-identically from `meridia.packet.build_packet(seed, out, PacketParams(), development)`.
