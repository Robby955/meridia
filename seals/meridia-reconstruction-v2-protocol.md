# Meridia reconstruction v2 sealed-world protocol

Registered before the full packet was generated or inspected.

- World: index 0 in `meridia-reconstruction-v2.json`, derived only through
  `meridia.sealing.sealed_seed`; the master key stays outside the repository.
- Packet: `meridia.packet.build_packet` with the default national parameters and
  `development=False`. The host directory name must not contain the derived seed.
- Frozen bars: `bars/national-v6/bars.json`, SHA-256
  `50406a1c9122f3f52e1e5ff59af7a2ca44533082aa87dc89264daad6f8bf603b`.
- Calibration A: `bars/national-v6/calibration_A.json`, SHA-256
  `5d1d36b8b58d8cbcf992244137cdbb829b7701840ae7445b819a28457234a1eb`.
- Calibration B: `bars/national-v6/calibration_B.json`, SHA-256
  `8c1fe31b7a8d778b305c212a361da22da1e423ba4ff0a72568278c459339848c`.
- Strong witness A: design-based method, 100 bootstrap replicates, calibration A.
- Strong witness B: Bayesian method, 400 sweeps and 100 burn-in sweeps,
  calibration B.
- Controls: all six names in `meridia.methods.controls.CONTROLS`, calibration A.
- Visibility: every method receives only the packet's `participant/` tree. The retained
  side is available only to the verifier.
- Surface: remove each method's legacy `detailed.csv` before verification and call
  `verify_release_projection_allocation`; exactly `release.csv`, `projection.csv`, and
  `allocation.csv` are accepted.
- Decision: both strong witnesses must pass every unchanged v6 bar and every control
  must fail at least one gate. A strong miss or passing control stops the seal. Bars are
  never changed from this confirmation.

Only one confirmation run is permitted on the full packet. No frontier trial may start
from this registration without Rob's separate authorization.
