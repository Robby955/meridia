# Meridia reconstruction v3 sealed-world protocol

Registered before the full packet was generated or inspected.

- World: index 4 in `meridia-reconstruction-v3.json` (SHA-256
  `044ed8c2ae6a8deabe809f31a6eacf55d5e57c8f3d5dcfba48c624c5d4e1f317`), derived only
  through `meridia.sealing.sealed_seed`; the master key stays outside the repository.
  Index 0 of the same manifest is the version-two world, re-registered with digests
  identical to `meridia-reconstruction-v2.json` (terrain and microdata are unchanged by
  version three) and retired from grading: its packet was used by a trial. Index 1 was
  registered first for version three; its one permitted confirmation stopped (both
  strong witnesses missed national mean adult income against the seven-world v7 bars),
  so it was retired to the qualification set as `qual-v3-8`. Index 2 was registered
  next; its confirmation stopped on projected national households (two migration waves
  inside the horizon) and on the design-based line's pooled median-income coverage, so
  it was retired as `qual-v3-9`. Index 3 was registered next; its confirmation stopped
  on the county count bars (a 719,435-person world whose smallest county holds 695
  persons, where the survey weights raked to the register's raw county distribution
  carried the misfiled trickle into the direct county estimate), so it was retired from
  grading. It was not taken into the qualification set: a freeze on ten worlds
  including it lifts the county count bars far enough that the `exact_key_union`
  control passes on three qualification worlds, and a bar wide enough to admit that
  control is not a bar. The qualification set stays at nine worlds and the small-county
  gap is recorded as open in `bars/national-v7/PROVENANCE.md`. The manifest was then
  re-registered with a fifth world, before the bars were re-derived and before this
  registration. Indices 0 to 3 reproduce byte for byte on each re-registration.
- Packet: `scripts/build_sealed_reconstruction_packet.py`, which calls
  `meridia.packet.build_packet` with the default national parameters,
  `PacketParams(regime="hidden")`, and `development=False`, and refuses a packet whose
  retained regime is not hidden. The host directory name must not contain the derived
  seed or a date.
- Packet built for index 4: manifest SHA-256
  `ca235b6d8c360e99d97371a8a18aeb1fafd02cc18d2fb03bcbe79c885fb611b6`. The registered
  digests replayed from the master key before the packet was written, the retained
  regime is hidden, and the derived seed appears in no participant file, no built
  archive, and no task file.
- Frozen bars: `bars/national-v7/bars.json`, SHA-256
  `f53983161245bb1eca4cb36a6a35325aba3c58262d9d46ef2cd97047d4f6aae9`.
- Calibration A: `bars/national-v7/calibration_A.json`, SHA-256
  `61cf3367661bee14155934be70e1afddf0ae4cf04c57494cd92537f6e2c0b107`.
- Calibration B: `bars/national-v7/calibration_B.json`, SHA-256
  `5fb058218393b113a2d2d16d7939aa0e69ca2a8f1041dac8954a9ab813dc6790`.
- Strong witness A: design-based method, 100 bootstrap replicates, calibration A.
- Strong witness B: Bayesian method, 400 sweeps and 100 burn-in sweeps,
  calibration B.
- Controls: all eight names in `meridia.methods.controls.CONTROLS`, calibration A,
  including `exact_key_union`, the count recipe that cleared version two.
- Visibility: every method receives only the packet's `participant/` tree. The retained
  side is available only to the verifier.
- Surface: remove each method's legacy `detailed.csv` before verification and call
  `verify_release_projection_allocation`; exactly `release.csv`, `projection.csv`, and
  `allocation.csv` are accepted.
- Decision: both strong witnesses must pass every unchanged v7 bar and every control
  must fail at least one gate. A strong miss or passing control stops the seal. Bars are
  never changed from this confirmation.

Only one confirmation run is permitted on the full packet. No frontier trial may start
from this registration without Rob's separate authorization.
