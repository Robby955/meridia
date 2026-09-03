# Meridia reconstruction v3 confirmation

The single confirmation permitted by `meridia-reconstruction-v3-protocol.md` on
registered world index 4 completed successfully. The full packet was built from that
world without printing or exposing its keyed seed. The frozen v7 bars and both
calibration files matched their pre-registered SHA-256 digests and were not changed.

Four earlier keyed worlds were registered for version three. Indices 1, 2 and 3 each
spent the one confirmation permitted for them and each one stopped. No bar was changed
after any stop; each was closed by a general improvement to the reference lines or by
retiring the world, as recorded in `bars/national-v7/PROVENANCE.md`.

- Index 1 (packet manifest SHA-256
  `d1b7c5420a43fc1722376e5edd687450156be7aa654cb2e14bf12954b4f29922`): STOP. Both
  witnesses missed national mean adult income against the seven-world bars. Retired
  as `qual-v3-8`.
- Index 2 (packet manifest SHA-256
  `a2ac927421ab431dc33e772e9db1e1911ef763c2c12757c7c59368e6ebf0cb03`): STOP. Both
  witnesses missed projected national households (two migration waves inside the
  horizon); the design-based witness missed pooled median-income coverage. Retired as
  `qual-v3-9`.
- Index 3 (packet manifest SHA-256
  `223456acf9f9a93f574b9346a19ba152a7efcdb1a7c9d457edbe9eea72ab5289`): STOP. Both
  witnesses missed the county count bars on a 719,435-person world whose smallest
  county holds 695 persons. Open: small-county register deconvolution. The world was
  retired from grading and kept out of the qualification set, because a freeze that
  includes it widens the county bars until the `exact_key_union` control clears them.
  All eight controls failed as required on all three worlds.

## Identity and receipts

- Seal manifest SHA-256:
  `044ed8c2ae6a8deabe809f31a6eacf55d5e57c8f3d5dcfba48c624c5d4e1f317`
- Packet manifest SHA-256:
  `ca235b6d8c360e99d97371a8a18aeb1fafd02cc18d2fb03bcbe79c885fb611b6`
- Full seed-free confirmation receipt SHA-256:
  `d8c246d9df4d6281de04507017aa1dde2056a5f94b3d3259f0bf9f85770ba58e`
- Frozen bars SHA-256:
  `f53983161245bb1eca4cb36a6a35325aba3c58262d9d46ef2cd97047d4f6aae9`

## Strong witnesses

- Design-based A: PASS. Tightest measured bar ratio 0.685522; persons/all worst error
  0.159378 and coverage 1.000000; projected persons/all worst error 0.193490 and
  coverage 1.000000; allocation regret 0.000000.
- Bayesian B: PASS. Tightest measured bar ratio 0.948890; persons/all worst error
  0.142162 and coverage 1.000000; projected persons/all worst error 0.181039 and
  coverage 1.000000; allocation regret 0.000000.

For witness A the tightest ratio was projected national tertiary share, for witness B
national county mean adult income. A ratio below one is inside the frozen bar.

## Weak controls

- register-only: FAIL on national persons accuracy (0.1186 against 0.061475) and pooled
  persons coverage (0.000).
- survey-only: FAIL on national persons accuracy (0.2144) and state persons accuracy
  (0.2909 against 0.23589).
- no-deduplication: FAIL on national persons accuracy (0.0907) and pooled persons
  coverage (0.024).
- inflated intervals: FAIL on the interval-score ceiling (tertiary share 0.7858 against
  0.09).
- static projection: FAIL on projected national children accuracy (0.1374 against
  0.097099) and projected pooled children coverage (0.659).
- uniform allocation: FAIL at regret 0.2217 above 0.023805.
- benchmark-only: FAIL on pooled persons coverage (0.146) and county households
  accuracy (0.5028 against 0.274994).
- exact-key union: FAIL on five gates, county persons 0.3413 against 0.338514, county
  children 0.4896 against 0.339383, and national elders 0.0724 against 0.05. This is
  the count recipe that cleared version two.

Result: both strong methods passed and all eight controls failed. This closes the fresh
scientific confirmation only; it does not authorize or constitute a frontier trial.
