# Meridia reconstruction v2 confirmation

The single confirmation permitted by `meridia-reconstruction-v2-protocol.md` completed
successfully. The full packet was built from registered world index 0 without printing
or exposing its keyed seed. The v6 bars and both calibration files matched their
pre-registered SHA-256 digests and were not changed.

## Identity and receipts

- Seal manifest SHA-256:
  `eb7877302b9defa9ac04e73db0f8f4457a4311411963e72332c88acfc64bb990`
- Packet manifest SHA-256:
  `d697053e246af6fbc1a2cfc978e9b4fa0f9c9b06413f4e0637d4fd92d4ca4915`
- Full seed-free confirmation receipt SHA-256:
  `c2bdef407c411a56121e4b75509b396f4f76c524a1434ed748891ff2b29cd708`

## Strong witnesses

- Design-based A: PASS. Tightest measured bar ratio 0.901382; persons/all
  worst error 0.190409 and coverage 0.902439; projected persons/all worst error
  0.288220 and coverage 0.878049; allocation regret 0.012804.
- Bayesian B: PASS. Tightest measured bar ratio 0.898237; persons/all worst
  error 0.171265 and coverage 1.000000; projected persons/all worst error 0.203714
  and coverage 1.000000; allocation regret 0.005902.

For both methods, the tightest ratio was projected national low-income-household share.
A ratio below one is inside the frozen bar.

## Weak controls

- register-only: FAIL on national persons accuracy and pooled coverage.
- survey-only: FAIL on national and county persons accuracy.
- no-deduplication: FAIL on persons and household coverage.
- inflated intervals: FAIL on the interval-score ceiling.
- static projection: FAIL on child projection accuracy and coverage.
- uniform allocation: FAIL at regret 0.4230 above 0.04346.

Result: both strong methods passed and all six controls failed. This closes the fresh
scientific confirmation only; it does not authorize or constitute a frontier trial.
