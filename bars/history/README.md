# Superseded bar sets

One line each. The shipping set is `bars/national-v14-standard`, described in
`bars/README.md`. Nothing under this directory can decide a verdict: the verifier refuses
any receipt that does not record `"frozen": true`. Seal protocols under `seals/` cite the
version-two and version-three sets at their former `bars/national-v6` and
`bars/national-v7` paths; those directories are now `bars/history/national-v6` and
`bars/history/national-v7`.

- `national-v0` through `national-v5` are the version-two calibration trajectory, one
  freeze each of the release-contract surface.
- The version-two set that the reconstruction protocol sealed against is `national-v6`,
  cited by digest in `seals/meridia-reconstruction-v2-protocol.md`.
- Version three, and the last set that gated a version-three submission, is `national-v7`,
  frozen on nine qualification worlds with the sealed world index 4 confirmed against it.
- First of the version-four attempts are `national-v8`, `national-v9` and `national-v10`,
  all recording `"frozen": false`.
- One evidence pass read under two profiles gave `national-v11-full` and
  `national-v11-lite`, both refused.
- Its successor gave `national-v12-full` and `national-v12-lite`, both refused.
- Immediately before the shipping pass came `national-v13-full`, `national-v13-lite` and
  `national-v13-standard`, all three refused.
- Reading the shipping evidence under the other two profiles gave `national-v14-full` and
  `national-v14-lite`. Both refused, and their receipts were dropped because an unfrozen
  receipt decides nothing; the freeze reports and provenance remain as the record that the
  same evidence was read under all three profiles.
