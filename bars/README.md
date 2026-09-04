# Frozen bars

Each directory is one freeze, and the shipping set is `national-v14-standard`.

`national-v14-standard` is the version-four set that decides. It records `"frozen": true`
and `"gate_profile": "standard"` in `bars.json`, and it is the only version-four set that
records `"frozen": true`. It was frozen under the `standard` gate profile on six
qualification worlds and carries its own blockers and caveats. See
`national-v14-standard/PROVENANCE.md` and `national-v14-standard/freeze_report.txt`.

The `lite` set reads the same evidence under its own profile. The `full` set refuses on
`reserve_skill/worst_regional_shortfall_probability` before evidence is compiled and
carries no bars. Both sit under `history/`, and both receipts were dropped because an
unfrozen receipt decides nothing; their freeze reports and provenance remain as the record
that the same evidence was read under all three profiles.

`history/` holds every superseded freeze, one line each in `history/README.md`, including
the version-three set `national-v7`, frozen on nine qualification worlds with the sealed
world index 4 confirmed against it, and the version-two set `national-v6` that the seal
protocols cite. The per-replicate submissions behind the superseded freezes are held in
authoring evidence outside this tree.
