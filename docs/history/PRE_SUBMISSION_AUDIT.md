# Pre-submission audit

Every item is checked and recorded before a Terminal-Bench Science reviewer sees the
task. The first two merged tasks burned review cycles on items in the second list; the
first list is what this task adds. A line is closed only with the artifact named beside it.

## Items this task adds

- Reviewer pattern 1, verifier not co-designed with the oracle: two strong methods from
  different families pass; the verifier reads declared outputs only; bars from the worse
  method with a margin and stated floors. Artifact: `docs/REVIEWER_GATES.md`, `bars/`.
- Reviewer pattern 2, no privileged generator knowledge: methods run on the participant
  side alone (tests copy `participant/` into an empty directory); public ranges and
  mechanism families only; mortality, coverage, and nonresponse estimated from records.
  Artifact: `tests/test_methods.py`, `tests/test_methods_b_and_controls.py`.
- Fresh-world rule: bars frozen on qualification worlds, then one unseen world, both
  strong methods and all controls run once, stop and report on any miss, bars never
  moved after the world is seen. Artifact: `bars/national-vN/PROVENANCE.md`, the freeze
  and confirmation logs, the relay entries.
- Instruction and verifier aligned: every scored output is named in the instruction and
  the contract; nothing promised that is not scored. Artifact: `docs/RELEASE_CONTRACT_V0.md`
  against `meridia/verify.py`.
- File set fails closed: exactly the declared files plus an optional totals file that is
  audited. Artifact: `meridia/verify.py`, red-team report.
- Red-team: independent attacks recorded with the gate each fails; the one that passed
  led to a verifier fix and is retained as evidence. Artifact: the packaging thread's
  red-team report and resolution note.
- Reference solutions run inside the participant container and receive reward 1 from the
  separate verifier image with zero accommodation. Artifact: Harbor witness receipts.
- Development worlds ship with truth (three); the hidden world exists only in the test
  image; no truth column reaches the participant side. Artifact: packet manifests,
  `FORBIDDEN_COLUMN_PREFIXES` in `meridia/packet.py`.
- Public vocabulary: world, state, county, sources, archives, snapshots; no agency
  framing anywhere in the task, the instruction, or the repository. Artifact: the voice
  gate run on every committed file.
- Independence and citations: every method from the public literature, cited where used.
  Artifact: `docs/INDEPENDENCE.md`.

## Items the first two tasks taught

- Instruction under 700 words, outcome not method; the contract carries the detail.
- Scientific vocabulary matches what the task does; no borrowed domain dress.
- Author expertise stated plainly in the proposal; understating it reads as "no
  specialist needed".
- `tests/test.sh` runs every test file; a green run must not hide a regression.
- Data regenerate from the generator; no hand-edited generated file.
- Timeouts and resource limits in the instruction match the configuration; changing one
  voids earlier trial evidence, so trials run after the final configuration.
- Local grading: a verifier that needs no network is graded from the test image; know
  the local runner's quirks before quoting a reward.
- Frontier trials: several per model on the final packet, pass rate reported as
  measured; if above the band, climb the registered ladder one rung and measure again.
- Everything above is done before the proposal form; the form quotes the measurements.
