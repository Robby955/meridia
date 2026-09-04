# Meridia version-four graded worlds confirmation

The three graded worlds registered in `meridia-v4-worlds.json` were minted and built at the
committed continuation ensemble size of 2048. Each seed was derived inside the sealed
builder from the master key and the seal manifest. No seed was printed, logged, written to
a participant tree, or committed, and the build log carries one line per world holding the
file count and the elapsed seconds and nothing else.

The freeze receipts authorized the build. Before any key was opened, the graded readiness
validator read both receipts out of the standard bar directory, being the frozen
`bars.json` and the accepted `reserve_calibration_accepted.json` under
`bars/national-v14-standard`, found no schema error on either, and reported a graded world
count of three at a compiled reserve rate of 3769.0 per person year. That rate is the one
the packet parameters carry, so the seal and the frozen profile agree on the law these
worlds are built under.

## Identity and receipts

```
4e63bca94b83b1cd574d9580c81a24326aecdfaee5d11d6d90c5f7fd86b248ce  the seal manifest
892505b85c3e45178c7434673f6e0959253d3180fe07c7af19082d958e21a20a  the frozen standard bars
179e2d50fd97f7076c601a9c43c335fd8f0e9b1fe9a133bc5ed72e5905b3aaf7  the accepted reserve calibration
cbe9003b38ea4ed0e822926b1e5149cdcffa9b8856d5fadab71004e739872214  graded-0/manifest.json, the world the package seals
118ab77fa7762fdffc7109b46f6880a8f72a1cab5d4b8a44f0b7314f7ba792ec  graded-1/manifest.json, held outside the package
281a987492872add87468e792110aa5b69a9285fb082d3fbd391944076517ff6  graded-2/manifest.json, also held out
```

## What was built

Each world holds fifteen participant files and seven retained files under the hidden source
regime. Elapsed build time was 3265 seconds at index 0, 2947 at index 1 and 2773 at index 2,
three worlds at once on one machine with no continuation cache. The seal manifest binds the
packet parameters, the generator and continuation source laws, the runtime law, and both
freeze receipts, so a world built under any other parameter set or any other bar file would
not authorize.

Registered index 0 is the world the task package seals into its images. The other two stay
outside the package and were minted so that a second and a third grading world exist under
one seal without a second mint.

## What this record does not say

The three graded worlds were minted after the freeze completed and none of them entered it.
Every bar was measured on the six qualification worlds, with the development diagnostics
measured separately on the twelve development worlds, and `freeze_report.txt` beside those
bars carries that measurement with a verdict line reading FROZEN under the standard profile.
No reference line and no control was run on a graded world here, so this file states no
pass, no fail and no measured value for any of them. It records the mint and the identity of
what was minted, and nothing further.
