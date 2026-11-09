# EW-v2 — post-freeze addendum

`FREEZE_ew.json` and `ew_v2_protocol.md` are hashed and read-only, so this addendum records changes to them rather than editing them.

## A1 (2026-11-09): public record replaces the OSF registration (user decision)

**Decision.** Both the protocol header and the FREEZE record `status` say:
"Test-A scoring only after OSF registration".
On 2026-11-09 the project author (Bryan Cheng) decided not to use OSF ("i dont think we need this"). The lead records that decision here.

**The public record is now the GitHub commit.** No OSF registration exists, and none will be claimed in any report or paper. Instead, the public, timestamped record of the frozen plan is commit `cef72b6` on branch `v2-rebuild` of https://github.com/austinjin1/SENTINEL-STOCKHOLM. It was pushed on 2026-11-09, before any Test-A row was loaded, and is mirrored to HF `bryan7264/SENTINEL` branch `v2-rebuild` (commit d5cb849…, files identical).

**Hashes, checked by the lead before the test pass.** The files in that commit are byte-identical to the local frozen files:

| File | sha256 | Matches commit |
|---|---|---|
| `docs/preregistration/FREEZE_ew.json` | `24716009f1a74442a55628ca51f9330ba6441f0d5e8ee6f3b1b6926a2cea2832` | yes (`git show cef72b6:…`) |
| `docs/preregistration/ew_v2_protocol.md` | `0d5c8f2e1b2c12d9796c08b1b0284da0b9fc97f450523e883601897c86e1a3a0` | yes |

The prepared OSF package (sha `02ad146b…`, 126 files) stays as an unused artifact.

**Limitation, to be stated wherever pre-registration is mentioned.** A git commit is weaker evidence than an OSF registration: history can in principle be rewritten, and it carries no DOI. Reports must say:
"pre-specified and frozen in a public GitHub commit before testing (not registered on OSF)".
They must not say "pre-registered on OSF".

**Nothing else changes.** The hypotheses, data, code, thresholds, the single-pass rule and the registered `score_command` are unchanged.

**Live trial.** The live-trial amendment planned for before 2026-12-01 will likewise be frozen in a dated public commit before 2026-12-01.
