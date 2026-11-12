# SENTINEL-Hydro (M6) — architecture round v2: lead confirmation of the code batch, and rulings on the code pins

**Written by the lead:** Wed 11 Nov 2026 1:15 PM EST.

**Status.** Read-only (0444). This file is the confirmation that entry-1 lead rulings §3 step 4 requires, together
with the rulings that go with it. It is not a stamp and it changes no recipe, constant or bound. It is never
edited; a correction is a new, dated lead file with its own registry row.

## 0. Files read, with their sha256 at writing

| File | sha256 |
|---|---|
| `docs/preregistration/hydromet_arch_v2_entry1_lead_rulings.md` | `0c20e00354fa29b16c2f7798e335bf10203951a4e635b5ed02aada08f483fe9e` |
| `docs/preregistration/hydromet_arch_v2_code_pins.json` | `574307a10dd417c03997890bf4022f6ac346be01de6e2a906acbfc1e350c0182` |
| `scratchpad/review_m6v2_batch/review.json` | `bdb6040cf93b7918a06eaf9deb96d71a5eb3308d186ac745b77188a030bb11f3` |
| `scratchpad/review_m6v2_batch/review.md` | `e62f8a9813155e3823c3845faa98acfa792b500873b4837274281d652d863a40` |

## 1. The check

Entry-1 §3 step 4, byte for byte with whitespace normalised: "The lead checks the batch against §2 and records the
code-pin sha256 in the LEAD_DECISIONS file. **The first D1 grid fit, any selection, gate G-D1a and any stack wait
for that confirmation.** If the check fails, the fits of step 3 made on the failing code are void and are repeated
after the fix."

An independent reviewer that wrote none of the code checked the batch against entry-1 §2. Its verdict is
`confirm_with_lead_rulings`, with no blocker. What it verified, rather than took on the engineer's word:

- Every §2 item (F1, F4, F6, F9, decision 2 condition 3, decision 3, the pin hash in the GBM, P and stack records,
  and every bullet of decision 4) is implemented and carries a unit test, with file:line evidence.
- The batch changed exactly the 8 code files and 3 test files the engineer listed and nothing else since the stamp
  row `e956ed69fd42`, checked file by file against that row's `code_sha256_at_stamp`.
- All five entry points call `assert_released` before any row is read. Only `--smoke` bypasses it, a smoke run
  never receives a `release.json` and never writes an inner or decision row, and the two cannot be combined; the
  single sanctioned exception is `--smoke --d2-kill-rule` (row `afc1fbaebe0d`).
- All 15 non-smoke run directories and all 38 M6V2 registry rows carry the release block with `e956ed69fd42`, pins
  `84958df5…c5d9`, code pins `574307a1…0182` and both change maps empty; the pool is read through
  `load_v2_train(require_pins=True)` with the fingerprint matched, train block, train roles.
- The code-pin record verifies clean (18 files, 0444); `verify_pins` is clean over 38 files; `thermox/nets.py` is
  `5e554c31…85d5`; the frozen M6 code hash recomputes to `5bd7d329…`.
- Tests: 134 passed, 0 failed, 0 skipped. Mutation set: 34 applied, 34 killed, 0 survived, on a scratch copy only.

## 2. Rulings

### B-1 — the batch is confirmed; the D1 work is released

The lead confirms the batch of entry-1 §3 step 1. The code-pin sha256
`574307a10dd417c03997890bf4022f6ac346be01de6e2a906acbfc1e350c0182`, the code-pin registry row `7744d6502add` and
the stamp row `e956ed69fd42` that released the fits are recorded in part 6 of
`docs/plans/architecture_research/LEAD_DECISIONS_20261110.md`, which is what step 4 asks of the lead. **The D1
grid, any selection on it, gate G-D1a and the stacks are released** and may start.

The check did not fail, so the second sentence of step 4 does not apply: the fits of step 3 stand. The 36
`M6V2/inner_v1` rows of R-TD, R-TD-W, G1 and G2 are **not void**, and neither is the physics arm now running.

### B-2 — four files of the non-smoke import closure are in neither record (finding R1)

The reviewer found four files that the non-smoke import closure reaches and that appear in neither the pins file
nor the code-pin record: `sentinel/hydromet_sota/gbm.py` (the LightGBM fit routine and the QStack that G1 and G2
use through `gbm_ref.py:37`), `sentinel/hydromet_sota/data.py` (the lever pool loader, whose output is
fingerprinted while its code is unpinned), `sentinel/hydromet_sota/runner.py` and
`sentinel/hydromet_sota/__init__.py`.

**Ruling.** The fits already made stay valid: the lead recomputed all four hashes and each file's modification
time is earlier than both the stamp (2026-11-11T01:01:16+00:00) and the first fit job (2026-11-10 20:49 EST), so
no fit of this round ran on a state of these files other than the one recorded here. Their hashes at this
confirmation are:

| File | sha256 at this confirmation |
|---|---|
| `sentinel/hydromet_sota/gbm.py` | `e97af8545638ff398b6c12325bf4478aa8e9d1be7361c0c5e9f3fa5b23a99126` |
| `sentinel/hydromet_sota/data.py` | `75ffc6ea47ffbb5f2bd0fd845e8d6887e2b732302490b84783f4da6c609b742d` |
| `sentinel/hydromet_sota/runner.py` | `751d2c71c4e2153db4a1202b9422e542ceceeb6089acdaa1478e38dd7b1015b8` |
| `sentinel/hydromet_sota/__init__.py` | `04b4b42c6b1024341bd78a9c546e6d46857d169fa96172a8cc11cfbe3ac4ed56` |

- From this file onward these four hashes are binding in the same way as a code pin: if any of them changes, every
  fit made after the change is void until the lead has read the diff and written a new ruling, and the engineer
  reports the change to the lead before the next job starts.
- They are added to `code_pin_targets()` and to the code-pin record when entry 2 is appended, so that the guard
  enforces what this file states. Until that append, this file is the enforcement.
- The engineer re-derives the non-smoke import closure after that change and reports any further file the closure
  reaches that neither record holds. A closure file that is in neither record is a defect of the record, not of
  the fit.

### B-3 — the `CODE PIN CHANGE` line (open point 1)

**Ruling.** The proposed form is accepted: `CODE PIN CHANGE: <label> sha256 <sha256>`, one line, parsed from the
deviations file. Four conditions, all of which the reviewer checked and the lead adopts:

- Whole lines inside fenced blocks are live to the parser, so every example of the form written anywhere in the
  record uses the literal placeholder `<sha256>` and never a real 64-character hash. A document that shows a real
  hash in an example is a defect and is corrected before the file is appended.
- Every malformed variant fails closed, and a wrong label has no effect: a line that does not parse releases
  nothing.
- Last line wins, which is safe only because the deviations file is append-only and is stamped whole; if that ever
  stops being true, this ruling stops with it.
- A `CODE PIN CHANGE` line for `gso_mt/runner.py` is written only after the lead has read the diff of that file
  and said so in writing. No other file of the fit-code class is changed under a line without the same step.

### B-4 — `code_changed_under_stamp` must be compared, by file class (open point 2)

The reviewer's point is correct and matters: `code_changed_under_stamp` is recorded in every run record but is not
compared in the combination check, so after the first `CODE PIN CHANGE` line the equality of `code_pins_sha256`
becomes vacuous, and seeds of one arm fitted under different trainer states could be averaged together.

**Ruling.** The combination check compares `code_changed_under_stamp` **by file class**. Equality is required on
the fit-code class — the files whose state can change a fitted model's predictions — and the entry for those files
is a hard condition of combining runs; for the remaining labels (gate, stack, inner and pin-script code) the entry
is recorded and reported but does not block combination. Conditions:

- The engineer writes the exact label list of the fit-code class from `code_pin_targets()`, including the four
  files of B-2, and sends it to the lead for approval **before** the rule is implemented. The lead does not
  enumerate labels it has not read in the code.
- The rule is implemented and tested together with the first `CODE PIN CHANGE` entry, and not later. Today both
  the old and the new rule agree, because `code_changed_under_stamp` is empty everywhere, so nothing already
  fitted is affected.
- A run whose fit-code class entry differs from another's is never averaged with it, never enters the same seed
  mean and never enters the same gate.

### B-5 — `gso_mt/synthetic.py` stays unpinned (open point 3)

**Ruling.** Accepted as the engineer proposed and the reviewer confirmed: `synthetic.py` is reachable only under
`if a.synthetic:` inside pinned drivers that refuse `--synthetic` without `--smoke`, and a smoke run writes no
inner or decision row. It therefore cannot reach a counted number and does not need a pin. If a driver ever
reaches it outside a smoke run, that is a defect and the file is pinned at once.

### B-6 — five mutations are killed only by the record's own self-check (finding T1)

**Ruling.** The five mutations the reviewer names (the prefix sha comparison, the any-length sha, the prefix
comparison in `assert_combinable`, `log_inner` on a smoke run and re-writing the record) are killed today only by
the hash self-check of the record itself, with no dedicated unit test behind them. A dedicated test for each is
written and passing before entry 2 is appended. This blocks nothing now: the mutations are killed, so the guard
behaves correctly; what is missing is the second line of defence.

## 3. What this file does not release

- Track T-E (England) stays on hold in full. No England reading is QC-summarised, aggregated, fitted or scored
  before the lead's stamped T-E entry.
- The D2 temperature stage waits for entry 2 and the lead's rulings on it.
- Tier 2 waits for the lead's answer to amendment §14 Q3.
- No look of any kind, and no validation-cell number of an arm, reference or stack without the lead's go for that
  step.

## 4. How this file is read

- Where this file and a lead message differ, this file wins.
- B-2's four hashes bind from this file; B-3, B-4 and B-6 are implemented with entry 2; B-5 changes nothing.
- The id of this file's registry row is recorded in the campaign decisions file, and entry 2 cites this file by
  name and sha256.
