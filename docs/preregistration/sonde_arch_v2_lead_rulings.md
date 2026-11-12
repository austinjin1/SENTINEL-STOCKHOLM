# M1 SENTINEL-Sonde — architecture round v2: lead review, rulings and stamp

**Written by:** the lead (AI-assisted), 2026-11-10, 17:24 UTC (12:24 EST).
**Stamps:** `docs/preregistration/sonde_arch_v2_amendment.md`
- sha256 `6c7b169d60ad74aa941965c67641d9a9e72faea804db6ac5b5b7b77d6bd60226`
- 326 lines, 26,071 bytes, last modified 2026-11-10 17:15:11 UTC.

The amendment is stamped **as written, byte for byte**, together with the rulings and clarifications below. Where
this file and the amendment differ, **this file wins**. Both files are read-only (0444) from the stamp onward.

## 1. What the lead checked before stamping

| Check | Result |
|---|---|
| File hash equals the hash M1 reported | yes |
| A validation number exists on a new arm or track | none found: the registry has no M1 row for any NWP, GEFS, SONDE-HF, R-NH, PF-GBM, E2, PhiDA, M1-H or StreamPULSE experiment, and `sentinel_runs/sonde` has no such run folder. The only M1 activity on 2026-11-10 is the Test-A rehearsal of the original track. |
| A bar is lowered | no. `m1n_protocol.md` §4 is unchanged; four references are added, which makes it harder. |
| Configuration cap | 5 + 2 + 4 + 2 + 3 = 16 ≤ 16 per variable group. References have their own parity allowance (≤ 5 each). |
| Comparator parity | stated for inputs, bias correction, statics, training data, auxiliary targets and tuning. |
| Bias correction and error model | fitted on train-block weather only; no validation or test target enters. |
| Gates and drop rules | pre-declared, on train-role sites × train block only; reported whichever way they fall. |
| Selection | early stopping never on the val block; E2 vs SONDE-HF is cross-fitted; the tie rule is pre-declared. |
| New blocks | M1-H uses the locked new-site block (no value read). M1-SP is lock-on-download and not yet downloaded. |
| Consistency with `LEAD_DECISIONS_20261110.md` | consistent, except the points ruled on below. |

## 2. Rulings on R1–R5

- **R1 — approved.** E2-B uses R-NH in place of "LSTM-w-NWP". E2-B = {GBM-NWP, PF-GBM, R-NH, Chronos-2-cov ZS},
  weighted by exactly the E2 rule.
- **R2 — the stricter option: no upstream-sensor inputs in M1-N v0.6.** No model in this amendment, candidate or
  reference, receives upstream-sensor features. Lever 3 of `m1n_protocol.md` §5 is suspended for v0.6. The date rule
  of amendment §2g is kept on record as the minimum condition for any later amendment, which would need its own
  stamp before any result.
- **R3 — approved, with conditions.** GEFSv12, control member c00 only, variables tmax / tmin / apcp / dswrf, lead
  days 1–7, about 0.6 TB streamed, less than 1 GB kept.
  1. **Pilot first:** one reforecast water year and one operational month. Log to the registry: a correctness check
     of the point extraction (values compared with an independent read of the same GRIB message at ≥ 20 sites),
     measured throughput, and the projected kept size.
  2. **Continue automatically** after the pilot unless the projected kept size exceeds 1 GB, or the projected
     wall-clock time exceeds 72 h, or the correctness check fails. In any of those cases, stop and report.
  3. Transient files: at most 5 GB on disk at any moment, deleted after each batch; `df -h /` ≥ 60 GB free checked
     before each batch.
  4. **Decoding environment:** eccodes / cfgrib go in a small **separate** conda environment, with its package list
     saved in the data manifest. **The `sentinel2` environment is not modified** (frozen looks pin it).
  5. ≤ 6 CPU threads; one M1 CPU job at a time.
  6. Members p01–p04 need a new ruling.
- **R4 — approved, with conditions.** M1-H is scored in the same single hardened look as M1-N: one unsealing of the
  new-site block, one claim.
  1. The M1-H candidate, its comparators and its decision rule are fixed in the **same** lead-written freeze record,
     before the claim.
  2. M1-H has its own decision and is outside the M1-N Holm family. Reports state that the look carried **two
     confirmatory families** (M1-N; M1-H) and give both results together, win or lose.
  3. If M1-H is not freeze-ready when M1-N is, the lead decides **in writing before the look** whether to wait or to
     score M1-N alone. If M1-N is scored alone, any later M1-H result on that block is **descriptive only**.
- **R5 — approved.** EW-H1h is descriptive unless the amendment §3 projection reaches ≥ 50 onsets with ≥ 2 basin
  groups before the freeze, and it must also reach ≥ 50 evaluable onsets at scoring. The disclosure sentence in §3 is
  mandatory wherever EW-H1h is reported.

## 3. Lead clarifications (binding)

- **L1 — time of writing.** The amendment header says "~17:40 UTC". The file was last modified at 17:15:11 UTC. The
  binding time is the timestamp of the `sonde/prereg_snapshot` registry row.
- **L2 — "best reference" (amendment §3, M1-H primary).** It means the worst case over **all** listed comparators
  on the scored block, at each lead: the candidate must beat every one of them. This is the rule `scripts/eval/sonde_score.py`
  applies for M1. The comparator is never chosen on validation.
- **L3 — cross-validation training data (amendment §2h and §3).** In each fold, network weights and tree models are
  fitted on the in-fold sites × **train block only**, with the 42-day embargo at the block edge. Val-block rows are
  used only as out-of-fold scores, and for calibration and ensemble weights fitted on out-of-fold predictions. If the
  existing CV code does anything else, M1 reports it to the lead before running any CV on a new arm.
- **L4 — deviations after the stamp.** The stamped amendment is not edited. A deviation goes in a new file,
  `docs/preregistration/sonde_arch_v2_deviations.md`, with its date and reason, and is stamped by the lead before the
  affected run produces a validation number.
- **L5 — public record.** The git commit of this stamp is **deferred**: a commit now would move an environment pin
  that a pending downstream look depends on. Until the commit, the record is the timestamped registry row plus the
  two read-only files. The commit and a public push must exist **before any freeze** on M1-N, M1-H or M1-SP. Reports
  describe the record as "pre-specified and frozen in a public GitHub commit before testing (not registered on OSF)".

## 4. What M1 may do from the stamp onward

In the amendment §5 order: step 1 (§2f) → references → SONDE-HF CV → M1-H models → gated families if the rule allows
→ finals, E2 and E2-B → numeric-forecast report on CV → freeze candidate → rehearsal → pre-test review → lead freeze
→ single look. GPUs 0–9 through the lock helper (`--min-free-mb 10000`, ≤ 2 h per process). Every run gets a
registry row, failed runs included. The original M1 track and its Test-A look are not delayed.
