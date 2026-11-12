# SENTINEL-Toxin (M4) architecture round v2 — lead rulings on the guard review (01)

Written by the lead on 2026-11-10 20:22 EST. **This file is not a stamp and releases nothing.** The pins file
`docs/preregistration/chemtox_arch_v2_pins.json` is NOT written: the independent review of the changed guard, of
addendum 01 and of the tests returned the verdict `fix_then_rereview`. `verify_freeze()` therefore keeps refusing,
and no `train_pool` row, auxiliary row or sealed row may be read (stamped rulings B1, S2, C-2).

Order of authority: the stamped amendment (sha256 `fb52cd4b…293b`) and the stamped lead rulings (sha256
`553be413…f060`) win over this file; this file only tightens them. Where this file and addendum 01 (sha256
`9a195e64…9f2e`, written by M4, not registered by the lead) differ, this file wins. Nothing here moves a bar, a
split, a block, a comparator, a statistic, a cap or an outcome wording.

## 1. What was reviewed

Independent review, read-only, started 2026-11-10 19:50:20 EST, written 20:10:49 EST; all probes and mutations on
synthetic frames. Kept as `docs/preregistration/reviews/arch_v2_20261110/M4v2_guard_review.json` (sha256
`bfa4a33c3b6622f76aa7478f9e510e5aa5f71ad313ee1015d8c8813b14274dba`) and `M4v2_guard_review.md` (sha256
`8e79016934e43d9a46c34b137d2488bb9bb5c20aa19ed603c3d6883015971163`). Finding numbers F1–F14 below are the
review's.

Files under review (sha256 re-computed by the lead when this file was written; equal to the review's):

| file | sha256 |
|---|---|
| `docs/preregistration/chemtox_arch_v2_addendum_01.md` | `9a195e643c420c39b7fd4451053fd3f482e738441d3ca626e85659cd342c9f2e` |
| `sentinel/models/chemtox/arch_v2_guard.py` | `c9620f372a471446721fca805b700483bda534ab7e99912fb6ce26d73f3268e9` |
| `sentinel/models/chemtox/arch_v2_h4.py` | `a21e517ad47c8f7c975c094913f0d57b3af66c7d73b5e44a5a1364982e768865` |
| `scripts/eval/chemtox_arch_v2_h4_report.py` | `a1267729f39d09c60c1eeaeb51973722dc8977a0fbafda22264df7211795092b` |
| `tests/test_chemtox_arch_v2_guard.py` | `61e26e7bf9231293b6b6507281f4a22ebcea770463fc91570597e8b6556ceff9` |
| `tests/test_chemtox_arch_v2.py` | `17065211bf508e79ace19303675caa9b877820208b770db4324c9420f69f240b` |
| `tests/test_chemtox_arch_v2_power.py` | `b9af9b24cf4829e7a629f5883f3d470868a95170fde79e898ffedcab2481f5b4` |
| `docs/preregistration/chemtox_arch_v2_pins.candidate.json` | `3accb83704aabb62eecce078fddcdbbedcf05d4ee9f010760178c41a9add72d6` |

Unchanged, as B1 requires (re-computed by the lead):

| file | sha256 |
|---|---|
| `docs/preregistration/chemtox_arch_v2_amendment.md` | `fb52cd4b072fa19c8c1e19b9fe1374a0e8abb71d9492eff764f2034cb26d293b` |
| `docs/preregistration/chemtox_arch_v2_lead_rulings.md` | `553be413bef252e5900023dc1dc983dd8f228a5825194ca70505cf6649c7f060` |
| `sentinel/models/chemtox/arch_v2.py` | `39ecc9cc39fb1c945a843ffa8e99d38eb73c3dce378d24d8a4566550e64fe980` |
| `sentinel/models/chemtox/arch_v2_power.py` | `0841903004361e52d0010b8be8e2c38745195e8d0629b0dd7e7c0c01243d4f7a` |

## 2. Rulings

- **G-1 (F1, blocking).** `verify_freeze()` must require the lead. It refuses unless ALL of the following hold:
  (a) the pins file has no write bit; (b) the registry holds a row of workstream `chemtox`, experiment
  `chemtox/prereg_snapshot`, with `extra.written_by == "lead"` and `extra.pins_sha256` equal to the sha256 of the
  pins file on disk (the two-source pattern of C-1); (c) the pins hold every key of `PINNED_REPO_FILES` and every
  pinned data path, each with a 64-hex value; (d) the pins hold no unknown entry and no null value, the
  protected-labels path excepted (G-6). Every reader writes the pins sha256 into the report it writes. Each
  condition has a test. M4 writes neither the pins file nor any row of that experiment (C-1).
- **G-2 (F2, blocking).** No test may read a real protected table, before or after the pins exist. A test that
  calls a reader on the real paths first asserts that the pins file is absent and skips, with the reason, when it
  is present; the skip comes before any reader call. The `aux_sandbox` fixture redirects `G.H4_RESULTS` and
  `G.H4_TESTS` as well. The tests that describe the pre-pins state skip once the pins exist; they are not deleted
  and are not made to pass by a read.
- **G-3 (F3, blocking).** H4 applies ALL SEVEN exclusions of amendment §10.2, in the stamped order, each with its
  own counter: the amendment says the list applies to H2, H3, H4, H5 and any future auxiliary target. Ruling N1
  (the pinned JSON reports govern over the prose) settles which count is cited; it does not remove a rule.
  `load_h4_rows()` gains the class join and both spent-id cuts. The pre-stamp H4 counts (10,050 / 2,984, measured
  with four steps by the old report script) are marked superseded; H4 is re-measured once, through the guard,
  after the pins exist, and the lead compares the new counts with the pinned v2 report. `load_h4_rows()` gets
  synthetic tests for its happy path, each of the seven exclusions, the T6 window, its re-check and its report.
  Its re-check is separate code from its exclusions (G-9).
- **G-4 (F4).** The guard itself joins the auxiliary target for exactly the `result_id`s it returns, after the
  re-checks, from the pinned `ecotox_aquatic.parquet` (H2, H3) or from the pinned raw tables (H4). No arm reads a
  label any other way. H5 (ACR) is computed from rows returned by `load_training_pool()` only, as amendment §10.2
  says; no guarded H3 row and no other row enters it. The revised addendum states, per reader, the columns
  returned and the targets refused.
- **G-5 (F5).** The single-look rule must survive a different runs root. `make_pins()` records the resolved runs
  root (the data root is already fixed by the path keys of the pins), and `verify_freeze()` refuses a different
  one. In addition `authorise()` records its claim in the registry before any sealed row is read, in a form that
  makes every later `authorise()` for that block refuse from any runs root, and that does not block the later
  steps of the same look. M4 chooses the form of the row (it is not a row of `chemtox/prereg_snapshot`); the delta
  review checks it. With tests, including the copied-FREEZE case of the review (probe P6).
- **G-6 (F6).** The protected-labels artefact stays (B1 (v)), but it is no longer trusted by path.
  `build_protected_labels()` refuses a file that already exists; it records the sha256 of the file it wrote in a
  write-once marker and in `AUTHORISED.json`; `load_test_block()` verifies that sha256 and asserts that the
  artefact's target equals the target of the pinned ecotox frame for every returned `result_id`. With tests for a
  planted and for a swapped file (probes P7, P7b).
- **G-7 (F7).** Every input that decides which rows come back is pinned by hash and written into every exclusion
  report: `prospective.parquet`, the ADORE split assignments, the raw `results.parquet` and `tests.parquet`,
  besides the split, the cluster table, the spent-id list and `ecotox_aquatic.parquet`. The reader compares the
  spent-ADORE set it derives from the registry with the pinned spent-id list and refuses on any difference. The
  lead sets the four data files 0444 when the pins are written; M4 does not rewrite them. Pinned code at the
  re-pin, besides the files already in `PINNED_REPO_FILES`: `scripts/eval/chemtox_arch_v2_h4_report.py`,
  `sentinel/models/chemtox/prospective.py`, the three test files of §1, the revised addendum (G-12), and the
  configuration, scorer, report-template, arm and comparator files that M4 lists with their sha256 in its report
  (lead message of 2026-11-11T01:08:38Z). `sentinel/core/paths.py` is shared by all workstreams and is not pinned;
  G-5 covers what it decides. The re-pin adds entries; it changes no hash of §1.1 of the stamped rulings. A later
  change to a pinned file needs a new lead-written pins file and row; the superseded pins file is kept.
- **G-8 (F8).** No report is overwritten. Each exclusion report and each H4 report is written with exclusive
  creation under a fixed directory of the runs root, with a counter in its name (as `ABORTED.k.json`), mode 0444,
  and holds the pins sha256, the time, the process id and the input hashes of G-7. The H4 script takes no output
  path from the caller.
- **G-9 (F9).** The re-check fails closed and covers all seven exclusions for H2, H3 and H4: a surviving row whose
  chemistry maps to no hold-out unit raises (it is not only counted); the T6 window, the T2 reserved classes and
  both spent-id lists are re-checked after the frozen loader's own filters, in separate code. Tests carry
  `T6a_test` and `T6b_test` rows, an H2 case, and held-out chemistry written in other forms (leading blank,
  trailing `.0`, leading zero, hyphenated, missing value, unknown CAS), each of which must raise.
- **G-10 (F10).** The revised addendum lists the five functions of the guard that read a protected table
  (`load_training_pool`, `build_protected_labels`, `load_test_block`, `load_auxiliary_rows`, `load_h4_rows`) and
  names the importable readers outside the guard (`arch_v2.load_usable_acute`, `build_universe`, `build_split`,
  `auxiliary_rows` called directly; `aux.ecotox_auxiliary`; `prospective.curate_ecotox`, `augmentation_rows`). Any
  use of one of those for T1/T6 work is a breach under N2 and voids the block concerned. The environment switch
  `M4_ALLOW_DIRECT_ECOTOX_READ` is removed: the 7 test items it opened stay in the file and skip always, with the
  reason; what they asserted is asserted by the reader's re-check on every real call (G-9) and is tested on
  synthetic frames. This replaces the last sentence of item (c) of ruling N2-A (annex).
- **G-11 (F11).** Ruling N2-A was given by message only. It is reproduced in the annex, and this file with its
  registry row is its lead-written record. The lead registers the addendum that is finally accepted, with its
  sha256, in the row written with the pins (C-2, C-7).
- **G-12 (F13).** Addendum 01 at sha256 `9a195e64…9f2e` is kept unchanged as the reviewed text and is not
  registered. M4 writes `docs/preregistration/chemtox_arch_v2_addendum_01_rev1.md`, which replaces it in full and
  opens with a list of every change from the reviewed text; the sentence that the H4 counts came from the same
  frozen loader is corrected. If the pre-rewrite v2 H4 script (sha256 `5653546f…` in `SHA256SUMS_v2`) can be
  recovered byte for byte it is restored beside the v1 copy; if not, the addendum says so. It is not rewritten
  from memory.
- **G-13 (F12, F14).** Noted. The candidate pins reproduce from `make_pins()`; byte identity shows nothing about
  the writer, which is why G-1 is needed. Modules written during the review (`arch_v2_arms.py`,
  `arch_v2_scorer.py` and later ones) do no file reads; the re-review checks that by search, and that no arch-v2
  module other than the guard opens a data file.

## 3. Order of events from here

1. M4 makes the changes of G-1 to G-10 and G-12, each with a test, on synthetic frames only. `arch_v2.py`,
   `arch_v2_power.py`, the amendment, the stamped rulings, the split, the cluster table and the spent-id list keep
   the hashes of §1.1 of the stamped rulings.
2. M4 regenerates the candidate pins and sends ONE report: sha256 of every file to be pinned, the revised
   addendum, the test files and results. It then edits none of those files.
3. The lead commissions one independent delta review of the changed guard, tests and revised addendum, with the
   mutation set of the first review re-run.
4. If it passes, the lead writes the pins file (0444), sets the data files of G-7 0444, and writes the registry
   row that carries the pins sha256 and the addendum sha256. From that moment S2 opens work on `train_pool`.
5. Until then: code and synthetic tests for arms and comparators, configuration files, licence checks, and the T2
   track under its own plan. No real protected row is read by anything.

## Annex — ruling N2-A as sent (lead message to M4, transcript time 2026-11-11T00:38:21Z = 19:38 EST; 3,010 characters; verbatim)

```text
Lead to M4 (19:39 EST): guard report received (guard 9bd7d60a…c066e, tests f314f53c…7d76, candidate pins 3eeac280…d51b — all three hashes re-computed by me and equal). Ruling on your point 1 below; I hold the independent guard review until it is done, so the guard is reviewed and pinned once, not twice.
RULING N2-A (auxiliary rows). You read N2 correctly: as stamped, arch_v2.auxiliary_rows() and scripts/eval/chemtox_arch_v2_h4_report.py are direct reads of ecotox_aquatic.parquet and may not run. No exemption. Fix by addendum under C-7, without touching arch_v2.py or arch_v2_power.py (hashes of §1.1 stay):
(a) Write docs/preregistration/chemtox_arch_v2_addendum_01.md (new, dated, short). It proposes one more sanctioned reader in the guard and nothing else: N2's list becomes load_training_pool(), build_protected_labels() and the new guarded auxiliary loader. State the stamped text, the collision, the change, and that no bar, split, block, comparator or statistic changes.
(b) Guard: add load_auxiliary_rows(target, track) to arch_v2_guard.py. It must: refuse unless verify_freeze() passes (so nothing runs before the pins file); call arch_v2.auxiliary_rows() with every path fixed to the pinned constants (no caller-supplied ecotox/prospective/registry/splits/cluster_table path); after the call, assert that no returned result_id is a result_id of T1_test, T6a_test or T6b_test, that no returned (test_cas, latin_name, duration_h) cell is a sealed cell, and that no returned test_cas maps to a held-out hold-out unit (re-checked independently of auxiliary_rows' own filters; hard error on any hit, counts only in the message); write the exclusion report (counts only) next to the pool outputs. H4 (BCF-family frame): the same guarded route — a guard function that owns the read; the h4 report script calls it and reads no parquet itself.
(c) Tests on synthetic/temporary paths only: the wrapper refuses without pins; refuses a caller path; a planted sealed result_id, sealed cell and held-out-unit chemistry each raise; the live guard still refuses everything. The 7 gated AUX-LEAK tests stay gated behind the environment variable and stay skipped on real data until the pins file exists.
(d) Do not run auxiliary_rows() or the h4 report on real data now. Regenerate the candidate pins. Report: addendum path + sha256, new guard sha256, test files + results, and the unchanged §1.1 hashes.
Then I commission ONE independent review of guard + addendum + tests, write the pins file and the registry rows. Nothing else changes in B1/C-1/C-2.
Your point 2 (reviews README hash 4c4e9ba9…): mine to fix; noted.
Correction to your GPU note: GPUs 1 and 3–9 carry this project's own jobs (M3 TabPFN, M5); only GPU 2 is another user's. GPU 0 is the idle one. Moot for T2 as you say.
T2 repair of chemeleon_lightgbm shards 4 and 20 of 32: accepted as described (no setting changed, merge validates by index coverage); record it in the T2 run log with the rc=124 cause. Then go on with item 2. No git commits.
```
