# Review: M4 arch-v2 guard, addendum 01 and tests

Reviewed 2026-11-10 19:50:20 EST to 2026-11-10 20:10:49 EST. Independent, read-only. All probes and mutations ran on synthetic frames in the scratch folder.

## Verdict: `fix_then_rereview`

No sealed row reaches a training or auxiliary set through the two new readers on the synthetic cases tested, and the three H2/H3 re-checks are real. But the gate itself (verify_freeze) can be passed by anyone with a two-key JSON file, one test will read real data as soon as the pins exist, the H4 reader is untested and applies four of the seven stamped exclusions while the addendum says seven, and the H2/H3 reader returns no target. These need guard, test and addendum changes, which change the hashes the pins would freeze.

## Hashes

All 12 files had the same sha256 at the start (19:50:20 EST) and at the end (20:08:13 EST); the 8 files under review equal the values the lead gave.

| file | sha256 |
|---|---|
| `docs/preregistration/chemtox_arch_v2_amendment.md` | `fb52cd4b072fa19c8c1e19b9fe1374a0e8abb71d9492eff764f2034cb26d293b` |
| `docs/preregistration/chemtox_arch_v2_lead_rulings.md` | `553be413bef252e5900023dc1dc983dd8f228a5825194ca70505cf6649c7f060` |
| `sentinel/models/chemtox/arch_v2.py` | `39ecc9cc39fb1c945a843ffa8e99d38eb73c3dce378d24d8a4566550e64fe980` |
| `sentinel/models/chemtox/arch_v2_power.py` | `0841903004361e52d0010b8be8e2c38745195e8d0629b0dd7e7c0c01243d4f7a` |
| `docs/preregistration/chemtox_arch_v2_addendum_01.md` | `9a195e643c420c39b7fd4451053fd3f482e738441d3ca626e85659cd342c9f2e` |
| `sentinel/models/chemtox/arch_v2_guard.py` | `c9620f372a471446721fca805b700483bda534ab7e99912fb6ce26d73f3268e9` |
| `sentinel/models/chemtox/arch_v2_h4.py` | `a21e517ad47c8f7c975c094913f0d57b3af66c7d73b5e44a5a1364982e768865` |
| `scripts/eval/chemtox_arch_v2_h4_report.py` | `a1267729f39d09c60c1eeaeb51973722dc8977a0fbafda22264df7211795092b` |
| `tests/test_chemtox_arch_v2_guard.py` | `61e26e7bf9231293b6b6507281f4a22ebcea770463fc91570597e8b6556ceff9` |
| `tests/test_chemtox_arch_v2.py` | `17065211bf508e79ace19303675caa9b877820208b770db4324c9420f69f240b` |
| `tests/test_chemtox_arch_v2_power.py` | `b9af9b24cf4829e7a629f5883f3d470868a95170fde79e898ffedcab2481f5b4` |
| `docs/preregistration/chemtox_arch_v2_pins.candidate.json` | `3accb83704aabb62eecce078fddcdbbedcf05d4ee9f010760178c41a9add72d6` |

## Findings

### F1 [blocking]

**Where.** sentinel/models/chemtox/arch_v2_guard.py:273-299 verify_freeze(); :260-270 make_pins()

**What.** verify_freeze() does not require the lead. It needs only that a file exists at docs/preregistration/chemtox_arch_v2_pins.json whose 'data' object has the ecotox path and the labels path as KEYS. It does not check the file mode, any registry row or written_by, that the PINNED_REPO_FILES entries are present, that the split is covered, or that any hash is non-null. A workstream agent under the same Unix user passes it with one write_text of a two-key JSON, or by copying the candidate (byte-identical to what the lead would write), or without touching the repo by setting G.REPO in-process. Audit control that remains: the file's mtime/ctime against the lead's registry row, and (looks only) pins_sha256 in AUTHORISED.json. load_training_pool() leaves no record at all; the aux exclusion report records split and ecotox sha but not the pins sha.

**Evidence.** RUN (synthetic sandbox, probes.log): P1a pins={'data':{ecotox:null,labels:null}}, mode 0666, no registry row, no freeze record -> verify_freeze passed, load_training_pool returned 2 rows, load_auxiliary_rows returned 3 rows; after rewriting the split to all train_pool, load_training_pool returned ids 1,2,3,4 (3,4 were sealed). P1b make_pins() dump written by the test process, mode 0666 -> passed. P1c pins with no split entry and empty 'files' -> passed. P1d honest pins + A.SPLIT_PATH pointed at an all-train_pool copy -> passed and returned the sealed ids (A.ECOTOX redirect IS refused). READ: C-2 says the lead writes the pins file (0444) and a registry row with its sha256; the guard checks neither. The repo's own _write_pins test helper (tests/test_chemtox_arch_v2_guard.py:81-91) is a non-lead writer that passes.

**Proposed ruling.** Before pins: verify_freeze() must (a) refuse a pins file with any write bit, (b) require a chemtox/prereg_snapshot registry row with extra.written_by=='lead' and extra.pins_sha256 == sha256(pins file) (the C-1 two-source pattern), (c) require every PINNED_REPO_FILES key and str(A.SPLIT_PATH), str(A.ECOTOX) with 64-hex values, (d) refuse unknown/extra-null entries except the labels path. Each reader should also write the pins sha into its report. Each with a test. This is still 'by rule and audit' (C-1), but it makes a workstream-written pins a deliberate forgery of a lead registry row instead of a one-line file write.

### F2 [blocking]

**Where.** tests/test_chemtox_arch_v2_guard.py:781-787, :523-534, :568-613; tests/test_chemtox_arch_v2.py:281-288

**What.** Once the lead writes the pins file, running the pinned test suite performs a real read. test_guard_still_refuses_the_new_readers_on_real_paths calls G.load_auxiliary_rows('H3','T1') on the REAL paths as its first statement inside pytest.raises: with pins present it will read ecotox_aquatic.parquet through the guard, write/overwrite $RUNS/chemtox/arch_v2/aux_exclusions/aux_H3_T1.json, and then fail 'DID NOT RAISE'. test_live_guard_still_refuses_everything and test_the_stamp_is_partial_so_nothing_is_unlocked fail by design after pins (they stop before a read). Separately, the aux_sandbox fixture does not redirect G.H4_RESULTS / G.H4_TESTS although its comment says it does: the only thing between test_aux_reader_refuses_without_pins and a read of the real raw ECOTOX tables is the verify_freeze() line under test.

**Evidence.** RUN: P2 (sandbox, pins file only, no FREEZE): load_training_pool, load_auxiliary_rows('H3','T1') and load_h4_rows('T6') all returned rows and wrote aux_H3_T1.json / aux_H4_T6.json; none raised. Mutation M5 (verify_freeze removed from load_h4_rows) made test_aux_reader_refuses_without_pins fail only because my run pointed SENTINEL_DATA at an empty scratch folder; on default paths the same mutation would open the real results.parquet. READ: fixture lines 568-613.

**Proposed ruling.** Fix before pins: real-path tests must assert `not pins_path().exists()` (or skip when it exists) BEFORE calling any reader; aux_sandbox must monkeypatch G.H4_RESULTS and G.H4_TESTS. Lead to rule whether the tests are pinned (C-2: 'the guard, its tests and the pins file') - they are absent from PINNED_REPO_FILES and from the candidate - and how 'pre-pins state' tests are retired after the pins exist.

### F3 [blocking]

**Where.** sentinel/models/chemtox/arch_v2_guard.py:774-838 load_h4_rows(); addendum_01 section 3 and section 4; amendment section 10.2 (lines 880-898)

**What.** The H4 reader applies four of the seven section-10.2 exclusions (held-out unit, not in split, same (CAS, species) cell, T6 window). It does not apply T2 reserved classes, spent prospective ids or spent ADORE ids; it never reads `class`. Amendment 10.2 says the list applies to H2, H3, H4, H5. Addendum section 4 says 'the seven AUX-LEAK exclusions ... are those of the stamped auxiliary_rows()' and that the H4 counts 'were measured before the stamp with the same frozen loader' - false for H4: the frozen loader raises on H4 (arch_v2.py:592-593); the H4 counts came from the old report script, whose pinned JSON also shows only the four steps. So the guard matches the pinned JSON but not the amendment text, and the addendum hides the difference. No test executes load_h4_rows past verify_freeze(): none of its exclusions, its re-check, or its happy path is tested, and the repo fixture's frame lacks species_number so the function could not run there. For H4 the 'independent' held-out-unit re-check uses the same map as the exclusion (not independent).

**Evidence.** RUN: P3 (synthetic): H4 T1 returned ids ['998','999','b1','b5','b8'] - the Bivalvia-species row (b5) and the rows whose result_id equals a spent prospective id (999) and a spent ADORE id (998) were kept; steps had only removed_held_out_unit_chemistry=1, removed_chemistry_not_in_split=1, removed_same_cell_as_sealed_row=1; the H3 path on the same world reported removed_t2_reserved_classes=1. Mutations M7 (no held-out exclusion), M8 (no H4 re-check), M9 (no cell exclusion), M14 (no T6 window) all left 60/60 tests passing. READ: h4_bcf_join_qc_v2.json steps (counts only).

**Proposed ruling.** Lead to rule: either (a) H4 applies all seven (guard gains the class join and both spent-id cuts; 10,050 / 2,984 are then re-measured and will change), or (b) rule in writing that the three are not applicable to BCF rows and why (N1: the pinned JSON governs). Either way the addendum text is corrected, and load_h4_rows gets synthetic tests for every exclusion, the re-check and the T6 window before pins.

### F4 [ruling_needed]

**Where.** sentinel/models/chemtox/arch_v2.py:673; sentinel/models/chemtox/arch_v2_guard.py:748-771; addendum_01 section 1 and 3

**What.** The H2/H3 reader returns no target. The frozen auxiliary_rows() drops log10_conc_mol_L before returning, so load_auxiliary_rows() hands back keys and covariates only. load_training_pool() returns pool rows only. There is therefore no sanctioned route by which ted / R2 / R3 / mmoe / A4 can obtain an H2 or H3 label, and no reader at all for H5 (ACR), although addendum section 1 names all of them. The addendum does not say what the readers return or refuse. As written, the first auxiliary fit needs either an unsanctioned read of ecotox_aquatic.parquet or an addendum 02 and a second re-pin.

**Evidence.** RUN: P3 'log10_conc_mol_L' in returned H3 columns -> False; H2 -> False. H4 frame does carry log10_bcf (its own target, from the raw table). READ: arch_v2.py:673.

**Proposed ruling.** Rule now, in the same re-pin: the guard joins the auxiliary target itself for exactly the returned result_ids (after the re-checks), and the addendum states per reader the columns returned and the targets refused (H5: computed from load_training_pool() rows plus guarded H3 rows, or struck).

### F5 [ruling_needed]

**Where.** sentinel/models/chemtox/arch_v2_guard.py:120 RUNS, :149-163, :420-450; sentinel/core/paths.py:22

**What.** A second look is possible through a different runs root. All single-look markers and the FREEZE path live under G.RUNS, which comes from env SENTINEL_RUNS at import and is not recorded in the pins. With a copy of the lead's FREEZE in a fresh runs root, authorise() succeeds again as long as the registry has no row for the block (always the case after an aborted look, and until the first look's rows are written). SENTINEL_DATA is different: an honest pins file refuses it (path keys no longer match).

**Evidence.** RUN: P6 (sandbox): look 1 aborted -> authorise() in the same root refused; with G.RUNS pointed at a fresh directory holding a copy of FREEZE, authorise() succeeded and load_test_block() returned the sealed rows a second time. Env probe: SENTINEL_RUNS moved G.RUNS, freeze_path and AUX_EXCLUSION_DIR; candidate pins contains no runs path.

**Proposed ruling.** Pin the runs root (make_pins records str(RUNS); verify_freeze refuses a different one), or write the AUTHORISED claim to a second fixed location (a registry row at authorise time). With a test. Not needed for pool work, but the guard is frozen by this re-pin.

### F6 [ruling_needed]

**Where.** sentinel/models/chemtox/arch_v2_guard.py:269, :283-287, :398-417, :479-494; candidate pins 'data' labels = null

**What.** 'labels = null' is not sound for the look stage. B1 (v) asks for the labels artefact to be pinned 'once built'; nothing does that. verify_freeze() skips null entries, require_labels=True is never used by the guard, build_protected_labels() returns any pre-existing file at LABELS_PATH unverified, and AUTHORISED.json records no labels hash. A labels file planted before the look, or swapped after it is built, is served by load_test_block().

**Evidence.** RUN: P7 labels planted before authorise -> verify_freeze and authorise passed; load_test_block(columns=[...]) returned the planted values (0.0, 0.0 instead of -6, -7). P7b labels rewritten after build -> verify_freeze and require_authorised passed; planted values returned. With columns=None the frame carries both the pinned-ecotox target and '<target>_protected'.

**Proposed ruling.** build_protected_labels() must refuse a pre-existing artefact it did not build in this authorisation, record its sha256 in a write-once marker, and load_test_block() must verify it (or take the target only from the pinned ecotox frame and drop the artefact). For pool work 'null' is harmless; decide now because the guard is frozen by the pins.

### F7 [ruling_needed]

**Where.** sentinel/models/chemtox/arch_v2_guard.py:108-118 PINNED_REPO_FILES, :660-661, :761-763, :768-769; arch_v2.py:115-117

**What.** Inputs that decide which rows come back are not pinned and not recorded. prospective.parquet (0664), adore/split_assignments.parquet (0664), the raw results.parquet and tests.parquet (0664) and the live registry (spent ADORE set is derived from it at call time, not from the pinned spent-id list) are constants but have no hash in the pins or in the exclusion report. The addendum's phrase 'pinned module constants' is inaccurate for them. Also absent from the pins: the three test files and scripts/eval/chemtox_arch_v2_h4_report.py (C-2 says the re-pin adds the guard and its tests), sentinel/core/paths.py and prospective.py (imported by arch_v2), and the new arch_v2_arms.py / arch_v2_scorer.py.

**Evidence.** RUN: P5 rewriting prospective.parquet between two calls changed the returned set from ['10','11','13'] to ['11'] with verify_freeze still passing; report hash keys are only ecotox_sha256 and split_sha256. Candidate pins: 9 files + 3 data entries (byte-identical to make_pins(), see F12).

**Proposed ruling.** Add the four data files to make_pins 'data' (hash only) and their sha256 to every exclusion report; have the reader compare the live spent-ADORE set with the pinned spent-id list and refuse on a difference; lead to say which tests/scripts are pinned.

### F8 [ruling_needed]

**Where.** sentinel/models/chemtox/arch_v2_guard.py:737-745 _write_exclusion_report(); scripts/eval/chemtox_arch_v2_h4_report.py:45-48

**What.** The 0444 exclusion report is written where claimed but is silently overwritten on every call: an existing report is chmod-ed to 0644 and rewritten, with no O_EXCL, no version and no history. The H4 script writes h4_bcf_join_qc_v3.json to a caller-supplied --out, not 0444, overwriting.

**Evidence.** RUN: P5 two calls -> one file aux_H3_T1.json, sha 179f1e475c8a then ebee819411d7, mode 0444 both times. Mutation M11 (no chmod 0444) is caught by a test.

**Proposed ruling.** Write each report content-addressed or with O_EXCL plus a counter (as ABORTED.k.json does), and include pins sha, time and pid.

### F9 [ruling_needed]

**Where.** sentinel/models/chemtox/arch_v2_guard.py:702-734 _assert_no_leak(); :678-688 _sealed_context()

**What.** The re-check is real for the three stated properties on H2/H3, but it is not fail-closed and covers three of seven exclusions. Chemistry that does not map to a hold-out unit is only counted (chemistry_unmapped), not refused; the T6 window, T2 classes and spent ids are not re-checked. Keys are compared by astype(str) with no normalisation, the same as the frozen loader, so the re-check is independent in logic but not in key handling. On real data the types line up (ecotox and split: result_id/test_cas string, duration_h double; cluster CSV test_cas int64 -> str), so this is a hardening gap, not an observed leak.

**Evidence.** RUN: mutations M1/M2/M3/M13 (each re-check deleted) each made the matching planted-leak test fail, so those tests do exercise the re-checks. P4a: a T6a_test cell in a retained unit is caught. P4b: planted held-out chemistry written ' 400', '400.0', '0400', '40-0', None, or unknown CAS -> NOT refused, chemistry_unmapped=1. M10 (held units from T1_test only) and M12 (sealed context without T6a_test) left 60/60 passing: no test has a T6a_test or T6b_test row.

**Proposed ruling.** Raise when chemistry_unmapped > 0; add the T6 window re-check; add tests with T6a_test and T6b_test rows and an H2 case.

### F10 [ruling_needed]

**Where.** addendum_01 section 3; sentinel/models/chemtox/arch_v2_guard.py:479-494, :50-55, :650-655; arch_v2.py:242-320, :431; aux.py:181-203; prospective.py:202-212, :653; tests/test_chemtox_arch_v2.py:29-33

**What.** The list of readers is not what the addendum says. Addendum: 'becomes three' (item 3 is two functions); guard docstring and H4 script: 'four'. load_test_block() (guard:489) also reads ecotox_aquatic.parquet and is in neither N2 nor the addendum. Other direct readers stay importable with no guard: frozen arch_v2.load_usable_acute / build_universe / build_split, and earlier-track aux.ecotox_auxiliary, prospective.curate_ecotox and prospective.augmentation_rows (which return labelled rows of the whole frame). The env switch M4_ALLOW_DIRECT_ECOTOX_READ=1 still un-skips 7 test items that call the frozen loader on real data. All of these are closed by rule N2 only.

**Evidence.** READ + grep over sentinel/, scripts/, tests/, configs/ (see q3_other_readers in review.json). No arch-v2 module or script in the tree calls any of them today: arch_v2_arms.py and arch_v2_scorer.py contain no read_parquet/open; chemtox_arch_v2_h4_report.py calls G.load_h4_rows only; arch_v2_h4.py imports re/numpy/pandas only.

**Proposed ruling.** State the full list in the addendum (five functions incl. load_test_block), name the unguarded importable readers as breaches if used for T1/T6 work, and remove or lead-gate the env switch.

### F11 [ruling_needed]

**Where.** results/registry.jsonl; docs/handoff/chemtox.md:1344; addendum_01 header

**What.** Ruling N2-A exists on disk only in M4-written files (addendum, guard, h4 module, script, tests, handoff). The registry has one chemtox/prereg_snapshot row (written_by lead, the stamp); it does not mention addendum 01, its sha 9a195e64, the guard sha or N2-A. C-7 requires a lead-written registry row for an addendum.

**Evidence.** RUN: registry scan (counts and names only): 73,614 rows; 0 rows with experiment starting chemtox_arch_v2; 0 rows whose split names T1_test/T6a_test/T6b_test; 1 row of chemtox/prereg_snapshot.

**Proposed ruling.** The lead records N2-A and the addendum sha in a lead-written file and registry row at the time of the pins, as C-2/C-7 say.

### F12 [note]

**Where.** docs/preregistration/chemtox_arch_v2_pins.candidate.json

**What.** The candidate reproduces byte-for-byte from make_pins(), and every hash in it equals the file on disk and the values in rulings section 1.1.

**Evidence.** RUN: json.dumps(make_pins(), indent=1, sort_keys=True)+'\n' == candidate bytes (1,974 bytes, sha 3accb837...); make_pins() only hashes files (guard:260-270).

**Proposed ruling.** None. Note that byte-identity means the file content cannot show who wrote it (see F1).

### F13 [note]

**Where.** addendum_01; $RUNS/chemtox/arch_v2/SHA256SUMS_v2:11; $RUNS/chemtox/arch_v2/h4_bcf_join_qc.py

**What.** Addendum facts otherwise check out: parent hashes, the six section-1.1 hashes, the section-10.1 counts (match the pinned JSON reports), the three earlier-track file hashes, and the state section. It moves no bar, split, block, comparator, statistic, cap or outcome wording (by reading). Small points: file mode is 0664 (not yet 0444); mtime 19:39:45 vs 'Dated 19:40'; 'the four tests' are 4 functions / 7 items. The claim that the H4 rules moved 'unchanged' cannot be verified: the pre-rewrite script (sha 5653546f... in SHA256SUMS_v2) is no longer on disk. Against the kept v1 script the rules are the same except DIMENSIONLESS gains '<NA>'; the new steps drop the v2 report's shared_with_train_pool_chemicals counter.

**Evidence.** READ and sha256sum. Not run on real data.

**Proposed ruling.** Ask M4 to restore the superseded v2 script beside the v1 copy; when H4 is re-measured through the guard, the lead compares with the pinned v2 counts.

### F14 [note]

**Where.** sentinel/models/chemtox/arch_v2_arms.py, arch_v2_scorer.py, tests/test_chemtox_arch_v2_arms.py, tests/test_chemtox_arch_v2_scorer.py

**What.** New arch-v2 modules and tests appeared during this review (mtimes 19:55-19:59 EST). They are not among the files under review, are not in the pins, and were not reviewed. By grep they do no I/O and call no reader. None of the eight files under review changed.

**Evidence.** ls / grep at 20:04 EST; sha256 arch_v2_arms.py 264e3bba..., arch_v2_scorer.py 2205246a... (still being edited).

**Proposed ruling.** Decide whether they belong in the pins or only in the per-look import-closure hash.

## Tests

| file | claimed | observed (start) | observed (end) |
|---|---|---|---|
| `tests/test_chemtox_arch_v2_guard.py` | 62 passed | 62 passed | 62 passed |
| `tests/test_chemtox_arch_v2.py` | 27 passed, 7 skipped | 27 passed, 7 skipped | 27 passed, 7 skipped |
| `tests/test_chemtox_arch_v2_power.py` | 18 passed | 18 passed | 18 passed |

CUDA_VISIBLE_DEVICES= OMP/MKL/OPENBLAS threads 4, -p no:cacheprovider, --basetemp in scratch, python -B, no SENTINEL_* or M4_* env

The 7 skips are the 4 functions gated by `M4_ALLOW_DIRECT_ECOTOX_READ`, which was not set.

### Mutation runs (scratch copies of the guard, 60 tests collected)

| mutation | result |
|---|---|
| M0_baseline | 60 passed |
| M1_no_result_id_recheck | 2 failed (killed) |
| M2_no_cell_recheck | 1 failed (killed) |
| M3_no_unit_recheck | 1 failed (killed) |
| M4_aux_no_verify_freeze | 1 failed (killed) |
| M5_h4_no_verify_freeze | 1 failed (killed) |
| M6_no_caller_path_check | 1 failed (killed) |
| M7_h4_no_held_out_exclusion | 60 passed (SURVIVED) |
| M8_h4_no_recheck | 60 passed (SURVIVED) |
| M9_h4_no_cell_exclusion | 60 passed (SURVIVED) |
| M10_held_units_T1_only | 60 passed (SURVIVED) |
| M11_report_not_0444 | 1 failed (killed) |
| M12_sealed_ctx_drops_T6a | 60 passed (SURVIVED) |
| M13_aux_no_recheck_at_all | 4 failed (killed) |
| M14_h4_no_t6_window | 60 passed (SURVIVED) |
| M15_verify_freeze_ignores_data_hashes | 1 failed (killed) |

mut/mutplugin.py swaps a mutated copy of the guard into sys.modules; repo file untouched; SENTINEL_DATA/RUNS pointed at empty scratch dirs; the two real-path tests deselected (60 collected)

### Claimed behaviours with no test

- load_h4_rows: happy path, each of its exclusions, its T6 window, its re-check, its report
- T6a_test and T6b_test rows in the sealed context (sandbox split has T1_test only)
- H2 target through load_auxiliary_rows
- refusal or handling of unmapped / NaN / differently formatted keys in the re-check
- exclusion report overwrite behaviour
- environment-variable redirection (SENTINEL_RUNS, SENTINEL_DATA)
- pins file written by a non-lead, with a write bit, incomplete, or with null hashes
- labels artefact pre-existing or changed (only the require_labels flag is tested)
- the H4 report script end to end

## Other readers (question 3)

| where | class | note |
|---|---|---|
| sentinel/models/chemtox/arch_v2_guard.py:409,471,489,697,793 (+761 via frozen loader) | guarded | build_protected_labels, load_training_pool, load_test_block, _sealed_cells, load_h4_rows, load_auxiliary_rows |
| sentinel/models/chemtox/arch_v2_guard.py:791-792 raw results.parquet/tests.parquet | guarded | load_h4_rows only; files unpinned (F7) |
| sentinel/models/chemtox/arch_v2.py:595 auxiliary_rows | frozen arch-v2 code; guarded when called via load_auxiliary_rows; direct call possible | direct callers in tree: only the 7 env-gated test items |
| sentinel/models/chemtox/arch_v2.py:248 load_usable_acute, :256 build_universe, :431 build_split | frozen arch-v2 code, unguarded, no caller in sentinel/scripts/tests | pre-stamp split builders; keys only, target dropped; hole by rule only |
| sentinel/models/chemtox/aux.py:181-203 ecotox_auxiliary; prospective.py:202-212 curate_ecotox, :653 augmentation_rows | frozen earlier-track code, unguarded, returns labelled rows | not imported by any arch-v2 arm today (arch_v2 imports prospective for parent_key helpers); hole by rule only |
| scripts/data/build_ecotox.py, build_chemicals.py, chemtox_build_prospective.py, chemtox_build_unseen_taxa.py | frozen earlier-track data builders | not arch-v2 entry points |
| scripts/eval/chemtox_arch_v2_h4_report.py | guarded | calls G.load_h4_rows only; no read_parquet |
| sentinel/models/chemtox/arch_v2_h4.py | no I/O | imports re, numpy, pandas; no read_*/open/Path |
| scripts/eval/chemtox_arch_v2_power_report.py:61-67 | unguarded read of the split (keys, no target), caller-supplied --split | pre-stamp power report |
| tests/test_chemtox_arch_v2.py:70,120-255 | test-only | reads real split and cluster CSV (keys); 7 items read ecotox via the frozen loader only if M4_ALLOW_DIRECT_ECOTOX_READ=1 |
| tests/test_chemtox_arch_v2_guard.py | test-only, synthetic | except the real-path refusal tests (F2) |
| tests/test_chemtox_models.py:425 | test-only | existence check |
| configs/chemtox/arch_v2.yaml:20 | config | names the split file only |

## State at review time (question 6)

- docs/preregistration/chemtox_arch_v2_pins.json: absent (start and end)
- docs/preregistration/chemtox_arch_v2_freeze_shas.json: absent
- $RUNS/chemtox/arch_v2/aux_exclusions/: absent
- $RUNS/chemtox/arch_v2/h4_bcf_join_qc_v3.json: absent
- $DATA/processed/chemtox_arch_v2/protected_labels_v2.parquet: absent
- block dirs / FREEZE / AUTHORISED / SCORED / ABORTED: none (look_state on all three blocks: exists False, registry_rows 0)
- registry: 73,614 rows; 0 with experiment starting chemtox_arch_v2; 0 with split naming T1_test/T6a_test/T6b_test; 1 chemtox/prereg_snapshot row (lead stamp)
- newest file in $RUNS/chemtox/arch_v2: SHA256SUMS_v2 14:53:37 EST (before the stamp)
- real loaders: refuse: 'no pins file ... not been stamped' (the passing real-path tests)

## Not checked

- Whether any process read ecotox_aquatic.parquet or the raw tables since the stamp: access times are not informative (relatime); only the absence of output artefacts was confirmed.
- Any behaviour on real data: no real loader was run past its refusal; all probes and mutations used synthetic frames.
- Value formats inside ecotox_aquatic.parquet, the split and the raw tables (CAS string form, species spelling): only schema types and row counts were read. The H4 same-cell exclusion compares raw ecotox test_cas with digit-normalised CAS; that they agree is inferred from the int64 cluster key and the 0.94 mapped fraction, not verified.
- The pre-rewrite v2 H4 script (not on disk), so 'rules moved unchanged' is unverified.
- arch_v2_arms.py, arch_v2_scorer.py and their tests (new during the review; grep only).
- tests/test_chemtox_arch_v2.py with M4_ALLOW_DIRECT_ECOTOX_READ=1 (forbidden).
- The statistical content of the amendment, the power module, and B1 (ii)-(iv) beyond what the probes touched.
- scored_pass / FREEZE two-source logic beyond probes P6 and P7; a per-block pre-test review is still required by C-2 (4).
- Whether gpboost_model.py is unchanged since the stamp (its current sha256 is cc3d1b26...; rulings leave it unsealed).
- Non-Python readers (notebooks, shell) outside sentinel/, scripts/, tests/, configs/.

## Evidence files (in this folder)

- `probes/test_probes.py`
- `probes.log`
- `mut/mutplugin.py`
- `mut/M*.log`
- `mut/M*.arch_v2_guard.py`
- `pytest_end_*.log`
- `hashes_start.txt`
- `hashes_end.txt`
