# SENTINEL-Lite (M7) — architecture round v2: lead rulings and stamp

**Written by the lead:** Wed 11 Nov 2026 1:01 PM EST.

**Status.** This file is set read-only (0444) at the stamp and is never edited. It is one of the four files named
in `sentinel/models/litev2/stamp.py` (`STAMPED_FILES`); the lead's registry row under workstream `M7-lite`,
experiment `lite/prereg_snapshot`, carries the sha256 of all four. Any correction is written in a new, dated file
and a new lead row, never by editing this one.

## 0. What is stamped

- The architecture-v2 amendment `docs/preregistration/lite_arch_v2_amendment.md` (DRAFT v0.3, sha256
  `2b369018d2fc2a340594790756b3edbef3b1de28a38cf6073eb5908111908a56`) is stamped **as written, byte for byte**.
  Its text is not edited. Where this file and the amendment differ, **this file wins**, and the superseded
  sentences are named here.
- Stamped with it, byte for byte: the counts-only label addendum `lite_arch_v2_label_addendum.md` (sha256
  `bcbd86de12ff116100f216f5c7c3c5de2e8f1de4d7bb567fd2badf5e12d29049`) and the deviations file
  `lite_arch_v2_deviations.md` (sha256 `23e6fd8705fe4bfc102df74160e46477c8c6b31b3ee38ff504370ed3fc00642c`), in the
  byte state the independent review read. The deviations file is append-only; an append after the stamp closes the
  guard until the lead stamps the new content.
- The two dated code-readings files are adopted as fixed records of what the code does:
  `lite_arch_v2_code_readings_20261110.md` (sha256
  `002d5d0822dab92d725467f59966b106b866c1c958855b801fef330aaf22a4d4`) and
  `lite_arch_v2_code_readings_export_20261110.md` (sha256
  `895dcefe314937bc21077e2c34d78cf2252fcee0a143b1ff4a4fb8957eb5454f`). Neither may change after the stamp without
  a stamped deviation.
- This stamp authorises no number. It opens the v2 fits of SENTINEL-Lite, in the order of §10, under the rulings
  below.

### 0.1 Files the lead read in full before writing this file, with their sha256 at writing

| File | sha256 | Mode at writing |
|---|---|---|
| `docs/preregistration/lite_arch_v2_amendment.md` | `2b369018d2fc2a340594790756b3edbef3b1de28a38cf6073eb5908111908a56` | 0o664 |
| `docs/preregistration/lite_arch_v2_label_addendum.md` | `bcbd86de12ff116100f216f5c7c3c5de2e8f1de4d7bb567fd2badf5e12d29049` | 0o664 |
| `docs/preregistration/lite_arch_v2_deviations.md` | `23e6fd8705fe4bfc102df74160e46477c8c6b31b3ee38ff504370ed3fc00642c` | 0o664 |
| `docs/preregistration/lite_arch_v2_code_readings_20261110.md` | `002d5d0822dab92d725467f59966b106b866c1c958855b801fef330aaf22a4d4` | 0o444 |
| `docs/preregistration/lite_arch_v2_code_readings_export_20261110.md` | `895dcefe314937bc21077e2c34d78cf2252fcee0a143b1ff4a4fb8957eb5454f` | 0o444 |
| `docs/preregistration/lite_protocol.md` | `24b695fc381e238a414d10a0c7ca24bfb41822da1a6b113e01741081d55a2cb4` | 0o664 |
| `docs/preregistration/lite_national_protocol.md` | `4ae319369a5998973e9caacb603ce53b77b07a4a58d16276ea5f9ad2fbf3d4b9` | 0o664 |
| `docs/preregistration/lite_split_national2b.json` | `03f0114a8683aa8fcc0d2b49901dc580c376ec3fdcdd2a8f41d1ee2242395663` | 0o664 |
| `docs/preregistration/lite_p2_lock.json` | `f3d4c63ebe503ac1774959bfcf943b45bc140883fe5b30933ee03f0cacca52be` | 0o444 |
| `$SENTINEL_RUNS/lite/counts_arch_v2/c2_teacher_scored_m2_test_lake_keys.csv` | `4786aa121eb673ced502d487fb4af714fff37027677677048d2024c960cfc699` | 0o444 |
| `$SENTINEL_RUNS/lite/counts_arch_v2/c2_teacher_scored_m2_test_lake_keys.meta.json` | `ea2b0b11bc0d804ef2701e9456a84e0050928769de997357d52242d35c2d5d94` | 0o444 |
| `$SENTINEL_RUNS/lite/counts_arch_v2/litev2_code_sha256_20261110.json` | `563894b93a9e27c0bbc3db37af16fbf49be2b55641c7e7a1e0d861c3fabcafe1` | 0o444 |

The independent pre-stamp review of the package is `scratchpad/review_m7v2_stamp/review.json` (sha256
`a7c68bb8c38bd4ff77c88459e26aa90ed8713dba75286ce71d0e78a178b11bdd`) and `review.md` (sha256
`141235bfd1908b9cdfe4a82a8ba1d62abdc2158f3e59e6af71961aeffbc2e43a`), written 2026-11-10T19:54:42-05:00, verdict
**stamp_with_lead_rulings**; its copy is filed read-only, with the same hashes, as
`docs/preregistration/reviews/arch_v2_20261110/M7v2_stamp_review.json` and `M7v2_stamp_review.md`. The earlier
verified pre-stamp review of the amendment is `docs/preregistration/reviews/arch_v2_20261110/M7_verify.json`.

## 1. Facts the lead established before the stamp

- **Code state.** The code-hash record `$SENTINEL_RUNS/lite/counts_arch_v2/litev2_code_sha256_20261110.json`
  (sha256 `563894b93a9e27c0bbc3db37af16fbf49be2b55641c7e7a1e0d861c3fabcafe1`, written by M7, 0444) lists 39 files.
  The lead re-hashed every one of them at the time in the header of this file: all 39 match. No file of the round
  may change between this file and the stamp row; a change voids the stamp until a new lead row exists.
- **Tests.** The lead ran the eleven `tests/test_litev2*.py` files on that code state: **120 passed, 0 failed, 0
  skipped** (pytest exit 0). The known-failing
  `tests/test_lite.py::test_scoring_hash_covers_scorer_import_closure` was run separately and still fails at
  `tests/test_lite.py:496`; it stays failing, as the record of deviation 10 (§9). This satisfies finding N-06.
- **Nothing of v2 has been fitted.** No v2 arm, comparator or re-run has been fitted on a real validation row.
  Immediately before this file was written, `python -m sentinel.models.litev2.stamp` reported `"stamped": false`,
  reason "not stamped: 1 of the four files do not exist (docs/preregistration/lite_arch_v2_lead_rulings.md)", with
  1 row of the experiment and 0 by the lead; the guard is checked again after the stamp row is written.
- **Pinned artefacts.** The split pin, the label table, the P2 index, the P2 lock, the C2 keys file and its meta
  file, and both protocol files were re-hashed by the lead and all match §8.

## 2. The verified pre-stamp review of the amendment (M7_verify.json)

The lead order of 2026-11-10 15:05 EST (Annex A) adopted every ruling sentence of `M7_verify.json`
(`verdicts[].ruling_text`, `points[]`, `missed[].suggested_ruling`) as a lead ruling except where that order said
otherwise. Those sentences bind from this file, in the form restated in §6. Where §4, §5, §6 or §7 of this file
changes one of them, the form in this file is the binding one.

## 3. Rulings R-01 to R-09 (independent review of the stamp package)

Each ruling below is the reviewer's proposed text, adopted by the lead. A lead remark follows where the lead rules
something the reviewer left open or where the lead decides differently.

### R-01

*Where:* docs/preregistration/lite_arch_v2_label_addendum.md:3-4, :55;
docs/preregistration/lite_arch_v2_deviations.md:45, :60, :77, :153; docs/licenses.md:53;
scripts/train/litev2_tabpfn_memtest.py:3-6; sentinel/models/lite/national_testlook.py (NATIONAL_CODE comment);
docs/preregistration/lite_arch_v2_code_readings_20261110.md:4

**Ruling.** The rulings file reproduces verbatim, with its time, every lead order the addendum, the deviations
file, docs/licenses.md and the v2 code cite (15:05 EST items 2(b), 2(c), 3, 5, 6, 7(ii); 18:26; 18:45 items 1-3;
19:38; 19:42). A cited order that this file does not reproduce is void and the choice resting on it is re-ruled
here before the stamp.

**Lead remark.** The five cited orders are reproduced verbatim in Annexes A, B, C, E and F of this file, with the
UTC timestamp of the message as sent; the relay of the M2 stamp that carries conflict C2 is Annex D, and the
pre-stamp order under which the code items A1-A7 were written is Annex G. The labels the package uses (15:05,
18:26, 18:45, 19:38, 19:42 EST) are M7's labels for those messages; the lead has checked each label against the
message it names, and the clock times in the transcript are 2026-11-10 14:59:48, 18:24:38, 18:33:07, 19:38:07 and
19:39:33 EST. No cited order is missing, so nothing is void under this ruling.

### R-02

*Where:* sentinel/models/litev2/stamp.py:36 (STAMPED_FILES), :83-88; sentinel/models/litev2/labels2b.py:57

**Ruling.** The lead's stamp row also records, under extra.pins, the sha256 of lite_split_national2b.json
(03f0114a...), labels_national2b.parquet (2969c729...), the P2 index (5c404be9...), lite_p2_lock.json
(f3d4c63e...), the C2 keys csv (4786aa12...) and its meta file (ea2b0b11...), lite_protocol.md (24b695fc...),
lite_national_protocol.md (4ae31936...) and lite_arch_v2_code_readings_20261110.md (002d5d08...); the split pin,
the label table and both protocols are set to 0444 at the stamp. A change of any pinned file voids the stamp until
a new lead row exists.

### R-03

*Where:* docs/preregistration/lite_arch_v2_deviations.md:163-170 (entry 10);
docs/preregistration/FREEZE_lite.json; sentinel/models/lite/guard.py (SHARED_CODE, load_freeze)

**Ruling.** FREEZE_lite.json, the 4-state frozen file list, the spent look's markers and tests/test_lite.py are
not edited; the failing closure test stays failing and is the record of deviation 10. The frozen 4-state scoring
code is the working tree with sentinel/models/orbit/data.py and sentinel/models/orbit/pipeline.py as at commit
34df74e; with those two files the hash a7a8be82...f9ff is reproduced (independent review, 2026-11-10). M7 appends
an entry to the deviations file that corrects entry 10 to name both files (preferably before the stamp, since the
stamp row hashes that file). No 4-state row is re-scored. The national freeze hashes the full import closure of
the national scorer (tests/test_litev2_national_closure.py must pass at that freeze).

**Lead remark.** The correction of deviation entry 10 is made **here** and not by appending to the deviations
file: the lead ordered M7 (Annexes E and G) not to append, because the stamp hashes the reviewed byte state
`23e6fd8705fe4bfc102df74160e46477c8c6b31b3ee38ff504370ed3fc00642c`. The correct reading is §9 of this file. Entry
10 of the deviations file is read as corrected by §9 in every write-up.

### R-04

*Where:* docs/preregistration/lite_arch_v2_label_addendum.md:204-217, :345-370;
scripts/train/litev2_comparators.py:144; sentinel/models/litev2/labels2b.py:409-446; M7_verify.json point Q1b

**Ruling.** Whichever configuration is selected must pass H1 against every applicable comparator fitted on that
configuration's own training rows and label version. If a configuration 1-6 is selected, S6-S11, P1 and P2a/b are
refitted on the rerun-view rows (CPU, no tuning) and their validation rows registered before the freeze is
authorised; the comparators fitted on the v2 rows (a superset, DenseNet P3a/b included) apply as well, so the
added rows can only remove a label. Every table states the training-row set and label version of each model, and
that slots 7-8 train on 10,651 rows against the rerun-view rows of configs 1-6.

### R-05

*Where:* docs/preregistration/lite_arch_v2_label_addendum.md:438-470 (power section; national test roles absent);
docs/preregistration/lite_protocol.md section 12 (role table and power check);
docs/preregistration/lite_national_protocol.md:36-55

**Ruling.** No floor, threshold, endpoint or family of lite_protocol.md or lite_national_protocol.md is changed by
the label rule. Test-N under national-2b has 379 / 460 / 262 / 350 labelled chips in 152 / 126 / 102 / 137 water
bodies (Secchi / turbidity / TSS / chl-a); the expected 95 % half-widths of section 12 become about 9 (Secchi),
13.5 (turbidity) and 17 (TSS) MdSA points, computed from counts only and optimistic, since the effective number of
water bodies is 55 / 30 / 30. The N3 floor on Test-N is applied by the scorer at the look, as written.

### R-06

*Where:* docs/preregistration/lite_arch_v2_label_addendum.md:342-343 (778), :295-296 (187 chips, 38 water bodies);
docs/preregistration/lite_arch_v2_amendment.md:407, :425

**Ruling.** Write the C2 sentence with '308 teacher pairings ... the 778 labelled chips (777 as built) stay in M7
training' and the P2 sentence with 'floor of at least 100 chips in at least 40 water bodies not met: 187 chips in
38 water bodies (189 as built); the floor is not lowered'.

### R-07

*Where:* docs/preregistration/lite_arch_v2_code_readings_20261110.md (whole file; section 7 l.1-4 of that
section); not in the package, not in STAMPED_FILES

**Ruling.** The readings of lite_arch_v2_code_readings_20261110.md (sha256 002d5d08...a4d4) are adopted as fixed
before any validation number; none may change after the stamp without a stamped deviation. Of its section 7: (1)
ANCHOR-Set is the architecture as written, 307,341 parameters; (2) 12 DenseNet runs, inside the ceiling; (3) the
stricter order holds: no candidate fit starts before every comparator validation row of H3 is registered; (4)
accepted as written. The validation score that enters selection S for slots 7-8 is computed on the same unit
(ensemble or seed) as it was for configs 1-6 and their re-runs; the carrier stays the INT8 pipeline of seed 0.

**Lead remark.** The unit the reviewer left open is now known and is fixed by ruling D-0 of §7: selection rule S
was computed for configurations 1-6 on the 3-seed ensemble mean, so the same unit is used for the re-runs and for
slots 7 and 8.

### R-08

*Where:* docs/preregistration/lite_arch_v2_deviations.md:172-181 (entry 11, five open points);
docs/preregistration/lite_arch_v2_label_addendum.md:365-369, :372-387, :468-470; docs/licenses.md:53

**Ruling.** One sentence each, as given under open_points.

**Lead remark.** The five sentences are §5 of this file.

### R-09

*Where:* tests/test_litev2_tabular.py:83 (the one test found); M7_verify.json point Q5

**Ruling.** Before the first comparator or candidate validation row is written, the three missing Q5 tests exist
and pass, and the lead is told their names: (i) the refit table holds no val, val_n, val_p2 or test-role chip or
water body; (ii) no train_p2 water body is the same water body as a national-2b training one under a different id
(as built: no train_p2 site within 2 km of a national-2b train chip); (iii) the evaluation anchor and S6 are one
artefact with one registered sha256.

**Lead remark.** The three tests were written before this stamp and are named here: (i)
`tests/test_litev2_checks.py::test_refit_table_holds_no_validation_or_test_chip_or_water_body` with its real-row
twin `::test_real_refit_table_is_clean`; (ii)
`tests/test_litev2_checks.py::test_no_train_p2_water_body_is_a_national_training_water_body_under_another_id` with
its real-row twin `::test_real_train_p2_rows_are_not_national_training_water_bodies`; (iii)
`tests/test_litev2_anchor_store.py::test_evaluation_anchor_and_s6_are_the_same_numbers_of_one_file` with
`::test_one_registered_artefact`. All three are inside the 120 passing tests of §1.

## 4. Rulings N-01 to N-06

### N-01

*Where:* docs/preregistration/lite_arch_v2_label_addendum.md:9-12 against :232-233;
docs/preregistration/lite_arch_v2_deviations.md:77-86 (entry 4)

**Ruling.** The sentence of addendum l.9-12 is read with one exception, stated here: the stored val_p2 and
train_p2 values of the two index builds were compared for equality and only the number of differences (0) was
kept.

### N-02

*Where:* docs/preregistration/lite_split_national2b.json (chip_role); label table columns water_body and
national2b_wb

**Ruling.** The water body of record for every role rule and every cluster bootstrap is the column water_body of
the pinned split, as in national-2. The national look also reports each primary metric without the test and test_n
chips whose national2b_wb group holds a training chip (169 and 79 chips), as a sensitivity row fixed now.

### N-03

*Where:* sentinel/models/litev2/stamp.py:83

**Ruling.** The lead records the id of the stamp row in the rulings file's companion note and in LEAD_DECISIONS; a
v2 run whose log cites another prereg_snapshot row id is void.

### N-04

*Where:* docs/preregistration/lite_protocol.md, docs/preregistration/lite_national_protocol.md (both mode 0664)

**Ruling.** As in R-02 (hashes in the stamp row, files 0444).

### N-05

*Where:* docs/preregistration/lite_arch_v2_deviations.md:12-44 (entry 1), :135-152 (entry 8), :153-162 (entry 9)

**Ruling.** No model object, checkpoint or prediction made before the stamp is used by any v2 run; every v2 fit
starts after the stamp row. No log of a v2 job is deleted.

### N-06

*Where:* sentinel/models/litev2/*.py, scripts/train/litev2_*.py, tests/test_litev2_*.py

**Ruling.** The lead (or M7 in the lead's sight) runs the litev2 tests once more on the code state that will run,
and records the count and the code hashes with the stamp row.

## 5. The five open points of deviations entry 11

Entry 11 of the deviations file put five points to the lead. Each is ruled here in one sentence group; these texts
are binding and are the only form in which the points may be cited.

### (a)

*Where:* docs/preregistration/lite_arch_v2_deviations.md:173-174;
docs/preregistration/lite_arch_v2_label_addendum.md:365-369 (re-runs reuse the KD target file as built)

**Ruling.** The re-runs of configurations 1-6 give KD weight 0 to the 308 teacher pairings on M2 Test-S
site-dates, through the same key-built mask as the v2 arms; this is the third and last exception to 'unchanged
settings' (with the corrected labels and the 4 reserved-site chips), and no other setting differs. No fit started
after the reservation uses a teacher value on a Test-S site-date. Tables of as-built against corrected validation
scores name these differences. A unit test shows that the re-run KD tensors hold no masked pairing, before the
first re-run starts.

### (b)

*Where:* docs/preregistration/lite_arch_v2_deviations.md:175-177; M7_verify.json F4, F5, DS-06 ruling texts (name
as-built checkpoints)

**Ruling.** Slots 7 and 8 start, seed for seed, from the re-run of configuration 3 on national-2b labels; their
no-KD twins start from the re-run of the no-KD twin of configuration 3; S11 takes its embeddings from the same
re-run checkpoints. The as-built checkpoints named in the verifier's F4 and F5 texts are loaded by no v2 job. The
sha256 of each re-run checkpoint is registered before the first slot-7 or slot-8 job starts; if a re-run
checkpoint is missing the slot does not run (restart only under the DS-11 rule), and there is no fallback to an
as-built checkpoint. The cut-off of DS-07 is unchanged. Write-ups state: slots 7-8 warm-start from configuration 3
as re-run on corrected labels with KD weight 0 on Test-S pairings; the as-built configuration 3 had been trained
with KD on those pairings before the reservation.

### (c)

*Where:* docs/preregistration/lite_arch_v2_deviations.md:178;
docs/preregistration/lite_arch_v2_label_addendum.md:372-387;
docs/preregistration/lite_arch_v2_amendment.md:497-508 (30 GPU-h)

**Ruling.** The hard ceiling of this round is 45 GPU-h, replacing 30 of amendment section 10, for exactly the runs
of the addendum's compute table; no other run is added under it, and a gated slot of section 6 opens only by its
own gate as written and counts under the same ceiling. GPU-hours are logged per job in the registry. Comparators,
re-runs and twins are run before any candidate, so that reaching the ceiling can never remove a reference; a
candidate run not finished under the ceiling is recorded as not run and still counts. The ceiling is never raised
after the first v2 validation number exists.

### (d)

*Where:* docs/preregistration/lite_arch_v2_deviations.md:179;
docs/preregistration/lite_arch_v2_label_addendum.md:468-470; docs/preregistration/lite_national_protocol.md:43-55
(N3)

**Ruling.** The N3 endpoints, thresholds and floors are unchanged. The sentence 'val_n has 48 positives in 18
water bodies' describes the as-built labels. After the stamp and before any v2 event number, the frozen scorer
recounts the E3a_turb positives and their water bodies on val_n under national-2b once, and the two counts are
registered; if val_n then has fewer than 30 positives or fewer than 10 water bodies with positives, E3a_turb is
descriptive only on the national track. The Test-N floor of N3 is applied at the look as written. No threshold is
changed in either case.

**Lead remark.** Ruling D-3 of §7 adds a second, stricter condition on top of this one and does not replace it:
the as-built view is counted once as well, and E3a_turb is confirmatory only if both views reach the floor. Where
the two readings differ, the stricter one holds.

### (e)

*Where:* docs/preregistration/lite_arch_v2_deviations.md:153-162 (entry 9); docs/licenses.md:53

**Ruling.** P3a and P3b are run and their results reported whatever they are; the licence question can never be a
reason to omit them. The DenseNet121 weight file, every checkpoint fine-tuned from it and every artefact derived
from it stay on the project disk: none is uploaded, released or served. Every write-up states that the weights are
torchvision DenseNet121_Weights.IMAGENET1K_V1, that the source states no licence for the file and that the
ImageNet-1k terms were not verified at download. Before any public write-up the terms are read, the licence row is
completed, and the user's written approval is recorded. The lead's approval of the one download (2026-11-10) is
recorded here with its time.

**Lead remark.** The approval of the one DenseNet121 download is the lead order of 2026-11-10 15:05 EST, item
7(ii), reproduced in Annex A: "Once the rows exist, I approve that single download ...". It is on file from this
stamp.

## 6. The verifier items, in the form that binds

The items below carry every item of the verified review as the independent reviewer restated it after reading the
package. `closed_in_package` means the package already does what the item requires and the sentence is adopted as
the wording for write-ups; `needs_ruling_sentence` means the sentence binds from this file. A reviewer note after
the sentence is part of the ruling.

### F1 — needs_ruling_sentence

VERIFIER: "The stamped text is DRAFT v0.3, sha256
2b369018d2fc2a340594790756b3edbef3b1de28a38cf6073eb5908111908a56, set to 0444 at the stamp; the hash
ca281d84...d771 (v0.2) is void. These v0.3 statements are superseded as statements of fact: l.29-32 (P2 code,
lock, index and 513 chips exist; l.68-75 is the state), l.383 (the eligible test_p2 count is known), and the
header sentence 'No rule, constant, model, comparator or threshold was changed' (v0.3 added the several-limits
sentence of section 3.3 and the all-zero-pixel criterion of section 8.5). The section 0 table is completed by the
no-KD twin of config 6 (validation rows registered 18:12 UTC). The P2 lock build read metadata columns of more
sources than columns_read lists; the full list is recorded in the deviations file." | NOTE: the lock-column list
is recorded in deviations entry 1 (l.24-36); the 0444 mode is the lead's act at the stamp.

### F2 — needs_ruling_sentence

VERIFIER: "Comparator S10 = TabPFN v2 regressor (tabpfn 2.2.1 and the checkpoint already on disk, loaded only
through sentinel.models.chemtox.tabpfn_model.load_tabpfn; no install, no download) on the 98 features of S6, with
the same v2 training rows, label rule, log10 targets and evaluation rows as S6: one model per target, FNU and NTU
handled as in S7, uncensored rows only, package defaults, no tuning, 3 seeds averaged. Where a target has more
than 10,000 training rows, whole water bodies are kept in ascending order of sha256('SENTINEL-Lite-tabpfn|' +
water_body) until the next one would exceed 10,000 rows, and the kept count is registered. S10 is a member of set
C on P2, of the strongest-simple family and of the national baseline list of Q1b; its validation rows are
registered before any candidate validation number. Amendment l.291, l.482, Q4 and the 'Package absent' row of
section 12 are superseded." | NOTE: the package adds a memory-test fallback (cap = largest of 10,000 / 8,000 /
6,000 that fits) on 'lead order item 6', which is not on file; the rulings file must carry that sentence or S10
runs at 10,000 only. No target has more than 10,000 uncensored training rows (largest 9,371).

### F3 — closed_in_package

Rule stated in addendum (i) l.26-31 and implemented through M2's reader (sentinel/models/orbit/wqp_surface.py,
unchanged wqp_surface_samples with depth); counts-only addendum supplied and recounted by me. The rulings file
should still adopt the rule sentence and the supersession of amendment section 3.4 item 1: VERIFIER: "Every WQP
label of national-2b and of P2, in every role and for every model and comparator, comes from the harmonised
QC-passed tables only (wqp_discrete/variable=*/); surface means depth_m <= 1.0 m or unknown; a result flagged
'bottom' is dropped at any depth; Secchi is exempt; the unchanged wqp_surface_samples applies it with depth_m
passed as result_depth; a sample with no QC-passed surface result is dropped and the reason is recorded; sonde
labels are unchanged; no other depth threshold is run or reported. Section 3.4 item 1 and the word 'unchanged' in
l.14-17 and section 3.4 item 4 are superseded: chip roles and water-body ids are unchanged, label values and label
hashes are not. Before the stamp M7 supplies a counts-only addendum: rows kept, dropped and changed per build,
target and role; the corrected section 3.1 table; the new national label and split hashes; the rebuilt P2 index
hash and eligible counts; and the list of existing models that are re-scored or re-run. No v2 arm or comparator is
fitted on as-built labels."

### F4 — needs_ruling_sentence

VERIFIER: "In every v2 arm, ablation and comparator (S8 and P3b included) a teacher pairing on an M2 Test-S
site-date (orbit-v3 test-role lake, in-lake, 2015-06-23 to 2023-12-31) carries KD weight 0 and supplies no
teacher-derived feature; the mask is built from keys by this rule, its count (308 as counted today) is registered
before the first v2 run and reconciled in writing with M2's count of 242 teacher-store chips; the 777 labelled
chips stay in M7 training; no M7 output is ever an input to anything scored on Test-S. The CIR-Lite-R gate cells
exclude the same lake-dates (114 val and 18 val_n chips are on Test-S lakes). Every write-up states that slots 7-8
warm-start from config 3, which was trained with KD on those pairings before the reservation." | NOTE: mask built,
308 registered (row lite/tests_mask) and reconciled with 242 in addendum (vi). Write 778 for 777 (R-06). The last
sentence must be reworded if open points (a)/(b) are ruled as recommended: the warm start is then from re-runs
trained after the reservation with KD weight 0 on those pairings.

### F5 — needs_ruling_sentence

VERIFIER: "A no-KD twin is the same configuration with every setting identical and KD weight 0. It is never a
selection candidate and never the carrier, it is named in a registry row before it runs, and it is not counted
against the cap of 8 (frozen lite_protocol.md l.317-318). Two exist now (configs 3 and 6; validation rows 17:26
and 18:12 UTC); every write-up states '8 counted configurations plus N uncounted no-KD twins' and lists all their
scores. For slot 7 or 8 the twin has KD weight 0 in the residual branch and starts its image branch from the
seed-matched no-KD checkpoint natstudent_mnv4s_raw_crop256_nokd_national2_20261110-115426; its 3 seeds, export and
registry rows must exist before the freeze is authorised, otherwise the slot cannot be selected." | NOTE: under
recommendation (b) the checkpoint named here (as-built, ...115426) is replaced by the corrected-label re-run of
the no-KD twin; state the two twins that exist as built and that both are re-run.

### F6 — needs_ruling_sentence

VERIFIER: "National slots 7-8 can earn only the label of frozen lite_protocol.md section 8a, decided in code and
unchanged: 'State of the art' is not applicable; 'First-in-class' needs the novelty search extended to the
GBM-anchored hybrid before the look, and H1 for Secchi and for turbidity or TSS against every applicable baseline,
which now includes S1-S11, P1, P2a/b and P3a/b, each with diff < 0 for each of the 3 seeds. The cap of 8
national-2 configurations is fixed by this ruling. Every write-up states that slots 7-8 were designed after the
national-2 validation numbers of configs 1-6 were seen, and names the model a GBM-anchored hybrid."

### F7 — closed_in_package

Deviation 5 and addendum (v): stricter reading, 128 -> 60 usable train_p2 rows, registry row e89036650629
(lite/p2_f7), unit test test_f7_removes_whole_water_bodies_and_draws_the_unseen_blocks. Verifier text for
adoption: VERIFIER: "Before any v2 run, every train_p2 water body with a chip or site in a 25-km block that holds
a national val, val_n, test or test_n role is removed from the v2 training rows by keys (25 eligible chips today:
13 in val-role and 12 in test-role blocks), and train_p2 rows in blocks without any Lite chip (96 chips) get the
national-1 salted block rule of section 3.4 item 2 and are removed if it gives a role other than train. The
removed counts are registered; the lock file is not changed; the same reduced row set serves every arm and
comparator."

### F8 — needs_ruling_sentence

VERIFIER: "P2 is descriptive in this round: one hardened look, no label of any kind; the sentence 'One scoped
label is possible' of section 2 and the confirmatory tests of section 8.8 are not applied; the floor is not
lowered and no water body changes role. Every P2 number is reported with 'rivers and streams only', the number of
test water bodies and the effective number of water bodies (about 10). Nothing is selected on val_p2 (effective
2.7 water bodies): the P2 headline model is config 1 of section 6 as written, all P2 configs are reported, and for
P3a/P3b both backbone learning rates are run with 3 seeds and both are reported." | NOTE: counts under the label
rule: 187 chips, 38 water bodies, 74 sites, effective 10.08 (R-06).

### F9 — needs_ruling_sentence

VERIFIER: "No lite or litev2 loader returns a val, val_n or val_p2 row to any v2 arm or comparator until a guard,
covered by a unit test, has verified a prereg_snapshot registry row that holds sha256 2b369018...8a56 of the
amendment and the sha256 of this rulings file; the guard uses the workstream name and field that the stamp row
actually has, and the wording of l.37-38 is superseded by that. Deviation 1 of lite_arch_v2_deviations.md records
that the P2 index and 513 chips were built, and 34 train_p2 and 12 val_p2 label values were printed to a log since
deleted, before the stamp; no test_p2 value was read. The eligibility constants (30 m, 50 m2, 1 % no-data) are
frozen as built." | NOTE: guard and tests exist (stamp.py; 19 tests pass) and deviation 1 holds the record; the
guard checks four files, not the split, labels or protocols (R-02).

### F10 — closed_in_package

Deviations 2 and 7: 4 lock sites excluded at load (0 index rows at them, verified), overlap counts recorded,
reserved-list sentence present. Verifier text for adoption: VERIFIER: "The lock is not changed. M7 checks the one
CBP_WQX train_p2 site against the M1 test-role site by coordinates and drops it from training at load time if they
are the same station. The Fuse and Track-N overlap counts are added to the record. test_p2 site-dates join the
campaign's reserved list: no workstream trains or tunes on them a model that is compared on, or fused for, P2. No
Lite output is an input to an M1, M2, M6 or Fuse model scored on its own test roles."

### F11 — closed_in_package

docs/licenses.md l.9 (NWIS names M7), l.51 (HydroLAKES), l.52 (MobileNetV4), l.53 (DenseNet121); registry row
lite/pretrained_weights. The DenseNet licence is 'not stated by the source' and the download approval is not on
file: see open point (e) and R-01. VERIFIER: "Before any v2 run docs/licenses.md gains rows for the
ImageNet-pretrained DenseNet121 weight file (source, licence, sha256), for the MobileNetV4 weights already in use
and for HydroLAKES, and names M7 as a user of NWIS sonde data. No weight file is downloaded before its row exists
and the lead or the user has approved the download."

### DS-06 — needs_ruling_sentence

VERIFIER: "Comparator S11 = GBM-stacked as the amendment describes it at l.289-290 (S6 features plus 16 principal
components of the pooled embedding of the config 3 network that slots 7-8 start from, seed-matched, PCA fitted on
v2 training rows only), fitted like S6. S11 is a member of set C on P2 and of the national baseline list of Q1b,
and its validation rows are registered before any candidate validation number. If the hybrid does not beat S11 (CI
includes 0), every write-up credits the gain to CNN features, not to the residual-anchor design. The words
'descriptive only' at l.288 no longer apply to GBM-stacked." | NOTE: 'the config 3 network that slots 7-8 start
from' is the corrected-label re-run under recommendation (b).

### DS-07 — needs_ruling_sentence

VERIFIER: "Slots 7-8 enter the national selection only if the registry holds, by 2026-11-16 23:59 EST by registry
timestamp, their 3-seed validation rows, the passing export-parity test of the seed-0 INT8 pipeline under the
frozen protocol's criterion, and the no-KD twin rows. The lead does not authorise the national freeze before that
time unless both slots are already registered or recorded as not run. The cut-off can be moved only by a stamped
deviation written before any validation number of slot 7 or 8 exists. A slot with any validation number is counted
and reported; 'recorded as unused' at l.319 is replaced by 'counted, reported, not selectable'."

### DS-09 — needs_ruling_sentence

VERIFIER: "On both tracks the carrier is the INT8 pipeline of seed 0 of the selected configuration, as in frozen
lite_protocol.md section 8a. The sentence of l.250-251 that makes the 3-seed ensemble the carrier is void; the
ensemble is reported descriptively and earns no label. No second, weaker latency or size tier is introduced."

### DS-10 — needs_ruling_sentence

VERIFIER: "Wherever a seed rule is applied, on the national track now or on P2 under a later stamped addendum,
every rejected comparison must also have diff < 0 for each of the 3 seeds individually against that same
comparator; the 'lowest test MdSA member' form of section 8.8 item 3 is void."

### DS-11 — needs_ruling_sentence

VERIFIER: "The DenseNet121 weights are torchvision DenseNet121_Weights.IMAGENET1K_V1, one file, its sha256 in a
registry row before use. On the national track a learning-rate tie (difference in S below 0.01) keeps 1e-4 and
both seed-0 runs are reported. The list of targets for which S9 runs is fixed from training-row counts under the
label rule and registered before any validation number; it is stated that S9 and the Tobit term do not exist for
FNU on P2. A restart is allowed only when the run directory holds no evaluation file and the registry holds no row
for that config and seed; both job ids are logged." | NOTE: done in the package: weight file sha registered (row
e7ebd23c8ede), S9 target list registered (lite/s9_targets: secchi, tss, chla; 47 / 83 / 51 censored rows per the
addendum). Still a sentence: the learning-rate tie rule and the restart rule.

### F12+F13+DS-14 — closed_in_package

Addendum (iii)/(iv) use 976b5be7... as the canonical hash and name the P2 rows under workstream 'lite'. VERIFIER:
"The hash 976b5be7... cited for lite_split_national2.json is the canonical split hash computed by
national2.split_sha256, not the file hash; the two P2 registry rows are under workstream 'lite'."

### Q1a — needs_ruling_sentence

VERIFIER: "Allowed, only under the rulings of F3, F5, F6, F7 and DS-07 together; if the lead does not adopt any
one of them, ANCHOR-Lite and A1 run on P2 only. No new national label type either way."

### Q1b — needs_ruling_sentence

VERIFIER: "Yes, and mandatory for whichever configuration is selected: S6-S9, S10 (TabPFN), S11 (GBM-stacked), P1,
P2a/b and P3a/b join the frozen list, each under the every-seed rule, fitted on the selected candidate's own
training rows and label version." | NOTE: not met by the package for a selected configuration 1-6 (R-04).

### Q2 — closed_in_package

Deviation 6 item 4; my recount: 0 recovered or censored labels on any test role. VERIFIER: "Not built, not loaded,
not reported. The descriptive table is rejected."

### Q3 — needs_ruling_sentence

VERIFIER: "Accepted as written (50 scene statistics, lat, lon, DOY; no site class, no HydroLAKES morphometry),
identical for candidate, anchor and every parity comparator; nothing is added later; recorded as a plan deviation;
claims say 'GBM with the same information', never 'strongest tabular model'."

### Q4 — needs_ruling_sentence

VERIFIER: "Not acceptable. S10 runs as specified in the F2 ruling."

### Q5 — open

Three of the four required unit tests were not found (R-09). VERIFIER: "Accepted. Before any validation number,
unit tests must show: no fold model is fitted on a row of the water body it predicts; the refit table holds no
evaluation-role chip or water body; national-2b additions and train_p2 take water-body ids from one grouping; the
evaluation anchor and S6 are one hash-pinned artefact. Anchors on time-held-out rows of training water bodies are
reported as a separate stratum. AFT accepted with the fixed parameters and a registered target list."

### Q6 — closed_in_package

Lock sha f3d4c63e... unchanged (0444); only the label filter changed in the index rebuild (deviation 4). VERIFIER:
"All stand exactly as locked: 40/20/40, sonde mean else median, 50 m2 within 30 m, floor 100 chips and 40 water
bodies, train_p2 years <= 2021. When the index is rebuilt for the label rule, only the label filter changes."

### Q7 — needs_ruling_sentence

VERIFIER: "Accepted, always worded 'CyFi-style' and 'HydroVision-style re-implementation on NAIP' with the
differences listed in every table; P2b and P3b are required; national learning-rate tie keeps 1e-4; on P2 both
learning rates are run and reported."

### Q8 — needs_ruling_sentence

VERIFIER: "One named file (torchvision DenseNet121_Weights.IMAGENET1K_V1), after its licence row exists and the
lead or the user approves the download; sha256 registered before use. This review grants nothing." | NOTE: the
file is already on disk (downloaded 18:43 EST, sha a639ec97... verified, 0444); the approval is cited, not on file
(R-01, open point e).

### Q9 — needs_ruling_sentence

VERIFIER: "No weaker tier. Carrier = INT8 pipeline of seed 0 on both tracks."

### Q10 — closed_in_package

Lock unchanged. VERIFIER: "Kept; they are in the immutable lock."

### Q11 — closed_in_package

No such rows in the v2 training rows; added chips carry recovered labels only (deviation 6 item 5). VERIFIER: "Not
used in this round. A later use needs its own stamped addendum and a new data version before any validation
number."

### Q12 — needs_ruling_sentence

VERIFIER: "As the lead already ruled: the floor stays, P2 is descriptive with one hardened look and no label, no
val or train water body moves to test, and M7 uses no 'state of the art' wording this round. Under the label rule
the eligible test count is expected to fall from 189 to 187 chips; the recount goes in the addendum." | NOTE: 187
confirmed by recount.

### P13 — needs_ruling_sentence

VERIFIER: "Stamp v0.3 (2b369018...8a56) byte for byte; the F1 ruling lists the superseded sentences. No v0.4 is
needed for this point." | NOTE: see F1.

### P14 — needs_ruling_sentence

VERIFIER: "The rulings file adds S10 with the F2 specification; four passages of the amendment are superseded." |
NOTE: see F2.

### P15 — closed_in_package

Addendum supplied; see F3. VERIFIER: "National-2, national-2b and P2 WQP labels all inherit the defect; the
amendment states no depth rule. Write the M2-identical project rule (F3 ruling), for every role, target, model and
comparator, before any national-2b or P2 validation number. The author's counts-only addendum is required before
the stamp."

### P16 — needs_ruling_sentence

VERIFIER: "X2 is not met by the amendment text. KD weight 0 and no teacher-derived feature on Test-S pairings,
mask by rule, count registered and reconciled with 242, CIR-Lite-R gate cells excluded, warm start disclosed (F4
ruling)." | NOTE: see F4.

### P17 — needs_ruling_sentence

VERIFIER: "Yes, the frozen section 8a label only ('First-in-class' at most; 'State of the art' not applicable),
under the F6 ruling and the widened baseline list. No target-scoped or new label."

### P18 — needs_ruling_sentence

VERIFIER: "Accepted as an uncounted H2 comparator, never selectable, listed in section 0 by ruling and counted
openly in every write-up (F5 ruling). Not a breach of lead rule 2."

### P19 — closed_in_package

Recounts are in the addendum; no registry row or run of a new arm found by me either (check 7). VERIFIER:
"Verified: no registry row or run on any new arm, comparator or P2 validation/test; lock and index counts and
hashes reproduce; section 3.1 row sums are right (FNU 128 train, 237 without role). Not accepted as final: 'at
most 365' becomes at most 316 and 128 becomes at most 115 under the label rule; the power figures are
arithmetic-correct but optimistic (about 10 effective water bodies)."

### M0 — closed_in_package

New label version national-2b pinned in lite_split_national2b.json (canonical 29f0dd00..., labels f4612981...) and
registry row lite/label_version; chip roles unchanged on the mapping alone (0 role changes of 32,820 chips,
recounted); 976b5be7... kept for the as-built version. VERIFIER: "Chip roles and water-body ids of national-2 are
unchanged and this is asserted on the chip_role mapping alone. The corrected labels are a new label version with
its own labels_sha256 and split hash, pinned in a new file and a registry row before any v2 run; 976b5be7... stays
as the pin of the as-built version and is used only to reproduce numbers already seen. The frozen-list baselines
are refitted on the corrected rows."

### M1 — needs_ruling_sentence

Package does the re-runs (addendum (vii), national2b_rerun.py refuses changed settings); the two stated exceptions
to 'unchanged' and open point (a) need the lead's sentence. VERIFIER: "Every counted national-2 configuration and
each no-KD twin is re-run with unchanged settings on the corrected labels (logged re-runs, not new
configurations), the selection rule S is applied once to the corrected validation rows, and the as-built and
corrected validation tables are both reported. No setting is changed between the two. If the lead does not
authorise the re-runs, configs 1-6 are only re-scored, each model is compared with baselines fitted on its own
label version, and the label version of every model is stated in each table."

### M2 — closed_in_package

Addendum (ii) recount. VERIFIER: "The counts of section 3.1 are replaced by the recount under the label rule,
given in the addendum per target and role; 'FNU: 128 rows now, at most 365' is read as 'at most 115 now, at most
316'. No text quotes the v0.3 counts as the training gain."

### M3 — closed_in_package

Addendum (ix): stratum 135 chips, 22 water bodies, registered (lite/p2_stratum); wording 'unseen water bodies' at
l.426. VERIFIER: "P2 reports say 'unseen water bodies', not unseen basins or regions, and give every P2 number
also for the stratum of test_p2 chips whose HUC8 and 25-km block hold no v2 training row; the stratum counts are
registered before the look."

### M4 — closed_in_package

Deviation 6 item 3 (19 Secchi chips dropped). VERIFIER: "One rule is used in both builds: a right-censored chip
with more than one distinct limit is not used as a censored row (the stricter, implemented rule); the
left-censored rule is as written."

### M5 — closed_in_package

Deviation 6 item 8. VERIFIER: "P2a and P2b keep the days-between feature as in the published recipe; tables mark
it as a label-side input that the candidate does not have."

### M6 — needs_ruling_sentence

Not done: both protocol files are still 0664 and their hashes are in no stamped file (N-04, R-02). VERIFIER: "The
rulings file records the sha256 of lite_protocol.md (24b695fc...) and lite_national_protocol.md (4ae31936...) as
the texts meant by 'frozen section 8a' and 'N1-N8', and both are set to 0444."

### M7 — needs_ruling_sentence

Ceiling restated at 45 GPU-h in the addendum (open point c); erratum on the spent look is in addendum (x).
VERIFIER: "The addendum restates the compute ceiling with the added runs. The v1 results stay reported as they
are, with a counts-only erratum on the label defect; no spent test row is re-scored without the lead's
authorisation."

Two items need a lead decision on top of the sentence:

- **F2 (TabPFN S10).** The memory-test fallback the package implements rests on "lead order item 6", which the
  reviewer could not find on file. It is on file: the 15:05 EST order, item 6, reproduced in Annex A, says "Only
  if 10,000 rows x 98 features does not fit in one 11 GB GPU, use the largest of 10,000 / 8,000 / 6,000 that fits,
  decided by a memory test on training rows only with no validation row loaded, and registered before any
  validation number." That sentence binds. S10 runs at 10,000 rows unless the memory test, run on training rows
  only with no validation row loaded and registered before any validation number, shows that 10,000 does not fit;
  the cap actually used is registered.
- **F4 and F5 (Test-S mask, no-KD twins).** The last sentence of F4 and the checkpoint named in F5 are replaced by
  §5(a) and §5(b): the warm start is from the corrected-label re-runs, which are trained after the reservation
  with KD weight 0 on the 308 Test-S pairings, and no as-built checkpoint is loaded by any v2 job.

## 7. The decisions M7 put to the lead

- **D-0 — the unit of selection rule S.** Selection rule S was computed for configurations 1-6 on the **3-seed
  ensemble** (the mean prediction over seeds 0, 1 and 2), not on single seeds. The lead checked this in the code
  and in the registry: `scripts/train/lite_train.py:112` forms `mu = np.mean(mus, 0)`, `:116` scores that mean
  into `scores`, `:123` writes it to the run's `summary.json`, and `:124-127` registers it with the note "3-seed
  ensemble (mean mu); fixed epochs, no val-based early stopping"; `scripts/eval/lite_select.py:50-52` reads
  exactly that `scores` block and forms S from it. The 32 `lite_val_national2` rows of the counted configurations
  and their twins all carry that note, each with per-seed statistics for seeds 0, 1 and 2. **Ruling:** the 3-seed
  ensemble mean is the unit of S for every selectable configuration of this round — the corrected-label re-runs of
  configurations 1-6, both no-KD twins and slots 7 and 8 — so that selection stays like-for-like, and a
  configuration with fewer than 3 finished seeds is not selectable. S decides selection only. It earns no label
  and is no carrier: the carrier of any label stays the INT8 pipeline of seed 0 on both tracks (DS-09), the H1
  comparisons and the every-seed rule of DS-10 are applied as the frozen protocol §8a and the amendment write
  them, and every write-up states that selection used the 3-seed ensemble while the reported carrier is seed 0.
- **D-1 — the time-held-out strata.** Accepted as reported: `test_time` and `test_time_n` hold later dates of
  training water bodies by design, the chips are disjoint from the training chips, and the water-body overlap is
  counted (1,761 training rows in 221 water bodies). They are reported as their own stratum, never pooled with an
  unseen-water-body stratum, and every table that uses them says that their water bodies are training water bodies
  at later dates and gives the overlap counts. This is the "separate stratum" of verifier item Q5.
- **D-2 — the two tests that read real key columns.** They stay as they are and run before the stamp; they are
  **not** marked skip-until-stamp, because a test that is skipped cannot protect the stamp. They read key columns
  only (water-body id, site, date), no label and no prediction, which the project rule on protected rows allows;
  they assert counts and nothing else, and they write no key of a test role to any file or registry row. The two
  are `tests/test_litev2_checks.py::test_real_refit_table_is_clean` and
  `::test_real_train_p2_rows_are_not_national_training_water_bodies`.
- **D-3 — E3a_turb.** M7's stricter form is adopted: the one-shot recount of §5(d) is run on val_n under
  national-2b, and the as-built view is counted once as well; E3a_turb is confirmatory on the national track only
  if **both** views give at least 30 positives in at least 10 water bodies, and descriptive only otherwise. No
  threshold, endpoint or floor changes either way, and the recount is run by the frozen scorer after the stamp
  through `tests/test_litev2_checks.py::test_e3a_count_step_is_stamp_guarded_one_shot_and_registered`'s step.
- **D-4 — the GPU ledger.** Accepted as built, under the hard ceiling of 45 GPU-h of §5(c). The per-job budgets
  reported by M7 (re-run 1.5 h, 2.0 h at launch; v2 net or twin 0.5 h per seed; DenseNet 2.0 h per seed;
  evaluation 0.25 h; embeddings 0.5 h; TabPFN memory test 0.5 h plus 0.75 h per block; worst case at admission of
  phase 1 = 42 h) are the pre-registered figures: a job that would need more than its budget is stopped and
  reported, it is not given more, and a job may not start if the hours already logged plus its own budget would
  pass 45. The refusal of GPU 2 and of any job whose `CUDA_VISIBLE_DEVICES` is not a single index stands; GPU 2
  carries another user's work and is never used by this project.
- **D-5 — the A4 ablation anchor.** Accepted: ablation A4 keeps its own reduced-row anchor in its run directory,
  hash-checked at load
  (`tests/test_litev2_anchor_store.py::test_a4_anchor_model_is_hash_checked_in_its_run_directory`). It is used by
  no other arm or comparator, and the single registered evaluation-anchor/S6 artefact of Q5(iv) is unchanged.

The intended rulings of §B of the pre-stamp order (Annex G) bind from here, word for word:

```text
INT8 carrier = scope="image" (INT8 image branch, FP32 context / pooling / heads / anchor sum), fixed now before any trained model; scope="graph" is also exported and its metrics reported, descriptive only; every write-up says "mixed precision", never "fully INT8". Export parity = your three conditions, thresholds fixed now and never loosened after a failure; in addition report, on real validation rows, the count > 1e-4 and the max of the float64-vs-float32 anchor difference (reported, not a pass condition). ANCHOR-Set stays as written (307,341 parameters). 12 DenseNet runs are inside the ceiling.
```

Item A5 of that order (the rerun-view path of `scripts/train/litev2_comparators.py`) is required before the
national freeze is authorised, not before the stamp, and R-04 is the rule it serves.

## 8. What the stamp row pins

The stamp row records, under `extra.sha256`, the four stamped files, and under `extra.pins` the artefacts below. A
change of any pinned file voids the stamp until a new lead row exists. The split pin, the label table, the P2
index and both protocol files are set to 0444 at the stamp; the others are already 0444.

| Artefact | sha256 |
|---|---|
| `docs/preregistration/lite_split_national2b.json` | `03f0114a8683aa8fcc0d2b49901dc580c376ec3fdcdd2a8f41d1ee2242395663` |
| `$SENTINEL_DATA/processed/lite_labels/labels_national2b.parquet` | `2969c729dc96dc9f796deb2c8a108f9e5e8d400ec987d9662a24c400917227e4` |
| `$SENTINEL_DATA/processed/naip_matchups/lite_p2/index.parquet` (the pin of `sentinel/models/litev2/p2_index.py:INDEX_SHA256`) | `5c404be9251a22ae1b1d082446879e1466fa1c3fd41658f1f9649230be7f31b2` |
| `docs/preregistration/lite_p2_lock.json` | `f3d4c63ebe503ac1774959bfcf943b45bc140883fe5b30933ee03f0cacca52be` |
| `$SENTINEL_RUNS/lite/counts_arch_v2/c2_teacher_scored_m2_test_lake_keys.csv` | `4786aa121eb673ced502d487fb4af714fff37027677677048d2024c960cfc699` |
| `$SENTINEL_RUNS/lite/counts_arch_v2/c2_teacher_scored_m2_test_lake_keys.meta.json` | `ea2b0b11bc0d804ef2701e9456a84e0050928769de997357d52242d35c2d5d94` |
| `docs/preregistration/lite_protocol.md` (the frozen §8a and §12 meant by this file) | `24b695fc381e238a414d10a0c7ca24bfb41822da1a6b113e01741081d55a2cb4` |
| `docs/preregistration/lite_national_protocol.md` (the frozen N1-N8) | `4ae319369a5998973e9caacb603ce53b77b07a4a58d16276ea5f9ad2fbf3d4b9` |
| `docs/preregistration/lite_arch_v2_code_readings_20261110.md` | `002d5d0822dab92d725467f59966b106b866c1c958855b801fef330aaf22a4d4` |
| `docs/preregistration/lite_arch_v2_code_readings_export_20261110.md` | `895dcefe314937bc21077e2c34d78cf2252fcee0a143b1ff4a4fb8957eb5454f` |
| `$SENTINEL_RUNS/lite/counts_arch_v2/litev2_code_sha256_20261110.json` (the 39 code files) | `563894b93a9e27c0bbc3db37af16fbf49be2b55641c7e7a1e0d861c3fabcafe1` |

## 9. Deviation 10, corrected

Entry 10 of the deviations file names one file edited after the 4-state freeze. Two were edited:
`sentinel/models/orbit/data.py` (2026-11-10 13:06 EST) and `sentinel/models/orbit/pipeline.py` (2026-11-10 11:53
EST). `FREEZE_lite.json` and `TEST_LOOK.json` record `scoring_code_sha256` `a7a8be82...f9ff`; the working tree now
gives `c048a73d...9f48`, and with those two files as at git commit `34df74e` the recorded hash `a7a8be82...f9ff`
is reproduced (checked by the independent review, 2026-11-10). `FREEZE_lite.json`, the frozen file list, the spent
look's markers and `tests/test_lite.py` are not edited; the failing closure test stays failing and is the record
of this deviation. No 4-state row is re-scored. The national freeze hashes the full import closure of the national
scorer, and `tests/test_litev2_national_closure.py` must pass at that freeze.

## 10. Order of work after the stamp

1. The TabPFN memory test (training rows only, no validation row loaded), and the registration of the cap actually
  used.
2. The comparators of set C and of the national baseline list, the corrected-label re-runs of configurations 1-6
  and both no-KD twins — all before any candidate fit, by §6 DS-06, Q1b and the stricter order of R-07(3),
  enforced in code by `tests/test_litev2_launch.py::test_no_candidate_fit_starts_before_the_comparators`.
3. The registration of each re-run checkpoint's sha256, then slots 7 and 8 and their twins, warm-started seed for
  seed from the corrected-label re-runs of configuration 3 and of its no-KD twin (§5(b)); S11 takes its embeddings
  from the same checkpoints. No fallback: a missing re-run checkpoint is a hard error and the slot does not run.
4. The one-shot E3a_turb recount (§5(d), D-3), the DenseNet comparators P3a and P3b (§5(e)), and the national
  sensitivity row of N-02.
5. Selection by rule S on the corrected validation rows, once (§5); then, if a configuration 1-6 is selected, the
  refits of R-04 on its own rerun-view rows before the lead is asked to authorise the freeze.
6. Every job through the GPU lock helper, at most 2 h per job, at most 6 CPU threads per job, never GPU 2, inside
  the 45 GPU-h ceiling, each with its ledger row.

## 11. How this file is read

- This file wins over the amendment, the addendum and the deviations file wherever they differ, and it names what
  it supersedes.
- No validation, test or event number of architecture round v2 exists before the stamp row. A v2 run whose log
  cites a `lite/prereg_snapshot` row id other than the one the lead records is void (N-03).
- The floors, thresholds, endpoints and families of `lite_protocol.md` and `lite_national_protocol.md` are
  unchanged by this round (R-05). Nothing in this file lowers a bar; where a count turns out to be below a floor,
  the claim is dropped, not the floor.
- P2 is descriptive this round: one hardened look, no label of any kind (F8).
- The labels this round can earn are the frozen §8a labels only: "State of the art" is not applicable to the
  national slots 7-8, and "First-in-class" needs the extended novelty search and H1 against every baseline of the
  widened list, each with diff < 0 for each of the 3 seeds (F6, DS-10).

## Annex A — the 15:05 EST order of 2026-11-10 (items 1-7)

Sent by the lead to the M7 workstream agent; transcript timestamp `2026-11-10T19:59:48.299Z`, 8027 characters,
reproduced verbatim.

```text
Lead order for M7 (15:05 EST): the verified review says "revise then stamp" — v0.3 is stamped byte for byte, but only together with a counts-only label addendum from you and my rulings file. Answers to your 7 points follow.

SOURCE. Read docs/preregistration/reviews/arch_v2_20261110/M7_verify.json in full (keys verdicts[].ruling_text, points[], missed[].suggested_ruling). Every ruling sentence in it is adopted as a lead ruling unless this message says otherwise. They become binding in docs/preregistration/lite_arch_v2_lead_rulings.md, which I write at the stamp. Do NOT edit lite_arch_v2_amendment.md (sha 2b369018…8a56); it must still have that hash when I stamp.

1. Addendum, not v0.4. Write docs/preregistration/lite_arch_v2_label_addendum.md, COUNTS ONLY (no label value, no per-row statistic of a val/test/lock row). Contents: (i) rows kept / dropped / changed per build (national-2, national-2b additions, P2), per target and per role, with drop reasons; (ii) the corrected section 3.1 table (the verifier expects FNU "at most 115 now, at most 316" — give your recount); (iii) the new national label version: labels_sha256, canonical split hash, pin file path + sha; (iv) the rebuilt P2 index hash and eligible counts per role (test_p2 expected about 187 chips — recount), plus the effective number of water bodies per role; (v) F7 removals (train_p2 water bodies in val/test-role 25-km blocks; salted-block rule for blocks without a Lite chip) as counts; (vi) the Test-S mask count (308 as counted) reconciled in writing with M2's 242 (different units — say which); (vii) the list of existing models re-run (see 2c) and the restated GPU-hour ceiling including S10, S11, the slot-7/8 twin and the re-runs; (viii) the S9 target list fixed from training-row counts under the label rule; (ix) the HUC8/25-km "no v2 training row" stratum counts for test_p2; (x) a counts-only erratum sentence for the spent v1 4-state look (scored on as-built labels; not re-scored).
   Also create docs/preregistration/lite_arch_v2_deviations.md (dated, numbered, append-only). Entry 1 = the pre-stamp builds per the F9 ruling (P2 index and 513 chips built; 34 train_p2 and 12 val_p2 label values printed to a log since deleted; no test_p2 value read; full columns_read list of the lock build; eligibility constants frozen as built).

2. WQP label rule for M7 (one project rule; F3 ruling text is the wording).
   (a) New label version. Chip roles and water-body ids of national-2 are unchanged — assert this on the chip_role mapping alone. The corrected labels are a new label version with its own labels_sha256 and canonical split hash, pinned in a NEW file plus a registry row before any run on it. 976b5be7… stays the pin of the as-built version and is used only to reproduce numbers already seen.
   (b) The 5,998 4-state chips inside national-2: no as-built WQP label may remain in any training or evaluation row of the new version. Re-derive each from the harmonised QC-passed tables (wqp_discrete/variable=*/) through your wqp_rule.py reader by a deterministic key (organisation, site, activity start date/time, characteristic); a chip with no QC-passed surface result under that key is dropped with a recorded reason. Do not keep any as built and do not use M2's side table for raw-unit keys. STOP and send me counts first if this would drop more than 5 % of the chips of any role or more than 10 % of any target's training rows — I will rule before you proceed.
   (c) Re-run, not re-score: I authorise re-running every counted national-2 configuration (1–6) and both no-KD twins on the corrected labels with every setting, seed and code path unchanged (logged re-runs, not new configurations, not counted against the cap of 8). The frozen-list baselines are refitted on the corrected rows. Selection rule S is applied once to the corrected validation rows; the as-built and corrected validation tables are both reported; nothing is changed between them. These re-runs start only AFTER my stamp (the label version must be pre-specified before any validation number on it).

3. P2 index: permission granted to rewire to the label rule and rebuild. Only the label filter changes; the lock file (f3d4c63e…52be) and every constant of Q6 stay; new index hash in the addendum; test_p2 stores no value, as now.

4. C2, the 103 already-scored teacher rows on M2 Test-S lakes: the six registry rows (b41eab495b6c, bc70b284498d, b9c6bb9ffafe, d89d96326ac8, d544efe3f909, b9f03a45c672) stay as they are (registry is append-only) and are disclosed in every M7 and M2 write-up. Nothing further is scored: no teacher output, and no Sentinel-2-input model of any kind, is scored by M7 code on an M2 Test-S site-date (orbit-v3 test-role lake, in-lake, 2015-06-23..2023-12-31) until I tell you M2S/test is sealed. The teacher step of scripts/eval/national_lite_score_test.py (l.178–189) must drop those site-dates through your exclusion flag and hard-fail if one reaches it; if that script is covered by any hash already pinned, make the change in a new file and say so, otherwise edit it and add a unit test. Write the 103 keys (keys and registry row ids only, no values) to a file under sentinel_runs/lite/counts_arch_v2/ and send me path + sha256 — I need it for the M2 ruling.

5. The 3 lock sites within 100 m of a reserved site under another id (2 test_p2, 1 val_p2; none has an index row or chip): the lock is not changed. All three are excluded from every role at load time (never trained on, never scored), by id list in the deviations file, and the loader hard-fails if any of them ever gets an index row. Same treatment for the one CBP_WQX train_p2 site if its coordinates match the M1 test-role station (F10).

6. TabPFN (S10) row cap: 10,000 (the stronger comparator), per the F2 ruling. Only if 10,000 rows x 98 features does not fit in one 11 GB GPU, use the largest of 10,000 / 8,000 / 6,000 that fits, decided by a memory test on training rows only with no validation row loaded, and registered before any validation number. tabpfn 2.2.1 and the checkpoint on disk only, through sentinel.models.chemtox.tabpfn_model.load_tabpfn; no install, no download.

7. (i) train_p2 block rule: the F7 ruling as written. (ii) DenseNet121: torchvision DenseNet121_Weights.IMAGENET1K_V1, one file. First add the rows to docs/licenses.md (DenseNet121 weight file: source URL, licence as stated by the source — look it up, do not write it from memory —, sha256 after download; the MobileNetV4 weights already in use; HydroLAKES; M7 added as NWIS user). Once the rows exist, I approve that single download: into its own new empty directory under the data root, nothing installed, sha256 into a registry row before use. (iii) P2 is descriptive this round (F8); carrier = INT8 seed 0 on both tracks (DS-09); cut-off for slots 7–8 stays 2026-11-16 23:59 EST by registry timestamp (DS-07).

GUARD (F9): no lite or litev2 loader returns a val, val_n or val_p2 row to a v2 arm, comparator or re-run until a guard with a unit test has verified a registry row with workstream "M7-lite", experiment "lite/prereg_snapshot", written_by "lead", whose extra.sha256 holds the amendment (2b369018…8a56), the addendum, the deviations file and lite_arch_v2_lead_rulings.md at their current hashes, all four read-only. Build it now; it must report "not stamped" today.

Before the stamp you may: write code and tests, build the label version / index / masks (protected rows: counts only), run synthetic or train-only smokes. No fit of any v2 arm or comparator on real validation rows, no re-run.

Order: label version + pin → P2 index → F7 and mask counts (registry rows) → guard + tests → licence rows → addendum + deviations entry 1 → send me the three paths + sha256 and the test summary. I then check, write the rulings, set lite_protocol.md and lite_national_protocol.md to 0444, and stamp. After the stamp: report at once (job name, GPU, PID, start time) when the first re-run or v2 reference fit starts — use the GPU lock helper, GPUs 7–9 are idle now.
```

## Annex B — the 18:26 EST message of 2026-11-10

Sent by the lead to the M7 workstream agent; transcript timestamp `2026-11-10T23:24:38.462Z`, 1213 characters,
reproduced verbatim.

```text
Lead: your run stopped on a usage limit at about 15:30 EST; it is now 18:26 EST on 2026-11-10. Please continue (you were writing litev2/stamp.py). First run `date`, re-read your handoff and re-check your files and processes. One ruling to build in now, coming with the M2 stamp I am writing: until M2's Test-S look (M2S/test) is sealed, no workstream scores a Sentinel-2-input model on a row of an M2 test-role water body dated ≤ 2023-12-31. For you this means (a) the national look's teacher step stays off for such rows (mask them by keys via your tests_mask module), and (b) I need from you a keys-only list of the rows of M2 test-role water bodies that the 2026-11-09 4-state look scored with the S2-input teacher (M2 water-body id, S2 sensing date, source; no label, no prediction, no error column), as one file with its sha256, because M2 must drop matching Test-S rows before its dev_v5m build. Write it from your existing keys files (tests_overlap_item4.json / supp_counts_item345.json); do not open a look prediction file for it. My rulings on your other waiting items (WQP rule for M7, TabPFN row cap, train_p2 block rule, the 3 lock sites) follow after the M2 stamp. Keep messages short; be economical.
```

## Annex C — the 18:45 EST rulings of 2026-11-10 (items 1-4)

Sent by the lead to the M7 workstream agent; transcript timestamp `2026-11-10T23:33:07.387Z`, 2324 characters,
reproduced verbatim.

```text
Lead rulings for M7 (18:45 EST). My 15:05 order stands in full; the last sentence of my 18:26 message ("rulings … follow") was wrong — they were already in the 15:05 order. You are right to work from it.
1. Keys file accepted as the superset: sha256 4786aa12…c699 verified by me (110 rows, keys only). I do NOT want the 103; do not open the prediction file. M2 will drop every match of all 110 rows. Set the csv and its meta.json read-only (0444) now and do not rewrite them; I pin the csv sha in the M2 rulings.
2. Finding (a) (whole-version effect: test_n 39 of 773, val_n 36 of 780, chl-a training rows −8.9 %): no change. The label rule is one project rule, fixed before any number on the new version and applied without looking at values, so the rows go. Addendum item (i) carries these counts by role, target and drop reason, and the power/MDE statements in the addendum are restated on the corrected counts. My 2(b) STOP check is passed as you report.
3. Finding (b) (4 national-2 train chips at sites of the reserved new-site block): yes, deviations entry. Wording to include: the reserved-site rule applies at load to every role; the "unchanged" re-runs of configurations 1–6 train on 4 fewer chips, and that is said wherever they are called unchanged; the as-built national-1/national-2 models were trained on those 4 chips (M7 labels at sites of M1's reserved block; no M1 sonde reading involved), disclosed in every M7 write-up. Add the count of chips at reserved sites in val, test and lock roles (I expect 0; if not 0, stop and tell me before anything else).
4. Finding (c) (S9 list secchi, tss, chla; none for turbidity 3 and ssc 0): accepted. In the addendum cite the amendment line that sets the minimum count; if no minimum was written, say so and state that any minimum from 4 to 47 gives the same list.
Speed: GPUs 3–9 are idle and the project owner wants them used (GPU 2 is another user's — never touch). Nothing of v2 may be fitted before my stamp, so the fastest path is the stamp package: pin row + tests, P2 index, F7/mask counts, teacher-step mask, licence rows, addendum + deviations with hashes. Send it as one message when complete. Plan the post-stamp runs so the re-runs of configurations 1–6, the twins and the v2 arms go out in parallel, one job per GPU through the lock helper, each ≤ 2 h.
```

## Annex D — the relay of the M2 stamp (conflict C2), 2026-11-10 evening

Sent by the lead to the M7 workstream agent; transcript timestamp `2026-11-10T23:59:39.670Z`, 1075 characters,
reproduced verbatim.

```text
Lead: one binding rule for M7 from the M2 stamp (registry row 51a841e48437, docs/preregistration/orbit_arch_v2_lead_rulings.md R-1 / C-1) — read at your next step, no need to stop running work.
Until M2S/test is sealed, no workstream, the M7 national look included, scores a Sentinel-2-input model on a row of an M2 test-role water body dated <= 2023-12-31. Conflict C2 applies to M2 as written for M7.
Your keys-only list sentinel_runs/lite/counts_arch_v2/c2_teacher_scored_m2_test_lake_keys.csv is now pinned in the M2 stamp at sha256 4786aa121eb673ced502d487fb4af714fff37027677677048d2024c960cfc699 (110 data rows). Do not edit, regenerate or move it; if you ever need a corrected list, write a new file and tell me first. M2 reads that list and nothing else of M7.
Put this rule into your stamp package / amendment text where C2 is described (new text only, nothing frozen edited), and confirm in your next report that no planned v2 fit or score touches such a row. Also: idle GPUs (8, 9 right now) are yours to use through the lock helper for approved work; never GPU 2.
```

## Annex E — the 19:38 EST message of 2026-11-10

Sent by the lead to the M7 workstream agent; transcript timestamp `2026-11-11T00:38:07.270Z`, 1790 characters,
reproduced verbatim.

```text
Lead to M7 (19:38 EST): keep your stricter default — no fit of any kind on real rows before my stamp, TabPFN memory test included.
Reason: my 18:45 order said nothing of v2 is fitted before the stamp, and the stamp is close (my own full read of amendment v0.3, addendum, deviations 1–11, both protocols and the guard is done; the independent review of the package is running now and is the only thing I am waiting for). Synthetic-tensor tests remain fine.
While you wait, three things, none of which touches a label:
1. Send the dated code-readings notes file (S8 columns, aux KD head position, sum over targets in the loss, …) with path + sha256 as soon as it is final. If any reading changes a stamped rule it becomes a deviation entry in a NEW file; do not append to lite_arch_v2_deviations.md now (the reviewed hash is 23e6fd87…642c and I stamp that byte state).
2. Deviation 10: check and tell me whether the hash list of the national-2 freeze candidate covers the scorer's full import closure, including sentinel/models/orbit/wqp_surface.py, sentinel/models/orbit/data.py and sentinel/data/matchups.py, and whether a test asserts it.
3. Prepare the launch order so nothing waits after the stamp: (a) TabPFN memory test, (b) comparators and the corrected-label re-runs of configs 1–6 and both twins, (c) slots 7–8. Plan (c) to start only after the config-3 re-runs exist — I expect to rule that slots 7–8 and their twins warm-start from the corrected-label re-runs, seed-matched (your proposal in entry 11); wait for the rulings file for the binding text.
From the user (18:50 EST): idle GPUs may be used to speed up approved work, inside the plan. Lock helper for every job, ≤ 2 h per job, never GPU 2 (another user's job); CPU is tight (load ~70), so ≤ 6 threads and prefer GPU work.
```

## Annex F — the 19:42 EST message of 2026-11-10

Sent by the lead to the M7 workstream agent; transcript timestamp `2026-11-11T00:39:33.853Z`, 663 characters,
reproduced verbatim.

```text
Lead to M7 (19:42 EST): do it. Add sentinel/data/matchups.py, sentinel/models/orbit/wqp_surface.py and sentinel/models/litev2/__init__.py to NATIONAL_CODE (not to guard.SHARED_CODE; FREEZE_lite.json and its hash list stay untouched), and add the national import-closure test in a new test file: it must fail if any repo file in the closure of scripts/eval/national_lite_score_test.py is missing from national_frozen_files(). Do not alter the failing 4-state test to make it pass; it stays as the record of deviation 10. Report the two file hashes and the test result in your next message. The rulings file will state this as the condition for the national freeze.
```

## Annex G — the pre-stamp order of 2026-11-10 (labelled 20:20 EST)

Sent by the lead to the M7 workstream agent; transcript timestamp `2026-11-11T01:14:44.618Z`, 4272 characters,
reproduced verbatim.

```text
Lead to M7 (20:20 EST, 2026-11-10): the independent review of your stamp package is back — verdict "stamp with lead rulings". I am writing lite_arch_v2_lead_rulings.md now. The stamp waits only on the code items below. Everything here is my INTENDED ruling; it binds only once it is in the rulings file.

DO NOT TOUCH: lite_arch_v2_amendment.md, lite_arch_v2_label_addendum.md, lite_arch_v2_deviations.md (I stamp the reviewed byte state 23e6fd87…642c — no append now; the correction of entry 10 and your X4/X5 reading go into my rulings file), both code-readings files, lite_split_national2b.json, labels_national2b.parquet, the P2 index, lite_p2_lock.json, the C2 csv + meta, FREEZE_lite.json, tests/test_lite.py. No fit on a real row, no validation row loaded.

A. Code to prepare now (synthetic tensors / train-role rows only), each with a unit test:
 1. Re-runs of configs 1–6 and both no-KD twins apply the Test-S KD mask (weight 0 on the 308 pairings) — the third and last exception to "unchanged settings". Test: a masked pairing contributes 0 to the KD term on the re-run path.
 2. Slots 7–8 warm-start, seed for seed, from the corrected-label re-run of config 3; their twins from the re-run no-KD twin; S11 embeddings from the same re-run checkpoints. No v2 job loads an as-built checkpoint; no fallback (missing re-run checkpoint = hard error). The checkpoint sha256 is registered before the first slot job. Test for the hard error.
 3. The three missing Q5 tests: (i) the refit table holds no val, val_n, val_p2 or test-role chip or water body; (ii) no train_p2 water body equals a national-2b training water body under a different id (as built: no train_p2 site within 2 km of a national-2b train chip); (iii) the evaluation anchor and S6 are one artefact with one registered sha256.
 4. Stricter order in code: no candidate FIT (not only evaluation) starts before every comparator validation row of H3 is registered for both blocks. That answers your 19:48 question: comparators, re-runs and twins first, in parallel on all idle GPUs; candidates after.
 5. scripts/train/litev2_comparators.py: a rerun-view path, so that if one of configs 1–6 is selected, S6–S11, P1, P2a/b can be refitted on the rerun-view training rows (CPU, no tuning). Needed before the national freeze, not before the stamp — do it last.
 6. E3a_turb: a one-shot count step (frozen scorer recounts positives and water bodies on val_n once after the stamp, registered; < 30 positives or < 10 water bodies → descriptive only on the national track). Code only now.
 7. Per-job GPU-hour logging against a hard ceiling of 45 GPU-h for the closed run list of the addendum's compute table (job refuses to start if the logged total + its own budget would pass 45).

B. Your 20:07 questions, intended rulings: INT8 carrier = scope="image" (INT8 image branch, FP32 context / pooling / heads / anchor sum), fixed now before any trained model; scope="graph" is also exported and its metrics reported, descriptive only; every write-up says "mixed precision", never "fully INT8". Export parity = your three conditions, thresholds fixed now and never loosened after a failure; in addition report, on real validation rows, the count > 1e-4 and the max of the float64-vs-float32 anchor difference (reported, not a pass condition). ANCHOR-Set stays as written (307,341 parameters). 12 DenseNet runs are inside the ceiling.

C. Report to me in ONE message when A1–A4, A6, A7 are done (A5 may follow later): (1) which unit selection rule S used for configs 1–6 — 3-seed ensemble or single seeds — with file:line and the registry row it is in; (2) the full pytest result of every litev2 / lite v2 test file (counts) and the one known-failing 4-state closure test; (3) sha256 of every file under sentinel/models/litev2/, every scripts/train/litev2_*.py, scripts/eval/national_lite_score_test.py and every tests/test_litev2_*.py, as one JSON file under sentinel_runs/lite/counts_arch_v2/ (path + sha256). After that message, change no code until the stamp row exists — I re-hash, run the tests myself and stamp.

After the stamp, launches use the lock helper on idle GPUs (never GPU 2), ≤ 2 h per job, ≤ 6 CPU threads per job (CPU load is above the core count — prefer GPU work). Keep it short; be economical.
```
