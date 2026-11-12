# M7 SENTINEL-Lite v2 — independent pre-stamp review

Reviewer: independent sub-agent, 2026-11-10, about 19:10–19:55 EST. I wrote none of the reviewed files.
Machine-readable copy: `review.json` in this directory (it holds the full verifier sentences and ruling texts).

## Verdict: `stamp_with_lead_rulings`

No blocking item was found: nothing in the package needs to change that a tightening lead ruling cannot settle.
Nine points need a ruling sentence (R-01 … R-09, of which R-08 is the package's own five open points). One
correction to the deviations file (entry 10) is best appended before the stamp, because the stamp row hashes that
file; a ruling sentence can carry it instead.

## Hashes

All six package items matched the expected sha256 at the start, at a mid-review re-hash (about 19:38 EST) and at
the end (19:53 EST and again when `review.json` was last built). The C2 keys csv is mode 0444.

| File | sha256 (start = end) |
|---|---|
| `docs/preregistration/lite_arch_v2_label_addendum.md` | `bcbd86de…9049` |
| `docs/preregistration/lite_arch_v2_deviations.md` (181 lines, entries 1–11) | `23e6fd87…642c` |
| `docs/preregistration/lite_split_national2b.json` | `03f0114a…5663` |
| `docs/preregistration/lite_arch_v2_amendment.md` (v0.3) | `2b369018…8a56` |
| `…/sentinel_runs/lite/counts_arch_v2/c2_teacher_scored_m2_test_lake_keys.csv` | `4786aa12…c699` |
| `sentinel/models/litev2/stamp.py` | `0ecf0f3c…f0fb` |

A seventh M7 file appeared during the review and is not in the package:
`docs/preregistration/lite_arch_v2_code_readings_20261110.md` (0444, 19:47 EST, sha256 `002d5d08…a4d4`) — see R-07.

## What the nine checks found

1. **Label rule, counts only.** The rule is applied through M2's reader and the unchanged
   `wqp_surface_samples` with depth; sidecar files are never read; the ambiguous "Chlorophyll" rows sit in
   `_unmapped.parquet` only. Every count file holds integers, percentages and effective numbers, no label
   statistic. One exception to "never by comparing values": the two P2 index builds were compared for equality on
   `train_p2` and `val_p2` and the number of differences (0) was reported (N-01). I could not check what the
   builders printed at run time: no build log exists.
2. **Split `lite_split_national2b.json`.**
   (i) On the water-body id of record, test, test_n, val, val_n and train share no water body. Under the regrouped
   id `national2b_wb` they do (169 of 1,178 test and 79 of 773 test_n chips are in a group that holds a training
   chip); this is inherited from national-2, no added chip is involved (N-02).
   (ii) No val, val_n or train water body of the earlier split is in a test role; 0 of 32,820 chips changed role.
   (iii) Every floor holds or is reported as missed and none is lowered: N2 holds for every test-role target; the
   P2 floor is missed (187 chips, 38 water bodies) and stays; the N3 floor on Test-N cannot be checked from counts.
   (iv) The split is built from the role mapping and keys; no test-role label value enters (code read).
   (v) My recount equals the addendum for every role and target.
3. **Verifier items.** Table below; 45 items: 16 closed in the package, 28 need a ruling sentence, 1 open (Q5).
4. **Deviations 1–11.** None lowers a bar, changes a hypothesis or family, or by itself changes comparator parity.

   | Entry | Assessment |
   |---|---|
   | 1 | True disclosure of pre-stamp builds; constants frozen as built. The deleted log cannot be re-read. |
   | 2 | Tightening: 4 lock sites excluded at load; 0 index rows at them (verified). |
   | 3 | Tightening: 4 train chips at reserved sites removed (10,595 → 10,591 at load); rests on an order not on file. |
   | 4 | Effect of the label rule on P2: index 513 → 511, usable test_p2 189 → 187 by key. Holds the value-equality count of N-01. |
   | 5 | Stricter reading of F7: 128 → 60 usable train_p2 rows, the same rows for every arm and comparator. |
   | 6 | Eight train-side label choices, applied equally to candidate and comparators; none adds evaluation freedom. |
   | 7 | Documentation of overlaps and the reserved-list rule. |
   | 8 | C2 disclosure and a wider teacher-score block; in-place edit of the not-yet-frozen national scorer. |
   | 9 | Pre-stamp download, licence unstated at source, approval not on file (open point e, R-01). |
   | 10 | Incomplete and partly wrong: see R-03. |
   | 11 | The five open points: see below. |
5. **Open points (a)–(e).** Recommendations and exact texts below.
6. **Guard.** `stamp_status` opens only when the four files (amendment, addendum, deviations, rulings) exist, are
   not writable, the amendment has the pinned hash, and one registry row has experiment `lite/prereg_snapshot`,
   workstream `M7-lite`, `extra.written_by == "lead"` and the four current hashes. Exercised on a temp registry
   and temp tree only. Today it is closed (no rulings file, no lead row). It does not cover the split pin, the
   label table, the P2 index, the protocols or code (R-02), and `written_by` is a plain string (N-03).
   Tests: run 1 (19:33–19:36 EST, 5 files) 81 passed, 1 failed; run 2 (19:43–19:45 EST, 7 files) 111 passed,
   1 failed. The one failure both times is `tests/test_lite.py::test_scoring_hash_covers_scorer_import_closure`
   (deviation 10). M7 was editing code during the review, so these are snapshots (N-06).
7. **Round rule 2.** No registry row and no run directory of any new arm, comparator, re-run or P2 validation/test
   exists. Only train-only caches exist (N-05).
8. **Parity, §15.1, power.** Comparators are fitted on the v2 rows only; a selected configuration 1–6 would have
   no added comparator on its own rows (R-04). Power from counts is given for P2 and national validation but not
   for Test-N (R-05).
9. **Other.** Lead orders the package relies on are not on file (R-01).

## Findings

| Id | Severity | Where | What | Proposed ruling (short) |
|---|---|---|---|---|
| R-01 | ruling_needed | addendum l.3–4, l.55; deviations l.45, l.60, l.77, l.153; `docs/licenses.md` l.53; code comments | Lead orders of 15:05, 18:26, 18:45, 19:38 and "19:42" EST are cited but on file nowhere; one is cited at a time that had not yet come. | The rulings file reproduces each cited order verbatim; an order not reproduced is void. |
| R-02 | ruling_needed | `sentinel/models/litev2/stamp.py` l.36, l.83–88; `labels2b.py` l.57 | The guard covers four markdown files; split pin and label table are 0664 and unguarded, protocols 0664. | Stamp row also records the sha256 of split pin, label table, P2 index, lock, C2 csv and meta, both protocols, code-readings file; split, labels and protocols set 0444. |
| R-03 | ruling_needed | deviations l.163–170 | Entry 10 names one edited file; two were edited (`orbit/data.py`, `orbit/pipeline.py`). The frozen hash `a7a8be82…` is reproducible from git commit `34df74e` (I reproduced it). | Frozen records and the failing test are not edited; the ruling states the two files and the commit; M7 appends a correcting entry; no 4-state re-score; the national freeze hashes its full closure. |
| R-04 | ruling_needed | addendum l.204–217, l.345–370; `scripts/train/litev2_comparators.py` l.144 | Two training-row sets; added comparators exist on one. Verifier Q1b is not met if a configuration 1–6 is selected. | Selected configuration must pass H1 against comparators on its own rows; S6–S11, P1, P2a/b refitted on rerun-view rows if a config 1–6 is selected; v2-row comparators apply as well. |
| R-05 | ruling_needed | addendum l.438–470 | No power statement from counts for Test-N after the label rule. | State Test-N counts 379 / 460 / 262 / 350 chips in 152 / 126 / 102 / 137 water bodies and half-widths about 9 / 13.5 / 17; no floor changed. |
| R-06 | ruling_needed | addendum l.342–343, l.295–296 | The intended ruling text has as-built counts. | Write 778 (777 as built) and 187 chips in 38 water bodies (189 as built). |
| R-07 | ruling_needed | `docs/preregistration/lite_arch_v2_code_readings_20261110.md` | 37 code readings written after the package was hashed; four touch a stamped number or sentence; outside the guard. | Adopt by hash; comparators' validation rows before any candidate fit; selection score on the same unit as configs 1–6. |
| R-08 | ruling_needed | deviations l.172–181 | The five open points. | See (a)–(e). |
| R-09 | ruling_needed | `tests/test_litev2_tabular.py` l.83; verifier Q5 | Three of the four unit tests Q5 requires before any validation number were not found. | The three tests exist and pass before the first validation row is written. |
| N-01 | note | addendum l.9–12 against l.232–233 | Value-equality comparison on `val_p2` (count 0) against "never by comparing values". | One sentence stating the exception. |
| N-02 | note | split pin; label table | Disjointness holds on `water_body`, not on `national2b_wb` (inherited). | Water body of record is `water_body`; sensitivity row without the 169 + 79 chips. |
| N-03 | note | `stamp.py` l.83 | `written_by` is unauthenticated; a deviations append closes the guard until a new lead row. | Lead records the stamp row id. |
| N-04 | note | both protocol files | Verifier M6 not done (0664, hashes unstamped). | As R-02. |
| N-05 | note | deviations entries 1, 8, 9 | Pre-stamp acts that cannot be undone or re-read (deleted log, scorer edit, download, train-only cache, `--train-smoke`). | Nothing made before the stamp is used by a v2 run; no v2 log is deleted. |
| N-06 | note | litev2 code and tests | Code changed during the review; test counts are snapshots. | Tests run once more on the code that will run; count and hashes recorded with the stamp. |

## Verifier items (`M7_verify.json`: 17 verdicts merged to 16 ids, 20 points, 8 missed)

| Status | Ids |
|---|---|
| closed_in_package (16) | F3, F7, F10, F11, F12+F13+DS-14, Q2, Q6, Q10, Q11, P15, P19, M0, M2, M3, M4, M5 |
| needs_ruling_sentence (28) | F1, F2, F4, F5, F6, F8, F9, DS-06, DS-07, DS-09, DS-10, DS-11, Q1a, Q1b, Q3, Q4, Q7, Q8, Q9, Q12, P13, P14, P16, P17, P18, M1, M6, M7 |
| open (1) | Q5 |
| dropped or contradicted | none |

The verifier's sentence for every item is quoted in full in `review.json` (`verifier_items[].sentence`). Changes
needed to those sentences:

- **F4:** 778 for 777; the warm-start sentence must be reworded if (a) and (b) are ruled as recommended.
- **F5:** the as-built checkpoint named there is replaced by the corrected-label re-run under (b).
- **F8, Q12:** 187 chips, 38 water bodies, 74 sites, effective 10.08.
- **F2:** the fallback caps 8,000 / 6,000 come from an order not on file; rule it or drop it.
- **DS-06:** "the config 3 network" is the re-run under (b).
- **Q1b:** see R-04. **M6:** not done. **Q5:** see R-09.

Consistency with the lead's intended rulings: TabPFN v2 through `load_tabpfn` — consistent (comparator code calls
it; cap rule as ruled). C2 — consistent (308 pairings masked; a masked pairing raises an error; `test_litev2`
tests pass), with 778 for 777. P2 descriptive, floor not lowered — consistent, with 187 for 189.

## Open points — recommendation and ruling text

**(a) Test-S KD mask on the re-runs** (deviations l.173–174; addendum l.365–369). Recommend: apply the mask.
F4 tolerated KD on those pairings only because it happened before the reservation and could not be undone without
retraining; the re-runs are that retraining and slots 7–8 would inherit it. Cost: the corrected table differs from
the as-built one by labels and by 308 KD pairings. Needs a wiring change in the re-run path.
> The re-runs of configurations 1-6 give KD weight 0 to the 308 teacher pairings on M2 Test-S site-dates, through
> the same key-built mask as the v2 arms; this is the third and last exception to "unchanged settings" (with the
> corrected labels and the 4 reserved-site chips), and no other setting differs. No fit started after the
> reservation uses a teacher value on a Test-S site-date. Tables of as-built against corrected validation scores
> name these differences. A unit test shows that the re-run KD tensors hold no masked pairing, before the first
> re-run starts.

**(b) Warm start of slots 7–8** (deviations l.175–177). Recommend: the corrected-label re-runs, fixed by rule, no
fallback.
> Slots 7 and 8 start, seed for seed, from the re-run of configuration 3 on national-2b labels; their no-KD twins
> start from the re-run of the no-KD twin of configuration 3; S11 takes its embeddings from the same re-run
> checkpoints. The as-built checkpoints named in the verifier's F4 and F5 texts are loaded by no v2 job. The sha256
> of each re-run checkpoint is registered before the first slot-7 or slot-8 job starts; if a re-run checkpoint is
> missing the slot does not run (restart only under the DS-11 rule), and there is no fallback to an as-built
> checkpoint. The cut-off of DS-07 is unchanged. Write-ups state: slots 7-8 warm-start from configuration 3 as
> re-run on corrected labels with KD weight 0 on Test-S pairings; the as-built configuration 3 had been trained
> with KD on those pairings before the reservation.

**(c) Compute ceiling 45 GPU-h** (deviations l.178; addendum l.372–387; amendment l.497–508). Recommend: accept
as a hard stop over a closed run list, never raised after the first number, references run first.
> The hard ceiling of this round is 45 GPU-h, replacing 30 of amendment section 10, for exactly the runs of the
> addendum's compute table; no other run is added under it, and a gated slot of section 6 opens only by its own
> gate as written and counts under the same ceiling. GPU-hours are logged per job in the registry. Comparators,
> re-runs and twins are run before any candidate, so that reaching the ceiling can never remove a reference; a
> candidate run not finished under the ceiling is recorded as not run and still counts. The ceiling is never raised
> after the first v2 validation number exists.

**(d) N3 event counts on `val_n`** (deviations l.179; addendum l.468–470; national protocol l.43–55). Recommend:
no recount before the stamp; a recount after it can only remove the endpoint.
> The N3 endpoints, thresholds and floors are unchanged. The sentence "val_n has 48 positives in 18 water bodies"
> describes the as-built labels. After the stamp and before any v2 event number, the frozen scorer recounts the
> E3a_turb positives and their water bodies on val_n under national-2b once, and the two counts are registered; if
> val_n then has fewer than 30 positives or fewer than 10 water bodies with positives, E3a_turb is descriptive only
> on the national track. The Test-N floor of N3 is applied at the look as written. No threshold is changed in
> either case.

**(e) DenseNet licence** (deviations l.153–162; `docs/licenses.md` l.53). Recommend: run and report in any case;
publish no weight or derived checkpoint; publication needs the user's written approval after the terms are read.
> P3a and P3b are run and their results reported whatever they are; the licence question can never be a reason to
> omit them. The DenseNet121 weight file, every checkpoint fine-tuned from it and every artefact derived from it
> stay on the project disk: none is uploaded, released or served. Every write-up states that the weights are
> torchvision DenseNet121_Weights.IMAGENET1K_V1, that the source states no licence for the file and that the
> ImageNet-1k terms were not verified at download. Before any public write-up the terms are read, the licence row
> is completed, and the user's written approval is recorded. The lead's approval of the one download (2026-11-10)
> is recorded here with its time.

## Not checked

- No label value, prediction or score of any validation, test or Test-S row was read. So the N3 positives, the size
  of the label changes and any value-level correctness of the rebuilt labels are unverified.
- What the label builders printed when they ran (no build log exists); the deleted P2 log of deviation 1.
- The lead orders of 15:05, 18:26, 18:45, 19:38 and 19:42 EST: not on file, so every "as ordered" statement is
  unverified.
- The P2 lock rebuild was not re-run (hash compared only). The count of stored `test_p2` values (0) is the
  verifier's, not repeated.
- The code-readings file was read but not checked against the code; `nets.py` and `fit.py` were not reviewed.
- Whether selection S for configs 1–6 used the ensemble or single seeds.
- ImageNet-1k terms and the torchvision weight licence (no network).
- The code state that will run after the stamp (code changed during the review).
- GPU memory behaviour of S10 and every compute estimate.
- `lite_arch_v2_change_list_draft.md` beyond its first 25 lines.

## Evidence files in this directory

`recount.py`, `recount.out`, `recount2.py`, `recount2.out` (key and count checks), `pytest.out`, `pytest2.out`,
`code_sha_at_test.txt`, `code_sha_at_test2.txt`, `guard_tmp/` (temp-registry guard exercise), `build_review.py`.
