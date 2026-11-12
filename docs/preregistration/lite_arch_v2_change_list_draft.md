# SENTINEL-Lite architecture round v2 — draft change list (M7, NOT a ruling, NOT stamped)

Status: author's working list, started 2026-11-10 14:47 EST. It changes nothing by itself.
The amendment `docs/preregistration/lite_arch_v2_amendment.md` is DRAFT v0.3, sha256
`2b369018d2fc2a340594790756b3edbef3b1de28a38cf6073eb5908111908a56`, frozen for review and not edited.
Sources: `docs/preregistration/reviews/arch_v2_20261110/M7_review_design.json` (DS-*),
`M7_review_integrity.json` (F*), `M7_verify.json` (verifier; "missed" items V-m0…V-m7), and the counts under
`/home/bcheng/STOCKHOLM/sentinel_runs/lite/counts_arch_v2/`. Every item waits for the lead's ruling; the wording
below is the author's proposal of what an addendum or a v0.4 would say, and the facts the author can supply.

No label value of a protected role and no validation number of a new arm, comparator or P2 is in this file.

## A. Statements of fact in v0.3 that are stale (F1, DS-01)

| Where in v0.3 | Stale statement | Fact on 2026-11-10 |
|---|---|---|
| §0 l.26–32 | P2 code, lock, index and chips "not built" | lock `lite_p2_lock.json` sha `f3d4c63e…52be` (registry row `03323db99ad9`); index 513 rows, sha `68688fac…ff44` (row `e6a469d28fe1`); 515 chips + 1 error row; §0 l.68–75 is the state |
| header | "No rule, constant, model, comparator or threshold was changed" (v0.2 → v0.3) | v0.3 added the several-limits sentence of §3.3 and the all-zero-pixel criterion of §8.5 |
| §0 table | lists the no-KD twin of config 3 only | a no-KD twin of config 6 also has validation rows (18:12 UTC); "6 counted configurations + 2 uncounted no-KD twins" |
| §0 l.37–38 | stamp guard reads `extra.sha256[<path>]` | the existing snapshot row `d2e93e086ce7` stores `extra.files[path].sha256`; the guard does not exist yet (to be written with a refusal test) |
| §8.3 l.364–365, l.383 | eligible test_p2 count unknown | known by counts (189 chips, 38 water bodies as built; changes under the label rule) |
| §8.4 last bullet | P2 chips not fetched | fetched |
| §3.3 l.140–143 | several right-censored limits: "smallest one for >" | `p2_index.py` keeps a right-censored chip only with one limit (`cens_n == 1`); P2 has 0 censored site-dates |
| l.16 | `976b5be7…` cited as hash of `lite_split_national2.json` | it is the canonical split hash of `national2.split_sha256`, not the file hash |
| l.291, l.482, Q4, §12 | TabPFN "package absent" | tabpfn 2.2.1 and its regressor checkpoint are on disk; loader `sentinel.models.chemtox.tabpfn_model.load_tabpfn` |
| file mode | 0664 | 0444 at the stamp (lead) |

Log event to record as deviation 1 (F9): `/home/bcheng/STOCKHOLM/sentinel_runs/lite/logs/p2_plan.log` held 34
train_p2 and 12 val_p2 label values and was deleted on 2026-11-10 18:08:03 UTC; no test_p2 value was read; no
model or comparator code of this amendment existed then (only `litev2/{__init__,p2,p2_index}.py` and
`scripts/data/build_lite_p2.py`). No log is deleted again.

## B. WQP label rule (F3, DS-02, V-m0, V-m1, V-m2)

Proposal: the project rule of LEAD_DECISIONS_20261110.md ("Shared WQP label defect") for every WQP label of
national-2b and P2, in every role, for every model and comparator, through M2's reader
`sentinel/models/orbit/wqp_surface.py` (imported by `sentinel/models/litev2/wqp_rule.py`; no second copy).

Facts (keys only; `wqp_sample_counts.json`, `wqp_defect_counts.json`, `supp_counts_item345.json`):

- `national_pm7` matchup table, 51,381 rows: 1,609 rows have no QC-passed surface result (would be dropped),
  1,971 rows keep their sample but lose member results. `national` table, 21,455 rows: 1,430 and 1,494.
- The counts of §3.1 do not survive the rule (FNU train 128 → at most 115; FNU without role 237 → at most 201).
- Open points for the ruling:
  1. **Pinned split hash.** `national2.FROZEN_SHA256 = 976b5be7…` covers `labels_sha256`; any corrected label
     changes it. Proposal: chip roles and water-body ids unchanged (asserted on `chip_role` alone); corrected
     labels are a new label version with its own hash in a new file; `976b5be7…` stays the pin of the as-built
     version.
  2. **4-state rows inside national-2** (`source=frozen`, 5,998 chips). They were built from `raw/wqp` with the
     WQX depth fields and raw units; their sample keys are not keys of M2's side table, so `wqp_surface.apply`
     refuses them. Depth rule was applied at build; QC-status filter of the harmonised tables was not. Options:
     keep as built (frozen, stated), or re-derive from the national table rows of the same chips.
  3. **Configs 1–6 and the selection already run** on as-built labels: re-run, or re-score only.
  4. **P2 index**: one change (`depth_m` read; `wqp_rule.surface_samples`); index hash `68688fac…` changes; the
     lock does not.
  5. §3.1 table and the S9 target list are recounted under the rule and registered before any validation number.

## C. M2 Test-S (F4, DS-05, conflict C2)

Proposal: KD weight 0 and no teacher-derived feature on every teacher pairing on an M2 Test-S site-date, in every
v2 arm, ablation and comparator (S8, P3b included); mask built by rule from keys
(`sentinel/models/litev2/tests_mask.py`); the labelled chips stay in training; CIR-Lite-R gate cells exclude the
same lake-dates; no M7 output is an input to anything scored on Test-S.

Facts (`tests_overlap_item4.json`): national-2 train chips on Test-S 777 (510 sites; 308 with a teacher pairing);
val 114 (41 with a pairing); val_n 18 (9). 308 (national-2 train pairings) and M2's 242 (chips of the 4-state
teacher store in Test-S lakes, any Lite role) count different populations. Reconciliation
(`supp_counts_item345.json`): the 4-state label table has 2,108 teacher pairings, 243 of them on Test-S (241
distinct S2 chips, 130 water bodies: train 96, val 41, test 67, test_time 39) — M2 counted 242 chips in 131 water
bodies from the store of 2,130 rows; national-2 has 10,442 pairings, 1,275 on Test-S (train 308, val 41, val_n 9,
test 67, test_n 20, test_time 39, test_time_n 16, excluded 775). Flag table (keys only):
`sentinel_runs/lite/tests_mask_national2/flags.parquet`, sha `ba7fcc60…3c0e`. The Lite-safe national teacher's
own 7,903 training points are all in orbit-v3 train-role geography (0 on a test-role lake).

Already happened (to disclose; F-M7-SCORED of `M2_review_leakage.json`): the 2026-11-09 4-state look scored the
S2-input teacher on 103 rows (65 `test` + 38 `test_time`) = 102 distinct (site, S2 date) in M2 Test-S lakes,
registry rows `b41eab495b6c`, `bc70b284498d`, `b9c6bb9ffafe`, `d89d96326ac8`, `d544efe3f909`, `b9f03a45c672`.
The national look script repeats the teacher step (`scripts/eval/national_lite_score_test.py` l.178–189): to be
masked or removed by ruling before any national look.

## D. Comparators and arms

- **S10 TabPFN v2** (F2, DS-04): via `load_tabpfn()` only, S6's 98 features, same rows/label rule/targets; row cap
  10,000 (verifier) against `MAX_TRAIN_ROWS_DEFAULT = 6000` (lead's 13:26 note) — needs one number; subsample rule
  by salted water-body hash; no fit before the stamp.
- **S11 GBM-stacked** (DS-06): member of the comparator sets, no longer "descriptive only".
- **No-KD twins** (F5, DS-08): uncounted, never selectable, named in a registry row before they run, always listed.
- **Slots 7–8** (F6, DS-07, DS-13): frozen §8a label only; cap of 8 fixed by ruling; entry by registry timestamp
  2026-11-16 23:59 EST; a slot with any validation number is counted.
- **Carrier** (DS-09): INT8 pipeline of seed 0 on both tracks; the ensemble sentence is void.
- **Seed rule** (DS-10): diff < 0 for each of the 3 seeds against the same comparator.
- **DenseNet121 / LR tie / S9 / restarts** (DS-11): torchvision `DenseNet121_Weights.IMAGENET1K_V1`; tie keeps
  1e-4; S9 target list registered; restart test = no evaluation file and no registry row.
- **CyFi-style days-between feature** (V-m5): kept, marked as a label-side input.

## E. P2

- Descriptive this round, one hardened look, no label (F8, DS-03, Q12); "rivers and streams only", number of
  test water bodies and effective number reported; nothing selected on val_p2.
- train_p2 block rule (F7): train_p2 water bodies in 25-km blocks holding a national val/val_n/test/test_n role
  are removed from v2 training rows by keys; salted national-1 rule for blocks without a Lite chip.
- Lock unchanged (F10): one CBP_WQX train_p2 site checked against an M1 test-role site by coordinates; Fuse and
  Track-N overlap counts recorded; test_p2 site-dates join the campaign's reserved list.
- "Unseen water bodies", not unseen basins (V-m3): stratum of test_p2 chips whose HUC8 and block hold no v2
  training row, counts registered before the look.
- Reserved 729 new sonde sites: 0 in the lock, the index and the chips (ids); by coordinates 3 lock sites lie
  within 100 m of a reserved site (test_p2 2, val_p2 1; none has an index row or a chip) — for the lead.

## F. Process

- Deviations file `docs/preregistration/lite_arch_v2_deviations.md` and a stamp guard in every litev2 entry point,
  with a refusal test (F9, DS-12).
- Licence rows before any v2 run: DenseNet121 weights, MobileNetV4 weights, HydroLAKES, NWIS for M7 (F11).
- Compute ceiling of §10 restated with S10, S11, the slot-7/8 twin and any label-rule re-run (V-m7).
- v1 4-state results stay as reported, with a counts-only erratum on the label defect (V-m7).
- Frozen texts' hashes recorded and both set to 0444 (V-m6; lead).
