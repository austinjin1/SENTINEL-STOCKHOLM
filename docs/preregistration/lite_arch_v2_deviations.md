# SENTINEL-Lite architecture round v2 — deviations and disclosures

- Dated, numbered, append-only. An entry is never edited or removed; a correction is a new entry that names the
  entry it corrects.
- Belongs to `lite_arch_v2_amendment.md` v0.3 (sha256
  `2b369018d2fc2a340594790756b3edbef3b1de28a38cf6073eb5908111908a56`), `lite_arch_v2_label_addendum.md` and the
  lead's `lite_arch_v2_lead_rulings.md`.
- Counts only: no label value and no per-row statistic of a validation, test or lock row. No site id of the
  reserved new-site block is written here.
- Entries 1–11 were written by the M7 workstream agent on 2026-11-10 19:01 EST (2026-11-11 00:01 UTC), before the stamp.

## 1. Builds and prints made before the stamp (ruling F9) — 2026-11-10

- The P2 lock (`docs/preregistration/lite_p2_lock.json`, sha256
  `f3d4c63ebe503ac1774959bfcf943b45bc140883fe5b30933ee03f0cacca52be`, created 2026-11-10T17:57:34Z, salt
  "SENTINEL-Lite-P2|2026-11-10", registry row `03323db99ad9`), the P2 index (first build, sha256
  `68688fac1af2c4e147070562f3dde8a035eb150360dd49759619456baf23ff44`, registry row `e6a469d28fe1`) and **513
  P2 chips** were built before the stamp.
- While the index was planned, **34 `train_p2` and 12 `val_p2` label values were printed** to the log
  `$SENTINEL_RUNS/lite/logs/p2_plan.log`. The log was deleted on 2026-11-10 at 18:08:03 UTC. No model was fitted on
  those values and no decision of the amendment used them. No log is deleted from now on.
- **No `test_p2` label value was read**: for that role the builds read keys and a present/absent flag only, and
  the index stores no `test_p2` value.
- **Columns read by the lock build** (the full list; no label value column):
  - `naip_matchups/matchups.parquet`, `naip_matchups/national/matchups.parquet`,
    `naip_matchups/national_pm7/matchups.parquet`: `site_id`, `lat`, `lon`, `name`, `site_class`, `huc8`;
  - `sonde_daily/index.parquet`: `site_no`, `var`, `first`, `last`;
  - `sites/sites.parquet`: `network`, `site_id`, `name`, `lat`, `lon`, `site_type`, `huc8`, `in_conus`;
  - `wqp_discrete/variable=turbidity_fnu/data.parquet`: `site_id`, `state`, `huc8`, `lat`, `lon`, `site_type`,
    `activity_start_date`;
  - `wqp_discrete/_stations.parquet`: `site_id`, `name`;
  - `docs/preregistration/sonde_split.json` (`site_role`), `docs/preregistration/hydromet_track_n_stations.json`
    (`role_of`), `docs/preregistration/sonde_newsites_list.json` (through `sentinel.data.reserved`);
  - HydroLAKES polygons and ids and the block roles of orbit split v3 (through
    `sentinel.models.orbit.split` and `sentinel.models.lite.data.hydrolakes_ids`).
  The lock file itself names only the sonde index and the WQP FNU columns; this entry completes its list.
- Lock pool: 1,963 sites in 636 water bodies; `test_p2` 252 water bodies / 738 sites, `val_p2` 122 / 410,
  `train_p2` 262 / 815. The lock is never rebuilt and the salt never changed.
- **Eligibility constants are frozen as built:** dark-NIR water within 30 m of the chip centre ≥ 50 m², outside-image
  (no-data) share ≤ 1 %.
- Also built before the stamp, counts only, no model fitted: the label version national-2b and its pin, the rebuilt
  P2 index (entry 4), the F7 removal list, the Test-S mask and the count files of the addendum. No validation number
  exists on national-2b, on P2 or on any v2 arm or comparator.

## 2. Lock sites excluded from every role at load time (lead order item 5; ruling F10) — 2026-11-10

The lock is not changed. These four lock sites are never trained on and never scored; none has an index row or a
chip. `p2_index` drops them before any label is read and hard-fails (`P2IndexError`) if one of them ever has an
index row (unit test).

| Lock site id | P2 role | Reason |
|---|---|---|
| `MDNR-3256/1.8` | `val_p2` | within 100 m of a site of the reserved new-site block that is held under another id |
| `USGS-383848122215501` | `test_p2` | the same |
| `USGS-374955122313301` | `test_p2` | the same |
| `CBP_WQX-01608500` | `train_p2` | the same station (25 m apart) as an M1 test-role sonde site held under a USGS id |

Lock sites after the exclusion: 1,959.

## 3. Chips at sites of the reserved new-site block (lead ruling 2026-11-10 18:45 EST, item 3) — 2026-11-10

- The reserved-site rule (`sentinel.data.reserved`) applies at load to every role of every lite and litev2 loader
  of this round.
- **4 national-2 `train` chips (3 sites) lie at sites of the reserved new-site block.** Their role in the pin file
  is unchanged; no loader of this round returns them.
- The "unchanged" re-runs of configurations 1–6 and of both no-KD twins therefore **train on 4 fewer chips** than
  the as-built runs. This is said wherever those re-runs are called unchanged.
- **The as-built national-1 and national-2 models were trained on those 4 chips.** These are M7 labels (WQP samples
  matched to NAIP) at sites of M1's reserved block; no M1 sonde reading is involved. This is disclosed in every M7
  write-up.
- Chips at reserved sites in the other roles: **val 0, val_n 0, test 0, test_n 0, test_time 0, test_time_n 0,
  excluded 0.** P2 lock sites at reserved sites: 0 in each P2 role; P2 index rows: 0 in each P2 role (registry row
  `68dfce1e2a72`).
- 5 role-less chips at reserved sites that would have been added to national-2b were dropped before the build.
- No reserved site id is written to any M7 registry row, run record or count file.

## 4. P2 index rebuilt under the label rule (lead order item 3) — 2026-11-10

- Only the WQP label filter changed (harmonised QC-passed surface results, ruling F3). The lock file and every
  constant of Q6 are unchanged.
- Index rows 513 → 511; new sha256 `5c404be9251a22ae1b1d082446879e1466fa1c3fd41658f1f9649230be7f31b2` (registry
  row `3ea898947cbd`). The first-build hash is kept in entry 1 for the record.
- `test_p2`: 2 chips lost their label (found by key and present/absent flag); usable 189 → 187. `train_p2` and
  `val_p2`: 0 values and 0 label sources changed between the builds.
- The chip store still holds the 513 chips of the first build; the 2 chips without an index row are not loaded.

## 5. How ruling F7 is implemented — 2026-11-10

- "Chip or site": a `train_p2` water body is removed whole if it has an index row, **or a `train_p2` lock site
  with or without an index row**, in a 25-km block whose Lite block role is val or test (the blocks of the `val`,
  `val_n`, `test` and `test_n` chips). Counting lock sites is the stricter reading: 8 water bodies have an index row
  in such a block, 35 lock water bodies have a site in one, and 12 water bodies of the index are removed (41 rows,
  38 usable).
- Block roles are those of every Lite split so far (4-state, national-1, national-2, and the 44 blocks first seen
  in national-2b, drawn there by the national-1 salted rule). A `train_p2` row in a block with no Lite role at
  all (111 rows, 96 usable, in 42 blocks) gets the same salted rule (national parameters and seed, weights = index
  rows per block; drawn 28 train, 8 val, 6 test) and is removed unless its block draws `train` (38 further rows, 30
  usable).
- The ruling's counts are confirmed by the recount: 25 usable rows in eval-role blocks (13 val, 12 test) and 96
  usable rows in blocks without a Lite chip.
- Result: 79 of 145 rows removed (68 of 128 usable); **60 usable `train_p2` rows** in 25 water bodies and 32
  sites. Registry row `e89036650629`.
- The same reduced row set serves every arm and comparator; `p2_index.load_index` applies it for every caller.

## 6. Label-build choices not spelled out in section 3 of the amendment — 2026-11-10

All fixed in code before any number on the version; none was chosen by looking at a value.

1. A censored row is used for a chip only if the chip has no uncensored value of that target.
2. Censored FNU and NTRU rows are not recovered (5 rows; section (ii) of the addendum).
3. A right-censored chip with more than one distinct limit is not used as a censored row (ruling on missed point
   4); this drops 19 Secchi chips.
4. Recovered labels are built on `train`, `val` and `val_n` only; NTRU and every censored row on `train` only;
   nothing recovered is built, loaded or reported on a test role or on `excluded` (Q2).
5. No original-target label (Secchi, NTU, TSS, chl-a) is built on an added chip; added chips carry recovered
   labels only.
6. Added chips get roles by the national-1 rules unchanged; none enters `val`, `val_n` or a test role.
7. The 4-state chips are re-derived by key (organisation-qualified site, date, target, characteristic) through the
   rule reader; M2's side table is not used for them (order 2(b)).
8. The P2 days-between feature of P2a/P2b is kept as in the published recipe and marked in tables as a label-side
   input that the candidate does not have (ruling on missed point 5).

## 7. Overlaps of the P2 lock with other workstreams' roles (ruling F10; counts of the verified review) — 2026-11-10

- M1 `sonde_split` test role: 0 exact id overlaps; 1 `CBP_WQX` `train_p2` id is the same station as an M1
  test-role site (excluded at load, entry 2).
- M1 train or val sondes in `test_p2`: 96 (88 + 8).
- M6 Track-N: test stations 0; val or excluded stations 19.
- Fuse `fuse_tier_c_split` site roles: `test_region` 13 in `train_p2`, 3 in `val_p2`, 16 in `test_p2`;
  `test_spatial` 1 in `train_p2`.
- `test_p2` site-dates join the campaign's reserved list: no workstream trains or tunes on them a model that is
  compared on, or fused for, P2. No Lite output is an input to an M1, M2, M6 or Fuse model scored on its own test
  roles.

## 8. Conflict C2: the teacher scored on M2 test-role lakes in the spent look — 2026-11-10

- The 4-state look of 2026-11-09 scored the Sentinel-2-input Lite-safe teacher on rows of M2 test-role lakes
  (registry rows `b41eab495b6c`, `bc70b284498d`, `b9c6bb9ffafe`, `d89d96326ac8`, `d544efe3f909`,
  `b9f03a45c672`; they stay, the registry is append-only). Disclosed in every M7 and M2 write-up.
- Keys-only list for M2: `$SENTINEL_RUNS/lite/counts_arch_v2/c2_teacher_scored_m2_test_lake_keys.csv`, sha256
  `4786aa121eb673ced502d487fb4af714fff37027677677048d2024c960cfc699`, 110 rows — a superset by keys, accepted by
  the lead (18:45 EST) and pinned in the M2 stamp (registry row `51a841e48437`, rulings R-1 and C-1). It is 0444
  and is never edited, regenerated or moved. No look prediction file was opened to write it.
- Rule in force until the lead says M2S/test is sealed: no workstream, the M7 national look included, scores a
  Sentinel-2-input model on a row of an M2 test-role water body dated ≤ 2023-12-31.
- `scripts/eval/national_lite_score_test.py` was edited in place (no hash pinned it: the national track is not
  frozen): its teacher step now drops such rows through `tests_mask.teacher_score_blocked` and hard-fails through
  `tests_mask.assert_teacher_scorable`. The block is wider than Test-S: buffer rows of the water body are included,
  there is no lower date bound, and a row without a date is blocked. `sentinel/models/litev2/tests_mask.py` was
  added to the national code list (`NATIONAL_CODE`) so the national freeze will hash it.
- Nothing further was scored. No v2 job runs a Sentinel-2-input model on any row (addendum, section on C2).

## 9. DenseNet121 weights (lead order item 7(ii)) — 2026-11-10

- One file downloaded on 2026-11-10 at 23:43 UTC, after its rows were added to `docs/licenses.md`:
  torchvision `DenseNet121_Weights.IMAGENET1K_V1`, into the new directory
  `$SENTINEL_DATA/raw/torchvision_densenet121_imagenet1k_v1/`, sha256
  `a639ec97d7c33b07ae66f0b5fb7d0192f95a3b11b7576c66c0126c2a727c4395` (registry row `e7ebd23c8ede`), 0444.
  Nothing was installed. The file has not been loaded; it will be opened only with `torch.load(weights_only=True)`.
- **Licence finding:** the source states no licence for the weight file, and the ImageNet terms page could not be
  reached; `docs/licenses.md` says so. The lead decides whether P3a/P3b results may be published with these weights.

## 10. A test of the frozen 4-state track fails because of another workstream's edit — 2026-11-10

- `tests/test_lite.py::test_scoring_hash_covers_scorer_import_closure` fails: the import closure of the frozen
  4-state scorer now reaches `sentinel/data/matchups.py` and `sentinel/models/orbit/wqp_surface.py`, which
  `FREEZE_lite.json` does not hash. The cause is an edit of `sentinel/models/orbit/data.py` by M2 on 2026-11-10
  (it now imports `wqp_surface`); no M7 file of the frozen list was edited.
- Effect: the scoring-code hash of the spent 4-state look can no longer be reproduced from the working tree. The
  look is spent and is not re-scored, so no result changes. Reported to the lead; not M7's file to change.

## 11. Open points sent to the lead with the stamp package (each needs a line in the rulings file) — 2026-11-10

1. Whether the Test-S KD mask (F4) also applies to the "unchanged" re-runs of configurations 1–6. Default of order
   2(c): unchanged — the re-runs use the KD target file as built, including the 308 masked-in-v2 pairings.
2. Whether slots 7–8 and their twin warm-start from the corrected-label re-runs (config 3 and its no-KD twin) or
   from the as-built checkpoints named in F4 / F5. M7's proposal: the corrected-label re-runs, seed-matched, since
   no v2 arm is fitted on as-built labels (F3).
3. The restated compute ceiling of 45 GPU-h (estimate 31; addendum section (vii)).
4. The N3 event counts on `val_n` are not recounted before the stamp (they need validation label values).
5. The licence finding of entry 9.
