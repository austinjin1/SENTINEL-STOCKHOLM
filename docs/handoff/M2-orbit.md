# Handoff — M2 SENTINEL-Orbit (updated 2026-11-11 13:12 EST)

## 2026-11-11 13:12 EST: dev_v5 built; v2 Test-S layer written (read first)

**v1 chain (dev_v5).** chain6b built dev_v5 on 2026-11-10 21:03 EST: `docs/preregistration/orbit_split_report_dev_v5.json`,
54,733 rows (251,984 with a chip: wqp 134,095, aquacorr 117,889), split file `orbit_split_v3.json`
sha256 `ffe288f2fafecada…`; protected cells are counted only (testxtest_lake chl-a 615 rows / 209 water bodies /
84 role blocks). `baselines_dev_v5` (fit scheme **base**) finished 07:27 EST (10.4 h, 52 fits); the ten SatWQ v5 seeds
finished 07:45–08:34 EST; four compare runs registered **1,762 rows under `orbit_val_dev_v5`** (base scheme only) and
the two power tables **72 rows each** under `orbit_power_dev_v5` (base) and `orbit_power_dev_v5_trackU` (track_u);
nothing failed. Val chl-a MdSA (valxval_lake): satwq_v5_noaug **48.56**, base 50.65, noaug+gbm 50.89, base+gbm 51.04,
MDN_B[val-sel] 52.49; seeds of the leader spread 48.14–53.27, so the §15.1 bar is not yet met and chain8's
seed-variance trigger fires. MDE80 vs MDN_B (base/track_u): chl-a 8.3/9.6, Secchi 5.2/5.9, TSS 10.6/13.7,
turbidity 33.7/37.3 — **turbidity stays descriptive (R-18, above the 15-point floor in both schemes)**.
Still running: `baselines_dev_v5_trackU` (PID 2619158, started 09:07 EST, fit scheme track_u); then the ten track-U
seeds, `chain6.done`, chain7 (selection → events → conformal → dev dry run → regenerated freeze candidate),
chain8, chain9. Only GPUs 0, 1 and 5 are free (another user holds 2, 3, 4, 6, 7, 8, 9): PAR stays ≤ 3 and
`nvidia-smi` is checked immediately before each launch.

**v1 selection row (lead, 13:10 EST).** No v1 selection row exists in the registry and **no member fit may start
until one does**. After `final_selection_dev_v5.json` is final (chain7, and chain8 if it changes the choice), write
exactly one row: workstream `M2-orbit` (never "M2 Orbit"), experiment `orbit_arch_v2/v1_selection`, carrying the §6
rule text, the candidate sha256 and `fit_scheme = "base"`; never superseded by an edit; its id goes to the lead
before the first comparator member fit, and `--fit-scheme base` is what the comparator script is passed (K-2).

**Loader leakage review (lead, 13:10 EST): `pass_with_lead_rulings`, no blocking finding.** No dev_v5 row is voided;
the six reviewed files were unchanged between the review's start and end hashes. Three findings await lead rulings
before the value-reading dev_v5m build: L-01 (no code gate ties `--mode build` to the review), L-03 (whether the
context-store pin's chip ids are a dev_v5m artefact), L-04 (the real-cache R-21 test). Review at
`…/scratchpad/review_m2v2_loader/review.{json,md}`. **The dev_v5m build stays held and the four loader files keep
their behaviour until the rulings file exists.**

**v2 code written since the stamp** (none of it has run on a real row; sha256, first 16):
- `sentinel/models/orbit_v2/testcells.py` `e0952d2c41fa3837` — amendment §1.6 keys and geometry: `scene_key`
  (same rule as the stamped `scripts/data/orbitv2_testS_keycounts.py`), `reference_scenes`, `scene_disjoint`
  (a row without a key is not disjoint), `primary_cell` (≥ 60 % of rows and ≥ 80 % of role blocks of the chl-a cell),
  `train_points`, `dist_train_km`.
- `scripts/eval/orbitv2_test_s.py` `5bf434a110fa012e` — the single M2S/test look: freeze and artefact checks
  (weights, comparator `SHA256SUMS`, K-2 fit-scheme equality of bundle and arm networks), claim → predict → seal →
  score → register, per-unit and per-network predictions (R-17), `--dry-run` on validation rows with a parity check
  against the stored `preds.npz` and `val_rows.parquet`, `--rescore-from-saved`, abort marker on any failure inside
  the claim. Still deferred inside it: the event track (R-13a) and the MDN_B interval comparison.
- `sentinel/models/orbit_v2/claims.py` `78d9387cc473dc7a` — a left-censored row (sealed `y` NaN, `y_dl` set) never
  enters an endpoint.
- Tests: `tests/test_orbit_v2_test_s.py` `3c314b39770be4c5` (10 tests: scene keys against the stamped script, the
  §1.6 rule, v1-ensemble and per-network parity, the dry run, pre-claim refusals, the real path on synthetic
  test-role rows in a temporary runs root, single-pass enforcement, the abort path, rescoring) — all pass;
  `tests/test_orbit_v2_claims.py` `6e7b666a76874814` (6) and `tests/test_orbit_v2_devm.py` `4aeb19f039115577` (37
  + 1 skipped) pass. The devm change is the L-04 fix: the real-cache R-21 test now re-derives the protected row and
  chip key sets with the builder's own functions, refuses empty sets and calls `audit_artefacts` on the written
  cache (keys and counts only; it stays skipped until the cache exists).
- Unchanged and already reported: `looks.py` `252af6b474432866`, `units.py` `bd1c9ba332b77f12`,
  `comparators.py` `5fc1ab46acc38c2c`, `scripts/train/orbitv2_power.py` `abb8a2b12b81e880` (registry row
  `6a9381a42a76`); `scripts/train/orbitv2_comparators.py` is now `450d722d27bd1728` (the bundle stores the kNN fit
  table).

**Held:** the dev_v5m build, every comparator fit, every A0/A1 network, any freeze, look or claim row, and anything
resting on readings (a)–(i), which the lead will answer in a lead file with a registry row.

## 2026-11-10 15:11 EST: architecture-round-v2 code before the stamp (read first)

**State.** dev_v5 is still not built (chain6b PID 2129931 waits for `wqp_hist.done`). The amendment is unchanged
(sha256 `5001373d…df8a`); no lead rulings on its reviews have arrived, so nothing from the reviews or from
`reviews/arch_v2_20261110/M2_verify.json` (verifier, 28 should-fix items, not a lead ruling) has been applied to
it. No stamp row. No validation number on any new arm. No network of any arm has been trained on real rows. No
new-target value column has been read on real data.

**Code written since 14:45 (all outside the v1 frozen globs; nothing in v1 edited).**

| File | What it does | Tests |
|---|---|---|
| `sentinel/models/orbit_v2/context.py` | Context-set rule of §3.4 item 3 (same water body, ±3..183 days, at most 6 dates, nearest first) from keys and image metadata | `tests/test_orbit_v2_context.py` (18) |
| `sentinel/models/orbit_v2/devm.py` | dev_v5m table and cache format, loader (`devm.load`), writer that refuses protected rows, context candidates (now with `platform`) | `tests/test_orbit_v2_devm.py` (28) |
| `scripts/data/orbitv2_build_dev_v5m.py` | Builder. `--mode keys` reads keys and image metadata only and prints counts; `--mode build` reads values and is not to be run yet. Two switches have no default and must be given: `--context-same-role`, `--join-same-waterbody` | keys mode run once on the dev_v4 base (below) |
| `sentinel/models/orbit_v2/arm.py` | Arms `a0`, `a1_c1`, `a1_c2`, `a1_c3`: configuration resolution, target matrix with censoring, head-drop rule, fit/inner cells, data sets, shared GLORIA-pretrained start, training of one network | `tests/test_orbit_v2_arm.py` (15) |
| `configs/orbit_v2/{a0,a1_c1,a1_c2,a1_c3}.yaml` | Amendment constants (refused if they differ from the code) and an empty `base` block | same |
| `scripts/train/orbitv2_arm.py` | One process = one network. A real run needs the stamp and a filled `base` block; `--smoke` needs `--data-dir` and `--smoke-base` and never reads the real dev_v5m cache before the stamp. Exit status 75 = time budget reached, rerun to continue | gate test in the same file |

All 90 v2 tests pass together (15:10 EST; arm 15, devm 28, context 18, targets 13, model 16). The tests use
synthetic rows, run on CPU and write only under pytest's temporary directory.

**Keys-only run of the builder on the dev_v4 base (design counts, not results; 15:07 EST).** Log and JSON:
`/tmp/claude-1016/-home-bcheng-STOCKHOLM/dbc009e4-477d-4528-910a-e89e964c5c0f/scratchpad/m2v2/devm_keys_dev_v4_smoke.{log,json}`.
Nothing was written to a cache; the shared lake cache was not written.
- 0 added rows. Expected: dev_v4 holds AquaCorr rows only, and the only WQP matchup set that exists today is
  2024 and later (test time block), so all 17,291 image-QC-passing candidates are protected and dropped.
- Join sites: 13,030 stations within 100 m of an AquaCorr row (13,001 development), 26,680 AquaCorr rows with a
  station, 8,506 development sites.
- Rows with a surface result key: TP 15,338, TN 6,607, DOC 2,166, turbidity-FNU 557, SSC 290, aCDOM 120.
- Context: 193,947 allow-listed chips of 289,931; 18,307 of 28,360 rows have at least one context date
  (2,604 have six); 19,611 distinct context chips. These counts change with dev_v5 and the 2015–2023 WQP chips.

**Choices made in code that the lead has not confirmed (a–i were reported at 14:43; j–r are new).**
(a) context image QC = `data.image_qc` with `same_acquisition`; (b) "Sen2Cor AOT" = `aot_c3`; (c) row date for the
context window = UTC sensing date of the row's chip; (d) an added row within 100 m and on the same date as a dev_v5
row is dropped; (e) one WQP site per row for the join; (f) v1 `turbidity_fnu` kept as `turbidity_fnu_v1`;
(g) AquaCorr DOC/aCDOM not used until OPEN-3 is ruled; (h) context stores `aquacorr` and `wqp` only; (i) the
value-reading dev_v5m build is held until dev_v5 exists and the lead rules on F-LOADER-V2;
(j) the "fewer than 200 fit values" rule is applied to all six new heads and counts quantified plus censored values;
(k) standardisation mean and SD use quantified values only;
(l) early stopping uses the inherited v1 `select_targets`, identical for A0 and A1;
(m) A1 always feeds day of year (latitude/longitude only in c1) and ignores an inherited `ctx_off`;
(n) A0 is fitted on the dev_v5 rows of dev_v5m only, A1 on every fit-cell row with a label of a kept head; both
stop on the same inner rows (dev_v5 rows);
(o) A0 and A1 networks with the same network seed start from the same cached GLORIA-pretrained v1 network;
(p) fp16 AMP for both arms (off on CPU);
(q) the per-network script computes no validation number; unit and arm scores belong to `orbitv2_units.py`;
(r) the `base` block of the arm files stays empty until the dev_v5 selection is registered, and a real run is
refused while it is empty.

**Still to write before the stamp (code and synthetic tests only).** `scripts/train/orbitv2_units.py` (unit and arm
mixtures, validation scores behind `require_stamp`, registry rows), parity comparators, `orbit_v2/looks.py` +
`scripts/eval/orbitv2_test_s.py`, the builder's real-run records (registry row, manifest, split report with
counts-only protected cells, test that no protected row or chip id is in a dev_v5m artefact),
`scripts/train/orbitv2_power.py` if GATE-2 is ruled.


## 2026-11-10 afternoon: WQP label rule, test-look fix, reviews (read first)

**State at 14:45 EST.** dev_v5 is not built. chain6b (PID 2129931) waits for `wqp_hist.done` (chip build PID
1419333, expected 20:30–21:00 EST), then builds `s2_wqp_matchups_2026` and dev_v5 under the label rule. The
amendment is frozen at sha256 `5001373d…df8a` and under independent verification: **do not edit it**. No validation
number exists on any new arm. Report sent to the lead at 14:43 EST.

**WQP label rule (lead rulings 2026-11-10: option A, and 13:21 EST sidecar exclusion).**
- Two defects in the shared loader (`sentinel/data/matchups.py`, not edited): depth was never read, and the
  sidecar files `_qc_rejects.parquet` / `_unmapped.parquet` were read as results.
- Rule: labels from the harmonised, QC-passed tables only; surface result = depth ≤ 1 m or unknown, never a result
  flagged "bottom"; Secchi exempt; a sample without such a result is dropped (`no_qc_passed_result` /
  `no_surface_result`). Code: `sentinel/models/orbit/wqp_surface.py`; `data.wqp_rows` calls `WS.apply`.
- Side table `processed/orbit_wqp_surface/surface_samples.parquet`, sha256 `8ed6a309…3a31` (builder
  `scripts/train/orbit_build_wqp_surface.py`). 12,289,657 samples; 620,469 + 278,039 dropped. It reproduces both
  built matchup tables exactly and covers 100 % of their keys.
- Records, in the ordered sequence: protocol change log (13:27; `orbit_protocol.md` sha256 `70225c35…c556`);
  registry `orbit_wqp_depth_rule` (summary row `242aa40e1913` + 48 count rows; cell recount rows `c6d15497e3ae`,
  `20f1a30d202a`, `56f7e3f762c5`, `96236565d7e0`); data pins `wqp_surface` and `wqp_discrete` in
  `testlook.DATA_PIN_SPECS`. `testlook.py` changed, so the v1 scoring-code hash changed: **the freeze candidate is
  stale until chain7 regenerates it.**
- Counts: `docs/preregistration/orbit_wqp_depth_rule_counts_20261110.json` (samples by table × target × role ×
  drop reason) and `docs/preregistration/orbit_wqp_rule_cellcounts_20261110.json` (protected cells; script
  `scripts/data/orbit_wqp_rule_cellcounts.py`). Headline cell `testxtest_lake`: chl-a 669 → 615 rows, 87 → 84
  blocks; TSS 221 → 166; turbidity 490 → 420; Secchi 1,442 unchanged. No pre-registered floor is crossed.
  `orbit_split_report_dev_v4.json` is superseded; the dev_v5 split report gives the binding counts.
- M7 uses the same rule through `wqp_surface.apply(samples)` (keys from `wqp_surface.sample_key`). Keep the public
  names of the module stable.

**Test-look tests.** A test had written `sentinel_runs/orbit/test_looks/M2_test/ABORTED.json` into the real runs
root on 2026-11-09 (`monkeypatch.undo()` restored the real `RUNS_ROOT`). `tests/test_orbit_testlook.py` now runs on
temporary roots with a guarded marker writer and asserts that the real `test_looks` tree is unchanged. **The lead
removes the stray file; do not touch it.**

**New v2 code.** `sentinel/models/orbit_v2/targets.py` + `tests/test_orbit_v2_targets.py` (13 tests): Track N /
auxiliary label rules of amendment §2.4. Values are read only for sites with a train or val role and dates
≤ 2023-12-31; protected rows are counted from key columns. Open for the lead: the AquaCorr 100 m join
(`same_waterbody` has no default) and the choice of the largest detection limit for a censored site-date.

**Reviews of the amendment** (`docs/preregistration/reviews/arch_v2_20261110/M2_review_{consistency,leakage,stats}.json`).
Read; unverified until the lead sends the verified rulings. The list below is a draft and is **not applied**.

| Group | Draft change (finding id) |
|---|---|
| Test-S integrity | Remove from Test-S the site-dates M7's sealed look already scored with an S2 model, by a keys-only list (F-M7-SCORED). State what the headline look and the teacher store hold of Test-S lakes (F-DISCLOSE, DISCLOSURES). |
| Scene rule | Define the reference set as all non-protected, non-buffer dev_v5m rows + context + SSL chips; the 58.1 % preview is below the 60 % threshold (F-SCENE-REF). |
| Loaders | Roles from keys before any value read; allow-list for the context cache; a test that no protected id is in a dev_v5m artifact; independent leakage review of `targets`/`context`/`devm` before the build (F-LOADER-V2, F-CTX-ROLE, F-SAME-LAKES). |
| Joins and fits | 100 m join only to a station of the same water body (F-JOIN-100M). Trophic-chain OLS on out-of-fold MDN_B predictions (F-OLS-INSAMPLE). |
| Stamp and freeze | Stamp row and `require_stamp` check the amendment and the rulings file (F-RULINGS-PIN); stamp row written by the lead only (STAMP-GATE). No A0/A1 run before the v1 dev_v5 selection is registered (INIT-A0). |
| Look code | `M2S/test` re-implements its look-specific parts in `orbit_v2` and never patches v1 at run time (TESTLOOK-IMPORT); a claim refuses a non-empty look directory (F-ABORT-MARKER; needs an edit of hashed `testlook.py`). |
| Power | New script `scripts/train/orbitv2_power.py`, Holm-adjusted α, MDE spec written out (POWER-SCRIPT, GATE-2, MDE-SPEC); kNN MDE80 of 33.7 makes the small-lake gate unreachable (POWER-KNN); H2-S power (H2S-POWER). |
| Comparators | Claim comparator = equal-weight average of MDN_B and GBM (CMP-1); GBM member with per-context-date features (CMP-4); event comparators under parity (CMP-5); literature check for TP/TN/DOC and no "state of the art" wording (CMP-3). |
| Statistics | +1 corrected bootstrap p and step-down Holm (STAT-1); TSS/NTU confirmatory only with ≥ 30 role blocks and MDE80 ≤ 15 (STAT-4); per-unit and per-network predictions sealed (SEED-1). |
| Configs and text | Complete config files in `configs/orbit_v2` with sha256 registered before the first validation number (OPEN-2); AquaCorr DOC/aCDOM unit check from metadata and fit rows only (OPEN-3); line 159 should point to §3.4 item 8 (TEXT); Track G draft conflicts (TRACKG-DRAFT). |

**Next.**
1. When chain6b has built dev_v5: check `protected_cell_counts`, the `wqp_depth_rule` record and key coverage in
   the split report; report hashes and counts to the lead. Restart chain6b by hand only if it died.
2. Pre-stamp code: see the 15:11 section above for what exists and what is still to write. Do not build
   dev_v5m before the lead's rulings on the loader findings.
3. After the stamp: A0 + comparator refits, then A1 c1–c3.


## 2026-11-10: architecture round v2 (read first)
Lead dispatch 2026-11-10: `docs/plans/architecture_research/LEAD_DECISIONS_20261110.md` (rules 1–6, M2 section) and
`docs/plans/architecture_research/M2_architecture_plan.md`. The decisions override the plan where they differ.

**State: amendment DRAFTED, waiting for the lead's stamp.** Until the stamp, only code, tests, data builds
(protected rows: counts only) and synthetic or train-only smoke tests are allowed. **No validation number on any
new arm or on any Track N target.** The headline v1 path (chain6b → chain7 → candidate → review → lead freeze →
`M2/test`) is not delayed by any of this.

| Item | Where | Status |
|---|---|---|
| Amendment (Test-S, Track N, arms A0/A1, seed unit + parity, selection, kill criterion, A2 gate, look `M2S/test`, dev_v5m) | `docs/preregistration/orbit_arch_v2_amendment.md`, sha256 `5001373d…` at report time | DRAFT; reported to the lead 12:35 EST with 12 points to confirm (its §8) |
| Track G amendments (plan §7.3 items 1–4) | `docs/preregistration/orbit_track_g_amendments_draft.md`, sha256 `13c6b5cb…` | DRAFT; not funded; no download |
| Test-S keys-only counts | `docs/preregistration/orbit_testS_keycounts_20261110.json`; script `scripts/data/orbitv2_testS_keycounts.py`; registry `orbit_archv2_keycounts` (67 rows) | done (AquaCorr exact; WQP 2015–2023 is an upper bound until dev_v5 exists) |
| Test-S reservation | `docs/plans/sota_campaign.md` ("Reserved unseen blocks added 2026-11-10") | lead-written |

**Test-S counts (keys only).** AquaCorr QC chips: 3,790 site-dates / 693 water bodies / 169 role blocks. WQP
2015–2023 upper bound (before scene match, chips, QC): 81,992 site-dates / 1,585 / 311. Scene-disjoint preview
(AquaCorr only): 69.5% of rows, 94.1% of blocks.

**Rules for v2 code (proposed in the amendment §5; confirm with the lead).**
- All new code goes **outside the v1 frozen globs** (`guard.FROZEN_GLOBS`): `sentinel/models/orbit_v2/`,
  `scripts/train/orbitv2_*.py`, `scripts/eval/orbitv2_*.py`, `scripts/data/orbitv2_*.py`, `configs/orbit_v2/`.
  It imports v1 primitives and never copies them. Do not edit a v1-hashed file before `M2/test` is sealed.
- The Test-S look `M2S/test` gets its own guard module, its own candidate and the lead-written
  `FREEZE_orbit_S.json`. Context chips, ring features and d_p of protected rows are computed only inside the claim.
- Scripts that need lake assignment for new points must not write the shared cache
  `$RUNS/orbit/cache/lake_assign_v3_*.parquet` while chain6b is building dev_v5 (see `lakes_readonly` in the
  counting script).

**Step-5 check (no training or tuning path reads Test-S).** Confirmed by reading every label-table and chip-store
reader in M2 code. Two notes sent to the lead: (1) the headline look holds all protected rows in memory inside the
claim (Test-S rows included) but predicts, writes and scores only the 2024–2026 cells; (2) the M7 teacher store has
242 imagery-only chips in Test-S lakes × archive years, tagged with their orbit role.

**Next.**
1. Pre-stamp: build `sentinel/models/orbit_v2/` (multi-target join, context-chip index, ring / d_p / glint features,
   SatWQ-LakeSet-MT, seed-unit trainer with EMA, parity comparators) with tests.
2. Build dev_v5m when chain6b has written dev_v5 (CPU ≤ 6 threads, one heavy job at a time).
3. After the stamp: A0 + comparator refits, then A1 c1–c3 (one process per network, ≤ 2 h, lock helper, GPUs 0–9).

## 2026-11-09 session summary (read first)
- **WQP 2015–2023 build had died** (09 Nov 10:37: a read timeout made it exit with code 3 so a restart loop would
  resume it, but it was launched without a loop). Relaunched by `logs/wqp_hist_loop.sh` (loop PID **1419331**,
  python 1419333; resume-safe; touches `wqp_hist.done`). At 12:54 it had 35,052 scenes left at about 1,100/h, so
  it should finish about **2026-11-10 21:00**. chain6 (PID 912737) is still waiting on it.
- **chain8** (`logs/chain8_seedvar_v5.sh`, PID **2135432**) waits for chain7 (protocol §6a round, conditional).
- **chain7** (`logs/chain7_freeze_v5.sh`, PID **2031839**, CPU only) waits for `chain6.done`. Then it:
  - selects the final configuration by the pre-registered rule and writes `final_selection_dev_v5.json`;
  - runs the dev_v5 event track (main + track U);
  - runs the val dry run of the single-look scorer;
  - overwrites `docs/preregistration/orbit_freeze_candidate.json` (dev_v5, with track U and data pins);
  - touches `chain7.done`.
  It never writes FREEZE_orbit.json.
- **Event track (B output) added**: protocol v0.4 §12, adapter `sentinel/events/orbit.py`, dev script
  `scripts/eval/events_orbit.py`, config `configs/events/orbit.yaml`. These files are outside the orbit globs and are
  added deliberately to the code hash (`guard.EVENT_CODE`). The adapter pins `sentinel/eval/events.py` v1.0
  (sha256 `4279d242…`). Labels are computed from linear values.
- **Single-look enforcement**: new `sentinel/models/orbit/testlook.py`, rewritten `scripts/eval/orbit_test.py`, and
  `guard.py` changes (protocol §9b):
  - only FREEZE_orbit.json, with status FROZEN and the lead's `authorised_by`/`authorised`/`candidate_sha256`, unlocks
    test;
  - registry refusal;
  - O_EXCL claim marker + `look_claimed` row;
  - protected rows only while a claim is open;
  - sealed predictions with SHA256SUMS;
  - SCORED marker;
  - content-hash `data_pins`;
  - `--rescore-from-saved`.
  Tests: `tests/test_orbit_testlook.py` (7) and `tests/test_orbit.py` pass.
- **Dry run on dev_v4**: `test_looks_dryrun/satwq_v4_noaug/M2_test`, registry `orbit_dryrun_dev_v4`. It reproduces
  orbit_compare (chl-a 48.26 vs 48.25) and the event metrics.
- **Interim candidate** `orbit_freeze_candidate.json` (dev_v4, no track U) was written for review of its structure.
  chain7 replaces it.
- **Lead step at freeze** (see `orbit_prepare_freeze.py` docstring): copy the candidate to FREEZE_orbit.json, set
  `"status": "FROZEN"`, and add `authorised_by`, `authorised` and `candidate_sha256`. Change no other key.

### Lead decisions 2026-11-09 (SOTA plan) and what was done
1. **Small lakes (< 1 km²) are a headline stratum.**
   - Pre-registered in protocol §11.4 before any val result by size existed. Only counts were computed: 34% of val
     chl-a rows, in 47 of 89 blocks.
   - Secondary family of 4 targets × 3 H1 comparators, Holm per cell (`holm_p_smalllake`).
   - Projected MDE is about 14–19 points, so it is **pre-declared descriptive**.
   - Implemented in `orbit_test.py` (`lake_area_km2` from HydroLAKES, strata `smalllake_lt1km2` / `lake_ge1km2`).
2. **Track G (international) has a draft pre-registration**, `docs/preregistration/orbit_track_g.md` (§11.5).
   - Sources G1–G5 (EEA Waterbase, SYKE, SLU MVM, Ontario, ECCC); LAGOS-US is used for de-duplication only.
   - Same matchup rule and frozen split v3.
   - Unseen block: non-US test-role lakes × 2024–25. Own freeze and look `M2G/test`.
   - Cap: 1 configuration and 8 GB.
   - Licence entries are PLANNED in `docs/licenses.md`. **Nothing has been downloaded.**
   - **Next:** verify the licences, freeze the file (sha256 into §11.5), then download. ETA 6–9 days.
3. **SSL (lever 4)**: protocol §11.6; `sentinel/models/orbit/ssl.py`; `ssl:` block in `configs/orbit/satwq.yaml`
   (off by default); hook in `orbit_satwq.py`.
   - The pixel set covers train-role × ≤ 2021 chips, excluding val/test/inner/buffer units by coordinates. The
     `v1_trackU` variant also excludes the held-out regions.
   - The exclusion manifest is `manifests/orbit_ssl_pixels_<v>.json` plus the chip lists.
   - Smoke test on 3 aquacorr shards: 38.5k pixels; the encoder updated and the head did not. The full aquacorr store
     has 34,428 eligible chips (excluded: val role 29.7k, val years 20.6k, test role 20.3k, inner 8.3k, buffer 2.1k,
     test years 1.5k).
   - **chain9** (`logs/chain9_ssl_v5.sh`, PID **2135433**):
     - after chain7 it builds the pixels (one CPU job at a time);
     - after chain8 it runs ssl1, ssl2 and ssl3 × 5 seeds (GPU lock helper), comparing each plain and +gbm
       (6 configurations, counted);
     - it then re-applies the §6 rule;
     - if an SSL configuration wins **and** it is within 4 days of the dev_v5 candidate, it trains the track-U refits
       and redoes the events, dry run and candidate. Otherwise SSL goes to Test-C.
4. **Landsat is deferred.**
7. **GPU rule update (2026-11-10; GPUs 0–9 open).**
   - `PL.setup_gpu` uses the helper's cap file (0–9) and lock files. It never takes a GPU on which another Unix
     user's process runs (`PL.foreign_gpus()`, fail-closed if nvidia-smi fails) and needs ≥ 8000 MiB free. It picks
     the most-free GPU first. There is no hard-coded GPU list.
   - Seeds now run in parallel, at most 3 at a time (each one GPU, ≤ 2 h, torch 2 threads, so about 6 CPU threads),
     through `logs/par.sh`. CPU-heavy steps (builds, baselines, compare) stay sequential.
   - chain6 (912737) was replaced **before any of its steps ran** by `logs/chain6b_dev_v5.sh` (PID **2129931**):
     same commands, parallel seeds.
   - chains 8 and 9 were relaunched with parallel seed loops (PIDs 2135432 / 2135433). chain7 is unchanged
     (2031839).
   - The disk guard was restarted (PID 2129935) and now watches 1419331 and 2129931.
6. **Split-conformal intervals (lead, 2026-11-10; protocol §5b, secondary).**
   - CQR offsets per (model, target, level), water-body-weighted, fitted on the val selection cells.
   - Val metrics cross-fitted by water body; the offsets are frozen in `conformal_orbit_<version>.json` (sha-pinned;
     track U has its own).
   - At test, raw and conformal coverage/width are reported side by side. Points and selection are unchanged; it is
     logged once in configs_tried.
   - Code: `sentinel/models/orbit/conformal.py`, `scripts/eval/orbit_conformal.py`, `orbit_test.py`
     (`--conformal-artifact` / FREEZE `conformal_artifact`), `orbit_prepare_freeze.py`.
   - chains 7/8/9 were relaunched with the conformal step (chain7 PID 2031839; chains 8/9 relaunched again, see item 7).
5. **Numeric values + intervals (user requirement, 2026-11-10).** `orbit_test.py` now seals the mixture quantiles
   (2.5/16/50/84/97.5%) of SatWQ-Net and MDN_B. In every cell, stratum and target it scores 68%/95% empirical coverage
   (role-block CI) and mean log10 width next to MdSA, SSPB, slope and RMSLE, and registers them (protocol §5).
   - Verified with a dev_v4 dry run: coverage equals orbit_compare's registered `orbit_val_dev_v4` values (chl-a
     SatWQ 0.722/0.960, MDN_B 0.690/0.955; Secchi 0.676/0.929).
   - The scratch run was deleted unregistered. Its small-lake metrics were never printed or read; only row counts
     were checked.

## SOTA plan (docs/plans/sota_campaign.md, 2026-11-09; for lead review; current candidate path unchanged)
**Diagnosis: why MDN_B is close** (dev_v4 val, chl-a, valxval_lake).
- **Same information.** Both models see the same L2A reflectance. MDN_B on the 12-band water median already captures
  most of the signal.
  - SatWQ-Net's extra inputs add little: context-off 50.0 vs 48.3 MdSA. Pixel sets and wavelength tokens give about
    4 points.
  - RGB+NIR-only scores 142, so the red edge carries the signal, and MDN_B sees it too.
- **Data-limited and noisy.**
  - Fit cell: 11.7k rows, about 840 lakes, mostly AquaCorr, with units unstated and mixed methods.
  - Seed spread is 49–56.5, about as large as the gain.
  - The atmospheric-correction error floor of Sen2Cor over water is shared by every model.
- **Source shift.** Dev is AquaCorr-heavy, while test is WQP-only (review W4).
- **Small cells.** Val has 1,157 rows in 89 blocks, so CIs are about ±10 points. The test MDE is 10–14 points, so H1 is
  under-powered even if the gain is real.
- **Conclusion (rule 5).** More capacity will not help. The levers are more same-source data (both sides get it), an
  information source MDN_B cannot use (unlabelled imagery), and pre-registered harder strata where the pixel-set
  inductive bias should matter.

**Levers, ranked by expected gain per GPU-hour.**

| # | Lever | GPU-h | Unseen test block | Cap | ETA |
|---|---|---|---|---|---|
| 1 | **dev_v5 data** (WQP 2015–2023, same source as test; running). All baselines are refitted. | ~6 (chain6) | headline (test role × 2024–25) + Test-B 2026 + pooled | 4 val configs (noaug/base × plain/+gbm) | candidate ~12–13 Nov |
| 2 | **Seed-variance round** (§6a, declared) | ~5 | same as 1 | 4 val configs (R1/R2 × plain/+gbm) | +1–2 d, only if triggered |
| 3 | **International matchups** (track G): EEA Waterbase lakes, Canadian federal/provincial lake monitoring, SYKE (Finland) and SLU Miljödata (Sweden), each where the licence permits. LAGOS-US is mostly WQP-derived, so it is used only for de-dup checks. All rows go through the same frozen split rule. | 0 for data, ~6 to refit | **new geography**: track-G test-role lakes × 2024–25 (never seen, new source) | refit the selected configuration + comparators only (1 config) | 6–9 d (sources, licences, S2 chips 1–3 d each) |
| 4 | **Self-supervised pretraining on unlabelled S2 water pixels** (masked band modelling). Uses train-role lakes × ≤ 2021 chips only (no test-role/test-year imagery, even unlabelled). It is the one information source MDN_B lacks. | ~15 (3 pretrain jobs ≤ 2 h, chained + 5-seed fine-tunes) | if adopted before the first freeze: the headline. Otherwise **Test-C** = WQP Oct 2026–Sep 2027 (prospective; scoreable mid-2027 after reporting lag) | 3 pretraining configs × 5 seeds | 4–5 d after dev_v5 |
| 5 | **Track S, small lakes** (HydroLAKES area < 1 km², a-priori, imagery/geometry only). Justification: mixed pixels and adjacency are worst there, and pixel sets + shore distance are SatWQ's inductive bias; the user cares about local ponds and reservoirs. | 0 (no selection on it) | Test-C (prospective). As headline strata only if the lead adds it before the freeze. | 0 configs; power check from protected counts first | 1 d to pre-register + power |
| 6 | Track U (frozen), Track E bloom strata (frozen), event track E4a (§12) | in chain6/7 | headline / U-test | as frozen | with the candidate |
| 7 | Track L, Landsat 8/9 cross-sensor (proposed in §11.3) | ~5 | new sensor × 2024–25 | 1 config | 5–7 d, lowest priority |

- **Rules applied**:
  - every track is pre-registered with its justification before any val result on it;
  - the original headline is always reported;
  - no spent block is reused (none is spent yet);
  - all configurations are counted.
- **Requested from the lead**: (a) approve levers 3–5; (b) decide whether track S becomes headline strata before the
  freeze or waits for Test-C.

### Lead decision 2026-11-09 (plan approved)
- Proceed dev_v5 → chain7 → full candidate (~12 Nov).
- If the per-seed rule fails against MDN_B on dev_v5, run ONE seed-variance round. It is pre-declared in protocol
  §6a with a cap of R1 dropout 0.2 and R2 dropout 0.2 + finetune lr 2.5e-4, 5 seeds each, plain and +gbm.
  - Automated as `logs/chain8_seedvar_v5.sh`, which waits for chain7 and checks the trigger itself.
  - If an R configuration wins, chain8 trains its track-U refits and redoes the events, dry run and candidate.
- Then request the independent pre-test review of testlook.py, orbit_test.py, guard.py and the event track.
- No FREEZE, no test.
- The §6a edit changed the code hash. The interim dev_v4 candidate is therefore stale, which is expected; chain7/8
  regenerate it.

### Disk (sota_campaign "Disk rule", checked 2026-11-09 17:45)
- Free space was 100 GB, and 108 GB after the clean-up.
- **Projected M2 growth until the candidate: about 6 GB or less**, so free space stays near 100 GB, far above 60 GB.

  | Item | Estimate |
  |---|---|
  | WQP chips (38 KB/chip; 101.5k chips now; hist cap 150k, about 86k left) | ≤ 3.3 GB |
  | 2026 build (≤ 30k chips) | ≤ 1.2 GB |
  | dev_v5 cache | ~0.1 GB |
  | baselines_dev_v5 + track U fitted pkl | ~0.8 GB |
  | 30 SatWQ runs | ~0.4 GB |

- `logs/disk_guard.sh` (PID 1588508) checks every 10 min. Below 62 GB free it pauses (SIGSTOP) the hist loop and
  chain6 with their children, writes `logs/disk_guard.TRIPPED`, and waits for the lead.
- Deleted: caches dev_v1–v3, lake_assign v1/v2, `smoke/`, and the two scratch dry runs (about 150 MB). The dev_v4
  fitted baselines (~0.4 GB) are kept until the dev_v5 candidate exists.

### M7-review items adopted (2026-11-09, protocol §9b item 8)
- **Scoring-code hash** now covers the full import closure of events.py (`hazards`, `ew_scoring`, `early_warning`,
  `bootstrap`, `metrics`, package `__init__`s). `test_event_import_closure_hashed` runs the event path in a fresh
  interpreter and checks the loaded modules against the hash.
- **testlook.py / orbit_test.py**:
  - a `predictions_sealed` registry row;
  - test access closes at sealing;
  - `--rescore-from-saved` is refused once scoring rows exist;
  - all artifacts are verified before the claim;
  - an event error is recorded as `event_track_run = 0` ("not run") and never burns the look;
  - the mdn_official weights are content-pinned;
  - (sota_campaign rule 7) the claim is bound to pid/host, protected access needs the claiming process, and any
    BaseException (incl. SIGTERM) between claim and seal writes ABORTED.json plus a `look_aborted` registry row.
- Tests: `tests/test_orbit_testlook.py` (11 tests) pass.
- The dev_v4 dry run was repeated twice with the patched/restructured scorer (not registered; scratch dirs deleted).
  The paired and event tables were identical to the registered `orbit_dryrun_dev_v4` run.

### Event-track validation (dev_v4, `orbit_events_val_dev_v4`, development only)
E4a on valxval_lake: 309 positives, 90 lakes. Water-body cross-fitted calibration.

| Score | AUPRC | BSS vs climatology |
|---|---|---|
| derived | 0.857 [0.806, 0.916] | 0.604 [0.547, 0.689] |
| direct | 0.848 | 0.590 |
| MDN_B P | 0.837 | 0.566 |
| GBM-direct | 0.797 | 0.496 |
| S2 CI analogue | 0.648 | 0.309 |
| climatology | 0.269 | — |

- **ΔBrier, derived − comparator**: CI < 0 against every comparator, including direct (−0.0030 [−0.0047, −0.0011])
  and MDN_B (−0.0081 [−0.0151, −0.0023]).
- **ΔAUPRC vs MDN_B**: +0.021 [−0.008, +0.059]. The CI includes 0.
- **Tiers**:
  - at 12 µg/L, results are similar;
  - at ≥ 50 µg/L (75 positives), derived ties direct and GBM-direct.
- **E3**: 0 val positives, so descriptive only.

### Open risk for H1 (dev_v4 val)
The ensemble beats MDN_B by 4.4 MdSA points [−13.1, +0.2]. Individual seeds fail §15.1(iii) against MDN_B (52.6):

| Seed | MdSA |
|---|---|
| s0 | 53.6 |
| s1 | 56.5 |
| s2 | 49.2 |

dev_v5 (5 seeds, WQP 2015–23 rows) is the next check.


## Reconciliation after the 2026-11-04/06 reboots
- **National WQP S2 rebuild for 2024–2025** finished on 2026-10-25 at 11:11 (manifest `s2_wqp_matchups`).
  - Coverage: 68,445 candidates and 37,231 chips.
  - Same-day QC-pass samples: chl-a 14,195; Secchi 14,326; turbidity 9,348; TSS 5,499.
- **chain4** finished: `chain4.done`, registry experiment `orbit_val_dev_v3_fit-recent`.
- Nothing else was missing.
- The **AquaCorr chip extraction** finished at 120,000 requests, with 117,889 chips ok (manifest
  `s2_chips_aquacorr`, 2026-10-25 13:39).

## Running jobs (started 2026-11-08)
| PID | What | Output / flag |
|---|---|---|
| 637422 (DIED 11-09 10:37; relaunched as loop 1419331) | National WQP × S2 build for **2015–2023**: `build_s2_wqp_matchups.py --start 2015-06-23 --end 2023-12-31 --window 0 --max-chips 150000 --read-workers 4 --scene-workers 2 --out-name s2_wqp_matchups_hist` (niced) | Log `$SENTINEL_RUNS/orbit/logs/wqp_hist_2015_2023.log`; tables `processed/s2_wqp_matchups_hist/`, chips appended to `processed/s2_chips/wqp` (≤ ~5 GB); flag `logs/wqp_hist.done`. Estimate 1.5–2 days (3,114 planning dates; the 2024–25 build took 10.7 h for 2 years) |
| 638293 (finished 14:43) | **chain5** (`logs/chain5_dev_v4.sh`, GPU 1): runs `baselines_dev_v4`, then `satwq_v4_noaug_s0..2`, then compare plain and +gbm (registered, `orbit_val_dev_v4`), then the power analysis | `logs/chain5.done` |
| 912737 | **chain6** (`logs/chain6_dev_v5.sh`, GPU 1). It waits for `wqp_hist.done` and `chain5.done`, then runs: the WQP 2026-01..09 Test-B matchup build (`--out-name s2_wqp_matchups_2026`, flag `wqp_2026.done`); build dev_v5; track counts; `baselines_dev_v5`; `satwq_v5_{noaug,base}_s0..4`; compare plain and +gbm for both; power (headline and track U); track-U baselines and SatWQ refits | `logs/chain6.done` |

If the hist build dies, chain6 waits forever. In that case kill PID 912737, fix the build and relaunch both.

## dev_v4 validation results (development, not claims; registry `orbit_val_dev_v4`, `orbit_power_dev_v4`)
- `satwq_v4_noaug` (3 seeds), chl-a on valxval_lake (1,157 rows, 89 blocks). Differences are MdSA points;
  negative means SatWQ-Net is better; CIs are role-block bootstrap 95%.

  | Comparator | Difference | 95% CI |
  |---|---:|---|
  | MDN_B (val-selected) | −4.4 | [−13.2, 0.2] |
  | GBM imagery + context (val-selected) | −8.0 | [−17.3, 0.7] |
  | location kNN | −78 | CI excludes 0 |

- SatWQ-Net MdSA is 48.3%, vs 52.6% for MDN_B and 56.3% for GBM.
- On valxany_lake (3,546 rows, 170 blocks), both CIs exclude 0: −7.5 vs MDN_B and −9.6 vs GBM.
- Secchi on valxval_lake: −4.7 [−8.6, −0.7] vs MDN_B and −5.1 [−12.7, 0.9] vs GBM.
- The +gbm blend is weaker on dev_v4: −1.2 vs MDN_B and −4.8 vs GBM.
- **Power (W4).** The headline test cell holds 669 chl-a rows in 87 blocks. The expected test SE is about
  3.5 for MDN_B and 5.0 for GBM, so the MDE at 80% power is about 10–14 MdSA points.
  - Power at the val gains: 0.22 vs MDN_B and 0.35 vs GBM for chl-a. H1 is **under-powered**.
  - Secchi vs MDN_B: power 0.91.
  - Track U: chl-a MDE about 12–15 points.

## Changes this session (all before any test scoring)
- `build_s2_wqp_matchups.py`: new `--out-name` option. The 2015–2023 tables go next to the 2024–25 table,
  so the Fuse consumers of `s2_wqp_matchups` are not disturbed.
- `data.py`:
  - `WQP_MATCHUP_SETS` reads both WQP sets.
  - `protected_cell_counts` writes counts only for protected cells (split report).
  - **Turbidity unit rule** (`apply_turbidity_unit_policy`): the target = NTU family + AquaCorr
    unspecified; FNU and mixed-unit values go to `turbidity_fnu`, which is never fitted; `turbidity_all_units`
    keeps the pooled value. A test was added.
- `orbit_compare.py` writes `val_row_preds.parquet`.
- New `scripts/train/orbit_power.py` (power analysis, W4).
- Harder tracks (RESEARCH_PLAN §15.5 amendment of 2026-11-08):
  - `sentinel/models/orbit/track_u.py`, the `track_u` fit scheme and `scripts/train/orbit_tracks.py`;
  - frozen rule file `docs/preregistration/orbit_track_u.json`, sha256
    `953948b7f0251057728adfc4d9a75065e1d0324c9ec3536eb6dae99ae957b2cf`;
  - counts in `orbit_tracks_counts_dev_v4.json`.
- `orbit_test.py`:
  - `--version` for the dry run;
  - blend support (FREEZE `blend_with`);
  - track-E strata;
  - `--track U`;
  - turbidity strata by unit class.
- New `scripts/eval/orbit_prepare_freeze.py`. It writes `FREEZE_orbit_CANDIDATE.json`, never
  `FREEZE_orbit.json`.
- Protocol is now **v0.3** (§2.1 unit rule, §2.3 dev_v4/dev_v5, §5a power, §11 tracks E/U/L). Model card:
  turbidity unit note.
- `baselines.py`: the GBM thread count comes from `ORBIT_N_THREADS`.

## Next steps (after chain6)
1. Read the `compare_satwq_v5_*` paired tables. Select the final configuration by the pre-registered rule:
   lowest val MdSA (chl-a, valxval_lake) among the configurations scored on val (§6). Report the
   configurations-tried count from `configs_tried.jsonl`.
2. Run the dry run, `orbit_test.py --dry-run-dev --version dev_v5 --satwq … --baselines baselines_dev_v5
   --comparators-json … [--blend-with gbm_imagery_context]`, and check that it reproduces the compare numbers.
3. (Done 2026-11-09) Event track: the bloom cut, calibrators and FPR cuts are frozen in the events artifact.
4. (Automated in chain7) Write the candidate with `orbit_prepare_freeze.py`, including the track-U artefacts. Then request the
   lead's review and authorisation. **Do not score test.**
5. If a retrained teacher is wanted for M7, write a **new versioned manifest** (`orbit_teacher_v2.json`) and
   tell the lead. `orbit_teacher.json` stays unchanged.
6. Track L (Landsat 8/9) is proposed only: a chip extraction of about 1–3 GB.

## Lead decisions (2026-11-08)
- chl-a stays the primary target and Secchi stays secondary; the endpoint is not switched.
- The Test-B cell and the pooled analysis were declared before any test look (protocol §9a).
  `orbit_test.py` scores the headline, `testB_2026` and `pooled_2024_2026` cells.
- The dev_v3 `fitted_baselines.pkl` files were deleted with approval (about 800 MB freed); the predictions
  are kept.

- **GPU budget is now GPUs 0 and 1, shared by everyone** (2026-11-08 16:00). `PL.setup_gpu` now goes
  through `sentinel.core.gpu.acquire_gpu`: it prefers GPU 1, honours the cap file, and lets the helper set
  `CUDA_VISIBLE_DEVICES`.
  - Every chain6 GPU step is its own process and stays within 2 h: one seed is about 15–45 min. The baseline
    runs (MDN grid + GBM) now run on **CPU** (6 threads, about 2–3 h each), so no GPU is held for them. The lock is released between steps.
  - Compare, power and track counts run on CPU.
  - Before chain6 reaches its GPU steps (about 2 days from now), check whether test/freeze-critical jobs
    from other workstreams are waiting. If they are, pause chain6 by PID (`kill -STOP`, later `kill -CONT`).

## Open issues
- `fitted_baselines.pkl` is about 400 MB per baseline run (dev_v4, dev_v5, track U).
- `orbit_test.py` was rewritten with single-look enforcement (2026-11-09). An independent review is still recommended
  before the freeze.
- AquaCorr turbidity, TSS and Secchi units are unstated (documented).

## 2026-11-08 23:30 — lead notice: sonde snapshot promotion (EW-v2 final chain)
`processed/sonde_daily` and `processed/sonde_hourly` are symlinks that now point to `sonde_daily_refresh_20261108` / `sonde_hourly_refresh_20261108` (USGS IV refresh adding WY2024–25 approvals). Identity check: all 5,500 daily and 749 hourly sites identical through 2023-09-30, so train/val are unchanged (`compare_vs_sonde_{daily,hourly}.json` in the new dirs). New manifest shas: daily `dcd359ea…`, hourly `74bdc126…` (superseded `f7ee51ee…`, `0893f730…`). If your manifests/records pin the old sha, re-pin with a dated note; WY2024–25 rows are test-role for EW-v2 — do not use them for selection.
