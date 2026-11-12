# M2 SENTINEL-Orbit — architecture round v2: lead review, rulings and stamp

**Written by:** the lead (AI-assisted), 2026-11-10. The binding stamp time is the timestamp of the registry row
(`M2-orbit` / `orbit/prereg_snapshot`) that carries the sha256 of this file.
**Stamps:** `docs/preregistration/orbit_arch_v2_amendment.md`
- sha256 `5001373da42652926c6ea4a84028ac46628aa0e6d8b6e99380244417aeb8df8a`
- 516 lines, 36,647 bytes, last modified 2026-11-10 12:34:02 EST.

The amendment is stamped **as written, byte for byte**, together with the rulings and clarifications below. Where
this file and the amendment differ, **this file wins**. Both files are read-only (0444) from the stamp onward.
Line numbers (l.…) and section numbers (§…) refer to the stamped amendment unless a file is named. "Protocol"
means `docs/preregistration/orbit_protocol.md`.

**The stamp is partial.** It releases code, tests, keys-only builds, configuration files, the power script, and —
each under the conditions of §2.6 and §4 below — the value-reading dev_v5m build and the comparator fits. It does
**not** release: a value-reading dev_v5m build before the loader review of R-21 has passed; any A0 or A1 network
on real rows before the event of R-19; any validation number before `require_stamp` checks both hashes (R-23);
arms A2, A3 and A4 (§3.8, each needs its own stamped addendum); Track G (draft, not funded); any freeze or look.

## 1. What the lead checked before stamping

| Check | Result |
|---|---|
| File hash | equals the hash reviewed (`5001373d…df8a`), re-computed by the lead immediately before the stamp and asserted by the stamp script |
| Full re-read | all 516 lines re-read by the lead on 2026-11-10 before writing this file, with `sentinel/models/orbit_v2/stamp.py` and M2's list of code choices not yet confirmed by the lead in `docs/handoff/M2-orbit.md` (section of 2026-11-10 15:11 EST) |
| Independent review | three independent reviewers (consistency, leakage, statistics) and one verifier, all on this exact hash (workflow run `wf_349862cd-ad0`; the verifier re-checked the hash three times, last at 19:36 UTC). 39 findings, all upheld by the verifier, none blocking; recommendation `stamp_with_rulings`. Review files (sha256): `M2_review_consistency.json` `cf5e30c5…3812`, `M2_review_leakage.json` `4fceb4af…14d6`, `M2_review_stats.json` `c89f7535…e093`, `M2_verify.json` `dc768972…4c04e`, all in `docs/preregistration/reviews/arch_v2_20261110/`. Every finding is closed by a ruling in §2; the verifier's binding sentences are adopted word for word unless marked "(lead)" |
| A validation, test or protected number on a new arm | none. The registry holds no row of an experiment, model or split naming `orbit_v2`, `orbitv2`, `M2S`, `dev_v5m` or LakeSet; no dev_v5m artefact exists; no network of any new arm has been trained on real rows; no new-target value column has been read on real data (asserted by the stamp script where it can be; counts in the stamp row) |
| Frozen code | no file returned by `guard.frozen_files()` is edited by this stamp. No `FREEZE_orbit*.json` exists: the stamp precedes the v1 freeze, as the verifier required |
| A bar is lowered | no. Every ruling below adds a comparator, a condition or a restriction, takes the stricter reading, or corrects a statement of fact |
| Stamp guard | `sentinel/models/orbit_v2/stamp.py` (sha256 at the stamp in the stamp row) checks the amendment hash only. R-23 orders its extension before any validation number |
| Living files this file relies on (sha256 at the stamp, in the stamp row) | the protocol (v0.4 with the change-log entry of 2026-11-10 13:27 EST), `docs/plans/sota_campaign.md`, `docs/plans/architecture_research/LEAD_DECISIONS_20261110.md`, `docs/plans/architecture_research/M2_architecture_plan.md`, `docs/preregistration/orbit_freeze_candidate.json` (dev_v4; stale, to be regenerated), `docs/preregistration/orbit_testS_keycounts_20261110.json`, the two Track G files, `docs/handoff/M2-orbit.md`, the ten files of `sentinel/models/orbit_v2/`, the four `orbitv2_*` scripts, the four files of `configs/orbit_v2/` and the six `tests/test_orbit_v2_*.py` files |

**Pre-stamp incident, recorded.** On 2026-11-09 a unit test wrote
`sentinel_runs/orbit/test_looks/M2_test/ABORTED.json` into the real runs root (88 bytes, sha256
`efa31da5bcd12a6b555725551ec3fa80e59973db09ecacb785c9693593cc5a32`, content: aborted 2026-11-09T17:26:57,
reason `KeyboardInterrupt()`, pid 1575747). It was the only file in the folder; no `FREEZE_orbit*.json` existed;
the registry held no M2 claim, seal, abort or look row. The lead moved it (`mv -n`) at 2026-11-10T20:20:28Z to
`sentinel_runs/orbit/stray_markers_removed/M2_test_ABORTED_20261109T172657_test_artefact.json`. No look was
attempted, started or spent. `tests/test_orbit_testlook.py` now runs on temporary roots (14 passed). The look
folder `M2_test` is empty at the stamp.

## 2. Rulings (binding)

Each ruling has a number (R-…) and, in brackets, the identifier used in `M2_verify.json`.

### 2.1 Answers to the amendment's §8 points

| Point | Ruling |
|---|---|
| 1. Scene-disjointness reference set (§1.6) | **Changed**: the widest set, R-6 |
| 2. Binding MDE after the stamp (§1.7, §2.6) | **Accepted only with** R-15 and R-16 |
| 3. Context date limit, two-sided context (§3.4 item 3) | **Confirmed** (≤ 2023-12-31, two-sided), **with** R-7. Every A1 or v2 result, event results included, carries the sentence "retrospective reconstruction with two-sided image context; not a nowcast" |
| 4. Parity reading (§3.5) | **Confirmed** as written, with R-10 and R-14 |
| 5. Multi-target-by-stacking GBM member (§3.5) | **Confirmed**: out-of-fold features from the fit cell only, folds grouped by role block, every member counted; R-11 applies the same rule to the OLS |
| 6. Track N arm when v2 = A0 or A1 fails (§2.7) | **Confirmed** (the default). The A1 configuration scored on Track N is chosen by validation chl-a only, never by a Track N validation number; it is not a third arm in H1-S or H2-S; the per-seed-unit rule applies to it; if A1 fails its kill criterion no SENTINEL claim is made on Track N |
| 7. Shared initialisation "at the moment the first A0 network starts" (§3.1) | **Changed** to an event: R-19 |
| 8. aCDOM(440) head (§3.4 item 8) | **Accepted** (the default): fine-tuning auxiliary head only, GLORIA pretraining unchanged, dropped before any training if the fit cell holds fewer than 200 values, descriptive |
| 9. H2-S family with small TSS and NTU counts (§6.3) | **Changed**: R-18 |
| 10. Chl-a method-class covariate (§4 item 4) | **Confirmed**: not adopted. Adding it later needs its own stamped addendum before any validation number |
| 11. Code separation (§5) | **Confirmed with conditions**: R-21, R-22, R-24, R-17 (sealed per-unit and per-network predictions), R-16 (new power script), R-4 |
| 12. Early stopping on the base targets; a-priori ranges (§3.4 item 9, §2.1) | **Confirmed**, read as in C-3. The ranges of §2.1 are fixed now and never revised after any value is seen; out-of-range values are set missing and counted per target |

### 2.2 Corrections of statements in the amendment

- **R-1 [F-M7-SCORED].** §0 l.24 and l.34–36 are corrected: on 2026-11-09 M7's sealed lite/test look scored a
  Sentinel-2-input SatWQ-Net (the Lite-safe teacher, trained on the M2 fit cell only) against Secchi, TSS and
  turbidity labels on up to 106 site-dates of M2 test-role lakes dated ≤ 2023; every Test-S row that matches an
  M7 look row (same M2 water body, S2 sensing date within 1 day) leaves every Test-S cell by a keys-only list
  fixed and hashed before any dev_v5m build, with its count appended; no M2 v2 code or decision reads any M7 look
  file; conflict C2 applies to M2 as written for M7; and until M2S/test is sealed no workstream, the M7 national
  look included, scores a Sentinel-2-input model on a row of an M2 test-role water body dated ≤ 2023-12-31.
  The list is fixed by C-1.
- **R-2 [DISCLOSURES].** Recorded in place of "no scorer reads them": `orbit_test.py` holds every protected row,
  Test-S included, in memory inside the M2/test claim, and the M2 teacher store holds imagery-only v1 predictions
  on chips of Test-S lakes (242 by the author's keys-only count); M2/test and M2U/test write, log, seal and score
  nothing from a test-role row dated ≤ 2023-12-31, which the pre-test review confirms by reading the code; and no
  teacher prediction and no other workstream's chip of a test-role water body is joined to a label or read by any
  v2 arm or comparator outside the open M2S/test claim.
- **R-3 [missed 1; replaces §2.4 l.179–181, and governs §2.6 and §4 item 1].** New-target labels are read by
  M2-owned code in `sentinel/models/orbit_v2` from the harmonised, QC-passed tables `wqp_discrete/variable=*/`
  only (never a sidecar), with `depth_m` passed as the result depth, under the lead's rule of 2026-11-10 word for
  word (surface = `depth_m` ≤ 1.0 m or unknown; a result flagged bottom is dropped at any depth; replicate median
  per site, local date, target, characteristic and unit); the same reader serves every role; the shared module is
  not edited; and the counts by target and role (protected roles: counts only) are appended before any Track N
  validation number.
- **R-4 [FROZEN-SET; missed 3].** "File hashed by the v1 freeze" means every file returned by
  `guard.frozen_files()` (45 on 2026-11-10) as recorded in the lead-written `FREEZE_orbit.json`; the protocol this
  file amends is `orbit_protocol.md` as hashed there (v0.4 with the 2026-11-10 13:27 change-log entry); v2 work
  edits none of these files and overwrites no artefact pinned in the v1 candidate; an edit before the v1 freeze is
  v1 work made under a lead ruling and regenerates the candidate; and any edit after M2/test is sealed to a file
  that v1 inference imports needs a registered check, before the Test-S freeze, that v1's validation-cell
  predictions are reproduced exactly.
- **R-5 [missed 2; governs §3.2 l.271, §3.3 l.279–280, §3.6, §6.2].** If the v1 recorded in `FREEZE_orbit.json`
  is a SatWQ + GBM blend, the statement "A0 − v1 is the effect of the seed unit and EMA" is not used; A0 is then
  also compared, descriptively, with the unblended v1 seed ensemble; the v1 per-seed rule at every look applies to
  each frozen single network combined with the frozen blend exactly as `FREEZE_orbit.json` defines the reported
  model; and the v1 seed count is the one recorded there.

### 2.3 Test-S data, scene rule, context and joins

- **R-6 [F-SCENE-REF; point 1].** The §1.6 reference set is the scene keys of every chip read in fitting, early
  stopping, selection, calibration or pretraining of any arm or comparator: all non-protected, non-buffer dev_v5m
  rows of every role and year, their context chips, and the SSL manifest chips if that initialisation is
  inherited; the 60 % and 80 % thresholds are unchanged; every label says "scene-disjoint" or "scenes shared with
  training"; and when all rows are primary a §15.1 label also needs the scene-disjoint point estimate to favour
  the arm against every claim comparator.
- **R-7 [F-CTX-ROLE; point 3].** A context chip must have the same frozen-rule water body and the same split role
  as its row (non-lake rows: same 200-km block, not in-lake, not buffer-excluded); the context stores are
  `s2_chips/aquacorr` and `s2_chips/wqp` only, content-pinned at the first dev_v5m build; `naip_teacher`,
  `mc_benchmark` and any store or chip added after that pin are excluded. (This replaces "any existing chip
  store", l.298.)
- **R-8 [F-JOIN-100M].** The 100 m join of §2.4 is made only to a WQP station whose frozen-rule water body and
  split role equal the row's, decided from keys before any value is loaded; otherwise the row carries no value
  for that target.
- **R-9 [OPEN-3].** AquaCorr `doc_value` and `acdom440_value` are used only if AquaCorr's metadata state the unit
  explicitly (mg/L as C; 1/m at 440 nm); the check reads metadata and fit-cell rows only, its outcome is written
  to the dev_v5m split report before any Track N validation number, and if it fails the AquaCorr values are used
  neither for fitting nor for scoring.
- **R-9a [missed 4].** Every binding count and every count rule of the amendment is evaluated on dev_v5m built
  under the lead's WQP rule (R-3); no threshold, floor or gate is changed because of a recount. The WQP counts of
  §1.4 and the 200-block and 282-block rows of §1.5 are upper bounds from a superseded build and decide nothing.
- **R-9b [missed 5].** Test-S WQP membership is the set realised by the 2015–2023 build as launched on
  2026-11-09 (same arguments); no later build or top-up adds a chip of a test-role water body dated
  ≤ 2023-12-31.
- **R-9c [missed 7].** The dev_v5m split report also gives, with no role in the rule, the §1.6 counts with the
  scene key replaced by (sensing date, relative orbit), and the look reports that subset as an added sensitivity.

### 2.4 Comparators and parity

- **R-10 [CMP-1; replaces l.373 "Descriptive only, no claim role" for the MDN_B + GBM average; single-network
  results of the arms stay descriptive].** For every base and Track N target, the
  equal-weight average in log10 space of the validation-selected MDN_B family member and the validation-selected
  GBM family member is an additional claim comparator in the intersection-union tests of H1-S, H2-S, Track N and
  the small-lake family, for v1 and v2 alike, and in the per-seed-unit rule; its predictions are sealed with the
  other comparators; and no label is issued for an arm that does not beat it.
- **R-11 [CMP-2].** The trophic-chain OLS is fitted on out-of-fold predictions (the 5 role-block-grouped folds of
  §3.5) of the validation-selected chl-a and Secchi members; a second chain built on the GBM's out-of-fold
  predictions is also a member; and the member with the lower validation MdSA on the target's selection cell is
  comparator 5.
- **R-12 [CMP-3].** Before the Test-S freeze a dated literature check in `docs/novelty_checks` lists published
  Sentinel-2 or Landsat retrieval methods for lake TP, TN and DOC; a reproducible one not covered by §2.5 is
  added as a comparator by dated addendum before any Track N validation number of the arm; otherwise any Track N
  label reads "better than every retrained reference of arch-v2 §2.5; no published method was re-run" and the
  words "state of the art" are not used for TP, TN or DOC. (lead) M2 writes the check; the lead reads it before
  the Test-S freeze.
- **R-13 [CMP-4].** The GBM family gains a member with per-context-date features for the same K = 6 dates in the
  same order (12 band water medians, q_d and Δt per slot, missing slots NaN), fitted and selected under the same
  rules as the other members.
- **R-13a [CMP-5].** At M2S/test the event comparators of protocol §12.3 are refitted on dev_v5m from the arch-v2
  §3.5 families (the direct GBM classifier also receives the ring and context summaries); E-H1 and E-H2 are each
  tested with Holm over the 2 arms; and the protocol §12.4 power rule is applied to the primary cell fixed by
  §1.6.
- **R-14 [PARITY-LETTER; point 4].** §3.5 as written replaces the words "same 3-init averaging" of LEAD_DECISIONS:
  MDN_B has 10 initialisations with both read-outs as family members, each GBM member is a 3-seed bag and the
  selected one a 9-seed bag, each seed unit is compared with the full comparator, and no comparator ensemble may
  be reduced later.
- **R-14a [REF-ORDER].** The comparator families of §3.5 and §2.5 are fitted on dev_v5m and their validation rows
  registered before, or in the same batch as, the first validation number of the arm they face.
- **R-14b [missed 6].** For TP, TN and DOC each GBM and OLS reference is fitted both with the DL / √2
  substitution and on quantified rows only; both are family members and the one with the lower validation MdSA
  on the target's selection cell is used.

### 2.5 Gates, seeds, statistics and wording

- **R-15 [GATE-1; point 2].** The gates of §1.7 and §2.6 are read as written: the largest MDE80 over every scored
  arm and every claim comparator (location kNN and the MDN_B + GBM average included) decides, and no gate,
  threshold or comparator list is re-read once any dev_v5m validation prediction exists; the lead records that on
  the registered dev_v4 numbers the small-lake stratum is expected to be descriptive; and if it is confirmatory,
  Holm runs over both arms' hypotheses together and a label for a target needs every comparator hypothesis of
  that target rejected plus the per-seed-unit rule.
- **R-16 [GATE-2; point 2].** Each status-setting MDE is computed once, by a new script
  `scripts/train/orbitv2_power.py` outside the v1 globs that calls `orbit_power.block_draws` unchanged
  (n_sim 2,000, seeds 0 and 1), on the validation rows of the same target and stratum in the comparator selection
  cell, with the role-block count of the primary cell fixed by §1.6, as (z(1 − α/2) + 0.84) standard errors at
  the smallest Holm α of the hypothesis's family; the maximum over every scored arm and every claim comparator
  governs; the Holm divisors stay 6 for H2-S and 3 for Track N whatever is declared descriptive; the script's
  sha256 is registered before the first validation number of any new arm; and the result is a dated addendum
  written before the Test-S freeze candidate.
- **R-17 [SEED-1].** For v2 the per-seed rule means that each of the 3 pre-declared seed units (network seeds
  0–2, 100–102, 200–202), taken alone, has a lower MdSA than every claim comparator in its full form on the
  endpoint's primary cell, for every endpoint on which a label is sought; for v1 it means every frozen single
  network, with the seed count recorded in `FREEZE_orbit.json`; no network or unit is replaced or added (a failed
  run is re-run with the same seed and configuration only, and a diverged network is kept and reported); the
  Test-S module seals per-unit and per-network predictions before scoring; and every v2 label carries the
  sentence "the per-seed rule was applied to 3-network seed units; k of 9 single networks individually beat every
  comparator".
- **R-17a [STAT-1].** In every Test-S family the per-comparison p-value is the two-sided paired role-block
  bootstrap p-value of `evaluate.paired_mdsa` with the +1 correction (B = 2,000, seed 0); a hypothesis's p-value
  is the largest over all its claim comparators (all of §2.5 for Track N) and counts only if every point
  difference favours the arm; Holm is step-down with monotone adjusted p-values; and the per-seed-unit rule
  applies to H2-S, Track N and small lakes exactly as to H1-S.
- **R-18 [STAT-4; point 9].** Secchi, TSS and turbidity-NTU are each confirmatory in H2-S only if the target's
  primary Test-S cell holds at least 30 role blocks and its binding MDE80 is at most 15 MdSA points against every
  claim comparator; otherwise the target is pre-declared descriptive in the freeze candidate; the H2-S Holm
  divisor stays 6 either way; and the turbidity endpoint uses NTU-class rows only.
- **R-18a [KILL-REF].** The kill criterion compares A1's 3-unit ensemble with the reported v1 model recorded in
  `FREEZE_orbit.json` (its seed ensemble, with the GBM blend if frozen so) on common `valxval_lake` rows; if it is
  first computed against a v1 candidate and the frozen v1 differs, it is recomputed and the stricter outcome
  stands.
- **R-18b [WORD-1].** Every Test-S label is written in full as "unseen lakes, archive years 2015–2023" with
  "scene-disjoint" or "scenes shared with training" as fixed by §1.6; every report states the number of
  confirmatory families and hypotheses in the look; and an endpoint whose binding MDE80 exceeds the arm's
  validation gain against any claim comparator is called under-powered in the freeze candidate and in the report.

### 2.6 Order of work, configurations, code, guards and looks

- **R-19 [OPEN-1; point 7; replaces l.247–250].** No A0 or A1 network (synthetic or train-only smoke tests
  excepted) starts before the v1 dev_v5 selection under protocol §6, §6a and §11.6 is registered and written into
  a regenerated v1 freeze candidate that has passed its independent review; A0 and A1 take the architecture,
  training recipe and initialisation named in that candidate, recorded by its sha256 in every run configuration;
  a chain9 SSL initialisation is inherited only if it is in that candidate; and nothing later changes either arm.
- **R-20 [OPEN-2].** The complete configuration files of A0, c1, c2 and c3 (every width, depth, loss weight and
  the seed list) are written to `configs/orbit_v2` and their sha256 registered before any non-smoke training run
  of a new arm; any later change that alters a prediction is a new configuration, and because the caps (A0 1,
  A1 3) are filled, none is allowed; a failed run is re-run unchanged or reported as failed.
- **R-21 [F-LOADER-V2; point 11].** No dev_v5m build reads a new-target value column until the orbit_v2 targets,
  context and devm modules have passed an independent leakage review; the builder assigns split roles from keys
  first and loads value columns only for non-protected keys; the development context cache is built from an
  allow-list of train-role and val-role chip ids; a test asserts that no protected row or chip id is in any
  dev_v5m artefact; for M2S/test only look-agnostic helpers of v1 `testlook.py` are imported, while the look list,
  look folder, claim, open-claim check, freeze loader and protected-row loader are re-implemented in
  `sentinel/models/orbit_v2` with no run-time patching of a v1 module and with every v1 guard test mirrored; and
  the M2S pins cover every file the protected loader reads, including `wqp_discrete/variable=*`, the context chip
  stores and `processed/orbit_wqp_surface`. (This replaces l.446 "hardened primitives … are imported, never
  copied" for the look-specific parts.) (lead) The review is commissioned by the lead; M2 does not choose the
  reviewer. It covers `orbit_v2/targets.py`, `context.py`, `devm.py` and `scripts/data/orbitv2_build_dev_v5m.py`
  at the hashes M2 reports after the changes ordered in this file.
- **R-22 [F-SAME-LAKES].** M2S/test is run for v1 and v2 whatever the M2/test outcome; from the moment M2/test or
  M2U/test is claimed, no per-row prediction, label, per-lake or per-block result of those looks is used in any
  v2 design, funding, selection, calibration or addendum decision; v2 selection, the kill criterion and every
  status rule are applied by registered code with no discretion; an arm whose addendum is stamped after the
  M2/test claim cannot enter the v2 selection for M2S/test (Test-C only); and the tie order of any added arm is
  fixed in its addendum.
- **R-23 [F-RULINGS-PIN].** The `orbit/prereg_snapshot` row and the Test-S freeze candidate record the sha256 of
  both the amendment and the lead-rulings file (`extra.sha256` maps both paths), both files are set 0444, and
  `orbit_v2.stamp.require_stamp` checks both hashes before any validation number. (lead) M2 extends
  `stamp_status` / `require_stamp` so that a row counts only if it maps both paths to the files' current sha256,
  both files are read-only, and `extra.written_by == "lead"`; with unit tests on a scratch registry. M2 writes no
  row of experiment `orbit/prereg_snapshot`. Until that extension passes its tests no validation number is
  produced.
- **R-24 [STRAY-ABORT].** Before `FREEZE_orbit.json` is written the lead moves
  `sentinel_runs/orbit/test_looks/M2_test/ABORTED.json` (sha `efa31da5…5a32`) out of the look folder and records
  the removal with date and reason (done: §1); the pre-test review of each look (M2/test, M2U/test, M2S/test)
  confirms by listing that its look folder is empty or absent; the M2S/test claim refuses to start if its look
  folder holds any file; and no test may write under the real runs root. (lead) v1 `testlook.py` is not edited for
  this; the lead lists each look folder immediately before authorising a look.
- **R-25 [COMPUTE-DISK].** LEAD rule 5 binds M2 in full: before each dev_v5m, context-cache or comparator build
  `df -h /` must show at least 60 GB free after the projected size, otherwise stop and report; all derived dev_v5m
  artefacts together stay under 10 GB; M2 CPU jobs run one at a time with at most 6 threads; and no imagery is
  downloaded. (lead) GPU jobs may run in parallel on GPUs 0–9 through the lock helper (≤ 2 h per job, never a GPU
  that holds another user's process); all M2 jobs running at the same time use at most 12 CPU threads in total.

### 2.7 Track G draft (recorded; not stamped)

`orbit_track_g.md` and `orbit_track_g_amendments_draft.md` are drafts and Track G is not funded. The verifier's
answers PG1 (archive-year G1–G5 rows of test-role lakes are counted by key, never used for training or tuning in
either build order, and left unused once M2S/test is sealed), PG2 (c2 is the single Track-G configuration whenever
v2 is an A1 configuration) and PG3 (data-version name fixed in the funding ruling; surface rule replaced by the
project-wide WQP rule) are the lead's intended rulings. They bind only at Track G's own stamp. Until then no
Track G fit, build with values, or number is released, and PG1 already holds as a restriction on M2: no
archive-year row of a test-role lake is used for training or tuning under any track.

## 3. Lead clarifications (binding)

- **C-1 — the M7 list of R-1.** The keys-only list is
  `sentinel_runs/lite/counts_arch_v2/c2_teacher_scored_m2_test_lake_keys.csv`, sha256
  `4786aa121eb673ced502d487fb4af714fff37027677677048d2024c960cfc699` (110 data rows, key columns only: 106 in the
  Test-S window and 4 buffer-excluded). M2 reads this file and nothing else of M7. A Test-S row leaves every
  Test-S cell if, for any of the 110 list rows, (i) its M2 water body equals `m2_waterbody` and its S2 sensing
  date is within 1 day of `s2_sensing_date`, or (ii) its site id and sample date equal `site_id` and
  `sample_date`. The builder refuses to run if the file's sha256 differs. The number of rows removed is written to
  the dev_v5m split report (a count; no value).
- **C-2 — strict values in code.** The builder switches `--context-same-role` and `--join-same-waterbody` are
  removed; the strict values of R-7 and R-8 are constants with no other setting, each with a test.
- **C-3 — early stopping (§3.4 item 9; point 12).** A0 and A1 stop on the inner-cell criterion of the selected v1
  configuration exactly as the regenerated v1 freeze candidate records it (its selection targets, cell, patience),
  computed with EMA weights, identical for both arms and on the same inner rows (the dev_v5 rows). The words "on
  the four base targets only" are a restriction: no new-target value enters the criterion. They add no target to
  v1's criterion.
- **C-4 — head-drop rule (§3.4 item 8).** The rule "fewer than 200 values in the fit cell" applies to each of the
  six new heads (TP, TN, DOC, FNU, SSC, aCDOM) and counts quantified plus left-censored values, because both
  enter the likelihood. It is applied once, by code, before any training. A dropped head's target is reported as
  counts only for the arm, as l.322 says; if it is a Track N target no SENTINEL result is given for it and the
  Track N Holm divisor stays 3. Standardisation mean and SD come from quantified fit-cell values only.
- **C-5 — context inputs (§3.4 item 6).** A1 always receives day of year; latitude and longitude only in c1; an
  inherited `ctx_off` is ignored by A1. A0 keeps the context setting of the selected v1 configuration. If that
  setting is `ctx_off`, every sentence that attributes A1 − A0 also names day of year.
- **C-6 — rows per arm.** A0 is fitted on the dev_v5 rows of dev_v5m; A1 on every fit-cell row that has a label
  of a kept head. Every sentence that attributes A1 − A0 (l.279–280) therefore names four things: context set,
  land ring, multi-target heads **and the added rows**. c3 − A0 carries the added rows without the context set.
  Networks of A0 and A1 with the same network seed start from the same initialisation fixed by R-19.
- **C-7 — AMP.** fp16 AMP for both arms on GPU. If the selected v1 configuration was trained without AMP, every
  sentence that attributes A0 − v1 also names AMP.
- **C-8 — label-blind definitions accepted as coded** (each is recorded in the dev_v5m data information):
  context image QC = `data.image_qc` with `same_acquisition`; "Sen2Cor AOT" = `aot_c3`; the row date for the
  context window is the UTC sensing date of the row's chip; an added row within 100 m of, and on the same date
  as, a dev_v5 row is dropped; the v1 column `turbidity_fnu` is carried as `turbidity_fnu_v1` and is not the FNU
  auxiliary target; the per-network script computes no validation number.
- **C-9 — join site and censored site-dates.** The join of R-8 uses one WQP station per row: the nearest
  qualifying station, ties by the lower station id, chosen from keys. For a site-date and target whose results
  are all left-censored, the label is "censored at the largest detection limit of that site-date" if that limit
  lies inside the a-priori range of §2.1; otherwise the site-date carries no value for that target. Both are
  counted per target in the split report.
- **C-10 — comparators come first.** M2's first value-reading jobs after dev_v5m exists are the comparator
  families (R-14a). `base` blocks of the arm files stay empty, and a real arm run is refused, until R-19 is met.
- **C-11 — addenda.** Every addendum this file orders (power result, literature check, counts, the M7 removal
  count) is a new dated file; nothing is appended to the stamped amendment or to this file. An addendum that
  changes a rule needs its own review and stamp; one that records a count or the outcome of a rule fixed here
  does not, and is registered with its sha256.

## 4. What M2 may do from the stamp onward

- **Now:** code and synthetic tests (units and arm mixtures, parity comparators, `orbit_v2/looks.py`, the Test-S
  module, the builder's records, `scripts/train/orbitv2_power.py`); the code changes ordered by R-21, R-23, C-1,
  C-2; keys-only builder runs; the literature check of R-12; train-only or synthetic smoke tests.
- **After the loader review of R-21 has passed and dev_v5 exists:** the value-reading dev_v5m build (once), under
  R-3, R-7, R-8, R-9, C-1, C-9 and R-25.
- **After the extension of R-23 passes its tests and dev_v5m exists:** comparator families on dev_v5m and their
  validation rows (R-10 to R-14b), each behind `require_stamp`; the power script's hash registered first (R-16).
- **After the event of R-19 and the registered configuration hashes of R-20:** A0 and A1 networks, the
  validation numbers of the arms, selection (§3.6) and the kill criterion (§3.7, R-18a), by registered code.
- **Not released by this stamp:** A2, A3, A4; Track G; `FREEZE_orbit_S.json`, any look, any claim row; any use of
  an M7 file other than the list of C-1; any imagery download.
- v1 work (dev_v5 build, v1 selection, the regenerated v1 freeze candidate and its review, M2/test, M2U/test)
  proceeds under the protocol and is not delayed by this file (round rule 4).
