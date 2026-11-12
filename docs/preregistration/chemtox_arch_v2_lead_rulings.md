# M4 SENTINEL-Toxin — architecture round v2 amendment (second text): lead review, rulings and stamp

**Written by:** the lead (AI-assisted), 2026-11-10. The binding stamp time is the timestamp of the registry row
(`chemtox` / `chemtox/prereg_snapshot`) that carries the sha256 of this file.
**Stamps:** `docs/preregistration/chemtox_arch_v2_amendment.md`
- sha256 `fb52cd4b072fa19c8c1e19b9fe1374a0e8abb71d9492eff764f2034cb26d293b`
- 1,150 lines, 76,777 bytes, last modified 2026-11-10 14:53:17 EST.

The amendment is stamped **as written, byte for byte**, together with the rulings below. Where this file and the
amendment differ, **this file wins**. Both files are read-only (0444) from the stamp onward. Line numbers (L…)
and section numbers (§…) refer to the stamped amendment unless a different file is named.

**The two files must be read together.** The stamped amendment contains sentences that the rulings below strike or
declare not yet true (the "closed" guard of §8.2, ground (ii) of §6.2, "by default" and every route to a label on
these blocks, the pre-stamp R1–R3 clause of §14, the header's freeze claim for `gpboost_model.py`, the "qualified"
label of §6.4). A reader of the amendment alone would be misled. Every citation of the amendment in a report,
paper or registry row cites this file with it.

**The stamp is partial.** It fixes the design: the universe, the hold-out unit, the split, the blocks, the
comparator ladder, the statistics and the wording of every outcome. It releases **no data access**. The lead does
not write `docs/preregistration/chemtox_arch_v2_pins.json` at this stamp, so `verify_freeze()` in
`sentinel/models/chemtox/arch_v2_guard.py` keeps raising and neither `load_training_pool()` nor `authorise()` can
run. No FREEZE file is written for any block. No look is cleared. All three blocks are descriptive only (F1).
§4 lists what M4 may do and when.

## 1. What the lead checked before stamping

| Check | Result |
|---|---|
| File hash | equals the hash reviewed (`fb52cd4b…293b`), re-computed by the lead before the stamp and asserted again by the stamp script |
| Full re-read | all 1,150 lines re-read by the lead on 2026-11-10 (19:04–19:06 EST) before writing this file, with the guard module `arch_v2_guard.py` (355 lines) |
| Independent review, first text | the first text (kept byte for byte as `chemtox_arch_v2_amendment_v1_unstamped.md`, sha256 `b9326d49aca711f0ceeec1362c21883030a94f83dd6971a2db15cba1babc9a9e`) was reviewed by three independent reviewers (consistency, leakage, statistics) and one verifier. The stamp was withheld and a second text ordered under rulings L1–L11 (§2.1). Files: `docs/preregistration/reviews/arch_v2_20261110/M4_review_*.json`, `M4_verify.json` |
| Independent review, this text | two independent reviewers (integrity, statistics) and one verifier on this exact hash. Verifier recommendation: `stamp_with_lead_rulings`, stated by the verifier to be a close call. 29 verdicts (18 findings, 11 checks of items carried from the first review), 13 lead points, 0 missed blocking issues, 2 missed non-blocking items. Verifier file: `docs/preregistration/reviews/arch_v2_20261110/M4v2_verify.json`, sha256 `b7e6757a62e830ba420ff87167d8dd091e74c8ad6854991db1397cc4d6877f70`. The two reviewers returned their findings as messages; the verifier file reproduces every finding id by id. Every finding is closed by a ruling in §2 |
| Items the verifier did not check independently | five, listed in §1.2. No ruling below depends on any of them |
| A validation, test or protected number on a new arm | none. The registry holds 0 rows whose experiment starts with `chemtox_arch_v2` and 0 rows whose split names `T1_test`, `T6a_test` or `T6b_test`. No pins file, protected-labels file, FREEZE, `AUTHORISED.json`, `SCORED.json` or `ABORTED.json` exists (asserted by the stamp script; counts in the stamp row). See also the disclosure in §2.5 |
| Pinned artefacts | every sha256 that the amendment states for a data file, a report or a code file was re-computed by the lead on 2026-11-10 and equals the stated value. The values that this stamp fixes are in §1.1 |
| Frozen code of the earlier tracks | `sentinel/models/chemtox/unseen_taxa.py` (sha256 `cf1493c60f694741528a3ef751fb467d89e5ff9428ee42788ed32f6c4f00a4de`, last modified 2026-11-09 05:23 EST) and `unseen_taxa_guard.py` (sha256 `e3721450bb97727cb7a1ee1c2401261302b4307d350bc09df7fffdec47edccc6`, last modified 2026-11-09 05:02 EST) were last modified before this round began (L5). `gpboost_model.py`: sha256 `cc3d1b2652c6ffedb3e1b3b12491e1f3a933979d1d72fb32c59ec66f80690700`, last modified 2026-10-24 (S4, C-6) |
| A bar is lowered | no. The rulings remove every route to a §15.1 label on the three blocks, raise B to 10,000 for every interval in a Holm family, require both clusterings for legs (a) and (b), and widen the baseline set of condition (b) |
| Stamp guard | the absent pins file. `verify_freeze()` raises "no pins file … the amendment has not been stamped" until the lead writes it (S2, C-2). This stamp row does not release anything in code |
| Living files (sha256 at the stamp, in the stamp row) | `arch_v2_guard.py` (`6e1ab9c17655afd197225c37d7ffb589127b7e3232fd9203b008ca3254f224fe` at the stamp; it will change under B1), `tests/test_chemtox_arch_v2.py`, `tests/test_chemtox_arch_v2_power.py`, `scripts/eval/chemtox_arch_v2_power_report.py`, `scripts/eval/chemtox_arch_v2_h4_report.py`, `docs/preregistration/chemtox_arch_v2_pins.candidate.json`, `docs/plans/sota_campaign.md`, `docs/plans/architecture_research/LEAD_DECISIONS_20261110.md` |

### 1.1 Hashes fixed by this stamp (S3)

| File | sha256 |
|---|---|
| `docs/preregistration/chemtox_arch_v2_amendment.md` | `fb52cd4b072fa19c8c1e19b9fe1374a0e8abb71d9492eff764f2034cb26d293b` |
| `sentinel/models/chemtox/arch_v2.py` | `39ecc9cc39fb1c945a843ffa8e99d38eb73c3dce378d24d8a4566550e64fe980` |
| `sentinel/models/chemtox/arch_v2_power.py` | `0841903004361e52d0010b8be8e2c38745195e8d0629b0dd7e7c0c01243d4f7a` |
| `$DATA/processed/chemtox_arch_v2/chemtox_t1t6_split_v2.parquet` (0444) | `ab71aace46260980a6f011a561c2280fe0025226978508daa57f96493c5323a3` |
| `docs/preregistration/chemtox_t1_clusters_v2.csv` (0444) | `a906aac1f9dd97577ff473981a1fce5aae1a509e3fb2dcb42f8726dea2d821f5` |
| `docs/preregistration/chemtox_arch_v2_spent_adore_ids_v2.json` (0444) | `3fa94b68ad96c9018737886430ae6d1efd3b1c52dc33b2d98b4857a87f5bd1d8` |
| `$DATA/processed/ecotox/ecotox_aquatic.parquet` (set 0444 at this stamp, C-5) | `beec1f81769aecad423e952c21bca2d0b3c899b2f0f40d80143fdab3cbf4fc9e` |
| `$RUNS/chemtox/arch_v2/t1t6_counts_v2.json` | `9fd67573287e62b6ff50cf4c673e2373a2795fa99bfbfe1224f5601516c6d4b4` |
| `$RUNS/chemtox/arch_v2/t1t6_mde_v2.json` | `f67d0376fa3440e7653c77fd0a74ed78afda9a214979f44fdb5af6bc32151475` |
| `$RUNS/chemtox/arch_v2/adore_partition_inventory_v2.json` | `d2c643b6857aba6c5c1e21a33c35db076ee90388912244b1e0ec29fb318aa15e` |
| `$RUNS/chemtox/arch_v2/aux_row_counts_v2.json` | `e250e48e942b0ccee6b3f73ce1f2dd9ec8fc8372da737417b6de7e7024c59c76` |
| `$RUNS/chemtox/arch_v2/h4_bcf_join_qc_v2.json` | `7dbdd501f47c29be973f7c424b86bec5532c382b36c55f1054c1b834d4ec527c` |

`$DATA` = `/home/bcheng/STOCKHOLM/sentinel_data`, `$RUNS` = `/home/bcheng/STOCKHOLM/sentinel_runs`. The guard's
hash is not fixed by this stamp; it is recorded at the re-pin of B1 (S3).

### 1.2 Items the verifier did not check independently

The verifier lists five: the bootstrap pairs of F7 for T6a_test and T6b_test; the statistics reviewer's
unit-balanced MDE at icc 0.5; the chemprop 0.070 / 0.085 pair and the `rf_emb_dur` hyper-parameters of F10; the
319 shared chemicals of N1 and the 232 chemicals / 619 rows of N2; and the test suites, which the verifier did not
run. None of these carries a ruling: F7 makes the analytic MDE govern and forbids quoting the bootstrap check as a
gate; the unit-balanced estimand is not adopted (lead point (f)); F10 and N1 make the pinned JSON reports govern
over the prose; N2 turns on the guard, not on a count. The test suites are run and reported by M4 with the guard
changes of B1, before the lead's re-review (§4).

### 1.3 Labelled scratch intermediates, recorded and deleted (L8, S1)

The lead recorded and then deleted the following files on 2026-11-10 at 19:11:37–38 EST. Each sha256 was asserted
in the same command, before the deletion. Directory:
`/tmp/claude-1016/-home-bcheng-STOCKHOLM/dbc009e4-477d-4528-910a-e89e964c5c0f/scratchpad/`.

| File | Bytes | sha256 | Last modified (EST) | Note |
|---|---|---|---|---|
| `m4_t1_rows.parquet` | 2,128,975 | `c303ec037abdacd0964aadd8976508070e21c9c36b2d066478dd8074cd1bdfcc` | 2026-11-10 11:47:41 | 98,709 rows; columns `result_id`, `cas`, `clust`, `log10_conc_mol_L` and an index column. It carried the target. The lead read the schema only, no value |
| `m4_t1_clusters.csv` | 30,115 | `2d8ade5edcd2b72eec9eed0c6f8647056ee25a9359376af6e8a2e389f5e4a9a4` | 2026-11-10 11:47:41 | columns `clust`, `rows`, `chem` (counts) |
| `t1t6/t1t6_counts.json` | 3,198 | `1c443fed0f9751ed29e44d1684c5fb98b999a2b015ab133cfb4df6f85b7c1f2d` | 2026-11-10 12:25:05 | byte-identical to `$RUNS/chemtox/arch_v2/t1t6_counts_v1_WITHDRAWN_label_stats.json`, which is kept as the record of the disclosed first-text read (L6) |
| `t1t6/t1t6_mde.json` | 3,981 | `c38a147dba3caf3b3f5516d9f0710b70e3d56a081d84bb9e86e253812d3a7e0f` | 2026-11-10 12:26:40 | byte-identical to `$RUNS/chemtox/arch_v2/t1t6_mde_v1_WITHDRAWN_label_stats.json`, kept for the same reason |

Left in place, because they hold no label: `t1t6/chemtox_t1_clusters.csv` (sha256 `a6e33634…fc4a`, identical to
`docs/preregistration/chemtox_t1_clusters_v1.csv`) and `t1t6/h4_bcf_join_qc.json` (sha256 `3ceead2f…57cd`,
identical to the copy in `$RUNS/chemtox/arch_v2/`). A scan of the scratch directory by the lead found no other
parquet or csv file with a T1/T6 label column. No pinned artefact lives in a temporary directory.

## 2. Rulings (binding)

### 2.1 Rulings L1–L11 of 2026-11-10 14:18 EST (copied from `LEAD_DECISIONS_20261110.md`; still binding)

- **L1.** Spent ADORE test rows leave the T1/T6 universe entirely: no test block, no training pool, no auxiliary,
  multitask, pretraining or ablation set. Spent partitions are derived by code from `results/registry.jsonl`,
  with counted assertions; the id list is frozen with its sha256. A reserved-but-unscored ADORE test partition is
  reported (counts only) and not subtracted without a lead ruling.
- **L2.** Hold-out unit = connected component over (t1_cluster, parent_key family). It is also the bootstrap and
  MDE resampling unit.
- **L3.** The split is re-derived with the same label-blind procedure (same clustering, cut-off, seed, draw rule).
  New artefacts are named `…_v2`. The sealed v1 split and v1 cluster file are not touched. The v1 text is kept
  byte for byte as `chemtox_arch_v2_amendment_v1_unstamped.md`; the v2 header says v1 was reviewed and not
  stamped.
- **L4.** From counts only, M4 tabulates the unit-level MDE at intraclass correlation 0.5 and 1.0 for T1 test
  shares of 15, 20, 25 and 30 % (and the T6a/T6b unit counts) and proposes one share with a written reason. The
  lead rules at the stamp. A block that cannot detect a realistic gain carries no §15.1 label (descriptive only).
- **L5.** `unseen_taxa.py` and `unseen_taxa_guard.py` are not edited. A new power-helper module with tests
  (it reproduces the closed-form unit-level value at icc = 1 and the row-level value at icc = 0).
- **L6.** No label-derived quantity of any v2 sealed block before its FREEZE. The v1 read is recorded as a
  disclosed deviation.
- **L7.** The auxiliary-leak and pretrained-cache verdicts as ruled by the verifier, extended by L1 and L2;
  filtered auxiliary counts are given in the amendment.
- **L8.** Path, size and sha256 of the labelled scratch intermediate are recorded, then it is deleted. No pinned
  artefact lives in a temporary directory.
- **L9.** The claim that the gpboost reference is frozen is struck. `sentinel/models/chemtox/gpboost_model.py`
  is not edited. The Bio-QSARs citation is corrected.
- **L10.** The T1/T6 guard module, pins, experiment key and protected-labels artefact are built as code now and
  reviewed before any look.
- **L11.** The lead adds the v2 blocks to `sota_campaign.md` at the stamp (file name, sha256, counts).

Every verdict of the first verifier (27 in all, `M4_verify.json`) and every item it lists as missed stays binding
as its ruling text says, except where a ruling below is stricter or corrects it (lead point (c)).

### 2.2 Rulings on the findings of the second review

The wording of each ruling is the verifier's recommended ruling, adopted by the lead word for word. The severity
in brackets is the verifier's after its own check.

- **B1 [blocking].** §8.2 items 1, 2 and 4 and the last bullet of §8.3 state REQUIRED behaviour, not the behaviour
  of arch_v2_guard.py at sha256 6e1ab9c17655afd197225c37d7ffb589127b7e3232fd9203b008ca3254f224fe; the word
  'closed' in the §8.2 heading is not yet true. No FREEZE file for T1_test, T6a_test or T6b_test is written until
  the guard (i) compares sha256(FREEZE) with a value the lead recorded in a 0444 file the workstream cannot write,
  (ii) re-runs verify_freeze() inside require_authorised() and refuses load_test_block() once SCORED.json or
  ABORTED.json exists for the block, (iii) calls verify_artifacts() inside scored_pass() before mark_scored(),
  (iv) writes into ABORTED.json what was actually written instead of a constant, and (v) pins
  ecotox_aquatic.parquet and, once built, the protected-labels artefact - each with a test - and has been
  re-reviewed and re-pinned by the lead. An abort after AUTHORISED.json exists spends that block's look; there is
  no second look. A file named FREEZE not written by the lead is void, and a look taken on one voids the block
  (spent, no label). The re-pin may change only the guard, its tests and the pins file: the sha256 of the
  amendment, the split, the cluster table, the spent-id list, arch_v2.py and arch_v2_power.py must be identical
  before and after, the superseded pins file is kept, and the new pins sha256 is written to a registry row.
- **B1, sequencing.** Sequencing (stricter, recommended): the lead does not write
  docs/preregistration/chemtox_arch_v2_pins.json until the guard meets this ruling, so verify_freeze() keeps
  raising and neither a pool read nor authorise() can run on the deficient guard. The lead adopts this sequencing;
  C-1 and C-2 say how.
- **B2 and F4 [blocking].** The hold-out share is 30 % of units on grounds (i) and (iii) of §6.2 only. Ground (ii)
  is struck and carries no weight: the only pre-declared 10 % criterion is the largest CLUSTER's share of the
  universe's rows (7.76 %, met); no per-block, per-unit criterion was ever pre-declared, and the one stated in
  ground (ii) fails for train_pool (13.46 %), T6a_test (14.37 %), T6b_test (13.52 %) and the in-domain subset of
  T1_test (10.75 %). No write-up may cite a '10 % of a block' criterion as met.
- **F1 [blocking].** Under L4 none of T1_test, T6a_test or T6b_test as constituted (micro-RMSE, governing MDE
  0.0959 / 0.1027 / 0.1033 at icc 0.5; 0.1911 / 0.2047 / 0.2044 at icc 1.0) can carry a §15.1 label: all three are
  descriptive only. The words 'by default' and every sentence making a label available 'if the realised difference
  exceeds the governing MDE' (L87-90, L657-660, §12.1 as a route to a label, §15.3) are struck. Conditions (a)-(j)
  are still computed and reported in full as the pre-registered descriptive reading; a result that would have met
  all of them is reported as 'exceeds the pre-registered MDE on an under-powered block; no SOTA label', and a
  result that does not is a measured null. A label on these questions needs a new, separately stamped track (lead
  point f).
- **F2 [blocking].** Condition (a) is read against whichever of mmoe and gpboost_published has the lower
  pool-validation score, whatever its §4.6 status; a §4.6 miss is disclosed beside the result and never removes
  that arm from (a); kill rule 10 is read as 'a comparator that misses §4.6 cannot be the ONLY arm carrying (a) or
  (b)', never as removing a stronger arm; if neither arm yields finite predictions on the full block, no §15.1
  label is available on that block.
- **F3 [blocking].** Condition (b) is read against the lowest pool-validation scorer among ALL entries that are
  neither a candidate nor a published-form arm: R1, R2, R3, xgboost_emb and xgboost_emb_dur, rf_emb and
  rf_emb_dur, lightgbm_noemb, readacross, the trivial, Haber and carry-forward baselines, and every matched
  baseline ensemble; the list is closed, and the chosen entry is named in writing before the look.
- **F5 [blocking].** Legs (a) and (b) must exclude 0 under BOTH clusterings (hold-out unit and reference_number).
  A candidate that passes (a)-(g), (i) and (j) and fails (h) carries no §15.1 label; it is reported as 'ahead on
  the full block, not separable from publication structure'. The word 'qualified' in §6.4 and §12.1 (h) names that
  report, not a class of label.
- **F6 [should fix].** The interval that carries (a) and (b) is the seed-mean paired difference (mean over seeds
  0, 1, 2 of candidate minus mean over the same seeds of the comparator, on identical rows), two-sided 95 %
  percentile cluster bootstrap; the Holm p-value for each leg is 2 x min(share of resamples <= 0, share >= 0) with
  (count + 1)/(B + 1), and B is raised to 10,000 for every interval that enters a Holm family (never lowered);
  seed s of the candidate is paired with seed s of the comparator; all_seeds_better is TRUE only if the point
  difference favours the candidate for each of seeds 0, 1 and 2 against each of the (a) and (b) comparators on the
  full block.
- **F7 [should fix].** The governing MDE is the analytic value of §6.2; the bootstrap check is not a gate and is
  never quoted as one; every write-up that quotes an MDE states both figures and that the percentile bootstrap's
  coverage on heavy-tailed unit sizes has not been verified.
- **F8 [should fix].** Outcome wording is fixed now: all of (a)-(j) met = 'exceeds the pre-registered MDE on an
  under-powered block; no SOTA label'; point difference in the candidate's favour with any condition unmet = 'not
  resolvable on this block'; point difference against the candidate = 'no gain', reported with the same
  prominence; a candidate behind a comparator with a CI excluding 0 = 'worse than <comparator>'. If no candidate
  is ahead of the (a) comparator on any block, RESEARCH_PLAN §15.5 applies and the next track must be harder or
  larger.
- **F9 [should fix].** The headline of each T6 look is the full block as defined in §3, never a subset of bands
  chosen after the look; no band carries a §15.1 label; the sentence 'the headline is scoped to the bands in which
  the sign holds' is struck; per-band results are printed for all five bands beside every headline; the earlier
  ruling stands that the 720-2160 h and > 2160 h bands are descriptive only on both blocks.
- **F10 [note].** Figures in the pinned t1t6_mde_v2.json govern wherever the text differs; rf_emb_dur uses the
  rf_emb configuration with the duration feature added and nothing else, written in the config file before any
  pool fit.
- **S1 [should fix].** Before the stamp the lead records path, size and sha256 of m4_t1_rows.parquet (2,128,975
  bytes, c303ec03...bdfcc) in the rulings file and deletes it together with scratchpad/t1t6/t1t6_counts.json,
  scratchpad/t1t6/t1t6_mde.json and scratchpad/m4_t1_clusters.csv; L8 applies to every labelled intermediate, not
  only the one §2.2 names.
  Done by the lead before the stamp: §1.3.
- **S2 [should fix].** The clause of §14 permitting R1-R3 fits before the stamp is struck; none ran. The stamp
  clears no read of a sealed block and no look. Work on train_pool rows obtained through load_training_pool()
  (fits, pool-validation numbers, and the embedding runs the amendment specifies) begins only when the lead writes
  docs/preregistration/chemtox_arch_v2_pins.json, and the lead writes that file only after the guard meets the B1
  ruling.
- **S3 [should fix].** The stamp record lists the sha256 of the amendment, arch_v2.py (39ecc9cc...fe980),
  arch_v2_power.py (08419030...d4f7a), the split, the cluster table, the spent-id list and ecotox_aquatic.parquet
  (beec1f81...fc9e, set 0444); the guard's hash is recorded only at the re-pin of the B1 ruling. Every addendum is
  a separate file with its own sha256 and registry row; nothing is appended to the stamped amendment.
- **S4 [should fix].** The header's mention of gpboost_model.py under 'the M5 BioAssess freeze' is struck; §11.2
  governs. The file is left unedited until every look whose import closure contains it is complete, and write-ups
  use the corrected citation.
- **N1 [note].** The pinned JSON reports govern over the prose; the count of chemicals shared by T1_test and
  T6b_test is stated in any cross-block statement, as §8.3 already requires.
- **N2 [note].** 'Physically cannot' is read as 'does not through the guard'; any read of ecotox_aquatic.parquet
  for T1/T6 work outside load_training_pool() and build_protected_labels() is a protocol breach that voids the
  block concerned.

### 2.3 Rulings on the lead points

- **(a) Hold-out share.** 30 % of units (609 of 2,029), seed 20261110, on grounds (i) and (iii) only; the v2
  artefacts stand and no _v3 is built.
- **(b) The 97 unscored ADORE test partitions.** The rows of the 97 unscored ADORE challenge x split test
  partitions stay in the T1/T6 universe (4,632 rows: 3,176 train_pool, 1,456 T1_test, 0 T6a_test, 0 T6b_test). No
  workstream scores any of the 97 partitions, and no model fitted on train_pool is scored on any ADORE partition,
  before all three looks are complete. After the looks an ADORE partition may be scored only with a lead ruling,
  with every result_id of T1_test, T6a_test and T6b_test removed from its test rows first (those ids are spent
  under rule 1) and the removed count reported. At each look the scored-partition set derived from the registry
  must still equal the eight partitions behind the frozen spent-id list, or the look does not proceed.
- **(c) The duration window.** The training window is the closed interval 24 <= duration_h <= 96; the earlier
  ruling's literal '(24, 96]' is corrected to '[24, 96]'. Rows at exactly 24 h (9,759) are window rows and belong
  to no T6 band.
- **(d) The largest unit.** Recorded as an observation, not a criterion: the largest cluster holds 7.76 % of the
  universe (pre-declared criterion met); the largest hold-out unit holds 10.09 % (5,938 rows: 4,321 in train_pool
  and 1,617 in T6a_test) and is a retained unit; the Butina cut-off stays 0.4 and the unit definition of L2 stays;
  no criterion on unit share is claimed as met.
- **(e) Reproduction rule.** As F2.
- **(f) Descriptive status and the path to a powered claim.** As F1. Path: a powered confirmatory claim is sought
  on a NEW track with more data or a harder task (RESEARCH_PLAN §15.5; sota_campaign rule 1: future data, a new
  geography or a new source), opened only by its own amendment, reviewed and stamped before any pool fit, whose
  label-free governing MDE at icc 1.0 is at or below an effect size the workstream commits to in writing before
  that power table is computed. Options (i)-(iii) of §15.3 (lower icc, finer unit, tighter cut-off) are refused.
  The stats lens's unit-balanced estimand is not adopted by this stamp: it may be proposed only as such a
  separately stamped addendum, with micro-RMSE still reported with equal prominence and a point difference in the
  candidate's favour on micro-RMSE also required.
- **T6b_test.** As F1; T6b_test is descriptive only, a full block with its own look, never re-described as a
  stratum of T6a.
- **What the stamp unlocks.** As the S2 ruling and the sequencing sentence of point (1).
- **`gpboost_model.py` (§15.4 item 3).** As S4: unedited and unsealed until every look whose import closure
  contains it is complete.
- **Condition (h).** As F5.

### 2.4 Items carried from the first review (verifier's check of each, and how it is closed)

- **BUILDER-PROVENANCE.** Closed by the S1 action and the S3 stamp record.
- **GUARD-GAP / L10.** Closed for the stamp by the B1 and S2 rulings; closed for a look only by the lead's written
  guard re-review.
- **GPBOOST-FREEZE-CLAIM / L9.** As S4.
- **L4 (share sweep and descriptive status).** As B2 and F1.
- **L8 (labelled scratch).** As S1.
- **T6B-POWER-CLAIM / Point 4.** As F1; T6b_test is descriptive only, a full block with its own look, never
  re-described as a stratum of T6a.
- **SHARED-REFERENCE-GATE.** As F5.
- **SEEDS / PRIMARY-METRIC-AND-CI.** As F6 and F7.
- **T6-BASELINE-LADDER.** As F3 and F10.
- **DURATION-FLOOR-AND-BANDS.** As lead point (c) and F9.
- **Missed 3 (reproduction tolerance).** As F2.

### 2.5 Fact and disclosure recorded at the stamp (lead point (5))

At the stamp `results/registry.jsonl` holds 0 rows whose experiment starts with `chemtox_arch_v2` and 0 rows whose
split names `T1_test`, `T6a_test` or `T6b_test`; no FREEZE, `AUTHORISED.json`, `SCORED.json`, `ABORTED.json` or
protected-labels file exists under `$RUNS/chemtox/arch_v2/` or at the guard's labels path. The stamp script asserts
this, and the stamp row carries the registry row count at that moment. Disclosure, in the verifier's words:

> The five adore_trivial_baselines rows with split '<challenge>/all' and model replicate_noise_floor (ids
> 50cd285b0c38, bcef2d0024ee, e09ebf4a40f4, 31ba96c50143, 5440e0c9e2f6) are label-repeatability summaries over
> whole ADORE challenges, not held-out scores; they spend no partition, and no such statistic is computed on a
> T1/T6 block before its look.

The second missed non-blocking item (the scratch parquet carried the target for all 26,718 sealed rows) is closed
by §1.3.

## 3. Lead clarifications (binding)

- **C-1 — "a 0444 file the workstream cannot write" (B1 (i)).** Every agent of this project runs under one
  operating-system account, so no file permission can stop a workstream from writing a file. The requirement is
  therefore met by rule and by audit: for each block the lead records sha256(FREEZE) in a lead-written 0444 file
  under `docs/preregistration/`, at a path that is a constant in the guard, **and** in a registry row with
  `extra.written_by == "lead"`. The guard checks both and refuses if either is missing or differs. M4 writes
  neither of them, nor the pins file, nor any FREEZE file, nor any row of experiment `chemtox/prereg_snapshot`.
  A file or row of these kinds written by anyone but the lead is void, as B1 says for a FREEZE.
- **C-2 — the pins file and the order of events.** (1) M4 changes the guard to meet B1 (i)–(v) and C-1, each with
  a test, and reports the new guard sha256, the test files and the test results. (2) The lead commissions one
  independent review of the changed guard and tests. (3) If it passes, the lead writes
  `docs/preregistration/chemtox_arch_v2_pins.json` (0444) and a registry row with its sha256; the superseded
  candidate file is kept. From that moment S2 opens work on `train_pool`. (4) A FREEZE for a block additionally
  needs that block's own freeze candidate and an independent pre-test review, as for every look in this project.
  The re-pin changes nothing in §1.1 except by adding the guard, its tests and the pins file.
- **C-3 — "pool-validation score" (F2, F3).** It is the selection score of §7.1 (mean micro-RMSE over the
  selection folds of the track concerned), computed on `train_pool` only; lower is better. The comparator of (a)
  and the baseline of (b) are fixed and named in writing in the freeze candidate, before the look.
- **C-4 — reservation (§8.5, L11).** At this stamp the lead adds `T1_test`, `T6a_test` and `T6b_test` to the
  section "Reserved unseen blocks added 2026-11-10" of `docs/plans/sota_campaign.md`: file
  `chemtox_t1t6_split_v2.parquet`, sha256 `ab71aace46260980a6f011a561c2280fe0025226978508daa57f96493c5323a3`;
  rows / hold-out units T1_test 11,363 / 546, T6a_test 11,249 / 694, T6b_test 4,106 / 289. No workstream trains
  or tunes on those `result_id`s; any overlap is reported to the lead as counts. The first-text split
  (`chemtox_t1t6_split_v1.parquet`) is superseded and is not a reservation.
- **C-5 — `ecotox_aquatic.parquet`.** Set 0444 by the lead at this stamp (it was 0664). The only script that
  writes it, `scripts/data/build_ecotox.py`, is not re-run by any workstream while a T1/T6 look is outstanding. A
  changed hash of this file stops every T1/T6 step until the lead rules.
- **C-6 — `gpboost_model.py`.** Its sha256 at the stamp is in §1. S4 holds from that value: a different hash
  before the last look whose import closure contains the file is a protocol breach that the lead rules on before
  any further look.
- **C-7 — addenda (L24–25 of the amendment, S3).** The words "dated addendum appended to this file" are read as
  "a separate, dated addendum file". Nothing is appended to the stamped amendment or to this file. An addendum
  has its own sha256, its own review where it changes a rule, and its own registry row written by the lead.
- **C-8 — the first-text read (L6).** The withdrawn reports of the first text
  (`t1t6_counts_v1_WITHDRAWN_label_stats.json`, `t1t6_mde_v1_WITHDRAWN_label_stats.json`) are kept as the record
  of the disclosed deviation. They are not opened for any design, selection or reporting decision of this round.
  Every write-up of a T1/T6 result states the deviation.

## 4. What M4 may do from the stamp onward

- **Now:**
  - the guard changes of B1 (i)–(v) and C-1, each with a test; this is the one item on the critical path;
  - code, unit tests and synthetic-data tests for every arm and comparator;
  - configuration files, written before any pool fit: `rf_emb_dur` as F10 says, the closed baseline list of F3,
    B = 10,000 and the seed pairing of F6 in the scorer, the outcome wording of F8 in the report template;
  - licence checks;
  - the T2 unseen-taxa track, unchanged, under its own pre-registered plan and pins. Its selection queue may be
    spread over idle GPUs (never a GPU that carries another user's process) with no change of any setting.
- **Not before the lead writes the pins file (C-2):** any read of `train_pool` rows (which goes through
  `load_training_pool()` only, N2), any fit on `train_pool`, any pool-validation number, any embedding run that
  the amendment specifies.
- **Not before a lead-written FREEZE for the block:** any read of a sealed block, any look.
- **Never under this amendment:** a score on any ADORE partition before all three looks are complete (lead point
  (b)); a label-derived statistic of a T1/T6 block before its look; a §15.1 label on `T1_test`, `T6a_test` or
  `T6b_test`; an edit of `unseen_taxa.py`, `unseen_taxa_guard.py` or `gpboost_model.py`.
- **Fixed files:** `arch_v2.py` and `arch_v2_power.py` stay at the hashes of §1.1 (B1). New arms, comparators and
  the scorer go into new modules. If a change to one of the two is needed, M4 proposes it to the lead as an
  addendum under C-7 before any edit.
- **A powered claim** is sought only on a new track under lead point (f): more data or a harder task, its own
  amendment, reviewed and stamped before any pool fit.
