# Review Report — SENTINEL v2, post-results reviews

**Review mode:** post-results (R10). Each section is an independent review of one workstream's test looks.
Reviewers make no git commits, do not re-run test looks, and re-score only from saved predictions into
scratch locations (never into `results/registry.jsonl`).

---

## Downstream — post-results review (2026-11-08 looks)

**Review mode:** post-results · **Date:** 2026-11-08 · **Reviewer:** independent reviewer, Autonomous Review Skill (fresh context; did not do the work)
**Scope:** the five single test looks run 2026-11-08 20:22–22:12 local (01:22–03:12 UTC on 11-09): D2 ToxinWatch it3,
D5 BioIntegrity, D1 Chicago (D1b qPCR nowcast), D1n national beaches, D6 MusselGuard.

**What I did**
- Read the FREEZE and FREEZE_CANDIDATE files, the five protocols (deviation and test-result sections), `testlook.py`,
  the five `scripts/train/downstream_<app>_test.py`, `wqp_track.VBMLR`, RESULTS.md (2026-11-08 entry), the model cards,
  the handoff note, the pre-test gate and delta re-review, RESEARCH_PLAN §8/§15, and `docs/research/07` §b.3.
- Recomputed every FREEZE hash against the current files and against the prereg archives.
- Diffed each archived protocol against the current one.
- Read the registry: `prereg_snapshot` rows, the 576 test rows, and all non-test rows after the first freeze.
- Re-scored every look from its saved `predictions.parquet` (sha-checked) with the frozen `score()` functions.
  Registry writes were monkey-patched to raise. Outputs are in
  `/tmp/claude-1016/-home-bcheng-STOCKHOLM/dbc009e4-477d-4528-910a-e89e964c5c0f/scratchpad/rev_ds/`.
  Each re-scored value was compared with every registry test row (value, CI and p, tolerance 1e-9).
- Recomputed the headline point estimates a second way, with sklearn `roc_auc_score` / `average_precision_score`.
- CPU only, nice 10, ≤ 4 threads, one job at a time. I opened no credential file and touched nothing of M5's.

### Summary

**Overall status: 🟡 PASS WITH CONCERNS** — sound process, but the D1n "SOTA" label is not supported by §15.1.

The single-pass machinery worked as designed:
- Each FREEZE file and `prereg_snapshot` row precedes its look's claim marker.
- Every hashed file matches its prereg archive. The only current-file mismatches are the five protocols, and each
  differs from its archived copy only by an appended test-result section.
- Each look claimed once, saved its predictions once and was scored once. There are no rescores, no duplicate rows,
  no `supersedes` and no validation rows after the first freeze.
- Every registry test row is reproduced exactly from the saved predictions (see the traceability table).
- No fabrication or test-set leakage was found.

The labels for D1 Chicago, D2, D5 and D6 follow their pre-registered rules. One label does not:
- **D1n is labelled "SOTA"** against a re-implemented Virtual-Beach MLR.
- The project's own literature review names a stronger published method family (random forest, Brooks et al. 2016),
  and that family was not in the comparison.
- The project's own GBM beats the claimed logistic on both test sets.
- So the label fails RESEARCH_PLAN §15.1(i).

### Critical issues (must fix before any number leaves the repo)

1. **🔴 C-PR-1 — The D1n "SOTA" label fails §15.1(i): the strongest published method was not in the comparison.**
   *Where:* RESULTS.md 2026-11-08 entry ("SOTA on the D1n track"); `downstream_d1n_beaches.md` §11; handoff
   ("D1n SOTA").
   - §15.1(i) requires "the strongest published method for the task", and R5 says labels follow §15.1 only.
   - The D1n §5 rule names only the VB-MLR incumbent and the strongest simple baseline. That narrows §15.1 rather
     than implementing it.
   - The project's own `docs/research/07_downstream_applications.md` §b.3 lists **"RF (Brooks 2016)"** as a
     must-report SafeSwim baseline. Its literature table (l.118) records Brooks et al. 2016 (USGS authors,
     14 methods): "random forest most accurate, then adaptive-lasso MLR". Searcy & Boehm 2021 also found RF best.
   - D1n has no RF comparator. Its incumbent is a ridge MLR with one shared α. It differs from operational Virtual
     Beach models: no per-site predictor selection, and no turbidity, wave or discharge predictors (disclosed as
     not nationally available). The PLS variant listed in §b.3 is also missing.
   - Evidence that a tree ensemble is stronger here: SENTINEL's own GBM (secondary) beats the claimed logistic on
     both test sets.
     - Registry: GBM − VB +0.0177 known / +0.0215 new, vs logistic − VB +0.0125 / +0.0157.
     - Post-hoc reviewer computation from the saved predictions (descriptive, unregistered, B = 1,000, HUC8):
       GBM − logistic **+0.0051 [0.0016, 0.0095]** known and **+0.0057 [0.0003, 0.014]** new.
   - **Fix:**
     - Relabel D1n everywhere as "**beats its re-implemented VB-MLR incumbent and the strongest simple baseline on
       the D1n track (pre-registered D1n §5 rule met); not SOTA under §15.1(i)** — no RF/PLS published-method
       comparator; margins +0.013/+0.016 AUROC".
     - It may still count for H-G4 "beats its incumbent (§8)": §8's D1 incumbent is the USGS nowcast, and the
       FREEZE pre-registered "as the D1 national track (D1 counted once)". State that it rests on a
       re-implementation.
     - A future SOTA claim needs a Brooks-style RF (with tuning parity) in the comparison and the unseen 2026
       reserve season.

### Warnings (should fix)

1. **🟡 W-PR-1 — The D5 look ran before the M5 freeze, and the M5 protocol's statement about that is wrong.**
   - D5 scored NRSA 2023-24 (= M5 Test C) at 20:25–20:32 local.
   - M5's FREEZE was prepared at 22:00:23 local; its `prereg_snapshot` is at 03:02:27 UTC.
   - `bioassess_protocol.md` says "Test C (2023-24) was not scored by any workstream at freeze (D5's 2023-24 test
     is pending; the post-results report records which pass ran first)". For the record: **D5 ran first, by about
     1.5 h.**
   - `BI.build_features` also builds the `chem` block (`chemistry_wide.parquet`, i.e. M5's targets) for all units,
     including 2023-24, inside the D5 test process. No D5 test model uses it and nothing is saved: there are no chem
     columns in `predictions.parquet` and no chem rows in the registry.
   - I found no feedback path to M5. The risk is negligible, but the M5 statement must be corrected by an addendum
     in M5's own post-results review. The D5 protocol §6a and the D5 model card should also note the order.
     Nothing was edited here: M5 is mid-pass and its files are frozen.
2. **🟡 W-PR-2 — RESULTS.md omits the configuration counts and selection notes that §15.5 guard 2 requires next to
   each claim.**
   - Configurations tried: D1n 20, D1 50, D5 92, D6 132, D2 72 (+16 D2n).
   - Selection notes to add: D1n chose logistic over GBM on validation; D1 selected over 15 feature sets.
   - The protocols carry these, but RESULTS.md does not. Add them.
3. **🟡 W-PR-3 — How D5 counts toward H-G4 is still open; the pre-registered text points to "do not count".**
   - The D5 rule (§8 W8, 2026-10-24 pre-training review; restored at the 2026-11-08 gate, §10) requires beating
     the Hill-2017 incumbent **and** the feature-matched RFs, with Holm over four. On test, Hill published/tuned
     are rejected (p 0.005/0.002), but feature-matched default/tuned are not (p 0.062/0.027 vs Holm thresholds
     0.05/0.025).
   - W8 exists precisely because the gain over Hill could come from the added geography features rather than the
     learner. Counting D5 under the bare H-G4 wording would relax a pre-registered rule after seeing the result.
   - Recommendation: D5 does **not** count toward H-G4 "beats incumbent", or counts only with the explicit
     qualifier "beats Hill-2017 RF, not a feature-matched RF".
   - The H-G4 verdict is unaffected: 1 (or at most 2) of the 4 needed → **H-G4 not met** either way.
4. **🟡 W-PR-4 — The handoff note contradicts itself.**
   - `docs/handoff/downstream.md` line 3 still says "Validation only everywhere; no test block has been loaded or
     scored", and the state table shows pre-look validation numbers as current.
   - Its last section records the five spent test blocks.
   - Fix the header so nobody re-uses a spent block.
5. **🟡 W-PR-5 — Several primary intervals rest on few effective clusters.** None of these carries a claim, so the
   labels are unaffected. Report them as caveats where the numbers are quoted.
   - D1 Chicago: the beach unit has **12 clusters** (the week unit has 32), so percentile CIs are likely too narrow.
   - D2: Lake Washington is about 60 % of known-lake rows, and one lake is about 39 % of new-lake rows. GBM AUROC
     CIs are 0.739–0.974 and 0.640–0.943.
   - D6 T = 2021: 66 positives in 38,390 front rows.
   - D1n new set: the last Holm step passes at p 0.016 vs 0.05, and the CI lower bound is +0.0023.

### Observations

- 🟢 **Freeze evidence keeps working only through the archives.**
  - The protocol `.md` is a hashed scoring file, so appending §11–§13 makes `check_roles` fail on the current tree.
    That blocks a rescore, which is acceptable.
  - Ordering evidence for the looks now rests only on the prereg archives plus the snapshot rows, which verify.
    Keep the archives.
- 🟢 **Model cards** (`docs/model_cards/downstream_*.md`) are templated. They carry no numbers and no label.
  - They omit the key caveats: re-implemented incumbents (D1n VB-MLR, D6 Kanankege, D2 "INLA-class"
    `SparseStGlmm`), "new = new HUC4" for D1n, and the D5/M5 overlap.
  - Add a one-line evaluation summary with registry ids.
- 🟢 **The D2 "INLA-class" incumbent** is a regularised GLMM-style logistic with splines, not INLA. The wording is
  acceptable because no claim is made; say "INLA-style re-implementation".
- 🟢 **D6 runtime:** scoring took 84 min (bootstrap over 163k rows × 9 sets), with registry rows written
  continuously. This matches a single process (PID 1094997, one claim marker) and is not a second batch.
- 🟢 **Point estimates rounded in the text** (e.g. D1 GM 0.7295 → "0.730" in the protocol, "−0.000" for −0.0001)
  are consistent with the registry.

### Verified (evidence)

**Ordering** (local time, EST; registry in UTC = local + 5 h)

| App | FREEZE written / snapshot row | TEST_LOOK claimed | predictions saved | first test row | last test row / SCORED | test rows |
|---|---|---|---|---|---|---|
| D1 Chicago (safeswim) | 20:21:59 / 01:21:59Z | 20:32:35 | 20:32:39 | 01:32:42Z | 01:33:41Z / 20:33:41 | 36 |
| D1n | 20:22:04 / 01:22:04Z | 20:33:48 | 20:34:31 | 01:34:44Z | 01:40:13Z / 20:40:13 | 52 |
| D5 | 20:22:09 / 01:22:09Z | 20:25:31 | 20:28:48 | 01:28:50Z | 01:32:29Z / 20:32:29 | 168 |
| D6 | 20:22:11 / 01:22:11Z | 20:40:17 | 20:48:30 | 01:48:59Z | 03:12:36Z / 22:12:36 | 234 |
| D2 | 20:22:17 / 01:22:17Z | 20:22:52 | 20:23:41 | 01:23:44Z | 01:25:25Z / 20:25:25 | 86 |

- **Markers:** each `TEST_LOOK.json` stores the sha256 of its FREEZE file, and all five equal the current FREEZE
  files (read-only, mode 0444).
- **Prediction files:** each `PREDICTIONS_SAVED.json` sha equals its `predictions.parquet`.
- **Archives:** all five prereg-archive shas match their FREEZE records. Every hashed file and the split file in each
  archive match the FREEZE per-file shas. `candidate_record_sha256` matches.
- **Protocol diffs:** each current protocol equals its archived copy plus a pure append (D5 +10 lines, D1n +12,
  D6 +9, D1 +9, D2 +8 lines). Every other hashed file matches the current tree.
- **Registry:** 576 test rows (36 + 52 + 168 + 234 + 86). There are no test rows before 01:22Z, no duplicate
  (experiment, model, metric, split) keys, and no `supersedes`.
- **Exactly once:** no `results_rescore.json` exists, `extra.rescore` is false on all rows, and there are
  **0 downstream non-test rows after the first freeze** (last validation row 22:25:52Z on 11-08).

**Traceability** — audit re-score from the saved predictions vs the registry

| App | registry test rows | reproduced exactly | mismatches | audit rows not in registry | claim_met (audit = recorded) |
|---|---|---|---|---|---|
| D1 Chicago | 36 | 36 | 0 | 0 | False = False |
| D2 | 86 | 86 | 0 | 0 | False = False |
| D1n | 52 | 52 | 0 | 0 | True = True |
| D5 | 168 | 168 | 0 | 0 | False = False (per-seed Δ vs the four RFs: 0.0111–0.0128, 0.0074–0.0092, all > 0) |
| D6 | 234 | 234 | 0 | 0 | False = False |

**All 576 test rows were reproduced exactly from the saved predictions**, with 0 mismatches and 0 extra rows.

- **Independent recomputation (sklearn):** the headline AUROC/AUPRC agree to 4 d.p.
  - D1n: 0.8152 / 0.8026 / 0.7577 / 0.8203 (known); 0.7860 / 0.7703 / 0.7086 / 0.7917 (new).
  - D5: ensemble 0.8608, Hill 0.8489 / 0.8479, feature-matched RF 0.8526 / 0.8515; n = 979, 600 Poor.
  - D1: 0.7294 / 0.6862 / 0.7295; n = 2,300, 101 positives.
  - D2: 0.9329 / 0.9284 / 0.8932 / 0.8830 and 0.8577 / 0.8352 / 0.7185 / 0.8312.
  - D6 front AUPRC: 2018 0.0608 vs 0.0166; 2021 0.0184 vs 0.0073.
- **Number check:** every number in the RESULTS.md entry and in the five protocol test sections matches a registry row
  (checked line by line). The model cards contain no numbers.

**Claim labels vs pre-registered rules**
- **D1 Chicago:** Holm m = 4 (two unscored primaries at p = 1). Week p 0.101 / 0.926; beach p 0.038 / 0.869. Nothing
  rejects → competitive ✔.
- **D1n:** the D1n §5 rule is met (Holm m = 4: p 0, 0, 0.004, 0.016 ≤ 0.0125 / 0.0167 / 0.025 / 0.05; deterministic
  model). The "SOTA" wording fails §15.1(i) → C-PR-1.
- **D5:** rule text in §8 W8 (dated 2026-10-24, before any fit, per the pre-training review) and §10 (2026-11-08,
  inside the archived protocol). p 0.002 / 0.005 / 0.027 / 0.062 against Holm 0.0125 / 0.0167 / 0.025 / 0.05 → two
  rejections. The 0.027 > 0.025 step fails → competitive ✔. The agent's "not met" reading is correct.
- **D6:** Holm m = 14. Eight p = 0 rejections (7 at 2018, plus the SC rule at 2021); the next p is 0.020 > 0.05/6, so
  stop. The both-origins rule fails → competitive ✔. Not counted toward H-G4 (pre-registered) ✔.
- **D2:** per-family Holm m = 2. Incumbent family p 0.211 / 0.053 → no claim → competitive ✔.
- **Seeds:** the seed rule holds for D2, D5 and D6 (per-seed Δ all > 0). D1 and D1n are deterministic (stated).
- **H-G4 tally:** at most 1 confirmed "beats incumbent" (D1 via D1n, resting on a re-implementation) of the 4 needed
  → **not met** ✔ (D4 TapRisk and D7 outstanding).

**Leakage**
- **D1n:**
  - Every `test_new` row lies in a test-role HUC4 and every `test_known` row in a train/val HUC4 (split file).
  - The two sets share no cluster.
  - Dates are 2021-02 to 2025-11 (2020 and 2026 absent).
  - All comparators, including the incumbent, are fit on train + val rows (as registered in §10 before the look).
  - Strata thresholds come from train-role rows.
- **D5:** fit = train + val cycles. "New" = location not in the fit set. Test labels enter only `ev`.
- **All five apps:** no registry or validation activity after the freeze, so no selection on the test block.
- **D5/M5 overlap:** 2018-19 is disclosed in RESULTS.md and in the M5 protocol. The 2023-24 order is not → W-PR-1.

**Statistics**
- **Bootstrap units:** cluster bootstrap units as registered: HUC8 (D1n, D6), HUC4 (D5, HUC8 secondary), lake (D2),
  ISO-week + beach (D1); B = 2,000, seeded, so exactly reproducible.
- **Non-estimable comparisons** are flagged and never registered. The audits found none, and every paired bootstrap
  had all 2,000 replicates valid in D1/D2.
- **p-values** are percentile-bootstrap (2 × min-tail). They are adequate at the thresholds used: B = 2,000 resolves
  0.05/14.

### Integrity verdict
- Fabrication detected: **No**
- Data leakage detected: **No** (W-PR-1 is an ordering/disclosure issue; no M5 target was used or output)
- All results traceable to code: **Yes** — see the traceability table
- Statistical claims valid: **Yes for the computations; No for the D1n "SOTA" label wording** (C-PR-1)

### Checklist summary

| Category | Items checked | Passed | Failed | N/A |
|---|---|---|---|---|
| Ordering / single pass | 8 | 8 | 0 | 0 |
| Data leakage | 6 | 5 | 0 | 1 |
| Experimental design (comparators) | 5 | 4 | 1 | 0 |
| Reproducibility (re-score) | 5 | 5 | 0 | 0 |
| Result authenticity / traceability | 6 | 6 | 0 | 0 |
| Statistical validity / labels | 8 | 7 | 1 | 0 |

---

## M4 ChemTox — post-results review (prospective pass, 2026-11-09)

**Review mode:** post-results · **Date:** 2026-11-09 · **Reviewer:** independent reviewer, Autonomous Review Skill (fresh context; did not do the work)
**Scope:** the single prospective pass (crashed attempt 2026-11-08 15:53–19:58, deviation D1; scored re-run 20:29:56 → 01:02:09 EST
on GPU 1); 189 registry rows `chemtox_prospective_test`; `prospective_results_primary.json` (sha256 `709803d0…6eb0b`) and
`score_primary.log` (`2e96d089…7a7`); `scripts/eval/chemtox_prospective_posthoc.py` and its 44 rows
`chemtox_prospective_test_posthoc`; `docs/preregistration/chemtox_prospective_plan.md` (stage 1, stage 2, W1–W3 addendum, D1);
the handoff and round-4 notes; RESULTS.md l.73; `docs/model_cards/chemtox.md` l.124–149; RESEARCH_PLAN §5.4, §15.1, §15.5;
the M4 sections of `REVIEW_REPORT_PRE_TRAINING.md`.

**What I did**
- Recomputed the sha256 of the data file, the frozen block's inputs (`selection.json`), the scorer, the W1 side file, both
  protected copies and the crashed-attempt archive (`SHA256SUMS -c`: OK). Checked file mtimes against the freeze and the pass.
- Read the scorer end to end (`main()`, `fit_predict`, `build_training_frame`, `track_mask`, Holm), and grepped every
  `sentinel/models/chemtox` model for any use of the target column in `predict(rows=…)`.
- Re-scored the pass from the saved `pred_*.npy` (CPU, ≤ 6 threads, nice, scratch only, no registry write), with my own track
  masks and two bootstraps (the shared `paired_bootstrap`, and my own cluster bootstrap with another seed). I compared every
  number with the protected report and every registry row. Scripts and outputs:
  `/tmp/claude-1016/-home-bcheng-STOCKHOLM/dbc009e4-477d-4528-910a-e89e964c5c0f/scratchpad/m4_review/`
  (`rescore.py`/`rescore.json`, `crust_fair.py`/`crust_fair.json`).
- Re-ran the D1 determinism comparison (all 14 archived models) and a seed-swap sensitivity analysis for `gpboost_multitaxa`.
- **Reviewer diagnostic. It is not pre-registered and is not a claim:** I refit the frozen `rf_published_mwlogp` and `rf`
  configurations (3 seeds, CPU) on exactly the 33,448 rows `gpboost_multitaxa` was trained on. This tests whether the
  `crust_novel` win is a method effect or a training-data effect. Its predictions are kept only in the scratch directory.
- I did not touch the frozen files, the M5-hashed `chemtox/{__init__,evaluate,features,tabpfn_model}.py`, the registry or git,
  and I opened no credential file. A `_trainpart` pass of M4 (PID 1185049) is queued on the GPU lock; I did not touch it.

### Summary

**Overall status: 🟡 PASS WITH CONCERNS. One critical label issue must be fixed before any number leaves the repository.**

The pass itself is clean:
- The freeze and every hash precede it.
- The crashed attempt produced no score.
- The re-run used the identical command and identical hashes.
- All 189 registry rows, every CI, p-value, per-seed Δ, `all_seeds_better` flag, best-baseline identity and Holm set are
  reproduced **exactly** from the saved predictions.
- No fabrication and no test-set leakage were found.

The primary label ("competitive, not SOTA") is correct under §15.1 and §4.4.

The only significant result is not a method improvement. On `crust_novel`, `gpboost_multitaxa` is the **only one of the 21
scored models that was trained on crustacean data**. Every comparator, including the published reference, the best-baseline RF
and the trivial baselines, was fit on fish rows only. Given the same rows, the published RF configuration does at least as well
(1.271 against 1.306).

### Critical issues (must fix before any number leaves the repo)

1. **🔴 C-M4P-1 — The `crust_novel` "beats the published reference and the best baseline" result is a training-data asymmetry,
   but RESULTS and the model card present it as a model improvement.**
   *Where:* RESULTS.md:73 ("`gpboost_multitaxa` … beats both the reference … and the best baseline RF …, Holm-rejected, all 3
   seeds"); `docs/model_cards/chemtox.md:132` (label "improved on a **secondary** track"); `docs/handoff/chemtox.md:24`.

   **Evidence:**
   - `build_training_frame(ds, cfg["extra"])` (scorer l.196–230) adds ADORE's `t-C2C` and `t-A2A` rows (7,334) only for
     configurations whose frozen `extra` is `adore_taxa_all`. In the frozen block that is `gpboost_multitaxa` alone.
   - All seven comparators have `extra=none`. The cycle-2 run directories contain no comparator search with extra rows
     (`rf_*`, `xgboost_*` and `readacross_*` exist only as `_extra-none*`). The trivial baselines are fit on
     `build_training_frame(ds0, "none")` (l.526).
   - 0 of the 9 `crust_novel` species occur in the fish-only training rows; all 9 occur in the multitaxa rows
     (`crust_fair.json`). Consequence: `species_mean` = `species_duration_mean` = `global_mean` = 1.854 on `crust_novel`. The
     trivial baselines fall back to the global mean.
   - Every comparator was therefore asked to extrapolate from fish to crustaceans, which is the cross-taxa weakness the ADORE
     pass already showed (`a-CA2F-diff`). Only the candidate saw the taxon.
   - Reviewer diagnostic (frozen configurations, unchanged, refit on the same 33,448 rows, 3 seeds; post hoc, **not** a result
     to report as a claim):

     | Model, trained on ADORE all taxa | `crust_novel` RMSE | Δ `gpboost_multitaxa` − this, CAS-paired 95% CI | `crust_novel_strict` | `fish_novel` |
     |---|---|---|---|---|
     | `rf_published_mwlogp` (published reference) | **1.271** | +0.035 [−0.115, +0.172] | 1.267 | 1.231 |
     | `rf` (re-tuned) | **1.273** | +0.033 [−0.091, +0.149] | 1.238 | 1.257 |
     | `gpboost_multitaxa` (scored) | 1.306 | — | 1.336 | 1.319 |

     The published method, given the same data, matches or beats the candidate. The scored −0.313 / −0.271 is the value of
     ADORE's crustacean rows, not of GPBoost.
   - Consistent with this, `gpboost_multitaxa` is significantly **worse** than the re-tuned XGBoost on the primary fish track:
     +0.101 [+0.024, +0.166] (report, `fish_novel`).

   **Why it is critical:**
   - §15.1(i) requires the strongest published method and the strongest simple baseline.
   - §15.5 guard 4 forbids under-equipped baselines.
   - Plan §2 promises that "every model … is trained on exactly the same rows, so the comparison is a model comparison". The
     `*_multitaxa` lever was pre-registered, but it was given only to a candidate.
   - The pre-registered decision fields are computed correctly. The **interpretation** printed next to them is not supported.

   **Fix (documentation only; no re-score, no change to the registry rows):**
   - Relabel the result in RESULTS.md:73, model card l.132/l.140–149 and handoff l.24, e.g.: "secondary `crust_novel`:
     `gpboost_multitaxa` — the only scored model trained on ADORE's crustacean/algae rows — has lower RMSE than the fish-only
     comparators (pre-registered fields: Holm-rejected within the track). This is a training-data effect, not a method
     improvement: no comparator was given those rows, and in a post-hoc reviewer diagnostic the published RF configuration
     refit on the same rows scores 1.271 (`REVIEW_REPORT_POST_RESULTS.md`)."
   - Do not use "beats the published method" or "improved" for this track.
   - Any future cross-taxa claim needs a pre-registered multitaxa comparator (`rf_multitaxa`, `xgboost_multitaxa`) on a new
     test block.

### Warnings (should fix)

1. **🟡 W-M4P-1 — `gpboost_multitaxa` seed 0 is non-deterministic. This does not create the `crust_novel` result, but it shows
   that the model is unstable.**
   - Verified: seed 0 differs on all rows (mean |Δ| 0.2195, max 1.30). Seeds 1–2 are bit-identical. The seed-mean differs by
     mean 0.073, max 0.433. The other 13 archived models are bit-identical, or (the three RFs) within ≤ 4.4e-15.
   - Seed-swap sensitivity on `crust_novel`, scored as pre-registered (Δ vs reference / vs best baseline; Holm with the other
     10 candidates' p unchanged):

     | Seed set | RMSE | Δ vs reference | Δ vs best baseline | Holm |
     |---|---|---|---|---|
     | scored (re-run seeds 0–2) | 1.306 | −0.313 [−0.540, −0.123] | −0.271 [−0.479, −0.090] | yes / yes |
     | crashed seed 0 + seeds 1–2 | 1.272 | −0.347 [−0.563, −0.164] | −0.305 [−0.507, −0.128] | yes / yes |
     | seeds 1–2 only | 1.284 | −0.335 | −0.293 | yes / yes |
     | re-run seed 0 alone | 1.379 | −0.240 [−0.504, −0.006] | −0.198 [−0.455, +0.026] | no / no |

     The scored seed 0 is the **worst** seed, so the non-determinism made the reported number more conservative, not less.
   - The seed spread on this track (1.279–1.379) is large. The run-to-run variance of a seed is of the same order as the
     between-seed variance. The three-seed rule (§15.1 iii) still holds: every per-seed Δ is < 0.
   - Both ensembles contain `gpboost_multitaxa`, so their seed 0 is not reproducible either. The effect on the primary track is
     ≤ 0.002 RMSE (stack 1.2079 → 1.2065 with the crashed seed 0). The D1 outcome lists only `gpboost_multitaxa`; add the
     ensembles.
   - The root cause is undiagnosed. `scratchpad/m4/gpbdiag/` contains a fold-0 CV diagnostic, but its logs are empty and it
     has no DONE flag. Either run it, or state "cause unknown".

2. **🟡 W-M4P-2 — "Parent-structure-clustered CIs (§4.5) change no conclusion" was never checked for the one conclusion that
   exists.**
   - *Where:* RESULTS.md:73; model card l.148–149.
   - `chemtox_prospective_posthoc.py` computes W2 only on `fish_novel` and `fish_novel_strict`, as §4.5 lists. But the
     `crust_novel` result is a §4.4-style claim, and §4.5 itself counts `crust_novel`'s 30 CAS in 11 parent groups.
   - The reviewer computed it: `crust_novel` parent-clustered CI vs reference [−0.614, −0.066], vs best baseline
     [−0.541, −0.045]. Both still exclude 0, so the sentence happens to be true.
   - Scope the sentence ("on the fish tracks; on `crust_novel` computed post hoc by the reviewer, also excluding 0"), or add the
     crust computation to the post-hoc script as a labelled post-hoc item.

3. **🟡 W-M4P-3 — "Holm-rejected" on `crust_novel` controls only within that track.**
   - The scorer applies Holm per track across the 11 candidates (l.656–670), as §4.2 implies.
   - Seven secondary tracks × 2 references are not corrected jointly. The `crust_novel` p-values (0.001 vs reference, 0.004
     vs best baseline; bootstrap resolution 0.001) would not survive a correction across tracks: Bonferroni over 7 × 11
     needs p ≤ 0.00065.
   - Write "Holm-rejected within the track (11 candidates); not corrected across the 9 tracks" wherever this result appears.

4. **🟡 W-M4P-4 — The model card's W1 sentence misstates the ranges.**
   - *Where:* `docs/model_cards/chemtox.md:147–148`: "the tree models and the stack score 1.11–1.19 on those rows and
     1.24–1.32 on the rest".
   - From `prospective_posthoc.json` (seen / unseen): `rf` 1.205/1.311, `lightgbm` 1.196/1.349, `lightgbm_aug` 1.149/1.413,
     `gpboost` 1.168/1.435, `gpboost_multitaxa` 1.170/1.458. The true ranges for the tree models and the stack are
     **1.11–1.21 / 1.24–1.46**.
   - The card should also say that `chemprop` is the best model of all on the ADORE-unseen rows (1.202, against stack 1.243
     and reference 1.287). That is descriptive only, but it is the opposite of the headline ordering, and it is the
     sub-population closest to "chemicals tested after ADORE".

5. **🟡 W-M4P-5 — The scored pass's predictions are not protected, and a second pass is queued in the same directory.**
   - The 18 `pred_*.npy` of the primary pass are mode 0664 and have no checksum file. W1, W2, this review and any future
     audit depend on them.
   - `m4_queue/trainpart_pass.sh` (PID 1185049, waiting on the GPU lock) will write into the same directory. It will
     overwrite `prospective_results.json` and `score.log` (the protected `_primary` copies exist) and add
     `pred_*_trainpart.npy`.
   - Fix: before the trainpart pass starts, `chmod 0444` the primary `pred_*.npy` and write a `SHA256SUMS`, as was done for
     the crashed archive.
   - Also: the handoff (l.27–28) says the trainpart pass needs the lead's go-ahead, while the script says "lead-approved
     2026-11-09". Reconcile the two.

6. **🟡 W-M4P-6 — The crashed attempt is missing from TRAINING_LOG.md.**
   - TRAINING_LOG has entries for the 14:56 pre-flight and the 20:30 re-run, but none for the 15:53 attempt. The segfault
     killed the process before `TrainingRun` wrote its entry.
   - D1 in the plan documents it, but the project rule is that failed runs are logged. Add a hand-written "FAILED rc 139"
     entry that points to `crashed_pass_20261108_1958/`.

### Observations

1. **"No prospective target value was read" (plan l.569, D1) is literally inaccurate. Its substance is correct.**
   - `pd.read_parquet(data_path)` (scorer l.350) loads **all** columns, including `result_conc1_mean_mol_log`, before any fit.
     The frame `pro` is then passed to `builder.transform(pro)` and `model.predict(Xpr, rows=pro)` for every model.
   - I checked every chemtox model's `predict`: none reads the target column. Chemprop's `_datapoints(X, rows)` uses
     `y=None`. The only target accesses are in `data.py`, `cycle.py` and `prospective.py`, on training or extra rows. The
     target is first used at l.576, after all fits.
   - Nothing derived from the target was computed, logged or written before the crash. 0 registry rows, and the crashed log
     ends at the `chemeleon_tabpfn seed 0 fitted` line.
   - Re-word as "loaded but not used or output". For any future sealed pass, drop the target columns before fitting.
2. Rounding: RESULTS and the model card give the strict-track stack CI as [−0.125, −0.008]; the report has −0.00747 (→ −0.007).
   RESULTS gives the `crust_novel_strict` vs-reference CI upper bound as +0.088; the report has +0.08747 (the model card
   correctly says +0.087).
3. The handoff's first line (`docs/handoff/chemtox.md:3`, "No prospective test row has been scored (0 rows)") contradicts its
   own status section. Update it.
4. Strict track: the stack's Δ vs reference, −0.059 [−0.125, −0.007] (p 0.016 > 0.05/11), is correctly reported as not
   Holm-rejected. Its `all_seeds_better` vs best baseline is False (seed 2 +0.004). The temporal track (chemprop 1.914 vs
   XGBoost 2.248, 29 chemicals) is correctly labelled underpowered, with no claim.
5. The stage-1 builder report computed the test set's target SD (1.730) and replicate floor (0.452) before the freeze. These
   are summary statistics of the test targets, known when configurations were frozen. They are disclosed and not used for
   selection; the risk is negligible.
6. W6 guard: the chemtox-scoped hash `8c5cd6b523a95fed` was identical at both pass starts. The package hash moved
   (`0be57783…` → `a8737ba3…`) because of other workstreams. As before, the authoritative values are the per-row
   `frozen_hash` and `data_sha256`, which are identical across all 189 rows.

### Verified (evidence)

**Ordering** (EST)

| Event | Time | Evidence |
|---|---|---|
| Freeze (`selection.json` written; frozen hash `630b43e905d9b852` recomputed by the scorer at both starts) | 11-08 14:20:41 | mtime; `FINALIZED` 14:20:46; sha256 `bf2ed61c…237328` ✓ |
| W1 side file `chemtox_adore_prefilter_seen.json` | 14:53:51 | sha256 `357571f9…d105` = plan l.512 ✓ |
| Dry-run pre-flight (16 models, "no target value was read") | 14:56–15:20 | `score_primary.log` l.~31–68; TRAINING_LOG 11655 |
| Attempt 1 (seals passed 15:53:36) → segfault during `chemeleon_tabpfn` seed 1 | 15:53:32 → 19:58:50, rc 139 | `crashed_pass_…/single_pass_wrapper.log`; 0 registry rows; no JSON |
| Re-run, same `single_pass.sh` command, same W6 hashes | 20:29:56 → 01:02:09, rc 0 | `scratchpad/m4/single_pass.log` |
| First / last registry row | 00:54:31 / 01:02:05 (05:54:31Z / 06:02:05Z) | after the last fit (chemprop seed 2, 00:53:05) |
| W3 copies + chmod 0444 | 01:02:09 | sha256 of both copies = RESULTS/plan ✓; identical to the unprotected originals |
| Post-hoc script / output | 01:12 / 01:13 | after the pass, implementing the pre-registered W1/W2 only |

No file under `sentinel/models/chemtox/`, the scorer or `sentinel/eval/bootstrap.py` is newer than `selection.json`. The scorer's
sha256[:16] is `a0cb0385226e9aaf` (= stage 2). The data file's sha256 is `3b2fc439…dc54d9` ✓.

**Traceability (re-score from the saved predictions)**
- 7 inference tracks × 18 models: point RMSE, per-seed RMSE, Δ / CI / p vs `published_reference` and `best_baseline`,
  `per_seed_delta` and `all_seeds_better`. Comparison against `prospective_results_primary.json`: **0 mismatches** at 1e-9.
- My own cluster bootstrap (seed 11) gives RMSE CIs within 0.03 of the reported ones on the four main tracks.
- Holm sets are identical. Families are 11 candidates per reference per track, as pre-registered (stage 2 l.448). Best-baseline
  identity per track is identical (XGBoost on the fish tracks; RF on the crust tracks).
- Registry: 189 rows = 9 tracks × 21 models; every value and `ci_low` equals the report. `is_baseline` is True for exactly the
  7 comparators + 3 trivial baselines. No `supersedes`. One `frozen_hash` and one `data_sha256` across all rows.
- Ensembles: `ensemble_mean` rebuilt from the components to 1.8e-15. Stack weights match the log (0.194/0.257/0.042/0.342/0.164).
- Post-hoc: 44 rows = 11 candidates × 2 references × 2 fish tracks. W2 shows no Holm rejection and no "qualified" flag on
  either fish track (stack parent-clustered CI vs reference: fish −0.029 [−0.112, +0.050], strict −0.059 [−0.129, −0.006]).

**Reported table vs source**

| Item (RESULTS l.73 / model card l.130–133) | Report value | Match |
|---|---|---|
| Primary stack 1.208 [1.003, 1.470]; reference 1.237; XGB 1.218 | 1.2079 [1.003, 1.470]; 1.2366; 1.2179 | ✓ |
| Δ vs reference −0.029 [−0.095, +0.039]; vs best baseline −0.010 [−0.070, +0.052]; no Holm | p 0.403 / 0.720; Holm sets empty | ✓ |
| Strict stack 1.205; Δ vs reference −0.059 [−0.125, −0.008]; vs best baseline −0.012 [−0.073, +0.043] | 1.2046; −0.0594 [−0.1245, −0.0075]; −0.012 [−0.073, +0.043] | ✓ (rounding, O2) |
| Published spec as published: XGB 1.528, RF 1.432 | 1.528, 1.432 | ✓ |
| `crust_novel` 1.306 [1.082, 1.538]; −0.313 [−0.540, −0.123]; −0.271 [−0.479, −0.090]; all 3 seeds | identical; per-seed Δ −0.240/−0.309/−0.340 | ✓ (label: C-M4P-1) |
| `crust_novel_strict` −0.070 [−0.278, +0.088] | −0.070 [−0.278, +0.0875] | ✓ (rounding) |
| Trivial baselines 1.83–1.84 on the primary track | 1.833 / 1.844 / 1.834 | ✓ |
| 51% of primary rows ADORE-pre-filter-seen (977 / 117) | posthoc: 977 rows / 117 chemicals | ✓ |

**Labels vs §15.1 and §4.4**
- Primary: no Holm rejection on (a) or (b), so §4.4 is not met → "competitive, not SOTA". Correct, and identical on the strict
  track.
- Seeds: 3 per stochastic model; `all_seeds_better` computed per comparison, with `null` handled for single predictions.
- Single test evaluation after the freeze: yes. The D1 re-run was pre-recorded with its conditions (plan l.559–583).

**Leakage**
- Novel tracks: CAS, InChIKey and `result_id` overlap with ADORE is 0 (verified in round 4). `assert_prospective_chemicals_absent`
  runs on every training frame, including the 7,334 multitaxa rows and the ECOTOX augmentation.
- No prospective row entered selection, NNLS weights (OOF files finite only on ADORE training-fold rows, round 4) or any
  transformer fit.
- W1 (977 primary rows that ADORE's pipeline saw and dropped) was disclosed before the pass with a hashed side file. It is not
  leakage: no pre-filter row is training data.

### Integrity verdict
- Fabrication detected: **No**
- Data leakage detected: **No**
- All results traceable to code: **Yes** (exact re-score, 0 mismatches)
- Statistical claims valid: **Yes for the computations and the primary label; No for the interpretation of the `crust_novel`
  result** (C-M4P-1)

### Checklist summary

| Category | Items checked | Passed | Failed | N/A |
|---|---|---|---|---|
| Ordering / single pass (freeze, hashes, D1, same command) | 8 | 7 | 0 | 1 (TRAINING_LOG gap = W-M4P-6, counted as a warning) |
| Data leakage | 6 | 6 | 0 | 0 |
| Experimental design (comparators, data parity) | 5 | 4 | 1 | 0 |
| Reproducibility (determinism, re-score, protection) | 6 | 4 | 0 | 2 (W-M4P-1, W-M4P-5 as warnings) |
| Result authenticity / traceability | 7 | 7 | 0 | 0 |
| Statistical validity / labels | 8 | 6 | 1 | 1 (cross-track multiplicity, W-M4P-3) |

---

## M5 BioAssess — post-results review (test pass 2026-11-09)

**Review mode:** post-results · **Date:** 2026-11-09 · **Reviewer:** independent reviewer, Autonomous Review Skill (fresh context; did not do the work)
**Scope:** the single authorised test pass. This covers: the run dir `sentinel_runs/bioassess/final_20261108T220242/` (`pred_test{A,B,C}.parquet`,
`pred_T1/T2.parquet`, `comparisons_*.json`, `holm_A.json`, `predictions_manifest.json`, `freeze_record.json`, `cache/` with 253 units);
the marker `final_scored.json` (started 22:30:09, finished 04:45:30 EST); the log `logs/final_test_pass_20261108.log`; and 2,180 registry rows
`bioassess_test`. I also read: `docs/preregistration/bioassess_freeze.json`, the freeze archive `bioassess_freeze_20261108T220227.tar.gz`
(+ `.sha256`), the registry row `bioassess/prereg_snapshot`, `bioassess_protocol.md` (§7, §9.1 claim table, §10), the post-freeze
addendum A1, `docs/handoff/bioassess.md`, RESEARCH_PLAN §15.1–15.2, the M5 sections of `REVIEW_REPORT_PRE_TRAINING.md`, RESULTS.md:33,
and the M5 entries in `logs/training_runs.jsonl`.

**What I did**
- **Hashes.** Recomputed the sha256 of all 31 frozen code files and all 9 frozen data files, plus the TabPFN checkpoint, the archive and the
  34 files of the prereg snapshot. I extracted the archive into scratch and compared it with the record.
- **Freeze record history.** Diffed the frozen record against the candidate record used by the 19:18 dev dry run, which predates D5's 2023-24
  look (20:28 EST).
- **Registry and run log.** Checked registry timing and contents: every `bioassess` experiment/split, every NRSA 2018-19 and 2023-24 row from
  any workstream, and the comparator-evidence row ids. Read the TrainingRun records.
- **Scorer code.** Read the scorer's pass, scoring and resume code: `run_pass`, `Predictor`, `score_frame`, `holm_adjust`, `paired_vs`,
  `paired_bootstrap`.
- **Independent re-computation from the saved predictions** (CPU, ≤ 6 threads, nice). I used my own vectorised cluster bootstrap (different
  RNG, B = 10,000 on Test A) and my own Holm. This covers:
  - pooled R²;
  - the 18-test SOTA family and the 6-test H-M5 family;
  - per-seed Δ;
  - 50-km-block and ecoregion sensitivity;
  - both C controls, on Tests A and C;
  - per-ecoregion Δ;
  - RMSE and median absolute error;
  - a full per-target ranking of all 21 models on A, B (all/new/re-sampled) and C;
  - T1 R² from `pred_T1.parquet`.
- **Exact re-score with the frozen scorer.** I ran `scripts/eval/bioassess_final.py --score-from-predictions` with `SENTINEL_RUNS` pointed
  at scratch. The new run dir and its `scratch_registry.jsonl` are in scratch. `log_result(registry_path=…)` writes only to that file.
  I checked afterwards that the main registry gained no `bioassess` row.
- Scripts and outputs: `/tmp/claude-1016/-home-bcheng-STOCKHOLM/dbc009e4-477d-4528-910a-e89e964c5c0f/scratchpad/m5rev/`
  (`recompute.py`/`recompute_out.txt`, `more.py`, `cmp.py`, `rescore.log`, `runs/`).
- **Not done:** I did not re-run the test pass and made no git operation. I opened no credential file and edited nothing except this report.

### Summary

**Overall status: 🟡 PASS WITH CONCERNS. The pass is clean and fully traceable. The labels need scoping before any number leaves the repository (C-M5P-1).**

**Integrity.**
- The freeze, the archive and the prereg-snapshot row (22:02:27) precede the marker (22:30:09) and the first log line (22:30:25).
- Every hash matches.
- There was exactly one real pass: no resume, 0 cached units reused, 253 fitted.
- No `bioassess_test` row existed before 09:43:59 UTC.
- The 2,180 rows match the dry-run row count, and `not_estimable` is empty.
- **The frozen scorer, re-run from the saved predictions into scratch, reproduces all 2,155 A/B/C registry rows exactly.** That is 0 value or CI mismatches at 1e-12, with byte-identical `holm_A.json` and `comparisons_{A,B_all,C}.json`. The 25 T1/T2 rows cannot be re-scored by design (O4).
- My independent bootstrap (different RNG) gives the same 18 SOTA and 6 H-M5 Holm decisions. All Δ agree to < 0.001, and all CIs to about 0.001.

**Lead's preliminary reading.** It is arithmetically correct:
- Candidate C is Holm-rejected in the 18-test SOTA family for chloride, pH, conductivity and TN on Test A, with all three seeds positive. It is
  not rejected for TP or turbidity.
- C beats its leave-BioSet-out control on 6/6 targets and its equal-size control on 5/6 (TP CI includes 0).
- Test C reproduces the same four targets.
- Candidate A meets H-M5 on 6/6 targets.

**What the reading leaves out.** The claims are much narrower than "SOTA on 4/6":
- The win belongs to a three-model ensemble whose other two members are the baselines. BioSet alone is significantly *worse* than the
  per-target comparator on TN, conductivity, chloride and pH.
- The gains are ΔR² of +0.011 to +0.024, about 1–2 % lower RMSE.
- TN does not survive ecoregion resampling.
- In the cycle-out Test B, the advantage disappears: a lat/lon/elevation RF beats every community model on TN, conductivity and chloride.

### Critical issues (must fix before any M5 number leaves the repo)

1. **🔴 C-M5P-1: Scope the "SOTA" labels to the pre-registered endpoint and the ensemble, or they mislead.**
   *Where:* the labels in `holm_A.json:2-30` (`sota_label: true` for C × {Cl, pH, cond, TN}). No M5 test result is in RESULTS.md, the model
   card or the paper yet; this is a condition on that write-up.

   **(a) Endpoint scope: the label holds only under leave-one-ecoregion-out.**
   - On Test B (cycle-out; fit on all of 2008-14, predict 2018-19), C's Δ vs the same SOTA comparators is about 0:

     | Target | Δ R² (B_all) | 95 % CI |
     |---|---|---|
     | TN | −0.003 | [−0.008, +0.002] |
     | conductivity | +0.001 | [−0.005, +0.007] |
     | chloride | −0.001 | [−0.007, +0.005] |
     | pH | +0.006 | [−0.004, +0.017] |

     Source: `comparisons_B_all.json`.
   - `RandomForest[geography]` (lat/lon/elevation only) is the best model of all 21 on B_all for TN, conductivity and chloride (R², from
     `pred_testB.parquet`):

     | Target | RF[geography] | C |
     |---|---|---|
     | TN | 0.680 | 0.619 |
     | conductivity | 0.740 | 0.712 |
     | chloride | 0.706 | 0.642 |

   - The same holds on re-sampled locations (TN 0.770 vs 0.651) and, for chloride, even on new locations (0.636 vs 0.619).
   - So whenever the region has been sampled before, community-based inference adds nothing over location for these targets.
   - **Required wording:** "SOTA under leave-one-ecoregion-out transfer to an unseen region (NRSA 2018-19, replicated on 2023-24)". Test B,
     including the geography-only result, must be reported next to it.

   **(b) Model scope: the label belongs to the ensemble, not to BioSet.**
   - C = mean(BioSet+geography, RF[community+geography], TabPFNv2). No single SENTINEL model is SOTA.
   - Candidate A (BioSet) minus the per-target comparator on Test A:

     | Target | Δ R² | 95 % CI |
     |---|---|---|
     | TN | −0.059 | [−0.081, −0.037] |
     | conductivity | −0.038 | [−0.062, −0.016] |
     | chloride | −0.031 | [−0.055, −0.009] |
     | pH | −0.036 | [−0.071, −0.003] |

   - BioSet+geography alone ties the comparator (chloride 0.466 vs 0.469, pH 0.439 vs 0.438).
   - The gain is therefore ensemble diversity. The controls do show it is BioSet-specific, not an effect of ensemble size:
     - leave-BioSet-out: Δ > 0 on 6/6 targets with all seeds positive;
     - equal-size control: Δ > 0 on 5/6, larger than against the leave-out ensemble.
   - **Required wording:** "BioSet adds signal to the strongest baselines (SOTA as an ensemble member)". Do not write "BioSet is SOTA".

   *Fix:* add both scopes, the Test B table and the candidate-A Δ to RESULTS, the model card and the handoff. Do not change the registry; it is
   correct.

### Warnings (should fix)

1. **🟡 W-M5P-1 (high): TN "SOTA" is fragile. It meets the pre-registered rule but no robustness check.**
   - On Test A, Δ = +0.0107 with location-cluster CI [+0.0043, +0.0171], Holm p = 0.021, and all seeds positive (+0.003, +0.011, +0.007).
   - The ecoregion-unit CI includes 0: [−0.013, +0.030] (`comparisons_A.json:776`). The 50-km block CI does exclude 0: [+0.004, +0.018].
   - Only 4 of 9 ecoregions are positive. The pooled gain is carried by WMT (+0.178); NAP is −0.064.
   - Median per-ecoregion R² is lower for C (0.163) than for the comparator (0.201).
   - Median absolute error is slightly *worse* (0.2176 → 0.2199 log10).
   - Test C repeats the pattern: ecoregion CI [−0.012, +0.032], 6/9 ecoregions positive.
   - §15.1(ii) asks for a "block-bootstrap" CI. The 50-km block CI satisfies it.
   - **Recommendation:** report TN as "SOTA by the pre-registered rule, not robust to ecoregion resampling". Do not count it among the robust
     targets. Chloride, conductivity and pH pass all three resampling units on both A and C. Chloride's ecoregion lower bound on A is only
     +0.001.

2. **🟡 W-M5P-2 (high): the effect sizes are small. Say so.**
   - ΔR² of +0.011 to +0.024 corresponds to RMSE reductions of 1.1 % (TN), 1.5 % (conductivity), 2.0 % (chloride) and 2.2 % (pH).
   - In real units: chloride RMSE 0.582 → 0.570 log10, a typical multiplicative error of about ×3.8 → ×3.7. pH RMSE 0.391 → 0.383 units.
   - The absolute skill is moderate. Pooled R² is 0.46–0.57, and median per-ecoregion R² is 0.16–0.34.
   - These gains are statistically solid for Cl, conductivity and pH. They are practically marginal for any management decision.
   - The substantive M5 result is H-M5: a community-only BioSet beats the blocked RF[community] by +0.045 to +0.133 R².

3. **🟡 W-M5P-3 (medium): H-M5 "6/6" depends on seed averaging.**
   - The pre-registered H-M5 rule is "Holm-rejected for ≥ 3 of 6" with no seed requirement, and it is met on 6/6 (`holm_A.json`, `h_m5|…`).
   - But single-seed BioSet beats RF[community] in all three seeds on only 4/6 targets. For TP the seed deltas are +0.020, +0.007, −0.013.
     For turbidity they are −0.013, +0.035, −0.004.
   - The seed-mean BioSet is a 3-network ensemble, while RF[community]'s seed fits are nearly identical.
   - On Test C (secondary), my Holm over 6 rejects 5/6: chloride is +0.025 [−0.002, +0.051] with *all three* seeds negative.
     Turbidity also has all three seeds negative, and TP and TN each have one negative seed.
   - **Recommendation:** report "H-M5 met (6/6 Holm-rejected on Test A; 4/6 with every seed positive; Test C 5/6)".

4. **🟡 W-M5P-4 (medium): §15.1(i) coverage of published methods is reasonable but thinly documented.**
   - The comparator set re-runs the classic transfer-function family (WA-PLS2, kNN-MAT), RF as used in bioassessment work, LightGBM and
     TabPFN v2. It also includes the simple baselines: ecoregion mean, nearest reference site and RF[geography].
   - All are scored on the same rows. WA-PLS2 and kNN-MAT are far below C on Test A (R² 0.10–0.32 vs 0.46–0.57).
   - The per-target comparators are the dev-LOEO argmax, which I verified. Every evidence row is dated ≤ 2026-10-25 03:27 UTC.
   - The one published, code-available method for exactly this task that was not re-run is the US EPA maximum-likelihood / taxon-response
     inference (Yuan's `bio.infer`, R package). It is typically similar to weighted averaging, so it is very unlikely to change a label. It
     should still be named.
   - The literature check is one sentence in protocol §10 (`bioassess_protocol.md:305`), not a dated document. Add
     `docs/novelty_checks/bioassess_*.md` with the search, and state that no published NRSA-LOEO benchmark number exists to compare against.
   - Post hoc, C is the top-ranked of all 21 models on Test A and Test C for all four claimed targets. No unpooled baseline (WA-PLS2,
     kNN-MAT, RF[context], …) beats the chosen comparator there.
   - Minor: on TN, the test-set-strongest baseline is RF[community+geography]+TabPFN (0.5042), not the dev-selected RF[community+context]+TabPFN
     (0.5037). C beats it too: +0.010 [+0.004, +0.016].

5. **🟡 W-M5P-5 (medium): A1 is honest. The wrong sentence still sits in 1,749 registry notes, and Test C rows carry no note.**
   - I confirmed the timing:
     - D5 `biointegrity/test_look` scored NRSA 2023-24 at 01:28:50–01:32:29 UTC (20:28–20:32 EST), 1.5 h before the M5 freeze at 22:02:27 EST.
     - The 19:18 dev-dry-run candidate record, which predates D5's look, has the same candidates, comparators, models, families and decision
       rule as the frozen record.
     - The only differences are the hashes (protocol, scorer, 7 newly hashed `__init__`/chemtox files), `exposure_note`, `resume_rule`,
       `expected_runtime_h`, status and timestamps.
   - So A1 does not affect any claim: no M5 selection used 2023-24, and Test C was a secondary replication.
   - The 2018-19 overlap with D5 (validation cycle since 2026-10-25 03:43 UTC) is also described correctly. All 60 comparator-evidence rows
     predate it, and the candidates were fixed on 2026-10-24.
   - However, every nrsa1819/transfer row (1,749) carries the frozen sentence "Test C (2023-24) was not scored by any workstream at freeze",
     and the 426 Test C rows carry no exposure note at all.
   - A1 requires the correction "alongside every Test C result". **Fix:** RESULTS and the model card must carry the A1 text wherever Test C
     numbers appear (RESULTS.md:33 already has it for the dev line).
   - Optionally add one dated registry annotation row. Do not edit the existing rows.

6. **🟡 W-M5P-6 (low): T1/T2 are labelled correctly as secondary, with a disclosed deviation, and the results are negative.**
   - T1 (UK → NRSA 2018-19, `RandomForest[family-log-class]`) has R² between −0.32 and +0.07 on all five shared targets. I reproduced this from
     `pred_T1.parquet`; for pH it is −0.011.
   - It is far below RF[geography]. The comparator is the Test B (cycle-out) geography fit, R² 0.43–0.74; the LOEO geography fit would give
     −0.02 to 0.20, which is still above T1 on every target except turbidity (T1 0.07 vs 0.03).
   - T2 (NRSA → UK reserved): TP 0.17 [0.07, 0.23]; all other targets negative, chloride −2.74.
   - Rows say "secondary, unadjusted" plus "Deviation from protocol §7.4 … family-level RF, not the encoder".
   - Report T1/T2 as **negative** (no transfer skill).
   - `--score-from-predictions` does not re-score T (O4 not done). My T1 check was manual. T2 was not re-scored: its observations are not in
     the saved file.

### Observations

- **joblib error.** The stderr `KeyError` in loky's `resource_tracker` (log tail, ~04:42) is the known benign memmap refcount warning. It
  appears after `block T predicted`, and scoring then completed:
  - `comparisons_B_resampled.json` at 04:43;
  - registry flush 09:43:59–09:45:30 UTC;
  - TrainingRun "✅ Completed, 2180 rows, 0 not estimable".

  The re-score reproduces every output. Harmless.
- **p-values of 0.** `holm_A.json` reports `p_raw: 0.0` for B = 10,000. Report these as p < 1e-4.
- **Comparator ordering on TP.** For TP, the dev-selected comparator (RF+LightGBM+TabPFN, 0.400) is weaker on test than TabPFN alone (0.407).
  Candidate B's near-miss (+0.018, Holm p = 0.056) would be about +0.011 against TabPFN. TP stays "competitive".
- **Test C overlap with Test A.** Test C shares 598 locations with Test A. This is harmless, because the training for both tests is
  development-only and LOEO. It does make A and C not fully independent replications; say "replicated on a later survey".
- **Last scorer edit not dry-run.** The scorer hash changed after the last dev dry run (19:18; R1/R2/W-b edits). The real pass ran cleanly
  and re-scores exactly, so this had no consequence.
- **Leakage controls.** These are in place and I verified them in code:
  - `run_pass` asserts no location overlap per fold;
  - Test A/C training is `survey ∈ {nrsa0809, nrsa1314}` minus ecoregion g;
  - the vocabulary is built from development tokens only.

  Test B "re-sampled" (50.4 %) is by design, and B is secondary.

### Labels I consider justified (Test A, confirmatory; LOEO endpoint)

| Target | Label | Basis |
|---|---|---|
| Chloride | **SOTA (as ensemble C; LOEO scope)** | +0.021 [+0.013, +0.028], Holm p < 1e-4, seeds +; block50 and ecoregion CIs > 0; C replicates; 7/9 ecoregions + |
| pH | **SOTA (as ensemble C; LOEO scope)** | +0.024 [+0.015, +0.032], Holm p < 1e-4, seeds +; all units > 0; C replicates; 8/9 ecoregions + |
| Conductivity | **SOTA (as ensemble C; LOEO scope)** | +0.013 [+0.006, +0.019], Holm p = 0.006, seeds +; all units > 0; C replicates; 7/9 ecoregions + |
| TN | **SOTA by the pre-registered rule, not robust**; I would headline it as "competitive" | +0.011, Holm p = 0.021, seeds +; ecoregion CI includes 0, 4/9 ecoregions +, median-region R² lower |
| TP | **competitive** | best candidate B +0.018, Holm p = 0.056; C seeds mixed |
| Turbidity | **competitive** | B/C +0.012, Holm p ≈ 0.23–0.26, seed 0 negative |
| H-M5 (A vs blocked RF) | **met** (6/6 Holm; 4/6 with all seeds positive) | — |
| T1/T2 transfer | **negative** | — |

All labels are for leave-one-ecoregion-out only. Under cycle-out (Test B), no candidate beats geography-only for TN, conductivity or chloride.

### Integrity verdict
- Fabrication detected: **No**
- Data leakage detected: **No**. Test C exposure via D5 is a disclosure error (A1), not leakage.
- All results traceable to code: **Yes**. The exact re-score matches 2,155/2,155 rows. The 25 T rows were checked manually for T1 only.
- Statistical claims valid: **Yes for the computations and the Holm decisions. Conditional for the labels:** they need the scope in C-M5P-1,
  and the TN caveat in W-M5P-1.

### Checklist summary

| Category | Items checked | Passed | Failed | N/A |
|---|---|---|---|---|
| Ordering / single pass (freeze → archive → snapshot → marker; hashes; one pass; no earlier rows; joblib error) | 9 | 9 | 0 | 0 |
| Data leakage (fold disjointness, dev-only training/vocab, 2018-19/2023-24 exposure, A1 impact) | 7 | 6 | 0 | 1 (registry note wording, W-M5P-5) |
| Experimental design (§15.1 comparators, simple baselines, controls) | 6 | 5 | 0 | 1 (bio.infer not re-run, W-M5P-4) |
| Reproducibility (exact re-score, independent recompute, T1 recompute) | 5 | 4 | 0 | 1 (T2 not re-scorable) |
| Result authenticity / traceability | 6 | 6 | 0 | 0 |
| Statistical validity / labels (Holm, seeds, CIs, sensitivity, scope, effect size) | 8 | 5 | 1 (C-M5P-1 scope) | 2 (TN, H-M5 seeds as warnings) |
