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

## EW-v2 Test-A — post-results review (2026-11-09)

**Review mode:** post-results · **Date:** 2026-11-09 · **Reviewer:** independent reviewer, Autonomous Review Skill (fresh context; did not do the work)
**Scope:** the single EW-v2 (M1 early-warning) Test-A scoring pass. This covers:
- the run dir `sentinel_runs/sonde/ew_v2_testA_score/`: `primary_endpoint.json`, `controls.csv`, `event_scores.csv`, `score_manifest.json`, `event_probs/` (18 parquet files), `reliability/`, `log.txt`, `data_info.json`;
- the wrapper log `sentinel_runs/sonde/logs/ew_v2_testA_score_20261109.log`;
- the marker `sentinel_runs/sonde/.ew_scored_test.json`;
- the 295 registry rows `ew_v2_test_score`;
- the RESULTS.md entry "2026-11-09 — EW-v2 Test-A single pass".

I also read `FREEZE_ew.json`, `ew_v2_protocol.md` (v0.3, all of §1–§12), addendum A1 (`ew_v2_post_freeze_addendum.md`), `ew_v2_osf_preregistration.md`, `docs/handoff/M1-sonde.md`, RESEARCH_PLAN §15.2, and the score-phase code (`scripts/train/sonde_events.py` `score`, `check_frozen_invocation`, `claim_single_pass`, `verify_snapshots`, `check_chloride_consistency`; `sentinel/models/sonde/guard.py` `load_freeze`).

**What I did**
- **Hashes.** `git show cef72b6:<path> | sha256sum` for `FREEZE_ew.json` (`24716009…`) and `ew_v2_protocol.md` (`0d5c8f2e…`): both byte-identical to the local files.
  - Recomputed `scoring_code_sha256` over the 22 frozen files with `guard.scoring_code_sha256`: `c110bcff…`, matches.
  - `git diff cef72b6 --` those 22 files is empty.
  - Calibration file sha `3c243c1b…` matches `thresholds_sha256`. Both `static_inputs` shas match, and their mtimes are 2026-10-22.
  - `origin/v2-rebuild` reflog: cef72b6 was pushed at 2026-11-09 10:03:19 −0500, before the marker (12:05:28).
- **Single pass / timing.** Compared wrapper start/end, marker `claimed`, registry timestamps, file mtimes in the run dir, and `log.txt` vs the wrapper log.
- **Earlier test access.** Parsed every registry row (66,704):
  - experiment prefix `ew_v2_test*`;
  - any `M1-sonde` row with "test" in split or experiment;
  - every row whose argv contains `--allow-test` or `--eval-blocks test`.
  - I also read the `access` block of all 73 other `data_info.json` under `sentinel_runs/sonde/`, listed every `.ew_scored*` marker, and searched for `*test*` artifacts.
- **Independent recomputation (CPU, nice, ≤ 4 threads).** From `event_probs/*_L3_test.parquet` (role = test) I recomputed the pooled BSS of stack vs climatology (H1, H2, H4, H1-t4). I also ran my own two-stage basin-group > site-month bootstrap (own RNG, B = 2000) for H1, H2 and H4.
  - Script: `/tmp/claude-1016/-home-bcheng-STOCKHOLM/dbc009e4-477d-4528-910a-e89e964c5c0f/scratchpad/ewrev/boot.py`.
- **Traceability.** Matched every number in the RESULTS entry against `primary_endpoint.json`, `controls.csv`, `event_scores.csv` (test-role, lead 3) and the registry.
- **Not done:** I did not re-run the scorer or create any marker, made no git operation, and did not open `keys.env`. I edited nothing except this report.

### Summary

**Overall status: 🟡 PASS WITH CONCERNS.** The pass itself is clean, the verdict is correct and every number traces. The RESULTS entry needs correcting before any number leaves the repository: it overstates the controls (C-EWP-1), omits outcomes the protocol requires (C-EWP-2), and frames the next steps in a way the protocol does not allow (W-EWP-1).

**Integrity of the pass: clean.**
- **Order of events.** The commit was pushed (10:03:19) before addendum A1 was written (12:04:33 local mtime). The wrapper started at 12:05:10. The marker was claimed and the `pass_claimed` row written at 12:05:28 (17:05:28Z). Scoring ended with 294 rows at 14:01:47, and the wrapper exited 0 at 14:01:52.
- **Row count.** 1 + 294 = 295 rows, which is exactly 1 claim + 43 control metrics × 6 cells + 9 primary metrics × 3 + 3 negative-control metrics × 3.
- **Exactly one pass.**
  - One `.ew_scored_*` marker exists (r--r--r--). There is no legacy `…_<sha12>` name.
  - There is one `pass_claimed` row, and no registry row follows the last `ew_v2_test_score` row.
  - The log has a single start/end and no restart or traceback.
  - `log.txt` equals the wrapper log minus the 3 wrapper lines.
  - Every output file's mtime is between 12:07 and 14:01.
- **Checks before the marker.**
  - The log line at 12:05:28 records the calibration sha and the four content-hash data pins (`match: True`).
  - Code-hash verification is not printed. It is enforced: `guard.load_freeze(check_code=True)` raises on a mismatch, and it runs before `claim_single_pass`. My recomputation confirms the hash.
  - Frozen artifacts are re-verified against the calibration file at load (`artifact … changed since calibration`).
  - The handoff records the lead's pre-pass check of `static_inputs`, and I confirm both shas now.
- **No test data was touched beforehand.**
  - There are 0 `ew_v2_test*` rows before 17:05:28Z, and no other M1-sonde row with test access.
  - All 73 other sonde runs have `blocks/roles ⊆ {train, val}` and `protected = []`.
  - The `--allow-test` rows from 2026-11-09 01:23–03:12Z belong to the downstream workstreams (toxinwatch, biointegrity, safeswim, d1n_beaches, musselguard), which do not load sonde data.
  - `sentinel_runs/events/feasibility_sonde` (13:11) is train/val-role, train/val-block only, and ran after the marker.
  - The disclosed prior-knowledge items still stand: W10 split eligibility conditioned on test-period availability, and W11 chloride datasheet counts.
- **Independent recompute.**

  | Hazard | BSS (mine = registered) | My two-stage CI | Registered CI | My p | Registered p |
  |---|---|---|---|---|---|
  | H1 | 0.01271 | [−0.0047, 0.0406] | [−0.0041, 0.0383] | 0.070 | 0.069 |
  | H4 | 0.01892 | [0.0001, 0.961] | [0.0001, 0.973] | 0.020 | 0.020 |
  | H2 | −0.01492 | [−29.4, 0.098] | [−27.5, 0.093] | 0.62 | 0.64 |

  Differences are within Monte Carlo error. The positive-row counts are consistent with 3 issue rows per evaluable onset (127/43, 27/9, 27/9).

**Decision rule: applied as frozen; the verdict is correct.**
- **Power rule.** H1 has 43 evaluable onsets in 11 basin groups, H2 has 9 in 3, and H4 has 9 in 2. All three are under 50 onsets, so all are underpowered (H4 passes only the ≥ 2-group half of the rule). `p_holm_input = 1`, `p_holm = 1` and `reject_holm = false` for each, and m = 3 is kept. `gate_equals_pod_set = true` for all three, so the C2 single onset definition holds.
- **Robustness.** Even with the descriptive bootstrap p (0.020, 0.069, 0.64), Holm at α = 0.05 would reject nothing: 0.020 > 0.05/3 = 0.0167. The negative verdict therefore does not depend on the power rule.
- **What the protocol declares for an underpowered primary.** Nothing that could replace it:
  - §9 pre-declares only a **secondary** pooled Test-A ∪ Test-B **H1** analysis (pass `test_ab`, not before 2027-06-30). It is "never replaced by the pooled analysis; the pooled result is labelled secondary in every report".
  - H2's confirmatory use moves to the live winter trial, which needs a separate amendment.
  - Tiers (H1 < 4, < 2) and H2-acute are **descriptive secondaries** (§7). There is no fallback population and no tier substitution.
  - The observed H1 count (43) matches the pre-stated projection (≈ 44, §1/§9).
- **Plan threshold.** RESEARCH_PLAN §15.2 "EW-v2: primary BSS endpoint passes for ≥ 2 hazards on Test-A …" is **failed**, not pending. Neither the pooled secondary nor the live trial is a Test-A primary, so neither can satisfy it as written.

### Critical issues (must fix in RESULTS.md before any number is quoted)

**C-EWP-1 — The control reading is overclaimed for H2 and leans on the weakest null.**
"The warnings carry real, above-chance timing signal" is stated for all hazards and supported mainly by the circular-shift surrogate.
- **Circular shift is a weak null.** A ≥ 30-day circular shift destroys seasonal alignment, so a pure site × season climatology also beats it.
- **H1.**
  - The signal is real at the alarm level, but the correct evidence is the **season-matched placebo**: POD_real 0.86 vs POD_placebo 0.33. The cluster bootstrap gives POD_real − POD_placebo = +0.53 [0.38, 0.68], p = 0.0005, over 11 basin groups.
  - It also beats **climatology at a similar alarm rate**: climatology POD 0.49 (21/43) at 1.92 alarm-days per 30 d, vs stack 0.86 (37/43) at 1.99.
- **The frozen alarm budget was not held out of sample.**
  - H1 stack realised **1.99 alarm-days per 30 d** against a budget of 1. `b1_meets_budget = False`, 39 % of sites were over budget, and 6.7 % of issue days were in alarm.
  - The ≥ 24 h POD is **0.70**, not 0.86.
  - The protocol (§7) requires that realised rates be reported "as they are". The entry omits them.
- **H2.**
  - "POD 1.0 vs 0.12" was achieved at **4.54 alarm-days per 30 d** (4.5× budget; 80 % of sites over budget; 16 % of days in alarm).
  - **Climatology alone detects 8/9 onsets** (POD 0.89 at 5.41 alarm-days per 30 d).
  - BSS is −0.015, and Diebold–Mariano vs climatology gives p = 0.22.
  - H2 shows no timing information beyond seasonality. Citing it as timing signal is misleading.
- **H4.** POD 0.89 (8/9) at 0.59 alarm-days per 30 d (within budget) vs climatology 0.33 (3/9). This is suggestive but rests on 9 onsets at 6 sites in 2 basin groups, where the cluster bootstrap is degenerate.
- **Unlabelled comparators.** "vs 0.12" and "vs 0.06" are circular-shift surrogate means but are not labelled as such.
- **The rest of the controls are read correctly:**
  - circular-shift mean 0.17, p 0.001 (the floor for 1,000 surrogates);
  - label permutation p 0.001;
  - Molchan skill 0.83;
  - DM vs climatology p 0.022, vs persistence p 0.17;
  - 0.0 alarm-days per 30 d at 11 negative-control sites with 0 onset rows.
- **Weaker tests than they look.**
  - Label permutation tests association, not skill over climatology. Its null mean BSS is −0.010.
  - DM uses a daily site-mean differential with Newey–West and does not account for basin clustering. That is why it gives 0.022 while the cluster-robust primary bootstrap gives 0.069.
  - The negative control (sites whose training DO min never fell below 6 mg/L) is an easy control.
- **Fix:** use the text below.

**C-EWP-2 — Required or material secondary outcomes are omitted.**
- **Approximate CIs not flagged.** Protocol §7 (W4) says that with fewer than 10 basin groups the CI "is stated to be approximate". The H2 (3 groups) and H4 (2 groups) CIs are not flagged. The H4 interval [+0.0001, +0.97] and p = 0.020 look like evidence at first glance: they are degenerate two-group bootstrap artefacts.
- **The H1 < 4 mg/L tier is not reported.** It is the only H1 cell with ≥ 50 evaluable onsets (56, 12 basin groups). Its pre-selected method (`event_gbm_mono`) has **BSS −0.002 [−0.020, +0.020]**, and DM vs climatology gives p = 0.70. The stack gives −0.005.
  - It is a descriptive secondary and cannot be a fallback claim.
  - Leaving out the one adequately sized cell, which is null, while reporting the favourable alarm controls, is selective reporting. It also directly supports "probabilistic skill … not established".
- **H2-acute must not be quoted as skill.** Its BSS of +0.25 has label-permutation p = 0.999: the observed BSS is below the null mean of +0.37, so the apparent skill is site-level label structure. It is not in the entry, but it should be marked "not skill" wherever it appears.
- **Relative economic value is not reported.** For H1 stack it is > 0 only at C/L 0.01 (0.24) and ≤ 0 for C/L ≥ 0.02.

### Warnings

- **W-EWP-1 — The "Next" line implies the pooled secondary and the live trial can rescue the claim.**
  - Per §9 the pooled Test-A ∪ Test-B H1 result is secondary and never replaces the primary.
  - About two-thirds of its onsets (Test-A) are now unblinded: 43 onsets, BSS +0.013.
  - Its Test-A rows will be re-derived from the 2027 snapshot.
  - It covers H1 only; H4 has no pre-declared route.
  - The live-trial amendment (`ew_v2_live_amendment.md`) **does not exist yet**. It is due before 2026-12-01, and RESEARCH_PLAN §15.2 also requires the live trial to be "registered and running".
  - The live trial scores a different, provisional-input variant (`ew_live_variant_cal_nosubdaily`), so it does not test the frozen Test-A stack.
  - The amendment is being written after Test-A unblinding. It must state that Test-A results were known, and it must not use them to choose sites, hazards or thresholds.
  - Thresholds are never re-tuned (§7). Re-calibrating the alarm threshold to fix the 2× budget overshoot seen on Test-A would be a post-hoc change and must be labelled as such.
- **W-EWP-2 — Wording.**
  - The entry says "pre-registered power rule" and "pre-registered pooled secondary". It never says OSF, and "GitHub commit `cef72b6`, not OSF (addendum A1)" is correct.
  - Addendum A1 prescribes the wording "pre-specified and frozen in a public GitHub commit before testing (not registered on OSF)". Use "pre-specified" in the EW entry.
  - "H-EW" is not defined anywhere. The plan calls the threshold "EW-v2" in RESEARCH_PLAN §15.2.
- **W-EWP-3 — Addendum A1 is local only.**
  - `ew_v2_post_freeze_addendum.md` is untracked, so it is in no public commit. The public cef72b6 protocol still says scoring happens only after OSF registration.
  - The only evidence that A1 predates the pass is a local mtime (12:04:33, 55 s before the wrapper start).
  - It should be committed and pushed with an honest note that it was written at 12:04 on 2026-11-09 and committed after the pass. That is the user's decision; I made no commit.
- **W-EWP-4 — `ew_v2_osf_preregistration.md` is a dead draft.** It is titled "pre-registered test" and cites protocol "v0.2". Mark it as unused or superseded so it is never quoted.

### Observations

- The frozen record still has the placeholders `"created": "[PENDING …]"` and `"git_commit": "[PENDING …]"`. The protocol header still reads "DRAFT v0.3". `frozen_at` and A1 cover these; they are cosmetic and must not be edited, because the files are hashed.
- The chloride consistency check uses a float tolerance (atol 1e-3 + rtol 1e-5). The observed maximum difference is 0.0026 mg/L at ~230 mg/L, which is within tolerance and consistent with float32 storage. The log reports 49 sites checked against 47 with an eligible regression, which is immaterial. Protocol wording ("differs … on any day") is stricter than the code; it is harmless.
- H2 had 9 evaluable onsets, above the a-priori upper scenario of about 7 (§9). This is reported as is, with no consequence.
- The test-role H1 population has 22 basin groups with rows and 150 sites. The protocol's metadata-based projection was 28 groups.
- `git_dirty: true` in the registry `code` block reflects untracked or other-workstream edits. None of the 22 EW scoring files differ from cef72b6.

### Integrity verdict
- Fabrication detected: **No.** Every RESULTS number traces exactly to `primary_endpoint.json`, `controls.csv` or `event_scores.csv`, and the BSS and CIs were recomputed independently.
- Data leakage detected: **No.** There was no test access before the single pass. The disclosed availability conditioning (W10) and chloride-datasheet counts (W11) remain.
- All results traceable to code: **Yes.**
- Statistical claims valid: **Yes for the primary decision** (power rule, Holm with p = 1, verdict). **No for the controls paragraph as written:** the H2 timing claim, missing alarm rates, the circular shift presented as the main null, and missing approximate-CI flags. Fix with the text below.

### Rounding check (RESULTS table vs `primary_endpoint.json`)
| Hazard | Onsets / groups | BSS | CI | p (descriptive) |
|---|---|---|---|---|
| H1 | 43 / 11 ✓ | 0.012713 → +0.013 ✓ | [−0.004095, 0.038256] → [−0.004, +0.038] ✓ | 0.069465 → 0.069 ✓ |
| H2 | 9 / 3 ✓ | −0.014922 → −0.015 ✓ | [−27.503, 0.092874] → [−27.5, +0.093] ✓ | 0.637681 → 0.64 ✓ |
| H4 | 9 / 2 ✓ | 0.018925 → +0.019 ✓ | [0.000101, 0.973369] → [+0.0001, +0.97] ✓ | 0.019990 → 0.020 ✓ |

The controls quoted in the entry are also correct: 0.8605 → 0.86, 0.1721 → 0.17, 0.000999 → 0.001, 0.8304 → 0.83, 0.02178 → 0.022, 0.1661 → 0.17, 0.0 alarm-days, 1.0 / 0.1172 → 0.12, 0.8889 → 0.89 / 0.0636 → 0.06. The process details also check out: 295 rows, 12:05–14:01 EST, and the exact registered `score_command`.

### Checklist summary

| Category | Items checked | Passed | Failed | N/A |
|---|---|---|---|---|
| Single pass / ordering (push → A1 → marker → rows → end; one marker; one claim row; no restart) | 7 | 7 | 0 | 0 |
| Freeze verification (commit = local ×2; code hash; calibration; data pins; artifacts; static inputs) | 7 | 7 | 0 | 0 |
| Data leakage / prior test access (registry, 73 run records, markers, other workstreams) | 5 | 5 | 0 | 0 |
| Decision rule (power rule, Holm p = 1, m = 3, gate = POD set, verdict, no fallback) | 6 | 6 | 0 | 0 |
| Result authenticity / traceability (table, controls, independent BSS and bootstrap) | 6 | 6 | 0 | 0 |
| Reporting fidelity (alarm rates, approximate CI, tiers, control reading, next steps, wording) | 6 | 1 | 4 (C-EWP-1, C-EWP-2 ×2, W-EWP-1) | 1 (OSF wording itself OK) |

### Required replacement text for the RESULTS.md entry

Replace the whole entry (header line through "Next:") with:

```markdown
- **2026-11-09 — EW-v2 Test-A single pass (M1 early warning; registry `ew_v2_test_score`, 295 rows; run `sentinel_runs/sonde/ew_v2_testA_score/`; marker `.ew_scored_test.json`). Post-results review: REVIEW_REPORT_POST_RESULTS.md "EW-v2 Test-A — post-results review (2026-11-09)" (integrity clean).**
  - **Pass:** the lead ran the registered `score_command` exactly once, CPU only, 12:05–14:01 EST. The plan was pre-specified and frozen in a public GitHub commit (`cef72b6`, pushed 10:03 EST) before testing (not registered on OSF; addendum A1).
  - **Primary family (lead 3 d, stack, test-role sites):** no hazard meets the pre-specified power rule (≥ 50 evaluable onsets and ≥ 2 basin groups with onsets), so every hazard enters Holm with p = 1 (m = 3). **No confirmatory claim. The RESEARCH_PLAN §15.2 EW-v2 threshold (≥ 2 hazards pass on Test-A) is not met.** Even with the descriptive p values, Holm would reject nothing (smallest 0.020 > 0.05/3). The H1 count (43) matches the pre-stated projection (≈ 44).

    | Hazard | Evaluable onsets | Basin groups with onsets | BSS vs climatology [95 % CI] | One-sided p (descriptive) |
    |---|---|---|---|---|
    | H1 hypoxia | 43 | 11 | +0.013 [−0.004, +0.038] | 0.069 |
    | H2 chloride | 9 | 3 | −0.015 [−27.5, +0.093]* | 0.64 |
    | H4 nitrate | 9 | 2 | +0.019 [+0.0001, +0.97]* | 0.020 |

    \* Fewer than 10 basin groups with onsets: two-stage CI approximate (protocol §7). The H4 interval and p come from a two-group bootstrap and are not evidence of skill.
  - **Descriptive secondaries (stack unless stated, lead 3, test-role).** Frozen budget 1 alarm-day per 30 d. Realised rates are reported as they are.
    - **H1, alarm level.** Realised 1.99 alarm-days per 30 d (budget exceeded; 39 % of sites over budget; 6.7 % of days in alarm).
      - POD 0.86 (37/43); ≥ 24 h warning 0.70.
      - Season-matched placebo POD 0.33. POD_real − POD_placebo +0.53 [0.38, 0.68], cluster bootstrap p 0.0005, 11 basin groups.
      - Climatology at a similar rate (1.92): POD 0.49.
      - Circular-shift surrogate mean POD 0.17 (p 0.001; this null does not control for season). Molchan skill 0.83.
      - 0.0 alarm-days per 30 d at 11 negative-control sites.
    - **H1, probabilistic.** Label permutation p 0.001 (association, not skill over climatology). Diebold–Mariano vs climatology p 0.022 and vs persistence p 0.17; DM is not basin-clustered, while the cluster-robust primary gives p 0.069. Relative economic value > 0 only at C/L 0.01.
    - **H1 tiers.** < 4 mg/L: 56 onsets, 12 basin groups, `event_gbm_mono` BSS −0.002 [−0.020, +0.020]. < 2 mg/L: 34 onsets, 8 groups, BSS +0.008 [−0.010, +0.045].
    - **H2.** POD 1.0 (9/9) at 4.54 alarm-days per 30 d (4.5× budget). Climatology alone: 8/9 at 5.41. No timing information beyond season.
    - **H4.** POD 0.89 (8/9) at 0.59 alarm-days per 30 d vs climatology 0.33 (3/9). 9 onsets, 6 sites, 2 basin groups.
    - **H2-acute.** BSS +0.25 is not skill: label-permutation p 0.999.
  - **Reading:**
    - For H1, alarms precede real onsets far more often than season-matched placebo dates and than climatology alarms at a similar rate. That is evidence of short-range timing information. It was obtained at about twice the frozen false-alarm budget and is descriptive only.
    - Probabilistic skill over climatology is small (BSS +0.013) and not established. It is null in the one adequately sized cell (H1 < 4 mg/L tier).
    - H2 and H4 are too small to interpret, and H2's alarms add nothing beyond seasonality.
  - **Next (secondary only; cannot change this verdict):**
    - The pre-specified pooled Test-A ∪ Test-B H1 analysis (pass `test_ab`, not before 2027-06-30) is labelled secondary in every report. Its Test-A part is already unblinded.
    - The live-trial amendment (provisional-input variant; H2's confirmatory route) must be frozen in a dated public commit before 2026-12-01. It must state that Test-A results were known and must not re-tune thresholds from them.
```

## M6 HydroMet — post-results review (2026-11-09)

**Review mode:** post-results · **Date:** 2026-11-09 · **Reviewer:** independent reviewer, Autonomous Review Skill (fresh context; did not do the work)
**Scope:** the single test looks `M6/test` and `M6N/test`, plus the pre-declared descriptive re-score `M6N/test_sens427`. This covers:
- the look dirs `sentinel_runs/hydromet/test_looks/{M6_test,M6N_test}/`: markers, sealed `predictions/` and `SHA256SUMS`, `score/`, `log.txt`;
- the logs `sentinel_runs/hydromet/logs/test_look_{M6,M6N}_test_20261109.log`;
- the registry experiments `M6/test` (1,785 rows), `M6N/test` (117) and `M6N/test_sens427` (116);
- the write-ups: `docs/preregistration/hydromet_test_results.md`, the RESULTS.md entry "2026-11-09 — M6 HydroMet TEST", `docs/model_cards/hydromet.md` and `docs/handoff/M6-hydromet.md`.

I also read:
- `FREEZE_hydromet.json` and the archive `freeze_archives/hydromet_freeze_20261109T161338.tar.gz`;
- the registry row `hydromet/prereg_snapshot`;
- `hydromet_protocol.md` §2, §5–§8 and §10–§12;
- RESEARCH_PLAN §15.1–§15.2 and §15.5;
- the code `scripts/eval/hydromet_test_look.py`, `sentinel/models/hydromet/{testlook,guard,evaluate,pipeline}.py` and `scripts/eval/hydromet_track_n.py` (score path).

**What I did**
- **Hashes.**
  - Extracted the archive into scratch. All **46/46** `frozen_files` are byte-identical across three copies: archive, working tree and the freeze record.
  - `FREEZE_hydromet.json` (sha `dd8b4376…`) and the candidate record (`d0de5033…` = `candidate_sha256`) also match the archive. The archive sha is `138dae5e…`, as stated in the handoff.
  - `guard.scoring_code_sha256()` now returns `5bd7d329…`, which equals the freeze.
  - `testlook.verify_data_pins` re-run now: all 5 `M6/test` pins and all 3 `M6N/test` pins match.
  - All 25 frozen artifacts re-hashed: they equal `weights_sha256`.
  - All 37 sealed npz headers carry `artifact_sha256 == weights_sha256[model]`, `freeze_sha256 = dd8b4376…` and `block = test`.
- **Sealing.** `sha256sum -c SHA256SUMS`: 25/25 (`M6_test`) and 12/12 (`M6N_test`) OK. The sha of each `SHA256SUMS` equals `PREDICTIONS_SAVED.json`. Neither folder holds a file outside `SHA256SUMS`, and all npz files are 0444.
- **Earlier test access.** I parsed every M6/hydromet registry row and every hydromet `data_info.json` (88) under `sentinel_runs`, plus every track-N log line that reports station roles.
- **Independent recompute** (CPU, ≤ 6 threads; scripts in `/tmp/claude-1016/-home-bcheng-STOCKHOLM/dbc009e4-477d-4528-910a-e89e964c5c0f/scratchpad/m6rev/scripts/`):
  - **Per-site metrics, `M6/test`** (`recompute_m6.py`). From the sealed npz plus raw `sonde_daily` (approved values only, all-model common mask, ≥ 60 days), with my own seed ensembles and blends, I recomputed per-site KGE and RMSE of all 36 models for `temp_c_mean` and `do_mgl_mean`. Result: 333 / 178 sites, identical n, max |ΔKGE| 4e-7 against `site_metrics_test_test.parquet`.
  - **Per-station metrics, `M6N/test`** (`recompute_m6n.py`). From the sealed npz plus `obs_daily.parquet` (≥ 8 days): 3,014 / 2,724 stations, identical n, max |ΔRMSE| 1e-6.
  - **Statistics** (`stats_m6.py`). My own basin-group bootstrap (different RNG), my own Holm and my own every-seed check, over all 28 + 4 + 4 hypotheses. Every Δ agrees exactly. CIs agree to ≤ 0.009 (bootstrap noise) and p to ≤ 0.12 at p > 0.3. Holm decisions, every-seed flags and `won` are identical in all three families.
- **Not done:** I did not re-run a look, create or modify a marker, edit a frozen file, open `keys.env` or use git. The only file I edited is this report.

### Summary

**Overall status: 🟡 PASS WITH CONCERNS.**
- Both looks are clean, single, verified before the claim, and fully traceable.
- The decision rule was applied exactly as frozen, and the label **"competitive"** (H-G1 not met for M6) is correct.
- Four write-up statements must be corrected before any number leaves the repository:
  - the track-S low-DO sentence (W1);
  - the scope of the blend's DO result (W2);
  - the track-N temperature label (W3);
  - the sensitivity wording (W4).
- Several protocol-required items are missing (W5–W6), and the model card and handoff contain stale or misleading text (W7–W8). None of these changes a number or the label.

**Integrity: clean.**
- **Order of events (UTC):**

  | Time | Event |
  |---|---|
  | 21:13:38 | Freeze `frozen_at` (file and archive mtime 16:13 EST) |
  | 21:13:48 | `prereg_snapshot` row (48 files) |
  | 21:13:55 | Wrapper start |
  | 21:14:09 | `M6/test` `TEST_LOOK.json` and `look_claimed` row |
  | 21:21:32 | Predictions sealed |
  | 21:32:45 | `SCORED.json` and last `M6/test` row |
  | 21:33:07 | `M6N/test` claimed |
  | 22:15:20 | `M6N/test` sealed |
  | 22:18:26 | `M6N/test` rows |
  | 22:19:57–22:20:03 | `test_sens427` rows |
  | 22:20:03 | `SCORED.json` |

- The driver runs `load_freeze` (split and code hash) and `verify_data_pins` before `claim`. `predict_artifacts` calls `guard.verify_weights` per artifact. The scorer refuses files whose `artifact_sha256` or `freeze_sha256` differ. These checks raise on failure and are not logged on success (O2). My re-verification above confirms that they hold now.
- **Counts:**
  - One claim row per look.
  - `M6/test`: 1 claim + 1,784 rows = 1,785.
  - `M6N/test`: 1 + 116 = 117.
  - `M6N/test_sens427`: 116, with no claim row (same look).
  - No `look_aborted` or `look_resumed` row, no `ABORTED*`, no `log_resume*`, no `log_rescore*`. One start/end per log, `exit 0`, and `log.txt` equals the wrapper log minus its 2–3 wrapper lines.
- **No earlier test access:**
  - No M6 registry row with a test block or test role exists before 21:14:09Z.
  - The only hydromet `data_info.json` files with a non-empty `access.protected` are the five under `test_looks/M6_test/`, all with `look: M6/test`.
  - Every other track-N log reports `{'val': 1365}`, `{'train': 6244}` or train+val only.
  - The four val rehearsals (`full_logs/rehearsal_*`) used val-role sites or stations × the val block.
- Comparators in the freeze equal `score_val_v4/comparators.json` (mtime 05:00 EST, val selection).
- The sens427 exclusion set (`basins_trackn.parquet`, mtime 06:14 EST, before the freeze) has 427 stations with method `local (NLDI basin unavailable)`. 87 of them are test-role, as logged. After the ≥ 8-day rule, 62 scored stations per endpoint are removed (3,014→2,952 and 2,724→2,662).

**Decision rule: applied exactly as frozen.**
- **`M6/test`:**
  - 28 hypotheses (7 models × 4 endpoints, all `tested`, 22–33 groups, all ≥ `MIN_TEST_GROUPS = 3`).
  - p_IUT = max of the two two-sided group-bootstrap p values. Holm over 28 at α 0.05: the smallest p_IUT is 0.029 (> 0.0018), so 0 rejections.
  - Every-seed vs GBM holds for no hypothesis.
  - `won` = 0, and `label_model_won` = [F, F, F, F] for `ealstm_ws_full`.
- **§15.1 for `ealstm_ws_full`:** all four Δgbm CIs include 0 (−0.006 [−0.020, 0.003], +0.004 [−0.007, 0.018], −0.000 [−0.020, 0.035], +0.013 [−0.018, 0.046]). "Competitive" is therefore the correct label, not "negative". SOTA is excluded by (ii)–(iii) and by the pre-declared published-method caveat (§8).
- **`M6N/test`:**
  - Holm over 4. Sorted p_IUT: 0.000 (`ealstm_ws_full` temp; rejects at 0.0125), then 0.026 (blend temp; exceeds 0.0167, so Holm stops), 0.038, 0.476.
  - `won` needs both CI_high < 0, every seed better and a Holm rejection, so 0 are won.
  - Sensitivity: same decisions.
- **Units and signs are correct throughout:**
  - KGE Δ = model − comparator, positive = better.
  - Track-N RMSE Δ = model − comparator, negative = better, in °C and mg/L.
  - The write-ups label "(worse)" correctly.

**Traceability: every number checked matches the registry and the sealed outputs.** I checked:
- all of hydromet_test_results.md §1–§3: tables, per-seed Δ, RMSE and KGE medians, the Holm walk-through, the sensitivity numbers;
- the RESULTS entry;
- the handoff's test numbers.

The one mismatch is O1 (a comparator mislabel; the number itself is correct).

### Warnings (should fix before any number leaves the repository)

**W1 — The track-S low-DO statement is mis-scoped in all four documents.**
- `M6S/val_v2` is a **validation** analysis of the cycle 1–4 models only: `ealstm`, `physdo`, `full`, `compact`, `gbmblend`, plus GBM and physics.
- The frozen label model `ealstm_ws_full` (and `ealstm_ws`) was **never evaluated** for low-DO detection, and nothing was evaluated on test (protocol §11 lists the val cell and track R only).
- The registered pooled CSI values for DO-min < 5 mg/L are 0.000–0.033, and the baselines are in that range too (`forcing_gbm` 0.033).
- "No M6 model detects low DO (CSI ≤ 0.03)" and "the val-era finding stands" therefore over-generalise in two ways: across models, and from val to test. The failure must still be stated, but scoped. Replacement text is given below.

**W2 — The blend's track-N DO result needs scoping (risk of overclaim).**
- The frozen DO-mean GBM comparator is `forcing_gbm` (1.351 mg/L). The blend's own GBM member `forcing_gbm_bag` has a lower median RMSE on this cell (1.336, registered).
- My descriptive paired group bootstrap of blend vs `forcing_gbm_bag` (not a frozen comparison) gives −0.020 [−0.037, +0.017] mg/L, with 2 of 3 seeds worse (+0.015, −0.012, +0.008).
- The every-seed "yes" vs `forcing_gbm` rests on a seed-0 margin of −0.0006 mg/L, and it fails in the sens427 re-score (seed 0: +0.002).
- "The GBM blend is better than the GBM on DO" must therefore read "lower median RMSE than the frozen GBM comparator `forcing_gbm`, not Holm-significant, not robust to the per-seed rule in the sensitivity re-score, and not lower than the bagged GBM it contains".

**W3 — The track-N temperature result is a §15.1 "negative" and should be named as such.**
- `ealstm_ws_full` is worse than the GBM with CI excluding 0: +0.161 [0.095, 0.246] °C median RMSE; every seed worse (+0.19, +0.26, +0.21).
- The write-ups state this as "worse" (good), but the track-N headline needs the label, because it is the only Holm "rejection" in either family.
- The rejection exists only because the frozen IUT uses **two-sided** p values, so a significant wrong-direction difference enters Holm.
  - In the registry, the rows of this hypothesis carry `holm_reject: true` in both `M6N/test` and `M6N/test_sens427`. Any downstream consumer (paper, console, fusion) must not read `holm_reject` without the sign of Δ.
  - Using two-sided p also relaxed the next Holm threshold (α/3 instead of α/4). This changed no decision: with one-sided p, blend DO ≈ 0.019 > 0.0125, and `M6/test` has no rejection either way.
  - As frozen, this is valid. Disclose it.

**W4 — "Unchanged" or "changes little" for sens427 is slightly too strong.** Every Δ and CI moves ≤ 0.005, and the Holm decisions are identical. But the blend's DO every-seed rule flips from met to not met. State this.

**W5 — Missing items the protocol requires next to the result:**
- **Configurations tried.** Protocol §8 ("both counts are reported") and RESEARCH_PLAN §15.5(2) require the counts next to the final claim: 6 val improvement cycles, 8 val-scored EA-LSTM variants plus the capped `ealstm_v3`, the EA-LSTM inner grid, and the GBM 3-point grid × 5 variants. They appear only in the val_v4 line.
- **Track-N power statement.** Protocol §12 says it is "stated with every track-N result". The test-cell group count (38 / 36) is in the table, but the pre-declared power framing is not.
- **Track-N secondaries (§12):**
  - per-station bias (registered; not reported);
  - new-region stations alone (not produced by the look).
  - For the lead's information, descriptive figures computed from the sealed `site_metrics_test.parquet` (unregistered; 4 basin groups, so not interpretable as evidence):
    - temperature, 921 stations: `ealstm_ws_full` 1.97 vs `forcing_gbm_bag` 1.85 °C;
    - DO, 810 stations: 1.32 vs `forcing_gbm` 1.40 mg/L.
  - Sonde-region stations alone: temperature 2.01 vs 1.84 °C; DO 1.35 vs 1.33 mg/L.
- **§7 secondaries on test.** They are registered but not summarised:
  - physics-DO head vs EA-LSTM: +0.009 [−0.022, 0.027] and −0.003 [−0.024, 0.012];
  - `ealstm_ws` vs `ealstm` (the upstream-forcing lever): −0.001 / +0.001 / −0.003 / −0.015, all CIs include 0;
  - site-resampling sensitivity (`*_site` columns);
  - secondary targets.

  The pre-registered static-input ablation (W4 of the pre-training review) was **not** carried to test: its artifacts are not in the freeze. Disclose it as val-only.

**W6 — The test write-up should say plainly that the val-era gains did not replicate.**
- Six val cycles selected `ealstm_ws_full` partly because upstream forcings added +0.012/+0.019 KGE on val temperature.
- On test, `ealstm_ws` − `ealstm` ≈ 0 on every endpoint, and the candidate's val Δgbm (+0.002 / −0.002 / −0.016 / +0.011) became −0.006 / +0.004 / −0.000 / +0.013.
- This is the expected selection effect. It belongs next to the label.
- Relatedly, "competitive is largely a power outcome" should not suggest that more power would flip the label. On temperature mean, the candidate's point estimate favours the GBM, and 4 of the 5 other standalone EA-LSTMs are significantly worse than it.

**W7 — The model card is stale and incomplete for a "tested v1.0.0" card:**
- the frozen model `ealstm_ws_full` (and `ealstm_ws`) is missing from the variants list;
- upstream-basin forcings are missing from Inputs, Manifests and the baseline ladder (`forcing_gbm_ws_bag`, the `_ws` physics members);
- "Evaluation cell" describes val only;
- the test path names `hydromet_predict.py` instead of the single-pass `hydromet_test_look.py`;
- "Registry" lists only `val_v1`/`val_v2`;
- the low-DO limitation has the W1 scoping problem.

**W8 — Handoff:**
- "The upstream-forcing lever helped the LSTM … yet on test the gap to the GBM stayed inside the CI" implies the lever worked on test. It did not (W6).
- The "Registered" and "Next steps 1" blocks still name `ealstm_ws` as candidate and the freeze as pending.
- The low-DO line has the W1 problem.

### Observations
- **O1.** hydromet_test_results.md §1, "DO mean: 1.00 vs 0.97 mg/L": 0.97 is `forcing_gbm_bag` (0.968). The frozen DO-mean GBM comparator is `forcing_gbm` (0.980). Say "1.00 vs 0.98 mg/L (`forcing_gbm`)".
- **O2.** Freeze, pin and weight verification is not logged on success. The evidence is the marker's freeze and code sha, the claim's position after the checks in the driver, and my re-verification. For future looks, log one line per check (this needs code outside the 46 hashed files, or a later freeze).
- **O3.** `basins_trackn.parquet`, which defines the sens427 exclusion set, is not covered by a data pin. Its mtime predates the freeze and the counts match the protocol (427 / 87), so this is a documentation gap only.
- **O4.** `hydromet_protocol.md` and `hydromet_freeze_candidate.json` are 0664. The lead may `chmod 0444` the 46 frozen files until the track-R freeze (contents and hash unchanged).
- **O5.** The physics/process family consists of simple regional baselines: air2stream without discharge, transferred by k-nearest donors, and DO_sat × monthly ratio. "Beats physics" should read "beats the pre-registered simple physics/process baselines".
- **O6.** The M6N registry rows do not carry the every-seed flags; they are only in `primary_test.csv`.

### Integrity verdict
- Fabrication detected: **No.** Every number traces to the sealed predictions, and I reproduced it from raw observations plus the sealed npz.
- Data leakage detected: **No.** There was no earlier test access, and protected data were loaded only under the open claim of this freeze.
- All results traceable to code: **Yes.**
- Statistical claims valid: **Yes.** They are as frozen. Two-sided-p caveat: W3. Scoping corrections: W1, W2 and W4.
- Frozen files byte-identical to the archive (needed for the track-R freeze): **Yes, 46/46.** Code hash `5bd7d329…`.

### Exact replacement text

**R1 — RESULTS.md: replace the whole bullet that starts "**2026-11-09 — M6 HydroMet TEST**" with:**

> - **2026-11-09 — M6 HydroMet TEST (single looks under `FREEZE_hydromet.json`; `M6/test` 1,785 rows, `M6N/test` 117, `M6N/test_sens427` 116; write-up `docs/preregistration/hydromet_test_results.md`; post-results review PASS WITH CONCERNS). §15.1 label for `ealstm_ws_full`: COMPETITIVE; M6 H-G1 not met.** Primary (test-role sites × WY2024–25, Holm over 28, 0 won, 0 Holm rejections): Δ median KGE vs best GBM −0.006 [−0.020, 0.003] (temp mean, 33 groups), +0.004 [−0.007, 0.018] (7-DADMax), −0.000 [−0.020, 0.035] (DO mean, 22 groups), +0.013 [−0.018, 0.046] (DO min); vs the pre-registered simple physics/process baselines +0.017 [0.006, 0.033], +0.014 [0.004, 0.031], +0.090 [0.054, 0.142], +0.071 [0.038, 0.104] (all CI > 0); every-seed rule met vs physics on all four, vs GBM on none; p_IUT 0.19 / 0.36 / 0.95 / 0.40. Four other standalone EA-LSTMs are below the GBM on temperature mean with CI excluding 0 (−0.011 to −0.026); the blend is level with the GBM. The val-era gains did not replicate: upstream forcings (`ealstm_ws` − `ealstm`) Δ −0.001 / +0.001 / −0.003 / −0.015 on test (all CIs include 0) after +0.012 / +0.019 on val temperature; selection over 6 val cycles (8 val-scored EA-LSTM variants + `ealstm_v3`; GBM 3-point grid × 5 variants; see the cycle-6 line) applies. **Track N (unsensed WQP stations, test cell 3,014 / 2,724 stations in 38 / 36 basin groups as in the pre-declared power check; Holm over 4, 0 won): `ealstm_ws_full` is WORSE than the GBM on temperature — §15.1 "negative" on track N — median per-station RMSE +0.161 °C [0.095, 0.246], every seed worse; this is the family's only Holm rejection and it arises because the frozen IUT uses two-sided p (wrong direction; registry `holm_reject: true` must not be read without the sign).** It is level with the GBM on DO (−0.012 [−0.063, 0.051] mg/L). `ealstm_gbmblend` has a lower DO median RMSE than the frozen GBM comparator `forcing_gbm` (−0.035 [−0.068, −0.002] mg/L, p_IUT 0.038, not Holm-significant, not won), but not than the bagged GBM it contains (1.316 vs 1.336 mg/L), and its every-seed margin (seed 0: −0.0006 mg/L) fails in the sensitivity re-score; on temperature it is worse than the GBM (+0.047 [0.005, 0.079]). All models beat the basin-forced physics members. Pre-declared sensitivity excluding the 87 test-role local-basin stations (62 scored stations per endpoint): Δ and CIs move ≤ 0.005 and Holm decisions are unchanged (`ealstm_ws_full` temp +0.157 [0.092, 0.240]; blend DO −0.033 [−0.066, −0.003], every-seed rule no longer met). Low-DO detection was not evaluated on test and was never evaluated for `ealstm_ws_full`: on val (`M6S/val_v2`, cycle 1–4 models and baselines) pooled CSI for daily-minimum DO < 5 mg/L was ≤ 0.033 for every model including the GBMs. Not reported on test: the pre-registered no-gauge static ablation (artifacts not frozen; val-only).

**R2 — hydromet_test_results.md.**

(a) In §1, replace "DO mean: 1.00 vs 0.97 mg/L." with:

> DO mean: 1.00 vs 0.98 mg/L (`forcing_gbm`, the frozen DO-mean comparator).

(b) In §1, replace "As pre-declared (§8), the test cell was expected to be underpowered for the val-sized gaps; a "SOTA" label was excluded in advance." with:

> As pre-declared (§8), the test cell was expected to be underpowered for the val-sized gaps, and a "SOTA" label was excluded in advance. More power would not be expected to change the label: on temperature mean, the candidate's point estimate favours the GBM, and four other standalone EA-LSTMs are significantly worse than it. The val-era gains did not replicate. On test, upstream forcings (`ealstm_ws` vs `ealstm`) give Δ −0.001 [−0.009, 0.010] / +0.001 [−0.014, 0.015] / −0.003 [−0.020, 0.014] / −0.015 [−0.031, 0.007], against +0.012 / +0.019 on val temperature. The candidate's val Δgbm (+0.002 / −0.002 / −0.016 / +0.011) became −0.006 / +0.004 / −0.000 / +0.013. The candidate was selected over 6 val cycles: 8 val-scored EA-LSTM variants plus the capped `ealstm_v3`, a 4-point EA-LSTM inner grid, and the GBM 3-point inner grid × 5 variants (RESULTS.md, cycle-6 line). "Physics" here means the pre-registered simple physics/process baselines: an air2stream-type model without discharge, transferred by k-nearest donors, and DO_sat × the monthly ratio.

(c) In §2, insert this paragraph directly before the table:

> Power (pre-declared, protocol §12): the test cell holds 3,014 / 2,724 stations in 38 / 36 basin groups, about 3× the val cell's 12 groups. Track N adds stations, not groups, for selection.

(d) In §2, replace the paragraph that starts "Median per-station RMSE: temperature" with:

> Median per-station RMSE: temperature `ealstm_ws_full` 2.01 °C [1.86, 2.18] vs `forcing_gbm_bag` 1.84 [1.74, 2.01] (KGE 0.812 vs 0.839); DO `ealstm_gbmblend` 1.32 mg/L vs `forcing_gbm` 1.35 (KGE 0.635 vs 0.615). Median per-station bias: temperature `ealstm_ws_full` +0.19 °C [−0.19, 0.42], `forcing_gbm_bag` +0.22 [0.03, 0.34]; DO `ealstm_ws_full` −0.31 mg/L [−0.50, −0.08], `ealstm_gbmblend` −0.37, `forcing_gbm` −0.37.
>
> **Track-N temperature is a §15.1 "negative" result for `ealstm_ws_full`:** it is worse than the GBM with CI excluding 0, and every seed is worse (+0.19, +0.26, +0.21 °C). This is the only Holm rejection in this family, and it is in the unfavourable direction. It arises because the frozen intersection–union test uses two-sided p values. The registry rows of this hypothesis carry `holm_reject: true` and must not be read without the sign of Δ. Using two-sided p changed no decision.
>
> The blend's DO result is narrow:
> - Its median RMSE is lower than that of the frozen comparator `forcing_gbm` (CI below 0; every seed lower, but seed 0 by only 0.0006 mg/L).
> - p_IUT = 0.038 does not survive Holm. After the first rejection (p < 0.001 ≤ 0.0125), the next p_IUT, 0.026 (blend temperature), exceeds 0.05/3 = 0.0167, so Holm stops. The result is not won.
> - It is not lower than the bagged GBM the blend contains (`forcing_gbm_bag` 1.336 vs blend 1.316 mg/L; not a frozen comparison).
> - Its every-seed margin fails in the sensitivity re-score.
>
> The new-region-only secondary of §12 was not produced by the look. It is not reported here (4 basin groups).

(e) In §2, replace the paragraph that starts "**Pre-declared sensitivity (`M6N/test_sens427`, descriptive):**" with:

> **Pre-declared sensitivity (`M6N/test_sens427`, descriptive).** This re-score excludes the 87 test-role stations whose basin fell back to the local delineation (of the 427 overall), which removes 62 scored stations per endpoint. Every Δ and CI moves by ≤ 0.005, and the Holm decisions are unchanged: `ealstm_ws_full` temp Δgbm +0.157 [0.092, 0.240], DO −0.011 [−0.061, 0.045]; blend temp +0.047 [0.004, 0.077], DO −0.033 [−0.066, −0.003]. One thing does change: the blend's DO every-seed rule vs `forcing_gbm` is no longer met (seed 0: +0.002 mg/L).

(f) In §3, replace bullets 2 and 3 ("Track N: …" and "Unchanged from validation …") with:

> - Track N: at unsensed reaches the candidate is **worse than the GBM on temperature (§15.1 "negative" on track N;** +0.16 °C median RMSE, CI excludes 0, every seed worse) and level on DO. The GBM blend has a lower DO median RMSE than the frozen GBM comparator (−0.035 mg/L, CI excludes 0), but it is not Holm-significant, not robust in the sensitivity re-score, and not lower than its own bagged GBM. 0 of 4 won.
> - Low-DO detection was **not evaluated on test**, and it was never evaluated for `ealstm_ws_full`. On validation (track S, `M6S/val_v2`; cycle 1–4 models and baselines), pooled CSI for daily-minimum DO < 5 mg/L was ≤ 0.033 for every model, including the GBMs: no evaluated model detects low DO.
> - §7 secondaries on test (registered, descriptive):
>   - physics-DO head vs `ealstm`: +0.009 [−0.022, 0.027] (DO mean) and −0.003 [−0.024, 0.012] (DO min);
>   - upstream forcings, `ealstm_ws` vs `ealstm`: all CIs include 0 (see §1).
>
>   The pre-registered no-gauge static ablation was not carried to test, because its artifacts are not in the freeze. It is val-only (`M6/val_v2`).

**R3 — docs/model_cards/hydromet.md.**

(a) In the front matter, replace the first two sentences of `limitations` ("Test result (2026-11-09): … (track S).") with:

> Test result (2026-11-09): "competitive". The model is level with the GBM baseline at unseen basin groups and better than the pre-registered simple physics/process baselines. At unsensed WQP stations (track N) it is worse than the GBM on temperature (a §15.1 "negative" on that track) and level on DO. Val-era gains from upstream forcings did not replicate on test. Low-DO detection was never evaluated for this model: on validation, the earlier variants and the GBM baselines all had CSI ≤ 0.033 for daily-minimum DO < 5 mg/L (track S).

(b) In Model details → Variants, add these bullets:

>   - `ealstm_ws` (v0.10): the same EA-LSTM with 11 upstream-basin-averaged gridMET channels (`ws_*`, NLDI/NHDPlus V2 basin of the StreamCat COMID; `processed/hydromet_upstream_forcing`) appended to the site-cell forcings (`configs/hydromet/lstm_ws.yaml`).
>   - **`ealstm_ws_full` (v0.11; the frozen, tested model):** `ealstm_ws` refitted on all train-role sites for each seed's inner-selected epoch count (`configs/hydromet/lstm_ws_full.yaml`).

(c) In Baseline ladder, add this bullet:

>   - Upstream-forcing parity: `forcing_gbm_ws_bag` (3-seed bag), `air2stream_regional_ws`, `air2water_{linear,mohseni}_ws` and `dosat_ratio[air2stream_regional]_ws`, driven by basin-mean forcings.

(d) In Inputs, add this bullet:

> - From v0.10: basin-mean gridMET forcings over the site's total upstream drainage basin (observed weather only).

(e) Replace "**Manifests:** `sonde_daily`, `forcings_daily`, `streamcat`." with:

> **Manifests:** `sonde_daily` (pinned to `sonde_daily_refresh_20261108`), `forcings_daily`, `hydromet_upstream_forcing`, `streamcat`, `sites`; for track N: `hydromet_wqp_national` and `hydromet_upstream_forcing_trackn` (data pins in `FREEZE_hydromet.json`).

(f) Replace the "Evaluation cell" bullet with:

> **Evaluation cells:**
> - Selection: val-role sites (unseen basin groups) × WY2022–2023, with a 42-day embargo, approved values only, identical cells for every model.
> - Test (single look, 2026-11-09): test-role sites × WY2024–2025 (`M6/test`), and test-role unsensed WQP stations × WY2024–2025 (`M6N/test`).

(g) Replace the "Test-role sites and test blocks" bullet with:

> **Test-role sites and test blocks:** only through the single-pass guard `scripts/eval/hydromet_test_look.py` (`sentinel/models/hydromet/testlook.py`). It verifies the freeze, code hash and data pins before an O_EXCL claim, seals the predictions, and scores each look once. Both Test-A looks are spent.

(h) Replace the "**Registry:**" bullet with:

> **Registry:**
> - Selection (val, selection-affected): `M6/val_v1`–`v4`, `M6S/val_v2`, `M6N/val_v1`, `M6N/inregion_val_v1`, `M6N/val_v2`.
> - Test: `M6/test`, `M6N/test`, `M6N/test_sens427`.
> - Configurations tried: `$SENTINEL_RUNS/hydromet/configs_tried.jsonl`.

(i) In Limitations, replace the bullet "Low-DO events are not detected by any M6 model (track S); warm extremes are under-predicted (cold bias)." with:

> - Low-DO detection has not been evaluated for this model or on test. On validation, every evaluated model (earlier EA-LSTM variants and the GBMs) had CSI ≤ 0.033 for daily-minimum DO < 5 mg/L (track S). On validation, warm extremes are under-predicted (cold bias).

(j) In Limitations, replace the bullet about the GBM blend with:

> - At unsensed WQP stations (track N) the frozen model is worse than the GBM on water temperature and level on DO. The GBM blend had a lower DO RMSE than the frozen GBM comparator (CI excludes 0). That gain was not Holm-significant, not robust to the per-seed rule in the sensitivity re-score, and not present against the bagged GBM the blend contains.

**R4 — docs/handoff/M6-hydromet.md.**

(a) In §1, replace the sentence that starts "The upstream-forcing lever helped the LSTM" with:

> The upstream-forcing lever helped the LSTM on val (+0.012–0.019 KGE on temperature) but did **not** replicate on test (`ealstm_ws` − `ealstm` −0.001 / +0.001 / −0.003 / −0.015, all CIs include 0). Treat val-cycle gains of this size as selection noise when ranking levers.

(b) In the DO bullet, replace "no model detects DO-min < 5 mg/L (CSI ≤ 0.03, track S)" with:

> on val no evaluated model (cycle 1–4 EA-LSTMs and the GBMs; `ealstm_ws_full` was never evaluated) detects DO-min < 5 mg/L (CSI ≤ 0.033, `M6S/val_v2`)

(c) In "Registered", replace "Freeze candidate (protocol §8): `ealstm_ws` … Re-evaluate after cycle 6 (`ealstm_ws_full`, val_v4)." with:

> Frozen and tested: `ealstm_ws_full` (cycle 6, `M6/val_v4`); see TEST RESULTS above.

(d) Mark "Next steps" item 1 as done: "Done: FREEZE_hydromet.json written 2026-11-09 21:13Z; `M6/test` and `M6N/test` scored once."

### Checklist summary

| Category | Items checked | Passed | Failed | N/A |
|---|---|---|---|---|
| Single-look integrity (claims, seals, markers, registry, timing) | 9 | 9 | 0 | 0 |
| Freeze verification (code hash, 46 files, pins, weights, npz provenance) | 6 | 6 | 0 | 0 |
| Data leakage / earlier test access | 4 | 4 | 0 | 0 |
| Decision rule (Holm 28/4, IUT, every-seed, MIN_TEST_GROUPS, label model, §15.1) | 7 | 7 | 0 | 0 |
| Result authenticity (independent recompute of metrics and statistics) | 5 | 5 | 0 | 0 |
| Fair reporting / completeness | 9 | 4 | 5 (W1–W5) | 0 |
| Documentation currency (card, handoff) | 2 | 0 | 2 (W7–W8) | 0 |

---

## M7 Lite — post-results review (2026-11-09)

**Review mode:** post-results · **Date:** 2026-11-09 · **Reviewer:** independent reviewer, Autonomous Review Skill.
It was read-only on every test output, ran nothing again, edited no frozen file and did not use git.

**Scope:** the single look `lite/test`, original 4-state track:
- `FREEZE_lite.json`, the archive `freeze_archives/lite_freeze_20261109T170530.tar.gz` and the registry row
  `lite/prereg_snapshot`;
- the markers, `predictions/` and `test_results.json` in `sentinel_runs/lite/test_look/`;
- the log `lite/logs/lite_test_look_20261109.log`;
- the `lite_test*` registry rows;
- the write-ups: `docs/preregistration/lite_test_results.md`, the RESULTS.md entry, the "Test results" section of
  `docs/model_cards/lite.md`, and the handoff;
- protocol v1.7 §3, §7, §8a and §13, against which they were checked.

### What I did
- **Freeze and code.**
  - `FREEZE_lite.json` (sha `a066b57a…`, 0444, mtime 22:05:18Z) is status FROZEN, with `authorised_by/at`.
  - It differs from the reviewed candidate (`415ad36f…` = `candidate_sha256`) only in `status`, `authorised_*`,
    `scoring_code_sha256`, `candidate_sha256` and `recovery_note_F1`. The F1 note is the text my E-patch check
    proposed.
  - The archive sha `c5bf623f…` matches its `.sha256`. All 82 files in it are byte-identical to the working tree: the
    80 hashed files plus the freeze and the candidate.
  - `guard.scoring_code_sha256()` now = `a7a8be82…` (80 files) = the freeze. **No hashed file has changed since the
    freeze.**
  - `verify_data_pins` re-run now: 7/7 match.
- **One claim, one seal, one score, in order.**
  - `TEST_LOOK.json` is dated 22:05:53Z (pid 1560213, host comet). The registry has exactly one `lite_test_claim`
    (22:05:53Z), one `lite_test_sealed` (22:24:03Z), no `lite_test_aborted`, and one flush of scoring rows at
    22:30:44–46Z:

    | Experiment | Rows |
    |---|---|
    | `lite_test` | 238 |
    | `lite_test_h1` | 12 |
    | `lite_test_h1_deployable` | 12 |
    | `lite_test_h2` | 6 |
    | `lite_test_events` | 36 |
    | `lite_test_verdicts` | 1 |

  - `SCORED.json` is dated 22:30:46Z.
  - The pre-claim checks ran: the log starts at 22:05:35Z, and the claim is 18 s later, which is the time hashing
    takes. The code order is fixed by the frozen hash.
  - `sha256sum` of the 3 sealed files equals `SHA256SUMS`. The sha of `SHA256SUMS` (`d4559783…`) equals both
    `PREDICTIONS_SAVED.json` and the `lite_test_sealed` registry anchor. Both name freeze `a066b57a…`. There are no
    unlisted files.
- **No earlier test access.**
  - No M7 registry row has a test split before 22:05Z.
  - The only test chip caches were created *after* the claim: `eval_test_*` at 17:05:58 EST = 22:05:58Z and
    `eval_test_time_*` at 22:17:17Z.
  - The log shows exactly one pass: 2 roles × the model list, then scoring, exit 0, ending 22:30:47Z.
- **Decision logic re-derived from `test_results.json` against §8a.**
  - H1 (Holm-6): every one-sided p ≥ 0.32, so Holm rejects nothing and no `reject` is true. The composite is false.
  - H1-deployable: the same.
  - H2: every p ≥ 0.74, and per-seed Secchi is −0.12 / +2.00 / −3.34, so the every-seed rule fails. False.
  - H-G6 = budgets (true) ∧ H1-dep (false) ∧ H2 (false) ∧ gap (true) = **false**.
  - The §15.1 label rule gives "negative" because a baseline is significantly better on Secchi: carrier vs GBM-RGBN
    +10.8 [+3.1, +15.3]; vs GBM+context +15.4 [+7.1, +20.3].
  - The families are confirmatory for `test` only. All 15 Test-B family rows are registered as
    `confirmatory: false`, and 0 rows have `reject: true`.
  - E4a: there is no decision key in the scorer output or the registry. The 36 event rows carry no
    confirmatory/preregistered field.
- **Traceability.**
  - All 238 `lite_test` rows equal `test_results.json` (value and both CI ends; 0 mismatches). All 30 family rows
    equal their comparisons.
  - Independently, from `pred_test.parquet` with my own MdSA, 100·(exp(median |ln p/y|) − 1), I reproduced to 3 dp:
    the ensemble, seed 1, INT8, no-KD, band-ratio, GBM, GBM+context and train-median MdSA for all three targets.
    Examples: ensemble Secchi 54.446, GBM 44.233.
  - My own water-body bootstrap (B = 2000, different RNG) agrees with the frozen code:

    | Comparison | Mine | Frozen code |
    |---|---|---|
    | Ensemble − GBM, Secchi | +10.21 [+4.71, +14.44] | [+4.45, +14.21] |
    | H2 Secchi | +1.55 [−1.78, +3.62] | [−1.74, +3.50] |
    | Turbidity vs GBM | +38.6 [−9.5, +82.9] | [−9.2, +79.7] |

    Groups: 387 / 44.
  - I checked every number in `lite_test_results.md`, the RESULTS.md entry and the model-card test section against
    `test_results.json`. All the scores, CIs, Δs, robustness values, red-edge values, slices, ablations, E4a values and
    Test-B values match, except the omissions and wording issues listed below.

### Findings
No critical issue. The negative result is reported plainly in all four documents: H-G6 FALSE, label "negative",
Secchi +10.2 [+4.4, +14.2], H2 not replicated and glint failing. E4a is called descriptive everywhere, and the stale
`"confirmatory": true` is explained in the write-up, RESULTS.md and the handoff. Test-B is kept descriptive.

The corrections below are reporting completeness and wording. Each gives exact replacement text, with numbers from
`test_results.json`.

- **🟡 P1. The future-year slice for turbidity and TSS rests on 8 and 3 chips but is printed like the others.**
  `test ∩ NAIP 2022–23` has Secchi 148 chips / 98 water bodies, turbidity **8 / 6** and TSS **3 / 2**. The scorer's
  ≥ 10 guard applies to the whole slice, not per target. In `lite_test_results.md` §"Secondary slices", replace
  the two rows
  `| Turbidity, future years (small n) | 110.0 | 102.1 | 42.8 | 14.1 |` and
  `| TSS, future years (small n) | 32.8 | 39.8 | 25.4 | 71.5 |`
  with:
  `| Turbidity, future years (n = 8 chips / 6 water bodies; not interpretable) | 110.0 | 102.1 | 42.8 | 14.1 |`
  `| TSS, future years (n = 3 chips / 2 water bodies; not interpretable) | 32.8 | 39.8 | 25.4 | 71.5 |`
  and change the Secchi future-years label to `Secchi, future years (Test-A ∩ NAIP 2022–23; n = 148 / 98)`.
- **🟡 P2. The H2 per-seed values for turbidity and TSS are marked "not reported", but they were computed.** In the
  H2 table, replace `| Turbidity | +12.3 [−15.5, +52.6] | not reported |` with
  `| Turbidity | +12.3 [−15.5, +52.6] | +40.1 / +18.4 / −10.5 |`, and `| TSS | +7.0 [−8.6, +26.7] | not reported |`
  with `| TSS | +7.0 [−8.6, +26.7] | +8.0 / +6.3 / −3.0 |`.
- **🟡 P3. The "every applicable baseline vs the carrier" table is incomplete.** Its "—" cells were computed, and TSS
  is missing. Replace the table with:

  | Target | GBM-RGBN + context | Location kNN | Location/season GBM | Train median |
  |---|---|---|---|---|
  | Secchi | +15.4 [+7.1, +20.3] | +5.0 [−4.9, +13.5] | +0.7 [−10.5, +9.6] | −11.7 [−26.1, −5.5] |
  | Turbidity | +36.2 [+3.0, +101.5] | −57.9 [−124.2, +42.1] | −29.6 [−99.1, +53.1] | −33.4 [−62.5, +56.2] |
  | TSS | +14.1 [−12.8, +66.8] | −5.3 [−95.9, +59.7] | −36.8 [−72.6, +14.4] | −36.2 [−93.3, +14.0] |
- **🟡 P4. Secondaries required by the protocol are missing from the write-up.** All of them are in
  `test_results.json`.
  - §3 / D1 requires lake-only turbidity and TSS as secondary, and results by site class.
  - §1 / §15.3 lists the chl-a trophic-class soft threshold.

  Add after "Test-A scores":

  > **Lake/reservoir-only turbidity and TSS (secondary, deviation D1; turbidity n = 58 / 26 water bodies, TSS
  > 62 / 30).** MdSA [95% CI]:
  > - ensemble: turbidity 124.3 [63.0, 227.6], TSS 69.3 [42.0, 116.7];
  > - INT8: 116.9 [59.3, 231.6], 66.6 [40.1, 111.4];
  > - band-ratio: 121.8 [85.7, 268.8], 69.0 [53.8, 118.2];
  > - GBM-RGBN: 87.2 [63.2, 229.5], 69.5 [45.0, 97.4];
  > - GBM + context: 91.9 [62.7, 196.3], 54.6 [44.9, 99.9].
  >
  > **Secchi, all site classes (secondary):** ensemble 55.8, GBM-RGBN 45.2 (n = 1,094).
  >
  > **By site class (point MdSA):** the non-lake classes are small. Secchi river/stream (n = 21): ensemble 146.3,
  > GBM 124.8. Turbidity estuary/coastal (n = 13): ensemble 155.9, GBM 80.8. Turbidity river/stream (n = 21):
  > ensemble 126.0, GBM 105.6. TSS river/stream (n = 29): ensemble 105.9, GBM 125.6.
  >
  > **Chl-a trophic class (soft threshold §15.3, balanced accuracy, lakes, n = 271).** Ensemble 0.334
  > [0.282, 0.389] vs band-ratio 0.311 [0.267, 0.356]. That is above on the point estimate only; there is no paired
  > test and the CIs overlap. GBM-RGBN is 0.381 [0.319, 0.451] and GBM + context 0.446.
- **🟡 P5. The red-edge wording covers Test-A only and overstates the result.**
  - In §"Red-edge gap", replace "The S2 teacher with red-edge is about 10–13 points better on Secchi than the 4-band
    student on the same chips." with:
    "On Secchi the S2 teacher with red-edge is better than the 4-band student on the same chips by 10–13 points on
    Test-A (n = 321) and 21–22 points on Test-B (n = 247). For TSS the teacher is also better. For turbidity the
    result is mixed (Test-B: the teacher is worse, 177.1 vs 161.6). These are descriptive point values with no CI,
    and turbidity/TSS n on Test-A is small (25 / 19)."
  - In "Interpretation", replace "S2 with red-edge is clearly more informative than NAIP RGBN on the same water."
    with "S2 with red-edge is more informative than NAIP RGBN for Secchi on the same water (descriptive, no CI). For
    turbidity and TSS the comparison is small-n and mixed."
- **🟡 P6. "CI excludes 0, in every seed" implies per-seed CIs that do not exist.** In "Interpretation", replace
  "GBM-RGBN is better by about 10 MdSA points on Secchi (CI excludes 0, in every seed)." with "GBM-RGBN is better by
  about 10 MdSA points on Secchi. The ensemble CI excludes 0, and the point difference favours GBM in each of the 3
  seeds (+8.9 / +10.0 / +7.1)."
- **🟡 P7. The model card is stale.**
  - In `docs/model_cards/lite.md`, line 10, replace "`docs/preregistration/lite_protocol.md` v1.3 (deviation D1 and
    the W8 teacher result in §3/§5/§11; national track §12)" with "`docs/preregistration/lite_protocol.md` v1.7
    (frozen 2026-11-09; deviation D1, the W8 teacher result, §8a decision rules, §13 E4a descriptive-only; national
    track §12)".
  - In its source list, "experiments `lite_val`, `lite_val_compare`, `lite_export`, `lite_robustness_val`,
    `lite_val_national`", add "`lite_test`, `lite_test_h1`, `lite_test_h1_deployable`, `lite_test_h2`,
    `lite_test_events`, `lite_test_verdicts`".

### Observations
- 🟢 The Test-B Δ table shows comparisons that favour the student, such as vs band-ratio on turbidity, but not the
  ensemble vs GBM-RGBN on turbidity or TSS, where GBM is better (116.3 / 54.5 vs 138.8 / 98.4 in the scores table).
  Suggested addition under the Test-B Δ table: "Against GBM-RGBN, the ensemble is worse on Test-B turbidity and TSS
  (scores above); no Test-B Δ carries a claim."
- 🟢 The W8 sentence "no sign that teacher proximity drove anything" is fair but weak evidence, because H2 is null
  overall. Better: "H2 restricted to water bodies with no teacher row within 5 km is also null (Secchi +0.3
  [−4.0, +3.0])."
- 🟢 The "Val → test" attribution to selection optimism and distribution shift is phrased as "consistent with", which
  is acceptable.
- 🟢 The frozen code computes the future-year slice per target without a per-target n floor. That cannot be changed
  now; P1 handles it in reporting. For the national freeze, add a per-target n ≥ 10 rule.

### Integrity verdict
- Fabrication detected: **No.** Every number checked traces to `test_results.json` and the registry. Key MdSA values
  and paired CIs were reproduced independently from the sealed predictions.
- Data leakage detected: **No.**
  - There was no test access before the claim.
  - The teacher overlap is 0 within 2 km for Test-A and Test-B, and the sensitivity slice is reported.
  - The code, data pins and artifacts are unchanged since the freeze.
- All results traceable to code: **Yes.** The single scorer at code hash `a7a8be82…` produced everything: one claim,
  one seal (with a registry anchor), one score.
- Statistical claims valid: **Yes.** The §8a decision rules were applied exactly. The negative/null conclusions are
  correct and plainly stated. P1 (small-n slices) and P5 (red-edge wording) need the corrections above before the
  numbers are reused.

**Verdict: 🟡 PASS WITH CONCERNS.**
- The single look is clean and valid. The headline is final: H-G6 FALSE, label "negative", H2 not replicated, glint
  fails.
- Apply P1–P7 (text-only edits to non-frozen write-ups; no re-run) before the M7 numbers enter the paper or the
  Console.
- No frozen file may be edited. A second look at Test-A is not allowed (§15.5 item 1).
