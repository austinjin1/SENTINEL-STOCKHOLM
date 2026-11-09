# SENTINEL-Fuse — pre-registration protocol (Goal 2)

**Status:** v1.1, 2026-11-08 (decision rules, hard strata, statistics and comparator amendments —
§6, §8, §8.1–8.3, §10, §11 — all written before any SENTINEL-Fuse model result existed and before any test
row was scored; the only outcomes seen were Stage-0 *comparator* dev-CV results on the v1 table).
v1.0, 2026-10-25. Written, and split files frozen (sha256 below), **before any Fuse model or
comparator was fitted**. Stage 0–1 work in this version uses development (validation) folds only. No
test row has been loaded by any modelling code; test scoring happens once per frozen configuration,
after the scoring script and every threshold are hashed into §10 (R6, plan §15.5 amendment).
**Implements:** RESEARCH_PLAN.md §3, §6, §14, §15.1, §15.2 (H-G2), §15.5; docs/research/05 §e.
**Code:** `sentinel/data/fuse_tiers.py`, `fuse_tier_c.py`, `fuse_tier_w.py` (tables);
`sentinel/models/fuse/` (`splits`, `data`, `baselines`, `stage0`, `model`, `train`, `plugins`);
`scripts/data/build_fuse_tiers.py`, `scripts/train/fuse_stage0.py`, `scripts/train/fuse_train.py`;
tests `tests/test_fuse.py`. Datasheets: `docs/datasheets/fuse_tier_{d,c,w}.md`.

---

## 1. Questions

1. Does any-subset deep fusion of ≥ 5 modalities predict water-quality variables and disaster-relevant
   events **better than the best single modality and better than GBM on concatenated features**, under
   spatial blocking, with paired CIs excluding 0 (H-G2)?
2. Does every modality contribute (leave-one-modality-out), and does the model degrade gracefully when
   modalities are missing?
3. Are its predictive distributions calibrated per availability pattern and region?

## 2. Data tiers (built 2026-10-25; manifests `fuse_tier_d`, `fuse_tier_c`, `fuse_tier_w`)

| Tier | Unit | Rows | Sites | Use |
|---|---|---|---|---|
| D (NEON) | site × local issue date, 2016-01-01 … 2025-06-30 (RELEASE-2026 only) | 105,117 (v2, 2026-11-08; v1 99,568) | 34 | P1, P3, P4, P5 |
| C (cyanotoxin) | site × local sample date with a microcystin result (WQP WQX 3.0 + EPA NLA 2007–2022) | 73,364 | 12,731 | P2, P4, P5 |
| W (wide) | WQP lab site-day at a USGS sonde site of the M1 split, WY2009–2025 | see datasheet | ≈1,340 | pretraining; nutrient/FIB virtual sensing (secondary) |

### 2.1 Tier D modalities and freshness rules (inputs dated ≤ issue day d; end-of-day issue)

| Modality | Source (NEON product) | Present when | Features |
|---|---|---|---|
| `sonde` | DP1.20288.001 water-quality sonde (DO, SC, pH, turbidity, fDOM, chl-a µg/L), downstream station (HOR 102 → 112; buoy 103) | ≥ 1 variable observed on d−6…d | per variable: last, lastmin/max, lag1, 3/7/30-d means, 7-d SD/min/max, last valid (≤ 30 d), age, 30-d coverage, d7 change — model space (log(x + offset) for SC, turbidity, fDOM, chl-a) |
| `nitrate` | DP1.20033.001 SUNA (NO3-N mg/L) | as sonde | same features |
| `temp` | DP1.20053.001 (streams) / DP1.20264.001 shallowest thermistor (lakes, rivers) | as sonde | same features |
| `hydro` | DP1.20016.001 surface-water elevation / DP4.00130.001 discharge | as sonde | same; elevation from one HOR/VER per site (most valid days) minus its trailing 365-d median (past-only); log(Q + 1 L/s) (built 2026-11-08) |
| `chem` | DP1.20093.001 lab chemistry (26 analytes) | latest event **strictly before d**, age ≤ 35 d | log value (published value, or detection limit when censored) + censored flag per analyte; age |
| `micro` | DP1.20141.002 16S (via `processed('microbe_wq')`) | latest event on or before d, age ≤ 65 d | genus CLR → PCA (32 PCs; prevalence filter, CLR and PCA fit on training rows of each fold only) + log reads/richness; age |
| `macroinv`, `periphyton` | DP1.20120.001 / DP1.20163.001 | latest event ≤ d, age ≤ 120 d | macroinv: log10 density, family richness, Shannon, EPT / Chironomidae / Oligochaeta / non-insect / Gastropoda / Amphipoda+Isopoda fractions; periphyton: benthic and water-column medians of log10 chl-a, log10 C, chl-a fraction, log C:N, N:P, C:chl-a, δ13C, δ15N, δ34S (built 2026-11-08; datasheet) |
| hub `wx` | gridMET (CONUS) / ERA5 (AK, PR) observed analysis, windows ending at d | source covers the site | precip sums 1/3/7/30 d, dry days, Tmax/Tmin means 3/7/30 d, srad/wind/RHmin/VPD 7-d means, PET 7-d sum |
| hub `static` | StreamCat (38 curated metrics) + site registry | always | coordinates, elevation, site type, land cover, nutrients inputs, climate/hydro/soils |
| hub `time` | — | always | day-of-year harmonics |

Same-event rule: a lab target on day d never sees the same day's chemistry (chem is strictly before d);
the same-day 16S sample may be an input (different measurement of the same water sample — this is the
co-located nowcast M3 studies). Operational latency (lab results and sequencing take weeks) is *not*
modelled in the primary analysis; a pre-registered sensitivity analysis shifts `chem` and `micro`
availability by +30 d.

### 2.2 Tier C modalities (same-day unless stated)

`pigment` (lab chl-a, phycocyanin), `nutrient` (TP, TN, TKN, NO3NO2, NH3/NH4, ortho-P, DOC), `physchem`
(temperature, DO, pH, SC, turbidity, Secchi), `satellite` (Sentinel-2 L2A water-masked band medians and
indices; `mc_benchmark_v2` qc_lenient ±7 d and `s2_wqp_matchups` qc_pass), `history` (prior microcystin
results at the site dated ≤ d − 7, 365-d window), NLA-only `profile`, `zoop`, `benthic`, `phyto`; hub `wx`
(nearest SENTINEL gridMET cell ≤ 50 km, windows ending d − 1), `static`, `time`. Other cyanotoxins in
the same sample are never inputs. `phyto` (cyanobacterial biovolume) is causally proximal to toxin
production; P2 is reported with and without it.

Labels: total microcystins (max result per site-day; censored → bound). **P2 primary label =
`y_gt8_total`** (8 µg/L on the total-microcystin basis; site-days whose only results are congener sums are
excluded from it — review W5); `y_gt8` (including congener-sum rows) is a sensitivity analysis. Duplicate
samples reported both to WQP and in NLA (same lake ≤ 1 km, same date) are kept once, NLA version (review W4;
rebuild 2026-11-08 changes row counts slightly; split roles unchanged). Site-days with only congener
results get the congener sum, which is used **only** for the 8 µg/L label (several programmes report
non-detect congeners at their reporting limit without a censor flag; `mc_basis` column). Thresholds:
**8 µg/L (primary; EPA 2019 recreational)**, 1.6 µg/L (v1 continuity), 0.3 µg/L (EPA 10-day drinking-water
health advisory, children).

## 3. Targets ("all water-quality features + disasters")

* Tier D sensor targets: daily means of DO, SC, pH, turbidity, fDOM, chl-a, nitrate, water temperature,
  and daily-min DO at h = 1, 3, 7, 14 d (model space as above).
* Tier D lab targets (same-day nowcast at chemistry events): TN, TP, DOC, NO3+NO2-N, NH4-N, ortho-P, TSS,
  Cl, SO4, Si, Ca — log(x + offset), left-censored likelihood below detection limits.
* Tier D events (EW-v2 definitions in `sentinel.eval.hazards`): H1 hypoxia (DO daily min < 3 mg/L; tier
  < 4 mg/L), H5b chlorophyll bloom (chl-a daily mean > 20 µg/L), as "any event day in (d, d+7]" and EW-v2
  onset labels. H4 nitrate (> 10 mg/L) has no positive at NEON and is not evaluated.
* Tier C: microcystin exceedance at the three thresholds; log10 microcystin (censored).

## 4. Splits (frozen; sha256 of the JSON files)

| File | sha256 | Rule |
|---|---|---|
| `fuse_tier_d_split_v2.json` (**current**) | `20adbe595ed42a46c490da31e16743353ce819e9d84cb8eedffa15303a26610a` | whole NEON domains; first domain (salted sha1 order) with a river, then with a lake, then fill to ≥ 7 test sites (caps 8 / 2 lakes / 1 river). **Test: D03, D07, D13, D14 = BARC, SUGG (lakes), FLNT (river), COMO, LECO, SYCA, WALK, WLOU.** Dev: 26 sites in 15 domains → 5 folds (6/5/5/5/5 sites) |
| `fuse_tier_d_split.json` (v1, superseded) | `b912650b6137becaf793baf880f1b9ff1f40b66fb5aef5101644916915099f9a` | hash-order rule without lake/river minimums gave an all-stream test set; replaced 2026-10-25 06:09 UTC **before any model was fitted** (composition only; no outcome inspected) |
| `fuse_tier_c_split.json` | `e2fd5de9da4079091445c0635f8ccf1febe7c878a7dc7a3beb679ad0b68e3f37` | Test-R = HUC2 {02, 05, 11, 12, 17} (8,134 rows); Test-S = 20-km blocks of the rest (9,790 rows); 552 sites within 5 km of a test site → `buffer` (unused); dev 7,959 sites / 51,157 rows with 5 folds over 20-km blocks (primary), 5-km blocks, and leave-HUC2-out |
| `fuse_tier_wp1_split.json` (W-P1 track, §12) | `b9e0d5c44d8a83c3bc4abc8517d66fb520186fcf6e64db59aa3e4d40e050288a` | M1 roles: train/val → dev (5 basin-group folds), test → test; Test-W-A = test sites × WY2024–2025 (42-d embargo) |
| Tier W | M1's `sonde_split.json` (`fd4bb14f…`) | site roles copied from M1 (train/val/test basin groups); test-role rows never used |

Leave-site-out is implied by the domain holdout (Tier D). The flow-connected NEON chain MAYF → BLWA →
TOMB lies in one domain (D08, dev). HUC codes are parsed as strings everywhere.

**Development protocol.** All selection (architecture, hyper-parameters, dropout schedule, encoders,
"best single modality") uses out-of-fold predictions over the development folds (Tier D: leave-domain-
group-out, 5 folds; Tier C: 20-km-block 5 folds primary). Early stopping uses an inner set of whole
training sites. Test-role rows are removed immediately after loading (`stage0.drop_protected`).

## 5. Cross-fitting rule for modality-model outputs used as fusion inputs

Any output of another workstream's model (prediction, embedding, uncertainty) that enters Fuse as a
modality must be **cross-fitted**: for every Tier row r at site s,

1. the producing model's training data excluded site s, its flow group, and (Tier D) its NEON domain —
   i.e. the output is out-of-fold or "never saw this site";
2. the producing model's training data excluded **every Fuse test site** (for both development and test
   rows), so no test label influences any fusion input;
3. every input the producing model used is dated ≤ the row's issue time (same freshness rules as §2.1);
4. for the final test fit, test-row inputs come from producers trained on development sites only.

This is checked mechanically (`plugins.check_crossfit`; a plugin declares `training_sites` and, if
fold-aware, per-row excluded sites). Consequences today: M1/M6 (trained on USGS sites only) are valid
at all NEON sites; M3's LODO out-of-fold predictions satisfy (1) but not (2) (its folds include Fuse
test domains) — Fuse will either refit M3 per Fuse fold through M3's API with a custom fold map, or
use M3's encoder only from a model fit on Fuse development sites. Engineered-feature tokenizers (this
version) are fit inside each Fuse fold.

## 6. Models and comparators (identical rows, identical folds, equal tuning budget)

| Row | Definition |
|---|---|
| trivial context (`hub`) | LightGBM on forcings + catchment + day-of-year only |
| persistence (Tier D sensor targets) | last valid same-variable value; σ = training RMSE |
| single modality (`single:<m>`) | LightGBM Gaussian (mean model + gamma model of OOF squared residuals) on modality m + hub |
| **best single modality** | the `single:<m>` with lowest development-CV CRPS for that target (chosen on validation); the test-set oracle-best single is also reported |
| **GBM concat** (`concat`) | LightGBM Gaussian on all modalities + hub, native NaN for absent modalities |
| stacking (`stack`) | Super Learner: LightGBM on inner-fold OOF single-modality predictions + presence flags |
| UME (`ume`) | moment-matched mixture of the single-modality predictions of the modalities present |
| per-pattern (`pattern`) | one model per availability pattern trained on all rows whose availability ⊇ the pattern (P4) |
| TabPFN v2 (`tabpfn`) | `tabpfn==2.2.1` regressor/classifier on concatenated features, ≤ 10k training rows (site-stratified subsample), ≤ 500 features |
| **SENTINEL-Fuse** | §7; 5 seeds |

**Persistence-anchored comparators (`<config>@a`, added v1.1).** For Gaussian targets with a same-variable
last value available at issue time (sensor targets: `<sensor>__<var>__lastvalid`; lab targets: the previous
chemistry event's value, ≤ 35 d, strictly before d), every configuration that contains the anchor's own
modality is also fit as μ = anchor + GBM(Δ), σ from a gamma GBM of the anchored residuals; rows without an
anchor use the level model. Reason (comparator strengthening, never weakening): on the v1 table the level
GBMs lost to persistence at h = 7 for DO and nitrate (registry `fuse_stage0_d`, `val_cv_lodgo5`). The
**best-single-modality comparator** is the lowest dev-CV-CRPS configuration among all `single:*` and
`single:*@a`; the **GBM-concat comparator** is the better of `concat` / `concat@a` on dev CV. SENTINEL-Fuse may
use the same anchor (`--anchored`; anchor applied only when its owner modality is present *and kept*, so
LOMO and modality dropout remove it as well).

**Censoring strategy of comparators.** GBMs are trained on the censoring bound (detection limit) as if it
were the value (their only option); censored rows are never scored for any model (review W1). Fuse uses
the Tobit likelihood.

Tuning parity: each GBM comparator uses the same early-stopping procedure; any hyper-parameter search
run for Fuse is matched by an equal-count random search for `concat` / `concat@a` (logged; review W10). The
number of Fuse configurations tried is counted from TRAINING_LOG.md and reported with every claim. TabPFN
(`tabpfn==2.2.1`) is run on ≤ 10k site-stratified training rows per fold; for Gaussian targets its predictive
CRPS is computed from its quantile output (99 quantiles). Tier C models are reported with and without the
`phyto` modality (`--exclude-mods phyto`).

## 7. SENTINEL-Fuse (configuration space for selection on validation)

Group tokenizers per modality (engineered features; plug-in encoders later via `plugins.py`), absent
tokens for missing modalities, HEALNet/Perceiver hybrid core (latents 16–64 × d 64–256, 1–4 layers,
modality-specific cross-attention), (target family × horizon) queries, Gaussian heads with β-NLL and
Tobit censored likelihood, Bernoulli event heads, modality-dropout schedule (20 % all / 20 % single + hub /
otherwise independent drop p ~ U(0.3, 0.5)), availability-pattern-weighted sampling, AdamW + one-cycle,
fp16 AMP. Stage 2 (masked multimodal pretraining + hub alignment) and Stage 4 (isotonic + Mondrian
conformal by pattern × region) follow. Every configuration tried is logged (TRAINING_LOG.md, registry)
and counted next to the final claim.

### 7.1 Declared configurations (log; each counted when run, dev CV only)

| # | Run | Declared | Definition |
|---|---|---|---|
| 1 | `fuse_d_v2a_s0` | 2026-11-08 | d 128, 32 latents, 2 layers, dropout 0.1, wd 1e-4, lr 1e-3, 30 epochs, patience 6, anchored, all 52 targets — **run; negative vs both comparators** |
| 2 | `fuse_d_v2b_s0` | 2026-11-08 | as 1 with d 64, 16 latents, dropout 0.3, wd 0.01, lr 3e-4, 20 epochs, patience 8 (regularisation for cross-site overfitting) — running |
| 3 | `fuse_pt_wp1` → `fuse_d_v3_*` | 2026-11-08 (before any code or fit) | **Stage 2 pretraining** on W-P1 *dev-subset* rows only (M1 train/val-role USGS sites, issue dates ≤ WY2023; no W-P1 test row, no NEON row). Objectives, equal weight: (a) supervised h7/h14 heads for DO, SC, turbidity, NO3+NO2 (shared families with Tier D); (b) masked-modality reconstruction — per row one present non-hub modality is hidden and its standardised engineered features are reconstructed from the latents by a modality-specific linear decoder (MSE on observed features); (c) hub-contrastive alignment — InfoNCE (τ = 0.1, in-batch negatives) between the mean-pooled latents from hub-only input and from the full input of the same row. Early stopping on W-P1 dev fold 0 sites. **Fine-tuning on Tier D** per development fold: core (latents, self-attention blocks), cross-attention blocks and tokenizer groups of modalities/variables present in both tiers (same name and size) and the decoder query embeddings of shared families are initialised from the pretrained model; Tier-D-only modalities (chem, micro, macroinv, periphyton) and groups start fresh; then config 2 hyper-parameters and the §7 dropout schedule. Ablation reported: same fine-tuning from random init (= config 2). |
| 4 | `fuse_d_v4_*` | 2026-11-08 (before any code or fit) | **Residual on the anchored GBM concat**: μ = μ_GBM + Δμ_net, σ from the net, where μ_GBM is `concat@a` — out-of-fold for every row the net trains on (inner cross-fit over the other development folds) and the Stage-0 OOF for evaluation rows. Comparators unchanged (the GBM concat itself is comparator (b), so a pass requires the residual net to add information). For LOMO/P3 the modality is removed from **both** parts (a `concat@a` refit without m per fold), so P3 stays valid. |

Selection among configurations uses dev-CV CRPS averaged over the P1 + lab targets; the number of configurations
run is reported with any claim (R6, plan §15.5).

## 8. Endpoints (exactly as RESEARCH_PLAN §6)

* **P1 (NEON, Tier D).** 7-day CRPS skill (1 − CRPS_fuse / CRPS_ref, model space) for **DO, turbidity,
  nitrate, chl-a** (`y__do_mgl__h7`, `y__turb_fnu__h7`, `y__no3n_mgl__h7`, `y__chl_ugl__h7`) vs (a) the
  best single modality (chosen on validation) and (b) GBM on concatenated features, leave-site-out
  (domain holdout) on the Tier D test sites. Paired rows = test rows with the target observed.
* **P2 (cyanotoxins, Tier C).** Microcystin > 8 µg/L AUPRC and AUROC on Test-S (20-km blocks; results
  also stratified by distance to the nearest training site 5–20 km / > 20 km, the "5-km" scheme) and
  Test-R (leave-region-out), vs best single modality and GBM concat.
* **P3 (≥ 5 modalities).** On paired Tier D test rows where modality m is present, inference-time
  leave-one-modality-out ΔCRPS_m = mean over the P1 targets and the lab targets {TN, TP, DOC} of
  (CRPS without m − CRPS with m) / CRPS_hub > 0 with CI excluding 0, for ≥ 5 modalities. Headline
  claims are confirmed by retrain-LOMO.
* **P4 (graceful degradation).** For every availability pattern with ≥ 200 test rows and ≥ 3 sites,
  Fuse CRPS ≤ the per-pattern model's CRPS (paired, one-sided 95 % bootstrap upper bound of the relative
  difference ≤ +2 %).
* **P5 (calibration).** 80 % and 90 % conformal intervals cover within ±3 points of nominal per
  availability-pattern group and per region (Tier D: NEON domain group; Tier C: HUC2).

**Primary family for multiplicity (Holm):** P1 (4 targets × 2 comparators) and P2 (2 test sets × 2
comparators, AUPRC) — 12 one-sided hypotheses "skill > 0", p-values from the site-cluster (Tier D) /
20-km-block (Tier C) bootstrap (share of resamples ≤ 0, two-sided ×2). H-G2 passes if P1 or P2 passes
against **both** comparators under spatial blocking (CI excluding 0) **and** P3 holds for ≥ 5 modalities.

**Decision rules (v1.1, review C2).**
* **P1 passes** if for **at least one** of the four targets **both** comparisons (vs best single, vs GBM
  concat) are Holm-significant (small-cluster-corrected 95 % CI, §10, excludes 0) **and** the point skill
  stays > 0 when any one test domain is removed (leave-one-domain-out). The number of passing targets
  (0–4) is reported; "P1 strong" = all four.
* **P2 passes** if for at least one test set (Test-S or Test-R) both AUPRC comparisons are Holm-significant
  (20-km-block bootstrap). The "5-km scheme" of the plan is implemented as (i) the 5-km buffer that removes
  all dev sites within 5 km of a Test-S site and (ii) Test-S results stratified by distance to the nearest
  training site (5–20 km / > 20 km) — reported, not a separate test.
* **P3** counts **non-hub** modalities only (the hub — forcings, catchment, time — is present for every row
  and is the reference, not a candidate). ΔCRPS_m is normalised by the hub model's CRPS on the same rows,
  averaged over the P1 and lab targets {TN, TP, DOC} available on rows where m is present (equal weight
  per target). A modality is **eligible** only if it is present on ≥ 200 test rows at ≥ 3 test sites
  (§8.2). Inference-time LOMO decides; retrain-LOMO confirms the headline.
* **P4** uses a non-inferiority margin of 2 % (relative CRPS) because exact "≥" cannot be tested; patterns
  with < 200 test rows or < 3 sites are not evaluated (counts reported).
* **P5** is descriptive per group (coverage within ±3 points); no multiplicity correction.

### 8.1 A-priori hard strata (secondary endpoints, v1.1; plan §15.5 items 5–7)

Fusion should matter most where single sources fail. These strata are defined from training-fold
quantiles and issue-time inputs only (never from model results) and reported for P1 targets vs both
comparators (skill + small-cluster CI + LODO range; no Holm, labelled secondary):

| Stratum | Definition (thresholds from training rows of the fold; for test: all dev rows) |
|---|---|
| `ext_p90`, `ext_p95` | target value above the training P90 / P95 (DO and DO-min: below P10 / P5) — extremes |
| `change_p90` | \|y − anchor\| above the training P90 of \|y − anchor\| — rapid change, where persistence fails |
| `own_absent` | the target's own sensor modality absent at issue time (the anchor is unavailable) |
| `sparse_le2` | ≤ 2 non-hub modalities present |
| `storm_p95` | 3-day antecedent precipitation above the training P95 |
| unseen domains | the whole Tier D test set is domain-held-out; results are also given per test domain |

Strata with < 50 rows or < 3 sites are reported as counts only (`sentinel/models/fuse/stats.py::hard_strata`).

### 8.2 P3 feasibility (presence counts only; no outcome inspected)

Tier D v2 test rows with a P1 or lab target: 18,983 at 8 sites; every one of the eight non-hub modalities
is present at all 8 test sites (rows: sonde 17,431, nitrate 14,800, temp 16,800, hydro 17,585, chem 16,703,
micro 12,216, macroinv 14,474, periphyton 11,873). P3 is therefore testable on Tier D; with 8 site clusters
only large effects reach significance. Dev-CV LOMO (26 sites, out-of-fold) is reported as a secondary,
selection-affected estimate.

### 8.3 Saturation check (plan §15.5 amendment 2026-11-08)

Tier D is not ceiling-saturated (persistence/GBM CRPS skill vs hub ≈ 0.4–0.5 at h = 7). The risk is the
other saturation mode — ties because the test set is small (8 sites). Proposed harder/larger track
(**not yet frozen; protocol section, split sha and power check to be added before any selection on it**):
"Tier W-P1" — **approved by the lead 2026-11-08 and pre-registered in §12** (split frozen, power checked,
no fit yet); reported alongside, never instead of, the Tier D endpoints.

**Honest fallback headline:** if deep fusion only ties GBM fusion, the claim is "missing-modality-robust
deep fusion matches strong GBM fusion and adds calibrated any-subset inference", with the full table.

## 9. Plug-and-play protocol and acceptance test

New modality = featurizer (cross-fitted, §5) + tokenizer + modality embedding + new modality-specific
cross-attention weights (`FuseModel.add_modality`, core frozen; optional LoRA later). **Regression
guarantee:** on rows where the new modality is absent, predictions are bit-identical to the model
before the addition (`torch.where` skip; verified in `tests/test_fuse.py::test_plug_and_play_bit_identical_when_absent`
and re-verified numerically on real development rows for every added modality). **Acceptance:** on
paired, blocked development rows where the new modality is present, relative ΔCRPS (or ΔAUPRC) > 0 with
site-bootstrap 95 % CI excluding 0 for ≥ 1 target, and no target worse than −1 % (lower CI bound of the
relative improvement ≥ −0.01); otherwise the modality is logged "non-contributing" and kept out of the
headline model (`plugins.acceptance_test`).

## 10. Statistics and freezing

Cluster bootstrap (B = 2,000 for final, 1,000 for development) of paired metric differences; units
(review W8): **NEON site** (Tier D; NEON domain used for the leave-one-domain-out robustness check),
**20-km spatial block** (Tier C, both test sets; matches the split construction). **Small-cluster correction
(review W9):** the percentile interval is widened about the point estimate by sqrt(G/(G − 1)) (G = number of
clusters; 1.069 for the 8 Tier D test sites) — `sentinel/models/fuse/stats.py::cluster_ci`. Diebold–Mariano (HAC) per site for P1, aggregated. Holm across
the primary family. ≥ 5 seeds for Fuse, seed variance included by resampling seeds within the bootstrap.
Before any test scoring: the scoring script, model checkpoints and thresholds are hashed into a
`fuse_freeze.json` next to this file; test rows are scored once; the result is logged as-is (R8).

## 11. Deviations log

* 2026-10-25 06:09 UTC — Tier D split v1 → v2 (composition rule; before any fit). See §4.
* 2026-11-08 — v1.1 amendments (before any Fuse result; test untouched): (a) decision rules made explicit
  (§8, review C2): P1 "≥ 1 target, both comparators, Holm" + LODO sign check; P3 non-hub only, normalised by
  CRPS_hub, eligibility rule; P2 "5-km" = buffer + distance strata; P4 2 % margin (plan says "≥"). (b) P2
  primary label `y_gt8_total` (W5); WQP/NLA duplicates removed (W4); Tier C satellite restricted to
  imagery on/before the sample date (W3). (c) Bootstrap units and small-cluster factor (§10, W8/W9).
  (d) Persistence-anchored comparators and anchored Fuse option (§6), added after Stage-0 v1-table comparator
  results showed persistence beating level GBMs — a comparator strengthening. (e) Hard strata §8.1 (lead
  directive 2026-11-08) and saturation note §8.3. (f) Tier D table v2 with hydro, macroinv, periphyton
  (NEON P2 builds finished); the split (site roles) is unchanged. (g) The 24 registry entries of the smoke
  run `smoke_stage0_d` (one fold) logged as `val_cv_lodgo` were superseded by entries with split `smoke` (W6).
  (h) `scripts/data/make_fuse_splits.py check` regenerates both frozen splits byte-identically (W11); the
  Tier C split was generated from the rows with a `y_gt8` label (5 sites whose only results are censored at
  DL > 8 µg/L have no role and are never used).
* 2026-11-08 — W-P1 harder track pre-registered (§12; split frozen `b9e0d5c4…`, power check by availability
  counts only; chl-a dropped from W-P1 for power before any fit).
* 2026-11-08 — §7.1 configuration log started; configs 3 (Stage-2 pretraining on W-P1 dev rows, lead suggestion)
  and 4 (residual on anchored GBM concat) declared before any code or fit, after config 1 was negative.

## 12. Harder / larger track W-P1 (pre-registered 2026-11-08, before any table build or fit)

**Why.** Tier D's test has 8 sites (risk of ties from a small test, plan §15.5 amendment). W-P1 is the P1
forecasting task at USGS continuous-sonde sites of the frozen M1 split, with unseen sites **and** unseen
years. Claims from it are labelled "W-P1".

**Split** `fuse_tier_wp1_split.json`, sha256 `b9e0d5c44d8a83c3bc4abc8517d66fb520186fcf6e64db59aa3e4d40e050288a`
(`scripts/data/make_fuse_splits.py`, regenerates byte-identically). Roles copied from `sonde_split.json`
(sha256 `fd4bb14fe730…`): M1 train/val → dev (2,114 sites, 78 basin groups), M1 test → **test** (564 sites,
42 basin groups; never trained on, never used for selection). Dev CV: 5 folds of whole M1 basin groups
(flow-connected stations together). Windows: dev issue dates WY2009–2023 with every target date ≤ 2023-09-30;
**Test-W-A (primary)** = test sites × issue dates 2023-11-12 … with target dates ≤ 2025-09-30 (42-d embargo);
Test-W-S (secondary) = test sites × WY2009–2023. M1 Test-B (WY2026) is not used.

**Table** (`sentinel/data/fuse_tier_wp1.py`; datasheet + manifest). One row per (site, issue day d) on a
fixed 3-day calendar grid (days since 2008-10-01 divisible by 3; label-free thinning of autocorrelated daily
rows, fixed before the build) with a target observed at d + 7 or d + 14. Modalities (inputs dated ≤ d; same engineered features as Tier D): `sonde` (DO, SC, pH,
turbidity FNU, fDOM), `nitrate` (NO3+NO2 sensor, mg/L N), `temp`, `hydro` (log discharge, gage height
anomaly), `labhist` (WQP discrete results strictly before d, ≤ 90 d, as Tier W), `satellite` (only where
S2 matchups exist; expected sparse); hub `wx` (gridMET observed, windows ending at d), `static`
(StreamCat + site registry), `time`. Units stay separate (FNU ≠ NTU; RFU never mixed with µg/L).

**Targets and power (availability counts only, `scripts/eval/fuse_wp1_power.py` →
`results/fuse/wp1_power_check.json`).** Daily means at h = 7 and 14 d of DO (Test-W-A: 182 test sites with
≥ 180 days, 23 basin groups), SC (237 / 30), turbidity FNU (64 / 11), NO3+NO2 (33 / 8). A target is
evaluated only with ≥ 20 test sites and ≥ 8 basin groups — chl-a (4 test sites) is **not** evaluated (reported
descriptively). Dev: DO 483 sites, SC 925, turbidity 302, nitrate 59 (≥ 1 year each).

**Endpoints (same comparators, rules and statistics as §6/§8/§10).** W-P1 primary: 7-day CRPS skill of Fuse
vs best single modality and vs GBM concat (both incl. the `@a` anchored variants) for the four targets on
Test-W-A; Holm across the 8 hypotheses (separate family from §8, reported separately); pass = ≥ 1 target
vs both comparators, and positive when any one test basin group is removed. 14-day horizon, the §8.1 hard
strata and Test-W-S are secondary. Cluster unit for CIs: M1 basin group (sites as sensitivity). Persistence
and the M1 SENTINEL-Sonde forecasts (cross-fitted per §5: M1 never trained on its test sites) are reported
as references; using M1 outputs as a Fuse input requires the §5 checks.
