# M7 SENTINEL-Lite — architecture round v2 amendment (data version national-2b, ANCHOR-Lite / WaterSet-Lite, re-run published comparators, track P2)

**Status:** DRAFT v0.3 (2026-11-10, ~18:15 UTC), written by M7 for the lead's review and stamp. AI-assisted.
- v0.2 (sha `ca281d84…d771`) differed from v0.1 (sha `0cf9fbcd…ba31`) only in §8.3–§8.4: the P2 lock list exists, and
  its hash and counts are filled in.
- v0.3 differs from v0.2 only in: the state note in §0; one sentence in §3.3 (limit when a chip has several censored
  results); the operational wording of the eligibility rule in §8.5; **the P2 index counts and their consequence in
  §8.6 (the confirmatory floor is not met, so P2 is descriptive)**; one sentence in §8.7; and Q12 in §11. No rule,
  constant, model, comparator or threshold was changed.
**Stamp:** pending. The lead records this file's SHA-256 in a `lite/prereg_snapshot` registry row and commits it.
**Implements:** `docs/plans/architecture_research/LEAD_DECISIONS_20261110.md` (rules 1–6 and the M7 section, which
override the plan where they differ) and `docs/plans/architecture_research/M7_architecture_plan.md` §4–§8. Differences
from the plan are listed in §12.
**Relation to frozen files.** This is a new file. It edits nothing that is frozen or hashed. `lite_protocol.md`
(frozen, 4-state look spent), the national split (`lite_split_national.json`), the national-2 split
(`lite_split_national2.json`, sha `976b5be7…ebe2`) and every file hashed by a FREEZE record are unchanged. It adds to
`lite_national_protocol.md` (v0.1, not frozen) by reference: where the two differ on a new arm, this file wins.
**Bars.** Nothing here lowers a bar. RESEARCH_PLAN §15.1 and `sota_campaign.md` rules 1–8 are unchanged. The national
label rule (`lite_protocol.md` §8a) is unchanged; the comparators added here can only make a label harder to earn.

**Lead ruling, recorded verbatim (2026-11-10):** the frozen `lite_protocol.md` §8a label rule stands for the national
track; no target-scoped label is added there after validation has been seen; a scoped SOTA claim is possible only on
P2. P3 France and P4 NLA are deferred.

## 0. What exists at the time of writing (and what does not)

- **No validation number exists on any arm or track in this file**: none on ANCHOR-Lite, WaterSet-Lite (ANCHOR-Set),
  their ablations, GBM-parity and its variants, the CyFi-style or HydroVision-style comparators, or track P2.
  - Checked 2026-11-10 17:31 UTC: the registry holds no row, and `$SENTINEL_RUNS/lite` holds no run directory, whose
    name contains `anchor`, `waterset`, `densenet`, `cyfi`, `hydrovision`, `gbm_parity`, `lite_p2` or `national2b`.
  - No code for these arms exists yet. No P2 site list, index or chip exists. No recovered label has been used by
    any model.
- **Guarantee until the stamp (rule 2).** Only code, unit tests, label/data builds and smoke tests are run.
  - Smoke tests use synthetic tensors, or train-role rows only: at most 2 epochs on at most 2,000 train chips, checking
    shapes, finite losses and a falling training loss. No candidate-vs-comparator metric is computed, not even on
    train folds.
  - The new entry points refuse to load `val`, `val_n` or `val_p2` rows unless a `lite/prereg_snapshot` registry row
    holds the SHA-256 of this file as it is on disk (`extra.sha256[<this path>]`).
  - Labels of protected rows (`test`, `test_n`, `test_time`, `test_time_n`, `test_p2`): counts only.
- **What is known (prior knowledge, disclosed).** All numbers are validation MdSA (%) on national val ∪ val_n,
  3-seed ensembles, models selected on validation (optimistic).

  | Model (national-2 training rows) | Secchi | Turbidity (NTU) | TSS |
  |---|---|---|---|
  | Config 1 crop128 KD | 50.7 | 94.8 | 98.8 |
  | Config 2 crop192 KD | 50.6 | 97.2 | 91.5 |
  | Config 3 crop256 KD | 48.9 | 91.0 | 91.6 |
  | Config 4 SSL init, crop128 | 53.6 | 103.0 | 103.3 |
  | Config 5 SSL init, crop256 | 50.7 | 98.0 | 102.5 |
  | Config 6 crop256 KD + scene/lat/lon/DOY FiLM | 38.0 | 89.8 | 96.6 |
  | No-KD twin of config 3 (comparator, not counted) | 51.7 | 100.9 | 99.8 |
  | GBM-RGBN | 47.3 | 116.8 | 97.9 |
  | GBM-RGBN + context (lat/lon/DOY) | 36.5 | 111.9 | 102.7 |
  | Location kNN / location-season GBM | 51.5 / 58.1 | 129.1 / 142.1 | 116.5 / 126.4 |
  | Band ratio (val-best form) / train median | 78.6 / 101.0 | 129.6 / 151.9 | 100.4 / 107.7 |

  - Selection among the six counted national-2 configs (frozen rule, S) was re-run at about 17:45 UTC: config 6 leads
    (S = 0.853; config 3: 0.916).
  - Config 6 vs GBM-RGBN + context, paired, val: Secchi +1.5 [−1.6, +5.0]; turbidity −22.1 [−52.7, +1.6]; TSS −6.1
    [−25.9, +10.2]. So on validation no current student beats the context GBM with a CI excluding 0 on any target.
    Location and season explain about 11 MdSA points of Secchi for the GBM and for the CNN alike.
  - The 4-state track look is spent (label "Negative"). No national test role has been loaded.
  - These numbers motivate ANCHOR-Lite: start from the strongest tabular model and let the image model learn only the
    residual. They also mean every design choice below was made knowing the national validation picture; this is why
    the lead's ruling confines any new scoped label to P2.
- **Recovered labels and the P2 pool were known by counts and metadata only** (§3.1, §8.4) when the design was
  fixed (v0.1). No value of any P2 site had been read before the lock list (§8.3) was written.
- **State at v0.3 (2026-11-10 18:15 UTC).**
  - The P2 lock list, the P2 index and 513 P2 chips exist (§8.3–§8.6). Code exists for the P2 lock and index only
    (`sentinel/models/litev2/p2.py`, `p2_index.py`, `scripts/data/build_lite_p2.py`). No model or comparator code of
    this amendment exists, and no model has seen a P2 chip or a recovered label.
  - After the lock row existed, the index build read `train_p2` and `val_p2` label values, as §8.3 allows. During a
    plan-only run a library warning printed 34 `train_p2` and 12 `val_p2` values (no site ids) into a build log. The
    log was deleted and the cause was fixed. **No `test_p2` value has been read**: test sites are read through a
    present/absent projection only, and the index stores no test value (checked).

## 1. Practical reasons (stated before any result)

| Arm / track | Practical reason |
|---|---|
| Data version national-2b (FNU, NTRU, SSC, censored rows) | USGS and many states report turbidity in FNU and sediment as SSC. The standard-unit filter dropped them, and it dropped "below detection" and "Secchi on bottom" rows, which are the clear-water and shallow-water cases a drone user meets. |
| ANCHOR-Lite | A field user gets one number per target from a phone-size model. The tabular model already uses colour statistics, scene statistics, place and season; the image branch should add only what spatial pattern adds (plumes, shore mixing, narrow channels). One hybrid model, not two products. |
| WaterSet-Lite (ANCHOR-Set) | Many targets are a property of the water pixels, not of the scene layout. A set model over water pixels is about ten times smaller than the CNN and tests whether spatial convolution is needed at all. |
| Re-run published comparators | §15.1 needs the strongest published method re-run on the same labels and inputs. Without it no "state of the art" wording is possible anywhere. |
| Track P2 | Drones resolve narrow rivers and small water bodies that 10–20 m satellites cannot. USGS sondes report FNU, and that is where operational turbidity monitoring happens. A model is useful there only if it works at water bodies it has never seen. |

## 2. Scope of claims

- **National track.** Rules unchanged (`lite_protocol.md` §8a, §12; `lite_national_protocol.md` N1–N7). "State of the
  art" does not apply there. Possible labels stay First-in-class / Competitive / Negative.
- **P2.** One scoped label is possible: "state of the art for FNU turbidity from 4-band 0.6–1 m imagery at unseen US
  water bodies (P2)". Rule in §8.8.
- **Wording.** A positive ANCHOR-Lite result is reported as "the GBM-anchored hybrid vs the GBM", never as "the CNN
  beats the GBM". Every arm and track is reported, wins and losses (rule 6).

## 3. Data version national-2b (recovered labels)

Source: `processed/naip_matchups/national_pm7/matchups.parquet` (manifest `naip_matchups_national_pm7`, already on
disk). No new imagery is expected. A recovered row whose chip is missing from the NAIP store is dropped and counted.

### 3.1 Counts (rows; counted without using any value)

| Recovered group | Rows | On train-role chips | On chips with no national-2 role | On val / val_n | On the four test roles | On `excluded` chips |
|---|---|---|---|---|---|---|
| Turbidity FNU | 471 | 128 | 237 | 16 | 87 | 3 |
| Turbidity NTRU (+4 censored = 323) | 319 | 25 | 260 | 13 | 19 | 2 |
| SSC mg/L (+3 censored, +5 in % = 456) | 448 | 49 | 306 | 41 | 50 | 2 |
| Censored Secchi | 316 | 32 | 245 | 9 | 11 | 19 |
| Censored TSS | 282 | 111 | 73 | 44 | 48 | 6 |
| Censored chl-a | 162 | 70 | 17 | 43 | 26 | 6 |
| Censored turbidity | 26 | 3 | 16 | 1 | 6 | 0 |
| Censored SSC | 3 | 0 | 2 | 1 | 0 | 0 |

- The 1,156 recovered rows without a role sit on at most 1,043 chips (the role-less chips of the build). Those chips
  also hold 272 other rows (JTU and other units, SSC in %, and standard-unit rows that the frozen label rules reject).
  These stay rejected; national-2b recovers only the groups in the table.
- Other turbidity units in the build: JTU 56, FNRU 5, NTMU 4, FNMU 2, blank or other 22.
- **The training gain is modest** (for example FNU: 128 rows now, at most 365 after §3.4). It is stated here so that
  no later text overstates it.

### 3.2 Targets and unit rules
Six targets, all in log10: `secchi` (m), `turbidity` (NTU), `turb_fnu` (FNU), `tss` (mg/L), `ssc` (mg/L), `chla`
(µg/L).

| Reported unit | Rule |
|---|---|
| NTU | `turbidity`, unchanged. The only turbidity label on the national evaluation blocks. |
| FNU | Own target `turb_fnu`. Never merged with NTU. |
| NTRU | Training only. Enters the NTU head with one learned additive log10 offset (initial value 0). Never an evaluation label. |
| JTU, FNRU, NTMU, FNMU, blank, other | Dropped, counted. |
| SSC mg/L | Own target `ssc`. Never merged with TSS. |
| SSC % | Dropped, counted. |

- Bounds: frozen `data.BOUNDS` for the four original targets; `turb_fnu` (0.05, 1000) FNU; `ssc` (0.1, 10000) mg/L.
  Out-of-bounds rows are dropped and counted.
- Chip label = median of the uncensored values of that target on the chip, as in the frozen `chip_labels`.
- The matching windows are unchanged: ±1 d, Secchi ±7 d on training rows only (N4).

### 3.3 Censored rows and the censored likelihood
- Direction is re-joined from `processed/wqp_discrete` (`censor_type`) by site, date, characteristic and unit: `<` is
  left-censored at `detection_limit`; `>` is right-censored (Secchi "visible on bottom"). Rows with a missing or mixed
  direction, a missing limit or a limit outside the bounds are dropped and counted. If a chip has several censored
  results for a target, the limit is the largest one for `<` and the smallest one for `>` (the conservative side).
- A censored row is used only when the chip has no uncensored value for that target.
- **Censored rows are training rows only.** They never enter MdSA or any other evaluation metric, on any block.
- **Likelihood (Tobit, log10 space).** With head outputs μ, σ and limit L:
  left-censored: −log Φ((log10 L − μ)/σ); right-censored: −log Φ((μ − log10 L)/σ). The term is multiplied by the
  detached σ^(2β), β = 0.5, exactly like the uncensored β-NLL term, and it is averaged with the uncensored terms of
  the same target.

### 3.4 Roles
Split file `docs/preregistration/lite_split_national2b.json` (new; its SHA-256 goes in a registry row).
1. Every national-2 chip keeps its role and its water-body id. Evaluation blocks are untouched, so every national
   evaluation label is unchanged.
2. A chip without a national-2 role is grouped with the frozen `water_body_groups` (national-2 parameters) over the
   union of national-2 sites and new sites. It becomes `train` only if all of these hold, otherwise `excluded`:
   - its water body holds no national-2 chip whose role is not `train`;
   - if its water body holds no national-2 chip at all, its 25-km block is a train block (national-1 block roles;
     unseen blocks by the national-1 salted rule);
   - NAIP year ≤ 2021;
   - it is more than 2 km from every evaluation-role site.
3. Recovered labels on `train` chips are used for training. Recovered labels on `val` / `val_n` chips are reported
   descriptively only (N2 minimum n applies) and never enter selection. **Recovered labels on the four national test
   roles are not built, not loaded and not reported** (ruling requested: Q2).
4. Checks at build: the 4-state, national and national-2 split hashes are unchanged; the uncensored rows of the four
   original targets on `train` are identical to national-2 (so the frozen-list baselines of
   `natbaselines_20261109-211150` keep parity for those targets; if the check fails they are refitted).
5. Sites of `national_pm7` never enter P2 (§8.2).

**Training rows of this round ("v2 training rows")** = national-2b `train` ∪ `train_p2` (§8). Every new arm and every
comparator is fitted on exactly these rows.

## 4. Models

### 4.1 Inputs (the "context-complete" set, same for every parity comparator)
- Chip: 4 × 256 × 256 uint8 NAIP (R, G, B, NIR), raw mode, and its ground sampling distance.
- **s, scene statistics (51 values):** for each band the DOQQ mean, std, p01, p02, p05, p25, p50, p75, p95, p98, p99
  and dark-NIR-water median (48), the scene valid fraction and dark-NIR fraction (2), and gsd (1).
- **c, location and season (5 values):** latitude and longitude as a unit-sphere vector (3); day of year of the image
  as sine and cosine (2).
- Nothing else. Site class, HydroLAKES morphometry, sample depth and any label-side field are not inputs.
- s and c are standardised with train-row means and standard deviations stored in the checkpoint.

### 4.2 Anchor (= comparator GBM-parity, §5)
- For each target t, LightGBM with the frozen `baselines.GBM_PARAMS`, mean of seeds 0, 1, 2, on 98 features: the 45
  frozen chip features, the 50 scene statistics and lat, lon, DOY.
- Turbidity is one pooled model over NTU, FNU and NTRU rows with two unit indicator features; it predicts the NTU
  and FNU anchors by setting the indicator. TSS and SSC are separate models.
- **Training rows get out-of-fold anchors**: 5 folds by water body, fold = `int(sha256("SENTINEL-Lite-anchor|" +
  water_body).hexdigest()[:8], 16) % 5`. **Evaluation rows get the model refitted on all v2 training rows.**
- Censored rows are not used by the anchor.
- At initialisation ANCHOR-Lite and ANCHOR-Set output exactly the anchor, so the paired difference against GBM-parity
  measures what the residual branch adds.

### 4.3 ANCHOR-Lite (rank 1)
- **Image branch:** the national-2 leader's network (MobileNetV4-Conv-S + 48-channel decoder at stride 4, class
  `LiteStudent`, unchanged), initialised from the seed-matched weights of config 3
  (`natstudent_mnv4s_raw_crop256_kd03_national2_20261110-020817`).
- **FiLM** on the decoder output from MLP([s, c]): 56 → 64 → 96, last layer zero-initialised.
- **Pooling** to 96 values: (a) the frozen centre-window water-weighted pooling (`features.pooling_weights`); (b) a
  learned attention pooling over the whole chip, logits = 1×1 conv of the features + a·r + b·r² (r = distance to the
  chip centre in metres; a, b learned, initial 0), restricted to dark-NIR water cells, all cells if the chip has
  none.
- **Heads:** e = MLP([s, c, a]) 62 → 64 → 64, where a is the vector of the six anchors. For each target:
  MLP([pooled, e]) 160 → 64 → 2 gives (Δ_t, log σ²_t). **μ_t = a_t + κ_t · tanh(Δ_t / κ_t)**, κ = 0.3 for Secchi and
  0.5 for the others (log10 units). The Δ output is zero-initialised; the log σ² bias starts at the log of the
  train out-of-fold residual variance of the anchor; log σ² is clamped to (−8, 6) as in the frozen model.
- **Loss:** Σ_t w_t · (β-NLL with β = 0.5, plus the Tobit term of §3.3) + 0.3 · KD + 0.01 · mean(Δ²); w = 1 for
  Secchi, NTU, FNU, TSS and 0.5 for SSC and chl-a.
- **KD:** the leader's dense 4-target map head is kept as an auxiliary head, trained only with the frozen
  `distill_nll` on the chips that have Lite-safe teacher targets (the existing file; no new teacher target is
  computed, none on any P2 chip). The auxiliary head is not used at inference.
- **Training:** 40 epochs, fixed; no early stopping and no checkpoint choice on validation. AdamW, 1e-3 for new
  modules, 5e-4 for backbone and decoder; batch 32; schedule, crop, AMP and augmentation as config 3. The per-band
  gain/gamma/offset augmentation is applied identically to the scene quantiles (quantiles commute with monotone maps;
  mean and std are recomputed approximately by the same map); the anchor is not recomputed under augmentation. Seeds
  0, 1, 2.
- **Output:** one (μ, σ) per target for the chip centre or tile. It does not produce a per-pixel map; that is a
  product limitation and it is stated wherever the model is reported.
- **Size (plan estimates, to be measured):** about 1.40 M network parameters, about 1.13 GMACs per 512² tile, INT8
  about 1.7 MB, plus the anchor trees (about 0.2 M numbers, about 1.1 MB in a compact float32 file).

### 4.4 WaterSet-Lite (ANCHOR-Set), the alternative residual branch
- Up to 1,024 dark-NIR water pixels sampled from the chip. If fewer than 64 exist, the pixels nearest the chip centre
  fill the set to 64.
- 19 features per pixel: log10 DN of the 4 bands; log ratios B/R, G/R, B/G, NIR/R; log(DN / scene p50) per band;
  distance to the chip centre (m); distance to the shore (m, distance transform of the water mask); local 5×5
  standard deviation per band; water flag.
- Linear 19 → 64; three global tokens from s, c and a; two induced set-attention blocks (16 inducing points, width 64,
  4 heads); pooling by multi-head attention with 4 seeds; then the same residual heads, bounds and loss as §4.3.
- No dense feature map exists, so there is no KD term (recorded as a difference, not hidden).
- Training resamples pixels every epoch. At evaluation the sample is fixed by a seed derived from the chip id.
- AdamW 1e-3, batch 64, 40 epochs fixed, seeds 0, 1, 2. About 0.15 M parameters.

### 4.5 Ablations
| ID | Change from ANCHOR-Lite | Question |
|---|---|---|
| A1 | No anchor: μ_t = train median + Δ_t (no tanh bound); FiLM and e use [s, c] only | Does the anchor matter, or is context enough? (context-complete direct student) |
| A3 | Anchor kept; s and c removed from FiLM and heads (e = MLP(a)) | Does the residual need the context? |
| A4 | Four original targets, uncensored rows only; anchors refitted on the same reduced rows; FNU is scored through the NTU head | What do the recovered labels add? |
| A5 | Centre-window pooling only | What does whole-chip attention pooling add? |

### 4.6 Budgets and the deployable carrier
- The frozen budgets (`lite_protocol.md` §5: ≤ 3 M parameters, ≤ 2 GMACs per 512² tile, INT8 ≤ 4 MB, proxy latency
  ≤ 100 ms) apply to **the whole deployable pipeline**: the INT8 network, the anchor trees and the feature code.
  - Parameters = network parameters + all tree thresholds and leaf values.
  - Size = INT8 ONNX file + the anchor file (compact float32 format written by our exporter; parity with LightGBM
    within 1e-4 log10 is a unit test).
  - Latency includes feature extraction and the trees.
- The carrier is the INT8 pipeline of seed 0. If the 3-seed ensemble of the selected config meets every budget as a
  whole, the carrier is the INT8 3-seed ensemble. This is decided by the budget arithmetic alone.
- A multi-input export wrapper (chip, s, c, a) lives in a new file outside every frozen hash. Export recipe and the
  INT8-vs-FP32 report are as in the frozen protocol.
- A config whose pipeline misses any budget is reported as "over budget" and cannot be a carrier.

## 5. Comparators, with full information parity (rule 3)

Common rules for every comparator:
- fitted on exactly the v2 training rows, with the same label rules (§3) and the same evaluation rows;
- 3 seeds, averaged, as for the candidate;
- hyper-parameters fixed below; nothing else is tuned;
- not limited by the on-device budgets;
- **built first:** every comparator's validation rows are in the registry before the first validation evaluation of
  ANCHOR-Lite, ANCHOR-Set or an ablation. The candidate's evaluation entry point checks this and refuses otherwise.

| ID | Comparator | Inputs | Notes |
|---|---|---|---|
| S1 | Train median | — | frozen code |
| S2 | Location kNN (k = 10) | lat, lon | frozen code |
| S3 | Location/season GBM | lat, lon, DOY | frozen code |
| S4 | GBM-RGBN | 45 chip features | frozen code |
| S5 | GBM + context | 45 + lat, lon, DOY | frozen code |
| S6 | **GBM-parity** | 45 + 50 scene statistics + lat, lon, DOY (98) | = the anchor (§4.2). Same information as the candidate. |
| S7 | GBM-parity, per unit | as S6 | separate NTU-only and FNU-only turbidity models (differs from S6 for turbidity only) |
| S8 | GBM-parity-MT | S6 + out-of-fold predictions of S6 for the other targets, for NTRU, and for the chip-mean teacher μ of the 4 teacher targets | parity of auxiliary targets and of the KD signal; folds as §4.2 |
| S9 | GBM-parity-AFT | as S6 | XGBoost `survival:aft`, normal distribution, scale 1.0, 400 rounds, learning rate 0.03, depth 4; uses the censored rows (parity of labels). Run for a target only if it has ≥ 20 censored training rows (count). |
| P1 | Band-ratio regressions (published forms) | chip band statistics | frozen forms; turbidity forms for FNU, TSS forms for SSC; every form is listed |
| P2a | **CyFi-style LightGBM**, faithful | water-pixel statistics of the chip: per band mean, min, max, p05, p95; B/R, G/R, B/G, NIR/R, NDVI, NDWI from band means; green p95/p05; percent water; month; days between image and sample | CyFi recipe (LightGBM on per-band water-pixel summaries and ratios, month, days-before-sample; no lat/lon). Differences: chip field (≈ 154 m) instead of a 2,000 m box; no land-cover and no AOT feature (not available for NAIP); frozen `GBM_PARAMS`. |
| P2b | CyFi-style + parity | P2a + s + c | |
| P3a | **HydroVision-style DenseNet121 regressor**, image only | chip | ImageNet weights, 4-band stem (NIR filter = mean of the RGB filters), global-pool regressor Linear(1024 → 2 per target). Same crops, augmentation, targets and loss (β-NLL + Tobit + NTRU offset) as the candidate. |
| P3b | HydroVision-style + parity | chip + s + c | pooled 1024 ⊕ MLP([s, c]) 64 → 256 → heads; auxiliary chip-level KD head on the teacher μ, weight 0.3 |

- **DenseNet training:** AdamW, heads 1e-3; batch 32; **80 epochs fixed** (the candidate's 40 epochs plus the 40 of
  the leader it starts from; in the comparator's favour). Backbone learning rate ∈ {1e-4, 3e-4}: both are run with seed
  0, the one with the lower validation score (national S for the national track, FNU MdSA for P2) is kept, and seeds 1
  and 2 use it. That is 8 runs for P3a + P3b, all logged. About 7–8 M parameters; reported as over the on-device
  budget.
- **Descriptive only (never in a label set):** the current national leaders (configs 3 and 6; they score FNU through
  the NTU head) and "GBM-stacked" (S6 + 16 principal components of the leader's pooled embedding, a check on whether
  the CNN's gain is only features).
- TabPFN (plan C2-TPF, optional) is not run: the package is not in the pinned environment, which is not modified.
- HydroVision's own data (fixed-camera RGB stills) and CyFi's (Sentinel-2) differ from ours. "Re-run" here means the
  published recipe refitted on our rows; the published numbers themselves are not comparable and are not quoted as
  comparators.

## 6. Configuration caps

| Track | Cap | Used / allocated |
|---|---|---|
| National (national-2), 8 configs × 3 seeds | 8 | 6 used (configs 1–6). **Slot 7 = ANCHOR-Lite, slot 8 = A1**, both subject to §7. The no-KD twins are comparators and are not counted. |
| P2, 8 configs × 3 seeds | 8 | 1 ANCHOR-Lite; 2 ANCHOR-Set; 3 A1; 4 A3; 5 A4; 6 A5; 7 reserved for CIR-Lite-R (gated, §9); 8 reserved for a SparK-initialised config (deferred, §9). |

- One training run per config and seed serves both tracks: the same trained ANCHOR-Lite and A1 are scored on the
  national validation blocks and on `val_p2`.
- Configs 2, 4, 5 and 6 of the P2 cap are not national candidates. Their national validation scores are reported
  (rule 6) but cannot be selected there.
- A slot that is not run is recorded as "not run" and still counts. No config is added, swapped or re-tuned. A crashed
  job may be restarted with the identical config and seed only if it produced no validation number; restarts are
  logged.
- Comparator runs are counted separately and listed in full (§5).

## 7. National track (rules unchanged)

- **Nothing queued is delayed.** Selection among configs 1–6 has been run and stands unless slots 7–8 qualify. The
  export, event calibrators, candidate builder, dry run and pre-test review of the selected config proceed now.
- **Slots 7–8 enter the national selection only if** the registry holds, by **2026-11-16 23:59 EST** and before the
  lead authorises the national freeze: (i) the 3-seed validation evaluation of both slots, and (ii) a passing
  export-parity test of the multi-input INT8 pipeline of seed 0. This is decided from timestamps, not from results.
  Otherwise the national track freezes without them and both slots are recorded as unused.
- If they qualify, the frozen selection rule (S; ties |ΔS| < 0.01 → smaller model, pipeline parameters counted as in
  §4.6) is re-run once over the eight configs.
- **Label.** `lite_protocol.md` §8a decides it, unchanged, in code. "State of the art" stays not applicable.
- **Added comparators.** S6–S9, P2a/b and P3a/b are fitted on the selected candidate's own training rows and targets
  (national-2 rows and the four targets if the candidate is one of configs 1–6) and scored in the same single
  national look. **Proposed:** they count as applicable baselines for First-in-class, in addition to the frozen
  list. That can remove a label and can never grant one (ruling requested: Q1b).
- The recovered targets have no role in the national label, selection or confirmatory tests.

## 8. Track P2: rivers and sondes, FNU turbidity at unseen water bodies

### 8.1 Question and endpoint
Can a phone-size model estimate FNU turbidity from 4-band 0.6–1 m imagery at water bodies that contributed no
training data, better than the strongest simple and the strongest published method?
**Primary endpoint:** MdSA of FNU turbidity (metric of `lite_protocol.md` §6) on all eligible `test_p2` chips, with
paired differences against each comparator.

### 8.2 Pool (built from identifiers, coordinates, classes and dates only)
- **Sources.** (a) USGS NWIS sites with a daily FNU series in `processed/sonde_daily` (statistic mean; daily median
  if the site has no mean series; sites with only max/min are out). (b) WQP sites with FNU results in
  `processed/wqp_discrete`. Only identifier, coordinate, site-type and date columns are read; the value columns are
  not loaded.
- **Kept:** CONUS; classes river/stream, canal, lake/reservoir, estuary/coastal (class from
  `matchups.site_class(site_type)`; NWIS codes mapped the same way); dates 2010-01-01 to 2023-09-30.
- **Excluded from the whole pool:**
  - the 729 reserved `sonde_newsites` (`sentinel.data.reserved`; `assert_no_reserved` on every list written);
  - M1 `sonde_split` test-role sites, and every date ≥ 2023-10-01 (M1's time block);
  - M6 Track-N test-role stations;
  - sites inside a lake that has the test role in M2's orbit split v3 (a superset of M2 Test-S);
  - every site of any Lite matchup build (4-state, national, `national_pm7`).
- **Unseen water bodies.** The frozen `water_body_groups` (2 km radius, same name within 40 km, same HUC8 for
  flowing waters, same HydroLAKES lake) is run over all Lite sites ∪ candidate sites. A P2 water body is a group
  with no Lite site in it. So every P2 site is more than 2 km from every Lite site, and a P2 river water body lies
  in a HUC8 with no flowing-water Lite site.
- France, NLA, transparency tube and DOC are not part of P2.

### 8.3 Lock (before any label value is read and before any chip is extracted)
- Role of a P2 water body: `u = int(sha256(salt + "|" + key).hexdigest()[:12], 16) / 16**12`, salt
  `"SENTINEL-Lite-P2|2026-11-10"`, key = the smallest site id (string order) in the group.
  **u < 0.40 → `test_p2`; u < 0.60 → `val_p2`; else `train_p2`.** The salt is never changed.
- The list (salt, rule, every water body with its sites and role) is written to
  `docs/preregistration/lite_p2_lock.json`, set read-only, and its SHA-256 is logged in a `lite/p2_lock` registry row.
- **Lock list SHA-256:** `f3d4c63ebe503ac1774959bfcf943b45bc140883fe5b30933ee03f0cacca52be` (written 2026-11-10
  17:57 UTC, mode 0444; registry row `03323db99ad9`, experiment `lite/p2_lock`; pinned in
  `sentinel/models/litev2/p2.py`). Built by `scripts/data/build_lite_p2.py lock`, which reads only the columns
  listed in the file's `columns_read` entry. The stamp covers this hash.
- Until the P2 look, `test_p2` label values are never read into memory beyond a present/absent flag, never printed,
  summarised or stored. They are joined to the sealed predictions inside the look.
- `train_p2` and `val_p2` values are read only after the lock row exists.

### 8.4 Counts in the lock list (metadata only; before the NAIP match)
Pool: 9,334 CONUS candidate sites (931 sonde sites with a daily FNU series in the window, 9,108 WQP FNU sites) →
7,876 in the four classes → 7,734 after the reserved 729 → 7,527 after M1 test-role sites → 7,239 after M6 Track-N
test stations → 6,571 after Lite-build sites → 6,316 after orbit-v3 test-role lakes → **1,963 sites in 636 unseen
water bodies** (the rest share a water body with a Lite site).

| Role | Water bodies | Sites | Water bodies with a sonde site | River / lake / estuary / canal (main class) |
|---|---|---|---|---|
| `test_p2` | 252 | 738 | 32 | 181 / 49 / 17 / 5 |
| `val_p2` | 122 | 410 | 20 | 96 / 13 / 10 / 3 |
| `train_p2` | 262 | 815 | 34 | 197 / 47 / 16 / 2 |

- Sites by source: WQP only 1,755; sonde 208 (106 of them also have WQP FNU results).
- The number with a NAIP image on a usable date is not known. Plan estimate: 150–400 test chips.

### 8.5 Index, labels and chips
- NAIP items 2010–2023 are matched to site-dates with the national build's code and chip recipe (256², centred on
  the site). The national build's JRC surface-water site filter is **not** applied: it removes the narrow channels
  and small ponds that P2 is about. The eligibility rule below replaces it.
- P2 rows carry the FNU label only. No other target is built at P2 sites.
- **Label:** the sonde daily value on the NAIP date; otherwise the within-day median of WQP FNU results on the
  nearest day within ±1 d. Sonde takes precedence. Bounds as §3.2. Censored WQP rows: Tobit rows in `train_p2`, not
  scored elsewhere (counted).
- **Water eligibility (label-free, same for every model):** the chip lies fully inside the image (the national
  build's validity rule) and holds at least 50 m² of dark-NIR water (`features.water_mask`) within 30 m of the site
  point. Failing chips are dropped before any model sees them and are counted by role.
  - As implemented (`p2_index.water_eligibility`, fixed before the index was built): the extractor status is `ok`
    (centre pixel inside the image); at most 1 % of the chip's pixels are zero in all four bands; and the dark-NIR
    water pixels inside the 30 m circle around the chip centre cover ≥ 50 m² (pixel count × gsd²).
- **Years:** `train_p2` keeps NAIP years ≤ 2021 (the national rule; protects the national time blocks). `val_p2` and
  `test_p2` keep all years to 2023.
- **Disk:** P2 adds at most 0.5 GB. If the index implies more, site-dates are thinned per site in the order of
  `sha256(salt + "|" + site + "|" + date)`, the same in every role, decided from counts.

### 8.6 Power, from counts only
- After the index is built, the number of `test_p2` chips with a present FNU label and the number of their water
  bodies are reported to the lead (counts only) and recorded in a registry row.
- **P2 is confirmatory only if `test_p2` has ≥ 100 such chips in ≥ 40 water bodies.** Otherwise the look is
  descriptive and no label can be claimed. (The plan's floor was 30 chips / 10 water bodies.) This floor was written
  in v0.1, before the index existed.
- **Index counts** (built 2026-11-10 18:11 UTC; registry row `e6a469d28fe1`, experiment `lite/p2_index`; index
  SHA-256 `68688fac1af2c4e147070562f3dde8a035eb150360dd49759619456baf23ff44`; counts only):

  | Role | Site-dates with a label and a NAIP image | Chips not eligible | **Eligible chips with a label** | **Water bodies** | Sites | Label from sonde / WQP | Chips in the largest water body |
  |---|---|---|---|---|---|---|---|
  | `test_p2` | 206 | 17 | **189** | **38** | 75 | 155 / 34 | 48 |
  | `val_p2` | 163 | 9 | 153 | 21 | 70 | 142 / 11 | 91 |
  | `train_p2` | 147 | 17 | 128 | 42 | 64 | 102 / 26 | 12 |

  - Of the 1,963 locked sites, 1,938 have a NAIP image (13,710 site–image-date pairs), but only 516 pairs have a label
    on the image date (sonde) or within ±1 d (WQP). One `train_p2` chip failed to extract, and two rows had no usable
    label (`train_p2` 1, `val_p2` 1). No censored site-date exists in any role, so P2 adds no Tobit row. The
    `test_p2` count is of present values; a value outside the bounds of §3.2 is dropped inside the look.
  - All 189 `test_p2` chips are river/stream sites. P2 as built is a river track; it has no lake or estuary test
    chip. `val_p2` has 1 estuary chip; `train_p2` has 4 lake and 2 canal chips.
  - **The floor is not met: 189 chips, but 38 water bodies (< 40).** The chips are also concentrated: the largest
    test water body holds 48 of the 189 chips, and the effective number of water bodies (1 / Σ share²) is about 10.
  - **Consequence, by the rule above: P2 is descriptive in this round.** Its single look is still run once, hardened
    (§8.9) and reported in full, but no label is claimed on P2 and the label rule of §8.8 is not applied. With the
    lead's ruling at the top of this file, this means **no "state of the art" wording is possible for M7 in this
    round** (ruling requested: Q12).
- Expected precision, scaled from the national validation turbidity comparison (615 chips, 162 water bodies, paired
  95 % CI half-width ≈ 22 MdSA points): about ±44 points at 40 water bodies, ±28 at 100, ±22 at 160. The largest
  validation difference seen so far between a student and a GBM on turbidity is about 27 points. **So P2 can confirm
  only a large difference; a null result will not show equivalence.** This is stated in advance.
- `val_p2` needs ≥ 30 chips in ≥ 10 water bodies for selection (§8.7).

### 8.7 Training, selection and freeze
- All P2 configs and all comparators are fitted on the v2 training rows (national-2b `train` ∪ `train_p2`).
- **Selection:** the config with the lowest `val_p2` FNU MdSA of the 3-seed ensemble, among configs whose pipeline
  meets the budgets; within 1 % relative, the one with fewer pipeline parameters. If `val_p2` is below the §8.6
  minimum, no selection is made and config 1 (ANCHOR-Lite) is the candidate. `val_p2` meets the minimum (153 chips,
  21 water bodies), but one water body holds 91 of its 153 chips; this is stated wherever a `val_p2` number is
  reported.
- No test-block number informs anything. Seeds, B = 2000 and the device come from the freeze record.

### 8.8 Statistics and label rule (decided in code)
- Paired water-body cluster bootstrap (clusters = P2 water bodies), B = 2000, as `evaluate.compare`.
- **Comparator set C** = S1–S9, every P1 form, P2a, P2b, P3a, P3b (§5). No comparator is chosen on validation: the
  carrier must beat each one on the test block (the worst case over the whole list, as in the lead's M1 ruling L2).
- **"State of the art (P2, FNU turbidity)"** iff all hold:
  1. §8.6 confirmatory counts are met;
  2. the carrier (§4.6, within every budget) has a paired MdSA difference below 0 against **every** member of C, with
     each 95 % CI excluding 0 and each Holm-adjusted one-sided p < 0.05 (Holm over C);
  3. seed rule: each of the FP32 seeds 0, 1, 2 has a difference below 0 against the member of C with the lowest
     test MdSA among S1–S9 and against the one with the lowest test MdSA among P1–P3b;
  4. the 3-seed FP32 ensemble also has a difference below 0 against every member of C.
- **"Negative (P2)"** if any member of C is better than the carrier with a CI excluding 0. **"Competitive (P2)"**
  otherwise. A result that holds for the FP32 ensemble but not for the carrier is reported as such and earns no
  label (ruling requested: Q9).
- **Secondary, descriptive, no label:** strata river/canal, lake/reservoir, estuary; sonde-labelled vs WQP-labelled;
  NAIP year ≤ 2021 vs 2022–23; narrow channel (water width at the site < 31 px from the water mask); FNU above the
  training P90 (threshold fixed at freeze from training rows); more than 10 km from any training site; 80 % interval
  coverage and width; every ablation and every comparator. N2 minimum n applies.

### 8.9 Hardened single look (as v1.4–v1.7 of the frozen protocol)
- `docs/preregistration/FREEZE_lite_p2.json` with status FROZEN and the lead's `authorised_by` / `authorised_at`;
  pins for code, configs, checkpoints, comparator artefacts, the lock list, the index and the label sources.
- The claim refuses if any `lite_p2_test*` registry row exists. It writes an exclusive, read-only `TEST_LOOK.json`.
- `test_p2` rows load only in the claiming process (pid and host checked).
- One pass writes every prediction (carrier, seeds, ensemble, ablations, all comparators); the files are sealed
  read-only with `SHA256SUMS`. Labels are joined after sealing. Scoring reads the sealed files only.
- A crash marks the look ABORTED and it stays spent. The write-up goes in a separate `lite_p2_test_results.md`.
- The P2 look is independent of the national look and needs its own authorisation.

## 9. Gates, kill rules and deferred items

| Item | Status under this amendment |
|---|---|
| **CIR-Lite-R** | **Gated, not run.** The calibration gate itself produces a validation number, so it needs its own stamped amendment first. Recorded now so it cannot move: per-scene monotone calibration must beat per-scene linear gain + offset by ≥ 10 % relative RMSE on Sentinel-2 water cells of val ∪ val_n; a physics-feature GBM is fitted before any CIR network result; if the gate fails the arm is closed and counted. P2 slot 7 is held for it. |
| **SparK SSL** | **Deferred, not run.** (SimSiam initialisation was negative on national-2: configs 4 and 5.) P2 slot 8 is held for it and needs its own stamped amendment. |
| P3 France, P4 NLA | Deferred. No build, no score. |
| Transparency tube, DOC, TabPFN | Not in this round. |

Kill and stop rules for the approved arms:
- **K1 tests first.** Before any validation number: unit tests pass for out-of-fold anchors (no water body in both
  fit and predict folds), the Tobit term (finite-difference gradient; limits of large and small σ), augmentation of
  scene quantiles, zero-initialisation (output equals the anchor), the NTRU offset, the anchor exporter and the
  multi-input export.
- **K2 comparators first** (§5), **K3 stamp guard** (§0).
- **K4 budgets** (§4.6): an over-budget pipeline cannot be a carrier.
- **K5 wall clock:** every job ≤ 2 h. A config that cannot finish a seed in one job is checkpointed and resumed; it
  is not shortened.
- **K6 national cut-off** (§7). **K7 P2 counts** (§8.6).
- **K8 reserved data:** any reserved site id in a P2 list, registry row or run record stops the build
  (`assert_no_reserved`).

## 10. Compute, disk and execution

- GPUs 0–9 through `python -m sentinel.core.gpu` (`--min-free-mb 10000`), ≤ 2 h per process, never on another user's
  GPU. CPU jobs sequential, ≤ 6 threads.
- Order after the stamp: comparators (S and P families) → ANCHOR-Lite and A1 → ANCHOR-Set, A3, A4, A5.
- Estimate: networks 6 configs × 3 seeds ≈ 5 GPU-h; DenseNet comparators 8 runs ≈ 11 GPU-h; GBMs < 1 CPU-h. Hard
  ceiling for this amendment: 30 GPU-h.
- Disk: `df` is checked before each build; work stops if free space would fall below 60 GB. New persistent data ≤ 1.0
  GB in total (P2 ≤ 0.5 GB), which keeps M7 inside its 10 GB cap once the national-1 training cache is deleted.
  Chip caches are deleted after each batch of runs. ImageNet DenseNet121 weights (≈ 31 MB) need one download (Q8).
- Registry rows for every run, comparator and count. No git commit by M7. `keys.env` is never read.
- M7 training site-dates on M2's Test-S lakes are reported to the lead as counts only; M7 training is not changed.

## 11. Rulings requested

- **Q1a.** ANCHOR-Lite and A1 as national slots 7–8 with the timestamp cut-off of §7 — or ANCHOR-Lite on P2 only?
- **Q1b.** May the added comparators (S6–S9, P2a/b, P3a/b) count as applicable baselines under §8a for the national
  label, for whichever config is selected? Proposed: yes (harder only).
- **Q2.** Recovered labels on national test-role chips: proposed not built and not reported. Alternative: a
  descriptive FNU/SSC table inside the national look.
- **Q3.** Context set = scene statistics + lat/lon/DOY only (your definition). The plan also listed site class and
  HydroLAKES morphometry; they are left out. Scene width, height and decimation are also left out (50 scene
  statistics + gsd; the plan counted 52).
- **Q4.** TabPFN is not run (not installed; environment pinned). Acceptable?
- **Q5.** Anchor for evaluation rows = refit on all training rows (plan: mean of the five fold models). XGBoost AFT as
  the censored-label parity comparator.
- **Q6.** P2 constants: fractions 40 / 20 / 40 (test / val / train); sonde statistic mean, else median; water
  eligibility 50 m² within 30 m; confirmatory floor 100 chips / 40 water bodies; `train_p2` years ≤ 2021.
- **Q7.** CyFi-style differences (no land cover, no AOT, chip field, frozen GBM parameters); DenseNet 80 epochs and
  the two-value learning-rate rule.
- **Q8.** One download of ImageNet DenseNet121 weights (timm or torchvision) into the model cache. They are not on
  disk.
- **Q9.** P2 carrier = budget-compliant INT8 pipeline; an ensemble-only win earns no label. Do you want a second,
  weaker tier?
- **Q10.** M6 Track-N test-role stations and orbit-v3 test-role lakes are excluded from the whole P2 pool
  (defensive; costs about 5 % of candidate sites).
- **Q11.** Not included, for your decision: FNU sonde/WQP site-dates at *existing* train-role Lite water bodies as
  extra training rows (same N4 rules). They would raise FNU training labels well above the few hundred available
  now, for every model alike, at about 0.3 GB. Default: not used.
- **Q12.** P2 misses the confirmatory floor (38 water bodies; the floor is 40). Proposed: the floor stays, P2 is
  descriptive, and M7 claims no "state of the art" label in this round. Alternatives, not recommended: (i) lower the
  floor before the stamp (only counts were seen, but the bar would move after the counts were known, and the expected
  CI half-width at 38 concentrated water bodies is above ±45 MdSA points); (ii) move `val_p2` water bodies into the
  test block (not clean: `val_p2` values have been read).

## 12. Differences from the plan, and deviations log

| Plan | This file | Why |
|---|---|---|
| Context includes site class and HydroLAKES morphometry | Scene statistics + lat/lon/DOY only | Lead's definition of "context-complete" |
| Transparency-tube and DOC heads on P2 | Not included | Not in the approved list |
| Evaluation anchors = mean of 5 fold models | Refit on all training rows | One anchor file in the deployable; standard stacking |
| TabPFN optional comparator | Not run | Package absent; environment pinned |
| P2 in a new `lite_p2_protocol.md` | P2 is §8 of this file | Lead's dispatch: one amendment file |
| P2 floor 30 chips / 10 water bodies | 100 chips / 40 water bodies | Power (§8.6) |
| P2 roles: 40 % test, rest unspecified | 40 / 20 / 40 with a `val_p2` block; `train_p2` years ≤ 2021 | Selection needs its own unseen block; protects the national time blocks |
| P2 label: beat the two val-designated comparators | Beat every listed comparator on the test block | Lead's M1 ruling L2 |
| FNU as a secondary on Test-N | Not built on national test roles | Lead's ruling on national labels |
| DenseNet 40 epochs | 80 epochs, two backbone learning rates | Parity with the candidate's warm start, in the comparator's favour |
| P2 data ≤ 0.5 GB; "52 scene statistics" | Same cap; 50 scene statistics + gsd | The scene file holds 48 band statistics + 2 fractions |

**Deviations after the stamp** go in a new file `docs/preregistration/lite_arch_v2_deviations.md`, stamped before the
affected run produces a validation number (lead's ruling L4). None so far.
