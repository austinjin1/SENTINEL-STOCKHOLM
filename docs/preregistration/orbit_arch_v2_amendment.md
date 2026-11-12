# SENTINEL-Orbit (M2): architecture round v2 pre-registration amendment

**Status: DRAFT, awaiting the lead's review and stamp** (sha256 + registry row `orbit/prereg_snapshot` + commit).
**Drafted:** 2026-11-10 (EST) by the M2 engineer, from `docs/plans/architecture_research/M2_architecture_plan.md`
(Appendix A.1–A.3, §4, §5, §7, §8) and `docs/plans/architecture_research/LEAD_DECISIONS_20261110.md` (rules 1–6
and the M2 section). Where the two differ, the lead decisions apply.

**What this file is.** An amendment to `docs/preregistration/orbit_protocol.md` v0.4. That file is hashed by the
v1 freeze and is **not edited**. Everything in the protocol that this amendment does not change still applies.
Sections are cited as "arch-v2 §n"; the protocol's own sections as "protocol §n".

**What this file does not change.**
- The headline hypotheses H1/H2, the headline cell, the v1 model, its comparators, its freeze and the `M2/test`
  look. They proceed on their own schedule (chain6b → chain7 → candidate → independent review → lead freeze) and
  are **not delayed** by anything here.
- The bar. RESEARCH_PLAN §15.1 and `docs/plans/sota_campaign.md` rules 1–8 apply unchanged to every arm and track
  below.

## 0. State of knowledge when this was written (disclosure)

Seen before this file was written, and used in the design:
- dev_v4 validation results of v1 and of its comparators, and the diagnostics of the architecture plan §1
  (seed variance, residual correlation with MDN_B and the GBM, per-lake bias). The new arms were designed from
  these. That is why they are tested on a block no model has been scored on (arch-v2 §1).
- Counts only, for protected rows: `protected_cell_counts` in `orbit_split_report_dev_v4.json`; the keys-only
  co-location rates of plan §5 (2024–25 build); row counts, site counts, unit / fraction / characteristic category
  counts and censoring fractions of `processed/wqp_discrete` (no value read by M2); the keys-only counts of
  arch-v2 §1.4.

Not in existence when this file was written:
- any validation or test number for arm A0 or A1, or for any model on TP, TN, DOC, FNU, SSC or aCDOM;
- any code for the new arms, the multi-target join or the Test-S loader;
- the dev_v5 data version (the 2015–2023 WQP matchup build was still running) and therefore dev_v5m;
- any evaluation of any model on a Test-S row. Test-S rows are protected rows (protocol §2.2): every development
  cache dropped them right after split assignment, no scorer reads them, and they were excluded from the SSL pixel
  manifest and from GLORIA pretraining;
- the `M2/test` look (it has not run; no test result of any kind exists for M2).

**Before the stamp** (lead rule 2): code, tests, data builds (protected rows: counts only) and smoke tests on
synthetic or train-role data only. No validation number on a new arm or on a Track N target.

---

## 1. Test-S: unseen lakes × archive years (new confirmatory block)

### 1.1 Definition
Cell **`testxarchive_lake`**: in-lake, non-buffer rows of **test-role** water bodies (frozen split v3, protocol
§2.2) with in-situ date **2015-06-23 to 2023-12-31** (the split's train and val time blocks), from every chip
source (AquaCorr chips, WQP 2015–2023 matchups), under the identical image QC and de-duplication of protocol §2.1.

Companion cell `testxarchive_nonlake` (test role, non-lake 200-km blocks, same dates): scored in the same look,
**descriptive only**, as the headline look does for its non-lake cell.

### 1.2 Practical reason (written before any result)
The main real use of a satellite water-quality archive is **retrospective reconstruction**: 2015–2023 histories
for lakes that were never monitored (trend assessment, trophic-status listing, restoration baselines). "Unseen
lakes, archive years" is exactly that use. The block also addresses the power problem recorded in protocol
§5a/§9a: the headline chl-a cell has 87 role blocks.

### 1.3 Scope of any claim
- A result on Test-S is labelled **"unseen lakes, archive years (2015–2023)"**. It tests spatial generalisation.
  Its years overlap the training years, so it is **not** a claim about future years.
- The headline (unseen lakes × 2024–2025) remains the strict claim. Test-S never replaces it, and both are reported
  whatever the outcome.

### 1.4 Counts (protected rows: counts only)

Labelled rows per target (dev_v4 split report; all AquaCorr-sourced):

| Target | test role × ≤ 2021 (rows / lakes / blocks) | test role × 2022–23 | Test-S total, dev_v4 | Headline, for reference |
|---|---|---|---|---|
| chl-a | 2,001 / 510 / 143 | 638 / 243 / 79 | **2,639 rows** | 669 / 225 / 87 |
| Secchi | 1,725 / 381 / 88 | 447 / 154 / 41 | **2,172 rows** | 1,442 / 394 / 92 |
| TSS | 79 / 24 / 23 | 13 / 6 / 6 | 92 | 221 / 79 / 57 |
| turbidity | 36 / 17 / 15 | 13 / 8 / 6 | 49 | 490 / 98 / 53 |

Keys-only counts made on 2026-11-10 (no label value read; site, date, chip and target-name keys only). Script
`scripts/data/orbitv2_testS_keycounts.py`; output `docs/preregistration/orbit_testS_keycounts_20261110.json`;
registry experiment `orbit_archv2_keycounts`.

| Source | Stage | Site-dates | Water bodies | Role blocks |
|---|---|---|---|---|
| AquaCorr chips, any target | chip present, image QC passed, de-duplicated | **3,790** | 693 | 169 |
| — of which dated ≤ 2021 / 2022–23 | same | 2,909 / 881 | 628 / 285 | 155 / 82 |
| — of which lakes < 1 km² | same | 886 | 320 | 91 |
| WQP 2015–2023 samples, any base target | **upper bound**: before the same-day scene match, chip extraction and image QC | 81,992 | 1,585 | 311 |
| — chl-a | upper bound, as above | 31,640 | 1,154 | 282 |
| — Secchi | upper bound | 71,059 | 1,385 | 262 |
| — TSS | upper bound | 14,887 | 545 | 206 |
| — turbidity (NTU family and FNU not yet separated) | upper bound | 21,762 | 618 | 233 |
| — lakes < 1 km², any base target | upper bound | 24,540 | 911 | 212 |

- The AquaCorr chl-a rows lie in 143 to 169 role blocks.
- The WQP rows are an upper bound only: most sample dates have no same-day Sentinel-2 acquisition, and chl-a
  image-QC pass was about 58% in the 2024–25 build. The WQP 2015–2023 matchup build was 75% complete when this was
  written.
- The final per-target counts come from the dev_v5m split report (arch-v2 §4.5) and are appended to this file as a
  dated, counts-only addendum.

### 1.5 Power (from protected counts only)
Rough scaling of the registered protocol §5a standard error (`orbit_power_dev_v4`: chl-a MDE80 10.3 MdSA points
against MDN_B and 14.3 against the GBM at 87 blocks) by √(87/B):

| Role blocks B | chl-a MDE80 vs MDN_B | chl-a MDE80 vs GBM | Basis |
|---|---|---|---|
| 87 | 10.3 | 14.3 | headline, registered |
| 143 | ≈ 8.0 | ≈ 11.2 | AquaCorr, ≤ 2021 rows only |
| 169 | ≈ 7.4 | ≈ 10.3 | AquaCorr, upper bound on blocks |
| 200 | ≈ 6.8 | ≈ 9.4 | if the WQP 2015–2023 rows add about 30 blocks |
| 282 | ≈ 5.7 | ≈ 7.9 | WQP chl-a sample keys, upper bound on blocks (before chips and QC) |

- The scaling assumes the per-block row distribution and the paired-difference variance of the dev_v4 val cell.
- No power "at the validation gain" is stated for a new arm, because no validation number of a new arm exists.
- **Binding calculation.** After the stamp, and before the Test-S freeze, `orbit_power.py` (protocol §5a method:
  validation per-row predictions plus protected **counts**) is re-run for each arm, endpoint and comparator on
  dev_v5m. It is registered as `orbit_power_S_dev_v5m`. It is a design quantity, never a result.

### 1.6 Scene-disjointness rule
A row of Test-S can share its Sentinel-2 scene with a training row of another lake. Atmospheric-correction error is
largely scene-specific, so the rule below fixes which rows are primary.
- **Scene key** = (MGRS tile, sensing date) of the row's chip, read from the STAC item id.
- **Reference set** = the scene keys of every fit-cell and inner-cell row of dev_v5m (any target, any water type).
- For each endpoint target, count the Test-S rows whose scene key is not in the reference set.
- **Rule.** If that scene-disjoint subset keeps **≥ 60% of the rows and ≥ 80% of the role blocks** of the chl-a
  cell, the scene-disjoint subset is the **primary** cell for every target and all rows are the sensitivity
  analysis. Otherwise the roles swap: all rows are primary and the scene-disjoint subset is the sensitivity.
- The decision is made **from counts only, before the freeze**, and written into the Test-S freeze candidate.
- Also reported (no role in the rule): the same counts with the val-cell scenes added to the reference set.
- Preview, AquaCorr keys only, any target (2026-11-10): 2,635 of 3,790 site-dates (69.5%) and 159 of 169 blocks
  (94.1%) are scene-disjoint from the AquaCorr fit and inner rows. The WQP 2015–2023 rows will add reference scenes,
  so the binding fractions will be lower.

### 1.7 Small-lake stratum
In-lake rows with HydroLAKES `Lake_area` < 1 km² (the protocol §11.4 definition), same family form as §11.4
(4 targets × the 3 H1 comparators, Holm within the family, per arm).
- It is **confirmatory only if** the binding chl-a MDE80 of arch-v2 §1.5, computed for this stratum, is
  **≤ 10 MdSA points against every one of the three H1 comparators**. Otherwise it is descriptive.
- Its status is fixed in the Test-S freeze candidate before the look.

### 1.8 Reservation and timing
- Reserved for all workstreams in `docs/plans/sota_campaign.md` ("Reserved unseen blocks added 2026-11-10"). No
  model that takes Sentinel-2 input, and no model fused with M2, is trained or tuned on these rows.
- Declared before the `M2/test` look runs, so the declaration is independent of any test result. Nothing in this
  file may be changed in response to the `M2/test` outcome. The only later additions are the count-determined
  items named here (arch-v2 §1.4, §1.6, §1.7, §2.6), each as a dated addendum written before the look.

---

## 2. Track N: new numeric targets on the Test-S block

### 2.1 Targets
| Target | Unit | Valid range (a priori) | Role |
|---|---|---|---|
| **TP** total phosphorus | mg/L as P | 0.001–10 | Track N family |
| **TN** total nitrogen | mg/L as N | 0.01–50 | Track N family |
| **DOC** dissolved organic carbon | mg/L as C | 0.1–200 | Track N family |
| Turbidity FNU | FNU | 0.05–5,000 | auxiliary head; descriptive |
| SSC | mg/L | 0.1–5,000 | auxiliary head; descriptive (mostly non-lake) |
| aCDOM(440) | 1/m | 0.01–50 | auxiliary head; descriptive (see arch-v2 §3.3 item 8) |

- Values outside the valid range are set missing, never clipped (as protocol §2.1).
- TOC is not a target and is not fitted. Phycocyanin, microcystin and surface temperature are excluded (plan §5).

### 2.2 Practical reason (written before any result)
Nutrient and organic-carbon status of unmonitored lakes: eutrophication listing, TMDL screening, drinking-water
source assessment. TP and TN are optically inactive, so any skill comes through trophic coupling. They are
**harder** targets, and a location/season model or the chl-a chain (arch-v2 §2.5) may well be the best available.

### 2.3 Block
The Test-S rows (`testxarchive_lake`, primary or sensitivity as fixed by arch-v2 §1.6) that carry a value of the
target. Test-role 2024–2025 rows are **not** used for any new target: those lakes and scenes are spent by the
headline look.

### 2.4 Label rules (applied identically to every model)
- **Source.** `processed/wqp_discrete` variables `tp`, `tn`, `doc`, `turbidity_fnu`, `ssc`, `acdom440`, in the
  units that build harmonised. For AquaCorr rows, AquaCorr's own co-located `doc_value` and `acdom440_value` are
  also used after a unit check.
- **Fraction.** TP and TN: `fraction_class = total` only. DOC: dissolved (all rows of the table).
- **Surface rule and replicates.** As `sentinel.data.matchups.wqp_surface_samples`: QC activities and rejected
  results dropped; depth ≤ 1 m, or unspecified and not flagged bottom; replicates → median per site, local date
  and characteristic.
- **Join.** WQP matchup rows: same WQP site and same local date. AquaCorr rows: the nearest WQP station within
  100 m on the same date (the protocol §2.1 de-duplication radius); a WQP value is preferred over an AquaCorr value.
- **Censoring.**
  - A left-censored result ("<" with a detection limit DL inside the valid range) enters every distributional
    model through the **interval likelihood**: the mixture CDF at log10 DL. Point-estimate baselines (GBM, OLS)
    are fitted with the substitution DL/√2.
  - Right-censored (">") results, and censored results without a DL, are dropped.
  - A site-date with both quantified and censored replicates uses the quantified median.
  - **Scoring.** MdSA and the other protocol §5 metrics use quantified, in-range rows only. Censored Test-S rows
    are counted, and each model's P(y ≤ DL) on them is reported descriptively.
- **Protected rows.** New-target values of protected rows are dropped right after split assignment, exactly as the
  base targets are. Only their counts are written.

### 2.5 Comparators (all fitted on the identical fit cell and data version; parity rules of arch-v2 §3.5)
For each of TP, TN and DOC, the claim is an intersection–union test against **every one** of:
1. the MDN_B family member selected on validation for that target;
2. the GBM family member selected on validation for that target;
3. location kNN (k from {4, 8, 16, 32});
4. the season + location GBM (no imagery);
5. the **trophic-chain baseline**: ordinary least squares, fitted on the fit cell, of the log target on the
   validation-selected MDN_B's predicted log chl-a and log Secchi (the classic chl-a–nutrient coupling). It uses
   the same form for all three targets.

**Selection cell for the comparators of a new target:** `valxval_lake` if it holds ≥ 200 quantified rows in ≥ 20
role blocks of that target; otherwise `valxany_lake`. (Validation rows are not protected; this is a count rule.)

### 2.6 Family, multiplicity and the MDE rule
- **Family:** TP, TN, DOC, with Holm across the family (arch-v2 §6.3).
- **MDE rule.** A target whose binding MDE80 (arch-v2 §1.5 method, with its own protected counts) **exceeds 15
  MdSA points against any comparator of arch-v2 §2.5** is pre-declared **descriptive** and leaves the family.
- The validation predictions that this calculation needs exist only after the stamp. The rule and the threshold are
  fixed now; each target's status is fixed in the Test-S freeze candidate, before the look, from validation
  predictions and protected counts only.
- Expected counts (keys only, plan §5): TP co-located at about 87% of chl-a matchup site-dates in the 2024–25 build,
  TN about 37%, DOC about 12%.
- Keys-only preview on Test-S, 2026-11-10. It is an **upper bound** (WQP 2015–2023 sample keys, before the same-day
  scene match, chips and image QC): a same-site, same-date sample of the new target exists at

  | Target | Test-S chl-a site-dates with the target (of 31,640) | Water bodies | Role blocks |
  |---|---|---|---|
  | TP (total) | 29,518 (93%) | 1,131 | 276 |
  | TN (total) | 8,724 (28%) | 435 | 182 |
  | DOC | 2,304 (7%) | 328 | 165 |
  | Turbidity FNU | 3,478 | 23 | 12 |
  | SSC | 1,161 | 31 | 12 |

  FNU and SSC sit in very few lake blocks, which is consistent with their descriptive role.
- The binding Test-S counts are appended with arch-v2 §1.4.

### 2.7 Which model is scored on Track N
The multi-target arm: the A1 configuration with the lowest validation chl-a MdSA (the protocol §6 rule restricted
to A1), provided A1 passed its kill criterion (arch-v2 §3.7).
- If that configuration is also v2 (arch-v2 §3.6), nothing is added.
- If v2 is A0, that A1 configuration is scored on the Track N targets; its base-target results on Test-S are then
  reported descriptively, with no claim role.
- If A1 failed its kill criterion, Track N is reported with the comparators only, and no SENTINEL claim is made
  on it.

---

## 3. New arms

### 3.1 Common rules
- **Data version:** dev_v5m (arch-v2 §4). Every model in a comparison, comparators included, is fitted on it.
- **Cells:** the fit, inner and selection cells of protocol §2.2, unchanged.
- **Pretraining and initialisation.** v1's GLORIA stage, unchanged. A0 and A1 share one initialisation recipe:
  that of the dev_v5 configuration selected by protocol §6/§6a/§11.6 at the moment the first A0 network starts
  training (recorded in the run configuration). If chain9 has selected an SSL configuration by then, both arms
  inherit its encoder initialisation. A later chain9 result does not change either arm.
- **No +GBM blend** is formed for any new arm.
- **Counting.** Every configuration scored on validation is appended to `configs_tried.jsonl` (235 rows on
  2026-11-10) and the count is reported next to any claim.
- **Compute.** One process per network, ≤ 2 h, checkpoint every 5 epochs, GPUs 0–9 through
  `sentinel/core/gpu.py` (`--min-free-mb`), never a GPU that holds another user's process. fp16 AMP; no bf16.
  CPU ≤ 6 threads per job.
- **Every arm and configuration is reported**, win or lose.

### 3.2 Seed unit (lead-approved with parity)
- A **seed unit** is 3 independently initialised networks. Each network keeps **EMA weights** (decay 0.999,
  started at the first fine-tuning step from the pretrained weights). The EMA weights are used for inner-cell early
  stopping and for inference. The patience-15 inner rule of v1 is unchanged.
- The unit's predictive distribution is the equal-weight **average of the 3 mixtures**; its point prediction is the
  median of that averaged mixture.
- Each arm configuration trains **3 seed units = 9 networks** (network seed = 100 × unit + init, units 0–2,
  inits 0–2). The reported model is the equal-weight mixture average of the 3 units.
- **Per-seed rule.** RESEARCH_PLAN §15.1(iii) (≥ 3 seeds, each individually better than the comparator on the
  point estimate) applies to **seed units** for the new arms. Reason: the published comparator's unit is itself an
  ensemble (MDN: 10 initialisations). The 9 single-network results of each configuration are also reported,
  descriptively.
- v1 keeps its own definition at every look (single networks, 5 seeds, as frozen).

### 3.3 Arm A0: variance control (cap: 1 configuration)
- **What.** The v1 architecture and training recipe exactly as selected on dev_v5 by protocol §6/§6a (the plain
  network of the frozen headline configuration, no blend), trained as seed units (arch-v2 §3.2). Four base targets
  only. 9 networks.
- **Practical reason.** On dev_v4 validation, 2 of 3 single seeds failed the per-seed rule against MDN_B while the
  seed ensemble passed on the point estimate. A deployed retrieval is an ensemble, so the unit of evidence should
  be one. A0 adds no information source. It is also the **control that attributes A1's gain**: A1 − A0 is the
  effect of context, ring and multi-target heads; A0 − v1 is the effect of the seed unit and EMA.

### 3.4 Arm A1: SatWQ-LakeSet-MT (cap: 3 configurations)
**Practical reason.** An agency reconstructing a lake's history holds that lake's whole image archive. Other
acquisitions of the same lake let a model separate the lake's persistent optical state from the scene-specific
atmospheric-correction error of one date. Shore adjacency is the main error source in small lakes. Managers need
nutrients and organic carbon next to chl-a and clarity.

**Architecture** (about 0.3 M parameters; v1 has 182,780). Every value below is fixed now; nothing is tuned.
1. **Pixel encoder: v1, unchanged.** Band token = MLP(asinh(ρ/0.01)) + Linear(sinusoidal SRF-centroid λ);
   3-layer, d = 64, 4-head pre-norm Transformer with a CLS token. The pixel-offset MLP gains one input,
   **log1p(d_p / 10 m)**: the pixel's distance to the nearest non-water pixel, from a distance transform of the
   SCL water mask over the full 64 × 64 chip (capped at the distance to the chip edge).
2. **Date embedding** = v1 attention pooling + mean pooling over the date's water pixels: up to 64 pixels for the
   target date, **8 for each context date**.
3. **Lake-context set C(w, t)**: imagery only, rule fixed now.
   - *Members.* Other chips of the same water body (in-lake rows: the same HydroLAKES id; non-lake rows: chips
     centred within 100 m of the row's point) that pass the image QC of protocol §2.1 and whose sensing date t′
     satisfies **3 d ≤ |t′ − t| ≤ 183 d**. Chips come from any existing chip store. **Only chip arrays and image
     metadata are read, never a label.** Buffer-excluded chips are never used.
   - *Date limit.* For every row dated ≤ 2023-12-31 (all development rows and all Test-S rows), context chips are
     dated **≤ 2023-12-31**. No chip of the headline years enters development or Test-S.
   - *One chip per date:* the chip nearest to the row's point (tie: lowest chip id). Same-date chips are excluded.
   - *Choice.* Up to **K = 6** dates: lowest Sen2Cor AOT first, ties by smaller |Δt|, then the earlier date.
   - *Token* for context date d: e_d + MLP(q_d) + MLP(Δt_d), with q_d = (cos SZA, cos VZA, cos RAA, AOT, glint
     angle, processing-baseline group {< 04.00, 04.xx, ≥ 05.00}).
   - *Aggregation.* 2 Set-Transformer blocks (SAB, 4 heads), then PMA with 1 seed vector → lake embedding ℓ.
   - *Training.* The whole set is dropped with p = 0.3; otherwise a random subset of 1…K members is used. A row
     with no context uses a learned "empty" embedding.
   - **No context-count feature** is given to any model. The count reflects monitoring intensity.
   - The context looks both ways in time. That fits retrospective reconstruction. It is not a nowcasting design,
     and no nowcasting claim is made from it.
4. **Land ring** (one vector per chip): MLP over the asinh median L2A spectrum (12 bands) of SCL vegetation / bare /
   unclassified pixels 30–300 m from the chip's water pixels, the land fraction of the chip, the land NDVI, and a
   "ring missing" flag (fewer than 5 ring pixels).
5. **Geometry:** v1 geometry, plus cos 2ω (sun–view angle), the specular glint angle and the Fresnel reflectance at
   ω for n = 1.34.
6. **Context:** DOY sin/cos. **Latitude/longitude only in configuration c1.**
7. **Fusion:** LayerNorm → MLP([chip_t, ℓ, chip_t − W·ℓ, geometry, ring, context]) → h (d = 64).
8. **Heads.** One K = 5 Gaussian mixture per target in standardised log10 (fit-cell mean and SD). Loss = masked
   NLL, weighted: chl-a, Secchi, TSS, turbidity-NTU **1**; TP, TN, DOC **0.5**; FNU, SSC, aCDOM(440) **0.25**.
   Left-censored values enter through the mixture CDF (arch-v2 §2.4). An auxiliary head whose fit cell holds
   fewer than 200 values is dropped before any training, and that target is then reported as counts only.
9. **Early stopping:** the v1 inner-cell criterion, computed on the four base targets only, with EMA weights.
10. **Seed units** as arch-v2 §3.2; initialisation as arch-v2 §3.1.

**Configurations (all three selectable, all counted).**
- **c1**: full model, with latitude/longitude.
- **c2**: full model, **coordinate-free** (also the Track-G candidate).
- **c3**: no context set (ring + d_p + multi-target heads + seed units), for attribution.

Each is 9 networks. Learning rate, batch size (128), epochs, patience, dropout and augmentation are those of the
selected v1 configuration.

**A-priori strata reported for A1:** rows with ≥ 1 context date versus none.

**Stated risks.**
- Context availability follows the sampling design (a chip exists near a date because someone sampled then). The
  rule is deterministic, no count is given, and the stratum above is reported. An imagery-only context bank from
  STAC would remove the dependence; it needs new imagery and is **not approved**.
- Multi-target negative transfer on the base targets. c3 against A0 measures it.

### 3.5 Comparators and parity (lead rule 3; mandatory)
Every comparator is fitted on the identical fit cell, rows, labels and data version as the arm it faces.

**Same ensembling as the candidate.**
- Every learned comparator is an ensemble of at least as many members as the candidate at both levels of
  comparison (the seed unit, 3 networks; the reported model, 9 networks).
- **MDN_B**: 10 initialisations per member, as published (≥ 9). Both read-outs are family members: the published
  one (median over initialisations of the top-component estimate) and the candidate's (median of the averaged
  mixture).
- **GBM** (every GBM, including season + location): each grid member is fitted as a **3-seed bag**; the
  validation-selected member is extended to a **9-seed bag** (3 × 3) for the comparison.
- Each seed unit of an arm is compared against the **full** comparator (10-initialisation MDN_B, 9-seed GBM bag).
- kNN, index algorithms and the trophic-chain OLS are deterministic.

**Same inputs.**
- **GBM family** = the existing 16-configuration grid × {v1 features; v1 features + ring + context summaries}
  × {single-target; **multi-target by stacking**}.
  - *Ring + context summaries:* the land-ring vector; the glint features; the median and minimum d_p of the centre
    water pixels; and the per-band median and IQR over the same K context chips' 3 × 3 water medians (the same set
    C(w, t); no count feature).
  - *Multi-target by stacking:* out-of-fold predictions (5 folds grouped by role block within the fit cell) of
    every other target, from the inner-best single-target GBM, are added as features. This gives the GBM the same
    auxiliary targets as the candidate (lead rule 3).
- **MDN_B family** = {7-band, 12-band, 12-band + context median + ring} × {5k, 10k, 20k iterations} × {linear, log
  input} × {single-target; **joint multi-target head** (the Pahlevan et al. 2022 form, with the interval
  likelihood for censored values)}.
- **Location kNN**: unchanged.
- Within each family the member with the lowest validation MdSA on the target's selection cell is the comparator.
  This is optimistic for the baselines. Each family has at least as many validation-scored members as the arms it
  faces (4).
- The same comparator set is used for v1 and v2 at the Test-S look.
- **Descriptive only, no claim role:** the average of MDN_B and GBM; single-network results of the arms.

### 3.6 Selection rule
- **v2** = the configuration with the lowest validation chl-a MdSA on `valxval_lake` (protocol §6) among A0, the
  A1 configurations (only if A1 passed its kill criterion), and any gated arm that was funded and run before the
  Test-S freeze. The compared quantity is the 3-unit ensemble. Tie within 0.05 points: the simpler arm, in the
  order A0, c3, c2, c1.
- **v1** is the configuration frozen for the headline. It is carried to the Test-S look unchanged, with the same
  weights.

### 3.7 A1 kill criterion (a development gate, not a selection rule)
A1 is carried to test only if its lowest-validation-chl-a configuration is **≥ 2 MdSA points better than v1**
(as frozen for the headline, dev_v5) on **both** validation chl-a and validation Secchi (`valxval_lake`, common
rows). Otherwise A1 stops, v2 = A0, and all A1 validation results are still reported.

### 3.8 Gated and not funded (listed so that nothing is added silently)
| Arm | Cap | Status |
|---|---|---|
| **A2** SatWQ-IOP-lite (physics decoder on the A1 encoder) | 1 selectable + 2 non-selectable ablations | **Gated, not funded.** Gate as written in plan §4.3: it may run only if a chain9 SSL configuration improves validation chl-a by ≥ 1 point over its non-SSL twin, **or** A1 fails its kill criterion. It then also needs the lead's funding decision, every physical constant checked against its primary source, and a GBM-phys member (QAA / Lee features) added to the GBM family. |
| **A3** SatWQ-TOA dual stream | 1 | Not funded (needs new L1C imagery). |
| **A4** AlphaEarth lake prior | 1 (ablation) | Not funded. |

- No validation number of A2, A3 or A4 is produced before a dated addendum to this file (architecture text,
  comparators, cap) is stamped by the lead.
- An arm that is not frozen before the Test-S look can be tested only on the prospective Test-C block.

---

## 4. Data version dev_v5m

1. **Definition.** dev_v5 plus the co-located multi-target join of arch-v2 §2.4, under the same split v3, image QC
   and de-duplication.
   - Every dev_v5 row is kept unchanged. The build asserts that row ids and the four base-target values are equal
     to dev_v5.
   - Added columns: the new-target values, censoring flags and detection limits; per-pixel d_p; the land-ring
     vector; glint features; the context-chip index C(w, t).
   - Added rows: chip-bearing, QC-passing, de-duplicated site-dates that carry only new-target values. No new
     imagery is requested.
2. **Protected rows** are dropped right after split assignment, as in `data.build_table`. The development context
   cache holds only chips of train-role or val-role water bodies dated ≤ 2023-12-31, and its loader refuses any
   other chip. Context chips of test-role water bodies are read only inside the open Test-S claim (arch-v2 §5).
3. **Turbidity units (amends protocol §2.1 for the new arms only).** FNU values may be **fitted** by the new arms
   and by their comparators, as a separate FNU target; NTU and FNU are still never mixed. The headline arm is
   unchanged. FNU is scored only on Test-S years: test-role 2024–25 FNU is spent as a descriptive stratum of the
   headline look.
4. **Chl-a method-class covariate** (plan §5 note): **not adopted** in this amendment. No model receives it.
5. **Outputs.** Manifest `manifests/orbit_dev_v5m.json`; split report
   `docs/preregistration/orbit_split_report_dev_v5m.json` with, as **counts only**: the per-target protected cell
   counts, including `testxarchive_lake` and its scene-disjoint subset, the small-lake stratum, censored rows, and
   context availability (rows with 0, 1–5 and 6 context dates).

---

## 5. Code, guard and freeze for the Test-S look

- **Separation from the v1 freeze.** The v1 scoring-code hash covers `sentinel/models/orbit/*.py`,
  `scripts/train/orbit_*.py`, `scripts/eval/orbit_*.py` and `configs/orbit/*.yaml`. So that v2 work can never
  change or invalidate the v1 candidate, all new code lives outside those globs: `sentinel/models/orbit_v2/`,
  `scripts/train/orbitv2_*.py`, `scripts/eval/orbitv2_*.py`, `configs/orbit_v2/`. No file hashed by the v1 freeze
  is edited before the `M2/test` look is sealed.
- **Look `M2S/test`.** Its own freeze file `docs/preregistration/FREEZE_orbit_S.json` (written only by the lead,
  from a reviewed `orbit_S_freeze_candidate.json`) and its own registry experiment. It applies every item of
  protocol §9b and of `sota_campaign.md` rule 7:
  - only the lead-written freeze unlocks protected rows;
  - the look refuses to start if any row of its experiment is in the registry;
  - content-hash pins of every data file the protected loader reads, now including the context chips and the
    `wqp_discrete` tables, verified before the claim;
  - every artifact verified before the claim;
  - an O_EXCL claim bound to pid and host;
  - sealed predictions with SHA256SUMS and a `predictions_sealed` registry row; access closes at sealing;
  - rescoring refused once a scoring row exists;
  - any BaseException between claim and seal aborts and spends the look;
  - a scoring-code hash over the full import closure (the v1 files it imports and all `orbit_v2` files).
- The hardened primitives of `sentinel/models/orbit/testlook.py` are imported, never copied.
- **Context chips, ring features and d_p of protected rows are computed only inside the open claim.**
- The freeze candidate records: this file's sha256; the scene-disjointness outcome; the MDE status of the
  small-lake stratum and of each Track N target; v1's weight hashes (equal to those in `FREEZE_orbit.json`); all
  v2 and comparator artifacts.
- A development dry run on the validation cell precedes the freeze, and the look code gets an **independent code
  review** before the lead writes the freeze.

---

## 6. The look `M2S/test`: endpoints and multiplicity

### 6.1 Arms scored
**v1** and **v2**, on the same rows, against the same comparators (arch-v2 §3.5). The look is run once.

### 6.2 Primary endpoint H1-S (chl-a, primary cell of arch-v2 §1.6)
- For each arm: MdSA lower than **each** of the validation-selected MDN_B family member, the validation-selected
  GBM family member and location kNN (intersection–union), with the paired **role-block** bootstrap of protocol §5
  (B = 2,000; the water-body bootstrap is a sensitivity), and the per-seed rule of arch-v2 §3.2.
- The arm's p-value is the largest of its three two-sided paired bootstrap p-values.
- **Holm over the 2 arms:** the smaller p-value is tested at 0.025 and the other at 0.05.
- A §15.1 label needs: Holm-adjusted p < 0.05, the difference in the arm's favour against every comparator, and the
  per-seed rule. The label is scoped as arch-v2 §1.3.

### 6.3 Secondary and track families (each with Holm; all reported)
| Family | Hypotheses | Holm over |
|---|---|---|
| H2-S | Secchi, TSS, turbidity-NTU × 2 arms; same intersection–union form | 6 |
| Track N | TP, TN, DOC × the multi-target arm (arch-v2 §2.7); comparators of arch-v2 §2.5 | ≤ 3 (targets declared descriptive by §2.6 leave) |
| Small lakes | as arch-v2 §1.7 | per arm, as protocol §11.4 |
| Events | E4a, derived route, E-H1/E-H2 of protocol §12 on `testxarchive_lake`; v2's event artifact is refitted with the same specification | secondary; protocol §12.4 power rule |

### 6.4 Descriptive, pre-declared
- FNU, SSC and aCDOM(440); `testxarchive_nonlake`; censored-row detection.
- Strata: ≤ 2021 versus 2022–23; AquaCorr versus WQP source; context ≥ 1 date versus none; rows more than 10 km
  from any train-role point; the sensitivity cell of arch-v2 §1.6.
- Interval coverage and width, raw and split-conformal (protocol §5, §5b, applied to v2 and to the
  validation-selected MDN_B with offsets fitted on validation).
- Single-network results; the MDN_B + GBM average; every A1 configuration not selected (validation only).

---

## 7. Reporting
- Every arm, configuration, track and family above is reported, wins and losses, each scoped to its track.
- Every number goes through `sentinel.core.registry`; every run through `sentinel.core.runlog.TrainingRun`.
- The write-up goes to `docs/preregistration/orbit_S_test_results.md`, never into a frozen file.

## 8. Points the lead should confirm or change before stamping
Each has a default written into the text above.
1. **Scene-disjointness reference set** (arch-v2 §1.6): fit + inner rows, as in the plan. Val-cell scenes are only
   reported. Alternative: add them to the reference set.
2. **Binding MDE after the stamp** (arch-v2 §1.7, §2.6): the Track N and small-lake MDE need validation
   predictions, so only the rule is fixed now.
3. **Context date limit and two-sided context** (arch-v2 §3.4 item 3): context ≤ 2023-12-31 for all development
   and Test-S rows.
4. **Parity reading** (arch-v2 §3.5): MDN_B keeps 10 initialisations (≥ 9) with both read-outs as members; GBMs are
   3-seed bags, 9-seed for the selected member; each seed unit faces the full comparator.
5. **Multi-target-by-stacking GBM member** (arch-v2 §3.5): added by me under lead rule 3 ("same auxiliary
   targets"); it is not in the plan.
6. **Track N arm when v2 = A0, or when A1 fails its kill criterion** (arch-v2 §2.7).
7. **Shared initialisation fixed when A0 starts** (arch-v2 §3.1). The plan's wording was "A1 inherits the chain9
   SSL initialisation if chain9 selects one"; chain9 may report after A0 starts.
8. **aCDOM(440) head** (arch-v2 §3.4 item 8): the plan lists aCDOM both as a GLORIA pretraining head and under
   "pretraining unchanged". Default: fine-tuning auxiliary head only, dropped if < 200 fit values.
9. **H2-S family** (arch-v2 §6.3): TSS and NTU stay in the family although their Test-S counts are small (92 and
   49 rows in dev_v4). Alternative: apply the MDE > 15 rule to them as well.
10. **Chl-a method-class covariate**: not adopted (arch-v2 §4 item 4).
11. **Code separation** (arch-v2 §5): new modules outside the v1 globs instead of editing `testlook.py` /
    `orbit_test.py` in place, so the v1 candidate hash cannot move.
12. **Early stopping on the four base targets only** (arch-v2 §3.4 item 9), and the a-priori valid ranges of
    arch-v2 §2.1.
