# D7 TroutRefuge — ecoregion-transfer track: design note (counts only)

**Status:** design note v0.1, written 2026-11-10 (19:40 UTC) at the lead's request (ruling of 2026-11-10 14:25 EST,
item 6). **Not yet a protocol.** Nothing in this track has been run: no transfer model has been fitted and no
transfer score exists on any cycle. If the lead approves, the design is copied as a dated amendment into
`downstream_troutrefuge.md` §8 **before** the first transfer fit, and both tracks are frozen in one FREEZE record and
looked at in one event. No freeze candidate is built here.

**What was read to write this note:** label counts of the training cycles (2008–09, 2013–14) and the validation
cycle (2018–19) only. Test-cycle rows (2023–24) were dropped (`common.drop_protected`, `assert_no_test`) before any
count; **no test value, test label or test count was read** — not even label-free unit counts. Validation-cycle
counts stand in for the unknown test-cycle counts throughout. Scripts and raw counts (scratchpad):
`d7/eco_counts.py`, `d7/eco_counts2.py`, `d7/eco_counts.json`.

## 1. Question
The main track asks: does the challenger predict coldwater-fish occupancy at **new locations inside ecoregions that
have fish surveys**? This track asks the harder question: does it predict occupancy in **an ecoregion that has no fish
survey at all**? Each ecoregion is held out in turn; the model that scores a reach has never seen a labelled reach of
that reach's ecoregion, and the scored visits come from a later survey cycle than every fitted visit.

## 2. Prior knowledge (disclosed; this is why the track exists)
The track was proposed **after** these validation and training-cycle numbers had been seen:
- Main track, validation new locations (n 806): LightGBM **0.9515** against RF occupancy incumbent **0.9441**, paired
  Δ **+0.0074 [−0.0027, +0.0188]**, p 0.16 → tie, "competitive". All first visits (n 1,594): Δ +0.0067
  [+0.0004, +0.0133], not Holm-rejected.
- Inner leave-one-ecoregion-out (LOEO) cross-validation **on the training cycles**, pooled out-of-fold AUROC of the
  selected configurations: LightGBM **0.9135** against RF occupancy incumbent **0.8939** (feature-matched RF 0.9025,
  L2 logistic 0.9092). These are each the best of 8 configurations, so both are selection-optimistic. All 8 LightGBM
  configurations (0.9085–0.9135) scored above all 8 RF-incumbent configurations (0.8859–0.8939).
- Ecoregion rate alone reaches AUROC 0.866 on validation new locations (0.862 on all validation first visits): a
  large part of any pooled AUROC is contrast **between** ecoregions.
- Numeric track: LOEO-calibrated intervals over-cover on new locations ("LOEO is harder than the new-location task").

Not seen: any transfer score on the validation cycle; any per-ecoregion score of any model on any cycle; any
confidence interval for the inner-LOEO difference; anything from the test cycle.

**Risk this creates.** Adding a track where the challenger's advantage looked larger, after the main track tied, is a
second chance at the same headline. The design answers this with (a) a rule fixed before any transfer score exists,
(b) a metric that is the already pre-registered inner criterion and not a new choice, (c) a correction for two
tracks (§8), and (d) a guard against a win that comes only from between-ecoregion contrast (§6).

## 3. Held-out blocks and events
- **Blocks:** the nine NARS aggregated ecoregions (`ag_eco9`): CPL, NAP, NPL, SAP, SPL, TPL, UMW, WMT, XER. Every unit
  has an ecoregion (0 missing). These nine are the full partition of the conterminous US, not a sample of regions.
- **Events** (roles from the frozen split, sha256 `587caeca…4a3c`, unchanged):

  | event | fitted on | scored on | when |
  |---|---|---|---|
  | validation transfer | training-cycle first visits **outside** ecoregion E | 2018–19 first visits **inside** E | once, before the freeze (rehearsal and validation evidence) |
  | test transfer | training + validation-cycle first visits outside E | 2023–24 first visits inside E | once, in the single look event, with the main track |

- **Scored rows:** all first visits of the evaluation cycle (not only new locations): every location in a held-out
  ecoregion is unseen by the model. No 1-km location cluster spans two ecoregions (0 of 3,918 in train + validation),
  so no fitted row shares a location with a scored row; the code still drops any such row and asserts.
- **Models:** the frozen phase-1 configurations, re-fitted, **no re-tuning**: LightGBM (num_leaves 15,
  min_child_samples 20, colsample 0.8; 3 seeds averaged), RF occupancy incumbent (500 trees, sqrt, leaf 1),
  feature-matched RF (max_features 0.2, leaf 1), L2 logistic (C 0.01; secondary). Imputation and scaling are fitted on
  the fitted rows only. The held-out ecoregion's one-hot column is constant in the fit and gets no special handling.
- **Primary challenger:** LightGBM (already fixed by inner CV on the training cycles; not re-chosen).

## 4. Counts per block (first visits; coldwater-occupied = positive)

| block | train n | train pos | train HUC4 with pos | val n | val pos | val neg | val prevalence | val HUC4 | val HUC4 with pos / with neg | Kish-effective HUC4, pos / neg | fit rows, val event (n / pos) | fit rows, test event (n / pos) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| CPL | 546 | 1 | 1 | 215 | 1 | 214 | 0.005 | 46 | 1 / 46 | 1.0 / 27.3 | 3,267 / 909 | 4,646 / 1,287 |
| NAP | 472 | 198 | 22 | 212 | 89 | 123 | 0.420 | 20 | 17 / 18 | 10.1 / 10.7 | 3,341 / 712 | 4,723 / 1,002 |
| NPL | 331 | 68 | 11 | 104 | 19 | 85 | 0.183 | 14 | 6 / 13 | 3.4 / 4.7 | 3,482 / 842 | 4,972 / 1,202 |
| SAP | 602 | 50 | 21 | 255 | 28 | 227 | 0.110 | 43 | 13 / 43 | 5.1 / 29.6 | 3,211 / 860 | 4,550 / 1,211 |
| SPL | 313 | 16 | 7 | 168 | 7 | 161 | 0.042 | 34 | 4 / 33 | 3.3 / 23.2 | 3,500 / 894 | 4,926 / 1,266 |
| TPL | 476 | 7 | 6 | 201 | 5 | 196 | 0.025 | 37 | 4 / 36 | 3.6 / 23.3 | 3,337 / 903 | 4,730 / 1,277 |
| UMW | 353 | 103 | 15 | 185 | 55 | 130 | 0.297 | 22 | 16 / 21 | 11.4 / 12.4 | 3,460 / 807 | 4,869 / 1,131 |
| WMT | 423 | 348 | 45 | 150 | 130 | 20 | 0.867 | 41 | 38 / 15 | 23.7 / 13.3 | 3,390 / 562 | 4,834 / 811 |
| XER | 297 | 119 | 30 | 104 | 45 | 59 | 0.433 | 33 | 21 / 23 | 17.6 / 13.1 | 3,516 / 791 | 5,006 / 1,125 |
| **pooled** | 3,813 | 910 | 115 | 1,594 | 379 | 1,215 | 0.238 | 187 | 89 / 172 | 54.9 / 103.4 | — | — |

Notes. (i) Kish-effective count = (Σ n_h)² / Σ n_h² over HUC4 clusters, per class: the number of equal-sized clusters
that carry the same information. (ii) One training row has no HUC4 and is counted here as its own cluster (the
protocol table says 114 training HUC4 with positives). (iii) 107 of 200 HUC4 span more than one ecoregion.
(iv) The "fit rows, test event" columns hold train + validation rows only; they are known today.
(v) HUC2 regions with validation rows: 18 (15 with positives; Kish-effective 8.6 positive / 10.4 negative).

**What the counts say.**
- Blocks are very unequal. CPL, SPL and TPL are almost all negatives (1, 7 and 5 positives); WMT is almost all
  positives (20 negatives). Holding out WMT removes 348 of 910 training positives (38 %): the hardest block.
- **Per-block claims are not feasible.** A block is *evaluable* only with ≥ 30 positives, ≥ 30 negatives and ≥ 10 HUC4
  with each class (the protocol's events power rule, applied to both classes because AUROC needs both). On
  validation-cycle counts only NAP, UMW and XER pass; WMT fails on negatives, SAP on positives (28), the rest on both.
- Therefore the inference must be pooled over blocks (§5–§7); per-block results are descriptive.

## 5. Primary metric
**Pooled AUROC of the out-of-ecoregion predictions**: the union of the nine blocks' scored rows, each row scored by
the model that never saw its ecoregion; one AUROC over that union. This is exactly the inner-CV selection criterion
already pre-registered and implemented (`troutrefuge_models.inner_select`), applied to a later cycle. Estimand:
paired Δ AUROC, LightGBM minus comparator.

## 6. Guard metric (part of the decision rule)
**Within-ecoregion AUROC** = share of concordant positive–negative pairs, counting only pairs **inside the same
ecoregion**, pooled over ecoregions (pair-weighted mean of the per-block AUROCs). It removes all between-ecoregion
contrast and all dependence on how the nine separately fitted models' score levels compare. Validation-cycle pairs:
33,644 within ecoregions (NAP 10,947; UMW 7,150; SAP 6,356; XER 2,655; WMT 2,600; NPL 1,615; SPL 1,127; TPL 980;
CPL 214) against 460,485 pooled pairs — only 7 % of the pooled pairs are within an ecoregion. Without this guard, a
"transfer" win could be purely a better ordering of whole ecoregions.

## 7. Resampling unit and minimum detectable effect
- **Unit: HUC4.** Paired cluster bootstrap, B = 2,000, whole HUC4s resampled over the pooled scored set (a HUC4 brings
  all its rows, in every ecoregion it spans); percentile interval. Same unit and B as the main track.
- **The nine blocks are fixed, not resampled.** With nine fitted models there is a block-level dependence that a HUC4
  bootstrap does not capture, and nine blocks are too few to resample. The claim is therefore worded as "each of these
  nine ecoregions held out in turn", not "any new region". Reported as a sensitivity, with no decision weight: the
  same interval with **HUC2** as the unit (18 clusters).
- **Minimum detectable paired Δ AUROC** (80 % power, two-sided 5 %; Hanley–McNeil standard error, MDE = 2.8 · se ·
  √(2(1 − ρ)), the protocol's formula; pooled validation-cycle counts, 379 positives / 1,215 negatives):

  | anticipated AUROC | ρ between the two models | row level (rows independent) | HUC4 level, worst case (Kish-effective 55 / 103) |
  |---|---|---|---|
  | 0.80 | 0.6 (protocol constants) | 0.036 | 0.099 |
  | 0.90 | 0.6 | 0.027 | 0.074 |
  | 0.90 | 0.8 | 0.019 | 0.052 |
  | 0.90 | 0.9 | 0.014 | 0.037 |

  The worst-case column assumes rows in one HUC4 carry no independent information. The main track shows how far
  from that the data are: on the same 1,594 validation rows the realised HUC4-bootstrap half-width was 0.0065
  (empirical MDE 0.009), which equals the row-level formula at ρ ≈ 0.90. With the two-track correction of §8
  (97.5 % intervals) every value above is multiplied by 1.10.
- **Reading.** The only effect size seen is the training-cycle inner-LOEO difference, +0.020, which is
  selection-optimistic. The track detects an effect of that size with about 80 % power only if ρ ≥ 0.8 and HUC4
  clustering is as mild as in the main track. Under the protocol's own constants (ρ 0.6) or the worst-case column it
  is underpowered, and "competitive" is the likely outcome. The realised interval width is first known from the
  validation transfer event.
- Per-block row-level MDE (AUROC 0.8, ρ 0.6): NAP 0.08, UMW 0.10, WMT 0.11, XER 0.11, SAP 0.13, NPL 0.16, SPL 0.26,
  TPL 0.30, CPL 0.67 — far above any plausible effect, which is why no per-block test is proposed.

## 8. Comparators and decision rule
**Comparators (the same family of six as the main track):**
1. RF occupancy incumbent — re-fitted without E.
2. MAST rule (score −MAST; mean annual stream temperature, Hill et al. 2013) — published score, no fit.
3. MSST rule (score −MSST; mean summer stream temperature) — published score, no fit; approved by the lead
   2026-11-10 as a comparator score only, under the `d7_mast` restrictions.
4. Air–water rule — OLS re-fitted on fitted rows outside E.
5. Feature-matched RF — re-fitted without E.
6. Strongest simple baseline — the better of location kNN (15 nearest fitted reaches, all outside E) and
   logistic-on-geography (re-fitted without E), chosen by pooled AUROC on the **validation** transfer event.
   Ecoregion rate and prevalence are not defined for an unseen ecoregion and are not candidates.

MAST and MSST come from a national model trained on temperature sites in every ecoregion, so they are **not** held
out; that favours the incumbents (conservative for the claim). They are scored on the rows where they are defined
(MAST 91–93 %, MSST 95–97 % of validation rows), as in the main track.

**Transfer claim ("beats incumbents under ecoregion transfer")** — all of the following on the test transfer event:
- paired Δ pooled AUROC > 0 with the interval excluding 0, against **each** of the six comparators, Holm over six;
- per-seed paired point estimate > 0 against each comparator for each of the 3 LightGBM seeds;
- guard: Δ **within-ecoregion** AUROC against the RF occupancy incumbent has a point estimate > 0 (its interval is
  reported; it need not exclude 0);
- the pooled scored set meets the events power rule (≥ 30 positives, ≥ 10 HUC4 with positives).

Otherwise the transfer track is "competitive" (no comparator significantly better than the challenger) or "negative".

**Two tracks, one headline (recommended rule).** D7 gets two chances at "beats incumbents". To keep the 5 % error rate
for that headline, each track is tested at 2.5 % (Bonferroni over two tracks: 97.5 % intervals; Holm within each
track at 2.5 %). D7 counts toward H-G4 if **either** track meets its full rule at that level, and the write-up names
the track. Outcomes:

| main track (new locations) | transfer track | D7 label |
|---|---|---|
| met | met | beats incumbents (new locations and ecoregion transfer) |
| not met | met | beats incumbents **under ecoregion transfer only**; ties the RF incumbent at new locations |
| met | not met | beats incumbents at new locations; transfer competitive/negative |
| not met | not met | competitive (or negative) |

Alternative for the lead: keep both tracks at 5 % but let only the originally pre-registered main track count toward
H-G4, the transfer result being a separate secondary claim. I recommend the Bonferroni rule because it does not
depend on which track is called primary after the validation numbers were seen.

**Reported without decision weight:** per-block AUROC of every model with the evaluable flag (§4 rule, decided from
the evaluation cycle's own counts); number of evaluable blocks with Δ > 0; HUC2-unit intervals; the L2 logistic
challenger; binary MAST/MSST ≤ 20 °C rules; all first-visit and salmonid-label versions.

## 9. Freeze and look
- One FREEZE record for D7 covers both tracks: one import-closure hash, one data pin set, one claim, one look. Both
  tracks' test predictions are written and sealed together before either is scored.
- The validation transfer event is run once before the freeze. **Its result cannot change this design**: the track is
  frozen and looked at whatever the validation transfer result is. Any change after it would be a dated, disclosed
  amendment needing the lead's approval.
- Configurations: no new configuration is created (frozen configurations re-fitted; the phase-1b LightGBM re-fit was
  likewise not counted). The log stays at 36 of 40. If the lead prefers to count the four re-fitted learned families,
  it is 40 of 40. Either way nothing is tuned in this track.
- No numeric (temperature) transfer track.

## 10. Limitations
- Inference is conditional on the nine ecoregions (§7); the pooled metric is dominated by between-ecoregion contrast
  (§6), which the guard only partly addresses (direction, not significance).
- NAP, UMW and SAP hold 73 % of the within-ecoregion pairs; the guard metric mostly reflects those three.
- All covariates (StreamCat, PRISM, gridMET) and both published temperature scores are national products that
  include the held-out ecoregion; only the fish labels are held out.
- Reaches with zero fish caught are unlabelled (main protocol §1).
- Test-cycle block counts are unknown; if a block's composition differs strongly from 2018–19, the MDE values change.

## 11. Decisions requested from the lead
1. Approve or change: pooled AUROC as primary with the within-ecoregion guard (alternative: within-ecoregion AUROC as
   primary — cleaner, no prior knowledge on it, lower power).
2. Two-track rule: Bonferroni over tracks (recommended) or main-track-only for H-G4.
3. Configuration count: 36 or 40 of 40.
4. Go-ahead for the validation transfer event (nothing is run before this).
