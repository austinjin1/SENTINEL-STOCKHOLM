# M2 SENTINEL-Orbit (SatWQ-Net): architecture research and SOTA plan

**Date:** 2026-11-10. **Type:** read-only research synthesis for the lead's decision. AI-assisted.
**Not:** a pre-registration, a result, or an edit to any protocol. Every protocol change below is a *proposal*. The
lead writes it as a dated amendment before any result on it exists (`sota_campaign.md` rule 2). Draft wording is in
Appendix A.
**Scope:** M2 numeric retrieval of chl-a, TSS, turbidity and Secchi (plus proposed new targets) with uncertainty,
and bloom state, from Sentinel-2 L2A over unseen lakes.

**How the evidence was checked (read this first).**
- **Repo numbers.** These were re-read this session from run files: `sentinel_runs/orbit/compare_satwq_v4_noaug/{val_row_preds.parquet,
  paired.csv, power_*.csv}`, `cache/dev_v4/table.parquet`, `docs/preregistration/orbit_split_report_dev_v4.json`
  (`protected_cell_counts`, counts only), `logs/satwq_v4_noaug_s0.log`, and the hist build log.
  - I **re-ran the three read-only pandas diagnostics** of the earlier workflow stage (`scratchpad/diag{1,2,3}.py`)
    on registered dev_v4 *validation* predictions and the dev cache.
  - I ran one **key-only count**: how many 2024–25 WQP matchup site-dates also carry TP/TN/DOC/… samples on the same
    date. It read site IDs and dates only, never values, so these protected rows contribute counts only.
  - Nothing was trained, no protected label value was read, and no file other than this one was written.
- **Web search.** The session's shared search budget ran out after **two** queries (Pahlevan 2020; Saranathan 2024).
  The other references were checked by **fetching the arXiv page, the Crossref DOI record, the conference page or the
  dataset landing page directly** this session. Tags:

  | Tag | Meaning |
  |---|---|
  | **[S]** | Found by web search this session |
  | **[F]** | Fetched directly this session (arXiv, Crossref, conference/dataset page) |
  | **[W]** | Verified by an earlier stage of this workflow (Crossref/arXiv). Not re-fetched by me |
  | **[R02]** | Carried from `docs/research/02_satellite_remote_sensing.md` (2026-10-22). **UNVERIFIED in this session** |
  | **[U]** | Unverified |

- **Design inputs were truncated.** I saw one full design (SatWQ-IOP, the "domain" angle) and none of the critics'
  text. Section 4 applies my own critique, grounded in repo evidence and the fetched literature, and names every change
  it makes to SatWQ-IOP.

---

## 0. Decision summary

1. **Simple ML ties because every model hits the same information floor, not because SatWQ-Net is badly built.**
   - On dev_v4 val chl-a, the residuals of SatWQ-Net, MDN_B and the GBM correlate **0.85–0.89**.
   - SatWQ-Net's single seeds average about 53 MdSA, the same as MDN_B (52.6). Its lead comes from the 3-seed ensemble
     (48.3).
   - A plain average of the two baselines, MDN_B and GBM, scores **49.7**.
2. **Where the error is.**
   - About **half of the error is a persistent per-lake bias.** Removing each lake's mean residual (an oracle) takes
     SatWQ-Net from 48.3 to **26.1** MdSA.
   - The models explain only about **30% of within-lake (temporal) chl-a variance**.
   - It is **not label noise**: two stations under 300 m apart on the same lake and date differ by about **9%** per
     sample.
3. **The binding constraint for a SOTA label is test power, as much as architecture.**
   - Headline chl-a: MDE 10.3 points vs MDN_B and 14.3 vs GBM; joint power at the val gains is **0.21**.
   - An unspent, unscored block already exists: **test-role lakes × 2015–2023**, "Test-S".
     - dev_v4 alone holds **2,639 chl-a rows in 143 or more role blocks** and **2,172 Secchi rows**.
     - dev_v5 adds WQP 2015–2023 rows.
     - It roughly doubles the number of test blocks. **Pre-register and reserve it before the M2/test look.** No GPU
       needed.
4. **More targets are free.** On the 2024–25 WQP matchup site-dates (keys only):
   - **TP is co-sampled at 87%** of chl-a site-dates;
   - TN at 37%, pheophytin at 31%, TOC at 17%, DOC at 11.5%.

   So a multi-target dataset (TP, TN, DOC, plus FNU/SSC as auxiliaries) costs **zero new imagery**.
5. **Ranked architecture bets** (Section 4):
   - **A1 SatWQ-LakeSet-MT** (7.0/10, about 16 GPU-h). Multi-date same-lake imagery context aimed at the per-lake bias,
     plus a shore/land ring for adjacency, multi-target heads, and variance-controlled seed units.
   - **A0** variance-control arm (6.0/10, about 3 GPU-h).
   - **A2 SatWQ-IOP-lite**, a physics decoder, cut down from the 45-GPU-h design (4.5/10, about 18 GPU-h; gated on the
     chain9 SSL result).
   - **A3** TOA dual-stream (4.0/10).
   - **A4** AlphaEarth lake prior (3.0/10).
   - Rejected: bigger models, fine-tuned EO foundation models as the core, GBM blends, and strata picked from val.
6. **Do not delay the headline freeze.** The current SatWQ-Net (v1) goes to M2/test as planned. The new arms target
   Test-S, Track N, Track G and Test-C, and v1 is scored again alongside v2 on Test-S (Holm over the 2 arms).
7. **Honest outlook** (subjective probabilities):
   - P(any chl-a SOTA label for M2 by early December) ≈ **0.25–0.35**;
   - P(at least one scoped M2 label on any track, most likely Secchi on Test-S or a new target) ≈ **0.4–0.5**;
   - headline H1 chl-a with v1: ≈ 0.10–0.15.

---

## 1. Diagnosis: why simple ML wins or ties

All numbers are dev_v4 **validation** (development, not claims; registry `orbit_val_dev_v4`). The cell is
`valxval_lake`. CIs are role-block bootstrap 95%; a negative difference means SatWQ-Net is better.

### 1.1 The numbers

| | chl-a (1,157 rows / 318 lakes / 89 blocks) | Secchi (919 / 183 / 48) |
|---|---|---|
| SatWQ-Net 3-seed ensemble | **48.3** (SSPB +0.7) | **27.5** |
| SatWQ-Net single seeds | 53.6 / 56.5 / 49.2 | 25.6 / 30.1 / 30.3 |
| MDN_B (val-selected) | 52.6 (SSPB **+17.4**) | 32.2 |
| GBM imagery+context (val-selected) | 56.3 (SSPB −8.1) | 32.7 |
| avg(MDN_B, GBM), post hoc, not a registered comparator | 49.7 | 30.0 |
| Paired difference vs MDN_B | −4.4 [−13.2, +0.2] | −4.7 [−8.6, −0.7] |
| Paired difference vs GBM | −8.0 [−17.3, +0.7] | −5.1 [−12.7, +0.9] |
| Per-seed difference vs MDN_B | +1.0 / +3.9 / −3.4 (**2 of 3 seeds fail**) | −6.7 / −2.2 / −2.0 (all pass) |
| Headline test MDE80 (vs MDN_B / vs GBM); joint power | 10.3 / 14.3; **0.21** | 4.0 / 8.0; 0.37 |

Other reference points:
- location kNN 126.5; index-opt 104.7; MDN-STREAM official weights 85.9;
- SatWQ-Net with context off 50.0; SatWQ-Net RGB+NIR only 142;
- on dev_v3, capacity d128 scored 55.6 vs base 51.2.

### 1.2 Seven findings

**F1. The lead is ensembling, not new information.**
- The residual correlation on chl-a is 0.888 between SatWQ-Net and MDN_B, 0.850 between SatWQ-Net and GBM, and 0.846
  between MDN_B and GBM. Secchi gives 0.85 / 0.87 / 0.78.
- Single SatWQ-Net seeds are no better than MDN_B. MDN_B itself is a 10-init median per model, so it is already an
  ensemble.
- The post-hoc baseline average (49.7) is within 1.4 points of the SatWQ-Net ensemble.
- On dev_v3 the gap to MDN_B was only 1.5 points (48.3 vs 49.8).

**F2. About half of the error is a persistent per-lake bias, and within-lake skill is weak.**
- Lakes with at least 3 rows, chl-a: the MSE splits into between-lake 47% (SatWQ-Net), 53% (MDN_B) and 46% (GBM).
- The oracle that removes each lake's mean residual gives 26.1 (SatWQ-Net), 29.8 (MDN_B) and 30.9 (GBM).
- Approximate share of variance explained (the label variance uses all rows, the residuals use lakes with ≥3 rows):
  - SatWQ-Net's within-lake residual MSE (0.055) is about 70% of the within-lake label variance (0.079), so roughly
    30% of temporal variation is explained;
  - between lakes, roughly 75% is explained.
- So the models know *which kind of lake* it is reasonably well, are biased per lake, and see within-lake change only
  weakly.

**F3. Label noise is not the floor.** Same lake, same date, different stations (`diag2.py`, chl-a):

| Station distance | Pairs | Per-sample MdSA-equivalent |
|---|---|---|
| < 300 m | 193 | ≈ 8.7% |
| 300–1,000 m | 487 | ≈ 13.8% |
| 1–3 km | 786 | ≈ 22.6% |

Secchi under 300 m is ≈ 3.3%. Error near 48% is therefore mostly retrieval error: atmospheric correction, optical
ambiguity and adjacency. It is not irreducible label noise. The caveat is that methods are mixed: only 3 such pairs are
lab-flagged on both sides.

**F4. The atmospheric-correction / geometry floor is shared.** Val MdSA rises with Sen2Cor AOT for every model. At
AOT > 0.4 (27 rows): SatWQ-Net 86, MDN_B 203, GBM 166. Physics-based AC over inland water is the documented weak link
(ACIX-Aqua, Pahlevan et al. 2021 [F]; Saranathan 2024 [S]).

**F5. Low-chl water is where S2 fails.** Rows at ≤ 2.6 µg/L (199 rows, 17%) score 137 / 200 / 168 MdSA (SatWQ-Net /
MDN_B / GBM).
> **Disclosure.** F4/F5 strata were *seen on val* by the earlier stage and again by me. Under `sota_campaign.md`
> rule 2 and §15.5 item 5 they **must not** become harder tracks or claim strata. They are design motivation only.

**F6. The data are thin and heterogeneous outside chl-a and Secchi.**
- dev_v4 holds 28,360 rows: chl-a 21,219, Secchi 18,030, but **TSS only 1,765 and turbidity 1,130, all
  unit-unspecified**.
- 93% of rows are AquaMatch, and only 1,437 chl-a values are lab-flagged.
- **Flag for Track G:** dev_v4 already holds **485 non-US rows**: Caravan-Qual 422 and GLORIA 63.
  - 151 are in-lake.
  - By role: 277 train-role and 208 val-role.
  - By cell: trainxtrain (fit + inner) 202, valxtrain 163, trainxval 75, valxval 45.

  This contradicts `orbit_track_g.md` §1 ("all current data are US") and protocol §10 ("US-only so far").

**F7. Optimisation noise inflates seed spread.**
- The s0 run stopped at epoch 41 (best epoch 25, inner MdSA 40.59), and the inner metric drifted to 44.3 by epoch 40.
- Early stopping on a 15% inner block set is a noisy selector. Combined with single-network seeds, this produces a
  per-seed spread (49–56.5) as large as the gain.

**Conclusion (rule 5 of `sota_campaign.md`).**
- More capacity will not help: d128 was worse, and the residuals are shared.
- What can help is (a) information the comparators do not use well, and (b) more power at test.
- The information sources ranked here:
  1. **The same lake's other acquisitions** (aimed at F2, the per-lake bias, and at averaging out scene-specific AC
     error);
  2. **shore/land pixels already in every 64×64 chip** (adjacency; SatWQ-Net reads only the centre 15×15);
  3. **co-measured targets** (TP, TN, DOC) for multi-task supervision;
  4. **unlabelled imagery** (SSL, chain9; physics ELBO);
  5. **less-processed radiometry** (L1C TOA).

  Comparator parity means GBM and MDN_B get 1–3 and 5 as features too, so the realistic *net* margin is the part
  that learned set aggregation and representation sharing add over hand-crafted summaries.

---

## 2. Literature: what the state of the art says

### 2.1 The comparators and their published behaviour
| Work | Citation | Tag | Result relevant to M2 |
|---|---|---|---|
| MDN for MSI/OLCI chl-a | Pahlevan et al. 2020, RSE 240:111604, doi:10.1016/j.rse.2019.111604 | [S] | n = 2,943 in-situ Rrs–chl pairs. The abstract states the model is "sensitive to uncertainties in remote-sensing reflectance". In-situ MSI MAPE ≈ 24% [R02] |
| MDN_B, matchup-trained (Landsat-8) | Smith et al. 2021, Front. Remote Sens. 1:623678, doi:10.3389/frsen.2020.623678 | [F] Crossref | Training on satellite matchups (Type B) beats in-situ training (Type A). SF Bay errors 54% (MDN_B anc), 41% (MDN_B ρ_s), 120% (MDN_A) [W, read from the NTRS PDF by the earlier stage]. **This is the precedent for MDN_B and for less-processed inputs (A3)** |
| Simultaneous multi-target MDN | Pahlevan et al. 2022, RSE 270:112860, doi:10.1016/j.rse.2021.112860 | [F] Crossref | Leave-one-dataset-out MSI chl-a MdSA 48.3%; the out-of-sample margin over the 2nd-best method shrinks to about 4–10 points. The multi-target head gives about 1 point [R02] |
| MDN vs BNN-MCD on real imagery | Saranathan et al. 2024, Front. Remote Sens. 5:1383147, doi:10.3389/frsen.2024.1383147 | [S] | On ACOLITE MSI matchups, chl-a MdSA ≈ 195% and TSS 110% (MDN). MDN chl-a interval coverage falls to 47.8% [R02/W] |
| BNN (MC dropout), oligo/mesotrophic lakes | Werther et al. 2022, RSE 283:113295, doi:10.1016/j.rse.2022.113295 | [F] Crossref | Gains came against off-the-shelf MDN weights, so the comparison is not like-for-like [R02] |
| ACIX-Aqua atmospheric-correction intercomparison | Pahlevan et al. 2021, RSE 258:112366, doi:10.1016/j.rse.2021.112366 | [F] Crossref | Global AC assessment for L8/S2 over inland and coastal water (title-level verification; numbers not read). Motivates F4, A2 and A3 |
| XGBoost + residual CNN, S2 L2A | Sherf, Efrati, Rozman, Harel, arXiv:2510.24124 (v2, 4 May 2026) | [F] | R² 0.79, MAE 13.52 mg m⁻³, slope 0.91; 13,626 matchups; 867 water bodies in testing. The abstract does not state the split; per [R02] the CNN stage used a random split. No MDN baseline |
| Spatiotemporal transformer, lake turbidity | Prasad, Verma, Khan 2026, NeurIPS 2026 CCAI workshop poster, https://neurips.cc/virtual/2026/174697 | [F] | Accuracy "drops sharply" when whole lakes or regions are held out. With lat/lon the model learns a location-specific correction that "collapses completely on another continent". **Removing 3 coordinate inputs fixes it with no measurable domestic cost.** The claimed wins over GBM and Dogliotti/Nechad are abstract-only |
| Spatial generalisation on GLORIA | Kumari, Alikas, Dev 2026, ICML 2026 poster, https://icml.cc/media/icml-2026/Slides/76520.pdf | [F] (PDF text extracted) | GLORIA n = 6,675. Chl-a leave-one-country-out R²: RF 0.61, XGB 0.59. aCDOM440: 0.61 random vs 0.13–0.14 LOCO. RF ≈ XGB (ΔR² < 0.02). The authors conclude "the bottleneck is data, not model architecture" |
| AquaSat | Ross et al. 2019, WRR 55, doi:10.1029/2019WR024883 | [F] Crossref | The earlier large Landsat matchup set. The Topp 2021 Secchi follow-up (static lake features carry much of the skill) is [R02] |
| AquaCorr-Global v1.0 | Zenodo, doi:10.5281/zenodo.21927415 | [F] | CC BY 4.0, published 2026-08-14. 1,984,387 same-calendar-day matchups (935,740 pass QC); 12.5 M in-situ records at 372,639 sites; 972.8 MB of files. Variables: chl-a, TSS, Secchi, aCDOM, turbidity, DOC, TOC |
| GLORIA | Lehmann et al. 2023, Sci. Data 10, doi:10.1038/s41597-023-01973-y | [F] Crossref | The pretraining corpus |

### 2.2 Methods borrowed by the proposals
| Idea | Citation | Tag | Use here |
|---|---|---|---|
| Permutation-invariant attention over sets | Lee et al. 2019, Set Transformer, ICML, arXiv:1810.00825 | [F] | A1's lake-context aggregator over dates |
| Weight averaging for flatter optima and lower variance | Izmailov et al. 2018, SWA, UAI, arXiv:1803.05407 | [F] | A0/A1 seed-unit variance control (EMA/SWA) |
| Conformalised quantile regression | Romano, Patterson, Candès 2019, arXiv:1905.03222 | [F] | Already in protocol §5b. A2's closure-conditional Mondrian variant |
| Physics-integrated VAE (regulariser limiting neural override of the physics) | Takeishi & Kalousis 2021, arXiv:2102.13156 | [F] | A2's discrepancy heads |
| Physics-decoder VAE on real S2 L2A (land, PROSAIL) | Zérah, Valero, Inglada 2024, RSE 312:114309, doi:10.1016/j.rse.2024.114309 | [F] Crossref | The closest precedent for A2. It covers land only and has no atmospheric nuisance model |
| QAA (IOPs from Rrs) | Lee, Carder, Arnone 2002, Appl. Opt. 41:5755, doi:10.1364/AO.41.005755 | [F] Crossref | A2 decoder; GBM-phys comparator features |
| Mechanistic Secchi model | Lee et al. 2015, RSE 169:139–149, doi:10.1016/j.rse.2015.08.002 | [F] Crossref | A2's Secchi operator |
| Gordon 1988; Lee 2013 Kd; Bricaud 1998; Pope & Fry 1997; Werdell 2013 GIOP; Sterckx 2011 adjacency; Hieronymi 2017 ONNS; Kennedy & O'Hagan 2001 | DOIs in Appendix C | [W] | A2 physics. **The numeric constants are UNVERIFIED** (recalled from memory by the design stage; publisher full texts returned 403) |
| Annual EO embedding fields | Brown et al. 2025, AlphaEarth Foundations, arXiv:2507.22291 | [F] | A4 lake prior. The abstract states annual layers for 2017–2024. Dimension (64-d) and licence (CC BY 4.0) are [R02] |
| Tabular foundation model | Hollmann et al. 2025, Nature 637, doi:10.1038/s41586-024-08328-6 | [F] Crossref | Optional descriptive "stronger simple baseline" check (§3) |
| Test-time training | Sun et al. 2020, ICML, arXiv:1909.13231 | [F] | Considered for per-lake adaptation; rejected for now (§4.6) |

### 2.3 What the literature implies for M2
- **No clean published SOTA exists for unseen-lake × unseen-year S2 L2A retrieval.** Published end-to-end S2 models use
  random or lake-level splits without a retrained MDN [R02, F: Sherf]. The bar is therefore the protocol's own:
  retrained MDN_B and a tuned GBM, with tuning parity.
- **Strong methods differ by single-digit points out of distribution.** Examples: the Pahlevan 2022 LODO margin of
  about 4–10 points; Kumari 2026 RF ≈ XGB; our val gaps of 4–8 points. A §15.1 win needs a test block big enough to
  resolve about 5–8 points, *and* a model that adds a few more points.
- **Coordinates are a liability out of region** (Prasad 2026). This matters for Track G and argues for
  coordinate-free variants.
- **Matchup training beats in-situ training on real imagery** (Smith 2021; Saranathan 2024). This is already
  exploited. The open lever on the input side is the AC floor: less-processed radiometry or learned AC.
- **Novelty status.** I found no inland-water retrieval model that conditions on the same lake's other acquisitions
  (multi-date set context) and is evaluated on unseen lakes. That absence is **UNVERIFIED**: the search budget was
  exhausted. A dated search in `docs/novelty_checks/` is required before any "first" wording.

---

## 3. Is the current architecture optimal? (honest)

**Verdict.** It is not demonstrably optimal, but it is probably close to the ceiling of its *inputs*: one date, the
centre 150 m, Sen2Cor L2A. The evidence:
- d128 was worse;
- context adds only 1.7 points;
- residuals are 0.85–0.89 correlated with two very different models;
- the ensemble edge is reproduced by averaging the baselines (F1).

Re-architecting the same inputs is unlikely to buy 5+ points.

**Concrete sub-optimalities (verified in `sentinel/models/orbit/satwq.py` and the config):**
1. **Only the centre 15×15 window is used.** The shoreline/land pixels already in every 64×64 chip, which are the
   adjacency source, are never seen by SatWQ-Net or the GBM.
2. **Single date.** The persistent per-lake bias (F2) cannot be separated from scene-specific AC error without other
   acquisitions of the same lake.
3. **A thin multi-task head.** It has 4 targets, two of them with fewer than 2k rows (TSS, turbidity), while
   co-measured TP/TN/DOC labels sit unused (Section 5).
4. **Variance.** One network per seed with early stopping on a noisy inner set (F7). By the comparator's own published
   definition, one MDN_B "model" is a median of 10 inits.
5. **Coordinates in the context vector.** The lat/lon unit vector is a known out-of-region failure mode (Prasad 2026).
   It matters for Track G.
6. **Random noise augmentation in GLORIA pretraining** instead of physically structured nuisance (AC residual, glint,
   adjacency). The drone-band augmentation already cost about 3 points (dev_v3: noaug 48.3 vs base 51.2).
7. **No pixel-to-pixel interaction before pooling** (minor; pixel sets gave about 4 points in total).

**What is fine and should be kept:**
- wavelength-tagged band tokens (needed for the M7 drone student and Track L);
- the mixture head;
- the masked multi-target NLL;
- GLORIA pretraining;
- whole-vector context dropout;
- the event route (E4a AUPRC 0.857, BSS 0.604).

**Honest note on the baseline bar.** A stronger simple baseline may exist: an equal-weight MDN_B + GBM average
(49.7), or a tabular foundation model (TabPFN, Hollmann 2025 [F]). Neither is a registered comparator, and the bar is
not changed. I recommend reporting avg(MDN_B, GBM) as a **pre-declared descriptive** comparator in every new look, so
that any claim is honest about it.

---

## 4. Ranked novel architecture proposals

The score (0–10) is my risk-adjusted expected contribution to a scoped §15.1 label per unit cost. GPU-h are on the
RTX 2080 Ti, scaled from the measured v1 timing:
- fine-tune about **8.4 s/epoch** on dev_v4 (335 s at epoch 40; early stop at epoch 41);
- GLORIA pretraining about 90 s;
- dev_v5 is assumed to be about 2–3× the fit rows.

| Rank | Arm | Score | GPU-h | Peak VRAM | New data | Primary unseen block | Status |
|---|---|---|---|---|---|---|---|
| 1 | **A1 SatWQ-LakeSet-MT** | **7.0** | ~16 (12–22) | ≤ 4 GB | 0 GB (phase A); optional ≤ 0.6 GB (phase B) | Test-S + Track N; then Track G, Test-C | Recommended |
| 2 | **A0 variance-control arm** (v1 + 3-init seed units + EMA) | **6.0** | ~3 | ~1.5 GB | 0 | Test-S | Recommended (control) |
| 3 | **A2 SatWQ-IOP-lite** (physics decoder, semi-supervised) | **4.5** | ~18 | ~6.5 GB | 0 | Test-S (if selected); Track G | Gated on chain9 SSL |
| 4 | **A3 SatWQ-TOA dual-stream** (L1C + L2A) | **4.0** | ~5 | ~2 GB | ~0.9 GB L1C centre windows (needs approval) | Test-C / Track G | Conditional |
| 5 | **A4 AlphaEarth lake prior** | **3.0** | ~3 | ~1.5 GB | < 0.2 GB | Test-S (ablation) | Optional |
| — | Fine-tuned EO foundation model as the core; bigger SatWQ; GBM blends; val-chosen strata; coordinate residual kriging; per-lake TTT | ≤ 1.5 | — | — | — | — | Rejected (§4.6) |

### 4.1 Rank 1: A1 SatWQ-LakeSet-MT (score 7.0/10)

**One line.** Keep SatWQ-Net's spectral-token encoder, and add three things:
- (i) a **set encoder over the same lake's other acquisitions** to estimate the lake's persistent optical state and
  bias separately from today's scene;
- (ii) the **land ring and per-pixel shore distance** from the full chip (adjacency);
- (iii) **multi-target heads**, adding the co-measured TP, TN and DOC (FNU and SSC as auxiliaries).

Each "seed" is a variance-controlled unit.

**Architecture (about 0.3 M parameters; v1 is 182,780).**
1. **Pixel encoder (unchanged from v1).**
   - Band token = MLP(asinh(ρ/0.01)) + Linear(sinusoidal SRF-centroid λ).
   - A 3-layer, d = 64, 4-head pre-norm Transformer with a CLS token.
   - The pixel-offset MLP input gains **log1p(d_p/10 m)**, the per-pixel distance to the nearest non-water pixel. It
     comes from an SCL/JRC distance transform over the full 64×64 chip.
2. **Date embedding** = v1 attention pooling + mean pooling over the date's water pixels:
   - up to 64 pixels for the target date;
   - **8 for each context date** (compute cap).
3. **Lake-context set** C(w, t). The rule is pre-registered and uses imagery only:
   - Membership: up to **K = 6** other QC-passing chips of the same water body w (same HydroLAKES id), with
     3 d ≤ |t′ − t| ≤ 183 d. Chips come from any chip store (aquacorr, wqp, wqp hist). **Only chip arrays are read,
     never their labels.**
   - Choice: the 6 with the lowest Sen2Cor AOT, with ties broken by |Δt|. Same-date chips (other stations) are
     excluded.
   - Each context date d is tokenised as e_d + MLP(q_d) + MLP(Δt_d), where q_d = (cos SZA, cos VZA, cos RAA, AOT,
     glint angle, processing-baseline group).
   - Aggregation: a 2-block **Set Transformer** (SAB, 4 heads) followed by PMA with 1 seed, giving the lake embedding
     ℓ.
   - Training: the whole set is dropped with p = 0.3, and a random 1…K subset is used, so single-visit lakes stay in
     distribution.
   - **No context-count feature.** The count encodes monitoring intensity, which correlates with trophic state.
4. **Land ring** (one vector per chip): an MLP over:
   - the asinh median L2A spectrum (12 bands) of SCL vegetation / bare / unclassified pixels 30–300 m from the water
     set;
   - the land fraction f_land and land NDVI.
5. **Geometry**: v1 geometry, plus the glint angle cos 2ω and its Fresnel reflectance.
6. **Context**: DOY sin/cos. **lat/lon only in config c1** (see below).
7. **Fusion**: LN → MLP([chip_t, ℓ, chip_t − W·ℓ, geo, ring, ctx]) → h (d = 64).
8. **Heads**: a K = 5 Gaussian mixture per target in standardised log10. The loss is masked NLL; left-censored
   TP/DOC values (detection limits) enter through the mixture CDF (interval likelihood). Targets and weights:
   - chl-a, Secchi, TSS, turbidity-NTU: weight 1;
   - TP, TN, DOC: weight 0.5;
   - FNU, SSC (auxiliary): weight 0.25.
9. **Seed unit = 3 independently initialised networks**, mixture-averaged, each with **EMA weights** (decay 0.999)
   used for inner-cell early stopping and inference. The existing patience-15 inner rule is unchanged. The arm's
   reported "seed s" is unit s, so ≥ 3 units are required (see §4.7 for the protocol point).
10. Pretraining: v1's GLORIA stage, unchanged. If chain9 selects an SSL configuration, A1 inherits its encoder
    initialisation (declared now, before the chain9 result).

**Configurations (cap 3, all selectable, all counted).**
- **c1**: full model with lat/lon.
- **c2**: full model, **coordinate-free** (Prasad 2026; also the Track-G candidate).
- **c3**: no context set. Ring + multi-target + seed units only, for attribution.

Each is 3 seed units × 3 inits = 9 networks.

**Inductive bias the GBM lacks.**
- (a) Learned, quality-weighted aggregation of a variable-size set of dated pixel sets. The GBM sees fixed
  medians/IQRs.
- (b) Explicit separation of persistent (ℓ) and date-specific (chip_t − Wℓ) components, which targets F2. AC error is
  largely scene-specific, so averaging over K dates attenuates it in ℓ.
- (c) A representation shared across 9 targets: a GBM fits each target separately.
- (d) A distributional output.

**Comparators (parity is mandatory; all fitted on the identical fit cell, inputs and data version).**
- **GBM family.** The existing 16-config grid × {v1 features, v1 + ring + context summaries}:
  - context summaries = per-band median and IQR over the same K context chips' 3×3 water medians;
  - no count feature.

  The val-selected member is the comparator. **Add a GBM-phys member** (QAA a(443), b_bp(555), Lee-2015 Zsd from
  ρ/π) if A2 runs.
- **MDN_B family.** The existing grid × {12-band, 12-band + context median + ring} × {single-target, **joint
  multi-target head** (the Pahlevan 2022 published form)}. The val-selected member is the comparator.
- **Location kNN**: unchanged.
- **New targets (TP, TN, DOC)**:
  - MDN_B per target;
  - GBM imagery+context;
  - kNN;
  - season+location GBM;
  - a **trophic-chain baseline**: fit-cell OLS of log TP (or TN) on the val-selected MDN_B's predicted log chl-a and
    log Secchi, the classic chl–TP coupling. It is a deliberately strong simple baseline for optically inactive
    targets.
- **Descriptive, pre-declared, no claim role**: avg(MDN_B, GBM); single-init seeds of A1.

**Expected effect (subjective, stated before any result).**
- Gain over the dev_v5 v1 on val chl-a: +1 to +5 MdSA points.
- Parity comparators given the same context/ring features may recover 1–3 points of that.
- Target for carrying A1 to test: **≥ 2 points better than v1-dev_v5 on val chl-a and Secchi** (a development kill
  criterion, not a selection rule).
- P(A1 reaches a val margin of ≥ 8 points over the strongest parity comparator) ≈ 0.3.

**Risks.**
- (i) Context availability follows the sampling design. Mitigations:
  - a deterministic rule;
  - no count feature;
  - a priori strata "≥ 1 context date / none";
  - optional **phase B**: a STAC-built bank of the 6 clearest May–Oct acquisitions per water body-year, centre 15×15
    only (about 6 KB per chip, about 90k chips ≈ 0.55 GB, 2–5 days of extraction), which removes the dependence on
    sampling.
- (ii) Context chips of test-role lakes are protected rows. They load only inside the open claim, like the scored
  rows. `testlook.py`'s protected loader must be extended, and that needs code review.
- (iii) Multi-target negative transfer. TP/TN are optically inactive, hence their low weights.

**Compute.**
- About 1.75× v1 per epoch (64 + 6×8 pixel encodings), so about 30 min per network on dev_v5.
- 3 configs × 9 networks ≈ 13.5 GPU-h, plus GLORIA pretraining (about 0.3 h) and a smoke test ≈ **16 GPU-h**.
- Each network is its own process (< 2 h), through `acquire_gpu`, with checkpoints every 5 epochs.
- ≤ 4 GB VRAM at batch 128 under fp16 AMP. No bf16 anywhere.
- CPU ≤ 6 threads. Comparators on CPU (2–3 h per family refit).

### 4.2 Rank 2: A0 variance-control arm (score 6.0/10)

The v1 architecture exactly as selected on dev_v5 by §6/§6a, with **seed unit = 3 inits (mixture average) + EMA**. Cap
**1 configuration**, about 3 GPU-h (9 networks of about 15 min each), ~1.5 GB.

- **Why.** On val, 2 of 3 single seeds fail the per-seed rule against MDN_B, while the 3-seed ensemble passes on the
  point estimate (F1, F7). A0 changes no information source, so the effect size stays near v1's. It does make
  criterion (iii) achievable, and it is the **control that attributes A1's gain**.
- **Protocol point for the lead.** The per-seed rule (§15.1(iii)) would apply to seed *units*.
  - Justification: the comparator's published unit is itself an ensemble (MDN = median of 10 inits).
  - This is a reinterpretation that must be decided and recorded **before any A0/A1 result**.
  - Single-network per-seed results are also reported (descriptive), so nothing is hidden.
  - If the lead rejects it, A0 is dropped and A1 uses single networks per seed.

### 4.3 Rank 3: A2 SatWQ-IOP-lite (score 4.5/10), merged from the SatWQ-IOP design with required changes

**What is kept from SatWQ-IOP.**
- An amortised posterior over water IOPs (log Chl, log a_g(440), log b_bp(555), η, S_g, ψ) as a K = 5 mixture in IOP
  space.
- A **fixed, differentiable fp32 forward model**: Bricaud-form a_ph, exponential CDOM, power-law b_bp, Gordon/Lee
  rrs(u), Rrs = 0.52 rrs/(1 − 1.7 rrs).
- A chip nuisance model: aerosol residual τ_a(λ/865)^−α, flat glint G·F_b, adjacency β0·exp(−d_p/ℓ)·(ρ_land − πRrs),
  gain g.
- Student-t reconstruction of the 11 L2A bands of every pixel, plus an ELBO on the 34,428 train-role, ≤ 2021 unlabelled
  chips of `manifests/orbit_ssl_pixels_v1.json`.
- Observation operators per target (chl-a, TSS/SSC via b*_bp, NTU vs FNU via η, Lee-2015 Secchi, DOC via a_g), each
  with a penalised discrepancy head (Takeishi & Kalousis 2021 [F]).
- The closure statistic r feeds a secondary closure-tertile Mondrian CQR.

**Required changes (mine; each is a condition for running A2).**
1. **Gate.** Start A2 only after chain9 reports. Run it if any ssl configuration improves val chl-a by ≥ 1 point over
   its non-SSL twin, **or** if A1 fails its kill criterion. Otherwise A2 goes to Test-C only. Unlabelled-data value is
   the premise of A2, and chain9 tests it cheaply.
2. **Same encoder as A1** (including d_p and the land ring), so that A2 = A1 + physics decoder and the attribution
   is clean.
3. **Scope cut** from about 45 to about 18 GPU-h:
   - drop the per-pixel Δz_p latents;
   - **1 selectable config** (λ_rec = 1.0 fixed a priori) × 3 seed units × 3 inits;
   - **2 non-selectable ablations** (no reconstruction; no nuisance), 3 seeds × 1 init.

   The "physics vs generic SSL" ablation is already supplied by chain9's ssl1.
4. **Constants.**
   - Every physical constant must be checked against the primary text **before the A2 freeze**: g0/g1,
     0.52/1.7, Lee 2013 m0–m3/γ, Lee 2015 0.14/0.013/2.5, Bricaud A/E, Pope & Fry a_w. They are UNVERIFIED (recalled
     from memory).
   - Learnable constants keep Gaussian priors fitted on GLORIA train-role ≤ 2021 rows only.
5. **Coordinate-free by default** (as designed; consistent with A1 c2).
6. **FNU fitting** is allowed **only in this arm and A1**. That needs a §2.1 amendment for the new arms; the headline
   arm is unchanged. FNU is tested **only on Test-S years**, because test-role 2024–25 FNU is spent as a descriptive
   stratum at the headline look.
7. **Parity**: add GBM-phys (QAA/Lee features) to the GBM family; MDN_B family as in A1.
8. Diagnostics (a)–(e) of the design are kept and reported descriptively. The closure-Mondrian CQR is secondary and
   never affects selection.

**Honest assessment.**
- Analysis-by-synthesis gives interpretability (IOP and nuisance products), a physically grounded uncertainty flag,
  and plausible extrapolation (humic boreal lakes in Track G; bloom extremes).
- But the forward model is approximate for turbid or CDOM-rich inland water, optically shallow lakes and scums.
- There is **no direct precedent** that a physics decoder beats a black-box MDN on real inland matchups. PROSAIL-VAE
  [F] covers land and is unsupervised.

P(A2 beats A1 on val) ≈ 0.2. Its main value may be scientific (an interpretable AC/adjacency decomposition) rather than
a SOTA margin.

### 4.4 Rank 4: A3 SatWQ-TOA dual-stream (score 4.0/10)

- **Idea.** Feed L1C TOA reflectance as a second token stream (shared wavelength tags plus a stream embedding) beside
  L2A, so the network can learn its own water-specific correction.
- **Why.**
  - Smith 2021 [F/W]: MDN_B on Rayleigh-corrected ρ_s beat AC Rrs.
  - AC is the shared floor (F1, F4).
- **Data.**
  - Earth Search `sentinel-2-l1c` exists [F]: licence is the ESA Sentinel legal notice; temporal extent from
    2015-06-27.
  - `sentinel-2-c1-l1c` returned **404** [F], so there is no Collection-1 L1C there.
  - Access cost / requester-pays status: **[U]**.
  - Centre 15×15 × 13 bands uint16 ≈ 6 KB per chip, so about 0.9 GB for 150k chips. This needs the lead's disk
    approval (M2 cap "as already built"; 87 GB free today vs the 60 GB floor).
- **Parity**: an MDN_B-ρ_TOA member (the published Smith 2021 variant) and GBM TOA features.
- **Risks**: L1C/L2A co-registration and offset traps (documented in report 02); several days of extraction.
- **Placement**: Test-C / Track G, unless it is ready before the v2 freeze.
- **Compute**: about 5 GPU-h.

### 4.5 Rank 5: A4 AlphaEarth lake prior (score 3.0/10)

- **Idea.** Use the previous calendar year's AlphaEarth annual embedding (causal: no same-year imagery), averaged over
  the water polygon, as a 64-d static prior. It goes to SatWQ-Net *and* the GBM (parity).
- **Data.** Under 0.2 GB. Annual layers cover 2017–2024 per arXiv [F]; [R02] says 2017–2025 are in the bucket. So
  2015–2017 rows have no prior.
- **Expected effect.** It helps lake-level bias for every model alike, so a SatWQ-specific margin is unlikely.
- **Value.** It answers gap G7 of report 02 (a first test of a foundation-model prior for inland retrieval). Run it
  only as a declared ablation if the GPU budget allows.

### 4.6 Rejected or dropped
| Idea | Why dropped |
|---|---|
| Larger SatWQ-Net (d128+) | dev_v3: 55.6 vs base 51.2; the residuals are shared (F1) |
| Fine-tuning Prithvi-EO-2.0 / Clay / Copernicus-FM as the core | Patch inputs (224 px) vs point samples; L1C/HLS inputs; 11 GB VRAM; no inland-WQ evidence (report 02 §a.6, G7 [R02]). At most a frozen-feature ablation later |
| SatWQ-Net + GBM blend as the v2 point estimate | Weaker on dev_v4 (−1.2 vs MDN_B); costs configuration counts |
| Tracks or strata on low chl (≤ 2.6 µg/L), high AOT, high SZA or processing baseline | **Seen on val** (F4/F5). Forbidden by rule 2 / §15.5 item 5 |
| Spatial residual kriging / stronger coordinate embeddings | Coordinate shortcut (Prasad 2026); test lakes sit in 50-km blocks away from training anyway; Track G risk |
| Per-lake test-time training (Sun 2020) | No inland-WQ evidence (the JAG 2026 TTT chl-a paper is [U]); adds an optimisation loop inside the single-look scorer |
| Phycocyanin / microcystin heads | S2 has no 620 nm band; WQP microcystin 2021–25 is reserved by D2n |
| SatWQ-IOP at full scope (7 ablations × 3 seeds, 45 GPU-h) | Cut down to A2 (§4.3) |

### 4.7 Cross-cutting rules for every new arm (fair-test plan)
1. **Data version.** dev_v5 plus the co-located multi-target join becomes a new version, **dev_v5m** (same split v3,
   same QC, same de-duplication). Every model, comparators included, is refitted on it.
2. **Counting.** Every val-scored configuration is appended to `configs_tried.jsonl` (235 rows today). Caps:
   - A0: 1;
   - A1: 3;
   - A2: 1 selectable + 2 non-selectable ablations;
   - A3: 1;
   - A4: 1 (ablation).

   That is at most 7 selectable plus 2 ablations. No +gbm blends for v2.
3. **Selection.** The protocol §6 rule (lowest val chl-a MdSA on `valxval_lake`) among the new arms gives "v2". v1 is
   frozen as authorised for the headline and carried unchanged.
4. **Comparator tuning parity.** Each comparator family gets at least as many val-scored members as the arm it
   faces. The val-selected member is the comparator (optimistic for the baselines).
5. **The look (M2S/test).**
   - It uses the protocol §9b guard with every item of rule 7 (pid/host-bound claim, BaseException abort, sealing,
     rescore refusal, import-closure hash, artifact verification before the claim).
   - **Two arms are scored, v1 and v2, with Holm over the 2 arms** within each endpoint.
   - The H1-type endpoint is an intersection–union test across the MDN_B family, the GBM family and kNN, with the
     per-seed-unit rule.
   - Secondary family: Secchi, TSS and NTU (Holm). Track N family: TP, TN, DOC (Holm).
   - E4a events: secondary. Small lakes: descriptive unless the count-based MDE ≤ 10 (§7.1).
6. **Disk.** No new imagery without the lead's approval and a `df -h` check (stop below 60 GB free).

---

## 5. New numeric predictive targets (multi-task)

**Co-location rate.** These were measured this session from **keys only** on the 2024–25 WQP S2 matchup build
(`s2_wqp_matchups`, all protected; no value read). The rate is the share of scene-screened matchup site-dates
(51,406 in total; 24,566 with chl-a) that also have a same-date WQP sample of the target.
- The QC-pass fraction for chl-a is about 58% (14,195 of 24,566).
- "Est. QC-pass 2024–25" = co-located at *any* matchup site-date × 0.58. It is an optimistic estimate for river-heavy
  targets, because river chips fail the JRC/SCL checks more often.
- The 2015–2023 hist build should add several times more.

| Target (unit) | Physical link to S2 | WQP label pool, all years / sites | Co-located: % of chl-a site-dates (% of all site-dates) | Est. QC-pass 2024–25 | Proposed role | Unseen test block |
|---|---|---|---|---|---|---|
| **TP** (mg/L P) | Indirect (trophic coupling with chl-a and Secchi) | 4,952,986 / 268,291 (4.8% censored) | **87.0%** (55.2%) | ~16k (~12k at chl-a site-dates) | **Track N primary**; auxiliary in A1 (weight 0.5) | Test-S years (test-role × 2015–23) |
| **TN** (mg/L N) | Indirect | 1,844,020 / 94,892 | 36.9% (23.1%) | ~7k | Track N primary | Test-S |
| **DOC** (mg/L C) | Via aCDOM(440) absorption | 673,329 / 60,886 | 11.5% (7.7%) | ~2.3k | Track N primary (if MDE ≤ 15) | Test-S |
| TOC (mg/L C) | Via CDOM, partly | 1,171,730 / 90,340 | 16.9% (10.7%) | ~3.2k | Auxiliary only (AquaCorr unit check) | — |
| Turbidity FNU | b_p near 860 nm | 496,902 / 12,531; plus USGS sondes: 958 sites, 2.07 M site-days | 4.4% (5.4%) | ~1.6k (lake share 0.32) | Auxiliary (A1/A2); descriptive test | Test-S years only (2024–25 FNU is spent at the headline) |
| SSC (mg/L) | b_bp (mineral) | 674,441 / 12,697 (89% USGS) | 2.4% (5.5%); lake share 0.18 | ~1.6k (mostly non-lake) | Auxiliary; mostly non-lake | Test-S non-lake (descriptive) |
| Transparency tube (cm) | Like Secchi (Tobit) | 632,927 / 27,597 (27.8% censored) | 1.1% (1.6%); lake share 0.14 | ~0.5k (~0.07k at lakes) | Descriptive only | — |
| Pheophytin (µg/L) | — | 795,345 / 69,963 | 31.4% | — | **Not a target**: use for the chl-a method class (corrected vs uncorrected) | — |
| aCDOM(440) (1/m) | Direct | WQP 5,996 / 1,716; GLORIA 3,344 after QC [R02] | n/a | < 100 | GLORIA pretraining head; descriptive | — |
| Carlson TSI / trophic class (derived) | From the chl-a, Secchi, TP predictive distributions | derived | — | — | Descriptive product (manager-facing) | Test-S |
| **Excluded** | phycocyanin (no 620 nm band on S2); microcystin (D2n reservation); surface temperature (no thermal band on S2) | | | | | |

**Notes.**
- **TP/TN are optically inactive.** Any skill comes through trophic coupling. The trophic-chain baseline (§4.1) is
  mandatory, and I expect location+season and the chain baseline to be strong. These are *harder* targets in the
  §15.5 sense, with clear practical value: nutrient management, eutrophication listing.
- **Chl-a method class.** Recommendation: once dev_v5's WQP rows carry the CharacteristicName (corrected /
  uncorrected / unknown), add it as a **data-level input to every model**. It is a label-protocol covariate, and parity
  applies.
  - The product is "chl-a, corrected".
  - At test, the row's recorded method is passed.
  - This is a lead decision. It reduces label heterogeneity (only 1,437 of 21,219 dev_v4 chl-a values are
    lab-flagged).

---

## 6. More data: verified sources, licence and size

| Source | Status / verification | Licence | New disk | What it adds |
|---|---|---|---|---|
| **WQP 2015–2023 S2 hist build (dev_v5)** | Running. At 11:33 on 10 Nov: 24,800 of 35,052 scenes, **87,216 chips ok**, about 1,080 scenes/h, so **ETA about 21:00 on 10 Nov** (read from the build log) | US federal public domain / WQP terms (`docs/licenses.md`) | ≤ 3.3 GB (approved) | Same-source (WQP) fit and val rows; test-role 2015–23 rows feed **Test-S** |
| **WQP WQX3 discrete, other targets** (`processed/wqp_discrete`, 2.8 GB on disk) | Counts from `_coverage_by_variable.csv`; co-location measured this session (§5) | As above | **0 GB** (labels only, joined to existing chips) | TP, TN, DOC, TOC, FNU, SSC, pheophytin |
| AquaCorr-Global v1.0 | [F] Zenodo: CC BY 4.0, 972.8 MB, 1,984,387 matchups. On disk; 117,889 S2 chips extracted | CC BY 4.0 | 0 | DOC/TOC/aCDOM columns not used yet (units unstated). Most remaining unchipped S2 rows duplicate WQP/AquaMatch. Low marginal value; do not extract more |
| USGS continuous sondes, daily FNU (M1's `processed/sonde_daily`) | On disk: `turb_fnu_*` at 958 sites (745 active since 2016), 2.07 M site-days | US federal public domain | 0 labels; chips ≈ 0.1–0.2 GB centre-only or 0.5–1 GB full | Unit-consistent FNU with daily min/max (stable-day QC). **Exclude the 729 reserved `sonde_newsites_list.json` sites.** WY2024–25 is EW-v2 test-role, so never use it for selection |
| GLORIA | [F] Lehmann 2023; on disk | CC BY 4.0 [R02] | 0 | Pretraining; aCDOM head; A2 priors |
| Track G: EEA Waterbase, SYKE, SLU MVM, Ontario, ECCC | `orbit_track_g.md` (DRAFT). Licences "to verify". **Not re-verified this session (no search budget)** | [U] until checked | ≤ 8 GB (approved cap) | New geography; unseen non-US block |
| S2 L1C TOA (A3) | [F] Earth Search `sentinel-2-l1c` exists; `sentinel-2-c1-l1c` 404; access cost [U] | ESA Sentinel legal notice | ~0.9 GB (needs approval) | Learned-AC input |
| Lake-context bank, phase B (A1) | Built by the existing chip pipeline | As S2 | ~0.55 GB (needs approval) | Sampling-independent context |
| AlphaEarth annual embeddings (A4) | [F] arXiv (annual layers 2017–2024); access/licence [R02] | CC BY 4.0 [R02] | < 0.2 GB | Static lake prior |
| EPA NLA 2017/2022 | On disk | US Government work | 0 | **Do not use as an M2 test block.** NLA 2022 first visits are the M5-X X-L test block (turbidity shared), and NLA reaches M2 through WQP anyway |
| LAGOS-US | Per Track G | — | 0 | De-duplication only |
| CAML / Tick Tick Bloom cyanobacteria counts | [R02] only (SeaBASS; about 23,570 points) | [U] | ~1 GB chips | A possible later bloom-severity target. Not proposed now |

**Disk.** 87 GB free at 11:35 on 10 Nov (`df -h`); the floor is 60 GB, shared by every workstream. Everything
proposed for the first two weeks needs **0 GB of new imagery**. Phase B, A3 and Track F together would need about
2–2.5 GB, all behind the lead's approval.

---

## 7. Harder tracks: practical justification, unseen block, power

### 7.1 Test-S: unseen lakes × archive years 2015–2023 (new; the strongest lever)

**Definition.** In-lake, non-buffer rows of **test-role** water bodies (frozen split v3) dated 2015-06-23 to
2023-12-31, from every chip source (AquaCorr and WQP hist), under the identical QC and de-duplication.

**Status.**
- These rows are protected (protocol §2.2: every test-role row in any year).
- They were dropped right after split assignment in every dev cache, and **no scorer touches them**. `orbit_test.py`
  scores only the headline, Test-B, pooled and track-U cells.
- They were excluded from the SSL manifest and from GLORIA pretraining.
- **They are unspent.**

**Practical justification** (written before any result; none exists). The dominant real use of a satellite
water-quality archive is **retrospective reconstruction** of 2015–2023 histories for lakes that were never monitored:
- trend assessment;
- trophic-status listing;
- baselines for restoration.

That is exactly "unseen lakes, archive years". It also answers the power problem documented in §5a/§9a: the headline
has 87 blocks.

**Counts** (dev_v4 split report, `protected_cell_counts`; counts only):

| Target | test role × train years (≤ 2021) | test role × val years (2022–23) | Test-S total (dev_v4) | Headline (2024–25) for reference |
|---|---|---|---|---|
| chl-a | 2,001 rows / 510 lakes / 143 blocks | 638 / 243 / 79 | **2,639 rows; 143–222 blocks** (union not computed) | 669 / 225 / 87 |
| Secchi | 1,725 / 381 / 88 | 447 / 154 / 41 | **2,172 rows; 88–129 blocks** | 1,442 / 394 / 92 |
| TSS | 79 / 24 / 23 | 13 / 6 / 6 | 92 (descriptive) | 221 / 79 / 57 |
| turbidity | 36 / 17 / 15 | 13 / 8 / 6 | 49 (descriptive) | 490 / 98 / 53 |

All dev_v4 Test-S rows are AquaCorr-sourced. dev_v5 adds the WQP 2015–2023 test-role rows; final counts come from the
dev_v5 split report.

**Power** (rough scaling of the registered §5a test SE by √(87/B); the §5a script must be re-run on the dev_v5
protected counts before freezing):

| Blocks B | chl-a MDE80 vs MDN_B | chl-a MDE80 vs GBM | Power at the val gains (−4.4 / −8.0) | Joint |
|---|---|---|---|---|
| 87 (headline) | 10.3 | 14.3 | 0.22 / 0.35 | 0.21 |
| 150 | ≈ 7.8 | ≈ 10.9 | ≈ 0.35 / ≈ 0.54 | ≈ 0.25–0.3 |
| 200 | ≈ 6.8 | ≈ 9.5 | ≈ 0.45 / ≈ 0.66 | ≈ 0.35 |

At a margin of +3 points more than v1 (vs both), the joint power at B ≈ 150 rises to about 0.6–0.65.

**Rules to pre-register.**
- (a) **Scene-disjointness.** Count the rows whose (MGRS tile, acquisition date) also occurs among fit/inner rows.
  - If the scene-disjoint subset keeps ≥ 60% of rows **and** ≥ 80% of blocks, it is the primary cell, with all rows
    as a sensitivity. Otherwise the roles swap.
  - Decided from counts only, before the freeze.
- (b) Small-lake stratum (HydroLAKES < 1 km²): confirmatory **only if** its count-based MDE ≤ 10; otherwise
  descriptive.
- (c) Claims are scoped "unseen lakes, archive years". The headline remains the strict claim.
- (d) **Reserve it now.** Add Test-S to `sota_campaign.md` "Reserved unseen blocks", so that no workstream ever trains
  on test-role rows of `s2_wqp_matchups_hist` or the AquaCorr chips. The Fuse consumers of WQP matchup tables in
  particular must be checked by the lead.
- (e) It must be declared **before the M2/test look runs**, so that the declaration is demonstrably independent of
  any test result.

### 7.2 Track N: new targets on the Test-S block
- **Block**: the Test-S rows (test-role lakes × 2015–2023) of TP, TN and DOC. FNU, SSC and aCDOM are descriptive.
  2024–25 test-role rows are **not** used for new targets: those lakes and scenes are spent by the headline look.
- **Practical justification**: nutrient and organic-carbon status of unmonitored lakes (TMDL / eutrophication
  screening). These targets are *harder* than chl-a because TP/TN are optically inactive.
- **Family**: TP, TN, DOC, with Holm. A target whose count-based MDE exceeds 15 MdSA points is pre-declared
  descriptive.
- **Power**: unknown until the dev_v5m join (counts only). By the co-location rates, TP should approach the chl-a WQP
  counts.

### 7.3 Track G: required amendments before its freeze (the file is still DRAFT; nothing downloaded)
1. **Correct the premise.** dev_v4 holds 485 non-US rows (Caravan-Qual 422, GLORIA 63; 151 in-lake; train-role 277,
   val-role 208).
   - Add de-duplication of G1–G5 against them (≤ 100 m, same date).
   - Restate the unseen block as non-US test-role lakes × 2024–25 *from G1–G5*. No test-role European rows exist in
     dev, since protected rows were dropped.
2. **Coordinate-free prediction for SatWQ-Net on Track G** (Prasad 2026). For v1, declare the context-off output
   primary. For v2, declare c2. Decide before any download.
3. **GBM comparator family on Track G** = {imagery-only, imagery+context}. The member that is stronger on the
   Track-G val cell is the comparator: optimistic for the baseline, and comparator selection only.
4. **The 1-configuration cap goes to** "the most recently frozen M2 configuration at the time of the Track-G freeze"
   (v1 or v2). Fix this rule now.
5. Power: as drafted, descriptive if under-powered. Expect few blocks. EEA/SYKE/SLU lakes cluster, so the MDE may be
   large.

### 7.4 Track F: USGS continuous FNU (more data; descriptive track)
- Use **train-role** sonde sites (by the M2 split rule) as training data for the FNU head, with stable-day QC
  (daily max/min < 1.5).
- Exclusions:
  - the 729 reserved M1 sites;
  - every WY2024–25 row for selection (EW-v2 test-role).
- The test is likely **descriptive**: river sites fall in 200-km non-lake blocks, and the non-lake test cells today
  have only 11–20 blocks.
- It needs about 0.1–1 GB of chips (approval) and coordination with M1.

### 7.5 Test-C (prospective, already planned)
WQP October 2026 – September 2027, scoreable around mid-2027. It is the fallback for any arm not frozen before the
M2S/test look (A3, A4, or A2 if it is gated out). It is the cleanest block for all targets.

### 7.6 Explicitly not proposed
Oligotrophic, high-AOT, high-SZA and processing-baseline strata were all **seen on val** (§1.2, F4/F5). The same goes
for any stratum chosen from `diag3.py`-style breakdowns.

---

## 8. Sequencing within the shared 2-GPU budget and the configuration caps

**Budget assumptions.**
- GPUs 0/1 are shared by about 7 workstreams, so M2 realistically gets about 3–4 GPU-h per day.
- Every job is ≤ 2 h, through `acquire_gpu`, and checkpointed.
- chain6–9 are already queued: dev_v5, candidate, seed-variance round, SSL.

| When (2026) | Step | GPU-h | Gate / prerequisite |
|---|---|---|---|
| **10–11 Nov** | **Lead decisions and pre-registration (Appendix A)**: Test-S + Track N + arms A0–A4 with caps, the seed-unit rule and the parity list; reserve Test-S in `sota_campaign.md`; Track G amendments 1–4. **Before the M2/test look and before any new-arm result** | 0 | Lead |
| 11–12 Nov (CPU, ≤ 6 threads) | Build dev_v5m: multi-target join (fit/val values; protected counts only); context-chip index; d_p/land-ring features from the existing chips (about 6–10 CPU-h); power for Test-S/N from protected counts (§5a) | 0 | dev_v5 cache (chain6) |
| 12–13 Nov | chain6/7 finish. **The headline freeze and the M2/test look proceed independently: do not wait for v2** | (chain6) | — |
| 13–15 Nov | A0 (9 networks) + comparator refits on dev_v5m (CPU) | ~3 | Seed-unit rule approved |
| 14–19 Nov | A1 c1–c3 (27 networks) | ~16 | A0 done (control) |
| 16–18 Nov | chain9 SSL outcome, then the **A2 gate** (§4.3 change 1) | (chain9 ~15) | — |
| 18–24 Nov | A2 if gated in (1 config + 2 ablations); otherwise skip to Test-C | ~18 | A2 constants verified |
| ~22–25 Nov | v2 = §6 selection among A0–A2; dry run on val; protected-count power; **independent review**; lead freeze `FREEZE_orbit_S.json`; **single look M2S/test** (Test-S + Track N; arms v1 + v2; Holm) | ~1 (inference) | All of §4.7 |
| After the Track G licence checks (+6–9 days) | Track G freeze, then download, then dev_v6 refit (1 config, coordinate-free), then M2G/test | ~3–6 | Track G amendments |
| Optional | A3 (L1C extraction, then 1 config) and/or A4 (ablation) | ~5 / ~3 | Disk approval; A1 shows the context/AC hypothesis has legs |
| Mid-2027 | Test-C prospective look for anything not frozen above | — | — |

**Totals.** The minimal path (A0 + A1 + Test-S look) is about **20 GPU-h**. The full plan (adding A2, Track G, A3 and
A4) is about **48 GPU-h**. At 3–4 GPU-h per day that is 1–2 weeks of M2's share for the minimal path.

**Pre-registration checklist** (in this order; each item dated in the protocol before the result it governs):
1. Test-S definition, scene-disjointness rule, MDE rules, scoping language, cross-workstream reservation.
2. Track N target list, unit rules, censoring treatment, Holm family, trophic-chain baseline.
3. Arms A0–A4: architecture text, configuration caps, the **seed-unit interpretation**, the parity comparator
   families (incl. joint-MDN and GBM-phys), the selection rule, the A1 kill criterion, the A2 gate, and the 2-arm Holm
   at the look.
4. §2.1 amendment: FNU may be fitted by the new arms only.
5. Track G amendments 1–4 (before its freeze and download).
6. The dev_v5m data version and the chl-a method-class covariate (if adopted, applied to all models).
7. `testlook.py`/`orbit_test.py` extensions (context chips loaded only inside the claim; new cells), followed by an
   independent code review.

---

## 9. Honest outlook (probability of earning a §15.1 label)

These are subjective estimates made before any new result. They include the winner's-curse shrinkage of val gains
selected from 235+ configurations.

| Track / endpoint | Model | P(SOTA label) | Main reason |
|---|---|---|---|
| Headline H1, chl-a (2024–25) | v1 | **0.10–0.15** | Joint power 0.21 at the (optimistic) val gains; 2 of 3 seeds fail the per-seed rule on val |
| H2, Secchi (headline) | v1 | 0.2–0.3 | 0.91 power vs MDN_B, but vs GBM the CI includes 0 (power 0.43); Holm over 3 targets |
| Event E-H2, E4a (ΔBrier *and* ΔAUPRC vs comparators 1–4) | v1 | 0.2–0.3 | ΔBrier vs MDN_B excludes 0 on val, but ΔAUPRC vs MDN_B does not (+0.021 [−0.008, +0.059]). E-H1 (BSS > 0) is very likely (≈ 0.9) |
| **Test-S, chl-a** | v1 / v2 | 0.2–0.3 (v1); **0.3–0.45** (v2, if A1 reaches its target) | Roughly doubled blocks; the per-seed-unit rule; A1 must add about 3 points under parity |
| **Test-S, Secchi** | v1 / v2 | **0.35–0.5** | Already per-seed robust on val; the GBM margin is the question |
| Track N: TP / TN | v2 | 0.1–0.2 each | Optically inactive; location/season and the trophic-chain baseline are strong |
| Track N: DOC | v2 | 0.15–0.3 | Optically linked but scarce; may end up descriptive by MDE |
| Track G (international) | v1 or v2, coordinate-free | 0.15–0.3 | Unknown counts; coordinate shortcut may hurt GBM+context, but the imagery-only GBM member is in the family |
| Track U | v1 | ≈ 0.1 | Joint power 0.16 |
| Test-C (mid-2027) | best frozen | ≈ the Test-S odds | Cleanest block, but late |
| **Any scoped M2 label by early December** | — | **≈ 0.4–0.5** | Several independent blocks; Secchi on Test-S is the most likely |
| **Any chl-a label by early December** | — | **≈ 0.25–0.35** | |

**What would change my mind.**
- If chain9 SSL gives ≥ 2 points on val, raise A2 and Test-S chl-a by about 0.1.
- If A1 c3 (no context) ≈ c1, the multi-date hypothesis is wrong: drop phase B and A3.
- If the Test-S scene-disjoint subset is small (< 60% of rows), Test-S power falls toward the headline's.
- If GBM-context-summaries match A1 on val, the result is still progress for SENTINEL (a better system), but not a
  SatWQ-specific SOTA. Report it as such.

---

## Appendix A: Draft pre-registration text for the lead (proposals; not written into any protocol by me)

**A.1 Protocol §11.7 Test-S (proposed).** "Declared 2026-11-__, before the M2/test look and before any model has
been evaluated on these rows.
- Cell `testxarchive_lake`: in-lake, non-buffer rows of test-role water bodies (split v3) dated 2015-06-23..2023-12-31,
  all chip sources, identical QC and de-duplication.
- Justification: retrospective reconstruction for unmonitored lakes; headline power (§5a).
- Primary/sensitivity by the scene-disjointness count rule (≥ 60% of rows and ≥ 80% of blocks).
- Small-lake stratum confirmatory only if MDE ≤ 10.
- Scored once in look `M2S/test` under the full §9b guard, with arms v1 (frozen headline config) and v2 (selected by
  §6 among the arms of §11.8) and Holm over the arms.
- Reserved for all workstreams from this date."

**A.2 §11.8 New arms (proposed).** The architecture text of §4.1–4.5, with these elements:
- caps: A0 1, A1 3, A2 1 + 2 ablations, A3 1, A4 1;
- seed unit = 3 inits + EMA, if the lead approves; otherwise single networks;
- parity families: MDN_B {7/12 bands, ±context/ring, single/joint head}; GBM {v1, +ring/context summaries, +phys};
  kNN; trophic-chain for TP/TN;
- the A1 kill criterion (≥ 2 points over v1-dev_v5 on val chl-a and Secchi);
- the A2 gate (chain9 ssl ≥ 1 point, or A1 fails);
- data version dev_v5m.

**A.3 §11.9 Track N (proposed).**
- Targets TP, TN, DOC (family, Holm); FNU, SSC, aCDOM descriptive.
- Block = Test-S rows of these targets.
- Censored values handled through the interval likelihood; the trophic-chain baseline is a comparator.
- MDE > 15 → descriptive.

**A.4 `orbit_track_g.md` amendments.** Items 1–4 of §7.3.

## Appendix B: Diagnostics provenance (this session; read-only)
- Python: `/home/bcheng/miniconda3/bin/python -I`, 2–4 threads, niced.
- `diag1.py <runs>/orbit/compare_satwq_v4_noaug/val_row_preds.parquet`: per-model MdSA/SSPB, residual correlations,
  between/within-lake MSE split.
- `diag3.py <same> <runs>/orbit/cache/dev_v4/table.parquet`: strata (seen on val; design motivation only) and lake
  oracles.
- `diag2.py <runs>/orbit/cache/dev_v4/table.parquet`: same lake, same date station pairs (noise floor).
- dev_v4 table profiling: providers, the 485 non-US rows, target counts.
- Key-only co-location count: `processed/s2_wqp_matchups/{matchups,samples}.parquet` (sample_key, site_id, date,
  site_class) joined to `processed/wqp_discrete/variable=*/data.parquet` (site_id, activity_start_date). No value
  column was read.
- Scripts are in `/tmp/claude-1016/-home-bcheng-STOCKHOLM/dbc009e4-477d-4528-910a-e89e964c5c0f/scratchpad/`.

## Appendix C: References and verification status
- [S] Pahlevan N. et al. (2020) RSE 240:111604. doi:10.1016/j.rse.2019.111604
- [S] Saranathan A.M., Werther M., Balasubramanian S.V., Odermatt D., Pahlevan N. (2024) Front. Remote Sens.
  5:1383147. doi:10.3389/frsen.2024.1383147
- [F] Smith B., Pahlevan N., et al. (2021) A Chlorophyll-a Algorithm for Landsat-8 Based on Mixture Density
  Networks. Front. Remote Sens. 1:623678. doi:10.3389/frsen.2020.623678
- [F] Pahlevan N. et al. (2022) Simultaneous retrieval of selected optical water quality indicators from Landsat-8,
  Sentinel-2, and Sentinel-3. RSE 270:112860. doi:10.1016/j.rse.2021.112860
- [F] Pahlevan N. et al. (2021) ACIX-Aqua. RSE 258:112366. doi:10.1016/j.rse.2021.112366
- [F] Werther M., Odermatt D., Simis S.G.H., et al. (2022) RSE 283:113295. doi:10.1016/j.rse.2022.113295
- [F] Sherf Y., Efrati B., Rozman G., Harel M. (2025; v2 2026) arXiv:2510.24124
- [F] Prasad A., Verma S., Khan M.A. (2026) A Spatiotemporal Transformer for Satellite Retrieval of Lake Turbidity.
  NeurIPS 2026 CCAI workshop. https://neurips.cc/virtual/2026/174697
- [F] Kumari S., Alikas K., Dev S. (2026) Geographic Imbalance and Spatial Generalisation in Machine Learning for Global
  Water Quality Prediction. ICML 2026 poster. https://icml.cc/media/icml-2026/Slides/76520.pdf
- [F] Lehmann M.K., Gurlin D., Pahlevan N., et al. (2023) GLORIA. Sci. Data 10. doi:10.1038/s41597-023-01973-y
- [F] Ross M.R.V., Topp S.N., Appling A.P., et al. (2019) AquaSat. WRR 55. doi:10.1029/2019WR024883
- [F] AquaCorr-Global v1.0 (2026) Zenodo. doi:10.5281/zenodo.21927415 (CC BY 4.0)
- [F] Lee Z., Carder K.L., Arnone R.A. (2002) QAA. Appl. Opt. 41:5755. doi:10.1364/AO.41.005755
- [F] Lee Z. et al. (2015) Secchi disk depth: a new theory and mechanistic model. RSE 169:139–149.
  doi:10.1016/j.rse.2015.08.002
- [F] Zérah Y., Valero S., Inglada J. (2024) Physics-constrained deep learning for biophysical parameter retrieval from
  Sentinel-2 images: inversion of the PROSAIL model. RSE 312:114309. doi:10.1016/j.rse.2024.114309
- [F] Takeishi N., Kalousis A. (2021) Physics-Integrated VAEs. arXiv:2102.13156
- [F] Brown C.F. et al. (2025) AlphaEarth Foundations. arXiv:2507.22291
- [F] Hollmann N. et al. (2025) Accurate predictions on small data with a tabular foundation model. Nature 637.
  doi:10.1038/s41586-024-08328-6
- [F] Romano Y., Patterson E., Candès E.J. (2019) Conformalized Quantile Regression. arXiv:1905.03222
- [F] Sun Y. et al. (2020) Test-Time Training with Self-Supervision. ICML 2020. arXiv:1909.13231
- [F] Lee J. et al. (2019) Set Transformer. ICML 2019. arXiv:1810.00825
- [F] Izmailov P. et al. (2018) Averaging Weights Leads to Wider Optima (SWA). UAI 2018. arXiv:1803.05407
- [F] Earth Search STAC `sentinel-2-l1c` (exists); `sentinel-2-c1-l1c` (404). https://earth-search.aws.element84.com/v1/collections/
- [W] Gordon H.R. et al. (1988) doi:10.1029/JD093iD09p10909; Lee Z. et al. (2013) doi:10.1002/jgrc.20308; Bricaud A.
  et al. (1998) doi:10.1029/98JC02712; Pope R.M., Fry E.S. (1997) doi:10.1364/AO.36.008710; Werdell P.J. et al.
  (2013) doi:10.1364/AO.52.002019; Sterckx S., Knaeps E., Ruddick K. (2011) doi:10.1080/01431161.2010.512930;
  Hieronymi M., Müller D., Doerffer R. (2017) doi:10.3389/fmars.2017.00140; Kennedy M.C., O'Hagan A. (2001)
  doi:10.1111/1467-9868.00294; Mensah P. et al. (2025) arXiv:2511.10387; Jeong (2026)
  doi:10.1016/j.envsoft.2025.106817. **Numeric constants from these are UNVERIFIED.**
- [R02] Topp et al. 2021 (ERL); Tick Tick Bloom / CAML; Dogliotti et al. 2015; Warren et al. 2019; Cao et al. 2020;
  Balasubramanian et al. 2025; Dawson et al. 2025; Prithvi-EO-2.0 / Clay / Copernicus-FM model cards. **UNVERIFIED in
  this session.**
- [U] "Pioneering test-time training … Chl-a from Sentinel-2 … China's lake eutrophication" (JAG 2026): unverified in
  report 02 as well. The absence of prior multi-date lake-context retrieval models is an UNVERIFIED absence.
