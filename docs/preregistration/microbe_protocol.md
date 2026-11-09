# M3 SENTINEL-Microbe — pre-registration protocol (16S community → co-located water quality)

**Status:** DRAFT v0.1, written 2026-10-24 **before the NEON data were available and before any
SENTINEL-Microbe model or baseline was fit** (phase 1 = implementation + cross-validation on
development data only). The final test (§8) is not scored in phase 1; it is scored once after the
configuration is frozen (§9). Data-dependent constants that can only be fixed once the data exist
(exact holdout cut, sample counts) are fixed in §2.3 by deterministic rules written here, and
recorded in §10 *before* model selection, with the sha256 of the frozen split file.
**Authors:** Bryan Cheng (AI-assisted). **Implements:** RESEARCH_PLAN.md §5.3, §14, §15.2 (H-G1 M3
primary), §15.5 amendment of 2026-10-24; docs/research/03 §d.1–d.4, §e.

## 1. Question, primary endpoint, claim limits

From a NEON surface-water 16S community sample, predict the water quality measured at the same
place and time: TN, TP, DOC, nitrate (grab chemistry), specific conductance, turbidity, dissolved
oxygen, chlorophyll-a (co-located sonde).

- **Primary endpoint (H-M3, RESEARCH_PLAN §15.2).** On the final test (§8: leave-one-NEON-domain-out
  × time-out), pooled R² of the frozen SENTINEL-Microbe model exceeds **both** (a) the tier-0
  season+location baseline `RF[DOY+location]` (the operational form of "DOY + latitude + site mean":
  under domain hold-out the held-out site's mean does not exist, see §6) and (b) `RF[genus-CLR]`,
  with the paired site-cluster bootstrap 95% CI of the R² difference excluding 0 for each, for ≥ 2 of
  the 8 targets, Holm-corrected across the 8 targets (intersection–union: a target counts only if
  both comparisons pass).
- **Alternative primary (first-in-class route).** Cross-primer transfer (train V4V5 → test V3V4 in a
  held-out domain, and reverse; §6) where the SeqSet model beats the tier-0 baseline and
  `RF[genus-CLR]` on ≥ 2 targets, CI excluding 0, with a dated novelty check
  (`docs/novelty_checks/microbe.md`).
- **Claim limits stated in advance.** Absolute skill is expected to be modest (the closest published
  comparison, Hermans et al. 2024, reported R² ≈ 0.02 for stream N under a random split). 16S is
  sampled monthly (streams) or bimonthly (lakes/rivers): **no daily early warning is claimed or
  attempted.** A target on which the tier-0 baseline matches the model is reported as "explained by
  season and location", with the number. All 8 targets are reported, including those with skill ≈ 0.

## 2. Data

### 2.1 Sources (NEON, CC BY 4.0, RELEASE-2026; provisional months excluded)

| Role | Product | Content used |
|---|---|---|
| Community | DP1.20141.002 (surface water community taxonomy, RELEASE-2026 OTUs `ay26_16s_otu.*`) | per-run taxon tables: dnaSampleID × OTU count, OTU representative sequence, RDP taxonomy |
| Community (cross-release only) | DP1.20141.001 (superseded, 2014–2018 ASVs, older pipeline) | per-sample ASV tables with sequences |
| Primer set / batch | DP1.20282.001 (marker gene sequences, metadata only) | PCR amplification table: forwardPrimer/reversePrimer/targetSubfragment per dnaSampleID — the authoritative primer label (never inferred from date) |
| Grab chemistry | DP1.20093.001 `swc_externalLabDataByAnalyte` | TN, TP, DOC, NO3+NO2-N (mg/L) |
| Sonde | DP1.20288.001 → `processed/neon_sonde_daily` | daily means of specific conductance (µS/cm), DO (mg/L), turbidity (FNU), chlorophyll-a (µg/L, never RFU) |
| Nitrate (secondary) | DP1.20033.001 → `processed/neon_nitrate` | SUNA daily mean NO3-N (secondary target only) |
| Site table | NEON `/sites` + locations | domain, aquatic type, latitude, longitude, elevation |

All inputs are read through their manifests (`sentinel.core.manifest`), cited by name in every
registry entry. The joined design (`processed/microbe_wq`, built by
`scripts/data/build_microbe_wq.py`) has its own manifest and datasheet.

### 2.2 Unit of analysis and join rules (fixed before any target was looked at)

- **Community event** = all 16S dnaSampleIDs of one site with the same local collection date and
  primer set (field replicates and technical replicates summed). Events with < 1,000 reads are
  excluded. All 16S reads are kept, including chloroplast/mitochondria OTUs (chloroplast reads are a
  legitimate phototroph signal for chl-a; stated, not filtered).
- **Grab chemistry match:** primary = the chemistry event with the **same `parentSampleID`** as one
  of the event's genetic samples (NEON's own field-visit key: the 16S filter and the chemistry
  bottles are sub-samples of one field super-parent at the same station). Fallback = the nearest
  chemistry event at the same site within **±24 h** of the microbe collection time
  (`chem_match = 'time'`; excluded in a sensitivity analysis). Otherwise missing — never
  interpolated, never borrowed from another site. Duplicate analyses → mean (data builder).
  Censored (below-detection) values → the published value when present, else the lab detection
  limit, then floored (§3) — substitution at modelling time only, flagged `<target>_censored`.
  (Amended 2026-10-24 before any data were joined: the original draft keyed on `namedLocation`;
  the NEON builder exposes `parentSampleID`, a stricter same-visit key.)
- **Sonde match:** the sensor co-located with the microbe sample (wadeable streams: downstream S2,
  HOR 102; lakes/rivers: buoy, HOR 103), **daily mean on the local-standard-time collection date**
  (days with < 18 hourly values are NaN, NEON final-QF-flagged values masked by the data builder).
  Ablation (reported, not used for selection): mean over collection date ± 1 day.
- Each match records Δt (hours), the matched named locations and the source sample ids.

### 2.3 Development / final-test partition (deterministic rule)

- **Final test (time-out):** events with collection date ≥ **2023-01-01**. If that holds out < 15%
  or > 30% of the eligible events, the cut moves to the 1 January closest to holding out 20% (rule
  fixed now; the realised cut and counts are recorded in §10 before model selection).
- **Development:** all earlier events. All fitting, CV and model selection use development events
  only; `sentinel.models.microbe.design.load_design` refuses holdout rows unless
  `allow_holdout=True`, which only the final scoring script may pass (unit-tested).
- **Cross-release set:** DP1.20141.001 events are used only in the cross-release scheme (§6) and never
  in training of the primary model.
- The partition and every development fold are serialised to `docs/preregistration/microbe_split.json`
  (`splits.split_record`); its sha256 is recorded in §10 **before** any model is compared.

## 3. Targets

`log_tn` (log10 mg/L, floor 0.005), `log_tp` (log10 mg/L, floor 0.0005), `log_doc` (log10 mg/L,
floor 0.05), `log_no3` (log10 NO3+NO2 as N mg/L, floor 0.001), `log_sc` (log10 µS/cm, floor 1),
`log_turb` (log10 FNU, floor 0.1), `do` (mg/L), `log_chla` (log10 µg/L, floor 0.05). Missing targets
are masked (multi-target set model) or the row is dropped for that target (single-target baselines).
Secondary targets (reported, not in the primary endpoint): SUNA nitrate; hypoxia (DO < 5 mg/L) and
trophic classes are deferred to phase 2 unless prevalence in development data ≥ 5%.

## 4. Inputs

- **Community (primary):** OTU/ASV counts and representative sequences. Tabular baselines use genus
  counts → training-fold prevalence filter (≥ 5% of training events, ≤ 500 most prevalent, others
  pooled) → CLR (pseudocount 0.5). The set model uses the 256 most abundant sequences per event
  (coverage re-measured and reported).
- **Community-side descriptors:** log10 reads, log10 number of sequences, primer-set one-hot.
- **Context (tier-0 and labelled "+context" variants only):** day-of-year harmonics, latitude,
  longitude, elevation, aquatic type.
- **Never inputs:** any chemistry or sensor value, site id, domain id, collection year.
- All filters, CLR vocabularies, imputers, standardisers and the id vocabulary are fit on training
  rows only (unit-tested in `tests/test_microbe.py`).

### 4.1 DNA encoder (decision)

Frozen, label-free sequence tables (`sentinel/models/microbe/dna.py`), computed once per unique
sequence:
1. **DNABERT-S** (Zhou et al. 2024; 117M parameters, Apache-2.0; `zhihan1996/DNABERT-S` revision
   `00e47f96cdea35e4b6f5df89e5419cbe47d490c6`, `pytorch_model.bin` sha256 `f3cfc3d0…a1e5f`), mean-pooled
   last layer (768-d), **fp32** inference (fp16 is avoidable error at negligible cost for a 117M
   inference-only model; the fp16-vs-fp32 agreement and the reference-parity check are computed by
   `scripts/eval/microbe_dnabert_parity.py`, registry experiment `microbe_encoder_checks`). Chosen because it is trained contrastively to place fragments of
   the same species together — the property cross-primer transfer needs — and fits an RTX 2080 Ti.
   The authors' `trust_remote_code` module does not instantiate under
   transformers 5.x, so the forward pass is re-implemented in plain PyTorch; parity with the
   reference implementation (instantiated without meta init) is checked by the same script
   (registry metric `parity_max_abs_diff_fp32`, 128 NEON sequences, fp32 CPU). The shipped `tokenizer.json` pads every input to
   2,000 tokens; padding/truncation are switched off (attention masking makes the output identical,
   only memory changes).
2. **k-mer profile** (k = 5, 1,024-d), deterministic.
3. **id** (ablation): learned per-sequence embedding, no DNA information.

A from-scratch CNN over raw sequence is not a phase-1 candidate (cost; may be added in an
improvement cycle, counted).

## 5. Metrics and uncertainty

Pooled out-of-fold R² (`sentinel.eval.metrics.r2`), RMSE, Spearman; per-held-out-domain R² (domains
with ≥ 10 rows) and their median; anomaly R² (within-domain skill net of domain bias); MSE skill vs
`RF[DOY+location]`. CIs: percentile cluster bootstrap, B = 2,000, **resampling unit = site**
(primary), domain-unit CI in `extra`. Paired comparisons: `paired_bootstrap` on rows where both
models predict. Holm across the 8 targets for the primary endpoint. Seeds: every stochastic model is
run with seeds 0, 1, 2; the evaluated prediction is the mean of the 3 seeded fits (same rule for
models and baselines); per-seed R² stored.

**Controls (reported, mandatory):**
- **Permutation null:** the target block is row-permuted (all 8 targets together) before fitting;
  `RF[genus-CLR]` under LODO with 20 permutations, the selected SeqSet with 3; reported as the null
  distribution of pooled R² next to the real score.
- **Site-identity probe:** 5-fold stratified (row-random) classification of site id from (i) genus
  CLR (logistic regression and RF) and (ii) the SeqSet sample embedding; accuracy vs chance. A high
  accuracy means random splits measure site recognition; only grouped schemes are interpreted.
- **Random 5-fold** (row-random) results are reported only as the leakage gap.

## 6. Cross-validation schemes (development data, phase 1)

| Scheme | Registry split | Definition | Role |
|---|---|---|---|
| **LODO** | `neon_dev/lodo_cv` | one fold per NEON domain | **primary; model selection** |
| LOSO | `neon_dev/loso_cv` | one fold per site | secondary |
| primer V4V5→V3V4 | `neon_dev/primer_v4v5_to_v3v4_cv` | train V4V5 events outside domain g, test V3V4 events in g (≥ 5) | transfer (empty inside .002, see §10) |
| cross-release = cross-primer | `neon_dev/release_002_to_001_cv` | train .002 (V4V5) events outside domain g, test .001 (V3V4, older-pipeline ASV) events of 2017–2018 in g | transfer (only sequence models and genus-level tables apply); descriptors disabled |
| cross-release reverse | `neon_dev/release_001_to_002_cv` | train .001 (2017–2018) outside g, test .002 events in g | secondary |
| random 5-fold | `neon_dev/random5_cv` | rows | leakage gap only |

Early stopping (neural models): 15% of the training fold's **domains** (≥ 2; sites under LOSO) held
out (`splits.inner_validation`, seed 0; amended §10); never the held-out fold.

### Models (identical folds and rows)

Baselines (fixed hyperparameters, `sentinel/models/microbe/baselines.py`):
`GlobalMean`; `Ridge[DOY+location]`; `RF[DOY+location]` (500 trees, min leaf 3, max_features 0.3);
`SiteMean+DOY` (training-site mean + DOY harmonics; defined only where the site is in training, i.e.
random 5-fold); `Ridge[genus-CLR]` (alpha by site-grouped inner CV); `RF[genus-CLR]`;
`XGB[genus-CLR]` (600 trees, lr 0.03, depth 4, subsample 0.8, colsample 0.3); `RF[genus-PA]`;
`RF[genus-CLR+DOY+location]`; `TabPFNv2[genus-CLR]` (tabpfn 2.2.1, 300 genera, 8 estimators). All
tabular community models also receive the community-side descriptors.

SENTINEL-Microbe **SeqSet** candidates (`configs/microbe/seqset.yaml`): `seqset_dnabert`,
`seqset_kmer`, `seqset_id` (2 SAB layers, d = 128, 4 heads, PMA + abundance-weighted mean, token
dropout 0.1, AdamW 5e-4 / wd 0.01, batch 64, ≤ 300 epochs, patience 30). DNA-encoded table
candidates: `SeqEmb-Ridge[DNABERT-S]` and `SeqEmb-RF[DNABERT-S]` (relative-abundance-weighted mean
of the frozen DNABERT-S vectors of all sequences in the event + descriptors; Ridge alpha by
site-grouped inner CV; RF as the baseline RF) — primer/release-agnostic like SeqSet. A `+context`
control run of the selected candidate is reported but is not a selection candidate.

**Selection rule (fixed):** among the DNA-encoded candidates (SeqSet ×3, SeqEmb ×2), the one with the
highest mean pooled LODO R² across the 8 targets on development data; ties within 0.005 → fewer
parameters. **Iteration (RESEARCH_PLAN §15.5
amendment):** if the selected model does not meet the §1 bar on development LODO, improvement cycles
continue (diagnose → literature → one change → re-run on development CV only). Every configuration
is logged (TRAINING_LOG.md, registry) and the count of configurations tried is reported with the
final claim; the selection rule above is applied to the full candidate set at freeze time.
Baselines are never weakened.

## 7. Integrity checks (unit tests)

Domains/sites never on both sides of a grouped fold; primer folds contain only the stated primer
sets; inner validation is site-disjoint and inside training; CLR/prevalence/context fills depend on
training rows only; targets never enter features; the development loader refuses holdout rows; the
frozen split hash is deterministic.

## 8. Final evaluation (phase 2 — NOT run in phase 1)

**Test (primary endpoint): LODO × time-out.** For each domain g: fit the frozen configuration on
development events outside g; predict holdout events (≥ cut date) in g. Training never contains a
holdout event or an event of domain g. All baselines re-run identically. Pooled, per-domain and
anomaly R² with site-cluster CIs; paired bootstrap vs `RF[DOY+location]` and `RF[genus-CLR]`; Holm.
Secondary: time-out with all development data (LOSO not required); cross-release.

## 9. Freezing

Before phase 2: selected configuration, this file, `configs/microbe/*.yaml`, `sentinel/models/microbe/*.py`
and `scripts/eval/microbe_final.py` are hashed and recorded here; the test is scored once.

## 10. Execution log (dated; nothing in §1–§8 is changed after CV results exist without an entry here)

- 2026-10-24: protocol drafted; code and unit tests written on synthetic arrays; NEON downloads in
  progress (separate data workstream).
- 2026-10-24 ~22:10 EDT, after `neon_microbe_16s` landed and BEFORE chemistry/sonde were joined or any
  target was inspected (community tables only):
  - DP1.20141.002 (RELEASE-2026) covers 2018-08 → 2024-11 and every event's read-weighted mean amplicon
    length is ~372 bp (V4V5, 515F-Y/926R after primer trimming); DP1.20141.001 covers 2014-08 → 2018-10
    at ~400 bp (V3V4, Pro341F/Pro805R). The 31 .001 samples with a PcrAmplification record are all V3V4
    and the length rule agrees with all 31. **Primer fallback (amendment):** when no PCR record is
    available (the DP1.20282.001 download was incomplete at build time), the primer set is assigned from
    the amplicon itself — read-weighted mean length < 300 bp → V4, < 386 → V4V5, < 440 → V3V4
    (`sentinel/data/microbe_wq.py::primer_from_length`; `primer_source` column) — never from the date.
  - Consequence: within .002 there is effectively one primer set, so the two within-release primer
    schemes are empty; **cross-primer transfer is evaluated as the cross-release scheme**
    (`release_002_to_001`: train V4V5/.002 events outside domain g → test V3V4/.001 events in g; and
    the reverse `release_001_to_002`), which is simultaneously a primer, pipeline (ASV vs 97% OTU) and
    time (2014–18 vs 2018–23) shift. Primer-set one-hots carry no information inside .002.
  - Sequencing depth differs ~10× between releases (median ~15.6k vs ~213k reads per event), so in the
    cross-release schemes the community-side descriptors (log reads, log richness, primer one-hot) are
    **disabled for every model** (`features.sample_descriptors(enabled=False)`).
  - Counts: 1,126 .002 events (≥ 1,000 reads) and 707 .001 events before target matching.
- 2026-10-24 ~22:15 EDT, **split frozen before any target was joined or inspected** (community-only build
  `scripts/data/build_microbe_wq.py --community-only`; `scripts/train/microbe_freeze_split.py`):
  - Amendment: the split is defined over **all community events** (≥ 1,000 reads), not over events
    with ≥ 1 target, so it is independent of the targets (rows without a target are masked per target).
    Events are site × local date (primer set no longer part of the key; 0 mixed-primer events).
  - Holdout rule outcome (§2.3): 2023-01-01 would hold out 44% (> 30%), so the cut moved to the
    1 January closest to 20%: **2024-01-01** (214 of 1,126 events = 19.0%; 33 sites, 19 domains).
    Development: 912 events (2018-08 → 2023-12), 33 sites, 19 domains (26–84 events per domain).
    Cross-release set: 707 DP1.20141.001 events (2014-08 → 2018-10), all in the development period.
  - Token coverage at 256 sequences: 92% of development events are truncated, keeping on average 80%
    (min 44%) of reads; genus-CLR tables use all reads.
  - `docs/preregistration/microbe_split.json` sha256
    **`4fac5438cf65536731fea7f2c5d4b679e2915a15f70eb3b05083334505951a7f`**
    (also `docs/preregistration/microbe_split.sha256`; audit `microbe_split_report.json`).
    `load_design()` refuses to run if the development rows differ from this file.
- 2026-10-24 ~22:35 EDT, **pre-training review fixes** (REVIEW_REPORT_PRE_TRAINING.md, "M3 Microbe";
  still before any target was joined or any model fit):
  - **C1 — cross-release final-test reserve.** DP1.20141.001 events collected **before 2017-01-01**
    (190 events, 15 domains, 24 sites; `configs/microbe/common.yaml: release_holdout`) are never used in
    phase 1. Phase-1 cross-release CV uses the 517 .001 events of 2017-01 → 2018-10 only. Final
    cross-release test (phase 2, once): for each domain g, fit on all .002 development events outside g
    (no .001 events), predict reserved .001 events in g. **Multiplicity:** the H-M3 decision is
    Holm-corrected over 16 hypotheses = 8 targets × 2 primary routes (LODO × time-out; cross-release
    V4V5 → V3V4 reserve); the reverse direction (V3V4 → V4V5) is secondary.
  - **W5 — resampling unit.** A target counts toward H-M3 only if the paired R² difference is positive
    and Holm-significant with **both** the site-cluster and the domain-cluster bootstrap.
  - **W6 — tuning parity.** Before any claim, if a DNA-encoded candidate beats the fixed-hyperparameter
    RF/XGB on development LODO, tuned versions are run with site-grouped inner CV over pre-specified
    grids — RF: min_samples_leaf ∈ {1, 3, 10} × max_features ∈ {0.1, 0.3, 0.6}; XGB: max_depth ∈ {2, 4, 6}
    × learning_rate ∈ {0.03, 0.1} (600 trees) — chosen by 3-fold site-grouped inner CV (inner fits use
    200 trees / 300 rounds for cost, the refit uses the full size) — and the stronger of tuned/untuned is
    the comparator.
    The tuning asymmetry (configurations tried for DNA candidates vs baselines) is reported.
  - **W1 — permutation null:** permute targets within each fold's training rows only; same seed rule
    (mean of seeds 0–2) as the real score.
  - **W3 — primer rule:** read-weighted **median** (not mean) amplicon length; the 4 .002 events that the
    mean rule called V3V4 (BARC, long chloroplast reads) become V4V5.
  - **W4 — chemistry fallback** (±24 h) additionally requires the same named location.
  - **W9 — wording:** the H-M3 comparator `RF[DOY+location]` is the "season + location" baseline of an
    *unseen domain*; in the time-out test `SiteMean+DOY` (site history allowed, sites seen in
    development) is also reported as a context row.
  - **O5 — early stopping** for SeqSet under domain hold-out uses whole training *domains* (15%, ≥ 2) as
    the inner validation (sites under LOSO).
  - Shared-host rule (CONTRIBUTING_V2): ≤ 6 CPU threads, one CPU-heavy job at a time; GPU jobs on GPU 4.
- 2026-10-24 ~23:20 EDT, **split re-frozen after an upstream data change** (before any model was fit
  and before any target value was used for anything but the join audit):
  - The NEON data workstream rebuilt `neon_microbe_16s` (taxon tables that were missing at the first
    build, plus 2,514 PCR records): the community events grew from 1,126 to **1,429** (.002) — the
    earlier frozen file (`4fac5438…`, kept in the scratch archive) no longer matched the data and the
    `load_design` guard refused it. Re-applying the unchanged §2.3 rule to the new community-only build:
    2023-01-01 → 34.6% held out (> 30%) → **2024-01-01** (214 / 1,429 = 15.0%, the 1 January closest to
    20%). Development **1,215 events** (33 sites, 19 domains). Primer sets now come from PCR records for
    all .002 samples (V4V5; the median length rule agrees on 99.9%) and 940/942 .001 samples (V3V4).
  - New `docs/preregistration/microbe_split.json` sha256
    **`aaeb05305afd8df9a1cc677f974d443b5417b1acdbd95ef1160368dd120626f8`** (supersedes `4fac5438…`).
  - Chemistry join fix (data-format finding, before modelling): `neon_chem/chem_event_wide` has one row
    per bottle (`*.RAW.NUT`, `*.FIL.NUT`, replicate `.2/.3` suffixes); from 2019 on, TN/TP/NO3 sit only on
    those rows. Chemistry is now collapsed to the **field visit** (`microbe_wq.visit_key`: strip
    fraction/replicate suffixes; mean over the visit's bottles reporting the analyte; censored = any)
    before the parentSampleID match. Development non-missing: TN 1,170, TP 1,183, DOC 1,138, NO3 1,180.
  - Sonde targets only for events whose 16S named location is S2 or a buoy (33 `reach`, 4 `inflow`,
    3 `littoral` events get no sonde target). Join audit: `docs/preregistration/microbe_join_audit.json`
    (all matched chemistry Δt ≤ 0.15 h; censored: TN 62, TP 6, DOC 1, NO3 151).
  - Sonde (`neon_sonde_daily`) not yet available: phase-1 runs start with the 4 chemistry targets; all
    models are re-run identically when the sonde targets land, and the selection rule (§6) is applied
    to whatever target set is complete at freeze time (stated with the result).
- 2026-10-24 ~23:31 EDT, after the tier-0 LODO baselines (GlobalMean, RF[DOY+location]) had run and
  **before any community model result existed**: `RF[DOY+location]` scored below `GlobalMean` on several
  targets (extrapolation to an unseen domain). To avoid an easy comparator, the H-M3 tier-0 reference
  is made **stricter**: per target, the best development-LODO scorer among {GlobalMean,
  Ridge[DOY+location], RF[DOY+location]} (chosen on development, then fixed for the test). The
  `RF[genus-CLR]` comparison (and, per W6, the stronger of tuned/untuned RF/XGB) is unchanged.
- 2026-10-25 ~01:45 EDT, `neon_sonde_daily` landed; design rebuilt with all 8 targets (frozen-split guard
  passes, 1,215 development events). Development non-missing: SC 730, turbidity 655, DO 683, chl-a 394
  (sonde only at S2/buoy events, ≥ 18 h/day). All phase-1 runs are (re-)run on the 8-target design;
  the earlier 4-target partial runs (`microbe_phase1_cv`, notes "chem targets only") are superseded and
  not used for selection. Data-workstream caveats recorded here (lead, 2026-10-25): no NEON sample was
  sequenced under both primer sets, and V3V4/.001 vs V4V5/.002 also differ in table version,
  bioinformatics pipeline and sequencing platform (MiSeq → NovaSeq); the only temporal overlap is
  Aug–Oct 2018. **The cross-primer transfer test is therefore a confounded primer × pipeline × platform ×
  time shift and is reported as such**; it uses sequences/taxonomy only, never OTU ids. 342 GMCF PCR
  records (2019-10 → 2021-10) list ITS primers but are 16S V4V5 (flagged upstream); our events carry the
  V4V5 label. Provisional records are excluded (none fall in the 2018-08 → 2024-11 .002 window used here).

### 10.1 Improvement cycle 1 (RESEARCH_PLAN §15.5 amendment) — declared 2026-10-25 ~02:30 EDT

Diagnosis from phase-1 development LODO (registry `microbe_phase1_cv`): SeqSet candidates early-stop
after ~5–20 epochs (≈ 1,000 training events) and trail the genus-CLR Ridge baseline; the mean-pooled
DNABERT-S SeqEmb tables lose most of the signal (averaging 768-d vectors over thousands of sequences);
k-mer SeqSet ≥ DNABERT-S SeqSet. Literature (docs/research/03 §a.2; Yoo & Rosen 2025): compositional
(CLR) linear/tree models on taxa are the hardest baseline; gains come from better *units* of
composition. Cycle-1 candidates (declared before any of their results exist; one change each):
1. `ridge_seqclust` — DNA-defined taxa: PCA(64) + k-means (K = 512) of the DNABERT-S vectors of the
   training-fold sequences (label-free), every sequence → nearest centroid, cluster CLR → Ridge
   (site-grouped alpha). Same as `ridge_clr` except that the composition units come from the DNA
   encoder instead of RDP genus labels (primer-agnostic: new sequences are binned by embedding).
2. `ridge_seqclust_kmer` — the same with 5-mer profiles.
3. `ens_<dna>+ridge_clr` — late fusion: unweighted mean of the out-of-fold predictions of the
   phase-1-selected DNA candidate and `Ridge[genus-CLR]` (the plan's "table branch + sequence branch",
   §5.3/research 03 §d.1, in its simplest form). Reported as a fusion model; if it is the selected
   SENTINEL model, the claim is "DNA-encoded sequence model adds to the CLR table", and it is compared
   with `Ridge[genus-CLR]` itself as well as RF/XGB.

### 10.2 Resumed 2026-11-08 — phase-1 completion, fusion control, improvement cycle 2 (enlarged-training track)

Recorded 2026-11-08 ~12:30 EST (after the two-week pause; host reboots killed nothing that had not finished):
- **Phase-1 complete on development LODO, 8 targets** (registry `microbe_phase1_cv`). Selection rule (§6) over
  SeqSet ×3 + SeqEmb ×2: **`seqset_kmer`** (mean pooled LODO R² 0.239; seqset_id 0.232, seqset_dnabert 0.174,
  rf_seqemb 0.095, ridge_seqemb −0.006; `phase1_selection.json`, registry `microbe_phase1_selection`).
  Strongest baselines: XGB[genus-CLR] 0.289, Ridge[genus-CLR] 0.284, RF[genus-CLR] 0.252.
- Note on wording: the phase-1 SeqSet registry rows carry `--notes` "chem targets only"; they were run after the
  8-target rebuild and score all 8 targets (the note is wrong, the rows are not).
- Cycle-1 item 3 (`ens_seqset_kmer+ridge_clr`, declared §10.1) scored 0.304. **Fusion control added after that
  result (stricter comparator, never a weaker one):** the same late fusion of two *baselines*,
  `ens_ridge_clr+xgb_clr`, scored 0.307 — the DNA fusion gain is an ensembling effect. Any fusion candidate is
  therefore compared with `ens_ridge_clr+xgb_clr` as well as with the H-M3 references. (An exploratory
  3-member `ens_seqset_kmer+seqset_id+ridge_clr`, 0.304, was also scored; counted as a configuration, not a
  candidate.)
- **H-M3 tier-0 reference implemented** (§10 amendment of 2026-10-24 ~23:31): `tier0_best` = per target the best
  development-LODO scorer of GlobalMean / Ridge[DOY+location] / RF[DOY+location]
  (`cv/lodo/tier0_best_choice.json`: Ridge[DOY+location] for TN, DOC, NO3, DO, chl-a; GlobalMean for TP, SC,
  turbidity), fixed for every scheme and the test. `microbe_compare.py --refs tier0_best rf_clr ...
  --pass-refs ...` (the pass criterion names its refs).
- **Saturation check (RESEARCH_PLAN §15.5 items 5–7): not saturated** — best development LODO R² ≈ 0.3 mean,
  strongest baseline far from ceiling. But the comparison is **under-powered**: the paired site-cluster
  bootstrap SE of R² differences on development LODO gives a minimum detectable difference (80% power, Holm
  worst case α = 0.05/8) of 0.10–0.20 R² per target (`microbe_split_lodo_x001_report.json`); the 214-event
  2024 holdout has less power still.

**Improvement cycle 2 — enlarged-training track `lodo_x001` (declared before any model was run on it).**
- Diagnosis: ~1,200 training events; SeqSet early-stops after ~5–20 epochs (data-limited); genus tables cannot
  use the older release well — only 49% of .001 (2017–18) reads fall in genera present in .002 (RDP taxonomy
  version change; 70% of .001 reads have a genus label), whereas sequence encoders (k-mer, DNABERT-S) embed every
  .001 sequence (100% coverage). Literature (2025): set models over DNA-LM sequence embeddings with abundance
  weighting (Yoo & Rosen 2025, arXiv 2508.11075; SetBERT, Bioinformatics 2025 btaf370) — release/primer-agnostic
  set inputs are the property that lets sequence models use heterogeneous data that taxonomy tables cannot.
- **One change:** the training set of every LODO fold additionally contains the DP1.20141.001 development events
  of 2017-01 → 2018-10 outside the held-out domain (517 events, 33 sites, 19 domains: TN 491, TP 502, DOC 490,
  NO3 496, SC 155, turbidity 87, DO 139, chl-a 140 extra rows). **Test rows are exactly the primary LODO test rows**
  (unit-tested; asserted at freeze), so results are paired with the primary track. The pre-2017 .001 reserve
  (C1) and the ≥ 2024 holdout never appear. Community-side descriptors stay enabled for every model (the primer
  one-hot then encodes release/primer; log reads differ ~13× between releases — same inputs for all models).
- Frozen fold record `docs/preregistration/microbe_split_lodo_x001.json`, sha256
  **`a61416982db82cd10323e2e565d00fda544fdb9c6bd339194f02486556e58caa`**; power check in
  `microbe_split_lodo_x001_report.json` (same test rows ⇒ same power as primary LODO).
- Models (all identical rows/folds; baselines get the enlarged data too — never weakened): GlobalMean,
  Ridge/RF[DOY+location], Ridge/RF/XGB[genus-CLR], SeqSet kmer/id/dnabert, and the cycle-1 SeqClust models.
  Hypothesis (stated before results): DNA-encoded models gain more from the added rows than genus tables.
- Selection on this track uses the §6 rule; the H-M3 comparators are the **stronger of the primary-track and
  enlarged-track versions** of each reference (per target), and `ens_ridge_clr+xgb_clr` for fusion candidates.
  Claims name the track. Phase-2 implication: a model selected on this track is refit for the test on .002
  development events + .001 2017–18 events outside g (still no reserve or holdout rows).

### 10.3 Improvement cycle 3 — primer-harmonised V4 units (declared 2026-11-08 ~15:45 EST, before any V4 result)

- **Cycle-1 outcome** (registry `microbe_phase1_cv`, iteration `cycle1`): SeqClust-Ridge[DNABERT-S] 0.256 mean
  LODO R² (DOC 0.352, best of all models on DOC), SeqClust-Ridge[kmer5] 0.245 — both below Ridge[genus-CLR]
  0.284. Updated §6 selection over the DNA-encoded set: `ridge_seqclust` (0.256) > `seqset_kmer` (0.239).
- **Cycle-2 outcome** (`lodo_x001`, enlarged training; hypothesis **rejected** for the DNA models): mean R²
  primary → enlarged: XGB[genus-CLR] 0.289 → 0.301, RF[genus-CLR] 0.252 → 0.262, Ridge[genus-CLR] 0.284 → 0.283,
  SeqClust-DNABERT 0.256 → 0.205, SeqClust-kmer 0.245 → 0.163 (SeqSet on this track still running).
- **Diagnosis (measured, fold domain=D01 training rows):** frozen encoders separate the two amplicon regions
  instead of aligning organisms. Read-mass overlap between releases of the 512 k-means units
  (Σ_u min(p_.001(u), p_.002(u))) is 0.04 for DNABERT-S and 0.00 for 5-mers, vs 0.27 for RDP genus labels — so the
  "primer-agnostic sequence embedding" premise (§4.1, RESEARCH_PLAN §5.3) does not hold for whole-amplicon
  embeddings across V3V4/V4V5.
- **Literature:** combining 16S data from different primer regions requires putting reads on a common
  coordinate system — fragment insertion / phylogenetic placement (SEPP, Janssen et al. 2018 mSystems) or
  in-silico trimming to the shared region; set models over DNA-LM embeddings (Yoo & Rosen 2025; SetBERT 2025)
  assume a single region.
- **One change: primer-harmonised exact V4 units.** `sentinel.models.microbe.dna.v4_window`: V4V5 (.002) reads
  start right after 515F (verified: 806R site at median position 253; no 515F site inside), V3V4 (.001) reads
  contain the 515F site (3′ 13-bp core `GC[AC]GCCGCGGT[AG]A` found in 73.6% of .001 sequences = 91.9% of .001
  reads, median end 166; 241 bp follow it). Both are cut to the same 230-bp window (E. coli ≈ 534–763); unit =
  sha1(window). Row-wise and label-free (sequence + its primer set only). Measured before modelling: cross-release
  read-mass overlap of exact V4 units **0.337** (vs genus 0.271); .001 read mass in shared units 0.67, .002 0.86.
- Candidates (each is its parent with only the composition unit changed; CLR / prevalence / pooling / descriptors
  / hyperparameters identical): `ridge_v4clr` (parent `ridge_clr`), `xgb_v4clr` (parent `xgb_clr`). DNA-encoded
  SENTINEL candidates under §6 (sequence-defined, release/primer-harmonised units; no taxonomy labels).
- Runs: `lodo` (primary), `lodo_x001` (enlarged), and the cross-release transfer route `release_002_to_001`
  (alternative primary, §1; with all its baselines, descriptors disabled for every model) and the reverse.
  Comparators unchanged (tier0_best, RF[genus-CLR], stronger of tuned/untuned RF/XGB, and for the transfer route
  the same baselines re-run on that scheme).
- Planned next if V4 units help (declared now, run only after the above): V4-window k-mer/DNABERT-S tables for the
  SeqSet (needs a new frozen embedding of ~150k V4 windows), and EMP 515F/806R V4 data for unlabeled pretraining
  (EMP's V4 region is the same window — the harmonisation is what makes EMP usable).
- 2026-11-08 ~19:35 EST, after the four V4 LODO runs (mean dev R²: V4-XGB 0.314 / 0.305 on lodo / lodo_x001;
  V4-Ridge 0.291 / 0.301; parents XGB 0.289 / 0.301, Ridge 0.284 / 0.283; TabPFN[genus-CLR] 0.291; SeqSet-kmer on
  lodo_x001 0.201): added, before it is scored, the fusion analogue `ens_ridge_v4clr+xgb_v4clr` (one change vs the
  baseline fusion `ens_ridge_clr+xgb_clr`: the unit). Fusion candidates are compared with
  `ens_ridge_clr+xgb_clr`; single models with the single-model references.

### 10.4 Improvement cycle 4 — community added to season + location (declared 2026-11-08 ~19:50 EST, before results)

- **Cycle-3 outcome (dev LODO, preview B=1000, not yet logged as comparisons):** V4-XGB beats RF[genus-CLR] and
  XGB[genus-CLR] on DOC and SC and the tier-0 reference on TP and SC (positive, Holm-significant with both site
  and domain units); the H-M3 dev analogue (all of tier0_best, RF, XGB) passes only for SC. Closest miss: DOC vs
  tier-0 (+0.12 R², p = 0.054 site / 0.062 domain). Tier-0 season+location is strong for DOC, NO3 and DO.
- **Diagnosis:** the community-only candidates must re-learn season/location from composition; the question
  H-M3 actually asks — does the community add skill beyond season + location in an unseen domain — is answered
  more directly by a model that has both.
- **Changes (one each; comparators made stricter, not weaker):**
  1. `xgb_v4clr_context` = `xgb_v4clr` + the tier-0 context features (DOY harmonics, lat, lon, elevation, site
     type). Parity baselines added: `xgb_clr_context` (XGB[genus-CLR+DOY+location]) and the existing
     `rf_clr_context`. For this candidate the H-M3 dev analogue requires beating tier0_best, RF[genus-CLR],
     XGB[genus-CLR] **and** the stronger +context genus baseline.
  2. Late fusion with the tier-0 model: `ens_ridge_v4clr+xgb_v4clr+context_ridge`, compared with the baseline
     fusion `ens_ridge_clr+xgb_clr+context_ridge` (and all single references).
- Claims from these candidates are worded "community (V4 units) adds skill beyond season + location".

### 10.5 Secondary time-out test block "Test-B" (2025 events) — pre-registered 2026-11-08 ~20:00 EST

Approved by the lead 2026-11-08. Pre-registered before any 2025 community or chemistry data exist locally
(the RELEASE-2026 build ends 2024-11-12; nothing after it has been downloaded or inspected by M3).
- **Role: secondary.** The primary endpoint (§1, §8) remains the 2024 holdout (214 events) together with the
  cross-release reserve (16-hypothesis Holm, §10 C1). Test-B never changes the primary decision.
- **Definition** (`configs/microbe/common.yaml: test_b`): DP1.20141.002 community events with collect date
  2024-01-01 → 2025-12-31 that are non-provisional in NEON RELEASE-2027 (or the first later release in which they
  are), and whose event id is **not** in `microbe_split.json` (neither development nor primary holdout). Built with
  the same builder, join rules (§2.2), read threshold and target definitions; if NEON changes the OTU pipeline or
  taxonomy version in RELEASE-2027, models consume sequences (V4 window) / genus labels exactly as frozen, and the
  change is reported. Rows of the primary holdout are not re-derived from the new release.
- **Fitting:** the frozen configuration's per-domain fits used for the primary test (development events outside
  domain g, plus .001 2017–18 rows if the frozen model is from the `lodo_x001` track) are saved with their sha256 at
  primary scoring and re-used for Test-B; if lost, a deterministic refit (same code, config, seeds, threads) is used
  after reproducing the primary-test predictions to max |Δ| < 1e-6. No refit on 2024 data. All comparators
  (tier0_best, RF/XGB[genus-CLR] tuned/untuned as frozen, baseline fusion where applicable) are treated identically.
- **Metrics:** as §5/§8 (pooled R², per-domain, anomaly R², paired site- and domain-cluster bootstrap B = 2,000).
  Holm across the 8 targets within Test-B (a separate, secondary family).
- **Pre-declared pooled analysis (secondary):** primary holdout + Test-B events pooled (one prediction per event,
  same fitted models); paired site- and domain-cluster bootstrap with clusters spanning both blocks; Holm over 8
  targets; reported next to — never instead of — the primary-holdout result.
- **Power check before scoring:** per-target event counts and the minimum detectable ΔR² (method of
  `microbe_split_lodo_x001_report.json`) are written to `docs/preregistration/microbe_test_b_report.json` with the
  Test-B id list sha256 before any prediction is made. Scored once.
- 2026-11-08 ~20:15 EST — **EMP freshwater added as unlabeled data** (lead-approved, ≤ 10 GB): `processed/emp_freshwater`
  (manifest + `docs/datasheets/emp_freshwater.md`; EMP Release 1 Deblur 150 bp, empo_3 = Water (non-saline): 4,723
  samples, 7 studies, 74,901 sequences; 0.19 GB raw). Alignment check (no labels involved): 81.8% of .002 and 73.3% of
  .001 read mass have their V4-window first 150 bp exactly present in EMP freshwater. No pretraining cycle is started
  before the permutation nulls finish (lead decision 2026-11-08).

### 10.6 Cycle-4 outcome and improvement cycle 5 (declared 2026-11-09 ~01:30 EST; cycle 5 runs only after the permutation nulls)

- **Cycle 4 (dev LODO, preview B = 1000):**
  - V4-XGB+context: mean R² 0.317 (DOC 0.398, DO 0.457). The parity baseline XGB[genus-CLR+context] scores 0.286.
  - DOC is positive and Holm-significant with both resampling units vs tier0_best, RF[genus-CLR] and RF/XGB+context.
    Vs XGB[genus-CLR] it is p = 0.012 / 0.042 (not Holm-significant).
  - SC passes vs tier-0, RF and XGB+context, but not vs XGB (p = 0.128).
  - The strict dev analogue therefore still passes for 0 targets for this candidate, and for 1 target (SC) for
    V4-XGB.
  - The unweighted fusion with Ridge[DOY+location] is negative: 0.303, and it hurts SC.
- **Cross-release transfer `release_002_to_001` (alternative primary route; descriptors off):**
  - Every community model has strongly negative R² on TP, TN and DOC (TP −0.88 to −1.49). This is consistent
    with a level shift between 2017–18 and 2018–23 (lab / pipeline era) that no input can see.
  - SC transfers best with V4-XGB (0.479, vs genus XGB 0.142 and RF 0.223). Paired CI vs tier-0 excludes 0; vs
    XGB the site-level CI is [+0.02, +0.74].
  - The transfer route is reported as failing for the nutrients.
- **Cycle 5 (one change):** `seqset_kmer_v4` = `seqset_kmer` with each token's frozen vector computed from the
  harmonised V4 window. Table `interim/microbe_seqemb/kmer5_v4.npy` (manifest `microbe_seqemb_kmer5_v4`; 81% of
  250,088 unique sequences have a window; zero vector otherwise).
  - Runs: `lodo` and `lodo_x001`, on GPU 0/1 in ≤ 2 h chunks, only when no test-critical job is waiting.
  - Hypothesis: unlike `seqset_kmer` (0.239 → 0.201), it gains rather than loses from the .001 rows.
- **Cycle 6 (planned, design to be declared before running):** label-free pretraining of the SeqSet encoder on EMP
  freshwater + NEON development communities (V4-window tokens), then fine-tuning under LODO. The pretraining data
  never include holdout events.
