// SENTINEL — Model catalog (5 encoders + downstream / deployment models).
//
// Data contract for the Models page. Every headline metric is REAL and
// held-out, cross-checked against /home/bcheng/SENTINEL/README.md and
// results/*.json. Retracted/fabricated figures are NOT reintroduced; where a
// number was corrected during integrity remediation, the old value is recorded
// in `corrects` so the page can show the honesty explicitly.
//
// Verified sources:
//   AquaSSM        results/benchmarks/aquassm_holdout.json (RMSE 0.8296 -> 0.83)
//                  results/trackA_leadtime/aquassm_specificity/expanded_matched_fpr.json
//                  (verdict.headline_5pct: cusum_det 11 / n_episodes 15, cusum_lead 36.1 d,
//                   cusum_fpr 0.0497; baseline_det 5, baseline_lead 25.51 d)
//                  results/trackA_leadtime/loso_generalization.json (leave-one-site-out:
//                   global tau 2.79±0.02σ, 9/12 same episodes, 29.2 d, clean-ctrl FPR 4.1%)
//                  results/trackA_leadtime/aquassm_specificity/loco_cusum_generalization.json
//                   (leave-one-control-out: same 11/15 episodes on every fit)
//                  results/trackA_leadtime/aquassm_specificity/candidates/csd-ews-channel.json
//                   (critical-slowing-down EWS channel: 11/15 @ median lead 41.3 d,
//                    ctrl held-out FPR 0.047; temporal split 41.6 d; robustness probe
//                    csd_ews_robustness_probe.txt — win holds across the hyperparam neighborhood)
//                  results/trackA_leadtime/usgs/cross_network_csd_ews.json (cross-network zero-shot:
//                   frozen NEON encoder on USGS NWIS — self-cal 14/17 @ 41.4 d @ 4.8% FPR;
//                   NEON operating point transfers 13/17 @ 38.1 d @ 2.8% FPR)
//   HydroViT       results/benchmarks/hydrovit_real.json (water-temp R² 0.7286)
//                  results/benchmarks/hydrovit_rededge.json (chl-a R² 0.32)
//   MicroBiomeNet  results/benchmarks/microbiomenet_domaingen*.json
//                  (raw-OTU 0.405, taxonomic 0.676, leakage 0.831)
//   ToxiGene       results/benchmarks/toxigene_realgeo.json
//                  (within-study AUROC 0.86, LOSO ~0.55 raw / 0.64 rank-norm)
//   BioMotion      results/benchmarks/biomotion_v2_holdout.json (AUROC 0.85)
//   Species Health results/benchmarks/species_mechanism_covered.json (same 5,411-covered-sites subset)
//                  (S1_streamcat.mussel_auroc 0.818, S3_geo_only.mussel_auroc 0.791,
//                   mussel S1−S0 delta 0.074 CI [0.016,0.139], S1 macro 0.889, S3 macro 0.898)
//                  NOTE: NOT species_mechanism.json (full 592-test-site subset), whose
//                  near-duplicate values differ (mussel 0.820/0.803, macro 0.883/0.900).

export type ModelStatus = 'validated' | 'region-limited' | 'simulation' | 'ceiling' | 'circular';

export type ModelKind = 'encoder' | 'fusion' | 'downstream' | 'deployment';

export interface CorrectedFigure {
  /** The retracted / superseded number, shown struck-through for transparency. */
  oldValue: string;
  /** Why it was wrong (leakage, synthetic data, circular label, invalid data). */
  reason: string;
}

export interface ModelEntry {
  id: string;
  /** Display name, e.g. "AquaSSM". */
  name: string;
  kind: ModelKind;
  /** Modality or role, e.g. "Sensor encoder". */
  role: string;
  /** One-line task description. */
  task: string;
  /** Human label for the headline metric, e.g. "Reconstruction RMSE". */
  metricLabel: string;
  /** The headline metric value as a display string, e.g. "0.83". */
  metricValue: string;
  /** Glossary term key for the metric (links to GLOSSARY_BY_TERM). */
  metricGlossary?: string;
  /** Holdout protocol used for the headline metric (display string). */
  holdout: string;
  /** Glossary term key for the holdout protocol. */
  holdoutGlossary?: string;
  /** Real, public data source(s). */
  dataSource: string;
  status: ModelStatus;
  /** One-line, plain-English honest note — the caveat or the nuance. */
  honestNote: string;
  /** Secondary verified metrics for a detail card (label + value). */
  secondaryMetrics?: { label: string; value: string; glossary?: string }[];
  /** A number that was corrected during integrity remediation, if any. */
  corrects?: CorrectedFigure;
  /** Architecture one-liner, for the detail view. */
  architecture?: string;
}

export const MODELS: ModelEntry[] = [
  // ============================ ENCODERS ==============================
  {
    id: 'aquassm',
    name: 'AquaSSM',
    kind: 'encoder',
    role: 'Sensor encoder',
    task: 'Sensor-stream reconstruction + label-free pre-onset anomaly detection',
    metricLabel: 'Reconstruction RMSE',
    metricValue: '0.83',
    metricGlossary: 'rmse',
    holdout: 'Station holdout (62 unseen sites)',
    holdoutGlossary: 'station_holdout',
    dataSource: 'USGS NWIS (127K sequences) + NEON continuous sonde',
    status: 'validated',
    honestNote:
      'The label-free detector flags real contamination episodes before onset with a controlled false-alarm rate, and — critically — this DETECTION generalizes broadly. On dissolved-oxygen hypoxia the rolling-CUSUM residual detector fires a sustained pre-onset alarm on 25 of 32 real episodes across two independent sensor networks (NEON 11/15, USGS 14/17) at a matched ≤5% held-out control false-alarm rate, versus 5/15 for a point-threshold baseline on NEON — a genuine label-free sensitivity gain, independently reproduced. HONEST SCOPE (from a rigorous self-audit): we do NOT claim a fixed "N-day lead time". A lead-versus-lookback analysis shows the measured median "lead" tracks the chosen search window (≈0.9× the lookback, correlation 0.997), and for most episodes the reconstruction residual is chronically elevated before onset rather than building toward it (only ~30% show a growing precursor) — so headlining a specific day-count would overstate the result. Tested directly against the official detection/advisory dates of 7 real historical events with co-located sensors (Toledo 2014, the 2023/2024 Mississippi saltwater intrusions, Delaware 2024, Lake Okeechobee 2016, and others), no verifiable lead emerged — these sites are chronically variable and the apparent leads were window-sensitive artifacts — so we make no early-warning-horizon claim. What IS robust is DISCRIMINATION of contamination-episode conditions from clean baselines at a matched ≤5% false-alarm rate, and its generalization: the operating point transfers across SITES (leave-one-site / leave-one-control-out — the same episodes on unseen sites, a near-constant 2.79σ threshold), across sensor NETWORKS (zero-shot NEON→USGS, threshold transfers at ≤5% FPR — 13/17 USGS), and across anomaly CLASSES — applied unchanged to 30 USGS urban road-salt sites it flags 26/30 salinization (high-conductance) excursions pre-onset, driven by the conductance residual channel alone (DO-only 14/30), with winter-only onset timing independently implicating road salt. So AquaSSM is a general, self-normalizing anomaly detector needing only an unlabeled clean baseline — strong and honest on detection and cross-site/network/class transfer, deliberately scoped away from a calibrated lead-time forecast.',
    secondaryMetrics: [
      { label: 'Pre-onset detection (NEON+USGS, ≤5% FPR)', value: '25 / 32 episodes' },
      { label: 'NEON hypoxia (pre-onset)', value: '11 / 15' },
      { label: 'Point-threshold baseline (same FPR)', value: '5 / 15' },
      { label: 'Cross-network (USGS, zero-shot, self-cal)', value: '14 / 17 pre-onset @ 4.8% FPR' },
      { label: 'Cross-network (NEON threshold transferred)', value: '13 / 17 @ 2.8% FPR' },
      { label: 'Second anomaly class (salinization, USGS)', value: '26 / 30 pre-onset @ 4.8% FPR' },
      { label: 'Matched control FPR (out-of-sample)', value: '≤ 5%', glossary: 'fpr' },
      { label: 'Threshold transfer (site + network)', value: 'same episodes, unseen sites/networks' },
      { label: 'Lead horizon', value: 'operating-window (45 d), not a fixed forecast' },
    ],
    architecture:
      'Mamba-based continuous-time state-space model on irregular multivariate streams; masked-parameter self-supervised pretraining, self-supervised NEON fine-tuning; rolling-CUSUM residual detector (with a critical-slowing-down variance/autocorrelation channel) for label-free pre-onset anomaly detection that transfers across sites, networks, and anomaly classes.',
  },
  {
    id: 'hydrovit',
    name: 'HydroViT',
    kind: 'encoder',
    role: 'Satellite encoder',
    task: 'Water-quality parameter prediction from Sentinel-2 imagery',
    metricLabel: 'Water-temp R²',
    metricValue: '0.729',
    metricGlossary: 'r2',
    holdout: 'Station holdout (real S2 × in-situ pairs)',
    holdoutGlossary: 'station_holdout',
    dataSource: 'ESA Sentinel-2 L2A (Planetary Computer) + in-situ WQP',
    status: 'validated',
    honestNote:
      'Chl-a is harder: R² 0.32 with red-edge bands B05–B07 — up from 0.18 on 4-band (CI excludes 0), a transferable-representation win. Replaces the retracted fabricated 0.893.',
    secondaryMetrics: [
      { label: 'Dissolved-oxygen R²', value: '0.58', glossary: 'r2' },
      { label: 'Specific-conductance R²', value: '0.59', glossary: 'r2' },
      { label: 'Chl-a R² (red-edge)', value: '0.32', glossary: 'r2' },
      { label: 'Chl-a R² (4-band)', value: '0.18', glossary: 'r2' },
    ],
    corrects: {
      oldValue: '0.893',
      reason: 'Fabricated fallback value; rebuilt on real Sentinel-2 × in-situ pairs under station holdout.',
    },
    architecture:
      'ViT-S/16 CNN-ViT hybrid, masked-autoencoder pretraining on 10-band Sentinel-2 L2A; red-edge indices (NDCI, CIre, MCIre, FAI) add the chl-a lift.',
  },
  {
    id: 'microbiomenet',
    name: 'MicroBiomeNet',
    kind: 'encoder',
    role: 'Microbial encoder',
    task: '8-class aquatic source attribution from 16S rRNA communities',
    metricLabel: 'Macro-F1',
    metricValue: '0.676',
    metricGlossary: 'macro_f1',
    holdout: 'True geographic holdout (full EMP release)',
    holdoutGlossary: 'geographic_holdout',
    dataSource: 'Earth Microbiome Project 16S rRNA (full release, ~19K samples)',
    status: 'validated',
    honestNote:
      'Transferable family/genus taxonomic features hit 0.676 under true geographic holdout, versus 0.405 on raw region-specific OTUs — higher taxa generalize across continents. The old 0.831 was random-split leakage.',
    secondaryMetrics: [
      { label: 'Raw-OTU baseline (geo holdout)', value: '0.405', glossary: 'macro_f1' },
      { label: 'Taxonomic features (geo holdout)', value: '0.676', glossary: 'macro_f1' },
      { label: 'Multi-seed mean (taxonomic)', value: '0.662 ± 0.018' },
    ],
    corrects: {
      oldValue: '0.831',
      reason: 'Per-sample random split leaked correlated same-region samples across train/test.',
    },
    architecture:
      'Aitchison-geometry transformer (CLR attention, simplex neural ODE); the domain-generalization result aggregates OTUs to phylum→genus CLR features — a transferable representation.',
  },
  {
    id: 'toxigene',
    name: 'ToxiGene',
    kind: 'encoder',
    role: 'Molecular encoder',
    task: 'Toxicant-exposure classification from zebrafish RNA-seq',
    metricLabel: 'Within-study AUROC',
    metricValue: '0.86',
    metricGlossary: 'auroc',
    holdout: 'Within-study CV / cross-study LOSO',
    holdoutGlossary: 'loso',
    dataSource: 'NCBI GEO — 12 real zebrafish (Danio rerio) series, real metadata labels',
    status: 'ceiling',
    honestNote:
      'Within a study, AUROC 0.86. Across studies (LOSO) it is only ~0.64 with per-sample rank-normalization (0.55 raw) — exposure signatures largely do not transfer to unseen experiments. An honest ceiling.',
    secondaryMetrics: [
      { label: 'Within-study AUROC', value: '0.86', glossary: 'auroc' },
      { label: 'Cross-study LOSO (rank-norm)', value: '0.64', glossary: 'auroc' },
      { label: 'Cross-study LOSO (raw)', value: '0.55', glossary: 'auroc' },
    ],
    corrects: {
      oldValue: '0.492',
      reason: 'Computed on invalid human/cattle disease GEO with hard-coded labels and placeholder gene names — not aquatic toxicology.',
    },
    architecture:
      'Reactome-constrained gene→pathway→process→outcome hierarchy; per-sample rank-normalization acts as batch correction for the modest cross-study transfer.',
  },
  {
    id: 'biomotion',
    name: 'BioMotion',
    kind: 'encoder',
    role: 'Behavioral encoder',
    task: 'Ecotoxicological effect classification from ECOTOX assays',
    metricLabel: 'AUROC',
    metricValue: '0.85',
    metricGlossary: 'auroc',
    holdout: 'Chemical-grouped split (novel chemicals)',
    holdoutGlossary: 'station_grouped_cv',
    dataSource: 'EPA ECOTOX — 29,421 real Daphnia dose-response assays',
    status: 'validated',
    honestNote:
      'A tabular GBM (AUROC 0.85) beats the 3M-param diffusion encoder (0.807), but a self-audit shows most of the signal is experimental-DESIGN metadata, not behavioral biology: design-only features reach 0.83, biology-only only 0.75, and a regularized logistic baseline reaches 0.82 (the GBM edge is ~0.04). It transfers across novel CHEMICAL classes (leave-one-class-out 0.87 full / 0.80 biology-only). Across SPECIES it degrades but does not collapse: leave-one-taxon-out AUROC is 0.68 on biology-only features (above the 0.5 chance line but weak — 0.66 is the anomaly prevalence, not a chance floor) and 0.78–0.81 with full features. Reported honestly; the transferable biological signal is weak.',
    secondaryMetrics: [
      { label: 'Cross-chemical-class (full)', value: '0.87', glossary: 'auroc' },
      { label: 'Cross-chemical-class (biology-only)', value: '0.80', glossary: 'auroc' },
      { label: 'Cross-species (biology-only)', value: '0.68 ≈ base rate', glossary: 'auroc' },
      { label: 'Diffusion-encoder baseline', value: '0.807', glossary: 'auroc' },
    ],
    architecture:
      'GradientBoosting on real ECOTOX endpoints (log-concentration spread, exposure/observation duration, effect-group one-hot); label-leaking endpoint fields are excluded.',
  },

  // ============================ FUSION ================================
  {
    id: 'fusion',
    name: 'Perceiver IO Fusion',
    kind: 'fusion',
    role: 'Multimodal fusion',
    task: 'Microcystin (optically-invisible cyanotoxin) detection from 3 modalities',
    metricLabel: '3-way fusion AUROC',
    metricValue: '0.82',
    metricGlossary: 'auroc',
    holdout: 'Station-grouped CV (within region)',
    holdoutGlossary: 'station_grouped_cv',
    dataSource: 'Sentinel-2 optics + in-situ pigment + WQP physicochemical drivers',
    status: 'region-limited',
    honestNote:
      'On microcystin, 3-modality fusion beats every single modality (0.82 vs 0.68) within a calibrated region — each modality adds, CI-significant, permutation-confirmed. But under cross-region holdout it collapses to chance (~0.49) because the toxin depends on the unobserved mcyD toxic-strain genotype. On chl-a/DO, fusion only ties the best single.',
    secondaryMetrics: [
      { label: 'Best single modality', value: '0.68', glossary: 'auroc' },
      { label: '3-way fusion (within region)', value: '0.82', glossary: 'auroc' },
      { label: 'Cross-region holdout', value: '~0.49 (chance)', glossary: 'cross_region_holdout' },
    ],
    architecture:
      'Perceiver IO cross-attention over 256 learned latents with confidence gating and per-modality temporal decay; see the Fusion page for the full per-threshold breakdown.',
  },

  // ========================== DOWNSTREAM =============================
  {
    id: 'species_health',
    name: 'Species Health Index',
    kind: 'downstream',
    role: 'Bioindicator occupancy',
    task: 'Keystone-species occupancy from environment',
    metricLabel: 'Macro-AUROC',
    metricValue: '0.949',
    metricGlossary: 'auroc',
    holdout: 'Leakage-free retrain (per-species)',
    holdoutGlossary: 'geographic_holdout',
    dataSource: 'USGS BioData (5,462 real sites); 4 species after dropping 2 always-absent',
    status: 'ceiling',
    honestNote:
      'Retrained leakage-free after the circular AUROC 1.000 (the input held the per-species counts that define the label). Honest macro-AUROC 0.949. With full COMID coverage (5,411 sites), real EPA StreamCat watershed environment is a genuine occupancy predictor: env-only significantly beats the crude proxy floor (mussel +0.074, CI [0.016, 0.139]) and matches pure geography (env-only macro 0.889 vs geography 0.898; mussel 0.818 vs 0.791) — confirming true environmental mechanism, not just a geographic range prior. But env and geography are largely redundant, so environment adds little on top of location (env−geo not significant).',
    secondaryMetrics: [
      { label: 'Env-only mussel (real StreamCat)', value: '0.818', glossary: 'auroc' },
      { label: 'Geography-only mussel', value: '0.791', glossary: 'auroc' },
      { label: 'Env-only vs proxy floor', value: '+0.074 (CI excl. 0)', glossary: 'auroc' },
      { label: 'Env-only macro-AUROC', value: '0.889', glossary: 'auroc' },
    ],
    corrects: {
      oldValue: '1.000',
      reason: 'Circular — the model input held the per-species counts that define the occupancy label.',
    },
  },
  {
    id: 'stream_gnn',
    name: 'Stream Network GNN',
    kind: 'downstream',
    role: 'Contamination propagation',
    task: 'Upstream–downstream propagation over river topology',
    metricLabel: 'Status',
    metricValue: 'Simulation only',
    holdout: 'Synthetic propagation labels',
    holdoutGlossary: 'simulation_only',
    dataSource: 'Real NHDPlus V2 topology (561 sites, 338 edges); synthetic labels',
    status: 'simulation',
    honestNote:
      'Real river topology, but trained and evaluated on synthetic propagation labels — a proof-of-concept for the architecture, NOT validated on real contamination events.',
  },
  {
    id: 'disease_forecast',
    name: 'Beach Nowcast (E. coli)',
    kind: 'downstream',
    role: 'Recreational-water risk',
    task: 'E. coli exceedance of the EPA Beach Action Value (235 CFU/100mL) from antecedent weather',
    metricLabel: 'Within-region AUROC',
    metricValue: '0.66–0.73',
    metricGlossary: 'auroc',
    holdout: 'Leave-sites-out (within region); cross-region tested',
    holdoutGlossary: 'geographic_holdout',
    dataSource: 'WQP E. coli — OH/MI/WI (~95K samples, 807 sites) + NOAA GHCN antecedent rainfall',
    status: 'region-limited',
    honestNote:
      'Replaces a retracted circular index (labels were a closed-form rule of the sensor inputs → fake 0.988). The real, non-circular task predicts E. coli exceedance from antecedent rainfall + season on independent channels. It has genuine LOCAL skill — leave-sites-out AUROC 0.63–0.69 across three real regions (MI 0.63 / OH 0.67 / WI 0.69). Cross-region transfer was ~0.55 (worse than persistence) with rainfall+season alone, but two region-invariant, fully-antecedent features close much of the gap: a coarse site-type (beach / inland lake / stream — genuinely transferable hydrology, streams exceed beaches in every region) and a per-region rank-normalized rainfall transform (label-free covariate-shift correction). These lift leave-one-region-out AUROC to 0.62 — now matching the persistence baseline (0.62), still below the 0.71 within-region ceiling. So cross-region is recoverable to persistence level with region-invariant features, though within-region calibration remains stronger.',
    secondaryMetrics: [
      { label: 'Within-region leave-sites-out (MI/OH/WI)', value: '0.63–0.69', glossary: 'auroc' },
      { label: 'Cross-region (rainfall+season only)', value: '~0.55 (< persistence)', glossary: 'cross_region_holdout' },
      { label: 'Cross-region (+ site-type + rank-rainfall)', value: '0.62 (= persistence)', glossary: 'cross_region_holdout' },
      { label: 'Transfer gap', value: '0.11 AUROC' },
      { label: 'Base rate (OH/MI/WI)', value: '20 / 7 / 14%' },
    ],
    corrects: {
      oldValue: '0.988',
      reason: 'Circular — pathogen labels were a closed-form function of the sensor inputs; replaced with a real, non-circular E. coli exceedance nowcast on independent weather predictors.',
    },
    architecture:
      'Gradient-boosted trees on NOAA GHCN prior-day rainfall (1/2/3/7-day) + day-of-year; label is an independent WQP E. coli culture result. Non-circular by construction (no FIB value is a predictor).',
  },
  {
    id: 'bloom_ew',
    name: 'Bloom Early-Warning',
    kind: 'downstream',
    role: 'Algal-bloom onset forecast',
    task: 'Forecast NEW cyanobacterial-bloom onset (chl-a exceedance) at the next in-situ visit',
    metricLabel: 'Onset AUROC (fusion)',
    metricValue: '0.733',
    metricGlossary: 'auroc',
    holdout: 'GroupKFold-by-site (within-region), antecedent-only',
    holdoutGlossary: 'station_grouped_cv',
    dataSource: 'Sentinel-2 optics + optical trajectory + prior in-situ chl-a + drivers + season',
    status: 'validated',
    honestNote:
      'A second-crisis forecast beyond hypoxia: given only strictly-antecedent signals (t0 Sentinel-2 optics + optical trajectory, prior chl-a + trend, drivers, season), it predicts a NEW WHO Alert-2 bloom (chl-a > 24 µg/L) at the next in-situ visit — median realized lead ~14 days, set by the WQP visit cadence, not the 5-day Sentinel-2 revisit. On the strict onset-only test (940 onset-eligible pairs, 207 true new-bloom onsets, where persistence-of-state is uninformative), fusion AUROC is 0.733 (site-clustered bootstrap 95% CI [0.691, 0.777], excluding the 0.5 no-skill line); at a ~14% false-alarm rate it recovers ~45% of new blooms. HONEST CAVEAT: on this onset test, fusion does NOT statistically beat the best single modality (delta +0.019, CI [-0.007, 0.046] includes 0). The forecast is single-modality-dominated by severity — antecedent biomass trajectory dominates at WHO Alert-2 (persistence-model 0.714), satellite optics dominates at the eutrophic chl>8 threshold (0.708) — and on the easy all-pairs task fusion (0.83) merely ties a trivial persistence baseline (0.822). This is a genuine early-warning capability for a second crisis, NOT a "fusion beats single modalities" claim.',
    secondaryMetrics: [
      { label: 'Onset AUROC (fusion, new blooms)', value: '0.733', glossary: 'auroc' },
      { label: 'Onset AUROC 95% CI', value: '[0.691, 0.777]', glossary: 'auroc' },
      { label: 'Best single (persistence-model)', value: '0.714', glossary: 'auroc' },
      { label: 'Fusion − best single (onset)', value: '+0.019 (CI incl. 0)', glossary: 'auroc' },
      { label: 'Onset-eligible pairs / true onsets', value: '940 / 207' },
      { label: 'Median realized lead', value: '~14 days', glossary: 'lead_time' },
    ],
    architecture:
      'Leakage-free within-region forecast (GroupKFold-by-site) over three antecedent blocks — Sentinel-2 optics + optical trajectory, prior in-situ chl-a + trend, and physicochemical drivers + season. Target is chl-a exceedance at the next in-situ visit; all predictors are strictly antecedent (no t1 optics/chl/drivers).',
  },
  {
    id: 'digital_twin',
    name: 'Digital Twin',
    kind: 'downstream',
    role: 'Ecosystem forecasting',
    task: 'Multi-horizon ecosystem state forecasting',
    metricLabel: '1-day R²',
    metricValue: '0.688',
    metricGlossary: 'r2',
    holdout: 'Temporal holdout vs persistence',
    holdoutGlossary: 'persistence_baseline',
    dataSource: '10 state variables, 6 horizons; real USGS',
    status: 'ceiling',
    honestNote:
      'The neural-ODE twin loses to persistence for absolute state at every horizon (overall R² 0.07 vs 0.92), and on reformulated targets the neural twin is at chance (exceedance AUROC 0.52). The one honest, modest positive is that reformulating the TARGET to threshold-EXCEEDANCE makes it seasonally predictable: a leak-free day-of-year climatology forecasts 90-day temperature>25 °C exceedance at AUROC ≈ 0.83, roughly flat across horizons. We do NOT headline the "0.93 vs 0.74 persistence-margin" comparison — persistence necessarily decays on a seasonal signal, so climatology is the correct null, and a learned combiner adds only ≈ +0.08 over climatology (no CI for that increment). We use the twin for counterfactual what-if simulation, not as a forecasting-accuracy claim.',
    secondaryMetrics: [
      { label: 'Neural twin overall R²', value: '0.07', glossary: 'r2' },
      { label: 'Persistence R²', value: '0.92', glossary: 'r2' },
      { label: 'Neural twin, 90-day exceedance', value: '0.52 (chance)', glossary: 'auroc' },
      { label: '90-day temp-exceedance, climatology (the skill)', value: '≈0.83', glossary: 'auroc' },
      { label: 'Conformal coverage (within-distribution)', value: '90%' },
    ],
  },

  // ========================== DEPLOYMENT ============================
  {
    id: 'sentinel_lite',
    name: 'SENTINEL-Lite (HydroDenseNet)',
    kind: 'deployment',
    role: 'Imagery-only screening',
    task: 'Water-quality screening from 4-band (RGB+NIR) imagery + a field pH strip',
    metricLabel: 'Temperature R²',
    metricValue: '0.74',
    metricGlossary: 'r2',
    holdout: 'Station holdout (unseen stations), site-clustered bootstrap CI',
    holdoutGlossary: 'spatial_holdout',
    dataSource: 'Real Sentinel-2 tiles × in-situ WQ; 372 stations',
    status: 'validated',
    honestNote:
      'Honest, audited scope (2026-07): the screen genuinely retrieves temperature (R² 0.74) and dissolved oxygen (R² 0.50 imagery-only; 0.59 with a field pH-strip input) — both show real within-station skill, so they are true optical retrievals. Turbidity is recoverable only on the log scale (log-R² 0.26; raw-NTU R²≈0 is a heavy-tailed-metric artifact). Two honest corrections to an earlier version: it uses an in-situ pH-strip reading as a network input (so it is not purely imagery-only), and specific conductance is NOT optically recoverable — an apparent SpCond R² 0.44 was a spurious brackish-vs-fresh regional proxy (within-station R² −1.67), not conductance retrieval, and is withdrawn. A gradient-boosted retrieval matches or beats the neural model on the same inputs.',
    secondaryMetrics: [
      { label: 'Dissolved-oxygen R² (imagery-only)', value: '0.50', glossary: 'r2' },
      { label: 'Dissolved-oxygen R² (+ pH strip)', value: '0.59', glossary: 'r2' },
      { label: 'Turbidity R² (log-scale)', value: '0.26', glossary: 'r2' },
      { label: 'Specific conductance', value: 'not optically recoverable' },
    ],
    architecture:
      'DenseNet121 + SpectralStem + CBAM + multi-scale FPN + FiLM pH-conditioning + per-target expert MLPs; a gradient-boosted spectral-feature model matches/beats it on the same inputs.',
  },
];

/** The five modality encoders, in canonical order (for the encoder grid). */
export const ENCODERS: ModelEntry[] = MODELS.filter((m) => m.kind === 'encoder');

/** Fast lookup by model id. */
export const MODELS_BY_ID: Record<string, ModelEntry> = Object.fromEntries(
  MODELS.map((m) => [m.id, m]),
);

/** Human labels + semantic status color for the four honesty tiers. */
export const STATUS_META: Record<
  ModelStatus,
  { label: string; tone: 'nominal' | 'watch' | 'alert' | 'neutral'; description: string }
> = {
  validated: {
    label: 'Validated',
    tone: 'nominal',
    description: 'Real, held-out data; genuine skill above baseline.',
  },
  'region-limited': {
    label: 'Region-limited',
    tone: 'watch',
    description: 'Real within a calibrated region; does not transfer cross-region.',
  },
  ceiling: {
    label: 'Genuine ceiling',
    tone: 'watch',
    description: 'Real but at the data ceiling; reported as an honest negative.',
  },
  simulation: {
    label: 'Simulation only',
    tone: 'neutral',
    description: 'Synthetic labels; proof-of-concept, not validated detection.',
  },
  circular: {
    label: 'Circular labels',
    tone: 'neutral',
    description: 'Labels are a closed-form function of the inputs; not a skill metric.',
  },
};
