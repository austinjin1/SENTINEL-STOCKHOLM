// SENTINEL — Methodology, holdout protocols, and the integrity story.
//
// This project just finished an integrity remediation. The honesty IS a
// feature and this module makes it presentable: what protocols we use, what was
// retracted and why, what is simulation, and the cross-cutting methodological
// finding (transferable representations). All content is REAL, cross-checked
// against README.md "Honest Scope" and "Key Findings".

// ---- Holdout protocols (what "held-out" means, per model) -------------
export interface ProtocolEntry {
  id: string;
  name: string;
  /** Glossary term key. */
  glossary: string;
  /** What it does. */
  description: string;
  /** Which models use it. */
  usedBy: string[];
}

export const PROTOCOLS: ProtocolEntry[] = [
  {
    id: 'station_holdout',
    name: 'Station holdout',
    glossary: 'station_holdout',
    description:
      'Whole monitoring stations are excluded from training and used only for test (GroupKFold by site). Blocks the model from memorizing a station baseline.',
    usedBy: ['AquaSSM', 'HydroViT'],
  },
  {
    id: 'geographic_holdout',
    name: 'Geographic holdout',
    glossary: 'geographic_holdout',
    description:
      'Whole geographic regions (k-means clusters over lat/lon) go entirely to train, val, or test. The hardest, most deployment-relevant test of generalization.',
    usedBy: ['MicroBiomeNet', 'Species Health Index'],
  },
  {
    id: 'loso',
    name: 'Leave-one-study-out',
    glossary: 'loso',
    description:
      'Train on all studies but one, test on the held-out study, rotate. Removes cross-study batch/platform leakage in toxicogenomics.',
    usedBy: ['ToxiGene'],
  },
  {
    id: 'station_grouped_cv',
    name: 'Station-grouped CV',
    glossary: 'station_grouped_cv',
    description:
      'Cross-validation folds hold out disjoint station sets; no station is split across train/test. CIs from site-clustered bootstrap.',
    usedBy: ['Perceiver IO Fusion', 'BioMotion'],
  },
  {
    id: 'cross_region_holdout',
    name: 'Cross-region (leave-one-HUC-out)',
    glossary: 'cross_region_holdout',
    description:
      'Leave-one-watershed-out over hydrologic units. The stress test where the microcystin fusion honestly collapses to chance.',
    usedBy: ['Perceiver IO Fusion (stress test)'],
  },
  {
    id: 'spatial_holdout',
    name: 'Spatial holdout',
    glossary: 'spatial_holdout',
    description:
      'Test stations are geographically separated from training stations, defeating spatial autocorrelation.',
    usedBy: ['SENTINEL-Lite'],
  },
];

// ---- The integrity / honesty story ------------------------------------
export interface RetractionEntry {
  id: string;
  /** What the retracted claim was. */
  claim: string;
  /** The fabricated/invalid value. */
  oldValue: string;
  /** Why it was wrong. */
  reason: string;
  /** The real, corrected result (or null if simply removed). */
  correctedTo: string | null;
}

export const RETRACTIONS: RetractionEntry[] = [
  {
    id: 'fusion_0992',
    claim: '"0.992 fusion beats single modality"',
    oldValue: '0.992',
    reason:
      'Computed on synthetic Gaussian streams with labels 60% blended from ground truth. Removed.',
    correctedTo:
      'Real fusion: microcystin 3-way 0.82 vs 0.68 best single (within region); ties best single on chl-a/DO.',
  },
  {
    id: 'case_studies_31',
    claim: '"31 events / 32-day median / zero false positives"',
    oldValue: '31 events, 32 d, 0 FP',
    reason: 'Produced on a synthetic simulation harness. Retracted.',
    correctedTo:
      'Real NEON: label-free rolling-CUSUM detector, median 36-day lead on 11/15 DO-hypoxia episodes at a matched 5.0% held-out FPR (Pareto-beats the point-threshold baseline, 5/15 @ 25.5 d).',
  },
  {
    id: 'microbiome_0831',
    claim: 'MicroBiomeNet source attribution 0.831',
    oldValue: '0.831',
    reason: 'Per-sample random split leaked correlated same-region samples across train/test.',
    correctedTo:
      'True geographic holdout: 0.676 with transferable taxonomic features (0.405 on raw OTUs).',
  },
  {
    id: 'hydrovit_0893',
    claim: 'HydroViT satellite WQ 0.893',
    oldValue: '0.893',
    reason: 'Fabricated fallback value. Rebuilt on real Sentinel-2 × in-situ pairs.',
    correctedTo: 'Station holdout: water-temp R² 0.729; chl-a R² 0.32 (red-edge).',
  },
  {
    id: 'species_1000',
    claim: 'Species Health Index AUROC 1.000',
    oldValue: '1.000',
    reason:
      'Circular — the input held the per-species counts that define the occupancy label.',
    correctedTo: 'Leakage-free retrain: macro-AUROC 0.949 (env-only floor ~0.85).',
  },
  {
    id: 'toxigene_0492',
    claim: 'ToxiGene toxicity 0.492',
    oldValue: '0.492',
    reason:
      'Invalid human/cattle disease GEO with hard-coded labels and placeholder gene names — not aquatic toxicology.',
    correctedTo:
      'Real zebrafish GEO: within-study AUROC 0.86; cross-study LOSO 0.64 (rank-normalized).',
  },
  {
    id: 'contrastive_recall',
    claim: 'Contrastive microbial–molecular Recall@1 = 1.000',
    oldValue: '1.000',
    reason: 'An N=1 evaluation on hash-seeded noise. Removed.',
    correctedTo: null,
  },
  {
    id: 'sensor_behavioral_mi',
    claim: 'Sensor↔behavioral mutual-information figure',
    oldValue: '—',
    reason: 'Computed on synthetic data. Removed.',
    correctedTo: null,
  },
];

// ---- What is simulation / proof-of-concept (not validated detection) --
export interface SimulationEntry {
  id: string;
  name: string;
  whatsReal: string;
  whatsSimulated: string;
}

export const SIMULATION_SCOPE: SimulationEntry[] = [
  {
    id: 'stream_gnn',
    name: 'Stream-Network GNN',
    whatsReal: 'NHDPlus river topology (561 sites, 338 edges).',
    whatsSimulated:
      'Contamination-propagation labels are synthetic (random injection + BFS propagation). Not validated on real events.',
  },
  {
    id: 'conformal',
    name: 'Conformal prediction pipeline',
    whatsReal: 'Real sensor coverage measured at 0.9367.',
    whatsSimulated: 'The 95% coverage guarantee is NOT met on real data.',
  },
  {
    id: 'causal',
    name: 'Causal screening',
    whatsReal: 'Lagged partial-correlation screening over real series.',
    whatsSimulated:
      'Not PCMCI+; 0 of 375 discovered "causal chains" validated against known chains.',
  },
  {
    id: 'disease_forecast',
    name: 'Disease-forecast risk index',
    whatsReal: 'Real USGS sensor inputs; WHO/CDC/EPA rule.',
    whatsSimulated: 'Labels are a closed-form function of the inputs (circular) — not a skill metric.',
  },
  {
    id: 'sentinel_lite_drone',
    name: 'SENTINEL-Lite drone payload',
    whatsReal: 'Imagery is real Sentinel-2; the vision model is trained/tested on it.',
    whatsSimulated: 'The dual-camera drone hardware is a proposed deployment concept, not a data source.',
  },
];

// ---- Honest negatives at the data ceiling (reported as findings) ------
export const HONEST_NEGATIVES: { id: string; finding: string }[] = [
  {
    id: 'twin',
    finding:
      'Digital Twin loses to a persistence baseline beyond 1 day (overall R² 0.07 vs persistence 0.92). Useful only at the 1-day horizon; uncertainty head conformal-calibrated to 90% coverage.',
  },
  {
    id: 'toxigene_transfer',
    finding: 'Cross-study molecular-toxicity transfer is near chance (LOSO AUROC ~0.55 raw).',
  },
  {
    id: 'predictability_audit',
    finding: 'Predictability audit: R² < 0 for DO / pH / turbidity from sensor data alone — they require multimodal signals.',
  },
  {
    id: 'arg',
    finding: 'ARG surveillance R² ≈ −0.008 — insufficient real metagenomic data (pseudo-labels only).',
  },
  {
    id: 'foundation',
    finding: 'Foundation / MoME joint-pretraining fusion AUROC 0.37–0.42 on test — limited multimodal co-occurrence in the data.',
  },
];

// ---- The cross-cutting methodological finding -------------------------
export const CENTRAL_FINDING = {
  title: 'Out-of-distribution skill comes from domain-informed pipelines, not raw model complexity',
  summary:
    "SENTINEL's honest cross-cutting methodological finding: on small, out-of-distribution-stressed tasks, naive under-regularized models on raw high-dimensional features transfer poorly, while domain-informed pipelines — regularization, supervised feature selection, batch-robust normalization, and, where physically warranted, additional informative bands — substantially recover OOD skill. Engineered representations (coarse taxonomy, rank-normalized genes, red-edge indices, self-normalized temporal residuals) are one effective, leakage-free route to that recovery — but a self-audit showed they are not uniquely responsible: a properly-regularized model on the raw features often recovers comparable skill. We therefore quantify representation gains against a regularized raw baseline, not an under-regularized straw-man, and deploy within-region where fusion is calibrated.",
  evidence: [
    {
      model: 'MicroBiomeNet',
      raw: 'Raw OTUs (geographic holdout): macro-F1 0.405',
      transferable: 'Family/genus taxonomy: macro-F1 0.676',
      note: 'Exact 90-bp ASVs are region-specific; higher taxa are shared across continents. Honest caveat: the 0.405 "collapse" is specific to an under-regularized model — a well-regularized MLP on raw OTUs reaches ~0.66 on the same holdout, comparable to the taxonomic features. So OOD recovery reflects regularization + representation together, not the taxonomic transform alone.',
    },
    {
      model: 'ToxiGene',
      raw: 'Raw expression (cross-study LOSO): AUROC 0.55',
      transferable: 'Per-sample rank-normalization: AUROC 0.64–0.68',
      note: 'Rank-normalization is batch correction — but one route, not uniquely necessary. A well-regularized raw-gene model (per-study standardized + supervised feature selection) reaches LOSO AUROC 0.66, inside this band, and rank-norm’s edge over it is not significant (Δ +0.03, CI includes 0). The 0.55 baseline was under-regularized (global-Z + variance selection).',
    },
    {
      model: 'HydroViT',
      raw: '4-band imagery (chl-a): R² 0.18',
      transferable: 'Red-edge (adds B05): R² 0.30',
      note: 'Confounded: ~36% of the gain is a literally new spectral band (B05 red-edge, absent from the 4-band set) and the ~64% index effect is contingent on it — on the 4-band information alone the indices do not help (Δ −0.02, null). So it is added information plus representation, not representation of the same features; 4-band chl-a caps near R² 0.10–0.12.',
    },
  ],
} as const;

// ---- The consolidated transfer-gap table (the project's spine) --------
// Every model, quantified: in-distribution vs out-of-distribution, and how much
// a transferable-representation fix recovers of the collapse. All numbers real,
// cross-checked against results/transfer_gap_analysis.json + the per-model JSONs.
export interface TransferGapRow {
  model: string;
  metric: string;
  /** In-distribution setting + score (score may be '—' where not a clean ID split). */
  idSetting: string;
  id: string;
  /** Out-of-distribution setting + raw-feature score (the honest collapse). */
  oodSetting: string;
  oodRaw: string;
  /** Score after the transferable-representation fix ('—' where no fix recovers it). */
  oodFixed: string;
  fix: string;
  /** Fraction (or absolute) of the ID→OOD gap the fix recovers; 'honest gap' when none does. */
  recovered: string;
  tone: 'nominal' | 'watch';
}

export const TRANSFER_GAP: TransferGapRow[] = [
  {
    model: 'MicroBiomeNet', metric: 'macro-F1',
    idSetting: 'random split', id: '0.831',
    oodSetting: 'geographic holdout', oodRaw: '0.405', oodFixed: '0.676',
    fix: 'Family/genus taxonomy (CLR)', recovered: '64% of gap', tone: 'nominal',
  },
  {
    model: 'ToxiGene', metric: 'AUROC',
    idSetting: 'within-study', id: '0.858',
    oodSetting: 'leave-one-study-out', oodRaw: '0.557', oodFixed: '0.642',
    fix: 'Per-sample rank-normalization', recovered: '28% of gap', tone: 'nominal',
  },
  {
    model: 'HydroViT', metric: 'chl-a R²',
    idSetting: 'station holdout', id: '—',
    oodSetting: '4-band imagery', oodRaw: '0.178', oodFixed: '0.320',
    fix: 'Red-edge bands B05–B07', recovered: '+0.14 abs', tone: 'nominal',
  },
  {
    model: 'Microcystin fusion', metric: 'AUROC',
    idSetting: 'within-region 3-way', id: '0.823',
    oodSetting: 'cross-region (HUC)', oodRaw: '0.485', oodFixed: '0.527',
    fix: 'None recovers (genotype spatially confined)', recovered: '12% of gap', tone: 'watch',
  },
  {
    model: 'Species Health', metric: 'macro-AUROC',
    idSetting: 'with geography', id: '0.944',
    oodSetting: 'env-only (no geo)', oodRaw: '0.853', oodFixed: '—',
    fix: 'None — occupancy is largely a range prior', recovered: 'honest gap', tone: 'watch',
  },
  {
    model: 'Disease nowcast', metric: 'AUROC',
    idSetting: 'within-region', id: '0.66–0.69',
    oodSetting: 'cross-region (state)', oodRaw: '~0.55', oodFixed: '—',
    fix: 'None — region-specific, needs local calibration', recovered: 'honest gap', tone: 'watch',
  },
];

export const TRANSFER_GAP_SUMMARY = {
  thesis:
    'Models look strong in-distribution (random / within-study / with-geography / within-region) and drop sharply out-of-distribution (geographic / leave-one-study-out / no-geography / cross-region). Domain-informed pipelines — regularization, supervised feature selection, batch-robust normalization, and where physically warranted a more informative band, often expressed as engineered representations (coarse taxonomy, rank normalization, red-edge indices) — recover a measured fraction of the gap, but never all of it. The honest deployment number is the OOD number.',
  meanRecovered: '~35%',
  meanRecoveredNote:
    'Fraction of the in-distribution→out-of-distribution collapse recovered by the domain-informed pipeline, across three benchmarks. IMPORTANT (self-audit): the recovery is NOT uniquely a representation effect — a well-regularized model on the RAW features recovers comparable OOD skill for MicroBiomeNet (~0.66 vs taxonomic 0.68) and ToxiGene (~0.66 vs rank-norm 0.66–0.69, Δ not significant), and HydroViT’s gain is partly a literally new band (B05). So the collapse baselines (0.405 / 0.55 / 0.18) were under-regularized straw-men; the honest attribution is regularization + feature-selection + normalization (+ bands), with engineered representations one effective route rather than the unique cause. Species and disease remain honest gaps.',
} as const;

// ---- The counterpoint: what DOES transfer — self-normalized lead time -----
// The transfer-gap table above is about learned CROSS-FEATURE relationships,
// which collapse OOD. The Track A lead-time detector transfers the opposite way:
// it learns nothing cross-site, only a self-normalized deviation from each site's
// own clean baseline, so its ALARM THRESHOLD is a near-universal constant and the
// operating point transfers to unseen sites. Verified by leave-one-out.
// Sources: results/trackA_leadtime/loso_generalization.json +
//          results/trackA_leadtime/aquassm_specificity/loco_cusum_generalization.json
export const LEADTIME_GENERALIZATION = {
  title: 'The self-normalized lead-time detector transfers across sites',
  summary:
    "The flip side of the transfer gap: because the Track A detector scores only a self-normalized deviation from each site's own clean baseline — never a learned cross-site relationship — its operating point is nearly site-invariant and transfers to streams it has never seen. We proved this by leaving sites out of calibration entirely.",
  evidence: [
    {
      test: 'Leave-one-site-out (per-site baseline detector)',
      finding:
        'The alarm threshold is a near-constant 2.79 ± 0.02σ. Transferred blind to each held-out site, it reproduces the per-site result: the same 9/12 episodes detected, identical 29.2-day median lead, and 4.1% held-out false-alarm rate on clean controls (vs 4.6% self-calibrated).',
    },
    {
      test: 'Leave-one-control-out (rolling-CUSUM headline)',
      finding:
        'The 11/15 headline threshold is calibrated only on control sites and applied to fully-unseen episode sites. Dropping any single control and re-calibrating catches the identical 11 episodes on every one of the 17 fits — detection is invariant to the calibration set.',
    },
  ],
  takeaway:
    'No site-specific threshold tuning is required — only an unlabeled clean baseline for self-normalization. The one limit is a minority of non-stationary streams that over-fire on clean water under any threshold (reported, not hidden); those use their own baseline for calibration.',
} as const;

/** One-paragraph honesty statement for the methodology page header. */
export const HONESTY_STATEMENT =
  'SENTINEL completed an integrity remediation. Every headline number on this site is real and held-out. Fabricated, synthetic-derived, and leakage-inflated figures were retracted and, where possible, replaced with the honestly-measured value. We show the corrections, the simulation-only components, and the honest negatives — because the honesty is the result.';
