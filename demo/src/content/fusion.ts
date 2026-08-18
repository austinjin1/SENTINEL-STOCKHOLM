// SENTINEL — The multimodal fusion story (microcystin).
//
// This is the headline scientific result and it is deliberately NUANCED:
// fusion genuinely wins within a calibrated region, and honestly collapses to
// chance across regions. Present both. All numbers are REAL, from
// station-grouped CV with site-clustered bootstrap CIs.
//
// Verified sources:
//   results/trackB_fusion/mc_fusion.json         (satellite / pigment / fusion AUROC + CIs)
//   results/trackB_fusion/mc_driver_fusion.json  (3rd modality = drivers; 3-way AUROC + bootstrap)
//   results/trackB_fusion/mc_region_transfer.json(cross-region collapse; no transform recovers)
//   results/trackB_fusion/mc_microbial_transfer.json (data-blocked genotype transfer path)
//   results/trackB_fusion/complementary_fusion.json  (chl-a/DO: fusion only ties best single)
//   results/trackB_fusion/FUSION_FINDINGS.md     (mechanism + conclusion)

export type ModalityKey = 'satellite' | 'pigment' | 'drivers';

export interface ModalityMeta {
  key: ModalityKey;
  label: string;
  /** What this modality physically measures. */
  measures: string;
  /** Why it is non-redundant for an optically-invisible toxin. */
  rationale: string;
}

export const MODALITIES: ModalityMeta[] = [
  {
    key: 'satellite',
    label: 'Satellite optics',
    measures: 'Sentinel-2 spectral band-ratios (chl-a, phycocyanin, red-edge indices)',
    rationale:
      'Sees surface bloom biomass, but the toxin itself is optically invisible — so optics alone is incomplete.',
  },
  {
    key: 'pigment',
    label: 'In-situ pigment',
    measures: 'In-situ chlorophyll-a and phycocyanin (CHLA, PC)',
    rationale:
      'Phycocyanin correlates with cyanobacteria (phycocyanin↔microcystin r≈0.73 vs chl-a↔microcystin r≈0.08) — a stronger toxin proxy than chl-a.',
  },
  {
    key: 'drivers',
    label: 'Physicochemical drivers',
    measures: 'Water temperature, phosphorus (and nitrogen where available)',
    rationale:
      'Temperature and nutrients drive toxin production — an orthogonal, causal signal that neither optics nor pigment captures.',
  },
];

export interface ThresholdRow {
  /** Machine key. */
  key: string;
  /** Display label, e.g. "> 1.6 µg/L (WHO drinking)". */
  label: string;
  /** Clinical/regulatory meaning of the threshold. */
  meaning: string;
  /** Number of positive samples at this threshold. */
  nPos: number;
  /** Number of positive sites. */
  nSitesPos?: number;
  /** Per-modality single-modality AUROC. */
  single: Record<ModalityKey, number>;
  /** Modalities NOT separately re-run at this threshold — displayed as not-measured, never as a real per-threshold AUROC. */
  singleNotMeasured?: Partial<Record<ModalityKey, boolean>>;
  /** 2-way fusion (satellite + pigment) AUROC. */
  twoWay: number;
  /** 3-way fusion (+ drivers) AUROC. */
  threeWay: number;
  /** 95% CI on the 3-way fusion AUROC. */
  threeWayCI: [number, number];
  /** Best single-modality AUROC (max of `single`). */
  bestSingle: number;
  /** 3-way minus best single (the headline gain). */
  gainVsBestSingle: number;
  /** Site-bootstrap CI on (3-way minus 2-way) — the marginal driver gain. */
  driverGainCI: [number, number];
  /** Negative control: shuffle drivers → fusion AUROC drops back toward 2-way. */
  permutationControl: number;
  /** Does the 3-way fusion beat best single with the CI excluding zero? */
  fusionHelps: boolean;
}

// Total co-located dataset: 3,649 usable Sentinel-2 tiles across 227 sites.
export const FUSION_DATASET = {
  task: 'microcystin',
  taskLabel: 'Microcystin (optically-invisible cyanotoxin)',
  nTiles: 3649,
  nSites: 227,
  cv: 'Station-grouped cross-validation, site-clustered bootstrap 95% CIs',
} as const;

// Per-threshold breakdown. Single/2-way/3-way values from mc_fusion.json +
// mc_driver_fusion.json. The WHO-drinking row (mc > 1.6) is the headline.
export const FUSION_THRESHOLDS: ThresholdRow[] = [
  {
    key: 'mc_gt1.6',
    label: '> 1.6 µg/L',
    meaning: 'WHO drinking-water guideline for microcystin-LR',
    nPos: 633,
    nSitesPos: 126,
    // Satellite arm shown at its best-achievable feature set (literature cyano indices NDCI/RE-ratio/MCI/FAI:
    // AUROC 0.708 per mc_satellite_arm.json), NOT the under-featured 0.65 band-mean arm. So the honest best
    // single is the improved satellite arm (0.708), above pigment (0.684) — the fusion gain is +0.08, not +0.138.
    single: { satellite: 0.708, pigment: 0.684, drivers: 0.626 },
    twoWay: 0.776,
    threeWay: 0.823,
    threeWayCI: [0.79, 0.851], // absolute 3-way AUROC, site-clustered bootstrap (mc_driver_fusion.json boot_3way_absolute)
    bestSingle: 0.708, // satellite-improved arm (mc_satellite_arm.json), above pigment 0.684
    gainVsBestSingle: 0.081, // 3-way vs best IMPROVED single, boot median CI [0.047,0.121] P=1.0 (mc_satellite_arm.json)
    driverGainCI: [0.028, 0.068], // 3-way vs 2-way, site bootstrap, P>0 = 1.0 — the CI-clean load-bearing increment
    permutationControl: 0.686,
    fusionHelps: true,
  },
  {
    key: 'mc_gt0.3',
    label: '> 0.3 µg/L',
    meaning: 'Detection / early-screening threshold',
    nPos: 1286,
    nSitesPos: 184,
    single: { satellite: 0.664, pigment: 0.718, drivers: 0.569 },
    twoWay: 0.786,
    threeWay: 0.8,
    threeWayCI: [0.769, 0.828], // absolute 3-way AUROC, site-clustered bootstrap (mc_driver_fusion.json boot_3way_absolute)
    bestSingle: 0.718,
    gainVsBestSingle: 0.082,
    driverGainCI: [0.005, 0.023],
    permutationControl: 0.708,
    fusionHelps: true,
  },
  {
    key: 'mc_gt8',
    label: '> 8 µg/L',
    meaning: 'WHO recreational-water guideline (severe bloom)',
    nPos: 286,
    nSitesPos: 72,
    single: { satellite: 0.669, pigment: 0.748, drivers: 0.669 }, // drivers NOT run at this threshold — placeholder, suppressed from display (see singleNotMeasured)
    singleNotMeasured: { drivers: true },
    twoWay: 0.792,
    threeWay: 0.792, // drivers not separately re-run at this threshold; 2-way = fusion
    threeWayCI: [0.76, 0.824],
    bestSingle: 0.748,
    gainVsBestSingle: 0.044,
    driverGainCI: [-0.013, 0.103], // CI includes 0 at this rarer threshold
    permutationControl: 0.713,
    fusionHelps: false, // CI includes 0 — honestly not a win at this threshold
  },
];

/** The headline row for hero copy (WHO drinking-water threshold). */
export const FUSION_HEADLINE = FUSION_THRESHOLDS[0];

// ---- The cross-region caveat (the honest boundary) --------------------
export interface RegionTransferRow {
  thresholdKey: string;
  thresholdLabel: string;
  /** Raw cross-region (leave-one-HUC-out) AUROC — collapses to chance. */
  raw: number;
  /** Best result any feature transform achieved (still ~chance). */
  bestTransform: number;
  bestTransformLabel: string;
  nCrossRegion: number;
  nRegions: number;
}

export const CROSS_REGION_TRANSFER: RegionTransferRow[] = [
  {
    thresholdKey: 'mc_gt1.6',
    thresholdLabel: '> 1.6 µg/L (WHO drinking)',
    raw: 0.485,
    bestTransform: 0.527,
    bestTransformLabel: '+domain (lat/lon/month)',
    nCrossRegion: 3594,
    nRegions: 5,
  },
  {
    thresholdKey: 'mc_gt0.3',
    thresholdLabel: '> 0.3 µg/L (detection)',
    raw: 0.49,
    bestTransform: 0.492,
    bestTransformLabel: 'fold-rank (train-only)',
    nCrossRegion: 3594,
    nRegions: 5,
  },
];

// ---- The mechanism (why it wins, and why it can't transfer) -----------
export const FUSION_MECHANISM = {
  whyItWins:
    'Microcystin is optically invisible, so satellite optics, in-situ pigment, and physicochemical drivers carry genuinely non-redundant information. Each modality adds real signal: 3-way > 2-way > best single, with 95% CIs excluding zero, and a shuffle-the-drivers negative control confirming it is signal rather than overfit capacity.',
  whyItCollapses:
    'Microcystin production depends on the toxin-producing strain fraction — the mcyD genotype — which varies dramatically between watersheds (from <6% to ~100% of cells) and predicts toxin better than any pigment. Optics, pigment, and drivers cannot see that genotype, so the learned relationship does not transfer to unseen regions. No feature transform (region-conditioning, within-site-relative, or global-rank) recovers it.',
  theTransferPath:
    "SENTINEL's own microbial modality directly measures the mcyE/ndaF toxin-gene genotype. Real co-located genotype data does exist at 25 of the 227 fusion sites (WQP mcyE/ndaF qPCR + cyanobacteria density), and adding it as a 4th modality raises the within-region fusion from 0.78 to ~0.86 — directionally confirming the mechanism (suggestive only: 23 positives, wide CI overlapping the baseline). But cross-region transfer stays structurally untestable: the genotype data is spatially confined to a single HUC region (3 monitoring programs), so no leave-one-region-out is possible. The science-grounded route — satellite + pigment + driver + genotype fusion — is gated on acquiring co-located mcyE at ≥2 regions (e.g. the Florida or Great-Lakes HAB sites).",
  deploymentRobustness:
    'For deployment, every one of the three modalities is a verified non-redundant contributor: leave-one-out and Shapley marginals both have site-clustered 95% CIs excluding zero (order: pigment > satellite > drivers). A per-sample reliability-gated variant degrades gracefully when a modality is missing at inference — dropping satellite, pigment, or drivers loses signal in exactly that measured order, with no failure. That variant trades accuracy for robustness (AUROC 0.75 vs the gradient-boosted 3-way stack at 0.82), so the headline number stays the boosted stack; the reliability gate is a graceful-degradation mechanism, not an accuracy gain.',
} as const;

// ---- Where fusion does NOT win (honest redundancy) --------------------
export interface RedundancyRow {
  task: string;
  taskLabel: string;
  horizon?: string;
  bestSingleLabel: string;
  bestSingle: number;
  fusion: number;
  persistence?: number;
  verdict: string;
}

export const FUSION_REDUNDANCY: RedundancyRow[] = [
  {
    task: 'new_onset_do_h7',
    taskLabel: 'DO-hypoxia new-onset forecast',
    horizon: '7-day',
    bestSingleLabel: 'sensor',
    bestSingle: 0.946,
    fusion: 0.949,
    persistence: 0.929,
    verdict: 'Fusion ties the best single (gain +0.003, CI includes 0). Sensor/persistence dominant — DO is chemistry, satellite-blind.',
  },
  {
    task: 'new_onset_do_h14',
    taskLabel: 'DO-hypoxia new-onset forecast',
    horizon: '14-day',
    bestSingleLabel: 'persistence',
    bestSingle: 0.902,
    fusion: 0.904,
    persistence: 0.902,
    verdict: 'Fusion ties persistence (gain +0.006, CI includes 0). Single-modality-dominated, as the literature predicts.',
  },
  {
    task: 'chla',
    taskLabel: 'Chlorophyll-a',
    bestSingleLabel: 'satellite spectral',
    bestSingle: 0.716,
    fusion: 0.708,
    verdict: 'Fusion matches but does NOT beat the best single. Chl-a is optical — satellite dominates. The retracted "0.992 fusion beats single" was synthetic.',
  },
];

// ---- Bloom early-warning: a second-crisis, antecedent-only forecast ---------
// SENTINEL forecasts a SECOND water crisis beyond hypoxia: NEW algal-bloom onset.
// Leakage-free within-region (GroupKFold-by-site); every predictor strictly
// antecedent (t0 Sentinel-2 optics + optical trajectory, prior in-situ chl + trend,
// drivers, season); target = chl-a exceedance at the NEXT in-situ visit.
// Source: results/trackB_fusion/bloom_ew/bloom_early_warning.json
//   scripts/experiments/trackB_fusion/bloom_ew/bloom_early_warning.py (+ verify_bloom_onset.py)
export const BLOOM_EARLY_WARNING = {
  title: 'Bloom early-warning: forecasting a second crisis',
  horizon: '~2 weeks (median realized lead 14 days)',
  horizonNote:
    'The ~2-week horizon is set by the WQP in-situ visit cadence, not the 5-day Sentinel-2 revisit.',
  // Onset-only test (restricted to visits where a bloom is NOT already present, so
  // persistence-of-state is uninformative), WHO Alert-2 threshold (chl-a > 24 µg/L).
  onsetFusionAuroc: 0.733,
  onsetFusionCI: [0.691, 0.777] as [number, number],
  nOnsetPairs: 940,
  nOnsets: 207,
  falseAlarmRate: 0.14,
  recallAtFar: 0.45,
  // The honest caveat: fusion does NOT statistically beat the best single modality.
  fusionMinusBestSingle: 0.019,
  fusionMinusBestSingleCI: [-0.007, 0.046] as [number, number],
  persistenceModelAuroc: 0.714, // antecedent biomass trajectory, WHO Alert-2
  satelliteEutrophicAuroc: 0.708, // satellite optics dominates at chl>8
  allPairsFusion: 0.83,
  allPairsPersistence: 0.822,
  summary:
    'SENTINEL forecasts a second water crisis beyond hypoxia: new algal-bloom onset. From strictly-antecedent signals — t0 Sentinel-2 optics and its trajectory, prior in-situ chl-a and its trend, physicochemical drivers, and season — it predicts a NEW WHO Alert-2 bloom (chl-a > 24 µg/L) at the next in-situ visit, a leakage-free within-region forecast (GroupKFold-by-site).',
  caveat:
    'This is a genuine early-warning capability, NOT a "fusion beats single modalities" claim. On the strict onset-only test, fusion does not statistically beat the best single modality (delta +0.019, CI [-0.007, 0.046] includes 0): the forecast is single-modality-dominated by severity — antecedent biomass trajectory dominates at WHO Alert-2 (persistence-model 0.714), satellite optics dominates at the eutrophic chl>8 threshold (0.708) — and on the easy all-pairs task fusion (0.83) merely ties a trivial persistence baseline (0.822).',
} as const;

/** One-sentence takeaway for the page header. */
export const FUSION_SUMMARY =
  'Multimodal fusion is a calibrated within-region capability: on microcystin, 3-way fusion beats every single modality (0.82, +0.08 over the best single — a literature-index satellite arm at 0.71 — at the WHO drinking threshold, CI excludes zero, permutation-confirmed), because the optically-invisible toxin makes optics, pigment, and drivers genuinely non-redundant. Its deployment scope is explicit — fusion is a within-region tool; cross-region transfer needs local calibration or the genotype modality (the same model reads ~0.49 across unseen watersheds, since the mcyD toxic-strain fraction is unobserved). On chl-a and DO, fusion only ties the best single. We report every case.';
