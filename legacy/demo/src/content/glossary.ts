// SENTINEL — Glossary of metrics, holdout protocols, and honesty terms.
//
// Every number surfaced in the UI should be explainable via a tooltip.
// These definitions are the single source of truth for those tooltips.
// Written to be read by a non-specialist judge without dumbing down the science.
//
// Source of truth: /home/bcheng/SENTINEL/README.md and results/*.json.
// REAL, verified content only — no fabricated figures.

export type GlossaryCategory = 'metric' | 'protocol' | 'honesty' | 'domain';

export interface GlossaryEntry {
  /** Stable key used to link a metric/UI element to its definition. */
  term: string;
  /** Short human-readable label. */
  label: string;
  /** Optional abbreviation as it appears next to numbers. */
  abbr?: string;
  category: GlossaryCategory;
  /** One or two sentences: what it is, and how to read it. */
  definition: string;
  /** Why it matters for trusting the SENTINEL numbers (the holdout protocol angle). */
  whyItMatters?: string;
}

export const GLOSSARY: GlossaryEntry[] = [
  // ---- Metrics ---------------------------------------------------------
  {
    term: 'auroc',
    label: 'AUROC',
    abbr: 'AUROC',
    category: 'metric',
    definition:
      'Area under the ROC curve. The probability that the model ranks a random positive (e.g. a contaminated sample) above a random negative. 0.5 is chance; 1.0 is perfect.',
    whyItMatters:
      'AUROC is threshold-free, so it summarizes ranking skill without cherry-picking an operating point. We report it with confidence intervals from site-clustered bootstrap resampling.',
  },
  {
    term: 'macro_f1',
    label: 'Macro-F1',
    abbr: 'macro-F1',
    category: 'metric',
    definition:
      'The F1 score (harmonic mean of precision and recall) averaged equally across every class. Unlike accuracy, it is not dominated by the largest class.',
    whyItMatters:
      'For 8-class source attribution with imbalanced classes, macro-F1 is the honest metric: a model that only predicts the majority class scores poorly.',
  },
  {
    term: 'r2',
    label: 'R-squared',
    abbr: 'R²',
    category: 'metric',
    definition:
      'The fraction of variance in the target that the model explains. 1.0 is perfect; 0.0 matches predicting the mean; negative means the model is worse than predicting the mean.',
    whyItMatters:
      'A negative R² is reported honestly as a failure (e.g. the Digital Twin beyond one day), not hidden.',
  },
  {
    term: 'rmse',
    label: 'RMSE',
    abbr: 'RMSE',
    category: 'metric',
    definition:
      'Root-mean-square error — the typical size of the prediction error, in the same units as the target (here, standardized sensor units). Lower is better.',
    whyItMatters:
      'AquaSSM reconstruction RMSE is measured on sequences from sites never seen in training.',
  },
  {
    term: 'fpr',
    label: 'False-positive rate',
    abbr: 'FPR',
    category: 'metric',
    definition:
      'The fraction of clean, non-event periods that the detector wrongly flags as alarms. Measured here as a sustained-alarm rate on held-out clean water.',
    whyItMatters:
      "A detector that fires constantly can look like it gives 'early warning' on every event. Fixing the FPR at a target specificity is what makes a lead-time credible.",
  },
  {
    term: 'lead_time',
    label: 'Lead time',
    category: 'metric',
    definition:
      'How many days before a verified event onset the calibrated detector first raised a sustained alarm. Credited only on episodes that also meet the false-positive-rate bar.',
    whyItMatters:
      "An 'early' alarm from a trigger-happy detector is not a real early warning, so lead time is only counted on specificity-validated episodes.",
  },
  {
    term: 'ci95',
    label: '95% confidence interval',
    abbr: '95% CI',
    category: 'metric',
    definition:
      'A range that, under repeated resampling, contains the true value about 95% of the time. Narrow intervals mean a well-estimated result.',
    whyItMatters:
      'SENTINEL intervals come from resampling whole sites (not individual rows), which is the honest way to reflect that samples from the same site are correlated.',
  },
  {
    term: 'pr_auc',
    label: 'PR-AUC',
    abbr: 'PR-AUC',
    category: 'metric',
    definition:
      'Area under the precision–recall curve. More informative than AUROC when positive events are rare, because it focuses on how well true events are found without drowning in false alarms.',
  },

  // ---- Holdout protocols ----------------------------------------------
  {
    term: 'station_holdout',
    label: 'Station holdout',
    category: 'protocol',
    definition:
      'Test data comes only from monitoring stations that were entirely excluded from training (GroupKFold by site). No station appears in both train and test.',
    whyItMatters:
      "This blocks the most common leakage in environmental ML: memorizing a station's baseline. It is stricter than a random row split.",
  },
  {
    term: 'geographic_holdout',
    label: 'Geographic holdout',
    category: 'protocol',
    definition:
      'Whole geographic regions (here, k-means clusters over latitude/longitude) are assigned entirely to train, validation, or test. The model is scored on regions it has never seen.',
    whyItMatters:
      'True cross-region generalization is the hardest and most deployment-relevant test. It is where several SENTINEL results honestly drop — and we report the drop.',
  },
  {
    term: 'cross_region_holdout',
    label: 'Cross-region (leave-one-HUC-out) holdout',
    category: 'protocol',
    definition:
      'Leave-one-region-out evaluation over hydrologic units (HUC watersheds): train on all regions but one, test on the held-out watershed, and rotate.',
    whyItMatters:
      'The microcystin fusion result passes within-region but collapses to chance (~0.49) here — the honest boundary of what transfers to unseen watersheds.',
  },
  {
    term: 'loso',
    label: 'Leave-one-study-out',
    abbr: 'LOSO',
    category: 'protocol',
    definition:
      'For toxicogenomics: train on all studies but one, test on the held-out study, and rotate across every study. Removes batch/platform leakage between studies.',
    whyItMatters:
      'Within-study performance can be inflated by shared lab conditions. LOSO measures whether an exposure signature actually transfers to a new experiment.',
  },
  {
    term: 'station_grouped_cv',
    label: 'Station-grouped cross-validation',
    category: 'protocol',
    definition:
      'Cross-validation where each fold holds out a disjoint set of stations, so no station is split across train and test in any fold.',
    whyItMatters:
      'The microcystin fusion win (0.82 vs 0.68) is measured under this protocol — real, non-leaked, within-region signal.',
  },
  {
    term: 'spatial_holdout',
    label: 'Spatial holdout',
    category: 'protocol',
    definition:
      'Test stations are geographically separated from training stations, so the model cannot exploit spatial autocorrelation with a nearby training site.',
    whyItMatters:
      'Used for SENTINEL-Lite (imagery-only screening): the 399 stations are split so test locations are unseen.',
  },
  {
    term: 'site_bootstrap',
    label: 'Site-clustered bootstrap',
    category: 'protocol',
    definition:
      'Confidence intervals computed by resampling whole sites with replacement, not individual rows.',
    whyItMatters:
      'Row-level bootstrap would give falsely tight intervals because samples from one site are correlated. Site-level resampling is the honest choice.',
  },

  // ---- Honesty / integrity terms --------------------------------------
  {
    term: 'circular_label',
    label: 'Circular label',
    category: 'honesty',
    definition:
      'A target that is a closed-form function of the model inputs, so a high score reflects recovering a known rule rather than predictive skill.',
    whyItMatters:
      'The Disease-Forecast risk index has circular labels; its high AUROC is flagged as NOT a skill metric.',
  },
  {
    term: 'data_leakage',
    label: 'Data leakage',
    category: 'honesty',
    definition:
      'When information from the test set (or a proxy for the label) leaks into training, inflating scores. Random splits across correlated sites are a common cause.',
    whyItMatters:
      "The retracted MicroBiomeNet 0.831 and Species-Health 1.000 were leakage. Leakage-free retraining gives the honest 0.676 and 0.949.",
  },
  {
    term: 'simulation_only',
    label: 'Simulation only',
    category: 'honesty',
    definition:
      'A model trained and evaluated on synthetic labels. It can demonstrate that an architecture learns a mechanism, but is not evidence of real-world detection skill.',
    whyItMatters:
      'The Stream-Network GNN is labeled simulation-only: real river topology, synthetic propagation labels.',
  },
  {
    term: 'ceiling',
    label: 'Genuine ceiling',
    category: 'honesty',
    definition:
      'A result that is real but at the limit of what the data can support — reported as an honest negative finding, not spun as a success.',
    whyItMatters:
      'Digital Twin < persistence beyond one day, and near-chance cross-study toxicity transfer, are reported as ceilings.',
  },
  {
    term: 'permutation_control',
    label: 'Permutation / negative control',
    category: 'honesty',
    definition:
      'A sanity check where a signal is shuffled (e.g. drivers randomized). If the model still scores high, the original result was overfitting, not real signal.',
    whyItMatters:
      'Shuffling the driver modality collapses the microcystin fusion back to the 2-way score — confirming the 3-way gain is real signal.',
  },
  {
    term: 'transferable_representation',
    label: 'Transferable representation',
    category: 'honesty',
    definition:
      'A feature encoding chosen to generalize across regions/studies (higher taxonomy, rank-normalized genes, red-edge spectral indices) instead of memorizing raw, region-specific features.',
    whyItMatters:
      "This is SENTINEL's central methodological finding: transferable representations recover out-of-distribution skill where raw features collapse.",
  },

  // ---- Domain terms ----------------------------------------------------
  {
    term: 'microcystin',
    label: 'Microcystin',
    category: 'domain',
    definition:
      'A liver toxin (hepatotoxin) produced by some cyanobacteria during harmful algal blooms. It is optically invisible — you cannot see it directly in satellite imagery — which is what makes multimodal fusion valuable.',
    whyItMatters:
      'Because the toxin is not directly visible, satellite optics, in-situ pigment, and physicochemical drivers each carry non-redundant information.',
  },
  {
    term: 'mcyd_genotype',
    label: 'mcyD toxic-strain genotype',
    category: 'domain',
    definition:
      'The fraction of a cyanobacterial population carrying the mcyD gene cluster that actually produces microcystin. It varies dramatically between watersheds (from <6% to ~100% of cells).',
    whyItMatters:
      'This unobserved variable is why microcystin fusion does not transfer cross-region: optics/pigment/drivers cannot see the toxic-strain fraction. Metagenomics can.',
  },
  {
    term: 'red_edge',
    label: 'Red-edge bands',
    category: 'domain',
    definition:
      'Sentinel-2 spectral bands B05–B07 (~705–783 nm) at the sharp reflectance transition of vegetation and algae. They are more sensitive to chlorophyll and phycocyanin than the standard visible bands.',
    whyItMatters:
      'Adding red-edge bands lifts chl-a R² from 0.18 (4-band) to 0.32 — a transferable-representation win with a CI that excludes zero.',
  },
  {
    term: 'persistence_baseline',
    label: 'Persistence baseline',
    category: 'domain',
    definition:
      "A naive forecast that predicts the future equals the present ('tomorrow = today'). Hard to beat for slowly-changing environmental variables.",
    whyItMatters:
      'The Digital Twin only beats persistence at the 1-day horizon; persistence wins beyond that, and we say so.',
  },
  {
    term: 'do_hypoxia',
    label: 'Dissolved-oxygen hypoxia',
    category: 'domain',
    definition:
      'A drop in dissolved oxygen (here, a verified onset below 4 mg/L) that stresses or kills aquatic life. The target event for the AquaSSM lead-time detector.',
  },
];

/** Fast lookup by term key, for tooltip wiring. */
export const GLOSSARY_BY_TERM: Record<string, GlossaryEntry> = Object.fromEntries(
  GLOSSARY.map((e) => [e.term, e]),
);
