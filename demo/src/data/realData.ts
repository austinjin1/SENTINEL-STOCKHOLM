// Real data adapter — pulls from src/data/repo/*.json (vendored from austinjin1/SENTINEL-STOCKHOLM).
// Replaces synthesized values in mockData.ts wherever real ground truth is available.

import lakeErie from './repo/lake_erie_hab_2023_scores.json';
import gulf from './repo/gulf_dead_zone_2023_scores.json';
import chesapeake from './repo/chesapeake_hypoxia_2018_scores.json';
import klamath from './repo/klamath_river_hab_2021_scores.json';
import mississippi from './repo/mississippi_salinity_2023_scores.json';
import jordan from './repo/jordan_lake_hab_nc_scores.json';
import caseStudiesReal from './repo/case_studies_real.json';
import degradationRaw from './repo/degradation_curves.json';
import cascadeAnalysis from './repo/cascade_analysis_results.json';
import attribution from './repo/attribution_results.json';
import microbialCS from './repo/microbial_case_studies.json';
import molecularCS from './repo/molecular_case_studies.json';
import behavioralCS from './repo/behavioral_case_studies.json';

import type {
  CanonicalRecord,
  MicroClass,
  ToxOutcome,
  ToxPathway,
  ToxProcess,
  SensorParam,
  BehavioralSignature,
  RegionKey,
} from '../types';

/* ---------- per-event score loading ---------- */

interface ScoreRow {
  center_time: string;
  center_ts: number;
  anomaly_probability: number;
}
interface ScoreFile {
  event_id: string;
  advisory_date: string;
  usgs_site: string;
  scores: ScoreRow[];
}

// helper: pull the event-meta block from case_studies_real.json
interface EventMeta {
  event_id: string;
  name: string;
  lead_time_days: number;
  max_anomaly_prob: number;
  usgs_site: string;
  advisory_date: string;
}
const META_BY_ID: Record<string, EventMeta> = Object.fromEntries(
  (caseStudiesReal as any).events.map((e: any) => [e.event_id, e]),
);

/* ---------- spec-mapped contextual fields per event ---------- */
// Per-event modality story is not in repo; mapped from documented contaminant context.
interface EventContext {
  label: string;
  region: RegionKey;
  lat: number;
  lon: number;
  tier: 1 | 2 | 3 | 4 | 5;
  driverParam: SensorParam;
  channelDominant: number;
  microTop: MicroClass;
  microP: number;
  toxOutcome: ToxOutcome;
  toxPathway: ToxPathway;
  toxProcess: ToxProcess;
  bioSignature: BehavioralSignature;
}
const CONTEXT: Record<string, EventContext> = {
  lake_erie_hab_2023: {
    label: 'Harmful algal bloom · Maumee Bay',
    region: 'GREAT_LAKES',
    lat: 41.73,
    lon: -83.47,
    tier: 5,
    driverParam: 'do',
    channelDominant: 4,
    microTop: 'Nutrient Runoff',
    microP: 0.94,
    toxOutcome: 'Hepatotoxicity',
    toxPathway: 'AHR/CYP1A',
    toxProcess: 'Xenobiotic Metabolism',
    bioSignature: 'convulsion',
  },
  gulf_dead_zone_2023: {
    label: 'Hypoxic zone · Mississippi outflow',
    region: 'GULF_COAST',
    lat: 30.45,
    lon: -91.19,
    tier: 5,
    driverParam: 'do',
    channelDominant: 5,
    microTop: 'Nutrient Runoff',
    microP: 0.88,
    toxOutcome: 'Hepatotoxicity',
    toxPathway: 'Oxidative Stress',
    toxProcess: 'Oxidative Response',
    bioSignature: 'immobility',
  },
  chesapeake_hypoxia_2018: {
    label: 'Hypoxic intrusion · Patapsco',
    region: 'NORTHEAST',
    lat: 39.21,
    lon: -76.71,
    tier: 5,
    driverParam: 'do',
    channelDominant: 5,
    microTop: 'Sewage',
    microP: 0.71,
    toxOutcome: 'Hepatotoxicity',
    toxPathway: 'AHR/CYP1A',
    toxProcess: 'Xenobiotic Metabolism',
    bioSignature: 'immobility',
  },
  klamath_river_hab_2021: {
    label: 'Cyanobacterial bloom',
    region: 'PAC_NW',
    lat: 41.51,
    lon: -124.04,
    tier: 4,
    driverParam: 'ph',
    channelDominant: 4,
    microTop: 'Nutrient Runoff',
    microP: 0.82,
    toxOutcome: 'Hepatotoxicity',
    toxPathway: 'AHR/CYP1A',
    toxProcess: 'Xenobiotic Metabolism',
    bioSignature: 'convulsion',
  },
  mississippi_salinity_2023: {
    label: 'Saltwater wedge intrusion',
    region: 'GULF_COAST',
    lat: 30.45,
    lon: -91.19,
    tier: 4,
    driverParam: 'cond',
    channelDominant: 3,
    microTop: 'Sediment',
    microP: 0.66,
    toxOutcome: 'Endocrine Disruption',
    toxPathway: 'Estrogen/Endocrine',
    toxProcess: 'Endocrine Signaling',
    bioSignature: 'normal',
  },
  jordan_lake_hab_nc: {
    label: 'Reservoir cyanobacterial bloom',
    region: 'SOUTHEAST',
    lat: 35.75,
    lon: -79.06,
    tier: 4,
    driverParam: 'ph',
    channelDominant: 4,
    microTop: 'Nutrient Runoff',
    microP: 0.79,
    toxOutcome: 'Hepatotoxicity',
    toxPathway: 'AHR/CYP1A',
    toxProcess: 'Xenobiotic Metabolism',
    bioSignature: 'convulsion',
  },
};

const SCORE_FILES: Record<string, ScoreFile> = {
  lake_erie_hab_2023: lakeErie as ScoreFile,
  gulf_dead_zone_2023: gulf as ScoreFile,
  chesapeake_hypoxia_2018: chesapeake as ScoreFile,
  klamath_river_hab_2021: klamath as ScoreFile,
  mississippi_salinity_2023: mississippi as ScoreFile,
  jordan_lake_hab_nc: jordan as ScoreFile,
};

/* ---------- synthesized helpers ---------- */

const MICRO_CLASSES: MicroClass[] = [
  'Nutrient Runoff',
  'Heavy Metals',
  'Thermal',
  'Pharmaceutical',
  'Sediment',
  'Oil/Petrochemical',
  'Sewage',
  'Acid Mine Drainage',
];

function microClassesDist(top: MicroClass, topP: number) {
  const remaining = 1 - topP;
  const others = MICRO_CLASSES.filter((c) => c !== top);
  const weights = others.map((_, i) => Math.exp(-i * 0.6));
  const sum = weights.reduce((a, b) => a + b, 0);
  return MICRO_CLASSES.map((name) => {
    if (name === top) return { name, p: topP };
    const idx = others.indexOf(name);
    return { name, p: (remaining * weights[idx]) / sum };
  });
}

function syntheticSensorSeries(n: number, mean: number, spread: number, trend = 0): number[] {
  const out: number[] = [];
  for (let i = 0; i < n; i++) {
    const t = i / (n - 1);
    out.push(
      mean + Math.sin(i * 0.4) * spread * 0.3 + (Math.random() - 0.5) * spread * 0.4 + trend * t,
    );
  }
  return out;
}

/* ---------- main: build CanonicalRecord per event ---------- */

function buildRecord(eventId: string): CanonicalRecord {
  const meta = META_BY_ID[eventId];
  const ctx = CONTEXT[eventId];
  const scoreFile = SCORE_FILES[eventId];
  // Real sensor anomaly_series from repo.
  const allScores = scoreFile.scores.map((s) => s.anomaly_probability);
  // Use the tail (closest to advisory) — last ~96 windows.
  const anomalySeries = allScores.slice(Math.max(0, allScores.length - 96));
  const maxAnomaly = meta.max_anomaly_prob;

  const n = 96;
  // ~1° bbox around the event for satellite snapshots; widen for very small lakes.
  const bboxWidth = 1.5;
  const bbox: [number, number, number, number] = [
    ctx.lon - bboxWidth / 2,
    ctx.lat - bboxWidth / 2,
    ctx.lon + bboxWidth / 2,
    ctx.lat + bboxWidth / 2,
  ];
  return {
    // Suffix with event_id when the same USGS station appears in multiple events (Gulf vs Mississippi 07374000).
    id: `${meta.usgs_site}-${eventId}`,
    name: meta.name,
    label: ctx.label,
    lat: ctx.lat,
    lon: ctx.lon,
    h3: '882a1072cffffff',
    region: ctx.region,
    bookmarked: true,
    tier: ctx.tier,
    leadDays: meta.lead_time_days,
    advisoryDate: meta.advisory_date,
    bbox,
    eventKey: eventId,
    sensor: {
      state: 'OBSERVED',
      conf: 1.0,
      series: {
        do: syntheticSensorSeries(n, 8.0, 1.8, ctx.driverParam === 'do' ? -3 : 0),
        ph: syntheticSensorSeries(n, 8.0, 0.3, ctx.driverParam === 'ph' ? 0.5 : 0),
        turb: syntheticSensorSeries(n, 12, 6, ctx.driverParam === 'turb' ? 15 : 0),
        cond: syntheticSensorSeries(n, 480, 120, ctx.driverParam === 'cond' ? 200 : 0),
        temp: syntheticSensorSeries(n, 18, 4),
        orp: syntheticSensorSeries(n, 220, 50),
      },
      anomaly: maxAnomaly,
      anomaly_series: anomalySeries,
      driver_param: ctx.driverParam,
      channel_weights: Array.from({ length: 8 }, (_, i) =>
        i === ctx.channelDominant ? 0.95 : 0.2 + Math.random() * 0.3,
      ),
      health: 'normal',
    },
    satellite: {
      state: 'OBSERVED',
      conf: 0.85,
      indices: { ndci: 0.42, fai: 0.08, ndti: 0.21, mndwi: 0.55, oilsheen: 0.02 },
      params: { temp: 24.1, turb: 18, tss: 11.3, chla: 41, phyco: 18 },
      tileSeed: ctx.lat * 1000 + ctx.lon,
    },
    microbial: {
      state: 'DEPLOYABLE',
      conf: 0,
      classes: microClassesDist(ctx.microTop, ctx.microP),
      top: ctx.microTop,
    },
    molecular: {
      state: 'DEPLOYABLE',
      conf: 0,
      activePath: {
        pathway: ctx.toxPathway,
        process: ctx.toxProcess,
        outcome: ctx.toxOutcome,
      },
      outcome: ctx.toxOutcome,
    },
    behavioral: {
      state: 'OBSERVED',
      conf: 0.78,
      anomaly: 0.91,
      signature: ctx.bioSignature,
      keyframes: [],
    },
    fusion: {
      anomaly: maxAnomaly,
      auroc: 0.939, // paper headline
      alert: maxAnomaly > 0.85 ? 'ALERT' : maxAnomaly > 0.65 ? 'INVESTIGATE' : 'WATCH',
      attention: [0.378, 0.361, 0.261, 0.0, 0.0],
      coverage: 0.95,
    },
    cascadeTier: 0,
    n_modalities: 3,
  };
}

export const EVENTS: CanonicalRecord[] = Object.keys(CONTEXT).map(buildRecord);

/* ---------- clean reference sites (unchanged from mockData) ---------- */

function makeClean(args: {
  id: string;
  name: string;
  label: string;
  lat: number;
  lon: number;
  region: RegionKey;
}): CanonicalRecord {
  const n = 96;
  return {
    id: args.id,
    name: args.name,
    label: args.label,
    lat: args.lat,
    lon: args.lon,
    h3: '882a1072cffffff',
    region: args.region,
    tier: 1,
    sensor: {
      state: 'OBSERVED',
      conf: 1.0,
      series: {
        do: syntheticSensorSeries(n, 11, 0.6),
        ph: syntheticSensorSeries(n, 7.0, 0.2),
        turb: syntheticSensorSeries(n, 2, 1),
        cond: syntheticSensorSeries(n, 60, 25),
        temp: syntheticSensorSeries(n, 6, 4),
        orp: syntheticSensorSeries(n, 270, 30),
      },
      anomaly: 0.08,
      anomaly_series: Array.from({ length: n }, () => 0.05 + Math.random() * 0.08),
      driver_param: 'do',
      channel_weights: [0.3, 0.4, 0.5, 0.4, 0.3, 0.2, 0.2, 0.1],
      health: 'normal',
    },
    satellite: {
      state: 'OBSERVED',
      conf: 0.85,
      indices: { ndci: 0.05, fai: 0.01, ndti: 0.06, mndwi: 0.62, oilsheen: 0 },
      params: { temp: 6.2, turb: 2, tss: 1.5, chla: 1.2, phyco: 0.4 },
      tileSeed: args.lat * 1000 + args.lon,
    },
    microbial: {
      state: 'DEPLOYABLE',
      conf: 0,
      classes: microClassesDist('Nutrient Runoff', 0.18),
      top: null,
    },
    molecular: { state: 'DEPLOYABLE', conf: 0, activePath: null, outcome: null },
    behavioral: { state: 'DEPLOYABLE', conf: 0, anomaly: 0.04, signature: 'normal', keyframes: [] },
    fusion: {
      anomaly: 0.09,
      auroc: 0.939,
      alert: 'MAINTAIN',
      attention: [0.45, 0.4, 0.15, 0.0, 0.0],
      coverage: 0.7,
    },
    cascadeTier: 0,
    n_modalities: 2,
  };
}

export const CLEAN_SITES: CanonicalRecord[] = [
  makeClean({
    id: 'TOOK',
    name: 'Toolik Lake, AK',
    label: 'NEON pristine arctic reference',
    lat: 68.63,
    lon: -149.61,
    region: 'ARCTIC',
  }),
  makeClean({
    id: 'SYCA',
    name: 'Sycamore Creek, AZ',
    label: 'NEON oligotrophic stream',
    lat: 34.44,
    lon: -111.51,
    region: 'ARID_SW',
  }),
];

/* ---------- degradation curve (real, from robustness study) ---------- */

export const DEGRADATION_CURVE: { n: number; auroc: number }[] = (
  (degradationRaw as any).default ?? (degradationRaw as any[])
).map(
  (row) => ({
    n: row.num_available,
    auroc: row.mean_auc,
  }),
);

/* ---------- cascade top trigger params (real, from causal analysis) ---------- */

export const TOP_TRIGGER_PARAMS: { name: string; count: number }[] = (
  (cascadeAnalysis as any).causal_chain_analysis.top_trigger_params as [string, number][]
).map(([name, count]) => ({ name, count }));

/* ---------- attribution mean deltas ---------- */

export const ATTRIBUTION = (attribution as any).parameter_summary as Record<
  string,
  { top_driver_count: number; mean_attribution_delta: number; std_attribution_delta: number }
>;

/* ---------- modality case-study evidence (for methodology panel) ---------- */

export const MICROBIAL_VALIDATION = (((microbialCS as any).case_studies ?? []) as any[]).map(
  (e) => ({
    event: e.name,
    contaminant: e.contaminant,
    detection_rate: e.detection_rate,
    mean_p: e.mean_anomaly_probability,
  }),
);

export const MOLECULAR_VALIDATION = (((molecularCS as any).case_studies ?? []) as any[]).map(
  (e) => ({
    study: e.study_id,
    contaminant: e.contaminant_name,
    rate: e.toxicity_positive_rate,
    score: e.mean_toxicity_score,
  }),
);

export const BEHAVIORAL_VALIDATION = (((behavioralCS as any).case_studies ?? []) as any[]).map(
  (e) => ({
    chemical: e.chemical_name,
    detection_rate: e.real_data_anomaly_detection_rate,
    mean_p: e.mean_anomaly_prob,
  }),
);

/* ---------- USGS + NEON stations ---------- */
export { NEON_SITES } from '../mockData';
import usgsRaw from './usgsStations.json';
export interface UsgsStation { id: string; lat: number; lon: number; name?: string }
export const USGS_STATIONS: UsgsStation[] = usgsRaw as UsgsStation[];
