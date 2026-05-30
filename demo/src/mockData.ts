import type { CanonicalRecord, MicroClass } from './types';

// Hardcoded placeholders matching spec §18.2 — will be replaced with repo JSON in Stage 0.

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

function microClasses(topName: MicroClass, topP: number) {
  const remaining = 1 - topP;
  const others = MICRO_CLASSES.filter((c) => c !== topName);
  return MICRO_CLASSES.map((name) => {
    if (name === topName) return { name, p: topP };
    const idx = others.indexOf(name);
    // distribute remaining as decay
    const weights = others.map((_, i) => Math.exp(-i * 0.6));
    const sum = weights.reduce((a, b) => a + b, 0);
    return { name, p: (remaining * weights[idx]) / sum };
  });
}

function syntheticSeries(n: number, mean: number, spread: number, trend = 0): number[] {
  const out: number[] = [];
  for (let i = 0; i < n; i++) {
    const t = i / (n - 1);
    out.push(mean + Math.sin(i * 0.4) * spread * 0.3 + (Math.random() - 0.5) * spread * 0.4 + trend * t);
  }
  return out;
}

function risingAnomaly(n: number, peak: number): number[] {
  const out: number[] = [];
  for (let i = 0; i < n; i++) {
    const t = i / (n - 1);
    const base = 0.15 + t * t * (peak - 0.15);
    out.push(Math.min(1, base + (Math.random() - 0.5) * 0.04));
  }
  return out;
}

function makeBookmarked(args: {
  id: string;
  name: string;
  label: string;
  lat: number;
  lon: number;
  region: any;
  tier: 1 | 2 | 3 | 4 | 5;
  leadDays: number;
  fusionAnomaly: number;
  microTop: MicroClass;
  microP: number;
  toxOutcome: any;
  toxPathway: any;
  toxProcess: any;
  bioSignature: 'convulsion' | 'immobility' | 'normal';
  driverParam: any;
  channelDominant: number;
}): CanonicalRecord {
  const n = 96;
  return {
    id: args.id,
    name: args.name,
    label: args.label,
    lat: args.lat,
    lon: args.lon,
    h3: '882a1072cffffff', // placeholder
    region: args.region,
    bookmarked: true,
    tier: args.tier,
    leadDays: args.leadDays,
    sensor: {
      state: 'OBSERVED',
      conf: 1.0,
      series: {
        do: syntheticSeries(n, 8.0, 1.8, args.driverParam === 'do' ? -3 : 0),
        ph: syntheticSeries(n, 8.0, 0.3, args.driverParam === 'ph' ? 0.5 : 0),
        turb: syntheticSeries(n, 12, 6, args.driverParam === 'turb' ? 15 : 0),
        cond: syntheticSeries(n, 480, 120, args.driverParam === 'cond' ? 200 : 0),
        temp: syntheticSeries(n, 18, 4),
        orp: syntheticSeries(n, 220, 50),
      },
      anomaly: args.fusionAnomaly,
      anomaly_series: risingAnomaly(n, args.fusionAnomaly),
      driver_param: args.driverParam,
      channel_weights: Array.from({ length: 8 }, (_, i) =>
        i === args.channelDominant ? 0.95 : 0.2 + Math.random() * 0.3,
      ),
      health: 'normal',
    },
    satellite: {
      state: 'OBSERVED',
      conf: 0.85,
      indices: { ndci: 0.42, fai: 0.08, ndti: 0.21, mndwi: 0.55, oilsheen: 0.02 },
      params: { temp: 24.1, turb: 18, tss: 11.3, chla: 41, phyco: 18 },
      tileSeed: args.lat * 1000 + args.lon,
    },
    microbial: {
      state: 'DEPLOYABLE',
      conf: 0,
      classes: microClasses(args.microTop, args.microP),
      top: args.microTop,
    },
    molecular: {
      state: 'DEPLOYABLE',
      conf: 0,
      activePath: { pathway: args.toxPathway, process: args.toxProcess, outcome: args.toxOutcome },
      outcome: args.toxOutcome,
    },
    behavioral: {
      state: 'OBSERVED',
      conf: 0.78,
      anomaly: 0.91,
      signature: args.bioSignature,
      keyframes: [],
    },
    fusion: {
      anomaly: args.fusionAnomaly,
      auroc: 0.939, // paper-canonical value
      alert: args.fusionAnomaly > 0.85 ? 'ALERT' : args.fusionAnomaly > 0.65 ? 'INVESTIGATE' : 'WATCH',
      attention: [0.378, 0.361, 0.261, 0.0, 0.0],
      coverage: 0.95,
    },
    cascadeTier: 0,
    n_modalities: 3,
  };
}

function makeClean(args: {
  id: string;
  name: string;
  label: string;
  lat: number;
  lon: number;
  region: any;
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
        do: syntheticSeries(n, 11, 0.6),
        ph: syntheticSeries(n, 7.0, 0.2),
        turb: syntheticSeries(n, 2, 1),
        cond: syntheticSeries(n, 60, 25),
        temp: syntheticSeries(n, 6, 4),
        orp: syntheticSeries(n, 270, 30),
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
    microbial: { state: 'DEPLOYABLE', conf: 0, classes: microClasses('Nutrient Runoff', 0.18), top: null },
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

export const EVENTS: CanonicalRecord[] = [
  makeBookmarked({
    id: '04199500',
    name: 'Lake Erie HAB 2023',
    label: 'Harmful algal bloom · Maumee Bay',
    lat: 41.73,
    lon: -83.47,
    region: 'GREAT_LAKES',
    tier: 5,
    leadDays: 59.3,
    fusionAnomaly: 0.997,
    microTop: 'Nutrient Runoff',
    microP: 0.94,
    toxOutcome: 'Hepatotoxicity',
    toxPathway: 'AHR/CYP1A',
    toxProcess: 'Xenobiotic Metabolism',
    bioSignature: 'convulsion',
    driverParam: 'do',
    channelDominant: 4, // 7d
  }),
  makeBookmarked({
    id: '07374000',
    name: 'Gulf Dead Zone 2023',
    label: 'Hypoxic zone · Mississippi outflow',
    lat: 30.45,
    lon: -91.19,
    region: 'GULF_COAST',
    tier: 5,
    leadDays: 87.2,
    fusionAnomaly: 0.993,
    microTop: 'Nutrient Runoff',
    microP: 0.88,
    toxOutcome: 'Hepatotoxicity',
    toxPathway: 'Oxidative Stress',
    toxProcess: 'Oxidative Response',
    bioSignature: 'immobility',
    driverParam: 'do',
    channelDominant: 5, // 30d
  }),
  makeBookmarked({
    id: '01589485',
    name: 'Chesapeake Hypoxia 2018',
    label: 'Hypoxic intrusion · Patapsco',
    lat: 39.21,
    lon: -76.71,
    region: 'NORTHEAST',
    tier: 5,
    leadDays: 89.8,
    fusionAnomaly: 0.999,
    microTop: 'Sewage',
    microP: 0.71,
    toxOutcome: 'Hepatotoxicity',
    toxPathway: 'AHR/CYP1A',
    toxProcess: 'Xenobiotic Metabolism',
    bioSignature: 'immobility',
    driverParam: 'do',
    channelDominant: 5,
  }),
  makeBookmarked({
    id: '11530500',
    name: 'Klamath River HAB 2021',
    label: 'Cyanobacterial bloom',
    lat: 41.51,
    lon: -124.04,
    region: 'PAC_NW',
    tier: 4,
    leadDays: 59.2,
    fusionAnomaly: 0.993,
    microTop: 'Nutrient Runoff',
    microP: 0.82,
    toxOutcome: 'Hepatotoxicity',
    toxPathway: 'AHR/CYP1A',
    toxProcess: 'Xenobiotic Metabolism',
    bioSignature: 'convulsion',
    driverParam: 'ph',
    channelDominant: 4,
  }),
  makeBookmarked({
    id: '07374000b',
    name: 'Mississippi Salinity 2023',
    label: 'Saltwater wedge intrusion',
    lat: 30.45,
    lon: -91.19,
    region: 'GULF_COAST',
    tier: 4,
    leadDays: 58.6,
    fusionAnomaly: 0.993,
    microTop: 'Sediment',
    microP: 0.66,
    toxOutcome: 'Endocrine Disruption',
    toxPathway: 'Estrogen/Endocrine',
    toxProcess: 'Endocrine Signaling',
    bioSignature: 'normal',
    driverParam: 'cond',
    channelDominant: 3, // 2d
  }),
  makeBookmarked({
    id: '02101726',
    name: 'Jordan Lake HAB 2022',
    label: 'Reservoir cyanobacterial bloom',
    lat: 35.75,
    lon: -79.06,
    region: 'SOUTHEAST',
    tier: 4,
    leadDays: 44.3,
    fusionAnomaly: 0.993,
    microTop: 'Nutrient Runoff',
    microP: 0.79,
    toxOutcome: 'Hepatotoxicity',
    toxPathway: 'AHR/CYP1A',
    toxProcess: 'Xenobiotic Metabolism',
    bioSignature: 'convulsion',
    driverParam: 'ph',
    channelDominant: 4,
  }),
];

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

// NEON tier sites (spec §18.3)
export interface NeonSite {
  id: string;
  name: string;
  lat: number;
  lon: number;
  tier: 1 | 2 | 3 | 4 | 5;
}

export const NEON_SITES: NeonSite[] = [
  { id: 'BARC', name: 'Barco Lake, FL', lat: 29.68, lon: -82.01, tier: 5 },
  { id: 'SUGG', name: 'Suggs Lake, FL', lat: 29.69, lon: -82.02, tier: 5 },
  { id: 'PRPO', name: 'Prairie Pothole, MN', lat: 47.13, lon: -99.25, tier: 5 },
  { id: 'MAYF', name: 'Mayfield Creek, AL', lat: 32.96, lon: -87.41, tier: 4 },
  { id: 'MCDI', name: 'McDiffett Creek, KS', lat: 38.95, lon: -96.44, tier: 4 },
  { id: 'PRIN', name: 'Pringle Creek, TX', lat: 33.38, lon: -97.78, tier: 4 },
  { id: 'TOMB', name: 'Lower Tombigbee, AL', lat: 31.85, lon: -88.16, tier: 2 },
  { id: 'WALK', name: 'Walker Branch, TN', lat: 35.96, lon: -84.28, tier: 2 },
  { id: 'TOOK', name: 'Toolik Lake, AK', lat: 68.63, lon: -149.61, tier: 1 },
  { id: 'SYCA', name: 'Sycamore Creek, AZ', lat: 34.44, lon: -111.51, tier: 1 },
  // remaining 22 → tier 3
  { id: 'ARIK', name: 'Arikaree, CO', lat: 39.76, lon: -102.45, tier: 3 },
  { id: 'BIGC', name: 'Big Cypress, FL', lat: 26.0, lon: -81.0, tier: 3 },
  { id: 'BLDE', name: 'Blacktail Deer, WY', lat: 44.95, lon: -110.59, tier: 3 },
  { id: 'BLUE', name: 'Blue River, OK', lat: 34.44, lon: -96.62, tier: 3 },
  { id: 'BLWA', name: 'Black Warrior, AL', lat: 32.54, lon: -87.8, tier: 3 },
  { id: 'CARI', name: 'Caribou Creek, AK', lat: 65.15, lon: -147.5, tier: 3 },
  { id: 'COMO', name: 'Como Creek, CO', lat: 40.04, lon: -105.54, tier: 3 },
  { id: 'CRAM', name: 'Crampton Lake, WI', lat: 46.21, lon: -89.47, tier: 3 },
  { id: 'CUPE', name: 'Rio Cupeyes, PR', lat: 18.11, lon: -66.99, tier: 3 },
  { id: 'FLNT', name: 'Flint River, GA', lat: 31.19, lon: -84.44, tier: 3 },
  { id: 'GUIL', name: 'Guilford Creek, MD', lat: 39.32, lon: -75.96, tier: 3 },
  { id: 'HOPB', name: 'Hop Brook, MA', lat: 42.47, lon: -72.33, tier: 3 },
  { id: 'KING', name: 'Kings Creek, KS', lat: 39.1, lon: -96.6, tier: 3 },
  { id: 'LECO', name: 'LeConte Creek, TN', lat: 35.69, lon: -83.5, tier: 3 },
  { id: 'LEWI', name: 'Lewis Run, VA', lat: 39.09, lon: -77.98, tier: 3 },
  { id: 'LIRO', name: 'Little Rock Lake, WI', lat: 45.99, lon: -89.7, tier: 3 },
  { id: 'MART', name: 'Martha Creek, WA', lat: 45.79, lon: -121.93, tier: 3 },
  { id: 'OKSR', name: 'Oksrukuyik, AK', lat: 68.67, lon: -149.14, tier: 3 },
  { id: 'POSE', name: 'Posey Creek, VA', lat: 38.89, lon: -78.15, tier: 3 },
  { id: 'REDB', name: 'Red Butte Creek, UT', lat: 40.78, lon: -111.79, tier: 3 },
  { id: 'WLOU', name: 'West St Louis Creek, CO', lat: 39.89, lon: -105.92, tier: 3 },
];

// Representative USGS scatter (~60 stations)
export interface UsgsStation {
  id: string;
  lat: number;
  lon: number;
}

export const USGS_STATIONS: UsgsStation[] = (() => {
  const rng = (seed: number) => {
    let s = seed;
    return () => {
      s = (s * 1664525 + 1013904223) % 4294967296;
      return s / 4294967296;
    };
  };
  const r = rng(42);
  const out: UsgsStation[] = [];
  // CONUS rough bbox
  for (let i = 0; i < 80; i++) {
    const lon = -125 + r() * 58;
    const lat = 26 + r() * 22;
    out.push({ id: `USGS-${(1000000 + i).toString()}`, lat, lon });
  }
  return out;
})();
