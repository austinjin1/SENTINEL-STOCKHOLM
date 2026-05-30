import type { CanonicalRecord, RegionKey, SensorParam } from '../types';
import { USGS_STATIONS, NEON_SITES, EVENTS, CLEAN_SITES } from '../data/realData';
import { latLngToCell } from 'h3-js';
import { mulberry32, seedFromLatLon } from './prng';

// §5.2 — region boxes
type Box = { region: RegionKey; lat: [number, number]; lon: [number, number] };
const BOXES: Box[] = [
  { region: 'GREAT_LAKES', lat: [41, 48], lon: [-93, -76] },
  { region: 'CORN_BELT', lat: [38, 45], lon: [-98, -84] },
  { region: 'GULF_COAST', lat: [25, 31], lon: [-98, -81] },
  { region: 'ARID_SW', lat: [31, 40], lon: [-120, -103] },
  { region: 'PAC_NW', lat: [42, 49], lon: [-125, -116] },
  { region: 'NORTHEAST', lat: [38, 43], lon: [-80, -69] },
  { region: 'SOUTHEAST', lat: [30, 38], lon: [-90, -76] },
  { region: 'MOUNTAIN', lat: [37, 45], lon: [-114, -104] },
  { region: 'ARCTIC', lat: [60, 72], lon: [-165, -140] },
];

export function classifyRegion(lat: number, lon: number): RegionKey {
  for (const b of BOXES) {
    if (lat >= b.lat[0] && lat <= b.lat[1] && lon >= b.lon[0] && lon <= b.lon[1]) {
      return b.region;
    }
  }
  return 'DEFAULT';
}

// §5.3 — region sensor baselines [mean, spread]
type ParamBaseline = Record<SensorParam, [number, number]>;
const BASELINES: Record<RegionKey, ParamBaseline> = {
  CORN_BELT: { do: [7.2, 1.5], ph: [8.0, 0.3], turb: [18, 10], cond: [720, 180], temp: [16, 8], orp: [180, 60] },
  ARID_SW: { do: [8.0, 1.0], ph: [8.2, 0.2], turb: [6, 4], cond: [950, 250], temp: [18, 9], orp: [220, 50] },
  PAC_NW: { do: [10.5, 0.8], ph: [7.4, 0.2], turb: [3, 2], cond: [90, 40], temp: [11, 5], orp: [250, 40] },
  GREAT_LAKES: { do: [9.0, 1.2], ph: [8.1, 0.3], turb: [9, 6], cond: [300, 120], temp: [14, 9], orp: [210, 55] },
  ARCTIC: { do: [11.5, 0.6], ph: [7.0, 0.2], turb: [2, 1], cond: [60, 25], temp: [6, 4], orp: [270, 30] },
  GULF_COAST: { do: [6.8, 1.3], ph: [8.0, 0.2], turb: [14, 8], cond: [2200, 600], temp: [23, 5], orp: [180, 50] },
  NORTHEAST: { do: [8.5, 1.2], ph: [7.6, 0.3], turb: [11, 6], cond: [400, 150], temp: [13, 8], orp: [200, 50] },
  SOUTHEAST: { do: [7.5, 1.4], ph: [7.4, 0.3], turb: [15, 8], cond: [340, 130], temp: [21, 6], orp: [190, 55] },
  MOUNTAIN: { do: [10.0, 0.9], ph: [7.8, 0.2], turb: [4, 3], cond: [200, 80], temp: [10, 5], orp: [240, 40] },
  DEFAULT: { do: [8.5, 1.5], ph: [7.7, 0.3], turb: [10, 6], cond: [500, 200], temp: [14, 8], orp: [200, 60] },
};

function gen(n: number, mean: number, spread: number, seed: number) {
  const out: number[] = [];
  let s = seed;
  for (let i = 0; i < n; i++) {
    s = (s * 1664525 + 1013904223) % 4294967296;
    const u = s / 4294967296;
    out.push(mean + Math.sin(i * 0.4) * spread * 0.3 + (u - 0.5) * spread * 0.6);
  }
  return out;
}

function distKm(a: { lat: number; lon: number }, b: { lat: number; lon: number }): number {
  const R = 6371;
  const dLat = ((b.lat - a.lat) * Math.PI) / 180;
  const dLon = ((b.lon - a.lon) * Math.PI) / 180;
  const x =
    Math.sin(dLat / 2) ** 2 +
    Math.cos((a.lat * Math.PI) / 180) *
      Math.cos((b.lat * Math.PI) / 180) *
      Math.sin(dLon / 2) ** 2;
  return 2 * R * Math.asin(Math.min(1, Math.sqrt(x)));
}

// Real H3 cell at resolution 8 (~0.7km edge) — coarse enough to dedupe nearby
// clicks, fine enough that distinct sites get distinct cells.
function h3Cell(lat: number, lon: number): string {
  return latLngToCell(lat, lon, 8);
}

// nearest reference (USGS station or NEON / event site)
function nearestStation(lat: number, lon: number) {
  let best: { id: string; lat: number; lon: number; dist: number; isEvent: boolean } | null = null;
  const all = [
    ...USGS_STATIONS.map((s) => ({ ...s, isEvent: false })),
    ...NEON_SITES.map((s) => ({ id: s.id, lat: s.lat, lon: s.lon, isEvent: false })),
    ...EVENTS.map((e) => ({ id: e.id!, lat: e.lat, lon: e.lon, isEvent: true })),
    ...CLEAN_SITES.map((e) => ({ id: e.id!, lat: e.lat, lon: e.lon, isEvent: true })),
  ];
  for (const s of all) {
    const d = distKm({ lat, lon }, { lat: s.lat, lon: s.lon });
    if (!best || d < best.dist) {
      best = { id: s.id, lat: s.lat, lon: s.lon, dist: d, isEvent: s.isEvent };
    }
  }
  return best!;
}

// Per-cell memo so repeat clicks return the *same* record (stable identity).
const resolveCache = new Map<string, CanonicalRecord>();

export function resolve(lat: number, lon: number): CanonicalRecord {
  const cell = h3Cell(lat, lon);
  const cached = resolveCache.get(cell);
  if (cached) return cached;

  const region = classifyRegion(lat, lon);
  const near = nearestStation(lat, lon);
  const isObserved = near.dist <= 5;
  const conf = Math.max(0, Math.min(1, Math.exp(-near.dist / 25)));
  const b = BASELINES[region];
  const seed = Math.floor((lat + 90) * 1000 + (lon + 180) * 1000);
  const n = 96;
  const series = {
    do: gen(n, b.do[0], b.do[1], seed + 1),
    ph: gen(n, b.ph[0], b.ph[1], seed + 2),
    turb: gen(n, b.turb[0], b.turb[1], seed + 3),
    cond: gen(n, b.cond[0], b.cond[1], seed + 4),
    temp: gen(n, b.temp[0], b.temp[1], seed + 5),
    orp: gen(n, b.orp[0], b.orp[1], seed + 6),
  };
  const rnd = mulberry32(seedFromLatLon(lat, lon));
  const anomalySeries = Array.from({ length: n }, () => 0.04 + rnd() * 0.08);
  const anomaly = anomalySeries[anomalySeries.length - 1];

  const fusionAlert: 'MAINTAIN' | 'WATCH' = anomaly > 0.25 ? 'WATCH' : 'MAINTAIN';

  const record: CanonicalRecord = {
    name: `Resolved cell · ${region}`,
    label: `nearest reference ${near.id} · ${near.dist.toFixed(1)} km`,
    lat,
    lon,
    h3: cell,
    region,
    sensor: {
      state: isObserved ? 'OBSERVED' : 'INFERRED',
      conf: isObserved ? 1.0 : conf,
      series,
      anomaly,
      anomaly_series: anomalySeries,
      driver_param: 'do',
      channel_weights: [0.3, 0.4, 0.5, 0.4, 0.3, 0.2, 0.2, 0.1],
      health: 'normal',
    },
    satellite: {
      state: 'OBSERVED',
      conf: 0.85,
      indices: { ndci: 0.1, fai: 0.02, ndti: 0.08, mndwi: 0.5, oilsheen: 0 },
      params: { temp: b.temp[0], turb: b.turb[0], tss: b.turb[0] * 0.6, chla: 4, phyco: 1 },
      tileSeed: lat * 1000 + lon,
    },
    microbial: {
      state: 'DEPLOYABLE',
      conf: 0,
      classes: [],
      top: null,
    },
    molecular: { state: 'DEPLOYABLE', conf: 0, activePath: null, outcome: null },
    behavioral: { state: 'DEPLOYABLE', conf: 0, anomaly: 0.05, signature: 'normal', keyframes: [] },
    fusion: {
      anomaly: anomaly * 0.9,
      auroc: 0.939,
      alert: fusionAlert,
      attention: [0.55, 0.45, 0, 0, 0],
      coverage: isObserved ? 0.9 : 0.6 + conf * 0.2,
    },
    cascadeTier: 0,
    n_modalities: 2,
  };
  resolveCache.set(cell, record);
  return record;
}
