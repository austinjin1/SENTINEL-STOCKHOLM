import type { CanonicalRecord, SensorParam } from '../types';
import { classifyRegion } from './resolve';
import { latLngToCell } from 'h3-js';
import { mulberry32, seedFromLatLon } from './prng';
import type { ParsedCsv } from './csvParse';
export { parseCsv, type ParsedCsv } from './csvParse';

export function buildRecordFromCsv(p: ParsedCsv): CanonicalRecord {
  const region = classifyRegion(p.lat, p.lon);
  const n = 96;

  // Track whether any required column was missing — drives sensor.state below.
  const EXPECTED: SensorParam[] = ['do', 'ph', 'turb', 'cond', 'temp', 'orp'];
  const missing = EXPECTED.filter((k) => !p.series[k] || p.series[k]!.length === 0);

  const rnd = mulberry32(seedFromLatLon(p.lat, p.lon));

  // Take the last n samples from each parsed series; if missing, generate gentle synth.
  function ensure(key: SensorParam, fallbackMean: number, spread: number): number[] {
    const v = p.series[key];
    if (v && v.length > 0) {
      const stride = Math.max(1, Math.floor(v.length / n));
      return v.filter((_, i) => i % stride === 0).slice(-n);
    }
    return Array.from({ length: n }, () => fallbackMean + (rnd() - 0.5) * spread);
  }

  const series = {
    do: ensure('do', 8.5, 1.5),
    ph: ensure('ph', 7.6, 0.3),
    turb: ensure('turb', 10, 6),
    cond: ensure('cond', 500, 200),
    temp: ensure('temp', 14, 6),
    orp: ensure('orp', 200, 60),
  };

  // Anomaly from recent deviation
  let agg = 0;
  let count = 0;
  (Object.keys(series) as SensorParam[]).forEach((k) => {
    if (!p.series[k]) return;
    const v = series[k];
    const head = v.slice(0, -8);
    const tail = v.slice(-8);
    if (head.length === 0) return;
    const mean = head.reduce((s, x) => s + x, 0) / head.length;
    const std =
      Math.sqrt(head.reduce((s, x) => s + (x - mean) ** 2, 0) / Math.max(1, head.length - 1)) || 1;
    const tMean = tail.reduce((s, x) => s + x, 0) / tail.length;
    agg += Math.min(1, Math.abs((tMean - mean) / std) / 3);
    count += 1;
  });
  const anomaly = count > 0 ? agg / count : 0.1;

  return {
    id: `sim-${Math.floor(rnd() * 1e6)}`,
    name: p.name,
    label: 'User-imported simulated event',
    lat: p.lat,
    lon: p.lon,
    h3: latLngToCell(p.lat, p.lon, 8),
    region,
    bookmarked: false,
    sensor: {
      // Downgrade to INFERRED when any expected column was imputed from synth defaults,
      // so the UI/user knows part of the series is fabricated.
      state: missing.length === 0 ? 'OBSERVED' : 'INFERRED',
      conf: missing.length === 0 ? 1.0 : Math.max(0.4, 1 - missing.length * 0.12),
      series,
      anomaly,
      anomaly_series: Array.from({ length: n }, (_, i) => {
        const t = i / (n - 1);
        return 0.08 + t * t * (anomaly - 0.08) + (rnd() - 0.5) * 0.03;
      }),
      driver_param: 'do',
      channel_weights: [0.3, 0.4, 0.5, 0.4, 0.3, 0.2, 0.2, 0.1],
      health: 'normal',
    },
    satellite: {
      state: 'INFERRED',
      conf: 0.6,
      indices: { ndci: 0.15, fai: 0.05, ndti: 0.1, mndwi: 0.5 },
      params: { temp: 14, turb: 10, tss: 6, chla: 4, phyco: 1 },
      tileSeed: p.lat * 1000 + p.lon,
    },
    microbial: { state: 'DEPLOYABLE', conf: 0, classes: [], top: null },
    molecular: { state: 'DEPLOYABLE', conf: 0, activePath: null, outcome: null },
    behavioral: { state: 'DEPLOYABLE', conf: 0, anomaly: 0.05, signature: 'normal', keyframes: [] },
    fusion: {
      anomaly: anomaly * 0.95,
      auroc: 0.939,
      alert:
        anomaly > 0.85 ? 'ALERT'
        : anomaly > 0.65 ? 'INVESTIGATE'
        : anomaly > 0.4 ? 'WATCH'
        : 'MAINTAIN',
      attention: [0.55, 0.0, 0.45, 0.0, 0.0],
      coverage: 0.6,
    },
    cascadeTier: 0,
    n_modalities: 1,
  };
}
