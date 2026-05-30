// Live USGS NWIS fetcher. Public, no-key, CORS-enabled.
// Docs: https://waterservices.usgs.gov/docs/instantaneous-values/
import type { CanonicalRecord, SensorParam } from '../types';
import { USGS_STATIONS } from '../data/realData';
import { mulberry32, seedFromLatLon } from './prng';

// USGS parameter codes we map to our 6 internal series.
const PARAM_MAP: Record<string, SensorParam> = {
  '00010': 'temp',    // water temperature, °C
  '00400': 'ph',      // pH
  '00300': 'do',      // dissolved oxygen, mg/L
  '00095': 'cond',    // specific conductance, µS/cm
  '00076': 'turb',    // turbidity, NTU
  '63680': 'turb',    // turbidity (alt sensor)
};

const ALL_PARAMS = Array.from(new Set(Object.keys(PARAM_MAP))).join(',');

// Cache so repeated clicks on the same area don't re-fetch.
// Keyed at ~110m precision (4 decimals) so adjacent clicks don't collide,
// with a TTL so stale data eventually refreshes.
const CACHE_TTL_MS = 30 * 60 * 1000; // 30 minutes
const cache = new Map<string, { at: number; value: NwisResponse }>();

function cacheGet(key: string): NwisResponse | undefined {
  const hit = cache.get(key);
  if (!hit) return undefined;
  if (Date.now() - hit.at > CACHE_TTL_MS) {
    cache.delete(key);
    return undefined;
  }
  return hit.value;
}

function cacheSet(key: string, value: NwisResponse) {
  cache.set(key, { at: Date.now(), value });
}

interface NwisResponse {
  stationId: string;
  stationName: string;
  stationLat: number;
  stationLon: number;
  series: Partial<Record<SensorParam, number[]>>;
  anomaly: number;
  ok: boolean;
  reason?: string;
}

function distKm(a: { lat: number; lon: number }, b: { lat: number; lon: number }) {
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

function nearestStations(lat: number, lon: number, k = 8) {
  return [...USGS_STATIONS]
    .map((s) => ({ ...s, dist: distKm({ lat, lon }, { lat: s.lat, lon: s.lon }) }))
    .sort((a, b) => a.dist - b.dist)
    .slice(0, k);
}

// Pick whichever of the top-k candidate stations actually has recent iv data.
export async function fetchLiveSensor(lat: number, lon: number): Promise<NwisResponse> {
  const cacheKey = `${lat.toFixed(4)},${lon.toFixed(4)}`;
  const cached = cacheGet(cacheKey);
  if (cached) return cached;

  const candidates = nearestStations(lat, lon, 8);
  for (const c of candidates) {
    try {
      const url = `https://waterservices.usgs.gov/nwis/iv/?format=json&sites=${c.id}&parameterCd=${ALL_PARAMS}&period=P30D`;
      const r = await fetch(url);
      if (!r.ok) continue;
      const j = await r.json();
      const ts = j?.value?.timeSeries as any[] | undefined;
      if (!ts || ts.length === 0) continue;

      const series: Partial<Record<SensorParam, number[]>> = {};
      let total = 0;
      for (const block of ts) {
        const code = block?.variable?.variableCode?.[0]?.value;
        const ours = PARAM_MAP[code];
        if (!ours) continue;
        const points = block?.values?.[0]?.value as any[] | undefined;
        if (!points || points.length === 0) continue;
        const nums = points
          .map((p) => parseFloat(p.value))
          .filter((v) => Number.isFinite(v) && v > -999); // -999999 is USGS no-data
        if (nums.length === 0) continue;
        // Decimate to ~96 points to match our internal series length.
        const stride = Math.max(1, Math.floor(nums.length / 96));
        const decimated = nums.filter((_, i) => i % stride === 0).slice(-96);
        if (decimated.length === 0) continue;
        if (!series[ours] || decimated.length > (series[ours]?.length ?? 0)) {
          series[ours] = decimated;
        }
        total += decimated.length;
      }
      if (total === 0) continue;

      // Crude anomaly score: stdev-normalized recent deviation, averaged across params.
      // Tuned to land in 0–1; not a model output, but visually plausible.
      let agg = 0;
      let n = 0;
      for (const k of Object.keys(series) as SensorParam[]) {
        const v = series[k]!;
        if (v.length < 8) continue;
        const recent = v.slice(-8);
        const baseline = v.slice(0, Math.max(8, v.length - 8));
        const mean = baseline.reduce((s, x) => s + x, 0) / baseline.length;
        const std =
          Math.sqrt(
            baseline.reduce((s, x) => s + (x - mean) ** 2, 0) / Math.max(1, baseline.length - 1),
          ) || 1;
        const recentMean = recent.reduce((s, x) => s + x, 0) / recent.length;
        const z = Math.abs((recentMean - mean) / std);
        agg += Math.min(1, z / 3);
        n += 1;
      }
      const anomaly = n > 0 ? agg / n : 0.05;

      const result: NwisResponse = {
        stationId: c.id,
        stationName: c.name ?? c.id,
        stationLat: c.lat,
        stationLon: c.lon,
        series,
        anomaly,
        ok: true,
      };
      cacheSet(cacheKey, result);
      return result;
    } catch {
      continue;
    }
  }

  const empty: NwisResponse = {
    stationId: candidates[0]?.id ?? '—',
    stationName: candidates[0]?.name ?? 'No station',
    stationLat: candidates[0]?.lat ?? lat,
    stationLon: candidates[0]?.lon ?? lon,
    series: {},
    anomaly: 0,
    ok: false,
    reason: 'No recent instantaneous-value data within 8 nearest stations.',
  };
  cacheSet(cacheKey, empty);
  return empty;
}

// Build a patch to apply onto an existing synthesized record.
export function patchFromLive(rec: CanonicalRecord, live: NwisResponse): Partial<CanonicalRecord> {
  if (!live.ok) return {};
  const merged = { ...rec.sensor.series };
  (Object.keys(live.series) as SensorParam[]).forEach((k) => {
    const v = live.series[k];
    if (v && v.length > 0) merged[k] = v;
  });
  // Pick the driver param as the one with the largest recent deviation.
  let driver: SensorParam = rec.sensor.driver_param;
  let bestZ = 0;
  (Object.keys(live.series) as SensorParam[]).forEach((k) => {
    const v = live.series[k];
    if (!v || v.length < 8) return;
    const tail = v.slice(-8);
    const head = v.slice(0, -8);
    if (head.length === 0) return;
    const mean = head.reduce((s, x) => s + x, 0) / head.length;
    const std = Math.sqrt(head.reduce((s, x) => s + (x - mean) ** 2, 0) / Math.max(1, head.length - 1)) || 1;
    const tMean = tail.reduce((s, x) => s + x, 0) / tail.length;
    const z = Math.abs((tMean - mean) / std);
    if (z > bestZ) { bestZ = z; driver = k; }
  });

  return {
    id: live.stationId,
    name: `USGS ${live.stationId}`,
    label: live.stationName,
    lat: live.stationLat,
    lon: live.stationLon,
    sensor: {
      ...rec.sensor,
      state: 'OBSERVED',
      conf: 1.0,
      series: merged,
      anomaly: live.anomaly,
      anomaly_series: (() => {
        const rnd = mulberry32(seedFromLatLon(live.stationLat, live.stationLon));
        return rec.sensor.anomaly_series.map((_, i, arr) => {
          const t = i / (arr.length - 1);
          return 0.1 + t * t * (live.anomaly - 0.1) + (rnd() - 0.5) * 0.03;
        });
      })(),
      driver_param: driver,
    },
    fusion: {
      ...rec.fusion,
      anomaly: Math.max(rec.fusion.anomaly, live.anomaly * 0.95),
      alert:
        live.anomaly > 0.85 ? 'ALERT'
        : live.anomaly > 0.65 ? 'INVESTIGATE'
        : live.anomaly > 0.4 ? 'WATCH'
        : 'MAINTAIN',
    },
  };
}
