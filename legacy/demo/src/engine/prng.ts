// Deterministic seeded RNG so repeated clicks / re-renders produce identical
// "noisy" series and screenshots stay reproducible. Mulberry32 — small, fast,
// and good enough for visual jitter (not cryptographic).

export function mulberry32(seed: number): () => number {
  let a = seed | 0;
  return function next() {
    a = (a + 0x6d2b79f5) | 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function seedFromLatLon(lat: number, lon: number): number {
  // 4-decimal granularity (~11m) so the same map click yields the same record.
  const s = `${lat.toFixed(4)}|${lon.toFixed(4)}`;
  let h = 2166136261;
  for (let i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return h >>> 0;
}
