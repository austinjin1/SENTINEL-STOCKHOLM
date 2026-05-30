// Time-aware sampling for the "Play story" feature.
//
// Sensor / anomaly series are stored as fixed-length number[] anchored at an event's
// advisory date — index 0 = oldest sample, last index = sample at advisory. To play
// them back chronologically we need to map a `currentDate` to the right index.

function parse(iso: string): number {
  return new Date(iso + 'T00:00:00Z').getTime();
}

const DAY_MS = 86400000;

/**
 * Return the value at `currentDate` from a series anchored at [windowStart, anchorDate].
 * Linearly interpolates between samples; clamps to the series ends if `currentDate`
 * falls outside the window.
 */
export function sampleSeriesAt(
  series: readonly number[],
  windowStart: string,
  anchor: string,
  currentDate: string,
): number {
  if (series.length === 0) return 0;
  const t0 = parse(windowStart);
  const t1 = parse(anchor);
  const t = parse(currentDate);
  if (t1 <= t0) return series[series.length - 1];
  if (t <= t0) return series[0];
  if (t >= t1) return series[series.length - 1];
  const frac = (t - t0) / (t1 - t0); // 0..1
  const idx = frac * (series.length - 1);
  const lo = Math.floor(idx);
  const hi = Math.min(series.length - 1, lo + 1);
  const w = idx - lo;
  return series[lo] * (1 - w) + series[hi] * w;
}

/**
 * Return a copy of `series` where every sample whose index falls past `currentDate`
 * is replaced with NaN — sparkline renderers treat NaN as a gap, which is exactly
 * the "future hidden" effect we want during playback.
 */
export function truncateAt(
  series: readonly number[],
  windowStart: string,
  anchor: string,
  currentDate: string,
): number[] {
  if (series.length === 0) return [];
  const t0 = parse(windowStart);
  const t1 = parse(anchor);
  const t = parse(currentDate);
  if (t1 <= t0 || t >= t1) return series.slice();
  if (t <= t0) return series.map(() => NaN);
  const frac = (t - t0) / (t1 - t0);
  const cutoff = Math.floor(frac * (series.length - 1));
  return series.map((v, i) => (i <= cutoff ? v : NaN));
}

/** Days between two ISO dates (b - a). */
export function daysBetween(a: string, b: string): number {
  return Math.round((parse(b) - parse(a)) / DAY_MS);
}
