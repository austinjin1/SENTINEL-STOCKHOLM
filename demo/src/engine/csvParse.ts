// Pure CSV parser, no dependencies on realData/resolve so it can run in a worker
// without dragging the full station catalog into the worker bundle.
import type { SensorParam } from '../types';

const HEADER_ALIASES: Record<string, SensorParam | 'timestamp'> = {
  timestamp: 'timestamp', time: 'timestamp', datetime: 'timestamp', date: 'timestamp',
  do: 'do', dissolved_oxygen: 'do', do_mgl: 'do',
  ph: 'ph',
  turb: 'turb', turbidity: 'turb', turbidity_ntu: 'turb',
  cond: 'cond', conductivity: 'cond', spc: 'cond', specific_conductance: 'cond',
  temp: 'temp', temperature: 'temp', water_temp: 'temp', temperature_c: 'temp',
  orp: 'orp',
};

export interface ParsedCsv {
  lat: number;
  lon: number;
  series: Partial<Record<SensorParam, number[]>>;
  rows: number;
  name: string;
}

export function parseCsv(text: string, opts: { lat: number; lon: number; name: string }): ParsedCsv {
  const lines = text.split(/\r?\n/).filter((l) => l.trim().length > 0);
  if (lines.length === 0) throw new Error('Empty CSV.');
  const headerRow = lines[0].split(/[,\t]/).map((c) => c.trim().toLowerCase());
  const mapping: (SensorParam | 'timestamp' | null)[] = headerRow.map(
    (h) => HEADER_ALIASES[h] ?? null,
  );

  const series: Partial<Record<SensorParam, number[]>> = {};
  for (let i = 1; i < lines.length; i++) {
    const cells = lines[i].split(/[,\t]/).map((c) => c.trim());
    for (let j = 0; j < cells.length; j++) {
      const m = mapping[j];
      if (!m || m === 'timestamp') continue;
      const v = parseFloat(cells[j]);
      if (!Number.isFinite(v)) continue;
      if (!series[m]) series[m] = [];
      series[m]!.push(v);
    }
  }
  if (Object.keys(series).length === 0) {
    throw new Error(
      'No recognized columns. Expected any of: timestamp, DO, pH, turbidity, conductivity, temperature, ORP.',
    );
  }
  return { lat: opts.lat, lon: opts.lon, series, rows: lines.length - 1, name: opts.name };
}
