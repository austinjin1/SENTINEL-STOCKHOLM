import type { CanonicalRecord } from '../types';

interface Props {
  record: CanonicalRecord | null;
  cascadeTier: number;
}

const REGION_LABEL: Record<string, string> = {
  GREAT_LAKES: 'Great Lakes',
  CORN_BELT: 'Corn Belt',
  GULF_COAST: 'Gulf Coast',
  ARID_SW: 'Arid Southwest',
  PAC_NW: 'Pacific Northwest',
  NORTHEAST: 'Northeast',
  SOUTHEAST: 'Southeast',
  MOUNTAIN: 'Mountain',
  ARCTIC: 'Arctic',
  DEFAULT: 'Continental',
};

export function BottomTelemetry({ record, cascadeTier }: Props) {
  if (!record) {
    return (
      <div
        style={{
          height: 32,
          background: 'var(--bg-panel)',
          borderTop: '1px solid var(--border-subtle)',
          display: 'flex',
          alignItems: 'center',
          padding: '0 20px',
          fontSize: 12,
          color: 'var(--text-tertiary)',
          flexShrink: 0,
        }}
      >
        Awaiting selection
      </div>
    );
  }
  const items: { label: string; value: string; mono?: boolean }[] = [
    { label: 'Lat', value: record.lat.toFixed(4), mono: true },
    { label: 'Lon', value: record.lon.toFixed(4), mono: true },
    { label: 'H3', value: record.h3, mono: true },
    { label: 'Region', value: REGION_LABEL[record.region] ?? record.region },
    { label: 'Tier', value: String(cascadeTier) },
    { label: 'Modalities', value: `${record.n_modalities}/5` },
    { label: 'Confidence', value: record.fusion.coverage.toFixed(2), mono: true },
  ];
  return (
    <div
      style={{
        height: 32,
        background: 'var(--bg-panel)',
        borderTop: '1px solid var(--border-subtle)',
        display: 'flex',
        alignItems: 'center',
        padding: '0 20px',
        gap: 24,
        fontSize: 12,
        color: 'var(--text-secondary)',
        flexShrink: 0,
      }}
    >
      {items.map((it) => (
        <span key={it.label} style={{ display: 'flex', gap: 6 }}>
          <span style={{ color: 'var(--text-tertiary)' }}>{it.label}</span>
          <span
            className={it.mono ? 'mono' : undefined}
            style={{ color: 'var(--text-primary)' }}
          >
            {it.value}
          </span>
        </span>
      ))}
    </div>
  );
}
