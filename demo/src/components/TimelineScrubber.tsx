import { useState } from 'react';
import { Play } from 'lucide-react';
import type { CanonicalRecord } from '../types';

interface Props {
  record: CanonicalRecord;
}

const TICKS = [-60, -45, -30, -14, -7, -3, 0];

export function TimelineScrubber({ record }: Props) {
  const [pos, setPos] = useState(record.leadDays ? -record.leadDays : 0);
  const lead = -pos;
  return (
    <div
      style={{
        padding: '14px 20px',
        borderTop: '1px solid var(--border-subtle)',
        position: 'sticky',
        bottom: 0,
        background: 'var(--bg-panel)',
      }}
    >
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline' }}>
        <div style={{ fontSize: 13, fontWeight: 600, color: 'var(--text-primary)' }}>Timeline</div>
        <div style={{ fontSize: 12, color: 'var(--accent-bright)' }}>
          Detected <span className="mono">{lead.toFixed(1)}d</span> early
        </div>
      </div>
      <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginTop: 10 }}>
        <button
          title="Play"
          style={{
            width: 28,
            height: 28,
            border: '1px solid var(--border-subtle)',
            background: 'var(--bg-elevated)',
            color: 'var(--text-secondary)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            borderRadius: 'var(--r-md)',
          }}
        >
          <Play size={12} strokeWidth={1.5} />
        </button>
        <div style={{ flex: 1, position: 'relative', height: 32 }}>
          <input
            type="range"
            min={-60}
            max={0}
            step={0.5}
            value={pos}
            onChange={(e) => setPos(Number(e.target.value))}
            style={{ width: '100%', accentColor: 'var(--accent)' }}
          />
          <div
            style={{
              position: 'absolute',
              top: 22,
              left: 0,
              right: 0,
              display: 'flex',
              justifyContent: 'space-between',
              pointerEvents: 'none',
            }}
          >
            {TICKS.map((t) => (
              <div key={t} className="mono" style={{ fontSize: 9, color: 'var(--text-tertiary)' }}>
                {t === 0 ? 'T-0' : `${t}d`}
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
