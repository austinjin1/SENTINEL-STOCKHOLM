import { useEffect, useState } from 'react';
import type { CanonicalRecord } from '../types';

interface Props {
  record: CanonicalRecord;
}

const MODALITY_DOTS = [
  { color: 'var(--m-sensor)', key: 'sensor' as const },
  { color: 'var(--m-satellite)', key: 'satellite' as const },
  { color: 'var(--m-behavioral)', key: 'behavioral' as const },
  { color: 'var(--m-microbial)', key: 'microbial' as const },
  { color: 'var(--m-molecular)', key: 'molecular' as const },
];

function useCountUp(target: number, dur = 800) {
  const [v, setV] = useState(0);
  useEffect(() => {
    const start = performance.now();
    let raf = 0;
    const step = (t: number) => {
      const p = Math.min(1, (t - start) / dur);
      const e = 1 - Math.pow(1 - p, 3);
      setV(target * e);
      if (p < 1) raf = requestAnimationFrame(step);
    };
    raf = requestAnimationFrame(step);
    return () => cancelAnimationFrame(raf);
  }, [target, dur]);
  return v;
}

export function FusionSummary({ record }: Props) {
  const anomaly = useCountUp(record.fusion.anomaly);
  const states = {
    sensor: record.sensor.state,
    satellite: record.satellite.state,
    microbial: record.microbial.state,
    molecular: record.molecular.state,
    behavioral: record.behavioral.state,
  };
  const sensorOnly = 0.92;
  const delta = record.fusion.anomaly - sensorOnly;
  return (
    <div
      style={{
        display: 'flex',
        padding: '16px 20px',
        gap: 24,
        borderBottom: '1px solid var(--border-subtle)',
      }}
    >
      <div style={{ flex: 1.4 }}>
        <div className="metric-xl" style={{ color: 'var(--text-primary)' }}>{anomaly.toFixed(3)}</div>
        <div style={{ fontSize: 11, color: 'var(--text-tertiary)', marginTop: 4 }}>
          Fused anomaly
        </div>
        <div
          style={{
            fontSize: 11,
            color: delta > 0 ? 'var(--accent-bright)' : 'var(--text-tertiary)',
            marginTop: 4,
          }}
        >
          <span className="mono">
            {delta >= 0 ? '+' : ''}{delta.toFixed(3)}
          </span> vs sensor only
        </div>
      </div>
      <div style={{ flex: 1.2, display: 'flex', flexDirection: 'column', justifyContent: 'space-between' }}>
        <div style={{ fontSize: 11, color: 'var(--text-tertiary)' }}>
          {record.n_modalities} of 5 modalities active
        </div>
        <div style={{ display: 'flex', gap: 8 }}>
          {MODALITY_DOTS.map((m) => {
            const active = states[m.key] !== 'DEPLOYABLE';
            return (
              <div
                key={m.key}
                title={m.key}
                style={{
                  width: 14,
                  height: 14,
                  borderRadius: 7,
                  background: active ? m.color : 'rgba(255,255,255,0.08)',
                  transition: 'all var(--dur-base) var(--ease-standard)',
                }}
              />
            );
          })}
        </div>
        <div style={{ fontSize: 11, color: 'var(--text-tertiary)' }}>
          attention rebalances on escalation
        </div>
      </div>
      <div style={{ flex: 1 }}>
        <div className="metric-l" style={{ color: 'var(--text-primary)' }}>
          {record.fusion.auroc.toFixed(3)}
        </div>
        <div style={{ fontSize: 11, color: 'var(--text-tertiary)', marginTop: 2 }}>
          Fusion AUROC
        </div>
        <div style={{ fontSize: 12, color: 'var(--text-secondary)', marginTop: 8 }}>
          Coverage <span className="mono">{(record.fusion.coverage * 100).toFixed(0)}%</span>
        </div>
        {record.leadDays !== undefined && (
          <div style={{ fontSize: 12, color: 'var(--accent-bright)', marginTop: 6 }}>
            <span className="mono">{record.leadDays.toFixed(1)}d</span> before advisory
          </div>
        )}
      </div>
    </div>
  );
}
