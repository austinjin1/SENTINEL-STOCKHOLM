import { ChevronDown, ChevronRight } from 'lucide-react';
import type { CanonicalRecord } from '../types';
import type { Action } from '../state';
import { DEGRADATION_CURVE } from '../data/realData';

interface Props {
  record: CanonicalRecord;
  open: boolean;
  dispatch: React.Dispatch<Action>;
}

const MOD_LABELS = [
  { label: 'sensor', color: 'var(--m-sensor)' },
  { label: 'satellite', color: 'var(--m-satellite)' },
  { label: 'behavioral', color: 'var(--m-behavioral)' },
  { label: 'microbial', color: 'var(--m-microbial)' },
  { label: 'molecular', color: 'var(--m-molecular)' },
];

const DECAY = [
  { name: 'behavioral', half: '5m', tau_s: 300, color: 'var(--m-behavioral)' },
  { name: 'sensor', half: '2h', tau_s: 7200, color: 'var(--m-sensor)' },
  { name: 'molecular', half: '3d', tau_s: 86400 * 3, color: 'var(--m-molecular)' },
  { name: 'satellite', half: '5d', tau_s: 86400 * 5, color: 'var(--m-satellite)' },
  { name: 'microbial', half: '7d', tau_s: 86400 * 7, color: 'var(--m-microbial)' },
];

// Real graceful-degradation curve from results/robustness/degradation_curves.json
const DEGRADATION = DEGRADATION_CURVE.filter((d) => d.n >= 1).sort((a, b) => b.n - a.n);

export function FusionDetail({ record, open, dispatch }: Props) {
  return (
    <div style={{ borderBottom: '1px solid var(--border-subtle)' }}>
      <button
        onClick={() => dispatch({ type: 'TOGGLE_FUSION_DETAIL' })}
        style={{
          width: '100%',
          padding: '12px 20px',
          display: 'flex',
          alignItems: 'center',
          gap: 8,
          color: 'var(--text-secondary)',
          background: 'transparent',
        }}
      >
        {open ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
        <span style={{ fontSize: 13, fontWeight: 600, color: 'var(--text-primary)' }}>Fusion detail</span>
      </button>
      {open && (
        <div style={{ padding: '0 20px 20px', display: 'flex', flexDirection: 'column', gap: 20 }}>
          {/* attention bar */}
          <div>
            <div style={{ fontSize: 12, color: 'var(--text-secondary)', marginBottom: 8, fontWeight: 500 }}>
              Attention weights
            </div>
            <div style={{ display: 'flex', height: 12, border: '1px solid var(--hairline)' }}>
              {record.fusion.attention.map((w, i) => (
                <div
                  key={i}
                  style={{
                    width: `${w * 100}%`,
                    background: MOD_LABELS[i].color,
                    transition: 'width 500ms var(--ease-entrance)',
                  }}
                />
              ))}
            </div>
            <div
              style={{
                display: 'flex',
                justifyContent: 'space-between',
                marginTop: 4,
                gap: 8,
              }}
            >
              {record.fusion.attention.map((w, i) => (
                <div
                  key={i}
                  className="mono"
                  style={{
                    fontSize: 9,
                    color: w > 0.01 ? MOD_LABELS[i].color : 'var(--text-tertiary)',
                  }}
                >
                  {MOD_LABELS[i].label} {w.toFixed(3)}
                </div>
              ))}
            </div>
            <div className="mono" style={{ fontSize: 9, color: 'var(--text-tertiary)', marginTop: 4 }}>
              microbial &amp; molecular weight ≈0 in routine detection — retained for source attribution.
            </div>
          </div>
          {/* decay curves */}
          <div>
            <div style={{ fontSize: 12, color: 'var(--text-secondary)', marginBottom: 8, fontWeight: 500 }}>
              Temporal decay (exp(−Δt/τ))
            </div>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(5, 1fr)', gap: 6 }}>
              {DECAY.map((d) => (
                <div key={d.name}>
                  <svg width="100%" height="34" viewBox="0 0 80 34" preserveAspectRatio="none">
                    {(() => {
                      const pts: string[] = [];
                      for (let i = 0; i <= 40; i++) {
                        const dt = (i / 40) * d.tau_s * 4;
                        const y = Math.exp(-dt / d.tau_s);
                        pts.push(`${i === 0 ? 'M' : 'L'}${(i * 2).toFixed(1)},${(34 - y * 32).toFixed(1)}`);
                      }
                      return <path d={pts.join(' ')} stroke={d.color} strokeWidth={1.2} fill="none" />;
                    })()}
                  </svg>
                  <div className="mono" style={{ fontSize: 8, color: 'var(--text-tertiary)' }}>
                    {d.name} · {d.half}
                  </div>
                </div>
              ))}
            </div>
          </div>
          {/* gates */}
          <div>
            <div style={{ fontSize: 12, color: 'var(--text-secondary)', marginBottom: 8, fontWeight: 500 }}>
              Confidence gates
            </div>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(5, 1fr)', gap: 8 }}>
              {MOD_LABELS.map((m, i) => {
                const states = [
                  record.sensor.conf,
                  record.satellite.conf,
                  record.behavioral.conf,
                  record.microbial.conf,
                  record.molecular.conf,
                ];
                const g = states[i] ?? 0;
                return (
                  <div key={m.label}>
                    <div className="mono" style={{ fontSize: 8, color: 'var(--text-tertiary)' }}>
                      {m.label}
                    </div>
                    <div style={{ height: 4, background: 'var(--inert-dim)', marginTop: 2 }}>
                      <div
                        style={{
                          width: `${g * 100}%`,
                          height: '100%',
                          background: m.color,
                          transition: 'width var(--dur-slow) var(--ease-standard)',
                        }}
                      />
                    </div>
                    <div className="mono" style={{ fontSize: 9, color: 'var(--text-mono)', marginTop: 2 }}>
                      {g.toFixed(2)}
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
          {/* degradation */}
          <div>
            <div style={{ fontSize: 12, color: 'var(--text-secondary)', marginBottom: 8, fontWeight: 500 }}>
              Graceful degradation (AUROC floor)
            </div>
            <div style={{ display: 'flex', gap: 12 }}>
              {DEGRADATION.map((d) => (
                <div key={d.n} className="mono" style={{ fontSize: 11, color: 'var(--text-mono)' }}>
                  <span style={{ color: 'var(--text-tertiary)' }}>{d.n}:</span> {d.auroc.toFixed(3)}
                </div>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
