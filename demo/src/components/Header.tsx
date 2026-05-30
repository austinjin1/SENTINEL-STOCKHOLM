import { X } from 'lucide-react';
import type { CanonicalRecord, AlertTier } from '../types';
import type { Action } from '../state';

interface Props {
  record: CanonicalRecord;
  dispatch: React.Dispatch<Action>;
}

const TIER_STYLES: Record<AlertTier, { fg: string; bg: string; label: string }> = {
  MAINTAIN: { fg: 'var(--nominal)', bg: 'var(--nominal-dim)', label: 'Maintain' },
  WATCH: { fg: 'var(--watch)', bg: 'var(--watch-dim)', label: 'Watch' },
  INVESTIGATE: { fg: 'var(--investigate)', bg: 'var(--investigate-dim)', label: 'Investigate' },
  ALERT: { fg: 'var(--alert)', bg: 'var(--alert-dim)', label: 'Alert' },
};

export function Header({ record, dispatch }: Props) {
  const tier = record.fusion.alert;
  const style = TIER_STYLES[tier];
  return (
    <div
      style={{
        padding: '20px 20px 16px',
        borderBottom: '1px solid var(--border-subtle)',
        position: 'relative',
      }}
    >
      <button
        onClick={() => dispatch({ type: 'CLOSE_RAIL' })}
        title="Close"
        style={{
          position: 'absolute',
          top: 14,
          right: 14,
          width: 28,
          height: 28,
          borderRadius: 'var(--r-md)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          color: 'var(--text-tertiary)',
          transition: 'background var(--dur-fast), color var(--dur-fast)',
        }}
        onMouseEnter={(e) => {
          e.currentTarget.style.background = 'rgba(255,255,255,0.06)';
          e.currentTarget.style.color = 'var(--text-primary)';
        }}
        onMouseLeave={(e) => {
          e.currentTarget.style.background = 'transparent';
          e.currentTarget.style.color = 'var(--text-tertiary)';
        }}
      >
        <X size={16} strokeWidth={1.5} />
      </button>
      <div style={{ display: 'flex', justifyContent: 'space-between', gap: 12, alignItems: 'flex-start' }}>
        <div style={{ paddingRight: 44 }}>
          <div style={{ fontWeight: 600, fontSize: 19, lineHeight: '24px' }}>
            {record.name}
          </div>
          <div style={{ fontSize: 13, color: 'var(--text-secondary)', marginTop: 4 }}>{record.label}</div>
          <div
            style={{
              fontSize: 12,
              color: 'var(--text-tertiary)',
              marginTop: 10,
              display: 'flex',
              gap: 12,
              flexWrap: 'wrap',
            }}
          >
            {record.id && (
              <span>
                USGS <span className="mono" style={{ color: 'var(--text-secondary)' }}>{record.id.split('-')[0]}</span>
              </span>
            )}
            <span className="mono" style={{ color: 'var(--text-secondary)' }}>
              {record.lat.toFixed(2)}, {record.lon.toFixed(2)}
            </span>
            <span>
              H3 <span className="mono" style={{ color: 'var(--text-secondary)' }}>{record.h3.slice(0, 8)}…</span>
            </span>
          </div>
        </div>
        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: 8,
            height: 28,
            padding: '0 12px',
            borderRadius: 'var(--r-pill)',
            background: style.bg,
            color: style.fg,
            fontSize: 12,
            fontWeight: 600,
            flexShrink: 0,
          }}
        >
          <span style={{ width: 7, height: 7, borderRadius: 4, background: style.fg }} />
          {style.label}
        </div>
      </div>
    </div>
  );
}
