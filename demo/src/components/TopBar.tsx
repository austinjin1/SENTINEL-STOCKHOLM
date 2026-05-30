import { useEffect, useState } from 'react';
import { Circle, FileText, Search, Eye } from 'lucide-react';
import type { Action } from '../state';

function useUtcClock() {
  const [t, setT] = useState(() => new Date());
  useEffect(() => {
    const id = setInterval(() => setT(new Date()), 1000);
    return () => clearInterval(id);
  }, []);
  return t.toISOString().slice(11, 19) + ' UTC';
}

interface Props {
  dispatch: React.Dispatch<Action>;
  colorBlind: boolean;
}

export function TopBar({ dispatch, colorBlind }: Props) {
  const clock = useUtcClock();
  return (
    <div
      style={{
        height: 56,
        background: 'rgba(255,255,255,0.72)',
        backdropFilter: 'blur(20px) saturate(180%)',
        WebkitBackdropFilter: 'blur(20px) saturate(180%)',
        borderBottom: '1px solid var(--border-subtle)',
        display: 'flex',
        alignItems: 'center',
        padding: '0 24px',
        flexShrink: 0,
        gap: 32,
      }}
    >
      <div style={{ display: 'flex', alignItems: 'baseline', gap: 12, minWidth: 220 }}>
        <span style={{ fontWeight: 600, fontSize: 17, letterSpacing: '-0.01em', color: 'var(--text-primary)' }}>
          SENTINEL
        </span>
        <span style={{ fontSize: 13, color: 'var(--text-tertiary)' }}>Water Quality Intelligence</span>
      </div>
      <div
        style={{
          flex: 1,
          maxWidth: 480,
          height: 36,
          background: 'rgba(0,0,0,0.04)',
          borderRadius: 'var(--r-md)',
          display: 'flex',
          alignItems: 'center',
          padding: '0 12px',
          gap: 8,
          color: 'var(--text-tertiary)',
        }}
      >
        <Search size={15} strokeWidth={1.75} />
        <span style={{ fontSize: 14 }}>Search a site, river, or coordinates…</span>
      </div>
      <div
        style={{
          marginLeft: 'auto',
          display: 'flex',
          alignItems: 'center',
          gap: 20,
          fontSize: 13,
          color: 'var(--text-secondary)',
        }}
      >
        <span className="mono" style={{ fontSize: 12, color: 'var(--text-tertiary)' }}>{clock}</span>
        <span style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
          <Circle size={8} fill="var(--nominal)" stroke="none" />
          Nominal
        </span>
        <button
          onClick={() => dispatch({ type: 'TOGGLE_COLORBLIND' })}
          title={colorBlind ? 'Switch to default palette' : 'Switch to color-blind-safe palette'}
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: 6,
            color: colorBlind ? 'var(--accent)' : 'var(--text-secondary)',
            fontSize: 13,
          }}
        >
          <Eye size={14} strokeWidth={1.75} />
          {colorBlind ? 'CB on' : 'CB'}
        </button>
        <button
          onClick={() => dispatch({ type: 'TOGGLE_METHODOLOGY' })}
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: 6,
            color: 'var(--text-link)',
            fontSize: 13,
          }}
        >
          <FileText size={14} strokeWidth={1.75} />
          Methodology
        </button>
      </div>
    </div>
  );
}
