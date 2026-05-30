import { useState } from 'react';
import { Star, ChevronDown } from 'lucide-react';
import type { CanonicalRecord } from '../types';
import type { Action } from '../state';

interface Props {
  events: CanonicalRecord[];
  cleanSites: CanonicalRecord[];
  dispatch: React.Dispatch<Action>;
}

const SHORT: Record<string, string> = {
  'Lake Erie HAB 2023': 'Lake Erie 2023',
  'Gulf Dead Zone 2023': 'Gulf Dead Zone 2023',
  'Chesapeake Hypoxia 2018': 'Chesapeake 2018',
  'Klamath River HAB 2021': 'Klamath 2021',
  'Mississippi Salinity 2023': 'Mississippi Salinity',
  'Jordan Lake HAB 2022': 'Jordan Lake 2022',
  'Jordan Lake HAB Nc': 'Jordan Lake 2022',
  'Toolik Lake, AK': 'Toolik Lake (clean)',
  'Sycamore Creek, AZ': 'Sycamore Creek (clean)',
};

export function StoryChips({ events, cleanSites, dispatch }: Props) {
  const [open, setOpen] = useState(false);
  return (
    <div
      style={{
        position: 'absolute',
        bottom: 20,
        left: 20,
        zIndex: 'var(--z-map-overlay)' as any,
      }}
    >
      <button
        onClick={() => setOpen((v) => !v)}
        style={{
          height: 36,
          padding: '0 16px',
          background: 'rgba(255,255,255,0.78)',
          backdropFilter: 'blur(20px) saturate(180%)',
          WebkitBackdropFilter: 'blur(20px) saturate(180%)',
          color: 'var(--text-primary)',
          fontSize: 13,
          fontWeight: 500,
          borderRadius: 'var(--r-pill)',
          boxShadow: 'var(--shadow-md)',
          display: 'flex',
          alignItems: 'center',
          gap: 8,
        }}
      >
        <Star size={14} strokeWidth={1.75} />
        Featured events
        <ChevronDown size={14} strokeWidth={1.75} style={{ transform: open ? 'rotate(180deg)' : 'none', transition: 'transform var(--dur-fast)' }} />
      </button>
      {open && (
        <div
          style={{
            marginTop: 8,
            background: 'var(--bg-surface)',
            borderRadius: 'var(--r-md)',
            boxShadow: 'var(--shadow-lg)',
            padding: 6,
            minWidth: 240,
            animation: 'fade-in-up var(--dur-base) var(--ease-entrance)',
          }}
        >
          {events.map((e) => (
            <Row key={e.id} onClick={() => { dispatch({ type: 'SELECT_SITE', record: e }); setOpen(false); }}>
              {SHORT[e.name] ?? e.name}
            </Row>
          ))}
          <div style={{ height: 1, background: 'var(--border-subtle)', margin: '6px 4px' }} />
          {cleanSites.map((e) => (
            <Row key={e.id} dim onClick={() => { dispatch({ type: 'SELECT_SITE', record: e }); setOpen(false); }}>
              {SHORT[e.name] ?? e.name}
            </Row>
          ))}
        </div>
      )}
    </div>
  );
}

function Row({ children, onClick, dim }: { children: React.ReactNode; onClick: () => void; dim?: boolean }) {
  return (
    <button
      onClick={onClick}
      style={{
        display: 'block',
        width: '100%',
        textAlign: 'left',
        padding: '8px 12px',
        fontSize: 13,
        color: dim ? 'var(--text-tertiary)' : 'var(--text-primary)',
        borderRadius: 'var(--r-sm)',
        transition: 'background var(--dur-fast)',
      }}
      onMouseEnter={(e) => (e.currentTarget.style.background = 'var(--bg-canvas)')}
      onMouseLeave={(e) => (e.currentTarget.style.background = 'transparent')}
    >
      {children}
    </button>
  );
}
