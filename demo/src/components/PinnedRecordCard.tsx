import { ArrowLeftRight, X, Pin } from 'lucide-react';
import type { CanonicalRecord } from '../types';
import type { Action } from '../state';

interface Props {
  pinned: CanonicalRecord;
  selected: CanonicalRecord | null;
  dispatch: React.Dispatch<Action>;
}

// Bottom-left floating chip showing the pinned record for A/B comparison.
// Click body to swap pinned ↔ selected; X clears the pin.
export function PinnedRecordCard({ pinned, selected, dispatch }: Props) {
  const delta = selected ? selected.fusion.anomaly - pinned.fusion.anomaly : 0;
  const deltaColor =
    Math.abs(delta) < 0.05
      ? 'var(--text-tertiary)'
      : delta > 0
        ? 'var(--alert)'
        : 'var(--nominal)';
  return (
    <div
      style={{
        position: 'absolute',
        bottom: 24,
        left: 24,
        maxWidth: 280,
        padding: '10px 14px',
        background: 'rgba(255,255,255,0.92)',
        backdropFilter: 'blur(20px) saturate(180%)',
        WebkitBackdropFilter: 'blur(20px) saturate(180%)',
        borderRadius: 'var(--r-md)',
        boxShadow: 'var(--shadow-md)',
        border: '1px solid var(--border-subtle)',
        display: 'flex',
        alignItems: 'center',
        gap: 10,
        zIndex: 'var(--z-map-overlay)' as any,
        animation: 'fade-in-up var(--dur-base) var(--ease-entrance)',
      }}
    >
      <Pin size={12} strokeWidth={2} color="var(--accent)" />
      <div style={{ minWidth: 0, flex: 1 }}>
        <div
          style={{
            fontSize: 12,
            fontWeight: 600,
            color: 'var(--text-primary)',
            whiteSpace: 'nowrap',
            overflow: 'hidden',
            textOverflow: 'ellipsis',
          }}
        >
          {pinned.name}
        </div>
        <div style={{ fontSize: 11, color: 'var(--text-tertiary)' }}>
          anomaly <span className="mono">{pinned.fusion.anomaly.toFixed(3)}</span>
          {selected && selected.id !== pinned.id && (
            <>
              {' · Δ '}
              <span className="mono" style={{ color: deltaColor }}>
                {delta >= 0 ? '+' : ''}{delta.toFixed(3)}
              </span>
            </>
          )}
        </div>
      </div>
      <button
        onClick={() => dispatch({ type: 'SWAP_PINNED' })}
        title="Swap pinned ↔ selected"
        disabled={!selected || selected.id === pinned.id}
        style={{
          width: 26, height: 26, display: 'flex', alignItems: 'center', justifyContent: 'center',
          borderRadius: 'var(--r-pill)', color: 'var(--text-secondary)',
          cursor: !selected || selected.id === pinned.id ? 'not-allowed' : 'pointer',
          opacity: !selected || selected.id === pinned.id ? 0.4 : 1,
        }}
      >
        <ArrowLeftRight size={13} strokeWidth={2} />
      </button>
      <button
        onClick={() => dispatch({ type: 'PIN_RECORD', record: null })}
        title="Clear pin"
        style={{
          width: 26, height: 26, display: 'flex', alignItems: 'center', justifyContent: 'center',
          borderRadius: 'var(--r-pill)', color: 'var(--text-tertiary)',
        }}
      >
        <X size={13} strokeWidth={2} />
      </button>
    </div>
  );
}
