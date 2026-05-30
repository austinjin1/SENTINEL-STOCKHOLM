import { ChevronUp, ChevronDown } from 'lucide-react';
import type { Action } from '../state';
import { TOP_TRIGGER_PARAMS } from '../data/realData';

interface Props {
  tier: 0 | 1 | 2 | 3;
  costK: number;
  dispatch: React.Dispatch<Action>;
}

const TIERS = [
  { label: 'Tier 0', adds: 'Sensor + Behavioral' },
  { label: 'Tier 1', adds: '+ Satellite' },
  { label: 'Tier 2', adds: '+ Microbial' },
  { label: 'Tier 3', adds: '+ Molecular' },
];

const TIER_COSTS = [0, 0.5, 15, 25];

const TOP_FEATURE = TOP_TRIGGER_PARAMS[0]?.name ?? 'sensor anomaly';
const FEATURE_BY_TIER: Record<number, string> = {
  0: `Top trigger ${TOP_FEATURE} (${TOP_TRIGGER_PARAMS[0]?.count}× across NEON sites)`,
  1: `Satellite chl-a > 25 µg/L, ${TOP_TRIGGER_PARAMS[1]?.name} elevated`,
  2: `Microbial source confidence > 0.90, ${TOP_TRIGGER_PARAMS[2]?.name} elevated`,
  3: `Toxicity outcome confirmed, ${TOP_TRIGGER_PARAMS[3]?.name} elevated`,
};

export function CascadeBar({ tier, costK, dispatch }: Props) {
  return (
    <div
      style={{
        padding: '14px 20px',
        borderBottom: '1px solid var(--border-subtle)',
        display: 'flex',
        flexDirection: 'column',
        gap: 8,
      }}
    >
      <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
        <div style={{ display: 'flex', flex: 1, gap: 4 }}>
          {TIERS.map((t, i) => {
            const active = i <= tier;
            const current = i === tier;
            return (
              <div
                key={t.label}
                style={{
                  flex: 1,
                  height: 38,
                  padding: '6px 10px',
                  display: 'flex',
                  flexDirection: 'column',
                  justifyContent: 'center',
                  background: current
                    ? 'var(--accent-soft)'
                    : active
                      ? 'var(--bg-elevated)'
                      : 'transparent',
                  border: `1px solid ${current ? 'var(--accent)' : 'var(--border-subtle)'}`,
                  borderRadius: 'var(--r-md)',
                  transition: 'all var(--dur-base) var(--ease-standard)',
                }}
              >
                <div
                  style={{
                    fontSize: 11,
                    fontWeight: 600,
                    color: current
                      ? 'var(--accent-bright)'
                      : active
                        ? 'var(--text-primary)'
                        : 'var(--text-tertiary)',
                  }}
                >
                  {t.label}
                </div>
                <div style={{ fontSize: 10, color: 'var(--text-tertiary)', marginTop: 1 }}>{t.adds}</div>
              </div>
            );
          })}
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'flex-end', lineHeight: 1.2 }}>
            <div style={{ fontSize: 13, color: 'var(--text-primary)' }}>
              <span className="mono">${costK.toFixed(1)}K</span>
              <span style={{ fontSize: 10, color: 'var(--text-tertiary)', marginLeft: 4 }}>total</span>
            </div>
            {tier < 3 && (
              <div style={{ fontSize: 10, color: 'var(--accent)' }}>
                next +<span className="mono">${TIER_COSTS[tier + 1].toFixed(1)}K</span>
              </div>
            )}
          </div>
          <button
            onClick={() => dispatch({ type: 'DEESCALATE' })}
            disabled={tier <= 0}
            title="De-escalate"
            style={{
              width: 30,
              height: 30,
              border: '1px solid var(--border-subtle)',
              background: 'var(--bg-elevated)',
              color: tier > 0 ? 'var(--text-secondary)' : 'var(--text-tertiary)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              cursor: tier > 0 ? 'pointer' : 'not-allowed',
              opacity: tier > 0 ? 1 : 0.4,
              borderRadius: 'var(--r-md)',
            }}
          >
            <ChevronDown size={14} strokeWidth={1.5} />
          </button>
          <button
            onClick={() => dispatch({ type: 'ESCALATE' })}
            disabled={tier >= 3}
            title="Escalate"
            style={{
              width: 30,
              height: 30,
              border: '1px solid var(--accent)',
              background: 'var(--accent-soft)',
              color: 'var(--accent-bright)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              cursor: tier < 3 ? 'pointer' : 'not-allowed',
              opacity: tier < 3 ? 1 : 0.4,
              borderRadius: 'var(--r-md)',
            }}
          >
            <ChevronUp size={14} strokeWidth={1.5} />
          </button>
        </div>
      </div>
      <div style={{ fontSize: 11, color: 'var(--text-tertiary)' }}>
        Policy · {FEATURE_BY_TIER[tier]}
      </div>
    </div>
  );
}
