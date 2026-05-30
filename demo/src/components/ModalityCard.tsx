import { useEffect, useRef, useState } from 'react';
import type { Coverage } from '../types';

interface Props {
  name: string;
  subtitle: string;
  accentColor: string;
  coverage: Coverage;
  conf?: number;
  headlineMetric?: React.ReactNode;
  headlineLabel?: string;
  secondary?: React.ReactNode;
  children?: React.ReactNode;
  deployableTier?: number;
  onActivate?: () => void;
  cardIndex?: number;
}

const COV_COLOR: Record<Coverage, string> = {
  OBSERVED: 'var(--obs)',
  INFERRED: 'var(--inf)',
  DEPLOYABLE: 'var(--dep)',
  PROJECTED: 'var(--accent)',
};

const COV_LABEL: Record<Coverage, string> = {
  OBSERVED: 'Observed',
  INFERRED: 'Inferred',
  DEPLOYABLE: 'Deployable',
  PROJECTED: 'Projected',
};

const COV_TOOLTIP: Record<Coverage, string> = {
  OBSERVED: 'Direct measurement from this site.',
  INFERRED: 'Estimated from nearby stations / regional baseline.',
  PROJECTED: 'Modeled forward from sensor + satellite signals.',
  DEPLOYABLE: 'Modality available but not yet activated — escalate to enable.',
};

export function ModalityCard({
  name,
  subtitle,
  accentColor,
  coverage,
  conf,
  headlineMetric,
  headlineLabel,
  secondary,
  children,
  deployableTier,
  cardIndex = 0,
}: Props) {
  const deployable = coverage === 'DEPLOYABLE';

  // PROCESSING scanline: when transitioning from DEPLOYABLE to a live state,
  // play a 700ms sweep over the viz well before the real content settles in.
  const prevCoverage = useRef(coverage);
  const [processing, setProcessing] = useState(false);
  useEffect(() => {
    if (prevCoverage.current === 'DEPLOYABLE' && coverage !== 'DEPLOYABLE') {
      setProcessing(true);
      const id = window.setTimeout(() => setProcessing(false), 700);
      prevCoverage.current = coverage;
      return () => window.clearTimeout(id);
    }
    prevCoverage.current = coverage;
    return;
  }, [coverage]);

  return (
    <div
      style={{
        background: 'var(--bg-elevated)',
        border: '1px solid var(--border-subtle)',
        borderRadius: 'var(--r-lg)',
        padding: 16,
        display: 'flex',
        flexDirection: 'column',
        gap: 10,
        opacity: deployable ? 0.55 : 1,
        boxShadow: 'var(--shadow-sm)',
        animation: `fade-in-up var(--dur-base) var(--ease-entrance) ${cardIndex * 45}ms both`,
      }}
    >
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
        <div>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <span
              style={{ width: 8, height: 8, borderRadius: 4, background: accentColor }}
            />
            <span style={{ fontWeight: 600, fontSize: 14, color: 'var(--text-primary)' }}>{name}</span>
          </div>
          <div style={{ fontSize: 11, color: 'var(--text-tertiary)', marginTop: 4, marginLeft: 16 }}>
            {subtitle}
          </div>
        </div>
        <CoverageChip coverage={coverage} conf={conf} />
      </div>
      <div
        style={{
          background: 'var(--bg-inset)',
          borderRadius: 'var(--r-md)',
          minHeight: 80,
          padding: 10,
          position: 'relative',
        }}
      >
        {deployable ? (
          <div
            style={{
              position: 'absolute',
              inset: 0,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              fontSize: 12,
              color: 'var(--text-tertiary)',
              textAlign: 'center',
              padding: 12,
            }}
          >
            Activate at Tier {deployableTier ?? '?'}
          </div>
        ) : (
          <>
            {children}
            {processing && (
              <>
                <div
                  style={{
                    position: 'absolute',
                    inset: 0,
                    background: 'var(--bg-inset)',
                    opacity: 0.6,
                    pointerEvents: 'none',
                  }}
                />
                <div
                  style={{
                    position: 'absolute',
                    left: 0,
                    right: 0,
                    height: 2,
                    background: `linear-gradient(90deg, transparent, ${accentColor}, transparent)`,
                    boxShadow: `0 0 10px ${accentColor}`,
                    animation: 'scanline-sweep 700ms linear forwards',
                    pointerEvents: 'none',
                  }}
                />
                <div
                  style={{
                    position: 'absolute',
                    top: 6,
                    left: 8,
                    fontSize: 10,
                    color: accentColor,
                    letterSpacing: '0.1em',
                    textTransform: 'uppercase',
                    pointerEvents: 'none',
                  }}
                >
                  Processing
                </div>
              </>
            )}
          </>
        )}
      </div>
      {headlineMetric && (
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline' }}>
          <div>
            <div className="metric-l" style={{ color: 'var(--text-primary)' }}>
              {headlineMetric}
            </div>
            {headlineLabel && (
              <div style={{ fontSize: 11, color: 'var(--text-tertiary)', marginTop: 2 }}>
                {headlineLabel}
              </div>
            )}
          </div>
          {secondary && (
            <div style={{ fontSize: 12, color: 'var(--text-secondary)', textAlign: 'right' }}>
              {secondary}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

function CoverageChip({ coverage, conf }: { coverage: Coverage; conf?: number }) {
  return (
    <div
      title={COV_TOOLTIP[coverage]}
      style={{
        display: 'flex',
        alignItems: 'center',
        gap: 6,
        height: 22,
        padding: '0 10px',
        background: 'rgba(255,255,255,0.04)',
        border: `1px solid ${COV_COLOR[coverage]}`,
        borderRadius: 'var(--r-pill)',
        fontSize: 11,
        color: 'var(--text-secondary)',
        cursor: 'help',
      }}
    >
      <span style={{ width: 6, height: 6, borderRadius: 3, background: COV_COLOR[coverage] }} />
      {COV_LABEL[coverage]}
      {coverage === 'INFERRED' && conf !== undefined && (
        <span className="mono" style={{ color: 'var(--text-tertiary)', marginLeft: 2 }}>
          {conf.toFixed(2)}
        </span>
      )}
    </div>
  );
}
