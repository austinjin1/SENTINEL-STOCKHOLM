import { X } from 'lucide-react';
import type { Action } from '../state';
import { MICROBIAL_VALIDATION, MOLECULAR_VALIDATION, BEHAVIORAL_VALIDATION } from '../data/realData';

interface Props {
  open: boolean;
  dispatch: React.Dispatch<Action>;
}

const ROWS: { metric: string; value: string; source: string }[] = [
  { metric: 'AquaSSM AUROC', value: '0.920', source: 'Paper §3.1 · CI [0.91, 0.93]' },
  { metric: 'HydroViT R² (water temperature)', value: '0.749', source: 'Paper §3.2 · 4,202 Sentinel-2 / in-situ pairs' },
  { metric: 'MicroBiomeNet F1 (8-class)', value: '0.913', source: 'Paper §3.3 · EMP 16S rRNA' },
  { metric: 'ToxiGene F1 (multi-label)', value: '0.894', source: 'Paper §3.4 · GEO + ECOTOX' },
  { metric: 'BioMotion AUROC', value: '1.000', source: 'Paper §3.5 · lab benchmark, down-weighted in fusion' },
  { metric: 'Fusion AUROC', value: '0.939', source: 'Paper §3.6 · CI [0.922, 0.956], p=0.002 vs sensor' },
  { metric: 'Lake Erie HAB lead time', value: '59.3d', source: 'Repo · case_studies_real/lake_erie_hab_2023_scores.json' },
  { metric: 'Cascade tier cost progression', value: '$15K / +$0.5K / +$15K / +$25K', source: 'Paper §4 cascade design' },
  { metric: 'Modality decay half-lives', value: '5m / 2h / 3d / 5d / 7d', source: 'Paper §2 fusion module' },
];

export function MethodologyPanel({ open, dispatch }: Props) {
  if (!open) return null;
  return (
    <div
      style={{
        position: 'fixed',
        inset: 0,
        background: 'var(--bg-scrim)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        zIndex: 'var(--z-demo)' as any,
        animation: 'fade-in var(--dur-base) var(--ease-standard)',
      }}
      onClick={() => dispatch({ type: 'TOGGLE_METHODOLOGY' })}
    >
      <div
        onClick={(e) => e.stopPropagation()}
        style={{
          width: 720,
          maxHeight: '82vh',
          background: 'var(--bg-surface)',
          borderRadius: 'var(--r-lg)',
          boxShadow: 'var(--shadow-lg)',
          padding: 32,
          overflowY: 'auto',
        }}
      >
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
          <div>
            <div style={{ fontSize: 24, fontWeight: 600, letterSpacing: '-0.02em' }}>
              What is real, what is modeled
            </div>
            <div style={{ fontSize: 14, color: 'var(--text-secondary)', marginTop: 6, maxWidth: 540 }}>
              Every number on screen carries a provenance tag. SENTINEL claims nothing it cannot defend.
            </div>
          </div>
          <button
            onClick={() => dispatch({ type: 'TOGGLE_METHODOLOGY' })}
            style={{
              width: 32,
              height: 32,
              borderRadius: 'var(--r-pill)',
              background: 'var(--bg-canvas)',
              color: 'var(--text-secondary)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
            }}
          >
            <X size={18} strokeWidth={1.75} />
          </button>
        </div>

        <div style={{ marginTop: 24 }}>
          {ROWS.map((r) => (
            <div
              key={r.metric}
              style={{
                display: 'grid',
                gridTemplateColumns: '1.6fr 1fr 2.4fr',
                gap: 16,
                padding: '12px 0',
                borderBottom: '1px solid var(--border-subtle)',
                alignItems: 'baseline',
              }}
            >
              <div style={{ fontSize: 13, color: 'var(--text-secondary)' }}>{r.metric}</div>
              <div className="mono" style={{ fontSize: 14, color: 'var(--text-primary)', fontWeight: 500 }}>
                {r.value}
              </div>
              <div style={{ fontSize: 12, color: 'var(--text-tertiary)' }}>{r.source}</div>
            </div>
          ))}
        </div>

        <div style={{ marginTop: 28 }}>
          <div style={{ fontSize: 13, fontWeight: 600, color: 'var(--text-primary)', marginBottom: 10 }}>
            Off-event validation evidence
          </div>
          <div style={{ fontSize: 13, color: 'var(--text-secondary)', lineHeight: 1.7 }}>
            <div style={{ marginBottom: 6 }}>
              <span style={{ color: 'var(--m-microbial)', fontWeight: 500 }}>Microbial</span>
              {' — '}
              {MICROBIAL_VALIDATION.slice(0, 3)
                .map((v) => `${v.event.split(' — ')[0]} ${(v.detection_rate * 100).toFixed(0)}%`)
                .join(' · ')}
            </div>
            <div style={{ marginBottom: 6 }}>
              <span style={{ color: 'var(--m-molecular)', fontWeight: 500 }}>Molecular</span>
              {' — '}
              {MOLECULAR_VALIDATION.slice(0, 3)
                .map((v) => `${v.contaminant} ${(v.rate * 100).toFixed(0)}%`)
                .join(' · ')}
            </div>
            <div>
              <span style={{ color: 'var(--m-behavioral)', fontWeight: 500 }}>Behavioral</span>
              {' — '}
              {BEHAVIORAL_VALIDATION.slice(0, 3)
                .map((v) => `${v.chemical.split('(')[0].trim()} ${(v.detection_rate * 100).toFixed(0)}%`)
                .join(' · ')}
            </div>
          </div>
        </div>

        <div
          style={{
            marginTop: 28,
            padding: 16,
            background: 'var(--bg-canvas)',
            borderRadius: 'var(--r-md)',
            fontSize: 12,
            color: 'var(--text-secondary)',
            lineHeight: 1.6,
          }}
        >
          <strong style={{ color: 'var(--text-primary)' }}>Observed</strong> · real cached sample at this cell.{' '}
          <strong style={{ color: 'var(--text-primary)' }}>Inferred</strong> · regional baseline with distance-decayed confidence.{' '}
          <strong style={{ color: 'var(--text-primary)' }}>Deployable</strong> · inactive until cascade activates.{' '}
          <strong style={{ color: 'var(--text-primary)' }}>Projected</strong> · activated cell with no physical sample; illustrative.
          <div style={{ marginTop: 8, color: 'var(--text-tertiary)' }}>
            No live network, no fetch, fully client-side. Cached and modeled data — not for operational use.
          </div>
        </div>
      </div>
    </div>
  );
}
