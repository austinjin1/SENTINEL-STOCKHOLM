import { X } from 'lucide-react';
import type { Action } from '../state';
import { MICROBIAL_VALIDATION, MOLECULAR_VALIDATION, BEHAVIORAL_VALIDATION } from '../data/realData';
import { MODELS_BY_ID } from '../content/models';
import { FUSION_HEADLINE, CROSS_REGION_TRANSFER } from '../content/fusion';

interface Props {
  open: boolean;
  dispatch: React.Dispatch<Action>;
}

// Values are sourced from the single-source-of-truth content modules
// (content/models.ts, content/fusion.ts) so this /live panel shows the
// remediated, held-out numbers — never a retracted value.
const aquassm = MODELS_BY_ID['aquassm'];
const hydrovit = MODELS_BY_ID['hydrovit'];
const microbiome = MODELS_BY_ID['microbiomenet'];
const toxigene = MODELS_BY_ID['toxigene'];
const biomotion = MODELS_BY_ID['biomotion'];
const crossRegionMc = CROSS_REGION_TRANSFER[0].raw; // >1.6 µg/L cross-region collapse

const ROWS: { metric: string; value: string; source: string }[] = [
  { metric: 'AquaSSM reconstruction RMSE', value: aquassm.metricValue, source: 'Real NEON station holdout. The label-free pre-onset anomaly detector is scored separately (see Track A pre-onset detection below), not on synthetic anomalies.' },
  { metric: 'HydroViT R² (water temperature)', value: hydrovit.metricValue, source: 'Real Sentinel-2 × in-situ pairs, station holdout. Replaces the retracted fabricated 0.893; chl-a is harder (R² 0.32, red-edge).' },
  { metric: 'MicroBiomeNet macro-F1', value: microbiome.metricValue, source: 'True geographic holdout with transferable family/genus taxonomy (0.405 on raw OTUs). The retracted 0.831 was random-split leakage.' },
  { metric: 'ToxiGene within-study AUROC', value: toxigene.metricValue, source: 'Real zebrafish GEO, within-study CV. Cross-study LOSO is only ~0.64 (rank-norm). The retracted 0.492 came from invalid non-aquatic GEO.' },
  { metric: 'BioMotion AUROC', value: biomotion.metricValue, source: 'Real EPA ECOTOX assays, chemical-grouped split (tabular GBM; 0.807 was the diffusion-encoder baseline). Does not transfer across species.' },
  { metric: 'Microcystin fusion AUROC (within region)', value: FUSION_HEADLINE.threeWay.toFixed(2), source: `3-way fusion beats best single (${FUSION_HEADLINE.bestSingle.toFixed(2)}) at the WHO drinking threshold under station-grouped CV — but collapses to ~${crossRegionMc.toFixed(2)} (chance) across unseen regions.` },
  { metric: 'Track A pre-onset detection', value: '25 / 32 episodes', source: 'Real NEON+USGS, label-free rolling-CUSUM detector fires a sustained pre-onset alarm on 25/32 DO-hypoxia episodes across two networks at a matched ≤5% held-out false-alarm rate (baseline point-threshold: 5/15 on NEON). Reported as detection, not a fixed lead-time: a lead-vs-lookback audit shows the measured "lead" tracks the 45-day operating window (corr 0.997), so we do not headline an N-day forecast horizon.' },
  { metric: 'Cascade tier cost progression', value: '$15K / +$0.5K / +$15K / +$25K', source: 'Illustrative cascade design.' },
  { metric: 'Modality decay half-lives', value: '5m / 2h / 3d / 5d / 7d', source: 'Fusion module design parameters.' },
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
        zIndex: 'var(--z-demo)',
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
            Off-event case-study screening (NOT validated detection)
          </div>
          <div style={{ fontSize: 11, color: 'var(--text-tertiary)', marginBottom: 10, lineHeight: 1.6 }}>
            These per-case rates are surrogate/screening outputs on illustrative case studies, not validated
            detection on a real held-out set. Real held-out metrics: MicroBiomeNet macro-F1 {microbiome.metricValue}{' '}
            (geographic holdout), ToxiGene within-study AUROC {toxigene.metricValue}, BioMotion AUROC {biomotion.metricValue}.
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
