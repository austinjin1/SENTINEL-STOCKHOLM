import {
  useMemo,
  useState,
  type CSSProperties,
  type ReactNode,
} from 'react';
import {
  ChevronRight,
  Info,
  Search,
  ArrowUpDown,
  ArrowUp,
  ArrowDown,
  X,
} from 'lucide-react';
import {
  SectionHeader,
  Callout,
  StatusBadge,
  Tooltip,
  CopyButton,
  type StatusKind,
} from '../ui';
import {
  MODELS,
  STATUS_META,
  type ModelEntry,
  type ModelKind,
  type ModelStatus,
} from '../content/models';
import { GLOSSARY_BY_TERM } from '../content/glossary';

/* ---------------------------------------------------------------------------
   Models — the core reference table. Every model (encoders, fusion, downstream,
   deployment) in one dense, sortable, filterable gnomAD/cBioPortal-style table.
   Each row expands to the honest note, the corrected-figure log, the secondary
   metrics, and a metric-vs-baseline bar. Every metric + protocol carries a
   glossary-backed tooltip. Real, verified numbers only.
   ------------------------------------------------------------------------- */

// --- kind metadata (order + label + viz color for the left rail tick) -------
const KIND_META: Record<
  ModelKind,
  { label: string; plural: string; color: string; blurb: string }
> = {
  encoder: {
    label: 'Encoder',
    plural: 'Modality encoders',
    color: 'var(--viz-satellite)',
    blurb: 'Per-modality representation learners.',
  },
  fusion: {
    label: 'Fusion',
    plural: 'Multimodal fusion',
    color: 'var(--viz-molecular)',
    blurb: 'Cross-modal integration.',
  },
  downstream: {
    label: 'Downstream',
    plural: 'Downstream tasks',
    color: 'var(--viz-microbial)',
    blurb: 'Applied prediction heads on the encoders.',
  },
  deployment: {
    label: 'Deployment',
    plural: 'Deployment',
    color: 'var(--viz-behavioral)',
    blurb: 'Field-deployable, imagery-only screening.',
  },
};

const KIND_ORDER: ModelKind[] = ['encoder', 'fusion', 'downstream', 'deployment'];

type StatusTone = 'nominal' | 'watch' | 'alert' | 'neutral';

// --- sorting ----------------------------------------------------------------
type SortKey = 'name' | 'kind' | 'metric' | 'status';
type SortDir = 'asc' | 'desc';

// Rank statuses from most- to least-validated for a meaningful status sort.
const STATUS_RANK: Record<ModelStatus, number> = {
  validated: 0,
  'region-limited': 1,
  ceiling: 2,
  simulation: 3,
  circular: 4,
};

// Parse a leading numeric out of a metric string ("0.82", "~0.49", "Simulation
// only") for numeric sorting; non-numeric metrics sort last.
function metricSortValue(m: ModelEntry): number {
  const match = m.metricValue.match(/-?\d+(\.\d+)?/);
  return match ? Number(match[0]) : Number.NEGATIVE_INFINITY;
}

// ---------------------------------------------------------------------------
// Small helpers / atoms
// ---------------------------------------------------------------------------

/** An info dot that opens a glossary tooltip; used next to metrics + protocols. */
function GlossaryInfo({ term, side = 'top' }: { term?: string; side?: 'top' | 'bottom' }) {
  if (!term) return null;
  const g = GLOSSARY_BY_TERM[term];
  if (!g) return null;
  return (
    <Tooltip
      side={side}
      label={
        <span style={{ display: 'block' }}>
          <span style={{ fontWeight: 600, display: 'block', marginBottom: 2 }}>
            {g.label}
          </span>
          <span style={{ display: 'block' }}>{g.definition}</span>
          {g.whyItMatters && (
            <span
              style={{
                display: 'block',
                marginTop: 6,
                opacity: 0.82,
                fontStyle: 'italic',
              }}
            >
              {g.whyItMatters}
            </span>
          )}
        </span>
      }
    >
      <Info
        size={12}
        strokeWidth={1.75}
        style={{ color: 'var(--text-tertiary)', display: 'block' }}
        aria-label={`What is ${g.label}?`}
      />
    </Tooltip>
  );
}

/** Map a model status to the StatusBadge tone. */
function statusTone(status: ModelStatus): StatusKind {
  return STATUS_META[status].tone as StatusKind;
}

// ---------------------------------------------------------------------------
// Metric-vs-baseline bar (the honest "does it beat baseline?" mini-chart)
// ---------------------------------------------------------------------------

interface BarDatum {
  label: string;
  value: number;
  /** display string (keeps "~0.49", "0.86" formatting) */
  display: string;
  tone: StatusTone;
  /** the headline / focus bar is emphasized */
  focus?: boolean;
  glossary?: string;
}

const TONE_FILL: Record<StatusTone, string> = {
  nominal: 'var(--nominal)',
  watch: 'var(--watch)',
  alert: 'var(--alert)',
  neutral: 'var(--neutral-status)',
};

/**
 * Build the bar series for a model from its secondary metrics. We only chart
 * models whose numbers live on a shared 0–1 scale (AUROC / F1 / R²); RMSE and
 * non-numeric status rows return null and render a note instead.
 */
function buildBars(m: ModelEntry): { bars: BarDatum[]; scaleNote: string } | null {
  // Curated per-model series so the comparison is always meaningful + honest.
  switch (m.id) {
    case 'hydrovit':
      return {
        scaleNote: 'R² · 0 = predicting the mean',
        bars: [
          { label: 'Water-temp R²', value: 0.729, display: '0.729', tone: 'nominal', focus: true, glossary: 'r2' },
          { label: 'Chl-a (red-edge)', value: 0.32, display: '0.32', tone: 'watch', glossary: 'red_edge' },
          { label: 'Chl-a (4-band)', value: 0.18, display: '0.18', tone: 'neutral', glossary: 'r2' },
        ],
      };
    case 'microbiomenet':
      return {
        scaleNote: 'macro-F1 · geographic holdout',
        bars: [
          { label: 'Taxonomic features', value: 0.676, display: '0.676', tone: 'nominal', focus: true, glossary: 'transferable_representation' },
          { label: 'Raw OTUs', value: 0.405, display: '0.405', tone: 'watch', glossary: 'macro_f1' },
          { label: 'Random-split leak', value: 0.831, display: '0.831', tone: 'alert', glossary: 'data_leakage' },
        ],
      };
    case 'toxigene':
      return {
        scaleNote: 'AUROC · 0.5 = chance',
        bars: [
          { label: 'Within-study', value: 0.86, display: '0.86', tone: 'nominal', focus: true, glossary: 'auroc' },
          { label: 'Cross-study (rank-norm)', value: 0.64, display: '0.64', tone: 'watch', glossary: 'loso' },
          { label: 'Cross-study (raw)', value: 0.55, display: '0.55', tone: 'neutral', glossary: 'loso' },
        ],
      };
    case 'biomotion':
      return {
        scaleNote: 'AUROC · 0.5 = chance',
        bars: [
          { label: 'Novel chemicals', value: 0.84, display: '0.84', tone: 'nominal', focus: true, glossary: 'auroc' },
          { label: 'Grouped GBM', value: 0.85, display: '0.85', tone: 'nominal', glossary: 'auroc' },
          { label: 'Diffusion encoder', value: 0.807, display: '0.807', tone: 'watch', glossary: 'auroc' },
        ],
      };
    case 'fusion':
      return {
        scaleNote: 'AUROC · 0.5 = chance',
        bars: [
          { label: '3-way fusion (in-region)', value: 0.82, display: '0.82', tone: 'nominal', focus: true, glossary: 'auroc' },
          { label: 'Best single modality', value: 0.68, display: '0.68', tone: 'watch', glossary: 'auroc' },
          { label: 'Cross-region holdout', value: 0.49, display: '~0.49', tone: 'alert', glossary: 'cross_region_holdout' },
        ],
      };
    case 'species_health':
      return {
        scaleNote: 'AUROC · 0.5 = chance',
        bars: [
          { label: 'Macro-AUROC', value: 0.949, display: '0.949', tone: 'watch', focus: true, glossary: 'auroc' },
          { label: 'Widespread mussel (genuine)', value: 0.88, display: '0.88', tone: 'nominal', glossary: 'auroc' },
          { label: 'Environment-only floor', value: 0.85, display: '~0.85', tone: 'neutral', glossary: 'auroc' },
          { label: 'Circular (retracted)', value: 1.0, display: '1.000', tone: 'alert', glossary: 'circular_label' },
        ],
      };
    case 'digital_twin':
      return {
        scaleNote: 'R² · persistence is the bar to beat',
        bars: [
          { label: '1-day horizon', value: 0.688, display: '0.688', tone: 'watch', focus: true, glossary: 'r2' },
          { label: 'Persistence baseline', value: 0.92, display: '0.92', tone: 'nominal', glossary: 'persistence_baseline' },
          { label: 'Overall (all horizons)', value: 0.07, display: '0.07', tone: 'alert', glossary: 'r2' },
        ],
      };
    case 'sentinel_lite':
      return {
        scaleNote: 'R² · spatial holdout (399 stations)',
        bars: [
          { label: 'Temperature', value: 0.776, display: '0.776', tone: 'nominal', focus: true, glossary: 'r2' },
          { label: 'Dissolved oxygen', value: 0.463, display: '0.463', tone: 'watch', glossary: 'r2' },
          { label: 'Sp. conductance', value: 0.442, display: '0.442', tone: 'watch', glossary: 'r2' },
          { label: 'Turbidity', value: 0.181, display: '0.181', tone: 'neutral', glossary: 'r2' },
        ],
      };
    default:
      // AquaSSM (RMSE) and simulation/circular rows: no shared 0–1 skill
      // chart; DetailPanel renders an explanatory note + the secondary metrics.
      return null;
  }
}

function MetricBar({ datum, scaleMax }: { datum: BarDatum; scaleMax: number }) {
  const pct = Math.max(0, Math.min(100, (datum.value / scaleMax) * 100));
  return (
    <div style={{ display: 'grid', gridTemplateColumns: '170px 1fr 56px', alignItems: 'center', gap: 'var(--s-3)' }}>
      <span
        style={{
          fontSize: 'var(--fs-12)',
          color: datum.focus ? 'var(--text-primary)' : 'var(--text-secondary)',
          fontWeight: datum.focus ? 600 : 400,
          display: 'inline-flex',
          alignItems: 'center',
          gap: 'var(--s-2)',
          minWidth: 0,
        }}
      >
        <span
          style={{
            overflow: 'hidden',
            textOverflow: 'ellipsis',
            whiteSpace: 'nowrap',
          }}
          title={datum.label}
        >
          {datum.label}
        </span>
        <GlossaryInfo term={datum.glossary} />
      </span>
      <span
        aria-hidden
        style={{
          position: 'relative',
          height: 10,
          background: 'var(--bg-inset)',
          borderRadius: 'var(--r-xs)',
          overflow: 'hidden',
        }}
      >
        <span
          style={{
            position: 'absolute',
            inset: 0,
            width: `${pct}%`,
            background: TONE_FILL[datum.tone],
            borderRadius: 'var(--r-xs)',
            transition: 'width var(--dur-slow) var(--ease-out)',
            opacity: datum.focus ? 1 : 0.72,
          }}
        />
      </span>
      <span
        className="mono"
        style={{
          fontSize: 'var(--fs-12)',
          textAlign: 'right',
          color: datum.tone === 'alert' ? 'var(--alert)' : 'var(--text-primary)',
          fontWeight: datum.focus ? 600 : 400,
        }}
      >
        {datum.display}
      </span>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Expanded detail row
// ---------------------------------------------------------------------------

function DetailPanel({ m }: { m: ModelEntry }) {
  const chart = buildBars(m);
  const scaleMax = chart ? Math.max(1, ...chart.bars.map((b) => b.value)) : 1;

  return (
    <div
      style={{
        padding: 'var(--s-6) var(--s-4) var(--s-6) var(--s-8)',
        background: 'var(--bg-subtle)',
        borderTop: '1px solid var(--border-subtle)',
        display: 'grid',
        gridTemplateColumns: 'minmax(0, 1.15fr) minmax(0, 1fr)',
        gap: 'var(--s-8)',
        animation: 'fade-in-up var(--dur-base) var(--ease-out)',
      }}
      className="models-detail-grid"
    >
      {/* Left column: honest note, correction log, architecture */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--s-4)', minWidth: 0 }}>
        <div>
          <div className="eyebrow" style={{ marginBottom: 'var(--s-2)' }}>
            Honest note
          </div>
          <p
            style={{
              margin: 0,
              fontSize: 'var(--fs-13)',
              lineHeight: 'var(--lh-normal)',
              color: 'var(--text-secondary)',
              maxWidth: '62ch',
            }}
          >
            {m.honestNote}
          </p>
        </div>

        {m.corrects && (
          <div
            style={{
              padding: 'var(--s-3) var(--s-4)',
              background: 'var(--alert-wash)',
              border: '1px solid var(--alert-border)',
              borderRadius: 'var(--r-md)',
            }}
          >
            <div
              className="eyebrow"
              style={{ color: 'var(--alert)', marginBottom: 'var(--s-2)' }}
            >
              Corrected during integrity remediation
            </div>
            <div
              style={{
                display: 'flex',
                alignItems: 'baseline',
                gap: 'var(--s-3)',
                flexWrap: 'wrap',
                marginBottom: 'var(--s-2)',
              }}
            >
              <span
                className="mono"
                style={{
                  fontSize: 'var(--fs-16)',
                  color: 'var(--alert)',
                  textDecoration: 'line-through',
                  textDecorationThickness: '1.5px',
                }}
              >
                {m.corrects.oldValue}
              </span>
              <ArrowRightGlyph />
              <span
                className="mono"
                style={{ fontSize: 'var(--fs-16)', color: 'var(--text-primary)', fontWeight: 600 }}
              >
                {m.metricValue}
              </span>
            </div>
            <p
              style={{
                margin: 0,
                fontSize: 'var(--fs-12)',
                lineHeight: 'var(--lh-normal)',
                color: 'var(--text-secondary)',
              }}
            >
              {m.corrects.reason}
            </p>
          </div>
        )}

        {m.architecture && (
          <div>
            <div className="eyebrow" style={{ marginBottom: 'var(--s-2)' }}>
              Architecture
            </div>
            <p
              style={{
                margin: 0,
                fontSize: 'var(--fs-13)',
                lineHeight: 'var(--lh-normal)',
                color: 'var(--text-secondary)',
                maxWidth: '62ch',
              }}
            >
              {m.architecture}
            </p>
          </div>
        )}
      </div>

      {/* Right column: metric-vs-baseline bar + secondary metric grid */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--s-5)', minWidth: 0 }}>
        {chart ? (
          <div>
            <div
              style={{
                display: 'flex',
                alignItems: 'baseline',
                justifyContent: 'space-between',
                gap: 'var(--s-3)',
                marginBottom: 'var(--s-3)',
              }}
            >
              <span className="eyebrow">Metric vs. baseline</span>
              <span
                style={{
                  fontSize: 'var(--fs-11)',
                  color: 'var(--text-tertiary)',
                  fontFamily: 'var(--font-sans)',
                }}
              >
                {chart.scaleNote}
              </span>
            </div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--s-3)' }}>
              {chart.bars.map((b) => (
                <MetricBar key={b.label} datum={b} scaleMax={scaleMax} />
              ))}
            </div>
          </div>
        ) : (
          <div
            style={{
              padding: 'var(--s-4)',
              background: 'var(--bg-inset)',
              border: '1px solid var(--border-subtle)',
              borderRadius: 'var(--r-md)',
              fontSize: 'var(--fs-12)',
              color: 'var(--text-tertiary)',
              lineHeight: 'var(--lh-normal)',
            }}
          >
            {m.status === 'simulation'
              ? 'No held-out skill chart: this model is trained on synthetic labels. It is a proof-of-concept for the architecture, not a validated detector.'
              : m.status === 'circular'
                ? 'No held-out skill chart: the labels are a closed-form function of the inputs, so any score recovers a known rule rather than predictive skill.'
                : 'Reported as reconstruction error (RMSE, lower is better) — not on a shared 0–1 skill scale. See the secondary metrics below.'}
          </div>
        )}

        {m.secondaryMetrics && m.secondaryMetrics.length > 0 && (
          <div>
            <div className="eyebrow" style={{ marginBottom: 'var(--s-3)' }}>
              Secondary metrics
            </div>
            <div
              style={{
                display: 'grid',
                gridTemplateColumns: 'repeat(auto-fill, minmax(150px, 1fr))',
                gap: 'var(--s-3)',
              }}
            >
              {m.secondaryMetrics.map((s) => (
                <div
                  key={s.label}
                  style={{
                    padding: 'var(--s-3)',
                    background: 'var(--bg-surface)',
                    border: '1px solid var(--border-subtle)',
                    borderRadius: 'var(--r-sm)',
                  }}
                >
                  <div
                    style={{
                      display: 'flex',
                      alignItems: 'center',
                      gap: 'var(--s-2)',
                      fontSize: 'var(--fs-11)',
                      color: 'var(--text-tertiary)',
                      marginBottom: 2,
                    }}
                  >
                    <span
                      style={{
                        overflow: 'hidden',
                        textOverflow: 'ellipsis',
                        whiteSpace: 'nowrap',
                      }}
                      title={s.label}
                    >
                      {s.label}
                    </span>
                    <GlossaryInfo term={s.glossary} />
                  </div>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--s-1)' }}>
                    <span
                      className="mono"
                      style={{ fontSize: 'var(--fs-16)', color: 'var(--text-primary)', fontWeight: 500 }}
                    >
                      {s.value}
                    </span>
                    <CopyButton value={s.value} label={s.label} size={12} />
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

function ArrowRightGlyph() {
  return (
    <span
      aria-hidden
      style={{
        color: 'var(--text-quaternary)',
        fontFamily: 'var(--font-mono)',
        fontSize: 'var(--fs-14)',
      }}
    >
      →
    </span>
  );
}

// ---------------------------------------------------------------------------
// Filter / control atoms
// ---------------------------------------------------------------------------

function FilterChip({
  active,
  onClick,
  children,
  tone,
  count,
}: {
  active: boolean;
  onClick: () => void;
  children: ReactNode;
  tone?: string;
  count?: number;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      aria-pressed={active}
      style={{
        display: 'inline-flex',
        alignItems: 'center',
        gap: 'var(--s-2)',
        padding: '4px 10px',
        borderRadius: 'var(--r-pill)',
        border: `1px solid ${active ? 'var(--accent)' : 'var(--border-strong)'}`,
        background: active ? 'var(--accent-wash)' : 'var(--bg-surface)',
        color: active ? 'var(--accent-strong)' : 'var(--text-secondary)',
        fontSize: 'var(--fs-12)',
        fontWeight: active ? 600 : 500,
        fontFamily: 'var(--font-sans)',
        lineHeight: 1.6,
        transition:
          'background var(--dur-fast) var(--ease-standard), border-color var(--dur-fast), color var(--dur-fast)',
      }}
      onMouseEnter={(e) => {
        if (!active) e.currentTarget.style.background = 'var(--bg-hover)';
      }}
      onMouseLeave={(e) => {
        if (!active) e.currentTarget.style.background = 'var(--bg-surface)';
      }}
    >
      {tone && (
        <span
          aria-hidden
          style={{ width: 6, height: 6, borderRadius: '50%', background: tone, flexShrink: 0 }}
        />
      )}
      {children}
      {count !== undefined && (
        <span
          className="mono"
          style={{
            fontSize: 'var(--fs-11)',
            color: active ? 'var(--accent-strong)' : 'var(--text-tertiary)',
            fontWeight: 500,
          }}
        >
          {count}
        </span>
      )}
    </button>
  );
}

/** A sortable table header cell. */
function SortHeader({
  label,
  columnKey,
  sortKey,
  sortDir,
  onSort,
  align = 'left',
  tooltip,
}: {
  label: string;
  columnKey: SortKey;
  sortKey: SortKey;
  sortDir: SortDir;
  onSort: (k: SortKey) => void;
  align?: 'left' | 'right';
  tooltip?: ReactNode;
}) {
  const active = sortKey === columnKey;
  const Icon = active ? (sortDir === 'asc' ? ArrowUp : ArrowDown) : ArrowUpDown;
  return (
    <button
      type="button"
      onClick={() => onSort(columnKey)}
      aria-label={`Sort by ${label}${active ? (sortDir === 'asc' ? ', ascending' : ', descending') : ''}`}
      style={{
        display: 'inline-flex',
        alignItems: 'center',
        gap: 'var(--s-2)',
        flexDirection: align === 'right' ? 'row-reverse' : 'row',
        color: active ? 'var(--text-primary)' : 'var(--text-tertiary)',
        fontFamily: 'var(--font-sans)',
        fontSize: 'var(--fs-11)',
        fontWeight: 600,
        letterSpacing: 'var(--tracking-wide)',
        textTransform: 'uppercase',
      }}
    >
      <span style={{ display: 'inline-flex', alignItems: 'center', gap: 'var(--s-1)' }}>
        {label}
        {tooltip}
      </span>
      <Icon
        size={12}
        strokeWidth={2}
        style={{ color: active ? 'var(--accent)' : 'var(--text-quaternary)', flexShrink: 0 }}
        aria-hidden
      />
    </button>
  );
}

// ---------------------------------------------------------------------------
// Page
// ---------------------------------------------------------------------------

const thBase: CSSProperties = {
  position: 'sticky',
  top: 0,
  zIndex: 1,
  textAlign: 'left',
  padding: 'var(--s-3) var(--s-4)',
  whiteSpace: 'nowrap',
  background: 'var(--bg-subtle)',
  borderBottom: '1px solid var(--border-strong)',
};

const tdBase: CSSProperties = {
  padding: 'var(--s-4)',
  verticalAlign: 'top',
  color: 'var(--text-primary)',
};

type KindFilter = ModelKind | 'all';
type StatusFilter = ModelStatus | 'all';

export default function Models() {
  const [query, setQuery] = useState('');
  const [kindFilter, setKindFilter] = useState<KindFilter>('all');
  const [statusFilter, setStatusFilter] = useState<StatusFilter>('all');
  const [sortKey, setSortKey] = useState<SortKey>('kind');
  const [sortDir, setSortDir] = useState<SortDir>('asc');
  const [expanded, setExpanded] = useState<string | null>('fusion'); // headline row open by default

  // counts per kind / status for the filter chips
  const kindCounts = useMemo(() => {
    const c: Record<string, number> = {};
    for (const m of MODELS) c[m.kind] = (c[m.kind] ?? 0) + 1;
    return c;
  }, []);
  const statusCounts = useMemo(() => {
    const c: Record<string, number> = {};
    for (const m of MODELS) c[m.status] = (c[m.status] ?? 0) + 1;
    return c;
  }, []);

  function onSort(k: SortKey) {
    if (k === sortKey) {
      setSortDir((d) => (d === 'asc' ? 'desc' : 'asc'));
    } else {
      setSortKey(k);
      // sensible default direction per column
      setSortDir(k === 'metric' ? 'desc' : 'asc');
    }
  }

  const rows = useMemo(() => {
    const q = query.trim().toLowerCase();
    let list = MODELS.filter((m) => {
      if (kindFilter !== 'all' && m.kind !== kindFilter) return false;
      if (statusFilter !== 'all' && m.status !== statusFilter) return false;
      if (!q) return true;
      const hay = `${m.name} ${m.role} ${m.task} ${m.dataSource} ${m.metricLabel}`.toLowerCase();
      return hay.includes(q);
    });

    const dir = sortDir === 'asc' ? 1 : -1;
    list = [...list].sort((a, b) => {
      let cmp = 0;
      switch (sortKey) {
        case 'name':
          cmp = a.name.localeCompare(b.name);
          break;
        case 'kind':
          cmp =
            KIND_ORDER.indexOf(a.kind) - KIND_ORDER.indexOf(b.kind) ||
            a.name.localeCompare(b.name);
          break;
        case 'metric':
          cmp = metricSortValue(a) - metricSortValue(b);
          break;
        case 'status':
          cmp = STATUS_RANK[a.status] - STATUS_RANK[b.status] || a.name.localeCompare(b.name);
          break;
      }
      return cmp * dir;
    });
    return list;
  }, [query, kindFilter, statusFilter, sortKey, sortDir]);

  const filtersActive = kindFilter !== 'all' || statusFilter !== 'all' || query.trim() !== '';

  function resetFilters() {
    setKindFilter('all');
    setStatusFilter('all');
    setQuery('');
  }

  const columnCount = 7;

  return (
    <>
      <SectionHeader
        as="h1"
        eyebrow="Model roster"
        title="Every model, one real metric, one honest holdout"
        description="The complete SENTINEL catalog — five modality encoders, the multimodal
        fusion head, downstream tasks, and the deployment model. Each row reports a single
        verified metric measured under its stated holdout protocol. Expand a row for the honest
        note, the metric-vs-baseline comparison, and any figure corrected during integrity
        remediation. Retracted headline numbers are never shown as results."
        actions={
          <a href="/docs" style={{ display: 'inline-flex', alignItems: 'center', gap: 4 }}>
            Holdout protocols
            <ChevronRight size={14} strokeWidth={1.9} />
          </a>
        }
      />

      <Callout tone="verified" title="How to read this table">
        Every metric and protocol has an{' '}
        <Info size={12} strokeWidth={2} style={{ verticalAlign: '-1px', color: 'var(--text-tertiary)' }} />{' '}
        tooltip that defines it and states why it matters. Status encodes the honesty tier, not
        a grade: <strong>validated</strong> means genuine held-out skill; <strong>region-limited</strong>{' '}
        and <strong>genuine ceiling</strong> are real but bounded; <strong>simulation</strong> and{' '}
        <strong>circular</strong> are labeled as not-yet-validated by design.
      </Callout>

      {/* --- controls ------------------------------------------------------ */}
      <div
        style={{
          display: 'flex',
          flexWrap: 'wrap',
          gap: 'var(--s-5)',
          alignItems: 'flex-end',
          justifyContent: 'space-between',
          margin: 'var(--s-7) 0 var(--s-5)',
        }}
      >
        <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--s-4)', minWidth: 0 }}>
          {/* kind filter */}
          <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--s-3)', flexWrap: 'wrap' }}>
            <span className="eyebrow" style={{ marginRight: 'var(--s-1)' }}>
              Group
            </span>
            <FilterChip active={kindFilter === 'all'} onClick={() => setKindFilter('all')} count={MODELS.length}>
              All
            </FilterChip>
            {KIND_ORDER.map((k) => (
              <FilterChip
                key={k}
                active={kindFilter === k}
                onClick={() => setKindFilter(kindFilter === k ? 'all' : k)}
                tone={KIND_META[k].color}
                count={kindCounts[k]}
              >
                {KIND_META[k].plural}
              </FilterChip>
            ))}
          </div>
          {/* status filter */}
          <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--s-3)', flexWrap: 'wrap' }}>
            <span className="eyebrow" style={{ marginRight: 'var(--s-1)' }}>
              Status
            </span>
            {(Object.keys(STATUS_META) as ModelStatus[]).map((s) => (
              <FilterChip
                key={s}
                active={statusFilter === s}
                onClick={() => setStatusFilter(statusFilter === s ? 'all' : s)}
                tone={TONE_FILL[STATUS_META[s].tone]}
                count={statusCounts[s]}
              >
                {STATUS_META[s].label}
              </FilterChip>
            ))}
          </div>
        </div>

        {/* search */}
        <div
          className="field-focus"
          style={{
            display: 'inline-flex',
            alignItems: 'center',
            gap: 'var(--s-2)',
            padding: '6px 10px',
            background: 'var(--bg-surface)',
            border: '1px solid var(--border-strong)',
            borderRadius: 'var(--r-md)',
            minWidth: 220,
          }}
        >
          <Search size={14} strokeWidth={1.9} style={{ color: 'var(--text-tertiary)', flexShrink: 0 }} aria-hidden />
          <input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Filter models, tasks, data…"
            aria-label="Filter the model table"
            style={{
              flex: 1,
              minWidth: 0,
              border: 'none',
              outline: 'none',
              background: 'transparent',
              color: 'var(--text-primary)',
              fontFamily: 'var(--font-sans)',
              fontSize: 'var(--fs-13)',
            }}
          />
          {query && (
            <button
              type="button"
              onClick={() => setQuery('')}
              aria-label="Clear filter"
              style={{ display: 'inline-flex', color: 'var(--text-tertiary)' }}
            >
              <X size={14} strokeWidth={1.9} />
            </button>
          )}
        </div>
      </div>

      {/* result count + reset */}
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          gap: 'var(--s-4)',
          marginBottom: 'var(--s-3)',
          fontSize: 'var(--fs-12)',
          color: 'var(--text-tertiary)',
        }}
      >
        <span>
          <span className="mono" style={{ color: 'var(--text-secondary)' }}>
            {rows.length}
          </span>{' '}
          of{' '}
          <span className="mono" style={{ color: 'var(--text-secondary)' }}>
            {MODELS.length}
          </span>{' '}
          models
        </span>
        {filtersActive && (
          <button
            type="button"
            onClick={resetFilters}
            style={{
              display: 'inline-flex',
              alignItems: 'center',
              gap: 4,
              color: 'var(--accent)',
              fontSize: 'var(--fs-12)',
              fontWeight: 500,
            }}
          >
            <X size={12} strokeWidth={2} /> Clear filters
          </button>
        )}
      </div>

      {/* --- the table ----------------------------------------------------- */}
      <div
        style={{
          border: '1px solid var(--border-subtle)',
          borderRadius: 'var(--r-md)',
          overflow: 'hidden',
          background: 'var(--bg-surface)',
        }}
      >
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 'var(--fs-13)', minWidth: 860 }}>
            <thead>
              <tr>
                <th style={{ ...thBase, width: 30 }} aria-label="Expand" />
                <th style={thBase}>
                  <SortHeader label="Model" columnKey="name" sortKey={sortKey} sortDir={sortDir} onSort={onSort} />
                </th>
                <th style={thBase}>
                  <span
                    className="eyebrow"
                    style={{ color: 'var(--text-tertiary)', letterSpacing: 'var(--tracking-wide)' }}
                  >
                    Task
                  </span>
                </th>
                <th style={{ ...thBase, textAlign: 'right' }}>
                  <SortHeader
                    label="Real metric"
                    columnKey="metric"
                    sortKey={sortKey}
                    sortDir={sortDir}
                    onSort={onSort}
                    align="right"
                    tooltip={
                      <Tooltip
                        label="One headline metric per model, measured under the holdout in the next column. Sorts by numeric value; RMSE and status rows sort last."
                      >
                        <Info size={11} strokeWidth={1.75} style={{ color: 'var(--text-tertiary)', display: 'block' }} aria-label="About the metric column" />
                      </Tooltip>
                    }
                  />
                </th>
                <th style={thBase}>
                  <span
                    className="eyebrow"
                    style={{ display: 'inline-flex', alignItems: 'center', gap: 'var(--s-1)', color: 'var(--text-tertiary)' }}
                  >
                    Holdout protocol
                    <Tooltip label="The evaluation split under which the metric was measured. Stricter splits (geographic, cross-region, leave-one-study-out) are the honest test of transfer.">
                      <Info size={11} strokeWidth={1.75} style={{ color: 'var(--text-tertiary)', display: 'block' }} aria-label="About holdout protocols" />
                    </Tooltip>
                  </span>
                </th>
                <th style={thBase}>
                  <span className="eyebrow" style={{ color: 'var(--text-tertiary)' }}>
                    Data
                  </span>
                </th>
                <th style={{ ...thBase, textAlign: 'left' }}>
                  <SortHeader label="Status" columnKey="status" sortKey={sortKey} sortDir={sortDir} onSort={onSort} />
                </th>
              </tr>
            </thead>
            <tbody>
              {rows.length === 0 ? (
                <tr>
                  <td
                    colSpan={columnCount}
                    style={{
                      padding: 'var(--s-9)',
                      textAlign: 'center',
                      color: 'var(--text-tertiary)',
                    }}
                  >
                    <div style={{ marginBottom: 'var(--s-3)' }}>
                      No models match these filters.
                    </div>
                    <button
                      type="button"
                      onClick={resetFilters}
                      style={{ color: 'var(--accent)', fontWeight: 500, fontSize: 'var(--fs-13)' }}
                    >
                      Clear filters
                    </button>
                  </td>
                </tr>
              ) : (
                rows.map((m, i) => {
                  const isOpen = expanded === m.id;
                  const meta = STATUS_META[m.status];
                  const zebra = i % 2 === 1;
                  const rowBg = isOpen
                    ? 'var(--accent-wash)'
                    : zebra
                      ? 'var(--bg-subtle)'
                      : 'var(--bg-surface)';
                  const lastRow = i === rows.length - 1;
                  return (
                    <ModelRow
                      key={m.id}
                      m={m}
                      meta={meta}
                      isOpen={isOpen}
                      rowBg={rowBg}
                      zebra={zebra}
                      lastRow={lastRow}
                      columnCount={columnCount}
                      onToggle={() => setExpanded(isOpen ? null : m.id)}
                    />
                  );
                })
              )}
            </tbody>
          </table>
        </div>
      </div>

      {/* --- cross-cutting finding ---------------------------------------- */}
      <div style={{ marginTop: 'var(--s-8)' }}>
        <Callout tone="info" title="The cross-cutting finding">
          Across encoders, the same lever recovers out-of-distribution skill: a{' '}
          <strong>transferable representation</strong> — higher taxonomy for microbiome,
          rank-normalized genes for toxicogenomics, red-edge spectral indices for imagery — generalizes
          where raw, region-specific features collapse. MicroBiomeNet 0.405 → 0.676 and HydroViT chl-a
          0.18 → 0.32 are the same story. That transferability is SENTINEL's central methodological
          contribution.
        </Callout>
      </div>

      {/* legend */}
      <div
        style={{
          marginTop: 'var(--s-6)',
          display: 'flex',
          flexWrap: 'wrap',
          gap: 'var(--s-5)',
          fontSize: 'var(--fs-12)',
          color: 'var(--text-tertiary)',
        }}
      >
        {(Object.keys(STATUS_META) as ModelStatus[]).map((s) => (
          <span key={s} style={{ display: 'inline-flex', alignItems: 'center', gap: 'var(--s-2)' }}>
            <StatusBadge kind={statusTone(s)}>{STATUS_META[s].label}</StatusBadge>
            <span style={{ maxWidth: '28ch' }}>{STATUS_META[s].description}</span>
          </span>
        ))}
      </div>

      {/* Detail-grid responsiveness + row-focus outline (scoped to this page). */}
      <style>{`
        @media (max-width: 760px) {
          .models-detail-grid { grid-template-columns: 1fr !important; }
        }
        .models-row-toggle:focus-visible {
          outline: 2px solid var(--border-focus);
          outline-offset: -2px;
        }
      `}</style>
    </>
  );
}

// ---------------------------------------------------------------------------
// A single (collapsible) model row + its detail panel.
// ---------------------------------------------------------------------------

function ModelRow({
  m,
  meta,
  isOpen,
  rowBg,
  zebra,
  lastRow,
  columnCount,
  onToggle,
}: {
  m: ModelEntry;
  meta: { label: string; tone: StatusTone; description: string };
  isOpen: boolean;
  rowBg: string;
  zebra: boolean;
  lastRow: boolean;
  columnCount: number;
  onToggle: () => void;
}) {
  const kindMeta = KIND_META[m.kind];
  const border = lastRow && !isOpen ? 'none' : '1px solid var(--border-subtle)';

  const baseBg = zebra ? 'var(--bg-subtle)' : 'var(--bg-surface)';

  return (
    <>
      <tr
        onClick={onToggle}
        style={{
          background: rowBg,
          cursor: 'pointer',
          transition: 'background var(--dur-fast) var(--ease-standard)',
        }}
        onMouseEnter={(e) => {
          if (!isOpen) e.currentTarget.style.background = 'var(--bg-hover)';
        }}
        onMouseLeave={(e) => {
          e.currentTarget.style.background = isOpen ? 'var(--accent-wash)' : baseBg;
        }}
      >
        {/* expand chevron */}
        <td style={{ ...tdBase, borderBottom: border, textAlign: 'center', paddingLeft: 'var(--s-3)', paddingRight: 0 }}>
          <button
            type="button"
            className="models-row-toggle"
            onClick={(e) => {
              e.stopPropagation();
              onToggle();
            }}
            aria-expanded={isOpen}
            aria-label={isOpen ? `Collapse ${m.name} detail` : `Expand ${m.name} detail`}
            style={{
              display: 'inline-flex',
              color: 'var(--text-tertiary)',
              borderRadius: 'var(--r-xs)',
            }}
          >
            <ChevronRight
              size={15}
              strokeWidth={2}
              style={{
                transform: isOpen ? 'rotate(90deg)' : 'none',
                transition: 'transform var(--dur-fast) var(--ease-standard)',
              }}
            />
          </button>
        </td>

        {/* model name + role, with kind tick */}
        <td style={{ ...tdBase, borderBottom: border }}>
          <div style={{ display: 'flex', gap: 'var(--s-3)', alignItems: 'flex-start' }}>
            <span
              aria-hidden
              title={kindMeta.label}
              style={{
                width: 3,
                alignSelf: 'stretch',
                minHeight: 30,
                borderRadius: 'var(--r-pill)',
                background: kindMeta.color,
                flexShrink: 0,
              }}
            />
            <span style={{ minWidth: 0 }}>
              <span
                className="mono"
                style={{
                  fontSize: 'var(--fs-13)',
                  fontWeight: 600,
                  color: 'var(--text-primary)',
                  display: 'block',
                }}
              >
                {m.name}
              </span>
              <span style={{ fontSize: 'var(--fs-12)', color: 'var(--text-tertiary)' }}>
                {m.role}
              </span>
            </span>
          </div>
        </td>

        {/* task */}
        <td
          style={{
            ...tdBase,
            borderBottom: border,
            maxWidth: 260,
            fontSize: 'var(--fs-13)',
            color: 'var(--text-secondary)',
            lineHeight: 'var(--lh-normal)',
          }}
        >
          {m.task}
        </td>

        {/* real metric (mono) */}
        <td style={{ ...tdBase, borderBottom: border, textAlign: 'right', whiteSpace: 'nowrap' }}>
          <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'flex-end', gap: 2 }}>
            <span style={{ display: 'inline-flex', alignItems: 'center', gap: 'var(--s-2)' }}>
              <GlossaryInfo term={m.metricGlossary} side="bottom" />
              <span
                className="mono"
                style={{
                  fontSize: 'var(--fs-16)',
                  fontWeight: 600,
                  color:
                    meta.tone === 'alert'
                      ? 'var(--alert)'
                      : m.metricValue.length > 8
                        ? 'var(--text-tertiary)'
                        : 'var(--text-primary)',
                  fontVariantNumeric: 'tabular-nums',
                }}
              >
                {m.metricValue}
              </span>
              <span onClick={(e) => e.stopPropagation()} style={{ display: 'inline-flex' }}>
                <CopyButton value={`${m.metricValue} (${m.metricLabel})`} label={m.metricLabel} size={12} />
              </span>
            </span>
            <span style={{ fontSize: 'var(--fs-11)', color: 'var(--text-tertiary)' }}>
              {m.metricLabel}
            </span>
          </div>
        </td>

        {/* holdout protocol */}
        <td
          style={{
            ...tdBase,
            borderBottom: border,
            maxWidth: 220,
            fontSize: 'var(--fs-12)',
            color: 'var(--text-secondary)',
          }}
        >
          <span style={{ display: 'inline-flex', alignItems: 'center', gap: 'var(--s-2)' }}>
            <span style={{ lineHeight: 'var(--lh-snug)' }}>{m.holdout}</span>
            <GlossaryInfo term={m.holdoutGlossary} />
          </span>
        </td>

        {/* data source */}
        <td
          style={{
            ...tdBase,
            borderBottom: border,
            maxWidth: 230,
            fontSize: 'var(--fs-12)',
            color: 'var(--text-tertiary)',
            lineHeight: 'var(--lh-snug)',
          }}
        >
          {m.dataSource}
        </td>

        {/* status badge */}
        <td style={{ ...tdBase, borderBottom: border, whiteSpace: 'nowrap' }}>
          <Tooltip label={meta.description} side="bottom">
            <StatusBadge kind={meta.tone as StatusKind}>{meta.label}</StatusBadge>
          </Tooltip>
        </td>
      </tr>

      {isOpen && (
        <tr>
          <td colSpan={columnCount} style={{ padding: 0, borderBottom: lastRow ? 'none' : '1px solid var(--border-subtle)' }}>
            <DetailPanel m={m} />
          </td>
        </tr>
      )}
    </>
  );
}
