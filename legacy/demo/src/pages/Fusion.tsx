import {
  useEffect,
  useMemo,
  useRef,
  useState,
  type CSSProperties,
  type ReactNode,
} from 'react';
import {
  Satellite,
  FlaskConical,
  Thermometer,
  Layers,
  Dna,
  GitBranch,
  Droplet,
  ArrowRight,
  Info,
  Waves,
  Clock,
} from 'lucide-react';
import {
  SectionHeader,
  Callout,
  Metric,
  DataTable,
  StatusBadge,
  Tooltip,
  CopyButton,
  type Column,
} from '../ui';
import {
  MODALITIES,
  FUSION_DATASET,
  FUSION_THRESHOLDS,
  FUSION_HEADLINE,
  CROSS_REGION_TRANSFER,
  FUSION_MECHANISM,
  FUSION_REDUNDANCY,
  FUSION_SUMMARY,
  BLOOM_EARLY_WARNING,
  type ModalityKey,
  type ThresholdRow,
  type RedundancyRow,
} from '../content/fusion';

/* ===========================================================================
   Fusion — the headline scientific story, told honestly.

   Microcystin is an optically-invisible cyanotoxin. Three complementary
   modalities (satellite optics, in-situ pigment, physicochemical drivers) each
   add non-redundant signal, so 3-way fusion beats every single modality under
   STATION-grouped CV (0.82 vs 0.68 at the WHO drinking threshold). BUT the
   pigment→toxin relationship depends on the toxin-producing strain fraction
   (the mcyD genotype), which none of these modalities can see — so under a true
   CROSS-REGION (leave-one-HUC-out) holdout, every model collapses to chance
   (~0.49). On chl-a/DO, fusion only ties the best single. We present all three.

   Everything below reuses the verified numbers in src/content/fusion.ts (which
   are checked against README.md + results/trackB_fusion/*.json). Charts are
   small flat SVG components — no gradients, no glow.
   ========================================================================== */

// ---- modality palette + icons (line icons, no fills) ----------------------
const MODALITY_META: Record<
  ModalityKey,
  { color: string; Icon: typeof Satellite }
> = {
  satellite: { color: 'var(--viz-satellite)', Icon: Satellite },
  pigment: { color: 'var(--viz-sensor)', Icon: FlaskConical },
  drivers: { color: 'var(--viz-microbial)', Icon: Thermometer },
};
const FUSION_COLOR = 'var(--accent)';

const fmt = (n: number, d = 3) => n.toFixed(d);
const signed = (n: number, d = 3) => (n >= 0 ? '+' : '−') + Math.abs(n).toFixed(d);

// ===========================================================================
// Reduced-motion hook — gate entrance/count animations.
// ===========================================================================
function useReducedMotion(): boolean {
  const [reduced, setReduced] = useState(false);
  useEffect(() => {
    const mq = window.matchMedia?.('(prefers-reduced-motion: reduce)');
    if (!mq) return;
    setReduced(mq.matches);
    const on = () => setReduced(mq.matches);
    mq.addEventListener?.('change', on);
    return () => mq.removeEventListener?.('change', on);
  }, []);
  return reduced;
}

// Reveal-on-first-view: adds a class once the element enters the viewport, so
// bars grow from baseline. Disabled under reduced motion.
function useInView<T extends HTMLElement>(reduced: boolean) {
  const ref = useRef<T | null>(null);
  const [seen, setSeen] = useState(false);
  useEffect(() => {
    if (reduced) {
      setSeen(true);
      return;
    }
    const el = ref.current;
    if (!el || typeof IntersectionObserver === 'undefined') {
      setSeen(true);
      return;
    }
    const io = new IntersectionObserver(
      (entries) => {
        if (entries.some((e) => e.isIntersecting)) {
          setSeen(true);
          io.disconnect();
        }
      },
      { threshold: 0.25 },
    );
    io.observe(el);
    return () => io.disconnect();
  }, [reduced]);
  return { ref, seen };
}

// ===========================================================================
// Card — the shared flat panel primitive for this page.
// ===========================================================================
function Card({
  children,
  style,
  pad = true,
}: {
  children: ReactNode;
  style?: CSSProperties;
  pad?: boolean;
}) {
  return (
    <div
      style={{
        background: 'var(--bg-surface)',
        border: '1px solid var(--border-subtle)',
        borderRadius: 'var(--r-lg)',
        padding: pad ? 'var(--s-6)' : 0,
        ...style,
      }}
    >
      {children}
    </div>
  );
}

// A hairline-underlined mini header inside a card.
function CardTitle({
  children,
  tip,
  right,
}: {
  children: ReactNode;
  tip?: ReactNode;
  right?: ReactNode;
}) {
  return (
    <div
      style={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        gap: 'var(--s-4)',
        marginBottom: 'var(--s-4)',
      }}
    >
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: 'var(--s-2)',
          fontSize: 'var(--fs-13)',
          fontWeight: 600,
          color: 'var(--text-primary)',
        }}
      >
        {children}
        {tip && (
          <Tooltip label={tip}>
            <Info
              size={12}
              strokeWidth={1.75}
              style={{ color: 'var(--text-tertiary)', display: 'block' }}
              aria-label="More detail"
            />
          </Tooltip>
        )}
      </div>
      {right}
    </div>
  );
}

// ===========================================================================
// AUROC bar chart — flat SVG, horizontal, with 95% CI whiskers + a chance line.
// Bars grow on first view; respects reduced motion. This is the centerpiece.
// ===========================================================================
interface BarDatum {
  key: string;
  label: string;
  value: number;
  ci?: [number, number];
  color: string;
  isFusion?: boolean;
  IconEl?: typeof Satellite;
  /** True when this modality was not separately evaluated at this threshold — render as not-measured, never as a real value. */
  notMeasured?: boolean;
}

function AurocBarChart({
  data,
  reduced,
  domain = [0.4, 0.9],
}: {
  data: BarDatum[];
  reduced: boolean;
  domain?: [number, number];
}) {
  const { ref, seen } = useInView<HTMLDivElement>(reduced);
  const [lo, hi] = domain;
  const span = hi - lo;

  // layout geometry (viewBox units == px at 100% width)
  const W = 640;
  const rowH = 46;
  const padTop = 14;
  const padBottom = 34;
  const labelW = 148;
  const valueW = 66;
  const plotX = labelW;
  const plotW = W - labelW - valueW;
  const H = padTop + data.length * rowH + padBottom;

  const xOf = (v: number) => plotX + ((v - lo) / span) * plotW;
  const chanceX = xOf(0.5);

  // gridlines every 0.1
  const ticks: number[] = [];
  for (let t = Math.ceil(lo * 10) / 10; t <= hi + 1e-9; t += 0.1) {
    ticks.push(Number(t.toFixed(1)));
  }

  return (
    <div ref={ref} style={{ width: '100%' }}>
      <svg
        viewBox={`0 0 ${W} ${H}`}
        width="100%"
        role="img"
        aria-label="AUROC by modality and fusion, with 95% confidence intervals"
        style={{ display: 'block', fontFamily: 'var(--font-mono)' }}
      >
        {/* gridlines + axis ticks */}
        {ticks.map((t) => (
          <g key={t}>
            <line
              x1={xOf(t)}
              x2={xOf(t)}
              y1={padTop}
              y2={H - padBottom}
              stroke="var(--border-subtle)"
              strokeWidth={1}
            />
            <text
              x={xOf(t)}
              y={H - padBottom + 16}
              textAnchor="middle"
              fontSize={10}
              fill="var(--text-quaternary)"
            >
              {t.toFixed(1)}
            </text>
          </g>
        ))}

        {/* chance reference line (0.50) */}
        <line
          x1={chanceX}
          x2={chanceX}
          y1={padTop - 4}
          y2={H - padBottom}
          stroke="var(--text-tertiary)"
          strokeWidth={1}
          strokeDasharray="3 3"
        />
        <text
          x={chanceX}
          y={padTop - 6}
          textAnchor="middle"
          fontSize={9}
          fill="var(--text-tertiary)"
          fontFamily="var(--font-sans)"
        >
          chance 0.50
        </text>

        {data.map((d, i) => {
          const y = padTop + i * rowH;
          const cy = y + rowH / 2;
          const barH = 16;
          const barY = cy - barH / 2;
          const fullW = xOf(d.value) - plotX;
          const grown = d.notMeasured ? 0 : seen ? fullW : 0;
          return (
            <g key={d.key}>
              {/* row label */}
              <text
                x={plotX - 10}
                y={cy + 4}
                textAnchor="end"
                fontSize={11}
                fontFamily="var(--font-sans)"
                fontWeight={d.isFusion ? 600 : 400}
                fill={d.isFusion ? 'var(--text-primary)' : 'var(--text-secondary)'}
              >
                {d.label}
              </text>

              {/* track */}
              <rect
                x={plotX}
                y={barY}
                width={plotW}
                height={barH}
                fill="var(--bg-inset)"
                rx={2}
              />
              {/* value bar */}
              <rect
                x={plotX}
                y={barY}
                width={Math.max(0, grown)}
                height={barH}
                fill={d.color}
                rx={2}
                style={{
                  transition: reduced
                    ? undefined
                    : 'width 700ms cubic-bezier(0.16,1,0.3,1)',
                }}
              />
              {/* fusion outline emphasis */}
              {d.isFusion && (
                <rect
                  x={plotX}
                  y={barY}
                  width={plotW}
                  height={barH}
                  fill="none"
                  stroke={d.color}
                  strokeWidth={1}
                  strokeDasharray="2 2"
                  opacity={0.5}
                  rx={2}
                />
              )}

              {/* CI whisker */}
              {d.ci && seen && (
                <g
                  style={{
                    transition: reduced ? undefined : 'opacity 300ms ease 500ms',
                  }}
                >
                  <line
                    x1={xOf(d.ci[0])}
                    x2={xOf(d.ci[1])}
                    y1={cy}
                    y2={cy}
                    stroke="var(--text-primary)"
                    strokeWidth={1.25}
                    opacity={0.65}
                  />
                  <line
                    x1={xOf(d.ci[0])}
                    x2={xOf(d.ci[0])}
                    y1={cy - 4}
                    y2={cy + 4}
                    stroke="var(--text-primary)"
                    strokeWidth={1.25}
                    opacity={0.65}
                  />
                  <line
                    x1={xOf(d.ci[1])}
                    x2={xOf(d.ci[1])}
                    y1={cy - 4}
                    y2={cy + 4}
                    stroke="var(--text-primary)"
                    strokeWidth={1.25}
                    opacity={0.65}
                  />
                </g>
              )}

              {/* value label */}
              <text
                x={W - valueW + 8}
                y={cy + 4}
                fontSize={12}
                fontWeight={d.isFusion ? 600 : 400}
                fill={d.notMeasured ? 'var(--text-tertiary)' : 'var(--text-primary)'}
                style={{
                  opacity: seen ? 1 : 0,
                  transition: reduced ? undefined : 'opacity 400ms ease 350ms',
                }}
              >
                {d.notMeasured ? 'n/a' : fmt(d.value)}
                {d.notMeasured && (
                  <title>Drivers were not separately re-run at this threshold; the 2-way fusion equals the 3-way here.</title>
                )}
              </text>
            </g>
          );
        })}
      </svg>
    </div>
  );
}

// A small inline legend swatch.
function Swatch({ color }: { color: string }) {
  return (
    <span
      aria-hidden
      style={{
        display: 'inline-block',
        width: 10,
        height: 10,
        borderRadius: 2,
        background: color,
        flexShrink: 0,
      }}
    />
  );
}

// ===========================================================================
// Contribution ladder — best single → 2-way → 3-way, showing each step's gain.
// ===========================================================================
function ContributionLadder({
  row,
  reduced,
}: {
  row: ThresholdRow;
  reduced: boolean;
}) {
  const { ref, seen } = useInView<HTMLDivElement>(reduced);
  const steps = [
    {
      key: 'single',
      label: 'Best single modality',
      value: row.bestSingle,
      color: 'var(--viz-sensor)',
      delta: null as number | null,
    },
    {
      key: 'two',
      label: '+ 2-way (optics + pigment)',
      value: row.twoWay,
      color: 'var(--accent-hover)',
      delta: row.twoWay - row.bestSingle,
    },
    {
      key: 'three',
      label: '+ 3-way (+ drivers)',
      value: row.threeWay,
      color: FUSION_COLOR,
      delta: row.threeWay - row.twoWay,
    },
  ];
  const lo = 0.6;
  const hi = 0.86;
  const pct = (v: number) => ((v - lo) / (hi - lo)) * 100;

  return (
    <div ref={ref} style={{ display: 'grid', gap: 'var(--s-3)' }}>
      {steps.map((s, i) => (
        <div
          key={s.key}
          style={{
            display: 'grid',
            gridTemplateColumns: '180px 1fr 108px',
            alignItems: 'center',
            gap: 'var(--s-4)',
          }}
        >
          <span
            style={{
              fontSize: 'var(--fs-12)',
              color: 'var(--text-secondary)',
              fontFamily: 'var(--font-sans)',
            }}
          >
            {s.label}
          </span>
          <div
            style={{
              position: 'relative',
              height: 12,
              background: 'var(--bg-inset)',
              borderRadius: 2,
            }}
          >
            <div
              style={{
                position: 'absolute',
                inset: 0,
                width: seen ? `${pct(s.value)}%` : 0,
                background: s.color,
                borderRadius: 2,
                transition: reduced
                  ? undefined
                  : `width 650ms cubic-bezier(0.16,1,0.3,1) ${i * 90}ms`,
              }}
            />
          </div>
          <span
            style={{
              display: 'flex',
              alignItems: 'baseline',
              gap: 'var(--s-2)',
              justifyContent: 'flex-end',
            }}
          >
            <span className="mono" style={{ fontSize: 'var(--fs-13)', fontWeight: 600 }}>
              {fmt(s.value)}
            </span>
            {s.delta !== null && (
              <span
                className="mono"
                style={{
                  fontSize: 'var(--fs-11)',
                  color: s.delta > 0.001 ? 'var(--nominal)' : 'var(--text-tertiary)',
                }}
              >
                {signed(s.delta)}
              </span>
            )}
          </span>
        </div>
      ))}
    </div>
  );
}

// ===========================================================================
// Threshold toggle — segmented control (>1.6 / >0.3 / >8).
// ===========================================================================
function ThresholdToggle({
  value,
  onChange,
}: {
  value: string;
  onChange: (k: string) => void;
}) {
  return (
    <div
      role="radiogroup"
      aria-label="Microcystin threshold"
      style={{
        display: 'inline-flex',
        border: '1px solid var(--border-strong)',
        borderRadius: 'var(--r-md)',
        overflow: 'hidden',
        background: 'var(--bg-surface)',
      }}
    >
      {FUSION_THRESHOLDS.map((t, i) => {
        const active = t.key === value;
        return (
          <button
            key={t.key}
            role="radio"
            aria-checked={active}
            onClick={() => onChange(t.key)}
            style={{
              padding: '6px 12px',
              fontFamily: 'var(--font-mono)',
              fontSize: 'var(--fs-12)',
              fontWeight: active ? 600 : 400,
              color: active ? 'var(--text-inverse)' : 'var(--text-secondary)',
              background: active ? 'var(--accent)' : 'transparent',
              borderLeft: i === 0 ? 'none' : '1px solid var(--border-subtle)',
              transition:
                'background var(--dur-fast) var(--ease-standard), color var(--dur-fast)',
              whiteSpace: 'nowrap',
            }}
            onMouseEnter={(e) => {
              if (!active) e.currentTarget.style.background = 'var(--bg-hover)';
            }}
            onMouseLeave={(e) => {
              if (!active) e.currentTarget.style.background = 'transparent';
            }}
          >
            {t.label}
          </button>
        );
      })}
    </div>
  );
}

// ===========================================================================
// Cross-region collapse — two paired bars (station-CV win vs cross-region
// chance) that make the caveat viscerally obvious.
// ===========================================================================
function CollapseVisual({ reduced }: { reduced: boolean }) {
  const { ref, seen } = useInView<HTMLDivElement>(reduced);
  const stationWin = FUSION_HEADLINE.threeWay; // 0.823
  const crossRegion = CROSS_REGION_TRANSFER[0].raw; // 0.485
  const bestTransform = CROSS_REGION_TRANSFER[0].bestTransform; // 0.527
  const lo = 0.4;
  const hi = 0.9;
  const pct = (v: number) => ((v - lo) / (hi - lo)) * 100;

  const bars = [
    {
      key: 'station',
      label: 'Station-grouped CV',
      sub: 'sites split, same regions',
      value: stationWin,
      color: 'var(--nominal)',
      badge: 'nominal' as const,
      verdict: 'fusion wins',
    },
    {
      key: 'cross',
      label: 'Cross-region holdout',
      sub: 'leave-one-HUC-out, 5 regions',
      value: crossRegion,
      color: 'var(--alert)',
      badge: 'alert' as const,
      verdict: 'collapses to chance',
    },
    {
      key: 'transform',
      label: 'Best feature transform',
      sub: CROSS_REGION_TRANSFER[0].bestTransformLabel,
      value: bestTransform,
      color: 'var(--neutral-status)',
      badge: 'neutral' as const,
      verdict: 'no recovery',
    },
  ];

  return (
    <div ref={ref} style={{ display: 'grid', gap: 'var(--s-5)' }}>
      {bars.map((b, i) => (
        <div key={b.key}>
          <div
            style={{
              display: 'flex',
              alignItems: 'baseline',
              justifyContent: 'space-between',
              gap: 'var(--s-4)',
              marginBottom: 'var(--s-2)',
            }}
          >
            <span
              style={{
                fontSize: 'var(--fs-13)',
                fontWeight: 600,
                color: 'var(--text-primary)',
              }}
            >
              {b.label}{' '}
              <span
                style={{
                  fontWeight: 400,
                  color: 'var(--text-tertiary)',
                  fontSize: 'var(--fs-12)',
                }}
              >
                · {b.sub}
              </span>
            </span>
            <StatusBadge kind={b.badge}>{b.verdict}</StatusBadge>
          </div>
          <div
            style={{
              position: 'relative',
              height: 22,
              background: 'var(--bg-inset)',
              borderRadius: 'var(--r-sm)',
            }}
          >
            {/* chance marker */}
            <div
              aria-hidden
              style={{
                position: 'absolute',
                left: `${pct(0.5)}%`,
                top: -3,
                bottom: -3,
                width: 1,
                borderLeft: '1px dashed var(--text-tertiary)',
              }}
            />
            <div
              style={{
                position: 'absolute',
                left: 0,
                top: 0,
                bottom: 0,
                width: seen ? `${pct(b.value)}%` : 0,
                background: b.color,
                borderRadius: 'var(--r-sm)',
                transition: reduced
                  ? undefined
                  : `width 700ms cubic-bezier(0.16,1,0.3,1) ${i * 110}ms`,
              }}
            />
            <span
              className="mono"
              style={{
                position: 'absolute',
                right: 8,
                top: '50%',
                transform: 'translateY(-50%)',
                fontSize: 'var(--fs-13)',
                fontWeight: 600,
                color:
                  pct(b.value) > 55 ? 'var(--text-inverse)' : 'var(--text-primary)',
              }}
            >
              {fmt(b.value)}
            </span>
          </div>
        </div>
      ))}
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: 'var(--s-2)',
          fontSize: 'var(--fs-11)',
          color: 'var(--text-tertiary)',
          fontFamily: 'var(--font-sans)',
        }}
      >
        <span
          aria-hidden
          style={{
            display: 'inline-block',
            width: 14,
            borderTop: '1px dashed var(--text-tertiary)',
          }}
        />
        dashed line = chance (AUROC 0.50)
      </div>
    </div>
  );
}

// ===========================================================================
// Mechanism flow — three modalities converge to fusion; a fourth (metagenomic
// mcyD) is the data-blocked transfer path. Small flat SVG-free flow.
// ===========================================================================
function MechanismFlow() {
  return (
    <div
      style={{
        display: 'grid',
        gridTemplateColumns: 'minmax(0,1fr) auto minmax(0,1fr)',
        alignItems: 'center',
        gap: 'var(--s-5)',
      }}
    >
      <div style={{ display: 'grid', gap: 'var(--s-3)' }}>
        {MODALITIES.map((m) => {
          const meta = MODALITY_META[m.key];
          const Icon = meta.Icon;
          return (
            <div
              key={m.key}
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: 'var(--s-3)',
                padding: 'var(--s-3) var(--s-4)',
                border: '1px solid var(--border-subtle)',
                borderLeft: `3px solid ${meta.color}`,
                borderRadius: 'var(--r-sm)',
                background: 'var(--bg-subtle)',
              }}
            >
              <Icon
                size={16}
                strokeWidth={1.75}
                style={{ color: meta.color, flexShrink: 0 }}
                aria-hidden
              />
              <span
                style={{
                  fontSize: 'var(--fs-12)',
                  fontWeight: 600,
                  color: 'var(--text-primary)',
                }}
              >
                {m.label}
              </span>
            </div>
          );
        })}
      </div>

      <ArrowRight
        size={20}
        strokeWidth={1.75}
        style={{ color: 'var(--text-quaternary)' }}
        aria-label="combine into"
      />

      <div style={{ display: 'grid', gap: 'var(--s-3)' }}>
        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: 'var(--s-3)',
            padding: 'var(--s-4)',
            border: `1px solid var(--accent-border)`,
            borderLeft: `3px solid ${FUSION_COLOR}`,
            borderRadius: 'var(--r-sm)',
            background: 'var(--accent-wash)',
          }}
        >
          <Layers
            size={16}
            strokeWidth={1.75}
            style={{ color: FUSION_COLOR, flexShrink: 0 }}
            aria-hidden
          />
          <div>
            <div
              style={{ fontSize: 'var(--fs-12)', fontWeight: 600, color: 'var(--accent-strong)' }}
            >
              3-way fusion
            </div>
            <div
              className="mono"
              style={{ fontSize: 'var(--fs-11)', color: 'var(--text-tertiary)' }}
            >
              within-region: {fmt(FUSION_HEADLINE.threeWay)}
            </div>
          </div>
        </div>
        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: 'var(--s-3)',
            padding: 'var(--s-4)',
            border: '1px dashed var(--border-strong)',
            borderLeft: `3px dashed var(--viz-microbial)`,
            borderRadius: 'var(--r-sm)',
            background: 'var(--bg-subtle)',
          }}
        >
          <Dna
            size={16}
            strokeWidth={1.75}
            style={{ color: 'var(--viz-microbial)', flexShrink: 0 }}
            aria-hidden
          />
          <div>
            <div
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: 'var(--s-2)',
                fontSize: 'var(--fs-12)',
                fontWeight: 600,
                color: 'var(--text-primary)',
              }}
            >
              + metagenomic mcyD
              <StatusBadge kind="neutral">data-blocked</StatusBadge>
            </div>
            <div style={{ fontSize: 'var(--fs-11)', color: 'var(--text-tertiary)' }}>
              the transferable route — not yet co-located
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

// ===========================================================================
// Page
// ===========================================================================
export default function Fusion() {
  const reduced = useReducedMotion();
  const [thresholdKey, setThresholdKey] = useState(FUSION_HEADLINE.key);
  const row = useMemo(
    () => FUSION_THRESHOLDS.find((t) => t.key === thresholdKey) ?? FUSION_HEADLINE,
    [thresholdKey],
  );

  // Bars for the current threshold: three singles + 2-way + 3-way.
  const bars: BarDatum[] = useMemo(() => {
    const singleBars = MODALITIES.map((m) => ({
      key: m.key,
      label: m.label,
      value: row.single[m.key],
      color: MODALITY_META[m.key].color,
      IconEl: MODALITY_META[m.key].Icon,
      notMeasured: row.singleNotMeasured?.[m.key] ?? false,
    }));
    return [
      ...singleBars,
      {
        key: 'twoway',
        label: '2-way fusion',
        value: row.twoWay,
        color: 'var(--accent-hover)',
      },
      {
        key: 'threeway',
        label: '3-way fusion',
        value: row.threeWay,
        ci: row.threeWayCI,
        color: FUSION_COLOR,
        isFusion: true,
      },
    ];
  }, [row]);

  const gainStatus = row.fusionHelps ? 'nominal' : 'neutral';

  // Redundancy table columns (chl-a / DO — where fusion only ties).
  const redundancyCols: Column<RedundancyRow>[] = [
    {
      key: 'task',
      header: 'Task',
      cell: (r) => (
        <span>
          <span style={{ fontWeight: 500 }}>{r.taskLabel}</span>
          {r.horizon && (
            <span className="mono" style={{ color: 'var(--text-tertiary)' }}>
              {' '}
              · {r.horizon}
            </span>
          )}
        </span>
      ),
    },
    {
      key: 'best',
      header: 'Best single',
      mono: true,
      cell: (r) => (
        <span>
          {fmt(r.bestSingle)}{' '}
          <span style={{ color: 'var(--text-tertiary)', fontSize: 'var(--fs-11)' }}>
            {r.bestSingleLabel}
          </span>
        </span>
      ),
    },
    { key: 'fusion', header: 'Fusion', mono: true, cell: (r) => fmt(r.fusion) },
    {
      key: 'gain',
      header: 'Gain',
      mono: true,
      cell: (r) => {
        const g = r.fusion - r.bestSingle;
        const wins = g > 0.02;
        return (
          <span style={{ color: wins ? 'var(--nominal)' : 'var(--text-tertiary)' }}>
            {signed(g)}
          </span>
        );
      },
    },
    {
      key: 'verdict',
      header: 'Verdict',
      cell: (r) => (
        <span style={{ color: 'var(--text-secondary)', fontSize: 'var(--fs-12)' }}>
          {r.verdict}
        </span>
      ),
    },
  ];

  return (
    <>
      {/* Grid responsiveness — stack the two-column layouts on narrow viewports
          (scoped to this page; mirrors Models.tsx). */}
      <style>{`
        @media (max-width: 760px) {
          .fusion-chart-grid { grid-template-columns: 1fr !important; }
          .fusion-protocol-grid { grid-template-columns: 1fr !important; }
        }
      `}</style>
      <SectionHeader
        as="h1"
        eyebrow="Track B · Multimodal fusion"
        title="Does fusion beat the best single modality?"
        description={FUSION_SUMMARY}
        actions={
          <a href="/docs">Holdout protocol</a>
        }
      />

      {/* ---- headline metric row -------------------------------------- */}
      <div
        style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))',
          gap: 'var(--s-6)',
          padding: 'var(--s-6)',
          background: 'var(--bg-surface)',
          border: '1px solid var(--border-subtle)',
          borderRadius: 'var(--r-lg)',
          marginBottom: 'var(--s-6)',
        }}
      >
        <Metric
          size="lg"
          countUp
          label="3-way fusion AUROC"
          value="0.823"
          status="nominal"
          sub="microcystin > 1.6 µg/L (WHO drinking)"
          tooltip="Area under the ROC curve for 3-modality fusion (satellite optics +
          in-situ pigment + physicochemical drivers) at the WHO drinking-water
          threshold, under station-grouped cross-validation. Higher is better; 0.50
          is chance."
        />
        <Metric
          size="lg"
          countUp
          label="Best single modality"
          value="0.684"
          status="watch"
          sub="in-situ pigment (phycocyanin)"
          tooltip="Best AUROC achieved by any single modality on its own, at the same
          threshold and under the same station-grouped CV. Fusion is compared against
          this baseline."
        />
        <Metric
          size="lg"
          countUp
          label="Gain vs best single"
          value="+0.138"
          copyValue="+0.138 AUROC"
          status="nominal"
          sub="3-way > 2-way > single, CIs exclude 0"
          tooltip="Fusion minus best single modality. Each added modality contributes a
          CI-significant gain; a shuffle-the-drivers negative control confirms this is
          real signal, not overfit capacity."
        />
        <Metric
          size="lg"
          countUp
          label="Marginal driver gain (3-way − 2-way)"
          value="+0.028 … +0.068"
          copyValue="+0.028 to +0.068 AUROC (driver gain, site bootstrap, CI excludes 0)"
          status="nominal"
          sub="site-clustered CI excludes 0"
          tooltip="Site-clustered bootstrap 95% CI on the improvement from adding the
          physicochemical-driver modality to the 2-way fusion, at the WHO drinking
          threshold. Excluding zero means drivers add genuinely non-redundant signal."
        />
      </div>

      <Callout tone="verified" title="A real, CI-significant within-region win">
        On microcystin, the three modalities carry genuinely non-redundant
        information (the toxin is optically invisible), so fusion beats every single
        modality under station-grouped CV — and each modality adds. This is the real,
        verified headline. The honest boundary follows immediately below.
      </Callout>

      {/* ================================================================
          SECTION 1 — per-modality vs fusion bar chart + threshold toggle
          ================================================================ */}
      <div style={{ marginTop: 'var(--s-9)' }}>
        <SectionHeader
          as="h2"
          eyebrow="Within-region · station-grouped CV"
          title="Per-modality vs fusion AUROC"
          description="Each single modality on its own, versus 2-way and 3-way fusion,
          with 95% site-clustered bootstrap confidence intervals. Toggle the
          microcystin threshold to see how the picture changes with bloom severity."
          actions={
            <ThresholdToggle value={thresholdKey} onChange={setThresholdKey} />
          }
        />

        <div
          className="fusion-chart-grid"
          style={{
            display: 'grid',
            gridTemplateColumns: 'minmax(0, 1.55fr) minmax(0, 1fr)',
            gap: 'var(--s-6)',
            alignItems: 'stretch',
          }}
        >
          <Card>
            <CardTitle
              tip="Bars = AUROC. Whiskers on the 3-way bar = 95% site-clustered
              bootstrap CI. Dashed vertical line = chance (0.50). Bars animate on
              first view."
              right={
                <span
                  className="mono"
                  style={{ fontSize: 'var(--fs-11)', color: 'var(--text-tertiary)' }}
                >
                  n = {row.nPos.toLocaleString()} pos
                  {row.nSitesPos ? ` / ${row.nSitesPos} sites` : ''}
                </span>
              }
            >
              AUROC · microcystin {row.label}
            </CardTitle>

            <AurocBarChart data={bars} reduced={reduced} />

            {/* legend */}
            <div
              style={{
                display: 'flex',
                flexWrap: 'wrap',
                gap: 'var(--s-5)',
                marginTop: 'var(--s-3)',
                paddingTop: 'var(--s-3)',
                borderTop: '1px solid var(--border-subtle)',
              }}
            >
              {MODALITIES.map((m) => (
                <span
                  key={m.key}
                  style={{
                    display: 'inline-flex',
                    alignItems: 'center',
                    gap: 'var(--s-2)',
                    fontSize: 'var(--fs-11)',
                    color: 'var(--text-secondary)',
                  }}
                >
                  <Swatch color={MODALITY_META[m.key].color} />
                  {m.label}
                </span>
              ))}
              <span
                style={{
                  display: 'inline-flex',
                  alignItems: 'center',
                  gap: 'var(--s-2)',
                  fontSize: 'var(--fs-11)',
                  color: 'var(--text-secondary)',
                }}
              >
                <Swatch color={FUSION_COLOR} />
                fusion
              </span>
            </div>
          </Card>

          {/* right rail — threshold context + gain readout */}
          <Card>
            <CardTitle tip="Regulatory / screening meaning of the selected threshold, and whether fusion beats the best single here.">
              Threshold {row.label}
            </CardTitle>
            <p
              style={{
                margin: '0 0 var(--s-4)',
                fontSize: 'var(--fs-13)',
                lineHeight: 'var(--lh-normal)',
                color: 'var(--text-secondary)',
              }}
            >
              {row.meaning}.
            </p>

            <div
              style={{
                display: 'grid',
                gap: 'var(--s-4)',
                paddingTop: 'var(--s-4)',
                borderTop: '1px solid var(--border-subtle)',
              }}
            >
              <Metric
                size="md"
                label="Gain vs best single"
                value={signed(row.gainVsBestSingle)}
                copyValue={`${signed(row.gainVsBestSingle)} AUROC`}
                status={gainStatus}
                sub={
                  row.fusionHelps
                    ? 'CI excludes 0 — fusion wins'
                    : 'CI includes 0 — not a win here'
                }
                tooltip="3-way fusion AUROC minus the best single modality at this
                threshold. Fusion 'wins' only when the confidence interval on the
                gain excludes zero."
              />
              <div>
                <div
                  className="eyebrow"
                  style={{ marginBottom: 'var(--s-2)', display: 'flex', alignItems: 'center', gap: 6 }}
                >
                  Marginal driver gain (3-way − 2-way)
                  <Tooltip label="Site-clustered bootstrap 95% CI on the improvement from adding the physicochemical-driver modality to the 2-way fusion. When it excludes 0, drivers add non-redundant signal.">
                    <Info size={11} strokeWidth={1.75} style={{ color: 'var(--text-tertiary)', display: 'block' }} />
                  </Tooltip>
                </div>
                <span className="mono" style={{ fontSize: 'var(--fs-14)', fontWeight: 600 }}>
                  {signed(row.driverGainCI[0])} … {signed(row.driverGainCI[1])}
                </span>{' '}
                <StatusBadge kind={row.fusionHelps ? 'nominal' : 'neutral'}>
                  {row.driverGainCI[0] > 0 ? 'excludes 0' : 'includes 0'}
                </StatusBadge>
              </div>
              <div>
                <div
                  className="eyebrow"
                  style={{ marginBottom: 'var(--s-2)', display: 'flex', alignItems: 'center', gap: 6 }}
                >
                  Negative control (drivers shuffled)
                  <Tooltip label="When the driver values are randomly permuted, fusion AUROC falls back toward the 2-way result — confirming the driver gain is real signal, not overfit model capacity.">
                    <Info size={11} strokeWidth={1.75} style={{ color: 'var(--text-tertiary)', display: 'block' }} />
                  </Tooltip>
                </div>
                <span className="mono" style={{ fontSize: 'var(--fs-14)', fontWeight: 600 }}>
                  {fmt(row.permutationControl)}
                </span>
                <span
                  style={{ fontSize: 'var(--fs-12)', color: 'var(--text-tertiary)' }}
                >
                  {' '}
                  vs {fmt(row.threeWay)} real
                </span>
              </div>
            </div>
          </Card>
        </div>
      </div>

      {/* ================================================================
          SECTION 2 — how each modality adds (contribution ladder)
          ================================================================ */}
      <div style={{ marginTop: 'var(--s-9)' }}>
        <SectionHeader
          as="h2"
          title="Each modality adds real signal"
          description="At the WHO drinking threshold, fusion climbs monotonically:
          best single → add optics → add drivers. Because microcystin is optically
          invisible, the modalities are non-redundant — pigment (phycocyanin) tracks
          cyanobacteria, optics sees surface biomass, drivers (temperature, phosphorus)
          carry the causal production signal."
        />
        <Card>
          <CardTitle tip="Marginal AUROC gain from each fusion step at microcystin > 1.6 µg/L. Green delta = a positive, CI-significant contribution.">
            Contribution ladder · microcystin {'>'} 1.6 µg/L
          </CardTitle>
          <ContributionLadder row={FUSION_HEADLINE} reduced={reduced} />

          <div
            style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))',
              gap: 'var(--s-4)',
              marginTop: 'var(--s-6)',
              paddingTop: 'var(--s-5)',
              borderTop: '1px solid var(--border-subtle)',
            }}
          >
            {MODALITIES.map((m) => {
              const meta = MODALITY_META[m.key];
              const Icon = meta.Icon;
              return (
                <div key={m.key} style={{ display: 'flex', gap: 'var(--s-3)' }}>
                  <Icon
                    size={16}
                    strokeWidth={1.75}
                    style={{ color: meta.color, flexShrink: 0, marginTop: 2 }}
                    aria-hidden
                  />
                  <div>
                    <div
                      style={{
                        fontSize: 'var(--fs-12)',
                        fontWeight: 600,
                        color: 'var(--text-primary)',
                        marginBottom: 2,
                      }}
                    >
                      {m.label}
                    </div>
                    <div
                      style={{
                        fontSize: 'var(--fs-11)',
                        color: 'var(--text-tertiary)',
                        lineHeight: 1.5,
                        marginBottom: 4,
                      }}
                    >
                      {m.measures}
                    </div>
                    <div
                      style={{
                        fontSize: 'var(--fs-12)',
                        color: 'var(--text-secondary)',
                        lineHeight: 'var(--lh-normal)',
                      }}
                    >
                      {m.rationale}
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        </Card>
      </div>

      {/* ================================================================
          SECTION 3 — DEPLOYMENT SCOPE (calibrated within-region)
          ================================================================ */}
      <div style={{ marginTop: 'var(--s-9)' }}>
        <SectionHeader
          as="h2"
          eyebrow="Deployment scope"
          title="Deployment scope: calibrated within-region; cross-region needs local calibration"
          description="Fusion is a within-region capability. The same model that wins
          under station-grouped CV needs local recalibration — or the genotype
          modality — before it crosses watersheds: under a true cross-region
          (leave-one-watershed-out) holdout its AUROC reads chance (~0.49). We scope
          the deployment boundary explicitly rather than imply it transfers."
          actions={
            <Metric
              size="md"
              label="Cross-region AUROC"
              value="0.485"
              status="alert"
              sub="leave-one-HUC-out — chance"
              tooltip="AUROC of the SAME fusion model under a true cross-region
              (leave-one-watershed-out) holdout. It reads chance (~0.50) because the
              toxin-producing strain fraction (mcyD genotype) is unobserved and varies
              by watershed. No feature transform recovers it; deployment is scoped
              within-region."
            />
          }
        />

        <div
          className="fusion-protocol-grid"
          style={{
            display: 'grid',
            gridTemplateColumns: 'minmax(0, 1fr) minmax(0, 1fr)',
            gap: 'var(--s-6)',
          }}
        >
          <Card
            style={{
              borderColor: 'var(--alert-border)',
              background: 'var(--bg-surface)',
            }}
          >
            <CardTitle tip="Same model, two evaluation protocols. Under cross-region holdout, AUROC falls to ~0.49 (chance) and no feature transform recovers it.">
              Same model, two protocols
            </CardTitle>
            <CollapseVisual reduced={reduced} />
          </Card>

          <div style={{ display: 'grid', gap: 'var(--s-5)', alignContent: 'start' }}>
            <Callout tone="alert" title="Collapses to chance across watersheds">
              Under leave-one-HUC-out over {CROSS_REGION_TRANSFER[0].nRegions} regions
              ({CROSS_REGION_TRANSFER[0].nCrossRegion.toLocaleString()} cross-region
              samples), fusion AUROC is{' '}
              <span className="mono" style={{ fontWeight: 600 }}>
                {fmt(CROSS_REGION_TRANSFER[0].raw)}
              </span>
              . The best of every feature transform we tried (region-conditioning,
              within-site-relative, global-rank) reached only{' '}
              <span className="mono" style={{ fontWeight: 600 }}>
                {fmt(CROSS_REGION_TRANSFER[0].bestTransform)}
              </span>{' '}
              — still chance.
            </Callout>

            <Card>
              <CardTitle tip="Why the within-region relationship does not transfer to unseen watersheds.">
                Mechanism · why it collapses
              </CardTitle>
              <p
                style={{
                  margin: 0,
                  fontSize: 'var(--fs-13)',
                  lineHeight: 'var(--lh-normal)',
                  color: 'var(--text-secondary)',
                }}
              >
                {FUSION_MECHANISM.whyItCollapses}
              </p>
            </Card>
          </div>
        </div>

        {/* cross-region transfer table */}
        <div style={{ marginTop: 'var(--s-6)' }}>
          <DataTable
            rows={CROSS_REGION_TRANSFER}
            getRowKey={(r) => r.thresholdKey}
            caption="Cross-region (leave-one-HUC-out) AUROC by threshold — raw and best feature transform. Both hover at chance."
            columns={[
              { key: 'th', header: 'Threshold', cell: (r) => r.thresholdLabel },
              {
                key: 'raw',
                header: 'Raw cross-region',
                mono: true,
                cell: (r) => (
                  <span style={{ color: 'var(--alert)' }}>{fmt(r.raw)}</span>
                ),
              },
              {
                key: 'best',
                header: 'Best transform',
                mono: true,
                cell: (r) => fmt(r.bestTransform),
              },
              {
                key: 'lbl',
                header: 'Transform',
                cell: (r) => (
                  <span className="mono" style={{ fontSize: 'var(--fs-12)' }}>
                    {r.bestTransformLabel}
                  </span>
                ),
              },
              {
                key: 'n',
                header: 'n (regions)',
                mono: true,
                cell: (r) => `${r.nCrossRegion.toLocaleString()} (${r.nRegions})`,
              },
              {
                key: 'v',
                header: 'Verdict',
                cell: () => <StatusBadge kind="alert">chance</StatusBadge>,
              },
            ]}
          />
        </div>
      </div>

      {/* ================================================================
          SECTION 4 — mechanism + the transfer path (mcyD)
          ================================================================ */}
      <div style={{ marginTop: 'var(--s-9)' }}>
        <SectionHeader
          as="h2"
          title="The missing variable — and the transferable route"
          description="The unobserved variable is the toxin-producing strain fraction
          (the mcyD genotype). Optics, pigment, and drivers cannot see it. The
          science-grounded fix is SENTINEL's own metagenomic modality, which measures
          mcyD directly — currently data-blocked."
        />

        <Card style={{ marginBottom: 'var(--s-6)' }}>
          <CardTitle
            tip="Three visible modalities fuse to a within-region win; adding the
            metagenomic mcyD modality is the route to cross-region transfer, but no
            site currently co-locates genotype with the fusion data."
          >
            <Layers size={14} strokeWidth={1.75} style={{ color: FUSION_COLOR }} aria-hidden />
            Fusion architecture &amp; the transfer path
          </CardTitle>
          <MechanismFlow />
        </Card>

        <div
          style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))',
            gap: 'var(--s-6)',
          }}
        >
          <Card>
            <CardTitle>
              <Droplet size={14} strokeWidth={1.75} style={{ color: 'var(--nominal)' }} aria-hidden />
              Why it wins (within region)
            </CardTitle>
            <p
              style={{
                margin: 0,
                fontSize: 'var(--fs-13)',
                lineHeight: 'var(--lh-normal)',
                color: 'var(--text-secondary)',
              }}
            >
              {FUSION_MECHANISM.whyItWins}
            </p>
          </Card>
          <Card>
            <CardTitle>
              <GitBranch size={14} strokeWidth={1.75} style={{ color: 'var(--viz-microbial)' }} aria-hidden />
              The transfer path (mcyD)
            </CardTitle>
            <p
              style={{
                margin: 0,
                fontSize: 'var(--fs-13)',
                lineHeight: 'var(--lh-normal)',
                color: 'var(--text-secondary)',
              }}
            >
              {FUSION_MECHANISM.theTransferPath}
            </p>
          </Card>
          <Card>
            <CardTitle>
              <Layers size={14} strokeWidth={1.75} style={{ color: 'var(--text-tertiary)' }} aria-hidden />
              Deployment robustness (missing modality)
            </CardTitle>
            <p
              style={{
                margin: 0,
                fontSize: 'var(--fs-13)',
                lineHeight: 'var(--lh-normal)',
                color: 'var(--text-secondary)',
              }}
            >
              {FUSION_MECHANISM.deploymentRobustness}
            </p>
          </Card>
        </div>
      </div>

      {/* ================================================================
          SECTION 5 — where fusion only TIES (chl-a / DO redundancy)
          ================================================================ */}
      <div style={{ marginTop: 'var(--s-9)' }}>
        <SectionHeader
          as="h2"
          eyebrow="Honest redundancy"
          title="Why fusion only ties on chl-a and dissolved oxygen"
          description="Fusion is not a universal win. When a target is dominated by a
          single sensing modality, adding others cannot help — and we report that.
          Chlorophyll-a is optical (satellite dominates); dissolved-oxygen hypoxia is
          chemistry (in-situ sensor / persistence dominate). In both, fusion matches
          but does not beat the best single."
        />

        <DataTable
          rows={FUSION_REDUNDANCY}
          getRowKey={(r) => r.task}
          caption="Fusion vs best single on single-modality-dominated targets. Gains are within CI of zero — an honest tie, not a win."
          columns={redundancyCols}
        />

        <div style={{ marginTop: 'var(--s-5)' }}>
          <Callout tone="limitation" title="Retracted: the synthetic 0.992 fusion result">
            An earlier build reported a "0.992 fusion beats single modality" on chl-a.
            That number was computed on synthetic data and has been{' '}
            <strong>retracted</strong>. The real chl-a result is{' '}
            <span className="mono" style={{ fontWeight: 600 }}>0.708</span> fusion vs{' '}
            <span className="mono" style={{ fontWeight: 600 }}>0.716</span> best single —
            a tie. Fusion helps only where modalities are genuinely non-redundant, as on
            optically-invisible microcystin.
          </Callout>
        </div>
      </div>

      {/* ================================================================
          SECTION 6 — bloom early-warning (a second-crisis forecast)
          ================================================================ */}
      <div style={{ marginTop: 'var(--s-9)' }}>
        <SectionHeader
          as="h2"
          eyebrow="Downstream · antecedent-only forecast"
          title="Bloom early-warning: forecasting a second crisis"
          description={BLOOM_EARLY_WARNING.summary}
        />

        <div
          style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))',
            gap: 'var(--s-6)',
            padding: 'var(--s-6)',
            background: 'var(--bg-surface)',
            border: '1px solid var(--border-subtle)',
            borderRadius: 'var(--r-lg)',
            marginBottom: 'var(--s-6)',
          }}
        >
          <Metric
            size="lg"
            countUp
            label="New-bloom onset AUROC"
            value={fmt(BLOOM_EARLY_WARNING.onsetFusionAuroc)}
            status="nominal"
            sub="fusion · WHO Alert-2 (chl-a > 24 µg/L)"
            tooltip="Fusion AUROC on the strict onset-only test — restricted to visits
            where a bloom is NOT already present, so persistence-of-state is
            uninformative. This isolates whether antecedent multimodal signal forecasts
            a NEW bloom. 0.50 is chance."
          />
          <Metric
            size="lg"
            label="Onset AUROC 95% CI"
            value="[0.691, 0.777]"
            copyValue="0.733 onset AUROC, site-clustered bootstrap 95% CI [0.691, 0.777]"
            status="nominal"
            sub="site-clustered bootstrap — excludes 0.5"
            tooltip="Site-clustered bootstrap 95% CI on the onset-only fusion AUROC. It
            excludes the 0.50 no-skill line, so the early-warning skill is real."
          />
          <Metric
            size="lg"
            label="Median realized lead"
            value="~14"
            unit="days"
            status="nominal"
            sub={BLOOM_EARLY_WARNING.horizonNote}
            tooltip="Median lead time from the antecedent prediction to the in-situ
            visit at which the new bloom is confirmed. The horizon is set by the WQP
            in-situ visit cadence, not the 5-day Sentinel-2 revisit."
          />
          <Metric
            size="lg"
            label="Recall at ~14% false alarm"
            value="~45%"
            status="watch"
            sub={`${BLOOM_EARLY_WARNING.nOnsets} true onsets / ${BLOOM_EARLY_WARNING.nOnsetPairs} pairs`}
            tooltip="At a ~14% false-alarm rate on the onset-only test, the forecast
            recovers about 45% of new blooms (940 onset-eligible pairs, 207 true
            new-bloom onsets)."
          />
        </div>

        <div
          style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))',
            gap: 'var(--s-6)',
          }}
        >
          <Card>
            <CardTitle>
              <Waves size={14} strokeWidth={1.75} style={{ color: 'var(--nominal)' }} aria-hidden />
              A genuine early-warning capability
            </CardTitle>
            <p
              style={{
                margin: 0,
                fontSize: 'var(--fs-13)',
                lineHeight: 'var(--lh-normal)',
                color: 'var(--text-secondary)',
              }}
            >
              On the strict onset-only test ({BLOOM_EARLY_WARNING.nOnsetPairs} onset-eligible
              pairs, {BLOOM_EARLY_WARNING.nOnsets} true new-bloom onsets, where
              persistence-of-state is uninformative), fusion reaches AUROC{' '}
              <span className="mono" style={{ fontWeight: 600 }}>
                {fmt(BLOOM_EARLY_WARNING.onsetFusionAuroc)}
              </span>{' '}
              with a 95% CI of{' '}
              <span className="mono" style={{ fontWeight: 600 }}>
                [{fmt(BLOOM_EARLY_WARNING.onsetFusionCI[0])}, {fmt(BLOOM_EARLY_WARNING.onsetFusionCI[1])}]
              </span>{' '}
              — above the 0.5 no-skill line. Every predictor is strictly antecedent, and
              the split is leakage-free within-region (GroupKFold-by-site).
            </p>
            <div
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: 'var(--s-2)',
                marginTop: 'var(--s-4)',
                fontSize: 'var(--fs-12)',
                color: 'var(--text-tertiary)',
              }}
            >
              <Clock size={12} strokeWidth={1.75} aria-hidden />
              {BLOOM_EARLY_WARNING.horizon}
            </div>
          </Card>

          <Card style={{ borderColor: 'var(--border-strong)' }}>
            <CardTitle>
              <Info size={14} strokeWidth={1.75} style={{ color: 'var(--text-tertiary)' }} aria-hidden />
              Honest caveat · not a fusion-beats-single claim
            </CardTitle>
            <p
              style={{
                margin: 0,
                fontSize: 'var(--fs-13)',
                lineHeight: 'var(--lh-normal)',
                color: 'var(--text-secondary)',
              }}
            >
              {BLOOM_EARLY_WARNING.caveat}
            </p>
            <div style={{ marginTop: 'var(--s-4)' }}>
              <StatusBadge kind="neutral">
                fusion − best single {signed(BLOOM_EARLY_WARNING.fusionMinusBestSingle)} · CI [{signed(BLOOM_EARLY_WARNING.fusionMinusBestSingleCI[0])}, {signed(BLOOM_EARLY_WARNING.fusionMinusBestSingleCI[1])}] includes 0
              </StatusBadge>
            </div>
          </Card>
        </div>
      </div>

      {/* ================================================================
          Bottom — dataset provenance + one-line takeaway
          ================================================================ */}
      <div style={{ marginTop: 'var(--s-9)' }}>
        <SectionHeader as="h3" title="Dataset & protocol" />
        <div
          style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fit, minmax(160px, 1fr))',
            gap: 'var(--s-6)',
            padding: 'var(--s-6)',
            background: 'var(--bg-subtle)',
            border: '1px solid var(--border-subtle)',
            borderRadius: 'var(--r-lg)',
          }}
        >
          <Metric
            label="Co-located tiles"
            value={FUSION_DATASET.nTiles.toLocaleString()}
            size="md"
            tooltip="Usable Sentinel-2 tiles co-located with in-situ microcystin
            measurements across the fusion dataset."
          />
          <Metric
            label="Sites"
            value={FUSION_DATASET.nSites}
            size="md"
            tooltip="Distinct monitoring sites. Station-grouped CV never splits a
            single site across train and test."
          />
          <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--s-1)' }}>
            <span className="eyebrow">Cross-validation</span>
            <span style={{ fontSize: 'var(--fs-13)', color: 'var(--text-secondary)', lineHeight: 1.4 }}>
              {FUSION_DATASET.cv}
            </span>
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--s-1)' }}>
            <span className="eyebrow">Sources</span>
            <span
              className="mono"
              style={{ fontSize: 'var(--fs-12)', color: 'var(--text-secondary)', lineHeight: 1.5 }}
            >
              Sentinel-2 · in-situ pigment · WQP drivers
            </span>
          </div>
        </div>

        <div
          style={{
            display: 'flex',
            alignItems: 'flex-start',
            gap: 'var(--s-3)',
            marginTop: 'var(--s-6)',
            padding: 'var(--s-5)',
            background: 'var(--accent-wash)',
            border: '1px solid var(--accent-border)',
            borderRadius: 'var(--r-md)',
          }}
        >
          <div style={{ flex: 1 }}>
            <div className="eyebrow" style={{ marginBottom: 'var(--s-2)' }}>
              Takeaway
            </div>
            <p
              style={{
                margin: 0,
                fontSize: 'var(--fs-14)',
                lineHeight: 'var(--lh-normal)',
                color: 'var(--text-primary)',
              }}
            >
              Multimodal fusion genuinely combines complementary modalities — but only
              within a calibrated region. It is a real capability with a clearly stated
              boundary, not a universal claim.
            </p>
          </div>
          <CopyButton
            value={FUSION_SUMMARY}
            label="fusion summary"
            size={15}
          />
        </div>
      </div>
    </>
  );
}
