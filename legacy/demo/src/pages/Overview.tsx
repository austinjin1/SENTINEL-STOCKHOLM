import {
  useCallback,
  useEffect,
  useRef,
  useState,
  type CSSProperties,
  type ReactNode,
} from 'react';
import { Link } from 'react-router-dom';
import {
  ArrowRight,
  Boxes,
  Compass,
  Database,
  FileText,
  Layers,
  MapPin,
  ShieldCheck,
  X,
} from 'lucide-react';
import { SectionHeader, Metric, Callout, StatusBadge, Tooltip } from '../ui';
import { DATA_SUMMARY } from '../content/datasets';
import { MODELS } from '../content/models';
import { RETRACTIONS } from '../content/methodology';

/**
 * Overview — the SENTINEL landing page (route /).
 *
 * A calm, factual masthead (what SENTINEL is, no hype), a compact "at a glance"
 * strip of the real headline metrics (mono, count-up), an honest validated-vs-
 * simulation scope callout, and cards linking to Models / Fusion / Data / Docs /
 * Live map. Includes the guided-tour entry point.
 *
 * Genomics-tool idiom (gnomAD homepage): flat fills, hairline borders, tight
 * elevation, monospace for every number/ID, one calm blue accent, semantic
 * status colors that encode DATA only. Every metric carries a tooltip that
 * explains what it is and the holdout protocol. Fully keyboard navigable;
 * respects prefers-reduced-motion.
 *
 * Every number is real and held-out (verified against README.md + results/*.json
 * via src/content/*). No retracted/fabricated figure is reintroduced.
 */

// Encoder-status roll-up for the honest-scope strip (from the model catalog).
const N_VALIDATED = MODELS.filter(
  (m) => m.status === 'validated' && m.kind === 'encoder',
).length;

// Model-catalog counts, derived so the card stats can never drift from the data.
const N_ENCODERS = MODELS.filter((m) => m.kind === 'encoder').length;
const N_DOWNSTREAM = MODELS.filter((m) => m.kind !== 'encoder').length;

export default function Overview() {
  const tour = useGuidedTour();

  return (
    <>
      {/* ── Masthead ─────────────────────────────────────────────────────── */}
      <section
        aria-labelledby="ov-title"
        style={{
          animation: 'fade-in-up var(--dur-base) var(--ease-out)',
        }}
      >
        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: 'var(--s-3)',
            marginBottom: 'var(--s-4)',
            flexWrap: 'wrap',
          }}
        >
          <StatusBadge kind="nominal">Integrity-remediated</StatusBadge>
          <span
            style={{
              fontSize: 'var(--fs-12)',
              color: 'var(--text-tertiary)',
              fontFamily: 'var(--font-sans)',
            }}
          >
            Every headline number is real and measured under a stated holdout.
          </span>
        </div>

        <div className="eyebrow" style={{ marginBottom: 'var(--s-3)' }}>
          Multimodal freshwater-contamination monitoring
        </div>
        <h1
          id="ov-title"
          style={{
            fontSize: 'var(--fs-40)',
            fontWeight: 'var(--fw-bold)' as unknown as number,
            letterSpacing: 'var(--tracking-tight)',
            lineHeight: 'var(--lh-tight)',
            color: 'var(--text-primary)',
            maxWidth: '20ch',
            margin: 0,
          }}
        >
          SENTINEL
        </h1>
        <p
          style={{
            margin: 'var(--s-4) 0 0',
            maxWidth: '72ch',
            fontSize: 'var(--fs-16)',
            lineHeight: 'var(--lh-normal)',
            color: 'var(--text-secondary)',
          }}
        >
          A research system that learns joint representations across five sensing
          modalities — in-situ sensors, satellite optics, aquatic microbiomes,
          molecular transcriptomics, and organism behavior — to detect and
          forecast freshwater-quality failures. Each model is benchmarked on
          real, public data under a stated held-out protocol. Where a model does
          not beat its baseline, or does not transfer across regions, we report
          it plainly.
        </p>

        <div
          style={{
            display: 'flex',
            gap: 'var(--s-3)',
            marginTop: 'var(--s-6)',
            flexWrap: 'wrap',
            alignItems: 'center',
          }}
        >
          <button
            type="button"
            onClick={tour.start}
            style={primaryButtonStyle}
            onMouseEnter={(e) =>
              (e.currentTarget.style.background = 'var(--accent-hover)')
            }
            onMouseLeave={(e) =>
              (e.currentTarget.style.background = 'var(--accent)')
            }
          >
            <Compass size={15} strokeWidth={1.9} aria-hidden />
            Take the guided tour
          </button>
          <Link to="/docs" style={secondaryLinkStyle}>
            Read the holdout protocol
            <ArrowRight size={14} strokeWidth={1.9} aria-hidden />
          </Link>
        </div>
      </section>

      {/* ── At a glance ──────────────────────────────────────────────────── */}
      <section
        id="tour-glance"
        aria-labelledby="ov-glance"
        style={{ marginTop: 'var(--s-10)', scrollMarginTop: 'calc(var(--nav-height) + var(--s-6))' }}
        {...tourTarget(tour, 'glance')}
      >
        <SectionHeader
          id="ov-glance"
          eyebrow="At a glance"
          title="Verified headline results"
          description="Held-out performance for the core models. Hover any metric for its definition and the exact holdout protocol; copy a value to cite it."
          actions={
            <Link to="/models" style={inlineLinkStyle}>
              All models
              <ArrowRight size={13} strokeWidth={1.9} aria-hidden />
            </Link>
          }
        />

        <div
          style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fit, minmax(210px, 1fr))',
            gap: 0,
            background: 'var(--bg-surface)',
            border: '1px solid var(--border-subtle)',
            borderRadius: 'var(--r-lg)',
            overflow: 'hidden',
            boxShadow: 'var(--shadow-sm)',
          }}
        >
          {HEADLINE_METRICS.map((m, i) => (
            <div
              key={m.key}
              style={{
                padding: 'var(--s-6)',
                borderRight: '1px solid var(--border-subtle)',
                borderBottom: '1px solid var(--border-subtle)',
                // The last cell in each row / the last row lose their extra
                // hairline visually via the container's overflow clip; a light
                // inset keeps the grid crisp at any column count.
                boxSizing: 'border-box',
                animation: `fade-in-up var(--dur-base) var(--ease-out) ${i * 40}ms both`,
              }}
            >
              <Metric
                size="lg"
                countUp
                label={m.label}
                value={m.value}
                unit={m.unit}
                status={m.status}
                sub={m.sub}
                tooltip={m.tooltip}
                copyValue={m.copyValue}
              />
            </div>
          ))}
        </div>

        <p
          style={{
            margin: 'var(--s-3) 0 0',
            fontSize: 'var(--fs-12)',
            color: 'var(--text-tertiary)',
            fontFamily: 'var(--font-sans)',
          }}
        >
          <span className="mono">{N_VALIDATED}</span> of{' '}
          <span className="mono">5</span> modality encoders are validated on real
          held-out data; the rest are reported as genuine ceilings, region-limited
          results, or simulation. See{' '}
          <Link to="/models" style={inlineLinkStyle}>
            Models
          </Link>{' '}
          for the full roster and every correction.
        </p>
      </section>

      {/* ── Honest scope ─────────────────────────────────────────────────── */}
      <section
        id="tour-scope"
        aria-labelledby="ov-scope"
        style={{ marginTop: 'var(--s-10)', scrollMarginTop: 'calc(var(--nav-height) + var(--s-6))' }}
        {...tourTarget(tour, 'scope')}
      >
        <SectionHeader
          id="ov-scope"
          eyebrow="Honest scope"
          title="What is validated, and where it stops"
          description="This project completed an integrity remediation. The honesty is a feature: we separate genuinely validated results from region-limited findings and simulation-only components."
        />

        <div
          style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))',
            gap: 'var(--s-4)',
            marginBottom: 'var(--s-4)',
          }}
        >
          <Callout tone="verified" title="Validated on real held-out data">
            AquaSSM reconstruction (RMSE{' '}
            <span className="mono">0.83</span>) and its label-free NEON
            lead-time detector; HydroViT water-temperature (R²{' '}
            <span className="mono">0.729</span>); BioMotion ECOTOX effect
            classification (AUROC <span className="mono">0.85</span>);
            MicroBiomeNet source attribution under a{' '}
            <em>true geographic holdout</em> (macro-F1{' '}
            <span className="mono">0.676</span>).
          </Callout>

          <Callout tone="verified" title="Generalizes across sites (self-normalized temporal detection)">
            The label-free lead-time detector transfers to streams it has never
            seen: it scores only a self-normalized deviation from each site's own
            clean baseline, so its alarm threshold is a near-constant{' '}
            <span className="mono">2.79σ</span> — invariant across every
            leave-one-control-out fit, catching the identical{' '}
            <span className="mono">11 / 15</span> NEON hypoxia episodes at a{' '}
            matched <span className="mono">5%</span> false-alarm rate on unseen
            sites, with no per-site tuning.
          </Callout>

          <Callout tone="limitation" title="Fusion is scoped within-region — cross-region needs local calibration">
            On microcystin, 3-modality fusion beats every single modality within
            a calibrated region (<span className="mono">0.82</span> vs{' '}
            <span className="mono">0.68</span>). Its deployment scope is explicit:
            it is a within-region tool — under a cross-region holdout it reads
            ~<span className="mono">0.49</span>, because the toxin depends on the
            unobserved <span className="mono">mcyD</span> toxic-strain genotype,
            so crossing regions needs local calibration or the genotype modality.
          </Callout>

          <Callout tone="info" title="Simulation only — labeled as such">
            The Stream-Network GNN uses real river topology but synthetic
            propagation labels; the conformal pipeline and causal screening are
            proofs-of-concept. The Digital Twin loses to persistence beyond a
            1-day horizon. None are validated detection.
          </Callout>
        </div>

        <div
          style={{
            display: 'flex',
            alignItems: 'flex-start',
            gap: 'var(--s-3)',
            padding: 'var(--s-4) var(--s-5)',
            background: 'var(--bg-subtle)',
            border: '1px solid var(--border-subtle)',
            borderRadius: 'var(--r-md)',
          }}
        >
          <ShieldCheck
            size={16}
            strokeWidth={1.9}
            style={{ color: 'var(--accent)', flexShrink: 0, marginTop: 2 }}
            aria-hidden
          />
          <p
            style={{
              margin: 0,
              fontSize: 'var(--fs-13)',
              lineHeight: 'var(--lh-normal)',
              color: 'var(--text-secondary)',
            }}
          >
            <strong style={{ color: 'var(--text-primary)' }}>
              Central methodological finding.
            </strong>{' '}
            Transferable representations — higher taxonomy, rank-normalized genes,
            red-edge spectral indices — recover out-of-distribution transfer where
            raw, region-specific features collapse (e.g. MicroBiomeNet macro-F1{' '}
            <span className="mono">0.405 → 0.676</span> from raw OTUs to family/
            genus taxonomy). The retraction log and every corrected figure live on{' '}
            <Link to="/docs" style={inlineLinkStyle}>
              Docs
            </Link>
            .
          </p>
        </div>
      </section>

      {/* ── Explore (section cards) ──────────────────────────────────────── */}
      <section
        id="tour-sections"
        aria-labelledby="ov-sections"
        style={{ marginTop: 'var(--s-10)', scrollMarginTop: 'calc(var(--nav-height) + var(--s-6))' }}
        {...tourTarget(tour, 'sections')}
      >
        <SectionHeader
          id="ov-sections"
          eyebrow="Explore"
          title="Sections"
          description="Each section presents real, within-protocol numbers with copy-to-cite values and per-metric tooltips."
        />
        <div
          style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))',
            gap: 'var(--s-4)',
          }}
        >
          {SECTION_CARDS.map((c) => (
            <SectionCard key={c.to} card={c} />
          ))}
        </div>
      </section>

      {/* ── Data provenance footnote ─────────────────────────────────────── */}
      <section
        style={{
          marginTop: 'var(--s-10)',
          paddingTop: 'var(--s-6)',
          borderTop: '1px solid var(--border-subtle)',
        }}
      >
        <p
          style={{
            margin: 0,
            fontSize: 'var(--fs-12)',
            color: 'var(--text-tertiary)',
            fontFamily: 'var(--font-sans)',
            lineHeight: 'var(--lh-normal)',
            maxWidth: '80ch',
          }}
        >
          Built entirely on real, public sources: USGS NWIS, NEON, ESA Sentinel-2
          (Microsoft Planetary Computer), the Earth Microbiome Project 16S, NCBI
          GEO, EPA ECOTOX, the Water Quality Portal / NOAA HABs, and NHDPlus.
          SENTINEL-DB harmonizes{' '}
          <Tooltip
            label={DATA_SUMMARY.rawRecordsNote}
            side="top"
          >
            <span className="mono" style={{ borderBottom: '1px dotted var(--text-quaternary)', cursor: 'help' }}>
              {DATA_SUMMARY.rawRecords}
            </span>
          </Tooltip>{' '}
          raw pre-QC records across{' '}
          <span className="mono">{DATA_SUMMARY.nSources}</span> sources and{' '}
          <span className="mono">{DATA_SUMMARY.nCountries}</span> countries.
          Post-QC volume is far smaller (
          <span className="mono">{DATA_SUMMARY.postQcRecords}</span>) — see{' '}
          <Link to="/data" style={inlineLinkStyle}>
            Data
          </Link>
          .
        </p>
      </section>

      {/* ── Guided tour overlay ──────────────────────────────────────────── */}
      <GuidedTour tour={tour} />
    </>
  );
}

/* ══════════════════════════════════════════════════════════════════════════
   Headline metrics — all REAL and held-out (verified vs README.md).
   ══════════════════════════════════════════════════════════════════════════ */
interface HeadlineMetric {
  key: string;
  label: string;
  value: string;
  unit?: string;
  status: 'nominal' | 'watch' | 'alert' | 'neutral';
  sub: ReactNode;
  tooltip: ReactNode;
  copyValue?: string;
}

const HEADLINE_METRICS: HeadlineMetric[] = [
  {
    key: 'leadtime',
    label: 'Median lead time',
    value: '36',
    unit: 'days',
    status: 'nominal',
    sub: '11 / 15 NEON DO-hypoxia episodes',
    copyValue: 'median 36-day early warning (NEON, 11/15 episodes, matched 5.0% FPR)',
    tooltip:
      'Median early warning of the label-free AquaSSM rolling-CUSUM detector ahead of the DO<4 mg/L hypoxia onset in the NEON record, sensor-only, at a matched 5.0% held-out false-alarm rate on clean NEON controls. It detects 11 of 15 real episodes — a Pareto improvement over the point-threshold baseline (5/15, 25.5-day lead) at every operating point.',
  },
  {
    key: 'fusion',
    label: 'Fusion AUROC (microcystin)',
    value: '0.82',
    status: 'nominal',
    sub: 'vs 0.68 best single modality',
    copyValue: '0.82 AUROC (3-modality microcystin fusion, station-grouped CV)',
    tooltip:
      'Three-modality fusion (satellite optics + in-situ pigment + physicochemical drivers) at the WHO drinking-water threshold, under station-grouped CV with site-clustered bootstrap CIs. Each modality adds CI-significant signal — but this collapses to ~0.49 across unseen regions.',
  },
  {
    key: 'hydrovit',
    label: 'Water-temp R²',
    value: '0.729',
    status: 'nominal',
    sub: 'HydroViT · Sentinel-2, station holdout',
    copyValue: '0.729 R² (HydroViT water temperature, station holdout)',
    tooltip:
      'HydroViT water-temperature prediction from real Sentinel-2 L2A imagery paired with in-situ measurements, under a station holdout (whole sites excluded from training). Replaces the retracted fabricated 0.893.',
  },
  {
    key: 'microbiome',
    label: 'Source attribution F1',
    value: '0.676',
    status: 'watch',
    sub: 'true geographic holdout',
    copyValue: '0.676 macro-F1 (MicroBiomeNet, true geographic holdout)',
    tooltip:
      'MicroBiomeNet 8-class aquatic source attribution under a TRUE geographic holdout, using transferable family/genus taxonomic features (0.405 on raw region-specific OTUs). The old 0.831 was random-split leakage.',
  },
  {
    key: 'aquassm',
    label: 'Sensor recon RMSE',
    value: '0.83',
    status: 'nominal',
    sub: 'AquaSSM · 62 unseen sites',
    copyValue: '0.83 RMSE (AquaSSM sensor reconstruction, station holdout)',
    tooltip:
      'AquaSSM self-supervised sensor-stream reconstruction error on held-out stations (lower is better). The reconstruction backbone underlies the label-free lead-time detector.',
  },
  {
    key: 'rawdata',
    label: 'Raw records (pre-QC)',
    value: '390',
    unit: 'M',
    status: 'neutral',
    sub: '14 public sources · 105 countries',
    copyValue: '~390M raw pre-QC records (14 public sources)',
    tooltip:
      'Raw, pre-QC record count (~85 GB) across 14 real public data sources, dominated by NEON high-frequency sonde data (351.7M raw). Usable post-QC volume is far smaller (~52M).',
  },
];

/* ══════════════════════════════════════════════════════════════════════════
   Section cards.
   ══════════════════════════════════════════════════════════════════════════ */
interface CardDef {
  to: string;
  title: string;
  Icon: typeof Boxes;
  blurb: string;
  stat: string;
  statLabel: string;
}

const SECTION_CARDS: CardDef[] = [
  {
    to: '/models',
    title: 'Models',
    Icon: Boxes,
    blurb:
      'Five modality encoders plus downstream models, each with its real held-out metric, holdout protocol, and every corrected figure.',
    stat: `${N_ENCODERS} + ${N_DOWNSTREAM}`,
    statLabel: 'encoders + downstream',
  },
  {
    to: '/fusion',
    title: 'Fusion',
    Icon: Layers,
    blurb:
      'The nuanced multimodal story on microcystin: fusion is a calibrated within-region win, scoped explicitly at the cross-region boundary. The full per-threshold breakdown.',
    stat: '0.82 / 0.49',
    statLabel: 'within-region / cross-region',
  },
  {
    to: '/data',
    title: 'Data',
    Icon: Database,
    blurb:
      'SENTINEL-DB: every source is real and public, with raw vs post-QC counts stated honestly and access routes for each dataset.',
    stat: '~390M',
    statLabel: 'raw records, 14 sources',
  },
  {
    to: '/docs',
    title: 'Docs',
    Icon: FileText,
    blurb:
      'Holdout protocols, metric definitions, the simulation-only scope, and a transparent retraction log from the integrity remediation.',
    stat: String(RETRACTIONS.length),
    statLabel: 'figures retracted & corrected',
  },
  {
    to: '/live',
    title: 'Live map',
    Icon: MapPin,
    blurb:
      'An operational console over real station geometry, replaying cached case-study events across the modality stack.',
    stat: 'MapLibre',
    statLabel: 'real station geometry',
  },
];

function SectionCard({ card }: { card: CardDef }) {
  const { Icon } = card;
  const [hover, setHover] = useState(false);
  return (
    <Link
      to={card.to}
      onMouseEnter={() => setHover(true)}
      onMouseLeave={() => setHover(false)}
      style={{
        display: 'flex',
        flexDirection: 'column',
        gap: 'var(--s-3)',
        padding: 'var(--s-5)',
        background: 'var(--bg-surface)',
        border: `1px solid ${hover ? 'var(--border-strong)' : 'var(--border-subtle)'}`,
        borderRadius: 'var(--r-lg)',
        textDecoration: 'none',
        boxShadow: hover ? 'var(--shadow-md)' : 'var(--shadow-xs)',
        transform: hover ? 'translateY(-1px)' : 'none',
        transition:
          'box-shadow var(--dur-fast) var(--ease-standard), border-color var(--dur-fast) var(--ease-standard), transform var(--dur-fast) var(--ease-standard)',
      }}
    >
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          gap: 'var(--s-3)',
        }}
      >
        <span
          style={{
            display: 'inline-flex',
            alignItems: 'center',
            justifyContent: 'center',
            width: 30,
            height: 30,
            borderRadius: 'var(--r-md)',
            background: 'var(--accent-wash)',
            border: '1px solid var(--accent-border)',
            color: 'var(--accent-strong)',
            flexShrink: 0,
          }}
        >
          <Icon size={16} strokeWidth={1.9} aria-hidden />
        </span>
        <ArrowRight
          size={16}
          strokeWidth={1.9}
          aria-hidden
          style={{
            color: hover ? 'var(--accent)' : 'var(--text-quaternary)',
            transform: hover ? 'translateX(2px)' : 'none',
            transition:
              'color var(--dur-fast) var(--ease-standard), transform var(--dur-fast) var(--ease-standard)',
          }}
        />
      </div>

      <div
        style={{
          fontSize: 'var(--fs-18)',
          fontWeight: 'var(--fw-semibold)' as unknown as number,
          color: 'var(--text-primary)',
        }}
      >
        {card.title}
      </div>
      <p
        style={{
          margin: 0,
          fontSize: 'var(--fs-13)',
          lineHeight: 'var(--lh-normal)',
          color: 'var(--text-secondary)',
          flexGrow: 1,
        }}
      >
        {card.blurb}
      </p>
      <div
        style={{
          display: 'flex',
          alignItems: 'baseline',
          gap: 'var(--s-2)',
          paddingTop: 'var(--s-2)',
          borderTop: '1px solid var(--border-subtle)',
        }}
      >
        <span className="metric-l" style={{ color: 'var(--text-primary)' }}>
          {card.stat}
        </span>
        <span
          style={{
            fontSize: 'var(--fs-11)',
            color: 'var(--text-tertiary)',
            fontFamily: 'var(--font-sans)',
          }}
        >
          {card.statLabel}
        </span>
      </div>
    </Link>
  );
}

/* ══════════════════════════════════════════════════════════════════════════
   Guided tour — a lightweight, keyboard-navigable spotlight walkthrough that
   highlights each section in turn. Persists a "seen" flag so it does not nag
   returning visitors, and respects prefers-reduced-motion.
   ══════════════════════════════════════════════════════════════════════════ */
interface TourStep {
  id: 'glance' | 'scope' | 'sections';
  title: string;
  body: ReactNode;
}

const TOUR_STEPS: TourStep[] = [
  {
    id: 'glance',
    title: 'Read the headline numbers',
    body: (
      <>
        Every figure here is real and measured under a stated holdout. Numbers
        render in monospace; hover the{' '}
        <span style={{ fontWeight: 600 }}>info</span> icon on any metric for its
        definition and exact protocol, and use the copy button to cite it.
      </>
    ),
  },
  {
    id: 'scope',
    title: 'Know where it stops',
    body: (
      <>
        We separate genuinely validated results from region-limited findings and
        simulation-only components — showing both what generalizes (cross-site
        temporal detection) and where fusion is scoped (within-region), plus the
        simulation scope, up front. The honesty is the result.
      </>
    ),
  },
  {
    id: 'sections',
    title: 'Go deeper',
    body: (
      <>
        Each section carries the full within-protocol tables, confidence
        intervals, and — on Docs — the retraction log of every corrected figure.
        You can revisit this tour any time from the masthead.
      </>
    ),
  },
];

const TOUR_SEEN_KEY = 'sentinel.overview.tour.seen.v1';

interface TourApi {
  active: boolean;
  index: number;
  start: () => void;
  next: () => void;
  prev: () => void;
  stop: () => void;
  registerTarget: (id: string, el: HTMLElement | null) => void;
  getTarget: (id: string) => HTMLElement | null;
}

function useGuidedTour(): TourApi {
  const [active, setActive] = useState(false);
  const [index, setIndex] = useState(0);
  const targets = useRef<Record<string, HTMLElement | null>>({});

  const registerTarget = useCallback((id: string, el: HTMLElement | null) => {
    targets.current[id] = el;
  }, []);

  const getTarget = useCallback((id: string) => targets.current[id] ?? null, []);

  const stop = useCallback(() => {
    setActive(false);
    try {
      localStorage.setItem(TOUR_SEEN_KEY, '1');
    } catch {
      /* storage may be unavailable — non-fatal */
    }
  }, []);

  const start = useCallback(() => {
    setIndex(0);
    setActive(true);
  }, []);

  // First-visit onboarding: auto-open the tour once, unless the visitor has
  // already seen (and dismissed) it. The "seen" flag is set on stop(). Runs on
  // an async timer so the sections have mounted and can be measured/spotlit.
  useEffect(() => {
    let seen: boolean;
    try {
      seen = localStorage.getItem(TOUR_SEEN_KEY) === '1';
    } catch {
      /* storage unavailable — treat as seen to avoid nagging */
      return;
    }
    if (seen) return;
    const id = window.setTimeout(() => {
      setIndex(0);
      setActive(true);
    }, 600);
    return () => window.clearTimeout(id);
  }, []);

  const next = useCallback(() => {
    setIndex((i) => {
      if (i >= TOUR_STEPS.length - 1) {
        stop();
        return i;
      }
      return i + 1;
    });
  }, [stop]);

  const prev = useCallback(() => setIndex((i) => Math.max(0, i - 1)), []);

  // Scroll the active step's target into view (respecting reduced motion).
  useEffect(() => {
    if (!active) return;
    const step = TOUR_STEPS[index];
    const el = targets.current[step.id];
    if (!el) return;
    const reduced = window.matchMedia?.(
      '(prefers-reduced-motion: reduce)',
    ).matches;
    el.scrollIntoView({
      behavior: reduced ? 'auto' : 'smooth',
      block: 'center',
    });
  }, [active, index]);

  // Keyboard controls while the tour is open.
  useEffect(() => {
    if (!active) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') stop();
      else if (e.key === 'ArrowRight' || e.key === 'Enter') next();
      else if (e.key === 'ArrowLeft') prev();
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [active, next, prev, stop]);

  return {
    active,
    index,
    start,
    next,
    prev,
    stop,
    registerTarget,
    getTarget,
  };
}

/** Props that wire a section element into the tour's spotlight registry. */
function tourTarget(tour: TourApi, id: TourStep['id']) {
  return {
    ref: (el: HTMLElement | null) => tour.registerTarget(id, el),
    'data-tour-active':
      tour.active && TOUR_STEPS[tour.index].id === id ? 'true' : undefined,
  };
}

function GuidedTour({ tour }: { tour: TourApi }) {
  const [rect, setRect] = useState<DOMRect | null>(null);
  const cardRef = useRef<HTMLDivElement>(null);
  const step = TOUR_STEPS[tour.index];

  const { active, getTarget } = tour;

  // Focus management: on open, move focus into the step card and trap Tab
  // within it; restore focus to the trigger on close. Esc/Arrow keys are
  // already wired in useGuidedTour.
  useEffect(() => {
    if (!active) return;
    const prevActive = document.activeElement as HTMLElement | null;
    const focusFirst = () => {
      const root = cardRef.current;
      if (!root) return;
      const focusable = root.querySelectorAll<HTMLElement>(
        'a[href], button:not([disabled]), textarea, input, select, [tabindex]:not([tabindex="-1"])',
      );
      // Prefer the primary Next/Done control (last button) as the initial focus.
      (focusable[focusable.length - 1] ?? focusable[0])?.focus();
    };
    const raf = requestAnimationFrame(focusFirst);
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key !== 'Tab') return;
      const root = cardRef.current;
      if (!root) return;
      const focusable = root.querySelectorAll<HTMLElement>(
        'a[href], button:not([disabled]), textarea, input, select, [tabindex]:not([tabindex="-1"])',
      );
      if (focusable.length === 0) return;
      const first = focusable[0];
      const last = focusable[focusable.length - 1];
      if (e.shiftKey && document.activeElement === first) {
        e.preventDefault();
        last.focus();
      } else if (!e.shiftKey && document.activeElement === last) {
        e.preventDefault();
        first.focus();
      }
    };
    document.addEventListener('keydown', onKeyDown, true);
    return () => {
      cancelAnimationFrame(raf);
      document.removeEventListener('keydown', onKeyDown, true);
      prevActive?.focus?.();
    };
    // Re-run only on open/close, not per step, so per-step Next clicks keep focus.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [active]);

  // Track the highlighted target's box so the card can sit beneath it and the
  // scrim can cut a hole around it. All measurements happen async (rAF / timer
  // / DOM events), never synchronously in the effect body.
  useEffect(() => {
    if (!active) return;
    const measure = () => {
      const el = getTarget(step.id);
      setRect(el ? el.getBoundingClientRect() : null);
    };
    // First measure on the next frame, then again after the smooth scroll
    // settles, and on any resize/scroll.
    const raf = requestAnimationFrame(measure);
    const id = window.setTimeout(measure, 280);
    window.addEventListener('resize', measure);
    window.addEventListener('scroll', measure, true);
    return () => {
      cancelAnimationFrame(raf);
      window.clearTimeout(id);
      window.removeEventListener('resize', measure);
      window.removeEventListener('scroll', measure, true);
    };
  }, [active, step.id, getTarget]);

  // While the tour is closed the whole overlay is unmounted, so a retained
  // `rect` is never rendered; on reopen the measure effect recomputes it on the
  // next frame. No reset effect is needed (and none should call setState).
  if (!tour.active) return null;

  const pad = 8;
  const highlight =
    rect &&
    ({
      position: 'fixed',
      top: rect.top - pad,
      left: rect.left - pad,
      width: rect.width + pad * 2,
      height: rect.height + pad * 2,
      border: '2px solid var(--accent)',
      borderRadius: 'var(--r-lg)',
      boxShadow: '0 0 0 9999px var(--bg-scrim)',
      pointerEvents: 'none',
      zIndex: 'var(--z-overlay)' as unknown as number,
      transition: 'all var(--dur-base) var(--ease-standard)',
    } as CSSProperties);

  return (
    <div
      role="dialog"
      aria-modal="true"
      aria-label={`Guided tour, step ${tour.index + 1} of ${TOUR_STEPS.length}`}
    >
      {/* Scrim: a full-screen dim, with a cut-out ring around the target. */}
      {highlight ? (
        <div style={highlight} aria-hidden />
      ) : (
        <div
          aria-hidden
          onClick={tour.stop}
          style={{
            position: 'fixed',
            inset: 0,
            background: 'var(--bg-scrim)',
            zIndex: 'var(--z-overlay)' as unknown as number,
          }}
        />
      )}

      {/* Step card, pinned to the bottom center. */}
      <div
        ref={cardRef}
        style={{
          position: 'fixed',
          bottom: 'var(--s-8)',
          left: '50%',
          transform: 'translateX(-50%)',
          width: 'min(440px, calc(100vw - var(--s-8)))',
          background: 'var(--bg-overlay)',
          border: '1px solid var(--border-strong)',
          borderRadius: 'var(--r-lg)',
          boxShadow: 'var(--shadow-lg)',
          zIndex: 'var(--z-popover)' as unknown as number,
          padding: 'var(--s-5)',
          animation: 'fade-in-up var(--dur-base) var(--ease-out)',
        }}
      >
        <div
          style={{
            display: 'flex',
            alignItems: 'flex-start',
            justifyContent: 'space-between',
            gap: 'var(--s-4)',
            marginBottom: 'var(--s-2)',
          }}
        >
          <div className="eyebrow">
            Guided tour · step{' '}
            <span className="mono">
              {tour.index + 1}/{TOUR_STEPS.length}
            </span>
          </div>
          <button
            type="button"
            onClick={tour.stop}
            aria-label="Close tour"
            style={{
              display: 'inline-flex',
              color: 'var(--text-tertiary)',
              padding: 2,
              borderRadius: 'var(--r-xs)',
            }}
          >
            <X size={15} strokeWidth={1.9} aria-hidden />
          </button>
        </div>

        <div
          style={{
            fontSize: 'var(--fs-16)',
            fontWeight: 'var(--fw-semibold)' as unknown as number,
            color: 'var(--text-primary)',
            marginBottom: 'var(--s-2)',
          }}
        >
          {step.title}
        </div>
        <p
          style={{
            margin: 0,
            fontSize: 'var(--fs-13)',
            lineHeight: 'var(--lh-normal)',
            color: 'var(--text-secondary)',
          }}
        >
          {step.body}
        </p>

        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            gap: 'var(--s-3)',
            marginTop: 'var(--s-5)',
          }}
        >
          <div style={{ display: 'flex', gap: 'var(--s-2)' }} aria-hidden>
            {TOUR_STEPS.map((s, i) => (
              <span
                key={s.id}
                style={{
                  width: i === tour.index ? 18 : 6,
                  height: 6,
                  borderRadius: 'var(--r-pill)',
                  background:
                    i === tour.index
                      ? 'var(--accent)'
                      : 'var(--border-strong)',
                  transition: 'width var(--dur-fast) var(--ease-standard)',
                }}
              />
            ))}
          </div>
          <div style={{ display: 'flex', gap: 'var(--s-2)' }}>
            {tour.index > 0 && (
              <button
                type="button"
                onClick={tour.prev}
                style={ghostButtonStyle}
              >
                Back
              </button>
            )}
            <button type="button" onClick={tour.next} style={smallPrimaryStyle}>
              {tour.index === TOUR_STEPS.length - 1 ? 'Done' : 'Next'}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

/* ══════════════════════════════════════════════════════════════════════════
   Local style tokens.
   ══════════════════════════════════════════════════════════════════════════ */
const primaryButtonStyle: CSSProperties = {
  display: 'inline-flex',
  alignItems: 'center',
  gap: 'var(--s-2)',
  padding: '8px 14px',
  background: 'var(--accent)',
  color: 'var(--text-inverse)',
  fontSize: 'var(--fs-14)',
  fontWeight: 600,
  borderRadius: 'var(--r-md)',
  boxShadow: 'var(--shadow-xs)',
  transition: 'background var(--dur-fast) var(--ease-standard)',
};

const smallPrimaryStyle: CSSProperties = {
  display: 'inline-flex',
  alignItems: 'center',
  padding: '6px 14px',
  background: 'var(--accent)',
  color: 'var(--text-inverse)',
  fontSize: 'var(--fs-13)',
  fontWeight: 600,
  borderRadius: 'var(--r-sm)',
};

const ghostButtonStyle: CSSProperties = {
  display: 'inline-flex',
  alignItems: 'center',
  padding: '6px 12px',
  color: 'var(--text-secondary)',
  fontSize: 'var(--fs-13)',
  fontWeight: 500,
  borderRadius: 'var(--r-sm)',
  border: '1px solid var(--border-strong)',
};

const secondaryLinkStyle: CSSProperties = {
  display: 'inline-flex',
  alignItems: 'center',
  gap: 'var(--s-2)',
  padding: '8px 12px',
  color: 'var(--text-link)',
  fontSize: 'var(--fs-14)',
  fontWeight: 500,
  borderRadius: 'var(--r-md)',
  textDecoration: 'none',
};

const inlineLinkStyle: CSSProperties = {
  display: 'inline-flex',
  alignItems: 'center',
  gap: 4,
  color: 'var(--text-link)',
  fontWeight: 500,
  textDecoration: 'none',
};
