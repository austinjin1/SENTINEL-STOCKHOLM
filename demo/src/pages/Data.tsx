import { useMemo, useState, type ReactNode } from 'react';
import {
  Database,
  Waves,
  Satellite,
  Microscope,
  Dna,
  Fish,
  Bird,
  Network,
  FlaskConical,
  HardDrive,
  Filter,
  ExternalLink,
} from 'lucide-react';
import {
  SectionHeader,
  DataTable,
  Callout,
  StatusBadge,
  Tooltip,
  CopyButton,
  Metric,
  type Column,
} from '../ui';
import {
  DATASETS,
  DATA_SUMMARY,
  type DatasetEntry,
  type DataModality,
} from '../content/datasets';

/**
 * Data — SENTINEL-DB, the data infrastructure page. A downloads/catalog view in
 * the Ensembl / UCSC idiom: a dense table of the real public sources with type,
 * raw record count, feeding modality, and public access route; the honest
 * ~390M raw → ~52M post-QC framing; and a compact coverage summary.
 *
 * Every figure is real and public (source of truth: src/content/datasets.ts,
 * cross-checked against README.md "Data Infrastructure: SENTINEL-DB"). No
 * fabricated or retracted number is reintroduced.
 */

// ---------------------------------------------------------------------------
// Modality presentation — one muted categorical color + a line icon each,
// pulled from the design tokens (--viz-*). Encodes which sensing channel a
// source feeds, never decoration.
// ---------------------------------------------------------------------------
const MODALITY: Record<
  DataModality,
  { label: string; color: string; Icon: typeof Waves }
> = {
  sensor: { label: 'Sensor', color: 'var(--viz-sensor)', Icon: Waves },
  satellite: { label: 'Satellite', color: 'var(--viz-satellite)', Icon: Satellite },
  microbial: { label: 'Microbial', color: 'var(--viz-microbial)', Icon: Microscope },
  molecular: { label: 'Molecular', color: 'var(--viz-molecular)', Icon: Dna },
  behavioral: { label: 'Behavioral', color: 'var(--viz-behavioral)', Icon: FlaskConical },
  biological: { label: 'Biological', color: 'var(--viz-behavioral)', Icon: Fish },
  topology: { label: 'Topology', color: 'var(--text-tertiary)', Icon: Network },
  discrete: { label: 'Discrete', color: 'var(--text-secondary)', Icon: Bird },
};

// Canonical order for the filter row (mirrors table sort roughly).
const MODALITY_ORDER: DataModality[] = [
  'sensor',
  'satellite',
  'discrete',
  'microbial',
  'molecular',
  'behavioral',
  'biological',
  'topology',
];

// Access host → https URL. Some access strings carry a parenthetical note;
// keep the host, drop the note, prefix https.
function accessUrl(access: string): string {
  const host = access.split(' ')[0].replace(/^https?:\/\//, '');
  return `https://${host}`;
}

// Small inline chip that colors a modality with its dot + label.
function ModalityChip({
  modality,
  size = 'md',
}: {
  modality: DataModality;
  size?: 'sm' | 'md';
}) {
  const m = MODALITY[modality];
  const Icon = m.Icon;
  return (
    <span
      style={{
        display: 'inline-flex',
        alignItems: 'center',
        gap: 'var(--s-2)',
        color: 'var(--text-secondary)',
        fontFamily: 'var(--font-sans)',
        fontSize: size === 'sm' ? 'var(--fs-11)' : 'var(--fs-12)',
        whiteSpace: 'nowrap',
      }}
    >
      <Icon
        size={size === 'sm' ? 12 : 13}
        strokeWidth={1.9}
        style={{ color: m.color, flexShrink: 0 }}
        aria-hidden
      />
      {m.label}
    </span>
  );
}

export default function Data() {
  // ---- modality filter (functional, keyboard-navigable) -------------------
  const [active, setActive] = useState<DataModality | null>(null);

  const counts = useMemo(() => {
    const c: Partial<Record<DataModality, number>> = {};
    for (const d of DATASETS) c[d.modality] = (c[d.modality] ?? 0) + 1;
    return c;
  }, []);

  const rows = useMemo(
    () => (active ? DATASETS.filter((d) => d.modality === active) : DATASETS),
    [active],
  );

  const columns: Column<DatasetEntry>[] = [
    {
      key: 'name',
      header: 'Source',
      align: 'left',
      cell: (r) => (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
          <span
            className="mono"
            style={{
              fontSize: 'var(--fs-13)',
              fontWeight: 'var(--fw-medium)' as unknown as number,
              color: 'var(--text-primary)',
            }}
          >
            {r.name}
          </span>
          <span style={{ fontSize: 'var(--fs-11)', color: 'var(--text-tertiary)' }}>
            {r.provider}
          </span>
        </div>
      ),
    },
    {
      key: 'modality',
      header: 'Modality',
      align: 'left',
      width: '128px',
      cell: (r) => <ModalityChip modality={r.modality} />,
    },
    {
      key: 'type',
      header: 'Record type',
      align: 'left',
      cell: (r) => (
        <span style={{ color: 'var(--text-secondary)', fontSize: 'var(--fs-12)' }}>
          {r.type}
        </span>
      ),
    },
    {
      key: 'records',
      header: 'Raw records',
      mono: true,
      width: '150px',
      title: 'Raw, pre-QC record / unit count as published by the source.',
      cell: (r) => (
        <span style={{ display: 'inline-flex', alignItems: 'center', gap: 'var(--s-1)' }}>
          <span>{r.records}</span>
          <CopyButton value={`${r.name}: ${r.records}`} label={`${r.name} record count`} size={12} />
        </span>
      ),
    },
    {
      key: 'access',
      header: 'Access',
      align: 'left',
      width: '210px',
      title: 'Public access route. All sources are open / freely available.',
      cell: (r) => (
        <a
          className="mono"
          href={accessUrl(r.access)}
          target="_blank"
          rel="noreferrer noopener"
          style={{
            display: 'inline-flex',
            alignItems: 'center',
            gap: 'var(--s-2)',
            fontSize: 'var(--fs-12)',
            color: 'var(--text-link)',
          }}
        >
          <span
            style={{
              overflow: 'hidden',
              textOverflow: 'ellipsis',
              whiteSpace: 'nowrap',
              maxWidth: 160,
            }}
          >
            {r.access}
          </span>
          <ExternalLink size={12} strokeWidth={1.75} style={{ flexShrink: 0 }} aria-hidden />
        </a>
      ),
    },
  ];

  return (
    <div style={{ animation: 'fade-in var(--dur-base) var(--ease-standard)' }}>
      <SectionHeader
        as="h1"
        eyebrow="Data infrastructure · SENTINEL-DB"
        title="Real public data sources"
        description="SENTINEL-DB harmonizes environmental records from 14 open data sources.
        Every source below is real and freely accessible. Headline volume is stated as a raw,
        pre-QC count; the usable post-QC volume is far smaller, and we say so."
        actions={
          <a href="/docs" style={{ display: 'inline-flex', alignItems: 'center', gap: 'var(--s-2)' }}>
            Holdout &amp; QC protocol
          </a>
        }
      />

      {/* ---- headline aggregate figures (honest raw vs post-QC) ------------- */}
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
          label="Raw records"
          value={DATA_SUMMARY.rawRecords}
          sub="~85 GB · pre-QC"
          copyValue={`${DATA_SUMMARY.rawRecords} raw environmental records (pre-QC)`}
          tooltip={DATA_SUMMARY.rawRecordsNote}
        />
        <Metric
          size="lg"
          countUp
          label="Post-QC records"
          value={DATA_SUMMARY.postQcRecords}
          status="watch"
          sub="after NEON quality-flag filtering"
          copyValue={`${DATA_SUMMARY.postQcRecords} post-QC records`}
          tooltip={DATA_SUMMARY.postQcNote}
        />
        <Metric
          size="lg"
          countUp
          label="Sources"
          value={DATA_SUMMARY.nSources}
          sub="all open / public"
          tooltip="Count of distinct public data programs harmonized into SENTINEL-DB.
          Each is listed in the catalog below with its public access route."
        />
        <Metric
          size="lg"
          countUp
          label="Countries"
          value={DATA_SUMMARY.nCountries}
          sub={`${DATA_SUMMARY.nSites} monitoring sites`}
          copyValue={`${DATA_SUMMARY.nCountries} countries, ${DATA_SUMMARY.nSites} sites`}
          tooltip="Geographic coverage across all sources. Site coverage is dominated by
          discrete-sample programs (WQP, GRQA); high-frequency sensor coverage is far sparser."
        />
      </div>

      {/* ---- QC funnel: the honesty framing, made visual -------------------- */}
      <div style={{ marginBottom: 'var(--s-6)' }}>
        <QcFunnel />
      </div>

      <Callout tone="limitation" title="Raw counts overstate usable volume">
        The <span className="mono">{DATA_SUMMARY.rawRecords}</span> headline is a raw, pre-QC
        count dominated by NEON high-frequency sonde data (
        <span className="mono">351.7M</span> raw). NEON raw dissolved-oxygen contained extreme
        outliers (max <span className="mono">4.95 billion</span>) removed by{' '}
        <span className="mono">FinalQF=0</span> filtering, leaving roughly{' '}
        <span className="mono">{DATA_SUMMARY.postQcRecords}</span> usable records. Report and cite
        the post-QC figure for anything downstream.
      </Callout>

      {/* ---- the source catalog ------------------------------------------- */}
      <div style={{ marginTop: 'var(--s-9)' }}>
        <SectionHeader
          as="h2"
          eyebrow="Catalog"
          title="Source table"
          description="The full SENTINEL-DB source inventory. Filter by the sensing modality a
          source feeds. Every access route is public; record counts are raw (pre-QC)."
          actions={
            <span style={{ display: 'inline-flex', alignItems: 'center', gap: 'var(--s-2)' }}>
              <Database size={14} strokeWidth={1.75} style={{ color: 'var(--text-tertiary)' }} aria-hidden />
              <span className="mono" style={{ fontSize: 'var(--fs-13)', color: 'var(--text-tertiary)' }}>
                {rows.length} / {DATASETS.length}
              </span>
            </span>
          }
        />

        {/* modality filter row */}
        <div
          role="group"
          aria-label="Filter sources by modality"
          style={{
            display: 'flex',
            flexWrap: 'wrap',
            alignItems: 'center',
            gap: 'var(--s-2)',
            marginBottom: 'var(--s-5)',
          }}
        >
          <span
            style={{
              display: 'inline-flex',
              alignItems: 'center',
              gap: 'var(--s-2)',
              marginRight: 'var(--s-1)',
              color: 'var(--text-tertiary)',
              fontSize: 'var(--fs-12)',
            }}
          >
            <Filter size={13} strokeWidth={1.75} aria-hidden />
            Filter
          </span>
          <FilterChip
            label="All sources"
            count={DATASETS.length}
            selected={active === null}
            onClick={() => setActive(null)}
          />
          {MODALITY_ORDER.filter((m) => counts[m]).map((m) => (
            <FilterChip
              key={m}
              label={MODALITY[m].label}
              count={counts[m] ?? 0}
              color={MODALITY[m].color}
              Icon={MODALITY[m].Icon}
              selected={active === m}
              onClick={() => setActive(active === m ? null : m)}
            />
          ))}
        </div>

        <DataTable
          rows={rows}
          columns={columns}
          getRowKey={(r) => r.id}
          emptyMessage="No sources match this modality."
          caption={
            <span>
              SENTINEL-DB source catalog · counts are raw pre-QC · cross-checked against{' '}
              <span className="mono">README.md → Data Infrastructure</span>
            </span>
          }
        />
      </div>

      {/* ---- coverage + how it feeds the models --------------------------- */}
      <div style={{ marginTop: 'var(--s-9)' }}>
        <SectionHeader
          as="h2"
          eyebrow="Coverage"
          title="What each modality supplies"
          description="How the sources map onto SENTINEL's five sensing channels, and where
          coverage is dense versus sparse. Sensor and satellite channels drive the fusion story;
          the rest supply held-out encoder benchmarks."
        />
        <div
          style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fill, minmax(280px, 1fr))',
            gap: 'var(--s-5)',
          }}
        >
          {COVERAGE.map((c) => (
            <CoverageCard key={c.modality} entry={c} />
          ))}
        </div>
      </div>

      {/* ---- provenance / simulation ledger ------------------------------- */}
      <div style={{ marginTop: 'var(--s-9)' }}>
        <SectionHeader
          as="h2"
          eyebrow="Provenance"
          title="Real data vs simulation"
          description="Every input is labeled. The encoders and the fusion benchmarks run on the
          real sources above under stated holdouts. Three components are simulation-only and are
          marked as such wherever they appear."
        />
        <div style={{ display: 'grid', gap: 'var(--s-4)' }}>
          <Callout tone="verified" title="Measured on real, public data">
            All five encoders (AquaSSM, HydroViT, MicroBiomeNet, ToxiGene, BioMotion) and the
            Perceiver&nbsp;IO fusion benchmarks are trained and evaluated on the real sources in
            the catalog above, under station-grouped or geographic holdouts. Reported numbers are
            within-protocol and independently re-derived from{' '}
            <span className="mono">results/*.json</span>.
          </Callout>
          <Callout tone="limitation" title="Simulation-only components (labeled everywhere)">
            <ul style={{ margin: 'var(--s-2) 0 0', paddingLeft: 'var(--s-6)', display: 'grid', gap: 'var(--s-2)' }}>
              <li>
                <span style={{ color: 'var(--text-primary)', fontWeight: 500 }}>Stream-Network GNN</span>{' '}
                — trained on <em>synthetic</em> propagation labels over real NHDPlus topology
                (<span className="mono">561</span> sites, <span className="mono">338</span> edges);
                not validated on real events.
              </li>
              <li>
                <span style={{ color: 'var(--text-primary)', fontWeight: 500 }}>Conformal pipeline</span>{' '}
                — real sensor coverage <span className="mono">0.9367</span>; the 95% guarantee is{' '}
                <strong>not</strong> met.
              </li>
              <li>
                <span style={{ color: 'var(--text-primary)', fontWeight: 500 }}>Causal screening</span>{' '}
                — lagged partial-correlation only (<span className="mono">0/375</span> chains
                validated; <em>not</em> PCMCI+).
              </li>
              <li>
                <span style={{ color: 'var(--text-primary)', fontWeight: 500 }}>Digital Twin</span>{' '}
                — loses to persistence beyond a <span className="mono">1</span>-day horizon.
              </li>
            </ul>
          </Callout>
        </div>
      </div>
    </div>
  );
}

// ===========================================================================
// QC funnel — a flat, three-stage bar showing raw → filtered → post-QC. No
// gradient, no glow: solid fills in the accent + status washes, hairline
// borders, mono figures. Widths encode the drop honestly (log-ish, clamped so
// the smallest stage stays visible).
// ===========================================================================
function QcFunnel() {
  const stages: {
    label: string;
    value: string;
    note: string;
    pct: number; // visual width share
    fill: string;
    fg: string;
    border: string;
  }[] = [
    {
      label: 'Raw ingest',
      value: '~390M',
      note: 'pre-QC, ~85 GB',
      pct: 100,
      fill: 'var(--accent-wash)',
      fg: 'var(--accent-strong)',
      border: 'var(--accent-border)',
    },
    {
      label: 'NEON FinalQF=0',
      value: '~52M',
      note: 'quality-flag filtered',
      pct: 42,
      fill: 'var(--watch-wash)',
      fg: 'var(--watch)',
      border: 'var(--watch-border)',
    },
    {
      label: 'Analysis-ready',
      value: '~52M',
      note: 'harmonized, deduped',
      pct: 30,
      fill: 'var(--nominal-wash)',
      fg: 'var(--nominal)',
      border: 'var(--nominal-border)',
    },
  ];

  return (
    <div
      style={{
        border: '1px solid var(--border-subtle)',
        borderRadius: 'var(--r-lg)',
        background: 'var(--bg-surface)',
        padding: 'var(--s-6)',
      }}
    >
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: 'var(--s-2)',
          marginBottom: 'var(--s-4)',
        }}
      >
        <HardDrive size={14} strokeWidth={1.75} style={{ color: 'var(--text-tertiary)' }} aria-hidden />
        <span className="eyebrow">Quality-control funnel</span>
        <Tooltip label="NEON raw dissolved-oxygen contained extreme outliers (max 4.95 billion) removed by FinalQF=0 filtering. The ~390M → ~52M drop is dominated by this single step.">
          <span
            className="mono"
            style={{
              fontSize: 'var(--fs-11)',
              color: 'var(--text-tertiary)',
              borderBottom: '1px dotted var(--border-strong)',
              cursor: 'help',
            }}
          >
            what dropped?
          </span>
        </Tooltip>
      </div>

      <div style={{ display: 'grid', gap: 'var(--s-3)' }}>
        {stages.map((s) => (
          <div key={s.label} style={{ display: 'flex', alignItems: 'center', gap: 'var(--s-4)' }}>
            <div
              style={{
                width: 132,
                flexShrink: 0,
                fontSize: 'var(--fs-12)',
                color: 'var(--text-secondary)',
              }}
            >
              {s.label}
            </div>
            <div style={{ flex: 1, minWidth: 0 }}>
              <div
                style={{
                  height: 30,
                  width: `${s.pct}%`,
                  minWidth: 96,
                  background: s.fill,
                  border: `1px solid ${s.border}`,
                  borderRadius: 'var(--r-sm)',
                  display: 'flex',
                  alignItems: 'center',
                  gap: 'var(--s-3)',
                  padding: '0 var(--s-3)',
                  transition: 'width var(--dur-slow) var(--ease-out)',
                }}
              >
                <span
                  className="mono"
                  style={{
                    fontSize: 'var(--fs-13)',
                    fontWeight: 'var(--fw-semibold)' as unknown as number,
                    color: s.fg,
                  }}
                >
                  {s.value}
                </span>
                <span style={{ fontSize: 'var(--fs-11)', color: 'var(--text-tertiary)' }}>
                  {s.note}
                </span>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

// ===========================================================================
// Filter chip — a flat, selectable pill. Selected = accent wash + border;
// otherwise neutral surface. Keyboard-focusable button, aria-pressed set.
// ===========================================================================
function FilterChip({
  label,
  count,
  color,
  Icon,
  selected,
  onClick,
}: {
  label: string;
  count: number;
  color?: string;
  Icon?: typeof Waves;
  selected: boolean;
  onClick: () => void;
}) {
  return (
    <button
      type="button"
      aria-pressed={selected}
      onClick={onClick}
      style={{
        display: 'inline-flex',
        alignItems: 'center',
        gap: 'var(--s-2)',
        padding: '3px 10px',
        borderRadius: 'var(--r-pill)',
        border: `1px solid ${selected ? 'var(--accent-border)' : 'var(--border-subtle)'}`,
        background: selected ? 'var(--accent-wash)' : 'var(--bg-surface)',
        color: selected ? 'var(--accent-strong)' : 'var(--text-secondary)',
        fontSize: 'var(--fs-12)',
        fontWeight: selected ? 600 : 500,
        lineHeight: 1.6,
        transition:
          'background var(--dur-fast) var(--ease-standard), border-color var(--dur-fast), color var(--dur-fast)',
      }}
      onMouseEnter={(e) => {
        if (!selected) e.currentTarget.style.background = 'var(--bg-hover)';
      }}
      onMouseLeave={(e) => {
        if (!selected) e.currentTarget.style.background = 'var(--bg-surface)';
      }}
    >
      {Icon && (
        <Icon
          size={12}
          strokeWidth={1.9}
          style={{ color: color ?? 'currentColor', flexShrink: 0 }}
          aria-hidden
        />
      )}
      {label}
      <span
        className="mono"
        style={{
          fontSize: 'var(--fs-11)',
          color: selected ? 'var(--accent-strong)' : 'var(--text-tertiary)',
        }}
      >
        {count}
      </span>
    </button>
  );
}

// ===========================================================================
// Coverage cards — one per SENTINEL sensing channel, describing what supplies
// it, coverage density, and the sources it draws on.
// ===========================================================================
interface CoverageEntry {
  modality: DataModality;
  title: string;
  density: 'dense' | 'moderate' | 'sparse';
  blurb: ReactNode;
  sources: string[];
}

const DENSITY_STATUS: Record<CoverageEntry['density'], { kind: 'nominal' | 'watch' | 'neutral'; label: string }> = {
  dense: { kind: 'nominal', label: 'dense' },
  moderate: { kind: 'watch', label: 'moderate' },
  sparse: { kind: 'neutral', label: 'sparse' },
};

const COVERAGE: CoverageEntry[] = [
  {
    modality: 'sensor',
    title: 'In-situ sensor streams',
    density: 'dense',
    blurb: (
      <>
        High-frequency sonde time series drive the AquaSSM lead-time detector and the in-situ
        arm of fusion. Temporally dense but spatially limited to gauged sites.
      </>
    ),
    sources: ['NEON Aquatic', 'USGS NWIS'],
  },
  {
    modality: 'satellite',
    title: 'Satellite optics',
    density: 'moderate',
    blurb: (
      <>
        Multispectral Sentinel-2 tiles (incl. red-edge B05–B07) and NOAA chl-a products feed
        HydroViT and the optical arm of fusion. Broad spatial reach, revisit-limited in time.
      </>
    ),
    sources: ['Sentinel-2', 'NOAA HABs'],
  },
  {
    modality: 'discrete',
    title: 'Discrete water-quality samples',
    density: 'dense',
    blurb: (
      <>
        Lab-analyzed grab samples (nutrients, microcystin, physicochemistry) supply fusion
        drivers and labels. Widest site coverage of any channel; irregular in time.
      </>
    ),
    sources: ['EPA WQP', 'GRQA v1.3', 'WQP HABs', 'Canada WQP'],
  },
  {
    modality: 'microbial',
    title: 'Microbial community',
    density: 'sparse',
    blurb: (
      <>
        16S rRNA OTU tables underpin MicroBiomeNet source attribution. Transferable family/genus
        taxonomy generalizes cross-region where raw OTUs collapse.
      </>
    ),
    sources: ['EMP 16S rRNA'],
  },
  {
    modality: 'molecular',
    title: 'Molecular / transcriptomic',
    density: 'sparse',
    blurb: (
      <>
        Real zebrafish GEO series (12) label ToxiGene toxicant classification. Rank-normalization
        recovers modest cross-study transfer (LOSO AUROC 0.64).
      </>
    ),
    sources: ['NCBI GEO'],
  },
  {
    modality: 'behavioral',
    title: 'Behavioral / ecotoxicology',
    density: 'moderate',
    blurb: (
      <>
        ECOTOX dose-response endpoints (Daphnia assays) train BioMotion effect classification.
        Signal is largely study-design driven, and survives dropping endpoint-category features.
      </>
    ),
    sources: ['EPA ECOTOX', 'USGS BioData', 'GBIF Freshwater'],
  },
  {
    modality: 'topology',
    title: 'Stream-network topology',
    density: 'sparse',
    blurb: (
      <>
        NHDPlus reach topology defines propagation edges for the Stream-Network GNN.{' '}
        <span style={{ color: 'var(--watch)' }}>Simulation-only</span> — trained on synthetic
        labels, not real events.
      </>
    ),
    sources: ['NHDPlus V2'],
  },
];

function CoverageCard({ entry }: { entry: CoverageEntry }) {
  const m = MODALITY[entry.modality];
  const Icon = m.Icon;
  const d = DENSITY_STATUS[entry.density];
  return (
    <div
      style={{
        border: '1px solid var(--border-subtle)',
        borderRadius: 'var(--r-lg)',
        background: 'var(--bg-surface)',
        padding: 'var(--s-5)',
        display: 'flex',
        flexDirection: 'column',
        gap: 'var(--s-3)',
        boxShadow: 'var(--shadow-xs)',
      }}
    >
      <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', gap: 'var(--s-3)' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--s-3)', minWidth: 0 }}>
          <span
            aria-hidden
            style={{
              display: 'inline-flex',
              alignItems: 'center',
              justifyContent: 'center',
              width: 30,
              height: 30,
              borderRadius: 'var(--r-sm)',
              border: '1px solid var(--border-subtle)',
              background: 'var(--bg-inset)',
              flexShrink: 0,
            }}
          >
            <Icon size={16} strokeWidth={1.75} style={{ color: m.color }} />
          </span>
          <span
            style={{
              fontSize: 'var(--fs-14)',
              fontWeight: 'var(--fw-semibold)' as unknown as number,
              color: 'var(--text-primary)',
              lineHeight: 'var(--lh-snug)',
            }}
          >
            {entry.title}
          </span>
        </div>
        <Tooltip
          label={
            entry.density === 'dense'
              ? 'Dense coverage: many observations across sites and/or time.'
              : entry.density === 'moderate'
                ? 'Moderate coverage: usable but revisit- or site-limited.'
                : 'Sparse coverage: few sources; treat cross-region transfer with care.'
          }
        >
          <StatusBadge kind={d.kind}>{d.label}</StatusBadge>
        </Tooltip>
      </div>

      <p
        style={{
          margin: 0,
          fontSize: 'var(--fs-13)',
          lineHeight: 'var(--lh-normal)',
          color: 'var(--text-secondary)',
        }}
      >
        {entry.blurb}
      </p>

      <div
        style={{
          display: 'flex',
          flexWrap: 'wrap',
          gap: 'var(--s-2)',
          marginTop: 'auto',
          paddingTop: 'var(--s-2)',
          borderTop: '1px solid var(--border-subtle)',
        }}
      >
        {entry.sources.map((s) => (
          <span
            key={s}
            className="mono"
            style={{
              fontSize: 'var(--fs-11)',
              color: 'var(--text-tertiary)',
              padding: '1px 7px',
              borderRadius: 'var(--r-sm)',
              background: 'var(--bg-inset)',
              border: '1px solid var(--border-subtle)',
            }}
          >
            {s}
          </span>
        ))}
      </div>
    </div>
  );
}
