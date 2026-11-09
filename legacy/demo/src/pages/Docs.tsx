import {
  useEffect,
  useMemo,
  useState,
  type ReactNode,
} from 'react';
import {
  BookOpen,
  GitBranch,
  FlaskConical,
  ListChecks,
  HelpCircle,
  Terminal,
  Search,
  ArrowUpRight,
} from 'lucide-react';
import { SectionHeader, Callout, DataTable, StatusBadge, CopyButton } from '../ui';
import type { Column } from '../ui';
import {
  PROTOCOLS,
  RETRACTIONS,
  SIMULATION_SCOPE,
  HONEST_NEGATIVES,
  CENTRAL_FINDING,
  LEADTIME_GENERALIZATION,
  TRANSFER_GAP,
  TRANSFER_GAP_SUMMARY,
  HONESTY_STATEMENT,
} from '../content/methodology';
import { GLOSSARY, type GlossaryEntry, type GlossaryCategory } from '../content/glossary';

/* ==========================================================================
   Docs — the SENTINEL documentation column (Ensembl / gnomAD help idiom).

   A sticky left in-page nav + a readable prose column. Sections:
     · How it works — the pipeline, modality by modality
     · How to reproduce — the real, copy-able commands
     · Methodology — holdout protocols + the integrity story as a strength
     · Glossary — searchable, from content/glossary
     · FAQ
   Flat, data-first, monospace for every command / metric / ID. Motion is
   subtle (scroll-spy underline, hover micro-states) and reduced-motion safe.
   ========================================================================== */

interface DocSection {
  id: string;
  label: string;
  Icon: typeof BookOpen;
}

const SECTIONS: DocSection[] = [
  { id: 'how-it-works', label: 'How it works', Icon: GitBranch },
  { id: 'reproduce', label: 'How to reproduce', Icon: Terminal },
  { id: 'methodology', label: 'Methodology', Icon: FlaskConical },
  { id: 'integrity', label: 'Integrity & retractions', Icon: ListChecks },
  { id: 'glossary', label: 'Glossary', Icon: BookOpen },
  { id: 'faq', label: 'FAQ', Icon: HelpCircle },
];

export default function Docs() {
  const active = useScrollSpy(SECTIONS.map((s) => s.id));

  return (
    <>
      <SectionHeader
        as="h1"
        eyebrow="Documentation"
        title="How SENTINEL works, and how to trust the numbers"
        description="The pipeline, the exact reproduction commands, the holdout
        protocols, a searchable glossary, and the full record of what was
        retracted during integrity remediation."
      />

      <div className="docs-grid">
        {/* Sticky in-page nav */}
        <nav className="docs-nav" aria-label="On this page">
          <div
            className="eyebrow"
            style={{ padding: '0 var(--s-3) var(--s-2)' }}
          >
            On this page
          </div>
          <ul style={{ listStyle: 'none', margin: 0, padding: 0 }}>
            {SECTIONS.map((s) => {
              const isActive = active === s.id;
              return (
                <li key={s.id}>
                  <a
                    href={`#${s.id}`}
                    aria-current={isActive ? 'true' : undefined}
                    style={{
                      display: 'flex',
                      alignItems: 'center',
                      gap: 'var(--s-3)',
                      padding: '7px var(--s-3)',
                      borderRadius: 'var(--r-sm)',
                      textDecoration: 'none',
                      fontSize: 'var(--fs-13)',
                      fontWeight: isActive
                        ? ('var(--fw-semibold)' as unknown as number)
                        : ('var(--fw-regular)' as unknown as number),
                      color: isActive ? 'var(--text-primary)' : 'var(--text-secondary)',
                      background: isActive ? 'var(--accent-wash)' : 'transparent',
                      boxShadow: isActive
                        ? 'inset 2px 0 0 0 var(--accent)'
                        : 'inset 2px 0 0 0 transparent',
                      transition:
                        'background var(--dur-fast) var(--ease-standard), color var(--dur-fast)',
                    }}
                    onMouseEnter={(e) => {
                      if (!isActive) e.currentTarget.style.background = 'var(--bg-hover)';
                    }}
                    onMouseLeave={(e) => {
                      if (!isActive) e.currentTarget.style.background = 'transparent';
                    }}
                  >
                    <s.Icon
                      size={14}
                      strokeWidth={1.9}
                      style={{
                        color: isActive ? 'var(--accent)' : 'var(--text-tertiary)',
                        flexShrink: 0,
                      }}
                      aria-hidden
                    />
                    {s.label}
                  </a>
                </li>
              );
            })}
          </ul>
        </nav>

        {/* Prose column */}
        <div className="docs-prose">
          <HowItWorks />
          <Reproduce />
          <Methodology />
          <Integrity />
          <Glossary />
          <Faq />
        </div>
      </div>

      <style>{DOCS_CSS}</style>
    </>
  );
}

/* ------------------------------------------------------------------------ *
 * Section: How it works                                                     *
 * ------------------------------------------------------------------------ */

const PIPELINE: { step: string; title: string; body: string }[] = [
  {
    step: '01',
    title: 'Ingest real public data',
    body: 'Pull from 14 public sources (USGS NWIS, NEON, ESA Sentinel-2 via the Planetary Computer, EMP 16S, NCBI GEO, EPA ECOTOX, WQP/NOAA HABs, NHDPlus). ~390M raw records pre-QC, dominated by NEON high-frequency sonde data.',
  },
  {
    step: '02',
    title: 'Quality-control & harmonize',
    body: 'Apply source quality flags (e.g. NEON FinalQF=0 removes extreme sonde outliers), align to sites/coordinates/time, and co-locate modalities where they overlap (e.g. Sentinel-2 tiles matched to in-situ WQP samples).',
  },
  {
    step: '03',
    title: 'Encode each modality',
    body: 'Five modality encoders learn representations: AquaSSM (sensor streams), HydroViT (satellite imagery), MicroBiomeNet (16S communities), ToxiGene (transcriptomics), BioMotion (ecotoxicology). Where possible, choose transferable representations over raw features.',
  },
  {
    step: '04',
    title: 'Fuse & forecast',
    body: 'Cross-attention fusion (Perceiver IO) combines modalities for optically-invisible targets like microcystin; downstream heads forecast events. Everything is scored under a stated holdout protocol with site-clustered confidence intervals.',
  },
  {
    step: '05',
    title: 'Report honestly',
    body: 'Publish held-out numbers with their protocol and CI. Where a model does not beat its baseline (persistence, best single modality, cross-region transfer), report the negative result rather than hide it.',
  },
];

const MODALITY_ROWS: {
  encoder: string;
  modality: string;
  input: string;
  representation: string;
}[] = [
  {
    encoder: 'AquaSSM',
    modality: 'Sensor',
    input: 'USGS/NEON continuous sonde streams',
    representation: 'Continuous-time state-space (Mamba) embeddings',
  },
  {
    encoder: 'HydroViT',
    modality: 'Satellite',
    input: 'Sentinel-2 L2A multispectral tiles',
    representation: 'MAE-pretrained ViT + red-edge spectral indices',
  },
  {
    encoder: 'MicroBiomeNet',
    modality: 'Microbial',
    input: '16S rRNA community tables',
    representation: 'CLR-geometry transformer over family/genus taxonomy',
  },
  {
    encoder: 'ToxiGene',
    modality: 'Molecular',
    input: 'Zebrafish RNA-seq / microarray',
    representation: 'Pathway-hierarchy features, rank-normalized per sample',
  },
  {
    encoder: 'BioMotion',
    modality: 'Behavioral',
    input: 'EPA ECOTOX dose-response assays',
    representation: 'Endpoint-structure gradient boosting',
  },
];

function HowItWorks() {
  const cols: Column<(typeof MODALITY_ROWS)[number]>[] = [
    { key: 'encoder', header: 'Encoder', mono: true, align: 'left', width: '140px' },
    { key: 'modality', header: 'Modality', align: 'left' },
    { key: 'input', header: 'Real input', align: 'left' },
    { key: 'representation', header: 'Learned representation', align: 'left' },
  ];

  return (
    <Section id="how-it-works" Icon={GitBranch} title="How it works" kicker="Pipeline">
      <p>
        SENTINEL is a multimodal water-quality early-warning system. It turns
        heterogeneous public environmental data into forecasts of water-quality
        failures — dissolved-oxygen hypoxia, cyanotoxin exceedances, contaminant
        exposure — and reports each result under an explicit holdout protocol.
      </p>

      <ol className="pipeline">
        {PIPELINE.map((p) => (
          <li key={p.step}>
            <span className="mono pipeline-step" aria-hidden>
              {p.step}
            </span>
            <div>
              <div className="pipeline-title">{p.title}</div>
              <p className="pipeline-body">{p.body}</p>
            </div>
          </li>
        ))}
      </ol>

      <h3 className="subhead">The five modality encoders</h3>
      <p>
        Each encoder learns a representation of one modality. Wherever a raw
        feature is region- or study-specific, SENTINEL substitutes a{' '}
        <em>transferable representation</em> (higher taxonomy, rank-normalized
        genes, red-edge indices) so the model generalizes rather than memorizes.
      </p>
      <DataTable rows={MODALITY_ROWS} getRowKey={(r) => r.encoder} columns={cols} />

      <Callout tone="info" title="The central methodological finding">
        {CENTRAL_FINDING.summary}
      </Callout>

      <div className="evidence-grid">
        {CENTRAL_FINDING.evidence.map((e) => (
          <div key={e.model} className="evidence-card">
            <div className="mono evidence-model">{e.model}</div>
            <div className="evidence-line evidence-raw">{e.raw}</div>
            <div className="evidence-line evidence-transfer">{e.transferable}</div>
            <p className="evidence-note">{e.note}</p>
          </div>
        ))}
      </div>

      <h3 className="subhead">{LEADTIME_GENERALIZATION.title}</h3>
      <p>{LEADTIME_GENERALIZATION.summary}</p>
      <div className="evidence-grid">
        {LEADTIME_GENERALIZATION.evidence.map((e) => (
          <div key={e.test} className="evidence-card">
            <div className="mono evidence-model">{e.test}</div>
            <p className="evidence-note">{e.finding}</p>
          </div>
        ))}
      </div>
      <Callout tone="verified" title="The operating point transfers to unseen sites">
        {LEADTIME_GENERALIZATION.takeaway}
      </Callout>

      <h3 className="subhead">The transfer gap, quantified across every model</h3>
      <p>{TRANSFER_GAP_SUMMARY.thesis}</p>
      <DataTable
        rows={TRANSFER_GAP}
        getRowKey={(r) => r.model}
        caption="In-distribution scores systematically overstate deployable skill — the honest number is the OOD column. Hover a score for its exact holdout setting."
        columns={[
          {
            key: 'model',
            header: 'Model',
            align: 'left',
            cell: (r) => (
              <span>
                <strong>{r.model}</strong>{' '}
                <span className="evidence-note">· {r.metric}</span>
              </span>
            ),
          },
          {
            key: 'id',
            header: 'In-distribution',
            mono: true,
            title: 'Optimistic in-distribution split',
            cell: (r) => <span title={r.idSetting}>{r.id}</span>,
          },
          {
            key: 'oodRaw',
            header: 'OOD (raw)',
            mono: true,
            title: 'Honest grouped/shifted holdout, raw features',
            cell: (r) => <span title={r.oodSetting}>{r.oodRaw}</span>,
          },
          {
            key: 'oodFixed',
            header: 'OOD (fixed)',
            mono: true,
            title: 'After the transferable-representation fix',
            cell: (r) => r.oodFixed,
          },
          { key: 'fix', header: 'Transferable fix', align: 'left', cell: (r) => r.fix },
          {
            key: 'recovered',
            header: 'Gap recovered',
            align: 'right',
            cell: (r) => (
              <span
                style={{
                  color: r.tone === 'nominal' ? 'var(--accent)' : 'var(--text-tertiary)',
                  fontVariantNumeric: 'tabular-nums',
                  whiteSpace: 'nowrap',
                }}
              >
                {r.recovered}
              </span>
            ),
          },
        ]}
      />
      <Callout
        tone="info"
        title={`Transferable representations recover ${TRANSFER_GAP_SUMMARY.meanRecovered} of the gap — never all of it`}
      >
        {TRANSFER_GAP_SUMMARY.meanRecoveredNote}
      </Callout>
    </Section>
  );
}

/* ------------------------------------------------------------------------ *
 * Section: How to reproduce                                                 *
 * ------------------------------------------------------------------------ */

const REPRO_STEPS: { title: string; note: ReactNode; cmd: string }[] = [
  {
    title: '1 · Clone and set up the environment',
    note: 'Python 3.11+. Installs the pinned dependency set used for every reported number.',
    cmd: 'git clone https://github.com/austinjin1/SENTINEL-STOCKHOLM.git\ncd SENTINEL-STOCKHOLM\npython -m venv .venv && source .venv/bin/activate\npip install -r requirements.txt',
  },
  {
    title: '2 · Fetch the public data',
    note: 'Downloads from the public sources listed on the Data page. Nothing is proprietary; API keys (Planetary Computer) are documented in the README.',
    cmd: 'python -m sentinel.data.fetch --sources all --out data/raw\npython -m sentinel.data.qc --in data/raw --out data/clean',
  },
  {
    title: '3 · Reproduce the encoder benchmarks',
    note: 'Each command writes a results JSON under results/benchmarks/ that matches the numbers on the Models page (held-out, with CIs).',
    cmd: 'python -m sentinel.eval.aquassm      --holdout station\npython -m sentinel.eval.hydrovit     --holdout station --bands rededge\npython -m sentinel.eval.microbiomenet --holdout geographic --features taxonomy\npython -m sentinel.eval.toxigene     --holdout loso --normalize rank\npython -m sentinel.eval.biomotion    --holdout chemical',
  },
  {
    title: '4 · Reproduce the fusion result (and its collapse)',
    note: 'Runs the microcystin 3-way fusion under station-grouped CV, then the cross-region (leave-one-HUC-out) stress test that collapses to chance.',
    cmd: 'python -m sentinel.fusion.microcystin --cv station-grouped --bootstrap site\npython -m sentinel.fusion.microcystin --cv leave-one-huc-out   # ~0.49, chance',
  },
  {
    title: '5 · Regenerate the site data',
    note: 'The demo reads results/*.json. This step re-exports the JSON the UI consumes, so the site always reflects the measured numbers.',
    cmd: 'python -m sentinel.report.export --out demo/src/data/repo',
  },
];

function Reproduce() {
  return (
    <Section id="reproduce" Icon={Terminal} title="How to reproduce" kicker="Commands">
      <p>
        Every headline number is reproducible from public data with the commands
        below. The evaluation scripts write result JSON that the site reads
        directly — there is no manual number entry between the benchmark and this
        page.
      </p>

      <Callout tone="info" title="Source of truth">
        Reported values live in <code className="mono">results/*.json</code> and
        are cross-checked against <code className="mono">README.md</code>. If a
        command produces a different number than the site shows, the command is
        authoritative — please open an issue.
      </Callout>

      <div className="repro-list">
        {REPRO_STEPS.map((s) => (
          <div key={s.title} className="repro-step">
            <div className="repro-head">
              <div className="repro-title">{s.title}</div>
              <p className="repro-note">{s.note}</p>
            </div>
            <CodeBlock code={s.cmd} />
          </div>
        ))}
      </div>
    </Section>
  );
}

/* ------------------------------------------------------------------------ *
 * Section: Methodology (holdout protocols)                                  *
 * ------------------------------------------------------------------------ */

function Methodology() {
  const cols: Column<(typeof PROTOCOLS)[number]>[] = [
    { key: 'name', header: 'Protocol', align: 'left', width: '210px',
      cell: (r) => <span style={{ fontWeight: 600 }}>{r.name}</span> },
    { key: 'description', header: 'What it does', align: 'left' },
    {
      key: 'usedBy',
      header: 'Used by',
      align: 'left',
      width: '220px',
      cell: (r) => (
        <span className="mono" style={{ fontSize: 'var(--fs-12)', color: 'var(--text-secondary)' }}>
          {r.usedBy.join(', ')}
        </span>
      ),
    },
  ];

  return (
    <Section id="methodology" Icon={FlaskConical} title="Methodology" kicker="Holdout protocols">
      <p>
        A number is only as trustworthy as the split it was measured on. SENTINEL
        never reports a random-row split for correlated environmental data. Each
        model is scored under the strictest protocol its data supports —
        excluding whole stations, regions, or studies from training.
      </p>

      <DataTable rows={PROTOCOLS} getRowKey={(r) => r.id} columns={cols} />

      <h3 className="subhead">Confidence intervals</h3>
      <p>
        Intervals come from <strong>site-clustered bootstrap</strong> —
        resampling whole sites with replacement, not individual rows. Row-level
        resampling would give falsely tight intervals because samples from one
        site are correlated. This is the honest choice, and it is why some of our
        intervals are wide.
      </p>

      <h3 className="subhead">What is simulation-only</h3>
      <p>
        Some components use real structure but synthetic labels. They demonstrate
        that an architecture can learn a mechanism; they are <em>not</em> evidence
        of real-world detection skill, and are labeled as such everywhere.
      </p>
      <div className="sim-grid">
        {SIMULATION_SCOPE.map((s) => (
          <div key={s.id} className="sim-card">
            <div className="sim-head">
              <span className="sim-name">{s.name}</span>
              <StatusBadge kind="neutral">simulation</StatusBadge>
            </div>
            <div className="sim-row">
              <span className="sim-label">Real</span>
              <span className="sim-val">{s.whatsReal}</span>
            </div>
            <div className="sim-row">
              <span className="sim-label">Simulated</span>
              <span className="sim-val">{s.whatsSimulated}</span>
            </div>
          </div>
        ))}
      </div>

      <h3 className="subhead">Honest negatives (reported as findings)</h3>
      <p>
        Where a model hits the ceiling of what the data can support, we report the
        negative result. These are findings, not failures to hide.
      </p>
      <ul className="negatives">
        {HONEST_NEGATIVES.map((n) => (
          <li key={n.id}>{n.finding}</li>
        ))}
      </ul>
    </Section>
  );
}

/* ------------------------------------------------------------------------ *
 * Section: Integrity & retractions                                          *
 * ------------------------------------------------------------------------ */

function Integrity() {
  const cols: Column<(typeof RETRACTIONS)[number]>[] = [
    {
      key: 'claim',
      header: 'Retracted claim',
      align: 'left',
      cell: (r) => (
        <div>
          <div style={{ color: 'var(--text-primary)' }}>{r.claim}</div>
          <div style={{ color: 'var(--text-tertiary)', fontSize: 'var(--fs-12)', marginTop: 2 }}>
            {r.reason}
          </div>
        </div>
      ),
    },
    {
      key: 'oldValue',
      header: 'Old',
      mono: true,
      width: '96px',
      cell: (r) => (
        <span
          className="mono"
          style={{ textDecoration: 'line-through', color: 'var(--alert)' }}
        >
          {r.oldValue}
        </span>
      ),
    },
    {
      key: 'correctedTo',
      header: 'Corrected to',
      align: 'left',
      cell: (r) =>
        r.correctedTo ? (
          <span style={{ color: 'var(--nominal)' }}>{r.correctedTo}</span>
        ) : (
          <StatusBadge kind="neutral">removed</StatusBadge>
        ),
    },
  ];

  return (
    <Section
      id="integrity"
      Icon={ListChecks}
      title="Integrity & retractions"
      kicker="The honesty is the result"
    >
      <Callout tone="verified" title="Integrity statement">
        {HONESTY_STATEMENT}
      </Callout>

      <p>
        This project completed an integrity remediation. The table below is the
        complete log: every figure that was retracted, why it was wrong, and the
        honestly-measured value that replaced it (or <em>removed</em> where no
        valid measurement was possible). None of the retracted numbers appear
        anywhere else on this site.
      </p>

      <DataTable
        rows={RETRACTIONS}
        getRowKey={(r) => r.id}
        columns={cols}
        caption={`${RETRACTIONS.length} retractions · struck-through values are gone from the site`}
      />
    </Section>
  );
}

/* ------------------------------------------------------------------------ *
 * Section: Glossary (searchable)                                            *
 * ------------------------------------------------------------------------ */

const CATEGORY_LABEL: Record<GlossaryCategory, string> = {
  metric: 'Metrics',
  protocol: 'Holdout protocols',
  honesty: 'Integrity terms',
  domain: 'Domain terms',
};

const CATEGORY_ORDER: GlossaryCategory[] = ['metric', 'protocol', 'honesty', 'domain'];

function Glossary() {
  const [query, setQuery] = useState('');
  const [cat, setCat] = useState<GlossaryCategory | 'all'>('all');

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase();
    return GLOSSARY.filter((e) => {
      if (cat !== 'all' && e.category !== cat) return false;
      if (!q) return true;
      return (
        e.label.toLowerCase().includes(q) ||
        (e.abbr?.toLowerCase().includes(q) ?? false) ||
        e.definition.toLowerCase().includes(q) ||
        (e.whyItMatters?.toLowerCase().includes(q) ?? false)
      );
    });
  }, [query, cat]);

  const grouped = useMemo(() => {
    const m = new Map<GlossaryCategory, GlossaryEntry[]>();
    for (const e of filtered) {
      const arr = m.get(e.category) ?? [];
      arr.push(e);
      m.set(e.category, arr);
    }
    return m;
  }, [filtered]);

  return (
    <Section id="glossary" Icon={BookOpen} title="Glossary" kicker="Definitions">
      <p>
        Every metric, protocol, and integrity term used on the site, written for
        a non-specialist without dumbing down the science. This is the single
        source of truth for the tooltips elsewhere.
      </p>

      <div className="glossary-controls">
        <div className="glossary-search">
          <Search size={15} strokeWidth={1.9} aria-hidden style={{ color: 'var(--text-tertiary)' }} />
          <input
            type="search"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Search terms, e.g. AUROC, geographic holdout…"
            aria-label="Search the glossary"
          />
        </div>
        <div className="glossary-filters" role="tablist" aria-label="Filter by category">
          <FilterChip active={cat === 'all'} onClick={() => setCat('all')}>
            All
          </FilterChip>
          {CATEGORY_ORDER.map((c) => (
            <FilterChip key={c} active={cat === c} onClick={() => setCat(c)}>
              {CATEGORY_LABEL[c]}
            </FilterChip>
          ))}
        </div>
      </div>

      {filtered.length === 0 ? (
        <div className="glossary-empty">
          No terms match <span className="mono">“{query}”</span>.
        </div>
      ) : (
        CATEGORY_ORDER.filter((c) => grouped.has(c)).map((c) => (
          <div key={c} className="glossary-group">
            <h3 className="glossary-cat">{CATEGORY_LABEL[c]}</h3>
            <dl className="glossary-list">
              {grouped.get(c)!.map((e) => (
                <div key={e.term} id={`term-${e.term}`} className="glossary-item">
                  <dt>
                    <span className="glossary-term">{e.label}</span>
                    {e.abbr && e.abbr !== e.label && (
                      <span className="mono glossary-abbr">{e.abbr}</span>
                    )}
                  </dt>
                  <dd>
                    <p className="glossary-def">{e.definition}</p>
                    {e.whyItMatters && (
                      <p className="glossary-why">
                        <span className="glossary-why-label">Why it matters</span>
                        {e.whyItMatters}
                      </p>
                    )}
                  </dd>
                </div>
              ))}
            </dl>
          </div>
        ))
      )}
    </Section>
  );
}

function FilterChip({
  active,
  onClick,
  children,
}: {
  active: boolean;
  onClick: () => void;
  children: ReactNode;
}) {
  return (
    <button
      type="button"
      role="tab"
      aria-selected={active}
      onClick={onClick}
      style={{
        padding: '4px 10px',
        borderRadius: 'var(--r-pill)',
        fontSize: 'var(--fs-12)',
        fontWeight: 'var(--fw-medium)' as unknown as number,
        color: active ? 'var(--accent-strong)' : 'var(--text-secondary)',
        background: active ? 'var(--accent-wash)' : 'var(--bg-surface)',
        border: `1px solid ${active ? 'var(--accent-border)' : 'var(--border-subtle)'}`,
        transition: 'background var(--dur-fast) var(--ease-standard), color var(--dur-fast)',
      }}
      onMouseEnter={(e) => {
        if (!active) e.currentTarget.style.background = 'var(--bg-hover)';
      }}
      onMouseLeave={(e) => {
        if (!active) e.currentTarget.style.background = 'var(--bg-surface)';
      }}
    >
      {children}
    </button>
  );
}

/* ------------------------------------------------------------------------ *
 * Section: FAQ                                                              *
 * ------------------------------------------------------------------------ */

const FAQ: { q: string; a: ReactNode }[] = [
  {
    q: 'Is any of the data synthetic?',
    a: (
      <>
        The data is real and public in every reported result. A few{' '}
        <em>components</em> use real structure with synthetic labels (the
        Stream-Network GNN, the conformal and causal pipelines, the disease-risk
        index) — these are clearly labeled <StatusBadge kind="neutral">simulation</StatusBadge>{' '}
        and are never counted as detection skill.
      </>
    ),
  },
  {
    q: 'Why does the fusion result collapse across regions?',
    a: (
      <>
        Microcystin production depends on the toxin-producing strain fraction
        (the <span className="mono">mcyD</span> genotype), which varies from
        under 6% to nearly 100% of cells between watersheds. Satellite optics,
        pigment, and drivers cannot see that genotype, so the within-region model
        does not transfer — it falls to chance (<span className="mono">~0.49</span>).
        We report the win (<span className="mono">0.82</span> within region) and
        the collapse together.
      </>
    ),
  },
  {
    q: 'What does "transferable representation" mean?',
    a: (
      <>
        A feature encoding chosen to generalize across regions or studies —
        higher taxonomy instead of exact OTUs, rank-normalized genes instead of
        raw expression, red-edge spectral indices instead of scene-specific
        color. This is SENTINEL's central finding: transferable representations
        recover out-of-distribution skill where raw features collapse.
      </>
    ),
  },
  {
    q: 'Why are some intervals so wide?',
    a: (
      <>
        Because they are honest. Confidence intervals come from resampling whole
        sites, not rows. Row-level bootstrap would look more precise but would be
        wrong — samples from one site are correlated.
      </>
    ),
  },
  {
    q: 'What was retracted, and where did it go?',
    a: (
      <>
        See the{' '}
        <a href="#integrity">Integrity &amp; retractions</a> log above. Every
        fabricated, synthetic-derived, or leakage-inflated figure is listed with
        the reason and the corrected value. None of them appear anywhere else on
        the site.
      </>
    ),
  },
  {
    q: 'Can I cite a number directly?',
    a: (
      <>
        Yes. Metrics across the site carry a copy button; the value copies with
        its unit. The exact protocol behind each number is in its tooltip and in
        the Methodology section above.
      </>
    ),
  },
];

function Faq() {
  return (
    <Section id="faq" Icon={HelpCircle} title="Frequently asked questions" kicker="FAQ">
      <div className="faq-list">
        {FAQ.map((item, i) => (
          <details key={i} className="faq-item" open={i === 0}>
            <summary>
              <span>{item.q}</span>
              <ArrowUpRight size={14} strokeWidth={1.9} aria-hidden className="faq-chevron" />
            </summary>
            <div className="faq-answer">{item.a}</div>
          </details>
        ))}
      </div>
    </Section>
  );
}

/* ------------------------------------------------------------------------ *
 * Shared building blocks                                                    *
 * ------------------------------------------------------------------------ */

function Section({
  id,
  Icon,
  title,
  kicker,
  children,
}: {
  id: string;
  Icon: typeof BookOpen;
  title: string;
  kicker: string;
  children: ReactNode;
}) {
  return (
    <section id={id} className="docs-section" aria-labelledby={`${id}-h`}>
      <div className="docs-section-head">
        <div className="docs-section-icon" aria-hidden>
          <Icon size={16} strokeWidth={1.9} />
        </div>
        <div>
          <div className="eyebrow">{kicker}</div>
          <h2 id={`${id}-h`} className="docs-section-title">
            {title}
          </h2>
        </div>
      </div>
      <div className="docs-section-body">{children}</div>
    </section>
  );
}

function CodeBlock({ code }: { code: string }) {
  return (
    <div className="codeblock">
      <div className="codeblock-copy">
        <CopyButton value={code} label="command" />
      </div>
      <pre>
        <code className="mono">{code}</code>
      </pre>
    </div>
  );
}

/* ------------------------------------------------------------------------ *
 * Scroll-spy: highlight the in-page nav entry for the section in view       *
 * ------------------------------------------------------------------------ */

function useScrollSpy(ids: string[]): string {
  const [active, setActive] = useState(ids[0]);
  // Stable dependency: the observer only needs to be rebuilt if the id set
  // actually changes. Joining to a primitive avoids a ref and satisfies the
  // exhaustive-deps / refs lint rules.
  const key = ids.join('|');

  useEffect(() => {
    const idList = key.split('|');
    const observer = new IntersectionObserver(
      (entries) => {
        // Pick the top-most section currently intersecting the upper viewport.
        const visible = entries
          .filter((e) => e.isIntersecting)
          .sort((a, b) => a.boundingClientRect.top - b.boundingClientRect.top);
        if (visible[0]) {
          setActive(visible[0].target.id);
        }
      },
      {
        // Bias the "active" band to the top third of the viewport.
        rootMargin: '-80px 0px -60% 0px',
        threshold: 0,
      },
    );
    for (const id of idList) {
      const el = document.getElementById(id);
      if (el) observer.observe(el);
    }
    return () => observer.disconnect();
  }, [key]);

  return active;
}

/* ------------------------------------------------------------------------ *
 * Scoped styles — flat, genomics-tool docs. Prose column + sticky nav.      *
 * ------------------------------------------------------------------------ */

const DOCS_CSS = `
.docs-grid {
  display: grid;
  grid-template-columns: 200px minmax(0, 1fr);
  gap: var(--s-9);
  align-items: start;
}
.docs-nav {
  position: sticky;
  top: calc(var(--nav-height) + var(--s-6));
  align-self: start;
}
.docs-prose { min-width: 0; }

.docs-section { scroll-margin-top: calc(var(--nav-height) + var(--s-6)); }
.docs-section + .docs-section { margin-top: var(--s-11); }

.docs-section-head {
  display: flex;
  align-items: center;
  gap: var(--s-4);
  padding-bottom: var(--s-4);
  margin-bottom: var(--s-6);
  border-bottom: 1px solid var(--border-subtle);
}
.docs-section-icon {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 32px;
  height: 32px;
  flex-shrink: 0;
  border-radius: var(--r-md);
  color: var(--accent);
  background: var(--accent-wash);
  border: 1px solid var(--accent-border);
}
.docs-section-title {
  font-size: var(--fs-22);
  font-weight: var(--fw-semibold);
  letter-spacing: var(--tracking-tight);
  color: var(--text-primary);
}

/* Prose defaults */
.docs-section-body { color: var(--text-secondary); }
.docs-section-body p {
  margin: 0 0 var(--s-5);
  max-width: 72ch;
  font-size: var(--fs-14);
  line-height: var(--lh-normal);
}
.docs-section-body p:last-child { margin-bottom: 0; }
.docs-section-body strong { color: var(--text-primary); font-weight: var(--fw-semibold); }
.docs-section-body em { font-style: italic; color: var(--text-primary); }
.docs-section-body a { color: var(--text-link); }
.docs-section-body code {
  font-family: var(--font-mono);
  font-size: 0.92em;
  padding: 1px 5px;
  border-radius: var(--r-xs);
  background: var(--bg-inset);
  border: 1px solid var(--border-subtle);
  color: var(--text-primary);
}
.subhead {
  font-size: var(--fs-16);
  font-weight: var(--fw-semibold);
  color: var(--text-primary);
  margin: var(--s-8) 0 var(--s-3);
}
.docs-section-body > :where(table, .evidence-grid, .sim-grid, .repro-list,
  .glossary-controls, .glossary-group, .faq-list, [role="note"], .pipeline,
  .negatives) {
  margin-bottom: var(--s-6);
}
[role="note"] { margin: var(--s-6) 0; }

/* Pipeline (ordered steps) */
.pipeline { list-style: none; margin: 0 0 var(--s-6); padding: 0; }
.pipeline li {
  display: flex;
  gap: var(--s-5);
  padding: var(--s-4) 0;
  border-bottom: 1px solid var(--border-subtle);
}
.pipeline li:last-child { border-bottom: none; }
.pipeline-step {
  font-size: var(--fs-13);
  color: var(--accent);
  font-weight: var(--fw-semibold);
  flex-shrink: 0;
  width: 24px;
  padding-top: 2px;
}
.pipeline-title {
  font-size: var(--fs-14);
  font-weight: var(--fw-semibold);
  color: var(--text-primary);
  margin-bottom: 2px;
}
.pipeline-body { margin: 0 !important; font-size: var(--fs-13) !important; }

/* Central-finding evidence cards */
.evidence-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
  gap: var(--s-4);
}
.evidence-card {
  padding: var(--s-4);
  background: var(--bg-surface);
  border: 1px solid var(--border-subtle);
  border-radius: var(--r-md);
}
.evidence-model {
  font-size: var(--fs-13);
  font-weight: var(--fw-semibold);
  color: var(--text-primary);
  margin-bottom: var(--s-3);
}
.evidence-line {
  font-size: var(--fs-12);
  padding: 3px 0;
  display: flex;
  align-items: baseline;
  gap: var(--s-2);
}
.evidence-raw { color: var(--text-tertiary); }
.evidence-transfer { color: var(--nominal); font-weight: var(--fw-medium); }
.evidence-note {
  margin: var(--s-3) 0 0 !important;
  font-size: var(--fs-12) !important;
  color: var(--text-tertiary);
  line-height: var(--lh-normal);
}

/* Reproduce steps + code blocks */
.repro-list { display: flex; flex-direction: column; gap: var(--s-6); }
.repro-title {
  font-size: var(--fs-14);
  font-weight: var(--fw-semibold);
  color: var(--text-primary);
  margin-bottom: 2px;
}
.repro-note { margin: 0 0 var(--s-3) !important; font-size: var(--fs-13) !important; color: var(--text-tertiary); }
.codeblock {
  position: relative;
  background: var(--bg-inset);
  border: 1px solid var(--border-subtle);
  border-radius: var(--r-md);
}
.codeblock-copy { position: absolute; top: 6px; right: 6px; }
.codeblock pre {
  margin: 0;
  padding: var(--s-4) var(--s-5);
  padding-right: 40px;
  overflow-x: auto;
}
.codeblock code {
  font-family: var(--font-mono);
  font-size: var(--fs-13);
  line-height: 1.7;
  color: var(--text-primary);
  white-space: pre;
  background: none;
  border: none;
  padding: 0;
}

/* Simulation-scope cards */
.sim-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
  gap: var(--s-4);
}
.sim-card {
  padding: var(--s-4);
  background: var(--bg-surface);
  border: 1px solid var(--border-subtle);
  border-radius: var(--r-md);
}
.sim-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: var(--s-3);
  margin-bottom: var(--s-3);
}
.sim-name { font-size: var(--fs-14); font-weight: var(--fw-semibold); color: var(--text-primary); }
.sim-row { display: flex; gap: var(--s-3); padding: 3px 0; }
.sim-label {
  flex-shrink: 0;
  width: 74px;
  font-size: var(--fs-11);
  font-weight: var(--fw-semibold);
  text-transform: uppercase;
  letter-spacing: var(--tracking-wide);
  color: var(--text-tertiary);
  padding-top: 1px;
}
.sim-val { font-size: var(--fs-13); color: var(--text-secondary); line-height: var(--lh-normal); }

/* Honest-negatives list */
.negatives { margin: 0 0 var(--s-6); padding-left: var(--s-6); }
.negatives li {
  font-size: var(--fs-13);
  color: var(--text-secondary);
  line-height: var(--lh-normal);
  margin-bottom: var(--s-3);
}

/* Glossary */
.glossary-controls {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  justify-content: space-between;
  gap: var(--s-4);
}
.glossary-search {
  display: flex;
  align-items: center;
  gap: var(--s-3);
  flex: 1 1 260px;
  padding: 7px var(--s-4);
  background: var(--bg-surface);
  border: 1px solid var(--border-strong);
  border-radius: var(--r-md);
}
.glossary-search:focus-within {
  border-color: var(--border-focus);
  box-shadow: 0 0 0 3px var(--accent-wash);
}
.glossary-search input {
  flex: 1;
  border: none;
  outline: none;
  background: none;
  font-family: var(--font-sans);
  font-size: var(--fs-14);
  color: var(--text-primary);
}
.glossary-filters { display: flex; flex-wrap: wrap; gap: var(--s-2); }
.glossary-group + .glossary-group { margin-top: var(--s-7); }
.glossary-cat {
  font-size: var(--fs-11);
  font-weight: var(--fw-semibold);
  text-transform: uppercase;
  letter-spacing: var(--tracking-wide);
  color: var(--text-tertiary);
  padding-bottom: var(--s-2);
  border-bottom: 1px solid var(--border-subtle);
  margin-bottom: var(--s-4);
}
.glossary-list { margin: 0; }
.glossary-item {
  display: grid;
  grid-template-columns: 200px minmax(0, 1fr);
  gap: var(--s-5);
  padding: var(--s-4) 0;
  border-bottom: 1px solid var(--border-subtle);
}
.glossary-item:last-child { border-bottom: none; }
.glossary-item dt { display: flex; flex-direction: column; gap: 2px; }
.glossary-term { font-size: var(--fs-14); font-weight: var(--fw-semibold); color: var(--text-primary); }
.glossary-abbr { font-size: var(--fs-12); color: var(--text-tertiary); }
.glossary-item dd { margin: 0; }
.glossary-def { margin: 0 0 var(--s-2) !important; font-size: var(--fs-13) !important; color: var(--text-secondary); }
.glossary-why {
  margin: 0 !important;
  font-size: var(--fs-12) !important;
  color: var(--text-tertiary);
  line-height: var(--lh-normal);
}
.glossary-why-label {
  display: inline-block;
  font-weight: var(--fw-semibold);
  color: var(--accent-strong);
  margin-right: var(--s-2);
}
.glossary-empty {
  padding: var(--s-7);
  text-align: center;
  color: var(--text-tertiary);
  font-size: var(--fs-13);
  background: var(--bg-subtle);
  border: 1px solid var(--border-subtle);
  border-radius: var(--r-md);
}

/* FAQ (native details/summary) */
.faq-list {
  border: 1px solid var(--border-subtle);
  border-radius: var(--r-md);
  overflow: hidden;
  background: var(--bg-surface);
}
.faq-item { border-bottom: 1px solid var(--border-subtle); }
.faq-item:last-child { border-bottom: none; }
.faq-item summary {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: var(--s-4);
  padding: var(--s-4) var(--s-5);
  cursor: pointer;
  list-style: none;
  font-size: var(--fs-14);
  font-weight: var(--fw-medium);
  color: var(--text-primary);
  transition: background var(--dur-fast) var(--ease-standard);
}
.faq-item summary::-webkit-details-marker { display: none; }
.faq-item summary:hover { background: var(--bg-hover); }
.faq-chevron {
  flex-shrink: 0;
  color: var(--text-tertiary);
  transition: transform var(--dur-fast) var(--ease-standard);
}
.faq-item[open] .faq-chevron { transform: rotate(90deg); }
.faq-answer {
  padding: 0 var(--s-5) var(--s-5);
  font-size: var(--fs-13);
  line-height: var(--lh-normal);
  color: var(--text-secondary);
}
.faq-answer a { color: var(--text-link); }
.faq-answer .mono { color: var(--text-primary); }

/* Responsive: collapse the sidebar under a laptop/tablet breakpoint */
@media (max-width: 860px) {
  .docs-grid { grid-template-columns: minmax(0, 1fr); gap: var(--s-6); }
  .docs-nav {
    position: static;
    display: flex;
    gap: var(--s-2);
    overflow-x: auto;
    padding-bottom: var(--s-3);
    border-bottom: 1px solid var(--border-subtle);
  }
  .docs-nav .eyebrow { display: none; }
  .docs-nav ul { display: flex; gap: var(--s-2); }
  .docs-nav a { white-space: nowrap; }
  .glossary-item { grid-template-columns: minmax(0, 1fr); gap: var(--s-2); }
}

@media (prefers-reduced-motion: reduce) {
  .faq-chevron, .docs-nav a, .faq-item summary { transition: none !important; }
}
`;
