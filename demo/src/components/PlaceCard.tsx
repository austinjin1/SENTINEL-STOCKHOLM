import { useEffect, useRef, useState } from 'react';
import { X, ChevronDown, ChevronRight, Zap, Clock, Pin } from 'lucide-react';
import type { CanonicalRecord, AlertTier } from '../types';
import type { AppState, Action } from '../state';
import { fetchLiveSensor, patchFromLive } from '../engine/liveUsgs';
import { useToasts } from './Toasts';
import { getEventTimeline } from '../data/eventTimelines';
import { sampleSeriesAt } from '../engine/timeSample';
import { EventBanner } from './EventBanner';
import { EventTimelinePanel } from './EventTimelinePanel';
import { AquaSSMCard } from './modalities/AquaSSM';
import { HydroViTCard } from './modalities/HydroViT';
import { MicroBiomeNetCard } from './modalities/MicroBiomeNet';
import { ToxiGeneCard } from './modalities/ToxiGene';
import { BioMotionCard } from './modalities/BioMotion';
import { EVENT_CONTENT } from '../data/eventContent';
import { WhatIfPanel } from './WhatIfPanel';

interface Props {
  state: AppState;
  dispatch: React.Dispatch<Action>;
  record: CanonicalRecord;
}

const ALERT_STYLES: Record<AlertTier, { fg: string; bg: string; label: string }> = {
  MAINTAIN: { fg: 'var(--nominal)', bg: 'var(--nominal-dim)', label: 'Maintain' },
  WATCH: { fg: 'var(--watch)', bg: 'var(--watch-dim)', label: 'Watch' },
  INVESTIGATE: { fg: 'var(--investigate)', bg: 'var(--investigate-dim)', label: 'Investigate' },
  ALERT: { fg: 'var(--alert)', bg: 'var(--alert-dim)', label: 'Alert' },
};

const TIER_COST = [0, 0.5, 15, 25]; // $K added per step
const TIER_DESC = [
  'Sensor + behavioral monitoring',
  'Sensor + behavioral + satellite',
  'Adds microbial source attribution',
  'Adds molecular toxicity profile',
];

function useCountUp(target: number, dur = 800) {
  const [v, setV] = useState(0);
  useEffect(() => {
    const start = performance.now();
    let raf = 0;
    const step = (t: number) => {
      const p = Math.min(1, (t - start) / dur);
      const e = 1 - Math.pow(1 - p, 3);
      setV(target * e);
      if (p < 1) raf = requestAnimationFrame(step);
    };
    raf = requestAnimationFrame(step);
    return () => cancelAnimationFrame(raf);
  }, [target, dur]);
  return v;
}

export function PlaceCard({ state, dispatch, record }: Props) {
  const [summaryOpen, setSummaryOpen] = useState(true);
  const [whatIfOpen, setWhatIfOpen] = useState(false);
  const [modsOpen, setModsOpen] = useState(false);
  const [timelineOpen, setTimelineOpen] = useState(false);
  const [expandedModality, setExpandedModality] = useState<string | null>(null);

  // Time-aware anomaly: when an EventTimeline is registered for this record, sample
  // anomaly_series at currentDate so the hero metric "plays" along with the timeline.
  const timeline = getEventTimeline(record.eventKey);
  const sampledAnomaly = timeline
    ? sampleSeriesAt(
        record.sensor.anomaly_series,
        timeline.windowStart,
        timeline.advisoryDate,
        state.currentDate,
      )
    : record.fusion.anomaly;
  // Fraction of the playback window already traversed — drives sparkline future mask.
  const maskPct = timeline
    ? Math.max(
        0,
        Math.min(
          1,
          (new Date(state.currentDate + 'T00:00:00Z').getTime() -
            new Date(timeline.windowStart + 'T00:00:00Z').getTime()) /
            (new Date(timeline.advisoryDate + 'T00:00:00Z').getTime() -
              new Date(timeline.windowStart + 'T00:00:00Z').getTime()),
        ),
      )
    : 1;
  const anomaly = useCountUp(sampledAnomaly);
  // Alert tier should follow the sampled value during playback, not the static peak.
  const liveAlert = timeline
    ? sampledAnomaly > 0.85
      ? 'ALERT'
      : sampledAnomaly > 0.65
        ? 'INVESTIGATE'
        : sampledAnomaly > 0.4
          ? 'WATCH'
          : 'MAINTAIN'
    : record.fusion.alert;
  const alert = ALERT_STYLES[liveAlert];
  const tier = state.cascadeTier;
  const canEscalate = tier < 3;
  const nextCost = canEscalate ? TIER_COST[tier + 1] : 0;
  const { push: pushToast } = useToasts();

  // Live USGS polling — only when the record was hydrated from a USGS station.
  // Backoff: 60s normal, doubles on error up to 10min, resets on success.
  const isLiveBacked = record.name?.startsWith('USGS ') ?? false;
  const backoffRef = useRef(60_000);
  useEffect(() => {
    if (!isLiveBacked) return;
    let cancelled = false;
    let timer: number | null = null;
    const poll = async () => {
      try {
        const live = await fetchLiveSensor(record.lat, record.lon);
        if (cancelled) return;
        const patch = patchFromLive(record, live);
        if (Object.keys(patch).length > 0) dispatch({ type: 'PATCH_RECORD', patch });
        backoffRef.current = 60_000;
      } catch {
        backoffRef.current = Math.min(600_000, backoffRef.current * 2);
      } finally {
        if (!cancelled) timer = window.setTimeout(poll, backoffRef.current);
      }
    };
    timer = window.setTimeout(poll, backoffRef.current);
    return () => {
      cancelled = true;
      if (timer != null) window.clearTimeout(timer);
    };
  }, [isLiveBacked, record.lat, record.lon, dispatch, record]);

  const pin = () => {
    dispatch({ type: 'PIN_RECORD', record });
    pushToast('info', `Pinned ${record.name} — pick another site to compare.`, 2400);
  };

  return (
    <div
      style={{
        position: 'absolute',
        top: 16,
        right: 16,
        bottom: 16,
        width: 420,
        background: 'rgba(255,255,255,0.88)',
        backdropFilter: 'blur(24px) saturate(180%)',
        WebkitBackdropFilter: 'blur(24px) saturate(180%)',
        borderRadius: 'var(--r-lg)',
        boxShadow: 'var(--shadow-card)',
        display: 'flex',
        flexDirection: 'column',
        overflow: 'hidden',
        animation: 'slide-in-right var(--dur-slow) var(--ease-entrance)',
        zIndex: 'var(--z-place-card)' as any,
      }}
    >
      {/* Event banner — pre-alert / alert / advisory */}
      {timeline && <EventBanner timeline={timeline} currentDate={state.currentDate} />}

      {/* Hero satellite image (events only) */}
      {record.bookmarked && record.bbox && record.advisoryDate && (
        <HeroImage
          bbox={record.bbox}
          date={state.currentDate}
          advisoryDate={record.advisoryDate}
        />
      )}

      {/* Header */}
      <div style={{ padding: '20px 24px 0', position: 'relative' }}>
        <button
          onClick={() => dispatch({ type: 'CLOSE_RAIL' })}
          title="Close"
          style={{
            position: 'absolute',
            top: 14,
            right: 14,
            width: 30,
            height: 30,
            borderRadius: 'var(--r-pill)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            color: 'var(--text-secondary)',
            background: 'var(--bg-canvas)',
          }}
          onMouseEnter={(e) => {
            e.currentTarget.style.background = 'rgba(0,0,0,0.08)';
            e.currentTarget.style.color = 'var(--text-primary)';
          }}
          onMouseLeave={(e) => {
            e.currentTarget.style.background = 'var(--bg-canvas)';
            e.currentTarget.style.color = 'var(--text-secondary)';
          }}
        >
          <X size={16} strokeWidth={1.75} />
        </button>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8, paddingRight: 80 }}>
          <div style={{ fontSize: 22, fontWeight: 600, letterSpacing: '-0.02em', flex: 1 }}>
            {record.name}
          </div>
          <button
            onClick={pin}
            title={state.pinnedRecord?.id === record.id ? 'Already pinned' : 'Pin for compare (P)'}
            disabled={state.pinnedRecord?.id === record.id}
            style={{
              width: 30, height: 30,
              display: 'flex', alignItems: 'center', justifyContent: 'center',
              borderRadius: 'var(--r-pill)',
              background: state.pinnedRecord?.id === record.id ? 'var(--accent-soft)' : 'var(--bg-canvas)',
              color: state.pinnedRecord?.id === record.id ? 'var(--accent)' : 'var(--text-secondary)',
              cursor: state.pinnedRecord?.id === record.id ? 'default' : 'pointer',
            }}
          >
            <Pin size={14} strokeWidth={1.75} />
          </button>
        </div>
        <div style={{ fontSize: 14, color: 'var(--text-secondary)', marginTop: 2 }}>
          {record.label}
        </div>
        <div
          style={{
            fontSize: 12,
            color: 'var(--text-tertiary)',
            marginTop: 10,
            display: 'flex',
            gap: 12,
            flexWrap: 'wrap',
          }}
        >
          {record.id && (
            <span>
              USGS{' '}
              <span className="mono" style={{ color: 'var(--text-secondary)' }}>
                {record.id.split('-')[0]}
              </span>
            </span>
          )}
          <span className="mono" style={{ color: 'var(--text-secondary)' }}>
            {record.lat.toFixed(3)}, {record.lon.toFixed(3)}
          </span>
        </div>
      </div>

      {/* Scrollable body */}
      <div style={{ flex: 1, overflowY: 'auto', minHeight: 0 }}>
        {/* Hero metric */}
        <div style={{ padding: '20px 24px 16px', position: 'relative' }}>
          {state.loadingLive && (
            <div
              style={{
                position: 'absolute',
                inset: 0,
                background:
                  'linear-gradient(90deg, transparent, rgba(255,255,255,0.6), transparent)',
                backgroundSize: '200% 100%',
                animation: 'shimmer 1.4s linear infinite',
                pointerEvents: 'none',
                borderRadius: 'var(--r-md)',
              }}
            />
          )}
          <div className="metric-hero" style={{ opacity: state.loadingLive ? 0.5 : 1, transition: 'opacity 240ms' }}>
            {anomaly.toFixed(3)}
          </div>
          <div style={{ fontSize: 13, color: 'var(--text-tertiary)', marginTop: 4 }}>
            {state.loadingLive ? 'Fetching live USGS data…' : 'Fused anomaly probability'}
            {record.fusion.auroc && (
              <>
                {' · '}AUROC <span className="mono">{record.fusion.auroc.toFixed(3)}</span>
              </>
            )}
          </div>
          <div style={{ display: 'flex', gap: 8, marginTop: 14, flexWrap: 'wrap' }}>
            <div
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: 6,
                height: 28,
                padding: '0 12px',
                borderRadius: 'var(--r-pill)',
                background: alert.bg,
                color: alert.fg,
                fontSize: 13,
                fontWeight: 600,
              }}
            >
              <span style={{ width: 7, height: 7, borderRadius: 4, background: alert.fg }} />
              {alert.label}
            </div>
            {record.leadDays !== undefined && (
              <div
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: 6,
                  height: 28,
                  padding: '0 12px',
                  borderRadius: 'var(--r-pill)',
                  background: 'var(--accent-soft)',
                  color: 'var(--accent)',
                  fontSize: 13,
                  fontWeight: 500,
                }}
              >
                <Clock size={12} strokeWidth={2} />
                <span className="mono">{record.leadDays.toFixed(1)}d</span> before advisory
              </div>
            )}
          </div>
        </div>

        {/* Editorial narrative + impact (events only) */}
        {record.eventKey && EVENT_CONTENT[record.eventKey] && (
          <EditorialBlock content={EVENT_CONTENT[record.eventKey]} />
        )}

        {/* Cascade action */}
        <div style={{ padding: '0 24px 16px' }}>
          <button
            onClick={() => dispatch({ type: 'ESCALATE' })}
            disabled={!canEscalate}
            style={{
              width: '100%',
              height: 44,
              borderRadius: 'var(--r-md)',
              background: canEscalate ? 'var(--accent)' : 'var(--bg-canvas)',
              color: canEscalate ? '#ffffff' : 'var(--text-tertiary)',
              fontSize: 14,
              fontWeight: 500,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              gap: 8,
              cursor: canEscalate ? 'pointer' : 'not-allowed',
              transition: 'background var(--dur-fast)',
            }}
            onMouseEnter={(e) => {
              if (canEscalate) e.currentTarget.style.background = 'var(--accent-bright)';
            }}
            onMouseLeave={(e) => {
              if (canEscalate) e.currentTarget.style.background = 'var(--accent)';
            }}
          >
            <Zap size={15} strokeWidth={2} />
            {canEscalate ? `Run deeper analysis · +$${nextCost.toFixed(1)}K` : 'All modalities active'}
          </button>
          <div style={{ fontSize: 12, color: 'var(--text-tertiary)', marginTop: 8, display: 'flex', justifyContent: 'space-between' }}>
            <span>Currently Tier {tier} · {TIER_DESC[tier]}</span>
            {tier > 0 && (
              <button
                onClick={() => dispatch({ type: 'DEESCALATE' })}
                style={{ color: 'var(--text-link)', fontSize: 12 }}
              >
                De-escalate
              </button>
            )}
          </div>
          <div style={{ fontSize: 12, color: 'var(--text-tertiary)', marginTop: 2 }}>
            Total spend <span className="mono">${state.totalCostK.toFixed(1)}K</span>
          </div>
        </div>

        {/* Detection summary */}
        <Section
          title="Detection summary"
          open={summaryOpen}
          onToggle={() => setSummaryOpen((v) => !v)}
        >
          <DetectionSummary record={record} />
        </Section>

        {/* What-if simulation */}
        <Section
          title="What-if simulation"
          open={whatIfOpen}
          onToggle={() => setWhatIfOpen((v) => !v)}
        >
          <WhatIfPanel record={record} dispatch={dispatch} />
        </Section>

        {/* Modalities */}
        <Section
          title={`Modality breakdown · ${record.n_modalities}/5 active`}
          open={modsOpen}
          onToggle={() => setModsOpen((v) => !v)}
        >
          <ModalityList
            record={record}
            tier={tier}
            expanded={expandedModality}
            setExpanded={setExpandedModality}
            maskPct={maskPct}
          />
        </Section>

        {/* Timeline (events only) — full milestone panel when registered, scrubber fallback otherwise */}
        {record.bookmarked && (
          <Section title="Timeline" open={timelineOpen} onToggle={() => setTimelineOpen((v) => !v)}>
            {timeline ? (
              <EventTimelinePanel
                timeline={timeline}
                state={state}
                dispatch={dispatch}
              />
            ) : (
              <TimelineInline record={record} />
            )}
          </Section>
        )}

        <div style={{ height: 8 }} />
      </div>

      {/* Footer */}
      <div
        style={{
          padding: '10px 24px',
          borderTop: '1px solid var(--border-subtle)',
          fontSize: 11,
          color: 'var(--text-tertiary)',
          background: 'var(--bg-subtle)',
        }}
      >
        Cached demonstration data ·{' '}
        <button
          onClick={() => dispatch({ type: 'TOGGLE_METHODOLOGY' })}
          style={{ color: 'var(--text-link)', fontSize: 11 }}
        >
          What's real, what's modeled
        </button>
      </div>
    </div>
  );
}

/* ---- subcomponents ---- */

function EditorialBlock({ content }: { content: import('../data/eventContent').EventContent }) {
  return (
    <div
      style={{
        padding: '0 24px 20px',
        animation: 'fade-in-up var(--dur-base) var(--ease-entrance)',
      }}
    >
      <p
        style={{
          fontSize: 14,
          lineHeight: 1.55,
          color: 'var(--text-secondary)',
          margin: 0,
        }}
      >
        {content.narrative}
      </p>

      {content.quote && (
        <blockquote
          style={{
            margin: '16px 0 0',
            padding: '12px 16px',
            borderLeft: '3px solid var(--accent)',
            background: 'var(--bg-canvas)',
            borderRadius: '0 var(--r-md) var(--r-md) 0',
            fontSize: 13,
            color: 'var(--text-primary)',
            fontStyle: 'italic',
            lineHeight: 1.5,
          }}
        >
          "{content.quote.text}"
          <div
            style={{
              fontStyle: 'normal',
              fontSize: 11,
              color: 'var(--text-tertiary)',
              marginTop: 8,
            }}
          >
            — {content.quote.cite}
          </div>
        </blockquote>
      )}

      <div style={{ marginTop: 16, display: 'flex', flexDirection: 'column', gap: 8 }}>
        {content.impact.map((row) => (
          <div
            key={row.label}
            style={{
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'baseline',
              fontSize: 13,
              gap: 12,
            }}
          >
            <span style={{ color: 'var(--text-tertiary)' }}>{row.label}</span>
            <span className="mono" style={{ color: 'var(--text-primary)', fontWeight: 500, textAlign: 'right' }}>
              {row.value}
            </span>
          </div>
        ))}
      </div>

      <div
        style={{
          marginTop: 14,
          paddingTop: 12,
          borderTop: '1px solid var(--border-subtle)',
          fontSize: 11,
          color: 'var(--text-tertiary)',
          lineHeight: 1.5,
        }}
      >
        <span style={{ fontWeight: 500, color: 'var(--text-secondary)' }}>Sources · </span>
        {content.sources.map((s, i) => (
          <span key={i}>
            {i > 0 && '; '}
            {s.url ? (
              <a href={s.url} target="_blank" rel="noreferrer" style={{ color: 'var(--text-link)' }}>
                {s.label}
              </a>
            ) : (
              s.label
            )}
          </span>
        ))}
      </div>
    </div>
  );
}

function HeroImage({
  bbox,
  date,
  advisoryDate,
}: {
  bbox: [number, number, number, number];
  date: string;
  advisoryDate: string;
}) {
  // NASA Worldview Snapshot API. Returns a static image for a given bbox + date + layers.
  // Docs: https://wiki.earthdata.nasa.gov/display/GIBS/GIBS+API+for+Developers
  const [west, south, east, north] = bbox;
  const layer = 'MODIS_Terra_CorrectedReflectance_TrueColor,Coastlines';
  const url = `https://wvs.earthdata.nasa.gov/api/v1/snapshot?REQUEST=GetSnapshot&LAYERS=${layer}&CRS=EPSG:4326&TIME=${date}&BBOX=${south},${west},${north},${east}&WIDTH=480&HEIGHT=200&FORMAT=image/jpeg`;
  const tOffset = Math.round(
    (new Date(date + 'T00:00:00Z').getTime() - new Date(advisoryDate + 'T00:00:00Z').getTime()) /
      86400000,
  );
  return (
    <div
      style={{
        position: 'relative',
        height: 180,
        background: '#0a1a2c',
        overflow: 'hidden',
        flexShrink: 0,
      }}
    >
      <img
        src={url}
        alt={`Satellite view at ${date}`}
        loading="lazy"
        style={{
          width: '100%',
          height: '100%',
          objectFit: 'cover',
          display: 'block',
          animation: 'fade-in 320ms var(--ease-entrance)',
        }}
        onError={(e) => {
          // GIBS sometimes returns a black tile for cloudy days — hide gracefully.
          (e.target as HTMLImageElement).style.opacity = '0.3';
        }}
      />
      {/* gradient scrim for readability of overlay text */}
      <div
        style={{
          position: 'absolute',
          inset: 0,
          background:
            'linear-gradient(180deg, rgba(0,0,0,0) 40%, rgba(0,0,0,0.55) 100%)',
        }}
      />
      <div
        style={{
          position: 'absolute',
          bottom: 10,
          left: 16,
          right: 16,
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'flex-end',
          color: '#ffffff',
          fontSize: 11,
        }}
      >
        <span style={{ opacity: 0.85 }}>NASA EOSDIS · MODIS Terra true color</span>
        <span className="mono" style={{ opacity: 0.85 }}>
          {tOffset === 0 ? 'Advisory day' : tOffset < 0 ? `T${tOffset}d` : `T+${tOffset}d`}
        </span>
      </div>
    </div>
  );
}

function Section({
  title,
  open,
  onToggle,
  children,
}: {
  title: string;
  open: boolean;
  onToggle: () => void;
  children: React.ReactNode;
}) {
  return (
    <div style={{ borderTop: '1px solid var(--border-subtle)' }}>
      <button
        onClick={onToggle}
        style={{
          width: '100%',
          padding: '12px 24px',
          display: 'flex',
          alignItems: 'center',
          gap: 8,
          color: 'var(--text-primary)',
          fontSize: 13,
          fontWeight: 600,
          transition: 'background var(--dur-fast)',
        }}
        onMouseEnter={(e) => (e.currentTarget.style.background = 'var(--bg-canvas)')}
        onMouseLeave={(e) => (e.currentTarget.style.background = 'transparent')}
      >
        {open ? <ChevronDown size={14} strokeWidth={2} /> : <ChevronRight size={14} strokeWidth={2} />}
        {title}
      </button>
      {open && <div style={{ padding: '0 24px 16px' }}>{children}</div>}
    </div>
  );
}

function DetectionSummary({ record }: { record: CanonicalRecord }) {
  const MOD_LABELS = [
    { label: 'Sensor', color: 'var(--m-sensor)' },
    { label: 'Satellite', color: 'var(--m-satellite)' },
    { label: 'Behavioral', color: 'var(--m-behavioral)' },
    { label: 'Microbial', color: 'var(--m-microbial)' },
    { label: 'Molecular', color: 'var(--m-molecular)' },
  ];
  const top = MOD_LABELS.map((m, i) => ({ ...m, w: record.fusion.attention[i] }))
    .filter((m) => m.w > 0)
    .sort((a, b) => b.w - a.w);
  const topNames = top.slice(0, 2).map((m) => m.label.toLowerCase()).join(' and ');
  const topShare = (top.slice(0, 2).reduce((s, m) => s + m.w, 0) * 100).toFixed(0);
  return (
    <>
      <div style={{ fontSize: 13, color: 'var(--text-secondary)', lineHeight: 1.5 }}>
        {top.length === 0
          ? 'Awaiting modality activation.'
          : `${record.n_modalities} of 5 modalities fused. ${topNames.charAt(0).toUpperCase() + topNames.slice(1)} carry ${topShare}% of the attention.`}
      </div>
      <div style={{ marginTop: 12, display: 'flex', height: 10, borderRadius: 5, overflow: 'hidden', background: 'var(--bg-canvas)' }}>
        {record.fusion.attention.map((w, i) => (
          <div
            key={i}
            title={`${MOD_LABELS[i].label} ${(w * 100).toFixed(1)}%`}
            style={{
              width: `${w * 100}%`,
              background: MOD_LABELS[i].color,
              transition: 'width 500ms var(--ease-entrance)',
            }}
          />
        ))}
      </div>
      <div style={{ fontSize: 11, color: 'var(--text-tertiary)', marginTop: 8 }}>
        Coverage <span className="mono">{(record.fusion.coverage * 100).toFixed(0)}%</span> · Microbial &
        molecular weight near zero in routine detection — retained for source attribution.
      </div>
    </>
  );
}

function ModalityList({
  record,
  tier,
  expanded,
  setExpanded,
  maskPct,
}: {
  record: CanonicalRecord;
  tier: 0 | 1 | 2 | 3;
  expanded: string | null;
  setExpanded: (id: string | null) => void;
  maskPct: number;
}) {
  const rows = [
    {
      id: 'sensor',
      name: 'AquaSSM',
      sub: 'Continuous sensor anomaly',
      color: 'var(--m-sensor)',
      coverage: record.sensor.state,
      headline: record.sensor.anomaly.toFixed(3),
      headlineLabel: 'anomaly',
      viz: <AquaSSMCard sensor={record.sensor} maskPct={maskPct} />,
    },
    {
      id: 'satellite',
      name: 'HydroViT',
      sub: 'Sentinel-2 spectral indices',
      color: 'var(--m-satellite)',
      coverage: record.satellite.state,
      headline: 'R² 0.749',
      headlineLabel: 'water temperature',
      viz: <HydroViTCard satellite={record.satellite} tier={tier} />,
    },
    {
      id: 'behavioral',
      name: 'BioMotion',
      sub: 'Daphnia behavioral assay',
      color: 'var(--m-behavioral)',
      coverage: record.behavioral.state,
      headline: record.behavioral.anomaly.toFixed(3),
      headlineLabel: 'AUROC',
      viz: <BioMotionCard bio={record.behavioral} />,
    },
    {
      id: 'microbial',
      name: 'MicroBiomeNet',
      sub: 'eDNA source attribution',
      color: 'var(--m-microbial)',
      coverage: record.microbial.state,
      headline: record.microbial.top ? `${record.microbial.classes.find((c) => c.name === record.microbial.top)?.p.toFixed(2) ?? ''}` : '—',
      headlineLabel: record.microbial.top ? `top · ${record.microbial.top.toLowerCase()}` : 'awaiting activation',
      viz: <MicroBiomeNetCard micro={record.microbial} tier={tier} />,
    },
    {
      id: 'molecular',
      name: 'ToxiGene',
      sub: 'Transcriptomic toxicity',
      color: 'var(--m-molecular)',
      coverage: record.molecular.state,
      headline: record.molecular.outcome ?? '—',
      headlineLabel: 'predicted outcome',
      viz: <ToxiGeneCard molecular={record.molecular} />,
    },
  ] as const;

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
      {rows.map((r) => {
        const isExpanded = expanded === r.id;
        const deployable = r.coverage === 'DEPLOYABLE';
        return (
          <div key={r.id}>
            <button
              onClick={() => setExpanded(isExpanded ? null : r.id)}
              style={{
                width: '100%',
                display: 'flex',
                alignItems: 'center',
                gap: 12,
                padding: '10px 12px',
                background: isExpanded ? 'var(--bg-canvas)' : 'transparent',
                borderRadius: 'var(--r-sm)',
                opacity: deployable ? 0.55 : 1,
                transition: 'background var(--dur-fast)',
              }}
              onMouseEnter={(e) => {
                if (!isExpanded) e.currentTarget.style.background = 'var(--bg-canvas)';
              }}
              onMouseLeave={(e) => {
                if (!isExpanded) e.currentTarget.style.background = 'transparent';
              }}
            >
              <span
                style={{
                  width: 8,
                  height: 8,
                  borderRadius: 4,
                  background: r.color,
                  flexShrink: 0,
                }}
              />
              <div style={{ flex: 1, textAlign: 'left', minWidth: 0 }}>
                <div style={{ fontSize: 13, fontWeight: 500, color: 'var(--text-primary)' }}>{r.name}</div>
                <div style={{ fontSize: 11, color: 'var(--text-tertiary)' }}>{r.sub}</div>
              </div>
              <div style={{ textAlign: 'right', flexShrink: 0 }}>
                <div className="mono" style={{ fontSize: 13, color: 'var(--text-primary)', fontWeight: 500 }}>
                  {r.headline}
                </div>
                <div style={{ fontSize: 10, color: 'var(--text-tertiary)' }}>{r.headlineLabel}</div>
              </div>
              <ChevronRight
                size={14}
                strokeWidth={2}
                style={{
                  color: 'var(--text-tertiary)',
                  flexShrink: 0,
                  transform: isExpanded ? 'rotate(90deg)' : 'none',
                  transition: 'transform var(--dur-fast)',
                }}
              />
            </button>
            {isExpanded && (
              <div
                style={{
                  marginTop: 4,
                  marginBottom: 8,
                  animation: 'fade-in-up var(--dur-base) var(--ease-entrance)',
                }}
              >
                {r.viz}
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}

function TimelineInline({ record }: { record: CanonicalRecord }) {
  const [pos, setPos] = useState(record.leadDays ? -record.leadDays : 0);
  const lead = -pos;
  return (
    <div>
      <div style={{ fontSize: 12, color: 'var(--text-secondary)', marginBottom: 10 }}>
        Drag to scrub from advisory back to early detection.
      </div>
      <input
        type="range"
        min={-60}
        max={0}
        step={0.5}
        value={pos}
        onChange={(e) => setPos(Number(e.target.value))}
        style={{ width: '100%', accentColor: 'var(--accent)' }}
      />
      <div
        style={{
          display: 'flex',
          justifyContent: 'space-between',
          fontSize: 10,
          color: 'var(--text-tertiary)',
          marginTop: 4,
        }}
      >
        <span>−60d</span>
        <span>−30d</span>
        <span>−14d</span>
        <span>Advisory</span>
      </div>
      <div style={{ fontSize: 12, color: 'var(--accent)', marginTop: 8 }}>
        Detected <span className="mono">{lead.toFixed(1)}d</span> early
      </div>
    </div>
  );
}
