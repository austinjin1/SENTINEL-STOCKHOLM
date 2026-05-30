import { useEffect } from 'react';
import { Play, Pause, FastForward, X, ChevronsRight } from 'lucide-react';
import type { CanonicalRecord, AlertTier } from '../types';
import type { AppState, Action } from '../state';
import type { EventTimeline, Milestone } from '../data/eventTimelines';
import { daysFromAdvisory } from '../data/eventTimelines';
import { sampleSeriesAt, daysBetween } from '../engine/timeSample';
import { EventBanner } from './EventBanner';
import { AquaSSMCard } from './modalities/AquaSSM';
import { HydroViTCard } from './modalities/HydroViT';
import { BioMotionCard } from './modalities/BioMotion';
import { MicroBiomeNetCard } from './modalities/MicroBiomeNet';
import { ToxiGeneCard } from './modalities/ToxiGene';

interface Props {
  timeline: EventTimeline;
  record: CanonicalRecord;
  state: AppState;
  dispatch: React.Dispatch<Action>;
  onClose: () => void;
}

const ALERT_STYLES: Record<AlertTier, { fg: string; bg: string; label: string }> = {
  MAINTAIN: { fg: 'var(--nominal)', bg: 'var(--nominal-dim)', label: 'Maintain' },
  WATCH: { fg: 'var(--watch)', bg: 'var(--watch-dim)', label: 'Watch' },
  INVESTIGATE: { fg: 'var(--investigate)', bg: 'var(--investigate-dim)', label: 'Investigate' },
  ALERT: { fg: 'var(--alert)', bg: 'var(--alert-dim)', label: 'Alert' },
};

const MS_PER_DAY = 86400000;
const isoDay = (t: number) => new Date(t).toISOString().slice(0, 10);
const parseDay = (s: string) => new Date(s + 'T00:00:00Z').getTime();

export function LivePlaybackPanel({ timeline, record, state, dispatch, onClose }: Props) {
  // Escape closes the panel and stops playback.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [onClose]);

  // Live-sampled scalars driving every "current" readout in the panel.
  const sampledAnomaly = sampleSeriesAt(
    record.sensor.anomaly_series,
    timeline.windowStart,
    timeline.advisoryDate,
    state.currentDate,
  );
  const liveAlert: AlertTier =
    sampledAnomaly > 0.85 ? 'ALERT'
    : sampledAnomaly > 0.65 ? 'INVESTIGATE'
    : sampledAnomaly > 0.4 ? 'WATCH'
    : 'MAINTAIN';
  const alert = ALERT_STYLES[liveAlert];

  const maskPct = Math.max(
    0,
    Math.min(
      1,
      (parseDay(state.currentDate) - parseDay(timeline.windowStart)) /
        Math.max(1, parseDay(timeline.advisoryDate) - parseDay(timeline.windowStart)),
    ),
  );

  // Virtual cascade tier — derives modality activation from where we are in the story
  // so cards "light up" without actually charging the user. Mirrors the paper's
  // cascade-controller monitor→investigate→alert progression.
  const virtualTier: 0 | 1 | 2 | 3 =
    state.currentDate >= timeline.advisoryDate ? 3
    : state.currentDate >= timeline.sentinelAlertDate ? 2
    : sampledAnomaly > 0.5 ? 1
    : 0;

  const daysToAdvisory = daysBetween(state.currentDate, timeline.advisoryDate);
  const playing = state.playingStory;
  const speed = state.playStorySpeed;

  const togglePlay = () => {
    if (playing) {
      dispatch({ type: 'STOP_STORY' });
      return;
    }
    const restart = state.currentDate >= timeline.windowEnd;
    dispatch({ type: 'START_STORY', from: restart ? timeline.windowStart : undefined });
  };

  const cycleSpeed = () => {
    const next = speed === 1 ? 4 : speed === 4 ? 16 : 1;
    dispatch({ type: 'SET_STORY_SPEED', speed: next });
  };

  // Slider state.
  const totalDays = Math.max(
    1,
    Math.round((parseDay(timeline.windowEnd) - parseDay(timeline.windowStart)) / MS_PER_DAY),
  );
  const dayOffset = Math.max(
    0,
    Math.min(
      totalDays,
      Math.round((parseDay(state.currentDate) - parseDay(timeline.windowStart)) / MS_PER_DAY),
    ),
  );

  const markerColor = (kind: Milestone['kind']) =>
    kind === 'sentinel-alert' ? 'var(--nominal, #2e8540)'
    : kind === 'official' ? 'var(--alert, #c0362c)'
    : kind === 'precursor' ? '#b06e1a'
    : kind === 'first-impact' ? '#b8430f'
    : kind === 'resolution' ? 'var(--accent, #0071e3)'
    : 'var(--text-tertiary)';

  return (
    <div
      style={{
        position: 'fixed',
        inset: 0,
        background: 'rgba(0,0,0,0.55)',
        backdropFilter: 'blur(8px)',
        zIndex: 9500,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        padding: 24,
        animation: 'fade-in var(--dur-base)',
      }}
      onClick={onClose}
    >
      <div
        onClick={(e) => e.stopPropagation()}
        style={{
          width: 'min(1100px, 100%)',
          maxHeight: '100%',
          background: 'rgba(255,255,255,0.97)',
          borderRadius: 'var(--r-lg)',
          boxShadow: 'var(--shadow-lg)',
          display: 'flex',
          flexDirection: 'column',
          overflow: 'hidden',
          animation: 'fade-in-up var(--dur-slow) var(--ease-entrance)',
        }}
      >
        {/* Header */}
        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: 12,
            padding: '14px 20px',
            borderBottom: '1px solid var(--border-subtle)',
          }}
        >
          <div style={{ flex: 1, minWidth: 0 }}>
            <div style={{ fontSize: 11, letterSpacing: '0.12em', textTransform: 'uppercase', color: 'var(--text-tertiary)' }}>
              Live playback
            </div>
            <div style={{ fontSize: 17, fontWeight: 600, letterSpacing: '-0.01em', color: 'var(--text-primary)' }}>
              {record.name}
            </div>
          </div>
          <div style={{ fontSize: 11, color: 'var(--text-tertiary)' }}>
            <span className="mono" style={{ color: 'var(--text-primary)', fontSize: 13, marginRight: 8 }}>
              {state.currentDate}
            </span>
            {daysToAdvisory > 0
              ? <>T<span className="mono">−{daysToAdvisory}</span>d to advisory</>
              : daysToAdvisory === 0 ? 'Advisory day'
              : <>T+<span className="mono">{-daysToAdvisory}</span>d</>}
          </div>
          <button
            onClick={onClose}
            title="Close (Esc)"
            style={{
              width: 30, height: 30, display: 'flex', alignItems: 'center', justifyContent: 'center',
              borderRadius: 'var(--r-pill)', background: 'var(--bg-canvas)', color: 'var(--text-secondary)',
            }}
          >
            <X size={16} strokeWidth={1.75} />
          </button>
        </div>

        {/* Banner */}
        <EventBanner timeline={timeline} currentDate={state.currentDate} />

        {/* Stage caption — large, plain-language "what is happening now" */}
        <StageCaption timeline={timeline} state={state} virtualTier={virtualTier} />

        {/* Stats strip */}
        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: 24,
            padding: '14px 20px',
            background: 'var(--bg-canvas)',
            borderBottom: '1px solid var(--border-subtle)',
          }}
        >
          <div>
            <div className="metric-hero" style={{ fontSize: 36, lineHeight: 1 }}>
              {sampledAnomaly.toFixed(3)}
            </div>
            <div style={{ fontSize: 11, color: 'var(--text-tertiary)', marginTop: 2 }}>
              Fused anomaly · sampled at T{daysFromAdvisory(timeline, state.currentDate) === 0 ? 0 : (daysFromAdvisory(timeline, state.currentDate) > 0 ? '+' : '') + daysFromAdvisory(timeline, state.currentDate)}d
            </div>
          </div>
          <div
            style={{
              display: 'flex', alignItems: 'center', gap: 6,
              height: 28, padding: '0 12px',
              borderRadius: 'var(--r-pill)',
              background: alert.bg, color: alert.fg,
              fontSize: 13, fontWeight: 600,
            }}
          >
            <span style={{ width: 7, height: 7, borderRadius: 4, background: alert.fg }} />
            {alert.label}
          </div>
          <div
            style={{
              display: 'flex', alignItems: 'center', gap: 6,
              height: 28, padding: '0 12px',
              borderRadius: 'var(--r-pill)',
              background: 'var(--accent-soft)', color: 'var(--accent)',
              fontSize: 12, fontWeight: 500,
            }}
          >
            Virtual cascade ·{' '}
            <span className="mono">T{virtualTier}</span>
          </div>
          <div style={{ marginLeft: 'auto', fontSize: 12, color: 'var(--text-tertiary)' }}>
            Lead time at threshold ·{' '}
            <span className="mono" style={{ color: 'var(--accent)' }}>
              {Math.abs(daysFromAdvisory(timeline, timeline.sentinelAlertDate))}d
            </span>
          </div>
        </div>

        {/* Modality grid */}
        <div
          style={{
            flex: 1,
            minHeight: 0,
            overflow: 'auto',
            padding: 16,
            display: 'grid',
            gridTemplateColumns: 'repeat(3, 1fr)',
            gap: 12,
            background: 'var(--bg-subtle, #f5f5f7)',
          }}
        >
          <AquaSSMCard sensor={record.sensor} cardIndex={0} maskPct={maskPct} />
          <HydroViTCard
            satellite={record.satellite}
            cardIndex={1}
            tier={virtualTier}
            date={state.currentDate}
            bbox={record.bbox}
          />
          <BioMotionCard bio={record.behavioral} cardIndex={2} />
          <MicroBiomeNetCard micro={record.microbial} cardIndex={3} tier={virtualTier} />
          <ToxiGeneCard molecular={record.molecular} cardIndex={4} />
          {/* Status tile */}
          <div
            style={{
              background: 'var(--bg-elevated)',
              border: '1px solid var(--border-subtle)',
              borderRadius: 'var(--r-lg)',
              padding: 16,
              display: 'flex',
              flexDirection: 'column',
              gap: 8,
              animation: 'fade-in-up var(--dur-base) var(--ease-entrance) 225ms both',
            }}
          >
            <div style={{ fontSize: 13, fontWeight: 600 }}>Cascade controller</div>
            <div style={{ fontSize: 11, color: 'var(--text-tertiary)' }}>
              PPO-trained gate, auto-escalating as evidence accumulates.
            </div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 6, marginTop: 6 }}>
              {[
                { tier: 0, label: 'Monitor · sensor + behavioral' },
                { tier: 1, label: 'Investigate · + satellite' },
                { tier: 2, label: 'Alert · + microbial' },
                { tier: 3, label: 'Confirm · + molecular' },
              ].map((t) => {
                const active = virtualTier >= t.tier;
                const current = virtualTier === t.tier;
                return (
                  <div
                    key={t.tier}
                    style={{
                      display: 'flex',
                      alignItems: 'center',
                      gap: 8,
                      padding: '6px 10px',
                      borderRadius: 'var(--r-sm)',
                      background: current ? 'var(--accent-soft)' : active ? 'var(--bg-canvas)' : 'transparent',
                      border: current ? '1px solid var(--accent)' : '1px solid transparent',
                      fontSize: 11,
                      color: active ? 'var(--text-primary)' : 'var(--text-tertiary)',
                      transition: 'all 250ms',
                    }}
                  >
                    <span
                      className="mono"
                      style={{
                        fontSize: 10,
                        fontWeight: 700,
                        color: current ? 'var(--accent)' : active ? 'var(--text-secondary)' : 'var(--text-tertiary)',
                      }}
                    >
                      T{t.tier}
                    </span>
                    {t.label}
                  </div>
                );
              })}
            </div>
          </div>
        </div>

        {/* Mini timeline */}
        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: 12,
            padding: '12px 20px',
            borderTop: '1px solid var(--border-subtle)',
            background: 'rgba(255,255,255,0.85)',
          }}
        >
          <button
            onClick={togglePlay}
            title={playing ? 'Pause' : 'Play'}
            style={{
              width: 36, height: 36,
              display: 'flex', alignItems: 'center', justifyContent: 'center',
              color: '#fff', background: 'var(--accent)',
              borderRadius: 'var(--r-pill)',
            }}
          >
            {playing ? <Pause size={15} strokeWidth={2} /> : <Play size={15} strokeWidth={2} style={{ marginLeft: 2 }} />}
          </button>
          <button
            onClick={cycleSpeed}
            title={`Speed ${speed}×`}
            style={{
              height: 28, padding: '0 10px',
              display: 'flex', alignItems: 'center', gap: 4,
              color: speed > 1 ? 'var(--accent)' : 'var(--text-secondary)',
              background: speed > 1 ? 'var(--accent-soft)' : 'transparent',
              border: '1px solid var(--border-subtle)',
              borderRadius: 'var(--r-pill)',
              fontSize: 11, fontWeight: 600,
            }}
          >
            <FastForward size={12} strokeWidth={2} />
            {speed}×
          </button>
          <button
            onClick={() => dispatch({ type: 'SET_DATE', date: timeline.advisoryDate })}
            title="Skip to advisory"
            style={{
              height: 28, padding: '0 10px',
              display: 'flex', alignItems: 'center', gap: 4,
              color: 'var(--text-secondary)',
              border: '1px solid var(--border-subtle)',
              borderRadius: 'var(--r-pill)',
              fontSize: 11, fontWeight: 500,
            }}
          >
            <ChevronsRight size={11} strokeWidth={2} />
            T0
          </button>
          <div style={{ flex: 1, position: 'relative' }}>
            <input
              type="range"
              min={0}
              max={totalDays}
              step={1}
              value={dayOffset}
              onChange={(e) => {
                const d = parseDay(timeline.windowStart) + Number(e.target.value) * MS_PER_DAY;
                dispatch({ type: 'SET_DATE', date: isoDay(d) });
              }}
              style={{ width: '100%', accentColor: 'var(--accent)' }}
            />
            {/* Milestone dots */}
            <div
              style={{
                position: 'absolute',
                top: 16,
                left: 0,
                right: 0,
                height: 12,
                pointerEvents: 'none',
              }}
            >
              {timeline.milestones.map((m) => {
                const pct =
                  ((parseDay(m.date) - parseDay(timeline.windowStart)) /
                    Math.max(
                      1,
                      parseDay(timeline.windowEnd) - parseDay(timeline.windowStart),
                    )) *
                  100;
                const isBig = m.kind === 'sentinel-alert' || m.kind === 'official';
                return (
                  <div
                    key={m.date + m.title}
                    title={`${m.date} · ${m.title}`}
                    onClick={(e) => {
                      e.stopPropagation();
                      dispatch({ type: 'SET_DATE', date: m.date });
                    }}
                    style={{
                      position: 'absolute',
                      left: `${pct}%`,
                      transform: 'translateX(-50%)',
                      width: isBig ? 9 : 6,
                      height: isBig ? 9 : 6,
                      borderRadius: '50%',
                      background: markerColor(m.kind),
                      border: '1px solid #fff',
                      pointerEvents: 'auto',
                      cursor: 'pointer',
                      boxShadow: '0 1px 2px rgba(0,0,0,0.25)',
                    }}
                  />
                );
              })}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

/* ------- stage caption: plain-language "what is happening now" ------- */

function StageCaption({
  timeline,
  state,
  virtualTier,
}: {
  timeline: EventTimeline;
  state: AppState;
  virtualTier: 0 | 1 | 2 | 3;
}) {
  // Find the most recent milestone whose date ≤ currentDate (the "we are here" one).
  const currentMilestone = (() => {
    let last: Milestone | null = null;
    for (const m of timeline.milestones) {
      if (m.date <= state.currentDate) last = m;
      else break;
    }
    return last;
  })();

  const beforeAlert = state.currentDate < timeline.sentinelAlertDate;
  const inAlertWindow =
    state.currentDate >= timeline.sentinelAlertDate &&
    state.currentDate < timeline.advisoryDate;

  const headline = beforeAlert
    ? 'Baseline · monitoring sensor + behavioral'
    : inAlertWindow
      ? `SENTINEL alert active · cascade ${virtualTier === 1 ? 'Investigate' : virtualTier === 2 ? 'Alert' : 'Monitor'}`
      : `Official advisory in effect · ${timeline.advisoryDate}`;

  const sub = currentMilestone
    ? `Now: ${currentMilestone.title}`
    : 'Awaiting first signal.';

  const accent = beforeAlert
    ? 'var(--text-tertiary)'
    : inAlertWindow
      ? 'var(--nominal, #2e8540)'
      : 'var(--alert, #c0362c)';

  return (
    <div
      style={{
        padding: '14px 20px',
        background: 'var(--bg-elevated, #fff)',
        borderBottom: '1px solid var(--border-subtle)',
        display: 'flex',
        flexDirection: 'column',
        gap: 4,
      }}
    >
      <div
        style={{
          fontSize: 11,
          letterSpacing: '0.14em',
          textTransform: 'uppercase',
          color: accent,
          fontWeight: 700,
        }}
      >
        {headline}
      </div>
      <div
        style={{
          fontSize: 15,
          fontWeight: 500,
          color: 'var(--text-primary)',
          lineHeight: 1.35,
        }}
      >
        {sub}
      </div>
      {currentMilestone?.body && (
        <div style={{ fontSize: 12, color: 'var(--text-secondary)', lineHeight: 1.45 }}>
          {currentMilestone.body}
        </div>
      )}
    </div>
  );
}
