import { useEffect, useRef } from 'react';
import { Play, Pause, RotateCcw, FastForward } from 'lucide-react';
import type { AppState, Action } from '../state';
import { getEventTimeline, type Milestone } from '../data/eventTimelines';

interface Props {
  state: AppState;
  dispatch: React.Dispatch<Action>;
}

const MS_PER_DAY = 86400 * 1000;

function isoDay(d: Date): string {
  return d.toISOString().slice(0, 10);
}

function parseDay(s: string): Date {
  return new Date(s + 'T00:00:00Z');
}

function fmtDay(s: string): string {
  const d = parseDay(s);
  return d.toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric', timeZone: 'UTC' });
}

export function TimeSlider({ state, dispatch }: Props) {
  // Window: if an EventTimeline is registered for the selected record, use its full
  // [windowStart, windowEnd] span. Otherwise fall back to T-60→T-0 around the advisory.
  const advisory = state.selectedRecord?.advisoryDate;
  const timeline = getEventTimeline(state.selectedRecord?.eventKey);
  const startISO = timeline ? timeline.windowStart : undefined;
  const endISO = timeline ? timeline.windowEnd : undefined;
  const anchor = advisory ?? state.currentDate;
  const anchorD = parseDay(anchor);
  const startD = startISO ? parseDay(startISO) : new Date(anchorD.getTime() - 60 * MS_PER_DAY);
  const endD = endISO ? parseDay(endISO) : anchorD;
  const totalDays = Math.max(
    1,
    Math.round((endD.getTime() - startD.getTime()) / MS_PER_DAY),
  );

  const currentD = parseDay(state.currentDate);
  let dayOffset = Math.round((currentD.getTime() - startD.getTime()) / MS_PER_DAY);
  if (dayOffset < 0) dayOffset = 0;
  if (dayOffset > totalDays) dayOffset = totalDays;

  // Playback is now globally owned (state.playingStory) so EventTimelinePanel and
  // the slider stay in sync. One loop, one source of truth.
  const playing = state.playingStory;
  const speed = state.playStorySpeed;
  const lastTick = useRef(0);

  useEffect(() => {
    if (!playing) return;
    let raf = 0;
    lastTick.current = performance.now();
    // 1× = 1 day per 400ms (2.5 days/sec) — deliberately slow so the user can
    // read each toast and watch the modality cards settle.
    const interval = 400 / speed;
    const tick = (t: number) => {
      if (t - lastTick.current >= interval) {
        lastTick.current = t;
        const cur = parseDay(state.currentDate).getTime();
        const next = cur + MS_PER_DAY;
        if (next > endD.getTime()) {
          dispatch({ type: 'STOP_STORY' });
          return;
        }
        dispatch({ type: 'SET_DATE', date: isoDay(new Date(next)) });
      }
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [playing, speed, state.currentDate, dispatch, endD]);

  const togglePlay = () => {
    if (playing) {
      dispatch({ type: 'STOP_STORY' });
      return;
    }
    const restart = state.currentDate >= isoDay(endD);
    dispatch({ type: 'START_STORY', from: restart ? isoDay(startD) : undefined });
  };

  const onSlide = (e: React.ChangeEvent<HTMLInputElement>) => {
    const offset = Number(e.target.value);
    const d = new Date(startD.getTime() + offset * MS_PER_DAY);
    dispatch({ type: 'SET_DATE', date: isoDay(d) });
  };

  const reset = () => {
    if (playing) dispatch({ type: 'STOP_STORY' });
    dispatch({ type: 'SET_DATE', date: isoDay(startD) });
    dispatch({ type: 'RESET_STORY_FIRES' });
  };

  // Hide entirely if no selection and user hasn't interacted with date — show always when event selected.
  const visible = !!advisory;
  if (!visible) return null;

  // T-offset relative to advisory (negative = before). Computed from the advisory anchor.
  const advisoryD = advisory ? parseDay(advisory) : anchorD;
  const tEvent = Math.round((currentD.getTime() - advisoryD.getTime()) / MS_PER_DAY);

  // Milestone markers — placed by % offset within the playable window.
  const markers: { milestone: Milestone; pct: number }[] = timeline
    ? timeline.milestones.map((m) => ({
        milestone: m,
        pct:
          Math.max(
            0,
            Math.min(
              1,
              (parseDay(m.date).getTime() - startD.getTime()) /
                (endD.getTime() - startD.getTime()),
            ),
          ) * 100,
      }))
    : [];

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
        position: 'absolute',
        top: 16,
        left: 280,
        right: 460,
        height: 48,
        background: 'rgba(255,255,255,0.78)',
        backdropFilter: 'blur(20px) saturate(180%)',
        WebkitBackdropFilter: 'blur(20px) saturate(180%)',
        borderRadius: 'var(--r-pill)',
        boxShadow: 'var(--shadow-md)',
        display: 'flex',
        alignItems: 'center',
        padding: '0 8px 0 16px',
        gap: 12,
        zIndex: 'var(--z-map-overlay)' as any,
        animation: 'fade-in-up var(--dur-base) var(--ease-entrance)',
      }}
    >
      <button
        onClick={reset}
        title="Reset to T-60"
        style={{
          width: 28,
          height: 28,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          color: 'var(--text-secondary)',
          borderRadius: 'var(--r-pill)',
        }}
        onMouseEnter={(e) => (e.currentTarget.style.background = 'var(--bg-canvas)')}
        onMouseLeave={(e) => (e.currentTarget.style.background = 'transparent')}
      >
        <RotateCcw size={14} strokeWidth={2} />
      </button>
      <button
        onClick={togglePlay}
        title={playing ? 'Pause' : 'Play'}
        style={{
          width: 32,
          height: 32,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          color: '#ffffff',
          background: 'var(--accent)',
          borderRadius: 'var(--r-pill)',
        }}
      >
        {playing ? <Pause size={14} strokeWidth={2} /> : <Play size={14} strokeWidth={2} style={{ marginLeft: 2 }} />}
      </button>
      <button
        onClick={() => {
          const next = speed === 1 ? 4 : speed === 4 ? 16 : 1;
          dispatch({ type: 'SET_STORY_SPEED', speed: next });
        }}
        title={`Playback speed (${speed}×)`}
        style={{
          height: 28,
          padding: '0 10px',
          display: 'flex',
          alignItems: 'center',
          gap: 4,
          color: speed > 1 ? 'var(--accent)' : 'var(--text-secondary)',
          fontSize: 11,
          fontWeight: 600,
          borderRadius: 'var(--r-pill)',
          background: speed > 1 ? 'var(--accent-soft)' : 'transparent',
        }}
      >
        <FastForward size={12} strokeWidth={2} />
        {speed}×
      </button>
      <div style={{ flex: 1, position: 'relative' }}>
        <input
          type="range"
          min={0}
          max={totalDays}
          step={1}
          value={dayOffset}
          onChange={onSlide}
          style={{ width: '100%', accentColor: 'var(--accent)' }}
        />
        {/* Milestone markers (only when an EventTimeline is registered) */}
        {markers.length > 0 && (
          <div
            style={{
              position: 'absolute',
              top: 14,
              left: 0,
              right: 0,
              height: 10,
              pointerEvents: 'none',
            }}
          >
            {markers.map(({ milestone: m, pct }) => (
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
                  width: m.kind === 'sentinel-alert' || m.kind === 'official' ? 8 : 5,
                  height: m.kind === 'sentinel-alert' || m.kind === 'official' ? 8 : 5,
                  borderRadius: '50%',
                  background: markerColor(m.kind),
                  border: '1px solid #fff',
                  pointerEvents: 'auto',
                  cursor: 'pointer',
                  boxShadow: '0 1px 2px rgba(0,0,0,0.25)',
                }}
              />
            ))}
          </div>
        )}
      </div>
      <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'flex-end', minWidth: 110 }}>
        <div className="mono" style={{ fontSize: 12, color: 'var(--text-primary)', fontWeight: 500 }}>
          {fmtDay(state.currentDate)}
        </div>
        <div
          className="mono"
          style={{
            fontSize: 10,
            color: tEvent === 0 ? 'var(--alert)' : tEvent > 0 ? 'var(--alert)' : 'var(--accent)',
          }}
        >
          {tEvent === 0 ? 'Advisory day' : tEvent > 0 ? `T+${tEvent}d` : `T${tEvent}d`}
        </div>
      </div>
    </div>
  );
}
