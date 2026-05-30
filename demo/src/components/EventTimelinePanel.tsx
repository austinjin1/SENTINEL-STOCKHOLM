import { Play, Pause, FastForward, ChevronsRight } from 'lucide-react';
import type { EventTimeline, MilestoneKind } from '../data/eventTimelines';
import { daysFromAdvisory } from '../data/eventTimelines';
import type { AppState, Action } from '../state';

interface Props {
  timeline: EventTimeline;
  state: AppState;
  dispatch: React.Dispatch<Action>;
}

const KIND_STYLE: Record<MilestoneKind, { color: string; label: string }> = {
  context: { color: 'var(--text-tertiary)', label: 'Context' },
  precursor: { color: '#b06e1a', label: 'Precursor' },
  'sentinel-alert': { color: 'var(--nominal, #2e8540)', label: 'SENTINEL alert' },
  'first-impact': { color: '#b8430f', label: 'First impact' },
  official: { color: 'var(--alert, #c0362c)', label: 'Official advisory' },
  resolution: { color: 'var(--accent, #0071e3)', label: 'Resolution' },
};

export function EventTimelinePanel({ timeline, state, dispatch }: Props) {
  const playing = state.playingStory;
  const speed = state.playStorySpeed;
  const currentDate = state.currentDate;

  // Playback loop is owned by TimeSlider — this component just dispatches actions
  // so the slider and panel always agree on what's "now".

  const onPlay = () => {
    if (playing) {
      dispatch({ type: 'STOP_STORY' });
      return;
    }
    // Restart from windowStart if we're at/past the end, else continue from current.
    const restart = currentDate >= timeline.windowEnd;
    dispatch({ type: 'START_STORY', from: restart ? timeline.windowStart : undefined });
  };

  // Index of the most-recent milestone whose date ≤ currentDate. Used to highlight.
  const currentIdx = (() => {
    let best = -1;
    for (let i = 0; i < timeline.milestones.length; i++) {
      if (timeline.milestones[i].date <= currentDate) best = i;
      else break;
    }
    return best;
  })();

  return (
    <div>
      {/* Controls */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 12 }}>
        <button
          onClick={onPlay}
          title={playing ? 'Pause story' : 'Play story from start'}
          style={{
            display: 'flex', alignItems: 'center', gap: 6,
            padding: '6px 12px',
            background: playing ? 'var(--alert-dim, rgba(192,54,44,0.12))' : 'var(--accent)',
            color: playing ? 'var(--alert, #c0362c)' : '#fff',
            borderRadius: 'var(--r-pill)',
            fontSize: 12, fontWeight: 600,
          }}
        >
          {playing ? <Pause size={12} strokeWidth={2} /> : <Play size={12} strokeWidth={2} />}
          {playing ? 'Pause' : 'Play story'}
        </button>
        <button
          onClick={() => {
            const next = speed === 1 ? 4 : speed === 4 ? 16 : 1;
            dispatch({ type: 'SET_STORY_SPEED', speed: next });
          }}
          title={`Story speed (${speed}×)`}
          style={{
            display: 'flex', alignItems: 'center', gap: 4,
            padding: '6px 10px',
            background: speed > 1 ? 'var(--accent-soft)' : 'transparent',
            border: '1px solid var(--border-subtle)',
            color: speed > 1 ? 'var(--accent)' : 'var(--text-secondary)',
            borderRadius: 'var(--r-pill)',
            fontSize: 11, fontWeight: 600,
          }}
        >
          <FastForward size={11} strokeWidth={2} />
          {speed}×
        </button>
        <button
          onClick={() => dispatch({ type: 'SET_DATE', date: timeline.advisoryDate })}
          title="Jump to advisory"
          style={{
            display: 'flex', alignItems: 'center', gap: 4,
            padding: '6px 10px',
            border: '1px solid var(--border-subtle)',
            color: 'var(--text-secondary)',
            borderRadius: 'var(--r-pill)',
            fontSize: 11, fontWeight: 500,
          }}
        >
          <ChevronsRight size={11} strokeWidth={2} />
          Skip to advisory
        </button>
      </div>

      {/* Milestone list */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
        {timeline.milestones.map((m, i) => {
          const past = m.date <= currentDate;
          const current = i === currentIdx;
          const fired = state.firedMilestones.includes(`${timeline.eventKey}|${m.date}`);
          const style = KIND_STYLE[m.kind];
          const offset = daysFromAdvisory(timeline, m.date);
          return (
            <button
              key={m.date + m.title}
              onClick={() => dispatch({ type: 'SET_DATE', date: m.date })}
              style={{
                display: 'flex',
                gap: 10,
                padding: '8px 10px',
                background: current
                  ? 'var(--accent-soft)'
                  : past
                    ? 'var(--bg-canvas)'
                    : 'transparent',
                border: `1px solid ${current ? style.color : 'var(--border-subtle)'}`,
                borderLeft: `3px solid ${style.color}`,
                borderRadius: 'var(--r-md)',
                opacity: past ? 1 : 0.55,
                textAlign: 'left',
                transition: 'background 200ms, opacity 200ms',
              }}
            >
              <div style={{ flex: 1, minWidth: 0 }}>
                <div style={{ display: 'flex', alignItems: 'baseline', gap: 8, marginBottom: 2 }}>
                  <span
                    className="mono"
                    style={{ fontSize: 11, color: 'var(--text-tertiary)' }}
                  >
                    {m.date} · {offset === 0 ? 'T0' : offset > 0 ? `T+${offset}d` : `T${offset}d`}
                  </span>
                  <span
                    style={{
                      fontSize: 9,
                      fontWeight: 700,
                      letterSpacing: '0.08em',
                      textTransform: 'uppercase',
                      color: style.color,
                    }}
                  >
                    {style.label}
                  </span>
                  {fired && (
                    <span
                      style={{
                        fontSize: 9,
                        color: 'var(--text-tertiary)',
                        marginLeft: 'auto',
                      }}
                    >
                      ✓ shown
                    </span>
                  )}
                </div>
                <div
                  style={{
                    fontSize: 13,
                    fontWeight: current ? 600 : 500,
                    color: 'var(--text-primary)',
                    lineHeight: 1.35,
                  }}
                >
                  {m.title}
                </div>
                {m.body && (
                  <div
                    style={{
                      fontSize: 12,
                      color: 'var(--text-secondary)',
                      marginTop: 3,
                      lineHeight: 1.45,
                    }}
                  >
                    {m.body}
                  </div>
                )}
                {m.source && (
                  <div style={{ fontSize: 10, color: 'var(--text-tertiary)', marginTop: 4 }}>
                    Source:{' '}
                    {m.source.url ? (
                      <a
                        href={m.source.url}
                        target="_blank"
                        rel="noreferrer"
                        style={{ color: 'var(--text-link)' }}
                      >
                        {m.source.label}
                      </a>
                    ) : (
                      m.source.label
                    )}
                  </div>
                )}
              </div>
            </button>
          );
        })}
      </div>

      {/* Lead-time callout */}
      <div
        style={{
          marginTop: 14,
          padding: '10px 12px',
          background: 'var(--accent-soft)',
          border: '1px solid var(--accent)',
          borderRadius: 'var(--r-md)',
          fontSize: 12,
          color: 'var(--text-primary)',
          lineHeight: 1.5,
        }}
      >
        <div style={{ fontWeight: 600, color: 'var(--accent)', marginBottom: 4 }}>
          Lead time ·{' '}
          <span className="mono">
            {Math.abs(daysFromAdvisory(timeline, timeline.sentinelAlertDate))}d
          </span>
        </div>
        {timeline.leadSummary}
      </div>
    </div>
  );
}
