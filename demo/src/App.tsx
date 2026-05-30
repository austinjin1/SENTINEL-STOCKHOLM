import { useEffect, useRef, useState } from 'react';
import './tokens.css';
import { TopBar } from './components/TopBar';
import { LeftRail } from './components/LeftRail';
import { MapCanvas } from './components/MapCanvas';
import { TimeSlider } from './components/TimeSlider';
import { PlaceCard } from './components/PlaceCard';
import { ImportDropZone } from './components/ImportDropZone';
import { PinnedRecordCard } from './components/PinnedRecordCard';
import { LivePlaybackPanel } from './components/LivePlaybackPanel';
import { getEventTimeline } from './data/eventTimelines';
import { useToasts } from './components/Toasts';
import { DemoOverlay } from './components/DemoOverlay';
import { MethodologyPanel } from './components/MethodologyPanel';
import { useAppState } from './state';
import { applyTierActivation } from './engine/escalate';
import { readInitialUrlState, useUrlSync } from './urlState';

const LAYER_KEYS = ['usgs', 'neon', 'satellite', 'chlorophyll', 'drought'] as const;
const MS_PER_DAY = 86400000;

function shiftDate(iso: string, days: number): string {
  const t = new Date(iso + 'T00:00:00Z').getTime() + days * MS_PER_DAY;
  return new Date(t).toISOString().slice(0, 10);
}

function App() {
  const [state, dispatch] = useAppState();
  const prevTier = useRef(state.cascadeTier);
  const urlHydrated = useRef(false);
  const [helpOpen, setHelpOpen] = useState(false);
  const { push: pushToast } = useToasts();

  // Story-playback milestone toaster: when the play head crosses a milestone we
  // haven't toasted yet, push a kind-coloured toast and mark it fired. Skipped
  // entirely when not in playingStory mode so manual scrubs don't spam toasts.
  useEffect(() => {
    if (!state.playingStory) return;
    const timeline = getEventTimeline(state.selectedRecord?.eventKey);
    if (!timeline) return;
    for (const m of timeline.milestones) {
      if (m.date > state.currentDate) break;
      const key = `${timeline.eventKey}|${m.date}`;
      if (state.firedMilestones.includes(key)) continue;
      const kind =
        m.kind === 'sentinel-alert' ? 'success'
        : m.kind === 'official' || m.kind === 'first-impact' ? 'error'
        : 'info';
      pushToast(kind, `${m.title}  ·  ${m.date}`, 5200);
      dispatch({ type: 'MARK_MILESTONE_FIRED', key });
      // Auto-escalate the real cascade tier if the milestone requests it. The cost
      // strip in PlaceCard now ticks up in lockstep with the cinematic story.
      if (m.escalateTo !== undefined && m.escalateTo > state.cascadeTier) {
        dispatch({ type: 'SET_CASCADE_TIER', tier: m.escalateTo });
      }
    }
  }, [
    state.playingStory,
    state.currentDate,
    state.selectedRecord,
    state.firedMilestones,
    pushToast,
    dispatch,
  ]);

  // Global keyboard shortcuts. Disabled while the user is typing in an input.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const target = e.target as HTMLElement | null;
      if (target && (target.tagName === 'INPUT' || target.tagName === 'TEXTAREA' || target.isContentEditable)) return;
      if (e.metaKey || e.ctrlKey || e.altKey) return;
      switch (e.key) {
        case 'e': case 'E': dispatch({ type: 'ESCALATE' }); break;
        case 'd': case 'D': dispatch({ type: 'DEESCALATE' }); break;
        case '1': case '2': case '3': case '4': case '5': {
          const layer = LAYER_KEYS[parseInt(e.key, 10) - 1];
          if (layer) dispatch({ type: 'TOGGLE_LAYER', layer });
          break;
        }
        case '[': dispatch({ type: 'SET_DATE', date: shiftDate(state.currentDate, -1) }); break;
        case ']': dispatch({ type: 'SET_DATE', date: shiftDate(state.currentDate, +1) }); break;
        case '?': setHelpOpen((v) => !v); break;
        case 'Escape':
          if (helpOpen) setHelpOpen(false);
          else if (state.selectedRecord) dispatch({ type: 'CLOSE_RAIL' });
          break;
        case 'p': case 'P':
          if (state.selectedRecord) dispatch({ type: 'PIN_RECORD', record: state.selectedRecord });
          break;
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [dispatch, state.currentDate, state.selectedRecord, helpOpen]);

  // Hydrate from URL exactly once.
  useEffect(() => {
    if (urlHydrated.current) return;
    urlHydrated.current = true;
    readInitialUrlState(dispatch, state);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useUrlSync(state);

  useEffect(() => {
    if (!state.selectedRecord) {
      prevTier.current = 0;
      return;
    }
    const record = state.selectedRecord;
    if (state.cascadeTier > prevTier.current) {
      const id = window.setTimeout(() => {
        const patch = applyTierActivation(record, state.cascadeTier);
        dispatch({ type: 'PATCH_RECORD', patch });
      }, 900);
      prevTier.current = state.cascadeTier;
      return () => clearTimeout(id);
    }
    prevTier.current = state.cascadeTier;
    return;
    // recordId guards the closure: switching records cancels the pending patch.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [state.cascadeTier, state.selectedRecord?.id]);

  return (
    <div style={{ height: '100vh', display: 'flex', flexDirection: 'column', background: 'var(--bg-canvas)' }}>
      <TopBar dispatch={dispatch} colorBlind={state.colorBlind} />
      <div style={{ flex: 1, display: 'flex', minHeight: 0, position: 'relative' }}>
        <LeftRail
          dispatch={dispatch}
          legendOpen={state.legendOpen}
          demoMode={state.demoMode}
          methodologyOpen={state.methodologyOpen}
        />
        <div style={{ flex: 1, position: 'relative', minWidth: 0 }}>
          <MapCanvas
            selected={state.selectedRecord}
            layers={state.layers}
            legendOpen={state.legendOpen}
            dispatch={dispatch}
            date={state.currentDate}
            colorBlind={state.colorBlind}
          />
          <TimeSlider state={state} dispatch={dispatch} />
          {state.selectedRecord && (
            <PlaceCard state={state} dispatch={dispatch} record={state.selectedRecord} />
          )}
          {state.pinnedRecord && (
            <PinnedRecordCard
              pinned={state.pinnedRecord}
              selected={state.selectedRecord}
              dispatch={dispatch}
            />
          )}
        </div>
      </div>
      <DemoOverlay active={state.demoMode} dispatch={dispatch} currentRecord={state.selectedRecord} />
      <MethodologyPanel open={state.methodologyOpen} dispatch={dispatch} />
      <ImportDropZone dispatch={dispatch} />
      {helpOpen && <KeyHelp onClose={() => setHelpOpen(false)} />}
      {(() => {
        const tl = getEventTimeline(state.selectedRecord?.eventKey);
        if (!tl || !state.selectedRecord || !state.playingStory) return null;
        return (
          <LivePlaybackPanel
            timeline={tl}
            record={state.selectedRecord}
            state={state}
            dispatch={dispatch}
            onClose={() => dispatch({ type: 'STOP_STORY' })}
          />
        );
      })()}
    </div>
  );
}

function KeyHelp({ onClose }: { onClose: () => void }) {
  const rows: [string, string][] = [
    ['E / D', 'Escalate / de-escalate tier'],
    ['1 – 5', 'Toggle USGS / NEON / satellite / chl-a / drought'],
    ['[ / ]', 'Step date ± 1 day'],
    ['P', 'Pin selected record (compare mode)'],
    ['Esc', 'Close rail / help'],
    ['?', 'Show this help'],
  ];
  return (
    <div
      onClick={onClose}
      style={{
        position: 'fixed', inset: 0, background: 'rgba(0,0,0,0.35)',
        display: 'flex', alignItems: 'center', justifyContent: 'center',
        zIndex: 10000, animation: 'fade-in var(--dur-fast)',
      }}
    >
      <div
        onClick={(e) => e.stopPropagation()}
        style={{
          background: 'rgba(255,255,255,0.98)',
          padding: 24, borderRadius: 'var(--r-lg)',
          boxShadow: 'var(--shadow-lg)', minWidth: 320,
        }}
      >
        <div style={{ fontSize: 16, fontWeight: 600, marginBottom: 12 }}>Keyboard shortcuts</div>
        {rows.map(([k, label]) => (
          <div key={k} style={{ display: 'flex', justifyContent: 'space-between', padding: '6px 0', fontSize: 13 }}>
            <span className="mono" style={{ color: 'var(--text-primary)' }}>{k}</span>
            <span style={{ color: 'var(--text-secondary)' }}>{label}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

export default App;
