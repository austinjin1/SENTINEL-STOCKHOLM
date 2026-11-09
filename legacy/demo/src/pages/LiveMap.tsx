import { useEffect, useRef, useState } from 'react';
import { X } from 'lucide-react';
import { TopBar } from '../components/TopBar';
import { LeftRail } from '../components/LeftRail';
import { MapCanvas } from '../components/MapCanvas';
import { TimeSlider } from '../components/TimeSlider';
import { PlaceCard } from '../components/PlaceCard';
import { ImportDropZone } from '../components/ImportDropZone';
import { PinnedRecordCard } from '../components/PinnedRecordCard';
import { LivePlaybackPanel } from '../components/LivePlaybackPanel';
import { getEventTimeline } from '../data/eventTimelines';
import { useToasts } from '../components/Toasts';
import { DemoOverlay } from '../components/DemoOverlay';
import { MethodologyPanel } from '../components/MethodologyPanel';
import { useAppState } from '../state';
import { applyTierActivation } from '../engine/escalate';
import { readInitialUrlState, useUrlSync } from '../urlState';

const LAYER_KEYS = ['usgs', 'neon', 'satellite', 'chlorophyll', 'drought'] as const;
const MS_PER_DAY = 86400000;

function shiftDate(iso: string, days: number): string {
  const t = new Date(iso + 'T00:00:00Z').getTime() + days * MS_PER_DAY;
  return new Date(t).toISOString().slice(0, 10);
}

/**
 * LiveMap — the interactive map dashboard, mounted at /live. This is the
 * original single-view app, lightly restyled to the genomics token system
 * (flat surfaces, hairline borders — no glass/gradient). It runs in a fixed,
 * non-scrolling viewport, so it toggles `body.app-fixed` while mounted.
 */
export default function LiveMap() {
  const [state, dispatch] = useAppState();
  const prevTier = useRef(state.cascadeTier);
  const urlHydrated = useRef(false);
  const [helpOpen, setHelpOpen] = useState(false);
  const { push: pushToast } = useToasts();

  // Fixed, non-scrolling viewport only for the map view.
  useEffect(() => {
    document.body.classList.add('app-fixed');
    return () => document.body.classList.remove('app-fixed');
  }, []);

  // Story-playback milestone toaster.
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
      if (m.escalateTo !== undefined && m.escalateTo > state.cascadeTier) {
        dispatch({ type: 'SET_CASCADE_TIER', tier: m.escalateTo });
      }
    }
    // cascadeTier is read but intentionally excluded to avoid re-running on tier changes.
    // eslint-disable-next-line react-hooks/exhaustive-deps
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
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [state.cascadeTier, state.selectedRecord?.id]);

  return (
    <div style={{ height: 'calc(100vh - var(--nav-height))', display: 'flex', flexDirection: 'column', background: 'var(--bg-canvas)' }}>
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
  const dialogRef = useRef<HTMLDivElement>(null);
  const closeRef = useRef<HTMLButtonElement>(null);
  const rows: [string, string][] = [
    ['E / D', 'Escalate / de-escalate tier'],
    ['1 – 5', 'Toggle USGS / NEON / satellite / chl-a / drought'],
    ['[ / ]', 'Step date ± 1 day'],
    ['P', 'Pin selected record (compare mode)'],
    ['Esc', 'Close rail / help'],
    ['?', 'Show this help'],
  ];

  // Focus management: move focus into the dialog on open, trap Tab within it,
  // and restore focus to the previously-focused element (the trigger) on close.
  useEffect(() => {
    const prevActive = document.activeElement as HTMLElement | null;
    closeRef.current?.focus();
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key !== 'Tab') return;
      const root = dialogRef.current;
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
      document.removeEventListener('keydown', onKeyDown, true);
      prevActive?.focus?.();
    };
  }, []);

  return (
    <div
      onClick={onClose}
      style={{
        position: 'fixed', inset: 0, background: 'var(--bg-scrim)',
        display: 'flex', alignItems: 'center', justifyContent: 'center',
        zIndex: 10000, animation: 'fade-in var(--dur-fast)',
      }}
    >
      <div
        ref={dialogRef}
        role="dialog"
        aria-modal="true"
        aria-label="Keyboard shortcuts"
        onClick={(e) => e.stopPropagation()}
        style={{
          background: 'var(--bg-surface)',
          padding: 24, borderRadius: 'var(--r-lg)',
          boxShadow: 'var(--shadow-lg)', minWidth: 320,
          border: '1px solid var(--border-subtle)',
        }}
      >
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: 12 }}>
          <div style={{ fontSize: 16, fontWeight: 600 }}>Keyboard shortcuts</div>
          <button
            ref={closeRef}
            onClick={onClose}
            aria-label="Close keyboard shortcuts"
            style={{
              width: 28, height: 28, borderRadius: 'var(--r-pill)',
              background: 'var(--bg-canvas)', color: 'var(--text-secondary)',
              display: 'flex', alignItems: 'center', justifyContent: 'center',
              border: '1px solid var(--border-subtle)', flexShrink: 0,
            }}
          >
            <X size={16} strokeWidth={1.75} />
          </button>
        </div>
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
