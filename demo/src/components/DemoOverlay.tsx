import { useEffect, useState } from 'react';
import type { Action } from '../state';
import { EVENTS, CLEAN_SITES } from '../data/realData';
import { applyTierActivation } from '../engine/escalate';

interface Step {
  at: number;
  caption: string;
  action?: () => void;
}

interface Props {
  active: boolean;
  dispatch: React.Dispatch<Action>;
  currentRecord: any;
}

export function DemoOverlay({ active, dispatch }: Props) {
  const [caption, setCaption] = useState('');
  const [progress, setProgress] = useState(0);

  useEffect(() => {
    if (!active) {
      setCaption('');
      setProgress(0);
      return;
    }
    const lakeErie = EVENTS[0];
    const toolik = CLEAN_SITES[0];
    const totalMs = 90000;
    const steps: Step[] = [
      { at: 0, caption: 'SENTINEL · operational console · cached data', action: () => dispatch({ type: 'CLOSE_RAIL' }) },
      { at: 8000, caption: 'Lake Erie HAB · July 2023 · the failure', action: () => dispatch({ type: 'SELECT_SITE', record: lakeErie }) },
      { at: 20000, caption: 'AquaSSM anomaly climbs to 0.997 — 59.3d before advisory' },
      { at: 30000, caption: 'Five modalities fuse · AUROC 0.939' },
      { at: 38000, caption: 'Escalate to Tier 2 — query microbial source attribution', action: () => {
        dispatch({ type: 'ESCALATE' }); // 0→1
        setTimeout(() => dispatch({ type: 'ESCALATE' }), 1200); // 1→2
        setTimeout(() => {
          const patch = applyTierActivation(lakeErie, 2);
          dispatch({ type: 'PATCH_RECORD', patch });
        }, 1800);
      }},
      { at: 50000, caption: 'Microbial source: nutrient runoff 0.94 — only spent lab cost because cheaper modalities flagged first' },
      { at: 58000, caption: 'Escalate to Tier 3 — molecular mechanism', action: () => {
        dispatch({ type: 'ESCALATE' });
        setTimeout(() => {
          const patch = applyTierActivation(lakeErie, 3);
          dispatch({ type: 'PATCH_RECORD', patch });
        }, 800);
      }},
      { at: 65000, caption: 'AHR/CYP1A → xenobiotic metabolism → hepatotoxicity' },
      { at: 75000, caption: 'Contrast: a pristine site', action: () => dispatch({ type: 'SELECT_SITE', record: toolik }) },
      { at: 85000, caption: 'All MAINTAIN · 0 alerts · bio modalities deployable — the system knows the difference' },
    ];
    const start = performance.now();
    const ids: number[] = [];
    steps.forEach((s) => {
      ids.push(window.setTimeout(() => {
        setCaption(s.caption);
        s.action?.();
      }, s.at));
    });
    const tick = () => {
      const elapsed = performance.now() - start;
      setProgress(Math.min(1, elapsed / totalMs));
      if (elapsed < totalMs) raf = requestAnimationFrame(tick);
      else dispatch({ type: 'TOGGLE_DEMO' });
    };
    let raf = requestAnimationFrame(tick);
    return () => {
      ids.forEach(clearTimeout);
      cancelAnimationFrame(raf);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [active]);

  useEffect(() => {
    if (!active) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') dispatch({ type: 'TOGGLE_DEMO' });
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [active, dispatch]);

  if (!active) return null;

  return (
    <>
      <div
        style={{
          position: 'fixed',
          inset: 0,
          background: 'rgba(13,17,23,0.18)',
          pointerEvents: 'none',
          zIndex: 'var(--z-demo)' as any,
          animation: 'fade-in var(--dur-base)',
        }}
      />
      <div
        style={{
          position: 'fixed',
          bottom: 32,
          left: '50%',
          transform: 'translateX(-50%)',
          padding: '16px 24px',
          background: 'var(--bg-surface)',
          borderRadius: 'var(--r-lg)',
          boxShadow: 'var(--shadow-lg)',
          color: 'var(--text-primary)',
          fontSize: 15,
          minWidth: 520,
          textAlign: 'center',
          zIndex: 'var(--z-demo)' as any,
          animation: 'fade-in-up var(--dur-base) var(--ease-entrance)',
        }}
      >
        {caption}
        <div style={{ height: 3, background: 'var(--bg-canvas)', marginTop: 12, borderRadius: 1.5 }}>
          <div
            style={{
              height: '100%',
              width: `${progress * 100}%`,
              background: 'var(--accent)',
              borderRadius: 1.5,
              transition: 'width 100ms linear',
            }}
          />
        </div>
      </div>
    </>
  );
}
