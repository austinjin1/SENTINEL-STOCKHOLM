import {
  useCallback,
  useEffect,
  useRef,
  useState,
} from 'react';
import { ArrowRight, ArrowLeft, X } from 'lucide-react';
import { DEFAULT_TOUR, type TourStep } from './tourSteps';

/* ==========================================================================
   GuidedTour — a lightweight, dismissible, accessible product tour.

   A short spotlight walk over the top navigation + a few key page elements,
   in the flat genomics-tool language: an opaque scrim with a cut-out over the
   target (box-shadow ring, NOT a blur/glow), a small anchored card with prev /
   next / done, keyboard support (Esc, ←, →, Enter), focus trapped in the card,
   and full prefers-reduced-motion respect (no transitions, instant placement).

   It does NOT edit the nav or App — steps target existing DOM via CSS selectors
   (with graceful fallback: a step whose target is missing is centered, and the
   spotlight is simply omitted). Trigger it from anywhere by rendering
   <GuidedTour open={...} onClose={...} />; the Overview page owns the button.
   ========================================================================== */

const PAD = 8; // spotlight padding around the target
const GAP = 12; // gap between target and card

function prefersReducedMotion(): boolean {
  return (
    typeof window !== 'undefined' &&
    !!window.matchMedia?.('(prefers-reduced-motion: reduce)').matches
  );
}

interface Rect {
  top: number;
  left: number;
  width: number;
  height: number;
}

export function GuidedTour({
  open,
  onClose,
  steps = DEFAULT_TOUR,
  onFinish,
}: {
  open: boolean;
  onClose: () => void;
  steps?: TourStep[];
  onFinish?: () => void;
}) {
  const [index, setIndex] = useState(0);
  const [rect, setRect] = useState<Rect | null>(null);
  const cardRef = useRef<HTMLDivElement>(null);
  const reduced = prefersReducedMotion();

  // Close + reset to the first step, so the next open always starts at step 0.
  // Resetting on the way out (rather than during render or in an effect) keeps
  // the component lint-clean and avoids cascading renders.
  const close = useCallback(() => {
    setIndex(0);
    onClose();
  }, [onClose]);

  const step = steps[index];
  const isFirst = index === 0;
  const isLast = index === steps.length - 1;

  // Measure the current target (and re-measure on resize / scroll). Writes
  // through requestAnimationFrame so no setState runs synchronously in an
  // effect body.
  const rafRef = useRef(0);
  const measure = useCallback(() => {
    const selector = open ? step?.target : undefined;
    cancelAnimationFrame(rafRef.current);
    rafRef.current = requestAnimationFrame(() => {
      if (!selector) {
        setRect(null);
        return;
      }
      const el = document.querySelector(selector);
      if (!el) {
        setRect(null);
        return;
      }
      const r = el.getBoundingClientRect();
      setRect({ top: r.top, left: r.left, width: r.width, height: r.height });
      if (r.top < 0 || r.bottom > window.innerHeight) {
        el.scrollIntoView({ block: 'center', behavior: reduced ? 'auto' : 'smooth' });
      }
    });
  }, [open, step, reduced]);

  useEffect(() => {
    measure();
    return () => cancelAnimationFrame(rafRef.current);
  }, [measure]);

  useEffect(() => {
    if (!open) return;
    const onChange = () => measure();
    window.addEventListener('resize', onChange);
    window.addEventListener('scroll', onChange, true);
    return () => {
      window.removeEventListener('resize', onChange);
      window.removeEventListener('scroll', onChange, true);
    };
  }, [open, measure]);

  const next = useCallback(() => {
    if (isLast) {
      onFinish?.();
      close();
    } else {
      setIndex((i) => Math.min(i + 1, steps.length - 1));
    }
  }, [isLast, onFinish, close, steps.length]);

  const prev = useCallback(() => {
    setIndex((i) => Math.max(i - 1, 0));
  }, []);

  // Keyboard: Esc closes, arrows navigate, Enter advances. Trap focus in card.
  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        e.preventDefault();
        close();
      } else if (e.key === 'ArrowRight' || e.key === 'Enter') {
        e.preventDefault();
        next();
      } else if (e.key === 'ArrowLeft') {
        e.preventDefault();
        prev();
      } else if (e.key === 'Tab') {
        const card = cardRef.current;
        if (!card) return;
        const focusables = card.querySelectorAll<HTMLElement>(
          'button, [href], [tabindex]:not([tabindex="-1"])',
        );
        if (focusables.length === 0) return;
        const first = focusables[0];
        const last = focusables[focusables.length - 1];
        if (e.shiftKey && document.activeElement === first) {
          e.preventDefault();
          last.focus();
        } else if (!e.shiftKey && document.activeElement === last) {
          e.preventDefault();
          first.focus();
        }
      }
    };
    document.addEventListener('keydown', onKey, true);
    return () => document.removeEventListener('keydown', onKey, true);
  }, [open, next, prev, close]);

  // Move focus into the card when the step changes.
  useEffect(() => {
    if (!open) return;
    const t = window.setTimeout(() => cardRef.current?.focus(), 20);
    return () => window.clearTimeout(t);
  }, [open, index]);

  if (!open || !step) return null;

  const card = computeCardPosition(rect, step.placement);
  const transition = reduced ? 'none' : 'all var(--dur-base) var(--ease-out)';

  return (
    <div
      role="dialog"
      aria-modal="true"
      aria-label="Product tour"
      style={{
        position: 'fixed',
        inset: 0,
        zIndex: 'var(--z-overlay)' as unknown as number,
      }}
    >
      {/* Scrim — with a cut-out over the target via a giant box-shadow ring.
          No blur, no glow: an opaque dark scrim with a hairline-outlined hole. */}
      {rect ? (
        <div
          aria-hidden
          style={{
            position: 'fixed',
            top: rect.top - PAD,
            left: rect.left - PAD,
            width: rect.width + PAD * 2,
            height: rect.height + PAD * 2,
            borderRadius: 'var(--r-md)',
            boxShadow: '0 0 0 9999px var(--bg-scrim)',
            outline: '2px solid var(--accent)',
            outlineOffset: 0,
            transition,
            pointerEvents: 'none',
          }}
        />
      ) : (
        <div
          aria-hidden
          onClick={close}
          style={{ position: 'fixed', inset: 0, background: 'var(--bg-scrim)' }}
        />
      )}

      {/* Click-catcher so clicking the scrim (outside the card) dismisses. */}
      {rect && (
        <div aria-hidden onClick={close} style={{ position: 'fixed', inset: 0 }} />
      )}

      {/* The step card. */}
      <div
        ref={cardRef}
        tabIndex={-1}
        style={{
          position: 'fixed',
          top: card.top,
          left: card.left,
          transform: card.transform,
          width: 340,
          maxWidth: 'calc(100vw - 24px)',
          background: 'var(--bg-overlay)',
          border: '1px solid var(--border-strong)',
          borderRadius: 'var(--r-lg)',
          boxShadow: 'var(--shadow-lg)',
          padding: 'var(--s-5)',
          outline: 'none',
          transition,
          animation: reduced ? 'none' : 'fade-in-up var(--dur-base) var(--ease-out)',
        }}
      >
        <div
          style={{
            display: 'flex',
            alignItems: 'flex-start',
            justifyContent: 'space-between',
            gap: 'var(--s-4)',
            marginBottom: 'var(--s-3)',
          }}
        >
          <div style={{ minWidth: 0 }}>
            <div
              className="mono"
              style={{
                fontSize: 'var(--fs-11)',
                color: 'var(--text-tertiary)',
                marginBottom: 'var(--s-1)',
              }}
            >
              {String(index + 1).padStart(2, '0')} / {String(steps.length).padStart(2, '0')}
            </div>
            <h2
              style={{
                fontSize: 'var(--fs-18)',
                fontWeight: 'var(--fw-semibold)' as unknown as number,
                color: 'var(--text-primary)',
                letterSpacing: 'var(--tracking-tight)',
              }}
            >
              {step.title}
            </h2>
          </div>
          <button
            type="button"
            onClick={close}
            aria-label="Close tour"
            style={{
              display: 'inline-flex',
              alignItems: 'center',
              justifyContent: 'center',
              width: 28,
              height: 28,
              borderRadius: 'var(--r-sm)',
              color: 'var(--text-tertiary)',
              flexShrink: 0,
              transition: 'background var(--dur-fast) var(--ease-standard)',
            }}
            onMouseEnter={(e) => {
              e.currentTarget.style.background = 'var(--bg-hover)';
            }}
            onMouseLeave={(e) => {
              e.currentTarget.style.background = 'transparent';
            }}
          >
            <X size={16} strokeWidth={1.9} />
          </button>
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

        {/* Progress dots + controls. */}
        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            gap: 'var(--s-4)',
            marginTop: 'var(--s-5)',
          }}
        >
          <div aria-hidden style={{ display: 'flex', gap: 6, alignItems: 'center' }}>
            {steps.map((_, i) => (
              <span
                key={i}
                style={{
                  width: i === index ? 16 : 6,
                  height: 6,
                  borderRadius: 'var(--r-pill)',
                  background: i === index ? 'var(--accent)' : 'var(--border-strong)',
                  transition: reduced
                    ? 'none'
                    : 'width var(--dur-fast) var(--ease-standard), background var(--dur-fast)',
                }}
              />
            ))}
          </div>

          <div style={{ display: 'flex', gap: 'var(--s-3)', alignItems: 'center' }}>
            {!isFirst && (
              <button
                type="button"
                onClick={prev}
                style={{
                  display: 'inline-flex',
                  alignItems: 'center',
                  gap: 'var(--s-2)',
                  padding: '6px 10px',
                  borderRadius: 'var(--r-sm)',
                  fontSize: 'var(--fs-13)',
                  fontWeight: 'var(--fw-medium)' as unknown as number,
                  color: 'var(--text-secondary)',
                  border: '1px solid var(--border-subtle)',
                  transition: 'background var(--dur-fast) var(--ease-standard)',
                }}
                onMouseEnter={(e) => {
                  e.currentTarget.style.background = 'var(--bg-hover)';
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.background = 'transparent';
                }}
              >
                <ArrowLeft size={14} strokeWidth={1.9} aria-hidden />
                Back
              </button>
            )}
            <button
              type="button"
              onClick={next}
              style={{
                display: 'inline-flex',
                alignItems: 'center',
                gap: 'var(--s-2)',
                padding: '6px 12px',
                borderRadius: 'var(--r-sm)',
                fontSize: 'var(--fs-13)',
                fontWeight: 'var(--fw-semibold)' as unknown as number,
                color: 'var(--text-inverse)',
                background: 'var(--accent)',
                transition: 'background var(--dur-fast) var(--ease-standard)',
              }}
              onMouseEnter={(e) => {
                e.currentTarget.style.background = 'var(--accent-hover)';
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.background = 'var(--accent)';
              }}
            >
              {isLast ? 'Done' : 'Next'}
              {!isLast && <ArrowRight size={14} strokeWidth={1.9} aria-hidden />}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

/**
 * computeCardPosition — place the card near the spotlighted target (respecting
 * placement) or centered when there is no target. Clamps to the viewport.
 */
function computeCardPosition(
  rect: Rect | null,
  placement: TourStep['placement'] = 'bottom',
): { top: number; left: number; transform: string } {
  const CARD_W = 340;
  const vw = typeof window !== 'undefined' ? window.innerWidth : 1200;
  const vh = typeof window !== 'undefined' ? window.innerHeight : 800;

  if (!rect) {
    return { top: vh / 2, left: vw / 2, transform: 'translate(-50%, -50%)' };
  }

  const cx = rect.left + rect.width / 2;

  if (placement === 'top') {
    const left = clamp(cx - CARD_W / 2, 12, vw - CARD_W - 12);
    return { top: rect.top - GAP, left, transform: 'translateY(-100%)' };
  }
  if (placement === 'right') {
    return {
      top: rect.top + rect.height / 2,
      left: Math.min(rect.left + rect.width + GAP, vw - CARD_W - 12),
      transform: 'translateY(-50%)',
    };
  }
  if (placement === 'left') {
    return {
      top: rect.top + rect.height / 2,
      left: Math.max(rect.left - GAP - CARD_W, 12),
      transform: 'translateY(-50%)',
    };
  }
  // bottom (default)
  const left = clamp(cx - CARD_W / 2, 12, vw - CARD_W - 12);
  return { top: rect.top + rect.height + GAP, left, transform: 'none' };
}

function clamp(v: number, lo: number, hi: number): number {
  return Math.max(lo, Math.min(hi, v));
}
