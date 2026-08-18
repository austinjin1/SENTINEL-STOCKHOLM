import { useId, useState, type ReactNode } from 'react';

/**
 * Tooltip — hover/focus popover for explaining a metric or control.
 *
 * Accessible: the trigger is described by the tooltip via aria-describedby,
 * and the tooltip shows on both hover AND keyboard focus. Opaque surface,
 * hairline border, tight shadow — no blur, no glow.
 *
 * Usage:
 *   <Tooltip label="AUROC on the held-out region. Higher is better.">
 *     <span className="mono">0.807</span>
 *   </Tooltip>
 */
export function Tooltip({
  label,
  children,
  side = 'top',
}: {
  label: ReactNode;
  children: ReactNode;
  side?: 'top' | 'bottom';
}) {
  const [open, setOpen] = useState(false);
  const id = useId();

  return (
    <span
      style={{ position: 'relative', display: 'inline-flex' }}
      onMouseEnter={() => setOpen(true)}
      onMouseLeave={() => setOpen(false)}
      onFocus={() => setOpen(true)}
      onBlur={() => setOpen(false)}
    >
      {/* The trigger is focusable so keyboard and screen-reader users get the
          explanation too (WCAG 1.4.13 / 2.1.1); Escape dismisses it. */}
      <span
        tabIndex={0}
        aria-describedby={open ? id : undefined}
        onKeyDown={(e) => {
          if (e.key === 'Escape' && open) {
            e.stopPropagation();
            setOpen(false);
          }
        }}
        style={{ display: 'inline-flex', borderRadius: 'var(--r-xs)' }}
      >
        {children}
      </span>
      {open && (
        <span
          role="tooltip"
          id={id}
          style={{
            position: 'absolute',
            [side === 'top' ? 'bottom' : 'top']: 'calc(100% + 6px)',
            left: '50%',
            transform: 'translateX(-50%)',
            zIndex: 'var(--z-tooltip)' as unknown as number,
            width: 'max-content',
            maxWidth: 260,
            padding: '6px 9px',
            background: 'var(--text-primary)',
            color: 'var(--text-inverse)',
            fontSize: 'var(--fs-12)',
            fontFamily: 'var(--font-sans)',
            fontWeight: 400,
            lineHeight: 1.4,
            borderRadius: 'var(--r-sm)',
            boxShadow: 'var(--shadow-md)',
            pointerEvents: 'none',
            animation: 'fade-in var(--dur-fast) var(--ease-standard)',
          }}
        >
          {label}
        </span>
      )}
    </span>
  );
}
