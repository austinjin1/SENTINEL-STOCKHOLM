import type { ReactNode } from 'react';

export type StatusKind = 'nominal' | 'watch' | 'alert' | 'neutral';

const KIND: Record<StatusKind, { fg: string; wash: string; border: string }> = {
  nominal: { fg: 'var(--nominal)', wash: 'var(--nominal-wash)', border: 'var(--nominal-border)' },
  watch: { fg: 'var(--watch)', wash: 'var(--watch-wash)', border: 'var(--watch-border)' },
  alert: { fg: 'var(--alert)', wash: 'var(--alert-wash)', border: 'var(--alert-border)' },
  neutral: {
    fg: 'var(--neutral-status)',
    wash: 'var(--neutral-status-wash)',
    border: 'var(--border-strong)',
  },
};

/**
 * StatusBadge — a compact pill that encodes DATA state (never decoration).
 * A filled dot in the status color + a short label, on a flat wash. Use in
 * tables and cards for pass/watch/fail, detected/missed, verified/simulated.
 *
 * Usage:  <StatusBadge kind="nominal">detected</StatusBadge>
 */
export function StatusBadge({
  kind = 'neutral',
  children,
  dot = true,
}: {
  kind?: StatusKind;
  children: ReactNode;
  dot?: boolean;
}) {
  const k = KIND[kind];
  return (
    <span
      style={{
        display: 'inline-flex',
        alignItems: 'center',
        gap: 'var(--s-2)',
        padding: '1px 8px',
        borderRadius: 'var(--r-pill)',
        background: k.wash,
        border: `1px solid ${k.border}`,
        color: k.fg,
        fontFamily: 'var(--font-sans)',
        fontSize: 'var(--fs-12)',
        fontWeight: 500,
        lineHeight: 1.6,
        whiteSpace: 'nowrap',
      }}
    >
      {dot && (
        <span
          aria-hidden
          style={{
            width: 6,
            height: 6,
            borderRadius: '50%',
            background: k.fg,
            flexShrink: 0,
          }}
        />
      )}
      {children}
    </span>
  );
}
