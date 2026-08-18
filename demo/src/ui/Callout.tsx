import type { ReactNode } from 'react';
import { AlertTriangle, CheckCircle2, Info, ShieldAlert } from 'lucide-react';

export type CalloutTone = 'info' | 'verified' | 'limitation' | 'alert';

const TONE: Record<
  CalloutTone,
  { fg: string; wash: string; border: string; Icon: typeof Info; label: string }
> = {
  info: {
    fg: 'var(--accent-strong)',
    wash: 'var(--accent-wash)',
    border: 'var(--accent-border)',
    Icon: Info,
    label: 'Note',
  },
  verified: {
    fg: 'var(--nominal)',
    wash: 'var(--nominal-wash)',
    border: 'var(--nominal-border)',
    Icon: CheckCircle2,
    label: 'Verified',
  },
  limitation: {
    fg: 'var(--watch)',
    wash: 'var(--watch-wash)',
    border: 'var(--watch-border)',
    Icon: AlertTriangle,
    label: 'Limitation',
  },
  alert: {
    fg: 'var(--alert)',
    wash: 'var(--alert-wash)',
    border: 'var(--alert-border)',
    Icon: ShieldAlert,
    label: 'Caution',
  },
};

/**
 * Callout — a bordered notice used to state facts honestly: verified results,
 * limitations (holdout collapse, what's simulation), or cautions. Flat wash
 * fill + hairline border in the semantic status color; a line icon; a bold
 * label. This is how the site foregrounds its integrity story.
 *
 * Usage
 *  <Callout tone="limitation" title="Cross-region generalization collapses">
 *    A model trained on one region does not transfer: held-out-region AUROC
 *    falls to chance. All headline numbers are within-protocol only.
 *  </Callout>
 */
export function Callout({
  tone = 'info',
  title,
  children,
  label,
}: {
  tone?: CalloutTone;
  title?: ReactNode;
  children?: ReactNode;
  label?: string;
}) {
  const t = TONE[tone];
  const Icon = t.Icon;
  return (
    <div
      role="note"
      style={{
        display: 'flex',
        gap: 'var(--s-3)',
        padding: 'var(--s-4) var(--s-5)',
        background: t.wash,
        border: `1px solid ${t.border}`,
        borderRadius: 'var(--r-md)',
      }}
    >
      <Icon
        size={16}
        strokeWidth={1.9}
        style={{ color: t.fg, flexShrink: 0, marginTop: 2 }}
        aria-hidden
      />
      <div style={{ minWidth: 0 }}>
        <div
          style={{
            fontSize: 'var(--fs-13)',
            fontWeight: 'var(--fw-semibold)' as unknown as number,
            color: t.fg,
            marginBottom: title || children ? 'var(--s-1)' : 0,
          }}
        >
          {label ?? t.label}
          {title ? <span style={{ color: 'var(--text-primary)' }}> — {title}</span> : null}
        </div>
        {children && (
          <div
            style={{
              fontSize: 'var(--fs-13)',
              lineHeight: 'var(--lh-normal)',
              color: 'var(--text-secondary)',
            }}
          >
            {children}
          </div>
        )}
      </div>
    </div>
  );
}
