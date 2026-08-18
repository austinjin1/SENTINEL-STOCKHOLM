import { useEffect, useRef, useState, type ReactNode } from 'react';
import { Info } from 'lucide-react';
import { Tooltip } from './Tooltip';
import { CopyButton } from './CopyButton';

export type MetricStatus = 'nominal' | 'watch' | 'alert' | 'neutral';

const STATUS_COLOR: Record<MetricStatus, string> = {
  nominal: 'var(--nominal)',
  watch: 'var(--watch)',
  alert: 'var(--alert)',
  neutral: 'var(--text-primary)',
};

/**
 * Metric — the primary way to present a single verified number.
 *
 * Genomics-tool signature: the value renders in MONOSPACE with tabular figures.
 * Every metric carries a tooltip (what it is + the holdout protocol) and a
 * copy-to-cite button, so the whole surface stays honest and citable.
 *
 * Props
 *  - label:    short caption under/over the value (e.g. "Median lead time")
 *  - value:    the number/string to show big and mono (e.g. "29.3")
 *  - unit:     optional trailing unit in muted mono (e.g. "days")
 *  - tooltip:  REQUIRED-in-spirit — explains the metric + holdout protocol
 *  - status:   colors the value to encode data state (default neutral)
 *  - sub:      optional caption line under the value (e.g. "n = 3 / 6 episodes")
 *  - copyValue:string written to clipboard (defaults to `${value} ${unit}`)
 *  - size:     'lg' hero | 'md' card (default) | 'sm' inline-in-table
 *  - countUp:  animate the number up on first view (numeric values only)
 *
 * Usage
 *  <Metric label="Median lead time" value="29.3" unit="days"
 *          status="nominal" sub="3 / 6 NEON DO-hypoxia episodes"
 *          tooltip="Median forecast lead over official advisory, sensor-only,
 *                   at held-out FPR 4.6%. Holdout = leave-one-region-out." />
 */
export function Metric({
  label,
  value,
  unit,
  tooltip,
  status = 'neutral',
  sub,
  copyValue,
  size = 'md',
  countUp = false,
  align = 'left',
}: {
  label: ReactNode;
  value: string | number;
  unit?: string;
  tooltip?: ReactNode;
  status?: MetricStatus;
  sub?: ReactNode;
  copyValue?: string;
  size?: 'lg' | 'md' | 'sm';
  countUp?: boolean;
  align?: 'left' | 'center';
}) {
  const valueClass =
    size === 'lg' ? 'metric-display' : size === 'sm' ? 'data-cell' : 'metric-xl';

  const numeric = typeof value === 'number' ? value : Number(value);
  const isNumeric = countUp && Number.isFinite(numeric);
  const decimals =
    typeof value === 'string' && value.includes('.')
      ? value.split('.')[1].length
      : isNumeric && !Number.isInteger(numeric)
        ? 1
        : 0;

  const display = useCountUp(isNumeric ? numeric : null, decimals);
  const shown = display === null ? String(value) : display;
  const clip = copyValue ?? `${value}${unit ? ' ' + unit : ''}`;

  return (
    <div
      style={{
        display: 'flex',
        flexDirection: 'column',
        gap: 'var(--s-1)',
        alignItems: align === 'center' ? 'center' : 'flex-start',
      }}
    >
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: 'var(--s-2)',
        }}
      >
        <span className="eyebrow">{label}</span>
        {tooltip && (
          <Tooltip label={tooltip}>
            <Info
              size={12}
              strokeWidth={1.75}
              style={{ color: 'var(--text-tertiary)', display: 'block' }}
              role="img"
              aria-label={`What is ${typeof label === 'string' ? label : 'this metric'}?`}
            />
          </Tooltip>
        )}
      </div>

      <div style={{ display: 'flex', alignItems: 'baseline', gap: 'var(--s-2)' }}>
        <span className={valueClass} style={{ color: STATUS_COLOR[status] }}>
          {shown}
        </span>
        {unit && (
          <span
            className="mono"
            style={{
              fontSize: size === 'lg' ? 'var(--fs-16)' : 'var(--fs-13)',
              color: 'var(--text-tertiary)',
              fontWeight: 'var(--fw-regular)' as unknown as number,
            }}
          >
            {unit}
          </span>
        )}
        <span style={{ alignSelf: 'center' }}>
          <CopyButton value={clip} label={typeof label === 'string' ? label : 'value'} />
        </span>
      </div>

      {sub && (
        <span
          style={{
            fontSize: 'var(--fs-12)',
            color: 'var(--text-tertiary)',
            fontFamily: 'var(--font-sans)',
          }}
        >
          {sub}
        </span>
      )}
    </div>
  );
}

/**
 * useCountUp — animate a number from 0 to `target` on first mount, once, when
 * it scrolls into view. Respects prefers-reduced-motion (jumps to target).
 * Returns null until the target is provided (non-numeric metrics skip animation).
 */
function useCountUp(target: number | null, decimals: number): string | null {
  // Start displayed at the target so a static render is always correct; the
  // effect only *lowers* then re-raises the value via async rAF frames on the
  // first mount, so no setState is called synchronously in the effect body.
  const [val, setVal] = useState<number | null>(target);
  const done = useRef(false);

  useEffect(() => {
    if (target === null || done.current) return;
    done.current = true;
    const reduced =
      typeof window !== 'undefined' &&
      window.matchMedia?.('(prefers-reduced-motion: reduce)').matches;
    if (reduced) return; // leave value at target

    const dur = 650;
    let start = 0;
    let raf = 0;
    const tick = (now: number) => {
      if (!start) start = now;
      const t = Math.min(1, (now - start) / dur);
      const eased = 1 - Math.pow(1 - t, 3); // ease-out cubic
      setVal(t < 1 ? target * eased : target);
      if (t < 1) raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [target]);

  if (val === null) return null;
  return val.toFixed(decimals);
}
