import { useState } from 'react';
import { Check, Copy } from 'lucide-react';

/**
 * CopyButton — copy-to-cite affordance for a value.
 *
 * Shows a small line-icon that swaps to a check for ~1.2s after copying.
 * `value` is what gets written to the clipboard (defaults to the visible text).
 *
 * Usage:
 *   <CopyButton value="29.3 days" label="lead time" />
 */
export function CopyButton({
  value,
  label,
  size = 13,
}: {
  value: string;
  label?: string;
  size?: number;
}) {
  const [copied, setCopied] = useState(false);

  async function copy() {
    try {
      await navigator.clipboard.writeText(value);
    } catch {
      // clipboard may be unavailable (e.g. insecure context) — fail quietly
    }
    setCopied(true);
    window.setTimeout(() => setCopied(false), 1200);
  }

  return (
    <button
      type="button"
      onClick={copy}
      aria-label={copied ? 'Copied' : `Copy ${label ?? 'value'} to cite`}
      title={copied ? 'Copied' : `Copy ${label ?? 'value'}`}
      style={{
        display: 'inline-flex',
        alignItems: 'center',
        justifyContent: 'center',
        width: size + 8,
        height: size + 8,
        borderRadius: 'var(--r-xs)',
        color: copied ? 'var(--nominal)' : 'var(--text-quaternary)',
        transition: 'color var(--dur-fast) var(--ease-standard), background var(--dur-fast)',
        verticalAlign: 'middle',
      }}
      onMouseEnter={(e) => {
        if (!copied) e.currentTarget.style.color = 'var(--text-secondary)';
        e.currentTarget.style.background = 'var(--bg-hover)';
      }}
      onMouseLeave={(e) => {
        if (!copied) e.currentTarget.style.color = 'var(--text-quaternary)';
        e.currentTarget.style.background = 'transparent';
      }}
    >
      {copied ? (
        <Check size={size} strokeWidth={2} />
      ) : (
        <Copy size={size} strokeWidth={1.75} />
      )}
    </button>
  );
}
