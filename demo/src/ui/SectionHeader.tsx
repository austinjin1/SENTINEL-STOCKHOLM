import type { ReactNode } from 'react';

/**
 * SectionHeader — a titled section header with a hairline underline, in the
 * gnomAD / UCSC idiom. Optional uppercase eyebrow, description, and a right-
 * aligned actions slot (e.g. a filter, a "copy citation" button, a link).
 *
 * Usage
 *  <SectionHeader
 *    eyebrow="Track A · Lead time"
 *    title="Early-warning performance"
 *    description="Sensor-only detection under leave-one-region-out holdout."
 *    actions={<a href="/docs">Holdout protocol</a>}
 *  />
 */
export function SectionHeader({
  title,
  eyebrow,
  description,
  actions,
  as: Tag = 'h2',
  id,
}: {
  title: ReactNode;
  eyebrow?: ReactNode;
  description?: ReactNode;
  actions?: ReactNode;
  as?: 'h1' | 'h2' | 'h3';
  id?: string;
}) {
  const titleSize =
    Tag === 'h1' ? 'var(--fs-28)' : Tag === 'h3' ? 'var(--fs-18)' : 'var(--fs-22)';

  return (
    <div
      style={{
        borderBottom: '1px solid var(--border-subtle)',
        paddingBottom: 'var(--s-4)',
        marginBottom: 'var(--s-6)',
      }}
    >
      <div
        style={{
          display: 'flex',
          alignItems: 'flex-end',
          justifyContent: 'space-between',
          gap: 'var(--s-5)',
          flexWrap: 'wrap',
        }}
      >
        <div style={{ minWidth: 0 }}>
          {eyebrow && (
            <div className="eyebrow" style={{ marginBottom: 'var(--s-2)' }}>
              {eyebrow}
            </div>
          )}
          <Tag
            id={id}
            style={{
              fontSize: titleSize,
              fontWeight: 'var(--fw-semibold)' as unknown as number,
              letterSpacing: 'var(--tracking-tight)',
              color: 'var(--text-primary)',
              lineHeight: 'var(--lh-snug)',
            }}
          >
            {title}
          </Tag>
        </div>
        {actions && (
          <div
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: 'var(--s-3)',
              fontSize: 'var(--fs-13)',
              flexShrink: 0,
            }}
          >
            {actions}
          </div>
        )}
      </div>
      {description && (
        <p
          style={{
            margin: 'var(--s-2) 0 0',
            maxWidth: '68ch',
            fontSize: 'var(--fs-14)',
            lineHeight: 'var(--lh-normal)',
            color: 'var(--text-secondary)',
          }}
        >
          {description}
        </p>
      )}
    </div>
  );
}
