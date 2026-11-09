import type { ReactNode } from 'react';

export interface Column<Row> {
  /** stable key; also used to read the cell if no `cell` is given */
  key: string;
  /** header label */
  header: ReactNode;
  /** render the cell; defaults to Row[key] as string */
  cell?: (row: Row, index: number) => ReactNode;
  /** monospace + right-align: use for numbers, IDs, coordinates */
  mono?: boolean;
  /** explicit alignment (mono defaults to 'right') */
  align?: 'left' | 'right' | 'center';
  /** fixed / min width, e.g. "120px" or "20%" */
  width?: string;
  /** optional header tooltip text (rendered as title attribute) */
  title?: string;
}

/**
 * DataTable — a dense, institutional data table (gnomAD / cBioPortal idiom):
 * hairline row separators, sticky header, optional zebra striping, mono numeric
 * columns with tabular figures, and hover row highlight. Presentational only —
 * sorting/filtering are owned by the caller (pass already-ordered `rows`).
 *
 * Usage
 *  <DataTable
 *    rows={sites}
 *    getRowKey={(r) => r.site}
 *    columns={[
 *      { key: 'site', header: 'NEON site', mono: true, align: 'left' },
 *      { key: 'lead', header: 'Lead (days)', mono: true,
 *        cell: (r) => r.lead.toFixed(1) },
 *    ]}
 *  />
 */
export function DataTable<Row>({
  columns,
  rows,
  getRowKey,
  zebra = true,
  compact = false,
  emptyMessage = 'No data.',
  caption,
  onRowClick,
}: {
  columns: Column<Row>[];
  rows: Row[];
  getRowKey: (row: Row, index: number) => string;
  zebra?: boolean;
  compact?: boolean;
  emptyMessage?: ReactNode;
  caption?: ReactNode;
  onRowClick?: (row: Row, index: number) => void;
}) {
  const padY = compact ? 'var(--s-2)' : 'var(--s-3)';
  const padX = 'var(--s-4)';

  return (
    <div
      style={{
        border: '1px solid var(--border-subtle)',
        borderRadius: 'var(--r-md)',
        overflow: 'hidden',
        background: 'var(--bg-surface)',
      }}
    >
      <div style={{ overflowX: 'auto' }}>
        <table
          style={{
            width: '100%',
            borderCollapse: 'collapse',
            fontSize: 'var(--fs-13)',
          }}
        >
          {caption && (
            <caption
              style={{
                captionSide: 'top',
                textAlign: 'left',
                padding: `${padY} ${padX}`,
                fontSize: 'var(--fs-12)',
                color: 'var(--text-tertiary)',
                borderBottom: '1px solid var(--border-subtle)',
              }}
            >
              {caption}
            </caption>
          )}
          <thead>
            <tr>
              {columns.map((c) => {
                const align = c.align ?? (c.mono ? 'right' : 'left');
                return (
                  <th
                    key={c.key}
                    title={c.title}
                    style={{
                      position: 'sticky',
                      top: 0,
                      zIndex: 1,
                      textAlign: align,
                      padding: `${padY} ${padX}`,
                      width: c.width,
                      whiteSpace: 'nowrap',
                      background: 'var(--bg-subtle)',
                      borderBottom: '1px solid var(--border-strong)',
                      fontFamily: 'var(--font-sans)',
                      fontSize: 'var(--fs-11)',
                      fontWeight: 600,
                      letterSpacing: 'var(--tracking-wide)',
                      textTransform: 'uppercase',
                      color: 'var(--text-tertiary)',
                    }}
                  >
                    {c.header}
                  </th>
                );
              })}
            </tr>
          </thead>
          <tbody>
            {rows.length === 0 ? (
              <tr>
                <td
                  colSpan={columns.length}
                  style={{
                    padding: 'var(--s-7)',
                    textAlign: 'center',
                    color: 'var(--text-tertiary)',
                    fontSize: 'var(--fs-13)',
                  }}
                >
                  {emptyMessage}
                </td>
              </tr>
            ) : (
              rows.map((row, i) => (
                <tr
                  key={getRowKey(row, i)}
                  onClick={onRowClick ? () => onRowClick(row, i) : undefined}
                  style={{
                    background:
                      zebra && i % 2 === 1 ? 'var(--bg-subtle)' : 'var(--bg-surface)',
                    cursor: onRowClick ? 'pointer' : 'default',
                    transition: 'background var(--dur-fast) var(--ease-standard)',
                  }}
                  onMouseEnter={(e) => {
                    e.currentTarget.style.background = 'var(--bg-hover)';
                  }}
                  onMouseLeave={(e) => {
                    e.currentTarget.style.background =
                      zebra && i % 2 === 1 ? 'var(--bg-subtle)' : 'var(--bg-surface)';
                  }}
                >
                  {columns.map((c) => {
                    const align = c.align ?? (c.mono ? 'right' : 'left');
                    const content = c.cell
                      ? c.cell(row, i)
                      : ((row as Record<string, unknown>)[c.key] as ReactNode);
                    return (
                      <td
                        key={c.key}
                        className={c.mono ? 'mono' : undefined}
                        style={{
                          textAlign: align,
                          padding: `${padY} ${padX}`,
                          borderBottom:
                            i === rows.length - 1
                              ? 'none'
                              : '1px solid var(--border-subtle)',
                          color: 'var(--text-primary)',
                          whiteSpace: c.mono ? 'nowrap' : 'normal',
                          fontVariantNumeric: c.mono ? 'tabular-nums' : undefined,
                        }}
                      >
                        {content}
                      </td>
                    );
                  })}
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}
