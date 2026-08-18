import type { ReactNode } from 'react';

/**
 * PageShell — the routed content column shared by all non-map pages. Centers a
 * max-width (~1200px) column with generous top/bottom rhythm. Page authors put
 * their SectionHeaders + cards + tables inside.
 */
export function PageShell({ children }: { children: ReactNode }) {
  return (
    <main
      id="main-content"
      tabIndex={-1}
      style={{
        maxWidth: 'var(--content-max)',
        margin: '0 auto',
        padding: 'var(--s-8) var(--s-6) var(--s-12)',
        outline: 'none',
      }}
    >
      {children}
    </main>
  );
}
