import { NavLink } from 'react-router-dom';
import { Activity } from 'lucide-react';

/**
 * NavBar — top navigation bar (gnomAD idiom): wordmark on the left, section
 * links with a subtle active underline, and a live-status affordance on the
 * right. Flat surface, single hairline bottom border, no blur, no shadow glow.
 */
const LINKS: { to: string; label: string; end?: boolean }[] = [
  { to: '/', label: 'Overview', end: true },
  { to: '/models', label: 'Models' },
  { to: '/fusion', label: 'Fusion' },
  { to: '/data', label: 'Data' },
  { to: '/docs', label: 'Docs' },
  { to: '/live', label: 'Live map' },
];

export function NavBar() {
  return (
    <header
      style={{
        height: 'var(--nav-height)',
        background: 'var(--bg-surface)',
        borderBottom: '1px solid var(--border-subtle)',
        position: 'sticky',
        top: 0,
        zIndex: 'var(--z-nav)' as unknown as number,
        flexShrink: 0,
      }}
    >
      <nav
        aria-label="Primary"
        style={{
          maxWidth: 'var(--content-max)',
          height: '100%',
          margin: '0 auto',
          padding: '0 var(--s-6)',
          display: 'flex',
          alignItems: 'center',
          gap: 'var(--s-7)',
        }}
      >
        <NavLink
          to="/"
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: 'var(--s-3)',
            textDecoration: 'none',
            flexShrink: 0,
          }}
        >
          <Activity size={17} strokeWidth={2} color="var(--accent)" aria-hidden />
          <span
            style={{
              fontFamily: 'var(--font-sans)',
              fontWeight: 700,
              fontSize: 'var(--fs-16)',
              letterSpacing: '0.02em',
              color: 'var(--text-primary)',
            }}
          >
            SENTINEL
          </span>
          <span
            className="mono"
            style={{
              fontSize: 'var(--fs-11)',
              color: 'var(--text-tertiary)',
              paddingTop: 2,
            }}
          >
            v0
          </span>
        </NavLink>

        <ul
          className="nav-scroll"
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: 'var(--s-2)',
            listStyle: 'none',
            margin: 0,
            padding: 0,
            height: '100%',
            // On narrow screens the links scroll horizontally instead of
            // overflowing off-canvas, so every section stays reachable.
            overflowX: 'auto',
            overflowY: 'hidden',
            scrollbarWidth: 'none',
            WebkitOverflowScrolling: 'touch',
          }}
        >
          {LINKS.map((l) => (
            <li key={l.to} style={{ height: '100%', flexShrink: 0 }}>
              <NavLink
                to={l.to}
                end={l.end}
                style={({ isActive }) => ({
                  display: 'inline-flex',
                  alignItems: 'center',
                  height: '100%',
                  padding: '0 var(--s-3)',
                  fontSize: 'var(--fs-14)',
                  fontWeight: isActive ? 600 : 500,
                  color: isActive ? 'var(--text-primary)' : 'var(--text-secondary)',
                  textDecoration: 'none',
                  boxShadow: isActive
                    ? 'inset 0 -2px 0 0 var(--accent)'
                    : 'inset 0 -2px 0 0 transparent',
                  transition:
                    'color var(--dur-fast) var(--ease-standard), box-shadow var(--dur-fast) var(--ease-standard)',
                })}
              >
                {l.label}
              </NavLink>
            </li>
          ))}
        </ul>

        <a
          href="https://github.com/austinjin1/SENTINEL-STOCKHOLM"
          target="_blank"
          rel="noreferrer"
          style={{
            marginLeft: 'auto',
            fontSize: 'var(--fs-13)',
            color: 'var(--text-tertiary)',
            textDecoration: 'none',
          }}
        >
          Source
        </a>
      </nav>
    </header>
  );
}
