import { useEffect, type ReactNode } from 'react';
import { Outlet, Route, Routes, useLocation } from 'react-router-dom';
import './tokens.css';
import { NavBar } from './components/NavBar';
import { PageShell } from './components/PageShell';
import Overview from './pages/Overview';
import Models from './pages/Models';
import Fusion from './pages/Fusion';
import Data from './pages/Data';
import Docs from './pages/Docs';
import LiveMap from './pages/LiveMap';

/**
 * RouteTransition — scrolls to the top on navigation and re-triggers a subtle
 * fade-in-up on the routed content whenever the path changes. Keying the wrapper
 * by pathname remounts it, so the CSS keyframe replays. prefers-reduced-motion
 * is honored globally in tokens.css (the animation collapses to ~0ms there), so
 * no motion runs for users who opt out.
 */
function RouteTransition({ children }: { children: ReactNode }) {
  const { pathname } = useLocation();
  useEffect(() => {
    window.scrollTo({ top: 0, left: 0, behavior: 'auto' });
  }, [pathname]);
  return (
    <div
      key={pathname}
      style={{ animation: 'fade-in-up var(--dur-base) var(--ease-out)' }}
    >
      {children}
    </div>
  );
}

/**
 * Layout for the content pages: sticky top nav + a centered, scrolling
 * max-width column. The /live map view uses its own full-viewport layout.
 */
function ContentLayout() {
  return (
    <>
      <a href="#main-content" className="skip-link">
        Skip to main content
      </a>
      <NavBar />
      <PageShell>
        <RouteTransition>
          <Outlet />
        </RouteTransition>
      </PageShell>
    </>
  );
}

/**
 * The /live map dashboard: sticky top nav above a fixed, non-scrolling map
 * viewport. LiveMap owns its own full-viewport layout.
 */
function LivePage() {
  return (
    <>
      <NavBar />
      <LiveMap />
    </>
  );
}

function NotFound() {
  return (
    <PageShell>
      <div style={{ padding: 'var(--s-10) 0', textAlign: 'center' }}>
        <div className="metric-xl" style={{ color: 'var(--text-tertiary)' }}>404</div>
        <p style={{ color: 'var(--text-secondary)' }}>No such page.</p>
        <a href="/">Return to overview</a>
      </div>
    </PageShell>
  );
}

export default function App() {
  return (
    <Routes>
      <Route element={<ContentLayout />}>
        <Route path="/" element={<Overview />} />
        <Route path="/models" element={<Models />} />
        <Route path="/fusion" element={<Fusion />} />
        <Route path="/data" element={<Data />} />
        <Route path="/docs" element={<Docs />} />
        <Route path="*" element={<NotFound />} />
      </Route>
      <Route path="/live" element={<LivePage />} />
    </Routes>
  );
}
