import type { ReactNode } from 'react';

/* ==========================================================================
   tourSteps — data + types for the GuidedTour, kept out of the component file
   so the component module only exports a component (fast-refresh friendly).
   ========================================================================== */

export interface TourStep {
  /** CSS selector for the element to spotlight. Omit for a centered step. */
  target?: string;
  /** Short step title. */
  title: string;
  /** One or two sentences of body copy. */
  body: ReactNode;
  /** Preferred placement of the card relative to the target. */
  placement?: 'bottom' | 'top' | 'right' | 'left';
}

/** The default walk: nav orientation → the honesty framing → where to dig in. */
export const DEFAULT_TOUR: TourStep[] = [
  {
    title: 'Welcome to SENTINEL',
    body: (
      <>
        A multimodal water-quality early-warning system. This 30-second tour
        shows you how the site is laid out and, more importantly, how to read
        the numbers — every figure here is real and measured under a stated
        holdout protocol.
      </>
    ),
  },
  {
    target: 'a[href="/models"]',
    title: 'Models',
    body: (
      <>
        The five modality encoders (sensor, satellite, microbial, molecular,
        behavioral) plus the downstream and deployment models. Each carries its
        headline metric, holdout protocol, and an honest note.
      </>
    ),
    placement: 'bottom',
  },
  {
    target: 'a[href="/fusion"]',
    title: 'Fusion — the headline story',
    body: (
      <>
        The nuanced result: on microcystin, 3-modality fusion beats every single
        modality within a region (0.82 vs 0.68), then collapses to chance across
        unseen watersheds. We present the win and the boundary.
      </>
    ),
    placement: 'bottom',
  },
  {
    target: 'a[href="/data"]',
    title: 'Data',
    body: (
      <>
        Real public sources only — USGS, NEON, ESA Sentinel-2, EMP 16S, NCBI
        GEO, EPA ECOTOX, and more. Raw and post-QC record counts are both shown.
      </>
    ),
    placement: 'bottom',
  },
  {
    target: 'a[href="/docs"]',
    title: 'Docs',
    body: (
      <>
        How the pipeline works, the exact commands to reproduce it, the holdout
        protocols, a glossary, and the full retraction log from our integrity
        remediation. The honesty is documented, not hidden.
      </>
    ),
    placement: 'bottom',
  },
  {
    target: '[data-tour="metric-tooltip"]',
    title: 'Every number is explained',
    body: (
      <>
        Hover the small info icon on any metric to see what it measures and the
        exact holdout protocol behind it. Use the copy button to cite the value.
      </>
    ),
    placement: 'bottom',
  },
  {
    title: "You're set",
    body: (
      <>
        Start with Models, then read the Fusion story. If a number ever looks
        surprising, its tooltip and the Docs page explain precisely how it was
        measured.
      </>
    ),
  },
];
