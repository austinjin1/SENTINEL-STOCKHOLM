import { useEffect, useRef } from 'react';
import type { AppState, Action, Layers } from './state';
import { EVENTS, CLEAN_SITES } from './data/realData';

const ALL_SITES = [...EVENTS, ...CLEAN_SITES];

// Compact 2-char codes for each layer key to keep URLs short.
const LAYER_CODES: Record<keyof Layers, string> = {
  usgs: 'us',
  neon: 'ne',
  satellite: 'sa',
  chlorophyll: 'ch',
  drought: 'dr',
};
const CODE_TO_LAYER = Object.fromEntries(
  (Object.entries(LAYER_CODES) as [keyof Layers, string][]).map(([k, v]) => [v, k]),
) as Record<string, keyof Layers>;

function encodeLayers(layers: Layers): string {
  return (Object.entries(layers) as [keyof Layers, boolean][])
    .filter(([, on]) => on)
    .map(([k]) => LAYER_CODES[k])
    .join(',');
}

function applyLayersFromUrl(raw: string, dispatch: React.Dispatch<Action>, current: Layers) {
  const want: Partial<Record<keyof Layers, boolean>> = {};
  for (const code of raw.split(',').filter(Boolean)) {
    const k = CODE_TO_LAYER[code];
    if (k) want[k] = true;
  }
  (Object.keys(LAYER_CODES) as (keyof Layers)[]).forEach((k) => {
    const target = !!want[k];
    if (target !== current[k]) dispatch({ type: 'TOGGLE_LAYER', layer: k });
  });
}

export function readInitialUrlState(dispatch: React.Dispatch<Action>, state: AppState) {
  const params = new URLSearchParams(window.location.search);
  const eventKey = params.get('site');
  const tier = params.get('tier');
  const date = params.get('date');
  const layers = params.get('layers');

  if (eventKey) {
    const rec = ALL_SITES.find((s) => s.eventKey === eventKey || s.id === eventKey);
    if (rec) {
      dispatch({ type: 'SELECT_SITE', record: rec });
      if (tier) {
        const t = Math.max(0, Math.min(3, parseInt(tier, 10))) as 0 | 1 | 2 | 3;
        if (!Number.isNaN(t) && t > 0) {
          // Step up one tier at a time so the activation animation can run if desired.
          for (let i = 0; i < t; i++) dispatch({ type: 'ESCALATE' });
        }
      }
    }
  }
  if (date && /^\d{4}-\d{2}-\d{2}$/.test(date)) {
    dispatch({ type: 'SET_DATE', date });
  }
  if (layers !== null) {
    applyLayersFromUrl(layers, dispatch, state.layers);
  }
}

export function useUrlSync(state: AppState) {
  const first = useRef(true);
  useEffect(() => {
    if (first.current) {
      first.current = false;
      return;
    }
    const params = new URLSearchParams();
    if (state.selectedRecord) {
      params.set('site', state.selectedRecord.eventKey ?? state.selectedRecord.id ?? '');
      if (state.cascadeTier > 0) params.set('tier', String(state.cascadeTier));
    }
    if (state.currentDate) params.set('date', state.currentDate);
    const layerCode = encodeLayers(state.layers);
    if (layerCode) params.set('layers', layerCode);
    const qs = params.toString();
    const url = qs ? `${window.location.pathname}?${qs}` : window.location.pathname;
    window.history.replaceState(null, '', url);
  }, [state.selectedRecord, state.cascadeTier, state.currentDate, state.layers]);
}
