import { useReducer } from 'react';
import type { CanonicalRecord } from './types';

export interface Layers {
  usgs: boolean;
  neon: boolean;
  satellite: boolean;   // GIBS MODIS true-color overlay (off → clean basemap)
  chlorophyll: boolean; // GIBS MODIS_Aqua_Chlorophyll_A overlay
  drought: boolean;     // USDM drought monitor overlay
}

export interface AppState {
  selectedRecord: CanonicalRecord | null;
  cascadeTier: 0 | 1 | 2 | 3;
  layers: Layers;
  demoMode: boolean;
  fusionDetailOpen: boolean;
  legendOpen: boolean;
  methodologyOpen: boolean;
  totalCostK: number;
  // ISO date string (YYYY-MM-DD) drives GIBS tile date and hero image
  currentDate: string;
  // True while fetchLiveSensor is in flight; PlaceCard shows a shimmer over its hero metric.
  loadingLive: boolean;
  // Optional pinned record for compare mode — rendered as a mini-card; click to swap with selectedRecord.
  pinnedRecord: CanonicalRecord | null;
  // Persisted UI prefs.
  colorBlind: boolean;
  // "Play story" mode: the EventTimeline panel + TimeSlider drive currentDate.
  playingStory: boolean;
  playStorySpeed: 1 | 4 | 16;
  // Track which milestones have already toasted (reset on new selection / story start).
  firedMilestones: string[];
}

export type Action =
  | { type: 'SELECT_SITE'; record: CanonicalRecord }
  | { type: 'CLOSE_RAIL' }
  | { type: 'TOGGLE_LAYER'; layer: keyof Layers }
  | { type: 'SET_CASCADE_TIER'; tier: 0 | 1 | 2 | 3 }
  | { type: 'ESCALATE' }
  | { type: 'DEESCALATE' }
  | { type: 'TOGGLE_DEMO' }
  | { type: 'TOGGLE_FUSION_DETAIL' }
  | { type: 'TOGGLE_LEGEND' }
  | { type: 'TOGGLE_METHODOLOGY' }
  | { type: 'PATCH_RECORD'; patch: Partial<CanonicalRecord> }
  | { type: 'SET_DATE'; date: string }
  | { type: 'LIVE_FETCH_BEGIN' }
  | { type: 'LIVE_FETCH_END' }
  | { type: 'PIN_RECORD'; record: CanonicalRecord | null }
  | { type: 'SWAP_PINNED' }
  | { type: 'TOGGLE_COLORBLIND' }
  | { type: 'START_STORY'; from?: string }
  | { type: 'STOP_STORY' }
  | { type: 'SET_STORY_SPEED'; speed: 1 | 4 | 16 }
  | { type: 'MARK_MILESTONE_FIRED'; key: string }
  | { type: 'RESET_STORY_FIRES' };

const TIER_COSTS = [0, 0.5, 15, 25];

// Default to a recent date with good MODIS coverage.
const DEFAULT_DATE = '2024-07-15';

export const initialState: AppState = {
  selectedRecord: null,
  cascadeTier: 0,
  layers: { usgs: true, neon: true, satellite: false, chlorophyll: false, drought: false },
  demoMode: false,
  fusionDetailOpen: true,
  legendOpen: false,
  methodologyOpen: false,
  totalCostK: 0,
  currentDate: DEFAULT_DATE,
  loadingLive: false,
  pinnedRecord: null,
  colorBlind: typeof localStorage !== 'undefined' && localStorage.getItem('sentinel.cb') === '1',
  playingStory: false,
  playStorySpeed: 1,
  firedMilestones: [],
};

function reducer(state: AppState, action: Action): AppState {
  switch (action.type) {
    case 'SELECT_SITE': {
      // When selecting a bookmarked event, jump the global date to its advisory_date.
      const advisory = action.record.advisoryDate;
      return {
        ...state,
        selectedRecord: action.record,
        cascadeTier: 0,
        totalCostK: 0,
        currentDate: advisory ?? state.currentDate,
        playingStory: false,
        firedMilestones: [],
      };
    }
    case 'CLOSE_RAIL':
      return {
        ...state,
        selectedRecord: null,
        cascadeTier: 0,
        totalCostK: 0,
        playingStory: false,
        firedMilestones: [],
      };
    case 'TOGGLE_LAYER':
      return { ...state, layers: { ...state.layers, [action.layer]: !state.layers[action.layer] } };
    case 'SET_CASCADE_TIER':
      return { ...state, cascadeTier: action.tier };
    case 'ESCALATE': {
      if (state.cascadeTier >= 3) return state;
      const next = (state.cascadeTier + 1) as 0 | 1 | 2 | 3;
      return { ...state, cascadeTier: next, totalCostK: state.totalCostK + TIER_COSTS[next] };
    }
    case 'DEESCALATE': {
      if (state.cascadeTier <= 0) return state;
      const next = (state.cascadeTier - 1) as 0 | 1 | 2 | 3;
      return { ...state, cascadeTier: next, totalCostK: Math.max(0, state.totalCostK - TIER_COSTS[state.cascadeTier]) };
    }
    case 'TOGGLE_DEMO':
      return { ...state, demoMode: !state.demoMode };
    case 'TOGGLE_FUSION_DETAIL':
      return { ...state, fusionDetailOpen: !state.fusionDetailOpen };
    case 'TOGGLE_LEGEND':
      return { ...state, legendOpen: !state.legendOpen };
    case 'TOGGLE_METHODOLOGY':
      return { ...state, methodologyOpen: !state.methodologyOpen };
    case 'PATCH_RECORD':
      if (!state.selectedRecord) return state;
      return { ...state, selectedRecord: { ...state.selectedRecord, ...action.patch } };
    case 'SET_DATE':
      return { ...state, currentDate: action.date };
    case 'LIVE_FETCH_BEGIN':
      return { ...state, loadingLive: true };
    case 'LIVE_FETCH_END':
      return { ...state, loadingLive: false };
    case 'PIN_RECORD':
      return { ...state, pinnedRecord: action.record };
    case 'SWAP_PINNED': {
      // Swap pinned ↔ selected. Reset tier when swapping in a different record.
      if (!state.pinnedRecord) return state;
      const wasSelected = state.selectedRecord;
      return {
        ...state,
        selectedRecord: state.pinnedRecord,
        pinnedRecord: wasSelected,
        cascadeTier: 0,
        totalCostK: 0,
      };
    }
    case 'TOGGLE_COLORBLIND': {
      const next = !state.colorBlind;
      try { localStorage.setItem('sentinel.cb', next ? '1' : '0'); } catch {
        /* localStorage unavailable (private mode / SSR); the preference just won't persist */
      }
      return { ...state, colorBlind: next };
    }
    case 'START_STORY':
      return {
        ...state,
        playingStory: true,
        firedMilestones: [],
        currentDate: action.from ?? state.currentDate,
      };
    case 'STOP_STORY':
      return { ...state, playingStory: false };
    case 'SET_STORY_SPEED':
      return { ...state, playStorySpeed: action.speed };
    case 'MARK_MILESTONE_FIRED':
      if (state.firedMilestones.includes(action.key)) return state;
      return { ...state, firedMilestones: [...state.firedMilestones, action.key] };
    case 'RESET_STORY_FIRES':
      return { ...state, firedMilestones: [] };
    // Selecting a different site resets story state so a fresh "Play story" works.
    default:
      return state;
  }
}

export function useAppState() {
  return useReducer(reducer, initialState);
}
