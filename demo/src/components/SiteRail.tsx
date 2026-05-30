import type { CanonicalRecord } from '../types';
import type { AppState, Action } from '../state';
import { Header } from './Header';
import { FusionSummary } from './FusionSummary';
import { CascadeBar } from './CascadeBar';
import { ModalityGrid } from './ModalityGrid';
import { FusionDetail } from './FusionDetail';
import { TimelineScrubber } from './TimelineScrubber';

interface Props {
  state: AppState;
  dispatch: React.Dispatch<Action>;
  record: CanonicalRecord;
}

export function SiteRail({ state, dispatch, record }: Props) {
  return (
    <div
      style={{
        width: 540,
        flexShrink: 0,
        background: 'var(--bg-panel)',
        borderLeft: '1px solid var(--border-subtle)',
        boxShadow: 'var(--shadow-rail)',
        display: 'flex',
        flexDirection: 'column',
        minHeight: 0,
        animation: 'slide-in-right var(--dur-base) var(--ease-entrance)',
      }}
    >
      <Header record={record} dispatch={dispatch} />
      <div style={{ flex: 1, overflowY: 'auto', minHeight: 0 }}>
        <FusionSummary record={record} />
        <CascadeBar tier={state.cascadeTier} costK={state.totalCostK} dispatch={dispatch} />
        <ModalityGrid record={record} tier={state.cascadeTier} />
        <FusionDetail record={record} open={state.fusionDetailOpen} dispatch={dispatch} />
        {record.bookmarked && <TimelineScrubber record={record} />}
      </div>
      <div
        style={{
          padding: '8px 20px',
          borderTop: '1px solid var(--border-subtle)',
          fontSize: 11,
          color: 'var(--text-tertiary)',
          display: 'flex',
          gap: 16,
          flexShrink: 0,
        }}
      >
        <span><span style={{ color: 'var(--obs)' }}>●</span> Observed</span>
        <span><span style={{ color: 'var(--inf)' }}>●</span> Inferred</span>
        <span><span style={{ color: 'var(--dep)' }}>●</span> Deployable</span>
        <span><span style={{ color: 'var(--accent)' }}>●</span> Projected</span>
      </div>
    </div>
  );
}
