import { Map as MapIcon, List, Layers, PlayCircle } from 'lucide-react';
import type { Action } from '../state';

interface Props {
  dispatch: React.Dispatch<Action>;
  legendOpen: boolean;
  demoMode: boolean;
  methodologyOpen: boolean;
}

const NavBtn: React.FC<{
  icon: React.ReactNode;
  label: string;
  active?: boolean;
  onClick?: () => void;
}> = ({ icon, label, active, onClick }) => (
  <button
    onClick={onClick}
    title={label}
    style={{
      width: 56,
      height: 56,
      display: 'flex',
      flexDirection: 'column',
      alignItems: 'center',
      justifyContent: 'center',
      gap: 4,
      background: active ? 'var(--accent-soft)' : 'transparent',
      color: active ? 'var(--accent)' : 'var(--text-secondary)',
      transition: 'background var(--dur-fast), color var(--dur-fast)',
    }}
    onMouseEnter={(e) => {
      if (!active) e.currentTarget.style.background = 'rgba(0,0,0,0.04)';
    }}
    onMouseLeave={(e) => {
      if (!active) e.currentTarget.style.background = 'transparent';
    }}
  >
    {icon}
    <span style={{ fontSize: 10, fontWeight: 500 }}>{label}</span>
  </button>
);

export function LeftRail({ dispatch, legendOpen, demoMode }: Props) {
  return (
    <div
      style={{
        width: 56,
        background: 'rgba(255,255,255,0.72)',
        backdropFilter: 'blur(20px) saturate(180%)',
        WebkitBackdropFilter: 'blur(20px) saturate(180%)',
        borderRight: '1px solid var(--border-subtle)',
        display: 'flex',
        flexDirection: 'column',
        justifyContent: 'space-between',
        flexShrink: 0,
      }}
    >
      <div>
        <NavBtn icon={<MapIcon size={18} strokeWidth={1.75} />} label="Map" active />
        <NavBtn icon={<List size={18} strokeWidth={1.75} />} label="Sites" />
      </div>
      <div>
        <NavBtn
          icon={<Layers size={18} strokeWidth={1.75} />}
          label="Legend"
          active={legendOpen}
          onClick={() => dispatch({ type: 'TOGGLE_LEGEND' })}
        />
        <NavBtn
          icon={<PlayCircle size={18} strokeWidth={1.75} />}
          label="Tour"
          active={demoMode}
          onClick={() => dispatch({ type: 'TOGGLE_DEMO' })}
        />
      </div>
    </div>
  );
}
