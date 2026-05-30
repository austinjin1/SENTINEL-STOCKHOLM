import type { Action, Layers } from '../state';

interface Props {
  layers: Layers;
  dispatch: React.Dispatch<Action>;
}

const CHIPS: { key: keyof Layers; label: string; color: string }[] = [
  { key: 'usgs', label: 'USGS network', color: 'var(--m-sensor)' },
  { key: 'neon', label: 'NEON sites', color: 'var(--m-microbial)' },
  { key: 'satellite', label: 'Satellite imagery', color: '#6f95c2' },
  { key: 'chlorophyll', label: 'Chlorophyll-a', color: '#3fa063' },
  { key: 'drought', label: 'Drought monitor', color: '#c89441' },
];

export function CoverageToggles({ layers, dispatch }: Props) {
  return (
    <div
      style={{
        position: 'absolute',
        top: 16,
        left: 16,
        display: 'flex',
        gap: 6,
        background: 'rgba(255,255,255,0.78)',
        backdropFilter: 'blur(20px) saturate(180%)',
        WebkitBackdropFilter: 'blur(20px) saturate(180%)',
        padding: 4,
        borderRadius: 'var(--r-pill)',
        boxShadow: 'var(--shadow-md)',
        zIndex: 'var(--z-map-overlay)' as any,
      }}
    >
      {CHIPS.map((c) => {
        const on = layers[c.key];
        return (
          <button
            key={c.key}
            onClick={() => dispatch({ type: 'TOGGLE_LAYER', layer: c.key })}
            style={{
              height: 30,
              padding: '0 14px',
              display: 'flex',
              alignItems: 'center',
              gap: 8,
              background: on ? 'var(--bg-canvas)' : 'transparent',
              color: on ? 'var(--text-primary)' : 'var(--text-tertiary)',
              fontSize: 13,
              borderRadius: 'var(--r-pill)',
              fontWeight: on ? 500 : 400,
              transition: 'all var(--dur-fast)',
            }}
          >
            <span
              style={{
                width: 8,
                height: 8,
                borderRadius: 4,
                background: on ? c.color : 'rgba(0,0,0,0.16)',
              }}
            />
            {c.label}
          </button>
        );
      })}
    </div>
  );
}
