import { useEffect, useMemo, useRef, useState } from 'react';
import { Sliders, Cloud, CloudRain, RotateCcw, Undo2 } from 'lucide-react';
import type { CanonicalRecord, SensorParam } from '../types';
import type { Action } from '../state';

interface Props {
  record: CanonicalRecord;
  dispatch: React.Dispatch<Action>;
}

// Slider config — each parameter, range, label, and HAB-relevance weight.
const PARAMS: {
  key: SensorParam;
  label: string;
  unit: string;
  min: number;
  max: number;
  step: number;
  weight: number;
}[] = [
  { key: 'do',   label: 'Dissolved O₂', unit: 'mg/L',  min: 0,   max: 14,   step: 0.1, weight: 1.4 },
  { key: 'ph',   label: 'pH',          unit: '',      min: 5,   max: 10,   step: 0.05, weight: 1.0 },
  { key: 'turb', label: 'Turbidity',    unit: 'NTU',   min: 0,   max: 80,   step: 1,    weight: 0.8 },
  { key: 'cond', label: 'Conductivity', unit: 'µS/cm', min: 0,   max: 2500, step: 25,   weight: 1.1 },
  { key: 'temp', label: 'Temperature', unit: '°C',     min: 0,   max: 35,   step: 0.5,  weight: 1.0 },
];

const SCENARIOS = [
  {
    name: 'Drought',
    icon: <Cloud size={13} strokeWidth={1.75} />,
    deltas: { do: -2, ph: 0.4, turb: -3, cond: 400, temp: 4, orp: -10 } as Partial<Record<SensorParam, number>>,
  },
  {
    name: 'Heavy rain',
    icon: <CloudRain size={13} strokeWidth={1.75} />,
    deltas: { do: -1, ph: -0.3, turb: 25, cond: -200, temp: -2, orp: 0 } as Partial<Record<SensorParam, number>>,
  },
];

export function WhatIfPanel({ record, dispatch }: Props) {
  // Baseline = mean of the last 8 samples (treated as "current" reading).
  const baseline = useMemo(() => {
    const out: Record<SensorParam, number> = { do: 0, ph: 0, turb: 0, cond: 0, temp: 0, orp: 0 };
    (Object.keys(out) as SensorParam[]).forEach((k) => {
      const v = record.sensor.series[k];
      if (!v || v.length === 0) return;
      const tail = v.slice(-8);
      out[k] = tail.reduce((s, x) => s + x, 0) / tail.length;
    });
    return out;
  }, [record.sensor.series]);

  const [values, setValues] = useState<Record<SensorParam, number>>(baseline);
  // Undo history: last ≤16 snapshots before the current one.
  const history = useRef<Record<SensorParam, number>[]>([]);

  // Reset when the underlying record changes.
  useEffect(() => {
    setValues(baseline);
    history.current = [];
  }, [baseline]);

  const commitChange = (updater: (cur: Record<SensorParam, number>) => Record<SensorParam, number>) => {
    setValues((cur) => {
      history.current = [...history.current.slice(-15), cur];
      return updater(cur);
    });
  };

  const undo = () => {
    if (history.current.length === 0) return;
    const prev = history.current[history.current.length - 1];
    history.current = history.current.slice(0, -1);
    setValues(prev);
  };

  // Forward model — weighted normalized distance from baseline.
  const anomaly = useMemo(() => {
    let sum = 0;
    let totalWeight = 0;
    for (const p of PARAMS) {
      const range = p.max - p.min;
      const dev = Math.abs(values[p.key] - baseline[p.key]) / range;
      sum += Math.min(1, dev * 2.4) * p.weight;
      totalWeight += p.weight;
    }
    return Math.min(0.999, sum / Math.max(0.0001, totalWeight));
  }, [values, baseline]);

  // Push the simulated anomaly back into the record's fusion + sensor so the UI updates live.
  useEffect(() => {
    const alert =
      anomaly > 0.85 ? 'ALERT'
      : anomaly > 0.65 ? 'INVESTIGATE'
      : anomaly > 0.4  ? 'WATCH'
      : 'MAINTAIN';
    dispatch({
      type: 'PATCH_RECORD',
      patch: {
        sensor: {
          ...record.sensor,
          anomaly: Math.max(record.sensor.anomaly * 0.4, anomaly),
        },
        fusion: { ...record.fusion, anomaly, alert },
      },
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [anomaly, dispatch]);

  const apply = (deltas: Partial<Record<SensorParam, number>>) => {
    commitChange((cur) => {
      const next = { ...cur };
      (Object.keys(deltas) as SensorParam[]).forEach((k) => {
        const d = deltas[k] ?? 0;
        next[k] = Math.max(0, baseline[k] + d);
      });
      return next;
    });
  };

  const reset = () => {
    commitChange(() => baseline);
  };

  return (
    <div>
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: 8,
          fontSize: 13,
          color: 'var(--text-secondary)',
          marginBottom: 10,
        }}
      >
        <Sliders size={13} strokeWidth={1.75} />
        Drag a parameter to see how the fused anomaly responds.
      </div>

      <div style={{ display: 'flex', flexDirection: 'column', gap: 10, marginBottom: 14 }}>
        {PARAMS.map((p) => {
          const v = values[p.key];
          const b = baseline[p.key];
          const delta = v - b;
          return (
            <div key={p.key}>
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12, marginBottom: 4 }}>
                <span style={{ color: 'var(--text-secondary)' }}>
                  {p.label} {p.unit && <span style={{ color: 'var(--text-tertiary)' }}>{p.unit}</span>}
                </span>
                <span className="mono" style={{ color: 'var(--text-primary)' }}>
                  {v.toFixed(p.step >= 1 ? 0 : p.step >= 0.5 ? 1 : 2)}
                  {delta !== 0 && (
                    <span
                      style={{
                        marginLeft: 6,
                        color: Math.abs(delta) / (p.max - p.min) > 0.15 ? 'var(--alert)' : 'var(--text-tertiary)',
                        fontSize: 11,
                      }}
                    >
                      {delta > 0 ? '+' : ''}{delta.toFixed(p.step >= 1 ? 0 : 1)}
                    </span>
                  )}
                </span>
              </div>
              <div style={{ position: 'relative' }}>
                <input
                  type="range"
                  min={p.min}
                  max={p.max}
                  step={p.step}
                  value={v}
                  onChange={(e) =>
                    commitChange((cur) => ({ ...cur, [p.key]: Number(e.target.value) }))
                  }
                  style={{ width: '100%', accentColor: 'var(--accent)' }}
                />
                {/* Baseline tick — fixed marker on the track at the unmodified value. */}
                <div
                  title={`Baseline ${b.toFixed(p.step >= 1 ? 0 : 2)}`}
                  style={{
                    position: 'absolute',
                    left: `${((b - p.min) / (p.max - p.min)) * 100}%`,
                    top: '50%',
                    width: 2,
                    height: 10,
                    marginLeft: -1,
                    marginTop: -5,
                    background: 'var(--text-tertiary)',
                    opacity: 0.7,
                    pointerEvents: 'none',
                  }}
                />
              </div>
            </div>
          );
        })}
      </div>

      <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
        {SCENARIOS.map((s) => (
          <button
            key={s.name}
            onClick={() => apply(s.deltas)}
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: 6,
              padding: '6px 12px',
              background: 'var(--bg-canvas)',
              border: '1px solid var(--border-subtle)',
              borderRadius: 'var(--r-pill)',
              fontSize: 12,
              color: 'var(--text-primary)',
              cursor: 'pointer',
            }}
          >
            {s.icon}
            {s.name}
          </button>
        ))}
        <button
          onClick={undo}
          disabled={history.current.length === 0}
          title="Undo last change"
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: 6,
            padding: '6px 12px',
            background: 'transparent',
            border: '1px solid var(--border-subtle)',
            borderRadius: 'var(--r-pill)',
            fontSize: 12,
            color: 'var(--text-tertiary)',
            cursor: history.current.length === 0 ? 'not-allowed' : 'pointer',
            opacity: history.current.length === 0 ? 0.45 : 1,
          }}
        >
          <Undo2 size={11} strokeWidth={1.75} />
          Undo
        </button>
        <button
          onClick={reset}
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: 6,
            padding: '6px 12px',
            background: 'transparent',
            border: '1px solid var(--border-subtle)',
            borderRadius: 'var(--r-pill)',
            fontSize: 12,
            color: 'var(--text-tertiary)',
            cursor: 'pointer',
          }}
        >
          <RotateCcw size={11} strokeWidth={1.75} />
          Reset
        </button>
      </div>
    </div>
  );
}
