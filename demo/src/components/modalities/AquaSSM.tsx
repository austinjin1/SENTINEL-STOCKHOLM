import type { Sensor } from '../../types';
import { ModalityCard } from '../ModalityCard';

interface Props {
  sensor: Sensor;
  cardIndex?: number;
  // 0..1 fraction of the series considered "past" — values beyond this index are
  // rendered as a faded dashed continuation, signalling "data not seen yet by SENTINEL".
  // Omitted (or 1) → full opaque chart (default).
  maskPct?: number;
}

const PARAM_LABELS: { key: keyof Sensor['series']; label: string }[] = [
  { key: 'do', label: 'DO mg/L' },
  { key: 'ph', label: 'pH' },
  { key: 'turb', label: 'TURB NTU' },
  { key: 'cond', label: 'COND µS/cm' },
  { key: 'temp', label: 'TEMP °C' },
  { key: 'orp', label: 'ORP mV' },
];

const CHANNELS = ['1h', '4h', '12h', '2d', '7d', '30d', '90d', '365d'];

function pathOf(
  values: number[],
  w: number,
  h: number,
  color: string,
  withBand = false,
  maskPct = 1,
) {
  const min = Math.min(...values);
  const max = Math.max(...values);
  const range = max - min || 1;
  const step = w / (values.length - 1);
  const point = (v: number, i: number) => {
    const x = i * step;
    const y = h - ((v - min) / range) * h;
    return `${x.toFixed(1)},${y.toFixed(1)}`;
  };
  // Split the line into a solid "past" segment + dashed "future" segment when
  // maskPct < 1, so the user sees that future data exists but the model only
  // knows up to currentDate.
  const cutoff = Math.max(0, Math.min(values.length - 1, Math.round((values.length - 1) * maskPct)));
  const pastValues = values.slice(0, cutoff + 1);
  const futureValues = values.slice(cutoff);
  const pastD =
    pastValues.length > 0
      ? pastValues.map((v, i) => `${i === 0 ? 'M' : 'L'}${point(v, i)}`).join(' ')
      : '';
  const futureD =
    futureValues.length > 1
      ? futureValues.map((v, i) => `${i === 0 ? 'M' : 'L'}${point(v, i + cutoff)}`).join(' ')
      : '';
  const lines = (
    <>
      {pastD && <path d={pastD} fill="none" stroke={color} strokeWidth={1.2} />}
      {futureD && maskPct < 1 && (
        <path
          d={futureD}
          fill="none"
          stroke={color}
          strokeWidth={1.2}
          strokeDasharray="2 2"
          opacity={0.35}
        />
      )}
    </>
  );
  if (!withBand) return lines;
  // ±1σ band: shaded horizontal stripe behind the line.
  const mean = values.reduce((s, x) => s + x, 0) / values.length;
  const variance =
    values.reduce((s, x) => s + (x - mean) ** 2, 0) / Math.max(1, values.length - 1);
  const sd = Math.sqrt(variance);
  const yMean = h - ((mean - min) / range) * h;
  const yHi = h - ((mean + sd - min) / range) * h;
  const yLo = h - ((mean - sd - min) / range) * h;
  return (
    <>
      <rect x={0} y={yHi} width={w} height={Math.max(0, yLo - yHi)} fill={color} opacity={0.12} />
      <line x1={0} y1={yMean} x2={w} y2={yMean} stroke={color} strokeWidth={0.4} opacity={0.5} strokeDasharray="2 2" />
      {lines}
    </>
  );
}

export function AquaSSMCard({ sensor, cardIndex, maskPct = 1 }: Props) {
  return (
    <ModalityCard
      name="AquaSSM"
      subtitle="Continuous-time SSM · 8 timescales · 25M params"
      accentColor="var(--m-sensor)"
      coverage={sensor.state}
      conf={sensor.conf}
      cardIndex={cardIndex}
      headlineMetric={sensor.anomaly.toFixed(3)}
      headlineLabel="Anomaly probability"
      secondary={<span style={{ color: 'var(--text-tertiary)' }}>Health nominal</span>}
    >
      {/* 6 sparklines */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 4 }}>
        {PARAM_LABELS.map((p) => {
          const v = sensor.series[p.key];
          const last = v[v.length - 1];
          const isDriver = sensor.driver_param === p.key;
          return (
            <div
              key={p.key}
              style={{
                background: 'transparent',
                position: 'relative',
                height: 34,
                border: '1px solid var(--hairline)',
              }}
            >
              <svg width="100%" height="100%" viewBox="0 0 150 34" preserveAspectRatio="none">
                {pathOf(v, 150, 34, isDriver ? 'var(--alert)' : 'var(--m-sensor)', isDriver, maskPct)}
              </svg>
              <div
                className="mono"
                style={{
                  position: 'absolute',
                  top: 1,
                  left: 3,
                  fontSize: 8,
                  color: isDriver ? 'var(--alert)' : 'var(--text-tertiary)',
                  letterSpacing: '0.04em',
                }}
              >
                {p.label}
              </div>
              <div
                className="mono"
                style={{
                  position: 'absolute',
                  top: 1,
                  right: 3,
                  fontSize: 8,
                  color: 'var(--text-mono)',
                }}
              >
                {last.toFixed(p.key === 'cond' ? 0 : 1)}
              </div>
            </div>
          );
        })}
      </div>
      {/* anomaly line */}
      <div style={{ marginTop: 6, height: 40, position: 'relative', border: '1px solid var(--hairline)' }}>
        <svg width="100%" height="100%" viewBox="0 0 220 40" preserveAspectRatio="none">
          <defs>
            <linearGradient id="anomGrad" x1="0" x2="0" y1="0" y2="1">
              <stop offset="0%" stopColor="var(--alert)" stopOpacity="0.4" />
              <stop offset="100%" stopColor="var(--alert)" stopOpacity="0" />
            </linearGradient>
          </defs>
          {/* threshold */}
          <line x1="0" y1={40 - 0.9 * 40} x2="220" y2={40 - 0.9 * 40} stroke="var(--text-tertiary)" strokeWidth="0.5" strokeDasharray="2 2" />
          {/* fill */}
          {(() => {
            const v = sensor.anomaly_series;
            const step = 220 / (v.length - 1);
            const cutoff = Math.max(0, Math.min(v.length - 1, Math.round((v.length - 1) * maskPct)));
            const pastPts = v
              .slice(0, cutoff + 1)
              .map((y, i) => `${i === 0 ? 'M' : 'L'}${(i * step).toFixed(1)},${(40 - y * 40).toFixed(1)}`)
              .join(' ');
            const futurePts = v
              .slice(cutoff)
              .map((y, i) => `${i === 0 ? 'M' : 'L'}${((i + cutoff) * step).toFixed(1)},${(40 - y * 40).toFixed(1)}`)
              .join(' ');
            const lastX = (cutoff * step).toFixed(1);
            const fill = `${pastPts} L${lastX},40 L0,40 Z`;
            return (
              <>
                <path d={fill} fill="url(#anomGrad)" />
                <path d={pastPts} fill="none" stroke="var(--alert)" strokeWidth={1.4} />
                {maskPct < 1 && (
                  <path
                    d={futurePts}
                    fill="none"
                    stroke="var(--alert)"
                    strokeWidth={1.4}
                    strokeDasharray="3 2"
                    opacity={0.3}
                  />
                )}
                {maskPct < 1 && maskPct > 0 && (
                  <line
                    x1={lastX}
                    y1={0}
                    x2={lastX}
                    y2={40}
                    stroke="var(--text-secondary)"
                    strokeWidth={0.6}
                    strokeDasharray="2 2"
                  />
                )}
              </>
            );
          })()}
        </svg>
        <div
          className="mono"
          style={{ position: 'absolute', top: 2, left: 4, fontSize: 8, color: 'var(--text-tertiary)' }}
        >
          ANOMALY
        </div>
      </div>
      {/* 8-channel strip */}
      <div style={{ marginTop: 6, display: 'flex', gap: 2, alignItems: 'flex-end', height: 28 }}>
        {sensor.channel_weights.map((w, i) => (
          <div key={i} style={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 2 }}>
            <div
              style={{
                width: '100%',
                height: 16,
                display: 'flex',
                alignItems: 'flex-end',
                background: 'var(--inert-dim)',
              }}
            >
              <div
                style={{
                  width: '100%',
                  height: `${w * 100}%`,
                  background: w > 0.8 ? 'var(--accent)' : 'var(--inert)',
                  transition: 'height var(--dur-slow) var(--ease-standard)',
                }}
              />
            </div>
            <div className="mono" style={{ fontSize: 7, color: 'var(--text-tertiary)' }}>
              {CHANNELS[i]}
            </div>
          </div>
        ))}
      </div>
    </ModalityCard>
  );
}
