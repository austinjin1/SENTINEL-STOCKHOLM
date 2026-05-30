import type { Satellite } from '../../types';
import { ModalityCard } from '../ModalityCard';

interface Props {
  satellite: Satellite;
  cardIndex?: number;
  tier: 0 | 1 | 2 | 3;
  // When date + bbox are both supplied, the card swaps its procedural HeatTile
  // for a NASA Worldview MODIS Aqua chlorophyll-a snapshot at that date — the
  // image physically updates as the live-playback head advances.
  date?: string;
  bbox?: [number, number, number, number]; // [west, south, east, north]
}

const INDICES: { key: keyof Satellite['indices']; label: string }[] = [
  { key: 'ndci', label: 'NDCI' },
  { key: 'fai', label: 'FAI' },
  { key: 'ndti', label: 'NDTI' },
  { key: 'mndwi', label: 'MNDWI' },
];

function HeatTile({ seed }: { seed: number }) {
  // procedural bloom hotspot
  const cells: { x: number; y: number; v: number }[] = [];
  const rng = (i: number) => {
    const s = Math.sin(seed * 9301 + i * 49297) * 233280;
    return s - Math.floor(s);
  };
  for (let y = 0; y < 8; y++) {
    for (let x = 0; x < 24; x++) {
      const dx = (x - 17) / 6;
      const dy = (y - 4) / 4;
      const r = Math.sqrt(dx * dx + dy * dy);
      const hotspot = Math.exp(-r * r) * 0.9;
      const noise = rng(y * 24 + x) * 0.2;
      cells.push({ x, y, v: Math.min(1, hotspot + noise * 0.4 + 0.1) });
    }
  }
  return (
    <svg width="100%" height="56" viewBox="0 0 240 56" preserveAspectRatio="none">
      {cells.map((c, i) => {
        const r = Math.round(0 + c.v * 80);
        const g = Math.round(40 + c.v * 200);
        const b = Math.round(120 - c.v * 80);
        return (
          <rect
            key={i}
            x={c.x * 10}
            y={c.y * 7}
            width={10}
            height={7}
            fill={`rgb(${r},${g},${b})`}
          />
        );
      })}
    </svg>
  );
}

function WorldviewChlA({ date, bbox }: { date: string; bbox: [number, number, number, number] }) {
  const [west, south, east, north] = bbox;
  const url =
    `https://wvs.earthdata.nasa.gov/api/v1/snapshot?REQUEST=GetSnapshot` +
    `&LAYERS=MODIS_Aqua_Chlorophyll_A,Coastlines` +
    `&CRS=EPSG:4326&TIME=${date}` +
    `&BBOX=${south},${west},${north},${east}` +
    `&WIDTH=480&HEIGHT=120&FORMAT=image/jpeg`;
  return (
    <div style={{ position: 'relative', width: '100%', height: 56, overflow: 'hidden', borderRadius: 2 }}>
      <img
        src={url}
        alt={`MODIS Aqua chl-a · ${date}`}
        loading="lazy"
        style={{
          width: '100%',
          height: '100%',
          objectFit: 'cover',
          display: 'block',
          animation: 'fade-in 280ms var(--ease-entrance)',
        }}
        onError={(e) => {
          // GIBS returns black tiles on cloudy days — fade so it doesn't dominate.
          (e.target as HTMLImageElement).style.opacity = '0.25';
        }}
      />
      <div
        className="mono"
        style={{
          position: 'absolute',
          left: 4,
          bottom: 3,
          fontSize: 8,
          color: '#fff',
          textShadow: '0 1px 2px rgba(0,0,0,0.7)',
        }}
      >
        MODIS Aqua chl-a · {date}
      </div>
    </div>
  );
}

export function HydroViTCard({ satellite, cardIndex, date, bbox }: Props) {
  return (
    <ModalityCard
      name="HydroViT"
      subtitle="ViT-S/16 · MAE pretrain · 13 bands"
      accentColor="var(--m-satellite)"
      coverage={satellite.state}
      conf={satellite.conf}
      cardIndex={cardIndex}
      headlineMetric={<>R² 0.749</>}
      headlineLabel="Water temperature"
      secondary={<span style={{ color: 'var(--text-tertiary)' }}>Chl-a <span className="mono" style={{ color: 'var(--text-secondary)' }}>{satellite.params.chla}</span> µg/L</span>}
    >
      {/* index meters */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 6 }}>
        {INDICES.map((idx) => {
          const v = satellite.indices[idx.key] ?? 0;
          const hot = idx.key === 'ndci' && v > 0.3;
          return (
            <div key={idx.key}>
              <div className="mono" style={{ fontSize: 8, color: 'var(--text-tertiary)' }}>
                {idx.label}
              </div>
              <div
                style={{
                  height: 4,
                  background: 'var(--inert-dim)',
                  marginTop: 2,
                  position: 'relative',
                }}
              >
                <div
                  style={{
                    height: '100%',
                    width: `${Math.min(100, v * 100)}%`,
                    background: hot ? 'var(--alert)' : 'var(--m-satellite)',
                  }}
                />
              </div>
              <div className="mono" style={{ fontSize: 10, color: 'var(--text-primary)', marginTop: 2 }}>
                {v.toFixed(2)}
              </div>
            </div>
          );
        })}
      </div>
      <div style={{ marginTop: 6 }}>
        {date && bbox ? (
          <WorldviewChlA date={date} bbox={bbox} />
        ) : (
          <HeatTile seed={satellite.tileSeed} />
        )}
        <div className="mono" style={{ fontSize: 8, color: 'var(--text-tertiary)', textAlign: 'right', marginTop: 2 }}>
          {date && bbox ? 'NASA EOSDIS Worldview · live tile' : 'derived indices (illustrative)'}
        </div>
      </div>
    </ModalityCard>
  );
}
