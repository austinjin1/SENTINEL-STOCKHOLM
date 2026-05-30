import type { Microbial } from '../../types';
import { ModalityCard } from '../ModalityCard';

interface Props {
  micro: Microbial;
  cardIndex?: number;
  tier: 0 | 1 | 2 | 3;
}

export function MicroBiomeNetCard({ micro, cardIndex, tier }: Props) {
  const sorted = [...micro.classes].sort((a, b) => b.p - a.p);
  const max = sorted[0]?.p ?? 1;
  return (
    <ModalityCard
      name="MicroBiomeNet"
      subtitle="Aitchison transformer + Simplex Neural-ODE · CLR space"
      accentColor="var(--m-microbial)"
      coverage={micro.state}
      conf={micro.conf}
      cardIndex={cardIndex}
      deployableTier={2}
      headlineMetric={<>F1 0.913</>}
      headlineLabel="Eight-class source attribution"
      secondary={micro.top ? <span style={{ color: 'var(--text-tertiary)' }}>Top {micro.top} <span className="mono" style={{ color: 'var(--text-secondary)' }}>{sorted[0].p.toFixed(2)}</span></span> : null}
    >
      <div style={{ display: 'flex', flexDirection: 'column', gap: 3 }}>
        {sorted.map((c, i) => {
          const w = (c.p / max) * 100;
          const top = i === 0;
          return (
            <div key={c.name} style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
              <div
                className="mono"
                style={{
                  width: 110,
                  fontSize: 9,
                  color: top ? 'var(--text-primary)' : 'var(--text-tertiary)',
                  letterSpacing: '0.02em',
                  textAlign: 'right',
                }}
              >
                {c.name}
              </div>
              <div style={{ flex: 1, height: 8, background: 'var(--inert-dim)' }}>
                <div
                  style={{
                    height: '100%',
                    width: `${w}%`,
                    background: top ? 'var(--m-microbial)' : 'var(--inert)',
                    transition: 'width var(--dur-slow) var(--ease-standard)',
                  }}
                />
              </div>
              <div className="mono" style={{ width: 32, fontSize: 9, color: 'var(--text-mono)', textAlign: 'right' }}>
                {c.p.toFixed(2)}
              </div>
            </div>
          );
        })}
      </div>
    </ModalityCard>
  );
}
