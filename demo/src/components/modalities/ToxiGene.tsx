import type { Molecular } from '../../types';
import { ModalityCard } from '../ModalityCard';

interface Props {
  molecular: Molecular;
  cardIndex?: number;
}

const PATHWAYS = [
  'AHR/CYP1A',
  'Metallothionein',
  'Estrogen/Endocrine',
  'Cholinesterase',
  'Oxidative Stress',
  'Heat Shock',
  'DNA Damage',
] as const;
const PROCESSES = [
  'Xenobiotic Metabolism',
  'Oxidative Response',
  'Endocrine Signaling',
  'Genotoxic Response',
] as const;
const OUTCOMES = [
  'Hepatotoxicity',
  'Neurotoxicity',
  'Reproductive Impairment',
  'Endocrine Disruption',
] as const;

function nodeY(i: number, total: number) {
  const padTop = 14;
  const padBot = 14;
  const usable = 150 - padTop - padBot;
  return padTop + (usable / Math.max(1, total - 1)) * i;
}

export function ToxiGeneCard({ molecular, cardIndex }: Props) {
  const active = molecular.activePath;
  const activeIdx = {
    pathway: active ? PATHWAYS.indexOf(active.pathway as any) : -1,
    process: active ? PROCESSES.indexOf(active.process as any) : -1,
    outcome: active ? OUTCOMES.indexOf(active.outcome as any) : -1,
  };
  return (
    <ModalityCard
      name="ToxiGene"
      subtitle="P-Net hierarchy · gene → pathway → process → outcome"
      accentColor="var(--m-molecular)"
      coverage={molecular.state}
      conf={molecular.conf}
      cardIndex={cardIndex}
      deployableTier={3}
      headlineMetric={<>F1 0.894</>}
      headlineLabel="Multi-label toxicity"
      secondary={molecular.outcome ? <span style={{ color: 'var(--text-tertiary)' }}>Outcome {molecular.outcome}</span> : null}
    >
      <svg width="100%" height={150} viewBox="0 0 260 150">
        {/* Column labels */}
        <text x="8" y="9" fontSize="7" fill="var(--text-tertiary)" fontFamily="IBM Plex Mono" letterSpacing="0.08em">
          PATHWAYS
        </text>
        <text x="130" y="9" fontSize="7" fill="var(--text-tertiary)" fontFamily="IBM Plex Mono" letterSpacing="0.08em">
          PROCESSES
        </text>
        <text x="232" y="9" fontSize="7" fill="var(--text-tertiary)" fontFamily="IBM Plex Mono" letterSpacing="0.08em" textAnchor="end">
          OUTCOMES
        </text>
        {/* edges (drawn first) */}
        {PATHWAYS.map((_, pi) =>
          PROCESSES.map((_, qi) => {
            const isActive = pi === activeIdx.pathway && qi === activeIdx.process;
            return (
              <line
                key={`pp${pi}-${qi}`}
                x1={56}
                y1={nodeY(pi, PATHWAYS.length)}
                x2={130}
                y2={nodeY(qi, PROCESSES.length)}
                stroke={isActive ? 'var(--m-molecular)' : 'var(--hairline)'}
                strokeWidth={isActive ? 1.4 : 0.5}
                opacity={isActive ? 1 : 0.4}
              />
            );
          }),
        )}
        {PROCESSES.map((_, qi) =>
          OUTCOMES.map((_, oi) => {
            const isActive = qi === activeIdx.process && oi === activeIdx.outcome;
            return (
              <line
                key={`qo${qi}-${oi}`}
                x1={180}
                y1={nodeY(qi, PROCESSES.length)}
                x2={232}
                y2={nodeY(oi, OUTCOMES.length)}
                stroke={isActive ? 'var(--m-molecular)' : 'var(--hairline)'}
                strokeWidth={isActive ? 1.4 : 0.5}
                opacity={isActive ? 1 : 0.4}
              />
            );
          }),
        )}
        {/* nodes */}
        {PATHWAYS.map((n, i) => {
          const lit = i === activeIdx.pathway;
          return (
            <g key={n} transform={`translate(8,${nodeY(i, PATHWAYS.length) - 6})`}>
              <rect
                width={48}
                height={12}
                fill={lit ? 'var(--m-molecular)' : 'var(--bg-panel)'}
                stroke={lit ? 'var(--m-molecular)' : 'var(--hairline-bright)'}
                strokeWidth={0.8}
                rx={1}
                fillOpacity={lit ? 0.2 : 1}
              />
              <text
                x={4}
                y={8}
                fontSize="6.5"
                fill={lit ? 'var(--text-primary)' : 'var(--text-secondary)'}
                fontFamily="IBM Plex Mono"
              >
                {n}
              </text>
            </g>
          );
        })}
        {PROCESSES.map((n, i) => {
          const lit = i === activeIdx.process;
          return (
            <g key={n} transform={`translate(130,${nodeY(i, PROCESSES.length) - 6})`}>
              <rect
                width={50}
                height={12}
                fill={lit ? 'var(--m-molecular)' : 'var(--bg-panel)'}
                stroke={lit ? 'var(--m-molecular)' : 'var(--hairline-bright)'}
                strokeWidth={0.8}
                rx={1}
                fillOpacity={lit ? 0.2 : 1}
              />
              <text
                x={3}
                y={8}
                fontSize="6.5"
                fill={lit ? 'var(--text-primary)' : 'var(--text-secondary)'}
                fontFamily="IBM Plex Mono"
              >
                {n}
              </text>
            </g>
          );
        })}
        {OUTCOMES.map((n, i) => {
          const lit = i === activeIdx.outcome;
          return (
            <g key={n} transform={`translate(232,${nodeY(i, OUTCOMES.length) - 6})`}>
              <rect
                x={-28}
                width={28}
                height={12}
                fill={lit ? 'var(--m-molecular)' : 'var(--bg-panel)'}
                stroke={lit ? 'var(--m-molecular)' : 'var(--hairline-bright)'}
                strokeWidth={0.8}
                rx={1}
                fillOpacity={lit ? 0.25 : 1}
              />
              <text
                x={-26}
                y={8}
                fontSize="6.5"
                fill={lit ? 'var(--text-primary)' : 'var(--text-secondary)'}
                fontFamily="IBM Plex Mono"
              >
                {n.slice(0, 14)}
              </text>
            </g>
          );
        })}
      </svg>
    </ModalityCard>
  );
}
