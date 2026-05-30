import type { CanonicalRecord } from '../types';
import { AquaSSMCard } from './modalities/AquaSSM';
import { HydroViTCard } from './modalities/HydroViT';
import { MicroBiomeNetCard } from './modalities/MicroBiomeNet';
import { ToxiGeneCard } from './modalities/ToxiGene';
import { BioMotionCard } from './modalities/BioMotion';

interface Props {
  record: CanonicalRecord;
  tier: 0 | 1 | 2 | 3;
}

export function ModalityGrid({ record, tier }: Props) {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 12, padding: 16 }}>
      <AquaSSMCard sensor={record.sensor} cardIndex={0} />
      <HydroViTCard satellite={record.satellite} cardIndex={1} tier={tier} />
      <BioMotionCard bio={record.behavioral} cardIndex={2} />
      <MicroBiomeNetCard micro={record.microbial} cardIndex={3} tier={tier} />
      <ToxiGeneCard molecular={record.molecular} cardIndex={4} />
    </div>
  );
}
