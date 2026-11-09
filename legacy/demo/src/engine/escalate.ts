import type { CanonicalRecord, Coverage } from '../types';

// Compute the patch to apply when escalating to the given tier.
// Activates the modality the tier adds, re-projects fusion attention/coverage.

export function applyTierActivation(
  record: CanonicalRecord,
  tier: 0 | 1 | 2 | 3,
): Partial<CanonicalRecord> {
  // tier 0: sensor + behavioral (already on)
  // tier 1: + satellite (already on by default)
  // tier 2: + microbial
  // tier 3: + molecular
  const patch: Partial<CanonicalRecord> = {};
  let microState: Coverage = record.microbial.state;
  let molState: Coverage = record.molecular.state;
  let microConf = record.microbial.conf;
  let molConf = record.molecular.conf;

  if (tier >= 2 && record.microbial.state === 'DEPLOYABLE') {
    microState = record.bookmarked ? 'OBSERVED' : 'PROJECTED';
    microConf = record.bookmarked ? 0.91 : 0.72;
  }
  if (tier >= 3 && record.molecular.state === 'DEPLOYABLE') {
    molState = record.bookmarked ? 'OBSERVED' : 'PROJECTED';
    molConf = record.bookmarked ? 0.86 : 0.64;
  }

  patch.microbial = { ...record.microbial, state: microState, conf: microConf };
  patch.molecular = { ...record.molecular, state: molState, conf: molConf };

  // attention shifts when bio modalities come online
  let att: [number, number, number, number, number] = [0.378, 0.361, 0.261, 0, 0];
  if (tier >= 2) att = [0.34, 0.32, 0.22, 0.12, 0];
  if (tier >= 3) att = [0.3, 0.28, 0.18, 0.14, 0.1];

  const n =
    2 + // sensor + behavioral always on
    (record.satellite.state !== 'DEPLOYABLE' ? 1 : 0) +
    (microState !== 'DEPLOYABLE' ? 1 : 0) +
    (molState !== 'DEPLOYABLE' ? 1 : 0);

  // Tiny corroboration nudge once biological modalities (microbial @ T2, molecular @ T3)
  // come online. 0.001 is deliberately small so it never reorders alerts on its own —
  // it just lets a record sitting on a threshold tick over when bio data agrees.
  const boost = tier >= 2 ? 0.001 : 0;
  patch.fusion = {
    ...record.fusion,
    attention: att,
    anomaly: Math.min(0.9999, record.fusion.anomaly + boost),
    coverage: Math.min(0.99, 0.7 + n * 0.06),
  };
  patch.n_modalities = n;
  patch.cascadeTier = tier;
  return patch;
}
