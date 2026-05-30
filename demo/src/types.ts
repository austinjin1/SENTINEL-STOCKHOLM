export type Coverage = 'OBSERVED' | 'INFERRED' | 'DEPLOYABLE' | 'PROJECTED';

export type RegionKey =
  | 'GREAT_LAKES'
  | 'CORN_BELT'
  | 'GULF_COAST'
  | 'ARID_SW'
  | 'PAC_NW'
  | 'NORTHEAST'
  | 'SOUTHEAST'
  | 'MOUNTAIN'
  | 'ARCTIC'
  | 'DEFAULT';

export type AlertTier = 'MAINTAIN' | 'WATCH' | 'INVESTIGATE' | 'ALERT';

export type SensorParam = 'do' | 'ph' | 'turb' | 'cond' | 'temp' | 'orp';

export type HealthState = 'normal' | 'drift' | 'fouling' | 'failure' | 'calibration';

export interface Sensor {
  state: Coverage;
  conf: number;
  series: Record<SensorParam, number[]>;
  anomaly: number;
  anomaly_series: number[];
  driver_param: SensorParam;
  channel_weights: number[]; // length 8
  health: HealthState;
}

export interface Satellite {
  state: Coverage;
  conf: number;
  indices: { ndci: number; fai: number; ndti: number; mndwi: number; oilsheen?: number };
  params: { temp: number; turb: number; tss: number; chla: number; phyco: number };
  tileSeed: number;
}

export type MicroClass =
  | 'Nutrient Runoff'
  | 'Heavy Metals'
  | 'Thermal'
  | 'Pharmaceutical'
  | 'Sediment'
  | 'Oil/Petrochemical'
  | 'Sewage'
  | 'Acid Mine Drainage';

export interface Microbial {
  state: Coverage;
  conf: number;
  classes: { name: MicroClass; p: number }[];
  top: MicroClass | null;
}

export type ToxOutcome =
  | 'Hepatotoxicity'
  | 'Neurotoxicity'
  | 'Reproductive Impairment'
  | 'Endocrine Disruption';

export type ToxPathway =
  | 'AHR/CYP1A'
  | 'Metallothionein'
  | 'Estrogen/Endocrine'
  | 'Cholinesterase'
  | 'Oxidative Stress'
  | 'Heat Shock'
  | 'DNA Damage';

export type ToxProcess =
  | 'Xenobiotic Metabolism'
  | 'Oxidative Response'
  | 'Endocrine Signaling'
  | 'Genotoxic Response';

export interface Molecular {
  state: Coverage;
  conf: number;
  activePath: { pathway: ToxPathway; process: ToxProcess; outcome: ToxOutcome } | null;
  outcome: ToxOutcome | null;
}

export type BehavioralSignature = 'normal' | 'convulsion' | 'immobility';

export interface Behavioral {
  state: Coverage;
  conf: number;
  anomaly: number;
  signature: BehavioralSignature;
  keyframes: number[][]; // procedural fallback
}

export interface FusionResult {
  anomaly: number;
  auroc: number;
  alert: AlertTier;
  attention: [number, number, number, number, number]; // sensor, satellite, behavioral, microbial, molecular
  coverage: number;
}

export interface CanonicalRecord {
  id?: string;
  name: string;
  label: string;
  lat: number;
  lon: number;
  h3: string;
  region: RegionKey;
  bookmarked?: boolean;
  tier?: 1 | 2 | 3 | 4 | 5;
  leadDays?: number;
  advisoryDate?: string; // YYYY-MM-DD — drives time slider + hero image
  bbox?: [number, number, number, number]; // [west, south, east, north] for satellite snapshots
  eventKey?: string; // matches EVENT_CONTENT key
  sensor: Sensor;
  satellite: Satellite;
  microbial: Microbial;
  molecular: Molecular;
  behavioral: Behavioral;
  fusion: FusionResult;
  cascadeTier: 0 | 1 | 2 | 3;
  n_modalities: number;
}
