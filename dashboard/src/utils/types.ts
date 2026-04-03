// ── Core domain types for SENTINEL dashboard ──

export type AlertLevel = 'normal' | 'low' | 'medium' | 'high' | 'critical';

export interface LatLng {
  lat: number;
  lng: number;
}

export interface Site {
  id: string;
  name: string;
  location: LatLng;
  waterBodyType: 'lake' | 'river' | 'reservoir' | 'estuary' | 'coastal';
  region: string;
  country: string;
  currentAlertLevel: AlertLevel;
  activeContaminants: string[];
  lastUpdated: string;          // ISO datetime
  description: string;
}

export interface SensorReading {
  timestamp: string;            // ISO datetime
  dissolvedOxygen: number | null;   // mg/L
  pH: number | null;
  conductivity: number | null;      // uS/cm
  temperature: number | null;       // C
  turbidity: number | null;         // NTU
  orp: number | null;               // mV
}

export interface AnomalyScore {
  timestamp: string;
  sensorAnomaly: number;        // 0-1
  satelliteAnomaly: number;     // 0-1
  microbialAnomaly: number;     // 0-1
  fusedScore: number;           // 0-1
  alertLevel: AlertLevel;
}

export interface SourceAttribution {
  source: string;
  probability: number;          // 0-1
  evidence: string[];
}

export interface PathwayActivation {
  pathway: string;
  activationLevel: number;      // 0-1
  description: string;
}

export interface TaxonAbundance {
  taxon: string;
  abundance: number;            // relative, 0-1
  isIndicator: boolean;
  indicatorOf?: string;
}

export interface BiosentinelPrediction {
  species: string;
  mortalityRisk: number;        // 0-1
  confidence: number;           // 0-1
  impactDescription: string;
}

export interface EscalationEvent {
  timestamp: string;
  tier: number;                 // 1-5
  trigger: string;
  action: string;
  details: string;
}

export interface SatelliteAcquisition {
  timestamp: string;
  imageUrl?: string;
  anomalyOverlayUrl?: string;
  anomalyDetected: boolean;
  description: string;
  cloudCover: number;           // 0-100 %
}

export interface CaseStudy {
  id: string;
  name: string;
  site: Site;
  description: string;
  eventType: string;
  startDate: string;
  endDate: string;
  authorityResponseDate: string;
  sentinelDetectionDate: string;
  leadTimeDays: number;
  sensorData: SensorReading[];
  anomalyScores: AnomalyScore[];
  sourceAttributions: SourceAttribution[];
  pathwayActivations: PathwayActivation[];
  microbialProfile: TaxonAbundance[];
  biosentinelPredictions: BiosentinelPrediction[];
  escalationHistory: EscalationEvent[];
  satelliteHistory: SatelliteAcquisition[];
  communityHealthScore: number; // 0-100
  reportSummary: string;
  uncertaintyFlags: string[];
}

export const ALERT_COLORS: Record<AlertLevel, string> = {
  normal: '#22c55e',
  low: '#eab308',
  medium: '#f97316',
  high: '#ef4444',
  critical: '#dc2626',
};

export const ALERT_BG_CLASSES: Record<AlertLevel, string> = {
  normal: 'bg-green-500',
  low: 'bg-yellow-500',
  medium: 'bg-orange-500',
  high: 'bg-red-500',
  critical: 'bg-red-700',
};

export const PARAMETER_COLORS: Record<string, string> = {
  dissolvedOxygen: '#3b82f6',
  pH: '#8b5cf6',
  conductivity: '#f59e0b',
  temperature: '#ef4444',
  turbidity: '#6b7280',
  orp: '#10b981',
};

export const PARAMETER_LABELS: Record<string, string> = {
  dissolvedOxygen: 'DO (mg/L)',
  pH: 'pH',
  conductivity: 'Cond. (uS/cm)',
  temperature: 'Temp (C)',
  turbidity: 'Turb. (NTU)',
  orp: 'ORP (mV)',
};
