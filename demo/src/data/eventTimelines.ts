// Per-event milestone timelines for the "Play story" feature.
//
// Each entry corresponds to one of the 6 events with real SENTINEL score data
// (src/data/repo/*_scores.json). Authoritative dates (advisory_date, lead_time_days)
// are pulled from case_studies_real.json — the milestone narrative around them is
// hand-curated from the cited public sources (EPA, NOAA, state agencies, peer-reviewed).
//
// To add a new event: add the score JSON, then add a new entry below with the same
// eventKey. The UI registers itself from this map.

export type MilestoneKind =
  | 'context'         // background driver, no SENTINEL signal yet (gray)
  | 'precursor'       // measurable upstream signal in sensor / satellite data (amber)
  | 'sentinel-alert'  // SENTINEL fused-alert threshold crossed (green, headline)
  | 'first-impact'    // first ecological or human impact reported (orange)
  | 'official'        // government advisory / declaration (red)
  | 'resolution';     // remediation milestone (blue)

export interface Milestone {
  date: string;       // YYYY-MM-DD
  kind: MilestoneKind;
  title: string;
  body?: string;
  source?: { label: string; url?: string };
  // When set + during playback, crossing this milestone auto-escalates the
  // cascade controller to the given tier (and charges the corresponding cost).
  // Mirrors the paper's PPO cascade controller: monitor → investigate → alert.
  escalateTo?: 0 | 1 | 2 | 3;
}

export interface EventTimeline {
  eventKey: string;
  windowStart: string;       // first date shown on the slider
  windowEnd: string;         // last date shown
  sentinelAlertDate: string; // when SENTINEL would have first fired (advisory − leadDays)
  advisoryDate: string;
  // Short narrative for the lead-time callout at the bottom of the panel.
  leadSummary: string;
  milestones: Milestone[];
}

// Helper: shift an ISO date by N days (positive or negative).
function shiftISO(iso: string, days: number): string {
  return new Date(new Date(iso + 'T00:00:00Z').getTime() + days * 86400000)
    .toISOString()
    .slice(0, 10);
}

export const EVENT_TIMELINES: Record<string, EventTimeline> = {
  /* ---------------------------------------------------------------- */
  lake_erie_hab_2023: {
    eventKey: 'lake_erie_hab_2023',
    windowStart: '2023-04-01',
    windowEnd: '2023-08-15',
    sentinelAlertDate: '2023-05-16',
    advisoryDate: '2023-07-15',
    leadSummary:
      'SENTINEL crossed its detection threshold 59 days before the EPA advisory; per-modality scoring shows HydroViT chlorophyll-a anomaly leading by 10 days and AquaSSM DO/turbidity by 5 days.',
    milestones: [
      {
        date: '2023-04-15',
        kind: 'context',
        title: 'Spring phosphorus loading begins',
        body: 'Maumee River discharge rises with snowmelt + fertilizer runoff; total P loading is the dominant Lake Erie HAB precursor.',
        source: { label: 'NOAA NCCOS HAB Bulletin 2023' },
      },
      {
        date: '2023-05-16',
        kind: 'sentinel-alert',
        title: 'SENTINEL fused-alert threshold crossed',
        body: 'AquaSSM continuous-time anomaly rises above 0.5; cascade controller escalates from monitor to investigate.',
        source: { label: 'SENTINEL benchmark, case_studies_real.json' },
        escalateTo: 1,
      },
      {
        date: '2023-07-05',
        kind: 'precursor',
        title: 'HydroViT chlorophyll-a anomaly (T−10d)',
        body: 'MAE-ViT chl-a reconstruction error spikes — paper Figure 5 trajectory.',
        source: { label: 'SENTINEL paper §5 Case-Study Timeline' },
      },
      {
        date: '2023-07-10',
        kind: 'precursor',
        title: 'AquaSSM DO + turbidity anomalies (T−5d)',
        body: 'Dissolved-oxygen swings + turbidity rise corroborate the satellite signal.',
        source: { label: 'SENTINEL paper §5' },
      },
      {
        date: '2023-07-12',
        kind: 'sentinel-alert',
        title: 'Cascade controller fires ALERT (+201.6h before advisory)',
        body: 'Fused anomaly probability crosses the operational alert threshold; deployment-relevant lead time.',
        source: { label: 'SENTINEL paper §5' },
        escalateTo: 2,
      },
      {
        date: '2023-07-15',
        kind: 'official',
        title: 'EPA / Ohio EPA advisory issued',
        body: 'Western Lake Erie HAB advisory affects 11M consumers in the Toledo / Cleveland metro corridor.',
        source: { label: 'NOAA Lake Erie HAB Forecast 2023', url: 'https://coastalscience.noaa.gov/' },
        escalateTo: 3,
      },
      {
        date: '2023-08-05',
        kind: 'first-impact',
        title: 'Peak bloom — recreational closures',
        body: 'Maumee Bay and Sandusky Bay close to swimming; Toledo water plant raises Carlson-index treatment.',
      },
    ],
  },

  /* ---------------------------------------------------------------- */
  gulf_dead_zone_2023: {
    eventKey: 'gulf_dead_zone_2023',
    windowStart: '2023-03-01',
    windowEnd: '2023-08-15',
    sentinelAlertDate: '2023-04-04',
    advisoryDate: '2023-07-01',
    leadSummary:
      'Longest validated lead time in the cohort: 87 days. The precursor signal is total-N loading from the Mississippi-Atchafalaya basin, weeks before any bottom-water hypoxia is measurable in situ.',
    milestones: [
      {
        date: '2023-02-20',
        kind: 'context',
        title: 'Mississippi River spring nitrate loading begins',
        body: 'Winter rainfall + corn-belt fertilizer runoff drives elevated TN in the lower river — the primary causal driver of summer Gulf hypoxia.',
        source: { label: 'USGS NAWQA, Hypoxia Task Force 2023' },
      },
      {
        date: '2023-04-04',
        kind: 'sentinel-alert',
        title: 'SENTINEL anomaly threshold crossed',
        body: 'AquaSSM detects the conductivity + TN anomaly signature; 87 days before NOAA field-survey confirmation.',
        source: { label: 'SENTINEL benchmark, case_studies_real.json' },
        escalateTo: 1,
      },
      {
        date: '2023-05-15',
        kind: 'precursor',
        title: 'LUMCON spring cruise samples low-DO bottom water',
        body: 'Louisiana Universities Marine Consortium reports early bottom-water hypoxia in transect samples.',
        source: { label: 'LUMCON Spring Hypoxia Cruise Report' },
      },
      {
        date: '2023-06-26',
        kind: 'precursor',
        title: 'NOAA forecast issued',
        body: 'NOAA NCCOS forecasts an "above-average" 4,155 sq-mi dead zone for summer 2023.',
        source: {
          label: 'NOAA NCCOS Gulf Hypoxia Forecast',
          url: 'https://coastalscience.noaa.gov/',
        },
        escalateTo: 2,
      },
      {
        date: '2023-07-01',
        kind: 'official',
        title: 'Hypoxia advisory window opens',
        body: 'Fisheries-management advisory window opens; recreational + commercial guidance follows.',
        escalateTo: 3,
      },
      {
        date: '2023-07-24',
        kind: 'first-impact',
        title: 'NOAA field survey confirms 3,058 sq-mi hypoxic zone',
        body: 'Equal to the area of New Jersey; eighth-smallest since 1985 but spatially intense near the LA coast.',
        source: { label: 'NOAA Hypoxia Task Force 2023' },
      },
    ],
  },

  /* ---------------------------------------------------------------- */
  chesapeake_hypoxia_2018: {
    eventKey: 'chesapeake_hypoxia_2018',
    windowStart: '2018-04-01',
    windowEnd: '2018-09-30',
    sentinelAlertDate: '2018-04-21',
    advisoryDate: '2018-07-20',
    leadSummary:
      'The largest lead time in the validation cohort (89.8 days). Anomalous freshwater discharge from Susquehanna in March-April 2018 was a dominant precursor of the July hypoxic volume.',
    milestones: [
      {
        date: '2018-03-15',
        kind: 'context',
        title: 'Wet spring on Susquehanna basin',
        body: 'Above-normal precipitation drives anomalously high freshwater discharge — a known stratification + nutrient-loading precursor.',
        source: { label: 'USGS Chesapeake Bay Streamflow Report 2018' },
      },
      {
        date: '2018-04-21',
        kind: 'sentinel-alert',
        title: 'SENTINEL detects anomalous discharge signature',
        body: 'AquaSSM scores conductivity + temperature drops characteristic of unusually large freshwater plume.',
        source: { label: 'SENTINEL benchmark, case_studies_real.json' },
        escalateTo: 1,
      },
      {
        date: '2018-06-10',
        kind: 'precursor',
        title: 'Bottom-water DO drops below 2 mg/L',
        body: 'Maryland DNR Eyes-on-the-Bay sondes show stratified hypoxic layer expanding mid-bay.',
        source: { label: 'Maryland DNR Bay Health Index 2018' },
        escalateTo: 2,
      },
      {
        date: '2018-07-20',
        kind: 'official',
        title: 'Maryland DNR hypoxia alert',
        body: 'DNR Bay Health Index reports the 3rd-worst hypoxic event on record at the time.',
        source: { label: 'Chesapeake Bay Program Annual Indicator Report 2018' },
        escalateTo: 3,
      },
      {
        date: '2018-08-15',
        kind: 'first-impact',
        title: 'Blue-crab fishery affected',
        body: 'Maryland DNR estimates ~$30M in fishery losses tied to the hypoxic season.',
      },
      {
        date: '2018-09-05',
        kind: 'resolution',
        title: 'Peak hypoxic volume 2.7 cu mi',
        body: 'Volume settles at 2.7 cu mi — confirmed in the annual indicator report.',
        source: { label: 'Chesapeake Bay Program Annual Indicator Report 2018' },
      },
    ],
  },

  /* ---------------------------------------------------------------- */
  klamath_river_hab_2021: {
    eventKey: 'klamath_river_hab_2021',
    windowStart: '2021-04-15',
    windowEnd: '2021-10-15',
    sentinelAlertDate: '2021-06-03',
    advisoryDate: '2021-08-01',
    leadSummary:
      'Drought-driven Microcystis bloom in Copco and Iron Gate reservoirs. SENTINEL flagged 59 days early on pH + DO swings characteristic of pre-bloom productivity surges.',
    milestones: [
      {
        date: '2021-04-20',
        kind: 'context',
        title: 'Extreme drought conditions',
        body: 'Klamath Basin enters exceptional drought; reservoir levels at <40% of historical normal.',
        source: { label: 'US Drought Monitor weekly bulletins' },
      },
      {
        date: '2021-06-03',
        kind: 'sentinel-alert',
        title: 'SENTINEL pH + DO anomaly threshold',
        body: 'Pre-bloom diurnal pH swings and afternoon DO supersaturation drive the AquaSSM score above threshold.',
        source: { label: 'SENTINEL benchmark, case_studies_real.json' },
        escalateTo: 1,
      },
      {
        date: '2021-07-08',
        kind: 'precursor',
        title: 'Microcystis surface scum observed',
        body: 'PacifiCorp + Karuk Tribe biologists report cyanobacterial scum in Copco Reservoir.',
        source: { label: 'PacifiCorp Klamath Reservoir Monitoring' },
        escalateTo: 2,
      },
      {
        date: '2021-08-01',
        kind: 'official',
        title: 'CA Water Boards recreational advisory',
        body: 'Cyanotoxin concentrations exceed California recreational thresholds; advisories posted at all reservoir access points.',
        source: { label: 'CA Water Boards HAB Incident Reports 2021' },
        escalateTo: 3,
      },
      {
        date: '2021-09-15',
        kind: 'first-impact',
        title: 'Yurok + Karuk subsistence closures',
        body: 'Tribal subsistence fishing closed for ceremonial and food use; advisory persists 60+ days.',
        source: { label: 'Yurok Tribe Environmental Program' },
      },
    ],
  },

  /* ---------------------------------------------------------------- */
  mississippi_salinity_2023: {
    eventKey: 'mississippi_salinity_2023',
    windowStart: '2023-06-01',
    windowEnd: '2023-11-15',
    sentinelAlertDate: '2023-08-03',
    advisoryDate: '2023-10-01',
    leadSummary:
      'Drought + low river flow allowed a salt wedge to migrate upriver toward New Orleans drinking-water intakes — a 1988-style emergency. SENTINEL flagged conductance trajectory anomalies 58 days early.',
    milestones: [
      {
        date: '2023-06-15',
        kind: 'context',
        title: 'Mississippi flow drops below threshold',
        body: 'Drought across Ohio Valley reduces lower Mississippi discharge below the ~300 kcfs salt-wedge threshold.',
        source: { label: 'USACE New Orleans District daily flow bulletins' },
      },
      {
        date: '2023-08-03',
        kind: 'sentinel-alert',
        title: 'SENTINEL conductance trajectory anomaly',
        body: 'AquaSSM detects the conductance + chloride trajectory signature 58 days before parish-level advisories.',
        source: { label: 'SENTINEL benchmark, case_studies_real.json' },
        escalateTo: 1,
      },
      {
        date: '2023-09-15',
        kind: 'precursor',
        title: 'Salt wedge crosses RM 64',
        body: 'Wedge front advances past Pointe-a-la-Hache; downstream intakes begin precautionary blending.',
        source: { label: 'USACE Saltwater Wedge Bulletins 2023' },
        escalateTo: 2,
      },
      {
        date: '2023-09-21',
        kind: 'official',
        title: 'Plaquemines Parish water advisory',
        body: 'Parish advises residents not to use tap water for drinking, cooking, or bathing.',
        source: { label: 'Louisiana Department of Health Public Notice' },
        escalateTo: 3,
      },
      {
        date: '2023-10-01',
        kind: 'official',
        title: 'Federal emergency declaration',
        body: 'FEMA approves Louisiana emergency declaration for drinking-water response; affects ~1.2M downstream consumers.',
        source: { label: 'FEMA-3593-EM-LA' },
      },
      {
        date: '2023-10-15',
        kind: 'resolution',
        title: 'Army Corps raises Mississippi River sill',
        body: 'USACE constructs / raises an underwater sill at Myrtle Grove to halt saline intrusion ($28M).',
        source: { label: 'USACE New Orleans District' },
      },
    ],
  },

  /* ---------------------------------------------------------------- */
  jordan_lake_hab_nc: {
    eventKey: 'jordan_lake_hab_nc',
    windowStart: '2022-04-15',
    windowEnd: '2022-09-15',
    sentinelAlertDate: '2022-05-31',
    advisoryDate: '2022-07-15',
    leadSummary:
      'Recurring cyanobacterial bloom in Jordan Lake — a primary drinking-water reservoir for the Research Triangle. SENTINEL flagged 44 days ahead of the state recreational advisory.',
    milestones: [
      {
        date: '2022-05-01',
        kind: 'context',
        title: 'Warm spring + elevated TN/TP loading',
        body: 'Above-normal April rainfall delivers nutrient pulse from Haw River and New Hope tributaries.',
        source: { label: 'NC DEQ Jordan Lake Watershed Assessment' },
      },
      {
        date: '2022-05-31',
        kind: 'sentinel-alert',
        title: 'SENTINEL fused-alert threshold crossed',
        body: 'AquaSSM detects the pre-bloom DO + chlorophyll trajectory 44 days before public notice.',
        source: { label: 'SENTINEL benchmark, case_studies_real.json' },
        escalateTo: 1,
      },
      {
        date: '2022-06-20',
        kind: 'precursor',
        title: 'Sentinel-2 chl-a anomaly visible from satellite',
        body: 'HydroViT identifies a green plume in the Haw River arm of the reservoir.',
        source: { label: 'Copernicus Sentinel-2 L2A imagery' },
        escalateTo: 2,
      },
      {
        date: '2022-07-15',
        kind: 'official',
        title: 'NC DEQ recreational advisory',
        body: 'Posted at Ebenezer Church, Vista Point, and Seaforth recreation areas.',
        source: { label: 'NC DEQ HAB Advisory Map' },
        escalateTo: 3,
      },
      {
        date: '2022-08-10',
        kind: 'first-impact',
        title: 'Cary + Apex drinking-water treatment escalation',
        body: 'Local utilities increase powdered-activated-carbon dosing; finished-water taste / odor complaints rise.',
      },
    ],
  },
};

// Convenience: lookup by record or eventKey, plus enriched milestone (with absolute
// date and sorted order) for UI consumption.
export function getEventTimeline(eventKey: string | undefined): EventTimeline | null {
  if (!eventKey) return null;
  return EVENT_TIMELINES[eventKey] ?? null;
}

// Day-offset relative to advisory (negative = before).
export function daysFromAdvisory(timeline: EventTimeline, iso: string): number {
  return Math.round(
    (new Date(iso + 'T00:00:00Z').getTime() -
      new Date(timeline.advisoryDate + 'T00:00:00Z').getTime()) /
      86400000,
  );
}

// Total span in days (used to drive the slider range).
export function timelineSpanDays(timeline: EventTimeline): number {
  return Math.round(
    (new Date(timeline.windowEnd + 'T00:00:00Z').getTime() -
      new Date(timeline.windowStart + 'T00:00:00Z').getTime()) /
      86400000,
  );
}

// Re-export shiftISO for any caller that needs to do its own date math.
export { shiftISO };
