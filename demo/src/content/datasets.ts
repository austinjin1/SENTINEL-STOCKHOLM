// SENTINEL — Real public data sources (SENTINEL-DB).
//
// Every source is real and public. The ~390M headline is a RAW, pre-QC count
// dominated by NEON high-frequency sonde data; usable post-QC volume is far
// smaller and we say so. Cross-checked against README.md "Data Infrastructure".
//
// Verified source: /home/bcheng/SENTINEL/README.md (SENTINEL-DB table).

export type DataModality =
  | 'sensor'
  | 'satellite'
  | 'microbial'
  | 'molecular'
  | 'behavioral'
  | 'biological'
  | 'topology'
  | 'discrete';

export interface DatasetEntry {
  id: string;
  /** Source name as published. */
  name: string;
  /** Provider / program. */
  provider: string;
  /** Raw record count as a display string (e.g. "351.7M"). */
  records: string;
  /** Numeric record count for sorting/aggregation where meaningful. */
  recordsNum?: number;
  /** Record type / description. */
  type: string;
  /** Which SENTINEL modality this feeds. */
  modality: DataModality;
  /** Public, accessible URL or access route. */
  access: string;
}

// Order roughly by raw record count (largest first), matching README table.
export const DATASETS: DatasetEntry[] = [
  {
    id: 'neon',
    name: 'NEON Aquatic',
    provider: 'National Ecological Observatory Network',
    records: '351.7M',
    recordsNum: 351_700_000,
    type: 'Continuous high-frequency sonde data (34 sites)',
    modality: 'sensor',
    access: 'data.neonscience.org',
  },
  {
    id: 'epa_wqp',
    name: 'EPA Water Quality Portal',
    provider: 'US EPA / USGS / NWQMC',
    records: '18.27M',
    recordsNum: 18_270_000,
    type: 'Discrete water-quality samples',
    modality: 'discrete',
    access: 'waterqualitydata.us',
  },
  {
    id: 'grqa',
    name: 'GRQA v1.3',
    provider: 'Global River Water Quality Archive',
    records: '17.99M',
    recordsNum: 17_990_000,
    type: 'Harmonized global river quality',
    modality: 'discrete',
    access: 'zenodo.org (GRQA v1.3)',
  },
  {
    id: 'wqp_hab',
    name: 'WQP (cyanotoxins, nutrients)',
    provider: 'Water Quality Portal',
    records: '755K',
    recordsNum: 755_000,
    type: 'HAB-related discrete samples (microcystin, nutrients)',
    modality: 'discrete',
    access: 'waterqualitydata.us',
  },
  {
    id: 'usgs_biodata',
    name: 'USGS BioData',
    provider: 'US Geological Survey',
    records: '701K',
    recordsNum: 701_000,
    type: 'Invertebrate + fish + WQP biological records',
    modality: 'biological',
    access: 'aquatic.biodata.usgs.gov',
  },
  {
    id: 'ecotox',
    name: 'EPA ECOTOX',
    provider: 'US EPA',
    records: '1.23M',
    recordsNum: 1_230_000,
    type: 'Ecotoxicology dose-response endpoints',
    modality: 'behavioral',
    access: 'cfpub.epa.gov/ecotox',
  },
  {
    id: 'canada_wqp',
    name: 'Canada WQP',
    provider: 'Environment and Climate Change Canada',
    records: '787K',
    recordsNum: 787_000,
    type: 'Discrete water-quality samples',
    modality: 'discrete',
    access: 'open.canada.ca',
  },
  {
    id: 'noaa_habs',
    name: 'NOAA HABs (ERDDAP chl-a)',
    provider: 'NOAA CoastWatch',
    records: '146K',
    recordsNum: 146_000,
    type: 'VIIRS + MODIS chlorophyll-a',
    modality: 'satellite',
    access: 'coastwatch.pfeg.noaa.gov/erddap',
  },
  {
    id: 'usgs_nwis',
    name: 'USGS NWIS',
    provider: 'US Geological Survey',
    records: '364K sequences',
    recordsNum: 364_000,
    type: 'Real-time sensor time series',
    modality: 'sensor',
    access: 'waterdata.usgs.gov/nwis',
  },
  {
    id: 'sentinel2',
    name: 'Sentinel-2',
    provider: 'ESA / Microsoft Planetary Computer',
    records: '2,986 tiles',
    recordsNum: 2986,
    type: 'Multispectral satellite imagery (L2A)',
    modality: 'satellite',
    access: 'planetarycomputer.microsoft.com',
  },
  {
    id: 'emp',
    name: 'EMP 16S rRNA',
    provider: 'Earth Microbiome Project',
    records: '20,288',
    recordsNum: 20_288,
    type: 'Microbiome OTU tables',
    modality: 'microbial',
    access: 'earthmicrobiome.org',
  },
  {
    id: 'ncbi_geo',
    name: 'NCBI GEO',
    provider: 'NCBI',
    records: '12 zebrafish series',
    type: 'Aquatic transcriptomics (RNA-seq / microarray)',
    modality: 'molecular',
    access: 'ncbi.nlm.nih.gov/geo',
  },
  {
    id: 'nhdplus',
    name: 'NHDPlus V2',
    provider: 'US EPA / USGS',
    records: '561 sites, 338 edges',
    type: 'Stream-network topology',
    modality: 'topology',
    access: 'epa.gov/waterdata/nhdplus',
  },
  {
    id: 'gbif',
    name: 'GBIF Freshwater',
    provider: 'Global Biodiversity Information Facility',
    records: '2,355',
    recordsNum: 2355,
    type: 'Bioindicator species occurrences',
    modality: 'biological',
    access: 'gbif.org',
  },
];

/** Headline aggregate figures, presented honestly (raw vs post-QC). */
export const DATA_SUMMARY = {
  rawRecords: '~390M',
  rawRecordsNote:
    'Raw, pre-QC count (~85 GB) dominated by NEON high-frequency sonde data (351.7M raw). Usable post-QC volume is far smaller.',
  postQcRecords: '~52M',
  postQcNote:
    'After NEON quality-flag filtering (351.7M → ~52M). NEON raw DO contained extreme outliers (max 4.95 billion) removed by FinalQF=0 filtering.',
  nSources: 14,
  nCountries: 105,
  nSites: '94,000+',
} as const;

/** Fast lookup by dataset id. */
export const DATASETS_BY_ID: Record<string, DatasetEntry> = Object.fromEntries(
  DATASETS.map((d) => [d.id, d]),
);
