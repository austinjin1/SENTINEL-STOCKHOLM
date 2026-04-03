# SENTINEL v2 — Data Pipeline Plan

## Current State Audit (10 files, ~4,800 lines)

### What exists and is solid:

| File | Lines | Quality | v2 Status |
|------|-------|---------|-----------|
| `satellite/download.py` | 463 | Excellent. GEE + Planetary Computer, AOI, DownloadRequest dataclasses, search/export/wait for S2 + Landsat thermal. Retry logic. | **KEEP + EXTEND** — needs Sentinel-3 OLCI + JRC water mask |
| `satellite/preprocessing.py` | 404 | Excellent. 5 spectral indices (NDCI, FAI, NDTI, MNDWI, Oil), tile grid, resize_tile, TemporalBuffer with disk persistence. | **KEEP + EXTEND** — needs S3 bands, multi-resolution fusion prep, 16 output params, water-pixel extraction for HydroViT MAE pretraining |
| `sensor/download.py` | 382 | Excellent. Station discovery by state, legacy+waterdata download, chunked 120-day requests, retry, Parquet output, station catalog JSON. | **KEEP + EXTEND** — needs EPA WQP, EU Waterbase, GEMS/Water ingest, irregular timestamp preservation for AquaSSM |
| `sensor/preprocessing.py` | 377 | Excellent. Quality flags, 15-min resampling, gap filling, rolling 90-day z-score, sliding window extraction. | **PARTIALLY REWRITE** — AquaSSM needs raw irregular timestamps (no resampling!), keep normalization but add irregular-time tensor format |
| `microbial/download.py` | 460 | Excellent. NARS ZIP download, Redbiom/Qiita EMP, NCBI SRA search+manifest, fasterq-dump script generation. | **KEEP + EXTEND** — needs MGnify download, citizen-science FreshWater Watch |
| `microbial/preprocessing.py` | 503 | Excellent. DADA2+Deblur via QIIME2 CLI, UCHIME chimera removal, SILVA 138 classification, CLR transform, abundance filtering, BIOM loading, metadata linkage. | **KEEP + EXTEND** — needs Aitchison-native output format, DNABERT-S sequence extraction, zero-inflation annotations |
| `molecular/download.py` | 547 | Excellent. GEO search+download via GEOparse, CTD bulk 4 datasets, ArrayExpress search+download, multi-species search. | **KEEP + EXTEND** — needs AOP-Wiki hierarchy download, ortholog mapping tables |
| `molecular/preprocessing.py` | 523 | Excellent. TPM normalization, log2 transform, quantile normalization, RMA, ComBat batch correction, 200-gene panel (7 pathways), pathway activation labeling, cross-platform integration. | **KEEP + EXTEND** — needs Reactome pathway hierarchy export, AOP graph structure for ToxiGene |
| `ecotox/download.py` | 395 | Excellent. Bulk ZIP download with progress, pipe-delimited parser, freshwater/aquatic filter, endpoint filter, key field extraction. | **KEEP AS-IS** |
| `ecotox/preprocessing.py` | 412 | Excellent. Unit conversion (mg/L, hours), missing data handling, species attribute imputation, chemical-stratified 80/10/10 split. | **KEEP AS-IS** |

### What's MISSING entirely:

1. **`sentinel/data/sentinel_db/`** — The entire SENTINEL-DB harmonization layer (6 new files)
2. **`sentinel/data/alignment/`** — Geographic alignment (empty, only __init__.py)
3. **`sentinel/data/case_studies/`** — Historical event collection (empty, only __init__.py)
4. **`sentinel/data/behavioral/`** — Behavioral video data (directory doesn't exist)
5. **`sentinel/data/satellite/download.py`** — Sentinel-3 OLCI support
6. **`sentinel/data/satellite/preprocessing.py`** — Water-pixel extraction for HydroViT
7. **`sentinel/data/sensor/preprocessing.py`** — Irregular-time format for AquaSSM
8. International data sources (EU Waterbase, GEMS/Water, CPCB India)

---

## Detailed Gap Analysis per Amended Plan Section

### Section 2: SENTINEL-DB (entirely new)

**2.2 Unified Ontology Layer** — Map 10,000+ parameter names across EPA/USGS/EU/GEMS to ~500 canonical params:
- `sentinel/data/sentinel_db/ontology.py`
  - Exact match lookup table (JSON)
  - Fuzzy string matching (rapidfuzz or fuzzywuzzy)
  - Unit harmonization rules
  - Source: existing v1 `PARAMETER_CODES` in sensor/download.py has 6 USGS codes — need to expand to cover EPA WQP's ~300 DO synonyms etc.

**2.2 Spatio-Temporal Alignment Engine** — H3 hexagonal grid:
- `sentinel/data/sentinel_db/spatial.py`
  - H3 indexing at resolution 8 (~0.74 km² hexagons, good for 500m co-registration)
  - Cross-source dedup within same H3 cell + ±3h temporal window
  - Satellite pixel co-registration to in-situ stations

**2.2 Quality Tiers** — Q1-Q4:
- `sentinel/data/sentinel_db/quality.py`
  - Q1: certified lab (USGS, EPA formal)
  - Q2: automated sensors with calibration
  - Q3: citizen science (default entry)
  - Q4: derived/estimated
  - Promotion logic (Q3→Q2 via satellite + sensor validation)

**2.2 Modality Linking** — Cross-modality co-location:
- `sentinel/data/sentinel_db/linking.py`
  - For each in-situ record: find satellite pixel (±500m, ±3h)
  - Find co-located microbiome samples (±10km, ±30d)
  - Cross-reference ECOTOX for species in that water body
  - Output: LinkedRecord with all available modalities

**2.2 Multi-Source Ingest**:
- `sentinel/data/sentinel_db/ingest.py`
  - EPA Water Quality Portal (https://www.waterqualitydata.us/) — uses `dataretrieval.wqp` module
  - EU Waterbase — bulk CSV download from EEA
  - GEMS/Water (GEMStat) — bulk download
  - Citizen science (FreshWater Watch) — API or bulk download
  - GRQA harmonized dataset (Zenodo DOI: 10.5281/zenodo.7056647)

**2.2 Schema**:
- `sentinel/data/sentinel_db/schema.py`
  - Pydantic models: WaterQualityRecord, SatelliteObservation, MicrobialSample, etc.
  - LinkedMultimodalRecord

### Section 3: AquaSSM data needs (sensor preprocessing changes)

The current `sensor/preprocessing.py` resamples to regular 15-min intervals. AquaSSM needs:
- **Raw irregular timestamps preserved** (the actual observation times, not binned)
- **Time gaps as explicit features** (Δt between consecutive observations)
- **Multi-scale decomposition prep** — no resampling, just normalization + gap annotation
- Keep the z-score normalization (rolling 90-day) but apply to raw observations
- Keep quality flag filtering
- Output format: list of (timestamp, values, Δt) tuples, not fixed-size windows

Changes to `sensor/preprocessing.py`:
- Add `preprocess_station_irregular()` alongside existing `preprocess_station()`
- Produce both formats: regular (for backward compat) + irregular (for AquaSSM)

### Section 4: HydroViT data needs (satellite preprocessing changes)

The current satellite pipeline handles S2 only. HydroViT needs:
- **Sentinel-3 OLCI download** — 21 bands, 300m resolution, daily revisit
  - GEE collection: `COPERNICUS/S3/OLCI`
  - Or Copernicus Marine Service
- **JRC Global Surface Water mask** — extract water-only pixels for pretraining
  - GEE: `JRC/GSW1_4/GlobalSurfaceWater`
- **Multi-resolution co-registration** — S2 (10m) + S3 (300m) for the same location
- **Water-pixel patch extraction** — for MAE pretraining (500M patches target)
- **16 parameter labels** — chl-a, turbidity, Secchi, CDOM, TSS, TN, TP, DO, NH3, NO3, pH, temp, phycocyanin, oil_prob, aCDOM, PAI
- **In-situ co-registration** — satellite pixels matched to SENTINEL-DB in-situ measurements (±3h, 500m)

Changes to `satellite/download.py`:
- Add `Sentinel3OLCIDownloader` class
- Add JRC water mask download

Changes to `satellite/preprocessing.py`:
- Add water-pixel extraction pipeline
- Add multi-resolution alignment (S2↔S3)
- Add in-situ label co-registration
- Keep existing spectral indices

### Section 5: MicroBiomeNet data needs

Current microbial pipeline is mostly fine. Additions needed:
- **DNABERT-S sequence extraction** — for each ASV, extract the representative 16S sequence and store alongside CLR vectors (for sequence encoding in model)
- **Zero-inflation annotation** — classify zeros as structural vs sampling
- **Aitchison-native output** — already have CLR, but need to also store raw compositions for simplex operations
- **MGnify download** — additional aquatic metagenome source
- **Temporal metadata** — ensure sample timestamps preserved for simplex ODE

### Section 7: BioMotion data needs (entirely new)

- `sentinel/data/behavioral/download.py`
  - DaphBASE behavioral datasets
  - Published Daphnia/fish ethology video datasets
  - EPA ECOTOX behavioral endpoint videos
  - Simulated behavioral data generation (for pretraining)

- `sentinel/data/behavioral/preprocessing.py`
  - SLEAP pose estimation pipeline invocation
  - Keypoint extraction (12 Daphnia, 8 mussel, 22 fish)
  - Behavioral feature computation (velocity, acceleration, turning angle, phototaxis, schooling)
  - 30Hz→1Hz downsampling
  - Trajectory normalization

### Geographic Alignment (empty, needs full implementation)

- `sentinel/data/alignment/geographic.py`
  - H3 hexagonal indexing (replaces HUC-8 from v1 plan)
  - HydroLAKES integration for lake boundaries
  - HydroSHEDS/HydroBASINS for watershed delineation
  - NLDI (existing in v1 plan) for US watershed navigation
  - Global coverage (not just US HUC-8)

### Case Studies (empty, needs full implementation)

- `sentinel/data/case_studies/collector.py`
  - 10+ historical contamination events
  - Multi-modal data collection per event
  - Timeline documentation (event onset, detection, notification)
  - EPA emergency response cross-referencing
- `sentinel/data/case_studies/events.py`
  - Event definitions (location, dates, contaminant, data availability)

---

## Agent Dispatch Plan (7 parallel agents)

### Agent 1: SENTINEL-DB Core (6 new files)
**Files**: All of `sentinel/data/sentinel_db/`
- `__init__.py`
- `schema.py` — Pydantic data models (WaterQualityRecord, SatelliteObservation, MicrobialSample, TranscriptomicSample, BehavioralRecording, LinkedMultimodalRecord, QualityTier enum)
- `ontology.py` — Parameter name ontology (10K→500 mapping, fuzzy matching, unit harmonization)
- `spatial.py` — H3 hexagonal indexing, cross-source dedup, satellite-to-station co-registration
- `quality.py` — Quality tier assignment (Q1-Q4), promotion rules, cross-source consistency scoring
- `linking.py` — Cross-modality linking engine (find satellite pixels, microbiome samples, ECOTOX records for each in-situ observation)
- `ingest.py` — EPA Water Quality Portal, EU Waterbase, GEMS/Water, FreshWater Watch, GRQA downloaders

**Context needed**: The existing `sensor/download.py` has USGS NWIS patterns to follow. Use `dataretrieval.wqp` for EPA WQP. H3 library: `h3-py`. Pydantic v2 models.

### Agent 2: Satellite Pipeline v2 (extend 2 existing files)
**Files**: Modify `satellite/download.py` + `satellite/preprocessing.py`
- Add `Sentinel3OLCIDownloader` class to download.py (GEE: `COPERNICUS/S3/OLCI`, 21 bands)
- Add JRC Global Surface Water mask download
- Add `extract_water_pixels()` to preprocessing.py — use JRC mask to extract water-only patches for MAE pretraining
- Add `MultiResolutionAligner` — co-register S2 (10m) and S3 (300m) tiles for same location
- Add `InSituCoRegistration` — match satellite pixels to SENTINEL-DB in-situ records (±3h, ±500m)
- Add 16-parameter label structure for HydroViT training data

**Context needed**: Existing satellite/download.py GEE patterns. The existing TemporalBuffer and tile grid code stays. Agent should ADD new classes, not replace existing ones.

### Agent 3: Sensor Pipeline v2 (extend preprocessing)
**Files**: Modify `sensor/preprocessing.py`, potentially `sensor/download.py`
- Add `preprocess_station_irregular()` — output raw (timestamp, values, Δt) tuples for AquaSSM
  - No resampling — preserve actual observation times
  - Compute Δt (inter-observation gaps) as explicit features
  - Same z-score normalization but on raw irregular observations
  - Output: list of named tuples (timestamp_seconds, parameter_values[6], delta_t_seconds, quality_mask[6])
- Add to download.py: EPA Water Quality Portal download via `dataretrieval.wqp`
- Add to download.py: EU Waterbase + GEMS/Water bulk download support
- Keep all existing regular-interval preprocessing for backward compat

**Context needed**: Existing sensor code is excellent. Agent should ADD new functions alongside existing ones. The `AquaSSM` model needs irregular timestamps and explicit gap features.

### Agent 4: Microbial Pipeline v2 (extend both files)
**Files**: Modify `microbial/download.py` + `microbial/preprocessing.py`
- Add MGnify aquatic metagenome download to download.py
- Add `extract_representative_sequences()` to preprocessing.py — for each ASV, extract the 16S rRNA sequence string and save alongside CLR vectors (needed for DNABERT-S encoding in MicroBiomeNet)
- Add `annotate_zero_inflation()` — classify each zero in the ASV table as structural (taxon absent from environment) vs sampling (below detection limit), using co-occurrence patterns
- Add `export_simplex_format()` — raw compositions (not CLR) for simplex ODE operations
- Add temporal metadata preservation — ensure sample collection timestamps are carried through all processing steps
- Keep all existing DADA2/CLR/BIOM processing

**Context needed**: Existing microbial code handles QIIME2 CLI well. DNABERT-S expects DNA sequences as input strings. Zero-inflation annotation uses co-occurrence: if taxon A is never observed at site X across 10+ samples, it's structural; if observed sometimes, zeros are sampling zeros.

### Agent 5: Molecular Pipeline v2 (extend both files)
**Files**: Modify `molecular/download.py` + `molecular/preprocessing.py`
- Add AOP-Wiki hierarchy download to download.py:
  - AOP-Wiki XML/JSON API: https://aopwiki.org/
  - Extract: gene → pathway → biological process → adverse outcome mapping
  - Parse Reactome pathway hierarchy (https://reactome.org/download-data)
- Add ortholog mapping tables download:
  - NCBI HomoloGene or Ensembl BioMart ortholog data
  - Map genes across zebrafish ↔ Daphnia ↔ fathead minnow ↔ rainbow trout ↔ human
  - Output: JSON/TSV of ortholog groups
- Add `build_hierarchy_graph()` to preprocessing.py — construct the P-NET-style gene→pathway→process→AOP sparse adjacency matrices for ToxiGene
- Add `build_ortholog_alignment()` — shared gene embedding alignment across species
- Keep all existing TPM/RMA/ComBat/panel code

**Context needed**: Existing molecular code is excellent. ToxiGene needs sparse adjacency matrices encoding which genes connect to which pathways. AOP-Wiki provides the hierarchy.

### Agent 6: Behavioral Pipeline (4 new files)
**Files**: All of `sentinel/data/behavioral/`
- `__init__.py`
- `download.py` — DaphBASE, published ethology datasets, EPA ECOTOX behavioral videos
  - DaphBASE: https://doi.org/10.5281/zenodo.XXXXXXX (find actual Zenodo DOI)
  - Published fish tracking datasets (DeepLabCut/SLEAP format)
  - Simulated trajectory generation for pretraining (random walk + realistic dynamics)
- `preprocessing.py` — SLEAP pose estimation pipeline
  - Invoke SLEAP CLI for pose estimation (12 Daphnia keypoints, 8 mussel, 22 fish)
  - Extract behavioral features: velocity, acceleration, turning angle, inter-individual distance, phototactic index
  - 30Hz raw → 1Hz summary statistics (mean, std, max, percentiles)
  - Trajectory normalization (center, scale, align heading)
  - Output: behavioral feature time series as numpy arrays
- `trajectory.py` — Trajectory data structures and augmentation
  - Trajectory dataclass (species, keypoints, features, timestamps)
  - Data augmentation: temporal jittering, spatial rotation, speed scaling
  - Normal vs anomalous trajectory labeling

**Context needed**: SLEAP (https://sleap.ai/) is installed via `pip install sleap`. Pose estimation runs on video files. Behavioral features are computed from pose sequences. The diffusion pretraining in BioMotion will corrupt these trajectories and learn to denoise them.

### Agent 7: Alignment + Case Studies (4 new files)
**Files**: `alignment/geographic.py` + `case_studies/collector.py` + `case_studies/events.py`
- `alignment/geographic.py`:
  - H3 hexagonal indexing (resolution 8) for all monitoring locations
  - HydroLAKES/HydroSHEDS integration for watershed delineation
  - NLDI navigation for US sites (adapt from v1 plan)
  - Cross-modality site matching: for each site, find all data sources within configurable radius
  - Global coverage (not just US)
- `case_studies/events.py`:
  - Dataclass definitions for 10+ historical contamination events
  - Event metadata: name, year, location bbox, contaminant class, data availability, documented timeline
  - Events: Gold King Mine 2015, Lake Erie HAB annual, Toledo 2014, Dan River 2014, Elk River 2014, Houston Ship Channel recurring, Flint MI 2014-2019, Gulf of Mexico dead zone, Chesapeake Bay blooms, East Palestine 2023
- `case_studies/collector.py`:
  - For each event: pull USGS sensor data (50km radius), satellite imagery (cloud-free), EPA records
  - Data window: 30 days before → 60 days after onset
  - Cross-reference EPA emergency response records
  - Save as structured event packages

**Context needed**: H3 library: `import h3`. HydroLAKES shapefile from https://www.hydrosheds.org/products/hydrolakes. Existing sensor and satellite downloaders should be reused for case study data collection.

---

## Checklist

### SENTINEL-DB (Agent 1)
- [ ] `sentinel/data/sentinel_db/__init__.py`
- [ ] `sentinel/data/sentinel_db/schema.py` — Pydantic data models
- [ ] `sentinel/data/sentinel_db/ontology.py` — 10K→500 parameter mapping + fuzzy match
- [ ] `sentinel/data/sentinel_db/spatial.py` — H3 indexing + dedup
- [ ] `sentinel/data/sentinel_db/quality.py` — Q1-Q4 tier assignment
- [ ] `sentinel/data/sentinel_db/linking.py` — Cross-modality linking
- [ ] `sentinel/data/sentinel_db/ingest.py` — EPA WQP, EU Waterbase, GEMS/Water, citizen science

### Satellite v2 (Agent 2)
- [ ] Add `Sentinel3OLCIDownloader` to `satellite/download.py`
- [ ] Add JRC water mask download to `satellite/download.py`
- [ ] Add `extract_water_pixels()` to `satellite/preprocessing.py`
- [ ] Add `MultiResolutionAligner` to `satellite/preprocessing.py`
- [ ] Add `InSituCoRegistration` to `satellite/preprocessing.py`
- [ ] Add 16-parameter label structure

### Sensor v2 (Agent 3)
- [ ] Add `preprocess_station_irregular()` to `sensor/preprocessing.py`
- [ ] Add EPA WQP download to `sensor/download.py`
- [ ] Add EU Waterbase download to `sensor/download.py`
- [ ] Add GEMS/Water download to `sensor/download.py`

### Microbial v2 (Agent 4)
- [ ] Add MGnify download to `microbial/download.py`
- [ ] Add `extract_representative_sequences()` to `microbial/preprocessing.py`
- [ ] Add `annotate_zero_inflation()` to `microbial/preprocessing.py`
- [ ] Add `export_simplex_format()` to `microbial/preprocessing.py`
- [ ] Add temporal metadata preservation

### Molecular v2 (Agent 5)
- [ ] Add AOP-Wiki hierarchy download to `molecular/download.py`
- [ ] Add Reactome pathway download to `molecular/download.py`
- [ ] Add ortholog mapping tables download to `molecular/download.py`
- [ ] Add `build_hierarchy_graph()` to `molecular/preprocessing.py`
- [ ] Add `build_ortholog_alignment()` to `molecular/preprocessing.py`

### Behavioral Pipeline (Agent 6)
- [ ] `sentinel/data/behavioral/__init__.py`
- [ ] `sentinel/data/behavioral/download.py` — DaphBASE, ethology datasets, simulated data
- [ ] `sentinel/data/behavioral/preprocessing.py` — SLEAP, feature extraction, downsampling
- [ ] `sentinel/data/behavioral/trajectory.py` — Data structures, augmentation, labeling

### Alignment + Case Studies (Agent 7)
- [ ] `sentinel/data/alignment/geographic.py` — H3, HydroLAKES, NLDI, global
- [ ] `sentinel/data/case_studies/events.py` — Event definitions (10+ events)
- [ ] `sentinel/data/case_studies/collector.py` — Multi-modal data collection per event

---

## Dependency Graph

```
SENTINEL-DB (Agent 1)
    ├── ontology.py ← needed by all ingest pipelines
    ├── spatial.py ← needed by linking.py and alignment
    ├── schema.py ← needed by everything
    └── ingest.py ← uses existing download code from sensor/satellite/microbial

Satellite v2 (Agent 2) ← independent, extends existing code
Sensor v2 (Agent 3) ← independent, extends existing code
Microbial v2 (Agent 4) ← independent, extends existing code
Molecular v2 (Agent 5) ← independent, extends existing code
Behavioral (Agent 6) ← independent, entirely new

Alignment (Agent 7) ← depends on spatial.py from Agent 1
Case Studies (Agent 7) ← depends on existing downloaders + alignment
```

All 7 agents CAN run in parallel because:
- Agents 2-6 only ADD to existing files (no conflicts)
- Agent 1 creates new files in a new directory
- Agent 7 creates new files in empty directories
- The dependency on Agent 1's spatial.py is soft — Agent 7 can import h3 directly

## Estimated Output
- ~25 new/modified files
- ~6,000-8,000 new lines of Python
- Total data pipeline: ~12,000-13,000 lines across ~35 files
