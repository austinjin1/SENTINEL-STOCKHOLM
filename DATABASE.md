# SENTINEL-DB — Data Acquisition Guide

SENTINEL is trained entirely on **public** data. No dataset is bundled in this
repo (raw data is git-ignored); every source is downloaded and processed with the
scripts in `scripts/data_acquisition/` and `scripts/preprocessing/`.

All commands are run from the repository root with the project conda env and
`PYTHONNOUSERSITE=1` (see README "Setup"). Raw downloads land in `data/raw/…`
and processed, training-ready files in `data/processed/…`.

```bash
export PYTHONNOUSERSITE=1
```

## Sources and scripts

| Source (paper Table 1) | Modality | Download | Preprocess → training data |
|------------------------|----------|----------|----------------------------|
| USGS NWIS | Sensor | `data_acquisition/discover_usgs_stations.py`, `data_acquisition/download_sensor.py` (`download_sensor_fast.py` for daily values) | `preprocessing/process_usgs_to_training.py`, `preprocessing/preprocess_sensor.py` |
| NEON Aquatic | Sensor | NEON portal (DP1.20288.001); stream large files with `data_acquisition/compress_neon_streaming.py` | `preprocessing/expand_neon_and_others.py` |
| GRQA v1.3 | Sensor | `data_acquisition/download_grqa.py` | `data_acquisition/ingest_grqa.py` |
| EPA WQP | Sensor | `data_acquisition/download_epa_wqp.py` | — |
| Sentinel-2 L2A | Satellite | `data_acquisition/download_satellite_real.py`, `download_nwis_s2_pairs.py`, `download_drone_s2_pairs.py` (bulk: `download_s2_massive.py`) | `data_acquisition/coregister_satellite_wq.py`, `preprocessing/expand_satellite_wq_pairs.py` |
| EMP 16S rRNA | Microbial | `data_acquisition/download_emp_microbiome.py`, `download_microbial_data.py` | `preprocessing/expand_emp_dataset.py`, `expand_microbiomenet_data.py` |
| NCBI GEO | Molecular | `data_acquisition/download_geo_transcriptomics.py` | `preprocessing/process_geo_zebrafish.py`, `prepare_geo_for_toxigene.py` |
| EPA ECOTOX | Molecular / Behavioral | ECOTOX ASCII bulk export (epa.gov/ecotox) into `data/raw/ecotox/` | `preprocessing/process_ecotox.py`, `process_ecotox_behavioral.py` |
| NOAA HABs / ERDDAP | Satellite / labels | `data_acquisition/download_noaa_habs.py`, `download_habs_wqp.py`, `download_habs_wqp_cyanotoxins.py` | — |
| USGS BioData | Biology | `data_acquisition/download_usgs_biodata.py` | `preprocessing/preprocess_biodata.py` |
| NHDPlusV2 | Stream topology | `data_acquisition/download_nhdplus.py` | — |
| GBIF | Biology | `data_acquisition/download_gbif.py` | — |

Supplementary sources also included in SENTINEL-DB:
`download_eu_waterbase.py`, `download_gems_water.py`, `download_hydrolakes.py`,
`download_modis_ocean_color.py`, `download_sentinel3.py`, `download_who_jmp.py`.

## One-shot download

`scripts/data_acquisition/download_all.py` chains the core downloads. Review and
edit the source list / credentials at the top before running:

```bash
PYTHONNOUSERSITE=1 python scripts/data_acquisition/download_all.py
```

## Quality-control notes (must match training)

- **NEON**: keep only `finalQF == 0` records (raw DO can spike to billions; filtered mean ≈ 9.52 mg/L).
- **USGS**: drop sentinel values (`-999999`); screen instrument drift with a rolling-σ threshold.
- **EMP 16S**: rarefy to 10,000 reads per sample to normalize sequencing depth.
- **Satellite**: match the closest cloud-free Sentinel-2 overpass within 48 h of the in-situ sample; 500 m buffer for sensor–satellite pairs.
- Missing parameters are **flagged, not imputed** — the fusion module gates missing modalities at inference.

## Per-modality processed outputs

```
data/processed/
  sensor/            # AquaSSM sequences (USGS NWIS, 15-min)
  satellite/         # HydroViT / HydroDenseNet S2 tiles + WQ pairs
  microbial/         # MicroBiomeNet 16S OTU vectors
  molecular/         # ToxiGene GEO expression + ECOTOX labels
  behavioral/        # BioMotion ECOTOX Daphnia dose-response
  biology/           # species-health + disease BioData
  hydrology/         # NHDPlus stream graph (stream GNN)
```
