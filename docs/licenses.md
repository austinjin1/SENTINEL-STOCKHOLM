# License register (data sources and models)

Every data source and pretrained model used by SENTINEL v2, with its license and any
redistribution constraint. Non-commercial models are used only as research baselines and are
not served by the public platform.

| Item | Kind | License / terms | Redistribution | Used in |
|---|---|---|---|---|
| USGS NWIS / Water Data APIs | data | US public domain | yes | M1, M6, EW-v2, D* |
| NEON data products | data | CC BY 4.0 (since 2026-06-30; previously CC0) | with attribution | M1, M3, M5, Fuse, M7 |
| NEON aquatic pull 2026-10-24 (DP1.20141.002/.001, DP1.20282.001, DP1.20093.001, DP1.20288.001, DP1.20033.001, DP1.20281.001, DP1.20264.001, DP1.20053.001, DP1.20016.001, DP4.00130.001, DP1.20120.001, DP1.20163.001, DP1.20107.001, DP1.20219.001, DP1.20278.001; RELEASE-2026 + flagged PROVISIONAL; 16S taxon tables from storage.neonscience.org) | data | CC BY 4.0 (NEON data policy; cite product DOIs of RELEASE-2026) | with attribution; provisional data not citable | M1, M3, M5, Fuse (`neon_*` datasets) |
| Water Quality Portal | data | public (multi-agency) | yes, cite providers | M2, D1, D2, Fuse |
| Sentinel-2 L2A (Copernicus) | data | Copernicus free & open | yes, attribution | M2, M7 |
| Landsat C2 | data | US public domain | yes | M2 |
| NAIP | data | US public domain | yes | M7 |
| GLORIA | data | CC BY 4.0 | with attribution | M2, M7 |
| ADORE | data | see dataset record | [verify] | M4 |
| EPA NARS (NRSA/NLA) | data | US public domain | yes | M5, D5, Fuse |
| UK EA ecology archive | data | OGL v3.0 | with attribution | M5 |
| Chronos-2 (amazon/chronos-2) | model | Apache-2.0 | yes | M1 |
| Other TSFMs, EO/molecular foundation models | model | [to fill at first use] | | |
| AquaCorr-Global v1.0 (Zenodo 21927415; Ghazi Zadeh Hashemi et al. 2026) | data | CC BY 4.0 (Zenodo record, checked 2026-10-22) | with attribution | M2 |
| Sentinel-2 L2A via Element 84 Earth Search (`sentinel-2-c1-l2a`) and Planetary Computer (`sentinel-2-l2a`) | data | Copernicus Sentinel data: free, full and open ("contains modified Copernicus Sentinel data") | yes, with attribution | M2, M7, Fuse |
| JRC Global Surface Water v1.3 (Pekel et al. 2016; Planetary Computer `jrc-gsw`) | data | free of charge, without restriction of use (EC JRC); cite doi:10.1038/nature20584 | yes, with citation | M2, M7 |
| ESA Sentinel-2 Spectral Response Functions v3.2 (COPE-GSEG-EOPG-TN-15-0007) | data/doc | ESA Copernicus documentation, public | yes, with citation | M2, M7 (GLORIA band simulation) |
| USGS Landsat 8 OLI / Landsat 9 OLI-2 band-average RSR (spectral characteristics viewer) | data | US public domain | yes | M2 (GLORIA band simulation) |
| MARIDA (Zenodo 5151941; Kikaki et al. 2022) | data | CC BY 4.0 (Zenodo record) | with attribution | M7 hazard heads |
| MADOS (Zenodo 10664073; Kikaki et al. 2024) | data | CC BY 4.0 (Zenodo record) | with attribution | M7 hazard heads |
| Oil Spill Drone / annotated port oil-spill RGB images (Zenodo 10555314; De Kerf et al. 2024) | data | CC BY 4.0 (Zenodo record) | with attribution | M7 hazard heads |
| Chesapeake Land Cover (LILA; Robinson et al. 2019) | data | CDLA-Permissive-1.0 (labels; LILA metadata); NAIP/Landsat public domain; building footprints ODbL (not downloaded) | yes, with attribution | M7 water masks |
| FloodNet supervised v1.0 (Rahnemoonfar et al. 2021) | data | CDLA-Permissive-1.0 (GitHub README) | yes, with attribution | M7 water masks |
| LADOS (Gkountakos et al. 2025) | data | article CC BY 4.0; images only on Roboflow Universe (account required), compiled from third-party sources | not downloaded | — |
| GBIF Backbone Taxonomy | data | CC BY 4.0 | with attribution | M5 taxonomy harmonisation |
| Delaware DNREC Non-tidal Fish Kill Reports (data.delaware.gov `8pfi-yck7`, retrieved 2026-10-25) | data | Public Domain (portal licence field) | yes | D3 KillAlert (`labels_delaware_fishkill`) |
| USGS Watershed Boundary Dataset HUC12 (WBD MapServer layer 6, retrieved 2026-10-24) | data | US public domain | yes | D6 MusselGuard (`wbd_huc12`) |
| US Census 2010 county gazetteer (population, internal points) | data | US Government work, public domain | yes | D6 MusselGuard (`census_county`) |
