# Availability note — EPA/NASA CyAN bloom labels (not built)

**Status (checked 2026-10-22): not downloaded.** The task scope assigns CyAN raster work to the satellite workstream. CyAN file downloads also require a NASA Earthdata login, and this workstream may not use logins.

| Item | Finding (2026-10-22) |
|---|---|
| Product | CyAN Cyanobacteria Index (CI_cyano), Sentinel-3 OLCI 300 m daily / 7-day composites, CONUS + Alaska tiles. Product v6 (DOI 10.5067/MERGED-S3/OLCI/L3M/CYAN/CI/6.0). MERIS 2002–2012. DN → CI: `CI = 10^(DN·0.011714 − 4.1870866)`; DN 0 = below detection, 254 = land, 255 = no data (per research note 09 §a.1) |
| File listing | **Open.** `POST https://oceandata.sci.gsfc.nasa.gov/api/cyan_file_search/` with `region, period, product, areaids, sdate, edate, addurl=1, results_as_file=1, wgetflag=1`. It returned URLs such as `https://oceandata.sci.gsfc.nasa.gov/getfile/L2025213.L3m_DAY_CYANAK_CI_cyano_CYAN_AK_300m.tif` |
| File download | **Login required.** `getfile/...` answers `HTTP 302 → https://oceandata.sci.gsfc.nasa.gov/getfile/urs/?next=…`, the NASA Earthdata URS login (free account) |
| EPA program status | The EPA CyAN page says the original project concluded in 2023 and moved to Sentinel-2 MSI in 2024. "While the Sentinel-3 portion of the project will conclude, the data will continue to be made publicly available and reprocessed annually" |
| EPA cyanoHAB forecast | `https://www.epa.gov/habs/hab-forecasts`. Weekly bloom probability for about 2,000 CyAN-resolvable lakes, April–November; bloom = median lake chl-a ≥ 12 µg/L with cyanobacteria dominance (Schaeffer et al. 2024). The page says the dashboards "provide past forecast information, forecast patterns and downloadable data". Dashboard export was not tested here |
| CyANWeb | `https://qed.epa.gov/cyanweb/` loads, but it is a JavaScript application. Its REST paths were not identified without a browser session (the `/rest/api/v1/info` path returns the app shell) |
| Label definition to use (EW-v2 H5) | CyAN lake-week bloom: median CI-derived chl-a ≥ 12 µg/L with cyanobacteria dominance (EPA forecast definition). Excludes clouds/ice, mixed shoreline pixels and small lakes |
| Who builds it | The satellite (M2) workstream, with an Earthdata account the user creates. Credentials go in env/keys file per CONTRIBUTING_V2. Label tables should then follow the vintage rules used for the NOAA SI (`labels_noaa_lake_erie_hab.md`), because CyAN is reprocessed annually |
