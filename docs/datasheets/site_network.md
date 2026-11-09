# Datasheet — `site_network` (NHDPlus V2 COMID links + flow-connected site edges)

**Files**
- `$SENTINEL_DATA/processed/site_network/nodes.parquet`: one row per site.
- `edges.parquet`: directed flow connections.
- `raw/nldi/comid_links.parquet`: every lookup result, by method.
- `interim/nldi/navigation_raw.parquet`: raw NLDI navigation hits.
- `raw/nhdplus_vaa/nhdplus_vaa_subset.parquet`: NHDPlus V2 VAA, 2.69 M flowlines, 20 columns.
- NLDI response cache (gzip JSON): `raw/nldi/cache/`.
- Manifest: `manifests/site_network.json`.

**Code:** `sentinel/data/sources/nldi.py`, `sentinel/data/sites.py` (attach_vaa,
vaa_downstream_edges, combine_edges, flow_components) · `scripts/data/download_nldi.py`
**Purpose:** HUC4 station folds with **flow-connected stations kept together**
(RESEARCH_PLAN §5.1, §14.2; docs/research/01 d.2), the COMID join key for StreamCat, and
upstream/downstream pairs for propagation studies (docs/research/06).

## Sources
- **USGS NLDI** `https://api.water.usgs.gov/nldi/linked-data` (moved from `labs.waterdata.usgs.gov`,
  which returns 404). No key. The USGS API gateway in front of it enforces
  **3,600 requests/hour per IP** (`x-ratelimit-limit: 3600`). On 2026-10-22 an early burst got HTTP 429
  with `Retry-After: 1714 s`. The client now spaces requests globally (1.25 s ⇒ ≤ 2,880/h, sharing the
  quota with other SENTINEL jobs), waits the full `Retry-After`, and caches every 200/404 response.
  At most 2–4 concurrent requests.
- **NHDPlus V2 VAA**: `nhdplusVAA.parquet` from HydroShare resource 6092c8a62fac45be97a09bfd0b0bf726
  (the table used by `nhdplusTools::get_vaa()`; md5 5691af50ffc1c1d66960a9dae690dc6e verified).
  Public domain (USGS/EPA NHDPlus V2).

## COMID linking (method recorded in `comid_method`)
1. `nldi_nwissite`: `/nwissite/USGS-<no>`. Reach-indexed hydrolocation with `reachcode` and
   `measure` (0 = downstream end, 100 = upstream end). Preferred for gauges. The nwissite source
   ("NWIS Surface Water Sites") holds ST/LK/ES-type sites; groundwater, spring, atmosphere and
   facility sites return 404.
2. `nldi_position`: `/comid/position?coords=POINT(lon lat)`, point in catchment, no measure.
   Used for NEON (sensor station S2 or buoy coordinates), for USGS sites missing from nwissite or
   with a null COMID, and, for comparison only, for a deterministic ~10 % sample of nwissite-linked
   sites (sha1(site_key) mod 10 = 0).
3. The same site can get different COMIDs from different lookups (docs/research/07 c.1). The
   agreement rate between the nwissite and position COMIDs on the sample is in
   `stats.nwissite_vs_position_same_comid`.

## Navigation and edges
- For every linked surface-water site (site types GW*, SB*, AT are not navigated), the client
  requests NLDI `navigation/DM/nwissite?distance=500` and then `navigation/UM/nwissite?distance=500`.
  The start is the nwissite feature, or the COMID for position-linked sites and NEON. Only hits that
  are other SENTINEL sites are kept.
- Each NHDPlus V2 VAA downstream main path (`dnhydroseq` chain, ≤ 500 km) is also walked locally.
  This cross-checks NLDI and connects sites NLDI's nwissite index cannot return (position-linked
  USGS sites, NEON ↔ NEON, e.g. Black Warrior → Tombigbee).
- `edges.parquet` columns:
  - `src` = upstream site, `dst` = downstream site (water flows src → dst).
  - `found_by`: every method that found the pair (`nldi_DM`, `nldi_UM`, `vaa_dm`, joined with `+`).
  - `distance_km`: along-channel distance = |pos(src) − pos(dst)|, where pos = VAA `pathlength` +
    `lengthkm` × measure/100 (km to the network outlet, along the main path).
  - `distance_method`: `vaa_pathlength_measure` when both ends have NLDI measures, else
    `vaa_pathlength_midpoint` (flowline midpoint).
  - `direction_check`: `ok` / `reversed` / `unknown`.
  - `travel_time_days`: difference of VAA `pathtimema`, i.e. travel time at mean-annual-flow
    velocity (approximate).
  - `src_comid`, `dst_comid`.
- `nodes.parquet`:
  - link columns: `comid`, `comid_method`, `reachcode`, `measure`, `mainstem` (geoconnex URI);
  - VAA columns `vaa_*`: hydroseq, dnhydroseq, levelpathi, terminalpa, pathlength, lengthkm,
    totdasqkm, streamorde, slope, pathtimema, ftype, vpuid;
  - `pos_km_to_outlet`, `pos_method`, `navigated`;
  - `flow_group`: connected component of the undirected edge graph. Use it to keep flow-connected
    sites in one fold;
  - `n_upstream_sites`, `n_downstream_sites`.

## Coverage and status (2026-10-22)
- **COMIDs:** 6,269 of 6,516 sites.
  - 5,268 `nldi_nwissite`, 1,001 `nldi_position`, 247 unlinked: AK 158 (no NHDPlus V2), NY 20,
    FL 14, ND 11, LA 11, GU 4, …
  - 6,145 linked sites have NHDPlus VAA attributes; HI and PR COMIDs have no VAA rows.
  - USGS stream (`ST*`) sites: 5,334 of 5,500 have a COMID.
- **Linking QC:**
  - nwissite vs position COMID agree for 93.8% of the 514 sampled sites.
  - VAA drainage area is within 0.5–2× of the NWIS drainage area for **97.2%** of nwissite-linked
    sites, but only **30%** of position-linked ones (`da_ratio_vaa_nwis` per node; treat
    position-linked edges with care).
  - nwissite COMIDs match the v1 NLDI map for 100% of the 440 overlapping sites.
- **`edges.parquet` (frozen provisional version, 2026-10-22 12:48 UTC):**
  - NLDI's gateway quota (3,600/h per IP, shared with other SENTINEL jobs) was exhausted, so the
    NLDI navigation responses were not yet available. This version comes from the complete local
    NHDPlus V2 VAA downstream-main-path traversal (`found_by = vaa_dm`), built with
    `--offline`.
  - 39,453 directed edges among 5,283 sites; distance median 116 km (max 500); 31,625 edges use
    measure-based distances, 7,752 midpoint-based.
  - `direction_check`: 39,377 ok, 76 unknown (different terminal path); 202 edges touch NEON,
    including MAYF → BLWA → TOMB.
  - 1,523 flow groups; the largest has 617 sites.
- **NLDI UM/DM cross-check:**
  - The full NLDI UM + DM navigation pass (11,636 requests) runs in the background with
    `--tag nldi_full`. It writes **separate** files `nodes_nldi_full.parquet` / `edges_nldi_full.parquet`
    (manifest `site_network_nldi_full`), so the frozen `edges.parquet` is never silently replaced.
  - At ≤ 2,880 requests/h, and with the shared quota, expect ≥ 4–8 h.
  - Swapping the split input to the NLDI version is a deliberate decision. Compare the two edge
    sets first; NLDI's DM should be close to `vaa_dm`.

## Known issues
- NLDI does not return along-channel distances, so distances come from VAA. VAA may differ from
  NLDI's network in a few places (e.g. updated network attributes). Pairs found only by one method
  are labelled.
- The 500-km cap is applied along the main path. Large rivers (Mississippi, Columbia) therefore
  form long chains of overlapping 500-km windows, and `flow_group` can link sites more than 500 km
  apart through intermediate sites. That is intended for leakage-safe grouping, but it creates a few
  very large groups. Combine with HUC4 as planned (RESEARCH_PLAN §14.2).
- Only main-stem relations (UM/DM) are captured. A site on a tributary is connected to sites
  downstream of the confluence, but two sites on sibling tributaries are not. They remain
  separate unless they share a downstream SENTINEL site.
- NHDPlus V2 has no Alaska network, and NLDI does not resolve the Guam sites. Most AK/GU sites
  therefore have no COMID and no edges.
- Position-linked COMIDs (no measure) can be a neighbouring small catchment. This matters most for
  lakes, estuaries and wells; see `comid_method`.

## Rebuild
```bash
PY="/home/bcheng/miniconda3/envs/sentinel2/bin/python -B"
$PY scripts/data/download_nldi.py --offline --distance 500           # frozen version: cache + VAA only, ~1 min
$PY scripts/data/download_nldi.py --workers 2 --distance 500 --min-interval 1.25 --tag nldi_full  # adds NLDI UM/DM
```
All responses are cached under `raw/nldi/cache/`, so a re-run only requests what is missing. A
fully cached rebuild takes a few minutes. A cold run takes about 8–9 h under the gateway quota:
≈ 7.6 k lookups + ≈ 11.8 k navigation requests at ≤ 2,880/h.
