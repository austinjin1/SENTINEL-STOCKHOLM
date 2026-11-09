# Datasheet — `labels_nas_dreissena` (USGS NAS zebra / quagga mussel occurrences)

Downstream application **D6 MusselGuard** (RESEARCH_PLAN §8; docs/research/07 §a.2 E7, §d.6).

| Item | Value |
|---|---|
| Publisher | U.S. Geological Survey, Nonindigenous Aquatic Species (NAS) program, Gainesville, Florida |
| API (verified 2026-10-22) | `https://nas.er.usgs.gov/api/v2/occurrence/search?species_ID={5,95,3118}&limit=5000&offset=…` (no key, JSON) |
| Species records | `https://nas.er.usgs.gov/api/v2/species/{5,95,3118}` |
| Documentation / terms | `https://nas.er.usgs.gov/api/documentation.aspx`, `https://nas.er.usgs.gov/disclaimer.aspx`, `https://nas.er.usgs.gov/about/DBCharacterization.aspx` (snapshots in raw) |
| Retrieval | 2026-10-22, 09:21 UTC (`retrieved_utc` per species in the stats/manifest) |
| Licence / terms | US Government work (public domain). NAS data disclaimer: data are provisional, "as-is", may be revised; refresh at least yearly; acknowledge contributors; NAS asks to be contacted for publications. Citation: *U.S. Geological Survey. [2026]. Nonindigenous Aquatic Species Database. Gainesville, Florida. Accessed [10/22/2026].* |
| Manifest | `$SENTINEL_DATA/manifests/labels_nas_dreissena.json` (sha256 of every file, request parameters, counts) |

## Species and access

Species IDs were confirmed with the species endpoint:

| speciesID | NAS scientific name | Common name | ITIS TSN |
|---|---|---|---|
| 5 | *Dreissena polymorpha* (Pallas, 1771) | zebra mussel | 81339 |
| 95 | *Dreissena bugensis* Andrusov, 1897 (NAS name for *D. rostriformis bugensis*) | quagga mussel | 567514 |
| 3118 | *Dreissena* sp. | a dreissenid mussel (not identified to species) | 81338 |

`species/search?genus=Dreissena` returns exactly these three taxa. 3118 is kept separately (`species_id`) so it can be excluded; it is included only in the `dreissena_any` first-detection group.

**Paging.** The documentation states that results are "ordered by species ID and occurrence record ID". The envelope field `count` is the page length (not a total). `endOfRecords` is the string `"true"`/`"false"`, and the page holding the last record already says `"true"`. The pager advances by the number of results returned. Check run on 2026-10-22: re-downloading with page size 2,000 instead of 5,000 returned an identical key set for all three species, with 0 duplicate keys (`paging_check` in the stats).

**Temporal accuracy (`yrAcc`).** Records do not carry their temporal-accuracy class. The API accepts it as a filter, so each class was queried separately (Actual / Estimated / Publication year). The classes partition every species exactly: 0 untagged keys, 0 conflicts.

## Files

| File | Rows | Content |
|---|---|---|
| `processed/labels_nas_dreissena/nas_dreissena_occurrences.parquet` | 12,865 | One row per NAS occurrence key (zebra 11,034 · quagga 1,523 · *Dreissena* sp. 308) |
| `processed/labels_nas_dreissena/nas_dreissena_first_detection_huc12.parquet` | 3,881 | Earliest record per (species group, HUC12): zebra 1,788 · quagga 204 · dreissena_any 1,889 |
| `processed/labels_nas_dreissena/nas_dreissena_references.parquet` | 11,349 | Occurrence → reference links (key, type, year, author, title, publisher) |
| `processed/labels_nas_dreissena/nas_dreissena_stats.json` | — | Counts, paging and yrAcc checks, website cross-check, citation |
| `raw/nas_dreissena/occurrence_species_{5,95,3118}.json.gz` | — | Every API page envelope as returned, with retrieval time and parameters (0.8 MB) |
| `raw/nas_dreissena/yracc_species_{id}.json` | — | Keys per temporal-accuracy class |
| `raw/nas_dreissena/species_*.json`, `api_documentation.html`, `disclaimer.html`, `db_characterization.html` | — | Provenance snapshots |

Disk use: about 2.2 MB (raw 1.2 MB, processed 1.0 MB).

## Coverage (measured 2026-10-22)

| | Zebra (5) | Quagga (95) | *Dreissena* sp. (3118) |
|---|---|---|---|
| Records | 11,034 | 1,523 | 308 |
| Record years | 1986–2026 | 1989–2026 | 1992–2023 |
| US / Canada / Mexico / no state | 8,390 / 2,550 / 0 / 94 | 1,218 / 304 / 1 / 0 | 102 / 178 / 0 / 28 |
| US states | 32 | 20 | 8 |
| Distinct HUC12 | 1,788 | 204 | 19 |
| Status: established (+ locally established) | 7,521 (+3) | 1,216 | 53 |
| Status: blank / unknown / collected | 2,836 / 520 / 105 | 166 / 121 / 0 | 253 / 0 / 0 |
| Status: extirpated / eradicated / failed | 32 / 9 / 8 | 6 / 1 / 13 | 0 / 0 / 2 |
| Record type: Literature / Pers. comm. / Specimen / NAS sighting report | 6,888 / 2,855 / 1,100 / 191 | 1,153 / 316 / 13 / 41 | 306 / 2 / 0 / 0 |
| yrAcc: Actual / Estimated / Publication year | 10,894 / 85 / 55 | 1,507 / 14 / 2 | 308 / 0 / 0 |
| Date precision: day / month | 11,023 / 11 | 1,462 / 61 | 307 / 1 |

**Records by decade.**

| Decade | Zebra | Quagga | *D.* sp. |
|---|---|---|---|
| 1980s | 125 | 1 | 0 |
| 1990s | 4,082 | 276 | 36 |
| 2000s | 2,692 | 334 | 62 |
| 2010s | 3,396 | 821 | 182 |
| 2020s | 739 | 91 | 28 |

**Spatial extent.** Coordinates are present and valid for all 12,865 records (latitude 29.25–54.98, longitude −121.45 to −66.88). 1.8% are flagged `Approximate` by NAS.

**Research comparison.** docs/research/07 reported zebra 11,034 records (7,521 established; 1,788 HUC12; "39 states") and quagga 1,523 (1,216 established; 204 HUC12; "23 states"). Records, established counts and HUC12 counts are reproduced exactly. The "states" figure in research counted distinct `state` strings, which include a blank and the Canadian codes MB, NB, ON, ONT, QC and QUE. The US-state counts are 32 and 20.

## Label definition

- **Occurrence (presence-only):** one NAS record means a documented observation or collection of the taxon at a place and date. The sources are literature, personal communication with NAS staff, museum specimens or a NAS sighting report. **There are no absences.**
- **`status`:** the *current* status of the introduced population at that location (lower-cased): established, locally established, collected, unknown, failed, extirpated, eradicated, or missing (blank in the source). `is_established` is True for established or locally established.
- **First detection per HUC12** (`nas_dreissena_first_detection_huc12.parquet`, for rolling-origin tests). Columns:
  - `first_year_any`: earliest record year, any status;
  - `first_year_actual`: earliest record whose yrAcc is Actual;
  - `first_year_established`: earliest record whose *current* status is established;
  - `first_key_*`, `first_status_any`, `n_records`, `n_established`, `states`.

  Ties on the earliest year are broken by the smallest NAS key. Groups are `zebra` (5), `quagga` (95) and `dreissena_any` (5 + 95 + 3118). `first_year_established` is not a historical establishment date, because the status is the one at download time.

## Schema (occurrences)

**Identifiers**
- `source` = `usgs_nas`.
- `record_id` = `nas_key`: the NAS occurrence key, as a string.
- `species_id` (string), `scientific_name`, `common_name`, `group`, `family`, `genus`, `species`.

**Dates**
- `date`: set only when the day is known. No day is invented.
- `date_start` / `date_end`: the uncertainty interval. Month-precision records span the whole month; year-only records span the year.
- `date_precision` ∈ {day, month, year, none}.
- `year`, `month`, `day` (Int64), `date_raw` (the NAS string, e.g. `1990-9-17`).
- `year_accuracy` ∈ {Actual, Estimated, Publication year}.
- `jan1_date_suspect`: True for 1 January dates (153 records, 151 of them literature), which are probably year-only placeholders.

**Status**
- `status`, `status_raw`, `is_established`, `record_type`.

**Coordinates**
- `latitude`, `longitude` (WGS84 decimal degrees as published).
- `coord_flag`: empty when OK; otherwise missing / zero / invalid / outside_bbox / swapped_fixed.
- `latlong_source` (reported / Map derived / GNIS / GPS), `latlong_accuracy` (Accurate / Approximate), `centroid_type`.

**Location**
- `state`: USPS code, or the Canadian province code.
- `state_raw`, `country` (US / CA / MX), `county`, `locality`.

**Hydrologic units**
- `huc8`, `huc10`, `huc12`: zero-padded digit strings. Blank or all-zero values become missing.
- `huc8_name`, `huc10_name`, `huc12_name`.
- `huc_flag`, e.g. `huc8_zero_placeholder`.

**Other fields**
- `fresh_marine_intro`, `n_references`, `reference_keys`, `reference_year_min`, `comments`, `museum_cat_number`, `disposal`, `uuid`.

## QC applied

1. Records are de-duplicated on the NAS key. The paging check is described above. `parse_nas_records` raises an error on duplicate keys.
2. Missing codes:
   - `huc8 = "00000000"` (332 Canadian records: ON 169, QC 149, MB 14) is NAS's placeholder for "outside the Watershed Boundary Dataset". It is set to missing and flagged.
   - Blank HUC10/HUC12 values (584 / 578 records) become missing.
   - Blank strings become missing everywhere.
3. State normalisation:
   - US full names map to USPS codes.
   - Canadian `ON`/`ONT` map to ON, `QC`/`QUE` to QC; MB and NB are kept.
   - "Mexico" maps to country MX.
4. Coordinates are validated against a North-America box (−170, 15, −50, 75). All passed; no (0, 0) values were found.
5. Dates: impossible month or day values lower the precision instead of producing an invalid date. Years after next year are rejected.
6. No values were imputed or synthesised. Raw API envelopes are kept for re-parsing.

## Known biases and caveats

- **Detection and reporting effort.** Records cluster where agencies, universities and boat-inspection or veliger programs look (Great Lakes, New York, the Upper Midwest). Boater traffic drives both the spread and the detection. Treat "no record" as unknown, not as absent.
- **Presence-only data with pseudo-replication.** 2,564 records share their species, coordinates and start date with another record. Single sites can contribute up to 62 records. Aggregate to site- or HUC12-year before modelling.
- **Ingestion lag.** New HUC12 first detections (`dreissena_any` group) fall from 25–59 per year (2019–2022) to 8 in 2025 and 5 in 2026. NAS Minnesota zebra records end in 2024, while the MN DNR list already holds 2025–2026 confirmations. The last ~2 years are incomplete, so test windows must end at least 2 years before the data vintage, or use state lists.
- **Temporal accuracy.** 57 records carry the *publication* year and 99 an *estimated* year (midpoint of a range). Use `first_year_actual` for strict onset analyses.
- **Status is current, not historical.** A site recorded as `collected` in 1993 may now be `established`.
- **Canada.** 3,032 records are Canadian. Most lack WBD HUCs, so drop them for US HUC12 work. Lake-wide Great Lakes HUC12 units (e.g. `042600000200` "Lake Erie") aggregate open-water records from both countries.
- **Website vs API.** On 2026-10-22 the NAS website's collection-info pages reported 11,314 (zebra), 1,589 (quagga) and 308 (*D.* sp.) records, while the API returned 11,034, 1,523 and 308. The first 25 specimen IDs listed on the quagga web page are all present in the API pull. The gap is **unresolved**: the website table needs ASP.NET postbacks and was not scraped. The API is the documented machine-readable interface and is the source used here.
- **Revisions.** NAS data are provisional and revised without notice. Results must cite the manifest (file sha256 values and retrieval time).

## How it joins to sites

- **HUC12 / HUC8.** Use the published WBD codes directly (12- and 8-character strings with leading zeros). They join to the WBD and to the HUC12 fields of other SENTINEL tables. The HUC is NAS-assigned; for river records it may differ from a fresh point-in-polygon at the stated coordinates.
- **NHDPlus V2 COMID.** Snap `latitude`/`longitude` with NLDI `https://api.water.usgs.gov/nldi/linked-data/comid/position?coords=POINT(lon lat)`. Record the snap distance and the method. Lake records need an NHD waterbody or LAGOS-US lake polygon rather than the nearest flowline catchment. Records flagged `Approximate`, or with `latlong_source` = GNIS, should be matched at HUC12 level only.
- **Minnesota.** Link to MN DNR DOW numbers (`labels_mn_infested`) by spatial join with MN DNR public-waters lake polygons, which carry the 8-digit DOW (field name not verified here), or LAGOS-US. Names are not reliable keys: many lakes share a name.
- **Upstream predictions.** Use the calcium proxies (SC, pH) and temperature for the HUC12 or lake over the years before the forecast origin.

## Rolling-origin usage (MusselGuard)

1. **Origin T.** The positives are HUC12s (or lakes) with `first_year_any`, or `first_year_actual`, in (T, T+k]. The candidates are units with no record by T. Units still unrecorded at T+k are **presumed negatives** (label noise: undetected infestations).
2. **Features.** Use only information dated ≤ T. That includes distance to infested units at T and network connectivity. Never use a status or a record added after T.
3. **Window.** End test windows at least 2 years before the data vintage (ingestion lag). Report results per data vintage.
4. **Baselines** (research/07 §b.3): distance to the nearest infested unit, the calcium-risk rule (Whittier et al. 2008), and co-kriging (Kanankege et al. 2018; AUC 0.78 in Minnesota).

## Rebuild

```bash
cd /home/bcheng/STOCKHOLM/SENTINEL-STOCKHOLM
/home/bcheng/miniconda3/envs/sentinel2/bin/python -B scripts/data/download_labels_nas_dreissena.py            # reuse raw pages if present
/home/bcheng/miniconda3/envs/sentinel2/bin/python -B scripts/data/download_labels_nas_dreissena.py --refresh  # new vintage (~1 min, ~25 requests)
/home/bcheng/miniconda3/envs/sentinel2/bin/python -B -m pytest tests/test_labels_dreissena.py
```

Code: `sentinel/data/sources/downstream_dreissena.py` (`nas_fetch_pages`, `records_from_pages`, `nas_year_accuracy_keys`, `parse_nas_records`, `nas_references`, `first_detection_by_huc12`). Tests use recorded real records in `tests/fixtures/labels_dreissena/`.
