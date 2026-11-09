# Datasheet — `labels_mn_infested` (Minnesota DNR Infested Waters List)

Downstream application **D6 MusselGuard** (RESEARCH_PLAN §8; docs/research/07 §a.2 E7, §d.6). This is the rolling-forecast label source: *which Minnesota waters were newly listed or confirmed in each year*.

| Item | Value |
|---|---|
| Publisher | Minnesota Department of Natural Resources (DNR), Invasive Species Program |
| Workbook (verified 2026-10-22) | `https://files.dnr.state.mn.us/eco/invasives/infested-waters.xlsx` (119,305 bytes; `Last-Modified: Wed, 21 Oct 2026 18:09:15 GMT`; ETag `"1d209-65e5da51e14ae"`) |
| Web page (definitions, headline counts) | `https://www.dnr.state.mn.us/invasives/ais/infested.html` |
| List version | **2026-10-21**, from the title row: "Minnesota Department of Natural Resources (DNR) List of Infested Waters - October 21, 2026" |
| Retrieval | 2026-10-22 |
| Access | Public download, no login. The DNR calls the Excel file "the most complete version of the infested waters list" and says it "should be used for regulatory or permitting purposes" |
| Licence / terms | **No open licence.** The DNR copyright notice (`https://www.dnr.state.mn.us/aboutdnr/disclaimers_and_policies.html`, snapshot in raw) claims copyright on site material. It allows viewing and printing for personal and fair use; republication requires DNR permission. Data are provided "as is" without warranty. → Use for research only. Do **not** redistribute the raw workbook; the test fixture is a 33-row excerpt. Cite the DNR list and its version date. |
| Manifest | `$SENTINEL_DATA/manifests/labels_mn_infested.json` |

## Files

| File | Rows | Content |
|---|---|---|
| `processed/labels_mn_infested/mn_infested_waters.parquet` | 1,524 | All water body × species listings |
| `processed/labels_mn_infested/mn_infested_zebra_mussel.parquet` | 737 | Zebra-mussel view (same schema) |
| `processed/labels_mn_infested/mn_infested_stats.json` | — | Counts, HTTP headers, web-page counts, parse metadata |
| `raw/mn_infested/infested-waters_2026-10-21.xlsx` | — | The workbook, stored under its list date so each vintage stays rebuildable |
| `raw/mn_infested/column_descriptions_2026-10-21.json` | — | The workbook's definitions sheet, as text |
| `raw/mn_infested/infested_page_2026-10-22.html`, `disclaimers_and_policies_2026-10-22.html` | — | Snapshots of the definitions page and the terms page |

Disk use: about 0.3 MB.

## Coverage (list of 2026-10-21)

**Rows.** 1,524 data rows. These sit between a title row, a header row and a trailing blank row plus an "End of Worksheet" row, which are dropped. Research's "1,526 rows" counted two non-data rows.

**Water bodies.** 997 distinct 8-digit DOW numbers. 193 rows are rivers or streams (DOW `NA`) and 23 are ponds or wetlands not on the public waters inventory (`none`).

**Species and listing basis.**

| Species | Rows | Confirmed | Connected |
|---|---|---|---|
| zebra mussel | 737 | 431 | 305 (+1 unknown) |
| Eurasian watermilfoil | 439 | 429 | 10 |
| spiny waterflea | 70 | 39 | 31 |
| faucet snail | 62 | 45 | 17 |
| flowering rush | 52 | 52 | 0 |
| silver carp | 44 | 7 | 37 |
| bighead carp | 44 | 9 | 35 |
| starry stonewort | 39 | 39 | 0 |
| grass carp | 12 | 7 | 5 |
| brittle naiad | 10 | 10 | 0 |
| ruffe | 3 | 2 | 1 |
| white perch | 3 | 2 | 1 |
| round goby | 3 | 3 | 0 |
| VHS virus | 3 | 1 | 2 |
| New Zealand mud snail | 2 | 2 | 0 |
| red swamp crayfish | 1 | 1 | 0 |

**Years.** `year_listed` runs from 1995 to 2026. Zebra-mussel `year_confirmed` runs from 1989 to 2026.

**Zebra-mussel rows by listing year.**

| Year listed | Rows | Year listed | Rows |
|---|---|---|---|
| 1995 | 12 | 2016 | 34 |
| 2000 | 2 | 2017 | 55 |
| 2001 | 1 | 2018 | 66 |
| 2004 | 2 | 2019 | 54 |
| 2006 | 20 | 2020 | 40 |
| 2007 | 8 | 2021 | 50 |
| 2009 | 23 | 2022 | 44 |
| 2010 | 27 | 2023 | 28 |
| 2011 | 6 | 2024 | 34 |
| 2012 | 18 | 2025 | 66 |
| 2013 | 34 | 2026 (to 21 Oct) | 24 |
| 2014 | 67 | | |
| 2015 | 22 | | |

**Check against the web page and research.** The DNR page ("As of September 2026") says zebra mussels are confirmed in **398 lakes and wetlands** and **302** bodies of water are listed because they are connected. The list of 21 October 2026 has:
- 406 confirmed zebra-mussel rows with a DOW number (405 distinct DOW8), plus 25 confirmed river rows;
- 305 connected rows (218 with a DOW number, 7 `none`, 80 rivers).

The small excess is consistent with later confirmations (Sept to 21 Oct 2026) and with some DOW-numbered river pools. The page also says "About 5% of Minnesota's 11,842 lakes" are infested with zebra mussels. That figure is the lake universe for negatives; the lake list itself is not part of this dataset.

## Label definition

These come from the workbook's column-description sheet, which is reproduced in raw.

- **`year_listed`:** "the year in which the water body was first added to the infested waters list".
  - 1995 = the original "List of Contaminated Waters". These rows are left-censored: earlier infestations are dated 1995.
  - 1996–2007: listing by rulemaking.
  - 2007–2014: listing by Commissioner's Order.
  - Since August 2014: the online list is official.
- **`listing_basis`**, from the column "Year species was first confirmed, or connected water body":
  - `confirmed`: the cell holds the year the DNR first confirmed a population. `year_confirmed` is the earliest 4-digit year in the cell, e.g. "1977; 2008; 2013; 2025" → 1977 and "2017 (dead); 2018 (live)" → 2017. `veliger_note` = True when the text mentions veligers (larvae, 22 zebra-mussel rows). `confirmed_flag` = `annotated` when the cell is more than a bare year. "year unknown" gives `year_unknown` (20 rows, none of them zebra mussel).
  - `connected`: text starting "connected…" (including the list's typo "conected…") or "part of … chain". The water body was listed **because it is connected** to a confirmed water. This is an **administrative listing, not a detection**. DOW numbers inside the text are never parsed as years.
  - `unknown`: blank or `NA` (1 zebra-mussel river row).
- **`listing_lag_years`** = `year_listed − year_confirmed` (confirmed rows only).
  - For zebra mussels the median is 0; the 10th and 90th percentiles are 0 and 1.
  - **39 zebra-mussel rows have a negative lag.** These waters were listed earlier (almost always as *connected*) and confirmed later; the list overwrites the basis column and keeps no history.
  - Hence the "year first known infested" is `min(year_listed, year_confirmed)`, but the confirmation event is `year_confirmed`.

## Schema

**Identifiers**
- `source` = `mn_dnr_infested_waters`.
- `record_id` = `mn_iw_<list_version>_<excel row>`. It is stable only within one list version.
- `list_version`, `row_number`.

**Water body**
- `waterbody_name`: whitespace-normalised. Lake names have no "Lake".
- `county`, `county_multiple`: True for "Multiple (…)"; 137 rows.

**Species**
- `species_common`: normalised lower case. "Eurasian watermilfoil " and "Eurasian Watermilfoil" are merged; "VHS" maps to "viral hemorrhagic septicemia (vhs)".
- `species_scientific` (e.g. *Dreissena polymorpha*), `species_raw`.

**Listing and confirmation**
- `year_listed` (Int64), `year_listed_raw`.
- `listing_basis`, `year_confirmed` (Int64), `years_confirmed_all`, `veliger_note`, `confirmed_flag`, `confirmed_or_connected_raw`, `listing_lag_years`.

**DOW number**
- `dow`: `CC-LLLL` or `CC-LLLL-BB`, zero-padded. The list's single-digit county `1-0179` becomes `01-0179`.
- `dow8`: 8-digit `CCLLLLBB`, with basin `00` for a whole lake. This is the format the DNR itself uses inside names, e.g. "Villard (61006700)".
- `dow_class` ∈ {dow, river_na, none, missing, unparsed}.
- `dow_raw`.

**Stray cells**
- `extra_cell`: the content of the hidden/unnamed 7th column. There is one value, `44277` at Excel row 204, kept but not interpreted.

## QC applied

- Placeholder strings "no data" (title row) are set to missing. Whitespace, including NBSP, is normalised. The title, header, blank and "End of Worksheet" rows are removed.
- DOW variants are parsed and classified. `NA` and `na` are rivers; `none`, `NONE` and "none, part of Winnibigoshish" are not on the PWI. No DOW values are left unparsed.
- Every one of the 16 species names maps to a scientific name.
- All 1,524 `year_listed` values are 4-digit years. `year_confirmed` uses explicit rules (above), tested on the real strings.
- Nothing is imputed. The raw workbook is kept, so the parse can be re-run (`--xlsx`).

## Known biases and caveats

- **Detection effort and boater traffic.** Confirmations follow DNR and partner surveys, lake-association reports and veliger sampling. Heavily used lakes are searched more. Uninfested lakes are only *presumed* negative.
- **"Connected" listings are administrative.** They are precautionary, made by connection rules, and correlate with the network position of known infestations by construction. Exclude them from detection targets, or model them as a separate, policy-driven label.
- **Listing lag and history loss.** The listing year can precede the confirmation (39 zebra-mussel rows) or follow it (36 rows at +1 year; up to +20). The workbook is a current snapshot, not a change log. To get true listing-date histories, keep each vintage (the downloader saves `infested-waters_<version>.xlsx`).
- **Left-censoring at 1995** for the original list.
- **Multi-county lakes** may appear once, under the DOW county, or more than once (e.g. Namakan 69-0693 has two rows). De-duplicate on (`dow8`, `species_common`) for lake-level labels.
- **Lake universe.** Negatives need the MN lake list (11,842 lakes; DNR hydrography / LAGOS-US), which is not included here.

## How it joins to sites

- **DOW number → lake polygon.** Join `dow8` to MN DNR public-waters / DNR Hydrography lake polygons (8-digit DOW lake-number field) or to the LAGOS-US LOCUS lake table via a spatial join. MN agencies use the DOW as the lake identifier.
- **Upstream SENTINEL predictions.** Use lake-level means of predicted SC/pH (calcium proxies) and temperature, from satellite lake pixels or WQP stations inside the polygon, over the years before the origin.
- **NAS cross-check.** USGS NAS (`labels_nas_dreissena`) holds 729 Minnesota zebra-mussel records (1989–2024) with coordinates and HUC12. Spatially join them to DOW polygons to compare detection dates. The DNR list is more current: it has 2025–2026 confirmations that NAS lacks.
- **Rivers** (`dow_class = river_na`) have only a descriptive name, e.g. "Pine River from the base of Cross Lake Dam to …", and need manual geocoding to NHD flowlines. Use lakes with a DOW for the primary task.

## Rolling-origin usage (MusselGuard)

1. **Origin T** (e.g. 2012, 2015, 2018).
   - Infested set at T: zebra-mussel waters with `listing_basis = confirmed` and `year_confirmed ≤ T`. Optionally add `connected` waters with `year_listed ≤ T`, as a separate feature.
   - **Positives:** lakes with `year_confirmed` in (T, T+5] that were not listed as confirmed by T.
   - **Negatives:** MN lakes not confirmed by the end of the window (presumed negative).
2. Use a single list vintage, and record it. The **2026-10-21** list has 431 confirmed zebra-mussel rows, with 57 confirmations in 2025–2026.
3. **Baselines:** distance to the nearest infested water at T, connectivity, the Ca-risk rule, and the co-kriging benchmark (Kanankege et al. 2018: AUC 0.78).

## Rebuild

```bash
cd /home/bcheng/STOCKHOLM/SENTINEL-STOCKHOLM
/home/bcheng/miniconda3/envs/sentinel2/bin/python -B scripts/data/download_labels_mn_infested.py              # download current list (+ page, terms)
/home/bcheng/miniconda3/envs/sentinel2/bin/python -B scripts/data/download_labels_mn_infested.py \
    --xlsx $HOME/STOCKHOLM/sentinel_data/raw/mn_infested/infested-waters_2026-10-21.xlsx                  # rebuild this vintage offline
/home/bcheng/miniconda3/envs/sentinel2/bin/python -B -m pytest tests/test_labels_dreissena.py
```

Code: `sentinel/data/sources/downstream_dreissena.py` (`parse_mn_infested`, `parse_mn_confirmed`, `parse_mn_dow`, `normalize_mn_species`, `mn_column_descriptions`, `mn_page_counts`, `mn_list_version`).
