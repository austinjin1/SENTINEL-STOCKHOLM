# Datasheet — `labels_delaware_fishkill` (Delaware DNREC non-tidal fish-kill reports)

| | |
|---|---|
| **Use in SENTINEL** | D3 KillAlert, pooled with Iowa (`docs/preregistration/downstream_killalert.md` §11, v2) |
| **Publisher** | Delaware Dept. of Natural Resources and Environmental Control (DNREC), Division of Fish & Wildlife, via the Delaware Open Data Portal |
| **Source** | Socrata dataset `8pfi-yck7` "Non-tidal Fish Kill Reports": `https://data.delaware.gov/resource/8pfi-yck7.json` (metadata `https://data.delaware.gov/api/views/8pfi-yck7.json`). The tidal companion `ycvg-4ys9` (234 rows) has **no coordinates** and is not used |
| **Licence** | "Public Domain" (portal licence field) |
| **Retrieved** | 2026-10-25 05:00 UTC; 2 requests, generic User-Agent |
| **Files** | raw `raw/delaware_fishkill/<YYYYMMDD>/{rows,metadata}.json`; processed `processed/labels_delaware_fishkill/{rows,events}.parquet`; manifest `labels_delaware_fishkill` |
| **Code** | `sentinel/data/sources/events_delaware_fishkill.py` · `scripts/data/download_events_delaware_fishkill.py` · tests `tests/test_downstream_killalert.py` |
| **Synthetic data** | none |

## Coverage
290 rows (one per kill × species group) → **184 events** after de-duplication on (location, date, lat, long);
142 geocoded (lat/long present, inside a Delaware box); 2004–2025. Waterbody types (derived from the
location name): Pond 89, Unspecified 48, River/Stream 26, Lake 21.

## Label harmonisation to the Iowa classes (keyword rules, auditable; `cause_rule` stored per event)
Applied to the free-text `probable_cause`, in this order: anthropogenic keywords (pesticide/herbicide/
insecticide, manure, sewage, spill, discharge, chemical, chlorine, fertiliser, treatment, algaecide, water
main, dredging, draw-down, …) → **Anthropogenic**; disease/parasite/epizootic/bacterial/gill → **Natural,
Disease/Parasite/Stress**; low DO / dissolved oxygen / winterkill / ice / algal bloom / temperature / heat /
turnover / low water / low flow / drought → **Natural, Environmental** (`is_env_wq`; hedged wording such as
"possibly", "likely" is flagged `env_keyword_hedged` but counted); "unknown" → Unknown; anything else →
"Other/unclassified" (origin Unknown). Result (2026-10-25): Environmental 98, Anthropogenic 36, Disease 15,
Unknown 19, Other 16. Iowa's "Unknown (Natural Suspected)" has no Delaware analogue (`is_natural_suspected`
False).

## Known issues
- Reports depend on public reporting and DNREC response (effort bias; mostly ponds).
- Only the calendar date of the report row is published (`date_dnrec_response` exists for some rows).
- Causes are investigator text; the keyword mapping is a modelling decision (documented, unit-tested).
- 42 events lack coordinates (excluded from KillAlert).

## Rebuild
```bash
/home/bcheng/miniconda3/envs/sentinel2/bin/python -B scripts/data/download_events_delaware_fishkill.py
```
