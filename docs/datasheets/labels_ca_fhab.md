# Datasheet — `labels_ca_fhab` (California FHAB bloom reports, advisories, cases, results)

**Role in SENTINEL v2:** secondary labels for D2 ToxinWatch (HAB-advisory forecast) and EW-v2
context (H5 lake cyanoHAB). Primary toxin labels come from WQP / the Washington database; these
records are voluntary, effort-driven and partly visual (see "Known biases").

## Source and access

| Item | Value |
|---|---|
| Publisher | California State Water Resources Control Board (Water Boards), Freshwater HAB (FHAB) Program |
| Dataset | "Surface Water – Freshwater Harmful Algal Blooms", data.ca.gov CKAN package `surface-water-freshwater-harmful-algal-blooms` |
| Landing page | https://data.ca.gov/dataset/surface-water-freshwater-harmful-algal-blooms |
| API used | `https://data.ca.gov/api/3/action/package_show?id=surface-water-freshwater-harmful-algal-blooms` → resource URLs; CSV downloads (HTTP 302 to a pre-signed object-store URL, then 200); fallback `https://data.ca.gov/api/3/action/datastore_search?resource_id=<id>` (paged) |
| Resources (ids on 2026-10-22) | Bloom Reports `c6a36b91-ad38-4611-8750-87ee99e497dd`; Cases `67648948-034f-4882-bbc0-c07c7d38daf9`; Responses `4283c060-c22f-48f5-a75c-8bccf0c54a99`; Results `9d4e1df4-0cd6-4165-9e63-effcafd9dccc`; Data Dictionary PDF `516bd039-…`; Disclaimer PDF `9a54fa52-…` |
| Retrieval date (vintage) | 2026-10-22 (package `metadata_modified` 2026-10-21T00:34:52; files named `*_2026-10-20.csv`; refreshed ~daily) |
| Access | Anonymous; no key or login. All four CSVs downloaded directly; row counts equal the datastore totals (3,728 / 1,484 / 14,437 / 2,080). |
| Licence / terms | CKAN licence "Other (Public Domain)" (`other-pd`). Disclaimer: data are voluntarily provided, "not exhaustive of all ambient conditions"; reports are suspected blooms until confirmed. |
| Advisory guidance | CCHAB "California Voluntary Guidance for Response to HABs in Recreational Inland Waters" (2016), trigger table at https://mywaterquality.ca.gov/habs/resources/response-guidance.html (verified 2026-10-22) |

## Files

- Raw (kept, 11 MB): `$SENTINEL_DATA/raw/ca_fhab/2026-10-22/` — `bloom_reports.csv`, `cases.csv`,
  `responses.csv`, `results.csv` (byte-identical published CSVs), `package_show.json`,
  `fetch_info.json` (method, row counts, datastore totals), `docs/habs-master-data-dictionary.pdf`,
  `docs/habs-disclaimer-for-data-dictionary.pdf`. One folder per retrieval date = label vintage.
- Processed (zstd Parquet, 1.1 MB): `$SENTINEL_DATA/processed/labels_ca_fhab/`
  `bloom_reports.parquet`, `advisories.parquet`, `cases.parquet`, `results.parquet`.
- Manifest: `$SENTINEL_DATA/manifests/labels_ca_fhab.json` (sha256 of every file, raw-frame sha1,
  request URLs, counts, flags).

## Coverage (vintage 2026-10-22)

| Table | Rows | Grain | Time | Space |
|---|---|---|---|---|
| `bloom_reports` | 3,713 (from 3,728 published rows) | one row per `report_id` | observation dates 2015-09-15 … 2026-10-19 | 940 waterbody names; 64 county strings (CA counties, multi-county strings such as "Calaveras; Tuolumne", plus "Arizona State" / "Nevada State"); 3,694 with valid coordinates |
| `advisories` | 14,437 | one row per advisory response action (`response_action_id`); 2,964 advisories | recommendation dates 2016-04-30 … 2026-10-20 | 801 waterbody names; 14,391 with coordinates |
| `cases` | 1,484 | one row per case (staff folder of reports) | case years 2016–2026 | — |
| `results` | 2,080 | one row per published result | sample dates 2022-04-18 … 2025-06-10 (1,911 rows undated) | — |

Bloom reports per observation year (published rows): 2016 90 · 2017 162 · 2018 211 · 2019 249 · 2020 369 ·
2021 560 · **2022 737** · 2023 457 · **2024 296** · 2025 265 · 2026 328 (to 19 Oct); one each in 2015,
1976, 2555, 2626. Advisory records per year: 2016–2021 ≤ 480/yr, 2022 3,599, 2023 2,657,
2024 2,391, 2025 2,356, 2026 2,129 (the 2022+ jump is a data-system change with routine-monitoring
updates, not more blooms).

Advisory tier of the 14,437 advisory records: None 6,242 · Caution 4,403 · Warning 478 · Danger 1,215 ·
"NA – refer to Report Details" 746 · algal-mat alert sign 570 · algal-mat general-awareness sign 236 ·
visual observation 330 · general awareness 160 · blank 57. At report level (3,713): None 1,424 ·
Caution 901 · Warning 87 · Danger 166 · mats 217 · other/unknown 918.

**End dates are rare:** bloom-report rows have an advisory start date in 3,297 of 3,728 rows but an
end date in only **238** (231 with both) — the research figure is confirmed. Advisory records: 325 end
dates among 13,856 starts. The data dictionary: "for waterbodies not regularly monitored, there may
never be an end date".

## Label definitions

| Column | Definition |
|---|---|
| `advisory_class` | Normalised `Advisory_Recommended`: `none`, `caution`, `warning`, `danger`, `algal_mat_alert`, `algal_mat_awareness`, `general_awareness`, `visual_observation`, `na_see_report`, `unknown` (blank = unknown, per the disclaimer). New values become `other_<text>`. |
| `tier_level` / `advisory_tier` | Planktonic CCHAB tier: 0 None < 1 Caution < 2 Warning < 3 Danger (ordered categorical). `<NA>` for mat signs, general awareness, visual observation, NA, blank — they are not on the planktonic scale. |
| `label_tier_ge_caution` | **Recommended secondary label**: tier ≥ Caution recommended (True), None recommended (False), otherwise `<NA>`. |
| `any_advisory` | Disclaimer semantics of a confirmed bloom: any advisory other than None (includes mat and awareness signs); `<NA>` for blank / "NA – refer to Report Details". |
| `observation_channel` | From `Case_Class`: `event_response` (complaint/report-driven), `monitoring` (`Monitoring_*`: local agency, state, field, community, tribe, federal, satellite → `monitoring_type`), `other`, `unknown`. |
| `basis_*` flags (advisories) | Parsed from `Advisory_Detail`: `basis_toxins` ("Advisory based on visual and toxins"), `basis_visual_only`, `below_triggers`, `lake_wide`, `routine_monitoring`, `under_investigation`; `<NA>` when the detail is blank. |

CCHAB trigger levels (any one criterion suffices; de-posting needs all "no advisory" criteria for ≥ 2 weeks):

| Criterion | Caution (Tier 1) | Warning (Tier 2) | Danger (Tier 3) |
|---|---|---|---|
| Total microcystins | 0.8 µg/L | 6 µg/L | 20 µg/L |
| Anatoxin-a | detected | 20 µg/L | 90 µg/L |
| Cylindrospermopsin | 1 µg/L | 4 µg/L | 17 µg/L |
| Potential toxin-producer cells | 4,000 cells/mL | – | – |
| Site-specific indicators | discolouration, scum, algal mats, soupy/paint-like appearance; suspected illness | – | – |

Building a waterbody-week label: take `advisories`, keep `tier_level` not null, assign each record to
the waterbody (name + coordinates) and ISO week of `date`, label = max tier in the week. Never treat
a week without a record as tier 0 — unmonitored ≠ no bloom; use only weeks with a record, or use
monitoring-channel waterbodies with regular records as the denominator.

## Schema highlights

- Dates: `date` is the local calendar date (observation date for reports, recommendation date for
  advisories, sample date for results). Every date has a `*_flag` (`''`, `missing`, `unparseable`,
  `out_of_range`, and `after_created` for reports observed > 1 day after the report was created —
  60 rows, date kept). `created_local`, `response_timestamp_local`, `case_timestamp_local` are naive
  system timestamps (Pacific time assumed, not documented).
- IDs (`report_id`, `case_id`, `advisory_id`, `response_action_id`, `result_id`) are strings.
- Coordinates: `latitude`/`longitude` (WGS84 decimal degrees as published; `datum` is mostly blank,
  WGS84 563 / NAD83 190 report rows), `coord_flag` (`''`, `missing`, `zero`, `invalid`, `outside_bbox`,
  `swapped_fixed`). Results carry `coord_source` (`sample` or `bloom_report`).
- Results: `value` (float), `unit` (as published: `uS`, `C`, `mg/L`; pH and one DO value have no
  unit), `censored` + `detection_limit` (no DL/2 substitution), `qualifier`, `value_category`
  (present/absent), `is_placeholder`.

## QC applied

1. Dates validated against 2014-01-01 … retrieval date + 1 day: observation years 1976, 2555 and 2626
   → NaT with `out_of_range`. The 2015-09-15 observation reported in 2016 is kept.
2. Coordinates outside California + 0.25° (W −124.7, S 32.3, E −113.85, N 42.25), (0, 0) or malformed
   (e.g. longitude −1215574, latitude 4010.319) → NaN with flag. Bloom reports: 15 zero, 3 invalid, 1
   missing; border waters (Klamath reservoirs in Oregon, Lake Tahoe NV side, Colorado River) are kept.
3. Published bloom-report grain is report × linked advisory: 14 reports occupy 29 rows (15 extra
   rows) — 6 reports link 2–3 distinct advisories (7 extra rows) and 8 rows repeat the same
   report–advisory pair. `bloom_reports` keeps the highest-tier (then latest) advisory
   per report and records `advisory_ids` and `n_linked_advisories`.
4. Personal data dropped from processed tables: `Response_Update_By` (staff e-mail) and `Case_Lead`
   (staff name). Raw CSVs are kept as published.
5. Results: placeholder rows (value "0" with no analyte and no measurement type; 1,883 rows) → value
   NaN; veterinary rows excluded (none in this vintage; the data dictionary says not to use them).

## Known issues and biases

- **Voluntary, effort-driven reporting.** Reports scale with visitors, media attention and staffing
  (737 reports observed in 2022 → 296 in 2024). Report source: public 1,214, agency/partner 1,185,
  unknown 1,314. Never treat "no report" as "no bloom".
- **Suspected until confirmed.** All reports are suspected blooms until staff recommend an advisory;
  1,424 reports ended with "None".
- **Label circularity.** Caution can be posted on visual indicators alone. Only 31.8% of ≥ Caution
  advisory records cite toxins in `Advisory_Detail`; 7.1% are explicitly visual-only; most give no
  basis. A model fed predicted chl-a can score well circularly — report a chl-a-only baseline.
- **Sampling channel.** Event-response vs monitoring records differ (share ≥ Caution: 57.5% vs
  47.7%); stratify by `observation_channel`.
- **Onset and duration.** Observation/first-report date ≠ bloom onset; end dates are almost always
  missing (above). Use "occurred within week w", never lead time to a report date.
- **2022 data-system change.** Advisory-record volume jumps ~7× in 2022 (routine-monitoring updates);
  do not compare counts across 2021/2022.
- **Results are not a toxin dataset.** The 171 lab rows give analyte/method (ELISA, qPCR, strips,
  microscopy) but no values; only 34 field point measurements (conductivity, temperature, DO, pH)
  have values. Toxin concentrations live in CEDEN (not pulled here).
- Waterbody names are free text (spelling variants; "Lake tahoe"/"Lake Tahoe"); several cases can
  exist per waterbody — group by name + coordinates, not by `case_id`.

## Join notes

- Points: `latitude`/`longitude` → NHDPlus waterbody polygon (lakes/reservoirs: point-in-polygon or
  nearest waterbody within ~200 m) or NLDI `comid/position` for stream reports (1,062 reservoir,
  493 wadeable-stream, 94 lake report rows; 1,953 blank type). Record the snap distance.
- CyAN / EPA cyanoHAB forecast lakes: join by NHD waterbody COMID of the polygon containing the point
  (only large, satellite-resolvable reservoirs, e.g. Clear Lake).
- WQP/CEDEN toxin results: spatial match (same waterbody, ≤ 1 km) and date window.
- No HUC/COMID is published; county and Regional Water Board are text.

## Rebuild

```bash
# download a new vintage (raw/ca_fhab/<today>/) and rebuild processed tables + manifest
/home/bcheng/miniconda3/envs/sentinel2/bin/python -B scripts/data/download_events_ca_fhab.py
# rebuild the 2026-10-22 vintage offline from raw
/home/bcheng/miniconda3/envs/sentinel2/bin/python -B scripts/data/download_events_ca_fhab.py --vintage 2026-10-22 --skip-download
# tests (fixtures: tests/fixtures/labels_ca_fhab/, real rows with staff columns blanked)
/home/bcheng/miniconda3/envs/sentinel2/bin/python -B -m pytest tests/test_labels_ca_fhab.py
```

Code: `sentinel/data/sources/events_ca_fhab.py` (parsers, label logic), `scripts/data/download_events_ca_fhab.py`.
