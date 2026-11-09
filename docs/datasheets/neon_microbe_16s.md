# Datasheet — neon_microbe_16s (NEON surface-water 16S community tables)

- **Source:** NSF NEON Data Portal API (`https://data.neonscience.org/api/v0`), products:
  - `DP1.20141.002` — Surface water microbe community taxonomy (https://data.neonscience.org/data-products/DP1.20141.002; RELEASE-2026 https://doi.org/10.48443/4xwg-p967)
  - `DP1.20141.001` — Surface water microbe community composition (superseded; 2014-2018) (https://data.neonscience.org/data-products/DP1.20141.001; RELEASE-2026 https://doi.org/10.48443/1dv2-9s65)
  - `DP1.20282.001` — Surface water microbe marker gene sequences (https://data.neonscience.org/data-products/DP1.20282.001; RELEASE-2026 https://doi.org/10.48443/dx8c-sy17)
- **Release:** `RELEASE-2026` (pinned; DOI per product on the product page) plus months only
  available as `PROVISIONAL`, flagged per row (`release`, `provisional`). Provisional data are
  not citable and may change; exclude them for frozen benchmarks unless stated.
- **License:** CC BY 4.0 (NEON data; since 2026-06-30, previously CC0). Attribution required: "NSF National Ecological Observatory Network (NEON). Data products as listed, RELEASE-2026 (product DOIs listed in each datasheet) and provisional data; https://data.neonscience.org (accessed 2026-10-24)."
- **Retrieved:** 2026-10-25 (API token, `X-API-Token` header; never stored).
- **Raw:** `$SENTINEL_DATA/raw/neon/<product>/...` (ledger `raw/neon/_ledger/<product>.jsonl`).
- **Processed:** `$SENTINEL_DATA/processed/neon_microbe_16s/`

Per-sample 16S community profiles co-collected with grab chemistry (join on `parentSampleID`/`eventID`, see `neon_chem`). `primer_set` comes from the DP1.20282.001 PcrAmplification record of each dnaSampleID (never inferred from the date).

## Files

| file | content |
|---|---|
| `samples.parquet` | DP1.20141.002 16S samples: dnaSampleID, site, domain, time_utc, date, parentSampleID, eventID, sequencerRunID, primer_set/forwardPrimer/reversePrimer, read numbers, field DO/SC/T, release, provisional |
| `counts.parquet` | long counts (dnaSampleID, sequencerRunID, sequenceName, count, in_metadata) — RELEASE-2026 OTU ids (`ay26_16s_otu.*`) |
| `otus.parquet` | sequenceName → sequence, domain…genus, scientificName, rank confidences |
| `samples_v1/counts_v1/otus_v1.parquet` | DP1.20141.001 (2014–2018) ASV tables (md5 sequence names) |
| `primers.parquet` | all 16S PCR records with primer_set and primer_record_issue (no provisional PCR records yet) |
| `primer_release_summary.parquet` | primer set × product summary |

## Coverage

```json
{
 "primers": {
  "rows": 2514
 },
 "counts": {
  "rows": 4285470
 },
 "otus": {
  "rows": 76081
 },
 "samples": {
  "rows": 1494,
  "first": "2018-08-14",
  "last": "2024-11-12",
  "sites": 33,
  "releases": {
   "RELEASE-2026": 1494
  },
  "primer_sets": {
   "16S_V4V5_515FY-926R": 1494
  },
  "with_counts": 1494,
  "domains": 19
 },
 "counts_v1": {
  "rows": 449911
 },
 "otus_v1": {
  "rows": 174141
 },
 "samples_v1": {
  "rows": 942,
  "first": "2014-08-05",
  "last": "2018-10-30",
  "sites": 33,
  "releases": {
   "RELEASE-2026": 942
  },
  "primer_sets": {
   "16S_V3V4_Pro341F-Pro805R": 940,
   "unknown": 2
  },
  "with_counts": 942,
  "domains": 19
 }
}
```

## QC applied

- Counts with individualCount <= 0 or missing dropped.
- Samples linked to taxon tables by dnaSampleID; `in_metadata=False` rows are samples in a run file whose metadata row is absent (other product/site).
- Staff identifiers (`*By`) dropped.

## Known issues

- OTU ids are NOT comparable across releases or between .002 and .001.
- DP1.20141.002 (RELEASE-2026) starts 2018-08: legacy 2014–2018 samples (older primer sets) exist only as DP1.20141.001 sequence variants, so a cross-primer test also crosses pipeline/table version — compare on sequences (`otus*.sequence`) or taxonomy, never on OTU ids.
- The RELEASE-2026 taxonTableLabSummary still points to the AY25 reference table (mct_sequence_reference_table_16S_20241111, `ay25_16S_otu.*`), whose ids match none of the `ay26_16s_otu.*` ids in the taxon tables; it is not used (sequences come from the taxon tables themselves).
- Some taxon tables spell sample IDs in lower case (`WLOU.ss.`); IDs are canonicalised case-insensitively to the metadata spelling (`dnaSampleID_raw` keeps the original).
- DP1.20141.001: RELEASE-2026 removed duplicate records with/without sequencerRunID; rows without a run id are single-run samples.
- Primer sets in surface water (PcrAmplification records): V3V4 Pro341F/Pro805R 2014-08→2018-10 (all DP1.20141.001 samples, MiSeq); V4V5 515F-Y/926R from 2018-08 (all DP1.20141.002 samples; MiSeq→NovaSeq). No V4 515F/806R surface-water records were found. Primer set is confounded with time, table version and pipeline; no sample was sequenced with both.
- 342 GMCF PCR records (2019-10→2021-10) list ITS1F/ITS2 primers on a 16S v4v5 record — a metadata error (taxonomy 98% Bacteria, same amplicon length as V4V5); labelled V4V5 with `primer_record_issue` set.
- Sequencing platform changed (MiSeq → NovaSeq) — see `instrument_model`.
- COVID-19 reduced sampling 2020–2022; YELL/BLDE flood gap 2022.
- ITS tables not downloaded (16S only); FASTQ not downloaded.

## Primer sets and releases (from PcrAmplification records)

| scope                             | primer_set               |    n |   sites | first               | last                | protocols                                                                                      |
|:----------------------------------|:-------------------------|-----:|--------:|:--------------------|:--------------------|:-----------------------------------------------------------------------------------------------|
| DP1.20282.001 PCR (all sequenced) | 16S_V3V4_Pro341F-Pro805R | 1017 |      33 | 2014-08-05 00:00:00 | 2018-10-30 00:00:00 | BMI_PcrSOP_v1; BMI_PcrSOP_v2                                                                   |
| DP1.20282.001 PCR (all sequenced) | 16S_V4V5_515FY-926R      | 1497 |      33 | 2018-08-14 00:00:00 | 2024-11-12 00:00:00 | BMI_markerGenes_16Sv4v5SOP_v1.1; BMI_markerGenes_16Sv4v5SOP_v1.2; GMCF_markerGenes_16SSOP_v2.0 |
| DP1.20141.002 samples             | 16S_V4V5_515FY-926R      | 1494 |      33 | 2018-08-14 00:00:00 | 2024-11-12 00:00:00 | BMI_markerGenes_16Sv4v5SOP_v1.1; BMI_markerGenes_16Sv4v5SOP_v1.2; GMCF_markerGenes_16SSOP_v2.0 |
| DP1.20141.001 samples_v1          | 16S_V3V4_Pro341F-Pro805R |  940 |      33 | 2014-08-05 00:00:00 | 2018-10-30 00:00:00 | BMI_PcrSOP_v1; BMI_PcrSOP_v2                                                                   |
| DP1.20141.001 samples_v1          | unknown                  |    2 |       2 | 2015-03-03 00:00:00 | 2017-07-11 00:00:00 |                                                                                                |

## Rebuild

```bash
python -B scripts/data/download_neon.py --products microbe_tax microbe_comp_v1 microbe_seq --externals
python -B scripts/data/build_neon.py microbe
```

## Microbe ↔ chemistry links

`chem_links.parquet`: dnaSampleID → chem_parentSampleID (`link='parentSampleID'` = same field event; `'location_date'` = same named location, |Δdate| ≤ 1 day, nearest). Stats: `{"samples": 2436, "linked": 2161, "by_link": {"parentSampleID": 2117, "location_date": 44}, "by_product": {"DP1.20141.001": 667, "DP1.20141.002": 1494}}`. Rebuild: `python -B scripts/data/build_neon.py links` (after `microbe` and `chem`).
