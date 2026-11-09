# Datasheet — neon_chem (NEON surface-water chemistry (grab samples))

- **Source:** NSF NEON Data Portal API (`https://data.neonscience.org/api/v0`), products:
  - `DP1.20093.001` — Chemical properties of surface water (https://data.neonscience.org/data-products/DP1.20093.001; RELEASE-2026 https://doi.org/10.48443/7b2c-ga32)
- **Release:** `RELEASE-2026` (pinned; DOI per product on the product page) plus months only
  available as `PROVISIONAL`, flagged per row (`release`, `provisional`). Provisional data are
  not citable and may change; exclude them for frozen benchmarks unless stated.
- **License:** CC BY 4.0 (NEON data; since 2026-06-30, previously CC0). Attribution required: "NSF National Ecological Observatory Network (NEON). Data products as listed, RELEASE-2026 (product DOIs listed in each datasheet) and provisional data; https://data.neonscience.org (accessed 2026-10-24)."
- **Retrieved:** 2026-10-25 (API token, `X-API-Token` header; never stored).
- **Raw:** `$SENTINEL_DATA/raw/neon/<product>/...` (ledger `raw/neon/_ledger/<product>.jsonl`).
- **Processed:** `$SENTINEL_DATA/processed/neon_chem/`

Lab chemistry 12–26×/yr per site, collected at the same events as the 16S samples (`parentSampleID` e.g. `ARIK.SS.20190806`, `eventID`).

## Files

| file | content |
|---|---|
| `chem_long.parquet` | one row per sample × analyte: value, unit, censored, qualifier (BDL/ND), detection_limit, lab, sample_fraction (FIL/RAW/PCN/POM), var, release |
| `chem_event_wide.parquet` | one row per parentSampleID: <var> values (mean of duplicate analyses) + <var>__censored, field DO/SC/T, lab pH/ANC |
| `field.parquet` | swc_fieldSuperParent (coordinates, depth, field sonde) |
| `field_samples.parquet` | swc_fieldData (sample ids, volumes, conditions) |
| `domain_lab.parquet` | swc_domainLabData (pH, ANC, alkalinity) |
| `lab_summary.parquet` | lab method detection limits per analyte/period |
| `units.parquet` | var → analyte, unit |

## Coverage

```json
{
 "long": {
  "rows": 244598,
  "first": "2012-01-31",
  "last": "2026-09-23",
  "sites": 34,
  "releases": {
   "RELEASE-2026": 205908,
   "PROVISIONAL": 38690
  }
 },
 "events": 20266,
 "event_sites": 34,
 "domains": 19,
 "censored_frac": 0.13337803252683997,
 "detection_limit_matched_frac": 0.8613112126836687,
 "vars_top40": {
  "tpc_ugl": 9395,
  "tpn_ugl": 9213,
  "tp_mgl": 8373,
  "toc_mgl": 8370,
  "tdp_mgl": 8346,
  "ortho_p_mgl": 8345,
  "no2_n_mgl": 8345,
  "nh4_n_mgl": 8345,
  "uv_absorbance_254_nm_au": 8340,
  "uv_absorbance_280_nm_au": 8339,
  "dic_mgl": 8336,
  "tds_mgl": 8335,
  "no3_no2_n_mgl": 8326,
  "si_mgl": 8317,
  "ca_mgl": 8317,
  "k_mgl": 8317,
  "na_mgl": 8317,
  "mn_mgl": 8316,
  "mg_mgl": 8316,
  "fe_mgl": 8316,
  "br_mgl": 8306,
  "so4_mgl": 8306,
  "f_mgl": 8306,
  "cl_mgl": 8306,
  "tss_mgl": 8282,
  "doc_mgl": 8188,
  "tn_mgl": 8163,
  "tdn_mgl": 8122,
  "tss_dry_mass_ug": 3631,
  "anc_meql": 3143,
  "specificconductance_uscm": 3143,
  "tpn_mg": 12,
  "tpc_mg": 12,
  "no3_no2_n_ugl": 9,
  "tdp_ugl": 9,
  "tp_ugl": 9,
  "ortho_p_ugl": 9,
  "nh4_n_ugl": 9,
  "no2_n_ugl": 9
 }
}
```

## QC applied

- `censored` = belowDetectionQF ∈ {BDL, ND}; `value` is the published concentration (NaN for ND); no DL/2 substitution.
- `detection_limit` = methodDetectionLimit of the same lab/analyte/unit whose lab-specific date window contains the collection date.
- `below_mdl` (derived) = value < detection_limit; early-year rows can be below the MDL without a NEON flag — `censored` keeps NEON's flag only.
- Units kept per row (`unit`); `var` names encode the unit (e.g. `tn_mgl`, `tpc_ugl`).
- POM isotope replicates kept only in the long table.
- Missing units (EcoCore_CSU: Br/Cl/F/SO4 2022-04→2023-09, DIC/DOC/TDN/TN/TOC 2020-02→03, TDS 2022-12→2023-09, UV absorbance always) filled only from the same lab's single unit for that analyte (UV → 'absorbance units'); `unit_imputed=True` on every filled row.

## Known issues

- Lab changes over time (EcoCore CSU, ISWS CAL, FIU, USGS NWQA) — check `lab` before pooling.
- `shipment_warm_qf`=1 rows (warm shipment) kept but flagged.
- `externalLabDataQF` carries format changes and lab remarks.

## Rebuild

```bash
python -B scripts/data/download_neon.py --products chem
python -B scripts/data/build_neon.py chem
```
